"""
Tests for the row-level quality flags, the run log, and the validation checks.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from xopr_gline.grounding import FrameSpan, GlacierProfile, ProfileSource
from xopr_gline.sweep import (_confidence, _snap_to_terminus, forget_runs,
                              log_run, sweep_log, to_geodataframe)
from xopr_gline.validate import (compare_to_reference, repeat_consistency,
                                 repeat_summary, scoreboard)


def _qc(**overrides) -> dict:
    base = {"edge_pinned": False, "at_window_edge": False,
            "detector_spread_km": 0.5, "bocpd_model_spread_km": 0.2,
            "onset_upflow": False, "snapped_to_terminus": False}
    return {**base, **overrides}


class _Pick:
    """Stand-in for a DetectionResult, which only its extra dict matters here."""

    def __init__(self, map_km, **extra):
        self.map_km = map_km
        self.extra = extra


# -- confidence ------------------------------------------------------------
def test_agreement_away_from_the_edges_is_high_confidence():
    assert _confidence(_qc()) == "high"


def test_an_edge_pinned_pick_is_never_trusted():
    """The boundary chose the answer, however tightly the detectors agree."""
    assert _confidence(_qc(edge_pinned=True)) == "low"


def test_detectors_far_apart_are_low_confidence():
    assert _confidence(_qc(detector_spread_km=9.0)) == "low"
    assert _confidence(_qc(bocpd_model_spread_km=9.0)) == "low"


def test_a_pick_at_a_search_limit_is_only_medium():
    assert _confidence(_qc(at_window_edge=True)) == "medium"
    assert _confidence(_qc(detector_spread_km=3.0)) == "medium"


def test_an_onset_upflow_of_the_changepoint_is_low():
    """The transition cannot start upflow of its own steepest point."""
    assert _confidence(_qc(onset_upflow=True)) == "low"


def test_confidence_survives_missing_spreads():
    """A run where only one detector answered still gets a verdict."""
    assert _confidence(_qc(detector_spread_km=np.nan,
                           bocpd_model_spread_km=np.nan)) == "high"


def test_a_pick_snapped_to_the_front_is_not_written_off_as_low():
    """
    The position is the calving front, which is measured. What is unresolved is
    only whether a distinct grounding point sits inland of it.
    """
    assert _confidence(_qc(edge_pinned=True, snapped_to_terminus=True)) \
        == "medium"


def test_a_snapped_pick_with_detectors_far_apart_stays_low():
    assert _confidence(_qc(edge_pinned=True, snapped_to_terminus=True,
                           detector_spread_km=9.0)) == "low"


# -- snapping to the terminus ----------------------------------------------
def _tidewater(dx=0.25, n_water=40, n_ice=80):
    """Open water, then ice: a glacier calving at its grounding zone."""
    n = n_water + n_ice
    x = np.arange(n) * dx
    thickness = np.concatenate([np.full(n_water, 1.0),
                                np.linspace(340, 900, n_ice)])
    surface = np.concatenate([np.full(n_water, 0.0),
                              np.linspace(40, 500, n_ice)])
    return GlacierProfile(
        x=x, amp=np.full(n, -80.0), h_surf=surface, h_bed=surface - thickness,
        lat=68.0 + x / 100.0, lon=-50.0 - x / 100.0,
    )


def _floating_tongue(dx=0.25, n_water=40, n_shelf=80, n_ground=80):
    """Front, then a long tongue at flotation, then grounding well inland."""
    n = n_water + n_shelf + n_ground
    x = np.arange(n) * dx
    thickness = np.concatenate([np.full(n_water, 1.0),
                                np.full(n_shelf, 400.0),
                                np.linspace(400, 900, n_ground)])
    # Afloat over the shelf, riding progressively higher once grounded.
    surface = np.concatenate([
        np.full(n_water, 0.0),
        np.full(n_shelf, 400.0 * 0.108),
        np.linspace(400, 900, n_ground) * 0.108
        + np.linspace(0, 300, n_ground),
    ])
    return GlacierProfile(
        x=x, amp=np.full(n, -80.0), h_surf=surface, h_bed=surface - thickness,
        lat=68.0 + x / 100.0, lon=-50.0 - x / 100.0,
    )


def test_a_pick_pinned_against_the_front_is_reported_at_the_front():
    """
    The search is clamped at the terminus and the trim reserves the first few
    samples, so a glacier grounded at its front comes back a fixed ~1.75 km
    inland on every glacier. That is a bias, not a measurement.
    """
    profile = _tidewater()
    terminus = profile.terminus_crossings_km()[0]

    pick = _Pick(terminus + 1.75, pinned_edge="lo", pinned_edge_km=terminus)
    out = _snap_to_terminus(pick, profile)

    assert out["snapped_to_terminus"]
    assert out["map_km"] == pytest.approx(terminus)
    assert out["raw_map_km"] == pytest.approx(terminus + 1.75)
    assert out["at_terminus"] and out["terminus_gap_km"] == 0.0
    # The reported point moves with it, or the geometry contradicts the km.
    assert out["lat"] == pytest.approx(profile.position_at(terminus)[0])


def test_a_pileup_at_the_front_is_not_enough_on_its_own():
    """
    The pin says the posterior wanted to be at or beyond the seaward limit. It
    does not say the ice is afloat there: a changepoint model piles mass at a
    truncation regardless. Where the geometry puts grounding well inland, the
    pick must be left where it is rather than moved onto the front.
    """
    # A floating tongue: the front, then km of ice at flotation, and only then
    # grounding. The flotation crossing lands far inland of the terminus.
    profile = _floating_tongue()
    terminus = profile.terminus_crossings_km()[0]
    crossing = profile.flotation_crossing_km()
    assert abs(crossing - terminus) > 5.0, "fixture should ground well inland"

    pick = _Pick(terminus + 1.75, pinned_edge="lo", pinned_edge_km=terminus)
    out = _snap_to_terminus(pick, profile)
    assert not out["snapped_to_terminus"]
    assert out["snap_declined_km"] > 5.0


def test_snapping_renames_the_frame_when_it_crosses_a_seam():
    """
    The frame is named from the position, so a row must not keep naming the
    frame the detector's own pick fell in once the point has moved out of it.
    """
    profile = _tidewater()
    terminus = profile.terminus_crossings_km()[0]
    profile.source = ProfileSource(
        collection="2012_Greenland_P3", segment="20120510_01",
        frames=(FrameSpan("20120510_01_010", 10, 0.0, terminus + 0.5),
                FrameSpan("20120510_01_011", 11, terminus + 0.5, 40.0)),
    )
    # The detector's pick sits in frame 011, the terminus in 010.
    pick = _Pick(terminus + 1.75, pinned_edge="lo", pinned_edge_km=terminus)
    assert _snap_to_terminus(pick, profile)["frame"] == "20120510_01_010"


def test_a_pick_pinned_against_something_that_is_not_the_front_is_left_alone():
    """Only a boundary that is itself a calving front may move a pick."""
    profile = _tidewater()
    pick = _Pick(25.0, pinned_edge="hi", pinned_edge_km=26.0)
    assert _snap_to_terminus(pick, profile) == {"snapped_to_terminus": False}


def test_an_unpinned_pick_is_left_alone():
    profile = _tidewater()
    terminus = profile.terminus_crossings_km()[0]
    pick = _Pick(terminus + 5.0, pinned_edge=None)
    assert _snap_to_terminus(pick, profile) == {"snapped_to_terminus": False}


def test_snapping_can_be_turned_off():
    profile = _tidewater()
    terminus = profile.terminus_crossings_km()[0]
    pick = _Pick(terminus + 1.75, pinned_edge="lo", pinned_edge_km=terminus)
    assert _snap_to_terminus(pick, profile, snap=False) == {
        "snapped_to_terminus": False}


# -- geometry from a chosen detector ---------------------------------------
def test_a_geopackage_can_be_keyed_on_a_detectors_own_pick():
    """
    A snapped row's `lat`/`lon` are the terminus, so the raw changepoint has to
    come from the detector's own coordinates or it cannot be mapped at all.
    """
    rows = pd.DataFrame({
        "status": ["ok"],
        "map_km": [30.0], "lat": [77.7], "lon": [-66.6],      # snapped
        "bocpd_km": [31.75], "bocpd_lat": [77.72], "bocpd_lon": [-66.64],
    })
    out = to_geodataframe(rows, detector="bocpd")
    assert out.geometry.iloc[0].y == pytest.approx(77.72)
    # The km column moves with the geometry, or the two disagree.
    assert out["map_km"].iloc[0] == pytest.approx(31.75)
    assert out["geometry_source"].iloc[0] == "bocpd"


def test_asking_for_a_detector_a_sweep_did_not_record_is_an_error():
    """Older sweeps carry only the reported position; say so, don't guess."""
    rows = pd.DataFrame({"status": ["ok"], "map_km": [30.0],
                         "lat": [77.7], "lon": [-66.6]})
    with pytest.raises(ValueError, match="bocpd"):
        to_geodataframe(rows, detector="bocpd")


# -- run log ---------------------------------------------------------------
def test_a_run_is_logged_beside_its_output(tmp_path):
    out = tmp_path / "sweep.jsonl"
    path = log_run(out, sweep_id="abc123", event="start", params={"pad_km": 4})
    assert path == tmp_path / "sweep.runs.jsonl"

    record = json.loads(path.read_text())
    assert record["sweep_id"] == "abc123"
    assert record["params"]["pad_km"] == 4
    # Enough to reproduce the run later.
    assert "commit" in record and "logged" in record and "host" in record


def test_the_log_appends_rather_than_replaces(tmp_path):
    out = tmp_path / "sweep.jsonl"
    log_run(out, sweep_id="one", event="start")
    log_run(out, sweep_id="one", event="finish")
    log_run(out, sweep_id="two", event="start")

    history = sweep_log(out)
    assert len(history) == 3
    assert history["sweep_id"].tolist() == ["one", "one", "two"]


def test_named_runs_can_be_forgotten_so_a_resume_redoes_them(tmp_path):
    """After a fix that changes correct rows, not failing ones."""
    out = tmp_path / "sweep.jsonl"
    with out.open("w") as fh:
        for run_id in ("a#1", "b#1", "c#1"):
            fh.write(json.dumps({"run_id": run_id, "status": "ok"}) + "\n")

    assert forget_runs(out, ["a#1", "c#1"]) == 2
    left = [json.loads(line) for line in out.open()]
    assert [r["run_id"] for r in left] == ["b#1"]


def test_forgetting_a_run_that_is_not_there_drops_nothing(tmp_path):
    out = tmp_path / "sweep.jsonl"
    out.write_text(json.dumps({"run_id": "a#1", "status": "ok"}) + "\n")
    assert forget_runs(out, ["zzz#9"]) == 0


def test_logging_without_an_output_is_a_no_op():
    assert log_run(None, sweep_id="x") is None
    assert sweep_log(None).empty


# -- validation ------------------------------------------------------------
def _points(lons, lats, ids, dates=None, status="ok") -> gpd.GeoDataFrame:
    frame = pd.DataFrame({
        "GlacierID": ids,
        "status": status,
        "date": dates or ["2010-04-20"] * len(ids),
    })
    return gpd.GeoDataFrame(frame, crs="EPSG:4326",
                            geometry=gpd.points_from_xy(lons, lats))


def test_repeat_visits_to_one_glacier_are_scored_against_each_other():
    """Three visits within a km, one 30 km away: the outlier should show."""
    points = _points([-50.0, -50.001, -50.002, -49.3],
                     [68.0, 68.001, 68.002, 68.0],
                     ids=[1, 1, 1, 1],
                     dates=["2010-04-20", "2012-04-20", "2014-04-20",
                            "2016-04-20"])
    scored = repeat_consistency(points, tolerance_km=3.0)
    assert scored["repeat_outlier"].sum() == 1
    assert scored.loc[3, "repeat_offset_km"] > 10.0
    assert scored["repeat_n"].eq(4).all()


def test_a_single_visit_is_neither_consistent_nor_not():
    points = _points([-50.0], [68.0], ids=[7])
    scored = repeat_consistency(points)
    assert not scored["repeat_outlier"].any()


def test_repeat_summary_lists_the_years_a_glacier_was_flown():
    points = _points([-50.0, -50.001], [68.0, 68.001], ids=[1, 1],
                     dates=["2010-04-20", "2014-04-20"])
    summary = repeat_summary(repeat_consistency(points))
    assert summary.loc[1, "years"] == [2010, 2014]
    assert summary.loc[1, "n"] == 2


def test_distance_to_a_reference_line_is_reported_in_km():
    points = _points([-50.0], [68.0], ids=[1])
    reference = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt(
            ["LINESTRING (-50.1 67.9, -50.1 68.1)"]), crs="EPSG:4326")
    out = compare_to_reference(points, reference)
    assert 0.0 < out["reference_distance_km"].iloc[0] < 10.0


def test_a_row_that_found_nothing_has_no_reference_distance():
    points = _points([-50.0], [68.0], ids=[1])
    points.loc[0, "geometry"] = None
    reference = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt(
            ["LINESTRING (-50.1 67.9, -50.1 68.1)"]), crs="EPSG:4326")
    assert compare_to_reference(points, reference)[
        "reference_distance_km"].isna().all()


def test_the_scoreboard_counts_only_rows_that_found_something():
    points = _points([-50.0, -50.1], [68.0, 68.1], ids=[1, 2],
                     status=["ok", "screened_out"])
    points["edge_pinned"] = [True, False]
    points["confidence"] = ["low", "high"]
    board = scoreboard(points)
    assert board["n"] == 1
    assert board["edge_pinned"] == 1.0
    assert board["confidence_low"] == 1.0

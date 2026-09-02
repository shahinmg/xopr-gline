"""
Tests for the segment sweep: task building and the never-raises contract.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xopr_gline.sweep import (_clusters, _frames_text, _pad, _parse_frames,
                              run_segment, tasks_from_points, to_geodataframe,
                              useable_frames)

HELHEIM = Path("/home/laserglaciers/igs_2026/"
               "greenland_v6_bocpd_cleaned_terminus_gap_helheim.gpkg")


def test_clusters_split_on_a_frame_gap():
    """One segment can cross several glaciers, and each is its own run."""
    assert _clusters([6, 7, 12, 13, 14, 20]) == [[6, 7], [12, 13, 14], [20]]
    assert _clusters([3]) == [[3]]
    assert _clusters([1, 2, 3]) == [[1, 2, 3]]


def test_clusters_honour_max_gap():
    assert _clusters([1, 3, 5], max_gap=2) == [[1, 3, 5]]
    assert _clusters([1, 3, 5], max_gap=1) == [[1], [3], [5]]


def test_padding_is_clipped_to_the_segment():
    assert _pad([1, 2], 1, last=10) == [1, 2, 3]
    assert _pad([9, 10], 1, last=10) == [8, 9, 10]
    assert _pad([5], None, last=10) is None
    assert _pad([5], 1, last=None) is None


def test_useable_frames_collapses_split_line_parts():
    """The gpkg splits some granules over several line parts."""
    frames = useable_frames()
    assert frames["granule"].is_unique
    assert len(frames) == frames["granule"].nunique()
    assert frames["segment"].str.match(r"^\d{8}_\d{2}$").all()


def test_a_segment_without_a_collection_never_runs():
    """Unresolvable segments become a row, not an exception."""
    row = run_segment(None, "20990101_01", [1, 2])
    assert row["status"] == "not_in_catalog"
    assert row["segment"] == "20990101_01"


def test_a_bad_segment_returns_a_row_rather_than_raising():
    row = run_segment("2010_Greenland_DC8", "20990101_99", [1])
    assert row["status"] == "no_profile"
    assert row["reason"]


def test_rows_are_json_serialisable():
    """run_sweep appends each row to JSONL, so nothing may be exotic."""
    row = run_segment(None, "20990101_01", [1, 2])
    assert json.loads(json.dumps(row, default=str))["status"] == "not_in_catalog"


def test_to_geodataframe_survives_a_sweep_that_found_nothing():
    rows = pd.DataFrame([{"segment": "a", "status": "no_profile"}])
    points = to_geodataframe(rows)
    assert points.crs == "EPSG:4326"
    assert points.geometry.isna().all()


def test_to_geodataframe_places_the_point():
    rows = pd.DataFrame([{"segment": "a", "status": "ok",
                          "lat": 80.5, "lon": -60.0}])
    point = to_geodataframe(rows).geometry.iloc[0]
    assert (point.x, point.y) == (-60.0, 80.5)


# -- rerunning an earlier sweep's frames -----------------------------------
def test_frame_spans_round_trip():
    """tasks_from_points reads back exactly what _frames_text wrote."""
    for text in ("004-006", "001-002", "041-044", "015-016"):
        assert _frames_text(_parse_frames(text)) == text
    assert _parse_frames("007") == [7]
    assert _parse_frames(None) is None
    assert _parse_frames("") is None


@pytest.mark.skipif(not HELHEIM.exists(), reason="helheim gpkg not present")
def test_tasks_from_points_reruns_the_frames_of_an_earlier_sweep():
    """A curated result set drives the same runs it was produced from."""
    tasks = tasks_from_points(HELHEIM)
    assert len(tasks) == 9
    assert tasks["segment"].nunique() == 7
    # run_ids are numbered per segment, as build_tasks does
    assert set(tasks[tasks["segment"] == "20080730_01"]["run_id"]) == {
        "20080730_01#1", "20080730_01#2"}
    assert tasks[tasks["run_id"] == "20080730_01#1"]["frames"].iloc[0] == [4, 5, 6]
    assert tasks["collection"].notna().all()


# -- buttressing diagnostic ------------------------------------------------
def _shelf(pedestal_m, n=240, dx=0.25, x_gl=30.0, thickness=800.0):
    """
    A shelf at flotation seaward of x_gl, held pedestal_m above it inland.

    x increases landward, so landward_sign() is +1 and the seaward window is
    the low-x side.
    """
    from xopr_gline.constants import DEFAULT_CONSTANTS
    from xopr_gline.grounding.profile import GlacierProfile

    x = np.arange(n) * dx
    residual = np.where(x >= x_gl, pedestal_m, 0.0)
    surf_msl = thickness * DEFAULT_CONSTANTS.flotation_factor + residual
    return GlacierProfile(x=x, amp=np.full(n, -110.0),
                          h_surf=surf_msl, h_bed=surf_msl - thickness,
                          geoid_separation_m=0.0)


class _Pick:
    def __init__(self, km):
        self.map_km = km


def test_a_buttressed_pick_reads_above_its_floating_level():
    """Helheim sits ~43 m above free flotation at the pick, ~0 m seaward."""
    from xopr_gline.sweep import _flotation_fields

    p = _shelf(pedestal_m=40.0)
    f = _flotation_fields(_Pick(32.0), p)          # 2 km inland of the step
    assert f["res_at_pick_m"] == pytest.approx(40.0, abs=0.5)
    assert f["res_seaward_m"] == pytest.approx(0.0, abs=0.5)
    assert f["res_offset_m"] == pytest.approx(40.0, abs=1.0)


def test_a_freely_floating_pick_reads_no_offset():
    """Petermann's wide tongue is at flotation on both sides of the pick."""
    from xopr_gline.sweep import _flotation_fields

    f = _flotation_fields(_Pick(32.0), _shelf(pedestal_m=0.0))
    assert f["res_offset_m"] == pytest.approx(0.0, abs=0.5)

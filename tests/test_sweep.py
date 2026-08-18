"""
Tests for the segment sweep: task building and the never-raises contract.
"""

import json

import numpy as np
import pandas as pd
import pytest

from xopr_gline.sweep import (_clusters, _pad, run_segment, to_geodataframe,
                              useable_frames)


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

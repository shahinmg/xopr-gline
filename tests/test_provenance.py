"""
Tests for frame provenance and grounding point geometry.
"""

import numpy as np
import pandas as pd
import pytest

from xopr_gline.grounding import (DetectionResult, FrameSpan, GlacierProfile,
                                  GradientDetector, ProfileSource)


def _source():
    """Two 10 km frames laid end to end, as from_xopr would build them."""
    return ProfileSource(
        collection="2010_Greenland_DC8", segment="20100420_03",
        frames=(
            FrameSpan("20100420_03_007", 7, 0.0, 10.0,
                      pd.Timestamp("2010-04-20T15:00"),
                      pd.Timestamp("2010-04-20T15:10")),
            FrameSpan("20100420_03_008", 8, 10.0, 20.0,
                      pd.Timestamp("2010-04-20T15:10"),
                      pd.Timestamp("2010-04-20T15:20")),
        ),
    )


def _profile(source=None, n=81):
    """A 0.25 km profile over 0-20 km with a track running north-west."""
    x = np.linspace(0.0, 20.0, n)
    return GlacierProfile(
        x=x,
        amp=np.linspace(-120.0, -80.0, n),
        h_surf=np.full(n, 300.0),
        h_bed=np.full(n, -600.0),
        lat=68.0 + x / 100.0,
        lon=-50.0 - x / 100.0,
        source=source,
    )


def test_n_frames_and_date_come_from_the_spans():
    source = _source()
    assert source.n_frames == 2
    assert source.date == pd.Timestamp("2010-04-20T15:00")
    assert str(source) == "2010_Greenland_DC8/20100420_03 (2 frames)"


def test_frame_at_km_names_the_frame_holding_a_point():
    source = _source()
    assert source.frame_at_km(3.0).frame == 7
    assert source.frame_at_km(17.0).frame == 8
    assert source.frame_at_km(25.0) is None


def test_frames_between_reports_every_frame_a_window_touches():
    source = _source()
    assert [f.frame for f in source.frames_between(8.0, 12.0)] == [7, 8]
    assert [f.frame for f in source.frames_between(1.0, 4.0)] == [7]
    assert source.frames_between(30.0, 40.0) == ()


def test_the_seam_between_frames_is_closed():
    """Consecutive frames leave a one-sample gap; a lookup there is not None."""
    from xopr_gline.grounding.profile import _close_seams
    spans = _close_seams([FrameSpan("a", 1, 0.0, 31.88),
                          FrameSpan("b", 2, 31.95, 81.16)])
    assert spans[0].hi_km == 31.95
    assert ProfileSource(frames=spans).frame_at_km(31.91).frame == 1


def test_a_missing_frame_stays_a_gap():
    """A gap wider than a sample seam is a frame that was not loaded."""
    from xopr_gline.grounding.profile import _close_seams
    spans = _close_seams([FrameSpan("a", 1, 0.0, 30.0),
                          FrameSpan("c", 3, 60.0, 90.0)])
    assert spans[0].hi_km == 30.0
    assert ProfileSource(frames=spans).frame_at_km(45.0) is None


def test_untimed_frames_give_no_date():
    """Synthetic profiles have no slow_time, and that is not an error."""
    assert ProfileSource(frames=(FrameSpan("a", 1, 0.0, 5.0),)).date is None


def test_position_at_interpolates_the_track():
    profile = _profile()
    lat, lon = profile.position_at(10.0)
    assert lat == pytest.approx(68.1)
    assert lon == pytest.approx(-50.1)


def test_position_at_is_nan_without_a_track():
    profile = GlacierProfile(x=np.arange(4.0), amp=np.zeros(4),
                             h_surf=np.full(4, 300.0), h_bed=np.full(4, -600.0))
    assert all(np.isnan(v) for v in profile.position_at(2.0))


def test_detector_carries_the_point_and_its_frame():
    """The geometry and provenance a geodataframe row needs, end to end."""
    profile = _profile(_source())
    result = GradientDetector().detect(profile, (0.0, 20.0))
    assert (result.lat, result.lon) == profile.position_at(result.map_km)
    assert result.frame.granule == _source().frame_at_km(result.map_km).granule
    assert [f.frame for f in result.frames_used] == [7, 8]


def test_a_windowed_profile_keeps_frame_positions():
    """window() slices x but not the axis, so frame bounds still apply."""
    profile = _profile(_source()).window(12.0, 20.0)
    result = GradientDetector().detect(profile, (12.0, 20.0))
    assert result.frame.frame == 8
    assert [f.frame for f in result.frames_used] == [8]


def test_result_without_provenance_has_no_frame():
    result = DetectionResult(map_km=5.0, x_km=np.arange(10.0))
    assert result.frame is None
    assert result.frames_used == ()
    assert np.isnan(result.lat)

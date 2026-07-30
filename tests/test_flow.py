"""
Tests for flight-track alignment with ice flow.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import pytest

from xopr_gline.grounding import GlacierProfile
from xopr_gline.grounding.flow import (
    _monotonic_fraction,
    _project,
    _runs,
    along_flow_runs,
    flow_angle_deg,
    longest_along_flow_run,
    select_flotation_leg,
    signed_cos,
    sinuosity,
    track_heading,
)

# Real flight tracks. Both live under the gitignored data/, so tests that need
# them skip on a fresh clone.
PETERMANN_CSV = Path("data/petermann_csv/petermann_20100420_03_surface.csv")
HELHEIM_CSV = Path("data/helheim_csv/Helheim_20080730_01_surface_float.csv")


def _track(csv: Path):
    if not csv.exists():
        pytest.skip(f"{csv} not present")
    frame = pd.read_csv(csv)
    return frame["Latitude"].values, frame["Longitude"].values


def _line(n=100, bearing_deg=0.0, length_m=50_000.0):
    """A straight track in EPSG:3413, returned as lat/lon."""
    theta = np.radians(bearing_deg)
    s = np.linspace(0, length_m, n)
    x = -272_710 + s * np.cos(theta)
    y = -983_360 + s * np.sin(theta)
    transformer = pyproj.Transformer.from_crs("EPSG:3413", "EPSG:4326",
                                              always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


def test_track_heading_is_constant_on_a_straight_line():
    """A straight leg has one heading, and it is a unit vector."""
    lat, lon = _line(bearing_deg=30.0)
    ex, ey = track_heading(lat, lon)
    assert np.allclose(np.hypot(ex, ey), 1.0)
    assert np.allclose(ex, ex[len(ex) // 2], atol=1e-3)
    assert np.allclose(ey, ey[len(ey) // 2], atol=1e-3)


def test_sinuosity_straight_versus_out_and_back():
    """1.0 for a straight leg; large for a track that doubles back."""
    lat, lon = _line()
    assert sinuosity(lat, lon) == pytest.approx(1.0, abs=1e-6)

    # Fly out and return along the same line, stopping just short of the start.
    back_lat = np.concatenate([lat, lat[::-1][1:]])
    back_lon = np.concatenate([lon, lon[::-1][1:]])
    assert sinuosity(back_lat[:-1], back_lon[:-1]) > 10.0


def test_monotonic_fraction_splits_an_out_and_back():
    """One-way is 1.0; an equal out-and-back is 0.5."""
    assert _monotonic_fraction(np.ones(10)) == pytest.approx(1.0)
    assert _monotonic_fraction(
        np.concatenate([np.ones(10), -np.ones(10)])
    ) == pytest.approx(0.5)
    assert np.isnan(_monotonic_fraction(np.full(5, np.nan)))


def test_runs_are_inclusive_index_pairs():
    mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert _runs(mask) == [(1, 2), (5, 7)]
    assert _runs(np.ones(3, dtype=bool)) == [(0, 2)]
    assert _runs(np.zeros(3, dtype=bool)) == []


def test_petermann_reads_as_along_flow():
    """The known-good transect: small angle, straight, one-way."""
    lat, lon = _track(PETERMANN_CSV)
    angle = flow_angle_deg(lat, lon)

    assert np.isfinite(angle).mean() > 0.8
    assert np.nanmedian(angle) < 15.0
    assert sinuosity(lat, lon) < 1.10
    assert _monotonic_fraction(signed_cos(lat, lon)) > 0.95


def test_rotating_a_track_makes_it_cross_flow():
    """
    Rotating the Petermann track 90 degrees about its centre should swing the
    angle towards across-flow, which pins the sampling and heading maths to the
    velocity field rather than to any assumed fjord axis.
    """
    lat, lon = _track(PETERMANN_CSV)
    to_3413 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3413",
                                          always_xy=True)
    to_4326 = pyproj.Transformer.from_crs("EPSG:3413", "EPSG:4326",
                                          always_xy=True)
    px, py = to_3413.transform(lon, lat)
    cx, cy = px.mean(), py.mean()
    lon_r, lat_r = to_4326.transform(cx - (py - cy), cy + (px - cx))

    across = np.nanmedian(flow_angle_deg(lat_r, lon_r))
    assert across - np.nanmedian(flow_angle_deg(lat, lon)) > 40.0


def _out_and_back_profile():
    """
    A synthetic out-and-back over Petermann's coordinates.

    Both legs cross flotation, but only the second carries bed power through
    its transition — the Helheim 20080730_01 situation in miniature.
    """
    lat, lon = _track(PETERMANN_CSV)
    c_lat, c_lon = lat[::20], lon[::20]
    full = len(c_lat)
    short = int(0.55 * full)     # the return leg is the shorter one

    # Out along the reversed centreline (seaward), then part way back.
    lat_ab = np.concatenate([c_lat[::-1], c_lat[1:short]])
    lon_ab = np.concatenate([c_lon[::-1], c_lon[1:short]])
    px, py = _project(lat_ab, lon_ab)
    x = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    x = x / 1000.0

    # Surface rises landward, so the flotation residual crosses on each leg.
    t = np.concatenate([np.arange(full)[::-1], np.arange(1, short)])
    h_surf = 60.0 + 240.0 * t / (full - 1)
    h_bed = np.full_like(h_surf, -600.0)

    amp = np.full_like(h_surf, -100.0)
    amp[:full] = np.nan          # the long leg has no bed picks

    return GlacierProfile(x=x, amp=amp, h_surf=h_surf, h_bed=h_bed,
                          lat=lat_ab, lon=lon_ab)


def test_selection_prefers_the_leg_with_data_over_the_longest():
    """
    The point of select_flotation_leg: an out-and-back crosses flotation once
    per leg, and the longest leg is not necessarily the usable one.
    """
    profile = _out_and_back_profile()
    legs = along_flow_runs(profile)
    assert len(legs) >= 2

    midpoint = 0.5 * (profile.x[0] + profile.x[-1])
    longest = longest_along_flow_run(profile)
    selected = select_flotation_leg(profile)

    assert selected is not None
    # The leg carrying bed power is the second half of the track.
    assert selected[0] >= midpoint
    # Length alone would have chosen the first, data-free leg.
    assert longest[0] < midpoint
    assert selected != longest


def test_selection_returns_none_without_a_crossing():
    """A profile that never approaches flotation yields no leg."""
    profile = _out_and_back_profile()
    flat = GlacierProfile(x=profile.x, amp=profile.amp,
                          h_surf=np.full_like(profile.x, 2000.0),
                          h_bed=np.full_like(profile.x, -600.0),
                          lat=profile.lat, lon=profile.lon)
    assert select_flotation_leg(flat) is None


def test_helheim_out_and_back_is_caught_despite_a_small_angle():
    """
    The Helheim repeat line flies up the flowline and straight back down it, so
    its folded angle looks fine (~7 deg) and only the sinuosity and monotonic
    checks reject it. This is the case the angle alone cannot see.
    """
    lat, lon = _track(HELHEIM_CSV)

    assert np.nanmedian(flow_angle_deg(lat, lon)) < 20.0
    assert sinuosity(lat, lon) > 10.0
    assert _monotonic_fraction(signed_cos(lat, lon)) == pytest.approx(0.5,
                                                                     abs=0.1)

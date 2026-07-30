"""
Tests for terminus detection and segment screening.
"""

import numpy as np
import pytest

from xopr_gline.grounding import GlacierProfile


def _profile(thickness, surface=300.0, geoid=0.0):
    """A profile with a prescribed thickness, on a 0.5 km grid."""
    thickness = np.asarray(thickness, dtype=float)
    h_surf = (np.full_like(thickness, surface) if np.isscalar(surface)
              else np.asarray(surface, dtype=float))
    return GlacierProfile(
        x=np.arange(len(thickness)) * 0.5,
        amp=np.full_like(thickness, -100.0),
        h_surf=h_surf,
        h_bed=h_surf - thickness,
        geoid_separation_m=geoid,
    )


def test_ice_mask_rejects_collapsed_picks():
    """Thickness at or below the floor is water, not thin ice."""
    profile = _profile([600, 600, 10, -5, 2, 600])
    assert profile.ice_mask(25.0).tolist() == [True, True, False, False,
                                               False, True]


def test_terminus_found_where_thickness_collapses():
    """The crossing sits on the ice side of the boundary."""
    profile = _profile([600] * 10 + [2] * 10)
    crossings = profile.terminus_crossings_km(min_thickness_m=25.0)
    assert crossings == [4.5]          # last ice sample, 9 * 0.5 km


def test_a_pick_gap_is_not_a_terminus():
    """
    NaN thickness is unknown, not open water. A bed-pick gap must not be
    mistaken for a calving front, which is the difference between a real
    terminus and a hole in the layer picks.
    """
    gapped = _profile([600] * 10 + [np.nan] * 10 + [600] * 10)
    assert gapped.terminus_crossings_km() == []

    # The same profile with measured water instead of a gap does cross.
    watery = _profile([600] * 10 + [2] * 10 + [600] * 10)
    assert len(watery.terminus_crossings_km()) == 2


def test_short_puddles_are_ignored():
    """A crossing needs a sustained open stretch, not one odd sample."""
    puddle = _profile([600] * 10 + [2] + [600] * 10)
    assert puddle.terminus_crossings_km(min_open_km=1.0) == []

    # Three samples span exactly 1.0 km, so they clear a 1.0 km bar but not 1.5.
    lake = _profile([600] * 10 + [2] * 3 + [600] * 10)
    assert lake.terminus_crossings_km(min_open_km=1.5) == []
    assert len(lake.terminus_crossings_km(min_open_km=1.0)) == 2


def test_open_water_would_read_as_near_flotation():
    """
    Why ice_mask exists: with thickness ~ 0 the flotation residual collapses to
    the water surface height above the geoid, so unmasked water reads as ice at
    flotation. These are the Helheim 20080730_01 numbers -- 300 m of ice, then
    a 55 m water surface over a 49 m geoid.
    """
    profile = _profile([600] * 10 + [1] * 10,
                       surface=[300.0] * 10 + [55.0] * 10,
                       geoid=49.0)
    residual = profile.flotation_residual

    assert abs(residual[-1]) < 15.0            # water reads as near flotation
    assert residual[0] > 100.0                 # real ice does not
    assert not profile.ice_mask()[-1]
    assert profile.ice_mask()[0]

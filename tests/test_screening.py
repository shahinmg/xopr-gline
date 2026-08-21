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


def test_flotation_window_is_cut_at_the_terminus():
    """
    The window must not run past the calving front. Unclamped it would extend
    over the water and hand the detectors an ice/water contrast to lock onto.
    """
    thickness = np.concatenate([
        np.linspace(900, 400, 30),     # grounded, well above flotation
        np.linspace(400, 330, 10),     # approaching flotation
        np.full(15, 1.0),              # open water past the front
    ])
    profile = _profile(thickness, surface=np.concatenate([
        np.linspace(400, 120, 30), np.linspace(120, 60, 10), np.full(15, 55.0),
    ]), geoid=49.0)

    terminus = profile.terminus_crossings_km()
    assert terminus, "fixture should contain a calving front"

    lo, hi = profile.flotation_window(margin_km=12.0)
    assert hi <= terminus[-1]
    # Unclamped, a 12 km margin would have reached the profile's end.
    assert hi < profile.x[-1]


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


# -- the calving front, as distinct from any other ice in the fjord --------
def _fjord_profile(seaward=True, n_ice=30):
    """
    Glacier tapering to flotation, calving front, open water, a berg, more water.

    The taper matters: the flotation crossing has to land inland of the front,
    as it does on real tidewater ice, or the window clamp has nothing to bite
    on. `seaward=False` reverses the line, which is the orientation
    `crossings[-1]` happens to get right.
    """
    h_surf = np.concatenate([np.linspace(800, 60, n_ice), np.zeros(20),
                             np.full(5, 20.0), np.zeros(15)])
    thickness = np.concatenate([np.linspace(900, 550, n_ice), np.zeros(20),
                                np.full(5, 200.0), np.zeros(15)])
    if not seaward:
        h_surf, thickness = h_surf[::-1], thickness[::-1]
    return GlacierProfile(
        x=np.arange(len(h_surf)) * 0.5,
        amp=np.full(len(h_surf), -100.0),
        h_surf=h_surf,
        h_bed=h_surf - thickness,
        geoid_separation_m=0.0,
    )


def test_the_calving_front_is_the_glacier_front_not_a_berg():
    """Bergs in the fjord each make a crossing; only the front is the terminus."""
    profile = _fjord_profile(seaward=True)
    assert profile.landward_sign() == -1
    assert profile.terminus_crossings_km() == [14.5, 25.0, 27.0]
    assert profile.calving_front_km() == 14.5


def test_the_calving_front_is_found_whichever_way_the_line_runs():
    """
    `crossings[-1]` is the front only where x runs landward. Reversed, the
    glacier sits at high x and its front is the largest crossing; running
    seaward it is the smallest, and `crossings[-1]` is a berg.
    """
    seaward = _fjord_profile(seaward=True)
    landward = _fjord_profile(seaward=False)
    assert seaward.landward_sign() == -1
    assert landward.landward_sign() == 1
    assert seaward.calving_front_km() == min(seaward.terminus_crossings_km())
    assert landward.calving_front_km() == max(landward.terminus_crossings_km())
    # The orientation the old code got wrong.
    assert seaward.calving_front_km() != seaward.terminus_crossings_km()[-1]


def test_the_calving_front_is_none_without_open_water():
    assert _profile([600] * 20).calving_front_km() is None


def test_a_reference_position_picks_the_first_front_seaward_of_it():
    profile = _fjord_profile(seaward=True)
    crossings = profile.terminus_crossings_km()
    assert profile.calving_front_km(0.0) == crossings[0]
    # From out in the fjord, past the front, the next ice edge seaward.
    assert profile.calving_front_km(crossings[0] + 0.25) == crossings[1]


def test_flotation_window_clamps_at_the_front_not_at_a_berg_beyond_it():
    """
    The clamp exists to keep open water out of the search. Assigning every
    crossing in turn let the last one win, so on a seaward-running line a berg
    stretched the window 12.5 km past the front, which is exactly the water the
    clamp is there to exclude.
    """
    profile = _fjord_profile(seaward=True)
    front = profile.calving_front_km()
    lo, hi = profile.flotation_window(margin_km=12.0)
    assert hi == front
    assert hi < profile.terminus_crossings_km()[-1]


def test_flotation_window_clamps_at_the_front_running_landward_too():
    profile = _fjord_profile(seaward=False)
    lo, hi = profile.flotation_window(margin_km=12.0)
    assert lo == profile.calving_front_km()


def test_the_calving_front_is_none_when_the_orientation_is_unknowable():
    """
    Which way is seaward comes from the flotation residual. A profile with none
    has no front to report, and must not raise: screening reaches here while
    deciding to throw the segment out, and turned a verdict into screen_failed
    on 20140424_01#4.
    """
    nan_profile = _profile([np.nan] * 20)
    with pytest.raises(ValueError):
        nan_profile.landward_sign()
    assert nan_profile.calving_front_km() is None

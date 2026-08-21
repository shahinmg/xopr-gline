"""
Invariants of the geometry panel's column model.

The figures these guard against had ocean drawn inside the glacier, ice with
its base above sea level, and a BedMachine seabed running through the middle of
the ice. Each came from a fill deriving its own mask, so the tests are on the
one classification every fill now reads from.
"""

import numpy as np
import pytest

from xopr_gline.constants import DEFAULT_CONSTANTS
from xopr_gline.grounding.plotting import (_GROUNDED, _MAX_BRIDGE_KM, _NOBASE,
                                           _build_section, _hold_front_bottom,
                                           _interp_gaps)
from xopr_gline.grounding.profile import GlacierProfile

GEOID = 30.0
SEA_LEVEL = GEOID
F = DEFAULT_CONSTANTS.flotation_factor


def _profile(bed_gap=None, thickness=800.0, n=240, dx=0.25):
    """
    A glacier running seaward: grounded inland, afloat, then open water.

    bed_gap is a (lo_km, hi_km) stretch whose bottom pick is blanked, which is
    the case the fills used to fall apart on.
    """
    x = np.arange(n) * dx
    h = np.linspace(thickness, 300.0, n)          # thinning seaward
    bed_msl = np.linspace(-300.0, -900.0, n)      # grounded inland, deep bed
    surf_msl = np.maximum(bed_msl + h, h * F)     # flotation once bed is deep
    bed_msl = surf_msl - h

    front = int(0.8 * n)                          # open water past here
    surf_msl[front:] = 0.0
    bed_msl[front:] = 0.0

    h_surf = surf_msl + GEOID
    h_bed = bed_msl + GEOID
    if bed_gap is not None:
        h_bed = np.where((x >= bed_gap[0]) & (x <= bed_gap[1]), np.nan, h_bed)

    amp = np.where(x < x[front], -110.0, -40.0)
    return GlacierProfile(x=x, amp=amp, h_surf=h_surf, h_bed=h_bed,
                          geoid_separation_m=GEOID)


def _section(profile, gp_km, bathymetry=None):
    return _build_section(profile, gp_km, SEA_LEVEL, bathymetry, floor=-1500.0)


def test_a_bed_pick_gap_is_not_filled_in_as_ocean():
    """The bug that put a sea-level water column inside the glacier."""
    gap = (20.0, 28.0)
    profile = _profile(bed_gap=gap)
    sec = _section(profile, gp_km=30.0)

    inside = (profile.x > gap[0] + _MAX_BRIDGE_KM) & (profile.x < gap[1])
    assert inside.any()
    assert not sec.submerged[inside].any()
    assert (sec.state[inside] == _NOBASE).all()


def test_a_gap_wider_than_the_bridge_limit_is_left_unbridged():
    x = np.arange(200) * 0.25
    y = np.where((x > 10.0) & (x < 20.0), np.nan, 100.0)
    filled, bridged = _interp_gaps(x, y, _MAX_BRIDGE_KM)

    wide = (x > 11.0) & (x < 19.0)
    assert not np.isfinite(filled[wide]).any()
    assert not bridged[wide].any()


def test_a_gap_inside_the_bridge_limit_is_bridged_and_marked():
    x = np.arange(200) * 0.25
    y = np.where((x > 10.0) & (x < 11.0), np.nan, 100.0)
    filled, bridged = _interp_gaps(x, y, _MAX_BRIDGE_KM)

    narrow = (x > 10.0) & (x < 11.0)
    assert np.isfinite(filled[narrow]).all()
    assert bridged[narrow].all()


def test_the_seabed_never_rises_through_the_ice_above_it():
    """A BedMachine bed disagreeing with the pick drew a line inside the ice."""
    profile = _profile()
    # Deliberately shallower than the ice base everywhere.
    bathymetry = np.full(profile.n, SEA_LEVEL - 50.0)
    sec = _section(profile, gp_km=30.0, bathymetry=bathymetry)

    floating = sec.ice & np.isfinite(sec.seabed)
    assert floating.any()
    assert (sec.seabed[floating] <= sec.base[floating]).all()


def test_the_seabed_never_breaks_the_sea_surface():
    profile = _profile()
    bathymetry = np.full(profile.n, SEA_LEVEL + 200.0)
    sec = _section(profile, gp_km=30.0, bathymetry=bathymetry)

    assert (sec.seabed[np.isfinite(sec.seabed)] <= SEA_LEVEL + 1e-9).all()


def test_a_bottom_pick_above_the_waterline_is_not_held_out_to_the_front():
    """
    20060608_01: the bottom pick sat on the surface across the whole shelf, and
    holding it drew 8 km of ice with its base 80 m above sea level.
    """
    profile = _profile()
    # Flat and above the waterline throughout, as it is on that shelf: there is
    # no deeper pick landward for the anchor to walk back to.
    bed = np.full(profile.n, SEA_LEVEL + 80.0)
    afloat = profile.x > 30.0
    collapsed = afloat & (profile.h_surf > SEA_LEVEL)

    held, marked = _hold_front_bottom(profile, bed, collapsed, SEA_LEVEL)

    assert not marked.any()
    assert not np.isfinite(held[afloat]).any()
    assert np.allclose(held[~afloat], bed[~afloat])


def test_a_plausible_bottom_pick_is_still_held_out_to_the_front():
    """The Helheim case the hold exists for: a real pick shoaling into the face."""
    profile = _profile()
    bed = profile.h_bed.copy()
    front = profile.x > 45.0
    bed[front] = SEA_LEVEL - 60.0           # shoaling, but still submerged
    collapsed = front & (profile.h_surf > SEA_LEVEL)

    held, marked = _hold_front_bottom(profile, bed, collapsed, SEA_LEVEL)

    assert marked.any()
    assert np.isfinite(held[marked]).all()
    assert (held[marked] < SEA_LEVEL).all()


def test_ice_is_never_drawn_with_its_base_above_sea_level():
    profile = _profile()
    afloat = profile.x > 40.0
    profile.h_bed[afloat] = profile.h_surf[afloat] - 30.0   # pick on the surface
    sec = _section(profile, gp_km=30.0)

    drawn = sec.ice & np.isfinite(sec.base)
    assert not (sec.base[drawn] > SEA_LEVEL).any()


def test_every_sample_holds_exactly_one_state():
    profile = _profile(bed_gap=(20.0, 28.0))
    sec = _section(profile, gp_km=30.0)

    assert not (sec.ice & sec.submerged & sec.unresolved).any()
    assert not (sec.ice & sec.unresolved).any()
    assert not (sec.unresolved & sec.submerged).any()


def test_grounded_ice_keeps_the_measured_bed_as_its_seabed():
    profile = _profile()
    sec = _section(profile, gp_km=30.0)

    grounded = sec.state == _GROUNDED
    assert grounded.any()
    assert np.allclose(sec.seabed[grounded], sec.base[grounded], equal_nan=True)


@pytest.mark.parametrize("gp_km", [10.0, 30.0, 50.0])
def test_the_section_survives_any_grounding_point(gp_km):
    profile = _profile(bed_gap=(20.0, 28.0))
    sec = _section(profile, gp_km=gp_km)

    assert sec.state.shape == profile.x.shape
    assert np.isfinite(sec.surf[sec.ice]).all()
    assert np.isfinite(sec.base[sec.ice]).all()

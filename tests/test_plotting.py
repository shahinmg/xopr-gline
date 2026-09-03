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
from xopr_gline.grounding.profile import (FrameSpan, GlacierProfile,
                                          ProfileSource)
from xopr_gline.grounding.radargram import (_PAD_VALUES, _anchor_shift,
                                            _regrid_columns)

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


# -- radargram ------------------------------------------------------------
def test_a_frame_gap_stays_a_gap_across_the_regrid():
    """Columns far from any trace are NaN, not the nearest trace smeared over."""
    x_src = np.r_[np.arange(0.0, 10.0, 0.3), np.arange(15.0, 20.0, 0.3)]
    power = np.tile(np.arange(len(x_src), dtype=float)[:, None], (1, 4))
    x, out = _regrid_columns(x_src, power, dx_km=0.3)

    blank = np.isnan(out[:, 0])
    assert blank.any()
    assert 10.0 <= x[blank].min() <= 11.5 and 13.5 <= x[blank].max() <= 15.0
    assert np.allclose(np.diff(x), 0.3)
    assert not np.isnan(out[x < 9.0]).any()


def test_the_radargram_is_registered_onto_the_profile_axis():
    """A different frame subset shifts the origin; the anchor takes it back."""
    anchor = ProfileSource(frames=(FrameSpan(frame=7, lo_km=100.0),
                                   FrameSpan(frame=8, lo_km=120.0)))
    spans = (FrameSpan(frame=7, lo_km=0.0), FrameSpan(frame=8, lo_km=20.0))
    assert _anchor_shift(spans, anchor) == 100.0
    assert _anchor_shift(spans, None) == 0.0


def test_frames_that_do_not_overlap_the_profile_are_refused():
    anchor = ProfileSource(frames=(FrameSpan(frame=7, lo_km=100.0),))
    with pytest.raises(ValueError, match="registered"):
        _anchor_shift((FrameSpan(frame=99, lo_km=0.0),), anchor)


def test_the_vertical_pad_values_are_not_drawn_as_echo():
    """imshow gets NaN where interpolate_to_vertical_grid padded a trace."""
    power = np.array([[-1.0, 5.0, -2.0], [3.0, -1.0, 4.0]])
    power[np.isin(power, _PAD_VALUES)] = np.nan
    assert np.isnan(power).sum() == 3
    assert np.nanmin(power) == 3.0


def _turning_profile(dx=0.25):
    """
    A seaward glacier reached across a long inland run whose residual rises.

    The whole-profile residual trend then points landward while the stretch
    being analysed runs seaward, which is 20080730_01 frames 15-16.
    """
    seaward = _profile(dx=dx, n=160)
    n_in = 480
    x_in = np.arange(n_in) * dx
    h = np.linspace(400.0, 1400.0, n_in)
    bed_msl = np.full(n_in, -300.0)
    surf_msl = bed_msl + h

    x = np.concatenate([x_in, x_in[-1] + dx + seaward.x])
    h_surf = np.concatenate([surf_msl + GEOID, seaward.h_surf])
    h_bed = np.concatenate([bed_msl + GEOID, seaward.h_bed])
    amp = np.concatenate([np.full(n_in, -110.0), seaward.amp])
    return GlacierProfile(x=x, amp=amp, h_surf=h_surf, h_bed=h_bed,
                          geoid_separation_m=GEOID), x_in[-1] + dx


def test_landward_sign_is_local_when_given_a_window():
    """One line over a segment that turns points whichever arm is longer."""
    profile, junction = _turning_profile()

    assert profile.landward_sign() == 1
    assert profile.landward_sign(window=(junction, profile.x[-1])) == -1
    assert profile.landward_sign(window=(0.0, junction)) == 1


def test_the_section_does_not_bridge_a_bottom_it_actually_picked():
    """
    The hatch that covered 7.75 km of picked bed on 20080730_01 frames 15-16.

    A sign fitted over the whole segment ran _hold_front_bottom in from the
    wrong end, so it held a front bottom inland and marked every sample it
    crossed as made up, real picks included.
    """
    profile, junction = _turning_profile()
    gp_km = junction + 30.0
    window = (junction, profile.x[-1])

    wrong = _build_section(profile, gp_km, SEA_LEVEL, None, floor=-1500.0,
                           sign=profile.landward_sign())
    right = _build_section(profile, gp_km, SEA_LEVEL, None, floor=-1500.0,
                           sign=profile.landward_sign(window=window))

    picked = np.isfinite(profile.h_bed)
    assert wrong.base_bridged[picked].sum() > 20
    assert not right.base_bridged[picked & (profile.x < gp_km)].any()


def test_the_grounded_side_follows_the_local_orientation():
    """Ocean was drawn under the inland ice when the sign came out flipped."""
    profile, junction = _turning_profile()
    gp_km = junction + 30.0
    sec = _build_section(profile, gp_km, SEA_LEVEL, None, floor=-1500.0,
                         sign=profile.landward_sign(window=(junction,
                                                           profile.x[-1])))

    assert not sec.submerged[profile.x < junction].any()

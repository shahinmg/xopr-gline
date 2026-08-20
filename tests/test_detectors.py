"""
Tests for the detectors, and mostly for the ways they can be fooled by the
edges of their own search window.
"""

import numpy as np
import pytest

from xopr_gline.grounding import (BOCPDDetector, GlacierProfile, OnsetDetector)
from xopr_gline.grounding.detectors import analysis_window
from xopr_gline.grounding.features import (FlotationResidual, _saturate,
                                           default_features)

# The shipped cutoffs are wavelengths in km tuned on ~140 km Petermann
# profiles; on a 30 km fixture a 71.6 km low-pass erases the very step under
# test. Scaled to the fixture, same job.
FEATURES = default_features(amp_cutoff_km=3.0, elev_cutoff_km=8.0)


def _shelf_then_grounded(n_float=60, n_ground=60, dx=0.25, water=0):
    """
    A profile that floats, grounds, and optionally opens into water.

    Bed power is bright over the floating section and dark over the grounded
    one, which is the contrast every detector here is looking for.
    """
    n = n_float + n_ground + water
    x = np.arange(n) * dx
    thickness = np.concatenate([
        np.full(water, 1.0),                       # open water, seaward
        np.linspace(300, 340, n_float),            # floating
        np.linspace(340, 900, n_ground),           # grounded, thickening
    ])
    # Floating ice sits at the hydrostatic surface; grounded ice rides above.
    surface = np.concatenate([
        np.full(water, 0.0),
        thickness[water:water + n_float] * 0.108,
        thickness[water + n_float:] * 0.108 + np.linspace(0, 400, n_ground),
    ])
    amp = np.concatenate([
        np.full(water, -40.0),                     # bright sea surface
        np.full(n_float, -60.0),                   # bright floating bed
        np.full(n_ground, -100.0),                 # dark grounded bed
    ])
    rng = np.random.default_rng(0)
    return GlacierProfile(
        x=x, amp=amp + rng.normal(0, 1.0, n),
        h_surf=surface, h_bed=surface - thickness,
        lat=68.0 + x / 100.0, lon=-50.0 - x / 100.0,
    )


# -- BOCPD window edges ----------------------------------------------------
def test_padding_stops_at_the_calving_front():
    """
    Past the front the "bed" is the sea surface, a far stronger changepoint
    than the grounding transition, so the pad must not reach it.
    """
    profile = _shelf_then_grounded(water=20)
    terminus = profile.terminus_crossings_km()
    assert terminus, "fixture should contain a calving front"

    lo, hi = terminus[0], terminus[0] + 8.0
    alo, ahi, padded_lo, padded_hi = analysis_window(profile, lo, hi, pad_km=4.0)
    assert alo >= terminus[0]
    assert not padded_lo             # blocked by the front
    assert padded_hi and ahi > hi    # free inland


def test_padding_is_taken_where_there_is_room():
    profile = _shelf_then_grounded()
    alo, ahi, padded_lo, padded_hi = analysis_window(
        profile, 8.0, 16.0, pad_km=3.0)
    assert (alo, ahi) == (5.0, 19.0)
    assert padded_lo and padded_hi


def test_padding_is_clipped_to_the_profile():
    profile = _shelf_then_grounded()
    lo = profile.extent[0]
    alo, _, padded_lo, _ = analysis_window(profile, lo, lo + 5.0, pad_km=3.0)
    assert alo == pytest.approx(lo)
    assert not padded_lo


def test_zero_pad_disables_padding():
    profile = _shelf_then_grounded()
    assert analysis_window(profile, 8.0, 16.0, pad_km=0.0) == (8.0, 16.0,
                                                               False, False)


def test_a_padded_window_does_not_pin_the_pick_to_its_own_trim():
    """
    The regression this whole change exists for: with the window edges trimmed
    and no padding, the maximum lands on the first surviving sample, so the
    trim picks the answer. Padded, the pick is free to sit where the data
    changes.
    """
    profile = _shelf_then_grounded()
    transition = 60 * 0.25                       # where the fixture grounds

    # A window whose seaward edge is well inside the floating section.
    window = (transition - 6.0, transition + 6.0)
    padded = BOCPDDetector(FEATURES, pad_km=4.0).detect(profile, window)
    assert not padded.extra["edge_pinned"]
    assert padded.map_km == pytest.approx(transition, abs=1.5)

    unpadded = BOCPDDetector(FEATURES, pad_km=0.0).detect(profile, window)
    assert unpadded.extra["padded_lo"] is False


def test_a_change_at_the_window_edge_is_reported_there_not_past_the_trim():
    """
    The failure the first Greenland sweep was full of. When the strongest
    change sits at or just outside the seaward limit, an unpadded run cannot
    put its maximum there, so it lands on the first sample the trim left
    standing: 7 samples in, a fixed 1.75 km at this spacing, which is what 84
    of its 99 rows did. Padded, the pick sits on the edge, which is the honest
    statement that the change is at or beyond the limit.
    """
    profile = _shelf_then_grounded()
    window = (15.5, 24.0)                        # transition at 15.0, outside

    unpadded = BOCPDDetector(FEATURES, pad_km=0.0).detect(profile, window)
    assert unpadded.extra["edge_pinned"]
    assert unpadded.extra["edge_distance_km"] == pytest.approx(
        7 * profile.dx, abs=1e-6)

    padded = BOCPDDetector(FEATURES, pad_km=4.0).detect(profile, window)
    assert not padded.extra["edge_pinned"]
    assert padded.extra["at_window_edge"]
    assert padded.map_km == pytest.approx(window[0])
    # And it is less spread out about it, not more.
    assert padded.posterior_entropy < unpadded.posterior_entropy


def test_the_reported_posterior_covers_only_the_requested_window():
    """Padding gives context, it does not widen the search."""
    profile = _shelf_then_grounded()
    window = (10.0, 18.0)
    result = BOCPDDetector(FEATURES, pad_km=4.0).detect(profile, window)
    assert result.x_km.min() >= window[0]
    assert result.x_km.max() <= window[1]
    assert window[0] <= result.map_km <= window[1]
    assert result.posterior.sum() == pytest.approx(1.0)


def test_an_unpadded_edge_pick_is_flagged():
    """A boundary the models could not see past is recorded, not hidden."""
    profile = _shelf_then_grounded()
    result = BOCPDDetector(FEATURES, pad_km=0.0).detect(profile, (12.0, 20.0))
    assert "edge_pinned" in result.extra
    assert "edge_distance_km" in result.extra


def test_model_maps_are_reported_per_likelihood():
    """The ensemble can outvote a model that was right; the spread flags it."""
    profile = _shelf_then_grounded()
    result = BOCPDDetector(FEATURES).detect(profile, (10.0, 20.0))
    assert set(result.extra["model_maps"]) == {"ifm", "fullcov"}


def test_posterior_entropy_is_low_for_a_sharp_transition():
    profile = _shelf_then_grounded()
    result = BOCPDDetector(FEATURES).detect(profile, (10.0, 20.0))
    assert 0.0 <= result.posterior_entropy < 0.5


# -- onset baseline --------------------------------------------------------
def test_the_onset_baseline_excludes_open_water():
    """
    Past the front thickness collapses, so the flotation residual reduces to
    the sea surface and floating_mask alone lets the bright water return set
    the baseline. That put the onset exactly on the terminus in 28 of the 44
    onset rows of the first Greenland sweep.
    """
    profile = _shelf_then_grounded(n_float=60, n_ground=60, water=40)
    terminus = profile.terminus_crossings_km()[0]
    transition = terminus + 60 * 0.25            # where the fixture grounds

    detector = OnsetDetector(feature=FEATURES[0], min_baseline_km=5.0,
                             baseline_km=20.0, persist_km=1.0)
    result = detector.detect(profile, (transition - 3.0, transition + 8.0))

    # The baseline must come from the floating ice near -60 dB, not the sea
    # surface near -40 dB.
    assert result.extra["baseline_dB"] == pytest.approx(-60.0, abs=3.0)
    assert result.map_km == pytest.approx(transition, abs=2.0)


def test_onset_refuses_when_there_is_no_floating_ice():
    """
    A glacier grounded to its calving front has no shelf to take a baseline
    from, and the honest answer is to decline rather than return the front.
    """
    profile = _shelf_then_grounded(n_float=4, n_ground=100, water=30)
    terminus = profile.terminus_crossings_km()[0]
    with pytest.raises(ValueError, match="floating ice"):
        OnsetDetector().detect(profile, (terminus, terminus + 10.0))


# -- features --------------------------------------------------------------
def test_the_flotation_channel_is_referenced_to_sea_level():
    """
    The channel BOCPD keys on must use the same datum as the screening, or the
    geoid ramp enters the detector as a trend.
    """
    profile = _shelf_then_grounded()
    flat = FlotationResidual(saturate_m=0.0).compute(profile)

    lifted = GlacierProfile(
        x=profile.x, amp=profile.amp,
        h_surf=profile.h_surf + 40.0, h_bed=profile.h_bed + 40.0,
        geoid_separation_m=40.0,
    )
    assert np.allclose(flat, FlotationResidual(saturate_m=0.0).compute(lifted))


def test_saturation_compresses_the_inland_ramp_but_keeps_the_sign():
    values = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    out = _saturate(values, 100.0)
    assert np.all(np.abs(out) <= 100.0)
    assert np.all(np.sign(out) == np.sign(values))
    # Small values pass through nearly untouched, which is where the grounding
    # transition lives.
    assert out[2] == 0.0
    assert abs(out[1] - values[1]) < 5.0
    assert np.all(np.diff(out) > 0)


def test_saturation_is_optional():
    values = np.array([-1000.0, 1000.0])
    assert np.array_equal(_saturate(values, 0.0), values)

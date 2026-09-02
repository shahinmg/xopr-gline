"""
Tests for the feature channels, mostly for the ways a channel can look like
signal when it is really an artefact of a gap or a filter.
"""

import numpy as np
import pytest

from xopr_gline.grounding import GlacierProfile
from xopr_gline.grounding.features import (AmplitudeRoughness, BedRoughness,
                                           FilterSpec, all_features,
                                           default_features)

# The shipped cutoffs are wavelengths tuned on ~140 km profiles; on a 40 km
# fixture they erase the step under test. Scaled to the fixture, same job.
DETREND = FilterSpec(3.0)


def _rough_then_smooth(n_each=80, dx=0.25, rough=6.0, quiet=0.5, gap=None):
    """
    A profile whose scatter, not level, changes halfway along.

    Both halves sit at the same mean bed power and on the same bed trend, so
    only a roughness channel can see the transition.
    """
    n = 2 * n_each
    x = np.arange(n) * dx
    rng = np.random.default_rng(0)
    thickness = np.concatenate([np.linspace(300, 340, n_each),
                                np.linspace(340, 900, n_each)])
    surface = np.concatenate([
        thickness[:n_each] * 0.108,
        thickness[n_each:] * 0.108 + np.linspace(0, 400, n_each)])
    amp = -60.0 + np.concatenate([rng.normal(0, rough, n_each),
                                  rng.normal(0, quiet, n_each)])
    bed = surface - thickness + np.concatenate(
        [rng.normal(0, 30.0, n_each), np.zeros(n_each)])
    if gap is not None:
        amp[gap] = np.nan
    return GlacierProfile(x=x, amp=amp, h_surf=surface, h_bed=bed)


def test_amplitude_roughness_sees_a_step_the_low_pass_hides():
    profile = _rough_then_smooth()
    half = profile.n // 2
    rough = AmplitudeRoughness(DETREND, window_km=2.0).compute(profile)
    assert np.median(rough[:half]) > 3 * np.median(rough[half:])

    # The level channel is blind to it, which is the whole point.
    level = default_features(amp_cutoff_km=3.0, elev_cutoff_km=8.0)[0]
    amp = level.compute(profile)
    assert abs(np.median(amp[:half]) - np.median(amp[half:])) < 2.0


def test_bed_roughness_sees_relief_not_slope():
    """The grounded half is steeper but smooth; only the relief should count."""
    profile = _rough_then_smooth()
    half = profile.n // 2
    rough = BedRoughness(DETREND, window_km=2.0).compute(profile)
    assert np.median(rough[:half]) > 3 * np.median(rough[half:])


def test_roughness_does_not_read_a_gap_as_smooth():
    """
    Interpolated bed power is perfectly smooth, so an unmasked gap would be the
    quietest stretch of the line and could be mistaken for the transition.
    """
    gap = slice(20, 60)                       # 10 km, inside the rough half
    profile = _rough_then_smooth(gap=gap)
    rough = AmplitudeRoughness(DETREND, window_km=2.0).compute(profile)
    half = profile.n // 2
    assert np.median(rough[gap]) > np.median(rough[half:])


def test_roughness_saturates():
    profile = _rough_then_smooth(rough=200.0)
    rough = AmplitudeRoughness(DETREND, window_km=2.0, saturate_db=6.0).compute(profile)
    assert rough.max() <= 6.0
    unsat = AmplitudeRoughness(DETREND, window_km=2.0, saturate_db=0.0).compute(profile)
    assert unsat.max() > 6.0


def test_roughness_is_finite_everywhere():
    """A channel with NaN in it kills every changepoint model downstream."""
    profile = _rough_then_smooth(gap=slice(20, 60))
    for feature in all_features():
        if feature.name.endswith("rough"):
            assert np.isfinite(feature.compute(profile)).all(), feature.name


def test_all_features_extends_the_defaults_without_changing_them():
    names = [f.name for f in all_features()]
    assert names[:4] == [f.name for f in default_features()]
    assert set(names) - set(f.name for f in default_features()) == {
        "amp_rough", "bed_rough"}


# -- unsmoothed channels ---------------------------------------------------
def test_a_zero_cutoff_disables_the_filter():
    """The zero end of a cutoff sweep: the channel is left unsmoothed."""
    profile = _rough_then_smooth()
    values = profile.h_bed_msl
    assert np.allclose(FilterSpec(0).apply(profile, values), values)
    assert np.allclose(FilterSpec(None).apply(profile, values), values)


def test_a_zero_cutoff_still_interpolates_gaps():
    """Unsmoothed still means gap-filled: no NaN reaches the detector."""
    profile = _rough_then_smooth()
    values = profile.h_bed_msl.copy()
    values[10:14] = np.nan

    out = FilterSpec(0).apply(profile, values)
    assert np.isfinite(out).all()
    # a straight line across the gap, not the measured values
    assert out[11] == pytest.approx(np.interp(11, [9, 14], [values[9], values[14]]))


# -- despiking -------------------------------------------------------------
def test_the_despiker_removes_a_spike_and_leaves_a_step():
    """A berg return is an isolated notch; the grounding step is not."""
    from xopr_gline.grounding.features import AmplitudeLevel, FilterSpec

    p = _rough_then_smooth()
    amp = np.where(p.x < p.x.mean(), -110.0, -35.0)     # the grounding step
    amp[len(amp) // 4] = -160.0                          # a berg spike
    p = GlacierProfile(x=p.x, amp=amp, h_surf=p.h_surf, h_bed=p.h_bed,
                       geoid_separation_m=0.0)

    clean = AmplitudeLevel(FilterSpec(0), despike_km=1.0).compute(p)
    assert clean[len(amp) // 4] == pytest.approx(-110.0, abs=1.0)  # spike gone
    # the step survives exactly: still two levels, still full contrast
    assert clean.min() == pytest.approx(-110.0, abs=1.0)
    assert clean.max() == pytest.approx(-35.0, abs=1.0)


def test_the_despiker_leaves_clean_data_untouched():
    """Only outliers are replaced, so a signal with none comes back unchanged."""
    from xopr_gline.grounding.features import _despike

    x = np.arange(400) * 0.25
    clean = -110.0 + 70.0 / (1 + np.exp(-(x - 50.0)))   # a smooth step, no noise
    assert np.allclose(_despike(clean, 0.25, 3.0), clean, atol=0.01)


def test_the_despiker_is_gentle_on_genuinely_rough_data():
    """Real bed roughness is signal, not spikes: rewrite a few % at most."""
    from xopr_gline.grounding.features import _despike

    rng = np.random.default_rng(0)
    rough = -110.0 + rng.normal(0, 6.0, 400)
    out = _despike(rough, 0.25, 3.0)
    assert (out != rough).mean() < 0.05


# -- the figure must show what the detector read ---------------------------
def test_the_amp_panel_labels_what_was_actually_done():
    """A figure drawn with the shipped cutoff on a tuned run misleads."""
    from xopr_gline.grounding.features import AmplitudeLevel, FilterSpec
    from xopr_gline.grounding.plotting import _amp_feature, _amp_label

    tuned = AmplitudeLevel(FilterSpec(0), despike_km=3.0)
    assert _amp_feature([tuned]) is tuned
    assert _amp_label(tuned) == "Despiked 3 km, no low-pass"
    assert _amp_label(AmplitudeLevel(FilterSpec(9.55))) == "9.55 km low-pass"
    # with no features given it falls back to the shipped default
    assert _amp_feature(None).filt.wavelength_km == 9.55

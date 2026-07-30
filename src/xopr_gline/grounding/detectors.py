"""
Grounding point detectors.

All take a GlacierProfile and return a DetectionResult, so methods can be
compared on the same data.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import scipy.misc
import scipy.special
from scipy.ndimage import uniform_filter1d

# The PyPI release of bayesian-changepoint-detection predates SciPy 1.0 and
# imports comb/logsumexp from scipy.misc, where they no longer live.
# TODO: change this soon. Use a more modern approach to handling scipy dependencies.
if not hasattr(scipy.misc, "comb"):
    scipy.misc.comb = scipy.special.comb
if not hasattr(scipy.misc, "logsumexp"):
    scipy.misc.logsumexp = scipy.special.logsumexp

from functools import partial

import bayesian_changepoint_detection.offline_changepoint_detection as offcd

from . import features as _features
from .profile import GlacierProfile
from .result import DetectionResult


class Detector(ABC):
    """Finds one grounding point in a profile."""

    name: str = "detector"

    @abstractmethod
    def detect(self, profile: GlacierProfile,
               window: Optional[tuple] = None) -> DetectionResult:
        ...

    @staticmethod
    def _resolve_window(profile: GlacierProfile, window) -> tuple:
        return window if window is not None else profile.flotation_window()


class BOCPDDetector(Detector):
    """
    Offline Bayesian changepoint detection (Fearnhead 2006).

    Runs the Gaussian, IFM and FullCov likelihoods and combines them as an
    equal-weight geometric mean. Features are used at their physical scales;
    normalising them gives the noisy amplitude channels equal weight with the
    flotation residual, which moves the answer without improving it.
    """

    name = "bocpd"

    def __init__(self, features: Sequence = _features.DEFAULT_FEATURES,
                 primary: str = "dfree", truncate: float = -50.0,
                 smooth_size: int = 3, normalise: bool = False,
                 models: Sequence[str] = ("ifm", "fullcov")):
        self.features = tuple(features)
        self.primary = primary
        self.truncate = truncate
        self.smooth_size = smooth_size
        self.normalise = normalise
        # The Gaussian model sees only the primary feature, which is already a
        # channel of IFM and FullCov, and was multimodal on Petermann. It is
        # available but not in the default ensemble.
        self.models = tuple(models)

    def detect(self, profile: GlacierProfile,
               window: Optional[tuple] = None) -> DetectionResult:
        lo, hi = self._resolve_window(profile, window)

        # Features are computed on the FULL profile, then sliced. Filtering a
        # short window instead produces edge transients that the changepoint
        # models happily mistake for the transition.
        channels = {f.name: f.compute(profile) for f in self.features}
        if self.primary not in channels:
            raise ValueError(
                f"primary feature {self.primary!r} not in "
                f"{sorted(channels)}"
            )
        if self.normalise:
            channels = {k: _standardise(v) for k, v in channels.items()}

        sel = (profile.x >= lo) & (profile.x <= hi)
        if sel.sum() < 3:
            raise ValueError(f"window [{lo}, {hi}] km selects {sel.sum()} samples")

        signal = np.column_stack([channels[f.name][sel] for f in self.features])
        observation = channels[self.primary][sel]

        x_win = profile.x[sel]
        n = len(x_win)
        prior = partial(offcd.const_prior, l=n + 1)
        x_cp = x_win[:-1]

        # Segments shorter than the channel count give a singular covariance in
        # the FullCov model, which spikes the posterior at the window edges. A
        # changepoint at the first or last sample is not a detection anyway.
        trim = max(signal.shape[1] + 2, 3)

        available = {
            "gauss": (observation, offcd.gaussian_obs_log_likelihood),
            "ifm": (signal, offcd.ifm_obs_log_likelihood),
            "fullcov": (signal, offcd.fullcov_obs_log_likelihood),
        }
        unknown = set(self.models) - set(available)
        if unknown:
            raise ValueError(f"unknown models {sorted(unknown)}")

        posteriors = {
            name: _run(available[name][0], prior, available[name][1],
                       self.truncate, trim)
            for name in self.models
        }

        combined = _geometric_mean(list(posteriors.values()))
        combined = uniform_filter1d(combined, size=self.smooth_size)
        combined = combined / combined.sum()

        return DetectionResult(
            map_km=float(x_cp[np.argmax(combined)]),
            x_km=x_cp,
            posterior=combined,
            detector=f"{self.name}{'-normalised' if self.normalise else ''}",
            source=profile.source,
            search_window=(lo, hi),
            extra={
                "model_maps": {
                    k: float(x_cp[np.argmax(v)]) for k, v in posteriors.items()
                },
                "model_posteriors": posteriors,
                "features": [f.name for f in self.features],
            },
        )


class OnsetDetector(Detector):
    """
    Where the bed echo first departs from the floating-section baseline.

    A changepoint detector targets the steepest part of the bright-to-dark
    ramp, which sits upflow of where the ice actually starts touching. This
    targets the downflow start of that ramp instead.

    The baseline comes from the profile's own floating section, identified by
    the flotation residual, so no along-track constants are needed.
    """

    name = "onset"

    def __init__(self, feature=None, k_sigma: float = 3.0,
                 persist_km: float = 2.0, float_threshold_m: float = 10.0,
                 baseline_km: float = 20.0, min_baseline_km: float = 8.0):
        self.feature = feature or _features.AmplitudeLevel()
        self.k_sigma = k_sigma
        self.persist_km = persist_km
        self.float_threshold_m = float_threshold_m
        # Bed power drifts along the shelf, so the reference is the stretch
        # immediately downflow of the window, not the whole floating section.
        self.baseline_km = baseline_km
        self.min_baseline_km = min_baseline_km

    def detect(self, profile: GlacierProfile,
               window: Optional[tuple] = None) -> DetectionResult:
        lo, hi = self._resolve_window(profile, window)
        amp_f = self.feature.compute(profile)

        # Reference stretch: afloat, and downflow of the search window so the
        # transition itself cannot contaminate the baseline.
        # "Downflow of the window" is decreasing x on a landward-running
        # profile and increasing x on a seaward-running one.
        sign = profile.landward_sign()
        if sign > 0:
            seaward = (profile.x < lo) & (profile.x >= lo - self.baseline_km)
        else:
            seaward = (profile.x > hi) & (profile.x <= hi + self.baseline_km)
        ref = (profile.floating_mask(self.float_threshold_m)
               & seaward
               & np.isfinite(amp_f))
        span = (profile.x[ref].max() - profile.x[ref].min()) if ref.any() else 0.0
        if span < self.min_baseline_km:
            raise ValueError(
                f"floating reference is {span:.1f} km, need "
                f"{self.min_baseline_km:.1f} km. The transect may not include "
                f"enough shelf downflow of the search window."
            )

        baseline = float(np.median(amp_f[ref]))
        # Noise from the high-pass residual: the filtered trend would otherwise
        # inflate sigma with the very decline being detected.
        sigma = float(np.nanstd(profile.amp[ref] - amp_f[ref]))
        threshold = baseline - self.k_sigma * sigma

        persist = max(3, int(round(self.persist_km / profile.dx)))
        below = (profile.x >= lo) & (profile.x <= hi) & (amp_f < threshold)

        # Scan landward from the seaward end of the window.
        order = np.arange(len(below)) if sign > 0 else np.arange(len(below))[::-1]

        onset_idx = None
        run = 0
        for i in order:
            run = run + 1 if below[i] else 0
            if run >= persist:
                onset_idx = i + (persist - 1) * (-1 if sign > 0 else 1)
                break
        if onset_idx is None:
            raise ValueError(
                f"bed power never drops {self.k_sigma:g} sigma below the "
                f"{baseline:.1f} dB baseline for {self.persist_km:g} km inside "
                f"the window"
            )

        return DetectionResult(
            map_km=float(profile.x[onset_idx]),
            x_km=profile.x,
            posterior=None,
            detector=self.name,
            source=profile.source,
            search_window=(lo, hi),
            extra={
                "baseline_dB": baseline,
                "sigma_dB": sigma,
                "threshold_dB": threshold,
                "baseline_span_km": float(span),
                "baseline_extent_km": (float(profile.x[ref].min()),
                                       float(profile.x[ref].max())),
            },
        )


class GradientDetector(Detector):
    """Steepest bed-power gradient in the window. Cheap reference method."""

    name = "gradient"

    def __init__(self, feature=None, trim: int = 3):
        self.feature = feature or _features.AmplitudeGradient()
        self.trim = trim

    def detect(self, profile: GlacierProfile,
               window: Optional[tuple] = None) -> DetectionResult:
        lo, hi = self._resolve_window(profile, window)
        # Same reason as BOCPDDetector: filter on the full profile, then slice.
        dA = self.feature.compute(profile)
        sel = (profile.x >= lo) & (profile.x <= hi)
        dA, x_win = dA[sel].copy(), profile.x[sel]
        if not np.any(np.isfinite(dA)):
            raise ValueError("gradient is all NaN in the search window")
        # Without this the steepest gradient is routinely the window boundary.
        if self.trim and len(dA) > 2 * self.trim:
            dA[:self.trim] = np.nan
            dA[-self.trim:] = np.nan
        idx = int(np.nanargmax(np.abs(dA)))
        return DetectionResult(
            map_km=float(x_win[idx]),
            x_km=x_win,
            posterior=None,
            detector=self.name,
            source=profile.source,
            search_window=(lo, hi),
            extra={"peak_gradient": float(dA[idx])},
        )


# -- helpers --------------------------------------------------------------
def _run(data, prior, likelihood, truncate, trim=0):
    _, _, pcp = offcd.offline_changepoint_detection(
        data, prior, likelihood, truncate=truncate
    )
    cp = np.exp(pcp).sum(0)
    if trim and len(cp) > 2 * trim:
        cp[:trim] = 0.0
        cp[-trim:] = 0.0
    total = cp.sum()
    if total <= 0:
        raise ValueError("posterior is empty after trimming the window edges")
    return cp / total


def _geometric_mean(posteriors):
    log_mean = sum(np.log(p + 1e-15) for p in posteriors) / len(posteriors)
    log_mean -= np.logaddexp.reduce(log_mean)
    return np.exp(log_mean)


def _standardise(values):
    return (values - np.mean(values)) / (np.std(values) + 1e-9)

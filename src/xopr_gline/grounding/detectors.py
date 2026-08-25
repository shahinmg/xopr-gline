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

    The models see a padded window and report inside the requested one, so the
    edge transient every likelihood produces at the start and end of its data
    falls in the padding instead of becoming the answer. Where the padding is
    blocked, by the profile running out or by the calving front, that side is
    trimmed as before and the result is flagged edge_pinned.
    """

    name = "bocpd"

    def __init__(self, features: Optional[Sequence] = None,
                 primary: str = "dfree", truncate: float = -50.0,
                 smooth_size: int = 3, normalise: bool = False,
                 models: Sequence[str] = ("ifm", "fullcov"),
                 pad_km: float = 4.0, edge_samples: int = 2,
                 min_thickness_m: float = 25.0):
        self.features = tuple(features if features is not None
                              else _features.DEFAULT_FEATURES)
        self.primary = primary
        self.truncate = truncate
        self.smooth_size = smooth_size
        self.normalise = normalise
        # The Gaussian model sees only the primary feature, which is already a
        # channel of IFM and FullCov, and was multimodal on Petermann. It is
        # available but not in the default ensemble.
        self.models = tuple(models)
        self.pad_km = pad_km
        self.edge_samples = edge_samples
        self.min_thickness_m = min_thickness_m

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

        alo, ahi, padded_lo, padded_hi = analysis_window(
            profile, lo, hi, self.pad_km, self.min_thickness_m)
        sel = (profile.x >= alo) & (profile.x <= ahi)
        if sel.sum() < 3:
            raise ValueError(f"window [{lo}, {hi}] km selects {sel.sum()} samples")

        signal = np.column_stack([channels[f.name][sel] for f in self.features])
        observation = channels[self.primary][sel]

        x_win = profile.x[sel]
        n = len(x_win)
        prior = partial(offcd.const_prior, l=n + 1)
        x_cp = x_win[:-1]

        available = {
            "gauss": (observation, offcd.gaussian_obs_log_likelihood, 1),
            "ifm": (signal, offcd.ifm_obs_log_likelihood, 1),
            "fullcov": (signal, offcd.fullcov_obs_log_likelihood,
                        signal.shape[1]),
        }
        unknown = set(self.models) - set(available)
        if unknown:
            raise ValueError(f"unknown models {sorted(unknown)}")

        # Only a genuinely artificial boundary is trimmed. A padded side has
        # real data beyond it, so its transient sits outside the reported range
        # and nothing needs discarding.
        posteriors, trims = {}, []
        for name in self.models:
            data, likelihood, width = available[name]
            trim = max(width + 2, 3)
            trims.append(trim)
            posteriors[name] = _run(data, prior, likelihood, self.truncate,
                                    trim_lo=0 if padded_lo else trim,
                                    trim_hi=0 if padded_hi else trim)
        # A zero in any model zeroes the geometric mean, so the widest trim in
        # the ensemble is the one that could have displaced the pick.
        trim = max(trims)

        # A model whose peak sits in the padding is saying there is no
        # changepoint inside the reported window. Renormalising it into that
        # range piles its mass on the boundary, and the geometric mean then
        # drags the combined pick there: on 20170331_01 frames 16-18, ifm peaks
        # at 72.01 km in the pad and reports 73.51 km, the window edge, pulling
        # the answer 2.5 km seaward of fullcov's 78.51 km. Such a model is kept
        # out of the combination unless every model is like that, in which case
        # there is nothing better to fall back on.
        peaks = {k: float(x_cp[np.argmax(v)]) for k, v in posteriors.items()}
        voting, dropped = _select_voting(peaks, self.models, lo, hi,
                                         self.pad_km)

        combined = _geometric_mean([posteriors[k] for k in voting])
        combined = uniform_filter1d(combined, size=self.smooth_size)

        # Report inside the requested window only: the padding exists to give
        # the models context, not to widen the search.
        report = (x_cp >= lo) & (x_cp <= hi)
        if not report.any():
            raise ValueError(f"window [{lo}, {hi}] km holds no changepoint "
                             f"positions")
        x_report = x_cp[report]
        posterior = combined[report]
        total = posterior.sum()
        if total <= 0:
            raise ValueError("posterior is empty inside the reported window")
        posterior = posterior / total
        map_km = float(x_report[np.argmax(posterior)])

        return DetectionResult.at(
            profile, map_km,
            x_km=x_report,
            posterior=posterior,
            detector=f"{self.name}{'-normalised' if self.normalise else ''}",
            search_window=(lo, hi),
            extra={
                # Reported inside the window like the combined pick, so the
                # spread between models is a disagreement about the answer and
                # not one model pointing into the padding.
                "model_maps": {
                    k: float(x_report[np.argmax(v[report])])
                    for k, v in posteriors.items()
                },
                "model_maps_analysis": {
                    k: float(x_cp[np.argmax(v)]) for k, v in posteriors.items()
                },
                # Sliced to the reported range so every array in here shares
                # x_km, which is what the plotting assumes.
                "model_posteriors": {k: _renormalise(v[report])
                                     for k, v in posteriors.items()},
                # Which models the combined posterior is actually made of, and
                # which were left out for peaking in the padding.
                "models_voting": tuple(voting),
                "models_dropped": tuple(dropped),
                "features": [f.name for f in self.features],
                "analysis_window": (float(alo), float(ahi)),
                "padded_lo": bool(padded_lo),
                "padded_hi": bool(padded_hi),
                **_edge_flags(map_km, lo, hi, profile.dx, self.edge_samples,
                              padded_lo, padded_hi, trim),
            },
        )


def analysis_window(profile: GlacierProfile, lo: float, hi: float,
                    pad_km: float, min_thickness_m: float = 25.0) -> tuple:
    """
    Widen [lo, hi] by pad_km where there is data to widen into.

    Returns (lo, hi, padded_lo, padded_hi). The pad stops at the profile ends
    and at a calving front, since past the front the "bed" is the sea surface
    and the ice/water contrast is a far stronger changepoint than the grounding
    transition it would drown out.
    """
    if not pad_km or pad_km <= 0:
        return float(lo), float(hi), False, False

    x0, x1 = profile.extent
    limit_lo, limit_hi = x0, x1
    for terminus in profile.terminus_crossings_km(min_thickness_m):
        if terminus <= lo:
            limit_lo = max(limit_lo, terminus)
        if terminus >= hi:
            limit_hi = min(limit_hi, terminus)

    alo = max(limit_lo, lo - pad_km)
    ahi = min(limit_hi, hi + pad_km)
    # A pad of less than one sample is no pad at all.
    return (alo, ahi,
            bool(lo - alo >= profile.dx),
            bool(ahi - hi >= profile.dx))


def _select_voting(peaks, models, lo, hi, pad_km) -> tuple:
    """
    Which models the combined posterior should be built from.

    A model whose peak sits in the padding is saying there is no changepoint
    inside the reported window; renormalising it into that range piles its mass
    on the boundary and drags the combined pick there. On 20170331_01 frames
    16-18, ifm peaks at 72.01 km in the pad and reports 73.51 km, the window
    edge, pulling the answer 2.5 km seaward of fullcov's 78.51 km.

    It is only dropped where the models are not describing the same change:
    peaks further apart than the context they were given cannot be one feature
    seen twice. A peak just outside the limit and one just inside agree that
    the change sits near the boundary, and there reporting the edge stays the
    honest answer.

    Returns (voting, dropped).
    """
    inside = [k for k in models if lo <= peaks[k] <= hi]
    outside = [k for k in models if k not in inside]
    if not inside or not outside:
        return list(models), []
    spread = max(abs(peaks[a] - peaks[b]) for a in inside for b in outside)
    return (inside, outside) if spread > pad_km else (list(models), [])


def _edge_flags(map_km, lo, hi, dx, edge_samples, padded_lo, padded_hi,
                trim) -> dict:
    """
    Whether the pick is sitting on a window boundary rather than on a change.

    edge_pinned is the serious one: the boundary was artificial and the trim,
    not the data, decided where the maximum could fall. at_window_edge is
    weaker and says only that the change is at or beyond the search limit.
    """
    reach = (trim + edge_samples) * dx
    pinned_lo = (not padded_lo) and (map_km - lo) <= reach
    pinned_hi = (not padded_hi) and (hi - map_km) <= reach
    pinned = pinned_lo or pinned_hi
    return {
        **_window_edge(map_km, lo, hi, dx, edge_samples),
        "edge_pinned": bool(pinned),
        # Which boundary, and its position, so a caller can ask what that
        # boundary was and whether the answer really belongs on it.
        "pinned_edge": "lo" if pinned_lo else ("hi" if pinned_hi else None),
        "pinned_edge_km": float(lo) if pinned_lo else (
            float(hi) if pinned_hi else np.nan),
        # How far the trim could have displaced the pick.
        "edge_reach_km": float(reach) if pinned else edge_samples * dx,
    }


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
                 baseline_km: float = 20.0, min_baseline_km: float = 8.0,
                 min_thickness_m: float = 25.0):
        self.feature = feature or _features.AmplitudeLevel()
        self.k_sigma = k_sigma
        self.persist_km = persist_km
        self.float_threshold_m = float_threshold_m
        # Bed power drifts along the shelf, so the reference is the stretch
        # immediately downflow of the window, not the whole floating section.
        self.baseline_km = baseline_km
        self.min_baseline_km = min_baseline_km
        self.min_thickness_m = min_thickness_m

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
        # ice_mask is what keeps the sea out of the baseline. Past the calving
        # front thickness collapses to ~0, so the flotation residual reduces to
        # the water surface and floating_mask is True over open water; the
        # bright sea-surface return then sets a baseline tens of dB too high
        # and the first ice sample clears the threshold on its own, putting the
        # onset exactly at the terminus.
        ref = (profile.floating_mask(self.float_threshold_m)
               & profile.ice_mask(self.min_thickness_m)
               & seaward
               & np.isfinite(amp_f))
        span = (profile.x[ref].max() - profile.x[ref].min()) if ref.any() else 0.0
        if span < self.min_baseline_km:
            raise ValueError(
                f"floating reference is {span:.1f} km of floating ice, need "
                f"{self.min_baseline_km:.1f} km. The transect may not include "
                f"enough shelf downflow of the search window, or the glacier "
                f"may be grounded to its calving front with no shelf at all."
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

        return DetectionResult.at(
            profile, float(profile.x[onset_idx]),
            x_km=profile.x,
            posterior=None,
            detector=self.name,
            search_window=(lo, hi),
            extra={
                "baseline_dB": baseline,
                "sigma_dB": sigma,
                "threshold_dB": threshold,
                "baseline_span_km": float(span),
                "baseline_extent_km": (float(profile.x[ref].min()),
                                       float(profile.x[ref].max())),
                **_window_edge(float(profile.x[onset_idx]), lo, hi, profile.dx),
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
        return DetectionResult.at(
            profile, float(x_win[idx]),
            x_km=x_win,
            posterior=None,
            detector=self.name,
            search_window=(lo, hi),
            extra={"peak_gradient": float(dA[idx]),
                   **_window_edge(float(x_win[idx]), lo, hi, profile.dx)},
        )


# -- helpers --------------------------------------------------------------
def _window_edge(map_km, lo, hi, dx, edge_samples: int = 2) -> dict:
    """How close a point estimate sits to the limits it was allowed to search."""
    near = edge_samples * dx
    return {
        "at_window_edge": bool((map_km - lo) <= near or (hi - map_km) <= near),
        "edge_distance_km": float(min(map_km - lo, hi - map_km)),
        "edge_reach_km": float(near),
    }


def _run(data, prior, likelihood, truncate, trim_lo=0, trim_hi=0):
    """
    One model's changepoint posterior, with artificial boundaries trimmed.

    Trimming is per side: a padded side keeps its full posterior, because the
    transient there falls outside the range the caller reports on.
    """
    _, _, pcp = offcd.offline_changepoint_detection(
        data, prior, likelihood, truncate=truncate
    )
    cp = np.exp(pcp).sum(0)
    if len(cp) > trim_lo + trim_hi:
        if trim_lo:
            cp[:trim_lo] = 0.0
        if trim_hi:
            cp[-trim_hi:] = 0.0
    total = cp.sum()
    if total <= 0:
        raise ValueError("posterior is empty after trimming the window edges")
    return cp / total


def _renormalise(values):
    """Rescale a posterior slice to sum to one, or leave it if it is all zero."""
    total = values.sum()
    return values / total if total > 0 else values


def _geometric_mean(posteriors):
    log_mean = sum(np.log(p + 1e-15) for p in posteriors) / len(posteriors)
    log_mean -= np.logaddexp.reduce(log_mean)
    return np.exp(log_mean)


def _standardise(values):
    return (values - np.mean(values)) / (np.std(values) + 1e-9)

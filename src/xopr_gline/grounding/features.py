"""
Features computed from a GlacierProfile

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .profile import GlacierProfile


@dataclass(frozen=True)
class FilterSpec:
    """
    Butterworth low-pass defined by wavelength in km.

    A wavelength of 0 or None disables the filter, so a channel can be run
    unsmoothed as the zero end of a cutoff sweep.
    """

    wavelength_km: Optional[float]
    order: int = 5

    def apply(self, profile: GlacierProfile, values: np.ndarray) -> np.ndarray:
        if not self.wavelength_km:
            return _fill(values)
        wn = profile.cutoff_to_wn(self.wavelength_km)
        b, a = butter(self.order, wn, btype="low", analog=False)
        filled = _fill(values)
        padlen = 3 * max(len(a), len(b))
        if len(filled) <= padlen:
            raise ValueError(
                f"profile has {len(filled)} samples, need > {padlen} to filter "
                f"at order {self.order}"
            )
        return filtfilt(b, a, filled)


def _fill(values: np.ndarray) -> np.ndarray:
    """Interpolate over NaN so the filter has something to chew on."""
    return pd.Series(values).interpolate().ffill().bfill().values


class Feature(ABC):
    """One channel fed to a detector."""

    name: str = "feature"

    @abstractmethod
    def compute(self, profile: GlacierProfile) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# The CSV-era script specified cutoffs as Butterworth Wn at dx = 0.716 km:
# AMP_WN 0.15 -> 9.55 km and SLOPE_WN 0.02 -> 71.6 km. Expressed as physical
# wavelengths they carry over to any sample spacing. They are defaults, not
# constants: 71.6 km is a long smoother on a 140 km profile, so sweep them
# through default_features() rather than treating them as settled.
AMP_CUTOFF_KM = 9.55
ELEV_CUTOFF_KM = 71.6

# Flotation residual beyond this many metres from zero is compressed. See
# _saturate.
SATURATE_M = 100.0

# Hampel threshold for the bed-power despiker. 5 rather than the textbook 3:
# measured on 20170422_01, a 1 km window at 3 sigma rewrites 15% of samples
# because a short-window MAD underestimates real bed roughness, while a 3 km
# window at 5 sigma rewrites 1.7% and still removes a 45 dB berg notch.
N_SIGMA = 5.0
DESPIKE_WINDOW_KM = 3.0


class AmplitudeLevel(Feature):
    """
    Bed echo power, dB, optionally despiked and low-pass filtered.

    `despike_km` runs a Hampel filter first, which removes isolated outliers —
    a berg return is a 50 dB single-sample notch — while leaving the step at
    the grounding line and the peak past it exactly as measured. The IFM and
    FullCov likelihoods sum squares about zero, so one such spike carries far
    more weight than it would in a mean-shift model.
    """

    name = "amp"

    def __init__(self, filt: FilterSpec = FilterSpec(AMP_CUTOFF_KM),
                 despike_km: float = 0.0, n_sigma: float = N_SIGMA):
        self.filt = filt
        self.despike_km = despike_km
        self.n_sigma = n_sigma

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        amp = profile.amp
        if self.despike_km:
            amp = _despike(amp, profile.dx, self.despike_km, self.n_sigma)
        return self.filt.apply(profile, amp)


class AmplitudeGradient(Feature):
    """
    d(amplitude)/dx, dB/km.

    Regions with no bed pick are masked from the data itself via
    profile.nan_blocks()
    """

    name = "dA"

    def __init__(self, filt: FilterSpec = FilterSpec(AMP_CUTOFF_KM),
                 max_abs: float = 5.0,
                 min_gap_km: float = 0.3):
        self.filt = filt
        self.max_abs = max_abs
        self.min_gap_km = min_gap_km

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        amp_f = self.filt.apply(profile, profile.amp)
        dA = np.gradient(amp_f, profile.x)

        for lo, hi in profile.nan_blocks(self.min_gap_km):
            dA[(profile.x >= lo) & (profile.x <= hi)] = np.nan
        dA[np.abs(dA) > self.max_abs] = np.nan

        # Interpolate across masked stretches rather than forward-filling them
        # to a constant, which would look like a zero-variance block.
        return pd.Series(dA).interpolate(limit_direction="both").values


class FlotationResidual(Feature):
    """
    h_surf_msl - H*(1 - rho_ice/rho_sw), m. Zero at hydrostatic flotation.

    Referenced to sea level like GlacierProfile.flotation_residual: the
    ellipsoidal surface carries a geoid ramp of tens of metres along track,
    which the detector would otherwise read as a trend in its primary channel.
    """

    name = "dfree"

    def __init__(self, filt: FilterSpec = FilterSpec(ELEV_CUTOFF_KM),
                 saturate_m: float = SATURATE_M):
        self.filt = filt
        self.saturate_m = saturate_m

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        h_surf = self.filt.apply(profile, profile.h_surf_msl)
        h_bed = self.filt.apply(profile, profile.h_bed_msl)
        thickness = h_surf - h_bed
        residual = h_surf - thickness * profile.constants.flotation_factor
        return _saturate(residual, self.saturate_m)


def _saturate(values: np.ndarray, scale: float) -> np.ndarray:
    """
    Compress values far from zero so the inland ramp cannot dominate.

    The flotation residual climbs past 1000 m inland while the grounding
    transition lives within tens of metres of zero; unsaturated, that ramp sets
    the covariance and pulls the changepoint onto bed topography. tanh rather
    than a clip, which leaves zero-variance blocks the covariance models choke
    on. Pass 0 or None to disable.
    """
    if not scale or not np.isfinite(scale):
        return values
    return scale * np.tanh(values / scale)


def _rolling_scatter(values: np.ndarray, dx_km: float,
                     window_km: float) -> np.ndarray:
    """Robust sigma (1.4826 * MAD) in a centred rolling window."""
    n = max(3, int(round(window_km / dx_km)) | 1)
    s = pd.Series(values)
    med = s.rolling(n, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(n, center=True, min_periods=3).median()
    return np.array(1.4826 * mad.interpolate(limit_direction="both").values,
                    dtype=float)


def _despike(values: np.ndarray, dx_km: float, window_km: float,
             n_sigma: float = N_SIGMA) -> np.ndarray:
    """
    Hampel filter: replace only samples beyond n_sigma robust sigma of the
    local median, leaving every other sample exactly as measured.

    A low-pass cannot do this — it trades the spike for a smeared step. NaN is
    left as NaN for the gap handling downstream.
    """
    n = max(3, int(round(window_km / dx_km)) | 1)
    med = pd.Series(values).rolling(n, center=True, min_periods=1).median()
    sigma = _rolling_scatter(values, dx_km, window_km)
    out = np.array(values, dtype=float, copy=True)
    spike = (np.abs(values - med.values) > n_sigma * np.maximum(sigma, 1e-9))
    spike &= np.isfinite(values)
    out[spike] = med.values[spike]
    return out


def _mask_gaps(profile: GlacierProfile, values: np.ndarray,
               min_gap_km: float) -> np.ndarray:
    """
    Blank stretches with no bed pick, then interpolate across them.

    Interpolated bed power and bed elevation are smooth by construction, so a
    gap would otherwise read as the smoothest — most floating-looking — part of
    the line.
    """
    for lo, hi in profile.nan_blocks(min_gap_km):
        values[(profile.x >= lo) & (profile.x <= hi)] = np.nan
    return pd.Series(values).interpolate(limit_direction="both").values


# Rolling window for the roughness channels, and the scales beyond which they
# saturate. Bed power scatter runs a few dB, bed relief a few tens of metres.
ROUGH_WINDOW_KM = 3.0
AMP_ROUGH_SATURATE_DB = 6.0
BED_ROUGH_SATURATE_M = 50.0


class AmplitudeRoughness(Feature):
    """
    Rolling scatter of bed power about its low-pass, dB.

    Some transitions are clutter-to-clean: the level barely moves but the
    variability collapses, which the amplitude low-pass removes by
    construction. This is the only channel that sees it.
    """

    name = "amp_rough"

    def __init__(self, filt: FilterSpec = FilterSpec(AMP_CUTOFF_KM),
                 window_km: float = ROUGH_WINDOW_KM,
                 saturate_db: float = AMP_ROUGH_SATURATE_DB,
                 min_gap_km: float = 0.3):
        self.filt = filt
        self.window_km = window_km
        self.saturate_db = saturate_db
        self.min_gap_km = min_gap_km

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        residual = _fill(profile.amp) - self.filt.apply(profile, profile.amp)
        rough = _rolling_scatter(residual, profile.dx, self.window_km)
        return _saturate(_mask_gaps(profile, rough, self.min_gap_km),
                         self.saturate_db)


class BedRoughness(Feature):
    """
    Rolling scatter of bed elevation about its low-pass, m.

    A floating tongue's bed is the cavity ceiling and is smooth; grounded bed
    is rough. Derived from the elevations rather than the power, so it fails
    differently from the amplitude channels.
    """

    name = "bed_rough"

    def __init__(self, filt: FilterSpec = FilterSpec(AMP_CUTOFF_KM),
                 window_km: float = ROUGH_WINDOW_KM,
                 saturate_m: float = BED_ROUGH_SATURATE_M,
                 min_gap_km: float = 0.3):
        self.filt = filt
        self.window_km = window_km
        self.saturate_m = saturate_m
        self.min_gap_km = min_gap_km

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        h_bed = profile.h_bed_msl
        residual = _fill(h_bed) - self.filt.apply(profile, h_bed)
        rough = _rolling_scatter(residual, profile.dx, self.window_km)
        rough[~np.isfinite(h_bed)] = np.nan
        return _saturate(_mask_gaps(profile, rough, self.min_gap_km),
                         self.saturate_m)


class SurfaceSlope(Feature):
    """d(h_surf)/dx, m/km."""

    name = "dsurf"

    def __init__(self, filt: FilterSpec = FilterSpec(ELEV_CUTOFF_KM)):
        self.filt = filt

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        return np.gradient(self.filt.apply(profile, profile.h_surf), profile.x)


def default_features(amp_cutoff_km: float = AMP_CUTOFF_KM,
                     elev_cutoff_km: float = ELEV_CUTOFF_KM,
                     saturate_m: float = SATURATE_M,
                     da_cutoff_km: Optional[float] = None,
                     despike_km: float = 0.0) -> tuple:
    """
    The default channel set with its filter cutoffs exposed for tuning.

    `da_cutoff_km` filters the gradient channel separately, defaulting to the
    amplitude one. They need to differ to run `amp` raw: a gradient of
    unsmoothed dB is noise, whatever the level channel wants.
    """
    amp = FilterSpec(amp_cutoff_km)
    grad = FilterSpec(amp_cutoff_km if da_cutoff_km is None else da_cutoff_km)
    elev = FilterSpec(elev_cutoff_km)
    return (
        AmplitudeLevel(amp, despike_km=despike_km),
        AmplitudeGradient(grad),
        FlotationResidual(elev, saturate_m=saturate_m),
        SurfaceSlope(elev),
    )


def optional_features(amp_cutoff_km: float = AMP_CUTOFF_KM,
                      rough_window_km: float = ROUGH_WINDOW_KM) -> tuple:
    """Channels available by name but not in the default set."""
    amp = FilterSpec(amp_cutoff_km)
    return (
        AmplitudeRoughness(amp, window_km=rough_window_km),
        BedRoughness(amp, window_km=rough_window_km),
    )


def all_features(amp_cutoff_km: float = AMP_CUTOFF_KM,
                 elev_cutoff_km: float = ELEV_CUTOFF_KM,
                 saturate_m: float = SATURATE_M,
                 rough_window_km: float = ROUGH_WINDOW_KM,
                 da_cutoff_km: Optional[float] = None,
                 despike_km: float = 0.0) -> tuple:
    """Every channel that can be selected by name, defaults first."""
    return (default_features(amp_cutoff_km, elev_cutoff_km, saturate_m,
                             da_cutoff_km, despike_km)
            + optional_features(amp_cutoff_km, rough_window_km))


DEFAULT_FEATURES = default_features()


def stack(profile: GlacierProfile, features=DEFAULT_FEATURES) -> np.ndarray:
    """(n, n_features) array in the given order."""
    return np.column_stack([f.compute(profile) for f in features])

"""
Features computed from a GlacierProfile
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .profile import GlacierProfile


@dataclass(frozen=True)
class FilterSpec:
    """Butterworth low-pass defined by wavelength in km."""

    wavelength_km: float
    order: int = 5

    def apply(self, profile: GlacierProfile, values: np.ndarray) -> np.ndarray:
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
# wavelengths they carry over to any sample spacing.
AMP_CUTOFF_KM = 9.55
ELEV_CUTOFF_KM = 71.6


class AmplitudeLevel(Feature):
    """Low-pass filtered bed echo power, dB."""

    name = "amp"

    def __init__(self, filt: FilterSpec = FilterSpec(AMP_CUTOFF_KM)):
        self.filt = filt

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        return self.filt.apply(profile, profile.amp)


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
    """h_surf - H*(1 - rho_ice/rho_sw), m. Zero at hydrostatic flotation."""

    name = "dfree"

    def __init__(self, filt: FilterSpec = FilterSpec(ELEV_CUTOFF_KM)):
        self.filt = filt

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        h_surf = self.filt.apply(profile, profile.h_surf)
        h_bed = self.filt.apply(profile, profile.h_bed)
        thickness = h_surf - h_bed
        return h_surf - thickness * profile.constants.flotation_factor


class SurfaceSlope(Feature):
    """d(h_surf)/dx, m/km."""

    name = "dsurf"

    def __init__(self, filt: FilterSpec = FilterSpec(ELEV_CUTOFF_KM)):
        self.filt = filt

    def compute(self, profile: GlacierProfile) -> np.ndarray:
        return np.gradient(self.filt.apply(profile, profile.h_surf), profile.x)


DEFAULT_FEATURES = (
    AmplitudeLevel(),
    AmplitudeGradient(),
    FlotationResidual(),
    SurfaceSlope(),
)


def stack(profile: GlacierProfile, features=DEFAULT_FEATURES) -> np.ndarray:
    """(n, n_features) array in the given order."""
    return np.column_stack([f.compute(profile) for f in features])

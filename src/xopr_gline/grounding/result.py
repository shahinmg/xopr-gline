"""
Detection results, with enough provenance to reproduce them.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .profile import ProfileSource


@dataclass
class DetectionResult:
    """
    Where a detector put the grounding point.

    Parameters
    ----------
    map_km: float
        Maximum a posteriori grounding point, km along-track.

    x_km: np.ndarray
        Positions the posterior is defined on.

    posterior: np.ndarray or None
        Normalised P(grounding point = x). None for detectors that return a
        point estimate only.

    detector: str
        Name of the method that produced this.

    source: ProfileSource or None
        Provenance of the profile it ran on.

    search_window: tuple
        (lo_km, hi_km) the detector was allowed to look in.

    extra: dict
        Per-detector diagnostics, e.g. individual model MAPs.
    """

    map_km: float
    x_km: np.ndarray
    posterior: Optional[np.ndarray] = None
    detector: str = "unknown"
    source: Optional[ProfileSource] = None
    search_window: Optional[tuple] = None
    extra: dict = field(default_factory=dict)

    def credible_interval(self, mass: float = 0.68) -> tuple:
        """Central interval containing `mass` of the posterior."""
        if self.posterior is None:
            return (self.map_km, self.map_km)
        cdf = np.cumsum(self.posterior)
        cdf = cdf / cdf[-1]
        tail = (1.0 - mass) / 2.0
        n = len(self.x_km)
        lo = int(np.searchsorted(cdf, tail))
        hi = int(np.searchsorted(cdf, 1.0 - tail))
        return (float(self.x_km[min(lo, n - 1)]),
                float(self.x_km[min(hi, n - 1)]))

    def summary(self) -> str:
        lo68, hi68 = self.credible_interval(0.68)
        lo95, hi95 = self.credible_interval(0.95)
        return (f"{self.detector}: MAP={self.map_km:.2f} km  "
                f"68%=[{lo68:.1f}, {hi68:.1f}]  95%=[{lo95:.1f}, {hi95:.1f}]")

    def __str__(self) -> str:
        return self.summary()


def transition_width_km(onset: DetectionResult,
                        changepoint: DetectionResult,
                        landward_sign: int = 1) -> float:
    """
    Distance from the onset of the bed-echo transition to its steepest point.

    Tidal flexure smears the bright-to-dark transition across the grounding
    zone, so this is a width, not an error: narrow means a sharply pinned
    grounding line, wide means a diffuse zone and low confidence in any single
    point estimate.

    landward_sign flips the convention for transects whose along-track runs
    seaward, where the onset sits at higher x than the changepoint. A negative
    result means the onset was found upflow of the changepoint, which should not
    happen and indicates one of the two is unreliable.
    """
    return float(landward_sign * (changepoint.map_km - onset.map_km))

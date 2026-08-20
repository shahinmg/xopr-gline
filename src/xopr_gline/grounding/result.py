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

    lat, lon: float
        The grounding point itself, interpolated on the profile track. NaN for
        profiles carrying no coordinates.
    """

    map_km: float
    x_km: np.ndarray
    posterior: Optional[np.ndarray] = None
    detector: str = "unknown"
    source: Optional[ProfileSource] = None
    search_window: Optional[tuple] = None
    extra: dict = field(default_factory=dict)
    lat: float = np.nan
    lon: float = np.nan

    @classmethod
    def at(cls, profile, map_km: float, **kwargs) -> "DetectionResult":
        """
        Build a result at a position on `profile`, filling in its provenance.

        Detectors should use this rather than the constructor so the point
        geometry cannot drift out of step with map_km.
        """
        lat, lon = profile.position_at(map_km)
        return cls(map_km=float(map_km), source=profile.source,
                   lat=lat, lon=lon, **kwargs)

    @property
    def frame(self):
        """The FrameSpan the grounding point falls in, or None."""
        return None if self.source is None else self.source.frame_at_km(self.map_km)

    @property
    def frames_used(self) -> tuple:
        """Frames overlapping the search window."""
        if self.source is None or self.search_window is None:
            return ()
        return self.source.frames_between(*self.search_window)

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

    @property
    def posterior_entropy(self) -> float:
        """
        Shannon entropy of the posterior over its own support, 0 to 1.

        0 is a delta at one sample and 1 is a flat posterior, so a very low
        value on a wide window means the model is certain, and often certain
        about a boundary artifact. NaN for point-estimate detectors.
        """
        if self.posterior is None or len(self.posterior) < 2:
            return np.nan
        p = np.asarray(self.posterior, dtype=float)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(len(self.posterior)))

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

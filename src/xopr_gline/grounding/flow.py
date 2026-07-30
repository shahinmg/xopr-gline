"""
Flight-track alignment with ice flow, from the ITS_LIVE velocity mosaic.

BOCPD assumes the profile crosses the grounding zone along flow. A cross-flow
or gridded (squiggly) transect samples the transition obliquely or repeatedly,
so its along-track coordinate is not a flowline distance and the detected
changepoint is not a grounding point. These functions score a track before it
is fed to a detector.
"""

from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from scipy.ndimage import uniform_filter1d

from ..geospatial import open_itslive_velocity
from .profile import GlacierProfile
from .profile import _mask_runs as _runs

FLOW_CRS = "EPSG:3413"


def _project(lat, lon) -> tuple:
    """lat/lon degrees to FLOW_CRS metres."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", FLOW_CRS,
                                              always_xy=True)
    return transformer.transform(np.asarray(lon, dtype=float),
                                 np.asarray(lat, dtype=float))


def sample_velocity(lat, lon, directory: Union[str, Path, None] = None,
                    buffer_m: float = 5_000.0) -> tuple:
    """
    Nearest-neighbour vx, vy in m/yr at each lat/lon.
    """
    px, py = _project(lat, lon)
    if not (np.isfinite(px).any() and np.isfinite(py).any()):
        raise ValueError("no finite coordinates to sample")

    box = (np.nanmin(px) - buffer_m, np.nanmin(py) - buffer_m,
           np.nanmax(px) + buffer_m, np.nanmax(py) + buffer_m)

    sampled = []
    for component in ("vx", "vy"):
        grid = open_itslive_velocity(component, directory=directory)
        window = grid.rio.clip_box(*box)
        point = window.sel(
            x=xr.DataArray(px, dims="point"),
            y=xr.DataArray(py, dims="point"),
            method="nearest",
        )
        sampled.append(point.values.astype(float))
    return sampled[0], sampled[1]


def track_heading(lat, lon, smooth: int = 5) -> tuple:
    """
    Unit tangent to the track in FLOW_CRS, as (ex, ey).

    Coordinates are smoothed over `smooth` samples before differencing; raw
    per-sample GPS jitter otherwise dominates the heading on a straight leg.
    """
    px, py = _project(lat, lon)
    if smooth > 1:
        px = uniform_filter1d(px, size=smooth, mode="nearest")
        py = uniform_filter1d(py, size=smooth, mode="nearest")

    ex = np.gradient(px)
    ey = np.gradient(py)
    norm = np.hypot(ex, ey)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ex / norm, ey / norm


def signed_cos(lat, lon, directory=None, smooth: int = 5,
               min_speed_m_yr: float = 50.0) -> np.ndarray:
    """
    Cosine between the track tangent and the flow direction, -1 to +1.

    Sign says which way along the flow the aircraft is heading, so a track that
    doubles back changes sign at the turn. NaN where the ice is slower than
    min_speed_m_yr, since the direction of near-stagnant ice is mostly noise.
    """
    vx, vy = sample_velocity(lat, lon, directory=directory)
    ex, ey = track_heading(lat, lon, smooth=smooth)

    speed = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (ex * vx + ey * vy) / speed
    cos[~np.isfinite(speed) | (speed < min_speed_m_yr)] = np.nan
    return np.clip(cos, -1.0, 1.0)


def flow_angle_deg(lat, lon, directory=None, smooth: int = 5,
                   min_speed_m_yr: float = 50.0) -> np.ndarray:
    """
    Per-sample angle between the track and the flow direction, 0-90 degrees.

    Folded to 0-90 because flying up-flow and down-flow are equally usable; 0
    is along flow, 90 is across it. Use signed_cos when the travel direction
    matters, e.g. to catch a track doubling back.
    """
    cos = signed_cos(lat, lon, directory=directory, smooth=smooth,
                     min_speed_m_yr=min_speed_m_yr)
    return np.degrees(np.arccos(np.abs(cos)))


def sinuosity(lat, lon) -> float:
    """
    Path length over straight-line distance between the endpoints.

    1.0 is a straight leg. Gridded surveys that turn inside a segment climb
    well above it, which is the signature to reject.
    """
    px, py = _project(lat, lon)
    steps = np.hypot(np.diff(px), np.diff(py))
    path = float(np.nansum(steps))
    chord = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
    if chord <= 0:
        return np.inf
    return path / chord


@dataclass(frozen=True)
class FlowAlignment:
    """How well one track follows the ice flow."""

    median_angle_deg: float
    p90_angle_deg: float
    sinuosity: float
    monotonic_fraction: float
    median_speed_m_yr: float
    fraction_valid: float

    def is_along_flow(self, max_angle_deg: float = 30.0,
                      max_sinuosity: float = 1.10,
                      min_monotonic_fraction: float = 0.95,
                      min_fraction_valid: float = 0.5) -> bool:
        """Whether this track is usable for grounding point detection."""
        return bool(
            np.isfinite(self.median_angle_deg)
            and self.median_angle_deg <= max_angle_deg
            and self.sinuosity <= max_sinuosity
            and self.monotonic_fraction >= min_monotonic_fraction
            and self.fraction_valid >= min_fraction_valid
        )

    def __str__(self) -> str:
        return (f"angle {self.median_angle_deg:.0f} deg (p90 "
                f"{self.p90_angle_deg:.0f}), sinuosity {self.sinuosity:.3f}, "
                f"monotonic {self.monotonic_fraction:.0%}, "
                f"speed {self.median_speed_m_yr:.0f} m/yr, "
                f"{self.fraction_valid:.0%} valid")


def assess_alignment(profile: GlacierProfile, directory=None,
                     smooth: int = 5,
                     min_speed_m_yr: float = 50.0) -> FlowAlignment:
    """Summarise a profile's alignment with the flow field."""
    if profile.lat is None or profile.lon is None:
        raise ValueError(
            "profile has no lat/lon; build it with from_xopr so coordinates "
            "are carried through"
        )

    cos = signed_cos(profile.lat, profile.lon, directory=directory,
                     smooth=smooth, min_speed_m_yr=min_speed_m_yr)
    angle = np.degrees(np.arccos(np.abs(cos)))
    vx, vy = sample_velocity(profile.lat, profile.lon, directory=directory)
    speed = np.hypot(vx, vy)
    valid = np.isfinite(angle)

    return FlowAlignment(
        median_angle_deg=float(np.nanmedian(angle)) if valid.any() else np.nan,
        p90_angle_deg=(float(np.nanpercentile(angle, 90)) if valid.any()
                       else np.nan),
        sinuosity=sinuosity(profile.lat, profile.lon),
        monotonic_fraction=_monotonic_fraction(cos),
        median_speed_m_yr=float(np.nanmedian(speed)),
        fraction_valid=float(valid.mean()),
    )


def _monotonic_fraction(cos: np.ndarray) -> float:
    """
    Fraction of the track travelling the same way along the flow.

    1.0 for a one-way transect, ~0.5 for an out-and-back repeat line. A
    fold-back reuses the same ground at two different along-track distances, so
    x stops being a flowline coordinate even though both legs look along-flow
    once the angle is folded to 0-90.
    """
    valid = np.isfinite(cos)
    if not valid.any():
        return np.nan
    signs = np.sign(cos[valid])
    signs = signs[signs != 0]
    if signs.size == 0:
        return np.nan
    return float(max((signs > 0).mean(), (signs < 0).mean()))


def along_flow_runs(profile: GlacierProfile,
                    max_angle_deg: float = 30.0,
                    min_length_km: float = 10.0,
                    directory=None,
                    smooth: int = 5,
                    min_speed_m_yr: float = 50.0) -> list:
    """
    Every contiguous along-flow leg, as a list of (lo_km, hi_km).

    Runs are split where the track reverses relative to the flow, so an
    out-and-back yields one leg per direction rather than both fused together.
    """
    cos = signed_cos(profile.lat, profile.lon, directory=directory,
                     smooth=smooth, min_speed_m_yr=min_speed_m_yr)
    angle = np.degrees(np.arccos(np.abs(cos)))
    aligned = np.isfinite(angle) & (angle <= max_angle_deg)
    if not aligned.any():
        return []

    legs = []
    for direction in (1.0, -1.0):
        good = aligned & (np.sign(cos) == direction)
        for start, stop in _runs(good):
            lo, hi = float(profile.x[start]), float(profile.x[stop])
            if hi - lo >= min_length_km:
                legs.append((lo, hi))
    return sorted(legs)


def longest_along_flow_run(profile: GlacierProfile, **kwargs) -> Optional[tuple]:
    """
    Longest along-flow leg, as (lo_km, hi_km), or None.

    Length alone is a poor criterion for grounding point work: an out-and-back
    can spend its longest leg inland and cross the grounding zone on the
    shorter one. Prefer select_flotation_leg.
    """
    legs = along_flow_runs(profile, **kwargs)
    if not legs:
        return None
    return max(legs, key=lambda leg: leg[1] - leg[0])


def select_flotation_leg(profile: GlacierProfile,
                         margin_km: float = 3.5,
                         max_angle_deg: float = 30.0,
                         min_length_km: float = 10.0,
                         directory=None,
                         smooth: int = 5,
                         min_speed_m_yr: float = 50.0) -> Optional[tuple]:
    """
    The along-flow leg that actually crosses the grounding zone, as (lo_km, hi_km).

    Considers only legs whose own flotation window resolves, then keeps the one
    with the most bed power inside that window. An out-and-back crosses the
    transition once per leg, and the legs are not equally usable: on Helheim
    20080730_01 the outbound leg reaches the terminus with its bed picks
    already gone, while the return leg carries them through. Scoring by data at
    the transition picks the leg a detector can work with, which is what a
    hand-tuned crop was doing.

    margin_km only scores the legs; it is not the detection window. Crop to the
    returned leg, then take a fresh flotation_window at whatever margin the
    glacier wants. Selection is insensitive to it in practice.

    Returns None if no leg resolves. Feed the result to GlacierProfile.window,
    then let the detector take its own flotation window inside it.
    """
    best = None
    best_score = -1
    for leg in along_flow_runs(profile, max_angle_deg=max_angle_deg,
                               min_length_km=min_length_km,
                               directory=directory, smooth=smooth,
                               min_speed_m_yr=min_speed_m_yr):
        try:
            sub = profile.window(*leg)
            lo, hi = sub.flotation_window(margin_km)
        except ValueError:
            continue          # no resolvable crossing on this leg
        inside = (sub.x >= lo) & (sub.x <= hi)
        score = int(np.isfinite(sub.amp[inside]).sum())
        if score > best_score:
            best, best_score = leg, score

    return best if best_score > 0 else None



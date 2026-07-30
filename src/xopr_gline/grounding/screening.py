"""
Segment screening: is this transect worth running a detector on?

A usable grounding point needs three things at once, and each one on its own
admits transects that cannot work:

  1. a terminus in range, found from the data where ice thickness collapses
  2. ice that actually approaches flotation, from the height above buoyancy
  3. a track running along flow rather than across it or doubling back

Screen first, detect second, so bulk runs spend their time on segments that can
produce an answer.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .flow import assess_alignment, select_flotation_leg
from .profile import GlacierProfile


@dataclass(frozen=True)
class SegmentScreen:
    """Verdict on one profile, with the numbers behind it."""

    leg_km: Optional[tuple]
    terminus_km: Optional[float]
    min_hab_m: float
    hab_crosses_zero: bool
    distance_to_terminus_km: Optional[float]
    median_angle_deg: float
    monotonic_fraction: float
    ice_fraction: float
    reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.reason

    def __str__(self) -> str:
        head = "USE " if self.usable else "SKIP"
        leg = (f"{self.leg_km[0]:.1f}-{self.leg_km[1]:.1f} km"
               if self.leg_km else "none")
        term = ("none" if self.terminus_km is None
                else f"{self.terminus_km:.1f} km")
        gap = ("" if self.distance_to_terminus_km is None
               else f", GZ {self.distance_to_terminus_km:.1f} km from terminus")
        return (f"{head} leg {leg}, terminus {term}{gap}, "
                f"min HAB {self.min_hab_m:.0f} m, angle "
                f"{self.median_angle_deg:.0f} deg"
                + (f"  [{self.reason}]" if self.reason else ""))


def screen_profile(profile: GlacierProfile,
                   margin_km: float = 3.5,
                   min_thickness_m: float = 25.0,
                   max_terminus_distance_km: Optional[float] = None,
                   hab_tolerance_m: float = 50.0,
                   max_angle_deg: float = 30.0,
                   directory=None) -> SegmentScreen:
    """
    Apply the three screens to a profile and say whether to run detectors.

    Reports the first failing reason rather than a bare False, so a bulk sweep
    can tell "no terminus in this frame slice" from "flew across the flow".

    max_terminus_distance_km is off by default because "grounding zone near the
    terminus" is a tidewater property, not a general one. Both regimes float;
    what differs is how far. Helheim carries a short floating section and
    grounds 0.3 km behind its front, Petermann ~76 km behind a long tongue, and
    both are valid targets. Gate on the distance only when you specifically
    want tidewater.

    Read min_hab_m as an approach to flotation, not a threshold: on Helheim it
    bottoms out near 10 m over ~730 m of ice, which is inside the error budget
    once firn air content (~10-15 m, not corrected here) and the geoid are
    accounted for. hab_tolerance_m is set accordingly.
    """
    ice = profile.ice_mask(min_thickness_m)
    ice_fraction = float(ice.mean())
    alignment = assess_alignment(profile, directory=directory)

    leg = select_flotation_leg(profile, margin_km=margin_km,
                               max_angle_deg=max_angle_deg,
                               directory=directory)

    # Everything below is judged on the leg, since that is what a detector
    # would be handed.
    scope = profile.window(*leg) if leg else profile
    hab = scope.height_above_buoyancy[scope.ice_mask(min_thickness_m)]
    min_hab = float(np.nanmin(np.abs(hab))) if hab.size else np.nan
    crosses = bool(hab.size and np.nanmin(hab) < 0 < np.nanmax(hab))

    crossings = scope.terminus_crossings_km(min_thickness_m)
    terminus = crossings[-1] if crossings else None

    distance = None
    if terminus is not None and leg is not None:
        try:
            lo, hi = scope.flotation_window(margin_km)
            distance = float(min(abs(terminus - lo), abs(terminus - hi)))
        except ValueError:
            distance = None

    reason = _first_failure(
        leg=leg, terminus=terminus, distance=distance, min_hab=min_hab,
        alignment=alignment, max_angle_deg=max_angle_deg,
        max_terminus_distance_km=max_terminus_distance_km,
        hab_tolerance_m=hab_tolerance_m,
    )

    return SegmentScreen(
        leg_km=leg,
        terminus_km=terminus,
        min_hab_m=min_hab,
        hab_crosses_zero=crosses,
        distance_to_terminus_km=distance,
        median_angle_deg=alignment.median_angle_deg,
        monotonic_fraction=alignment.monotonic_fraction,
        ice_fraction=ice_fraction,
        reason=reason,
    )


def _first_failure(leg, terminus, distance, min_hab, alignment,
                   max_angle_deg, max_terminus_distance_km,
                   hab_tolerance_m) -> str:
    """The first screen that fails, worded for a sweep log."""
    if leg is None:
        return "no along-flow leg crossing flotation"
    if not np.isfinite(alignment.median_angle_deg):
        return "flow direction undefined (ice too slow)"
    if alignment.median_angle_deg > max_angle_deg:
        return f"across flow ({alignment.median_angle_deg:.0f} deg)"
    if terminus is None:
        return "no terminus in range"
    if not np.isfinite(min_hab):
        return "no ice on the leg"
    if min_hab > hab_tolerance_m:
        return f"never near flotation (min HAB {min_hab:.0f} m)"
    if (max_terminus_distance_km is not None and distance is not None
            and distance > max_terminus_distance_km):
        return f"terminus {distance:.0f} km from the transition"
    return ""

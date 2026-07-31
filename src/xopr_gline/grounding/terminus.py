"""
Did the detector land on the calving front rather than a grounding transition?

Two different things put a grounding point at the terminus and they need
opposite responses. A glacier can genuinely ground at its front: Helheim
20080730_01 puts the changepoint 1.3 km inboard of its calving face, and there
the grounding point *is* the terminus. Or a floating tongue reaches the sea, in
which case a detection at the front is the detector locking onto the ice/water
contrast -- a large step in bed power that says nothing about grounding -- and
the result should not be trusted.

Distance alone cannot separate them, and neither can the flotation residual at
the face. Ice at a grounding zone is by definition near flotation, and so is a
shelf: Helheim's residual is only +20 m in the last 2 km before it calves,
inside the error budget once firn air content (10-15 m, uncorrected) is
allowed for. What separates them is what sits behind the front. Helheim climbs
to +98 m 5-10 km inland and +218 m by 10-20 km, because grounded ice rises
fast above flotation. A floating tongue stays near zero for tens of km until
its real grounding zone. So the question asked here is "is there a floating
section behind this front?", measured on a stretch well inland of the
flexure-dominated face.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .profile import GlacierProfile
from .result import DetectionResult

INLAND = "inland"
GROUNDED_TERMINUS = "grounded_terminus"
TERMINUS_ARTIFACT = "terminus_artifact"
TERMINUS_UNKNOWN = "terminus_unknown"
BEYOND_TERMINUS = "beyond_terminus"
NO_TERMINUS = "no_terminus"


@dataclass(frozen=True)
class TerminusVerdict:
    """Where a detection sits relative to the calving front, and the numbers."""

    classification: str
    terminus_km: Optional[float]
    distance_km: Optional[float]
    residual_at_front_m: float
    residual_inland_m: float
    interval_spans_terminus: bool

    @property
    def at_terminus(self) -> bool:
        return self.classification in (GROUNDED_TERMINUS, TERMINUS_ARTIFACT,
                                       TERMINUS_UNKNOWN)

    @property
    def suspect(self) -> bool:
        """Whether the detection should be held back from a results table."""
        return self.classification in (TERMINUS_ARTIFACT, BEYOND_TERMINUS)

    def __str__(self) -> str:
        if self.terminus_km is None:
            return "no calving front in range"
        where = ("at the front" if abs(self.distance_km) < 1e-9 else
                 f"{abs(self.distance_km):.2f} km "
                 f"{'inland of' if self.distance_km > 0 else 'seaward of'} "
                 f"the front")
        residual = ("residual unmeasured" if not np.isfinite(self.residual_inland_m)
                    else f"residual {self.residual_at_front_m:+.0f} m at the "
                         f"front, {self.residual_inland_m:+.0f} m inland")
        return (f"{self.classification}: terminus {self.terminus_km:.2f} km, "
                f"{where}, {residual}"
                + (", 95% interval spans it" if self.interval_spans_terminus else ""))


def classify_terminus(profile: GlacierProfile,
                      result: DetectionResult,
                      tol_km: float = 2.0,
                      front_km: float = 2.0,
                      inland_km: tuple = (5.0, 15.0),
                      min_inland_span_km: float = 2.0,
                      grounded_threshold_m: float = 30.0,
                      min_thickness_m: float = 25.0,
                      min_open_km: float = 1.0,
                      smooth_km: float = 5.0) -> TerminusVerdict:
    """
    Classify a detection against the nearest calving front.

    tol_km is how close counts as "at the terminus". The default of 2 km is set
    by Helheim, whose changepoint sits 1.3 km inboard of a front it genuinely
    grounds at; tightening it below that would call a real grounded terminus
    inland.

    inland_km is the stretch behind the front the verdict is actually decided
    on, in km along flow. It starts at 5 km to clear tidal flexure and the
    grounding zone itself, where grounded and floating ice are equally close to
    hydrostatic balance.

    grounded_threshold_m matches GlacierProfile.flotation_window, so "grounded"
    means the same thing here as it does when the search window is built. It is
    applied inland, not at the face, where it would reject every real grounded
    terminus.

    Returns a verdict rather than raising, including when there is no terminus
    in range, so a bulk sweep gets a row either way.
    """
    crossings = profile.terminus_crossings_km(min_thickness_m, min_open_km)
    if not crossings:
        return TerminusVerdict(NO_TERMINUS, None, None, np.nan, np.nan, False)

    # Nearest to the detection, not the seaward-most: a segment can cross open
    # water more than once, and the front that matters is the one the detector
    # may have mistaken for a grounding transition.
    terminus = min(crossings, key=lambda c: abs(c - result.map_km))

    try:
        sign = profile.landward_sign(smooth_km)
    except ValueError:
        sign = 1
    distance = float(sign * (result.map_km - terminus))

    # Water samples drop out with the ice mask, which matters because with
    # H ~ 0 the residual collapses to the water surface height and reads as
    # near flotation for free.
    residual = profile.smoothed_residual(smooth_km)
    ice = profile.ice_mask(min_thickness_m)
    inland = sign * (profile.x - terminus)      # + is inland along the flow

    at_front = ice & (np.abs(profile.x - terminus) <= front_km)
    behind = ice & (inland >= inland_km[0]) & (inland <= inland_km[1])

    front_residual = (float(np.nanmedian(residual[at_front])) if at_front.any()
                      else np.nan)
    # A couple of stray samples cannot carry the verdict, so require the
    # stretch to actually span some ground before it is allowed to decide.
    span = (float(profile.x[behind].max() - profile.x[behind].min())
            if behind.any() else 0.0)
    inland_residual = (float(np.nanmedian(residual[behind]))
                       if span >= min_inland_span_km else np.nan)

    lo, hi = result.credible_interval(0.95)
    spans = bool(lo <= terminus <= hi)

    return TerminusVerdict(
        classification=_classify(distance, inland_residual, tol_km,
                                 grounded_threshold_m),
        terminus_km=float(terminus),
        distance_km=distance,
        residual_at_front_m=front_residual,
        residual_inland_m=inland_residual,
        interval_spans_terminus=spans,
    )


def _classify(distance_km, inland_residual_m, tol_km, grounded_threshold_m) -> str:
    """
    The verdict, from the signed distance and the flotation state behind the front.

    interval_spans_terminus is reported but does not decide: a posterior
    spanning the terminus is evidence the two cannot be told apart, yet letting
    it vote would drag plainly inland detections into the at-terminus class
    whenever the posterior is diffuse.
    """
    if distance_km < -tol_km:
        # Seaward of the calving front is open water. Nothing grounds there.
        return BEYOND_TERMINUS
    if abs(distance_km) > tol_km:
        return INLAND
    if not np.isfinite(inland_residual_m):
        # No measurable ice behind the front, so whether a shelf sits there is
        # unanswered. Not the same as knowing the detection is bad.
        return TERMINUS_UNKNOWN
    if inland_residual_m > grounded_threshold_m:
        return GROUNDED_TERMINUS
    return TERMINUS_ARTIFACT

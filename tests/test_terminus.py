"""
Tests for classifying a detection against the calving front.
"""

import numpy as np

from xopr_gline.constants import DEFAULT_CONSTANTS
from xopr_gline.grounding import GlacierProfile, classify_terminus
from xopr_gline.grounding.result import DetectionResult
from xopr_gline.grounding.terminus import (BEYOND_TERMINUS, GROUNDED_TERMINUS,
                                           INLAND, NO_TERMINUS,
                                           TERMINUS_ARTIFACT, TERMINUS_UNKNOWN)


def _profile(thickness, surface, geoid=0.0):
    """A profile with prescribed geometry, on a 0.5 km grid."""
    thickness = np.asarray(thickness, dtype=float)
    surface = (np.full_like(thickness, surface) if np.isscalar(surface)
               else np.asarray(surface, dtype=float))
    return GlacierProfile(
        x=np.arange(len(thickness)) * 0.5,
        amp=np.full_like(thickness, -100.0),
        h_surf=surface,
        h_bed=surface - thickness,
        geoid_separation_m=geoid,
    )


def _result(map_km):
    return DetectionResult(map_km=map_km, x_km=np.array([map_km]),
                           detector="test")


def _front(profile):
    """The calving front the fixture was built with."""
    crossings = profile.terminus_crossings_km()
    assert len(crossings) == 1, "fixture should have exactly one front"
    return crossings[0]


WATER = 20                     # 10 km of open water at the seaward end


def _with_offset(thickness, offset_m):
    """
    Ice of a given thickness sitting offset_m above hydrostatic flotation.

    Fixtures are written in flotation offsets rather than raw elevations
    because that is the quantity the classifier reads; the surface is derived.
    Open water is appended past the front: thickness 1 m under a 2 m surface,
    which is how a collapsed pick looks.
    """
    thickness = np.asarray(thickness, dtype=float)
    surface = thickness * DEFAULT_CONSTANTS.flotation_factor + offset_m
    return _profile(np.concatenate([thickness, np.full(WATER, 1.0)]),
                    np.concatenate([surface, np.full(WATER, 2.0)]))


def _grounded_front():
    """
    Helheim's shape: near flotation at the face, climbing steeply inland.

    x increases seaward, so inland is decreasing x. The offset follows a curve
    rather than a ramp because Helheim's does: +20 m over the last 2 km, +29 m
    at 2-5 km, then +98 m at 5-10 km and +218 m by 10-20 km.
    """
    inland = np.linspace(19.5, 0.0, 40)         # km behind the face
    return _with_offset(np.linspace(700, 1400, 40), 20.0 + 2.0 * inland ** 2)


def _floating_front():
    """
    Petermann's shape: a 20 km shelf at flotation, grounded ice behind it.

    The grounded section is what fixes the profile's orientation -- a bare
    shelf has a flat residual, leaving landward_sign undetermined, which no
    real transect crossing a grounding zone does.
    """
    shelf = np.full(40, 600.0)                  # 20 km at flotation
    grounded = np.linspace(1400, 700, 20)       # 10 km of grounded ice behind
    return _with_offset(
        np.concatenate([grounded, shelf]),
        np.concatenate([np.linspace(400.0, 0.0, 20), np.zeros(40)]),
    )


def test_no_terminus_returns_a_verdict_not_an_error():
    """A bulk sweep needs a row back even with no front in the frame slice."""
    profile = _profile(np.full(30, 800.0), np.linspace(600, 300, 30))
    verdict = classify_terminus(profile, _result(5.0))
    assert verdict.classification == NO_TERMINUS
    assert verdict.terminus_km is None
    assert not verdict.at_terminus and not verdict.suspect


def test_grounded_front_is_a_real_grounding_point():
    """
    Ice climbing above flotation behind the face means the glacier grounds at
    its terminus. That is an answer, not a failure, so it is not suspect.
    """
    profile = _grounded_front()
    verdict = classify_terminus(profile, _result(_front(profile) - 1.0))
    assert verdict.classification == GROUNDED_TERMINUS
    assert verdict.terminus_km == _front(profile)
    assert verdict.at_terminus
    assert not verdict.suspect
    assert verdict.residual_inland_m > 30.0


def test_floating_front_is_an_ice_water_artifact():
    """
    The same detection with a shelf behind the front is the ice/water contrast,
    not a grounding transition, and has to be held back from a results table.
    """
    profile = _floating_front()
    verdict = classify_terminus(profile, _result(_front(profile) - 1.0))
    assert verdict.classification == TERMINUS_ARTIFACT
    assert verdict.at_terminus
    assert verdict.suspect


def test_the_face_alone_cannot_tell_them_apart():
    """
    Why the verdict is decided inland: both fixtures are near flotation in the
    last 2 km, so a threshold applied at the face would classify them alike.
    """
    g, f = _grounded_front(), _floating_front()
    grounded = classify_terminus(g, _result(_front(g) - 1.0))
    floating = classify_terminus(f, _result(_front(f) - 1.0))

    assert abs(grounded.residual_at_front_m) < 30.0
    assert abs(floating.residual_at_front_m) < 30.0
    assert grounded.classification != floating.classification


def test_detection_seaward_of_the_front_is_rejected():
    """Nothing grounds in open water past the calving face."""
    profile = _grounded_front()
    verdict = classify_terminus(profile, _result(_front(profile) + 4.0))
    assert verdict.classification == BEYOND_TERMINUS
    assert verdict.distance_km < 0
    assert verdict.suspect


def test_detection_well_inland_is_untouched():
    """The ordinary case: a grounding point with ice between it and the sea."""
    verdict = classify_terminus(_grounded_front(), _result(2.0))
    assert verdict.classification == INLAND
    assert verdict.distance_km > 2.0
    assert not verdict.at_terminus and not verdict.suspect


def test_tolerance_is_wide_enough_for_helheim():
    """
    Helheim's changepoint sits 1.3 km inboard of a front it genuinely grounds
    at. A default tolerance that called that inland would be useless.
    """
    profile = _grounded_front()
    verdict = classify_terminus(profile, _result(_front(profile) - 1.3))
    assert verdict.classification == GROUNDED_TERMINUS


def test_no_ice_behind_the_front_is_unknown_not_rejected():
    """
    A frame slice that stops just behind the face cannot say whether a shelf
    sits there. Unknown must not be lumped in with suspect, or short slices
    would silently drop real grounded termini.
    """
    stub = _profile(np.concatenate([np.full(8, 900.0), np.full(WATER, 1.0)]),
                    np.concatenate([np.full(8, 120.0), np.full(WATER, 2.0)]))
    front = stub.terminus_crossings_km()[0]
    verdict = classify_terminus(stub, _result(front - 1.0))
    assert verdict.classification == TERMINUS_UNKNOWN
    assert verdict.at_terminus
    assert not verdict.suspect


def test_sign_convention_follows_the_flow_not_x():
    """
    The other fixtures run seaward, so ice sits at low x. Flip them and inland
    is now high x: the signed distance has to follow the flow, not the axis, or
    every verdict inverts on half the transects.
    """
    assert _grounded_front().landward_sign() == -1

    landward = _profile(
        np.concatenate([np.full(20, 900.0), np.full(20, 1.0)])[::-1].copy(),
        np.concatenate([np.full(20, 400.0), np.full(20, 2.0)])[::-1].copy(),
    )
    assert landward.landward_sign() == 1
    terminus = landward.terminus_crossings_km()[0]

    inland = classify_terminus(landward, _result(terminus + 5.0))
    assert inland.classification == INLAND
    assert inland.distance_km > 0

    beyond = classify_terminus(landward, _result(terminus - 5.0))
    assert beyond.classification == BEYOND_TERMINUS
    assert beyond.distance_km < 0

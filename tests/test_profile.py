"""GlacierProfile grid invariants."""

import numpy as np
import pytest

from xopr_gline.grounding.profile import GlacierProfile


def _profile(x):
    n = len(x)
    return GlacierProfile.from_arrays(
        x, np.full(n, -60.0), np.full(n, 100.0), np.full(n, -400.0)
    )


def test_uniform_grid_accepted():
    p = _profile(np.arange(0.0, 20.0, 0.716))
    assert p.dx == pytest.approx(0.716)


def test_near_duplicate_sample_rejected():
    """A 0.19 m step in a 716 m grid makes np.gradient spike, so refuse it."""
    x = np.arange(0.0, 20.0, 0.716)
    x[10] = x[9] + 0.00019
    with pytest.raises(ValueError, match="uniformly spaced"):
        _profile(x)


def test_error_names_the_offending_sample():
    x = np.arange(0.0, 20.0, 0.5)
    x[7] = x[6] + 0.01
    with pytest.raises(ValueError, match=r"x\[7\] - x\[6\]"):
        _profile(x)


def test_jitter_within_tolerance_accepted():
    rng = np.random.default_rng(0)
    steps = 0.5 * (1 + rng.uniform(-0.005, 0.005, 39))
    _profile(np.concatenate([[0.0], np.cumsum(steps)]))


def test_non_monotonic_still_rejected_first():
    x = np.arange(0.0, 10.0, 0.5)
    x[4], x[5] = x[5], x[4]
    with pytest.raises(ValueError, match="strictly increasing"):
        _profile(x)


def test_window_keeps_a_valid_grid():
    p = _profile(np.arange(0.0, 40.0, 0.5))
    w = p.window(10.0, 20.0)
    assert w.dx == pytest.approx(0.5)

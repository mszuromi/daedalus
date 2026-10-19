"""The Taylor norm bounds of the state-space route are reproducible: the
1-norm estimate used beyond the exact range drew unseeded random start
vectors, so two identical runs could pick different Taylor parameters and
differ in the last bits (seen on Linux CI in
test_delay_loops_memory_61::test_discard_is_bit_identical_and_releases)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.integration.time_domain import statespace as SS  # noqa: E402


def _matrix(n, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    return A - 2.0 * np.eye(n)


def test_norms_are_reproducible_in_the_estimated_range(monkeypatch):
    monkeypatch.setattr(SS, '_EXACT_NORM_N', 50)        # force the estimator
    A = _matrix(120)
    first = SS._onenorm_powers(A)
    np.random.seed(1)
    for _ in range(5):
        np.random.rand(7)                                # disturb the global stream
        assert SS._onenorm_powers(A) == first
    # the caller's random stream is untouched by the call
    np.random.seed(5)
    a = np.random.rand(3)
    np.random.seed(5)
    SS._onenorm_powers(A)
    assert np.array_equal(np.random.rand(3), a)


def test_exact_range_is_the_true_norm_and_p_bounds_are_upper_bounds():
    A = _matrix(350)                                     # n > 300: exact since the fix
    d = SS._onenorm_powers(A)
    assert d[2] == float(np.abs(A @ A).sum(axis=0).max()) ** 0.5
    P = A.copy()
    for p in range(2, 6):
        P = P @ A
        assert d[p] >= float(np.abs(P).sum(axis=0).max()) ** (1.0 / p) * (1 - 1e-12)

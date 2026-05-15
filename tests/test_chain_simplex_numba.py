"""
tests/test_chain_simplex_numba.py
=================================
Parity tests for the numba-compiled chain simplex
(``_exp_over_chain_simplex_fast`` → ``_exp_over_chain_simplex_numba_core``)
against the pure-Python reference (``_exp_over_chain_simplex``).

The numba version operates on numpy buffers and uses the same closed
form, so results should agree to floating-point round-off across the
configurations that exercise the analytic path in production.  The
last test compares them on a representative grid of cases including
non-degenerate, near-degenerate, and overflow scenarios.

Run with::

    sage -python -m pytest tests/test_chain_simplex_numba.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.time_domain import final_integral as _fi
from msrjd.integration.time_domain.final_integral import (
    _exp_over_chain_simplex,
    _exp_over_chain_simplex_fast,
)


def _both_paths(alphas, lower, upper):
    """Run Python ref and numba dispatcher; return both."""
    py = _exp_over_chain_simplex(alphas, lower, upper)
    nb = _exp_over_chain_simplex_fast(alphas, lower, upper)
    return py, nb


def _assert_parity(py, nb, label=''):
    """Both None, or both finite with relative diff < 1e-12."""
    if py is None and nb is None:
        return
    assert py is not None and nb is not None, (
        f'{label}: one is None, the other is not. py={py}, nb={nb}'
    )
    if py == 0 and nb == 0:
        return
    denom = max(abs(py), 1e-30)
    rel = abs(py - nb) / denom
    assert rel < 1e-12, (
        f'{label}: py={py}, nb={nb}, |Δ|={abs(py - nb):.3e}, '
        f'rel={rel:.3e}'
    )


# ─── Small specific cases ─────────────────────────────────────────
def test_empty_chain():
    py, nb = _both_paths([], 0.0, 1.0)
    assert py == 1.0 + 0.0j
    assert nb == 1.0 + 0.0j


def test_empty_interval():
    py, nb = _both_paths([-0.3 + 0.0j], 1.0, 0.0)
    assert py == 0.0 + 0.0j
    assert nb == 0.0 + 0.0j


def test_single_variable():
    py, nb = _both_paths([-0.5 + 0.2j], 0.0, 5.0)
    _assert_parity(py, nb, 'single var')


def test_degenerate_inner_returns_none():
    """Innermost α=0 → Python returns None; numba should match."""
    py, nb = _both_paths([0.0 + 0.0j, -0.5 + 0.0j], 0.0, 5.0)
    assert py is None
    assert nb is None


# ─── Parameterised grid (non-degenerate cases) ─────────────────────
@pytest.mark.parametrize('alphas,L,U', [
    ([-0.3 + 0.1j, -0.5 + 0.0j], 0.0, 5.0),
    ([-0.2 + 0.3j, -0.4 - 0.1j, -0.6 + 0.2j], -1.0, 3.0),
    ([0.4 - 0.1j, -0.7 + 0.8j, -1.1 + 0.0j], 0.0, 4.0),
    ([-0.05 + 0.0j, -0.10 + 0.0j, -0.15 + 0.0j, -0.20 + 0.0j], 0.0, 10.0),
    ([-0.5 + 1.0j, -0.7 - 0.5j, -1.0 + 0.3j, -1.3 - 0.2j], -2.0, 4.0),
])
def test_parity_nondegenerate(alphas, L, U):
    py, nb = _both_paths(alphas, L, U)
    _assert_parity(py, nb, f'alphas={alphas}')


# ─── Near-degenerate (some β values close to but not at zero) ─────
def test_parity_near_degenerate():
    """Pole values arranged so cumulative β is small but nonzero
    (1e-6).  Python returns a value (numerically poor); numba should
    match it bit-for-bit."""
    eps_small = 1e-6
    alphas = [-0.5 + 0.0j, -0.5 + eps_small + 0.0j]  # β_outer ≈ 1e-6
    py, nb = _both_paths(alphas, 0.0, 3.0)
    _assert_parity(py, nb, 'near-degenerate')


# ─── Overflow case (large interval) ─────────────────────────────────
def test_parity_overflow():
    """Large interval × large |Re β| → both should return None."""
    py, nb = _both_paths([-10.0 + 0.0j, -10.0 + 0.0j], -100.0, 0.0)
    # Either both None (Python overflow guard) or both finite — but
    # they must agree.
    if py is None:
        assert nb is None, f'py=None but nb={nb}'
    else:
        _assert_parity(py, nb, 'overflow')


# ─── Speed sanity check ───────────────────────────────────────────
def test_numba_is_faster():
    """Trigger numba JIT compile, then time a tight loop against the
    Python version.  Pass if numba is at least 5× faster (numba should
    typically be 30-100× but be conservative to avoid CI flakes)."""
    if not _fi._HAVE_NUMBA:
        pytest.skip('numba not available')

    import time

    alphas = [-0.3 + 0.1j, -0.5 + 0.0j, -0.7 - 0.2j, -1.0 + 0.4j]
    L, U = 0.0, 5.0

    # Warm up JIT compile.
    _exp_over_chain_simplex_fast(alphas, L, U)

    N_ITERS = 5000
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _exp_over_chain_simplex(alphas, L, U)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _exp_over_chain_simplex_fast(alphas, L, U)
    t_nb = time.perf_counter() - t0

    speedup = t_py / max(t_nb, 1e-9)
    print(f'\n  python: {t_py:.3f}s, numba: {t_nb:.3f}s, speedup: {speedup:.1f}×')
    assert speedup >= 5.0, (
        f'numba speedup {speedup:.1f}× below 5× threshold; either numba '
        f'is not being used or compile failed silently'
    )

"""
tests/test_chain_simplex_precision_fix.py
=========================================
Regression tests for the chain-simplex close-pole precision fix
(``USE_CHAIN_SIMPLEX_PRECISION_FIX`` flag at
``msrjd/integration/time_domain/final_integral.py:1017+``).

Background
----------
At spike-reset k=2 ell=1 with canonical fixture params, the float64
chain-simplex closed form ``_exp_over_chain_simplex`` overestimates the
1-loop value by ~4× due to cancellation between Term A (coefficient
``C/b_inner``) and Term B (coefficient ``-C·exp(b_inner·lower)/b_inner``)
when ``b_inner`` is small (e.g., 0.022i from poles 0.329i - 0.351i = 0.022i).

The fix detects this regime via a cheap subset-sum scan at the
dispatcher entry and routes affected calls to
``_exp_over_chain_simplex_mpmath`` running at 50-digit precision.

Run with::

    sage -python -m pytest tests/test_chain_simplex_precision_fix.py -v
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
    _exp_over_chain_simplex_mpmath,
    _min_subset_sum_abs,
    _CHAIN_SIMPLEX_CANCEL_THRESHOLD,
)


# ─── Helper ────────────────────────────────────────────────────────


def _wide_pole_alphas():
    """Three α-values with no close pairs; float64 should be accurate."""
    return [0.5j, 1.5j, 3.0j]


def _close_pole_alphas():
    """Three α-values including a sum that's small (0.329j - 0.351j = -0.022j)."""
    return [0.329j, -0.351j, 0.494j]


# ─── Tests ─────────────────────────────────────────────────────────


def test_min_subset_sum_abs_detects_close_pair():
    """``_min_subset_sum_abs`` should find the small pair-sum of close poles."""
    alphas = _close_pole_alphas()
    mss = _min_subset_sum_abs(alphas)
    assert mss < _CHAIN_SIMPLEX_CANCEL_THRESHOLD, (
        f'expected min_subset_sum < {_CHAIN_SIMPLEX_CANCEL_THRESHOLD}, '
        f'got {mss}'
    )
    # The smallest non-empty subset sum is |0.329j + (-0.351j)| = 0.022.
    assert abs(mss - 0.022) < 1e-10, f'expected 0.022, got {mss}'


def test_min_subset_sum_abs_wide_poles_above_threshold():
    """Wide-pole α-vector should NOT trigger the fix."""
    mss = _min_subset_sum_abs(_wide_pole_alphas())
    assert mss >= _CHAIN_SIMPLEX_CANCEL_THRESHOLD, (
        f'wide poles should not trigger fix; got min_subset_sum={mss}')


def test_mpmath_matches_python_on_wide_poles():
    """When poles are wide, float64 and mpmath should agree at machine precision."""
    alphas = _wide_pole_alphas()
    L, U = 0.0, 5.0
    v_py = _exp_over_chain_simplex(alphas, L, U)
    v_mp = _exp_over_chain_simplex_mpmath(alphas, L, U)
    assert v_py is not None
    assert v_mp is not None
    diff = abs(v_py - v_mp)
    rel = diff / max(abs(v_mp), 1e-30)
    assert rel < 1e-12, (
        f'wide-pole disagreement: python={v_py}, mpmath={v_mp}, rel={rel}'
    )


def test_dispatcher_routes_close_poles_to_mpmath():
    """When the close-pole-mpmath fix is enabled, dispatcher should
    return the mpmath value (not the float64 value).

    Note: the flag is False by default (2026-05-17) because it did not
    address the spike-reset k=2 ell=1 bug — that turned out to be a
    cap-mismatch issue handled by ``USE_POSET_CAP_MATCH_SCIPY``.  The
    code path is preserved for possible future use; this test
    temporarily enables it.
    """
    saved = _fi.USE_CHAIN_SIMPLEX_PRECISION_FIX
    _fi.USE_CHAIN_SIMPLEX_PRECISION_FIX = True
    try:
        alphas = _close_pole_alphas()
        L, U = 0.0, 5.0
        v_dispatch = _exp_over_chain_simplex_fast(alphas, L, U)
        v_mp = _exp_over_chain_simplex_mpmath(alphas, L, U)
        assert v_dispatch is not None
        assert v_mp is not None
        diff = abs(v_dispatch - v_mp)
        # Dispatcher should call mpmath directly so values should be IDENTICAL.
        assert diff < 1e-14, (
            f'expected dispatcher to route to mpmath; got '
            f'dispatch={v_dispatch}, mpmath={v_mp}'
        )
    finally:
        _fi.USE_CHAIN_SIMPLEX_PRECISION_FIX = saved


def test_flag_off_recovers_pre_fix_behaviour():
    """Setting ``USE_CHAIN_SIMPLEX_PRECISION_FIX = False`` must recover
    bit-identical pre-fix behaviour, regardless of α-vector."""
    alphas = _close_pole_alphas()
    L, U = 0.0, 5.0

    saved = _fi.USE_CHAIN_SIMPLEX_PRECISION_FIX
    try:
        _fi.USE_CHAIN_SIMPLEX_PRECISION_FIX = False
        v_dispatch = _exp_over_chain_simplex_fast(alphas, L, U)
    finally:
        _fi.USE_CHAIN_SIMPLEX_PRECISION_FIX = saved

    v_py = _exp_over_chain_simplex(alphas, L, U)
    assert v_dispatch is not None
    assert v_py is not None
    # When the fix is off and numba is unavailable, dispatcher falls
    # through to the Python ref; values must be bit-identical.
    # When numba is available, numba is supposed to match Python to
    # 1e-12 by separate parity test.
    diff = abs(v_dispatch - v_py)
    assert diff < 1e-10, (
        f'flag-off should recover pre-fix path; '
        f'got dispatch={v_dispatch}, python_ref={v_py}, diff={diff}'
    )


def test_mpmath_handles_trivial_cases():
    """mpmath function handles N=0 and U≤L correctly."""
    assert _exp_over_chain_simplex_mpmath([], 0.0, 5.0) == 1.0 + 0.0j
    v = _exp_over_chain_simplex_mpmath([0.5j], 3.0, 3.0)
    assert abs(v) < 1e-15  # U == L → integral is 0
    v = _exp_over_chain_simplex_mpmath([0.5j], 5.0, 3.0)
    assert abs(v) < 1e-15  # U < L → integral is 0


def test_short_chain_skips_gate():
    """N < 3 should skip the close-pole check entirely (cheap fast path)."""
    # Even with an artificially small alpha, len < 3 → no routing to mpmath.
    alphas = [1e-6 + 0j, 0.5 + 0j]  # tiny first, but only 2 elements
    L, U = 0.0, 5.0
    # We can't easily assert "mpmath wasn't called", but we can confirm
    # the dispatcher returns the same as the direct float64 ref.
    v_dispatch = _exp_over_chain_simplex_fast(alphas, L, U)
    v_py = _exp_over_chain_simplex(alphas, L, U)
    if v_py is not None and v_dispatch is not None:
        diff = abs(v_dispatch - v_py)
        assert diff < 1e-10, (
            f'short chain should match python ref; '
            f'got dispatch={v_dispatch}, python={v_py}'
        )

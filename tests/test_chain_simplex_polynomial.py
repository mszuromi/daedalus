"""
tests/test_chain_simplex_polynomial.py
======================================
Direct unit tests for ``_exp_over_chain_simplex_polynomial`` — the
Stage 3b-extended polynomial-prefactor closed form for the chain-
simplex integral.

Coverage:
* Empty / trivial cases (N=0, N=1, upper ≤ lower).
* Non-degenerate β: agrees with the original closed form bit-identically.
* Single-level degeneracy (innermost, middle, outermost).
* Cumulative-pole-sum vanishing (α_inner + α_outer = 0).
* Multiple simultaneous degenerate levels.
* Cross-check against scipy.nquad on the actual chain simplex.

Run with::

    sage -python -m pytest tests/test_chain_simplex_polynomial.py -v
"""
from __future__ import annotations

import os
import sys
import math
import cmath

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.integration.time_domain.final_integral import (
    _exp_over_chain_simplex,
    _exp_over_chain_simplex_polynomial,
)


# ─── Trivial / boundary cases ──────────────────────────────────────
def test_empty_chain_returns_one():
    """N = 0: empty product = 1."""
    val = _exp_over_chain_simplex_polynomial([], -1.0, 2.0)
    assert val == 1.0 + 0.0j


def test_empty_interval_returns_zero():
    """upper ≤ lower: zero-measure domain."""
    val = _exp_over_chain_simplex_polynomial(
        [0.5 + 0.0j, -0.3 + 0.2j], 2.0, 1.0,
    )
    assert val == 0.0 + 0.0j


def test_single_variable_nondegenerate():
    """N=1: ∫_L^U exp(α s) ds = (exp(αU) − exp(αL))/α."""
    α = -0.4 + 0.7j
    L, U = -1.0, 3.0
    val = _exp_over_chain_simplex_polynomial([α], L, U)
    expected = (cmath.exp(α * U) - cmath.exp(α * L)) / α
    assert abs(val - expected) < 1e-13


def test_single_variable_degenerate():
    """N=1 with α=0: ∫_L^U 1 ds = U − L."""
    val = _exp_over_chain_simplex_polynomial([0.0 + 0.0j], -2.0, 5.0)
    assert abs(val - 7.0) < 1e-13


# ─── Non-degenerate: match the original closed form ───────────────
@pytest.mark.parametrize('alphas,L,U', [
    ([-0.5 + 0.7j, -0.3 - 0.2j], -1.0, 4.0),
    ([0.4 - 0.1j, -0.7 + 0.8j, -1.1 + 0.0j], 0.0, 3.0),
    ([-0.2j, -0.3 + 0.5j, -0.5 - 0.1j, -0.8 + 0.4j], -2.0, 5.0),
])
def test_nondegenerate_matches_original(alphas, L, U):
    """For non-degenerate β, polynomial path must match original."""
    poly_val = _exp_over_chain_simplex_polynomial(alphas, L, U)
    orig_val = _exp_over_chain_simplex(alphas, L, U)
    assert orig_val is not None, 'original should not return None here'
    assert poly_val is not None
    assert abs(poly_val - orig_val) < 1e-12


# ─── Degenerate at one level ───────────────────────────────────────
def test_innermost_alpha_zero():
    """α_1 = 0, α_2 ≠ 0.

    ∫_L^U ds_2 exp(α_2 s_2) · ∫_L^{s_2} 1 ds_1
      = ∫_L^U (s_2 − L) exp(α_2 s_2) ds_2
    Closed form (IBP):
      = (U−L)·exp(α_2·U)/α_2 − exp(α_2·U)/α_2² + exp(α_2·L)/α_2²
    """
    α_2 = -0.6 + 0.5j
    L, U = -1.0, 2.0
    val = _exp_over_chain_simplex_polynomial(
        [0.0 + 0.0j, α_2], L, U,
    )
    expected = ((U - L) * cmath.exp(α_2 * U) / α_2
                - cmath.exp(α_2 * U) / α_2**2
                + cmath.exp(α_2 * L) / α_2**2)
    assert val is not None
    assert abs(val - expected) < 1e-12


def test_outermost_alpha_zero():
    """α_1 ≠ 0, α_2 = 0.

    ∫_L^U ds_2 · ∫_L^{s_2} exp(α_1 s_1) ds_1
      = ∫_L^U (exp(α_1 s_2) − exp(α_1 L))/α_1 ds_2
      = (exp(α_1 U) − exp(α_1 L))/α_1² − (U−L) exp(α_1 L)/α_1
    """
    α_1 = 0.4 - 0.3j
    L, U = 0.0, 3.0
    val = _exp_over_chain_simplex_polynomial(
        [α_1, 0.0 + 0.0j], L, U,
    )
    expected = ((cmath.exp(α_1 * U) - cmath.exp(α_1 * L)) / α_1**2
                - (U - L) * cmath.exp(α_1 * L) / α_1)
    assert val is not None
    assert abs(val - expected) < 1e-12


def test_cumulative_pole_sum_zero_at_outermost():
    """N=2 with α_1 + α_2 = 0 (but individually nonzero).

    The merged β at the outermost level is 0.  The original closed
    form handles this case in its outermost branch ((U-L) for β≈0),
    so it does NOT return None — but the polynomial path must give
    the same answer.
    """
    α_1 = 0.5 + 0.2j
    α_2 = -α_1
    L, U = -0.5, 2.5
    val = _exp_over_chain_simplex_polynomial([α_1, α_2], L, U)
    assert val is not None
    expected = ((U - L) / α_1
                - (cmath.exp(α_1 * L)
                   * (cmath.exp(α_2 * U) - cmath.exp(α_2 * L))
                   / (α_1 * α_2)))
    assert abs(val - expected) < 1e-12
    # Original also handles this (β-degenerate-at-outermost branch).
    orig = _exp_over_chain_simplex([α_1, α_2], L, U)
    assert orig is not None
    assert abs(orig - expected) < 1e-12


def test_innermost_alpha_zero_original_returns_none():
    """Original returns None on innermost α=0; polynomial rescues."""
    α_2 = -0.6 + 0.5j
    L, U = -1.0, 2.0
    orig = _exp_over_chain_simplex([0.0 + 0.0j, α_2], L, U)
    assert orig is None, 'original path should bail on innermost α=0'
    poly = _exp_over_chain_simplex_polynomial(
        [0.0 + 0.0j, α_2], L, U,
    )
    assert poly is not None


def test_intermediate_merged_beta_zero():
    """N=3 with α_0 + α_1 = 0 (intermediate merged β is 0).

    After integrating s_0, the upper-piece carries merged β = α_0+α_1 = 0
    into level 1.  The original's level-1 ``abs(b_inner) < eps`` check
    bails to None.  The polynomial path handles it.
    """
    α_0 = 0.5 + 0.2j
    α_1 = -α_0
    α_2 = -0.7 + 0.4j
    L, U = -0.5, 2.0
    orig = _exp_over_chain_simplex([α_0, α_1, α_2], L, U)
    assert orig is None, 'original path should bail on intermediate merged β=0'
    poly = _exp_over_chain_simplex_polynomial(
        [α_0, α_1, α_2], L, U,
    )
    assert poly is not None
    # Cross-check against recursive scipy.quad reference.
    ref = _chain_simplex_recursive_quad([α_0, α_1, α_2], L, U)
    assert abs(poly - ref) < 1e-8 * max(1.0, abs(ref))


# ─── Cross-check against recursive scipy.quad ─────────────────────
def _chain_simplex_recursive_quad(alphas, lower, upper):
    """Reference: compute the chain-simplex integral via nested 1D
    scipy.quad calls (innermost → outermost), using complex
    arithmetic.  Slower than the analytic closed form but easier to
    audit than nquad's bounds-callback convention."""
    from scipy.integrate import quad

    N = len(alphas)
    if N == 0:
        return 1.0 + 0.0j

    def _inner(level, s_outer):
        # Returns ∫_lower^{s_outer} ds_level exp(α_level · s_level)
        #         · _inner(level-1, s_level)   if level > 0
        # Returns ∫_lower^{s_outer} ds_0 exp(α_0 · s_0)               if level = 0
        if level == 0:
            def f_re(s):
                return (cmath.exp(alphas[0] * s)).real
            def f_im(s):
                return (cmath.exp(alphas[0] * s)).imag
            r, _ = quad(f_re, lower, s_outer,
                        epsrel=1e-10, epsabs=1e-13, limit=200)
            i, _ = quad(f_im, lower, s_outer,
                        epsrel=1e-10, epsabs=1e-13, limit=200)
            return complex(r, i)
        else:
            def g_re(s):
                return (cmath.exp(alphas[level] * s)
                        * _inner(level - 1, s)).real
            def g_im(s):
                return (cmath.exp(alphas[level] * s)
                        * _inner(level - 1, s)).imag
            r, _ = quad(g_re, lower, s_outer,
                        epsrel=1e-10, epsabs=1e-13, limit=200)
            i, _ = quad(g_im, lower, s_outer,
                        epsrel=1e-10, epsabs=1e-13, limit=200)
            return complex(r, i)

    return _inner(N - 1, upper)


@pytest.mark.parametrize('alphas,L,U', [
    # Non-degenerate baseline.
    ([-0.4 + 0.1j, -0.6 - 0.2j, -0.3 + 0.4j], -1.0, 2.0),
    # Innermost degenerate.
    ([0.0 + 0.0j, -0.5 + 0.3j, -0.7 - 0.1j], 0.0, 2.0),
    # Outermost degenerate.
    ([-0.3 + 0.2j, -0.4 - 0.1j, 0.0 + 0.0j], -1.0, 1.5),
    # Cumulative sum zero (α_0 + α_1 = 0).
    ([0.5 - 0.2j, -0.5 + 0.2j, -0.3 + 0.4j], 0.0, 2.0),
    # Two degenerate levels.
    ([0.0 + 0.0j, 0.0 + 0.0j, -0.5 + 0.3j], 0.0, 2.0),
    # Three degenerate levels (pure polynomial).
    ([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], 0.0, 2.0),
])
def test_matches_recursive_quad(alphas, L, U):
    """End-to-end: polynomial closed form matches nested-quad numerics."""
    poly_val = _exp_over_chain_simplex_polynomial(alphas, L, U)
    assert poly_val is not None
    ref = _chain_simplex_recursive_quad(alphas, L, U)
    # scipy.quad with tight tolerances should match the analytic
    # closed form to ~1e-9 on integrals of order 1.
    tol = 1e-8 * max(1.0, abs(ref))
    assert abs(poly_val - ref) < tol, (
        f'analytic {poly_val} vs scipy {ref} differs by {abs(poly_val - ref):.3e} '
        f'(tol {tol:.3e})'
    )


# ─── Multiple degenerate levels ───────────────────────────────────
def test_all_alphas_zero_pure_polynomial():
    """All α = 0: integrand is constant 1.

    Chain-simplex volume = (U − L)^N / N!.
    """
    for N in (1, 2, 3, 4, 5):
        L, U = 0.0, 2.5
        val = _exp_over_chain_simplex_polynomial(
            [0.0 + 0.0j] * N, L, U,
        )
        expected = (U - L) ** N / math.factorial(N)
        assert val is not None
        assert abs(val - expected) < 1e-12, f'N={N}: got {val}, want {expected}'


def test_two_consecutive_degenerate_levels():
    """α_1 = α_2 = 0, α_3 ≠ 0.

    ∫∫∫_{L ≤ s_1 ≤ s_2 ≤ s_3 ≤ U}  exp(α_3 s_3)  ds_1 ds_2 ds_3
    = ∫_L^U ds_3 exp(α_3 s_3) · (s_3 - L)²/2
    """
    α_3 = -0.4 + 0.3j
    L, U = 0.0, 2.0
    val = _exp_over_chain_simplex_polynomial(
        [0.0 + 0.0j, 0.0 + 0.0j, α_3], L, U,
    )
    # Compute ∫_L^U (s - L)²/2 · exp(α_3 s) ds via IBP closed form.
    # ∫_0^{U-L} u²/2 · exp(α_3 (u + L)) du
    # = exp(α_3 L) / 2 · I_2(U-L)
    # where I_2(T) = T²·exp(α_3 T)/α_3 - 2T·exp(α_3 T)/α_3² + 2·exp(α_3 T)/α_3³ - 2/α_3³.
    T = U - L
    I2 = (T**2 * cmath.exp(α_3 * T) / α_3
          - 2 * T * cmath.exp(α_3 * T) / α_3**2
          + 2 * cmath.exp(α_3 * T) / α_3**3
          - 2 / α_3**3)
    expected = cmath.exp(α_3 * L) / 2 * I2
    assert val is not None
    assert abs(val - expected) < 1e-12


# ─── Smoke / overflow ─────────────────────────────────────────────
def test_overflow_returns_none():
    """Large real α with large interval should trigger overflow guard."""
    α = 100.0 + 0.0j   # exp(100 · 10) = exp(1000) overflows
    val = _exp_over_chain_simplex_polynomial(
        [α, α], 0.0, 20.0,
    )
    assert val is None

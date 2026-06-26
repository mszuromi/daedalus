"""
tests/test_chain_intermediate_uppers.py
========================================
Unit tests for ``_chain_with_intermediate_uppers`` — the Stage-3b
maximality fix that handles scalar upper bounds on non-maximal poset
variables (i.e. positions before the chain top in any given linear
extension).  Without this helper, those subsets bailed to scipy.

Run with::

    sage -python -m pytest tests/test_chain_intermediate_uppers.py -v
"""
from __future__ import annotations

import os
import sys
import cmath

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.integration.time_domain.final_integral import (
    _chain_with_intermediate_uppers,
    _exp_over_chain_simplex,
    _exp_over_chain_simplex_polynomial,
)


# ─── No intermediate uppers ────────────────────────────────────────
def test_no_uppers_matches_standard_chain():
    """When no scalar uppers, helper short-circuits to the standard
    chain simplex."""
    alphas = [-0.3 + 0.0j, -0.5 + 0.0j, -0.7 + 0.0j]
    L, U_top = 0.0, 10.0
    val = _chain_with_intermediate_uppers(alphas, L, {}, U_top)
    ref = _exp_over_chain_simplex(alphas, L, U_top)
    assert val is not None
    assert abs(val - ref) < 1e-13


def test_upper_only_on_chain_top():
    """Upper on chain top (= maximal element) is same as setting U_top."""
    alphas = [-0.3 + 0.0j, -0.5 + 0.0j]
    L = 0.0
    U_top = 10.0
    U_chain_top_explicit = 5.0
    val = _chain_with_intermediate_uppers(
        alphas, L, {1: U_chain_top_explicit}, U_top,
    )
    ref = _exp_over_chain_simplex(alphas, L, U_chain_top_explicit)
    assert val is not None
    assert abs(val - ref) < 1e-13


# ─── Intermediate upper on a non-maximal position ─────────────────
def test_m2_intermediate_upper_at_position_0_matches_direct():
    """m=2 with upper U_0 on position 0 (innermost).

    Direct integration:
       ∫_0^{U_top} ds_1 exp(α_1 s_1) · ∫_0^{min(U_0, s_1)} ds_0 exp(α_0 s_0)
    """
    α_0 = -0.4 + 0.0j
    α_1 = -0.6 + 0.0j
    L = 0.0
    U_0 = 5.0
    U_top = 20.0

    val = _chain_with_intermediate_uppers(
        [α_0, α_1], L, {0: U_0}, U_top,
    )
    assert val is not None

    # Direct closed-form:
    # Case s_1 ≤ U_0: integrand piece is exp(α_1 s_1)·(exp(α_0 s_1) - 1)/α_0
    # Case s_1 > U_0: integrand piece is exp(α_1 s_1)·(exp(α_0 U_0) - 1)/α_0
    # First: ∫_0^{U_0} (exp((α_0+α_1) s) - exp(α_1 s)) / α_0 ds
    #      = [(exp((α_0+α_1) U_0) - 1)/(α_0+α_1) - (exp(α_1 U_0) - 1)/α_1] / α_0
    # Second: (exp(α_0 U_0) - 1)/α_0 · (exp(α_1 U_top) - exp(α_1 U_0))/α_1

    part_1 = (((cmath.exp((α_0 + α_1) * U_0) - 1) / (α_0 + α_1))
              - ((cmath.exp(α_1 * U_0) - 1) / α_1)) / α_0
    part_2 = ((cmath.exp(α_0 * U_0) - 1) / α_0
              * (cmath.exp(α_1 * U_top) - cmath.exp(α_1 * U_0)) / α_1)
    expected = part_1 + part_2

    assert abs(val - expected) < 1e-12, (
        f'helper={val}, direct={expected}, |Δ|={abs(val - expected):.3e}'
    )


def test_m3_intermediate_upper_at_position_1_matches_recursive_quad():
    """m=3 with upper on position 1 (middle).  Cross-check against
    nested scipy.quad on the same chain integrand."""
    from scipy.integrate import quad

    α_0 = -0.3 + 0.0j
    α_1 = -0.5 + 0.0j
    α_2 = -0.7 + 0.0j
    L = 0.0
    U_1 = 3.0
    U_top = 8.0

    val = _chain_with_intermediate_uppers(
        [α_0, α_1, α_2], L, {1: U_1}, U_top,
    )
    assert val is not None

    def f0(s0):
        return cmath.exp(α_0 * s0)
    def f1(s1):
        re_v, _ = quad(lambda s0: (cmath.exp(α_1 * s1)
                                    * f0(s0)).real, L, s1)
        im_v, _ = quad(lambda s0: (cmath.exp(α_1 * s1)
                                    * f0(s0)).imag, L, s1)
        return complex(re_v, im_v)
    def f2_re(s2):
        re_v, _ = quad(lambda s1: (cmath.exp(α_2 * s2) * f1(s1)).real,
                       L, min(U_1, s2))
        return re_v
    def f2_im(s2):
        im_v, _ = quad(lambda s1: (cmath.exp(α_2 * s2) * f1(s1)).imag,
                       L, min(U_1, s2))
        return im_v

    # Outermost integration over s_2 ∈ [L, U_top]:
    re_total, _ = quad(f2_re, L, U_top,
                       epsrel=1e-9, epsabs=1e-12, limit=200)
    im_total, _ = quad(f2_im, L, U_top,
                       epsrel=1e-9, epsabs=1e-12, limit=200)
    ref = complex(re_total, im_total)
    assert abs(val - ref) < 1e-6 * max(1.0, abs(ref)), (
        f'helper={val}, scipy={ref}, |Δ|={abs(val - ref):.3e}'
    )


def test_m3_two_intermediate_uppers_matches_recursive_quad():
    """m=3 with uppers on positions 0 AND 1 (both non-maximal).
    Exercises 2^2 = 4 case enumeration."""
    from scipy.integrate import quad

    α_0 = -0.2 + 0.0j
    α_1 = -0.4 + 0.0j
    α_2 = -0.6 + 0.0j
    L = 0.0
    U_0 = 2.0
    U_1 = 4.0
    U_top = 10.0

    val = _chain_with_intermediate_uppers(
        [α_0, α_1, α_2], L, {0: U_0, 1: U_1}, U_top,
    )
    assert val is not None

    def inner_at_s1(s1):
        # ∫_0^{min(U_0, s1)} exp(α_0 s_0) ds_0
        upper_0 = min(U_0, s1)
        if upper_0 <= L:
            return 0.0 + 0.0j
        # ∫_L^{upper_0} exp(α_0 s) ds = (exp(α_0 upper_0) - exp(α_0 L)) / α_0
        return ((cmath.exp(α_0 * upper_0) - cmath.exp(α_0 * L)) / α_0)

    def mid_at_s2(s2):
        # ∫_0^{min(U_1, s2)} exp(α_1 s_1) · inner(s1) ds_1
        upper_1 = min(U_1, s2)
        if upper_1 <= L:
            return 0.0 + 0.0j

        def integrand_re(s1):
            return (cmath.exp(α_1 * s1) * inner_at_s1(s1)).real

        def integrand_im(s1):
            return (cmath.exp(α_1 * s1) * inner_at_s1(s1)).imag

        re_v, _ = quad(integrand_re, L, upper_1,
                       epsrel=1e-9, epsabs=1e-12, limit=200)
        im_v, _ = quad(integrand_im, L, upper_1,
                       epsrel=1e-9, epsabs=1e-12, limit=200)
        return complex(re_v, im_v)

    def outer_re(s2):
        return (cmath.exp(α_2 * s2) * mid_at_s2(s2)).real

    def outer_im(s2):
        return (cmath.exp(α_2 * s2) * mid_at_s2(s2)).imag

    re_total, _ = quad(outer_re, L, U_top,
                       epsrel=1e-9, epsabs=1e-12, limit=200)
    im_total, _ = quad(outer_im, L, U_top,
                       epsrel=1e-9, epsabs=1e-12, limit=200)
    ref = complex(re_total, im_total)
    assert abs(val - ref) < 1e-5 * max(1.0, abs(ref)), (
        f'helper={val}, scipy={ref}, |Δ|={abs(val - ref):.3e}'
    )


# ─── Empty-domain and edge cases ───────────────────────────────────
def test_empty_chain_returns_one():
    val = _chain_with_intermediate_uppers([], 0.0, {}, 1.0)
    assert val == 1.0 + 0.0j


def test_upper_below_lower_returns_zero():
    """If any effective upper is ≤ L, integration domain is empty."""
    val = _chain_with_intermediate_uppers(
        [-0.5 + 0.0j], 5.0, {0: 3.0}, 10.0,
    )
    assert val == 0.0 + 0.0j


def test_redundant_upper_short_circuits():
    """When U_intermediate ≥ U_top, the intermediate upper is redundant.
    Should give the same answer as no intermediate upper."""
    alphas = [-0.3 + 0.0j, -0.5 + 0.0j]
    L, U_top = 0.0, 5.0
    # U_0 = 10 > U_top = 5 — redundant.
    val_with = _chain_with_intermediate_uppers(
        alphas, L, {0: 10.0}, U_top,
    )
    val_without = _chain_with_intermediate_uppers(
        alphas, L, {}, U_top,
    )
    assert val_with is not None and val_without is not None
    assert abs(val_with - val_without) < 1e-13


# ─── Complex α (typical Hawkes pole) ──────────────────────────────
def test_complex_alphas_with_intermediate_upper():
    """Realistic Hawkes pole values with intermediate upper."""
    α_0 = -0.4 + 0.2j
    α_1 = -0.6 - 0.1j
    α_2 = -0.5 + 0.3j
    L = 0.0
    U_1 = 4.0
    U_top = 10.0

    val = _chain_with_intermediate_uppers(
        [α_0, α_1, α_2], L, {1: U_1}, U_top,
    )
    assert val is not None
    assert abs(val.real) > 0 or abs(val.imag) > 0  # non-trivial result

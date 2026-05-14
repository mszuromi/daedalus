"""
tests/test_1d_polytope_modesum.py
==================================
Direct unit tests for the m=1 analytic 1D interval integrator (Phase J
refactor Stage 4a-perdiag).  Each test poses a closed-form analytic
question and checks ``_integrate_1d_polytope_modesum`` returns the
right answer to machine precision.

Run with::

    sage -python -m pytest tests/test_1d_polytope_modesum.py -v
"""
from __future__ import annotations

import os
import sys
import cmath
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.time_domain.final_integral import (
    EdgeModeSum,
    _integrate_1d_polytope_modesum,
)


def _ems(modes):
    """EdgeModeSum stub with given (residue, pole) modes; other fields
    are placeholders since the 1D integrator reads only ``modes``."""
    return EdgeModeSum(
        ri=0, pi=0,
        delta_coeff=0.0 + 0.0j,
        modes=tuple(modes),
    )


# ─── Single-edge, single-pole, finite interval ─────────────────────
def test_single_pole_finite_interval():
    """∫_L^U C·exp(λ·(s - c0)) ds  with  Δt = s - c0.

    Constraint  s - c0 > 0  ⇒  L = c0.  Take U as a tighter bound
    from a second constraint  -s + u0 > 0  ⇒  U = u0.
    """
    C = 0.7 - 0.3j
    lam = -0.5 + 1.2j           # Re < 0 — retarded
    c0_edge_1 = -3.0            # Δt_1 = s + 3
    c0_edge_2 =  5.0            # Δt_2 = -s + 5
    # Per-edge mode lists.
    edges = [_ems([(C, lam)]),
             _ems([(1.0 + 0.0j, 0.0 + 0.0j)])]  # second edge contributes 1 (no s-dep in α_s)
    # subset_constraint_data: each row is (a_int, a_ext, c0).
    cdata = [
        ([1.0],  [], c0_edge_1),       # Δt_1 = s + (-3),  i.e. s > 3? No: c_eff = -3, constraint -3+s>0 → s>3
        ([-1.0], [], c0_edge_2),       # constraint -s + 5 > 0 → s < 5
    ]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
    )
    # Analytic: integrand at multi-index (0, 0) is
    #   C · exp(lam · (s + c0_edge_1)) · 1 · exp(0 · (-s + c0_edge_2))
    #   = C · exp(lam·s + lam·c0_edge_1)
    # α_s = lam · 1 + 0 · -1 = lam
    # γ   = lam · c0_edge_1 + 0 · c0_edge_2 = lam · (-3)
    L, U = 3.0, 5.0
    alpha_s = lam
    gamma = lam * c0_edge_1
    expected = C * (cmath.exp(alpha_s * U + gamma)
                    - cmath.exp(alpha_s * L + gamma)) / alpha_s
    assert val is not None
    assert abs(val - expected) < 1e-13


# ─── Single-edge, multi-pole, finite interval ──────────────────────
def test_multi_pole_sum_finite_interval():
    """Pole decomposition: integral is the sum over poles."""
    modes = [(1.0 + 0.0j, -0.5 + 1j),
             (0.3 + 0.2j, -0.8 - 0.4j),
             (-0.1 + 0.0j, -1.5 + 2j)]
    edges = [_ems(modes)]
    cdata = [
        ([1.0], [], 0.0),    # constraint s > 0
    ]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=2.0 - 1.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
        bbox_cap=50.0,       # not used because the helper now tracks ∞
    )
    # Expected: sum over poles of pref · C · (∫_0^∞ exp(λ·s) ds)
    # = sum over poles of pref · C · (-1/λ)   (since exp(λ·∞) = 0 for Re λ < 0)
    pref = 2.0 - 1.0j
    expected = sum(pref * C * (-1.0 / lam) for (C, lam) in modes)
    assert val is not None
    assert abs(val - expected) < 1e-13


# ─── Multi-edge Cartesian (multi-index expansion) ──────────────────
def test_two_edges_cartesian_product():
    """Two smooth edges, each with two poles, multi-index expansion."""
    modes_a = [(1.0 + 0.0j, -0.5 + 0.7j),
               (0.4 - 0.1j, -1.2 + 0.0j)]
    modes_b = [(0.9 + 0.0j, -0.3 + 0.5j),
               (0.2 + 0.3j, -1.0 - 0.2j)]
    edges = [_ems(modes_a), _ems(modes_b)]
    # Both edges have a_int_e = 1.0; constraint s > 0 (lower bound).
    cdata = [
        ([1.0], [], 0.0),
        ([1.0], [], 0.0),
    ]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
    )
    # Cartesian expected: sum over (a, b) of C_a · C_b · ∫_0^∞ exp((λ_a+λ_b)·s) ds
    # = sum over (a, b) of C_a · C_b · (-1/(λ_a + λ_b))
    expected = sum(
        Ca * Cb * (-1.0 / (la + lb))
        for (Ca, la) in modes_a
        for (Cb, lb) in modes_b
    )
    assert val is not None
    assert abs(val - expected) < 1e-13


# ─── Override via pole_tuples (grouped path API) ───────────────────
def test_pole_tuples_override():
    """The pole_tuples parameter must take precedence over the
    Cartesian iterator built from smooth_edge_modes.  Inject custom
    coefficients (the merged-residue tensor in real grouped use)."""
    edges = [_ems([(1.0 + 0.0j, -0.5 + 0.0j)]),
             _ems([(1.0 + 0.0j, -0.7 + 0.0j)])]
    cdata = [
        ([1.0], [], 0.0),
        ([1.0], [], 0.0),
    ]
    # Override yields a single multi-index with coefficient 42 + 0j.
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
        pole_tuples=[(42.0 + 0.0j, (-0.5 + 0.0j, -0.7 + 0.0j))],
    )
    # Expected: 42 · (-1/(λ_a + λ_b)) = 42 · (-1/(-1.2)) = 35
    assert val is not None
    assert abs(val - 35.0) < 1e-13


# ─── Empty interval (L >= U) ────────────────────────────────────────
def test_empty_interval_returns_zero():
    """Mutually inconsistent constraints  s > 5  AND  s < 2."""
    edges = [_ems([(1.0 + 0.0j, -0.5 + 0.0j)]),
             _ems([(1.0 + 0.0j, -0.5 + 0.0j)])]
    cdata = [
        ([1.0],  [], -5.0),   # s > 5
        ([-1.0], [], 2.0),    # s < 2
    ]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
    )
    assert val == 0.0 + 0.0j


# ─── Infeasible constant constraint ────────────────────────────────
def test_infeasible_constant_constraint_returns_zero():
    """A constraint with a=0 and c_eff <= 0 makes the polytope empty."""
    edges = [_ems([(1.0 + 0.0j, -0.5 + 0.0j)]),
             _ems([(1.0 + 0.0j, -0.5 + 0.0j)])]
    cdata = [
        ([0.0], [], -1.0),    # 0·s - 1 > 0  ⇒  infeasible
        ([1.0], [], 0.0),
    ]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
    )
    assert val == 0.0 + 0.0j


# ─── α_s ≈ 0: constant integrand ────────────────────────────────────
def test_alpha_s_zero_constant_integrand():
    """When all a_int_e[0] are zero except for one edge whose pole
    is also zero, α_s = 0 and integrand is constant in s."""
    # One edge with non-trivial pole but a_int = 0 (no s-dependence)
    # plus another edge with pole = 0 and a_int = 1 (also no s-dep in α_s).
    edges = [_ems([(0.5 + 0.0j, 0.0 + 0.0j)])]
    cdata = [
        ([1.0], [], 0.0),     # constraint s > 0 ⇒ L=0
    ]
    # A second constraint to bound s from above.
    cdata.append(([-1.0], [], 5.0))   # s < 5 ⇒ U=5
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=2.0 + 0.0j,
        subset_constraint_data=cdata[:1],   # use only the first edge / constraint
        free_ext_vals=[],
    )
    # Add the s<5 constraint as a second edge entry of length matching modes.
    # Easier: re-do with two edges, both poles 0, a_int 1 and -1.
    edges2 = [_ems([(1.0 + 0.0j, 0.0 + 0.0j)]),
              _ems([(1.0 + 0.0j, 0.0 + 0.0j)])]
    val2 = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges2,
        prefactor_complex=3.0 + 0.0j,
        subset_constraint_data=[
            ([1.0],  [], 0.0),    # s > 0
            ([-1.0], [], 5.0),    # s < 5
        ],
        free_ext_vals=[],
    )
    # α_s = 0·1 + 0·(-1) = 0; γ = 0; integrand = 3 · 1 · 1 · exp(0) = 3
    # ∫_0^5 3 ds = 15.
    assert val2 is not None
    assert abs(val2 - 15.0) < 1e-13


# ─── Free-ext-time dependence ──────────────────────────────────────
def test_free_ext_time_in_c_eff():
    """c_ext for each edge includes the external-time contributions.

    Constraint  s > t_free[0]  ⇒  L = t_free[0].
    Smooth-factor argument depends on t_free[0] too.
    """
    edges = [_ems([(1.0 + 0.0j, -0.4 + 0.0j)])]
    cdata = [
        ([1.0], [-1.0], 0.0),    # s - t_free[0] > 0  ⇒ L = t_free[0]
    ]
    free = [2.5]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=free,
    )
    # Δt = s - 2.5; α_s = -0.4; γ = -0.4 · (-2.5) = 1.0
    # ∫_2.5^∞ exp(-0.4 s + 1.0) ds = exp(1.0) · (- exp(-0.4·2.5) / -0.4)
    #   = exp(1) · exp(-1) / 0.4 = 1/0.4 = 2.5
    assert val is not None
    assert abs(val - 2.5) < 1e-13


# ─── Divergence: unbounded interval with wrong-sign Re(α_s) ────────
def test_divergent_unbounded_returns_none():
    """L = -∞ with Re(α_s) < 0 → divergent integral; should return None
    so the caller falls back to scipy."""
    edges = [_ems([(1.0 + 0.0j, -0.5 + 0.0j)])]   # Re(λ) < 0
    cdata = [
        ([-1.0], [], 5.0),   # -s + 5 > 0 ⇒ s < 5; no lower bound
    ]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
    )
    # a_int = -1, so α_s = -0.5 · -1 = +0.5  (Re > 0).
    # L = -∞, U = 5.  At -∞, exp(0.5·s) → 0 ✓.  Should NOT diverge.
    # Let me invert sign: take λ = +0.5 (Re > 0) so α_s = +0.5 · -1 = -0.5
    # then exp(-0.5·s) → ∞ at -∞ → diverges.
    edges_div = [_ems([(1.0 + 0.0j, 0.5 + 0.0j)])]
    val_div = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges_div,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
    )
    assert val_div is None
    # And the convergent case returns a finite value.
    assert val is not None


# ─── Smoke: zero coefficient short-circuit ─────────────────────────
def test_zero_coefficient_short_circuit():
    """When C = 0 for every pole, the integral is zero."""
    edges = [_ems([(0.0 + 0.0j, -0.5 + 0.0j),
                   (0.0 + 0.0j, -0.7 + 0.0j)])]
    cdata = [
        ([1.0], [], 0.0),
    ]
    val = _integrate_1d_polytope_modesum(
        smooth_edge_modes=edges,
        prefactor_complex=1.0 + 0.0j,
        subset_constraint_data=cdata,
        free_ext_vals=[],
    )
    assert val == 0.0 + 0.0j

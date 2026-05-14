"""
tests/test_polygon_m2_integrator.py
====================================
Direct unit tests for the m=2 polygon integrator (Phase J refactor
Stage 3a-full).  These cover the geometry primitives and the
analytic-over-triangle formula in isolation, with known-answer test
cases.  They run independently of the full Phase J pipeline so
correctness is established before the integrator is wired in.

Run with::

    sage -python -m pytest tests/test_polygon_m2_integrator.py -v
"""
from __future__ import annotations

import os
import sys
import cmath

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.time_domain.final_integral import (
    EdgeModeSum,
    _exp_over_unit_triangle,
    _exp_over_triangle,
    _clip_polygon_to_halfplane,
    _polygon_from_2d_constraints,
    _enumerate_pole_tuples,
    _integrate_2d_polygon_modesum,
)


# ───────────────────────────────────────────────────────────────────
# J(p, q) = ∫₀¹ du ∫₀^{1-u} dw exp(p·u + q·w)
# ───────────────────────────────────────────────────────────────────

def _J_numerical(p, q, n=400):
    """Reference J(p, q) via Riemann sum on the unit triangle.

    Slow but transparent; used to sanity-check the closed form."""
    du = 1.0 / n
    total = 0.0 + 0.0j
    for i in range(n):
        u = (i + 0.5) * du
        # w ∈ [0, 1-u], split that range into n bins
        if 1 - u <= 0:
            continue
        dw = (1 - u) / n
        for j in range(n):
            w = (j + 0.5) * dw
            total += cmath.exp(p * u + q * w) * du * dw
    return total


def test_J_zero_zero_is_half():
    """J(0, 0) = area of unit triangle = 1/2."""
    assert abs(_exp_over_unit_triangle(0.0, 0.0) - 0.5) < 1e-12


def test_J_taylor_branch_matches_numerical_for_small_pq():
    """For tiny p, q the Taylor expansion should agree with a direct
    Riemann-sum reference to ~1e-7."""
    p = 1e-7 + 2e-7j
    q = -3e-7 + 1e-7j
    val = _exp_over_unit_triangle(p, q)
    ref = _J_numerical(p, q, n=200)
    assert abs(val - ref) < 1e-9, f'val={val}, ref={ref}'


def test_J_general_real_values():
    """J(1, 2) has the closed form (e − 1)² / 2 ≈ 1.4762."""
    expected = (cmath.e - 1) ** 2 / 2
    val = _exp_over_unit_triangle(1.0, 2.0)
    assert abs(val - expected) < 1e-12


def test_J_symmetric_under_pq_swap():
    """J(p, q) = J(q, p) by u↔w symmetry of the unit triangle."""
    for p, q in [(0.7, 1.3), (2.0 + 1j, -0.5j), (0.1, 0.1)]:
        a = _exp_over_unit_triangle(p, q)
        b = _exp_over_unit_triangle(q, p)
        assert abs(a - b) < 1e-12, f'p={p}, q={q}: a={a}, b={b}'


def test_J_q_zero_branch():
    """J(p, 0) = (e^p − 1 − p) / p²."""
    p = 1.5
    val = _exp_over_unit_triangle(p, 0.0)
    expected = (cmath.exp(p) - 1 - p) / (p * p)
    assert abs(val - expected) < 1e-12


def test_J_pq_equal_branch():
    """J(p, p) = ((p − 1) e^p + 1) / p²."""
    p = 0.8
    val = _exp_over_unit_triangle(p, p)
    expected = ((p - 1) * cmath.exp(p) + 1) / (p * p)
    assert abs(val - expected) < 1e-12


def test_J_complex_general():
    """Complex p, q: verify against direct Riemann sum (loose tolerance
    since the reference is ~n^-2 accurate)."""
    p = 0.5 + 0.7j
    q = -0.3 + 1.2j
    val = _exp_over_unit_triangle(p, q)
    ref = _J_numerical(p, q, n=200)
    # n=200 Riemann on a 2D triangle has ~1e-4 error.
    assert abs(val - ref) < 1e-3, f'val={val}, ref={ref}'


# ───────────────────────────────────────────────────────────────────
# ∫∫_T exp(α·x + β·y) dA over an arbitrary triangle
# ───────────────────────────────────────────────────────────────────

def test_exp_over_triangle_unit_triangle_matches_J():
    """For the triangle (0,0), (1,0), (0,1) the integral is exactly
    J(α, β)."""
    alpha = 0.3
    beta = -0.7
    val = _exp_over_triangle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                             alpha, beta)
    expected = _exp_over_unit_triangle(alpha, beta)
    assert abs(val - expected) < 1e-12


def test_exp_over_triangle_constant_integrand_equals_area():
    """α = β = 0 ⇒ integrand is 1, integral = triangle area."""
    # Right triangle with legs 2 and 3 → area 3.
    val = _exp_over_triangle((0.0, 0.0), (2.0, 0.0), (0.0, 3.0),
                             0.0, 0.0)
    assert abs(val - 3.0) < 1e-12


def test_exp_over_triangle_translation_invariance():
    """Translating the triangle by (h_x, h_y) multiplies the
    integral by exp(α·h_x + β·h_y)."""
    alpha = 0.4 + 0.1j
    beta = -0.2 + 0.5j
    base = _exp_over_triangle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                              alpha, beta)
    shifted = _exp_over_triangle(
        (5.0, 7.0), (6.0, 7.0), (5.0, 8.0), alpha, beta,
    )
    expected = base * cmath.exp(alpha * 5.0 + beta * 7.0)
    assert abs(shifted - expected) < 1e-10


# ───────────────────────────────────────────────────────────────────
# Sutherland-Hodgman polygon clipping
# ───────────────────────────────────────────────────────────────────

def test_clip_polygon_keeps_entirely_inside_intact():
    """Polygon entirely in the keep half-space: vertices unchanged."""
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    # Keep x > -10 (everything satisfies it)
    clipped = _clip_polygon_to_halfplane(square, 1.0, 0.0, 10.0)
    assert clipped == square


def test_clip_polygon_drops_entirely_outside():
    """Polygon entirely in the reject half-space: empty."""
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    # Keep x > 10 (no point satisfies it)
    clipped = _clip_polygon_to_halfplane(square, 1.0, 0.0, -10.0)
    assert clipped == []


def test_clip_polygon_partial_keep_half_unit_square():
    """Clip the unit square against x > 0.5: result is the right
    half [0.5, 1] × [0, 1], area 0.5."""
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    clipped = _clip_polygon_to_halfplane(square, 1.0, 0.0, -0.5)
    # Polygon area via shoelace.
    n = len(clipped)
    assert n >= 3
    area = 0.0
    for i in range(n):
        x0, y0 = clipped[i]
        x1, y1 = clipped[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    area = abs(area) / 2
    assert abs(area - 0.5) < 1e-12


# ───────────────────────────────────────────────────────────────────
# Polygon from constraints
# ───────────────────────────────────────────────────────────────────

def test_polygon_from_constraints_unit_square():
    """Constraints  s_0 > 0,  s_1 > 0,  -s_0 > -1,  -s_1 > -1
    define [0, 1]²."""
    constraints = [
        ([1.0, 0.0], [], 0.0),   #  s_0 > 0
        ([0.0, 1.0], [], 0.0),   #  s_1 > 0
        ([-1.0, 0.0], [], 1.0),  # -s_0 + 1 > 0 ⟺ s_0 < 1
        ([0.0, -1.0], [], 1.0),  # -s_1 + 1 > 0 ⟺ s_1 < 1
    ]
    polygon = _polygon_from_2d_constraints(constraints, [], bbox_cap=10.0)
    # Compute polygon area; should be 1.0 (unit square).
    n = len(polygon)
    assert n >= 3
    area = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    area = abs(area) / 2
    assert abs(area - 1.0) < 1e-10, f'polygon={polygon}, area={area}'


def test_polygon_from_constraints_with_ext_time():
    """Constraint  s_0 + t_ext > 0  ⇔  s_0 > -t_ext.  With t_ext=2
    and the bounding box ±5, the polygon spans s_0 ∈ [-2, 5]."""
    constraints = [
        ([1.0, 0.0], [1.0], 0.0),   # s_0 + 1·t_ext + 0 > 0
    ]
    polygon = _polygon_from_2d_constraints(
        constraints, free_ext_vals=[2.0], bbox_cap=5.0,
    )
    # Polygon is a (5 + 2) × 10 rectangle.  Area = 7 × 10 = 70.
    n = len(polygon)
    area = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    area = abs(area) / 2
    assert abs(area - 70.0) < 1e-10


# ───────────────────────────────────────────────────────────────────
# Pole-tuple enumeration
# ───────────────────────────────────────────────────────────────────

def test_enumerate_pole_tuples_two_edges_two_modes():
    """Two edges, each with two modes — yields 4 tuples in lex order."""
    ems_list = [
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, 0.1j), (2.0 + 0j, 0.2j))),
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((3.0 + 0j, 0.3j), (4.0 + 0j, 0.4j))),
    ]
    out = list(_enumerate_pole_tuples(ems_list))
    assert len(out) == 4
    # Products of residues: 1·3, 2·3, 1·4, 2·4 (inner-edge varies fastest).
    # Compare as multisets keyed by real part (residues are all real here).
    got_products = sorted(p.real for (p, _) in out)
    expected = sorted([3.0, 6.0, 4.0, 8.0])
    assert got_products == expected


def test_enumerate_pole_tuples_empty_yields_one():
    """No edges → exactly one empty tuple."""
    out = list(_enumerate_pole_tuples([]))
    assert len(out) == 1
    assert out[0] == (1.0 + 0j, ())


# ───────────────────────────────────────────────────────────────────
# End-to-end polygon integrator vs scipy.nquad reference
# ───────────────────────────────────────────────────────────────────

def test_polygon_integrator_matches_scipy_simple_case():
    """Two smooth edges, two retardation constraints defining a
    triangle in (s_0, s_1) space; compare the analytic polygon-mode-
    sum result to scipy.integrate.nquad on the equivalent closure.

    Constraints: s_0 > 0, s_1 - s_0 > 0  (so polygon = {0 < s_0 < s_1},
    capped at bbox).  Single mode per edge.

    Mode-sum:
      edge 0:  Δt_0 = s_0       → 1 mode with C=1, λ=-1 (decay)
      edge 1:  Δt_1 = s_1 - s_0 → 1 mode with C=1, λ=-2 (decay)
    Integrand = exp(-s_0) · exp(-2(s_1 - s_0)) = exp(s_0 - 2 s_1)
    """
    from scipy.integrate import nquad
    smooth_edge_modes = [
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, -1.0 + 0j),)),
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, -2.0 + 0j),)),
    ]
    # Constraint data: (a_int, a_ext, c0).  a_int is the linear form
    # for Δt in (s_0, s_1).
    subset_constraint_data = [
        ([1.0, 0.0], [], 0.0),    # Δt_0 = s_0
        ([-1.0, 1.0], [], 0.0),   # Δt_1 = s_1 - s_0
    ]
    free_ext_vals = []
    poly_val = _integrate_2d_polygon_modesum(
        smooth_edge_modes,
        prefactor_complex=1.0 + 0j,
        subset_constraint_data=subset_constraint_data,
        free_ext_vals=free_ext_vals,
        bbox_cap=50.0,  # big enough that e^{-50} < 1e-21 — negligible
    )

    # Reference: scipy.nquad over the same domain (capped at 50).
    import math

    def integrand(s_0, s_1):
        return math.exp(s_0 - 2 * s_1)

    def bounds_s0(s_1):
        # s_0 ∈ (0, s_1) intersected with bbox
        return (max(0.0, -50.0), min(s_1, 50.0))

    ref, _ = nquad(integrand, [bounds_s0, (0.0, 50.0)],
                    opts={'limit': 200})
    # Closed form: ∫_0^∞ ∫_0^{s_1} exp(-s_0 - 2(s_1-s_0)) ds_0 ds_1
    #            = ∫_0^∞ ∫_0^{s_1} exp(s_0 - 2 s_1) ds_0 ds_1
    #            = ∫_0^∞ exp(-2 s_1) · (exp(s_1) - 1) ds_1
    #            = ∫_0^∞ (exp(-s_1) - exp(-2 s_1)) ds_1
    #            = 1 - 1/2 = 0.5
    assert abs(poly_val - 0.5) < 1e-6, f'poly={poly_val}, expected 0.5'
    assert abs(poly_val - ref) < 1e-4, f'poly={poly_val}, scipy={ref}'


def test_polygon_integrator_two_poles_per_edge_vs_scipy():
    """Two smooth edges, EACH with two poles — exercises the pole-
    expansion machinery (2² = 4 single-exp terms).  Compares analytic
    polygon-modesum to scipy.nquad on the equivalent un-expanded
    closure.

    Each edge's smooth factor is  C₁ exp(λ₁·Δt) + C₂ exp(λ₂·Δt) ,
    which is the realistic shape of a multi-pole retarded propagator.
    Constraints  s₀ > 0,  s₁ - s₀ > 0  → polygon = {0 < s₀ < s₁ <
    bbox}.
    """
    from scipy.integrate import nquad
    import math

    # Edge 0: residues C and poles p chosen so the smooth factor
    # decays as e^{-s_0} + 0.3 e^{-3 s_0}.  λ = i·p in our
    # convention, so to get e^{-x} we need λ = -1.  We pass the
    # final λ-values directly into modes; the integrator treats
    # them as the EdgeModeSum's λ_α field.
    smooth_edge_modes = [
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, -1.0 + 0j),
                           (0.3 + 0j, -3.0 + 0j))),
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((0.7 + 0j, -2.0 + 0j),
                           (0.5 + 0j, -4.0 + 0j))),
    ]
    subset_constraint_data = [
        ([1.0, 0.0], [], 0.0),    # Δt_0 = s_0
        ([-1.0, 1.0], [], 0.0),   # Δt_1 = s_1 - s_0
    ]
    free_ext_vals = []
    poly_val = _integrate_2d_polygon_modesum(
        smooth_edge_modes,
        prefactor_complex=1.0 + 0j,
        subset_constraint_data=subset_constraint_data,
        free_ext_vals=free_ext_vals,
        bbox_cap=50.0,
    )

    # Reference via scipy.nquad on the explicit closure.
    def integrand(s_0, s_1):
        dt_0 = s_0
        dt_1 = s_1 - s_0
        # Each edge: Σ_k C_k · exp(λ_k · Δt_e)
        edge0 = (1.0 * math.exp(-1.0 * dt_0)
                 + 0.3 * math.exp(-3.0 * dt_0))
        edge1 = (0.7 * math.exp(-2.0 * dt_1)
                 + 0.5 * math.exp(-4.0 * dt_1))
        return edge0 * edge1

    def bounds_s0(s_1):
        return (max(0.0, -50.0), min(s_1, 50.0))

    ref, _ = nquad(integrand, [bounds_s0, (0.0, 50.0)],
                    opts={'limit': 200})
    assert abs(poly_val - ref) < 1e-4, \
        f'poly={poly_val}, scipy={ref}, |Δ|={abs(poly_val - ref):.3e}'


def test_polygon_integrator_with_external_time_dependence():
    """Constraints involve a free external time; verify the polygon
    shifts with the external value and the integrator handles it
    correctly.

    Δt_0 = s_0 - t_ext  ⇒  retardation s_0 > t_ext.
    Δt_1 = s_1 - s_0    ⇒  retardation s_1 > s_0.

    Single mode per edge, decay rates -1 and -2.
    """
    from scipy.integrate import nquad
    import math

    smooth_edge_modes = [
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, -1.0 + 0j),)),
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, -2.0 + 0j),)),
    ]
    subset_constraint_data = [
        ([1.0, 0.0], [-1.0], 0.0),   # Δt_0 = s_0 + (-1)·t_ext + 0
        ([-1.0, 1.0], [0.0], 0.0),   # Δt_1 = s_1 - s_0
    ]
    t_ext = 1.5
    free_ext_vals = [t_ext]
    poly_val = _integrate_2d_polygon_modesum(
        smooth_edge_modes,
        prefactor_complex=1.0 + 0j,
        subset_constraint_data=subset_constraint_data,
        free_ext_vals=free_ext_vals,
        bbox_cap=50.0,
    )

    def integrand(s_0, s_1):
        dt_0 = s_0 - t_ext
        dt_1 = s_1 - s_0
        return math.exp(-1.0 * dt_0) * math.exp(-2.0 * dt_1)

    def bounds_s0(s_1):
        return (max(t_ext, -50.0), min(s_1, 50.0))

    ref, _ = nquad(integrand, [bounds_s0, (t_ext, 50.0)],
                    opts={'limit': 200})
    assert abs(poly_val - ref) < 1e-4, \
        f'poly={poly_val}, scipy={ref}, t_ext={t_ext}'

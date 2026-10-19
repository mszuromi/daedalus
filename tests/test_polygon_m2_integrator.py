"""
tests/test_polygon_m2_integrator.py
====================================
Direct unit tests for the m=2 polygon integrator (Phase J refactor
Stage 3a-full): the pole-tuple enumeration and the integrator against
scipy references.  (The tests of the pre-0.2.0 clipped-polygon primitives,
``_exp_over_unit_triangle``, ``_exp_over_triangle``,
``_clip_polygon_to_halfplane`` and ``_polygon_from_2d_constraints``, went
with them when audit W7 retired the boxed routes; the box-free region and
its kernels are tested in tests/test_phase_j_box_free.py and
tests/test_expdd.py.)

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

from engine.integration.time_domain.final_integral import (
    EdgeModeSum,
    _enumerate_pole_tuples,
    _integrate_2d_polygon_modesum,
)


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
    the box-free region; the scipy reference is capped at 50).  Single mode per edge.

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
    assert abs(poly_val - ref) < 1e-12, \
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
    )

    def integrand(s_0, s_1):
        dt_0 = s_0 - t_ext
        dt_1 = s_1 - s_0
        return math.exp(-1.0 * dt_0) * math.exp(-2.0 * dt_1)

    def bounds_s0(s_1):
        return (max(t_ext, -50.0), min(s_1, 50.0))

    ref, _ = nquad(integrand, [bounds_s0, (t_ext, 50.0)],
                    opts={'limit': 200})
    assert abs(poly_val - ref) < 1e-12, \
        f'poly={poly_val}, scipy={ref}, t_ext={t_ext}'

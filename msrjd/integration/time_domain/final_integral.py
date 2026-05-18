"""
msrjd.integration.time_domain.final_integral
=============================================
Vertex-time integration on a tree-level diagram via explicit numerical
quadrature, with proper handling of the δ(t) component of any
"instantaneous" propagator entry.

MVP scope
---------
Only tree-level (loop_number == 0) typed diagrams are handled. Tree-level
is the right proving ground for the Phase J evaluation layer because it
exercises:

- time-domain retarded propagator lookup per edge,
- polytope extraction from explicit Heaviside factors,
- **δ-edge subset enumeration** (δ(t) components of instantaneous
  couplings like `ñ × δn` in MSR-JD),
- vertex-time integration over the retarded polytope,
- global translation invariance / origin pinning,
- numerical dispatch,

**without** needing the kernel reduction / caching / contraction
machinery. Loop cases are deferred to Extension 1.

Integration strategy
--------------------
The retarded propagator for an edge `(u -> v)` with matrix index
`(phys=p, resp=r)` decomposes into two pieces:

    G_R[p, r](t_v - t_u)
      =  delta_coeff[p, r] · δ(t_v - t_u)
       + Θ(t_v - t_u) · smooth[p, r](t_v - t_u)

The δ piece encodes an instantaneous response (nonzero whenever the
frequency-domain entry has a nonzero `ω → ∞` limit, which is the case
for any instantaneous coupling in the MSR-JD action). The smooth piece
is the usual pole-residue sum.

For a tree diagram with `|E|` edges, the full integrand is a product
of `|E|` such factors. Expanding the product yields `2^|E|` terms; we
enumerate them by picking a subset `S ⊆ edges` to take in its δ form
and the complement in its smooth form:

  Σ_{S ⊆ edges}  ∫ ds_1 … ds_m
                 · (∏_{e ∈ S} delta_coeff[e] · δ(t_{v_e} - t_{u_e}))
                 · (∏_{e ∉ S} Θ(t_{v_e} - t_{u_e}) · smooth[e](t_{v_e} - t_{u_e}))
                 · combined_prefactor

For each subset:

1. **Delta-edge equations** `t_{v_e} − t_{u_e} = 0` for `e ∈ S` are
   solved to eliminate integration variables by substitution. In the
   MVP star-tree case (single source vertex, all edges connecting source
   to leaves), one δ-edge pins the source time to a specific leaf time;
   two or more δ-edges force equality among several leaf times, which
   is a shot-noise δ(τ=0) contribution and is **skipped** for the
   continuous-callable return type.

2. **Smooth edges** contribute both a `fast_callable`-compiled factor
   (JIT'd over `CDF`) and a linear retardation constraint `t_v > t_u`
   that's added to the polytope.

3. The per-subset contribution is then a numerical quadrature over the
   reduced set of integration variables with the reduced polytope.

4. All subset contributions are summed into the final callable.

The public entry point `integrate_tree_diagram` returns a Python
callable `contribution(*ext_time_values) -> complex`.

Numerical overflow mitigation
-----------------------------
Each per-subset smooth product is `.expand()`'d into a sum of
single-exponential terms before JIT-compilation, so
`scipy.integrate.quad` cannot overflow IEEE doubles when sampling at
large negative `s`. (See the 2026-04-08 overflow fix in the CHANGELOG.)
"""

import math
import functools as _functools
from dataclasses import dataclass

from sage.all import SR, fast_callable, CDF, solve as sage_solve

from msrjd.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
    G_t_delta_coeff,
)
from msrjd.core.vertices import NoiseSourceType, ConvVertexType


# ───────────────────────────────────────────────────────────────────────
# Canonical mode-sum representation of a propagator edge
# ───────────────────────────────────────────────────────────────────────
# Every retarded-propagator edge in a Feynman diagram decomposes as:
#
#     G_R[pi, ri](Δt)  =  delta_coeff · δ(Δt)
#                       + Θ(Δt) · Σ_α  C_α · exp(λ_α · Δt)
#
# where ``λ_α = i·p_α`` for our Fourier convention and ``C_α`` is the
# residue of the propagator matrix at pole ``p_α`` at position (pi, ri).
#
# The smooth-part data is currently re-extracted inside
# ``_build_fast_subset_evaluator`` on every (diagram, subset) call,
# even though it depends only on the (pi, ri) of the edge.  Lifting
# the extraction to a per-edge step at the top of
# ``integrate_diagram`` eliminates the redundant work and gives the
# downstream integrators a clean, JSON-able data structure to consume.
#
# Spatial-extension hook: in a future spatial theory ``λ_α`` and
# ``C_α`` become callables of momentum ``k`` rather than complex
# scalars.  The integrator backends will then gain an outer loop over
# momentum, but the EdgeModeSum interface stays the same.

@dataclass(frozen=True)
class EdgeModeSum:
    """Numerical mode-sum representation of one propagator edge.

    Fields
    ------
    ri, pi : int
        Response-column and physical-row indices into the propagator
        matrix.  Convention: ``G[pi, ri] = ⟨φ_pi  ñ_ri⟩``.
    delta_coeff : complex
        Coefficient of the δ(Δt) component (= the ω → ∞ limit of
        ``G_FT[pi, ri]``, captured in ``propagator_data['D_delta']``).
    modes : tuple of (complex C_α, complex λ_α)
        Pole-residue pairs.  For our codebase's Fourier convention
        (e^{-iωt}, retarded poles in Im(ω) > 0), ``λ_α = i · p_α``
        where ``p_α ∈ propagator_data['pole_vals']`` and
        ``C_α = C_mats[α][pi, ri]``.  The smooth-time-domain
        propagator is ``Σ_α C_α · exp(λ_α · Δt)``.
    dt_c0, dt_int_pairs, dt_ext_pairs :
        Sparse linear form for ``Δt`` in terms of (integration vars,
        free external times).  Filled in at the per-subset stage —
        leave empty in the per-edge build, populate when the subset
        is known.  Kept here so the same EdgeModeSum carries through
        the whole evaluation chain.
    """
    ri: int
    pi: int
    delta_coeff: complex
    modes: tuple                 # tuple[tuple[complex, complex], ...]
    dt_c0: float = 0.0
    dt_int_pairs: tuple = ()     # tuple[tuple[int, float], ...]
    dt_ext_pairs: tuple = ()     # tuple[tuple[int, float], ...]


def _build_edge_mode_sums(edge_info, propagator_data):
    """Build one EdgeModeSum per entry of ``edge_info`` by extracting
    the per-pole residue from ``propagator_data['C_mats']`` ONCE per
    edge.

    The ``dt_*`` fields are NOT populated here — those depend on
    which integration variables survive δ-elimination in each subset
    and are filled in at the subset level (see
    ``_attach_subset_dt`` below).

    Returns a list parallel to ``edge_info`` (same length, same
    order).  If the propagator data is incomplete (missing ``pole_vals``
    or ``C_mats``), returns ``None`` to signal that the caller should
    fall back to the SR-symbolic path.
    """
    pole_vals = propagator_data.get('pole_vals')
    C_mats = propagator_data.get('C_mats')
    if pole_vals is None or C_mats is None:
        return None

    # Convert poles to complex once.
    try:
        modes_lambdas = tuple(complex(CDF(SR(p))) * 1j for p in pole_vals)
    except Exception:
        return None
    n_poles = len(modes_lambdas)

    edge_mode_sums = []
    for ei in edge_info:
        ri, pi = ei['ri'], ei['pi']
        try:
            residues = tuple(
                complex(CDF(SR(C_mats[k][pi, ri])))
                for k in range(n_poles)
            )
        except Exception:
            return None
        modes = tuple(zip(residues, modes_lambdas))
        try:
            d_c = complex(ei['delta_coeff'])
        except Exception:
            try:
                d_c = complex(CDF(SR(ei['delta_coeff'])))
            except Exception:
                return None
        edge_mode_sums.append(EdgeModeSum(
            ri=ri, pi=pi,
            delta_coeff=d_c,
            modes=modes,
        ))
    return edge_mode_sums


def _extract_exp_mode(sr_expr, tau_sym):
    """Extract ``(C, λ)`` from an SR expression of the form
    ``C · exp(λ · tau_sym)`` (single-exponential kernel).

    Uses the log-derivative trick: λ = d/dτ log(g(τ)) evaluated at
    τ=0, then C = g(0).  Works for any single-exponential expression
    after Heaviside-stripping and num-params substitution.

    Returns ``(C_complex, λ_complex)`` or ``None`` if the expression
    can't be reduced to this form (e.g. polynomial-prefactor kernels
    like the alpha kernel ``τ/τ_g² · exp(-τ/τ_g)``, which would need
    a multi-mode decomposition).
    """
    try:
        c_at_zero = sr_expr.subs({tau_sym: 0})
        C = complex(CDF(SR(c_at_zero)))
    except Exception:
        return None
    if C == 0:
        # Polynomial-prefactor kernel (e.g. alpha) — single-exponential
        # extraction doesn't apply.  Caller should fall back to the
        # SR + scipy path until multi-mode kernels are supported.
        return None
    try:
        deriv = sr_expr.diff(tau_sym)
        lam_sr = (deriv / sr_expr).subs({tau_sym: 0})
        lam = complex(CDF(SR(lam_sr)))
    except Exception:
        return None
    return (C, lam)


def _attach_subset_dt(edge_mode_sum, a_int, a_ext, c0):
    """Return a copy of ``edge_mode_sum`` with the per-subset Δt
    linear form (sparse coefficients on integration vars + free
    external times, plus constant) populated.
    """
    int_pairs = tuple(
        (i, float(a)) for i, a in enumerate(a_int)
        if abs(float(a)) > 1e-15
    )
    ext_pairs = tuple(
        (i, float(a)) for i, a in enumerate(a_ext)
        if abs(float(a)) > 1e-15
    )
    return EdgeModeSum(
        ri=edge_mode_sum.ri,
        pi=edge_mode_sum.pi,
        delta_coeff=edge_mode_sum.delta_coeff,
        modes=edge_mode_sum.modes,
        dt_c0=float(c0),
        dt_int_pairs=int_pairs,
        dt_ext_pairs=ext_pairs,
    )


# ───────────────────────────────────────────────────────────────────────
# Analytic ∫∫_polygon exp(α·x + β·y) dA  (Stage 3a-full)
# ───────────────────────────────────────────────────────────────────────
# Replaces scipy.nquad on the m=2 polytope.  The integrand factors
# after pole-expansion as a sum of single-exponential terms
# A·exp(α·s_0 + β·s_1 + γ).  Each term is integrated analytically:
#
#   1. The polytope is a convex polygon (intersection of half-planes
#      from the retardation constraints).  Computed once per
#      (subset, τ-point) via Sutherland-Hodgman clipping.
#   2. Fan-triangulate the polygon from vertex 0.
#   3. Per triangle: affine-map to the unit triangle 0 ≤ u, w ≤ 1,
#      u + w ≤ 1.  The integrand reduces to ``exp(α₀ + p·u + q·w)``
#      with α₀, p, q expressible from (α, β, vertices).
#   4. Unit-triangle integral ``J(p, q) = ∫₀¹ ∫₀^{1-u} exp(p·u + q·w)
#      dw du`` has the closed form
#
#         J(p, q) = [(eᵖ − e^q)/(p − q) − (eᵖ − 1)/p] / q
#
#      with stable Taylor fallbacks when |p|, |q|, or |p−q| → 0.
#   5. Sum over pole tuples ``(α_e)_e``: each contributes
#      A · exp(γ) · |det| · exp(α₀) · J(p, q).
#
# Compared to scipy.nquad on the un-expanded integrand: O(n_poles^|E_smooth|
# × n_triangles) complex-exp evaluations per (subset, τ-point) instead
# of ~10⁴ adaptive samples per (subset, τ-point), each itself doing
# n_poles · |E_smooth| complex-exps.  Net speedup typically 10-100×.

USE_POLYGON_M2_INTEGRATOR = True
POLYGON_BBOX_CAP = 200.0  # bounding-box for unbounded polygons

# ───────────────────────────────────────────────────────────────────────
# Physical fallback margin for unbounded polytope sides (Stage 3b)
# ───────────────────────────────────────────────────────────────────────
# When the polytope has no explicit scalar lower/upper, the chain
# simplex needs *some* finite L / U.  POLYGON_BBOX_CAP=200 is too loose:
# combined with retarded poles (Re β < 0), |Re β · L|=β·200 overflows
# the exp guard at |Re β|>3 (sum of 3–4 poles).  Physically, retarded
# propagators decay over a few correlation times beyond the earliest
# external time, so the integrand is negligible past
# ``min(0, free_ext_vals) − POSET_PHYSICAL_MARGIN`` (lower) and
# ``max(0, free_ext_vals) + POSET_PHYSICAL_MARGIN`` (upper).
# 50 is generous (≈ several correlation times for typical Hawkes
# τ_v ~ 10) but tight enough to avoid the overflow guard at typical
# pole magnitudes.
POSET_PHYSICAL_MARGIN = 50.0

# ───────────────────────────────────────────────────────────────────────
# Runtime path counters (diagnostic; zero perf impact when not read)
# ───────────────────────────────────────────────────────────────────────
# Increment at decision points inside the analytic evaluators so we can
# tell whether the intended analytic path actually completed or fell
# back to scipy.nquad at runtime.  The ``_evaluator_label`` recorded in
# ``subset_diagnostics`` is INTENT (set at subset setup time); these
# counters are RUNTIME.  Call ``_reset_runtime_counters()`` before a
# timed run, then read ``_RUNTIME_COUNTERS`` after.
_RUNTIME_COUNTERS = {
    # m=2 polygon path
    'polygon_attempted': 0,
    'polygon_returned_none': 0,
    # m≥3 poset path
    'poset_attempted': 0,
    'poset_extract_returned_none': 0,
    'poset_consistent_lower_failed': 0,
    'poset_maximality_failed': 0,
    'chain_simplex_fast_returned_none': 0,
    'chain_simplex_polynomial_called': 0,
    'chain_simplex_polynomial_returned_none': 0,
    'poset_returned_none_total': 0,
    # m=1 interval path
    'interval_attempted': 0,
    'interval_returned_none': 0,
    # scipy.nquad fallback (counted at _integrate_polytope entry for m≥1)
    'scipy_nquad_called_m1': 0,
    'scipy_nquad_called_m2': 0,
    'scipy_nquad_called_mge3': 0,
}


def _reset_runtime_counters():
    """Zero out ``_RUNTIME_COUNTERS`` before a timed run."""
    for k in _RUNTIME_COUNTERS:
        _RUNTIME_COUNTERS[k] = 0

# ───────────────────────────────────────────────────────────────────────
# Analytic ∫_L^U exp(α·s + γ) ds  (Stage 4a-perdiag, m=1)
# ───────────────────────────────────────────────────────────────────────
# Per-pole-tuple closed-form 1D exponential integral.  Replaces
# scipy.quad on the pole-residue closure callable for m=1 subsets.
# Same correctness guarantees as the polygon/poset paths (exact for
# rational propagators).
USE_1D_INTEGRATOR = True
# Threshold below which we switch to a 4th-order Taylor expansion of
# J(p, q) to avoid catastrophic cancellation in the formula's denominator.
_J_TAYLOR_EPS = 1e-6


def _exp_over_unit_triangle(p, q):
    r"""Closed-form value of

        J(p, q) = ∫₀¹ du ∫₀^{1-u} dw  exp(p·u + q·w)

    for complex ``p``, ``q``.

    Derivation: do the w-integral first to get
    ``∫₀¹ du exp(p·u) · (exp(q(1-u)) − 1) / q``, then split and
    integrate in u.  Falls back to a 4th-order Taylor expansion when
    any of |p|, |q|, |p − q| drops below ``_J_TAYLOR_EPS`` to avoid
    catastrophic cancellation.
    """
    import cmath
    eps = _J_TAYLOR_EPS
    abs_p = abs(p)
    abs_q = abs(q)
    abs_pq = abs(p - q)

    if abs_p < eps and abs_q < eps:
        # 4th-order Taylor of the double integral.  Coefficients are
        # the moments of the unit triangle:
        #   ∫∫ 1 = 1/2
        #   ∫∫ u  = ∫∫ w = 1/6
        #   ∫∫ u² = ∫∫ w² = 1/12,   ∫∫ uw = 1/24
        #   ∫∫ u³ = ∫∫ w³ = 1/20,   ∫∫ u²w = ∫∫ uw² = 1/60
        return (0.5
                + (p + q) / 6.0
                + (p * p) / 24.0 + (q * q) / 24.0 + (p * q) / 24.0
                + (p**3 + q**3) / 120.0
                + (p * p * q + p * q * q) / 120.0)

    if abs_p < eps:
        # J(0, q) = (e^q − 1 − q) / q²
        return (cmath.exp(q) - 1 - q) / (q * q)

    if abs_q < eps:
        # J(p, 0) = (e^p − 1 − p) / p²
        return (cmath.exp(p) - 1 - p) / (p * p)

    if abs_pq < eps:
        # J(p, p) = ((p − 1) e^p + 1) / p²
        return ((p - 1) * cmath.exp(p) + 1) / (p * p)

    # General case.
    ep = cmath.exp(p)
    eq = cmath.exp(q)
    return ((ep - eq) / (p - q) - (ep - 1) / p) / q


def _exp_over_triangle(v0, v1, v2, alpha, beta):
    r"""``∫∫_T exp(α·x + β·y) dA`` for triangle ``T = (v0, v1, v2)``.

    The vertices ``v0``, ``v1``, ``v2`` are 2-tuples of floats; ``α``,
    ``β`` are complex.  Affine-maps to the unit triangle and uses
    ``_exp_over_unit_triangle`` for the closed-form integral.

    Returns ``None`` if the per-term exponential would overflow IEEE
    double range — the polygon-modesum caller treats ``None`` as a
    signal to abort the analytic path and fall back to scipy.nquad.
    """
    import cmath
    e1x = v1[0] - v0[0]
    e1y = v1[1] - v0[1]
    e2x = v2[0] - v0[0]
    e2y = v2[1] - v0[1]
    det = e1x * e2y - e1y * e2x  # signed parallelogram area
    if abs(det) < 1e-15:
        return 0.0 + 0.0j  # degenerate
    p = alpha * e1x + beta * e1y
    q = alpha * e2x + beta * e2y
    v0_term = alpha * v0[0] + beta * v0[1]
    # Overflow guard — bilateral.  ``cmath.exp(z)`` blows past double
    # range for Re(z) > ~709 (overflow) AND underflows to 0 for Re(z)
    # < ~-745.  For most analytic integrators the underflow direction
    # is harmless (term decays correctly), but ``_exp_over_unit_
    # triangle`` (called below) has structural cancellation —
    # ``(exp(p) - exp(q)) / (p - q) - (exp(p) - 1) / p`` — which can
    # produce floating-point-noise-dominated values when one of
    # ``exp(p)`` or ``exp(q)`` underflows and the other doesn't.
    # Stage 4a opt #3 (2026-05-15 commit 7f0bf05) relaxed this to one-
    # sided which appears to have introduced wrong-direction loop
    # corrections in spike-reset k=2 ell=1 (m=2 polygon path is
    # exercised heavily there).  Reverting per Agent 4's audit
    # recommendation: keep the guard bilateral so previously-bailed
    # cases route to scipy.nquad as they did in Stage 3b.
    EXP_REAL_LIMIT = 600.0
    if (abs(p.real) > EXP_REAL_LIMIT
            or abs(q.real) > EXP_REAL_LIMIT
            or abs(v0_term.real) > EXP_REAL_LIMIT):
        return None
    try:
        J = _exp_over_unit_triangle(p, q)
        return abs(det) * cmath.exp(v0_term) * J
    except (OverflowError, ValueError):
        return None


def _clip_polygon_to_halfplane(polygon, a, b, c):
    r"""Sutherland-Hodgman clip a CCW polygon against the half-plane
    ``a·x + b·y + c > 0``.

    ``polygon`` is a list of ``(x, y)`` tuples.  Returns the clipped
    polygon as a new list; may be empty if the polygon lies entirely
    in the rejected half-space.

    Vertices exactly on the boundary (``a·x + b·y + c = 0``) are
    treated as "on the inside" — for analytic integration over the
    polygon interior, the measure-zero boundary contribution is 0
    regardless of which side we assign.
    """
    if not polygon:
        return []
    n = len(polygon)
    output = []
    prev = polygon[-1]
    prev_f = a * prev[0] + b * prev[1] + c
    for i in range(n):
        curr = polygon[i]
        curr_f = a * curr[0] + b * curr[1] + c
        prev_in = prev_f >= 0.0
        curr_in = curr_f >= 0.0
        if curr_in:
            if not prev_in:
                # Edge enters: add intersection.
                t = prev_f / (prev_f - curr_f)
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                output.append((ix, iy))
            output.append(curr)
        else:
            if prev_in:
                # Edge exits: add intersection.
                t = prev_f / (prev_f - curr_f)
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                output.append((ix, iy))
            # curr discarded
        prev = curr
        prev_f = curr_f
    return output


def _polygon_from_2d_constraints(constraint_data, free_ext_vals, bbox_cap):
    r"""Build the 2D convex polygon defined by the retardation
    constraints, starting from a CCW bounding box ``±bbox_cap`` and
    clipping with each constraint.

    Each constraint is ``(a_int, a_ext, c0)`` with
    ``a_int·(s_0, s_1) + (c0 + a_ext·free_ext_vals) > 0``.

    Returns a CCW polygon vertex list (possibly empty if the
    intersection is empty).
    """
    polygon = [
        (-bbox_cap, -bbox_cap),
        (bbox_cap, -bbox_cap),
        (bbox_cap, bbox_cap),
        (-bbox_cap, bbox_cap),
    ]
    for (a_int, a_ext, c0) in constraint_data:
        c_eff = float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        a0 = float(a_int[0])
        a1 = float(a_int[1])
        polygon = _clip_polygon_to_halfplane(polygon, a0, a1, c_eff)
        if not polygon:
            return []
    return polygon


def _enumerate_pole_tuples(edge_mode_sums):
    r"""Cartesian product over per-edge modes.

    Yields ``(C_product, lambdas)`` per pole tuple:
      * ``C_product``: complex, the product of residues ``∏_e C_α_e``
      * ``lambdas``:  tuple of complex ``(λ_α₁, λ_α₂, …)`` — one per
        smooth edge, in the same order as ``edge_mode_sums``.

    For ``len(edge_mode_sums) == 0`` yields exactly one tuple
    ``(1.0+0j, ())`` representing the empty product / no exponential.
    """
    if not edge_mode_sums:
        yield (1.0 + 0.0j, ())
        return
    n_edges = len(edge_mode_sums)
    mode_counts = [len(ems.modes) for ems in edge_mode_sums]
    # Stack-based product (avoids itertools.product overhead).
    idx = [0] * n_edges
    while True:
        C_prod = 1.0 + 0.0j
        lambdas = []
        for e in range(n_edges):
            C, lam = edge_mode_sums[e].modes[idx[e]]
            C_prod *= C
            lambdas.append(lam)
        yield C_prod, tuple(lambdas)
        # Advance index.
        e = 0
        while e < n_edges:
            idx[e] += 1
            if idx[e] < mode_counts[e]:
                break
            idx[e] = 0
            e += 1
        else:
            return  # all overflowed


# ───────────────────────────────────────────────────────────────────────
# Per-subset plan cache (Stage 4a-plan, 2026-05-15)
# ───────────────────────────────────────────────────────────────────────
# All three analytic modesum integrators (m=1 interval, m=2 polygon,
# m≥3 poset) iterate over pole tuples and per tuple compute
#
#   α_s[v]     = Σ_e λ_e · a_int_e[v]      for v in 0..m-1
#   γ_const[t] = Σ_e λ_e · c0_e
#   γ_slope[t][j] = Σ_e λ_e · a_ext_e[j]   for j in 0..n_ext-1
#
# and finally
#
#   γ(free_vals) = γ_const + Σ_j γ_slope[j] · free_vals[j].
#
# ``λ_e``, ``a_int_e[v]``, ``c0_e``, ``a_ext_e[j]`` are all functions of
# ``(smooth_edge_modes, subset_constraint_data)`` — strictly per-subset.
# ``free_vals`` is the only τ-grid-varying input.  Previously each
# integrator rebuilt the per-tuple α_s / γ_const / γ_slope on every
# ``_contrib(free_vals)`` call, which compounds linearly across the τ
# grid.  The plan caches them once at subset setup and threads the
# cached arrays into the per-call inner loop, so the per-τ work
# reduces to the much cheaper
#
#   γ_per_tuple[t] = γ_const[t] + Σ_j γ_slope[t][j] · free_vals[j]
#
# while reusing α_s, polygon-vertex constraints, and the EdgeModeSum
# cache.  Scales with (N_τ − 1) — biggest win on τ-dense sweeps
# (k=1 max_ell=2 with 10+ probes, etc.).
def _build_modesum_plan(smooth_edge_modes, subset_constraint_data,
                        m, n_ext):
    """Pre-compute the τ-invariant per-pole-tuple data used by the
    analytic modesum integrators.

    Returns a dict with the following keys; all values are tuples
    (immutable, cheap to share across closures):

    ``pole_tuples``: tuple of (C_prod, lambdas)
        Pre-enumerated cartesian product over per-edge modes; identical
        contents to ``_enumerate_pole_tuples(smooth_edge_modes)`` but
        materialised so iteration over the τ grid pays the construction
        cost once.

    ``alphas_per_tuple``: tuple, len = n_tuples
        Each entry is a tuple of ``m`` complex values
        ``(α_s[0], …, α_s[m-1])`` for the corresponding pole tuple.
        Replaces the per-call inner edge-loop
        ``α_s[v] += λ_e · a_int_e[v]``.

    ``gamma_const_per_tuple``: tuple, len = n_tuples
        Each entry is the complex constant ``Σ_e λ_e · c0_e`` for that
        pole tuple — i.e. γ at ``free_vals = 0``.

    ``gamma_slope_per_tuple_per_ext``: tuple of tuples
        ``slope[t][j] = Σ_e λ_e · a_ext_e[j]`` for tuple t, external-
        time index j.  γ at general free_vals is then
        ``γ_const[t] + Σ_j slope[t][j] · free_vals[j]``.
    """
    pole_tuples = tuple(_enumerate_pole_tuples(smooth_edge_modes))
    # Pre-extract per-edge linear coefficients in floats.  The arity
    # of ``a_int_e`` is ``m`` and ``a_ext_e`` is ``n_ext``.  We tolerate
    # short ``a_int`` lists (degenerate constraints with no internal
    # coefficients) by padding with 0.0.
    a_int_per_edge = []
    a_ext_per_edge = []
    c0_per_edge = []
    for (a_int, a_ext, c0) in subset_constraint_data:
        a_int_pad = tuple(
            float(a_int[v]) if v < len(a_int) else 0.0 for v in range(m)
        )
        a_ext_pad = tuple(
            float(a_ext[j]) if j < len(a_ext) else 0.0
            for j in range(n_ext)
        )
        a_int_per_edge.append(a_int_pad)
        a_ext_per_edge.append(a_ext_pad)
        c0_per_edge.append(float(c0))

    alphas_per_tuple = []
    gamma_const_per_tuple = []
    gamma_slope_per_tuple_per_ext = []
    n_smooth = len(smooth_edge_modes)
    for (C_prod, lambdas) in pole_tuples:
        alphas = [0.0 + 0.0j] * m
        gamma_const = 0.0 + 0.0j
        gamma_slope = [0.0 + 0.0j] * n_ext
        for e in range(n_smooth):
            lam = lambdas[e]
            for v in range(m):
                alphas[v] += lam * a_int_per_edge[e][v]
            gamma_const += lam * c0_per_edge[e]
            for j in range(n_ext):
                gamma_slope[j] += lam * a_ext_per_edge[e][j]
        alphas_per_tuple.append(tuple(alphas))
        gamma_const_per_tuple.append(gamma_const)
        gamma_slope_per_tuple_per_ext.append(tuple(gamma_slope))

    return {
        'pole_tuples':                  pole_tuples,
        'alphas_per_tuple':             tuple(alphas_per_tuple),
        'gamma_const_per_tuple':        tuple(gamma_const_per_tuple),
        'gamma_slope_per_tuple_per_ext': tuple(gamma_slope_per_tuple_per_ext),
    }


# ───────────────────────────────────────────────────────────────────────
# Causal poset extraction + linear-extension enumeration (Stage 3b prep)
# ───────────────────────────────────────────────────────────────────────
# For m ≥ 3 (deeply-nested integrals), the retardation constraint set
# typically forms a directed acyclic graph (DAG) on the integration
# variables — each constraint  s_v − s_u > 0  is an edge ``u → v``.
# The full polytope decomposes into a disjoint union of simplex
# regions, one per linear extension (topological sort) of the DAG.
# Each simplex region has the form
#
#     L ≤ s_{σ(1)} ≤ s_{σ(2)} ≤ … ≤ s_{σ(N)} ≤ U
#
# (when all variables share the same scalar lower bound L and the
# top variable has a scalar upper bound U).  The integral over each
# region factors into nested 1D exponential integrals — a closed
# form lives in ``_exp_over_chain`` (Stage 3b-nested).
#
# This file implements the structural part: extract the DAG + scalar
# bounds from the constraint list, enumerate linear extensions.

@dataclass(frozen=True)
class _CausalPoset:
    """DAG on ``m`` integration variables plus per-variable scalar
    bounds, extracted from a list of retardation constraints.

    Fields
    ------
    m : int
        Number of integration variables.
    edges : tuple of (int, int)
        Pairs ``(u, v)`` with ``s_v > s_u``.  Duplicate edges removed.
    scalar_lowers : tuple of (int, float)
        ``(var_idx, c)`` meaning ``s_var > c``.  Multiple lower bounds
        on the same variable are kept; the effective lower is the
        max.
    scalar_uppers : tuple of (int, float)
        ``(var_idx, c)`` meaning ``s_var < c``.  Effective upper is
        the min.

    Notes
    -----
    Edge (u, v) means u precedes v in the integration time ordering.
    """
    m: int
    edges: tuple
    scalar_lowers: tuple
    scalar_uppers: tuple


def _extract_causal_poset(subset_constraint_data, free_ext_vals, m,
                          tol=1e-12):
    r"""Build a ``_CausalPoset`` from a list of retardation
    constraints.

    Each constraint is ``(a_int, a_ext, c0)`` representing
    ``a_int · s + c_eff > 0`` where
    ``c_eff = c0 + a_ext · free_ext_vals``.

    Constraint shape recognised:
    * **Inter-axis**: ``a_int`` has exactly two nonzero entries
      summing to 0 (one +1, one −1), and ``c_eff ≈ 0`` (within tol).
      Adds edge ``(u, v)`` where ``a_int[u] = −1`` (lower) and
      ``a_int[v] = +1`` (upper).
    * **Scalar lower**: ``a_int`` has exactly one +1 entry, all else 0.
      Adds (var, −c_eff) to ``scalar_lowers``.
    * **Scalar upper**: ``a_int`` has exactly one −1 entry, all else 0.
      Adds (var, c_eff) to ``scalar_uppers``.
    * **Anything else** (multiple inter-axis couplings, mixed
      coefficients, inter-axis with nonzero c_eff, etc.) → return
      ``None``.  Caller falls back to scipy.nquad.

    Returns ``_CausalPoset`` or ``None``.
    """
    edges = []
    scalar_lowers = []
    scalar_uppers = []
    for (a_int, a_ext, c0) in subset_constraint_data:
        c_eff = float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        # Find positions and signs of nonzero entries in a_int.
        nz = [(i, float(a_int[i])) for i in range(len(a_int))
              if abs(float(a_int[i])) > tol]
        if not nz:
            # Pure constant constraint.  Should be satisfied
            # (c_eff > 0); if violated, polytope is empty (signal
            # via None — caller falls back, which will also detect
            # the empty polytope correctly).
            if c_eff <= 0:
                return None
            continue
        if len(nz) == 1:
            (var_idx, coef) = nz[0]
            # Pure scalar bound.  Only accept ±1 coefficients.
            if abs(coef - 1.0) < tol:
                # +1 · s_var + c_eff > 0  ⇔  s_var > −c_eff
                scalar_lowers.append((var_idx, -c_eff))
            elif abs(coef + 1.0) < tol:
                # −1 · s_var + c_eff > 0  ⇔  s_var < c_eff
                scalar_uppers.append((var_idx, c_eff))
            else:
                # Non-unit coefficient — not a clean ±1 constraint.
                return None
            continue
        if len(nz) == 2:
            # Need exactly one +1 and one −1, plus c_eff ≈ 0.
            (i, a_i), (j, a_j) = nz
            if abs(c_eff) > 1000 * tol:
                # Inter-axis constraint with extra constant — would
                # be a "shifted" ordering like s_v > s_u + c.  Not
                # supported in the simple poset model; bail.
                return None
            if abs(a_i - 1.0) < tol and abs(a_j + 1.0) < tol:
                # +1 at i, −1 at j  ⇔  s_i > s_j  ⇔  edge (j → i)
                edges.append((j, i))
            elif abs(a_i + 1.0) < tol and abs(a_j - 1.0) < tol:
                # −1 at i, +1 at j  ⇔  s_j > s_i  ⇔  edge (i → j)
                edges.append((i, j))
            else:
                return None
            continue
        # 3+ nonzero entries — mixed constraint not supported.
        return None

    # Deduplicate edges.
    edges_set = tuple(sorted(set(edges)))
    return _CausalPoset(
        m=m,
        edges=edges_set,
        scalar_lowers=tuple(scalar_lowers),
        scalar_uppers=tuple(scalar_uppers),
    )


def _enumerate_linear_extensions(poset):
    r"""Yield each linear extension (topological ordering) of a
    ``_CausalPoset``.

    A linear extension is a permutation σ of [0, m) such that for
    every edge (u, v) of the poset, σ⁻¹(u) < σ⁻¹(v) — i.e., u appears
    before v in the ordering.

    Standard recursive Kahn-style enumeration: at each step, pick
    one of the currently-source nodes (no remaining predecessors)
    and recurse.

    Yields tuples of length ``poset.m``.
    """
    m = poset.m
    # Predecessor counts (per variable).
    in_count = [0] * m
    successors = [[] for _ in range(m)]
    for (u, v) in poset.edges:
        in_count[v] += 1
        successors[u].append(v)

    available = sorted(i for i in range(m) if in_count[i] == 0)

    def _recurse(order, in_count_local, available_local):
        if len(order) == m:
            yield tuple(order)
            return
        for var in list(available_local):
            new_available = [a for a in available_local if a != var]
            new_in_count = list(in_count_local)
            for succ in successors[var]:
                new_in_count[succ] -= 1
                if new_in_count[succ] == 0:
                    # Insert preserving sorted order for deterministic
                    # output (canonical lex enumeration).
                    inserted = False
                    for k, x in enumerate(new_available):
                        if x > succ:
                            new_available.insert(k, succ)
                            inserted = True
                            break
                    if not inserted:
                        new_available.append(succ)
            order.append(var)
            yield from _recurse(order, new_in_count, new_available)
            order.pop()

    yield from _recurse([], in_count, available)


def _causal_poset_consistent_scalar_lower(poset, tol=1e-9):
    r"""Compute the single effective scalar lower bound shared by
    variables that have one.

    Returns ``(L, True)`` if every variable WITH A SCALAR LOWER has
    the same value ``L`` (within ``tol``), or no scalar lower at all
    (returns ``(None, True)`` — caller uses a bbox cap).  Variables
    WITHOUT an explicit scalar lower inherit ``L`` via the chain
    ordering of any linear extension: the chain-simplex form
    integrates each non-bottom variable from L (or the previous
    variable's value, whichever is greater).  For retardation-style
    constraint structure where every internal vertex is reachable
    from an external-leaf-pinned ancestor, this is correct.

    Returns ``(None, False)`` if scalar lowers exist on multiple
    variables with DIFFERENT values — in that case the chain
    simplex over a single L would over- or under-include regions
    and the caller should fall back to scipy.nquad.
    """
    per_var_max = {}
    for (var, c) in poset.scalar_lowers:
        cur = per_var_max.get(var)
        per_var_max[var] = c if cur is None else max(cur, c)
    if not per_var_max:
        return (None, True)
    vals = list(per_var_max.values())
    Lmin, Lmax = min(vals), max(vals)
    if Lmax - Lmin > tol:
        return (None, False)
    # All variables that have a scalar lower agree on value Lmax.
    # Variables without one inherit via the chain ordering.
    return (Lmax, True)


def _causal_poset_consistent_scalar_upper(poset, tol=1e-9):
    r"""Companion to ``_causal_poset_consistent_scalar_lower`` for
    the upper-bound side, but only the TOP variable in each linear
    extension needs an upper bound (others are bounded by the next
    variable in the chain).

    Returns the smallest scalar upper found (across any variable),
    intended to be used as the cap for whichever variable ends up
    at the top of the linear extension — IF that variable has its
    own scalar upper.  When no variable has a scalar upper, the
    caller must supply a fallback cap.
    """
    per_var_min = {}
    for (var, c) in poset.scalar_uppers:
        cur = per_var_min.get(var)
        per_var_min[var] = c if cur is None else min(cur, c)
    return per_var_min


def _exp_over_chain_simplex(alphas, lower, upper, eps=1e-9):
    r"""Closed-form value of the nested integral on the chain
    simplex  ``{lower ≤ s_1 ≤ s_2 ≤ … ≤ s_N ≤ upper}``::

       ∫_{lower}^{upper}        ds_N  exp(α_N · s_N)
         · ∫_{lower}^{s_N}      ds_{N-1}  exp(α_{N-1} · s_{N-1})
         · …
         · ∫_{lower}^{s_2}      ds_1  exp(α_1 · s_1)

    where ``alphas = [α_1, α_2, …, α_N]`` with ``α_k`` the exponent
    coefficient for the k-th variable in the chain (1-indexed
    mathematically, 0-indexed in the list).

    Derivation: integrate inside-out.  After each step, the running
    integrand is a sum of terms of the form
    ``C · exp(β · s_outer + (constant depending on already-integrated
    bounds))``.  Each inner integration produces two new terms — one
    "upper-bound" piece (whose ``β`` merges with the next-outer
    variable's coefficient) and one "lower-bound" piece (a numerical
    prefactor ``exp(β · lower)`` falls out, leaving the outer
    coefficient unchanged).  After N steps the sum has 2^N constant
    terms; their sum is the integral.

    Parameters
    ----------
    alphas : sequence of complex
        Effective exponent coefficients, innermost first.
    lower, upper : float
        Common scalar lower bound and the cap on the outermost var.
    eps : float
        Threshold below which a ``β`` is treated as degenerate (the
        formula's ``1/β`` factor would amplify roundoff).  Returns
        ``None`` in that case so the caller falls back to scipy.

    Returns
    -------
    complex or None
        The integral value, or ``None`` if any intermediate
        coefficient ``β`` is too close to zero.
    """
    import cmath
    N = len(alphas)
    if N == 0:
        # Empty product of integrals — by convention 1.
        return 1.0 + 0.0j
    # Empty chain simplex (``upper ≤ lower``): integration domain has
    # zero measure, integral is 0.  Crucial for τ < 0 configurations
    # where the chain-bottom scalar lower (e.g. from leaf t = 0) and
    # the chain-top scalar upper (e.g. from leaf t = τ < 0) cross.
    if upper <= lower:
        return 0.0 + 0.0j

    # Each term: (complex coefficient, list of remaining β values).
    # At level k (about to integrate the (k+1)-th variable in the
    # chain, which is the innermost remaining), ``beta[0]`` is the
    # effective coefficient on that variable, ``beta[1:]`` are the
    # coefficients on the outer variables we haven't touched yet.
    terms = [(1.0 + 0.0j, list(alphas))]

    # Overflow guard.  The 2^N term expansion can produce intermediate
    # ``exp(b · L_or_U)`` factors whose Re(b · arg) exceeds the IEEE
    # double's exp range (~709).  The TRUE integral is finite — the
    # individual large terms cancel — but the cancellation is fragile
    # and depends on bit-exact arithmetic the closed-form can't
    # guarantee.  Safe path: detect the overflow risk, return None,
    # let the caller fall back to scipy.nquad which handles the
    # well-conditioned integral natively.
    # ``EXP_REAL_LIMIT = 600`` leaves a margin below the 709 hard
    # limit so accumulated roundoff doesn't push us over.
    EXP_REAL_LIMIT = 600.0

    try:
        # Integrate variables 1, 2, …, N-1 (each bounded above by next
        # variable in the chain).
        for _ in range(N - 1):
            new_terms = []
            for (C, beta) in terms:
                b_inner = beta[0]
                if abs(b_inner) < eps:
                    return None
                if abs((b_inner * lower).real) > EXP_REAL_LIMIT:
                    return None
                # Term A — upper-bound piece.  exp(b_inner · s_outer)
                # merges with the existing exp(beta[1] · s_outer)
                # factor, so the new β on the next-outer variable is
                # ``b_inner + beta[1]``.
                beta_A = list(beta[1:])
                beta_A[0] = b_inner + beta_A[0]
                new_terms.append((C / b_inner, beta_A))
                # Term B — lower-bound piece.  Just pulls out a
                # constant ``exp(b_inner · lower)`` factor; outer β
                # unchanged.
                beta_B = list(beta[1:])
                new_terms.append((
                    -C * cmath.exp(b_inner * lower) / b_inner,
                    beta_B,
                ))
            terms = new_terms

        # Outermost integration: s_N from lower to upper (both
        # constants).
        total = 0.0 + 0.0j
        for (C, beta) in terms:
            b = beta[0]
            if abs(b) < eps:
                # ∫_L^U exp(0 · s) ds = U − L
                total += C * (upper - lower)
            else:
                if (abs((b * upper).real) > EXP_REAL_LIMIT
                        or abs((b * lower).real) > EXP_REAL_LIMIT):
                    return None
                total += C * (cmath.exp(b * upper)
                              - cmath.exp(b * lower)) / b
    except (OverflowError, ValueError):
        return None
    return total


# ───────────────────────────────────────────────────────────────────────
# Numba-compiled chain simplex (fast-path companion to the function above)
# ───────────────────────────────────────────────────────────────────────
# Same algorithm as ``_exp_over_chain_simplex`` translated to a numba
# ``@njit`` function operating on pre-allocated complex128 numpy
# buffers.  Targets the chain simplex inner loop which post-Stage-3b is
# the dominant per-(diagram, subset, pole-tuple, linear-extension)
# cost.  Expected 30-100× per call vs the pure-Python version.
#
# Semantics MUST match the Python version bit-for-bit on every
# converged result.  The Python version stays in place as the
# reference / fallback (selected by ``USE_NUMBA_CHAIN_SIMPLEX = False``
# or by the wrapper on a numba-import failure).
import numpy as np

try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    _numba = None

USE_NUMBA_CHAIN_SIMPLEX = True

# ─── Chain-simplex precision-loss fix (2026-05-16) ────────────────────
#
# When the chain simplex contains close-paired effective coefficients
# (e.g. spike-reset propagator with poles 0.329i, 0.351i, so some
# ``b_inner`` in the 2^N recursion is ~0.022), the closed-form formula
# in ``_exp_over_chain_simplex`` suffers cancellation-driven precision
# loss.  Empirically observed as a 4× aggregate overestimate of the
# 1-loop value at spike-reset k=2 ell=1 (76% rel diff vs the converged
# grouped-scipy reference), with per-subset audit measuring an aggregate
# analytic/scipy ratio of 1.086 across 30 sampled m=3 subsets that
# extrapolates to the observed 4× across 313 subsets.
#
# Fix: when any subset sum of ``alphas`` falls below a threshold (cheap
# 2^N scan of subset-sum magnitudes), route the entire chain-simplex
# evaluation to ``_exp_over_chain_simplex_mpmath`` running at 50-digit
# precision.  The mpmath path is 50-500× slower per call but only fires
# when the float64 path was already wrong; unaffected configurations
# (quad model, spike-reset k≤1 ell≤1, etc.) skip the gate entirely.
#
# Full audit at ``docs/m_ge3_precision_bug_audit.md``;
# design at ``docs/m_ge3_chain_simplex_fix_proposal.md``.
#
# To revert exactly to the pre-fix behaviour (bit-identical), set the
# flag to ``False``.  No other code changes required.
# Set to True to enable the close-pole mpmath dispatch in chain simplex.
# Empirically this did NOT address the spike-reset k=2 ell=1 bug (which
# turned out to be a cap-mismatch issue, not chain-simplex precision —
# see USE_POSET_CAP_MATCH_SCIPY below).  Keeping the code path for
# possible future use.  Off by default.
USE_CHAIN_SIMPLEX_PRECISION_FIX = False

# Threshold below which any subset-sum magnitude of ``alphas`` triggers
# the high-precision (mpmath) path.  Calibrated to catch the spike-reset
# 0.022i pole-difference case (which causes ~10% per-subset bias) while
# not unduly slowing well-conditioned theories.  Lower → fewer mpmath
# calls (faster but less safe); higher → more mpmath calls (safer but
# slower).
_CHAIN_SIMPLEX_CANCEL_THRESHOLD = 0.1


def _min_subset_sum_abs(alphas):
    """Minimum |Σ_{i ∈ S} α_i| over non-empty subsets S of ``alphas``.

    Cheap 2^N scan used to detect when the chain-simplex recursion will
    encounter a small ``b_inner`` (cumulative-pole-sum) at some level,
    triggering precision-loss in the closed-form formula.

    Returns ``inf`` for empty input so the threshold gate is a no-op.

    Cached via ``_min_subset_sum_abs_cached`` keyed on the alphas tuple,
    since the chain-simplex dispatcher is called many times with the
    same α-vector.  Cache eliminates the per-call Python overhead that
    would otherwise erase the numba speedup on well-conditioned inputs.
    """
    n = len(alphas)
    if n == 0:
        return float('inf')
    # Use the alphas directly as a tuple; complex elements are hashable.
    # On numpy arrays this requires a manual tuple() conversion.
    return _min_subset_sum_abs_cached(tuple(alphas))


@_functools.lru_cache(maxsize=4096)
def _min_subset_sum_abs_cached(alphas_tuple):
    """Cached 2^N subset-sum-magnitude scan."""
    n = len(alphas_tuple)
    min_abs = float('inf')
    for mask in range(1, 1 << n):
        s = 0.0 + 0.0j
        for i in range(n):
            if mask & (1 << i):
                s = s + alphas_tuple[i]
        a = abs(s)
        if a < min_abs:
            min_abs = a
    return min_abs


def _exp_over_chain_simplex_mpmath(alphas, lower, upper, eps=1e-9, dps=50):
    r"""High-precision (mpmath, default 50 digits) evaluation of the
    same closed-form integral as ``_exp_over_chain_simplex``.

    Used as the dispatch target for ``_exp_over_chain_simplex_fast``
    when ``USE_CHAIN_SIMPLEX_PRECISION_FIX`` is enabled AND the
    cheap subset-sum scan detects a close-pole condition.

    Semantics match ``_exp_over_chain_simplex`` exactly — same input/
    output types, same ``None`` return on overflow / degenerate β at
    the float64-effective threshold ``eps``.  The difference is that
    intermediate arithmetic uses ``mpmath.mpc`` at 50 decimal digits,
    so the 2^N cancellation that loses ~14 digits of float64 precision
    leaves ~36 digits intact — well below any plausible aggregation
    tolerance.

    Parameters
    ----------
    alphas, lower, upper, eps : as in ``_exp_over_chain_simplex``.
    dps : int, default 50
        Decimal-digit precision for the mpmath workspace.  50 digits
        comfortably exceeds float64's 16, with margin for cancellation
        of any plausible close-pole pair (e.g., spike-reset's 0.022i
        loses ~14 digits → 50 − 14 = 36 left).
    """
    try:
        from mpmath import mp, mpc, exp as mp_exp
    except ImportError:
        # mpmath unavailable → behave as if the fix were off; caller
        # already has bit-exact recovery via flag.
        return _exp_over_chain_simplex(alphas, lower, upper, eps)

    n = len(alphas)
    if n == 0:
        return 1.0 + 0.0j
    if upper <= lower:
        return 0.0 + 0.0j

    EXP_REAL_LIMIT = 600.0

    saved_dps = mp.dps
    mp.dps = dps
    try:
        L = mpc(float(lower), 0.0)
        U = mpc(float(upper), 0.0)
        alphas_mp = [mpc(complex(a).real, complex(a).imag) for a in alphas]
        terms = [(mpc(1, 0), list(alphas_mp))]

        for _ in range(n - 1):
            new_terms = []
            for (C, beta) in terms:
                b_inner = beta[0]
                # Use the float64 magnitude for the threshold check —
                # mpmath's abs is exact but slow; complex(b_inner) is
                # fine since the test is "is this MUCH bigger than eps".
                if abs(complex(b_inner)) < eps:
                    return None
                bl_real = float((b_inner * L).real)
                if abs(bl_real) > EXP_REAL_LIMIT:
                    return None
                beta_A = list(beta[1:])
                beta_A[0] = b_inner + beta_A[0]
                new_terms.append((C / b_inner, beta_A))
                beta_B = list(beta[1:])
                new_terms.append((
                    -C * mp_exp(b_inner * L) / b_inner,
                    beta_B,
                ))
            terms = new_terms

        total = mpc(0, 0)
        for (C, beta) in terms:
            b = beta[0]
            if abs(complex(b)) < eps:
                total = total + C * (U - L)
            else:
                bu_real = float((b * U).real)
                bl_real = float((b * L).real)
                if abs(bu_real) > EXP_REAL_LIMIT or abs(bl_real) > EXP_REAL_LIMIT:
                    return None
                total = total + C * (mp_exp(b * U) - mp_exp(b * L)) / b
        return complex(float(total.real), float(total.imag))
    except Exception:
        return None
    finally:
        mp.dps = saved_dps


if _HAVE_NUMBA:
    @_numba.njit(cache=True)
    def _exp_over_chain_simplex_numba_core(alphas, lower, upper, eps):
        """Returns ``(status, value)``:
            status = 0  → ``value`` is the integral
            status = 1  → degenerate β (caller should fall back to
                          polynomial-prefactor path)
            status = 2  → overflow (caller returns None)
        """
        N = alphas.shape[0]
        if N == 0:
            return 0, 1.0 + 0.0j
        if upper <= lower:
            return 0, 0.0 + 0.0j

        EXP_REAL_LIMIT = 600.0

        # Buffer size = max possible term count = 2^(N-1).
        max_terms = 1 << max(N - 1, 0)

        # Ping-pong buffers for the doubling term list.
        coefs_a = np.zeros(max_terms, dtype=np.complex128)
        betas_a = np.zeros((max_terms, N), dtype=np.complex128)
        coefs_b = np.zeros(max_terms, dtype=np.complex128)
        betas_b = np.zeros((max_terms, N), dtype=np.complex128)

        coefs_a[0] = 1.0 + 0.0j
        for j in range(N):
            betas_a[0, j] = alphas[j]
        n_terms = 1

        for level in range(N - 1):
            n_remaining = N - level
            new_n = 0
            for i in range(n_terms):
                b_inner = betas_a[i, 0]
                if abs(b_inner) < eps:
                    return 1, 0.0 + 0.0j
                if abs((b_inner * lower).real) > EXP_REAL_LIMIT:
                    return 2, 0.0 + 0.0j
                exp_lower_val = np.exp(b_inner * lower)
                C_i = coefs_a[i]

                # Term A (upper-bound piece):
                #   coef  = C / b_inner
                #   β_new = (b_inner + β_old[1], β_old[2], …)
                coefs_b[new_n] = C_i / b_inner
                betas_b[new_n, 0] = b_inner + betas_a[i, 1]
                for j in range(2, n_remaining):
                    betas_b[new_n, j - 1] = betas_a[i, j]
                new_n += 1

                # Term B (lower-bound piece):
                #   coef  = -C · exp(b_inner · lower) / b_inner
                #   β_new = (β_old[1], β_old[2], …)
                coefs_b[new_n] = -C_i * exp_lower_val / b_inner
                for j in range(1, n_remaining):
                    betas_b[new_n, j - 1] = betas_a[i, j]
                new_n += 1

            # Swap a ↔ b for next level.
            tmp_c = coefs_a
            coefs_a = coefs_b
            coefs_b = tmp_c
            tmp_b = betas_a
            betas_a = betas_b
            betas_b = tmp_b
            n_terms = new_n

        # Outermost integration: s_N from lower to upper.
        total = 0.0 + 0.0j
        for i in range(n_terms):
            b = betas_a[i, 0]
            C_i = coefs_a[i]
            if abs(b) < eps:
                total += C_i * (upper - lower)
            else:
                if (abs((b * upper).real) > EXP_REAL_LIMIT
                        or abs((b * lower).real) > EXP_REAL_LIMIT):
                    return 2, 0.0 + 0.0j
                total += C_i * (np.exp(b * upper) - np.exp(b * lower)) / b

        return 0, total


def _exp_over_chain_simplex_fast(alphas, lower, upper, eps=1e-9):
    """Dispatcher: numba version when available + enabled, else Python.

    Returns the integral value or ``None`` (degenerate β / overflow),
    matching ``_exp_over_chain_simplex`` semantics exactly.

    When ``USE_CHAIN_SIMPLEX_PRECISION_FIX`` is enabled (default) and
    a cheap subset-sum scan detects a close-pole condition (any non-
    empty subset of ``alphas`` summing to magnitude
    < ``_CHAIN_SIMPLEX_CANCEL_THRESHOLD``), the entire call is routed
    to ``_exp_over_chain_simplex_mpmath`` for high-precision evaluation.
    The float64 path is precision-limited in that regime and produces
    a systematic overestimate that compounds across many subsets at
    spike-reset k≥2 ell≥1.  Set the flag to ``False`` to disable the
    routing and recover the pre-fix behaviour bit-identically.
    """
    if (USE_CHAIN_SIMPLEX_PRECISION_FIX
            and len(alphas) >= 3
            and _min_subset_sum_abs(alphas) < _CHAIN_SIMPLEX_CANCEL_THRESHOLD):
        return _exp_over_chain_simplex_mpmath(alphas, lower, upper, eps)
    if not (_HAVE_NUMBA and USE_NUMBA_CHAIN_SIMPLEX):
        return _exp_over_chain_simplex(alphas, lower, upper, eps)
    try:
        arr = np.asarray(alphas, dtype=np.complex128)
        if arr.ndim != 1:
            return _exp_over_chain_simplex(alphas, lower, upper, eps)
        status, val = _exp_over_chain_simplex_numba_core(
            arr, float(lower), float(upper), float(eps),
        )
    except Exception:
        return _exp_over_chain_simplex(alphas, lower, upper, eps)
    if status != 0:
        return None
    return complex(val)


def _exp_over_chain_simplex_polynomial(alphas, lower, upper, eps=1e-9):
    r"""Polynomial-prefactor extension of ``_exp_over_chain_simplex``.

    Same closed-form integral as ``_exp_over_chain_simplex`` but DOES
    NOT return ``None`` when an intermediate cumulative-pole-sum β
    vanishes.  In that case the level's antiderivative is a polynomial
    in s (rather than ``exp(β·s)/β``); the polynomial is carried
    through subsequent levels, with closed-form treatment of
    ``∫ u^k · exp(β·u) du`` at non-degenerate levels.

    Math sketch
    -----------
    Each level produces one of two outcomes:

    * **Non-degenerate (β ≠ 0).**  For each ``u^k · exp(β·s)`` term in
      the running integrand:

          ∫_lower^{s_outer} (s - lower)^k · exp(β·s) ds
          = exp(β·lower) · Σ_{j=0}^{k} (-1)^j · k!/(k-j)! · u_outer^{k-j}
                                       · exp(β·u_outer) / β^{j+1}
            + (-1)^{k+1} · k! · exp(β·lower) / β^{k+1}

      yields (k+1) "upper" terms carrying β into the next-outer
      variable's exponent, plus 1 "lower" constant term with no β.

    * **Degenerate (β ≈ 0).**  The integrand is just a polynomial:

          ∫_lower^{s_outer} (s - lower)^k ds  =  u_outer^{k+1} / (k+1)

      yields a single term whose polynomial degree has grown by 1 and
      whose β-list is unchanged (no exp factor carries forward).

    The final outermost integration replaces ``s_outer`` with a numeric
    upper bound and accumulates the per-term contributions.

    Term representation
    -------------------
    Each running term is a tuple ``(C, poly, β_list)``:
    * ``C`` — complex scalar coefficient.
    * ``poly`` — tuple of complex polynomial coefficients in basis
      ``(s - lower)^k``, index = degree.
    * ``β_list`` — tuple of complex β values for the variables
      remaining to integrate, innermost first.

    Initial state: ``[(1+0j, (1+0j,), tuple(alphas))]``.

    Parameters and return value mirror ``_exp_over_chain_simplex``;
    returns ``None`` only on overflow (real-exponent magnitude beyond
    ``EXP_REAL_LIMIT`` = 600), never on degenerate β.
    """
    import cmath
    import math

    N = len(alphas)
    if N == 0:
        return 1.0 + 0.0j
    if upper <= lower:
        return 0.0 + 0.0j

    EXP_REAL_LIMIT = 600.0

    terms = [(1.0 + 0.0j, (1.0 + 0.0j,), tuple(alphas))]

    try:
        # Inner-loop integrations: s_1, s_2, …, s_{N-1}.
        for _level in range(N - 1):
            new_terms = []
            for (C, poly, β_list) in terms:
                β_inner = β_list[0]
                β_rest = β_list[1:]
                # β_rest always has ≥ 1 element while in this inner loop.

                if abs(β_inner) < eps:
                    # Degenerate level: polynomial integration.
                    new_poly = [0.0 + 0.0j] * (len(poly) + 1)
                    for k, a in enumerate(poly):
                        new_poly[k + 1] = a / (k + 1)
                    new_terms.append((C, tuple(new_poly), β_rest))
                else:
                    # Non-degenerate: closed-form polynomial × exp.
                    if abs((β_inner * lower).real) > EXP_REAL_LIMIT:
                        return None
                    exp_lower = cmath.exp(β_inner * lower)
                    # Upper β list: β_inner merges into next-outer β.
                    β_upper_rest = (β_inner + β_rest[0],) + β_rest[1:]
                    β_lower_rest = β_rest

                    for k, a in enumerate(poly):
                        if a == 0:
                            continue
                        k_fact = math.factorial(k)
                        # (k+1) upper terms carrying β forward.
                        for j in range(k + 1):
                            sign = -1 if (j & 1) else 1
                            falling = k_fact // math.factorial(k - j)
                            coef = C * a * sign * falling / (β_inner ** (j + 1))
                            deg = k - j
                            up_poly = tuple(
                                (1.0 + 0.0j) if i == deg else (0.0 + 0.0j)
                                for i in range(deg + 1)
                            )
                            new_terms.append((coef, up_poly, β_upper_rest))
                        # 1 lower term (constant, β-list reduced by one).
                        sign_last = -1 if ((k + 1) & 1) else 1
                        coef_lower = (C * a * sign_last * k_fact
                                       / (β_inner ** (k + 1)) * exp_lower)
                        new_terms.append(
                            (coef_lower, (1.0 + 0.0j,), β_lower_rest)
                        )
            terms = new_terms

        # Outermost integration: s_N from `lower` to `upper`.
        total = 0.0 + 0.0j
        u_top = upper - lower
        for (C, poly, β_list) in terms:
            β = β_list[0] if β_list else (0.0 + 0.0j)
            if abs(β) < eps:
                # Pure polynomial: ∫_lower^{upper} Σ a_k (s-lower)^k ds
                for k, a in enumerate(poly):
                    if a == 0:
                        continue
                    total += C * a * (u_top ** (k + 1)) / (k + 1)
            else:
                # ∫_lower^{upper} Σ a_k (s-lower)^k · exp(β·s) ds
                # = Σ_k a_k · [Σ_{j=0..k} (-1)^j k!/(k-j)! u_top^{k-j} exp(β·upper)/β^{j+1}
                #              + (-1)^{k+1} k! · exp(β·lower)/β^{k+1}]
                if abs((β * lower).real) > EXP_REAL_LIMIT:
                    return None
                if abs((β * upper).real) > EXP_REAL_LIMIT:
                    return None
                exp_top = cmath.exp(β * upper)
                exp_low = cmath.exp(β * lower)
                for k, a in enumerate(poly):
                    if a == 0:
                        continue
                    k_fact = math.factorial(k)
                    contrib = 0.0 + 0.0j
                    for j in range(k + 1):
                        sign = -1 if (j & 1) else 1
                        falling = k_fact // math.factorial(k - j)
                        contrib += (sign * falling * (u_top ** (k - j))
                                    * exp_top / (β ** (j + 1)))
                    sign_last = -1 if ((k + 1) & 1) else 1
                    contrib += sign_last * k_fact * exp_low / (β ** (k + 1))
                    total += C * a * contrib
        return total
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def _chain_with_intermediate_uppers(
    alphas_chain,
    L,
    upper_per_position,
    U_chain_top,
):
    r"""Chain-simplex integral with scalar uppers at arbitrary positions
    (Stage 3b-maximality, generalisation of ``_exp_over_chain_simplex``).

    Computes

       ∫_{L ≤ s_{σ(0)} ≤ … ≤ s_{σ(m-1)} ≤ U_chain_top  ∧  s_{σ(k)} ≤ U_k for k ∈ keys}
            ∏_k exp(α_k · s_{σ(k)})  ds

    where ``upper_per_position[k] = U_k`` is the scalar upper bound on
    position ``k`` in the chain (positions not in the dict are bounded
    only by chain ordering + ``U_chain_top``).

    Math
    ----
    Chain ordering + per-position scalar uppers give an "effective upper"
    at each position:

        effective_upper[k] = min over j ≥ k of (upper_per_position[j], U_chain_top)

    which is non-decreasing in ``k``.  Group consecutive positions
    with identical ``effective_upper`` into "levels"; for ``q+1``
    levels the top level has the largest upper (= ``effective_upper[m-1]``).

    For each lower level (i = 0..q-1, value ``U_i < U_top``), the chain
    crosses ``U_i`` at some position ``c_i ∈ [level_i.start, m-1]``.
    By chain ordering, the crossings are monotonic:
    ``c_0 ≤ c_1 ≤ … ≤ c_{q-1}``.  Enumerating valid tuples
    ``(c_0, c_1, …, c_{q-1})`` and computing a piece-product per
    tuple decomposes the integral exactly.

    Pieces from a cut tuple:
    * Piece ``i`` (``i = 0..q-1``): positions ``[c_{i-1}+1 … c_i]``
      with bounds ``[U_{i-1}, U_i]`` (where ``U_{-1} = L``).  May be
      empty when ``c_{i-1} = c_i`` (consecutive cuts at same position).
    * Final piece: positions ``[c_{q-1}+1 … m-1]`` with bounds
      ``[U_{q-1}, U_top]``.  May be empty when ``c_{q-1} = m-1``.

    Each non-empty piece is an independent chain simplex; the product
    gives the case's contribution.

    Returns ``None`` if every case produces ``None`` from the
    underlying chain simplex (genuine overflow); otherwise returns
    the analytic closed-form sum.
    """
    m = len(alphas_chain)
    if m == 0:
        return 1.0 + 0.0j
    if upper_per_position is None:
        upper_per_position = {}

    # Compute effective upper at each position.
    effective_upper = [U_chain_top] * m
    running_min = U_chain_top
    for k in range(m - 1, -1, -1):
        u_k = upper_per_position.get(k)
        if u_k is not None:
            running_min = min(running_min, float(u_k))
        effective_upper[k] = running_min

    if any(effective_upper[k] <= L for k in range(m)):
        return 0.0 + 0.0j

    # Group consecutive positions with identical effective_upper into
    # levels.  ``levels[i] = (start, end_inclusive, upper_value)``.
    levels = []
    k = 0
    while k < m:
        v = effective_upper[k]
        start = k
        while k < m and effective_upper[k] == v:
            k += 1
        levels.append((start, k - 1, v))

    # If only one level, no intermediate uppers: standard chain.
    if len(levels) == 1:
        top = levels[0][2]
        v = _exp_over_chain_simplex_fast(alphas_chain, L, top)
        if v is None:
            v = _exp_over_chain_simplex_polynomial(alphas_chain, L, top)
        return v

    # q non-top levels (with strictly smaller uppers than the top level).
    q = len(levels) - 1

    # For each non-top level, find the LATEST position with a direct
    # constraint (i.e., upper_per_position[k] is set).  This is the
    # minimum cut position for the level: the cut must happen at or
    # after the constraint's original position, not just at the level's
    # extended start (positions before the constraint inherit the
    # upper via chain ordering, not via a direct constraint).
    cut_min_per_level = []
    for i in range(q):
        start, end, _ = levels[i]
        latest_direct = None
        for k in range(start, end + 1):
            if k in upper_per_position:
                latest_direct = k
        cut_min_per_level.append(
            latest_direct if latest_direct is not None else start
        )

    total = 0.0 + 0.0j
    any_returned = False

    # Enumerate cut tuples (c_0, c_1, …, c_{q-1}) with each
    # c_i ∈ [cut_min_per_level[i], m-1] and c_0 ≤ c_1 ≤ … ≤ c_{q-1}.
    def _gen_cuts(i, prev_c):
        if i == q:
            yield ()
            return
        lo = max(prev_c, cut_min_per_level[i])
        for c in range(lo, m):
            for rest in _gen_cuts(i + 1, c):
                yield (c,) + rest

    for cuts in _gen_cuts(0, 0):
        pieces = []
        prev_end = -1
        prev_upper = L
        for i, c in enumerate(cuts):
            positions = list(range(prev_end + 1, c + 1))
            upper_val = levels[i][2]
            pieces.append({
                'positions': positions,
                'lower': prev_upper,
                'upper': upper_val,
            })
            prev_end = c
            prev_upper = upper_val
        # Final piece: top-level positions with [last_cut_upper, top_upper].
        top_upper = levels[-1][2]
        pieces.append({
            'positions': list(range(prev_end + 1, m)),
            'lower': prev_upper,
            'upper': top_upper,
        })

        # Multiply chain simplex evaluations across non-empty pieces.
        case_value = 1.0 + 0.0j
        case_ok = True
        for piece in pieces:
            if not piece['positions']:
                continue
            if piece['upper'] <= piece['lower']:
                case_value = 0.0 + 0.0j
                break
            alphas_piece = [alphas_chain[k] for k in piece['positions']]
            v = _exp_over_chain_simplex_fast(
                alphas_piece, piece['lower'], piece['upper'],
            )
            if v is None:
                v = _exp_over_chain_simplex_polynomial(
                    alphas_piece, piece['lower'], piece['upper'],
                )
            if v is None:
                case_ok = False
                break
            case_value *= v

        if case_ok:
            total += case_value
            any_returned = True

    return total if any_returned else None


USE_POSET_INTEGRATOR = True

# ─── 2026-05-17: bbox-cap consistency between analytic and scipy ──────
#
# Before this fix, the analytic m≥3 poset evaluator used
# ``L = earliest_ext - POSET_PHYSICAL_MARGIN`` (default 50.0) for the
# lower bound on the chain when no scalar lower constraint was present,
# while the scipy fallback used ``L = -OUTER_CAP`` (default 200.0).
# The two paths therefore integrated different domains on the unbounded
# direction of the polytope.  For theories with strictly retarded poles
# (Re β ≪ 0) the smooth integrand decays fast enough that the difference
# is negligible.  But for marginal-stability theories (Re β ≈ 0 — e.g.
# spike-reset near the firing-rate fixed point) the integrand oscillates
# without decay and the integral is genuinely cap-dependent: analytic
# at L=-50 and scipy at L=-200 disagree by ~10-13% per m≥3 subset, which
# compounds to a 4× aggregate error in the 1-loop value at spike-reset
# k=2 ell=1.
#
# When True (default), the scipy.nquad m≥3 path uses ``OUTER_CAP =
# POSET_PHYSICAL_MARGIN`` (50.0) instead of the hard-coded 200.0,
# matching the analytic poset path's lower-bound fallback.  This keeps
# the analytic path fast (it never overflows at the tighter cap) and
# brings the grouped Phase J scipy reference into agreement with the
# per-diag analytic value.  Both paths now compute the same regularized
# integral for unbounded-below polytopes.
#
# Set the flag to False to recover the pre-fix behaviour bit-identically
# (scipy at cap=200, analytic at cap=50).
# See ``docs/m_ge3_precision_bug_audit.md`` for the full evidence chain.
# Empirically the cap mismatch is NOT the dominant source of the
# spike-reset k=2 ell=1 disagreement: aligning the caps at 50 does not
# bring per-diag (+2.66e-3) into agreement with grouped (+6.38e-4).  The
# grouped path's analytic poset uses pre-summed (cancelled-within-group)
# pole tuples, which produces a smaller integrand magnitude and hence a
# different value than per-diag's individual-diagram pole tuples
# summed AFTER integration.  By linearity these should be equal, so the
# 4× discrepancy points to a bookkeeping difference in pole-tuple
# construction between the two paths.  Off by default; see
# ``docs/m_ge3_precision_bug_audit.md``.
USE_POSET_CAP_MATCH_SCIPY = False

# ─── Experimental: mpmath accumulation in the m≥3 poset evaluator ─────
#
# Hypothesis (2026-05-16): the per-diag analytic 1-loop at spike-reset
# k=2 ell=1 is 4× too high because ``_integrate_nd_polytope_poset_modesum``
# sums O(6^n_smooth) pole-tuple terms whose individual magnitudes can be
# large (close-paired poles produce O(1/Δp) residues; products of 5 such
# can be O(10^8)).  In float64, the cancellation down to the actual O(1)
# integral loses many digits per subset; the bias compounds across 313
# m≥3 subsets to give the observed 4× aggregate error.
#
# When the flag is True, the OUTER pole-tuple sum is accumulated in
# mpmath at 50-digit precision while individual term computation stays
# in float64.  Costs only the additions (negligible vs the chain simplex
# work).  When False (default), behaviour is bit-identical to pre-fix.
#
# This flag is provisional pending validation against
# ``test_grouped_vs_perdiag.py`` and the spike-reset k=2 ell=1 fixture.
# Empirically this did NOT address the spike-reset bug either — the
# poset accumulation has cancellation factor only ~21× (well within
# float64 precision).  Keeping the code path for possible future use
# on configs with bigger cancellation.  Off by default.
USE_POSET_MPMATH_ACCUMULATION = False


def _integrate_nd_polytope_poset_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    m,
    bbox_cap=POLYGON_BBOX_CAP,
    pole_tuples=None,
    plan=None,
):
    r"""Analytic ``∫_{polytope} Π_e [Σ_α C_α exp(λ_α · Δt_e)] · pref
                                ds_1 … ds_m`` for m ≥ 3 via causal-
    poset decomposition.

    Procedure:
      1. Extract the causal poset (DAG + scalar bounds) from the
         retardation constraints.  Fail (return ``None``) if any
         constraint is mixed (not a clean inter-axis or scalar
         bound).
      2. Resolve the COMMON scalar lower bound ``L`` for all
         integration variables.  Fail if scalar lowers differ across
         variables (the simple chain simplex form would over- or
         under-include regions).  Fall back to ``-bbox_cap`` when no
         scalar lower is present.
      3. Enumerate every linear extension σ of the poset.  Each is a
         disjoint chain simplex
           L ≤ s_{σ(0)} ≤ s_{σ(1)} ≤ … ≤ s_{σ(m-1)} ≤ U_σ
         where U_σ is the scalar upper of σ(m-1) (or ``bbox_cap``).
      4. For each pole tuple (α_e)_e:
           α_v = Σ_e λ_α_e · a_int_e[v]                (per orig var)
           γ   = Σ_e λ_α_e · c_ext_e                   (ext-time part)
           α_chain[k] = α_{σ(k)}                       (permute)
           contribution = pref · ∏ C_α_e · exp(γ) ·
                          _exp_over_chain_simplex(α_chain, L, U_σ)
      5. Sum across extensions and tuples.  Fail if any chain
         simplex returns ``None`` (degenerate β).

    Returns ``complex`` or ``None``.  ``None`` triggers a fallback to
    scipy.nquad in the caller.
    """
    import cmath
    if m < 3:
        return None  # m=2 has its own dedicated path
    _RUNTIME_COUNTERS['poset_attempted'] += 1
    n_smooth = len(smooth_edge_modes)
    if len(subset_constraint_data) != n_smooth:
        _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
        return None

    poset = _extract_causal_poset(
        subset_constraint_data, free_ext_vals, m,
    )
    if poset is None:
        _RUNTIME_COUNTERS['poset_extract_returned_none'] += 1
        _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
        return None

    L_value, lower_ok = _causal_poset_consistent_scalar_lower(poset)
    if not lower_ok:
        _RUNTIME_COUNTERS['poset_consistent_lower_failed'] += 1
        _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
        return None
    # ── Lower bound: tight physical fallback (Stage 3b-bounds) ──────
    # When ``_causal_poset_consistent_scalar_lower`` returns no scalar
    # lower (L_value is None), the integrand still extends "to the
    # past" only as far as the retarded propagator chain decays.
    # Using ``-bbox_cap = -200`` here was too loose: combined with
    # cumulative pole sums |Re β| > 3 (sum of 3+ retarded poles),
    # ``exp(β · L)`` in the closed form's lower-bound term overflows
    # past the 600 real-exponent threshold and the path bails to
    # scipy.  A physical bound — earliest external time minus a few
    # correlation times — is more than adequate.
    if L_value is not None:
        L = float(L_value)
    else:
        earliest_ext = min(list(free_ext_vals) + [0.0])
        L = earliest_ext - POSET_PHYSICAL_MARGIN

    upper_per_var = _causal_poset_consistent_scalar_upper(poset)

    # Pre-extract per-edge linear data (constant in pole tuple).
    c_ext_per_edge = [
        float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        for (a_int, a_ext, c0) in subset_constraint_data
    ]
    a_int_per_edge = [
        tuple(float(a_int[i]) for i in range(m))
        for (a_int, _a_ext, _c0) in subset_constraint_data
    ]

    pref = complex(prefactor_complex)
    # Experimental: accumulate the pole-tuple sum in mpmath to preserve
    # precision when the per-tuple terms have large opposite-signed
    # magnitudes that cancel down to a small result.  When the flag is
    # off (default), behaviour is bit-identical to the float64 path.
    use_mp_accum = USE_POSET_MPMATH_ACCUMULATION
    if use_mp_accum:
        from mpmath import mp as _mp, mpc as _mpc
        _saved_dps = _mp.dps
        _mp.dps = 50
        total_mp = _mpc(0, 0)
    total = 0.0 + 0.0j
    extensions = list(_enumerate_linear_extensions(poset))
    if not extensions:
        return None

    # Pre-resolve the chain-top upper bound per extension.
    # The upper-bound fallback stays at bbox_cap — retarded β gives
    # Re(β · U) < 0 here, which the exp underflows safely; the
    # closed-form's truncation at U=bbox_cap is negligible for typical
    # Hawkes pole magnitudes.  The lower-bound fix (above) is what
    # addresses the overflow guard firing in the closed form.
    upper_for_ext = [
        upper_per_var.get(sigma[m - 1], float(bbox_cap))
        for sigma in extensions
    ]
    # Pre-resolve the per-position upper map per extension.  Maps
    # chain position k → scalar upper on σ[k] (for the intermediate-
    # upper-aware chain integrator below).
    upper_per_position_per_ext = []
    for sigma in extensions:
        upp = {}
        for k in range(m):
            v = sigma[k]
            if v in upper_per_var:
                upp[k] = float(upper_per_var[v])
        upper_per_position_per_ext.append(upp)

    # ── Plan-cache fast path (Stage 4a-plan, 2026-05-15) ────────────
    # When the caller threads a pre-built plan through, the per-tuple
    # ``alphas_orig`` and the γ decomposition have been computed once
    # at subset setup.  Per call we only contract γ_slope with
    # free_ext_vals and reorder alphas per linear extension.  The
    # poset structure, scalar bounds, linear extensions and cut tuples
    # are all τ-dependent (they read free_ext_vals via c_eff) and
    # stay above this branch — they're rebuilt per call regardless.
    if plan is not None:
        pole_iter = plan['pole_tuples']
        alphas_per_tuple = plan['alphas_per_tuple']
        gamma_const_per_tuple = plan['gamma_const_per_tuple']
        gamma_slope_per_tuple_per_ext = plan[
            'gamma_slope_per_tuple_per_ext'
        ]
        n_ext = len(free_ext_vals)
        for t_idx, (C_prod, _lambdas) in enumerate(pole_iter):
            alphas_orig = alphas_per_tuple[t_idx]
            gamma = gamma_const_per_tuple[t_idx]
            slope_row = gamma_slope_per_tuple_per_ext[t_idx]
            for j in range(n_ext):
                gamma = gamma + slope_row[j] * free_ext_vals[j]
            if gamma.real > 600.0:
                return None
            try:
                term_const = pref * C_prod * cmath.exp(gamma)
            except (OverflowError, ValueError):
                return None
            if term_const == 0:
                continue
            for sigma, U_ext, upp_per_pos in zip(
                    extensions, upper_for_ext,
                    upper_per_position_per_ext):
                alphas_chain = [alphas_orig[sigma[k]] for k in range(m)]
                chain_val = _chain_with_intermediate_uppers(
                    alphas_chain, L, upp_per_pos, float(U_ext),
                )
                if chain_val is None:
                    _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
                    _RUNTIME_COUNTERS[
                        'chain_simplex_polynomial_returned_none'
                    ] += 1
                    if use_mp_accum:
                        _mp.dps = _saved_dps
                    return None
                if use_mp_accum:
                    _term_f64 = term_const * chain_val
                    total_mp = total_mp + _mpc(_term_f64.real, _term_f64.imag)
                else:
                    total += term_const * chain_val
        if use_mp_accum:
            _result = complex(float(total_mp.real), float(total_mp.imag))
            _mp.dps = _saved_dps
            return _result
        return total

    # ── Legacy path (no plan) ─────────────────────────────────────
    pole_iter = (
        pole_tuples if pole_tuples is not None
        else _enumerate_pole_tuples(smooth_edge_modes)
    )
    for C_prod, lambdas in pole_iter:
        # α_v for each ORIGINAL integration variable.
        alphas_orig = [0.0 + 0.0j] * m
        gamma = 0.0 + 0.0j
        for e in range(n_smooth):
            lam = lambdas[e]
            for v in range(m):
                alphas_orig[v] += lam * a_int_per_edge[e][v]
            gamma += lam * c_ext_per_edge[e]
        # Overflow guard on the γ-prefactor.  Stage 4a optim
        # (2026-05-15): only positive Re(γ) overflows ``cmath.exp``;
        # negative direction underflows to 0 (correct for decayed
        # integrand).  Matches the polygon/interval guards.
        if gamma.real > 600.0:
            return None
        try:
            term_const = pref * C_prod * cmath.exp(gamma)
        except (OverflowError, ValueError):
            return None
        if term_const == 0:
            continue
        # Sum across linear extensions of the poset.
        for sigma, U_ext, upp_per_pos in zip(
                extensions, upper_for_ext, upper_per_position_per_ext):
            alphas_chain = [alphas_orig[sigma[k]] for k in range(m)]

            # Check whether any non-maximal variable has a scalar upper.
            # If so, route through the intermediate-uppers helper which
            # handles the chain split via 2^p case enumeration.  When
            # all scalar uppers (if any) sit on the chain-top variable
            # σ[m-1], the helper short-circuits to the standard chain
            # closed form (no case enumeration).  This subsumes the
            # old "maximality bail" — we never return None on that
            # ground anymore.
            chain_val = _chain_with_intermediate_uppers(
                alphas_chain, L, upp_per_pos, float(U_ext),
            )
            if chain_val is None:
                # All sub-pieces overflowed.  Fall back to scipy.
                _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
                _RUNTIME_COUNTERS[
                    'chain_simplex_polynomial_returned_none'
                ] += 1
                if use_mp_accum:
                    _mp.dps = _saved_dps
                return None
            if use_mp_accum:
                _term_f64 = term_const * chain_val
                total_mp = total_mp + _mpc(_term_f64.real, _term_f64.imag)
            else:
                total += term_const * chain_val
    if use_mp_accum:
        _result = complex(float(total_mp.real), float(total_mp.imag))
        _mp.dps = _saved_dps
        return _result
    return total


def _integrate_1d_polytope_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    bbox_cap=POLYGON_BBOX_CAP,
    pole_tuples=None,
    plan=None,
):
    r"""Analytic ``∫_L^U Π_e [Σ_α C_α exp(λ_α · Δt_e)] · prefactor ds``
    for ``m = 1``.

    After pole-expansion the integrand is a sum of single-exponential
    terms ``A · exp(α_s · s + γ)`` where
        α_s = Σ_e λ_α_e · a_int_e[0]
        γ   = Σ_e λ_α_e · c_ext_e,  c_ext_e = c0_e + Σ_j a_ext_e[j]·t_free[j]
    Each term is integrated in closed form over the polytope interval
    ``[L, U]``.

    The interval bounds come from the smooth-edge retardation
    constraints ``a_int_e[0]·s + c_ext_e > 0``.  Unbounded sides are
    tracked as ±∞: the corresponding closed-form boundary term
    evaluates to 0 if the integrand decays in that direction
    (``sign(Re α_s)`` matches), else ``None`` (caller falls back to
    scipy.quad which handles the unbounded endpoint via the standard
    adaptive-quadrature substitution).

    ``pole_tuples`` (optional): pre-built iterable of ``(C_prod,
    lambdas)`` pairs that replaces ``_enumerate_pole_tuples
    (smooth_edge_modes)``.  Used by the grouped Phase J path to
    inject merged residues ``B_α = Σ_td cp_td · Π_e C^{(td)}_{α_e, e}``.

    Returns ``complex`` or ``None`` on overflow / divergent unbounded
    integrand.
    """
    import cmath
    import math
    n_smooth = len(smooth_edge_modes)
    _RUNTIME_COUNTERS['interval_attempted'] += 1
    if len(subset_constraint_data) != n_smooth:
        _RUNTIME_COUNTERS['interval_returned_none'] += 1
        return None

    # Resolve the feasible interval — track unboundedness exactly.
    L = -math.inf
    U = +math.inf
    for (a_int, a_ext, c0) in subset_constraint_data:
        a = float(a_int[0]) if a_int else 0.0
        c_eff = float(c0) + sum(
            float(a_ext[i]) * float(free_ext_vals[i])
            for i in range(len(a_ext))
        )
        if abs(a) < 1e-15:
            if c_eff <= 0:
                return 0.0 + 0.0j
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if L >= U:
        return 0.0 + 0.0j
    L_inf = math.isinf(L)
    U_inf = math.isinf(U)

    pref = complex(prefactor_complex)
    total = 0.0 + 0.0j

    # ── Plan-cache fast path (Stage 4a-plan, 2026-05-15) ────────────
    # When the caller threads a pre-built plan through, the per-tuple
    # ``α_s`` and ``γ`` decomposition has been computed once at
    # subset setup — only the γ slope contraction with free_ext_vals
    # remains.  See ``_build_modesum_plan``.
    if plan is not None:
        pole_iter = plan['pole_tuples']
        alphas_per_tuple = plan['alphas_per_tuple']
        gamma_const_per_tuple = plan['gamma_const_per_tuple']
        gamma_slope_per_tuple_per_ext = plan[
            'gamma_slope_per_tuple_per_ext'
        ]
        n_ext = len(free_ext_vals)
        for t_idx, (C_prod, _lambdas) in enumerate(pole_iter):
            alpha_s = alphas_per_tuple[t_idx][0]
            gamma = gamma_const_per_tuple[t_idx]
            slope_row = gamma_slope_per_tuple_per_ext[t_idx]
            for j in range(n_ext):
                gamma = gamma + slope_row[j] * free_ext_vals[j]
            # Same overflow guard / closed-form as the non-plan path
            # below.  Duplicated rather than fall-through so the hot
            # path stays branch-free on (plan is not None).
            if gamma.real > 600.0:
                return None
            try:
                term_const = pref * C_prod
            except (OverflowError, ValueError):
                return None
            if term_const == 0:
                continue
            try:
                if abs(alpha_s) < 1e-15:
                    if L_inf or U_inf:
                        return None
                    contrib = (U - L) * cmath.exp(gamma)
                else:
                    if U_inf:
                        if alpha_s.real >= 0:
                            return None
                        term_U = 0.0 + 0.0j
                    else:
                        arg = alpha_s * U + gamma
                        if arg.real > 600.0:
                            return None
                        term_U = cmath.exp(arg)
                    if L_inf:
                        if alpha_s.real <= 0:
                            return None
                        term_L = 0.0 + 0.0j
                    else:
                        arg = alpha_s * L + gamma
                        if arg.real > 600.0:
                            return None
                        term_L = cmath.exp(arg)
                    contrib = (term_U - term_L) / alpha_s
            except (OverflowError, ValueError):
                return None
            total += term_const * contrib
        return total

    # ── Legacy path (no plan): rebuild per-edge data per call ──────
    c_ext_per_edge = [
        float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        for (a_int, a_ext, c0) in subset_constraint_data
    ]
    a_int_per_edge = [
        float(a_int[0]) if a_int else 0.0
        for (a_int, _a_ext, _c0) in subset_constraint_data
    ]

    pole_iter = (
        pole_tuples if pole_tuples is not None
        else _enumerate_pole_tuples(smooth_edge_modes)
    )
    for C_prod, lambdas in pole_iter:
        alpha_s = 0.0 + 0.0j
        gamma = 0.0 + 0.0j
        for e in range(n_smooth):
            lam = lambdas[e]
            alpha_s += lam * a_int_per_edge[e]
            gamma += lam * c_ext_per_edge[e]
        # Overflow guard on the γ-prefactor (matches polygon path).
        # Stage 4a optim (2026-05-15): only positive Re(γ) overflows
        # ``cmath.exp``; negative direction underflows to 0 (which
        # gives the correct result for a fully-decayed integrand).
        if gamma.real > 600.0:
            return None
        try:
            term_const = pref * C_prod
        except (OverflowError, ValueError):
            return None
        if term_const == 0:
            continue
        try:
            if abs(alpha_s) < 1e-15:
                # α_s ≈ 0: integrand is constant exp(γ) in s.
                if L_inf or U_inf:
                    return None  # diverges
                contrib = (U - L) * cmath.exp(gamma)
            else:
                if U_inf:
                    if alpha_s.real >= 0:
                        return None  # would diverge at +∞
                    term_U = 0.0 + 0.0j
                else:
                    arg = alpha_s * U + gamma
                    if arg.real > 600.0:
                        return None
                    term_U = cmath.exp(arg)
                if L_inf:
                    if alpha_s.real <= 0:
                        return None  # would diverge at -∞
                    term_L = 0.0 + 0.0j
                else:
                    arg = alpha_s * L + gamma
                    if arg.real > 600.0:
                        return None
                    term_L = cmath.exp(arg)
                contrib = (term_U - term_L) / alpha_s
        except (OverflowError, ValueError):
            return None
        total += term_const * contrib
    return total


def _integrate_2d_polygon_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    bbox_cap=POLYGON_BBOX_CAP,
    pole_tuples=None,
    plan=None,
):
    r"""Analytic ∫∫_polygon Π_e [Σ_α C_α exp(λ_α · Δt_e)] · prefactor
                  ds_0 ds_1.

    Each per-edge mode sum is pole-expanded; the resulting sum of
    single-exponential terms ``A · exp(α·s_0 + β·s_1 + γ)`` is
    integrated analytically over the polygon defined by
    ``subset_constraint_data``.

    Returns ``complex`` or ``None`` if construction fails (e.g.
    polygon empty / degenerate, or the per-edge data is missing).

    Δt_e for each smooth edge is expressed as
        Δt_e = c0_e + a_int_e[0]·s_0 + a_int_e[1]·s_1 + Σ_j a_ext_e[j]·t_free[j]
    Substituting into ``Σ_α λ_α · Δt_e`` and rearranging gives the
    exponent ``γ + α_s·s_0 + β_s·s_1`` where
        α_s = Σ_e a_int_e[0] · λ_α_e
        β_s = Σ_e a_int_e[1] · λ_α_e
        γ   = Σ_e λ_α_e · (c0_e + Σ_j a_ext_e[j]·t_free[j])

    ``pole_tuples`` (optional): pre-built iterable of ``(C_prod, lambdas)``
    pairs that replaces ``_enumerate_pole_tuples(smooth_edge_modes)``.
    Used by the grouped Phase J path
    (``msrjd.integration.time_domain.grouped_integral``) to inject a
    merged-residue tensor ``B_α = Σ_td cp_td · Π_e C^{(td)}_{α_e, e}``
    in place of the per-edge Cartesian product.  When ``None``, the
    per-diagram default iterator runs.
    """
    import cmath
    n_smooth = len(smooth_edge_modes)
    _RUNTIME_COUNTERS['polygon_attempted'] += 1
    if len(subset_constraint_data) != n_smooth:
        # Smooth-edges-to-constraints mismatch shouldn't happen — the
        # caller built both from ``smooth_edges`` in lock-step.  Bail
        # to scipy.nquad fallback.
        _RUNTIME_COUNTERS['polygon_returned_none'] += 1
        return None

    # Polygon is shared across all pole tuples.
    polygon = _polygon_from_2d_constraints(
        subset_constraint_data, free_ext_vals, bbox_cap,
    )
    if len(polygon) < 3:
        # Empty or degenerate polygon → integral is zero.
        return 0.0 + 0.0j

    # Fan triangulation.
    triangles = [
        (polygon[0], polygon[i], polygon[i + 1])
        for i in range(1, len(polygon) - 1)
    ]

    total = 0.0 + 0.0j
    pref = complex(prefactor_complex)

    # ── Plan-cache fast path (Stage 4a-plan, 2026-05-15) ────────────
    # Per-tuple α_s, β_s and γ decomposition pre-computed at subset
    # setup; only γ_slope contraction and the per-triangle integral
    # remain per τ.  See ``_build_modesum_plan``.
    if plan is not None:
        pole_iter = plan['pole_tuples']
        alphas_per_tuple = plan['alphas_per_tuple']
        gamma_const_per_tuple = plan['gamma_const_per_tuple']
        gamma_slope_per_tuple_per_ext = plan[
            'gamma_slope_per_tuple_per_ext'
        ]
        n_ext = len(free_ext_vals)
        for t_idx, (C_prod, _lambdas) in enumerate(pole_iter):
            alpha_s, beta_s = alphas_per_tuple[t_idx]
            gamma = gamma_const_per_tuple[t_idx]
            slope_row = gamma_slope_per_tuple_per_ext[t_idx]
            for j in range(n_ext):
                gamma = gamma + slope_row[j] * free_ext_vals[j]
            if gamma.real > 600.0:
                return None
            try:
                term_const = pref * C_prod * cmath.exp(gamma)
            except (OverflowError, ValueError):
                return None
            if term_const == 0:
                continue
            tri_sum = 0.0 + 0.0j
            for (v0, v1, v2) in triangles:
                tri_contrib = _exp_over_triangle(
                    v0, v1, v2, alpha_s, beta_s
                )
                if tri_contrib is None:
                    return None
                tri_sum += tri_contrib
            total += term_const * tri_sum
        return total

    # ── Legacy path (no plan): rebuild per-edge data per call ──────
    # Precompute per-edge "ext-time c-contribution":
    #   c_ext_e = c0_e + Σ_j a_ext_e[j] · t_free[j]
    # so that γ for a given pole tuple is Σ_e λ_α_e · c_ext_e.
    c_ext_per_edge = [
        float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        for (a_int, a_ext, c0) in subset_constraint_data
    ]
    a_int_per_edge = [
        (float(a_int[0]), float(a_int[1]))
        for (a_int, _a_ext, _c0) in subset_constraint_data
    ]

    pole_iter = (
        pole_tuples if pole_tuples is not None
        else _enumerate_pole_tuples(smooth_edge_modes)
    )
    for C_prod, lambdas in pole_iter:
        # α_s, β_s, γ for this pole tuple.
        alpha_s = 0.0 + 0.0j
        beta_s = 0.0 + 0.0j
        gamma = 0.0 + 0.0j
        for e in range(n_smooth):
            lam = lambdas[e]
            a0, a1 = a_int_per_edge[e]
            alpha_s += lam * a0
            beta_s += lam * a1
            gamma += lam * c_ext_per_edge[e]
        # Overflow guard on the γ-prefactor.  cmath.exp overflows for
        # Re(γ) > ~709 and underflows to 0 for Re(γ) < ~-745;
        # underflow is the right behaviour (term decays to 0).
        # Stage 4a optim (2026-05-15): check only the positive-
        # overflow side, matching the fixed-direction guard inside
        # ``_exp_over_triangle``.
        if gamma.real > 600.0:
            return None
        try:
            term_const = pref * C_prod * cmath.exp(gamma)
        except (OverflowError, ValueError):
            return None
        if term_const == 0:
            continue
        # Triangle sum.  ``_exp_over_triangle`` returns ``None`` when
        # any per-term exp would overflow — propagate that as a
        # whole-subset fallback signal.
        tri_sum = 0.0 + 0.0j
        for (v0, v1, v2) in triangles:
            tri_contrib = _exp_over_triangle(v0, v1, v2, alpha_s, beta_s)
            if tri_contrib is None:
                return None
            tri_sum += tri_contrib
        total += term_const * tri_sum
    return total


# ───────────────────────────────────────────────────────────────────────
# Quadrature accuracy knob
# ───────────────────────────────────────────────────────────────────────
# Controls scipy.integrate.quad / nquad parameters for the vertex-time
# integrals. Loosen these for fast iterative checks; tighten for
# publication-quality results.
#
# Usage from a notebook cell (BEFORE running the numerics cell):
#   from msrjd.integration.time_domain import final_integral
#   final_integral.QUAD_OPTS = {'limit': 30, 'epsrel': 1e-3}
#
QUAD_OPTS = {
    'limit': 200,      # max subintervals for scipy.integrate.quad / nquad
}

# ───────────────────────────────────────────────────────────────────────
# Cumulant-kernel τ_v integration cap (non-local noise sources)
# ───────────────────────────────────────────────────────────────────────
# Diagrams with a NoiseSourceType vertex carry an extra integration
# variable τ_v parametrising the relative time between the source's
# legs (per-leg time map for non-local cumulant kernels).  The
# kernel itself decays on its natural timescale (e.g., σ for a
# Gaussian), so integrating τ_v over a half-infinite range
# (retard_L, +∞) — which is what the polytope alone gives —
# leaves scipy.quad's tan-substitution coordinate transform free
# to compress the kernel's central peak near the boundary, where
# adaptive sampling intermittently misses it.  Capping τ_v ∈
# (-CAP, +CAP) collapses the range to a finite interval where
# adaptive quadrature is well-behaved.  ±50 is safe for kernels
# with σ ≤ 5; loosen to ±200 (the polytope OUTER_CAP) if your
# kernel has heavy tails:
#   from msrjd.integration.time_domain import final_integral
#   final_integral.TAU_KERNEL_CAP = 200.0
TAU_KERNEL_CAP = 50.0


# ───────────────────────────────────────────────────────────────────────
# Heaviside guard mode
# ───────────────────────────────────────────────────────────────────────
# The polytope integrators wrap their integrand in
# ``_make_heaviside_filtered_integrand`` which returns 0 whenever any
# retarded ``Δt_e`` constraint is violated.  This is a defensive belt-
# and-braces measure: the polytope BOUNDS we pass to scipy.quad / nquad
# should already constrain the integration to the feasible region, so
# the wrapper SHOULD be redundant — except in cases where the bounds
# fall back to ``±OUTER_CAP`` (m=2 without a pure-s_1 constraint, or
# m≥3 with deferred-inner constraints).  In those cases the cap is a
# superset of the true polytope and the filter is what enforces
# correctness.
#
# For paths where the bounds are EXACT (m=1, m=2 with pure_s_1), the
# wrapper is pure overhead — ~few µs per integrand call.  At millions
# of calls per τ sweep the cumulative cost is meaningful.
#
# Default: ``DEBUG_HEAVISIDE_GUARD = False`` skips the filter on the
# exact-bound paths; the cap-fallback paths always apply it regardless
# (correctness is non-negotiable).
#
# Set to ``True`` to force the filter on every path — useful when
# validating a refactor that touches the polytope bound logic and you
# want a belt-and-braces sanity check.
DEBUG_HEAVISIDE_GUARD = False


# ───────────────────────────────────────────────────────────────────────
# Tree-level vertex-time integration
# ───────────────────────────────────────────────────────────────────────

def _loop_number_from_graph(typed_diagram):
    """Compute loop number from the diagram's graph structure.

    For a connected graph: L = |E| - |V| + 1.
    This avoids any frequency-domain dependency.
    """
    D = typed_diagram.prediagram[0]
    return D.num_edges() - D.num_verts() + 1


def integrate_diagram(
    typed_diagram,
    propagator_data,
    combined_prefactor,
    ext_time_vars,
    num_params=None,
    origin_leaf_idx=0,
    external_fields=None,
    representative_ir=None,  # deprecated, kept for backward compat
):
    r"""
    Vertex-time integration for a typed Feynman diagram at ANY loop
    order, evaluated via explicit numerical quadrature.

    Previously this function was named ``integrate_tree_diagram`` and
    asserted tree-level.  The core algorithm -- assign a time to each
    internal vertex, integrate over those times, and enforce the
    retarded Heaviside on every edge -- works identically for loop
    diagrams because our enumeration produces DAGs (the "loop" in
    Feynman terminology is a topological cycle in the underlying
    undirected graph, not a cyclic directed path).  Multi-edges
    between the same vertex pair are already supported through the
    3-tuple ``(u, v, label)`` edge keys used by
    ``_lookup_prop_indices``.

    Returns a dict whose `contribution` value is a Python callable

        f(*ext_time_values) -> complex

    taking `k` positional arguments in **canonical** order: position i
    is the time of ``external_fields[i]``.  If `origin_leaf_idx` is
    not None, the value supplied at that position is ignored (it was
    pinned to zero during integrand construction).

    Parameters
    ----------
    typed_diagram : TypedDiagram
        The typed diagram.  Any loop order is accepted; the
        integrator treats every internal vertex as an integration
        variable and every edge as a retarded-propagator factor.
        Needed for the prediagram `D`, leaf list, and
        ``propagator_indices``.
    propagator_data : dict
        Must contain `'pole_vals'`, `'C_mats'`, and optionally
        `'D_delta'` (for delta-coefficient detection).
    combined_prefactor : SR or numeric
        Sum of scalar prefactors over diagrams in the kernel group.
    ext_time_vars : list of SR
        `k` external time variables in canonical order:
        ``ext_time_vars[i]`` is the time of ``external_fields[i]``.
    num_params : dict or None
        Numerical parameter substitutions for the propagator matrix AND
        the combined prefactor. Required if either the propagator
        entries or the prefactor contain free symbolic parameters — the
        JIT-compiled integrand cannot be built until every symbol
        except the integration variables and external times has been
        substituted.
    origin_leaf_idx : int or None
        Which canonical position to pin to zero (i.e., which entry of
        `external_fields` provides the base time). Default 0.
    external_fields : list of tuple or None
        The canonical external field list as specified by the user,
        e.g. ``[('dn',1), ('dn',1), ('dn',2)]``.  Used to map each
        leaf to its canonical position so that ``contribution(t_1,
        t_2, t_3)`` always has position i = time of
        ``external_fields[i]``, regardless of the diagram's internal
        leaf ordering.  If None, falls back to position-based mapping
        (leaf j → ext_time_vars[j]).

    Returns
    -------
    dict with keys:
        'status' : 'ok' | 'empty_polytope' | 'failed'
        'contribution' : callable
            `f(*ext_time_values) -> complex`. For status != 'ok' this
            may be `None`.
        'integration_vars' : list of SR
            The non-leaf vertex time symbols that were integrated out.
        'stripped_integrand' : SR
            The symbolic integrand WITHOUT the Heaviside factors, for
            debugging.
        'constraints' : list of SR
            One SR expression per edge; each expression `dt_e` is the
            linear combination `t_head - t_tail` that must be positive
            for the integrand to be nonzero.
    """
    loop_number = _loop_number_from_graph(typed_diagram)

    D = typed_diagram.prediagram[0]
    leaves = list(typed_diagram.prediagram[2])
    leaf_set = set(leaves)

    if len(ext_time_vars) != len(leaves):
        raise ValueError(
            f"ext_time_vars has length {len(ext_time_vars)} but "
            f"the diagram has {len(leaves)} leaves."
        )

    # ── 1. Numerical G(t) matrix (smooth + delta parts) ──────────
    t_sym = SR.var('_t_td_')
    G_t_obj = build_G_t_matrix(propagator_data, t_sym, num_params=num_params)

    # ── 2. Enumerate inter-vertex Wick contractions ───────────────
    # For correlators with repeated external field types (e.g. two
    # dn₁ legs at different spacetime points), each DISTINCT way to
    # assign canonical positions to leaves is a separate Wick
    # contraction that contributes to the connected correlator.
    #
    # The dedup in symmetry.py merges diagrams that differ only in
    # which same-type leaf connects to which vertex (they have the
    # same field-type multiset per vertex).  We compensate here by
    # summing over all such permutations.
    #
    # For same-vertex permutations (e.g. 2 dn₁ both at one vertex),
    # the integrand is invariant under swap (commutative product),
    # so summing 2! mappings gives 2x overcounting → divide by 2!.
    # For cross-vertex permutations (dn₁ at different vertices), the
    # integrands differ → summing gives the full answer.
    import itertools as _itertools
    from math import factorial as _factorial

    if external_fields is not None and len(external_fields) == len(leaves):
        _leaf_fields = [typed_diagram.external_legs.get(lf) for lf in leaves]
        # Group canonical positions and leaves by field type
        _cp_by_field = {}
        _leaves_by_field = {}
        for cp, field in enumerate(external_fields):
            _cp_by_field.setdefault(field, []).append(cp)
        for j, field in enumerate(_leaf_fields):
            _leaves_by_field.setdefault(field, []).append(j)

        # Enumerate all canonical-to-leaf mappings
        _all_mappings = [{}]
        for field in sorted(_cp_by_field.keys(), key=str):
            cps = _cp_by_field[field]
            lfs = _leaves_by_field.get(field, [])
            if len(cps) != len(lfs):
                _all_mappings = [{j: j for j in range(len(leaves))}]
                break
            perms = list(_itertools.permutations(lfs))
            new_mappings = []
            for m in _all_mappings:
                for perm in perms:
                    nm = dict(m)
                    for cp, lf_idx in zip(cps, perm):
                        nm[cp] = lf_idx
                    new_mappings.append(nm)
            _all_mappings = new_mappings

        # Compensation: for each internal vertex V, for each same-type
        # field group at V with n_V leaves, divide by n_V!.  This
        # removes overcounting from within-vertex permutations (which
        # give the same integrand by commutativity).
        _vertex_of_leaf = {}
        for ek in typed_diagram.edge_types:
            u, v = ek[0], ek[1]
            if u in leaf_set and v not in leaf_set:
                _vertex_of_leaf[u] = v
            elif v in leaf_set and u not in leaf_set:
                _vertex_of_leaf[v] = u
        _vertex_field_counts = {}
        for lf in leaves:
            v = _vertex_of_leaf.get(lf)
            if v is None:
                continue
            field = typed_diagram.external_legs.get(lf)
            _vertex_field_counts.setdefault(v, {}).setdefault(field, 0)
            _vertex_field_counts[v][field] += 1
        _compensation = 1
        for v, fcounts in _vertex_field_counts.items():
            for field, count in fcounts.items():
                _compensation *= _factorial(count)
    else:
        # SAFETY WARNING: falling back to identity leaf→position
        # mapping.  For diagrams whose leaves have MIXED field types
        # (e.g. one δn_1 and one δv_2), the enumeration's leaf order
        # is not guaranteed to match ``external_fields`` — a mismatch
        # here produces a τ → −τ mirror image of the physical
        # correlator.  Always pass ``external_fields`` when k ≥ 2.
        _leaf_field_list = [typed_diagram.external_legs.get(lf)
                            for lf in leaves]
        if len(set(_leaf_field_list)) > 1:
            import warnings as _warnings
            _warnings.warn(
                "integrate_tree_diagram: external_fields not provided "
                "for a diagram with mixed leaf field types "
                f"({_leaf_field_list}).  The canonical leaf→position "
                "mapping will fall back to identity, which may produce "
                "a τ → −τ mirror image of the physical correlator.  "
                "Pass external_fields to fix.",
                stacklevel=2,
            )
        _all_mappings = [{j: j for j in range(len(leaves))}]
        _compensation = 1

    # Build vertex_time for the FIRST mapping; subsequent mappings
    # handled by permuting the positional arguments in the wrapper.
    vertex_time = {}
    _first_mapping = _all_mappings[0]
    for cp, leaf_idx in _first_mapping.items():
        lf = leaves[leaf_idx]
        t_ext = ext_time_vars[cp]
        if origin_leaf_idx is not None and cp == origin_leaf_idx:
            t_ext = SR(0)
        vertex_time[lf] = t_ext

    internal_vertices = [v for v in D.vertices() if v not in leaf_set]
    integration_vars = []
    for v in internal_vertices:
        s_v = SR.var(f's_v{v}_td_', latex_name=rf's_{{v_{{{v}}}}}')
        vertex_time[v] = s_v
        integration_vars.append(s_v)

    # ── 2b. Non-local cumulant noise sources: per-leg time map ────
    # For each NoiseSourceType vertex, the response legs sit at
    # *independent* times coupled by a non-local kernel κ^{(n)}(τ).
    # We anchor the vertex at its existing time symbol (= leg-0
    # time) and introduce one extra integration variable τ per
    # noise source, with leg-1 time = anchor − τ.  Edges leaving
    # this vertex through leg 1 (matched by their resp_leg pop_idx)
    # are routed through the leg-1 time symbol.  Plain SourceType
    # vertices (cortical Poisson, GTaS auto-cumulant) keep the
    # single-time semantics — vertex_leg_time stays empty for them
    # and the existing edge-build code path is unaffected.
    #
    # ConvVertexType vertices (conductance-style interaction
    # vertices, e.g. ``vt·v·Conv(g,n)``) extend the same scaffold
    # to one PHYSICAL leg per kernel attachment.  The shape of
    # ``vertex_leg_time[v]`` is identical (``{pop_idx_0based: SR time}``)
    # but ``vertex_leg_kind[v]`` tells the edge-routing code below
    # whether to match against ``edge_resp_leg`` (NoiseSourceType)
    # or ``edge_phys_leg`` (ConvVertexType).
    vertex_leg_time = {}        # {v: {leg_pop_idx_0based: SR time}}
    vertex_leg_kind = {}        # {v: 'response' | 'physical'}
    noise_source_specs = {}     # {v: list of cumulant_specs dicts}
    conv_vertex_specs  = {}     # {v: list of (tau_sym, att_dict) pairs}
    extra_tau_syms = []         # list of (tau_sym, vertex) pairs
    vertex_assignments = (
        getattr(typed_diagram, 'vertex_assignments', None) or {}
    )
    for v, vtype in vertex_assignments.items():
        if not isinstance(vtype, NoiseSourceType):
            continue
        if not vtype.cumulant_specs:
            continue
        # All specs on this vertex share the same leg multiset
        # (they were grouped by extract_source_types).  Read the
        # 0-based leg ordering from the first spec.
        legs0 = vtype.cumulant_specs[0]['legs']    # e.g. (0, 1)
        anchor_leg = legs0[0]
        other_leg  = legs0[1] if len(legs0) > 1 else legs0[0]
        if anchor_leg == other_leg:
            # Auto-cumulant survived as non-local (kernel had no
            # delta, but legs are equal) — treat both legs at the
            # anchor time and leave τ-integration in.  Rare in
            # practice; keeps the code uniform.
            other_leg = anchor_leg
        # Anchor time = the existing internal-vertex symbol
        anchor_time = vertex_time[v]
        # τ symbol — distinct per noise vertex
        tau_sym = SR.var(
            f's_v{v}_tau_td_', latex_name=rf'\tau_{{v_{{{v}}}}}'
        )
        extra_tau_syms.append((tau_sym, v))
        integration_vars.append(tau_sym)
        # Per-leg time map: leg-0 = anchor; leg-1 = anchor − τ
        vertex_leg_time[v] = {
            anchor_leg: anchor_time,
            other_leg:  anchor_time - tau_sym,
        }
        vertex_leg_kind[v] = 'response'
        noise_source_specs[v] = list(vtype.cumulant_specs)

    # ── 2c. Conductance-style interaction vertices ────────────────
    # ConvVertexType is the interaction-vertex analogue of
    # NoiseSourceType: one PHYSICAL leg per kernel attachment sits
    # at ``anchor_time − τ`` linked to the rest of the vertex via
    # the synaptic kernel ``g(τ)``.  See
    # ``docs/conductance_vertex_kernels_design.md`` for the math.
    #
    # A single ConvVertexType can carry several attachments (e.g.
    # ``vt · v · Conv(g1, n1) · Conv(g2, n2)``) — one τ per
    # attachment.  Each kernel-attached leg's pop-idx is recorded
    # under leg_index in the attachment dict so vertex_leg_time
    # can route edges through the right time symbol.
    for v, vtype in vertex_assignments.items():
        if not isinstance(vtype, ConvVertexType):
            continue
        if not vtype.kernel_attachments:
            continue
        if v in vertex_leg_time:
            # Vertex was already promoted by the NoiseSourceType
            # block — bigrade overlap is structurally impossible
            # (NoiseSourceType is n_phys=0, ConvVertexType is
            # n_phys≥1) but guard against the misclassification
            # rather than silently merging two leg maps.
            raise RuntimeError(
                f"vertex {v} appears both as NoiseSourceType and "
                f"ConvVertexType — these are mutually exclusive."
            )
        anchor_time = vertex_time[v]
        leg_time_map = {}
        # Track this vertex's τ↔attachment pairs locally — DON'T mutate
        # the attachment dict.  The same ConvVertexType instance can be
        # bound to multiple graph vertices in the typing engine, so any
        # mutation here would leak τ symbols across vertices in the
        # same diagram (vertex 2's τ overwriting vertex 4's, etc.).
        att_tau_pairs = []
        for att_idx, att in enumerate(vtype.kernel_attachments):
            # ``leg_index`` is the position within physical_legs
            # the kernel attaches to.  We key vertex_leg_time on
            # the leg's pop_idx (matching the edge-routing
            # convention used for NoiseSourceType).
            leg_tuple = att['leg']
            phys_pop_idx = leg_tuple[1] - 1  # 0-based
            tau_sym = SR.var(
                f's_v{v}_gtau{att_idx}_td_',
                latex_name=rf'\tau^{{g}}_{{v_{{{v}}},{att_idx}}}'
            )
            extra_tau_syms.append((tau_sym, v))
            integration_vars.append(tau_sym)
            # Kernel-attached leg sits at anchor − τ.
            leg_time_map[phys_pop_idx] = anchor_time - tau_sym
            att_tau_pairs.append((tau_sym, att))
        vertex_leg_time[v] = leg_time_map
        vertex_leg_kind[v] = 'physical'
        conv_vertex_specs[v] = att_tau_pairs

    # ── 3. Gather per-edge info: ri, pi, dt, delta_coeff, smooth factor
    def _resolve_leg_time(vert, edge_key, default_time):
        """Pick the right time symbol for ``vert``'s end of an edge.

        For NoiseSourceType (``vertex_leg_kind == 'response'``) the
        per-leg map is keyed on the edge's RESPONSE leg pop-idx; for
        ConvVertexType (``vertex_leg_kind == 'physical'``) it's keyed
        on the PHYSICAL leg pop-idx.  Plain vertices keep their
        single ``vertex_time`` entry — the routing falls through to
        ``default_time``.
        """
        if vert not in vertex_leg_time:
            return default_time
        edge_resp_leg, edge_phys_leg = typed_diagram.edge_types[edge_key]
        if vertex_leg_kind.get(vert) == 'physical':
            leg = edge_phys_leg
        else:
            leg = edge_resp_leg
        pop_idx = leg[1] - 1  # 0-based
        return vertex_leg_time[vert].get(pop_idx, default_time)

    edges = list(D.edges())
    edge_info = []
    for (u, v, lbl) in edges:
        ri, pi = _lookup_prop_indices(typed_diagram, (u, v, lbl))
        edge_key = (u, v, lbl)
        # Route edge tail / head through per-leg time when the
        # vertex carries a non-local kernel (noise source or
        # conductance interaction); otherwise use the standard
        # single-time map.
        t_u = _resolve_leg_time(u, edge_key, vertex_time[u])
        t_v = _resolve_leg_time(v, edge_key, vertex_time[v])
        dt = SR(t_v - t_u)
        delta_c = G_t_delta_coeff(G_t_obj, pi, ri)
        smooth_factor = G_t_entry(G_t_obj, pi, ri, dt, include_heaviside=False)
        edge_info.append({
            'u': u, 'v': v, 'lbl': lbl,
            'ri': ri, 'pi': pi,
            'dt_sym': dt,
            'delta_coeff': delta_c,
            'smooth_factor': smooth_factor,
        })

    # ── 3a. Mode-sum cache (Stage 2 of Phase J refactor) ─────────
    # Build one ``EdgeModeSum`` per edge, extracting the per-pole
    # residue from ``propagator_data['C_mats']`` ONCE.  The fast
    # subset evaluator (the hot path for plain diagrams) reuses
    # this across all 2^|branch| subsets instead of re-extracting
    # residues from SR on every call.  ``None`` if the propagator
    # data is incomplete, in which case the fast path stays on
    # the legacy tuple-based extractor.
    edge_mode_sums = _build_edge_mode_sums(edge_info, propagator_data)

    # Combined prefactor (numerical)
    cp = SR(combined_prefactor) if combined_prefactor is not None else SR(1)
    if num_params:
        cp = cp.subs(num_params)

    # ── 3b. Non-local cumulant kernel substitution ────────────────
    # For each NoiseSourceType vertex, replace each placeholder
    # symbol ``z_kappa_<noise>_<order>_<i>_<j>`` in ``cp`` with the
    # actual kernel SR expression returned by the user's kernel_fn,
    # evaluated at the per-vertex τ integration symbol.  The signs
    # and combinatorial factors that ``_build_cumulant_action``
    # multiplied onto each placeholder (typically -1/2) are already
    # in ``cp``; the substitution carries them through.  The result
    # is a cp that is now an explicit function of the τ symbols,
    # which the existing fast_callable / nquad path handles
    # naturally because each τ is in ``integration_vars``.
    if noise_source_specs:
        kappa_subs = {}
        for v, specs in noise_source_specs.items():
            tau_sym_v = next(
                (ts for ts, vv in extra_tau_syms if vv == v), None
            )
            if tau_sym_v is None:
                continue
            for spec in specs:
                i_leg, j_leg = spec['legs'][0], spec['legs'][1]
                kappa_subs[spec['symbol']] = SR(
                    spec['kernel_fn'](i_leg, j_leg, tau_sym_v)
                )
        if kappa_subs:
            cp = cp.subs(kappa_subs)
            if num_params:
                # Re-substitute num_params now that kernel symbols
                # like ns.lambda_X, ns.mu_shift_diff have been
                # introduced via kernel_fn evaluation.
                cp = cp.subs(num_params)
            # Try to combine like terms (e.g. ordered pairs (i,j) and
            # (j,i) with symmetric kernels collapse to a single term).
            # simplify_full can be expensive but the cumulant prefactor
            # is a small SR expression so it's cheap here.
            try:
                cp = cp.simplify_full()
            except (ValueError, RuntimeError, AttributeError):
                pass

    # ── 3c. Conductance-vertex kernel substitution ────────────────
    # For each ConvVertexType, replace the kernel SR symbol in ``cp``
    # with the time-domain kernel ``g(τ)`` evaluated at the
    # per-attachment τ symbol introduced in section 2c.  Same shape
    # as the noise-source path above — once substituted, ``cp`` is
    # an explicit function of the τ symbols and flows through
    # fast_callable / nquad without further special handling.
    # ``conv_kernel_extracted`` flags whether all ConvVertex kernels were
    # successfully decomposed into single-exponential pseudo-edges.  When
    # True, ``cp`` has the kernel symbols replaced by ``1`` and the kernel
    # weights live in the per-subset mode-sum (analytic path).  When False
    # (e.g. polynomial-prefactor alpha kernel that single-exp extraction
    # rejects), we fall back to the legacy ``cp.subs(g(τ))`` substitution
    # and the slower SR + scipy.nquad path.
    conv_kernel_extracted = False
    # ``conv_extracted_modes`` collects ``(tau_sym, C, lam)`` triples for
    # the per-subset pseudo-edge build.  Populated only when extraction
    # succeeds; empty otherwise.
    conv_extracted_modes = []
    if conv_vertex_specs:
        from sage.all import heaviside as _sage_heaviside
        _all_extractable = True
        _modes_buffer = []
        for v, att_tau_pairs in conv_vertex_specs.items():
            for tau_sym, att in att_tau_pairs:
                td_fn = att.get('kernel_td_fn')
                if td_fn is None:
                    _all_extractable = False
                    break
                kernel_sr = SR(td_fn(tau_sym))
                kernel_sr = kernel_sr.substitute_function(
                    _sage_heaviside, lambda _x: SR(1)
                )
                if num_params:
                    kernel_sr = kernel_sr.subs(num_params)
                mode = _extract_exp_mode(kernel_sr, tau_sym)
                if mode is None:
                    _all_extractable = False
                    break
                C, lam = mode
                _modes_buffer.append((tau_sym, C, lam))
            if not _all_extractable:
                break
        conv_kernel_extracted = _all_extractable
        if _all_extractable:
            conv_extracted_modes = _modes_buffer

        # Build the kernel-symbol → SR substitution dict.  When the
        # kernel will live in a pseudo-edge, substitute with ``1`` so
        # the analytic mode-sum doesn't double-count it.  Otherwise
        # substitute with the full ``g(τ)`` SR expression so the SR +
        # scipy.nquad path sees a complete integrand.
        g_subs = {}
        for v, att_tau_pairs in conv_vertex_specs.items():
            for tau_sym, att in att_tau_pairs:
                td_fn = att.get('kernel_td_fn')
                if td_fn is None:
                    continue
                if conv_kernel_extracted:
                    g_subs[att['symbol']] = SR(1)
                else:
                    td_expr = SR(td_fn(tau_sym))
                    td_expr = td_expr.substitute_function(
                        _sage_heaviside, lambda _x: SR(1)
                    )
                    g_subs[att['symbol']] = td_expr
        if g_subs:
            cp = cp.subs(g_subs)
            if num_params:
                cp = cp.subs(num_params)
            try:
                cp = cp.simplify_full()
            except (ValueError, RuntimeError, AttributeError):
                pass

    # ── 4. External-time bookkeeping ─────────────────────────────
    free_ext_idx = [
        j for j in range(len(ext_time_vars))
        if (origin_leaf_idx is None or j != origin_leaf_idx)
    ]
    free_ext_syms = [ext_time_vars[j] for j in free_ext_idx]

    n_edges = len(edge_info)

    # Pre-classify edges by whether they CAN be chosen as δ.  An edge
    # with ``|delta_coeff| < 1e-15`` contributes nothing if placed in
    # the δ subset, so the inner subset loop only branches on edges
    # that have a nonzero δ part.  Edges with zero δ are always in
    # ``smooth_edges``.  Pre-classification turns a 2^|E| enumeration
    # into 2^|branch| with no behaviour change relative to the old
    # ``continue`` guard inside the loop.
    branch_edge_indices: list[int] = []
    forced_smooth_indices: list[int] = []
    for i in range(n_edges):
        if abs(complex(edge_info[i]['delta_coeff'])) < 1e-15:
            forced_smooth_indices.append(i)
        else:
            branch_edge_indices.append(i)
    n_branch = len(branch_edge_indices)
    n_subsets_total = 2 ** n_branch

    # Expose the |S|=0 (all smooth) symbolic integrand and constraints
    # for debugging / display, matching the pre-fix return shape.
    display_stripped = cp
    for ei in edge_info:
        display_stripped = display_stripped * ei['smooth_factor']
    try:
        display_stripped = display_stripped.expand()
    except Exception:
        pass
    display_constraints = [ei['dt_sym'] for ei in edge_info]

    # Accumulators
    subset_contributions = []   # continuous smooth contributions (callable)
    delta_contributions = []    # shot-noise δ spikes (structured dicts)
    n_shotnoise_skipped = 0
    subset_diagnostics = []

    for branch_bits in range(n_subsets_total):
        # Δ subset: branch-edges with bit set, in original edge-index order.
        delta_edges = [
            branch_edge_indices[k] for k in range(n_branch)
            if (branch_bits >> k) & 1
        ]
        # Smooth subset: forced-smooth edges + branch-edges with bit unset.
        # Preserve original edge-index order so downstream constraint
        # extraction and zip(edge_info, ...) sees the same ordering as
        # the pre-Stage-1a code path.
        branch_smooth = [
            branch_edge_indices[k] for k in range(n_branch)
            if not ((branch_bits >> k) & 1)
        ]
        smooth_edges = sorted(forced_smooth_indices + branch_smooth)

        # ── Solve the δ-edge equalities: eliminate integration vars
        # by substitution. For each δ edge, set dt_e = 0 and solve for
        # an integration variable appearing in the equation; if no
        # integration variable is available, the equation becomes a
        # constraint among external times (shot-noise δ, skip).
        substitutions = {}
        remaining_int_vars = list(integration_vars)
        ext_time_equalities = []  # residual constraints on ext times

        subset_infeasible = False
        for ei_idx in delta_edges:
            eq_expr = edge_info[ei_idx]['dt_sym'].subs(substitutions)
            eq_expr = SR(eq_expr)
            # Find an integration variable to solve for
            int_var_to_eliminate = None
            try:
                eq_vars = set(eq_expr.variables())
            except AttributeError:
                eq_vars = set()
            for iv in remaining_int_vars:
                if iv in eq_vars:
                    int_var_to_eliminate = iv
                    break
            if int_var_to_eliminate is not None:
                try:
                    sol = sage_solve(
                        eq_expr == 0, int_var_to_eliminate,
                        solution_dict=True,
                    )
                except Exception:
                    sol = []
                if not sol:
                    subset_infeasible = True
                    break
                new_rhs = sol[0][int_var_to_eliminate]
                substitutions[int_var_to_eliminate] = new_rhs
                remaining_int_vars.remove(int_var_to_eliminate)
                # Resolve transitively: apply the new substitution to
                # the RHS of every existing entry so a chain like
                # ``{a: f(b), b: g(c)}`` collapses to ``{a: f(g(c)), b: g(c)}``.
                # Sage's ``.subs(dict)`` is a parallel one-pass operation
                # and does NOT chain substitutions — without this fixup
                # ``cp.subs(substitutions)`` would leave ``b`` exposed in
                # the result, breaking the integrator's free-symbol
                # audit downstream.  Pre-existing concern that becomes
                # load-bearing with multi-τ ConvVertexType diagrams.
                # Cheap early-skip: only chain-resolve if some EXISTING
                # RHS actually mentions the variable we just eliminated.
                # For typical non-ConvVertex diagrams the chain almost
                # never forms (the per-edge ``eq_expr.subs(substitutions)``
                # above already applies prior subs before solving), so
                # the inner SR.subs() loop is pure overhead.  Walk
                # variables once per existing entry — much cheaper than
                # blindly calling SR.subs.
                affected_keys = []
                for _k, _rhs in substitutions.items():
                    if _k == int_var_to_eliminate:
                        continue
                    try:
                        _rhs_vars = SR(_rhs).variables()
                    except (AttributeError, TypeError):
                        continue
                    if int_var_to_eliminate in _rhs_vars:
                        affected_keys.append(_k)
                if affected_keys:
                    _chain_subs = {int_var_to_eliminate: new_rhs}
                    for _k in affected_keys:
                        substitutions[_k] = SR(
                            substitutions[_k]
                        ).subs(_chain_subs)
            else:
                # No integration variable to eliminate → this is a
                # constraint on external times alone. If it's
                # identically zero, the δ is satisfied trivially; if
                # not, it's a shot-noise δ(τ)-style contribution.
                ext_time_equalities.append(eq_expr)

        if subset_infeasible:
            continue

        # Shot-noise check: any nontrivial residual equality among
        # external times means this subset contributes a δ(τ) spike
        # at the hypersurface where that equality holds. Instead of
        # skipping it, compute a structured δ-contribution: a numeric
        # coefficient together with the linear equality and any
        # retardation half-space constraints, so downstream code can
        # insert it into a discrete τ grid.
        has_shotnoise = False
        nontrivial_equalities = []
        for eq in ext_time_equalities:
            try:
                if bool(eq.is_zero()):
                    continue
            except Exception:
                pass
            # Nontrivial equation → shot-noise
            has_shotnoise = True
            nontrivial_equalities.append(SR(eq))
        if has_shotnoise:
            n_shotnoise_skipped += 1
            subset_diagnostics.append({
                'delta_edges': delta_edges,
                'smooth_edges': smooth_edges,
                'status': 'shotnoise',
                'ext_time_equalities': ext_time_equalities,
            })

            # For the MVP we only support the single-equality case
            # (one δ(a · τ + c) spike per subset). Multi-equality cases
            # would correspond to δ(τ_a − τ_b) · δ(τ_c − τ_d) style
            # "double-delta" spikes which are rare at tree level and
            # deferred to Extension 1.
            if len(nontrivial_equalities) != 1:
                continue

            # Build the numeric coefficient: combined_pf × ∏ δ-coeffs
            #                               × (smooth factors with δ subs applied)
            subset_factor_delta = cp
            for ei_idx in delta_edges:
                subset_factor_delta = (
                    subset_factor_delta
                    * SR(edge_info[ei_idx]['delta_coeff'])
                )
            for ei_idx in smooth_edges:
                subset_factor_delta = (
                    subset_factor_delta
                    * edge_info[ei_idx]['smooth_factor']
                )
            subset_factor_delta = subset_factor_delta.subs(substitutions)
            # NOTE: ``.subs(num_params)`` used to run here but it is
            # redundant: ``build_G_t_matrix(propagator_data, t_sym,
            # num_params=num_params)`` has already substituted
            # ``num_params`` into every ``edge_info[i]['smooth_factor']``
            # and ``edge_info[i]['delta_coeff']``, and ``cp`` was
            # substituted at the top of ``integrate_diagram``.  The only
            # remaining free variables here are ext-time symbols and
            # integration variables -- neither of which is in
            # ``num_params``.  Removed 2026-04-21 (audit Fix #A, ~6%
            # speedup on k=2 ell=1 quadratic Hawkes).
            try:
                subset_factor_delta = subset_factor_delta.expand()
            except Exception:
                pass

            # At the shot-noise hypersurface, any remaining integration
            # variables must already have been eliminated (otherwise it
            # wouldn't be a pure δ contribution — m_sub > 0 after all
            # deltas applied means we'd need to integrate further).
            # For the MVP star tree, this condition always holds.
            if remaining_int_vars:
                subset_diagnostics.append({
                    'status': 'shotnoise_with_remaining_int_vars',
                    'delta_edges': delta_edges,
                    'remaining': list(remaining_int_vars),
                })
                continue

            # Extract the linear form of the equality in terms of
            # free_ext_syms: a · x + c = 0.
            eq = nontrivial_equalities[0]
            try:
                eq_a = [float(eq.coefficient(s)) for s in free_ext_syms]
                eq_c = float(eq.subs({s: 0 for s in free_ext_syms}))
            except (TypeError, ValueError):
                continue

            # Check the equality is actually nontrivial (not all zero)
            if all(abs(a) < 1e-15 for a in eq_a):
                # Pure numeric residual: if nonzero, no contribution;
                # if zero, it's trivially satisfied (shouldn't happen
                # because we already filtered `is_zero()` above).
                continue

            # The symbolic factor evaluated at the δ surface is just
            # subset_factor_delta — it already has all the pin-substitutions
            # applied and can be evaluated numerically once free_ext_vals
            # are supplied. For the MVP case where remaining_int_vars is
            # empty, subset_factor_delta depends only on free_ext_syms
            # (or is a constant).
            try:
                coeff_free_vars = set(subset_factor_delta.variables())
            except AttributeError:
                coeff_free_vars = set()
            unexpected_in_coeff = coeff_free_vars - set(free_ext_syms)
            if unexpected_in_coeff:
                # Can't build a callable coefficient with free params left
                continue

            try:
                coeff_fc = fast_callable(
                    subset_factor_delta,
                    vars=list(free_ext_syms),
                    domain=CDF,
                )
            except Exception:
                continue

            # Retardation constraints at the δ point: for the MVP's
            # shot-noise subset (all edges δ, no smooth), there are
            # no retardation constraints. But in principle a mixed
            # δ+smooth shot-noise subset could have them.
            retard_data_delta = []
            for ei_idx in smooth_edges:
                c_retard = SR(
                    edge_info[ei_idx]['dt_sym']
                ).subs(substitutions)
                try:
                    a_ext = [
                        float(c_retard.coefficient(s))
                        for s in free_ext_syms
                    ]
                    c0 = float(c_retard.subs(
                        {s: 0 for s in free_ext_syms}
                    ))
                except (TypeError, ValueError):
                    continue
                retard_data_delta.append((a_ext, c0))

            delta_contributions.append({
                'coeff_fc': coeff_fc,
                'equality_a': eq_a,
                'equality_c': eq_c,
                'equality_symbolic': eq,
                'retardation_data': retard_data_delta,
                'delta_edges': list(delta_edges),
                'free_ext_idx': list(free_ext_idx),
            })
            continue

        # ── Stage 4a optim (2026-05-15): early prefactor build +
        # analytic-eligibility check.  When the subset can be served
        # by `_fast_eval` (pole/residue closure) + analytic modesum
        # integrators, the SR-based `subset_factor` build + `.expand()`
        # + `fast_callable()` compile is dead weight (Stage-3b
        # profiling: ~52% of integrate_diagram wall time on the k=2
        # ell=1 quad config, JIT tree never queried).  Skip it when
        # eligible; fall back to the full SR build only when needed.
        prefactor_num = cp
        for _ei_idx in delta_edges:
            prefactor_num = prefactor_num * SR(
                edge_info[_ei_idx]['delta_coeff']
            )

        try:
            _prefactor_c = complex(CDF(SR(prefactor_num)))
            _prefactor_is_numerical = True
        except Exception:
            _prefactor_c = None
            _prefactor_is_numerical = False

        # Numerical zero-skip (structural; no simplify_full).
        if _prefactor_is_numerical and _prefactor_c == 0:
            continue

        # Build retardation constraints for smooth edges (with δ subs applied).
        # Always needed for the polytope path.
        subset_retard = []
        for ei_idx in smooth_edges:
            c = SR(edge_info[ei_idx]['dt_sym']).subs(substitutions)
            subset_retard.append(c)

        m_sub = len(remaining_int_vars)
        fc_vars_sub = list(remaining_int_vars) + list(free_ext_syms)

        # `_analytic_eligible` ⇒ the per-call evaluator goes through
        # `_fast_eval` (built below from pole/residue cache) for any
        # residual scipy.nquad path, and the analytic modesum
        # integrators (m=1/2/≥3) handle the closed-form path.
        # Neither needs `integrand_fc_sub`, so we skip the entire
        # SR + `.expand()` + `fast_callable()` build chain.
        #
        # Conductance vertices (ConvVertexType) are eligible whenever
        # their kernels decompose into single-exponential pseudo-edges
        # (``conv_kernel_extracted``); the per-attachment ``(C, λ)``
        # mode is appended to ``smooth_edge_modes`` below, alongside
        # a polytope constraint ``τ > 0`` from the pseudo-edge's
        # ``dt = +τ`` linear form.  NoiseSourceType kernels remain on
        # the slow path until they get an analogous extraction.
        _conv_only_leg_times = (
            bool(conv_vertex_specs)
            and not noise_source_specs
            and conv_kernel_extracted
        )
        _analytic_eligible = (
            (not vertex_leg_time or _conv_only_leg_times)
            and edge_mode_sums is not None
            and _prefactor_is_numerical
        )

        if _analytic_eligible:
            subset_factor = None
            integrand_fc_sub = None
        else:
            # NoiseSourceType kernel diagrams, or non-numerical
            # prefactor, or missing edge_mode_sums cache — build the
            # full SR + fast_callable path for the residual scipy
            # integrand.  NOTE: ``.subs(num_params)`` removed
            # 2026-04-21 (audit Fix #A); see commit notes.
            subset_factor = cp
            for ei_idx in delta_edges:
                subset_factor = subset_factor * SR(
                    edge_info[ei_idx]['delta_coeff']
                )
            for ei_idx in smooth_edges:
                subset_factor = (
                    subset_factor * edge_info[ei_idx]['smooth_factor']
                )
            subset_factor = subset_factor.subs(substitutions)
            try:
                subset_factor = subset_factor.expand()
            except Exception:
                pass

            # Structural zero-check (avoids Maxima simplify_full,
            # which can hang or blow up for complex Hawkes integrands).
            try:
                if subset_factor.is_trivial_zero():
                    continue
            except AttributeError:
                if str(subset_factor) == '0':
                    continue

            # Free-symbol audit — catches num_params pass-through bugs.
            try:
                subset_free_vars = set(subset_factor.variables())
            except AttributeError:
                subset_free_vars = set()
            unexpected = subset_free_vars - set(fc_vars_sub)
            if unexpected:
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'stripped_integrand': display_stripped,
                    'constraints': display_constraints,
                    'reason': (
                        f"[subset {bin(branch_bits)}] stripped integrand "
                        f"contains unexpected free symbols {unexpected}; "
                        f"pass them via num_params."
                    ),
                }

            try:
                integrand_fc_sub = fast_callable(
                    subset_factor, vars=fc_vars_sub, domain=CDF,
                )
            except Exception as exc:
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'stripped_integrand': display_stripped,
                    'constraints': display_constraints,
                    'reason': (
                        f"[subset {bin(branch_bits)}] fast_callable "
                        f"failed: {exc}"
                    ),
                }

        # Extract linear coefficients for the polytope
        subset_constraint_data = []
        constraint_err = None
        for c in subset_retard:
            c_sr = SR(c)
            try:
                a_int = [float(c_sr.coefficient(v))
                         for v in remaining_int_vars]
                a_ext = [float(c_sr.coefficient(s)) for s in free_ext_syms]
                zero_subs = {v: 0
                             for v in list(remaining_int_vars)
                             + list(free_ext_syms)}
                c0 = float(c_sr.subs(zero_subs))
            except (TypeError, ValueError) as exc:
                constraint_err = exc
                break
            subset_constraint_data.append((a_int, a_ext, c0))

        # ── Cap each surviving τ_v integration variable to a finite
        # range ──────────────────────────────────────────────────────
        # Cumulant kernels (Gaussian, etc.) decay rapidly on a
        # kernel-natural timescale.  Without a finite cap, scipy.quad
        # integrates over (retard_L, +∞), and the adaptive Cauchy /
        # tan-substitution coordinate transform compresses the
        # kernel's central peak near the upper boundary of the
        # transformed parameter range.  The peak then gets
        # intermittently missed by the sampling — producing the
        # spurious spikes seen in non-local diagram contributions
        # at τ values where the external time t puts the kernel
        # peak in the "danger zone."  Capping τ_v ∈ (−CAP, +CAP)
        # collapses the integration to a finite interval where
        # adaptive quad is well-behaved.  Outside ±5σ a Gaussian
        # is < 1e-6 of its peak; ±50 with σ ~ 1 is overkill and
        # safe for any kernel with σ < 10.
        if extra_tau_syms and remaining_int_vars:
            n_iv = len(remaining_int_vars)
            n_ext = len(free_ext_syms)
            # Which τ symbols come from a ConvVertexType.  These are
            # causal-synaptic-kernel τ = vertex_time − leg_time, which
            # by construction have support τ ≥ 0; the heaviside(τ) was
            # stripped from cp above and the lower bound 0 (not −CAP)
            # is what carries the causality constraint into the
            # polytope.
            conv_tau_syms = set()
            for _v, _att_pairs in conv_vertex_specs.items():
                for _tau, _att in _att_pairs:
                    conv_tau_syms.add(_tau)
            for (tau_s, _v) in extra_tau_syms:
                if tau_s not in remaining_int_vars:
                    continue
                idx = remaining_int_vars.index(tau_s)
                is_conv_tau = tau_s in conv_tau_syms
                # Upper cap:  -τ_v + CAP > 0  ⇒  τ_v < CAP
                a_up  = [0.0] * n_iv
                a_up[idx] = -1.0
                subset_constraint_data.append(
                    (a_up, [0.0] * n_ext, TAU_KERNEL_CAP)
                )
                # Lower cap.  For NoiseSourceType: +τ_v + CAP > 0
                # (symmetric around 0).  For ConvVertexType: +τ_v > 0
                # (causal — kernel support starts at τ = 0).
                a_lo  = [0.0] * n_iv
                a_lo[idx] = +1.0
                lo_c0 = 0.0 if is_conv_tau else TAU_KERNEL_CAP
                subset_constraint_data.append(
                    (a_lo, [0.0] * n_ext, lo_c0)
                )
        if constraint_err is not None:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'stripped_integrand': display_stripped,
                'constraints': display_constraints,
                'reason': (
                    f"[subset {bin(branch_bits)}] constraint not "
                    f"linear: {constraint_err}"
                ),
            }

        # ── Conv-vertex kernel pseudo-edges ──────────────────────────
        # Each surviving ``(τ, C, λ)`` mode contributes one synthetic
        # smooth edge with ``Δt = +τ`` and a single mode-sum pole at
        # ``λ = −1/τ_g``.  Appended to ``smooth_edge_modes`` and
        # ``subset_constraint_data`` at the analytic-mode-sum call
        # sites below.  The pseudo-edge's polytope constraint
        # ``Δt > 0`` SUBSUMES the one-sided ``τ > 0`` cap added above,
        # so we drop the redundant cap entry to avoid double-counting
        # the constraint in the poset extraction.
        conv_pseudo_edges = []
        conv_pseudo_constraints = []
        if conv_kernel_extracted and conv_extracted_modes:
            n_iv_sub = len(remaining_int_vars)
            n_ext_sub = len(free_ext_syms)
            for (tau_sym, C, lam) in conv_extracted_modes:
                try:
                    tau_idx_sub = remaining_int_vars.index(tau_sym)
                except ValueError:
                    # τ was eliminated by δ-edge substitution — can't
                    # happen for a ConvVertex kernel τ (no edge ever
                    # has dt = τ_kernel alone in this code path), but
                    # guard defensively.
                    conv_pseudo_edges = None
                    conv_pseudo_constraints = None
                    break
                a_int_pe = [0.0] * n_iv_sub
                a_int_pe[tau_idx_sub] = 1.0
                conv_pseudo_edges.append(EdgeModeSum(
                    ri=-1, pi=-1,            # synthetic — never indexed
                    delta_coeff=complex(0.0),
                    modes=((C, lam),),
                ))
                conv_pseudo_constraints.append(
                    (a_int_pe, [0.0] * n_ext_sub, 0.0)
                )
            # Drop the redundant one-sided ``τ > 0`` cap (added in the
            # bounds loop above) so the polytope doesn't carry the same
            # constraint twice — the poset integrator treats them as
            # independent dim-1 retardation walls and rejects identical
            # ones.
            if conv_pseudo_edges is not None:
                _conv_tau_set = {tau for (tau, _, _) in conv_extracted_modes}
                _to_keep = []
                for c_tuple in subset_constraint_data:
                    a_int_c, a_ext_c, c0_c = c_tuple
                    # Identify "+τ + 0 > 0" rows for our τs (a_ext all
                    # zero, c0 == 0, a_int has a single +1 at the τ
                    # column).
                    nonzero = [(j, v) for j, v in enumerate(a_int_c) if v != 0]
                    is_pure_tau_pos = (
                        c0_c == 0.0
                        and not any(v != 0 for v in a_ext_c)
                        and len(nonzero) == 1
                        and nonzero[0][1] == 1.0
                        and remaining_int_vars[nonzero[0][0]] in _conv_tau_set
                    )
                    if not is_pure_tau_pos:
                        _to_keep.append(c_tuple)
                subset_constraint_data = _to_keep

        # Fix E (2026-04-21): direct numerical per-edge evaluator
        # reconstructs P · Π_e Σ_k C_e^{(k)} · exp(I · p_k · Δt_e)
        # from the propagator's pole / residue data plus the
        # already-extracted ``subset_constraint_data``, without
        # materialising the distributed |edges|^|poles|-term sum
        # that fast_callable would have to compile.  Overflow-safe
        # by edge-product bound Σ_k |C_e^{(k)}| for Δt_e ≥ 0.
        #
        # Stage 4a optim (2026-05-15): ``prefactor_num`` is now
        # computed early above (used for the analytic-eligibility
        # check).  Only the ri/pi lookup remains here.
        smooth_edges_ri_pi = [
            (edge_info[_ei_idx]['ri'], edge_info[_ei_idx]['pi'])
            for _ei_idx in smooth_edges
        ]
        if vertex_leg_time and not _conv_only_leg_times:
            # NoiseSourceType vertex with non-rational (Gaussian, etc.)
            # kernel — the fast pole/residue evaluator assumes
            # P · Π_e Σ_k C_e^{(k)} exp(i p_k Δt_e), which the
            # cumulant κ-factor breaks.  Fall through to the generic
            # fast_callable path.  ConvVertex kernels with successful
            # single-exp extraction take the pseudo-edge analytic
            # path instead (handled below).
            _fast_eval = None
        else:
            # Stage 2: prefer the pre-built ``edge_mode_sums`` cache
            # (residues + λ_α extracted once per edge at the top of
            # integrate_diagram).  Falls back to the legacy in-call
            # extraction path if the cache wasn't built (incomplete
            # propagator_data).  For ConvVertex-only diagrams we
            # extend the smooth-edge list with the per-attachment
            # kernel pseudo-edges + their dt constraints.
            if edge_mode_sums is not None:
                smooth_edge_modes = [
                    edge_mode_sums[_ei_idx] for _ei_idx in smooth_edges
                ]
                if conv_pseudo_edges:
                    smooth_edge_modes = smooth_edge_modes + conv_pseudo_edges
                    subset_constraint_data = (
                        subset_constraint_data + conv_pseudo_constraints
                    )
                _fast_eval = _build_fast_subset_evaluator_from_modes(
                    prefactor_num,
                    smooth_edge_modes,
                    subset_constraint_data,
                    m_sub,
                )
            else:
                _fast_eval = _build_fast_subset_evaluator(
                    propagator_data,
                    prefactor_num,
                    smooth_edges_ri_pi,
                    subset_constraint_data,
                    m_sub,
                )
        integrand_for_quad = (
            _fast_eval if _fast_eval is not None else integrand_fc_sub
        )

        # Capture the smooth-edge EdgeModeSum subset + complex
        # prefactor for the analytic mode-sum paths:
        #   m = 1  → ``_integrate_1d_polytope_modesum``     (Stage 4a-perdiag)
        #   m = 2  → ``_integrate_2d_polygon_modesum``      (Stage 3a-full)
        #   m ≥ 3  → ``_integrate_nd_polytope_poset_modesum`` (Stage 3b)
        # ``None`` means the closure-only fallback path applies for
        # both (the constraints either can't be extracted or have
        # a non-numerical prefactor still).
        _smooth_edge_modes = None
        _modesum_prefactor_c = None
        _modesum_enabled = (
            (USE_1D_INTEGRATOR and m_sub == 1)
            or (USE_POLYGON_M2_INTEGRATOR and m_sub == 2)
            or (USE_POSET_INTEGRATOR and m_sub >= 3)
        )
        # ``_prefactor_c`` was already computed early (Stage 4a optim
        # 2026-05-15); reuse it instead of re-converting through SR.
        _pole_tuples_cache = None
        _modesum_plan = None
        if (_modesum_enabled
                and (not vertex_leg_time or _conv_only_leg_times)
                and edge_mode_sums is not None
                and _prefactor_is_numerical):
            _modesum_prefactor_c = _prefactor_c
            _smooth_edge_modes = [
                edge_mode_sums[_ei_idx] for _ei_idx in smooth_edges
            ]
            if conv_pseudo_edges:
                # Conv-vertex kernels enter as single-mode pseudo-edges
                # appended to the smooth-edge list, with their own
                # ``Δt = +τ`` polytope constraint already merged into
                # subset_constraint_data above.
                _smooth_edge_modes = _smooth_edge_modes + conv_pseudo_edges
            # Stage 4a-plan (2026-05-15): pre-compute the τ-invariant
            # per-pole-tuple data (alphas, γ_const, γ_slope) for the
            # entire τ grid.  Each analytic integrator threads this
            # plan through its inner loop and skips the per-call
            # edge-loop that would otherwise rebuild α_s / γ on every
            # ``_contrib(free_vals)`` invocation.  Trades a single
            # subset-setup cost for (N_τ − 1) call-time recomputations.
            _modesum_plan = _build_modesum_plan(
                _smooth_edge_modes,
                subset_constraint_data,
                m_sub,
                len(free_ext_syms),
            )
            # Keep the legacy cache populated so any caller still
            # passing ``pole_tuples=`` keeps working unchanged.
            _pole_tuples_cache = _modesum_plan['pole_tuples']

        # Build this subset's contribution callable
        def _make_subset_contrib(fc, cdata, m_val,
                                  modes=None, pref_c=None,
                                  pole_tuples=None, plan=None):
            def _contrib(free_vals):
                # m=1 analytic 1D interval (Stage 4a-perdiag).
                if (modes is not None and pref_c is not None
                        and m_val == 1):
                    interval_val = _integrate_1d_polytope_modesum(
                        smooth_edge_modes=modes,
                        prefactor_complex=pref_c,
                        subset_constraint_data=cdata,
                        free_ext_vals=free_vals,
                        pole_tuples=pole_tuples,
                        plan=plan,
                    )
                    if interval_val is not None:
                        return interval_val
                # m=2 analytic polygon (Stage 3a-full).
                if (modes is not None and pref_c is not None
                        and m_val == 2):
                    poly_val = _integrate_2d_polygon_modesum(
                        smooth_edge_modes=modes,
                        prefactor_complex=pref_c,
                        subset_constraint_data=cdata,
                        free_ext_vals=free_vals,
                        pole_tuples=pole_tuples,
                        plan=plan,
                    )
                    if poly_val is not None:
                        return poly_val
                # m≥3 analytic causal-poset chain simplex (Stage 3b).
                if (modes is not None and pref_c is not None
                        and m_val >= 3):
                    poset_val = _integrate_nd_polytope_poset_modesum(
                        smooth_edge_modes=modes,
                        prefactor_complex=pref_c,
                        subset_constraint_data=cdata,
                        free_ext_vals=free_vals,
                        m=m_val,
                        pole_tuples=pole_tuples,
                        plan=plan,
                    )
                    if poset_val is not None:
                        return poset_val
                # Closure-only fallback via scipy.nquad.
                resolved = []
                for (a_int, a_ext, c0) in cdata:
                    c_eff = c0 + sum(a_ext[i] * free_vals[i]
                                     for i in range(len(a_ext)))
                    resolved.append((list(a_int), c_eff))
                return _integrate_polytope(fc, resolved, free_vals, m_val)
            return _contrib

        subset_contributions.append(
            _make_subset_contrib(
                integrand_for_quad, subset_constraint_data, m_sub,
                modes=_smooth_edge_modes,
                pref_c=_modesum_prefactor_c,
                pole_tuples=_pole_tuples_cache,
                plan=_modesum_plan,
            )
        )
        # ``_evaluator_label`` tags the INTENDED analytic path for
        # this subset.  At runtime the closure may still fall back
        # to scipy.nquad if the analytic path returns None (mixed
        # constraint, non-uniform bounds, degenerate β).  The label
        # records the design intent; runtime falls through to
        # 'fast_numpy' silently.
        if _smooth_edge_modes is not None and m_sub == 1:
            _evaluator_label = 'interval_modesum'
        elif _smooth_edge_modes is not None and m_sub == 2:
            _evaluator_label = 'polygon_modesum'
        elif _smooth_edge_modes is not None and m_sub >= 3:
            _evaluator_label = 'poset_modesum'
        elif _fast_eval is not None:
            _evaluator_label = 'fast_numpy'
        else:
            _evaluator_label = 'fast_callable'
        subset_diagnostics.append({
            'delta_edges': delta_edges,
            'smooth_edges': smooth_edges,
            'status': 'evaluated',
            'm_after_delta': m_sub,
            'evaluator': _evaluator_label,
        })

    # ── Build the final contribution callable ─────────────────────
    # The subset_contributions were built using the FIRST mapping.
    # To include all inter-vertex Wick contractions, we evaluate the
    # same integrand with the input arguments permuted for each
    # alternative mapping, then sum and divide by the compensation
    # factor (which removes overcounting of same-vertex permutations).
    _m0 = _all_mappings[0]
    _m0_inv = {v: k for k, v in _m0.items()}
    _k = len(ext_time_vars)

    # For each alternative mapping m, compute the permutation that
    # converts canonical-order inputs into the order the first-mapping
    # integrand expects.
    _perms = []
    for _m in _all_mappings:
        _perm = [0] * _k
        for _cp in range(_k):
            _perm[_m0_inv[_m[_cp]]] = _cp
        _perms.append(tuple(_perm))

    _comp = _compensation

    def contribution(*ext_time_values):
        if len(ext_time_values) != len(ext_time_vars):
            raise ValueError(
                f"contribution() expects {len(ext_time_vars)} positional "
                f"arguments (one per ext_time_var); got "
                f"{len(ext_time_values)}."
            )
        total = 0.0 + 0.0j
        for perm in _perms:
            permuted = [ext_time_values[perm[j]] for j in range(_k)]
            free_vals = [float(permuted[j]) for j in free_ext_idx]
            for cfn in subset_contributions:
                total = total + complex(cfn(free_vals))
        return total / _comp

    # If non-local cumulant kernels were substituted in cp, expose
    # the τ-dependent prefactor (= the substituted, simplified cp,
    # already num_params-substituted) so the display layer can place
    # it inside the integral with the propagator factors.  When no
    # NoiseSourceType vertex is present, this is just cp (==
    # combined_prefactor.subs(num_params)) and the display layer
    # treats it as a τ-independent prefactor outside the integral.
    has_cumulant_kernel = bool(noise_source_specs)
    return {
        'status': 'ok',
        'contribution': contribution,
        'delta_contributions': delta_contributions,
        'integration_vars': integration_vars,
        'stripped_integrand': display_stripped,
        'constraints': display_constraints,
        'edge_info': edge_info,
        'n_subsets_evaluated': len(subset_contributions),
        'n_delta_contributions': len(delta_contributions),
        'n_shotnoise_skipped': n_shotnoise_skipped,
        'subset_diagnostics': subset_diagnostics,
        'cumulant_prefactor':       cp if has_cumulant_kernel else None,
        'has_cumulant_kernel':      has_cumulant_kernel,
    }


def eval_delta_contributions_on_tau_grid(
    delta_contributions,
    tau_grid,
    free_ext_dim=1,
    vary_index=0,
    fixed_values=None,
):
    r"""
    Convert a list of symbolic δ-spike contributions (from
    `integrate_tree_diagram` or `compute_correction_td`) into a
    discretized contribution on a 1D τ grid, optionally restricted to
    a slice of a higher-dimensional external-time space.

    Each delta contribution stores a linear equality
    `a · x + c = 0` in the free-external-time vector `x` (of length
    `free_ext_dim`). On a 1D slice where all-but-one entry of `x` is
    fixed, the equality collapses to `a_vary · τ + c' = 0` where
    `c' = c + Σ_{j ≠ vary_index} a_j · fixed_values[j]`. We solve
    `τ_fire = −c' / a_vary`, evaluate the coefficient callable at the
    (fixed + varying) point, check any retardation half-spaces, and
    deposit `coeff / |a_vary| / Δτ` into the nearest bin on
    `tau_grid`.

    Parameters
    ----------
    delta_contributions : list of dict
        As returned by `integrate_tree_diagram` under
        `delta_contributions` or `compute_correction_td` under the
        same key. Each entry has `equality_a`, `equality_c`,
        `coeff_fc`, and `retardation_data`.
    tau_grid : 1-D numpy array
        Uniformly spaced grid of τ values along the axis being varied.
        Bin width is inferred as `tau_grid[1] - tau_grid[0]`.
    free_ext_dim : int, default 1
        Number of free-external-time dimensions that each
        `delta_contribution['equality_a']` is parameterized over.
        For k=2 with one leaf pinned this is 1; for k=3 it is 2.
    vary_index : int, default 0
        Which component of the free-external-time vector is swept by
        `tau_grid`. All other components are pinned to the
        corresponding value in `fixed_values`.
    fixed_values : dict or None
        Mapping `{j: value}` for indices `j ≠ vary_index`. Any index
        not supplied is pinned to 0.0. For the common k=3 "slice
        through the origin" the default (all zeros) is usually what
        the caller wants.

    Returns
    -------
    numpy.ndarray (complex)
        An array the same length as `tau_grid`, with zeros everywhere
        except at the bins where δ contributions fire.

    Notes
    -----
    A δ contribution whose equality is IDENTICALLY zero on the chosen
    slice (i.e., `a_vary == 0` but the remaining slice residual is
    also zero) corresponds to a "δ along the whole slice" — the
    contribution is continuous along the varying axis rather than
    concentrated at a single bin. These contributions are **skipped**
    by this helper with a silent pass; they should be handled by a
    full 2D grid evaluator or by adding an explicit continuous
    contribution to the smooth total. Callers can inspect
    `delta_contributions` directly to detect this case.
    """
    import numpy as np

    if free_ext_dim < 1:
        raise ValueError(f"free_ext_dim must be >= 1, got {free_ext_dim}")
    if not (0 <= vary_index < free_ext_dim):
        raise ValueError(
            f"vary_index={vary_index} out of range for "
            f"free_ext_dim={free_ext_dim}"
        )
    if fixed_values is None:
        fixed_values = {}

    tau_grid = np.asarray(tau_grid, dtype=float)
    if tau_grid.size < 2:
        raise ValueError("tau_grid must have at least 2 points")
    dtau = float(tau_grid[1] - tau_grid[0])
    out = np.zeros_like(tau_grid, dtype=complex)

    # Build the full fixed-values vector (length free_ext_dim, with
    # vary_index slot filled in per-evaluation).
    other_indices = [j for j in range(free_ext_dim) if j != vary_index]
    fixed_vec_template = [0.0] * free_ext_dim
    for j in other_indices:
        fixed_vec_template[j] = float(fixed_values.get(j, 0.0))

    for dc in delta_contributions:
        eq_a = dc['equality_a']
        eq_c = dc['equality_c']
        coeff_fc = dc['coeff_fc']
        if len(eq_a) != free_ext_dim:
            # Dimension mismatch between the delta contribution and
            # the caller's advertised free_ext_dim. Silently skip.
            continue

        a_vary = eq_a[vary_index]
        c_eff = eq_c + sum(
            eq_a[j] * fixed_vec_template[j] for j in other_indices
        )

        if abs(a_vary) < 1e-15:
            if abs(c_eff) > 1e-12:
                # Infeasible: the δ surface doesn't intersect this
                # slice at all → zero contribution.
                continue
            # DEGENERATE: the δ equality is satisfied EVERYWHERE on
            # this slice. The contribution is NOT a spike — it's a
            # smooth continuous function along the varying axis:
            #   C_degenerate(τ) = coeff_fc(τ) (no 1/dtau divisor)
            # This is the "pair-driven" piece: e.g. for two identical
            # pop-1 fields at the same time (δ(τ₁) on the τ₁=0
            # slice), the remaining smooth propagator to pop-2 gives
            # a decaying function of τ₂.
            for i_tau, tau_val in enumerate(tau_grid):
                eval_vec = list(fixed_vec_template)
                eval_vec[vary_index] = float(tau_val)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue
                # Check retardation constraints at this point
                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False
                        break
                    r_val = c0 + sum(
                        a_list[j] * eval_vec[j]
                        for j in range(free_ext_dim)
                    )
                    if r_val <= 0:
                        retard_ok = False
                        break
                if retard_ok:
                    out[i_tau] = out[i_tau] + val
            continue
        tau_fire = -c_eff / a_vary

        # Build the full point in free-ext-time space for evaluating
        # the coefficient callable
        eval_vec = list(fixed_vec_template)
        eval_vec[vary_index] = float(tau_fire)

        coeff_fc = dc['coeff_fc']
        try:
            coeff_val = complex(coeff_fc(*eval_vec))
        except Exception:
            continue

        # Check retardation constraints at the fire point
        retard_ok = True
        for (a_list, c0) in dc.get('retardation_data', []):
            if len(a_list) != free_ext_dim:
                retard_ok = False
                break
            val = c0 + sum(a_list[j] * eval_vec[j] for j in range(free_ext_dim))
            if val <= 0:
                retard_ok = False
                break
        if not retard_ok:
            continue

        # δ(a_vary · τ + c_eff) = δ(τ − τ_fire) / |a_vary|
        weight = coeff_val / abs(a_vary)

        # Find the nearest grid bin and add weight / dtau
        idx = int(np.argmin(np.abs(tau_grid - tau_fire)))
        out[idx] = out[idx] + weight / dtau

    return out


def eval_delta_contributions_on_2d_grid(
    delta_contributions,
    tau1_grid,
    tau2_grid,
    free_ext_dim=3,
    grid_axes=(1, 2),
    fixed_values=None,
):
    r"""
    Discretize δ-spike contributions onto a 2D grid for heatmap display.

    For each delta contribution with linear equality `a · x + c = 0`
    in the free-external-time vector `x`, this helper finds all 2D
    grid points where the equality is approximately satisfied (within
    half a grid step) and deposits the coefficient value there.

    Parameters
    ----------
    delta_contributions : list of dict
        As returned by `compute_correction_td` under `delta_contributions`.
    tau1_grid, tau2_grid : 1-D numpy arrays
        The two axes of the 2D heatmap grid. Must be uniformly spaced.
    free_ext_dim : int
        Total number of free external-time dimensions.
    grid_axes : tuple of two ints
        Which components of the free-ext vector correspond to the two
        grid axes. Default (1, 2) for k=3 with t_a=0 pinned.
    fixed_values : dict or None
        Values for any free-ext components NOT on the grid axes.
        Default: {0: 0.0} (the reference field at time 0).

    Returns
    -------
    numpy.ndarray (complex), shape (len(tau1_grid), len(tau2_grid))
        The discretized delta contributions. Add to the smooth `total_C`
        heatmap to get the full theory prediction.
    """
    import numpy as np

    if fixed_values is None:
        fixed_values = {0: 0.0}

    tau1 = np.asarray(tau1_grid, dtype=float)
    tau2 = np.asarray(tau2_grid, dtype=float)
    dt1 = float(tau1[1] - tau1[0]) if len(tau1) > 1 else 1.0
    dt2 = float(tau2[1] - tau2[0]) if len(tau2) > 1 else 1.0
    ax1, ax2 = grid_axes

    out = np.zeros((len(tau1), len(tau2)), dtype=complex)

    for dc in delta_contributions:
        eq_a = dc['equality_a']
        eq_c = dc['equality_c']
        coeff_fc = dc['coeff_fc']
        if len(eq_a) != free_ext_dim:
            continue

        a1 = eq_a[ax1]  # coefficient on the first grid axis
        a2 = eq_a[ax2]  # coefficient on the second grid axis

        # Contribution from fixed (non-grid) axes to the equality
        c_fixed = eq_c
        for idx in range(free_ext_dim):
            if idx != ax1 and idx != ax2:
                c_fixed += eq_a[idx] * float(fixed_values.get(idx, 0.0))

        # The equality on the grid is: a1·τ₁ + a2·τ₂ + c_fixed = 0
        # This is a line in the (τ₁, τ₂) plane. We find grid bins
        # that this line passes through.

        if abs(a1) < 1e-15 and abs(a2) < 1e-15:
            # No dependence on grid axes — either always or never fires
            if abs(c_fixed) < 1e-12:
                # Fires everywhere — evaluate coeff at every point
                for i, t1 in enumerate(tau1):
                    for j, t2 in enumerate(tau2):
                        eval_vec = [0.0] * free_ext_dim
                        for k_idx, v in fixed_values.items():
                            eval_vec[k_idx] = float(v)
                        eval_vec[ax1] = float(t1)
                        eval_vec[ax2] = float(t2)
                        try:
                            val = complex(coeff_fc(*eval_vec))
                        except Exception:
                            continue
                        # Check retardation
                        retard_ok = True
                        for (a_list, c0) in dc.get('retardation_data', []):
                            if len(a_list) != free_ext_dim:
                                retard_ok = False; break
                            rv = c0 + sum(a_list[m] * eval_vec[m]
                                          for m in range(free_ext_dim))
                            if rv <= 0:
                                retard_ok = False; break
                        if retard_ok:
                            # This is a 2D degenerate — the delta fires
                            # on the full 2D grid. No 1/dt divisor
                            # (the delta is in a direction orthogonal
                            # to the grid plane).
                            out[i, j] += val
            continue

        # The delta line a1·τ₁ + a2·τ₂ + c_fixed = 0 crosses the grid.
        # For each row i (fixed τ₁), solve for τ₂:
        #   τ₂_fire = -(a1·τ₁[i] + c_fixed) / a2
        # OR for each column j (fixed τ₂), solve for τ₁:
        #   τ₁_fire = -(a2·τ₂[j] + c_fixed) / a1
        # Use whichever axis has the larger coefficient (better resolved).

        if abs(a2) >= abs(a1):
            # Sweep along τ₁ axis, solve for τ₂ at each row
            for i, t1 in enumerate(tau1):
                t2_fire = -(a1 * t1 + c_fixed) / a2
                j = int(np.argmin(np.abs(tau2 - t2_fire)))
                if abs(tau2[j] - t2_fire) > dt2:
                    continue  # not within a grid cell

                eval_vec = [0.0] * free_ext_dim
                for k_idx, v in fixed_values.items():
                    eval_vec[k_idx] = float(v)
                eval_vec[ax1] = float(t1)
                eval_vec[ax2] = float(t2_fire)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue

                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False; break
                    rv = c0 + sum(a_list[m] * eval_vec[m]
                                  for m in range(free_ext_dim))
                    if rv <= 0:
                        retard_ok = False; break
                if not retard_ok:
                    continue

                # The delta δ(a2·τ₂ + ...) has Jacobian 1/|a2|.
                # On the grid with spacing dt2, the bin density is
                # coeff / (|a2| · dt2).
                weight = val / (abs(a2) * dt2)
                out[i, j] += weight
        else:
            # Sweep along τ₂ axis, solve for τ₁ at each column
            for j, t2 in enumerate(tau2):
                t1_fire = -(a2 * t2 + c_fixed) / a1
                i = int(np.argmin(np.abs(tau1 - t1_fire)))
                if abs(tau1[i] - t1_fire) > dt1:
                    continue

                eval_vec = [0.0] * free_ext_dim
                for k_idx, v in fixed_values.items():
                    eval_vec[k_idx] = float(v)
                eval_vec[ax1] = float(t1_fire)
                eval_vec[ax2] = float(t2)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue

                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False; break
                    rv = c0 + sum(a_list[m] * eval_vec[m]
                                  for m in range(free_ext_dim))
                    if rv <= 0:
                        retard_ok = False; break
                if not retard_ok:
                    continue

                weight = val / (abs(a1) * dt1)
                out[i, j] += weight

    return out


# ───────────────────────────────────────────────────────────────────────
# Fast numerical subset-integrand evaluator (Fix E, 2026-04-21)
# ───────────────────────────────────────────────────────────────────────

def _build_fast_subset_evaluator(
    propagator_data,
    prefactor_num,
    smooth_edges_ri_pi,
    subset_constraint_data,
    m_sub,
):
    r"""Return a Python callable that evaluates a subset's smooth
    integrand numerically, without going through
    ``fast_callable(subset_factor.expand())``.

    The subset integrand is structurally

        P · Π_e  Σ_k  C_e^{(k)} · exp(I · p_k · Δt_e)

    where ``P`` is the product of the combined prefactor and all
    δ-edge coefficients, the outer product runs over smooth edges,
    and the inner sum runs over poles.  ``fast_callable``'s overflow-
    safe form (Fix from 2026-04-08, commit 388fa7c) was to first call
    ``subset_factor.expand()``, distributing the ``|edges|^|poles|``
    cross-product into a sum of single exponentials.  For a 5-edge
    diagram with 4 poles that's ~988 single-exp terms, all compiled
    into the JIT tree.  cProfile of the k=2 ell=1 quadratic Hawkes
    V=5 diagram (2026-04-21) shows ~18 µs per ``fast_callable`` call,
    × 750 k samples per τ point, = the full 13 s of Phase J wall time.

    This evaluator skips the expansion entirely: each edge contributes
    ``Σ_k C_e^{(k)} exp(I p_k Δt_e)`` computed independently and
    multiplied in.  The product is bounded term-by-term by ``|C_e^{(k)}|``
    for Δt_e ≥ 0 (which the Heaviside filter guarantees -- see
    ``_make_heaviside_filtered_integrand``), so the pre-cancellation
    overflow that motivated the expand fix cannot occur.

    Returns None if the numerical extraction fails (e.g., ``prefactor``
    isn't purely numerical after ``num_params`` subs, or the propagator
    data lacks ``pole_vals`` / ``C_mats``).  Callers should fall back
    to ``fast_callable(subset_factor.expand())`` in that case.
    """
    import cmath as _cmath

    # ── Prefactor → complex scalar ──
    try:
        pref_c = complex(CDF(SR(prefactor_num)))
    except Exception:
        return None

    # ── Pole list (shared across edges) ──
    pole_vals = propagator_data.get('pole_vals')
    C_mats = propagator_data.get('C_mats')
    if pole_vals is None or C_mats is None:
        return None
    try:
        poles_tuple = tuple(complex(CDF(SR(p))) for p in pole_vals)
    except Exception:
        return None
    n_poles = len(poles_tuple)

    # ── Per-edge (residues, c0, int_pairs, ext_pairs) ──
    edge_data = []
    for (ri, pi), (a_int, a_ext, c0) in zip(
        smooth_edges_ri_pi, subset_constraint_data
    ):
        try:
            residues = tuple(
                complex(CDF(SR(C_mats[k][pi, ri])))
                for k in range(n_poles)
            )
        except Exception:
            return None
        # Sparse (position, coef) pairs — retardation ``Δt`` vectors
        # have exactly 1–2 nonzero entries regardless of m_sub.
        int_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_int)
            if abs(float(a)) > 1e-15
        )
        ext_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_ext)
            if abs(float(a)) > 1e-15
        )
        edge_data.append(
            (poles_tuple, residues, float(c0), int_pairs, ext_pairs)
        )
    edge_data_t = tuple(edge_data)
    m_offset = m_sub          # index into args where external times begin
    _cexp = _cmath.exp

    def evaluator(*args):
        # args = (s_0, ..., s_{m_sub-1}, t_free_0, t_free_1, ...)
        result = pref_c
        for (poles, residues, c0, int_pairs, ext_pairs) in edge_data_t:
            dt = c0
            for (i, a) in int_pairs:
                dt += a * args[i]
            for (i, a) in ext_pairs:
                dt += a * args[m_offset + i]
            # Σ_k r_k · exp(i·p_k·dt)
            edge_val = 0.0 + 0.0j
            for (p, r) in zip(poles, residues):
                edge_val += r * _cexp(1j * p * dt)
            result *= edge_val
        return result

    return evaluator


def _build_fast_subset_evaluator_from_modes(
    prefactor_num,
    smooth_edge_modes,
    subset_constraint_data,
    m_sub,
):
    """Stage 2 variant of ``_build_fast_subset_evaluator``.

    Same per-call evaluator semantics as the legacy function, but
    consumes pre-built ``EdgeModeSum`` objects (residues + λ_α
    extracted once per edge at the top of ``integrate_diagram``)
    instead of re-extracting them from ``propagator_data`` on every
    call.  ``smooth_edge_modes`` is the subset of the diagram's
    per-edge mode sums corresponding to the current smooth-set;
    ``subset_constraint_data`` carries the Δt linear forms after
    δ-elimination.

    The two builders return numerically identical closures (same
    edge-product loop, same complex-exp arithmetic); this one just
    skips the per-call SR → complex coercion in the legacy hot path.
    """
    import cmath as _cmath

    # ── Prefactor → complex scalar ──
    try:
        pref_c = complex(CDF(SR(prefactor_num)))
    except Exception:
        return None

    # Edge-data tuples are constructed once here from the pre-built
    # mode-sum cache + per-subset Δt constraint data.  No SR → CDF
    # coercion happens — that already ran when the EdgeModeSum list
    # was built.
    edge_data = []
    for ems, (a_int, a_ext, c0) in zip(
        smooth_edge_modes, subset_constraint_data
    ):
        # Modes are already (C_α, λ_α) with λ_α = i·p_α; the legacy
        # evaluator multiplies ``1j * p`` per-call which would double
        # the imaginary factor.  Split modes back into separate
        # poles/residues tuples for the SAME inner loop shape as the
        # legacy evaluator.
        residues = tuple(C for (C, _lam) in ems.modes)
        # λ_α = i·p_α  ⇒  p_α = -i·λ_α = λ_α / 1j
        poles = tuple((_lam / 1j) for (_C, _lam) in ems.modes)
        int_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_int)
            if abs(float(a)) > 1e-15
        )
        ext_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_ext)
            if abs(float(a)) > 1e-15
        )
        edge_data.append(
            (poles, residues, float(c0), int_pairs, ext_pairs)
        )
    edge_data_t = tuple(edge_data)
    m_offset = m_sub
    _cexp = _cmath.exp

    def evaluator(*args):
        result = pref_c
        for (poles, residues, c0, int_pairs, ext_pairs) in edge_data_t:
            dt = c0
            for (i, a) in int_pairs:
                dt += a * args[i]
            for (i, a) in ext_pairs:
                dt += a * args[m_offset + i]
            edge_val = 0.0 + 0.0j
            for (p, r) in zip(poles, residues):
                edge_val += r * _cexp(1j * p * dt)
            result *= edge_val
        return result

    return evaluator


# ───────────────────────────────────────────────────────────────────────
# Polytope-integration helpers
# ───────────────────────────────────────────────────────────────────────

def _integrate_polytope(integrand_callable, s_constraints, free_ext_vals, m):
    """
    Integrate `integrand_callable(s_1, ..., s_m, *free_ext_vals)` over
    the polytope `{s : a_int · s + c_eff > 0 for all constraints}`.

    s_constraints is a list of tuples `(a_int_list_of_len_m, c_eff)`.
    """
    if m == 1:
        _RUNTIME_COUNTERS['scipy_nquad_called_m1'] += 1
    elif m == 2:
        _RUNTIME_COUNTERS['scipy_nquad_called_m2'] += 1
    elif m >= 3:
        _RUNTIME_COUNTERS['scipy_nquad_called_mge3'] += 1
    if m == 0:
        # Zero integration variables — the "integrand" is just a number.
        # Still have to check the constraints (they may be vacuous or
        # infeasible).  Θ(0) = 0 convention: boundary c_eff = 0 is OUTSIDE
        # the feasible region (the half-space is strictly open at Δt = 0).
        for (a_int, c_eff) in s_constraints:
            if c_eff <= 0:
                return 0.0 + 0.0j
        val = integrand_callable(*free_ext_vals)
        return complex(val)

    if m == 1:
        return _integrate_1d_polytope(
            integrand_callable, s_constraints, free_ext_vals
        )

    if m == 2:
        return _integrate_2d_polytope(
            integrand_callable, s_constraints, free_ext_vals
        )

    return _integrate_nd_polytope(
        integrand_callable, s_constraints, free_ext_vals, m
    )


def _make_heaviside_filtered_integrand(integrand_callable, s_constraints,
                                        free_ext_vals, m):
    r"""
    Wrap `integrand_callable` with an explicit Heaviside-product check.

    The polytope bounds we pass to `scipy.nquad` are only an approximation
    of the true polytope when some constraints couple multiple integration
    axes: cross-axis constraints must be deferred to an inner axis, and
    when the bounds function for an outer axis loses its lower or upper
    bound from such deferred constraints, we fall back to ±OUTER_CAP.
    That fallback admits regions geometrically OUTSIDE the true polytope.

    Physically the retarded propagator `G^R(Δt) = Θ(Δt) · G^sm(Δt)`
    vanishes on those regions via the Heaviside.  But our JIT-compiled
    integrand contains ONLY `G^sm`, never `Θ` — it relies entirely on the
    polytope bounds for retardation.  When the bounds overshoot, the
    integrand evaluates `G^sm` on a region where it should be zero, and
    for retarded poles (Im(ω) > 0) `G^sm(Δt) = C · exp(-γ Δt)` GROWS for
    `Δt < 0`, producing a spurious positive contribution.

    This wrapper explicitly multiplies by the Heaviside product, i.e.
    returns 0.0 whenever any `a_int · s + c_eff < 0`.  The polytope
    bounds then serve only as an optimization: they tighten the
    quadrature domain for speed, but correctness no longer depends on
    them being exact.

    Parameters
    ----------
    integrand_callable : callable
        The JIT-compiled smooth integrand `f(s_0, ..., s_{m-1},
        *free_ext_vals)`.
    s_constraints : list of (a_int, c_eff)
        Polytope constraints `a_int · s + c_eff > 0` for each retarded
        edge still active at this subset (strict inequality per Θ(0) = 0
        convention — the boundary Δt = 0 is OUTSIDE the feasible region).
        `c_eff` already has the current external-time values substituted
        in (via the caller).
    free_ext_vals : list
        Free external time values, passed through to `integrand_callable`.
    m : int
        Number of integration axes (`s_0, ..., s_{m-1}`).

    Returns
    -------
    callable f(*s_vals) → complex, with the Heaviside filter applied.
    """
    # Convention: Θ(0) = 0.  A constraint `a_int · s + c_eff > 0` is
    # STRICTLY required: the boundary `Δt = 0` is excluded.  Use `dt <= 0`
    # to kill both the exterior (strictly infeasible) AND the boundary.
    #
    # Fix D (2026-04-21): pre-extract constraints in SPARSE form so the
    # hot loop skips zero-coefficient axes entirely.  For retarded
    # propagators each constraint is `t_v − t_u > 0` which has exactly
    # two nonzero entries in `a_int` (regardless of `m`), so this
    # collapses an inner `for j in range(m)` loop into 2 iterations.
    #
    # A pre-check here handles constraints that are purely trivial
    # (all `a` zero): if `c_eff > 0` the constraint is always satisfied
    # and we drop it; if `c_eff <= 0` the polytope is empty and the
    # filter always returns 0.
    sparse = []
    always_empty = False
    for (a_int, c_eff) in s_constraints:
        c_eff_f = float(c_eff)
        pairs = tuple((j, float(a)) for j, a in enumerate(a_int)
                      if abs(float(a)) > 1e-15)
        if not pairs:
            # Pure constant constraint.  Θ(0) = 0: strict c_eff > 0.
            if c_eff_f <= 0.0:
                always_empty = True
                break
            # Trivially satisfied — drop.
            continue
        sparse.append((c_eff_f, pairs))
    # Tuple-of-tuples for slightly faster iteration than list-of-tuples
    # (CPython's FOR_ITER has a specialized path for tuples).
    sparse_constraints = tuple(sparse)
    free_ext_tuple = tuple(free_ext_vals)

    if always_empty:
        def filtered_empty(*s_vals):
            return 0.0 + 0.0j
        return filtered_empty

    # Capture `free_ext_vals` as a LIST (not tuple) in the closure so
    # the argument-packing style matches the pre-Fix D code exactly —
    # `integrand_callable(*args)` with `args = list(s_vals) +
    # free_ext_list`.  Measured: `integrand_callable(*s_vals,
    # *free_ext_tuple)` (Python 3.5+ multi-unpack) triggers a slow
    # path inside Sage's fast_callable on the 1-loop m=3 workload
    # (~2× wall-clock regression vs. baseline) for reasons not yet
    # root-caused.  Empirically the single-unpack form matches
    # baseline performance while still benefiting from the sparse-
    # scan speedup below.
    free_ext_list = list(free_ext_vals)

    def filtered(*s_vals):
        # Heaviside check: every constraint must be > 0 (Θ(0) = 0).
        for c_eff, nzs in sparse_constraints:
            dt = c_eff
            for (j, a) in nzs:
                dt += a * s_vals[j]
            if dt <= 0.0:
                return 0.0 + 0.0j
        return complex(integrand_callable(*(list(s_vals) + free_ext_list)))

    return filtered


def _complex_quad(integrand_callable, s_slot_index, other_args, lower, upper):
    """
    1D quadrature of `integrand_callable` along `s_slot_index`, with
    the other arguments fixed to `other_args`, over `[lower, upper]`.

    Real and imaginary parts are integrated separately via
    `scipy.integrate.quad`, which handles ±inf bounds natively.
    """
    from scipy.integrate import quad

    def _eval(s_val):
        args = list(other_args)
        args.insert(s_slot_index, float(s_val))
        val = integrand_callable(*args)
        return complex(val)

    def f_re(s_val):
        return _eval(s_val).real

    def f_im(s_val):
        return _eval(s_val).imag

    re_val, _ = quad(f_re, lower, upper, **QUAD_OPTS)
    try:
        im_val, _ = quad(f_im, lower, upper, **QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _resolve_1d_bounds(s_constraints, s_index):
    """
    For the given integration-variable index `s_index`, intersect all
    half-line constraints `a_i s_{s_index} + (other terms already
    substituted) + c_eff > 0` into a single interval `[L, U]`.

    Each element of `s_constraints` is `(a_int_list, c_eff_scalar)`
    where `a_int_list` has length equal to the number of integration
    variables. For this 1D-on-one-axis pass we assume the other axes'
    coefficients are zero (i.e., the caller has already substituted
    them).

    Returns (L, U). If the intersection is infeasible we return a
    DEGENERATE empty interval `(0.0, 0.0)` rather than a flipped
    (inf, -inf) sentinel. This matters because `scipy.quad(f, 0, 0)`
    returns 0 correctly, while `scipy.quad(f, +inf, -inf)` returns
    `-quad(f, -inf, +inf)` — the full real-line integral with a
    sign flip, which silently poisons any outer quadrature
    (`scipy.nquad`) that feeds this bounds function into
    `scipy.quad`. The 1D code path catches infeasible ranges
    up-front via `if L >= U: return 0`, so this degenerate form is
    indistinguishable from the old sentinel for the 1D path; but
    the 2D path needs the degenerate-empty form so that the inner
    integral returns 0 where the projection is empty.
    """
    L, U = -math.inf, math.inf
    infeasible = False
    for (a_int, c_eff) in s_constraints:
        a = a_int[s_index]
        if abs(a) < 1e-15:
            # Degenerate constraint (no dependence on s_index).  Under
            # Θ(0) = 0, the inequality is strict: `c_eff > 0` required.
            # Boundary c_eff = 0 is infeasible.
            if c_eff <= 0:
                infeasible = True
                break
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if infeasible or L >= U:
        return 0.0, 0.0   # degenerate empty interval
    return L, U


def _integrate_1d_polytope(integrand_callable, s_constraints, free_ext_vals):
    """Single integration variable. The polytope is an interval.

    A single axis is always cleanly bounded by the polytope (no deferred
    constraints possible since there's no inner axis to defer to), so
    the bounds (L, U) returned by ``_resolve_1d_bounds`` are exact.
    When ``DEBUG_HEAVISIDE_GUARD`` is False (production default) we
    skip the Heaviside-filter wrapper entirely — it's redundant given
    exact bounds and adds ~few µs per integrand call.  Flip the flag
    to True if you want belt-and-braces validation.
    """
    L, U = _resolve_1d_bounds(s_constraints, s_index=0)
    if L >= U:
        return 0.0 + 0.0j

    if DEBUG_HEAVISIDE_GUARD:
        # Wrapped path: filter every integrand call against the
        # retarded constraints.  No-op on the (L, U) interior but
        # catches any drift if the bound resolver is wrong.
        filt = _make_heaviside_filtered_integrand(
            integrand_callable, s_constraints, free_ext_vals, m=1,
        )

        def f_re(s_0):
            return filt(s_0).real

        def f_im(s_0):
            return filt(s_0).imag
    else:
        # Fast path: bounds are exact so the filter is redundant.
        free_ext_list = list(free_ext_vals)

        def f_re(s_0):
            return complex(integrand_callable(s_0, *free_ext_list)).real

        def f_im(s_0):
            return complex(integrand_callable(s_0, *free_ext_list)).imag

    from scipy.integrate import quad
    re_val, _ = quad(f_re, L, U, **QUAD_OPTS)
    try:
        im_val, _ = quad(f_im, L, U, **QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _integrate_2d_polytope(integrand_callable, s_constraints, free_ext_vals):
    """
    Two integration variables s_0, s_1.

    Use `scipy.integrate.nquad` with `s_0` as the innermost integral
    and `s_1` as the outermost. The `s_0` bounds depend on s_1; the
    `s_1` bounds are extracted from constraints with zero coefficient
    on `s_0` (if any), defaulting to `(-inf, +inf)` if there are no
    pure-`s_1` constraints.
    """
    from scipy.integrate import nquad

    # ── Pre-split 2D constraints by role w.r.t. s_0 (Fix D) ──────
    # For each constraint `a_0 s_0 + a_1 s_1 + c_eff > 0` we
    # classify:
    #   pure_0      : a_1 ≈ 0, a_0 ≠ 0 → precomputed (L, U) on s_0.
    #   mixed       : both a_0 ≠ 0 and a_1 ≠ 0 → resolve per s_1 call.
    #   s1_only     : a_0 ≈ 0, a_1 ≠ 0 → pure residual inequality in s_1.
    #   constant    : a_0 ≈ 0, a_1 ≈ 0 → constant check (handled at build).
    pure_0_L = -math.inf
    pure_0_U = math.inf
    mixed_s0 = []       # list of (a_0, c_eff, a_1)
    s1_residual = []    # list of (a_1, c_eff): constraint reduces to a_1 s_1 + c_eff > 0
    bounds_s0_always_empty = False
    for (a_int, c_eff) in s_constraints:
        a_0 = float(a_int[0])
        a_1 = float(a_int[1])
        c_f = float(c_eff)
        if abs(a_0) < 1e-15 and abs(a_1) < 1e-15:
            # Constant constraint; Θ(0) = 0 wants c_eff > 0.
            if c_f <= 0.0:
                bounds_s0_always_empty = True
                break
            # Trivially satisfied — drop.
            continue
        if abs(a_0) < 1e-15:
            s1_residual.append((a_1, c_f))
            continue
        if abs(a_1) < 1e-15:
            bound = -c_f / a_0
            if a_0 > 0.0:
                if bound > pure_0_L:
                    pure_0_L = bound
            else:
                if bound < pure_0_U:
                    pure_0_U = bound
        else:
            mixed_s0.append((a_0, c_f, a_1))

    mixed_s0_t = tuple(mixed_s0)
    s1_residual_t = tuple(s1_residual)

    if bounds_s0_always_empty or pure_0_L >= pure_0_U:
        # Entire polytope empty.
        def bounds_s0(s_1_val):
            return 0.0, 0.0
    else:
        def bounds_s0(s_1_val):
            # Check s_1-only residual constraints (pure a_1 s_1 + c > 0).
            for (a_1, c_f) in s1_residual_t:
                if a_1 * s_1_val + c_f <= 0.0:
                    return 0.0, 0.0
            L = pure_0_L
            U = pure_0_U
            for (a_0, c_f, a_1) in mixed_s0_t:
                bound = -(c_f + a_1 * s_1_val) / a_0
                if a_0 > 0.0:
                    if bound > L:
                        L = bound
                else:
                    if bound < U:
                        U = bound
            if L >= U:
                return 0.0, 0.0
            return L, U

    # Bounds on s_1: use constraints where a_0 = 0 AND a_1 != 0
    # (genuinely pure-s_1); else fall back to the ±OUTER_CAP
    # Heaviside-filtered quadrature domain.
    #
    # Subtle: a constraint with BOTH a_int[0] ≈ 0 and a_int[1] ≈ 0 is
    # NOT a bound on s_1 at all — it's a pure-external inequality that
    # either kills the polytope (c_eff ≤ 0 under Θ(0) = 0) or is
    # trivially satisfied (c_eff > 0).  Such constraints must NOT set
    # the `pure_s1_found` flag, otherwise we skip the OUTER_CAP fallback
    # and scipy.nquad runs on an unbounded axis, oversampling the
    # infinite-domain transform grid and biasing the integral.  At k=4
    # with distinct fields, δ-sifting can pin an integration variable
    # to external times and leave residual pure-external constraints
    # that trigger this path — the overshoot source for ~12% of
    # theory-vs-sim at k=4.
    L1, U1 = math.inf, -math.inf
    pure_s1_found = False
    tmp_L, tmp_U = -math.inf, math.inf
    for (a_int, c_eff) in s_constraints:
        if abs(a_int[0]) < 1e-15:
            a = a_int[1]
            if abs(a) < 1e-15:
                # Pure-external constraint: NOT a bound on s_1.
                # Θ(0) = 0: strict c_eff > 0 required; boundary infeasible.
                if c_eff <= 0:
                    return 0.0 + 0.0j
                continue
            # Genuine pure-s_1 constraint.
            pure_s1_found = True
            bound = -c_eff / a
            if a > 0 and bound > tmp_L:
                tmp_L = bound
            elif a < 0 and bound < tmp_U:
                tmp_U = bound
    if pure_s1_found:
        L1, U1 = tmp_L, tmp_U
        if L1 >= U1:
            return 0.0 + 0.0j
    else:
        # Fallback cap: ±200 is ample with Heaviside-filtered integrand
        # (see _make_heaviside_filtered_integrand — correctness doesn't
        # depend on the cap being tight, only large enough to contain
        # the decaying tail).
        L1, U1 = -200.0, 200.0

    # Heaviside-filtered integrand: correctness no longer depends on
    # the (L1, U1) cap being exactly the true polytope projection.
    # Skip the filter when bounds are EXACT (pure_s1_found) — the
    # inner ``bounds_s0`` callable is exact too, so the integration
    # domain matches the polytope precisely.  Cap-fallback path
    # always needs the filter regardless of DEBUG_HEAVISIDE_GUARD.
    needs_filter = (not pure_s1_found) or DEBUG_HEAVISIDE_GUARD
    if needs_filter:
        filt = _make_heaviside_filtered_integrand(
            integrand_callable, s_constraints, free_ext_vals, m=2,
        )

        def f_re(s_0, s_1):
            return filt(s_0, s_1).real

        def f_im(s_0, s_1):
            return filt(s_0, s_1).imag
    else:
        free_ext_list = list(free_ext_vals)

        def f_re(s_0, s_1):
            return complex(
                integrand_callable(s_0, s_1, *free_ext_list)
            ).real

        def f_im(s_0, s_1):
            return complex(
                integrand_callable(s_0, s_1, *free_ext_list)
            ).imag

    re_val, _ = nquad(f_re, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
    try:
        im_val, _ = nquad(f_im, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _integrate_nd_polytope(integrand_callable, s_constraints, free_ext_vals, m):
    """
    General `m >= 3` case via `scipy.integrate.nquad` with nested
    bound functions.

    Variable ordering for nquad: the FIRST argument of the integrand
    is the INNERMOST integration variable.  We integrate s_0 first
    (innermost), then s_1, ..., up to s_{m-1} (outermost).

    For each variable s_k, its bounds are computed by:
      1. Substituting all OUTER variables (s_{k+1}, ..., s_{m-1}) and
         the external times into the linear constraints
         `a · s + c_eff > 0`.
      2. Resolving the remaining 1D polytope on s_k via
         `_resolve_1d_bounds`.
    """
    from scipy.integrate import nquad

    def _make_bound_fn(k_var):
        """Return a function `bounds_k(*outer_vals)` that returns (L, U)
        for variable s_{k_var} given all the OUTER variable values.

        nquad calls bound functions with the outer variables in
        REVERSE order (s_{k+1}, s_{k+2}, ..., s_{m-1}).

        Important subtlety — constraints that still couple to a MORE-
        INNER axis s_j (j < k_var) must be SKIPPED here.  Those
        constraints are genuine bounds on the inner variable, not on
        s_{k_var}, and they will be resolved when nquad nests into the
        deeper (smaller-index) integration axis and s_j becomes the
        resolution target.  Failing to filter them is a latent bug:
        `_resolve_1d_bounds` inspects only `a_int[s_index]` and treats
        `|a_int[k_var]| < 1e-15` as a pure-residual check, which then
        spuriously declares the polytope infeasible whenever the
        accumulated residual (still containing the unresolved inner
        coefficient) is negative.  This mirrors the filter
        `abs(a_int[0]) < 1e-15` that `_integrate_2d_polytope` already
        applies to the outer-axis bound (lines ~1240–1255).

        Regression: test_phase_J_nd_polytope_preserves_deferred_constraints
        in tests/test_time_domain.py.

        Fix D (2026-04-21): constraint classification and sparse-
        coefficient extraction happen ONCE at closure-build time, not
        on every call.  Constraints with only `s_{k_var}` nonzero
        (no outer coupling) contribute a fixed slice to (L, U) that we
        precompute.  The per-call path now only iterates constraints
        whose bound actually varies with the outer values.
        """
        # ── Classify constraints by their role w.r.t. axis k_var ──
        # Four buckets (after filtering deferred-inner constraints):
        #   pure_k      : a[k_var] != 0, no outer coupling → precomputed
        #                 contribution to (L, U).
        #   mixed       : a[k_var] != 0, some outer axes coupled →
        #                 resolve per call using outer vals.
        #   outer_only  : a[k_var] == 0, some outer axes coupled →
        #                 pure residual inequality; kills polytope when
        #                 negative at this outer point.
        #   (trivial satisfied or trivially infeasible constraints
        #    are resolved at build time.)
        pure_k_L = -math.inf
        pure_k_U = math.inf
        mixed = []         # list of (a_k, c_eff, outer_pairs)
        outer_only = []    # list of (c_eff, outer_pairs)
        infeasible_at_build = False
        for (a_int, c_eff) in s_constraints:
            # Skip constraints that still couple to a more-inner axis.
            deferred = False
            for j in range(k_var):
                if abs(a_int[j]) >= 1e-15:
                    deferred = True
                    break
            if deferred:
                continue
            a_k = float(a_int[k_var])
            # Sparse outer-axis coefficients: tuple of (outer_index, coeff)
            # where outer_index is the position in *outer_vals (not the
            # absolute axis index).  nquad passes outer_vals with
            # outer_vals[0] = s_{k_var+1}, [1] = s_{k_var+2}, etc.,
            # which matches the original `j = k_var + 1 + i_outer`
            # indexing.
            outer_pairs = tuple(
                (j - (k_var + 1), float(a_int[j]))
                for j in range(k_var + 1, len(a_int))
                if abs(float(a_int[j])) >= 1e-15
            )
            c_eff_f = float(c_eff)
            if abs(a_k) < 1e-15:
                if not outer_pairs:
                    # Constant constraint: Θ(0)=0 wants c_eff > 0.
                    if c_eff_f <= 0.0:
                        infeasible_at_build = True
                        break
                    # Trivially satisfied → drop.
                    continue
                outer_only.append((c_eff_f, outer_pairs))
            else:
                if outer_pairs:
                    mixed.append((a_k, c_eff_f, outer_pairs))
                else:
                    bound = -c_eff_f / a_k
                    if a_k > 0.0:
                        if bound > pure_k_L:
                            pure_k_L = bound
                    else:
                        if bound < pure_k_U:
                            pure_k_U = bound

        # If the polytope is empty under Θ(0)=0 irrespective of outer
        # values, or the pure-k_var bounds are crossed, return a
        # constant-zero bounds function — scipy.quad(a, a) is 0, so
        # this cleanly kills the outer integral over the empty region.
        if infeasible_at_build or pure_k_L >= pure_k_U:
            def _bounds_infeas(*outer_vals):
                return 0.0, 0.0
            return _bounds_infeas

        mixed_t = tuple(mixed)
        outer_only_t = tuple(outer_only)

        def bounds_k(*outer_vals):
            # Start from the precomputed pure-axis bounds (may be ±inf
            # if no pure-k constraint is present on that side — mixed
            # constraints below may finitely bound the axis instead).
            L = pure_k_L
            U = pure_k_U
            # Check outer-only constraints first (cheap polytope kill).
            for c_eff, outer_pairs in outer_only_t:
                val = c_eff
                for (oi, a) in outer_pairs:
                    val += a * outer_vals[oi]
                if val <= 0.0:
                    return 0.0, 0.0
            # Apply mixed (outer-coupled on-axis) constraints.
            for a_k, c_eff, outer_pairs in mixed_t:
                total_c = c_eff
                for (oi, a) in outer_pairs:
                    total_c += a * outer_vals[oi]
                bound = -total_c / a_k
                if a_k > 0.0:
                    if bound > L:
                        L = bound
                else:
                    if bound < U:
                        U = bound
            if L >= U:
                return 0.0, 0.0
            # Cap infinite bounds only when still unbounded.  Matches
            # the original post-`_resolve_1d_bounds` behaviour where a
            # cap never clips a finite constraint-derived bound.
            if math.isinf(L):
                L = -OUTER_CAP
            if math.isinf(U):
                U = OUTER_CAP
            return L, U
        return bounds_k

    # Fallback cap for any axis whose bounds resolve to (-inf, +inf).
    # With the Heaviside-filtered integrand (below), correctness does
    # NOT depend on this cap being tight — the filter kills any
    # contribution from the region outside the true polytope.  The cap
    # only needs to be large enough that the decaying integrand has
    # effectively vanished by the boundary.  For retarded propagators
    # with time constant τ, ~10τ gives exp(-10) ≈ 5e-5 tail.  Hawkes
    # τ=10 ⟹ ±200 is ample.  A wider cap is harmless (the filter
    # zeros out the extra volume) but slightly slower to quadrature
    # through.
    #
    # 2026-05-17: when USE_POSET_CAP_MATCH_SCIPY is enabled (default),
    # we tie OUTER_CAP to POSET_PHYSICAL_MARGIN so the m≥3 scipy
    # fallback integrates over the same domain as the analytic poset
    # path's unbounded-below fallback.  For marginal-stability theories
    # (Re β ≈ 0) the integral is genuinely cap-dependent and the two
    # paths must agree on the cap or the grouped/per-diag comparison
    # disagrees.  For strongly-decaying theories the integrand is
    # negligible at |s| > 50 anyway, so this is harmless.
    if USE_POSET_CAP_MATCH_SCIPY:
        OUTER_CAP = POSET_PHYSICAL_MARGIN
    else:
        OUTER_CAP = 200.0

    # Outermost variable s_{m-1}: bounds computed from constraints with
    # zero coefficient on all inner variables (pure-s_{m-1}).  If no
    # such constraints exist, fall back to ±OUTER_CAP.
    L_out, U_out = _outer_bounds(s_constraints, m - 1)
    if L_out >= U_out:
        return 0.0 + 0.0j
    if not math.isfinite(L_out):
        L_out = -OUTER_CAP
    if not math.isfinite(U_out):
        U_out = OUTER_CAP

    # Build the list of bound specifications for nquad.
    # Order: innermost first.  For variables s_0, ..., s_{m-2}, use
    # callable bounds (functions of outer vars).  For s_{m-1}, use the
    # constant tuple (L_out, U_out).
    bound_specs = [_make_bound_fn(k) for k in range(m - 1)]
    bound_specs.append((L_out, U_out))

    # Heaviside-filtered integrand: evaluates to 0 outside the true
    # polytope, guarding against any cap overshoot or deferred-
    # constraint leak.
    filt = _make_heaviside_filtered_integrand(
        integrand_callable, s_constraints, free_ext_vals, m=m,
    )

    def f_re(*all_args):
        # nquad passes integration variables (s_0, ..., s_{m-1}).
        return filt(*all_args).real

    def f_im(*all_args):
        return filt(*all_args).imag

    re_val, _ = nquad(f_re, bound_specs, opts=QUAD_OPTS)
    try:
        im_val, _ = nquad(f_im, bound_specs, opts=QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _outer_bounds(s_constraints, k_var):
    """Compute bounds on s_{k_var} from constraints whose coefficients on
    all OTHER integration variables vanish (pure-s_{k_var} constraints).

    Used for the outermost integration variable in nD polytope
    integration, where there are no further outer variables to
    substitute.  If no pure constraint exists, returns (-inf, +inf).
    """
    L, U = -math.inf, math.inf
    pure_found = False
    for (a_int, c_eff) in s_constraints:
        # Check that a_int is purely k_var-dependent
        is_pure = True
        for j in range(len(a_int)):
            if j == k_var:
                continue
            if abs(a_int[j]) >= 1e-15:
                is_pure = False
                break
        if not is_pure:
            continue
        pure_found = True
        a = a_int[k_var]
        if abs(a) < 1e-15:
            # Degenerate pure-k_var constraint.  Θ(0) = 0: boundary
            # c_eff = 0 is infeasible; strict c_eff > 0 required.
            if c_eff <= 0:
                return math.inf, -math.inf  # infeasible
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if pure_found and L >= U:
        return math.inf, -math.inf
    if not pure_found:
        # No pure constraints — let inner bounds clip via callables.
        return -math.inf, math.inf
    return L, U


# ───────────────────────────────────────────────────────────────────────
# Edge → propagator-index lookup (matches _resolve_edge_propagator_data)
# ───────────────────────────────────────────────────────────────────────

def format_td_integral_latex(
    tree_result,
    typed_diagram=None,
    combined_prefactor=None,
    ext_time_vars=None,
    label=None,
):
    r"""
    Build a LaTeX string describing the time-domain integral that
    Phase J evaluates for a given tree diagram, in the same spirit as
    the notebook's `show_integral` helper for the frequency-domain
    Phase I integrand.

    This helper is intended for debugging and documentation only —
    numerical evaluation goes through `tree_result['contribution']`.

    Parameters
    ----------
    tree_result : dict
        Output of `integrate_tree_diagram`. Must contain the
        `edge_info`, `integration_vars`, and `constraints` keys.
    typed_diagram : TypedDiagram, optional
        Used for the leaf/internal-vertex display. If not supplied the
        vertex-time assignment is inferred from `tree_result`.
    combined_prefactor : SR or numeric, optional
        Prefactor to display in front of the integral. If None, the
        prefactor is folded into the integrand and not displayed
        separately.
    ext_time_vars : list of SR, optional
        External time symbols, used for display only. If not provided,
        default labels `t_1, t_2, ...` are inferred.
    label : str, optional
        A short label for the diagram (e.g., 'Tree-1', 'Hawkes-k2').
        Rendered as a prefix in the output.

    Returns
    -------
    str
        A LaTeX string, ready to pass to `display(Math(...))` in a
        Jupyter notebook. Includes:
          - vertex-time assignment block
          - the integral expression
            `C_Γ = pref · ∫ ds_1 ... ds_m  ∏ G_R[…](…)`
          - the retarded polytope constraints `Θ(t_v − t_u)`
          - any δ-edge summary (edges with a nonzero δ coefficient)
    """
    from sage.all import latex

    edge_info = tree_result.get('edge_info', [])
    integration_vars = tree_result.get('integration_vars', [])
    constraints = tree_result.get('constraints', [])

    lines = []
    if label:
        lines.append(r'\text{' + str(label) + r'} \;:')

    # Integration variables
    if integration_vars:
        int_vars_tex = r' \wedge '.join(latex(v) for v in integration_vars)
        lines.append(
            r'\text{non-leaf vertex times: } \{' + int_vars_tex + r'\}'
        )
        integrals_tex = ''
        for v in integration_vars:
            integrals_tex += r'\int\!d' + latex(v) + r'\;'
    else:
        integrals_tex = ''

    # Combined prefactor.  For diagrams with a non-local cumulant
    # kernel (NoiseSourceType vertex), the substituted, simplified
    # τ-dependent prefactor lives in ``tree_result['cumulant_prefactor']``
    # and goes INSIDE the integral as a kernel factor.  For ordinary
    # diagrams the kappa machinery is absent and the user-supplied
    # ``combined_prefactor`` (τ-independent scalar) goes OUTSIDE.
    pref_tex = ''
    inside_kernel_tex = ''
    cumulant_pref = tree_result.get('cumulant_prefactor', None)
    has_cumulant = tree_result.get('has_cumulant_kernel', False)
    if has_cumulant and cumulant_pref is not None:
        # The cumulant prefactor is τ-dependent — render it inside the
        # integral, parenthesised, before the propagator product.
        inside_kernel_tex = (
            r'\bigl[' + latex(SR(cumulant_pref)) + r'\bigr] \cdot '
        )
    elif combined_prefactor is not None:
        pref_tex = latex(combined_prefactor) + r'\;'

    # Build the edge-factor product
    factor_bits = []
    delta_summary_bits = []
    for ei in edge_info:
        u = ei['u']
        v = ei['v']
        ri = ei['ri']
        pi = ei['pi']
        dt = ei['dt_sym']
        delta_c = ei.get('delta_coeff', 0)
        # G_R[pi, ri](dt)
        factor_bits.append(
            r'G^{R}_{' + str(pi) + ',' + str(ri) + r'}\!\bigl('
            + latex(dt) + r'\bigr)'
        )
        try:
            if abs(complex(delta_c)) > 1e-15:
                delta_summary_bits.append(
                    r'c_{' + str(pi) + ',' + str(ri) + r'} = '
                    + latex(delta_c)
                )
        except Exception:
            pass

    product_tex = r' \cdot '.join(factor_bits) if factor_bits else r'1'

    # Main equation: C_Γ = pref × ∫ds × [κ(τ)] × ∏ G_R × Θ
    lines.append(
        r'C_{\Gamma}(t) \;=\; '
        + pref_tex
        + integrals_tex
        + inside_kernel_tex
        + product_tex
    )

    # Retardation constraints (as Θ factors)
    if constraints:
        theta_bits = [
            r'\Theta\bigl(' + latex(SR(c)) + r'\bigr)'
            for c in constraints
        ]
        lines.append(
            r'\text{retardation: } '
            + r' \cdot '.join(theta_bits)
        )

    # δ-edge summary, if any
    if delta_summary_bits:
        lines.append(
            r'\text{instantaneous } \delta\text{-edge coefficients: } '
            + r', \; '.join(delta_summary_bits)
        )

    n_subsets = tree_result.get('n_subsets_evaluated')
    n_skipped = tree_result.get('n_shotnoise_skipped', 0)
    if n_subsets is not None:
        lines.append(
            r'\text{δ-edge subsets evaluated: } '
            + str(n_subsets)
            + (
                r'\;\;(\text{shot-noise skipped: } '
                + str(n_skipped) + r')'
                if n_skipped else ''
            )
        )

    return r' \\ '.join(lines)


def _lookup_prop_indices(typed_diagram, edge_key):
    """
    Look up (resp_row, phys_col) propagator indices for a prediagram
    edge, using the same fallback order as
    `_resolve_edge_propagator_data` in `msrjd/integration/symbolic.py`.
    """
    u, v, lbl = edge_key
    prop_indices = typed_diagram.propagator_indices
    td_edge_keys = list(typed_diagram.edge_types.keys())

    # 1. Exact match on (u, v, lbl)
    for td_ek in td_edge_keys:
        if td_ek == (u, v, lbl):
            return prop_indices[td_ek]

    # 2. Match on (u, v, None)
    for td_ek in td_edge_keys:
        if (td_ek[0], td_ek[1]) == (u, v) and (
            len(td_ek) < 3 or td_ek[2] is None
        ):
            return prop_indices[td_ek]

    # 3. First edge matching (u, v) regardless of label
    for td_ek in td_edge_keys:
        if (td_ek[0], td_ek[1]) == (u, v):
            return prop_indices[td_ek]

    raise KeyError(
        f"No propagator indices found for edge ({u}, {v}, {lbl}) in "
        f"typed diagram."
    )


# ───────────────────────────────────────────────────────────────────────
# Backward-compatibility alias
# ───────────────────────────────────────────────────────────────────────
# ``integrate_tree_diagram`` was the original name for what is now
# ``integrate_diagram`` (generalised to any loop order).  Keep the old
# name resolvable so tests and external callers don't break.
integrate_tree_diagram = integrate_diagram

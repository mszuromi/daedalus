"""
tests/test_causal_poset.py
==========================
Stage 3b-prep tests: partial-order extraction from retardation
constraints + linear-extension enumeration.  Each test isolates one
piece so failures are easy to localise before the closed-form
integrator (Stage 3b-nested) is built on top.

Run with::

    sage -python -m pytest tests/test_causal_poset.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.time_domain.final_integral import (
    _CausalPoset,
    _extract_causal_poset,
    _enumerate_linear_extensions,
    _causal_poset_consistent_scalar_lower,
    _causal_poset_consistent_scalar_upper,
    _exp_over_chain_simplex,
    _integrate_nd_polytope_poset_modesum,
    EdgeModeSum,
)


# ───────────────────────────────────────────────────────────────────
# _extract_causal_poset
# ───────────────────────────────────────────────────────────────────

def test_extract_simple_chain_3_vars():
    """Constraints  s_1 > s_0,  s_2 > s_1  → edges (0,1), (1,2)."""
    constraints = [
        ([-1.0, 1.0, 0.0], [], 0.0),   # s_1 > s_0
        ([0.0, -1.0, 1.0], [], 0.0),   # s_2 > s_1
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=3)
    assert poset is not None
    assert poset.m == 3
    assert poset.edges == ((0, 1), (1, 2))
    assert poset.scalar_lowers == ()
    assert poset.scalar_uppers == ()


def test_extract_with_scalar_lower_bound():
    """Constraint  s_0 > 5.0  encoded as +1·s_0 + (−5.0) > 0."""
    constraints = [
        ([1.0, 0.0], [], -5.0),
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is not None
    assert poset.edges == ()
    assert poset.scalar_lowers == ((0, 5.0),)
    assert poset.scalar_uppers == ()


def test_extract_with_scalar_upper_bound():
    """Constraint  s_1 < 7.0  encoded as −1·s_1 + 7.0 > 0."""
    constraints = [
        ([0.0, -1.0], [], 7.0),
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is not None
    assert poset.scalar_uppers == ((1, 7.0),)


def test_extract_with_external_time_in_lower():
    """Constraint  s_0 > t_ext  with a_ext = [+1], c0 = 0.

    For t_ext = 2.5, lower is 2.5.
    """
    constraints = [
        ([1.0, 0.0], [-1.0], 0.0),  # s_0 - t_ext > 0
    ]
    poset = _extract_causal_poset(
        constraints, free_ext_vals=[2.5], m=2,
    )
    assert poset is not None
    assert len(poset.scalar_lowers) == 1
    var, val = poset.scalar_lowers[0]
    assert var == 0
    assert abs(val - 2.5) < 1e-12


def test_extract_rejects_mixed_constraint():
    """Constraint with three nonzero coefficients (not a clean ±1
    pair) → mixed → returns None."""
    constraints = [
        ([1.0, 1.0, -1.0], [], 0.0),  # not a pure ordering
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=3)
    assert poset is None


def test_extract_rejects_shifted_inter_axis():
    """Inter-axis constraint with c_eff ≠ 0  (e.g.  s_v > s_u + c
    for c ≠ 0) → not supported in the chain form → None."""
    constraints = [
        ([-1.0, 1.0], [], 3.0),  # s_1 > s_0 + 3 — not a pure poset edge
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is None


def test_extract_dedup_repeated_edges():
    """Two redundant copies of the same edge are deduplicated."""
    constraints = [
        ([-1.0, 1.0], [], 0.0),
        ([-1.0, 1.0], [], 0.0),
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is not None
    assert poset.edges == ((0, 1),)


# ───────────────────────────────────────────────────────────────────
# _enumerate_linear_extensions
# ───────────────────────────────────────────────────────────────────

def test_linear_extensions_empty_poset():
    """No edges, m=3 → all 3! = 6 permutations are extensions."""
    poset = _CausalPoset(m=3, edges=(), scalar_lowers=(),
                         scalar_uppers=())
    exts = list(_enumerate_linear_extensions(poset))
    assert len(exts) == 6
    # Should include every permutation of (0, 1, 2)
    import itertools
    expected = set(itertools.permutations(range(3)))
    assert set(exts) == expected


def test_linear_extensions_chain_3_vars():
    """Chain  0 → 1 → 2  has exactly ONE linear extension."""
    poset = _CausalPoset(m=3, edges=((0, 1), (1, 2)),
                         scalar_lowers=(), scalar_uppers=())
    exts = list(_enumerate_linear_extensions(poset))
    assert exts == [(0, 1, 2)]


def test_linear_extensions_y_poset():
    """Y-shaped DAG: 0 → 2, 1 → 2 (two sources, one sink).
    Linear extensions: (0, 1, 2) and (1, 0, 2)."""
    poset = _CausalPoset(m=3, edges=((0, 2), (1, 2)),
                         scalar_lowers=(), scalar_uppers=())
    exts = list(_enumerate_linear_extensions(poset))
    assert sorted(exts) == [(0, 1, 2), (1, 0, 2)]


def test_linear_extensions_diamond():
    """Diamond:  0 → 1, 0 → 2, 1 → 3, 2 → 3.
    Linear extensions: (0, 1, 2, 3) and (0, 2, 1, 3)."""
    poset = _CausalPoset(
        m=4, edges=((0, 1), (0, 2), (1, 3), (2, 3)),
        scalar_lowers=(), scalar_uppers=(),
    )
    exts = list(_enumerate_linear_extensions(poset))
    assert sorted(exts) == [(0, 1, 2, 3), (0, 2, 1, 3)]


def test_linear_extensions_count_matches_known_formula():
    """For ``m`` totally unrelated vars, the number of extensions
    equals m! — sanity check for m=4, 5."""
    import math
    for m in (4, 5):
        poset = _CausalPoset(m=m, edges=(), scalar_lowers=(),
                              scalar_uppers=())
        exts = list(_enumerate_linear_extensions(poset))
        assert len(exts) == math.factorial(m), \
            f'm={m}: got {len(exts)} extensions, expected {math.factorial(m)}'


# ───────────────────────────────────────────────────────────────────
# _causal_poset_consistent_scalar_lower / upper
# ───────────────────────────────────────────────────────────────────

def test_consistent_scalar_lower_all_equal():
    """All variables share L = 1.0 → returns (1.0, True)."""
    poset = _CausalPoset(
        m=3, edges=(), scalar_lowers=((0, 1.0), (1, 1.0), (2, 1.0)),
        scalar_uppers=(),
    )
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is True
    assert abs(L - 1.0) < 1e-12


def test_consistent_scalar_lower_missing_var():
    """Some variables have a scalar lower, others don't.  As long
    as the set ones agree, the missing ones can inherit L via the
    chain ordering — accept and return that common L.
    """
    poset = _CausalPoset(
        m=3, edges=(), scalar_lowers=((0, 1.0), (1, 1.0)),
        scalar_uppers=(),
    )
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is True
    assert abs(L - 1.0) < 1e-12


def test_consistent_scalar_lower_differing_values():
    """Two variables have different lower bounds → ``False``."""
    poset = _CausalPoset(
        m=2, edges=(), scalar_lowers=((0, 1.0), (1, 2.0)),
        scalar_uppers=(),
    )
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is False


def test_consistent_scalar_lower_no_lowers_at_all():
    """No scalar lowers → returns ``(None, True)``  (caller will
    supply a bbox cap)."""
    poset = _CausalPoset(m=3, edges=(), scalar_lowers=(),
                         scalar_uppers=())
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is True
    assert L is None


def test_consistent_scalar_upper_per_var_min():
    """Per-variable scalar uppers are returned as a dict.  When a
    variable has multiple uppers the tightest (min) wins."""
    poset = _CausalPoset(
        m=2, edges=(),
        scalar_lowers=(),
        scalar_uppers=((0, 5.0), (0, 3.0), (1, 10.0)),
    )
    uppers = _causal_poset_consistent_scalar_upper(poset)
    assert uppers == {0: 3.0, 1: 10.0}


# ───────────────────────────────────────────────────────────────────
# _exp_over_chain_simplex
# ───────────────────────────────────────────────────────────────────

def test_chain_simplex_N1_matches_1d_quad():
    """N=1: ∫_L^U exp(α · s) ds = (exp(α·U) − exp(α·L)) / α."""
    import cmath
    alpha = 0.7
    L, U = 0.5, 2.5
    expected = (cmath.exp(alpha * U) - cmath.exp(alpha * L)) / alpha
    val = _exp_over_chain_simplex([alpha], L, U)
    assert abs(val - expected) < 1e-12


def test_chain_simplex_N2_real_values_closed_form():
    """N=2: ∫_0^1 ds_2 exp(q·s_2) ∫_0^{s_2} ds_1 exp(p·s_1)

    Closed form: (1/p) · [(e^{p+q} − 1)/(p+q) − (e^q − 1)/q]
    """
    import cmath
    p, q = 0.3, 0.8
    expected = (1 / p) * (
        (cmath.exp(p + q) - 1) / (p + q)
        - (cmath.exp(q) - 1) / q
    )
    val = _exp_over_chain_simplex([p, q], 0.0, 1.0)
    assert abs(val - expected) < 1e-12


def test_chain_simplex_N3_matches_scipy_nquad():
    """N=3 chain simplex 0 < s_1 < s_2 < s_3 < 1 with random α's.
    Compare closed form to scipy.integrate.nquad."""
    from scipy.integrate import nquad
    import math
    alphas = [0.5, -0.3, 0.9]
    L, U = 0.0, 1.0
    val = _exp_over_chain_simplex(alphas, L, U)
    assert val is not None

    def integrand(s_1, s_2, s_3):
        return math.exp(
            alphas[0] * s_1 + alphas[1] * s_2 + alphas[2] * s_3
        )

    def bounds_s1(s_2, s_3):
        return (L, s_2)

    def bounds_s2(s_3):
        return (L, s_3)

    ref, _ = nquad(integrand,
                    [bounds_s1, bounds_s2, (L, U)],
                    opts={'limit': 200})
    assert abs(val - ref) < 1e-6, f'closed={val}, scipy={ref}'


def test_chain_simplex_complex_alphas():
    """Complex α values (typical for our pole-expanded use case):
    chain integral with α_k = i · p_k."""
    from scipy.integrate import nquad
    import math
    # Real and imaginary parts of α
    alphas_complex = [0.2 + 0.5j, -0.1 + 0.8j, 0.3 + 0.2j]
    val = _exp_over_chain_simplex(alphas_complex, 0.0, 1.0)
    assert val is not None

    def integrand_re(s_1, s_2, s_3):
        import cmath
        z = sum(a * s for a, s in zip(alphas_complex, [s_1, s_2, s_3]))
        return cmath.exp(z).real

    def integrand_im(s_1, s_2, s_3):
        import cmath
        z = sum(a * s for a, s in zip(alphas_complex, [s_1, s_2, s_3]))
        return cmath.exp(z).imag

    def bounds_s1(s_2, s_3):
        return (0.0, s_2)
    def bounds_s2(s_3):
        return (0.0, s_3)

    re_ref, _ = nquad(integrand_re,
                       [bounds_s1, bounds_s2, (0.0, 1.0)],
                       opts={'limit': 200})
    im_ref, _ = nquad(integrand_im,
                       [bounds_s1, bounds_s2, (0.0, 1.0)],
                       opts={'limit': 200})
    ref = complex(re_ref, im_ref)
    assert abs(val - ref) < 1e-6, f'closed={val}, scipy={ref}'


def test_chain_simplex_returns_none_on_degenerate_beta():
    """If two consecutive α's are exact negatives of each other,
    the intermediate β = α_1 + α_2 hits zero and the closed-form
    1/β factor blows up.  Function should return None."""
    val = _exp_over_chain_simplex([1.0, -1.0, 0.5], 0.0, 1.0)
    # β at level 1 = 1.0 (α_1) → non-degenerate, integrate s_1 fine.
    # β at level 2 = 1.0 + (-1.0) = 0.0 → degenerate.
    assert val is None


def test_chain_simplex_returns_none_on_overflow():
    """If the closed-form's per-term ``exp(b · upper)`` would overflow
    IEEE double range, the integrator returns None and the caller
    falls back to scipy.  Trigger by combining a large positive-real
    α with a large positive upper.
    """
    # α = +2 with upper = 1000 ⇒ exp(2000) overflows.
    val = _exp_over_chain_simplex(
        [2.0 + 0j, 0.1 + 0j, 0.1 + 0j], 0.0, 1000.0,
    )
    assert val is None


def test_chain_simplex_translation_invariance():
    """Shifting (L, U) by a common offset h multiplies the result
    by exp((Σ α) · h)."""
    import cmath
    alphas = [0.4 + 0.1j, -0.2j, 0.5]
    base = _exp_over_chain_simplex(alphas, 0.0, 1.0)
    shifted = _exp_over_chain_simplex(alphas, 2.0, 3.0)
    expected = base * cmath.exp(sum(alphas) * 2.0)
    assert abs(shifted - expected) < 1e-10


def test_chain_simplex_N4_matches_scipy():
    """N=4 sanity check vs scipy.nquad.  4-fold nested adaptive
    quadrature is slow but a good cross-check."""
    from scipy.integrate import nquad
    import math
    alphas = [0.5, -0.3, 0.9, 0.2]
    val = _exp_over_chain_simplex(alphas, 0.0, 1.0)
    assert val is not None

    def integrand(s_1, s_2, s_3, s_4):
        return math.exp(
            alphas[0] * s_1 + alphas[1] * s_2
            + alphas[2] * s_3 + alphas[3] * s_4
        )

    def bounds_s1(s_2, s_3, s_4):
        return (0.0, s_2)
    def bounds_s2(s_3, s_4):
        return (0.0, s_3)
    def bounds_s3(s_4):
        return (0.0, s_4)

    ref, _ = nquad(integrand,
                    [bounds_s1, bounds_s2, bounds_s3, (0.0, 1.0)],
                    opts={'limit': 100})
    # 4D adaptive scipy is loose; closed-form should still agree
    # within scipy's accuracy floor.
    assert abs(val - ref) < 1e-5, f'closed={val}, scipy={ref}'


# ───────────────────────────────────────────────────────────────────
# End-to-end: _integrate_nd_polytope_poset_modesum vs scipy.nquad
# ───────────────────────────────────────────────────────────────────

def test_poset_integrator_m3_chain_two_modes_vs_scipy():
    """3 internal vertices in a chain (typical 2-loop topology with
    a sequential causal chain), 3 retarded edges, 2 modes per edge.

    Constraint set mirrors what Phase J actually produces:
      * Δt_0 = s_0  — leaf at t=0 → vertex 0 (scalar lower on s_0)
      * Δt_1 = s_1 - s_0  — vertex 0 → vertex 1 (inter-axis edge)
      * Δt_2 = s_2 - s_1  — vertex 1 → vertex 2 (inter-axis edge)

    No explicit scalar bounds on s_1, s_2 — they inherit via the
    chain ordering.  Compare the analytic poset answer to
    scipy.nquad on the same integrand.
    """
    from scipy.integrate import nquad
    import math

    # Each edge gets a DISJOINT pole pair so the cumulative sums of
    # α's never coincidentally vanish.  For chain structure
    #     Δt_0 = s_0,  Δt_1 = s_1 − s_0,  Δt_2 = s_2 − s_1
    # the α's are
    #     α_0 = λ_0 − λ_1
    #     α_1 = λ_1 − λ_2
    #     α_2 = λ_2
    # Non-degeneracy requires λ_0 ≠ λ_1, λ_0 ≠ λ_2, λ_1 ≠ λ_2,
    # λ_0 ≠ 0, λ_2 ≠ 0.  We additionally use a narrow pole spectrum
    # (0.05–1.0 range) so cumulative β·upper stays well below the
    # double-precision exp overflow threshold (~709).  Disjoint pole
    # sets ensure non-degeneracy.  These ranges match realistic
    # Hawkes propagators.
    smooth_edge_modes = [
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, -0.1 + 0j),
                           (0.3 + 0j, -0.2 + 0j))),
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((0.7 + 0j, -0.3 + 0j),
                           (0.4 + 0j, -0.4 + 0j))),
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((0.9 + 0j, -0.5 + 0j),
                           (0.2 + 0j, -0.6 + 0j))),
    ]
    subset_constraint_data = [
        ([1.0, 0.0, 0.0], [], 0.0),       # s_0 > 0
        ([-1.0, 1.0, 0.0], [], 0.0),      # s_1 > s_0
        ([0.0, -1.0, 1.0], [], 0.0),      # s_2 > s_1
    ]
    free_ext_vals = []
    val = _integrate_nd_polytope_poset_modesum(
        smooth_edge_modes,
        prefactor_complex=1.0 + 0j,
        subset_constraint_data=subset_constraint_data,
        free_ext_vals=free_ext_vals,
        m=3,
    )
    assert val is not None, 'poset integrator should have fired'

    # Reference via scipy.nquad on the explicit closure form.  Cap
    # the outer s_2 integration at 50 to match what the poset
    # integrator does internally (POLYGON_BBOX_CAP / fallback cap).
    def integrand(s_0, s_1, s_2):
        dt0 = s_0
        dt1 = s_1 - s_0
        dt2 = s_2 - s_1
        edge0 = (1.0 * math.exp(-0.1 * dt0)
                 + 0.3 * math.exp(-0.2 * dt0))
        edge1 = (0.7 * math.exp(-0.3 * dt1)
                 + 0.4 * math.exp(-0.4 * dt1))
        edge2 = (0.9 * math.exp(-0.5 * dt2)
                 + 0.2 * math.exp(-0.6 * dt2))
        return edge0 * edge1 * edge2

    def bounds_s0(s_1, s_2):
        return (0.0, s_1)
    def bounds_s1(s_2):
        return (0.0, s_2)

    # The poset integrator uses POLYGON_BBOX_CAP (=200) as the
    # outer cap when no scalar upper is given.  Match that here.
    from msrjd.integration.time_domain.final_integral import (
        POLYGON_BBOX_CAP,
    )
    ref, _ = nquad(
        integrand,
        [bounds_s0, bounds_s1, (0.0, POLYGON_BBOX_CAP)],
        opts={'limit': 200},
    )
    assert abs(val - ref) < 1e-4, (
        f'poset={val}, scipy={ref}, |Δ|={abs(val - ref):.3e}'
    )


def test_poset_integrator_returns_none_on_mixed_constraint():
    """If any constraint is non-poset (e.g. has 3 nonzero a_int
    entries), the integrator should return None — caller falls back
    to scipy."""
    smooth_edge_modes = [
        EdgeModeSum(ri=0, pi=0, delta_coeff=0,
                    modes=((1.0 + 0j, -1.0 + 0j),)),
    ]
    subset_constraint_data = [
        # Mixed: 3 nonzero entries.
        ([1.0, 1.0, -1.0], [], 0.0),
    ]
    val = _integrate_nd_polytope_poset_modesum(
        smooth_edge_modes,
        prefactor_complex=1.0 + 0j,
        subset_constraint_data=subset_constraint_data,
        free_ext_vals=[],
        m=3,
    )
    assert val is None


def test_poset_integrator_skips_m_less_than_3():
    """The poset integrator only applies to m ≥ 3; lower m delegates
    to the polygon (m=2) or 1D paths.  Calling it with m=2 should
    return None to signal "wrong path, try another"."""
    val = _integrate_nd_polytope_poset_modesum(
        smooth_edge_modes=[],
        prefactor_complex=1.0 + 0j,
        subset_constraint_data=[],
        free_ext_vals=[],
        m=2,
    )
    assert val is None

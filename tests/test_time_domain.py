"""
tests/test_time_domain.py
=========================
Phase J (time-domain integration) MVP tests.

Scope: only the Phase J evaluation layer — tree-level diagrams handled
by `msrjd.integration.time_domain`. Loop kernel reduction, kernel
caching, and parent-diagram contraction are NOT yet implemented and
those paths should raise or be marked 'skipped'.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_time_domain.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import (
    SR, I, matrix, DiGraph, solve as sage_solve, heaviside, exp,
)

from msrjd.diagrams.type_assignment import TypedDiagram
from msrjd.core.vertices import VertexType, SourceType
from msrjd.integration.symbolic import (
    build_integrand_stationary,
    group_diagrams_by_kernel,
    integrate_to_time_domain,
)
from msrjd.integration.time_domain import (
    build_G_t_matrix,
    G_t_entry,
    identify_loop_subgraphs,
    integrate_tree_diagram,
    compute_correction_td,
)


# ───────────────────────────────────────────────────────────────────────
# Fixture: 1×1 propagator — single damped mode
# ───────────────────────────────────────────────────────────────────────

def _propagator_data_1pop():
    r"""
    Minimal 1×1 propagator for a single damped mode.

    Kernel K(ω) = 1 + iω (τ = 1). This is the frequency-domain form of
    the retarded time-domain equation (1 + ∂_t) G_R(t) = δ(t), whose
    solution is G_R(t) = Θ(t) exp(-t).

    Pole of det K = 0: ω = I (Im > 0 ✓). Residue matrix C = 1 (1×1).
    Time-domain propagator via pole sum: G(t) = exp(-t).
    """
    omega = SR.var('omega')
    K = matrix(SR, 1, 1, [1 + I * omega])
    adj = K.adjugate()
    D_omega = K.det()
    D_prime = D_omega.diff(omega)

    pole_eqs = sage_solve(D_omega == 0, omega)
    pole_vals = [eq.rhs() for eq in pole_eqs]

    C_mats = []
    for w in pole_vals:
        C_data = [[(I * adj[0, 0].subs({omega: w})
                    / D_prime.subs({omega: w})).factor()]]
        C_mats.append(matrix(SR, C_data))

    G_ft = K.inverse()

    return {
        'G_ft': G_ft,
        'G_ft_explicit': True,
        'adj_ft': adj,
        'D_omega': D_omega,
        'pole_vals': pole_vals,
        'C_mats': C_mats,
        'nf': 1,
    }


# ───────────────────────────────────────────────────────────────────────
# Fixture: diagonal 2×2 propagator — two independent damped modes
# ───────────────────────────────────────────────────────────────────────

def _propagator_data_2pop_diagonal():
    r"""
    Diagonal 2×2 propagator: K(ω) = diag(1 + iω, 1 + 2 iω).

    Pole analysis must be done carefully for a diagonal K because
    det(K) = (1 + iω)(1 + 2iω) has poles at ω₁ = I and ω₂ = I/2. The
    residue matrices at each pole are not diagonal in general — at
    ω₁ = I the (0,0) entry dominates and the (1,1) entry vanishes (and
    vice versa at ω₂).
    """
    omega = SR.var('omega')
    K = matrix(SR, 2, 2, [
        [1 + I * omega, 0],
        [0, 1 + 2 * I * omega],
    ])
    adj = K.adjugate()
    D_omega = K.det().expand()
    D_prime = D_omega.diff(omega)

    pole_eqs = sage_solve(D_omega == 0, omega)
    pole_vals = [eq.rhs() for eq in pole_eqs]

    C_mats = []
    for w in pole_vals:
        nrows, ncols = 2, 2
        C_data = [[SR(0)] * ncols for _ in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                num = adj[i, j].subs({omega: w})
                den = D_prime.subs({omega: w})
                if num != 0:
                    C_data[i][j] = (I * num / den).factor()
        C_mats.append(matrix(SR, C_data))

    G_ft = K.inverse()

    return {
        'G_ft': G_ft,
        'G_ft_explicit': True,
        'adj_ft': adj,
        'D_omega': D_omega,
        'pole_vals': pole_vals,
        'C_mats': C_mats,
        'nf': 2,
    }


# ───────────────────────────────────────────────────────────────────────
# Fixture: a minimal tree-level k=2 diagram (1 source → 2 leaves)
# ───────────────────────────────────────────────────────────────────────

def _tree_k2_single_population(prop_idx=(0, 0)):
    r"""
    Tree-level k=2 diagram with a single source vertex emitting two
    outgoing ñ legs to two distinct leaves.

    Edges: (0 → 1), (0 → 2). Vertex 0 = source. Vertices 1, 2 = leaves.

    The source carries prefactor 1 (we absorb the (1/2) n* symmetry
    factor into `combined_prefactor` at the kernel-group level so the
    test can compare raw integrals without worrying about that factor).

    prop_idx selects which propagator entry to associate with each
    edge. For a 1×1 propagator pass (0, 0). For a 2×2 diagonal
    propagator pass (i, i) to pick the i-th diagonal entry.
    """
    ri, pi = prop_idx
    st = SourceType(SR(1), [('nt', 1), ('nt', 1)], (2, 0))

    D = DiGraph()
    D.add_edges([(0, 1), (0, 2)])
    G = D.to_undirected()
    pd = (D, G, [1, 2], [0])

    return TypedDiagram(
        prediagram=pd,
        vertex_assignments={0: st},
        edge_types={
            (0, 1, None): (('nt', 1), ('dn', 1)),
            (0, 2, None): (('nt', 1), ('dn', 1)),
        },
        external_legs={1: ('dn', 1), 2: ('dn', 1)},
        propagator_indices={
            (0, 1, None): (ri, pi),
            (0, 2, None): (ri, pi),
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Test 1: build_G_t_matrix produces the expected exponential for 1×1
# ═══════════════════════════════════════════════════════════════════════

def test_G_t_matrix_single_pole():
    """
    For K(ω) = 1 + iω, the time-domain propagator should be exp(-t).
    Numerically check at t = 1 against exp(-1).
    """
    pd = _propagator_data_1pop()
    t = SR.var('t')
    G_t = build_G_t_matrix(pd, t)

    val = G_t[0, 0].subs({t: 1})
    expected = float(exp(-1))
    assert abs(float(val.real()) - expected) < 1e-12
    # Symbolically, simplify_full should give exp(-t)
    assert bool((G_t[0, 0] - exp(-t)).simplify_full().is_zero())


# ═══════════════════════════════════════════════════════════════════════
# Test 2: G_t_entry enforces retardation via Heaviside
# ═══════════════════════════════════════════════════════════════════════

def test_G_t_entry_retarded():
    """
    G_t_entry with t_expr = -1 must be killed by Heaviside (return 0);
    with t_expr = +1 must return exp(-1).
    """
    pd = _propagator_data_1pop()
    t = SR.var('t')
    G_t = build_G_t_matrix(pd, t)

    e_pos = G_t_entry(G_t, phys_idx=0, resp_idx=0,
                      t_expr=1.0, include_heaviside=True)
    e_neg = G_t_entry(G_t, phys_idx=0, resp_idx=0,
                      t_expr=-1.0, include_heaviside=True)

    val_pos = float(SR(e_pos).real())
    val_neg = float(SR(e_neg).real())

    assert abs(val_pos - float(exp(-1))) < 1e-12
    assert abs(val_neg - 0.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════════
# Test 3: identify_loop_subgraphs returns [] for tree-level
# ═══════════════════════════════════════════════════════════════════════

def test_subgraph_tree_case_returns_empty():
    """
    A tree-level diagram has no free (loop) frequencies, so
    identify_loop_subgraphs should return [] without raising.
    """
    td = _tree_k2_single_population()
    pd = _propagator_data_1pop()
    ir = build_integrand_stationary(td, pd, k=2)
    # Tree-level: free_freqs is empty.
    assert ir['loop_number'] == 0
    result = identify_loop_subgraphs(ir, td)
    assert result == []


# ═══════════════════════════════════════════════════════════════════════
# Test 4: k=2 tree integrates to the analytical (1/2) exp(-|τ|)
# ═══════════════════════════════════════════════════════════════════════

def test_k2_tree_single_integration_analytical():
    """
    For G_R(t) = Θ(t) exp(-t), the integral

        I(t₁, t₂) = ∫ ds G_R(t₁ - s) G_R(t₂ - s)
                  = ∫_{-∞}^{min(t₁,t₂)} ds exp(-(t₁+t₂) + 2s)
                  = (1/2) exp(-|t₁ - t₂|)

    is the standard convolution result. The Phase J tree evaluator with
    combined_prefactor = 1 should reproduce this at several τ values.
    """
    td = _tree_k2_single_population()
    pd = _propagator_data_1pop()
    ir = build_integrand_stationary(td, pd, k=2)

    t1, t2 = SR.var('t1'), SR.var('t2')
    result = integrate_tree_diagram(
        typed_diagram=td,
        representative_ir=ir,
        propagator_data=pd,
        combined_prefactor=SR(1),
        ext_time_vars=[t1, t2],
        num_params=None,
        origin_leaf_idx=1,  # pin t2 → 0
        timeout_sec=60,
    )

    assert result['status'] == 'ok', (
        f"Tree integration did not succeed: status={result['status']}, "
        f"got {result['contribution']}"
    )

    contribution = result['contribution']
    # With t2 pinned to 0, the result should depend only on t1.
    # Compare against (1/2) exp(-|t1|) at a handful of points.
    for t1_val in (0.5, 1.0, 2.5):
        phase_j_val = float(SR(contribution).subs({t1: t1_val}).real())
        expected = 0.5 * float(exp(-abs(t1_val)))
        assert abs(phase_j_val - expected) < 1e-8, (
            f"At t1={t1_val}: Phase J = {phase_j_val}, "
            f"expected = {expected}"
        )
    # And for negative t1 (origin leaf still at 0):
    for t1_val in (-0.5, -2.0):
        phase_j_val = float(SR(contribution).subs({t1: t1_val}).real())
        expected = 0.5 * float(exp(-abs(t1_val)))
        assert abs(phase_j_val - expected) < 1e-8


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Translation invariance of the k=2 tree result
# ═══════════════════════════════════════════════════════════════════════

def test_k2_tree_translation_invariance():
    """
    The k=2 tree result must depend only on τ = t1 - t2 (not on t1 and
    t2 separately). Run the integration without pinning the origin and
    verify that evaluating at (t1=1, t2=0) and (t1=6, t2=5) gives the
    same number.
    """
    td = _tree_k2_single_population()
    pd = _propagator_data_1pop()
    ir = build_integrand_stationary(td, pd, k=2)

    t1, t2 = SR.var('t1'), SR.var('t2')
    result = integrate_tree_diagram(
        typed_diagram=td,
        representative_ir=ir,
        propagator_data=pd,
        combined_prefactor=SR(1),
        ext_time_vars=[t1, t2],
        num_params=None,
        origin_leaf_idx=None,  # keep both free
        timeout_sec=60,
    )

    assert result['status'] == 'ok'
    contribution = SR(result['contribution'])

    val_a = float(contribution.subs({t1: 1, t2: 0}).real())
    val_b = float(contribution.subs({t1: 6, t2: 5}).real())

    assert abs(val_a - val_b) < 1e-8, (
        f"Translation invariance violated: "
        f"(t1=1,t2=0) → {val_a}, (t1=6,t2=5) → {val_b}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Test 6: End-to-end Phase J vs Phase I on the k=2 tree
# ═══════════════════════════════════════════════════════════════════════

def test_phase_J_vs_phase_I_linear_hawkes_tree():
    """
    End-to-end MVP validation.

    For the same tree k=2 diagram and the same propagator data:
      - Phase I (integrate_to_time_domain) computes C(t₁, t₂) via
        residue integration in frequency space.
      - Phase J (compute_correction_td) computes C(t₁, t₂) via
        vertex-time integration in the time domain.

    Both must agree to within 1e-6 absolute tolerance at a handful of
    τ values.
    """
    td = _tree_k2_single_population()
    pd = _propagator_data_1pop()

    # Phase I residue result
    ir = build_integrand_stationary(td, pd, k=2)
    td_result = integrate_to_time_domain(ir)
    phase_i_expr = td_result['time_domain_result']
    phase_i_t1, phase_i_t2 = ir['ext_times']

    # Phase J time-domain result — reuse the same kernel group structure
    kernel_groups = group_diagrams_by_kernel([td], pd, k=2)
    assert len(kernel_groups) == 1
    assert kernel_groups[0]['loop_number'] == 0

    j_t1, j_t2 = SR.var('t1'), SR.var('t2')
    j_result = compute_correction_td(
        kernel_groups=kernel_groups,
        propagator_data=pd,
        k=2,
        num_params=None,
        ext_time_vars=[j_t1, j_t2],
        origin_leaf_idx=1,
        timeout_sec=60,
    )
    assert not j_result['skipped_kernel_ids']
    phase_j_expr = SR(j_result['total_C'])

    # Compare at several τ > 0 values (with t2 = 0 pinned in Phase J).
    for t1_val in (0.25, 0.75, 1.5, 3.0):
        pi_val = float(SR(phase_i_expr).subs(
            {phase_i_t1: t1_val, phase_i_t2: 0}
        ).real())
        pj_val = float(phase_j_expr.subs({j_t1: t1_val}).real())
        assert abs(pi_val - pj_val) < 1e-6, (
            f"Phase I vs Phase J mismatch at τ={t1_val}: "
            f"PI={pi_val}, PJ={pj_val}"
        )

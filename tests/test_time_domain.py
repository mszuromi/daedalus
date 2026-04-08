"""
tests/test_time_domain.py
=========================
Phase J (time-domain integration) MVP tests.

Scope: only the Phase J evaluation layer — tree-level diagrams handled
by `msrjd.integration.time_domain`. Loop kernel reduction, kernel
caching, and parent-diagram contraction are NOT yet implemented and
those paths should raise or be marked 'skipped'.

The tree evaluator uses explicit numerical quadrature
(`scipy.integrate.quad` / `nquad`) on a `fast_callable` version of the
integrand, with polytope bounds extracted from the retarded Heaviside
factors. It returns a Python callable `f(*ext_time_values) -> complex`,
not a symbolic SR expression — these tests call that callable directly.

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

    Poles at ω₁ = I and ω₂ = I/2. Not used by every test but useful
    as a sanity-check fixture.
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

    is the standard convolution result. The Phase J tree evaluator
    with combined_prefactor = 1 should reproduce this at several τ
    values. The returned contribution is a Python callable
    (numerical quadrature) — we invoke it directly.
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
    )

    assert result['status'] == 'ok', (
        f"Tree integration did not succeed: status={result['status']}"
    )

    contribution = result['contribution']
    # Positional call: (value_for_t1, value_for_t2) — the t2 slot is
    # ignored because it was pinned to 0 internally.
    for t1_val in (0.5, 1.0, 2.5):
        phase_j_val = complex(contribution(t1_val, 0.0))
        expected = 0.5 * float(exp(-abs(t1_val)))
        assert abs(phase_j_val.real - expected) < 1e-8, (
            f"At t1={t1_val}: Phase J = {phase_j_val.real}, "
            f"expected = {expected}"
        )
        assert abs(phase_j_val.imag) < 1e-10
    for t1_val in (-0.5, -2.0):
        phase_j_val = complex(contribution(t1_val, 0.0))
        expected = 0.5 * float(exp(-abs(t1_val)))
        assert abs(phase_j_val.real - expected) < 1e-8


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
    )

    assert result['status'] == 'ok'
    contribution = result['contribution']

    val_a = complex(contribution(1.0, 0.0))
    val_b = complex(contribution(6.0, 5.0))

    assert abs(val_a.real - val_b.real) < 1e-8, (
        f"Translation invariance violated: "
        f"(t1=1,t2=0) → {val_a.real}, (t1=6,t2=5) → {val_b.real}"
    )
    assert abs(val_a.imag) < 1e-10 and abs(val_b.imag) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
# Test 6b: Nondiagonal 2×2 propagator — regression test for the
#          numerical-overflow bug that shipped on 2026-04-08.
# ═══════════════════════════════════════════════════════════════════════

def _propagator_data_2pop_nondiagonal():
    r"""
    Non-diagonal 2×2 propagator with cross-population coupling:

        K(ω) = [[1 + iω,   -3/10 ],
                [ -2/10,   1 + iω]]

    This is the minimal fixture whose time-domain integrand is a
    product of **multiple** sums of exponentials — exactly the
    configuration where the fast_callable evaluator overflows IEEE
    double precision at large negative `s` if the integrand is not
    expanded into a sum of single exponentials before JIT-compiling.

    Pole check: det K = (1+iω)² − 6/100, so poles are at
    ω = i(1 ± √6/10), both with Im > 0 (retarded).
    """
    omega = SR.var('omega')
    K = matrix(SR, 2, 2, [
        [1 + I * omega,       -SR(3)/10],
        [-SR(2)/10,            1 + I * omega],
    ])
    adj = K.adjugate()
    D_omega = K.det().expand()
    D_prime = D_omega.diff(omega)

    pole_eqs = sage_solve(D_omega == 0, omega)
    pole_vals = [eq.rhs() for eq in pole_eqs]

    C_mats = []
    for w in pole_vals:
        Cm = [[SR(0)] * 2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                num = adj[i, j].subs({omega: w})
                den = D_prime.subs({omega: w})
                if num != 0:
                    Cm[i][j] = (I * num / den).factor()
        C_mats.append(matrix(SR, Cm))

    return {
        'G_ft': K.inverse(),
        'G_ft_explicit': True,
        'adj_ft': adj,
        'D_omega': D_omega,
        'pole_vals': pole_vals,
        'C_mats': C_mats,
        'nf': 2,
    }


def _tree_k2_cross_population():
    r"""
    Tree-level k=2 diagram whose two outgoing edges from the source
    point at DIFFERENT physical fields (dn_1 and dn_2), forcing the
    tree evaluator to multiply two time-domain propagators with
    DIFFERENT pole decompositions — reproducing the nondiagonal case
    that triggered the fast_callable overflow bug.

    Edge (0, 1): (resp_row=0, phys_col=0) → G_t[0, 0]
    Edge (0, 2): (resp_row=0, phys_col=1) → G_t[1, 0]   (off-diagonal)
    """
    st = SourceType(SR(1), [('nt', 1), ('nt', 1)], (2, 0))
    D = DiGraph()
    D.add_edges([(0, 1), (0, 2)])
    pd = (D, D.to_undirected(), [1, 2], [0])
    return TypedDiagram(
        prediagram=pd,
        vertex_assignments={0: st},
        edge_types={
            (0, 1, None): (('nt', 1), ('dn', 1)),
            (0, 2, None): (('nt', 1), ('dn', 2)),
        },
        external_legs={1: ('dn', 1), 2: ('dn', 2)},
        propagator_indices={
            (0, 1, None): (0, 0),
            (0, 2, None): (0, 1),
        },
    )


def test_phase_J_nondiagonal_2x2_does_not_overflow():
    r"""
    Regression test for the 2026-04-08 overflow bug.

    Phase J on a 2×2 nondiagonal propagator used to return `nan` (or,
    after silently failing to expand the exponential product, to
    return a symmetric curve that was half the correct amplitude).
    Verify that the tree evaluator now returns finite values at
    several τ points AND that the result is **asymmetric** in τ (the
    off-diagonal coupling breaks the τ → −τ symmetry present only for
    fully diagonal kernels).
    """
    from sage.all import fast_callable, CDF

    pd = _propagator_data_2pop_nondiagonal()
    td = _tree_k2_cross_population()

    kernel_groups = group_diagrams_by_kernel([td], pd, k=2)
    assert len(kernel_groups) == 1
    j_result = compute_correction_td(
        kernel_groups=kernel_groups,
        propagator_data=pd,
        k=2,
        num_params=None,
        origin_leaf_idx=1,
    )
    assert not j_result['skipped_kernel_ids']
    total_C = j_result['total_C']

    # Finite values at a grid of τ points (no NaN)
    tau_vals = [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0]
    results = {}
    for tv in tau_vals:
        val = complex(total_C(tv, 0.0)).real
        assert not (val != val), f'Phase J returned NaN at τ={tv}'
        assert abs(val) < 1e3, f'Phase J returned unreasonable value at τ={tv}: {val}'
        results[tv] = val

    # Asymmetry: the nondiagonal propagator breaks τ → −τ symmetry.
    # The absolute values at ±τ should differ at the several-percent level.
    for pos_t in (0.3, 1.0, 3.0):
        diff = abs(results[pos_t] - results[-pos_t])
        rel = diff / max(abs(results[pos_t]), abs(results[-pos_t]), 1e-12)
        assert rel > 1e-3, (
            f'Phase J output symmetric in τ at ±{pos_t}, but the '
            f'nondiagonal propagator should break symmetry: '
            f'C(+{pos_t}) = {results[pos_t]}, C(-{pos_t}) = {results[-pos_t]}'
        )

    # Cross-check against a direct numerical FFT of the frequency-domain
    # integrand (notebook-style Phase I reference). They should agree to
    # FFT-grid-resolution accuracy (~1e-3 at N=4096, Omega_max=80).
    ir = build_integrand_stationary(td, pd, k=2)
    ext_var = ir['ext_freqs'][0]
    f_spectrum = fast_callable(ir['integrand'], vars=[ext_var], domain=CDF)

    import numpy as np
    N, Omega_max = 16384, 200.0
    d_omega = 2 * Omega_max / N
    omega_grid = np.linspace(-Omega_max + d_omega / 2,
                             Omega_max - d_omega / 2, N)
    S_vals = np.array([complex(f_spectrum(w)) for w in omega_grid])
    Delta_omega = omega_grid[1] - omega_grid[0]
    tau_grid = np.fft.fftshift(
        np.fft.fftfreq(N, d=Delta_omega / (2 * np.pi)))
    scale = N * Delta_omega / (2 * np.pi)
    C_fft = (np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(S_vals)))
             * scale * float(ir['scalar_prefactor']))

    for tv in tau_vals:
        idx = int(np.argmin(np.abs(tau_grid - tv)))
        fft_val = C_fft[idx].real
        pj_val = results[tv]
        assert abs(fft_val - pj_val) < 5e-3, (
            f'Phase J disagrees with notebook FFT IFT at τ={tv}: '
            f'FFT={fft_val}, PhaseJ={pj_val}'
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
        numerical vertex-time integration in the time domain.

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

    # Phase J numerical result — reuse the same kernel group structure
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
    )
    assert not j_result['skipped_kernel_ids']
    total_C_fn = j_result['total_C']

    # Compare at several τ > 0 values (with t2 = 0 pinned in Phase J).
    for t1_val in (0.25, 0.75, 1.5, 3.0):
        pi_val = float(SR(phase_i_expr).subs(
            {phase_i_t1: t1_val, phase_i_t2: 0}
        ).real())
        pj_val = complex(total_C_fn(t1_val, 0.0)).real
        assert abs(pi_val - pj_val) < 1e-6, (
            f"Phase I vs Phase J mismatch at τ={t1_val}: "
            f"PI={pi_val}, PJ={pj_val}"
        )

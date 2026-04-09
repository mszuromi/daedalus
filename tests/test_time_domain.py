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
    eval_delta_contributions_on_tau_grid,
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
    For K(ω) = 1 + iω, the smooth time-domain propagator should be
    exp(-t), and the δ(t) coefficient should be zero (this propagator
    has no instantaneous response — G_ft decays as 1/ω at ω → ∞).
    """
    pd = _propagator_data_1pop()
    t = SR.var('t')
    G_t_obj = build_G_t_matrix(pd, t)

    assert isinstance(G_t_obj, dict)
    smooth = G_t_obj['smooth']
    delta = G_t_obj['delta']

    val = smooth[0, 0].subs({t: 1})
    expected = float(exp(-1))
    assert abs(float(val.real()) - expected) < 1e-12
    assert bool((smooth[0, 0] - exp(-t)).simplify_full().is_zero())
    # No δ component for a 1/(α + iω) propagator (decays as 1/ω)
    assert complex(delta[0, 0]) == 0


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
# Test 6c: δ(t) component fix — Hawkes-style "instantaneous response"
# ═══════════════════════════════════════════════════════════════════════

def _propagator_data_instantaneous_pair():
    r"""
    Minimal 2×2 kernel whose frequency-domain inverse has an entry
    with a nonzero `ω → ∞` limit — i.e., a δ(t) component in the
    retarded time-domain propagator:

        K(ω) = [[1,      1       ],
                [-a,     1 + iω  ]]

    `det K = (1 + iω) + a`, so the unique pole is at
    `ω = i·(a − 1)`; it sits in the upper half plane for `a < 1`.

    Inverse: G_ft[0,0] = (1 + iω) / (1 + iω + a) → 1 as ω → ∞, so
    `G_R[0,0](t)` has a δ(t) component of weight 1. This mimics the
    `ñ × δn` coupling in the MSR-JD Hawkes action, where a ñ source
    at time t produces an immediate δn response at the same time t.

    `G_ft[1,0] = a / (1 + iω + a)` has lim → 0, so `G_R[1,0](t)` is
    smooth (no δ).
    """
    omega = SR.var('omega')
    a = SR(3) / 10
    K = matrix(SR, 2, 2, [
        [SR(1),   SR(1)],
        [-a,      1 + I * omega],
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


def test_G_t_matrix_detects_delta_component():
    """
    Regression test for the 2026-04-08 "missing δ(t) component" bug.

    `build_G_t_matrix` must return a dict with both 'smooth' and
    'delta' keys. The delta matrix must be nonzero exactly on the
    (row, col) entries whose frequency-domain propagator has a
    nonzero `ω → ∞` limit.
    """
    pd = _propagator_data_instantaneous_pair()
    t = SR.var('t')
    obj = build_G_t_matrix(pd, t)

    assert isinstance(obj, dict)
    assert 'smooth' in obj and 'delta' in obj

    delta = obj['delta']
    d00 = complex(delta[0, 0])
    d01 = complex(delta[0, 1])
    d10 = complex(delta[1, 0])
    d11 = complex(delta[1, 1])

    # G_ft[0,0] = (1+iω)/(1+iω+a) → 1 as ω→∞ → δ(t) coeff = 1
    assert abs(d00 - 1.0) < 1e-6, (
        f'Expected δ coeff at [0,0] to be ~1, got {d00}'
    )
    # All other entries have lim → 0 → no δ
    assert abs(d01) < 1e-6
    assert abs(d10) < 1e-6
    assert abs(d11) < 1e-6


def test_phase_J_delta_component_asymmetric_cross_correlator():
    r"""
    End-to-end regression: the δ(t) component of an instantaneous
    propagator entry was missing from Phase J before the 2026-04-08
    fix, causing the resulting C(τ) to be symmetric and ~half the
    correct amplitude. After the fix, Phase J should match the
    analytical closed-form answer derived by hand from the δ-edge
    subset expansion.

    Fixture: 2×2 "instantaneous" K (see _propagator_data_instantaneous_pair).
    Tree diagram: single source with two outgoing edges to distinct
    leaves, using different matrix entries — one with a δ component
    (G_ft[0,0]) and one without (G_ft[1,0]).

    NB: we do NOT compare against `scipy.integrate.quad` IFT for this
    fixture because the spectrum decays only as 1/ω at infinity
    (from the `+1` constant in G_ft[0,0]), so the oscillatory integral
    converges very slowly and scipy's adaptive quadrature is unreliable
    as a reference. The analytical formula is derived from explicit
    decomposition of G_R into its δ and smooth parts:

        G_R[0,0](t) = δ(t) − a · exp(-(1+a)·t) · Θ(t)
        G_R[1,0](t) =         a · exp(-(1+a)·t) · Θ(t)

    with a = 3/10. Convolving (∫ G_R[0,0](t_1 − s) · G_R[1,0](−s) ds)
    and multiplying by the pipeline's scalar_prefactor gives:

        τ > 0:  pref · (−a² / (2(1+a))) · exp(−(1+a)τ)
        τ < 0:  pref · a · exp((1+a)τ) · (2 + a) / (2(1+a))

    These are asymmetric because the δ edge contribution at τ < 0 is
    not mirrored at τ > 0.
    """
    from sage.all import exp as sexp
    import math

    pd = _propagator_data_instantaneous_pair()

    # Tree k=2 diagram: source → leaf_1 via (ri=0, pi=0), source → leaf_2 via (ri=0, pi=1)
    st = SourceType(SR(1), [('nt', 1), ('nt', 1)], (2, 0))
    D = DiGraph()
    D.add_edges([(0, 1), (0, 2)])
    pd_prediag = (D, D.to_undirected(), [1, 2], [0])
    td = TypedDiagram(
        prediagram=pd_prediag,
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
    total_C_J = j_result['total_C']

    # Pipeline prefactor (same as what Phase J uses internally)
    ir = build_integrand_stationary(td, pd, k=2)
    pref = float(ir['scalar_prefactor'])

    a = 0.3
    one_plus_a = 1.0 + a

    def C_analytic(tau):
        if tau > 0:
            return pref * (-a * a / (2.0 * one_plus_a)) * math.exp(-one_plus_a * tau)
        elif tau < 0:
            coeff = a * (2.0 + a) / (2.0 * one_plus_a)
            return pref * coeff * math.exp(one_plus_a * tau)
        else:
            # τ = 0: shot-noise δ contribution, not captured by callable.
            # Phase J returns the limit of the smooth τ → 0⁺ piece.
            return pref * (-a * a / (2.0 * one_plus_a))

    tau_vals = [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0]
    for tv in tau_vals:
        exp_val = C_analytic(tv)
        pj_val = complex(total_C_J(tv, 0.0)).real
        assert abs(exp_val - pj_val) < 1e-10, (
            f'Phase J does not match analytic at τ={tv}: '
            f'analytic={exp_val}, PhaseJ={pj_val}, '
            f'diff={abs(exp_val-pj_val):.2e}'
        )

    # Also verify the result is actually ASYMMETRIC (the old bug made
    # it symmetric). Compare C(+τ) vs C(-τ) at two τ values.
    for tv in (1.0, 3.0):
        c_pos = complex(total_C_J(tv, 0.0)).real
        c_neg = complex(total_C_J(-tv, 0.0)).real
        rel = abs(c_pos - c_neg) / max(abs(c_pos), abs(c_neg), 1e-12)
        assert rel > 1e-2, (
            f'Phase J output is symmetric at ±{tv}, but the δ edge '
            f'must break τ → −τ symmetry: '
            f'C(+{tv}) = {c_pos}, C(-{tv}) = {c_neg}'
        )


# ═══════════════════════════════════════════════════════════════════════
# Test 6d: Shot-noise δ(τ) spike on the k=2 autocorrelator
# ═══════════════════════════════════════════════════════════════════════

def test_phase_J_autocorrelator_delta_spike_at_origin():
    r"""
    Regression for the "missing δ(τ) spike" symptom on a k=2
    autocorrelator `⟨δn₁(t₁) δn₁(t₂)⟩`.

    With both tree edges reading the same instantaneous matrix entry
    `G_ft[0,0]`, the subset `S = {both edges}` pins s = t₁ = t₂ and
    produces a pure δ(t₁ - t₂) shot-noise spike at τ = 0. The
    continuous Phase J callable cannot represent this; instead, the
    spike is returned as a structured `delta_contributions` entry
    that the caller can insert into a discrete τ grid via
    `eval_delta_contributions_on_tau_grid`.

    This test:
      1. Runs Phase J on the 2×2 instantaneous fixture, autocorrelator edges.
      2. Asserts exactly one `delta_contributions` entry is produced.
      3. Asserts the equality fires at τ = 0.
      4. Asserts the coefficient is `combined_prefactor × c_δ²` where
         `c_δ = lim_{ω→∞} G_ft[0,0] = 1`.
      5. Runs the discrete-grid helper and verifies that the spike
         height times the bin width recovers the analytic total weight.
    """
    import numpy as np

    pd = _propagator_data_instantaneous_pair()

    # Autocorrelator: both edges → G_ft[0,0] (has δ component)
    st = SourceType(SR(1), [('nt', 1), ('nt', 1)], (2, 0))
    D = DiGraph()
    D.add_edges([(0, 1), (0, 2)])
    pd_prediag = (D, D.to_undirected(), [1, 2], [0])
    td = TypedDiagram(
        prediagram=pd_prediag,
        vertex_assignments={0: st},
        edge_types={
            (0, 1, None): (('nt', 1), ('dn', 1)),
            (0, 2, None): (('nt', 1), ('dn', 1)),
        },
        external_legs={1: ('dn', 1), 2: ('dn', 1)},
        propagator_indices={
            (0, 1, None): (0, 0),
            (0, 2, None): (0, 0),
        },
    )

    kernel_groups = group_diagrams_by_kernel([td], pd, k=2)
    assert len(kernel_groups) == 1

    j_result = compute_correction_td(
        kernel_groups=kernel_groups,
        propagator_data=pd,
        k=2,
        num_params=None,
        origin_leaf_idx=1,
    )

    delta_contribs = j_result['delta_contributions']
    assert len(delta_contribs) == 1, (
        f'Expected exactly one δ contribution for the autocorrelator, '
        f'got {len(delta_contribs)}'
    )

    dc = delta_contribs[0]
    # Equality: should fire at τ = 0 (the ±t₁ = 0 constraint)
    a0 = dc['equality_a'][0]
    c0 = dc['equality_c']
    assert abs(a0) > 1e-12
    tau_fire = -c0 / a0
    assert abs(tau_fire) < 1e-12, (
        f'δ spike should fire at τ=0, got {tau_fire}'
    )

    # Coefficient at the fire point
    coeff_val = complex(dc['coeff_fc'](float(tau_fire)))
    # For this diagram:
    #   combined_prefactor = scalar_prefactor = -2 (the standard MSR-JD
    #     source factor for SourceType(SR(1), [ñ,ñ], (2,0)))
    #   c_δ = G_ft[0,0](∞) = 1
    #   both-δ weight = combined_pf × c_δ × c_δ × (no smooth factors) = -2
    assert abs(coeff_val.real - (-2.0)) < 1e-10, (
        f'δ coefficient at τ=0 should be -2, got {coeff_val}'
    )
    assert abs(coeff_val.imag) < 1e-10

    # Discretize onto a τ grid and check the spike integrates correctly
    tau_grid = np.linspace(-5.0, 5.0, 501)
    spikes = eval_delta_contributions_on_tau_grid(
        delta_contribs, tau_grid, free_ext_dim=1,
    )
    dtau = float(tau_grid[1] - tau_grid[0])

    # Most bins should be zero
    nonzero_idx = np.nonzero(np.abs(spikes) > 1e-10)[0]
    assert len(nonzero_idx) == 1, (
        f'Expected spike in exactly one bin, got {len(nonzero_idx)}'
    )
    # The nonzero bin should be at τ = 0
    assert abs(tau_grid[nonzero_idx[0]]) < 1e-10, (
        f'Spike should be at τ=0, got τ={tau_grid[nonzero_idx[0]]}'
    )
    # Bin height × bin width should recover the analytic weight (-2)
    integrated_weight = complex(spikes[nonzero_idx[0]]) * dtau
    assert abs(integrated_weight.real - (-2.0)) < 1e-10, (
        f'Integrated δ weight should be -2, got {integrated_weight}'
    )
    assert abs(integrated_weight.imag) < 1e-10


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

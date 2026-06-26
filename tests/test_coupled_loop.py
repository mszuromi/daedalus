"""
tests/test_coupled_loop.py
==========================
Dyson 3c (loop-level coupled integrator via spectral assignments):

  * **TREE ANCHOR** — the ell=0 record driven through the full assignment
    machinery (fpairs → projector elements → batched spectral kinematic, pv
    as enumerated, NO 2^{−n_C}) equals the spectral-Lyapunov 2-point
    ``coupled_two_point`` — pins weights, pv convention, fpairs threading and
    the two-segment C representation against 3a's sim-validated result;
  * **DECOUPLED-LIMIT CROSS-VALIDATION** — a 2-species theory with tiny cross
    coupling (g=h=1e−8, symbolically nonzero ⇒ routes through the coupled
    spectral-assignment path) reproduces the TRUSTED single-field generic loop
    path (max_ell=1) for C_aa — the strongest end-to-end check: every loop
    diagram, both paths, must agree;
  * coupled nonlinear e2e smoke through compute_cumulants (max_ell=1):
    runs, real, finite, and the cubic loop correction reduces C_aa(0,0).

Run:  sage -python -m pytest tests/test_coupled_loop.py -q
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.core.field_theory import FieldTheory                  # noqa: E402
from api._propagator import build_propagator                # noqa: E402
from api.theory import SpatialTheoryBuilder                 # noqa: E402

_MUA, _MUB, _D, _TA, _TB = 1.5, 1.2, 0.8, 1.0, 0.7


def _two_species(g, h, ga=0.0, gb=0.0):
    b = (SpatialTheoryBuilder('coupled2_loop')
         .physical_field('a', spatial_dim=1)
         .physical_field('b', spatial_dim=1)
         .parameter('mua', default=_MUA, domain='positive')
         .parameter('mub', default=_MUB, domain='positive')
         .parameter('D', default=_D, domain='positive')
         .parameter('g', default=g)
         .parameter('h', default=h)
         .parameter('ga', default=ga)
         .parameter('gb', default=gb)
         .parameter('Ta', default=_TA, domain='positive')
         .parameter('Tb', default=_TB, domain='positive'))
    action = ('at*((Dt+mua-D*Laplacian)*a + g*b + ga*a^3) '
              '+ bt*((Dt+mub-D*Laplacian)*b - h*a + gb*b^3) '
              '- Ta*at^2 - Tb*bt^2')
    return (b.set_action_text(action)
            .equation(lhs='(Dt+mua-D*Laplacian)*a + g*b + ga*a^3', rhs='0')
            .equation(lhs='(Dt+mub-D*Laplacian)*b - h*a + gb*b^3', rhs='0')
            .boundary('infinite').initial('stationary').build())


def _fund(g, h, ga=0.0, gb=0.0):
    return {'mua': _MUA, 'mub': _MUB, 'D': _D, 'g': g, 'h': h,
            'ga': ga, 'gb': gb, 'Ta': _TA, 'Tb': _TB}


def _setup(model):
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    return ft, prop


@pytest.mark.parametrize('legs,qv,tau', [
    ((('a', 1), ('a', 1)), 0.0, 0.4),
    ((('a', 1), ('b', 1)), 0.9, 0.0),
    ((('a', 1), ('b', 1)), 0.6, 0.7),
])
def test_tree_anchor_assignment_machinery_equals_lyapunov(legs, qv, tau):
    """ell=0 record × spectral assignments == coupled_two_point (q-space)."""
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.full_integrator import (
        diagram_kinematic_spectral, spectral_rows, external_times_2pt)
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _norm_sr)
    from engine.integration.spatial.spatial_correlator import extract_noise_matrix
    from engine.integration.spatial.heat_kernel import reaction_diffusion_matrices
    from engine.integration.spatial.spectral_propagator import (
        build_reference, coupled_two_point, spectral_projectors)
    from sage.all import SR, var

    g, h = 0.4, 0.3
    model = _two_species(g, h)
    ft, prop = _setup(model)
    fund = _fund(g, h)

    # exact M, N from the symbolic K_ft / noise sector
    omega, k_var = var('omega'), var('k')
    lap = var('Laplacian')
    M_sr, D_sr, _ = reaction_diffusion_matrices(prop['K_ft'], omega, k_var, lap)
    sub = {var(kk): vv for kk, vv in fund.items()}
    Mn = np.array([[float(M_sr[i, j].subs(sub)) for j in range(2)]
                   for i in range(2)])
    Dn = np.array([[float(D_sr[i, j].subs(sub)) for j in range(2)]
                   for i in range(2)])
    ref = build_reference(Mn, Dn)
    N = extract_noise_matrix(ft, {kk: vv for kk, vv in fund.items()})
    eig, proj = spectral_projectors(Mn)

    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(list(legs), phys_idx)
    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=0)
    trees = by_ell.get(0, [])
    assert trees, 'no tree record enumerated'

    nps_sr = _norm_sr(fund)
    ei_, ej_ = int(phys_idx[ext_int[0]]), int(phys_idx[ext_int[-1]])
    total = 0.0 + 0.0j
    for td, pre in trees:
        pv = float(SR(pre).subs(nps_sr))
        dd = diagram_to_cstack(td)
        rows = spectral_rows(dd)
        n_rows = len(rows)
        elems = np.empty((n_rows, 2), dtype=complex)
        for r, (ei, e, half) in enumerate(rows):
            assert e.fpairs, 'tree edge lost its fpairs'
            ri_, pi_ = e.fpairs[0] if half in ('R', 'Cu') else e.fpairs[1]
            for a_i in range(2):
                elems[r, a_i] = proj[a_i][pi_, ri_]
        assign = np.array(list(itertools.product(range(2), repeat=n_rows))).T
        Wgt = np.prod(elems[np.arange(n_rows)[:, None], assign], axis=0)
        mass_table = eig[assign]
        # orientation-resolved external times (the i≠j rule): the i leaf sits
        # at +τ (C_ij(τ) = ⟨φ_i(t+τ)φ_j(t)⟩); mirror records swap leaves
        leaves = list(dd.external_legs)
        if ei_ == ej_:
            et = external_times_2pt(dd, tau)
        else:
            lf = {leaf: int(phys_idx[fld])
                  for leaf, fld in td.external_legs.items()}
            if (lf[leaves[0]], lf[leaves[1]]) == (ei_, ej_):
                et = {leaves[0]: tau, leaves[1]: 0.0}
            else:
                et = {leaves[0]: 0.0, leaves[1]: tau}
        I = diagram_kinematic_spectral(
            dd, [qv], et, mass_table, ref.D0, n_s=40)
        total += pv * (Wgt @ I)

    # Lyapunov / fluctuation-regression oracle (3a, sim-validated)
    expected = coupled_two_point(ref, N, qv * qv, tau)[ei_, ej_]
    assert abs(total.imag) < 1e-10
    assert total.real == pytest.approx(float(expected.real), rel=2e-5), \
        f'assignment tree {total.real:.8f} vs Lyapunov {expected.real:.8f}'


def test_decoupled_limit_matches_single_field_loop(monkeypatch):
    """g=h=1e-8 (symbolically coupled ⇒ spectral-assignment path) at max_ell=1
    == the trusted single-field generic loop path on the equivalent scalar
    theory.  Cross-validates every loop diagram through both drivers.

    Both paths run at SPATIAL_GRID_NT/NS = 40 (the shared env override): at
    the 22-node default each path carries ~1-4% quadrature error on the {C,R}
    diagram at τ=0.8 — in opposite directions because the coupled driver's
    window scale is min Re eig (=μ_b) while the scalar path's is μ_a.  At
    n_t=40 both are within ~5e-4 of the n_t=60 converged value (probed); at
    equal (n_t,n_s,mu_scale) they agree bit-for-bit."""
    from api import compute_cumulants

    monkeypatch.setenv('SPATIAL_GRID_NT', '40')
    monkeypatch.setenv('SPATIAL_GRID_NS', '40')
    eps = 1e-8
    ga = 0.4
    sg = np.array([0.0, 0.8])
    kw = dict(k=2, max_ell=1, tau_max=0.8, tau_step=0.4,
              parallel=False, verbose=False, use_cache=False)

    model_c = _two_species(eps, eps, ga=ga, gb=0.3)
    th_c = compute_cumulants(model=model_c, fundamental=_fund(eps, eps, ga, 0.3),
                             external_fields=[('a', 1), ('a', 1)],
                             spatial_grid=sg, **kw)
    assert th_c['spatial_info'].get('coupled_loop'), \
        'coupled theory did not route through the spectral-assignment loop path'
    C_c = np.asarray(th_c['C_tau_x']).real

    scalar = (SpatialTheoryBuilder('single_a')
              .physical_field('phi', spatial_dim=1)
              .parameter('mu', default=_MUA, domain='positive')
              .parameter('D', default=_D, domain='positive')
              .parameter('gph', default=ga)
              .parameter('T', default=_TA, domain='positive')
              .set_action_text('phit*((Dt+mu-D*Laplacian)*phi + gph*phi^3) '
                               '- T*phit^2')
              .equation(lhs='(Dt+mu-D*Laplacian)*phi + gph*phi^3', rhs='0')
              .boundary('infinite').initial('stationary').build())
    th_s = compute_cumulants(model=scalar,
                             fundamental={'mu': _MUA, 'D': _D, 'gph': ga,
                                          'T': _TA},
                             external_fields=[('phi', 1), ('phi', 1)],
                             spatial_grid=sg, **kw)
    C_s = np.asarray(th_s['C_tau_x']).real

    assert C_c.shape == C_s.shape
    scale = np.max(np.abs(C_s))
    assert np.max(np.abs(C_c - C_s)) < 4e-3 * scale, \
        f'decoupled-limit mismatch:\ncoupled\n{C_c}\nvs single-field\n{C_s}'


def test_cross_correlator_loop_vs_hartree_oracle(monkeypatch):
    """The DISCRIMINATING cross-correlator check (reviewer's instrument): for a
    coupled CUBIC theory the 1-loop correction is exactly the first-order
    response to the Hartree mass shift δM = diag(3g_i·C_ii(x=0,τ=0)).  Compare
    δC_ab(x,τ) from the spectral-assignment driver against the matrix
    finite-difference of the exact tree correlator — at ±τ (the cross
    correlator is genuinely τ-ASYMMETRIC; the old symmetrized double-count
    would fail this at both signs and by 2× at τ=0)."""
    from scipy.linalg import expm, solve_continuous_lyapunov
    from api import compute_cumulants

    monkeypatch.setenv('SPATIAL_GRID_NT', '40')
    monkeypatch.setenv('SPATIAL_GRID_NS', '40')
    g, h, ga, gb = 0.4, 0.3, 0.3, 0.25
    model = _two_species(g, h, ga=ga, gb=gb)
    xs = np.array([0.0, 0.8])
    th = compute_cumulants(model=model, k=2, max_ell=1,
                           fundamental=_fund(g, h, ga, gb),
                           external_fields=[('a', 1), ('b', 1)],
                           tau_max=0.7, tau_step=0.7, spatial_grid=xs,
                           parallel=False, verbose=False, use_cache=False)
    info = th['spatial_info']
    assert info.get('coupled_loop')
    taus = np.asarray(th['tau_grid'])
    dC = info['C_by_order'][1] - info['C_by_order'][0]   # the 1-loop piece

    # exact x-space tree correlator C_ij(x,τ; M) by q-quadrature
    M0 = np.array([[_MUA, g], [-h, _MUB]])
    Nz = np.diag([2 * _TA, 2 * _TB])
    qg = np.linspace(1e-3, 80.0, 8000)
    dq = qg[1] - qg[0]

    def C_x(Mm, x, tau, i, j):
        ii, jj = (i, j) if tau >= 0 else (j, i)
        vals = np.empty(qg.size)
        for iq, q in enumerate(qg):
            A = Mm + _D * q * q * np.eye(2)
            Sig = solve_continuous_lyapunov(A, Nz)
            vals[iq] = (expm(-A * abs(tau)) @ Sig)[ii, jj]
        return float(np.sum(np.cos(qg * x) * vals) * dq / np.pi)

    Caa0 = C_x(M0, 0.0, 0.0, 0, 0)
    Cbb0 = C_x(M0, 0.0, 0.0, 1, 1)
    dM = np.diag([3 * ga * Caa0, 3 * gb * Cbb0])
    eps = 1e-4
    for it, tau in enumerate(taus):
        for ix, x in enumerate(xs):
            oracle = (C_x(M0 + eps * dM, x, float(tau), 0, 1)
                      - C_x(M0 - eps * dM, x, float(tau), 0, 1)) / (2 * eps)
            assert abs(dC[it, ix] - oracle) < 0.03 * max(abs(oracle), 1e-3), \
                (f'(τ={tau}, x={x}): driver δC_ab={dC[it, ix]:.6f} vs '
                 f'Hartree oracle {oracle:.6f}')
    # genuine τ-asymmetry must survive (the old bug symmetrized it away)
    i_p = int(np.argmin(np.abs(taus - 0.7)))
    i_m = int(np.argmin(np.abs(taus + 0.7)))
    assert abs(dC[i_p, 0] - dC[i_m, 0]) > 0.2 * max(abs(dC[i_p, 0]), 1e-4), \
        f'δC_ab(±0.7) suspiciously symmetric: {dC[i_p, 0]} vs {dC[i_m, 0]}'


def test_coupled_nonlinear_loop_e2e_smoke():
    """Coupled (g,h≠0) cubic theory at max_ell=1: runs through compute_cumulants,
    real output, and the stabilizing cubic REDUCES C_aa(0,0) vs tree."""
    from api import compute_cumulants

    g, h, ga, gb = 0.4, 0.3, 0.3, 0.3
    model = _two_species(g, h, ga=ga, gb=gb)
    sg = np.array([0.0, 1.0])
    th = compute_cumulants(model=model, k=2, max_ell=1,
                           fundamental=_fund(g, h, ga, gb),
                           external_fields=[('a', 1), ('a', 1)],
                           tau_max=0.8, tau_step=0.4, spatial_grid=sg,
                           parallel=False, verbose=False, use_cache=False)
    info = th['spatial_info']
    assert info.get('coupled_loop') and info.get('n_live_diagrams', 0) > 0
    C = np.asarray(th['C_tau_x']).real
    assert np.all(np.isfinite(C))
    assert info['imag_frac'] < 1e-6
    Cbo = info['C_by_order']
    i0 = int(np.argmin(np.abs(np.asarray(th['tau_grid']))))
    c_tree = Cbo[0][i0, 0]
    c_1l = Cbo[1][i0, 0]
    assert c_tree > 0 and c_1l > 0
    assert c_1l < c_tree, ('stabilizing cubic must reduce the variance: '
                           f'tree {c_tree:.5f} -> 1-loop {c_1l:.5f}')

"""
tests/test_spatial_conventions.py
=================================
Spatial-engine conventions pinned against independent references (audit
wave 2, Oct 2026):

  * **k=2 ±τ completion = orbit–stabilizer rule.**  A k=2 single-field
    diagram contributes ``(Γ(τ)+Γ(−τ))/comp`` with ``comp =
    external_wick_compensation(td)``; the external-leg kinds do not decide it.
    From two loops on there are ``{R,R}``-external diagrams without a
    leaf-swap automorphism (comp = 1) that the old kind rule counted once.
    Pinned against the general-k mapping-sum driver and by τ-evenness.
  * **Lag convention** ``C_ab(τ) = ⟨φ_a(0) φ_b(τ)⟩`` on the coupled spatial
    route (tree), against a q-integrated Lyapunov oracle at both lag signs.
  * **total_C** of the spatial route is the returned grid (loops included),
    ``total_C_by_ell`` the per-order increments; checked against the closed
    form strict one-loop (tadpole) correlator at τ = 0.

Run:  sage -python -m pytest tests/test_spatial_conventions.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('sage.all')

from api.model import ModelBuilder, SpatialModelBuilder          # noqa: E402


# ── k=2 ±τ completion at two loops ─────────────────────────────────────────

def _cubic_quartic_records(max_ell=2):
    from sage.all import SR
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    b = (ModelBuilder('cq_conv', n_populations=0)
         .physical_field('p', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('DD', default=1.0, domain='positive')
         .parameter('g', default=0.1, domain='real')
         .parameter('lam', default=0.1, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
         .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)+g*p^2+lam*p^3) - T*pt^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=2 + 2 * max_ell + 1)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    _, pidx = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = _legs_to_phys_idx([('p', 1)] * 2, pidx)
    base = {SR.var('mu'): 1.0, SR.var('DD'): 1.0, SR.var('g'): 0.1,
            SR.var('lam'): 0.1, SR.var('T'): 1.0, SR.var('pstar1'): 0.0}
    by_ell = build_pipeline_records(ft, b, prop, ext, max_ell=max_ell, k=2,
                                    verbose=False)
    out = []
    for ell, recs in by_ell.items():
        for td, pre in recs:
            pv = float(SR(pre).subs(base))
            if pv != 0.0:
                out.append((ell, td, diagram_to_cstack(td), pv))
    return out


@pytest.fixture(scope='module')
def two_loop_records():
    return _cubic_quartic_records(max_ell=2)


def test_two_loop_rr_diagrams_without_leaf_swap_exist(two_loop_records):
    """The case the old leg-kind rule missed: {R,R} external legs, comp = 1."""
    from engine.diagrams.symmetry import external_wick_compensation
    bad = [r for r in two_loop_records if r[0] == 2
           and sorted(e.kind for e in r[2].edges if e.external) == ['R', 'R']
           and external_wick_compensation(r[1]) == 1]
    assert len(bad) >= 1
    # one loop: the kind rule and the orbit index coincide
    from engine.integration.spatial.full_integrator import _is_retarded_type
    for ell, td, dd, _pv in two_loop_records:
        if ell <= 1:
            assert (external_wick_compensation(td) == 1) == \
                _is_retarded_type(dd)


def test_k2_completion_equals_general_k_mapping_sum(two_loop_records):
    """diagram_correlator_x(comp) == diagram_correlator_pts over both
    field-respecting mappings ÷ comp (the Boltzmann-validated general-k
    architecture), for the comp = 1 {R,R} two-loop diagrams, at ±τ; and the
    single-field correlator is τ-even per diagram."""
    from engine.diagrams.symmetry import external_wick_compensation
    from engine.integration.spatial.full_integrator import (
        diagram_correlator_x, diagram_correlator_pts,
        field_respecting_mappings)
    picked = [r for r in two_loop_records if r[0] == 2
              and sorted(e.kind for e in r[2].edges if e.external) == ['R', 'R']
              and external_wick_compensation(r[1]) == 1][:2]
    assert picked
    xs = np.array([0.0, 0.7])
    kw = dict(n_t=5, n_s=5)
    for _ell, td, dd, pv in picked:
        comp = external_wick_compensation(td)
        maps = field_respecting_mappings(['p', 'p'], ['p', 'p'])
        assert len(maps) == 2
        for tau in (0.6, -0.6, 0.0):
            k2 = diagram_correlator_x(dd, pv, xs, tau, 1.0, 1.0, spatial_dim=1,
                                      comp=comp, **kw)
            xp = np.stack([np.zeros_like(xs), xs], axis=1)
            gk = diagram_correlator_pts(dd, pv, xp, np.array([0.0, tau]), 1.0,
                                        1.0, spatial_dim=1, mappings=maps,
                                        comp=comp, **kw)
            assert np.allclose(k2, gk, rtol=1e-12, atol=1e-300)
        cp = diagram_correlator_x(dd, pv, xs, 0.6, 1.0, 1.0, spatial_dim=1,
                                  comp=comp, **kw)
        cm = diagram_correlator_x(dd, pv, xs, -0.6, 1.0, 1.0, spatial_dim=1,
                                  comp=comp, **kw)
        assert np.allclose(cp, cm, rtol=1e-12, atol=1e-12)


def test_two_loop_correlator_needs_comp(two_loop_records):
    """Without comp the >= 2-loop k=2 completion refuses (the leg-kind
    fallback is exact only up to one loop)."""
    from engine.integration.spatial.full_integrator import diagram_correlator
    dd, pv = next((r[2], r[3]) for r in two_loop_records if r[0] == 2)
    with pytest.raises(ValueError, match='external_wick_compensation'):
        diagram_correlator(dd, pv, 0.0, 0.5, 1.0, 1.0, spatial_dim=1,
                           n_t=4, n_s=4)


# ── lag convention on the coupled route ─────────────────────────────────────

_MA, _MB, _DC, _TA, _TB, _G, _H = 1.1, 1.7, 0.6, 0.9, 0.5, 0.7, -0.45


def _coupled_linear():
    ea = '(Dt+ma-D*Laplacian)*a + g*b'
    eb = '(Dt+mb-D*Laplacian)*b + h*a'
    return (SpatialModelBuilder('conv_lin2')
            .physical_field('a', spatial_dim=1)
            .physical_field('b', spatial_dim=1)
            .parameter('ma', default=_MA, domain='positive')
            .parameter('mb', default=_MB, domain='positive')
            .parameter('D', default=_DC, domain='positive')
            .parameter('g', default=_G).parameter('h', default=_H)
            .parameter('Ta', default=_TA, domain='positive')
            .parameter('Tb', default=_TB, domain='positive')
            .set_action_text(f'at*({ea}) + bt*({eb}) - Ta*at^2 - Tb*bt^2')
            .equation(lhs=ea, rhs='0').equation(lhs=eb, rhs='0')
            .boundary('infinite').initial('stationary').build())


def _lyapunov_cross_x(x, tau, i, j):
    """⟨φ_i(0,0) φ_j(x,τ)⟩ by q-quadrature of Σ(q) e^{−A(q)ᵀτ} (τ ≥ 0) /
    e^{−A(q)|τ|} Σ(q) transposed (τ < 0); independent of the package."""
    from scipy.integrate import quad
    from scipy.linalg import expm, solve_continuous_lyapunov
    A0 = np.array([[_MA, _G], [_H, _MB]])
    N = np.diag([2 * _TA, 2 * _TB])

    def f(q):
        A = A0 + _DC * q * q * np.eye(2)
        S = solve_continuous_lyapunov(A, N)
        M = S @ expm(-A.T * tau) if tau >= 0 else expm(-A * abs(tau)) @ S
        return M[i, j] * math.cos(q * x) / math.pi
    return quad(f, 0.0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)[0]


@pytest.mark.parametrize('legs,i,j', [((('a', 1), ('b', 1)), 0, 1),
                                      ((('b', 1), ('a', 1)), 1, 0)])
def test_coupled_tree_lag_convention(legs, i, j):
    from api import compute_cumulants
    fund = {'ma': _MA, 'mb': _MB, 'D': _DC, 'g': _G, 'h': _H,
            'Ta': _TA, 'Tb': _TB}
    xs = np.array([0.0, 0.5])
    th = compute_cumulants(model=_coupled_linear(), k=2, max_ell=0,
                           fundamental=fund, external_fields=list(legs),
                           tau_grid=np.array([-0.6, 0.0, 0.6]),
                           spatial_grid=xs, parallel=False, verbose=False,
                           use_cache=False)
    C = np.real(np.asarray(th['C_tau_x']))
    for it, tau in enumerate(th['tau_grid']):
        for ix, x in enumerate(xs):
            ref = _lyapunov_cross_x(float(x), float(tau), i, j)
            assert C[it, ix] == pytest.approx(ref, rel=1e-10, abs=1e-12), \
                (legs, tau, x)
    # genuinely τ-asymmetric (else the test could not see a mirror)
    assert abs(C[0, 0] - C[2, 0]) > 0.1 * abs(C[2, 0])


# ── total_C carries the loops ───────────────────────────────────────────────

def test_spatial_total_c_is_the_returned_loop_grid():
    """d=1 φ⁴ (Allen–Cahn-type) at one loop: total_C(τ, x) == C_tau_x on the
    grid, total_C_by_ell = per-order increments, and at τ = 0 the result is
    the closed-form tadpole mass shift δμ ∂C₀/∂μ with δμ = 3λ C₀(0, 0)."""
    from api import compute_cumulants
    mu, D, T, lam = 1.7, 0.8, 0.6, 0.04
    m = (ModelBuilder('ac_tc', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=mu, domain='positive')
         .parameter('D', default=D, domain='positive')
         .parameter('T', default=T, domain='positive')
         .parameter('lam', default=lam)
         .equation(lhs='(Dt + mu - D*Laplacian)*phi + lam*phi^3', rhs='0')
         .set_action_text(
             'phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
         .boundary('infinite').initial('stationary').build())
    xs = np.array([0.0, 0.9])
    taus = np.array([0.0, 0.5])
    r = compute_cumulants(m, k=2, max_ell=1,
                          fundamental={'mu': mu, 'D': D, 'T': T, 'lam': lam},
                          external_fields=[('phi', 1), ('phi', 1)],
                          spatial_grid=xs, tau_grid=taus, verbose=False,
                          use_cache=False, parallel=False)
    C = np.asarray(r['C_tau_x'])
    for it, tau in enumerate(taus):
        for ix, x in enumerate(xs):
            assert r['total_C'](tau, x) == C[it, ix]
            parts = sum(f(tau, x) for f in r['total_C_by_ell'].values())
            assert parts == pytest.approx(C[it, ix], rel=1e-14)
    assert r['total_C'](0.0) == C[0, 0]
    assert sorted(r['total_C_by_ell']) == [0, 1]
    with pytest.raises(ValueError, match='not on the evaluated grid'):
        r['total_C'](0.25, 0.0)
    with pytest.raises(ValueError, match='finite'):
        r['total_C'](0.0, float('nan'))
    # closed form at τ=0: C₀(x) = T/(2√(μD)) e^{−|x|√(μ/D)}
    c00 = T / (2 * math.sqrt(mu * D))
    dmu = 3 * lam * c00
    for ix, x in enumerate(xs):
        k_ = math.sqrt(mu / D)
        c0 = c00 * math.exp(-abs(x) * k_)
        dc0 = c0 * (-0.5 / mu - abs(x) / (2 * math.sqrt(mu * D)))
        assert r['total_C_by_ell'][0](0.0, x).real == pytest.approx(c0, rel=1e-12)
        assert r['total_C_by_ell'][1](0.0, x).real == \
            pytest.approx(dmu * dc0, rel=1e-5)


def test_spatial_tree_total_c_off_grid_is_exact():
    """Single-field d=1 tree: off-grid points use the exact closed form."""
    from api import compute_cumulants
    mu, D, T = 1.3, 0.7, 0.9
    m = (ModelBuilder('ou_tc', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=mu, domain='positive')
         .parameter('D', default=D, domain='positive')
         .parameter('T', default=T, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2')
         .boundary('infinite').initial('stationary').build())
    r = compute_cumulants(m, k=2, max_ell=0,
                          fundamental={'mu': mu, 'D': D, 'T': T},
                          external_fields=[('phi', 1), ('phi', 1)],
                          spatial_grid=np.array([0.0]),
                          tau_grid=np.array([0.0]), verbose=False,
                          use_cache=False, parallel=False)
    x = 0.4
    ref = T / (2 * math.sqrt(mu * D)) * math.exp(-x * math.sqrt(mu / D))
    assert r['total_C'](0.0, x).real == pytest.approx(ref, rel=1e-12)

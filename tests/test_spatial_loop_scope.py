"""
tests/test_spatial_loop_scope.py
================================
Scope guards of the single-field spatial loop integrators (audit wave 2,
Oct 2026), each against an independent reference or a loud refusal:

  * a multi-field model with a DIAGONAL kinetic block whose loop edges carry
    another field (fields mixed only through the noise) routes to the coupled
    spectral-assignment driver (per-edge masses) instead of giving every edge
    the external field's (μ, D).  Exact reference: ``b`` is independent of
    ``a`` and ``C_aa`` is linear in a's noise, so ``C_aa = (1 + c⟨b²⟩) C⁰_aa``
    to all orders, ``⟨b²⟩ = T_b / (2√(μ_b D))``;
  * the k-point driver refuses such a model with the 'single-field' message
    that hands it to the coupled k-point driver;
  * periodic-boundary loops refuse (the loop integral is the infinite-domain
    one; only the tree carries the image sum).

Run:  sage -python -m pytest tests/test_spatial_loop_scope.py -q
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

_MUA, _MUB, _D, _C, _TA, _TB = 1.1, 3.0, 0.6, 0.4, 0.8, 0.9


def _noise_mixed():
    return (SpatialModelBuilder('noise_mixed_scope')
            .physical_field('a', spatial_dim=1).physical_field('b', spatial_dim=1)
            .parameter('mua', default=_MUA, domain='positive')
            .parameter('mub', default=_MUB, domain='positive')
            .parameter('D', default=_D, domain='positive')
            .parameter('c', default=_C)
            .parameter('Ta', default=_TA, domain='positive')
            .parameter('Tb', default=_TB, domain='positive')
            .set_action_text('at*((Dt+mua-D*Laplacian)*a) '
                             '+ bt*((Dt+mub-D*Laplacian)*b) '
                             '- Ta*at^2*(1 + c*b^2) - Tb*bt^2')
            .equation(lhs='(Dt+mua-D*Laplacian)*a', rhs='0')
            .equation(lhs='(Dt+mub-D*Laplacian)*b', rhs='0')
            .boundary('infinite').initial('stationary').build())


_FUND = {'mua': _MUA, 'mub': _MUB, 'D': _D, 'c': _C, 'Ta': _TA, 'Tb': _TB}


def test_loop_edges_of_another_field_get_their_own_mass():
    from api import compute_cumulants
    xs = np.array([0.0, 0.7])
    th = compute_cumulants(model=_noise_mixed(), fundamental=_FUND,
                           external_fields=[('a', 1), ('a', 1)], k=2,
                           max_ell=1, tau_grid=np.array([0.0, 0.4]),
                           spatial_grid=xs, parallel=False, verbose=False,
                           use_cache=False)
    info = th['spatial_info']
    assert info.get('coupled_loop')
    Cb = info['C_by_order']
    b2 = _TB / (2 * math.sqrt(_MUB * _D))
    for it in range(2):
        for ix in range(2):
            c0 = Cb[0][it, ix]
            # (τ=0, x=0.7): 7e-4 from the fixed default chamber grid (the
            # fixed-quadrature item of the same audit; 1e-5 at n_t=n_s=40)
            tol = 2e-3 if (it, ix) == (0, 1) else 1e-6
            assert Cb[1][it, ix] - c0 == pytest.approx(_C * b2 * c0, rel=tol)
    # τ=0, x=0 tree: T_a/(2√(μ_a D)) (the tree itself is untouched)
    assert Cb[0][0, 0] == pytest.approx(_TA / (2 * math.sqrt(_MUA * _D)),
                                        rel=1e-10)


def test_kpoint_refuses_foreign_field_edges_with_routing_message():
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.pipeline_bridge import compute_spatial_kpoint
    m = _noise_mixed()
    ft = FieldTheory(m, taylor_order=6)
    ft.expand()
    prop = build_propagator(ft, m, use_cache=False, verbose=False)
    with pytest.raises(NotImplementedError, match='single-field'):
        compute_spatial_kpoint(ft, m, prop,
                               dict(_FUND, astar1=0.0, bstar1=0.0),
                               [('a', 1)] * 4,
                               np.zeros((1, 3, 2)), max_ell=1)


def test_periodic_loops_refuse():
    from api import compute_cumulants
    m = (ModelBuilder('ac_pbc_scope', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .parameter('lam', default=0.1)
         .parameter('L', default=4.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi + lam*phi^3', rhs='0')
         .set_action_text(
             'phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
         .boundary('periodic', length='L').initial('stationary').build())
    fund = {'mu': 1.0, 'D': 1.0, 'T': 1.0, 'lam': 0.1, 'L': 4.0}
    with pytest.raises(NotImplementedError, match='periodic'):
        compute_cumulants(m, k=2, max_ell=1, fundamental=fund,
                          external_fields=[('phi', 1), ('phi', 1)],
                          spatial_grid=np.array([0.0]),
                          tau_grid=np.array([0.0]), verbose=False,
                          use_cache=False, parallel=False)
    # the tree on the ring stays available
    r = compute_cumulants(m, k=2, max_ell=0, fundamental=fund,
                          external_fields=[('phi', 1), ('phi', 1)],
                          spatial_grid=np.array([0.0]),
                          tau_grid=np.array([0.0]), verbose=False,
                          use_cache=False, parallel=False)
    # ring variance: T/(2√(μD)) coth(L√(μ/D)/2)
    assert np.real(r['C_tau_x'][0, 0]) == pytest.approx(
        0.5 / math.tanh(2.0), rel=1e-6)


# ── k-point driver: leg labels and liveness ─────────────────────────────────

def _quad_rd(T):
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    b = (ModelBuilder('rd_quad_scope', n_populations=0)
         .physical_field('p', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('DD', default=1.0, domain='positive')
         .parameter('g', default=0.1, domain='real')
         .parameter('T', default=T, domain='positive')
         .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
         .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)+g*p^2) - T*pt^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=5)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    return ft, b, prop, {'mu': 1.0, 'DD': 1.0, 'g': 0.1, 'T': T, 'pstar1': 0.0}


def test_kpoint_leg_labels_and_unit_scale():
    """k=3 tree: ('p', 1..3) labels give the same (nonzero) value as
    ('p', 1)×3 (they used to enumerate nothing → exact 0), and the value
    scales exactly as T² when T → 1e-8·T (an absolute 1e-14 prefactor cut
    used to return 0 there)."""
    from engine.integration.spatial.pipeline_bridge import compute_spatial_kpoint
    pts = np.array([[[0.3, 0.2], [-0.4, 0.5]]])
    ft, b, prop, npar = _quad_rd(1.0)
    v1, _ = compute_spatial_kpoint(ft, b, prop, npar, [('p', 1)] * 3, pts,
                                   max_ell=0)
    v2, _ = compute_spatial_kpoint(ft, b, prop, npar,
                                   [('p', 1), ('p', 2), ('p', 3)], pts,
                                   max_ell=0)
    assert abs(v1[0]) > 1e-6
    assert v2[0] == pytest.approx(v1[0], rel=1e-13)
    s = 1e-8
    fts, bs, props, nps = _quad_rd(s)
    vs, _ = compute_spatial_kpoint(fts, bs, props, nps, [('p', 1)] * 3, pts,
                                   max_ell=0)
    assert vs[0] == pytest.approx(s * s * v1[0], rel=1e-10)


def test_noise_extractors_refuse_unresolved_entries():
    """A noise coefficient that does not reduce to a number raises in both
    extractors (it used to be dropped to 0 silently, changing C)."""
    from api.compute import FieldTheory
    from engine.integration.spatial.heat_kernel import SpatialPropagatorError
    from engine.integration.spatial.spatial_correlator import (
        extract_noise_coefficients, extract_noise_matrix)
    m = _noise_mixed()
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    full = dict(_FUND, astar1=0.0, bstar1=0.0)
    N = extract_noise_matrix(ft, full)
    assert np.allclose(N, np.diag([2 * _TA, 2 * _TB]))
    for drop in ('Tb', 'bstar1'):
        part = {k_: v for k_, v in full.items() if k_ != drop}
        with pytest.raises(SpatialPropagatorError, match=drop):
            extract_noise_matrix(ft, part)
        with pytest.raises(SpatialPropagatorError, match=drop):
            extract_noise_coefficients(ft, part)

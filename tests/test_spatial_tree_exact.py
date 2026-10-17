"""
tests/test_spatial_tree_exact.py
================================
Spatial tree correlators without fixed grids (audit wave 2, Oct 2026), each
against an independent reference:

  * d = 2, 3 single-field tree = the exact heat-kernel time integral (it was a
    fixed q_cut = 40 / n_q = 2000 radial transform, 1 % ringing at unit
    parameters and O(1) errors for stiff masses): Bessel-K closed form at
    τ = 0 and an independent radial q-quadrature at τ ≠ 0, including a stiff
    mass far outside the old grid's range; the UV-divergent equal point
    x = τ = 0 refuses;
  * periodic boxes: converged image sums, checked against a direct mode sum;
  * coupled (scalar-D) tree at d = 2 and on a large d = 1 ring (it was a fixed
    1201-mode sum, 9 % low at L = 400) against Lyapunov references.

Run:  sage -python -m pytest tests/test_spatial_tree_exact.py -q
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


def _lin(d, mu, D, T, bc='infinite', L=None):
    b = (ModelBuilder(f'lin{d}', n_populations=0)
         .physical_field('phi', spatial_dim=d)
         .parameter('mu', default=mu, domain='positive')
         .parameter('D', default=D, domain='positive')
         .parameter('T', default=T, domain='positive'))
    if L is not None:
        b = b.parameter('L', default=L, domain='positive')
    b = (b.equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2'))
    b = b.boundary('periodic', length='L') if bc == 'periodic' else b.boundary('infinite')
    return b.initial('stationary').build()


def _radial_ref(d, mu, D, T, r, tau):
    """∫ d^dq/(2π)^d e^{iq·x} T/(μ+Dq²) e^{−(μ+Dq²)|τ|} by radial quadrature."""
    from scipy.integrate import quad
    from scipy.special import j0, spherical_jn

    def f(q):
        cq = T / (mu + D * q * q) * math.exp(-(mu + D * q * q) * abs(tau))
        if d == 2:
            return q * j0(q * r) * cq / (2 * math.pi)
        return q * q * spherical_jn(0, q * r) * cq / (2 * math.pi ** 2)
    return quad(f, 0.0, np.inf, limit=800, epsabs=1e-15, epsrel=1e-12)[0]


@pytest.mark.parametrize('d,mu,D,T', [(2, 1.3, 0.7, 0.9), (3, 0.8, 1.4, 1.1),
                                      (2, 400.0, 0.05, 1.0)])
def test_dge2_tree_is_exact(d, mu, D, T):
    from api import compute_cumulants
    rs = np.array([0.3, 1.1])
    taus = np.array([0.0, 0.4])
    r = compute_cumulants(_lin(d, mu, D, T), k=2, max_ell=0,
                          fundamental={'mu': mu, 'D': D, 'T': T},
                          external_fields=[('phi', 1), ('phi', 1)],
                          spatial_grid=rs, tau_grid=taus, verbose=False,
                          use_cache=False, parallel=False)
    C = np.real(np.asarray(r['C_tau_x']))
    from engine.integration.spatial.spatial_correlator import (
        free_correlator_static_closed_form as cf)
    for ix, x in enumerate(rs):
        ref0 = cf(x, mu, D, T, d)
        assert C[0, ix] == pytest.approx(ref0, rel=1e-11, abs=1e-300)
        ref1 = _radial_ref(d, mu, D, T, x, 0.4)
        assert C[1, ix] == pytest.approx(ref1, rel=1e-8, abs=1e-14)


def test_dge2_equal_point_refuses():
    from api import compute_cumulants
    from engine.integration.spatial.heat_kernel import SpatialPropagatorError
    with pytest.raises(SpatialPropagatorError, match='UV-divergent'):
        compute_cumulants(_lin(2, 1.0, 1.0, 1.0), k=2, max_ell=0,
                          fundamental={'mu': 1.0, 'D': 1.0, 'T': 1.0},
                          external_fields=[('phi', 1), ('phi', 1)],
                          spatial_grid=np.array([0.0]),
                          tau_grid=np.array([0.0]), verbose=False,
                          use_cache=False, parallel=False)


def test_d2_periodic_box_equals_mode_sum():
    """Image sum (production) vs a direct lattice mode sum at τ > 0 (where the
    mode sum converges as e^{−Dk²τ})."""
    from api import compute_cumulants
    mu, D, T, L, tau = 0.9, 0.6, 1.0, 3.0, 0.3
    xs = np.array([0.4, 1.2])
    r = compute_cumulants(_lin(2, mu, D, T, 'periodic', L), k=2, max_ell=0,
                          fundamental={'mu': mu, 'D': D, 'T': T, 'L': L},
                          external_fields=[('phi', 1), ('phi', 1)],
                          spatial_grid=xs, tau_grid=np.array([tau]),
                          verbose=False, use_cache=False, parallel=False)
    C = np.real(np.asarray(r['C_tau_x']))[0]
    n = np.arange(-60, 61)
    k1, k2 = np.meshgrid(2 * np.pi * n / L, 2 * np.pi * n / L, indexing='ij')
    m = mu + D * (k1 ** 2 + k2 ** 2)
    cq = T / m * np.exp(-m * tau)
    for ix, x in enumerate(xs):
        ref = float(np.sum(np.cos(k1 * x) * cq)) / L ** 2
        assert C[ix] == pytest.approx(ref, rel=1e-11)


_MA, _MB, _DC, _TA, _TB, _G, _H = 1.1, 1.7, 0.6, 0.9, 0.5, 0.7, -0.45


def _coupled(d, bc='infinite', L=None):
    ea = '(Dt+ma-D*Laplacian)*a + g*b'
    eb = '(Dt+mb-D*Laplacian)*b + h*a'
    b = (SpatialModelBuilder('tree_exact_c2')
         .physical_field('a', spatial_dim=d)
         .physical_field('b', spatial_dim=d)
         .parameter('ma', default=_MA, domain='positive')
         .parameter('mb', default=_MB, domain='positive')
         .parameter('D', default=_DC, domain='positive')
         .parameter('g', default=_G).parameter('h', default=_H)
         .parameter('Ta', default=_TA, domain='positive')
         .parameter('Tb', default=_TB, domain='positive'))
    if L is not None:
        b = b.parameter('L', default=L, domain='positive')
    b = (b.set_action_text(f'at*({ea}) + bt*({eb}) - Ta*at^2 - Tb*bt^2')
         .equation(lhs=ea, rhs='0').equation(lhs=eb, rhs='0'))
    b = b.boundary('periodic', length='L') if bc == 'periodic' else b.boundary('infinite')
    return b.initial('stationary').build()


def _lyap_q(qsq, tau):
    """⟨φ(0)φ(τ)ᵀ⟩ in q-space: Σ e^{−Aᵀτ} (τ≥0), e^{−A|τ|}Σ (τ<0)."""
    from scipy.linalg import expm, solve_continuous_lyapunov
    A = np.array([[_MA, _G], [_H, _MB]]) + _DC * qsq * np.eye(2)
    S = solve_continuous_lyapunov(A, np.diag([2 * _TA, 2 * _TB]))
    return S @ expm(-A.T * tau) if tau >= 0 else expm(-A * abs(tau)) @ S


_FUND = {'ma': _MA, 'mb': _MB, 'D': _DC, 'g': _G, 'h': _H, 'Ta': _TA, 'Tb': _TB}


def test_coupled_tree_d2_exact():
    from scipy.integrate import quad
    from scipy.special import j0
    from api import compute_cumulants
    xs = np.array([0.5])
    taus = np.array([-0.5, 0.0, 0.5])
    r = compute_cumulants(model=_coupled(2), k=2, max_ell=0, fundamental=_FUND,
                          external_fields=[('a', 1), ('b', 1)], tau_grid=taus,
                          spatial_grid=xs, parallel=False, verbose=False,
                          use_cache=False)
    C = np.real(np.asarray(r['C_tau_x']))
    for it, tau in enumerate(taus):
        ref = quad(lambda q: q * j0(q * 0.5) * _lyap_q(q * q, tau)[0, 1]
                   / (2 * math.pi), 0, np.inf, limit=800, epsabs=1e-14,
                   epsrel=1e-11)[0]
        assert C[it, 0] == pytest.approx(ref, rel=1e-8)


def test_coupled_tree_large_ring_d1():
    """L = 400 ring: exact (image sum) vs the direct mode sum over enough
    modes; the fixed 1201-mode sum was 9 % low here."""
    from api import compute_cumulants
    L, tau = 400.0, 0.2
    xs = np.array([0.0, 1.0])
    r = compute_cumulants(model=_coupled(1, 'periodic', L), k=2, max_ell=0,
                          fundamental=dict(_FUND, L=L),
                          external_fields=[('a', 1), ('a', 1)],
                          tau_grid=np.array([tau]), spatial_grid=xs,
                          parallel=False, verbose=False, use_cache=False)
    C = np.real(np.asarray(r['C_tau_x']))[0]
    n = np.arange(-20000, 20001)
    k = 2 * np.pi * n / L
    caa = np.array([_lyap_q(kk * kk, tau)[0, 0] for kk in k])
    for ix, x in enumerate(xs):
        ref = float(np.sum(np.cos(k * x) * caa)) / L
        assert C[ix] == pytest.approx(ref, rel=1e-9)

"""
tests/test_coupled_tree_e2e.py
==============================
Dyson step 3b: a COUPLED multi-field scalar-diffusion theory flows end-to-end to
C_ij(x,τ) via the dedicated coupled tree-level driver
(``pipeline_bridge.compute_coupled_tree_correlator``), which reads ``prop['K_ft']``
(built even when the diagonal heat-kernel block is rejected), extracts M/𝒟/N, and
FTs the spectral-Lyapunov 2-point.

Validation:
  * **chain vs trusted oracle** — a DECOUPLED 2-field theory (g=h=0) run through
    the coupled driver reproduces ``free_two_point`` (the analytic diagonal
    correlator) → validates M/𝒟/N extraction + Lyapunov + numerical FT together;
  * **coupling extraction** — a coupled theory's off-diagonal reaction M is read
    correctly and the cross-correlation C_ab is nonzero and even in x;
  * **noise matrix** — extract_noise_matrix recovers the diagonal (= 2·κ) and the
    cross noise.

Run:  sage -python -m pytest tests/test_coupled_tree_e2e.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.core.field_theory import FieldTheory                 # noqa: E402
from pipeline._propagator import build_propagator               # noqa: E402
from pipeline.theory import SpatialTheoryBuilder                # noqa: E402
from msrjd.integration.spatial.spatial_correlator import (      # noqa: E402
    free_two_point, extract_noise_matrix,
)
from msrjd.integration.spatial.pipeline_bridge import (         # noqa: E402
    compute_coupled_tree_correlator,
)

_MUA, _MUB, _D, _TA, _TB = 1.5, 1.2, 0.8, 1.0, 0.7


def _two_species(g, h, *, cross_noise=0.0):
    b = (SpatialTheoryBuilder('coupled2')
         .physical_field('a', spatial_dim=1)
         .physical_field('b', spatial_dim=1)
         .parameter('mua', default=_MUA, domain='positive')
         .parameter('mub', default=_MUB, domain='positive')
         .parameter('D', default=_D, domain='positive')
         .parameter('g', default=g)
         .parameter('h', default=h)
         .parameter('Ta', default=_TA, domain='positive')
         .parameter('Tb', default=_TB, domain='positive')
         .parameter('Tab', default=cross_noise))
    action = ('at*((Dt+mua-D*Laplacian)*a + g*b) '
              '+ bt*((Dt+mub-D*Laplacian)*b - h*a) '
              '- Ta*at^2 - Tb*bt^2 - Tab*at*bt')
    return (b.set_action_text(action)
            .equation(lhs='(Dt+mua-D*Laplacian)*a + g*b', rhs='0')
            .equation(lhs='(Dt+mub-D*Laplacian)*b - h*a', rhs='0')
            .boundary('infinite').initial('stationary').build())


def _setup(model):
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    return ft, prop


def _fund(g, h, cross_noise=0.0):
    return {'mua': _MUA, 'mub': _MUB, 'D': _D, 'g': g, 'h': h,
            'Ta': _TA, 'Tb': _TB, 'Tab': cross_noise}


def test_decoupled_matches_free_two_point():
    """g=h=0 ⇒ the coupled driver's C_aa(x,τ) == free_two_point(μa, D, Ta)."""
    model = _two_species(0.0, 0.0)
    ft, prop = _setup(model)
    taus = np.array([0.3, 0.8])
    xs = np.array([0.0, 0.7, 1.5])
    C_aa, info = compute_coupled_tree_correlator(
        ft, model, prop, _fund(0.0, 0.0), [('a', 1), ('a', 1)], taus, xs,
        q_cut=120.0, n_q=12000)
    for it, tau in enumerate(taus):
        for ix, x in enumerate(xs):
            ref = free_two_point(_MUA, _D, _TA, float(x), float(tau)).real
            assert abs(C_aa[it, ix].real - ref) < 5e-3, \
                f'(τ={tau},x={x}): driver {C_aa[it,ix].real:.5f} vs free {ref:.5f}'
    # diagonal extraction (order-robust)
    assert sorted(np.round(np.diag(info['M']), 6)) == sorted([_MUA, _MUB])
    assert sorted(np.round(np.diag(info['N']), 6)) == sorted([2 * _TA, 2 * _TB])
    assert info['D0'] == pytest.approx(_D)
    assert np.allclose(info['Dhat'], 0.0)                       # scalar diffusion


def test_coupled_extraction_and_cross_correlation():
    g, h = 0.4, 0.3
    model = _two_species(g, h)
    ft, prop = _setup(model)
    taus = np.array([0.0, 0.5])
    xs = np.array([-1.0, 0.0, 1.0])
    C_ab, info = compute_coupled_tree_correlator(
        ft, model, prop, _fund(g, h), [('a', 1), ('b', 1)], taus, xs,
        q_cut=120.0, n_q=12000)
    M = info['M']
    assert sorted(np.round(np.diag(M), 6)) == sorted([_MUA, _MUB])
    offs = sorted([round(abs(M[0, 1]), 6), round(abs(M[1, 0]), 6)])
    assert offs == sorted([round(g, 6), round(h, 6)])           # off-diagonal reaction
    assert np.max(np.abs(C_ab.real)) > 1e-3                     # genuine cross-correlation
    for it in range(len(taus)):                                 # even in x
        assert C_ab[it, 0].real == pytest.approx(C_ab[it, 2].real, abs=2e-3)


def test_coupled_routes_through_compute_cumulants():
    """compute_cumulants (public API, max_ell=0) on a coupled theory routes to the
    coupled driver and returns C(x,τ) matching a direct driver call."""
    from pipeline import compute_cumulants
    g, h = 0.4, 0.3
    model = _two_species(g, h)
    sg = np.array([0.0, 0.5, 1.0])
    th = compute_cumulants(
        model=model, k=2, max_ell=0, fundamental=_fund(g, h),
        external_fields=[('a', 1), ('a', 1)], tau_max=1.0, tau_step=0.5,
        spatial_grid=sg, parallel=False, verbose=False, use_cache=False)
    C = np.asarray(th['C_tau_x']).real
    ft, prop = _setup(model)
    Cd, info = compute_coupled_tree_correlator(
        ft, model, prop, _fund(g, h), [('da', 1), ('da', 1)],
        np.asarray(th['tau_grid']), sg)
    assert info['coupled'] is True
    assert np.allclose(C, Cd.real, atol=1e-6)               # routed to the driver
    i0 = int(np.argmin(np.abs(np.asarray(th['tau_grid']))))
    assert C[i0, 0] > 0                                     # C_aa(0,0) is a variance


def test_noise_matrix_diagonal_and_cross():
    model = _two_species(0.4, 0.3, cross_noise=0.5)
    ft, _ = _setup(model)
    N = extract_noise_matrix(ft, _fund(0.4, 0.3, cross_noise=0.5))
    assert sorted(np.round(np.diag(N), 6)) == sorted([2 * _TA, 2 * _TB])
    # cross noise: action −Tab·ã b̃ ⇒ N_ij = −coeff = +Tab
    off = [N[0, 1], N[1, 0]]
    assert off[0] == pytest.approx(off[1])                      # symmetric
    assert abs(off[0]) == pytest.approx(0.5, abs=1e-6)

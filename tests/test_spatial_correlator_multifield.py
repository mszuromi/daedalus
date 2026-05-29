"""
tests/test_spatial_correlator_multifield.py
===========================================
Additional spatial-correlator coverage on NEW theories, beyond
``test_spatial_correlator.py``:

  * a 2-field DECOUPLED spatial theory — each field's auto-correlator
    must match its own free closed form (exercises the multi-field
    Tier-1 / block-diagonal heat-kernel propagator path)
  * single-field linear diffusion at several (μ, D, T) — parametric
    robustness of the closed form C(x,0) = (T/2√(μD)) e^{-|x|/ξ}
  * Allen-Cahn at tree level (max_ell=0) must equal the free linear
    closed form and be INDEPENDENT of λ (the cubic is a loop effect)

Run:  sage -python -m pytest tests/test_spatial_correlator_multifield.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.theory import TheoryBuilder
from pipeline import compute_cumulants


def _closed_equal_time(x, mu, D, T):
    xi = math.sqrt(D / mu)
    return T / (2 * math.sqrt(mu * D)) * np.exp(-np.abs(x) / xi)


def _two_field_model():
    return (
        TheoryBuilder('two-field decoupled spatial', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .physical_field('psi', spatial_dim=1)
        .parameter('mu1', default=1.0, domain='positive')
        .parameter('D1', default=1.0, domain='positive')
        .parameter('mu2', default=2.0, domain='positive')
        .parameter('D2', default=0.5, domain='positive')
        .parameter('T1', default=1.0, domain='positive')
        .parameter('T2', default=1.5, domain='positive')
        .set_action_text(
            'phit*((Dt+mu1-D1*Laplacian)*phi) '
            '+ psit*((Dt+mu2-D2*Laplacian)*psi) '
            '- T1*phit^2 - T2*psit^2')
        .equation(lhs='(Dt+mu1-D1*Laplacian)*phi', rhs='0')
        .equation(lhs='(Dt+mu2-D2*Laplacian)*psi', rhs='0')
        .boundary('infinite').initial('stationary').build())


_FUND2 = {'mu1': 1.0, 'D1': 1.0, 'mu2': 2.0, 'D2': 0.5, 'T1': 1.0, 'T2': 1.5}


@pytest.mark.parametrize('field,mu,D,T', [
    ('phi', 1.0, 1.0, 1.0),
    ('psi', 2.0, 0.5, 1.5),
])
def test_two_field_decoupled_each_matches_closed_form(field, mu, D, T):
    """Each field of a decoupled 2-field spatial theory reproduces its
    own free equal-time correlator (block-diagonal heat-kernel path)."""
    model = _two_field_model()
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    th = compute_cumulants(
        model=model, k=2, max_ell=0, fundamental=_FUND2,
        external_fields=[(field, 1), (field, 2)],
        tau_max=1.0, tau_step=1.0, spatial_grid=xs,
        parallel=False, verbose=False, use_cache=True)
    it0 = int(np.argmin(np.abs(th['tau_grid'])))
    C = np.asarray(th['C_tau_x']).real[it0]
    np.testing.assert_allclose(C, _closed_equal_time(xs, mu, D, T),
                               rtol=1e-9, atol=1e-12)


def _linear_model():
    return (
        TheoryBuilder('linear diffusion param', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text('phit*((Dt+mu-D*Laplacian)*phi) - T*phit^2')
        .equation(lhs='(Dt+mu-D*Laplacian)*phi', rhs='0')
        .boundary('infinite').initial('stationary').build())


@pytest.mark.parametrize('mu,D,T', [
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),    # shorter ξ
    (1.0, 4.0, 1.0),    # longer ξ
    (0.5, 2.0, 3.0),    # all three varied
])
def test_linear_diffusion_varied_params(mu, D, T):
    model = _linear_model()
    xs = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    th = compute_cumulants(
        model=model, k=2, max_ell=0, fundamental={'mu': mu, 'D': D, 'T': T},
        external_fields=[('phi', 1), ('phi', 2)],
        tau_max=1.0, tau_step=1.0, spatial_grid=xs,
        parallel=False, verbose=False, use_cache=True)
    it0 = int(np.argmin(np.abs(th['tau_grid'])))
    C = np.asarray(th['C_tau_x']).real[it0]
    np.testing.assert_allclose(C, _closed_equal_time(xs, mu, D, T),
                               rtol=1e-9, atol=1e-12)


def _allen_cahn_model():
    return (
        TheoryBuilder('allen-cahn tree test', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .parameter('lam', default=0.3, domain='positive')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt+mu-D*Laplacian)*phi + lam*phi^3) - T*phit^2')
        .equation(lhs='(Dt+mu-D*Laplacian)*phi', rhs='-lam*phi^3')
        .stability_analysis(True)
        .boundary('infinite').initial('stationary').build())


@pytest.mark.parametrize('lam', [0.0, 0.3, 1.0])
def test_allen_cahn_tree_is_lambda_independent_free_form(lam):
    """At tree level (max_ell=0) the λφ³ vertex contributes nothing, so
    C(x,0) must equal the free linear closed form for ANY λ."""
    model = _allen_cahn_model()
    xs = np.array([0.0, 1.0, 2.0])
    th = compute_cumulants(
        model=model, k=2, max_ell=0,
        fundamental={'mu': 1.0, 'D': 1.0, 'lam': lam, 'T': 1.0},
        external_fields=[('phi', 1), ('phi', 2)],
        tau_max=1.0, tau_step=1.0, spatial_grid=xs,
        parallel=False, verbose=False, use_cache=True)
    it0 = int(np.argmin(np.abs(th['tau_grid'])))
    C = np.asarray(th['C_tau_x']).real[it0]
    np.testing.assert_allclose(C, _closed_equal_time(xs, 1.0, 1.0, 1.0),
                               rtol=1e-9, atol=1e-12)

"""
tests/test_spatial_correlator_multifield.py
===========================================
Additional spatial-correlator coverage on NEW models, beyond
``test_spatial_correlator.py``:

  * a 2-field DECOUPLED spatial model — each field's auto-correlator
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

from api.model import ModelBuilder
from api import compute_cumulants


def _closed_equal_time(x, mu, D, T):
    xi = math.sqrt(D / mu)
    return T / (2 * math.sqrt(mu * D)) * np.exp(-np.abs(x) / xi)


def _two_field_model():
    return (
        ModelBuilder('two-field decoupled spatial', n_populations=0)
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
    """Each field of a decoupled 2-field spatial model reproduces its
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
        ModelBuilder('linear diffusion param', n_populations=0)
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
        ModelBuilder('allen-cahn tree test', n_populations=0)
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


def test_coupled_multifield_scalar_diffusion_supported():
    """A coupled (off-diagonal) multi-field theory with EQUAL (scalar) diffusion
    is now handled by the spectral-Lyapunov coupled driver (Dyson 3b):
    compute_cumulants returns a finite C(x,τ) instead of raising."""
    model = (
        ModelBuilder('coupled spatial (scalar D)', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .physical_field('psi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .parameter('g', default=0.3, domain='real')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt+mu-D*Laplacian)*phi + g*psi) '
            '+ psit*((Dt+mu-D*Laplacian)*psi + g*phi) '
            '- T*phit^2 - T*psit^2')
        .equation(lhs='(Dt+mu-D*Laplacian)*phi', rhs='-g*psi')
        .equation(lhs='(Dt+mu-D*Laplacian)*psi', rhs='-g*phi')
        .boundary('infinite').initial('stationary').build())
    th = compute_cumulants(
        model=model, k=2, max_ell=0,
        fundamental={'mu': 1.0, 'D': 1.0, 'g': 0.3, 'T': 1.0},
        external_fields=[('phi', 1), ('phi', 2)],
        tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0, 1.0]),
        parallel=False, verbose=False, use_cache=False)
    C = np.asarray(th['C_tau_x']).real
    assert np.all(np.isfinite(C))
    i0 = int(np.argmin(np.abs(np.asarray(th['tau_grid']))))
    assert C[i0, 0] > 0                                # C_phiphi(0,0) is a variance


def test_coupled_multifield_unequal_diffusion_raises_clean():
    """Coupled WITH UNEQUAL diffusion (𝒟̂≠0) still needs the Dyson series, so
    compute_cumulants raises a CLEAR NotImplementedError."""
    model = (
        ModelBuilder('coupled spatial (unequal D)', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .physical_field('psi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D1', default=1.0, domain='positive')
        .parameter('D2', default=0.5, domain='positive')
        .parameter('g', default=0.3, domain='real')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt+mu-D1*Laplacian)*phi + g*psi) '
            '+ psit*((Dt+mu-D2*Laplacian)*psi + g*phi) '
            '- T*phit^2 - T*psit^2')
        .equation(lhs='(Dt+mu-D1*Laplacian)*phi', rhs='-g*psi')
        .equation(lhs='(Dt+mu-D2*Laplacian)*psi', rhs='-g*phi')
        .boundary('infinite').initial('stationary').build())
    with pytest.raises(NotImplementedError, match='scalar-diffusion'):
        compute_cumulants(
            model=model, k=2, max_ell=0,
            fundamental={'mu': 1.0, 'D1': 1.0, 'D2': 0.5, 'g': 0.3, 'T': 1.0},
            external_fields=[('phi', 1), ('phi', 2)],
            tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0, 1.0]),
            parallel=False, verbose=False, use_cache=False)


def _mixed_dim_model():
    """A spatial field ``phi`` (dim=1) coexisting with a time-only,
    spatially-averaged auxiliary ``m`` (dim=0), DECOUPLED.  Decision D1
    of the spatial design doc explicitly allows mixing dim=0 with
    dim>=1 fields in v1."""
    return (
        ModelBuilder('mixed dim0/dim1', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .physical_field('m', spatial_dim=0)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .parameter('a', default=2.0, domain='positive')
        .parameter('T', default=1.0, domain='positive')
        .parameter('Tm', default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt+mu-D*Laplacian)*phi) + mt*((Dt+a)*m) '
            '- T*phit^2 - Tm*mt^2')
        .equation(lhs='(Dt+mu-D*Laplacian)*phi', rhs='0')
        .equation(lhs='(Dt+a)*m', rhs='0')
        .boundary('infinite').initial('stationary').build())


_FUND_MIXED = {'mu': 1.0, 'D': 1.0, 'a': 2.0, 'T': 1.0, 'Tm': 1.0}


def test_mixed_dim0_dim1_spatial_field_matches_closed_form():
    """In a model mixing a spatial field (dim=1) with a time-only
    auxiliary (dim=0), the SPATIAL field's correlator still reproduces
    its free closed form — the dim=0 block must not perturb it."""
    model = _mixed_dim_model()
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    th = compute_cumulants(
        model=model, k=2, max_ell=0, fundamental=_FUND_MIXED,
        external_fields=[('phi', 1), ('phi', 2)],
        tau_max=1.0, tau_step=1.0, spatial_grid=xs,
        parallel=False, verbose=False, use_cache=True)
    it0 = int(np.argmin(np.abs(th['tau_grid'])))
    C = np.asarray(th['C_tau_x']).real[it0]
    np.testing.assert_allclose(C, _closed_equal_time(xs, 1.0, 1.0, 1.0),
                               rtol=1e-9, atol=1e-12)


def test_mixed_dim0_field_spatial_request_raises_clean():
    """Requesting the SPATIAL correlator of a time-only (dim=0) field
    must raise a CLEAR SpatialPropagatorError (its heat-kernel block has
    B=0 diffusion), not a bare ZeroDivisionError from 1/sqrt(4*pi*B)."""
    from engine.integration.spatial.heat_kernel import SpatialPropagatorError
    model = _mixed_dim_model()
    with pytest.raises(SpatialPropagatorError, match='time-only'):
        compute_cumulants(
            model=model, k=2, max_ell=0, fundamental=_FUND_MIXED,
            external_fields=[('m', 1), ('m', 2)],
            tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0, 1.0]),
            parallel=False, verbose=False, use_cache=False)


def test_higher_derivative_k4_raises_clean():
    """A higher-derivative (Laplacian² → k⁴) spatial model is not Tier-1
    (the heat kernel needs dispersion λ = -(A + B·k²)).  It must raise a
    clear NotImplementedError whose recorded reason names the offending
    k-power — NOT a misleading 'coupled multi-field' message, since a
    single-field k⁴ model has no field coupling at all."""
    model = (
        ModelBuilder('k4 dispersion', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .parameter('E', default=0.5, domain='positive')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt + mu - D*Laplacian + E*Laplacian^2)*phi) - T*phit^2')
        .equation(lhs='(Dt + mu - D*Laplacian + E*Laplacian^2)*phi', rhs='0')
        .boundary('infinite').initial('stationary').build())
    with pytest.raises(NotImplementedError, match='higher-derivative'):
        compute_cumulants(
            model=model, k=2, max_ell=0,
            fundamental={'mu': 1.0, 'D': 1.0, 'E': 0.5, 'T': 1.0},
            external_fields=[('phi', 1), ('phi', 2)],
            tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0, 1.0]),
            parallel=False, verbose=False, use_cache=False)


def test_negative_diffusion_raises_anti_diffusive_error():
    """A wrong-sign Laplacian ("+ D*Laplacian") makes B<0 (anti-diffusion).
    The error must identify it as anti-diffusive / ill-posed and hint at the
    sign — NOT mislabel it as a zero-diffusion time-only (dim=0) field."""
    from engine.integration.spatial.heat_kernel import SpatialPropagatorError
    model = (
        ModelBuilder('neg diffusion', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text('phit*((Dt + mu + D*Laplacian)*phi) - T*phit^2')
        .equation(lhs='(Dt + mu + D*Laplacian)*phi', rhs='0')
        .boundary('infinite').initial('stationary').build())
    with pytest.raises(SpatialPropagatorError, match='anti-diffusive'):
        compute_cumulants(
            model=model, k=2, max_ell=0,
            fundamental={'mu': 1.0, 'D': 1.0, 'T': 1.0},
            external_fields=[('phi', 1), ('phi', 2)],
            tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0, 1.0]),
            parallel=False, verbose=False, use_cache=False)


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

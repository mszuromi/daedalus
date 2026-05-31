"""
tests/test_spatial_correlator.py
================================
Phase 5 (tree-level) + Phase 6 acceptance: the spatial two-point
correlator C(x, τ) through ``compute_cumulants(spatial_grid=...)``,
validated against closed forms on several models.

Closed forms (free / tree-level, d=1):
  * infinite domain:  C(x, 0) = (T / 2√(μD)) e^{-|x|/ξ},  ξ = √(D/μ)
  * periodic (ring L): C(0, 0) = (T / 2√(μD)) coth((L/2)√(μ/D))
    (image sum == exact periodic correlator, by Poisson summation)
  * two-time:  matches direct 2-D (t_v, x_v) quadrature

Models exercised: Allen-Cahn infinite, Allen-Cahn periodic, and the
linear Edwards-Wilkinson interface (distinct params, no φ³).

Run::

    sage -python -m pytest tests/test_spatial_correlator.py -v
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_REPO = os.path.join(os.path.dirname(__file__), '..')


def _load(theory_file):
    path = os.path.join(_REPO, 'theories', theory_file)
    spec = importlib.util.spec_from_file_location('m', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _compute(model, fundamental, spatial_grid, tau_max=2.0, tau_step=1.0,
             max_ell=0, k=2):
    from pipeline import compute_cumulants
    return compute_cumulants(
        model=model, k=k, max_ell=max_ell, fundamental=fundamental,
        external_fields=[('phi', 1), ('phi', 2)],
        tau_max=tau_max, tau_step=tau_step,
        spatial_grid=np.asarray(spatial_grid, dtype=float),
        parallel=False, verbose=False, use_cache=True)


# ── Infinite domain: equal-time + two-time ────────────────────────
def test_allen_cahn_infinite_equal_time():
    model = _load('allen_cahn_1d_subcritical_infinite.theory.py')
    mu = D = T = 1.0
    xi = math.sqrt(D / mu)
    xs = np.linspace(-3, 3, 13)
    th = _compute(model, {'mu': mu, 'D': D, 'lam': 0.1, 'T': T}, xs)
    assert th['C_tau_x'].shape == (len(th['tau_grid']), len(xs))
    it0 = int(np.argmin(np.abs(th['tau_grid'])))
    for ix, x in enumerate(xs):
        c = th['C_tau_x'][it0, ix].real
        analytic = T / (2 * math.sqrt(mu * D)) * math.exp(-abs(x) / xi)
        assert abs(c - analytic) / max(abs(analytic), 1e-300) < 1e-10


def test_edwards_wilkinson_equal_time():
    """Linear theory, distinct params (μ=0.5, D=2)."""
    model = _load('edwards_wilkinson_1d.theory.py')
    mu, D, T = 0.5, 2.0, 1.0
    xi = math.sqrt(D / mu)
    xs = np.linspace(-6, 6, 13)
    th = _compute(model, {'mu': mu, 'D': D, 'T': T}, xs)
    it0 = int(np.argmin(np.abs(th['tau_grid'])))
    for ix, x in enumerate(xs):
        c = th['C_tau_x'][it0, ix].real
        analytic = T / (2 * math.sqrt(mu * D)) * math.exp(-abs(x) / xi)
        assert abs(c - analytic) / max(abs(analytic), 1e-300) < 1e-10


def test_two_time_vs_direct_quadrature():
    from scipy import integrate
    model = _load('allen_cahn_1d_subcritical_infinite.theory.py')
    mu = D = T = 1.0

    def Gh(t, x):
        return 0.0 if t <= 0 else (
            1 / math.sqrt(4 * math.pi * D * t)
            * math.exp(-x * x / (4 * D * t) - mu * t))

    def C_direct(x, tau):
        def inner(tv):
            f = lambda xv: Gh(abs(tau) - tv, x - xv) * Gh(-tv, -xv)
            v, _ = integrate.quad(f, -50, 50, limit=200, epsabs=1e-12)
            return v
        val, _ = integrate.quad(inner, -50, 0.0, limit=200, epsabs=1e-12)
        return 2 * T * val

    xs = np.array([0.0, 1.0, 2.0])
    th = _compute(model, {'mu': mu, 'D': D, 'lam': 0.1, 'T': T}, xs,
                  tau_max=2.0, tau_step=1.0)
    for it, tau in enumerate(th['tau_grid']):
        if tau <= 0:
            continue
        for ix, x in enumerate(xs):
            c = th['C_tau_x'][it, ix].real
            ref = C_direct(x, tau)
            assert abs(c - ref) / max(abs(ref), 1e-300) < 1e-8


# ── Periodic: equal-time variance vs exact coth, + → infinite ─────
def test_pbc_variance_matches_coth():
    model = _load('allen_cahn_1d_subcritical_pbc.theory.py')
    mu = D = T = 1.0
    for L in [4.0, 10.0, 20.0]:
        th = _compute(model, {'mu': mu, 'D': D, 'lam': 0.1, 'T': T, 'L': L},
                      np.array([0.0]), tau_max=1.0, tau_step=1.0)
        it0 = int(np.argmin(np.abs(th['tau_grid'])))
        var = th['C_tau_x'][it0, 0].real
        # Exact periodic equal-time variance.
        exact = (T / (2 * math.sqrt(mu * D))
                 * (1.0 / math.tanh((L / 2) * math.sqrt(mu / D))))
        assert abs(var - exact) / exact < 1e-8, f'L={L}: {var} vs {exact}'


def test_pbc_approaches_infinite():
    model = _load('allen_cahn_1d_subcritical_pbc.theory.py')
    mu = D = T = 1.0
    inf_var = T / (2 * math.sqrt(mu * D))
    prev = None
    for L in [4.0, 8.0, 20.0]:
        th = _compute(model, {'mu': mu, 'D': D, 'lam': 0.1, 'T': T, 'L': L},
                      np.array([0.0]), tau_max=1.0, tau_step=1.0)
        it0 = int(np.argmin(np.abs(th['tau_grid'])))
        diff = abs(th['C_tau_x'][it0, 0].real - inf_var)
        if prev is not None:
            assert diff < prev
        prev = diff
    assert prev < 1e-5


# ── API behaviour: warnings + errors ──────────────────────────────
def test_max_ell_1_computes_tadpole_loop():
    """max_ell=1 on Allen-Cahn now computes the 1-loop tadpole (Stage C):
    ⟨φ²⟩ is suppressed below the tree value 0.5 and matches the strict-1-loop
    0.4625 at λ=0.1.  (Previously this warned and returned tree-level.)"""
    model = _load('allen_cahn_1d_subcritical_infinite.theory.py')
    th0 = _compute(model, {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
                   np.array([0.0, 1.0]), max_ell=0)
    th1 = _compute(model, {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
                   np.array([0.0, 1.0]), max_ell=1)
    i0 = int(np.argmin(np.abs(th1['tau_grid'])))
    tree = float(np.asarray(th0['C_tau_x']).real[i0, 0])
    loop = float(np.asarray(th1['C_tau_x']).real[i0, 0])
    assert th1['C_tau_x'].shape[1] == 2
    assert abs(tree - 0.5) < 1e-9
    assert abs(loop - 0.4625) < 1e-4          # strict 1-loop ⟨φ²⟩
    assert loop < tree                         # the loop suppresses ⟨φ²⟩
    si = th1['spatial_info']
    assert si.get('Sigma') is not None
    # M(Γ) comes FROM the pipeline (not hardcoded): g = 3λ = 0.3, and it is
    # q-independent (the signature of a tadpole / pure mass shift).  The ~1e-6
    # residual is the pipeline's pole-numerics precision.
    assert abs(si['self_energy_coeff_g'] - 0.3) < 1e-4
    assert si['g_q_spread'] < 1e-4
    assert abs(si['phi2_0'] - 0.5) < 1e-9      # loop integral ⟨φ²⟩₀ (closed form)


def test_max_ell_above_1_raises():
    """max_ell > 1 spatial needs the per-edge ∫dℓ integrator (Stage C.5+);
    it must raise a clear NotImplementedError, not silently truncate."""
    model = _load('allen_cahn_1d_subcritical_infinite.theory.py')
    with pytest.raises(NotImplementedError, match='max_ell'):
        _compute(model, {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
                 np.array([0.0, 1.0]), max_ell=2)


def test_k_not_2_raises():
    from pipeline import compute_cumulants
    model = _load('allen_cahn_1d_subcritical_infinite.theory.py')
    # 3 external legs so we pass the len(external_fields)==k check and
    # reach the spatial k!=2 guard.
    with pytest.raises(NotImplementedError, match='k=2'):
        compute_cumulants(
            model=model, k=3, max_ell=0,
            fundamental={'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
            external_fields=[('phi', 1), ('phi', 2), ('phi', 3)],
            tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0]),
            parallel=False, verbose=False, use_cache=True)


# ── the d-general q→x output transform (the remaining d>1 piece) ──
@pytest.mark.parametrize('d,tol', [(1, 2e-3), (2, 1e-2), (3, 3e-2)])
def test_radial_inverse_ft_matches_closed_form(d, tol):
    """radial_inverse_ft applied to the free C(q,0)=T/(μ+Dq²) reproduces the exact
    static correlator in d=1,2,3: (T/2√μD)e^{−κr} (d=1), (T/2πD)K₀(κr) (d=2),
    (T/4πD)e^{−κr}/r (d=3), κ=√(μ/D).  This is the EXTERNAL output transform that
    turns a self-energy-dressed δC(|q|,τ) into the real-space correlator in any d.
    (d=3 sinc converges more slowly under plain trapz — FFTLog would tighten it;
    the result is cutoff-limited, Regime 1.)"""
    from msrjd.integration.spatial.spatial_correlator import (
        radial_inverse_ft, free_correlator_static_closed_form as oracle,
    )
    mu = D = T = 1.0
    q = np.linspace(0.0, 120.0, 12000)
    Cq = T / (mu + D * q * q)
    for r in (0.5, 1.0, 2.0, 3.0):
        got = radial_inverse_ft(q, Cq, r, d)
        ref = oracle(r, mu, D, T, d)
        assert abs(got - ref) <= tol * abs(ref)

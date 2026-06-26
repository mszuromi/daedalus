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
    from api import compute_cumulants
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
    # The GENERIC pipeline sums every enumerated ell=1 diagram (no bubble/tadpole
    # branch).  Allen-Cahn at φ*=0 has exactly ONE live diagram — the φ̃φ³ tadpole
    # — and its δC reproduces the validated mass-shift 0.4625 (𝒮(Γ) from the
    # enumeration, NOT g=3λ hardcoded).
    si = th1['spatial_info']
    assert si.get('one_loop') is True and si.get('generic') is True
    assert si.get('n_live_diagrams') == 1


def test_max_ell_above_2_raises():
    """v1 supports tree/1-loop/2-loop through the full-diagram integrator;
    max_ell > 2 must raise a clear NotImplementedError, not silently truncate.
    (Higher loops work by construction but are deliberately gated on cost.)"""
    model = _load('allen_cahn_1d_subcritical_infinite.theory.py')
    with pytest.raises(NotImplementedError, match='max_ell'):
        _compute(model, {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
                 np.array([0.0, 1.0]), max_ell=3)


def test_k_not_2_raises():
    from api import compute_cumulants
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
    from engine.integration.spatial.spatial_correlator import (
        radial_inverse_ft, free_correlator_static_closed_form as oracle,
    )
    mu = D = T = 1.0
    q = np.linspace(0.0, 120.0, 12000)
    Cq = T / (mu + D * q * q)
    for r in (0.5, 1.0, 2.0, 3.0):
        got = radial_inverse_ft(q, Cq, r, d)
        ref = oracle(r, mu, D, T, d)
        assert abs(got - ref) <= tol * abs(ref)


@pytest.mark.parametrize('d', [1, 2, 3])
def test_periodic_inverse_ft_limits(d):
    """periodic_inverse_ft (discrete-momentum lattice sum on a period-L cubic
    box) reproduces the infinite-domain radial_inverse_ft at large L, and gives
    a LARGER correlator at small L (the field's periodic images add)."""
    from engine.integration.spatial.spatial_correlator import (
        radial_inverse_ft, periodic_inverse_ft)
    mu = D = T = 1.0
    q = np.linspace(60.0 / 24000, 60.0, 6000)
    Cq = T / (mu + D * q * q)
    rs = np.array([0.3, 0.8, 1.5])
    inf = np.real(radial_inverse_ft(q, Cq, rs, d))
    big = np.real(periodic_inverse_ft(q, Cq, rs, d, L=30.0))
    sml = np.real(periodic_inverse_ft(q, Cq, rs, d, L=4.0))
    assert np.max(np.abs(big - inf)) < 5e-3            # L→∞ recovers infinite
    assert np.all(sml >= inf - 1e-9)                   # periodic enhancement
    assert sml[-1] > inf[-1] + 1e-4                    # strictly larger at large r


def test_d2_periodic_through_compute_cumulants():
    """END-TO-END: a d=2 PERIODIC linear theory runs through compute_cumulants;
    large L matches the infinite-domain value and small L exceeds it — via the
    d≥2 periodic lattice-sum path + the model-boundary fallback in _bc_from_prop."""
    from api.compute import compute_cumulants
    from api.theory import TheoryBuilder

    def build(bc, L):
        b = (TheoryBuilder('p2d', n_populations=0)
             .physical_field('phi', spatial_dim=2)
             .parameter('mu', default=1.0, domain='positive')
             .parameter('D', default=1.0, domain='positive')
             .parameter('T', default=1.0, domain='positive')
             .parameter('L', default=L, domain='positive')
             .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
             .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2'))
        b = (b.boundary('periodic', length='L') if bc == 'periodic'
             else b.boundary('infinite'))
        return b.initial('stationary').build()

    rs = np.array([0.3, 1.0, 1.8])

    def run(m, fund):
        th = compute_cumulants(model=m, k=2, max_ell=0, fundamental=fund,
                               external_fields=[('dphi', 1), ('dphi', 1)],
                               spatial_grid=rs, tau_max=0.0, tau_step=1.0,
                               verbose=False)
        return np.real(np.asarray(th['C_tau_x']))[0]

    inf = run(build('infinite', 6.0), {'mu': 1, 'D': 1, 'T': 1, 'L': 6.0})
    perL = run(build('periodic', 30.0), {'mu': 1, 'D': 1, 'T': 1, 'L': 30.0})
    perS = run(build('periodic', 4.0), {'mu': 1, 'D': 1, 'T': 1, 'L': 4.0})
    assert np.max(np.abs(perL - inf)) < 5e-3                 # large L ≈ infinite
    assert np.all(perS >= inf - 1e-9)                        # periodic ≥ infinite
    assert perS[-1] > inf[-1] + 1e-4                         # strictly enhanced


def test_d2_tree_through_compute_cumulants():
    """END-TO-END: a d=2 linear theory runs through ``compute_cumulants`` (tree,
    max_ell=0) and the returned ``C(r,0)`` matches the exact d=2 free correlator
    (T/2πD)·K₀(r√(μ/D)).  Confirms the d=2 spatial dispatch (relaxed d≠1 gate +
    radial q→x transform) works through the public API."""
    from api.compute import compute_cumulants
    from api.theory import TheoryBuilder
    from engine.integration.spatial.spatial_correlator import (
        free_correlator_static_closed_form as oracle,
    )
    m = (TheoryBuilder('lin2d', n_populations=0)
         .physical_field('phi', spatial_dim=2)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2')
         .boundary('infinite').initial('stationary').build())
    rs = np.array([0.5, 1.0, 2.0, 3.0])
    out = compute_cumulants(
        m, k=2, max_ell=0, fundamental={'mu': 1.0, 'D': 1.0, 'T': 1.0},
        external_fields=[('phi', 1), ('phi', 1)], spatial_grid=rs,
        tau_max=1.0, tau_step=1.0, verbose=False, use_cache=False,
        mf_dae_n_starts=4)
    C = np.real(out['C_tau_x'])
    mid = C.shape[0] // 2                         # τ=0 row (grid is −1,0,1)
    for i, r in enumerate(rs):
        assert abs(C[mid, i] - oracle(r, 1.0, 1.0, 1.0, 2)) <= 2e-2 * oracle(r, 1.0, 1.0, 1.0, 2)
    # total_C closure also works at d=2 (single-point query)
    assert abs(out['total_C'](0.0, 1.0).real - oracle(1.0, 1.0, 1.0, 1.0, 2)) <= 2e-2 * oracle(1.0, 1.0, 1.0, 1.0, 2)


def test_d2_bubble_loop_through_compute_cumulants():
    """END-TO-END: a d=2 reaction-diffusion (φ̃φ²) theory runs through
    ``compute_cumulants(max_ell=1)`` and returns tree + 1-loop bubble.  The loop
    correction δC(r,0)=C₁−C₀ is positive and q-dependent, and scales as g²
    (δC(2g)/δC(g)≈4) — confirming the d=2 bubble path (the d=2 ∫d²ℓ self-energy,
    validated vs the C-stack, + the d-independent Dyson collapse + radial q→x)."""
    from api.compute import compute_cumulants
    from api.theory import TheoryBuilder

    def rd2(g):
        return (TheoryBuilder('rd2', n_populations=0)
                .physical_field('phi', spatial_dim=2)
                .parameter('mu', default=1.0, domain='positive')
                .parameter('D', default=1.0, domain='positive')
                .parameter('g', default=g, domain='real')
                .parameter('T', default=1.0, domain='positive')
                .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2')
                .set_action_text('phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
                .boundary('infinite').initial('stationary').build())

    rs = np.array([0.5, 1.0, 2.0])
    kw = dict(k=2, external_fields=[('phi', 1), ('phi', 1)], spatial_grid=rs,
              tau_max=1.0, tau_step=1.0, verbose=False, use_cache=False,
              mf_dae_n_starts=4)
    o1 = compute_cumulants(rd2(0.2), max_ell=1,
                           fundamental={'mu': 1.0, 'D': 1.0, 'g': 0.2, 'T': 1.0}, **kw)
    o0 = compute_cumulants(rd2(0.2), max_ell=0,
                           fundamental={'mu': 1.0, 'D': 1.0, 'g': 0.2, 'T': 1.0}, **kw)
    o2 = compute_cumulants(rd2(0.4), max_ell=1,
                           fundamental={'mu': 1.0, 'D': 1.0, 'g': 0.4, 'T': 1.0}, **kw)
    # GENERIC pipeline: the d=2 1-loop now SUMS the bubbles AND the (Regime-1
    # cutoff) tadpole — the previously-dropped diagram — through one path.
    assert o1['spatial_info']['generic'] is True
    assert o1['spatial_info']['n_live_diagrams'] >= 2   # 2 bubbles (+ tadpole)
    mid = o1['C_tau_x'].shape[0] // 2
    dC_g = np.real(o1['C_tau_x'])[mid] - np.real(o0['C_tau_x'])[mid]
    dC_2g = np.real(o2['C_tau_x'])[mid] - np.real(o0['C_tau_x'])[mid]  # tree(0.4)≈tree(0.2)
    # g²-scaling: every ell=1 diagram (bubble AND tadpole) is ∝ g², so the full
    # 1-loop correction scales as g² (tree is g-independent at φ*=0).
    ratio = dC_2g[0] / dC_g[0]
    assert abs(ratio - 4.0) <= 0.15

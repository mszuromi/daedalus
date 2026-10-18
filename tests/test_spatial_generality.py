"""
tests/test_spatial_generality.py
================================
Spatial generality (audit W6s): population legs, parameters inside operators,
and a constant drift.

* Population legs (GEN-10).  A multi-population spatial field's leg
  ``('p', 2)`` is population 2, never population 1 (the leg->field map used to
  strip the population digit): each population matches its single-population
  twin and the closed form; an absent population or an ambiguous bare name
  raises.
* Parameters inside operators (GEN-11).  At a NON-ZERO saddle,
  ``Lap(lam*p^2)`` and ``lam*Lap(p^2)`` are the same model (the constant
  ``Lap(lam*pstar^2)`` is annihilated): identical results, and the tree equals
  the closed form with ``D_eff = DD - 2*lam*p*``.
* Constant drift (GEN-12, ACC-74).  A drift ``v*Dx(phi)`` shared by every field
  is removed exactly by the Galilean boost, ``C_v(x, tau) = C_0(x - v*tau, tau)``
  at every loop order and every k (convention ``C(x, tau) = <phi(0,0)
  phi(x,tau)>``).  The tree matches the independent erfc closed form (infinite
  line and periodic image sum) and its exact time/rate-rescaled twins; the
  coupled k=2 and k=3 drivers (k>=3 used to drop the drift) match the
  drift-free model at the boosted points; unequal drifts, cross-advection and
  a real k-linear term refuse with thresholds relative to the model's
  velocity scale.

Reference closed form (independent of the package):
``C(x,t) = T/(4 sqrt(mu D)) [e^{-ax} erfc(sqrt(mu|t|) - x/(2 sqrt(D|t|)))
+ e^{ax} erfc(sqrt(mu|t|) + x/(2 sqrt(D|t|)))]``, ``a = sqrt(mu/D)``.

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_spatial_generality.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR
from scipy.special import erfc

from api.model import ModelBuilder
from api import compute_cumulants
from engine.integration.spatial.heat_kernel import SpatialPropagatorError
from engine.integration.spatial.pipeline_bridge import (
    _drift_velocity, _legs_to_phys_idx)


def C_ou(x, tau, mu, D, T):
    """Exact stationary 1-d OU-diffusion correlator (erfc closed form)."""
    x, tau = abs(float(x)), abs(float(tau))
    a = math.sqrt(mu / D)
    if tau == 0.0:
        return T / (2.0 * math.sqrt(mu * D)) * math.exp(-a * x)
    s, r = math.sqrt(mu * tau), x / (2.0 * math.sqrt(D * tau))
    return T / (4.0 * math.sqrt(mu * D)) * (
        math.exp(-a * x) * erfc(s - r) + math.exp(a * x) * erfc(s + r))


_KW2 = dict(k=2, use_cache=False, parallel=False, spatial_parallel=False,
            verbose=False)


def _k2(model, legs, taus, xs, max_ell=0, **params):
    out = compute_cumulants(model, max_ell=max_ell, external_fields=legs,
                            tau_grid=np.asarray(taus, float),
                            spatial_grid=np.asarray(xs, float),
                            parameters=params or None, **_KW2)
    return np.real(np.asarray(out['C_tau_x'])), out


# ── GEN-10: population legs ───────────────────────────────────────────
def _two_pop():
    return (ModelBuilder('two-pop spatial', n_populations=2)
            .physical_field('p', spatial_dim=1)
            .parameter('mu', default=[1.0, 4.0], indexed='vector',
                       domain='positive')
            .parameter('DD', default=1.0, domain='positive')
            .parameter('T', default=1.0, domain='positive')
            .set_action_text('sum(pt[i]*((Dt+mu[i]-DD*Laplacian)*p[i])'
                             ' - T*pt[i]^2 for i in pop)')
            .equation(lhs='(Dt + mu[i] - DD*Laplacian)*p[i]', rhs='0')
            .boundary('infinite').initial('stationary').build())


def test_population_leg_is_its_own_field():
    taus, xs = [0.0, 0.5], [0.0, 0.7]
    m = _two_pop()
    for pop, mu in ((1, 1.0), (2, 4.0)):
        C, _ = _k2(m, [('p', pop), ('p', pop)], taus, xs)
        ref = np.array([[C_ou(x, t, mu, 1.0, 1.0) for x in xs] for t in taus])
        assert np.max(np.abs(C - ref)) < 1e-13, (pop, C, ref)


def test_population_leg_resolution_refuses_wrong_or_ambiguous_legs():
    pidx = {('dp', 1): 0, ('dp', 2): 1, ('dq', 1): 2}
    assert _legs_to_phys_idx([('dp', 2), ('p', 1)], pidx) == [('dp', 2),
                                                              ('dp', 1)]
    assert _legs_to_phys_idx(['dp2', ('q', 7)], pidx) == [('dp', 2),
                                                          ('dq', 1)]
    with pytest.raises(SpatialPropagatorError, match='populations'):
        _legs_to_phys_idx([('p', 3)], pidx)
    with pytest.raises(SpatialPropagatorError, match='populations'):
        _legs_to_phys_idx(['dp'], pidx)
    with pytest.raises(SpatialPropagatorError, match='does not name'):
        _legs_to_phys_idx([('r', 1)], pidx)
    with pytest.raises(SpatialPropagatorError):
        _k2(_two_pop(), [('p', 3), ('p', 3)], [0.0], [0.0])


# ── GEN-11: a parameter inside an operator at a non-zero saddle ───────
def _op_model(term):
    return (ModelBuilder('param inside operator', n_populations=0)
            .physical_field('p', spatial_dim=1)
            .parameter('mu', default=1.0, domain='positive')
            .parameter('DD', default=1.0, domain='positive')
            .parameter('lam', default=0.3, domain='real')
            .parameter('h', default=1.0, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='h')
            .set_action_text(f'pt*(Dt(p)+mu*p-DD*Lap(p)+{term}-h) - T*pt^2')
            .operator_ir().boundary('infinite').initial('stationary').build())


def test_kill_means_treats_parameters_as_constants():
    from api.spatial_operator_ir import Lap, kill_means
    pbar, dp, lam, x = SR.var('pbar dp lam x')
    keep = [dp, x]
    assert kill_means(Lap(lam * pbar ** 2), [pbar], keep_syms=keep) == 0
    assert kill_means(Lap(lam), [pbar], keep_syms=keep) == 0
    # a fluctuation or a coordinate inside the operator is not a constant
    assert kill_means(Lap(lam * dp ** 2), [pbar], keep_syms=keep) != 0
    assert kill_means(Lap(x * pbar), [pbar], keep_syms=keep) != 0
    # without keep_syms the old rule (means only) is unchanged
    assert kill_means(Lap(lam * pbar ** 2), [pbar]) != 0
    assert kill_means(Lap(pbar ** 2), [pbar]) == 0


def test_parameter_inside_operator_at_nonzero_saddle():
    taus, xs = [0.0, 0.5, 1.0], [0.0, 0.6, 1.5]
    A, _ = _k2(_op_model('Lap(lam*p^2)'), [('p', 1), ('p', 1)], taus, xs)
    B, _ = _k2(_op_model('lam*Lap(p^2)'), [('p', 1), ('p', 1)], taus, xs)
    assert np.array_equal(A, B)
    Deff = 1.0 - 2.0 * 0.3 * 1.0                   # DD - 2*lam*p*, p* = h/mu
    ref = np.array([[C_ou(x, t, 1.0, Deff, 1.0) for x in xs] for t in taus])
    assert np.max(np.abs(A - ref)) < 1e-12, (A, ref)


# ── GEN-12 / ACC-74: constant drift (Galilean boost) ──────────────────
def _advected(v=None, lam=None, L=None):
    b = (ModelBuilder('advected OU', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive'))
    nl, rhs, adv = '', '0', ''
    if lam is not None:
        b = b.parameter('lam', default=lam, domain='positive')
        nl, rhs = ' + lam*phi^3', '-lam*phi^3'
    if v is not None:
        b = b.parameter('v', default=v, domain='real')
        adv = ' + v*Dx(phi, 0)'
    b = (b.equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs=rhs)
         .set_action_text(f'phit*(Dt(phi) + mu*phi - D*Lap(phi){adv}{nl})'
                          ' - T*phit^2').operator_ir())
    b = b.boundary('periodic', length=L) if L else b.boundary('infinite')
    return b.initial('stationary').build()


_TD = np.array([-1.0, 0.0, 0.5, 2.0])
_XD = np.array([-2.0, 0.0, 0.4, 1.5])
_LEG = [('phi', 1), ('phi', 1)]


@pytest.mark.parametrize('s', [1.0, 1e-6, 1e8])
def test_drift_tree_is_the_boosted_closed_form(s):
    """Exact up to an exact change of time units (rates*s, tau/s)."""
    v = 0.3
    C, out = _k2(_advected(v), _LEG, _TD / s, _XD, mu=s, D=s, T=s, v=v * s)
    ref = np.array([[C_ou(x - v * t, t, 1.0, 1.0, 1.0) for x in _XD]
                    for t in _TD])
    assert np.max(np.abs(C - ref)) < 1e-14, (C, ref)


def test_drift_tree_periodic_box_is_the_boosted_image_sum():
    v, L = 0.3, 6.0
    C, _ = _k2(_advected(v, L=L), _LEG, _TD, _XD)
    ref = np.array([[sum(C_ou(x - v * t + n * L, t, 1.0, 1.0, 1.0)
                         for n in range(-40, 41)) for x in _XD] for t in _TD])
    assert np.max(np.abs(C - ref)) < 1e-14, (C, ref)


def test_drift_one_loop_is_the_boosted_drift_free_loop():
    v, lam = 0.3, 0.1
    taus, xs = np.array([0.0, 0.5, -0.5]), np.array([0.0, 0.6])
    C, _ = _k2(_advected(v, lam=lam), _LEG, taus, xs, max_ell=1)
    ref = np.array([_k2(_advected(lam=lam), _LEG, [t], xs - v * t,
                        max_ell=1)[0][0] for t in taus])
    assert np.array_equal(C, ref)
    # the drift is not dropped (the boosted points differ)
    C0, _ = _k2(_advected(lam=lam), _LEG, taus, xs, max_ell=1)
    assert np.max(np.abs(C - C0)) > 1e-3


def _coupled(va, vb):
    da = f' + {va}*Dx(a, 0)' if va else ''
    db = f' + {vb}*Dx(b, 0)' if vb else ''
    return (ModelBuilder('coupled drift', n_populations=0)
            .physical_field('a', spatial_dim=1)
            .physical_field('b', spatial_dim=1)
            .parameter('ma', default=1.0, domain='positive')
            .parameter('mb', default=1.7, domain='positive')
            .parameter('DD', default=1.0, domain='positive')
            .parameter('c', default=0.4, domain='real')
            .parameter('g', default=0.25, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .equation(lhs='(Dt+ma-DD*Laplacian)*a + c*b', rhs='0')
            .equation(lhs='(Dt+mb-DD*Laplacian)*b - c*a', rhs='0')
            .set_action_text(
                f'at*(Dt(a)+ma*a-DD*Lap(a){da}+c*b+g*a^2) - T*at^2'
                f' + bt*(Dt(b)+mb*b-DD*Lap(b){db}-c*a) - T*bt^2')
            .operator_ir().boundary('infinite').initial('stationary')
            .build())


def test_coupled_drift_k2_and_k3_are_boosted():
    """ACC-74: the coupled k>=3 driver used to drop a drift; the coupled k=2
    tree crashed on it.  Both are the drift-free model at x - v*tau."""
    v = 0.35
    taus, xs = np.array([-0.8, 0.6]), np.array([0.0, 0.7])
    legs = [('a', 1), ('b', 1)]
    C, _ = _k2(_coupled(v, v), legs, taus, xs)
    ref = np.array([_k2(_coupled(0, 0), legs, [t], xs - v * t)[0][0]
                    for t in taus])
    assert np.array_equal(C, ref)
    pts = np.array([[[0.0, 0.0], [0.5, 0.4]], [[1.0, -0.3], [-0.2, 0.7]]])
    kw = dict(k=3, max_ell=0, use_cache=False, parallel=False, verbose=False,
              external_fields=[('a', 1), ('b', 1), ('a', 1)])
    Cv = compute_cumulants(_coupled(v, v), spatial_points=pts, **kw)['C_kpoint']
    pb = pts.copy()
    pb[:, :, 0] -= v * pb[:, :, 1]
    C0b = compute_cumulants(_coupled(0, 0), spatial_points=pb, **kw)['C_kpoint']
    C0 = compute_cumulants(_coupled(0, 0), spatial_points=pts, **kw)['C_kpoint']
    assert np.array_equal(Cv, C0b)
    assert np.max(np.abs(Cv - C0)) > 1e-5 * np.max(np.abs(C0))
    with pytest.raises(NotImplementedError, match='different velocities'):
        compute_cumulants(_coupled(v, 0.1), spatial_points=pts, **kw)


def test_drift_velocity_rules_are_unit_invariant():
    M = np.diag([1.0, 1.7]) + np.array([[0, 0.4], [-0.4, 0]])
    D = np.eye(2)
    for s in (1.0, 1e-12, 1e12):
        V = 1j * 0.3 * s * np.eye(2)
        assert _drift_velocity(V, M * s, D * s) == pytest.approx(0.3 * s,
                                                                 rel=1e-15)
        # a roundoff-level mismatch is a common drift; a real one is not
        Vn = V + np.diag([0.0, 1e-14j * s])
        assert _drift_velocity(Vn, M * s, D * s) == pytest.approx(0.3 * s)
        with pytest.raises(NotImplementedError, match='different velocities'):
            _drift_velocity(V + np.diag([0.0, 1e-3j * s]), M * s, D * s)
        with pytest.raises(NotImplementedError, match='off-diagonal'):
            _drift_velocity(V + np.array([[0, 1e-3j * s], [0, 0]]), M * s,
                            D * s)
        with pytest.raises(NotImplementedError, match='REAL part'):
            _drift_velocity(V + 1e-3 * s * np.eye(2), M * s, D * s)
    # a drift far below the velocity scale is no drift (relative, not 1e-9)
    assert _drift_velocity(1e-14j * np.eye(1), np.eye(1), np.eye(1)) == 0.0
    assert _drift_velocity(1e-14j * np.eye(1), 1e-20 * np.eye(1),
                           1e-20 * np.eye(1)) != 0.0


# ── MEM-04: one calibrated memory guard inside every chamber ──────────
def _records(action, d, ell, rhs='0'):
    from sage.all import SR as _SR
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _formfactor_callable)
    m = (ModelBuilder('mem', n_populations=0).physical_field('p', spatial_dim=d)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('c', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*p', rhs=rhs)
         .set_action_text(action).operator_ir().boundary('infinite')
         .initial('stationary').build())
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, m, use_cache=False, verbose=False)
    _, pidx = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = _legs_to_phys_idx([('p', 1), ('p', 1)], pidx)
    base = {_SR.var(n): v for n, v in (('mu', 1.0), ('D', 1.0), ('c', 0.3),
                                       ('T', 1.0), ('pstar', 0.0),
                                       ('pstar1', 0.0))}
    vt = [{'weight': float(_SR(t['weight']).subs(base)), 'n_phys': t['n_phys'],
           'chain': t['chain'], 'mode': t['mode']}
          for t in (getattr(ft._ns, '_operator_ir_vertex_terms', None) or [])]
    out = []
    for td, _pv in build_pipeline_records(ft, m, prop, ext, max_ell=ell,
                                          verbose=False).get(ell, []):
        out.append((diagram_to_cstack(td),
                    _formfactor_callable(td, vt, d=d) if vt else None))
    return out


def _peak_and_estimate(dd, ff, d, xs, n_t, n_s, q=0.8):
    import tracemalloc
    from engine.integration.spatial import full_integrator as FI
    et = FI.external_times_2pt(dd, 0.5)
    n_V = len(dd.internal_vertices)
    n_C = sum(1 for e in dd.edges if e.kind == 'C')
    a = np.array([e.a for e in dd.edges], dtype=float).reshape(len(dd.edges),
                                                               -1)
    P_bound = (n_t ** n_V) * (n_s ** n_C) * FI.time_grid_split_factor(
        dd, et, n_t, min(n_t, 8))
    est = FI.chamber_peak_bytes(
        P_bound, n_V=n_V, n_leaf=len(et), n_C=n_C, E=len(dd.edges),
        L=a.shape[1], n_ext=1, n_x=0 if xs is None else len(xs),
        spatial_dim=d, formfactor=ff)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        b0 = tracemalloc.get_traced_memory()[0]
        FI.diagram_kinematic(dd, [q], et, 1.0, 1.0, spatial_dim=d, n_t=n_t,
                             n_s=n_s, formfactor=ff, xs=xs)
        peak = tracemalloc.get_traced_memory()[1] - b0
    finally:
        tracemalloc.stop()
    return peak, est


def test_chamber_estimate_covers_the_measured_peak():
    """Plain x-path (n_x = 30) and the d = 2 form-factor q-path (its
    Gauss–Hermite grid): the per-chamber estimate is above the tracemalloc
    peak and not wildly above it."""
    from engine.integration.spatial.full_integrator import (
        uv_divergent_subgraphs)
    dd, ff = _records('pt*(Dt(p)+mu*p-D*Lap(p)+c*p^3)-T*pt^2', 1, 1,
                      rhs='-c*p^3')[0]
    peak, est = _peak_and_estimate(dd, ff, 1, np.linspace(0, 2, 30), 22, 24)
    assert 0.3 * est <= peak <= est, (peak, est)
    recs = [r for r in _records('pt*(Dt(p)+mu*p-D*Lap(p)-c*Lap(p^2))-T*pt^2',
                                2, 1, rhs='c*Laplacian*p^2')
            if r[1] is not None and not uv_divergent_subgraphs(r[0], 2)]
    dd, ff = max(recs, key=lambda r: getattr(r[1], 'gh_order_needed', 1))
    peak, est = _peak_and_estimate(dd, ff, 2, None, 22, 24)
    assert 0.3 * est <= peak <= est, (peak, est)


def test_kpoint_and_gh_paths_are_guarded(monkeypatch):
    """The k-point driver and the d >= 2 form-factor q-path used to run with
    no guard (> 5 GB measured at k = 3, one loop, two points)."""
    from engine.integration.spatial import full_integrator as FI
    from engine.integration.spatial.full_integrator import (
        SpatialMemoryBudgetError, uv_divergent_subgraphs)
    monkeypatch.setenv('SPATIAL_MEM_BUDGET_GB', '0.01')
    m = (ModelBuilder('k3 guard', n_populations=0)
         .physical_field('p', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('c', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*p', rhs='-c*p^2')
         .set_action_text('pt*(Dt(p)+mu*p-D*Lap(p)+c*p^2)-T*pt^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    pts = np.array([[[0.0, 0.0], [0.5, 0.4]], [[1.0, 0.0], [-0.5, 0.4]]])
    with pytest.raises(SpatialMemoryBudgetError, match='chamber'):
        compute_cumulants(m, k=3, max_ell=1, spatial_points=pts,
                          use_cache=False, parallel=False, verbose=False,
                          external_fields=[('p', 1)] * 3)
    recs = [r for r in _records('pt*(Dt(p)+mu*p-D*Lap(p)-c*Lap(p^2))-T*pt^2',
                                2, 1, rhs='c*Laplacian*p^2')
            if r[1] is not None and not uv_divergent_subgraphs(r[0], 2)]
    dd, ff = recs[0]
    with pytest.raises(SpatialMemoryBudgetError, match='chamber'):
        FI.diagram_kinematic(dd, [0.8], FI.external_times_2pt(dd, 0.5), 1.0,
                             1.0, spatial_dim=2, formfactor=ff)


# ── ACC-44: the unequal-D (Dyson) tree has no q-grid / mode cap ───────
def test_laguerre_gauss_transform_closed_form():
    """F_p(x; a) = ∫dᵈq/(2π)ᵈ e^{iq·x}|q|^{2p}e^{−aq²} against direct
    quadrature (cosine, Hankel J0, spherical sinc), and the two periodic
    branches (images / dual modes) against each other."""
    from scipy.integrate import quad
    from scipy.special import j0
    from engine.integration.spatial.dyson_dressing import _F_p, _periodic_1d
    a = 0.37
    for p in range(4):
        for r in (0.0, 0.6, 1.7):
            got = [float(_F_p(p, np.array([r]), a, d)[0]) for d in (1, 2, 3)]
            g = lambda q: q ** (2 * p) * np.exp(-a * q * q)
            ref1 = quad(lambda q: np.cos(q * r) * g(q), 0, np.inf)[0] / np.pi
            ref2 = quad(lambda q: q * j0(q * r) * g(q), 0, np.inf)[0] / (2 * np.pi)
            ref3 = quad(lambda q: q * q * np.sinc(q * r / np.pi) * g(q), 0,
                        np.inf)[0] / (2 * np.pi ** 2)
            for gv, rv in zip(got, (ref1, ref2, ref3)):
                assert abs(gv - rv) <= 1e-10 * max(abs(rv), 1e-3), (p, r, gv, rv)
    L, xs = 3.0, np.array([0.0, 0.4, 1.5, 2.9])
    for p in range(3):
        img = _periodic_1d(p, xs, 0.999 * L * L / (4 * np.pi), L)  # images
        mod = _periodic_1d(p, xs, 1.001 * L * L / (4 * np.pi), L)  # modes
        # a changes by 0.2 % across the switch: compare each against its own
        # opposite-branch evaluation at the same a via the closed image sum
        a_m = 1.001 * L * L / (4 * np.pi)
        img_m = sum(_F_p(p, xs + n * L, a_m, 1) for n in range(-60, 61))
        assert np.max(np.abs(mod - img_m)) <= 1e-12 * np.max(np.abs(img_m))
        assert np.all(np.isfinite(img))


def _exact_unequal_D_line(M, Dm, N, tau, x, legs):
    from scipy.integrate import quad
    from scipy.linalg import expm, solve_continuous_lyapunov
    i, j = legs
    ii, jj = (j, i) if tau >= 0 else (i, j)

    def Cq(q):
        A = M + Dm * q * q
        return np.real((expm(-A * abs(tau))
                        @ solve_continuous_lyapunov(A, N))[ii, jj])
    if x == 0.0:
        return quad(Cq, 0, np.inf, limit=400, epsabs=1e-14)[0] / np.pi
    return quad(Cq, 0, np.inf, weight='cos', wvar=abs(x), limit=400)[0] / np.pi


def test_dressed_tree_x_converges_to_the_exact_unequal_D_tree():
    """The x-space Dyson tree vs the exact Lyapunov tree on the line and on
    a periodic box (image sum of the line): the truncation error falls
    geometrically with the order; no fixed-grid floor."""
    from engine.integration.spatial.dyson_dressing import dressed_tree_C_x
    M = np.array([[1.5, 0.4], [-0.3, 1.2]])
    Dm = np.diag([0.6, 1.0])
    D0 = 0.8
    N = np.diag([2.0, 1.4])
    legs, xs = (0, 1), np.array([0.0, 0.5, 2.0])
    for tau in (0.7, -0.7):
        ex = np.array([_exact_unequal_D_line(M, Dm, N, tau, x, legs)
                       for x in xs])
        errs = [np.max(np.abs(np.real(dressed_tree_C_x(
            xs, tau, M, Dm - D0 * np.eye(2), D0, N, n, legs)) - ex))
            for n in (1, 3, 6)]
        assert errs[0] > errs[1] > errs[2] and errs[2] < 3e-8, errs
    L, tau = 3.0, 0.4
    ex = np.array([sum(_exact_unequal_D_line(M, Dm, N, tau, x + n * L, legs)
                       for n in range(-8, 9)) for x in xs])
    got = np.real(dressed_tree_C_x(xs, tau, M, Dm - D0 * np.eye(2), D0, N, 6,
                                   legs, L=L))
    assert np.max(np.abs(got - ex)) < 3e-8, (got, ex)


# ── k >= 3 derivative vertices: the form factor reaches the k-point driver ─
def _k3_model(term, rhs):
    return (ModelBuilder('k3 form factor', n_populations=0)
            .physical_field('p', spatial_dim=1)
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .parameter('c', default=0.3, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .equation(lhs='(Dt + mu - D*Laplacian)*p', rhs=rhs)
            .set_action_text(f'pt*(Dt(p)+mu*p-D*Lap(p)+{term})-T*pt^2')
            .operator_ir().boundary('infinite').initial('stationary').build())


_MB = ('-c*Lap(p^2)', 'c*Laplacian*p^2')
_PL = ('c*p^2', '-c*p^2')


def _k3(model, pts):
    return compute_cumulants(model, k=3, max_ell=0, spatial_points=pts,
                             use_cache=False, parallel=False, verbose=False,
                             external_fields=[('p', 1)] * 3)['C_kpoint']


def test_kpoint_applies_the_derivative_form_factor():
    """compute_spatial_kpoint used to drop the form factor: Model B's
    conserved c*Lap(p^2) came out as exactly minus the plain c*p^2 twin."""
    import sympy as sp
    pts = np.array([[[0.5, 0.3], [-0.4, 0.0]], [[1.2, 0.3], [0.2, 0.0]],
                    [[0.0, 0.6], [0.9, 0.2]]])
    CB, CP = _k3(_k3_model(*_MB), pts), _k3(_k3_model(*_PL), pts)
    assert np.max(np.abs(CB + CP)) > 0.1 * np.max(np.abs(CP)), (CB, CP)
    # every tree record: the multivariate moment exists and each external
    # symbol q_j is read from q[j] (by index, not by position)
    from sage.all import SR as _SR
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _formfactor_callable, diagram_form_factor)
    m = _k3_model(*_MB)
    ft = FieldTheory(m, taylor_order=5)
    ft.expand()
    prop = build_propagator(ft, m, use_cache=False, verbose=False)
    _, pidx = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = _legs_to_phys_idx([('p', 1)] * 3, pidx)
    base = {_SR.var('c'): 0.3}
    vt = [{'weight': float(_SR(t['weight']).subs(base)), 'n_phys': t['n_phys'],
           'chain': t['chain'], 'mode': t['mode']}
          for t in ft._ns._operator_ir_vertex_terms]
    seen_partial = False
    for td, _p in build_pipeline_records(ft, m, prop, ext, max_ell=0,
                                         k=3).get(0, []):
        ff = _formfactor_callable(td, vt, d=1)
        R = sp.expand(diagram_form_factor(td, vt, d=1))
        if R == 0:
            continue
        assert ff.moment_x_multi is not None
        q = [0.3, 1.1]
        want = complex(R.subs({sp.Symbol('q0'): q[0], sp.Symbol('q1'): q[1]}))
        got = complex(np.asarray(ff(np.zeros((1, 0)), q)).ravel()[0])
        assert abs(got - want) <= 1e-12 * max(abs(want), 1.0), (R, got, want)
        # a form factor in fewer than all k-1 external momenta (the Model B
        # tree uses q0 only) used to get no multivariate moment
        seen_partial |= len([s for s in R.free_symbols
                             if str(s)[0] == 'q']) < 2
    assert seen_partial


@pytest.mark.slow
def test_kpoint_model_b_conservation_sum_rule():
    """int dx1 dx2 kappa3 = kappa3(q = 0) = 0 for the conserved vertex (the
    O(h^2) trapezoid residual shrinks 4x per halving); the plain twin's
    integral is O(1).  Main (form factor dropped) gave +1.00 of int|kappa3|."""
    def ratio(model, h):
        xg = np.arange(-10, 10 + h / 2, h)
        tot = ab = 0.0
        for x1 in xg:
            pts = np.zeros((len(xg), 2, 2))
            pts[:, 0, 0], pts[:, 0, 1], pts[:, 1, 0] = x1, 0.3, xg
            v = _k3(model, pts)
            tot, ab = tot + np.sum(v), ab + np.sum(np.abs(v))
        return tot / ab
    r5, r25 = ratio(_k3_model(*_MB), 0.5), ratio(_k3_model(*_MB), 0.25)
    assert abs(r5) < 0.06 and abs(r25) < 0.02 and abs(r25) < abs(r5) / 3
    assert abs(ratio(_k3_model(*_PL), 0.5)) > 0.9

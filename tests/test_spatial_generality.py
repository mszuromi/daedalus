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

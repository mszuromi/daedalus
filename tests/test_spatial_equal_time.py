"""
tests/test_spatial_equal_time.py
================================
Equal external times: the radial-simplex rule of the spatial chamber integral.

At equal external times (k = 2 at tau = 0, k-point events at one time) every
edge weight is homogeneous in (vertex-time gaps, Schwinger sigmas), so the
radial direction -- which carries the heat-kernel factor e^{-x^2/4B} that a
product grid resolves only jointly in (t, sigma) -- is a closed-form Bessel-K,
and the remaining simplex is integrated spectrally (tanh-sinh, stick
breaking) at an adaptive order.  The product grid was 1.2e-5 relative off the
exact one-loop Allen-Cahn mass shift at tau = 0 (1e-12 at tau = 1).

References: the exact tadpole mass shift (Allen-Cahn, d = 1), a fine product
grid (bubbles, d = 1 and 2; the k = 3 tree), the grid itself where the rule
does not apply (SPATIAL_EQTIME=0 restores it bit for bit).

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_spatial_equal_time.py -q
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR

from api.model import ModelBuilder
from api import compute_cumulants
from engine.integration.spatial import full_integrator as FI

_MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')


def _ac():
    spec = importlib.util.spec_from_file_location(
        'ac', os.path.join(_MODELS, 'allen_cahn_1d_subcritical_infinite.model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _q2(d=1):
    return (ModelBuilder('phi2 bubble', n_populations=0)
            .physical_field('p', spatial_dim=d)
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=0.7, domain='positive')
            .parameter('c', default=0.3, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .mf_residual(lhs='(Dt + mu - D*Laplacian)*p', rhs='-c*p^2')
            .set_action_text('pt*(Dt(p)+mu*p-D*Lap(p)+c*p^2)-T*pt^2')
            .operator_ir().boundary('infinite').initial('stationary').build())


def _C_eq(x, mu, D, T):
    """Equal-time free correlator T/(2 sqrt(mu D)) e^{-|x| sqrt(mu/D)}."""
    return T / (2.0 * math.sqrt(mu * D)) * math.exp(-abs(x) * math.sqrt(mu / D))


def test_allen_cahn_one_loop_equal_time_is_the_exact_mass_shift():
    """W5's oracle case: tau = 0 one loop = d/dmu of the tree times the
    tadpole mass shift 3 lam <phi^2>_0 (exact in d = 1)."""
    mu, D, T, lam = 1.5, 0.5, 0.8, 0.15
    xs = np.array([0.0, 0.5, 1.0, 2.0])
    out = compute_cumulants(_ac(), k=2, max_ell=1,
                            parameters=dict(mu=mu, D=D, T=T, lam=lam),
                            external_fields=[('phi', 1), ('phi', 1)],
                            tau_grid=np.array([0.0]), spatial_grid=xs,
                            use_cache=False, parallel=False,
                            spatial_parallel=False, verbose=False)
    dC = np.real(np.asarray(out['C_tau_x']))[0] - np.array(
        [_C_eq(x, mu, D, T) for x in xs])
    dmu = 3.0 * lam * _C_eq(0.0, mu, D, T)
    # d/dmu of T/(2 sqrt(mu D)) e^{-|x| sqrt(mu/D)}
    ref = np.array([dmu * _C_eq(x, mu, D, T)
                    * (-0.5 / mu - abs(x) / (2.0 * math.sqrt(mu * D)))
                    for x in xs])
    assert np.max(np.abs(dC - ref) / np.abs(ref)) < 1e-11, (dC, ref)
    eq = out['spatial_info']['equal_time']
    assert eq['calls'] >= 1 and eq['unconverged'] == 0
    assert eq['max_rel_change'] < 1e-9


def _records(model, params, field, ell, k=2):
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.diagrams.symmetry import external_wick_compensation
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    _, pidx = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = _legs_to_phys_idx([(field, 1)] * k, pidx)
    base = {SR.var(n): v for n, v in params.items()}
    out = []
    for td, p in build_pipeline_records(ft, model, prop, ext, max_ell=ell,
                                        k=k).get(ell, []):
        pv = float(SR(p).subs(base))
        if pv != 0.0:
            out.append((diagram_to_cstack(td), pv,
                        external_wick_compensation(td)))
    return out


@pytest.mark.parametrize('d, xs', [(1, [0.0, 0.5, 1.0, 2.0]),
                                   (2, [0.5, 1.0, 2.0])])
def test_bubble_equal_time_matches_a_fine_grid(d, xs, monkeypatch):
    """phi^2 bubbles at tau = 0 (the UV-finite records at d = 2): the rule
    agrees with the (44, 48) product grid to that grid's own accuracy, and
    is converged in its order."""
    xs = np.asarray(xs)
    params = dict(mu=1.0, D=0.7, c=0.3, T=1.0, pstar1=0.0, pstar=0.0)
    recs = [r for r in _records(_q2(d), params, 'p', 1)
            if not FI.uv_divergent_subgraphs(r[0], d)]
    assert recs
    FI.eqtime_ledger_reset()
    rule = sum(FI.diagram_correlator_x(dd, pv, xs, 0.0, 1.0, 0.7,
                                       spatial_dim=d, comp=cp)
               for dd, pv, cp in recs)
    led = FI.eqtime_ledger_collect()
    assert led['calls'] >= len(recs) and led['max_rel_change'] < 1e-8
    monkeypatch.setenv('SPATIAL_EQTIME', '0')
    fine = sum(FI.diagram_correlator_x(dd, pv, xs, 0.0, 1.0, 0.7,
                                       spatial_dim=d, comp=cp, n_t=44, n_s=48)
               for dd, pv, cp in recs)
    dflt = sum(FI.diagram_correlator_x(dd, pv, xs, 0.0, 1.0, 0.7,
                                       spatial_dim=d, comp=cp)
               for dd, pv, cp in recs)
    scale = np.max(np.abs(fine))
    # the grid converges algebraically TO the rule: d = 1 (22,24) 8e-6 ->
    # (44,48) 4e-8; d = 2 (22,24) 6e-4 -> (44,48) 4e-5 -> (60,64) 1e-5
    tol = 1e-6 if d == 1 else 5e-5
    assert np.max(np.abs(rule - fine)) < tol * scale, (rule, fine)
    assert np.max(np.abs(rule - fine)) < 0.2 * np.max(np.abs(rule - dflt))


def test_k3_tree_equal_time_matches_a_fine_grid(monkeypatch):
    pts = np.zeros((3, 2, 2))
    pts[:, 0, 0] = [0.0, 0.5, 1.0]
    pts[:, 1, 0] = [0.0, -0.4, 0.2]
    kw = dict(k=3, max_ell=0, spatial_points=pts, use_cache=False,
              parallel=False, verbose=False, external_fields=[('p', 1)] * 3)
    out = compute_cumulants(_q2(), **kw)
    assert out['spatial_info']['equal_time']['calls'] >= 1
    rule = np.asarray(out['C_kpoint'])
    monkeypatch.setenv('SPATIAL_EQTIME', '0')
    orig = FI.diagram_kinematic

    def fine(*a, **k):
        k['n_t'], k['n_s'] = 56, 60
        return orig(*a, **k)
    monkeypatch.setattr(FI, 'diagram_kinematic', fine)
    ref = np.asarray(compute_cumulants(_q2(), **kw)['C_kpoint'])
    assert np.max(np.abs(rule - ref)) < 2e-7 * np.max(np.abs(ref)), (rule, ref)


def test_equal_time_switch_and_high_dimension_note(monkeypatch):
    """SPATIAL_EQTIME=0 is the product grid exactly; above the rule's
    dimension the grid is used and the ledger says so."""
    params = dict(mu=1.0, D=0.7, c=0.3, T=1.0, pstar1=0.0, pstar=0.0)
    dd, pv, cp = _records(_q2(), params, 'p', 1)[0]
    xs = np.array([0.0, 0.7])
    monkeypatch.setenv('SPATIAL_EQTIME', '0')
    FI.eqtime_ledger_reset()
    g = sum(FI.diagram_correlator_x(d_, p_, xs, 0.0, 1.0, 0.7, comp=c_)
            for d_, p_, c_ in _records(_q2(), params, 'p', 1))
    assert FI.eqtime_ledger_collect() is None
    # the switch IS the product grid: bit for bit against the dispatch
    # forced off (same platform)
    monkeypatch.delenv('SPATIAL_EQTIME')
    monkeypatch.setattr(FI, '_eqtime_applies', lambda *a, **k: False)
    g_grid = sum(FI.diagram_correlator_x(d_, p_, xs, 0.0, 1.0, 0.7, comp=c_)
                 for d_, p_, c_ in _records(_q2(), params, 'p', 1))
    assert np.array_equal(g, g_grid), (g, g_grid)
    monkeypatch.undo()
    # and main's product-grid value (origin/main c8b869c, macOS): the same
    # algorithm, so it agrees to cross-platform BLAS / libm round-off
    main = np.array([0.04642867565324972, 0.03777113863107688])
    assert np.max(np.abs(g - main) / np.abs(main)) < 1e-13, (g, main)
    monkeypatch.delenv('SPATIAL_EQTIME', raising=False)
    monkeypatch.setattr(FI, 'EQTIME_MAX_DIM', 0)          # force "too big"
    FI.eqtime_ledger_reset()
    FI.diagram_correlator_x(dd, pv, xs, 0.0, 1.0, 0.7, comp=cp)
    led = FI.eqtime_ledger_collect()
    assert led['calls'] == 0 and led['grid_calls'] >= 1
    assert 'not certified' in led['grid_note']
    assert np.all(np.isfinite(g))


# ── accuracy that is not certified is never 'ok' ──────────────────────
def _kpz():
    return (ModelBuilder('kpz', n_populations=0)
            .physical_field('p', spatial_dim=1)
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=0.7, domain='positive')
            .parameter('c', default=0.3, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .mf_residual(lhs='(Dt + mu - D*Laplacian)*p', rhs='0')
            .set_action_text('pt*(Dt(p)+mu*p-D*Lap(p)-(c/2)*Dx(p,0)^2)-T*pt^2')
            .operator_ir().boundary('infinite').initial('stationary').build())


def test_derivative_vertex_loop_is_approximate_with_its_spread():
    """The form factor makes the loop-shrink corner an integrable
    rho^(-1/2) singularity: the product grid converges ~n^-2 at every tau
    (KPZ C-C bubble at q = 0.5, tau = 1: 2.81e-3 / 2.70e-3 / 2.68e-3 at
    22/24, 44/48, 66/72).  Such a result is 'approximate' with the measured
    0.7x-coarser spread, a G6 finding, and strict policy raises."""
    from api.compute import SpatialAccuracyError
    kw = dict(k=2, max_ell=1, external_fields=[('p', 1)] * 2,
              tau_grid=np.array([0.0, 1.0]), spatial_grid=np.array([0.5]),
              use_cache=False, parallel=False, spatial_parallel=False,
              verbose=False)
    with pytest.warns(UserWarning, match='approximate'):
        r = compute_cumulants(_kpz(), **kw)
    assert r['status'] == 'approximate'
    rec, = r['spatial_accuracy']
    assert rec['reason'] == 'derivative_vertex_loop'
    assert 0.0 < rec['coarse_spread'] < 1e-2
    assert any(f.code == 'G6_SPATIAL_DERIVATIVE_LOOP'
               for f in r['analysis'].findings)
    with pytest.raises(SpatialAccuracyError):
        compute_cumulants(_kpz(), policy={'mode': 'strict'}, **kw)
    # a plain-vertex result stays ok
    r2 = compute_cumulants(_q2(), **kw)
    assert r2['status'] == 'ok' and r2['spatial_accuracy'] == []


def test_uv_count_includes_the_form_factor_powers():
    """omega = (V-1) + n_C - dL/2 - r/2 with r the (even part of the)
    form-factor degree along the subgraph's loop directions: the KPZ C-C
    bubble l^2 (l+q)^2 is finite (omega = +1/2); the same record with the
    square of that form factor (r = 8) is divergent."""
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    m = _kpz()
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, m, use_cache=False, verbose=False)
    _, pidx = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = _legs_to_phys_idx([('p', 1)] * 2, pidx)
    vt = [{'weight': float(SR(t['weight']).subs({SR.var('c'): 0.3})),
           'n_phys': t['n_phys'], 'chain': t['chain'], 'mode': t['mode']}
          for t in ft._ns._operator_ir_vertex_terms]
    seen = 0
    for td, _p in build_pipeline_records(ft, m, prop, ext, max_ell=1).get(1, []):
        ff = _formfactor_callable(td, vt, d=1)
        if getattr(ff, 'rcal', None) is None or ff.rcal == 0:
            continue
        dd = diagram_to_cstack(td)
        assert FI.uv_divergent_subgraphs(dd, 1, ff) == []
        if ff.rcal.as_poly().total_degree() >= 4:

            class _Sq:
                rcal = ff.rcal ** 2
            bad = FI.uv_divergent_subgraphs(dd, 1, _Sq())
            assert bad and min(om for _s, om in bad) < 0
            seen += 1
    assert seen


def test_coupled_equal_time_loop_is_marked():
    """The coupled (spectral) loop driver keeps the product grid at tau = 0:
    its result says so."""
    spec = importlib.util.spec_from_file_location(
        'crd', os.path.join(_MODELS, 'coupled_rd_2species_1d.model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.warns(UserWarning, match='approximate'):
        r = compute_cumulants(mod.build(), k=2, max_ell=1,
                              external_fields=[('a', 1), ('a', 1)],
                              tau_grid=np.array([0.0]),
                              spatial_grid=np.array([0.0, 1.0]),
                              use_cache=False, parallel=False,
                              spatial_parallel=False, verbose=False)
    assert r['status'] == 'approximate'
    assert [x['reason'] for x in r['spatial_accuracy']] == [
        'equal_time_coupled_loop']


def test_plain_diagrams_of_a_mixed_model_have_no_form_factor():
    """Allen-Cahn + Model B + KPZ: a diagram whose nodes are all phi^3 is a
    plain diagram (no constant form factor: radial-simplex rule at equal
    times, plain sigma grid, not counted as a derivative loop)."""
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable,
        diagram_form_factor)
    spec = importlib.util.spec_from_file_location(
        'mix', os.path.join(_MODELS, 'combined_allencahn_modelb_kpz_1d.model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.build()
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, m, use_cache=False, verbose=False)
    _, pidx = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    field = str(list(pidx)[0][0])
    ext = _legs_to_phys_idx([(field, 1)] * 2, pidx)
    sub = {SR.var(str(p['name'])): float(p['default'])
           for p in m.get('parameters', []) if p.get('default') is not None
           and not isinstance(p.get('default'), (list, tuple))}
    sub[SR.var('phistar1')] = 0.0                  # the phi* = 0 saddle
    vt = [{'weight': float(SR(t['weight']).subs(sub)), 'n_phys': t['n_phys'],
           'chain': t['chain'], 'mode': t['mode']}
          for t in ft._ns._operator_ir_vertex_terms]
    kinds = set()
    for td, _p in build_pipeline_records(ft, m, prop, ext, max_ell=1).get(1, []):
        R = diagram_form_factor(td, vt, d=1)
        ff = _formfactor_callable(td, vt, d=1)
        if R == 1:
            assert ff is None
            kinds.add('plain')
        elif R == 0:
            assert getattr(ff, 'identically_zero', False)
            kinds.add('zero')
        else:
            assert ff is not None and not getattr(ff, 'identically_zero', False)
            kinds.add('derivative')
    assert kinds >= {'plain', 'derivative'}, kinds


def test_drift_boost_keeps_the_equal_time_record():
    """The Galilean boost runs one drift-free evaluation per tau; the tau = 0
    one (not the first tau) carries the equal-time record."""
    m = (ModelBuilder('advected AC', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('lam', default=0.1, domain='positive')
         .parameter('v', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .mf_residual(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
         .set_action_text('phit*(Dt(phi) + mu*phi - D*Lap(phi) + v*Dx(phi, 0)'
                          ' + lam*phi^3) - T*phit^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('phi', 1)] * 2,
                          tau_grid=np.array([-0.5, 0.0, 0.5]),
                          spatial_grid=np.array([0.0, 0.6]), use_cache=False,
                          parallel=False, spatial_parallel=False, verbose=False)
    eq = r['spatial_info']['equal_time']
    assert eq['calls'] >= 1 and eq['unconverged'] == 0
    assert r['status'] == 'ok'


def test_monte_carlo_integrator_is_approximate(monkeypatch):
    monkeypatch.setenv('SPATIAL_INTEGRATOR', 'mc')
    monkeypatch.setenv('SPATIAL_MC_N', '20000')
    with pytest.warns(UserWarning, match='approximate'):
        r = compute_cumulants(_q2(), k=2, max_ell=1,
                              external_fields=[('p', 1)] * 2,
                              tau_grid=np.array([0.5]),
                              spatial_grid=np.array([0.0]), use_cache=False,
                              parallel=False, spatial_parallel=False,
                              verbose=False)
    assert r['status'] == 'approximate'
    assert 'monte_carlo_integrator' in [x['reason'] for x in r['spatial_accuracy']]

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
            .equation(lhs='(Dt + mu - D*Laplacian)*p', rhs='-c*p^2')
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
    g = FI.diagram_correlator_x(dd, pv, xs, 0.0, 1.0, 0.7, comp=cp)
    assert FI.eqtime_ledger_collect() is None
    monkeypatch.delenv('SPATIAL_EQTIME')
    monkeypatch.setattr(FI, 'EQTIME_MAX_DIM', 0)          # force "too big"
    FI.eqtime_ledger_reset()
    FI.diagram_correlator_x(dd, pv, xs, 0.0, 1.0, 0.7, comp=cp)
    led = FI.eqtime_ledger_collect()
    assert led['calls'] == 0 and led['grid_calls'] >= 1
    assert 'not certified' in led['grid_note']
    assert np.all(np.isfinite(g))

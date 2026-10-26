"""The Pade order cache of the delay route: an automatic order search is run once
per model, parameter point and option set; later entries and lag grids replay
its order (one run, the search's truncation tolerance) instead of searching."""
import warnings

import numpy as np
import pytest

from api import compute_cumulants
from api import delay as dl
from api.noise import Gaussian
from api.sde import SDE

TAUS = np.array([0.0, 0.5, 1.0, 2.0])
OPTS = dict(tol=1e-6)


def _model(name='delay order cache', d=0.5):
    return (SDE(name).parameter('a', default=1.0).parameter('c', default=-2.0)
            .parameter('d', default=d, domain='positive').parameter('eps', default=0.02)
            .parameter('D', default=0.1).field('x')
            .drift('x', '-a*x + c*x(t - d) - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def _run(m, taus=TAUS, ext=(('x', 1), ('x', 1)), **delay):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(m, k=2, max_ell=1, external_fields=list(ext), tau_grid=taus,
                                 parallel=False, verbose=False, use_cache=False,
                                 taylor_order=4, delay={**OPTS, **delay})


def _same(a, b):
    for ell in a['C_tau_by_ell']:
        assert np.array_equal(a['C_tau_by_ell'][ell], b['C_tau_by_ell'][ell])
    assert np.array_equal(a['C_tau'], b['C_tau'])


def test_second_call_replays_the_order_bit_identically():
    m = _model()
    first = _run(m, order_cache=False)
    assert first['delay']['orders_run'] == [4, 8] and not first['delay']['order_from_cache']
    fresh = _run(m)                                   # searches, fills the cache
    assert not fresh['delay']['order_from_cache']
    hit = _run(m)
    info = hit['delay']
    assert info['order_from_cache'] and info['orders_run'] == [8] and info['n'] == 8
    assert info['est_rel_error'] == first['delay']['est_rel_error']
    assert info['target_met'] is True and info['est_by_order'] == first['delay']['est_by_order']
    assert info['truncation_tol'] == first['delay']['truncation_tol']
    _same(first, hit)                                 # the search's own result, bit for bit
    _same(fresh, hit)


def test_other_lag_grid_reuses_the_order_and_equals_its_own_search():
    m = _model()
    _run(m)
    taus = np.array([0.25, 0.75, 3.0])
    hit = _run(m, taus=taus)
    own = _run(m, taus=taus, order_cache=False)
    assert hit['delay']['order_from_cache'] and not own['delay']['order_from_cache']
    assert hit['delay']['n'] == own['delay']['n']
    _same(hit, own)


def test_cache_is_keyed_by_parameter_point_and_options():
    m = _model()
    _run(m)
    assert not _run(_model(d=0.6))['delay']['order_from_cache']       # another delay
    assert not _run(m, tol=1e-5)['delay']['order_from_cache']         # another target
    assert not _run(m, n_max=8)['delay']['order_from_cache']          # another cap
    assert _run(m)['delay']['order_from_cache']                       # the first key still hits
    m2 = (SDE('delay order cache eps').parameter('a', default=1.0).parameter('c', default=-2.0)
          .parameter('d', default=0.5, domain='positive').parameter('eps', default=0.03)
          .parameter('D', default=0.1).field('x')
          .drift('x', '-a*x + c*x(t - d) - eps*x^3')
          .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    assert not _run(m2)['delay']['order_from_cache']                  # another model / point


def test_opt_out_and_clear():
    m = _model()
    _run(m)
    assert not _run(m, order_cache=False)['delay']['order_from_cache']
    assert _run(m)['delay']['order_from_cache']
    dl.clear_order_cache()
    assert not _run(m)['delay']['order_from_cache']


def test_fixed_order_never_touches_the_cache():
    m = _model()
    r = _run(m, order=8)
    assert r['delay']['orders_run'] == [8] and not r['delay']['order_from_cache']
    assert not _run(m)['delay']['order_from_cache']


def test_capped_search_is_pinned_by_the_cache():
    """A wall-capped search stops where the machine's speed put it; the replay
    does not re-decide (the delayed-alpha bit-identity test needed this)."""
    m = _model()
    first = _run(m, wall_max=0.0)                    # any prediction exceeds the cap
    assert first['delay']['orders_run'] == [4] and first['status'] == 'target_missed'
    again = _run(m, wall_max=0.0)
    assert again['delay']['order_from_cache'] and again['delay']['n'] == 4
    assert again['status'] == 'target_missed'
    _same(first, again)


def test_unfingerprinted_model_is_not_cached():
    m = dict(_model())
    m['spec_signature'] = None
    _run(m)
    assert not _run(m)['delay']['order_from_cache']


def test_strictly_proper_approximant_is_replayed(monkeypatch):
    """A hard delay at a vertex leg that Phase J cannot dress falls back on the
    [n-1/n] approximant; the cache remembers the approximant, so the replay
    runs it directly (one run, not the (n, n) pass first)."""
    from engine.integration.time_domain import final_integral as FI
    from tests.test_delay_nonlinear import EXT, TAU, _product_model
    monkeypatch.setattr(FI, 'DRESS_CONV_KERNELS', False)
    m = _product_model('dnl order cache proper')
    kw = dict(k=2, max_ell=1, external_fields=EXT, tau_grid=TAU, parallel=False,
              verbose=False, use_cache=False, delay={'tol': 1e-3})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        first = compute_cumulants(m, **kw)
        hit = compute_cumulants(m, **kw)
    assert first['delay']['approximant'] == 'pade_proper' and not first['delay']['order_from_cache']
    assert hit['delay']['order_from_cache'] and hit['delay']['approximant'] == 'pade_proper'
    assert hit['delay']['orders_run'] == [first['delay']['n']]
    _same(first, hit)

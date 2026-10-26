"""The Pade order cache of the delay route: an automatic order search is run once
per entry at a model, parameter point and option set; the same entry replays it
(one run), another entry starts its own search at the cached order and measures
its own estimate."""
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
    assert first['delay']['orders_run'] == [4, 8] and first['delay']['order_source'] == 'search'
    fresh = _run(m)                                   # searches, fills the cache
    assert fresh['delay']['order_source'] == 'search'
    hit = _run(m)
    info = hit['delay']
    assert info['order_source'] == 'replay' and info['orders_run'] == [8] and info['n'] == 8
    assert info['est_rel_error'] == first['delay']['est_rel_error']
    assert info['target_met'] is True and info['est_by_order'] == first['delay']['est_by_order']
    assert info['truncation_tol'] == first['delay']['truncation_tol']
    _same(first, hit)                                 # the search's own result, bit for bit
    _same(fresh, hit)


SHORT, MID = np.array([0.0, 0.05, 0.1]), np.array([0.0, 0.5, 1.0, 2.0])


def test_other_entry_measures_its_own_estimate_and_escalates():
    """The estimate is relative to max |C| of the entry's lag grid: at tol = 5e-7 the
    short grid settles at n = 8 (1.6e-7), the mid grid needs n = 16 (8.3e-7 at n = 8).
    The mid entry must not inherit the short entry's estimate or order."""
    m = _model()
    short = _run(m, taus=SHORT, tol=5e-7)
    assert short['delay']['n'] == 8 and short['delay']['target_met'] is True
    mid = _run(m, taus=MID, tol=5e-7)
    own = _run(m, taus=MID, tol=5e-7, order_cache=False)
    info = mid['delay']
    assert info['order_source'] == 'verified' and info['order_from_cache']
    assert info['n'] == 16 and own['delay']['n'] == 16            # escalated past the cache
    assert info['est_by_order'][8] == pytest.approx(own['delay']['est_by_order'][8], rel=1e-3)
    assert info['est_by_order'][8] > 5e-7 > short['delay']['est_rel_error']
    assert info['est_rel_error'] == info['est_by_order'][16]       # its own, not the short grid's
    np.testing.assert_allclose(mid['C_tau'], own['C_tau'], rtol=1e-6, atol=0)
    # the entries now each replay their own search
    assert _run(m, taus=MID, tol=5e-7)['delay']['order_source'] == 'replay'
    assert _run(m, taus=SHORT, tol=5e-7)['delay']['n'] == 8


def test_other_entry_that_agrees_at_the_cached_order_stays_there():
    m = _model()
    _run(m, taus=MID, tol=1e-6)                    # settles at 8 (8.3e-7)
    r = _run(m, taus=np.array([3.0, 5.0, 8.0]), tol=1e-6)
    own = _run(m, taus=np.array([3.0, 5.0, 8.0]), tol=1e-6, order_cache=False)
    assert r['delay']['order_source'] == 'verified' and r['delay']['n'] == own['delay']['n'] == 8
    assert r['delay']['target_met'] is True and r['delay']['orders_run'] == [4, 8]
    assert r['delay']['est_rel_error'] == pytest.approx(own['delay']['est_rel_error'], rel=1e-3)


def test_cache_is_keyed_by_parameter_point_and_options():
    m = _model()
    _run(m)
    assert _run(_model(d=0.6))['delay']['order_source'] == 'search'       # another delay
    assert _run(m, tol=1e-5)['delay']['order_source'] == 'search'     # another target
    assert _run(m, n_max=8)['delay']['order_source'] == 'search'      # another cap
    assert _run(m)['delay']['order_source'] == 'replay'               # the first key still hits
    m2 = (SDE('delay order cache eps').parameter('a', default=1.0).parameter('c', default=-2.0)
          .parameter('d', default=0.5, domain='positive').parameter('eps', default=0.03)
          .parameter('D', default=0.1).field('x')
          .drift('x', '-a*x + c*x(t - d) - eps*x^3')
          .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    assert _run(m2)['delay']['order_source'] == 'search'              # another model / point


def test_opt_out_and_clear():
    m = _model()
    _run(m)
    assert _run(m, order_cache=False)['delay']['order_source'] == 'search'
    assert _run(m)['delay']['order_source'] == 'replay'
    dl.clear_order_cache()
    assert _run(m)['delay']['order_source'] == 'search'


def test_fixed_order_never_touches_the_cache():
    m = _model()
    r = _run(m, order=8)
    assert r['delay']['orders_run'] == [8] and r['delay']['order_source'] == 'search'
    assert _run(m)['delay']['order_source'] == 'search'


def test_capped_search_is_pinned_by_the_cache():
    """A wall-capped search stops where the machine's speed put it; the replay
    does not re-decide (the delayed-alpha bit-identity test needed this)."""
    m = _model()
    first = _run(m, wall_max=0.0)                    # any prediction exceeds the cap
    assert first['delay']['orders_run'] == [4] and first['status'] == 'target_missed'
    again = _run(m, wall_max=0.0)
    assert again['delay']['order_source'] == 'replay' and again['delay']['n'] == 4
    assert again['status'] == 'target_missed'
    _same(first, again)


def test_unfingerprinted_model_is_not_cached():
    m = dict(_model())
    m['spec_signature'] = None
    _run(m)
    assert _run(m)['delay']['order_source'] == 'search'


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
    assert first['delay']['approximant'] == 'pade_proper' and first['delay']['order_source'] == 'search'
    assert hit['delay']['order_source'] == 'replay' and hit['delay']['approximant'] == 'pade_proper'
    assert hit['delay']['orders_run'] == [first['delay']['n']]
    _same(first, hit)

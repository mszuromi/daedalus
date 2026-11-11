"""The Pade order cache of the delay route: an automatic order search is run once
per entry at a model, parameter point and option set; the same entry replays it
(one run), another entry starts its own search at the cached order and measures
its own estimate."""
import warnings

import numpy as np
import pytest

from daedalus.api import compute_cumulants
from daedalus.api import delay as dl
from daedalus.api.noise import Gaussian
from daedalus.api.sde import SDE

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
    # (n = 4 is its error estimate)
    assert r['delay']['orders_run'] == [4, 8] and r['delay']['order_source'] == 'search'
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


def test_the_order_does_not_depend_on_the_wall_time(monkeypatch):
    """No wall-time cap by default: the estimate alone picks the order, so a
    slow or loaded machine settles on the order a quiet one does.  An explicit
    wall_max is a reported stop that says it depends on the machine."""
    assert dl.DEFAULTS['wall_max'] is None
    m = _model('delay order no wall cap')
    quiet = _run(m, order_cache=False)
    monkeypatch.setattr(dl, '_predicted_growth', lambda *a, **k: (1e9, 'a loaded machine'))
    busy = _run(m, order_cache=False)
    assert busy['delay']['n'] == quiet['delay']['n'] == 8
    assert busy['delay']['target_met'] is True and busy['status'] == 'ok'
    _same(quiet, busy)
    capped = _run(m, order_cache=False, wall_max=900.0)
    info = capped['delay']
    assert info['n'] == 4 and capped['status'] == 'target_missed'
    assert "speed and load" in info['cap'] and info['stop_reason'] == info['cap']


def test_fixed_order_is_estimated_against_the_order_below():
    m = _model('delay fixed order estimate')
    with pytest.warns(dl.DelayOrderWarning, match='estimated against n = 4'):
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
                              tau_grid=TAUS, parallel=False, verbose=False, use_cache=False,
                              taylor_order=4, delay=dict(order=8, tol=1e-12))
    info = r['delay']
    assert info['orders_run'] == [4, 8] and info['estimate_order'] == 4 and info['n'] == 8
    assert info['target_met'] is False and r['status'] == 'target_missed'
    assert info['est_rel_error'] == info['est_by_order'][8] > 1e-12
    # the returned run is the one the pinned order gave without an estimate
    alone = _run(m, order=8, estimate=False)
    assert alone['delay']['orders_run'] == [8] and alone['delay']['est_by_order'] == {}
    assert alone['delay']['target_met'] is None and alone['status'] == 'ok'
    _same(r, alone)
    # the estimate is the search's: the same number at the same pair of orders
    met = _run(m, order=8)
    auto = _run(m, order_cache=False)
    assert met['delay']['target_met'] is True and met['status'] == 'ok'
    assert met['delay']['est_rel_error'] == pytest.approx(auto['delay']['est_by_order'][8],
                                                          rel=1e-3)


def test_pinned_order_of_a_long_delay_reports_its_error():
    # one-stage delayed negative feedback at d = 5: the one-loop term moves by
    # ~10 % from n = 8 to n = 16, and a pinned n = 8 used to report no
    # estimate.  Its estimate bounds that move.
    from tests.test_delay_loops_memory_61 import _negative_feedback
    m = _negative_feedback('hard')
    kw = dict(k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
              tau_grid=np.array([-1.0, 0.0, 1.0]), parallel=False, verbose=False)
    with pytest.warns(dl.DelayOrderWarning, match='at n = 8'):
        r8 = compute_cumulants(m, delay=dict(order=8), **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r16 = compute_cumulants(m, delay=dict(order=16, estimate=False), **kw)
    move = (np.max(np.abs(np.asarray(r8['C_tau']) - np.asarray(r16['C_tau'])))
            / np.max(np.abs(np.asarray(r16['C_tau']))))
    l8, l16 = (np.real(np.asarray(r['C_tau_by_ell'][1])) for r in (r8, r16))
    assert np.max(np.abs(l8 - l16)) > 0.05 * np.max(np.abs(l16))
    assert r8['status'] == 'target_missed' and r8['delay']['est_rel_error'] > move


def test_unfingerprinted_model_is_not_cached():
    m = dict(_model())
    m['spec_signature'] = None
    _run(m)
    assert _run(m)['delay']['order_source'] == 'search'


def test_strictly_proper_approximant_is_replayed(monkeypatch):
    """A hard delay at a vertex leg that Phase J cannot dress falls back on the
    [n-1/n] approximant; the cache remembers the approximant, so the replay
    runs it directly (one run, not the (n, n) pass first)."""
    from daedalus.engine.integration.time_domain import final_integral as FI
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


def test_discard_run_is_its_own_entry_with_the_same_order_and_estimate():
    """A k >= 3 discard run reads the controller's estimate at the same sample points but is
    called with ``points``, not a lag grid: another entry.  Started from the cache it takes
    two orders (the cached one and the one before), reports the estimate of the pair it ran
    (est_by_order holds the orders that HAVE an estimate), and agrees with the search."""
    m = (SDE('delay order cache k3').parameter('a', default=1.0).parameter('c', default=-2.0)
         .parameter('d', default=0.5, domain='positive').parameter('b', default=0.3)
         .parameter('D', default=0.1).field('x').drift('x', '-a*x + c*x(t - d) - b*x^2')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    kw = dict(k=3, max_ell=0, external_fields=[('x', 1)] * 3, parallel=False, verbose=False,
              use_cache=False, delay={'tol': 5e-5})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = compute_cumulants(m, tau_grid=np.array([0.0]), **kw)
        b = compute_cumulants(m, points=[(0.0, 0.0, 0.0)], discard=True, **kw)
    assert a['delay']['order_source'] == 'search' and b['delay']['order_source'] == 'verified'
    assert b['delay']['n'] == a['delay']['n']
    n = a['delay']['n']
    assert b['delay']['est_by_order'][n] == pytest.approx(a['delay']['est_by_order'][n], rel=1e-3)
    assert b['delay']['est_rel_error'] == pytest.approx(a['delay']['est_rel_error'], rel=1e-3)

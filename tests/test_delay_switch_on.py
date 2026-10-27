"""Switch-on transients (``t_min``) of a model with a delay kernel.

The fluctuations and the history of every delayed field are zero before ``t_min``.  The exact delayed tree
(``daedalus.api._delay_exact``) is the stationary covariance, so a transient takes the Pade approximant for the tree too: the
loops and the tree converge together as the order grows, with the slower rate of the first delay after ``t_min``
(measured: 7e-3, 2e-3, 2e-4 relative at n = 4, 8, 16 for tau < d).  The order controller reports that honestly
(``target_met``, ``status``).  References: the closed-form Ornstein-Uhlenbeck transient inside the first delay (the
delayed term is still zero), and the stationary exact tree far from ``t_min``.
"""
import math
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

from daedalus.api import compute_cumulants                                      # noqa: E402
from daedalus.api.delay import DelayOrderWarning                                # noqa: E402
from daedalus.api._delay_exact import ExactTree                                 # noqa: E402
from tests.test_delay_kernels import A_, D_, _exact_ou, _ou           # noqa: E402

EXT = [('x', 1), ('x', 1)]
D_DELAY = 0.5


def _run(model, times, t_min=0.0, ell=0, **kw):
    kw.setdefault('delay', {'order': 8})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=2, max_ell=ell, external_fields=EXT, t_min=t_min,
                                 times=np.asarray(times, dtype=float), parallel=False,
                                 verbose=False, use_cache=False, **kw)


def test_first_delay_window_matches_the_ou_transient_and_converges():
    """For t - t_min < d the delayed term is zero: Var = (D/a)(1 - exp(-2 a s))."""
    m = _ou('delay switch-on ou', d=D_DELAY, eps=0.0)
    ts = [[0.1, 0.1], [0.25, 0.25], [0.4, 0.4]]
    ref = np.array([(D_ / A_) * (1 - math.exp(-2 * A_ * t[0])) for t in ts])
    err = {n: np.max(np.abs(np.real(_run(m, ts, delay={'order': n})['C_times']) - ref) / ref)
           for n in (4, 8, 16)}
    assert err[16] < 1e-3 and err[16] < err[8] < err[4]


def test_far_from_t_min_the_transient_is_the_stationary_exact_tree():
    m = _ou('delay switch-on stationary', d=D_DELAY, eps=0.0)
    r = _run(m, [[0.0, 0.0], [0.5, 0.0], [1.5, 0.0]], t_min=-40.0, delay={'order': 16})
    want = _exact_ou(np.array([0.0, 0.5, 1.5]), D_DELAY)
    got = np.real(r['C_times'])
    # t1 >= t2 here: C(t1, t2) = C(tau = t1 - t2)
    assert np.max(np.abs(got - want)) <= 2e-4 * np.max(np.abs(want))


def test_record_is_honest_about_the_pade_tree_and_the_target():
    m = _ou('delay switch-on record', d=D_DELAY)
    ts = [[0.1, 0.1], [0.4, 0.4], [2.0, 2.0]]
    r = _run(m, ts, ell=1, delay={'tol': 1e-8})
    info = r['delay']
    assert info['tree'] == 'pade' and 'switch-on transient' in info['tree_note']
    assert info['target_met'] is False and r['status'] == 'target_missed'
    assert info['est_rel_error'] > 1e-8 and info['orders_run'][-1] == info['n']
    assert np.max(np.abs(r['C_times_by_ell'][1])) > 0
    assert np.allclose(np.real(r['C_times']), sum(np.real(v) for v in r['C_times_by_ell'].values()),
                       rtol=1e-12, atol=0)
    with pytest.warns(DelayOrderWarning):
        compute_cumulants(m, k=2, max_ell=1, external_fields=EXT, t_min=0.0,
                          times=np.asarray(ts), parallel=False, verbose=False, use_cache=False,
                          delay={'tol': 1e-8})


def test_a_loose_target_is_met_by_the_search():
    m = _ou('delay switch-on loose', d=D_DELAY)
    r = _run(m, [[0.4, 0.4], [2.0, 2.0]], ell=1, delay={'tol': 5e-3})
    assert r['delay']['target_met'] is True and r['status'] == 'ok'
    assert r['delay']['n'] in (8, 16)


def test_initial_covariance_with_a_delay_is_refused_by_name():
    from daedalus.engine.integration.time_domain.statespace import SwitchOnUnsupported
    m = _ou('delay switch-on c0', d=D_DELAY)
    with pytest.raises(SwitchOnUnsupported, match='initial_covariance is not available'):
        _run(m, [[0.4, 0.4]], initial_covariance=np.array([[0.1]]))

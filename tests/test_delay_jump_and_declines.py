"""
Three honesty properties of the delay route (``daedalus.api.delay``) and the
Gaussian closure's delays (``daedalus.api.hartree``).

* The covariance's jump at ``|tau| = d``: a linear Hawkes unit whose kernel is
  ``Kernel.delay(d) * Kernel.exponential(tau)`` has a covariance density that
  jumps by ``w Lambda / tau`` at ``|tau| = d``.  Every Pade order smooths it over
  ``~d / n``, and lags next to it carry about half the jump at every order,
  which no comparison of two orders sees.  The record ``result['delay']['jump']``
  fits the jump, gives the error per lag and marks the lags in the layer, and
  ``est_rel_error`` covers the actual error (the exact delayed tree is the
  reference: one delay).
* Declined orders: an order of the automatic search whose realization the
  state-space route declines as ill-conditioned is replaced by a neighbouring
  order the route takes (``result['delay']['declined_orders']``).
* The closure keeps the all-pass (n, n) image for a delayed filter of a spike
  train: its reported fields are dynamical ones, and the all-pass image
  reproduces the delay of an exogenous input exactly at every order.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.api import compute_cumulants                                      # noqa: E402
from daedalus.api import delay as D                                             # noqa: E402
from daedalus.api.delay import Kernel                                           # noqa: E402
from daedalus.api.noise import Gaussian                                         # noqa: E402
from daedalus.api.sde import SDE                                                # noqa: E402

EXT = [('n', 1), ('n', 1)]
MU, W, DL, TK = 0.6, -0.2, 1.3, 0.5
TAUS = np.array([0.0, 0.3, 0.8, 1.29, 1.31, 2.0, 3.0])


def _hawkes(mu=MU, w=W, d=DL, tk=TK, name='jump layer unit'):
    return (SDE(name).parameter('mu', default=mu).parameter('w', default=w)
            .parameter('d', default=d).parameter('tk', default=tk)
            .define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tk'))
            .point_process('n', 'mu + w*Conv(g, n)')).build()


def _run(model, taus=TAUS, **delay):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = compute_cumulants(model, k=2, max_ell=0, external_fields=EXT, tau_grid=taus,
                              verbose=False, delay=delay or None)
    return r, [w_ for w_ in rec if issubclass(w_.category, D.DelayOrderWarning)]


@pytest.fixture(scope='module')
def exact_tree():
    r, _ = _run(_hawkes())
    assert r['delay']['tree'] == 'exact'
    return np.real(np.asarray(r['C_tau']))


# ── the Pade step and ramp responses ────────────────────────────────────────

@pytest.mark.parametrize('m,n', [(3, 4), (7, 8), (15, 16), (19, 20)])
def test_step_and_ramp_responses(m, n):
    """``sigma(x) -> Theta(x - 1)`` away from the jump, ``sigma(0+) = 0`` for the
    strictly proper image, ``rho' = sigma``; n = 20 takes the mpmath path."""
    x = np.array([1e-9, 0.3, 0.7, 1.4, 2.0, 8.0])
    s, r = D.pade_step_ramp(m, n, x)
    assert abs(s[0]) < 1e-6
    assert abs(s[-1] - 1.0) < 1e-6 and abs(r[-1] - 7.0) < 1e-5
    h = 1e-5
    for x0 in (0.4, 0.95, 1.2):
        sp, rp = D.pade_step_ramp(m, n, [x0 - h, x0, x0 + h])
        assert (rp[2] - rp[0]) / (2 * h) == pytest.approx(sp[1], abs=1e-5)
    # the error near the jump does not shrink with n: about half the jump
    s1, _ = D.pade_step_ramp(m, n, [1.0 - 0.1 / n, 1.0 + 0.1 / n])
    assert 0.3 < s1[0] < 0.7 and 0.3 < s1[1] < 0.7


# ── the jump at |tau| = d ───────────────────────────────────────────────────

@pytest.mark.parametrize('n', [8, 16])
def test_jump_layer_estimate_covers_the_error_on_a_pade_tree(n, exact_tree):
    r, warns = _run(_hawkes(), tree='pade', order=n)
    dl = r['delay']
    j = dl['jump']
    v = np.real(np.asarray(r['C_tau']))
    scale = float(np.max(np.abs(v)))
    actual = np.abs(v - exact_tree)
    # the error sits next to the jump (about half of it) ...
    assert np.argmax(actual) in (3, 4) and actual[3] > 0.3 * abs(W * MU / (1 - W) / TK)
    # ... and the estimate covers it (it did not: 0.17 at n = 16 against 0.78)
    assert dl['est_rel_error'] >= np.max(actual) / scale
    assert dl['est_rel_error'] == pytest.approx(j['est_rel_error'])
    assert j['est_rel_error'] < 2.0 * np.max(actual) / scale
    # the fitted jump is w Lambda / tau, the layer is the two lags next to d
    lam = MU / (1 - W)
    fit = j['lags'][0]['fit']['+']
    assert j['lags'][0]['d'] == pytest.approx(DL) and j['lags'][0]['kernels'] == ['z_g']
    assert fit['jump'] == pytest.approx(W * lam / TK, rel=0.12)
    assert list(TAUS[np.array(j['near_jump'])]) == [1.29, 1.31]
    err = np.array(j['error_by_tau'])
    assert np.all(err[3:5] >= actual[3:5])
    # visible, and the result is marked
    assert r['status'] == 'target_missed'
    assert any('1.29, 1.31' in str(w_.message) and 'jumps at |tau| = 1.3' in str(w_.message)
               for w_ in warns)


def test_jump_layer_with_negative_lags_and_both_sides():
    """A symmetric grid: both sides are fitted, the layer is marked at -d and
    +d, and the estimate still covers the error."""
    taus = np.linspace(-2.0, 2.0, 41)
    ex, _ = _run(_hawkes(), taus=taus)
    r, _ = _run(_hawkes(), taus=taus, tree='pade', order=8)
    j = r['delay']['jump']
    v = np.real(np.asarray(r['C_tau']))
    actual = np.max(np.abs(v - np.real(np.asarray(ex['C_tau'])))) / np.max(np.abs(v))
    assert r['delay']['est_rel_error'] >= actual
    fit = j['lags'][0]['fit']
    assert set(fit) == {'+', '-'}
    assert fit['+']['jump'] == pytest.approx(fit['-']['jump'], rel=1e-6)
    near = taus[np.array(j['near_jump'])]
    assert np.all(np.abs(np.abs(near) - DL) <= D.JUMP_LAYER * DL / 8 + 1e-12)
    assert np.any(near < 0) and np.any(near > 0)


def test_no_jump_record_without_an_approximated_part():
    """The exact tree (one delay, no loops) has no approximated part, and a
    delayed filter of a continuous field puts no jump into the covariance."""
    r, _ = _run(_hawkes())
    assert r['delay']['jump'] is None
    ou = (SDE('delayed filter of a continuous field').parameter('a', default=1.0)
          .parameter('c', default=0.5).parameter('d', default=0.8).parameter('tk', default=0.4)
          .define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tk'))
          .field('x').drift('x', '-a*x + c*Conv(g, x)')
          .noise('xi', Gaussian(var='0.5'), couples={'x': 1})).build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(ou, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                              tau_grid=np.array([0.0, 0.79, 0.81]), verbose=False,
                              delay={'tree': 'pade', 'order': 8})
    assert r['delay']['proper_kernels'] == [] and r['delay']['jump'] is None


# ── declined orders ─────────────────────────────────────────────────────────

def test_search_replaces_a_declined_order_by_a_neighbour(monkeypatch):
    """Order 8 of the search declined (simulated): the neighbour 7 runs in its
    place, the ladder goes on to 16, and the record says so."""
    real = D._declined_reasons

    def fake(r):
        name = str((r.get('config') or {}).get('model_name') or '')
        if '[delay Pade 8]' in name:
            return {'realization_ill_conditioned': 2}
        return real(r)

    monkeypatch.setattr(D, '_declined_reasons', fake)
    r, _ = _run(_hawkes(name='declined order unit'), tree='pade', order_cache=False)
    dl = r['delay']
    assert dl['orders_run'] == [4, 7, 16]
    rec = dl['declined_orders'][8]
    assert rec['replaced_by'] == 7 and rec['reasons'] == {'realization_ill_conditioned': 2}
    assert 7 in rec['tried']
    # a pinned order runs as given
    r2, _ = _run(_hawkes(name='declined order unit'), tree='pade', order=8, order_cache=False)
    assert r2['delay']['n'] == 8 and r2['delay']['declined_orders'] == {}


@pytest.mark.slow
def test_two_delay_model_stays_on_the_statespace_route():
    """One field, two hard delays, Ito multiplicative noise, one loop: order 16
    of the default search is declined as ill-conditioned (its subsets would run
    on mode sums, the spectra's quadrature for minutes); order 15 runs in its
    place on the state-space route, and the spectra are exact."""
    P = dict(mu=0.9, a=1.4, b1=-0.6, d1=0.7, b2=0.35, d2=1.9, s0=0.3, s1=0.55)
    m = SDE('two delays, multiplicative noise')
    for k_, v_ in P.items():
        m = m.parameter(k_, default=v_)
    model = (m.field('x').drift('x', 'mu - a*x + b1*x(t - d1) + b2*x(t - d2)')
             .noise('w0', Gaussian(var='s0^2'), into={'x': 1})
             .noise('w1', Gaussian(var='s1^2'), into={'x': 'x'})).build()
    D.clear_order_cache()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(model, k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
                              tau_grid=np.array([0.0, 0.7]), verbose=False,
                              spectra={'omega': np.array([0.0, 0.8]), 'T': np.array([3.0])})
    dl = r['delay']
    assert dl['declined_orders'][16]['replaced_by'] == 15
    assert dl['n'] == 15 and dl['statespace_share_by_order'][15] == 1.0
    assert set(r['spectra']['method_by_ell'].values()) == {'exact'}
    assert 'advisory' not in (r['spectra'].get('quad_cost') or {})


# ── the closure keeps the all-pass image ────────────────────────────────────

def test_hartree_keeps_the_all_pass_image_for_a_spike_train_input():
    """A Poisson input through ``delay * exponential`` into a cubic unit: the
    delay of an exogenous input is invisible in the driven field's stationary
    statistics; the all-pass (n, n) image reproduces that at every order, the
    strictly proper one would not (1e-3 at n = 8)."""
    from daedalus.api.hartree import hartree

    def model(delayed):
        m = SDE('Poisson input' + (', delayed' if delayed else ''))
        for k_, v_ in dict(ts=0.5, lam=2.0, p=0.7, w=1.3, mu=1.0, eps=0.4, d=0.9).items():
            m.parameter(k_, default=v_, domain='positive')
        m.field('v')
        g = Kernel.exponential('ts')
        m.define_kernel('g', kernel=Kernel.delay('d') * g if delayed else g)
        m.point_process('n', 'lam', jump_size='p')
        m.drift('v', '-mu*v - eps*v^3 + w*Conv(g, n)')
        return m.build()

    ref = hartree(model(False), warn=False)
    tau = np.array([0.0, 0.3, 1.0, 2.5])
    for order in (5, 8):
        r = hartree(model(True), warn=False, delay={'order': order})
        assert r.info['delay']['approximant'] == 'pade'
        assert r.info['delay']['proper_kernels'] == []
        assert np.allclose(r.cov, ref.cov, rtol=1e-12, atol=0)
        assert np.allclose(r.C_matrix(tau), ref.C_matrix(tau), rtol=1e-12, atol=1e-14)

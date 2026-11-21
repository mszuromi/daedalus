"""
Delayed first-order filters of spike trains on the Pade route
(``daedalus.api.delay``), and the estimate that checks it.

* A kernel ``Kernel.delay(d) * Kernel.exponential(tau)`` filtering a spike
  train: the all-pass (n, n) image is ``(-1)^n + (strictly proper)``, its
  direct term passes through the filter, and the approximated kernel starts at
  ``(-1)^n / tau`` instead of 0.  A point-process covariance holds
  ``Lambda K(tau)`` at short lags, so ``C(0)`` would carry an error
  ``(-1)^n w Lambda / tau`` at every order.  These kernels take the strictly
  proper ``[n-1/n]`` approximant: ``C(0)`` converges with ``n`` to the exact
  delayed tree and to an independent reference (the Hawkes integral equation
  for the covariance density, solved on a grid here).
* The doubling estimate compares orders of one parity; one order of the other
  parity checks it (``result['delay']['parity_probe']``), so a two-delay
  model (no exact tree) reports an estimate at least of the size of its
  actual error.
* A nonlinear Hawkes unit with such a kernel runs with loops (its Pade tree
  converges to the exact tree, so the tree-vs-exact guard lets it through).
* The remedy of a refusal "no Pade order is usable" says when a higher
  ``n_max`` cannot help.
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
from daedalus.api.delay import Kernel                                           # noqa: E402
from daedalus.api.noise import Gaussian                                         # noqa: E402
from daedalus.api.sde import SDE                                                # noqa: E402

EXT = [('n', 1), ('n', 1)]


def _hawkes_covariance(mu, kernel, taus, h, T):
    """Covariance density ``c(tau)`` (``tau >= 0``, the delta at 0 left out)
    of a stationary linear Hawkes unit with rate ``mu + int K(t - s) dN(s)``:
    ``c(t) = Lambda K(t) + int_0^inf K(t + s) c(s) ds + int_0^t K(t - u) c(u) du``,
    trapezoid rule on ``[0, T]`` with step ``h``, linear interpolation."""
    grid = np.arange(0.0, T + h / 2, h)
    n = grid.size
    lam = mu / (1.0 - _integral(kernel, T))
    wts = np.full(n, h)
    wts[0] = wts[-1] = h / 2
    tp = grid[:, None] + grid[None, :]
    tm = grid[:, None] - grid[None, :]
    A = kernel(tp) * wts[None, :]
    lower = np.tril(np.ones((n, n)))
    wl = np.where(np.eye(n, dtype=bool), h / 2, h) * lower
    wl[:, 0] = np.where(np.arange(n) > 0, h / 2, 0.0)
    A += kernel(np.where(tm >= 0, tm, 0.0)) * wl
    c = np.linalg.solve(np.eye(n) - A, lam * kernel(grid))
    return np.interp(taus, grid, c), lam


def _integral(kernel, T, m=400001):
    t = np.linspace(0.0, T, m)
    v = kernel(t)
    return float(np.sum((v[1:] + v[:-1]) * np.diff(t)) / 2)


def _reference(mu, kernel, taus, T=16.0):
    """Richardson extrapolation of ``_hawkes_covariance`` in the step."""
    c1, lam = _hawkes_covariance(mu, kernel, taus, 0.01, T)
    c2, _ = _hawkes_covariance(mu, kernel, taus, 0.005, T)
    return 2 * c2 - c1, lam


def _delay_exp(w, d, tk):
    return lambda t: np.where(t > d, w / tk * np.exp(-(np.asarray(t) - d) / tk), 0.0)


def _delay_gamma2(w, d, m):
    # Kernel.gamma(2, m): mean m, (1 + i omega m / 2)^-2
    b = 2.0 / m
    return lambda t: np.where(t > d, w * b * b * (np.asarray(t) - d)
                              * np.exp(-b * (np.asarray(t) - d)), 0.0)


def _one_unit(mu, w, d, tk):
    return (SDE('delayed exp filter of a spike train')
            .parameter('mu', default=mu).parameter('w', default=w)
            .parameter('d', default=d).parameter('tk', default=tk)
            .define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tk'))
            .point_process('n', 'mu + w*Conv(g, n)')).build()


MU, W, D, TK = 0.45, 0.3, 0.9, 0.6
TAUS = np.array([0.0, 0.25, 2.5])


@pytest.fixture(scope='module')
def one_unit():
    return _one_unit(MU, W, D, TK)


@pytest.fixture(scope='module')
def one_unit_reference():
    return _reference(MU, _delay_exp(W, D, TK), TAUS)


def test_reference_agrees_with_the_exact_tree(one_unit, one_unit_reference):
    """The grid reference and the exact delayed tree agree (both independent
    of the Pade route)."""
    ref, lam = one_unit_reference
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(one_unit, k=2, max_ell=0, external_fields=EXT,
                              tau_grid=TAUS, verbose=False)
    assert r['delay']['tree'] == 'exact'
    assert np.allclose(np.real(r['C_tau']), ref, atol=2e-4 * W * lam / TK)


def test_pade_tree_at_zero_lag_converges(one_unit, one_unit_reference):
    """C(0) of the Pade tree: the error shrinks with n at both parities and is
    far below the direct-term size w Lambda / tau of the (n, n) image."""
    ref, lam = one_unit_reference
    jump = W * lam / TK
    err = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for n in (4, 5, 8, 9):
            r = compute_cumulants(one_unit, k=2, max_ell=0, external_fields=EXT,
                                  tau_grid=TAUS, verbose=False,
                                  delay={'tree': 'pade', 'order': n})
            assert [k_.endswith('g') for k_ in r['delay']['proper_kernels']] == [True]
            err[n] = abs(float(np.real(r['C_tau'])[0]) - ref[0])
    for n, e in err.items():
        assert e < 0.05 * jump, (n, e, jump)
    assert err[8] < err[4] and err[9] < err[5]
    # far from the delay the tree is accurate as well
    assert abs(float(np.real(r['C_tau'])[2]) - ref[2]) < 0.05 * jump


def test_continuous_field_keeps_the_all_pass_image():
    """A delayed first-order filter of a continuous field is not sampled at
    its jump: it keeps the (n, n) approximant."""
    m = (SDE('delayed exp filter of a voltage')
         .parameter('a', default=1.2).parameter('b', default=-0.4)
         .parameter('d', default=0.8).parameter('tk', default=0.5)
         .parameter('s', default=0.3)
         .define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tk'))
         .field('x').drift('x', '-a*x + b*Conv(g, x)')
         .noise('xi', Gaussian(var='s^2'), into={'x': 1})).build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                              tau_grid=np.array([0.0, 1.0]), verbose=False,
                              delay={'tree': 'pade', 'order': 4})
    assert r['delay']['proper_kernels'] == []


def test_two_delays_estimate_is_honest():
    """Two distinct delays (no exact tree): the automatic order's estimate is
    of the size of its actual error against the reference, and C(0) has the
    right sign and size."""
    mu, w1, w2 = 0.4, 0.3, -0.35
    m = (SDE('two delayed filters of a spike train')
         .parameter('mu', default=mu).parameter('w1', default=w1).parameter('w2', default=w2)
         .parameter('d1', default=0.5).parameter('m1', default=0.8)
         .parameter('d2', default=1.1).parameter('t2', default=0.4)
         .define_kernel('g1', kernel=Kernel.delay('d1') * Kernel.gamma(2, 'm1'))
         .define_kernel('g2', kernel=Kernel.delay('d2') * Kernel.exponential('t2'))
         .point_process('n', 'mu + w1*Conv(g1, n) + w2*Conv(g2, n)')).build()
    taus = np.array([0.0, 0.3, 0.8, 2.0, 3.0])
    k1, k2 = _delay_gamma2(w1, 0.5, 0.8), _delay_exp(w2, 1.1, 0.4)
    ref, lam = _reference(mu, lambda t: k1(t) + k2(t), taus)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=0, external_fields=EXT, tau_grid=taus,
                              verbose=False)
    dl = r['delay']
    assert dl['tree'] == 'pade' and [k_.endswith('g2') for k_ in dl['proper_kernels']] == [True]
    assert dl['parity_probe'] is not None and dl['parity_probe']['est'] is not None
    v = np.real(r['C_tau'])
    actual = float(np.max(np.abs(v - ref)) / np.max(np.abs(ref)))
    assert dl['est_rel_error'] >= 0.5 * actual, (dl['est_rel_error'], actual)
    assert abs(v[0] - ref[0]) < 0.25 * abs(ref[0]), (v[0], ref[0])


def test_parity_probe_raises_a_blind_estimate(monkeypatch, one_unit):
    """With the strictly proper choice switched off, the (n, n) image's
    parity error at tau = 0 (about w Lambda / tau, the same sign at n // 2
    and n) cancels in the same-parity difference; the opposite-parity probe
    reports it."""
    from daedalus.api import delay as D
    monkeypatch.setattr(D, 'contact_delay_kernels', lambda *a, **k: [])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(one_unit, k=2, max_ell=0, external_fields=EXT,
                              tau_grid=TAUS, verbose=False,
                              delay={'tree': 'pade', 'order': 8})
    dl = r['delay']
    assert dl['proper_kernels'] == []
    lam = MU / (1 - W)
    v = np.real(r['C_tau'])
    probe = dl['parity_probe']
    assert probe['order'] == 5
    # |C_8(0) - C_5(0)| is about twice the direct-term error w Lambda / tau
    assert probe['est'] * np.max(np.abs(v)) > W * lam / TK
    assert dl['est_rel_error'] == probe['est']


def test_nonlinear_hawkes_with_loops_runs():
    """An exp-rate Hawkes unit with a delayed exp filter of its own spikes and
    one loop: its Pade tree is close to the exact tree, so the order runs, and
    the loop at tau = 0 has no error of alternating parity."""
    m = (SDE('exp-rate unit with a delayed filter')
         .parameter('b', default=0.1).parameter('w', default=-0.4)
         .parameter('d', default=1.0).parameter('tk', default=0.6)
         .define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tk'))
         .point_process('n', 'exp(b + w*Conv(g, n))')).build()
    loops = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for n in (8, 9):
            r = compute_cumulants(m, k=2, max_ell=1, external_fields=EXT,
                                  tau_grid=np.array([0.0, 2.0]), verbose=False,
                                  delay={'order': n, 'estimate': False})
            assert r['delay']['tree'] == 'exact' and [k_.endswith('g') for k_ in r['delay']['proper_kernels']] == [True]
            assert r['delay']['tree_vs_pade_rel'] < 0.2
            loops[n] = np.real(np.asarray(r['C_tau_by_ell'][1]))
    assert np.all(np.isfinite(loops[8]))
    # no error that alternates with the parity of n in the loop at tau = 0
    assert abs(loops[9][0] - loops[8][0]) < 0.2 * abs(loops[8][0])


def test_unusable_remedy_says_when_n_max_cannot_help():
    from daedalus.api.delay import _unusable_remedy
    assert "raise delay={'n_max'" in _unusable_remedy({4: 3.5})
    assert "raise delay={'n_max'" in _unusable_remedy({4: 3.5, 8: 0.4})
    txt = _unusable_remedy({4: 3.54, 8: 3.54})
    assert 'will not help' in txt and 'n = 8: 3.54' in txt

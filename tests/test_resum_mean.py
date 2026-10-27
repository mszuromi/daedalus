"""
tests/test_resum_mean.py
========================
The screened series at ``k = 1``: the resummed MEAN of one field
(``dd.Config(k=1, resum='screened')``, ``daedalus/api/resum.py``).

A single field with an odd vertex, ``dx = (-mu x + g x^2 - eps x^3) dt + noise``
(additive Gaussian noise of variance ``2 D``): its stationary density is
``exp(-U / D)`` with ``U = mu x^2 / 2 - g x^3 / 3 + eps x^4 / 4``, so the exact
mean is one quadrature.

* the screened mean equals the plain loop series order by order as the
  couplings go to zero (their difference is of the next order);
* it stays accurate where the plain series turns away from the exact mean;
* at ``mass='hartree'``, order 1 is the Hartree mean up to higher order;
* the propagator (and the PMS ``kappa``) is that of the ``k = 2`` run of the
  same model: both read the equal-time covariance;
* the result carries the mean by order, the spread and the plain series under
  the keys of ``k = 2`` (``C_tau_by_order``, ``C_tau_spread``,
  ``C_tau_loops_by_ell``, ...);
* the refusals of the other options stay.
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import daedalus as dd  # noqa: E402
from daedalus.api.hartree import hartree, HartreeUnsupported  # noqa: E402
from daedalus.api.noise import Gaussian  # noqa: E402
from daedalus.api.sde import SDE  # noqa: E402

TAU = np.array([0.0])


def odd_quartic(g, eps=0.2, mu=1.0, D=0.5):
    m = SDE('odd quartic')
    for p, d in dict(mu=mu, g=g, eps=eps, D=D).items():
        m.parameter(p, default=d, domain='positive')
    m.field('x')
    m.drift('x', '-mu*x + g*x^2 - eps*x^3')
    m.noise('xi', Gaussian(var='2*D'), into={'x': 1})
    return m.build()


def exact_mean(g, eps=0.2, mu=1.0, D=0.5):
    from scipy.integrate import quad
    U = lambda x: (mu * x * x / 2 - g * x ** 3 / 3 + eps * x ** 4 / 4) / D
    Z = quad(lambda x: math.exp(-U(x)), -15, 15, limit=400)[0]
    return quad(lambda x: x * math.exp(-U(x)), -15, 15, limit=400)[0] / Z


_RUNS = {}


def run(g, N, eps=0.2, k=1, resum='screened'):
    key = (g, N, eps, k, repr(resum))
    if key not in _RUNS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _RUNS[key] = dd.run(odd_quartic(g, eps), dd.Config(
                k=k, max_ell=N, resum=resum, tau_grid=TAU,
                external_fields=([('x', 1)] if k == 1 else [('x', 1), ('x', 1)]),
                verbose=False))
    return _RUNS[key]


def plain(r, N):
    return sum(float(np.real(r['C_tau_loops_by_ell'][e][0]))
               for e in range(N + 1))


def test_equals_the_plain_series_as_the_couplings_vanish():
    """Screened minus plain at order 2 is of the next order in the couplings:
    its size relative to the plain order-2 mean falls like the square of the
    coupling."""
    rel = {}
    for d in (0.05, 0.025):
        r = run(d, 2, eps=d)
        rel[d] = abs(r['mean'] - plain(r, 2)) / abs(plain(r, 2))
    assert rel[0.025] < 0.02
    assert rel[0.05] / rel[0.025] > 3        # -> 4: second order in the coupling


def test_order_one_against_the_exact_stationary_density():
    r = run(0.05, 1)
    ex = exact_mean(0.05)
    assert abs(r['mean'] - ex) < 0.1 * ex
    assert abs(r['mean'] - ex) < abs(plain(r, 1) - ex) / 3


def test_order_two_beats_the_plain_series_at_moderate_coupling():
    g = 0.3
    ex = exact_mean(g)
    r = run(g, 2)
    assert abs(r['mean'] - ex) < 0.05 * ex                  # 3 % (plain: 34 %)
    assert abs(r['mean'] - ex) < abs(plain(r, 2) - ex) / 5


@pytest.mark.slow
def test_screened_converges_where_the_plain_series_oscillates():
    g = 0.3
    ex = exact_mean(g)
    errs = {N: abs(run(g, N)['mean'] - ex) for N in (2, 3)}
    perr = {N: abs(plain(run(g, N), N) - ex) for N in (2, 3)}
    assert errs[3] < 0.01 * ex
    assert errs[3] < errs[2]
    assert perr[3] > 0.05 * ex


@pytest.mark.slow
def test_strong_coupling_beats_the_plain_series():
    g = 1.0
    ex = exact_mean(g)
    errs = [abs(run(g, N)['mean'] - ex) for N in (1, 2, 3)]
    perr3 = abs(plain(run(g, 3), 3) - ex)
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.3 * ex
    assert errs[2] < perr3 / 3       # the plain series has turned away


def test_order_one_at_the_hartree_mass_is_the_hartree_mean():
    r = run(0.05, 1, resum={'method': 'screened', 'mass': 'hartree'})
    h = hartree(odd_quartic(0.05), warn=False)
    assert r['resummation']['kappa'] == 1.0
    assert abs(r['mean'] - h.mean[0]) < 1e-3 * abs(h.mean[0])


def test_k1_and_k2_share_the_propagator_and_the_pms_mass():
    r1, r2 = run(0.3, 2), run(0.3, 2, k=2)
    i1, i2 = r1['resummation'], r2['resummation']
    assert i1['kappa'] == i2['kappa'] and i1['kappa_source'] == i2['kappa_source']
    assert np.array_equal(i1['A_trial'], i2['A_trial'])
    assert np.array_equal(i1['Q_trial'], i2['Q_trial'])
    assert np.array_equal(i1['pms_curve'][2], i2['pms_curve'][2])


def test_result_keys_and_the_mean_by_order():
    N = 2
    r = run(0.3, N)
    assert set(r['C_tau_by_order']) == {0, 1, 2}
    assert set(r['C_tau_loops_by_ell']) == {0, 1, 2}
    assert np.allclose(r['C_tau'], r['C_tau_by_order'][N])
    assert np.allclose(r['C_tau_spread'],
                       r['C_tau_by_order'][2] - r['C_tau_by_order'][1])
    assert 'spread_last_two' in r['resummation']
    saddle = 0.0                   # mu x - g x^2 + eps x^3 = 0 at the origin
    assert r['mean'] == pytest.approx(
        saddle + float(np.real(r['C_tau'][0])))
    assert r['mean_by_order'][N] == pytest.approx(r['mean'])
    assert r['config']['resum'] == 'screened' and r['config']['k'] == 1
    # the plain series is the k = 1 loop path
    from daedalus.api import compute_cumulants
    base = compute_cumulants(odd_quartic(0.3), k=1, max_ell=N,
                             external_fields=[('x', 1)], tau_grid=TAU,
                             verbose=False)
    for e in range(N + 1):
        assert np.allclose(r['C_tau_loops_by_ell'][e], base['C_tau_by_ell'][e])


def _shifted(two_fields):
    """``dx = (a - mu x + g x^2 - eps x^3) dt + noise`` with a mean-field
    saddle away from 0, optionally with a linear follower ``y``."""
    m = SDE('shifted odd quartic')
    for p, d in dict(mu=1.0, g=0.3, eps=0.2, D=0.5, a=0.8, nu=2.0).items():
        m.parameter(p, default=d, domain='positive')
    if two_fields:
        m.field('y')
    m.field('x')
    if two_fields:
        m.drift('y', '-nu*y + 0.1*x')
        m.noise('eta', Gaussian(var='0.3'), into={'y': 1})
    m.drift('x', 'a - mu*x + g*x^2 - eps*x^3' + (' + 0.05*y' if two_fields else ''))
    m.noise('xi', Gaussian(var='2*D'), into={'x': 1})
    return m.build()


def _screened_mean(model, field, N=2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return dd.run(model, dd.Config(
            k=1, max_ell=N, resum='screened', tau_grid=TAU,
            external_fields=[(field, 1)], verbose=False))


def test_a_saddle_away_from_zero_and_a_second_field():
    from scipy.integrate import quad
    U = lambda x: (-0.8 * x + x * x / 2 - 0.3 * x ** 3 / 3
                   + 0.2 * x ** 4 / 4) / 0.5
    Z = quad(lambda x: math.exp(-U(x)), -15, 15, limit=400)[0]
    ex = quad(lambda x: x * math.exp(-U(x)), -15, 15, limit=400)[0] / Z
    r = _screened_mean(_shifted(False), 'x')
    saddle = r['mean'] - float(np.real(r['C_tau'][0]))
    assert abs(saddle - 0.897) < 1e-3                  # the absolute mean adds it
    assert abs(r['mean'] - ex) < 0.01 * ex
    # two fields, the follower y declared first: <y> = 0.1 <x> / nu exactly
    rx = _screened_mean(_shifted(True), 'x', N=1)
    ry = _screened_mean(_shifted(True), 'y', N=1)
    assert ry['mean'] == pytest.approx(0.1 * rx['mean'] / 2.0, rel=1e-7)
    assert rx['resummation']['kappa'] == ry['resummation']['kappa']


def test_refusals_stay():
    with pytest.raises(ValueError, match='k = 1'):
        dd.Config(k=1, max_ell=1, resum='screened',
                  spectra={'omega': [0.0]})
    with pytest.raises(ValueError, match='covariance'):
        dd.Config(k=1, max_ell=1, resum='screened', covariance={'fields': None})
    with pytest.raises(ValueError, match='k = 1'):
        dd.Config(k=3, max_ell=1, resum='screened')
    with pytest.raises(ValueError, match='max_ell'):
        dd.Config(k=1, max_ell=0, resum='screened')
    # kernels are still refused at k = 1
    with pytest.raises(HartreeUnsupported):
        dd.run(dd.load_model('single_population_quad_exp_test')[0],
               dd.Config(k=1, max_ell=1, resum='screened'))

"""
tests/test_audit_wave1_noise.py
===============================
Regression tests for the Oct 2026 audit's fix wave 1, noise library front
end (``api.noise``, ``api.sde``).  Inline public models only.

* the uniform jump's CGF enters the action as a Taylor polynomial: the built
  model records its order and compute_cumulants refuses (k, max_ell) that
  can use noise vertices above it (they were silently absent);
* theta inside a jump-law parameter is refused like in a source parameter;
* sequence parameters refuse a string (``list('Dn')`` gave ``D``, ``n``);
* the round-1 sources and jump laws check their parameter domains.
"""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                                      # noqa: E402
from api import noise as N                                             # noqa: E402
from api.noise import NoiseError                                       # noqa: E402
from api.sde import SDE                                                # noqa: E402


def _uniform_ou(order, mu=1.3, lam=2.0, h=0.7):
    return (SDE(f'aw1 uniform ou {order}').parameter('mu', default=mu)
            .parameter('lam', default=lam).parameter('h', default=h).field('x')
            .drift('x', '-mu*x')
            .noise('eta', N.CompoundPoisson('lam', N.Uniform(0, 'h')), couples={'x': 1})
            .series_order(order).build())


def _cc(model, k, ell):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=k, max_ell=ell, external_fields=[('x', 1)] * k,
                                 tau_grid=np.array([0.4, 1.0]) if k <= 2 else None,
                                 parallel=False, verbose=False, use_cache=False)


def test_uniform_series_order_recorded_and_enforced():
    m = _uniform_ou(3)
    assert m['noise_series_order'] == {'order': 3, 'sources': ['eta']}
    # k = 4 tree needs the theta^4 vertex, absent from an order-3 action
    with pytest.raises(ValueError, match=r'SDE.series_order\(4\)'):
        _cc(m, 4, 0)
    with pytest.raises(ValueError, match=r'k \+ 2\*max_ell = 5'):
        _cc(m, 3, 1)
    # within the order: the exact tree, lam E[J^2] / (2 mu) e^{-mu tau}
    r = _cc(m, 2, 0)
    ex = 2.0 * 0.7 ** 2 / 3 / (2 * 1.3) * np.exp(-1.3 * np.array([0.4, 1.0]))
    assert np.max(np.abs(np.real(r['C_tau']) / ex - 1)) < 1e-13
    # regular laws record nothing
    g = (SDE('aw1 regular').parameter('mu', default=1.0).field('x').drift('x', '-mu*x')
         .noise('W', N.Gaussian(var=1), couples={'x': 1}).build())
    assert 'noise_series_order' not in g


@pytest.mark.parametrize('make', [
    lambda: N.CompoundPoisson(1, N.Dirac('theta')),
    lambda: N.CompoundPoisson('r', N.Exponential('2*theta')),
    lambda: N.GaussianJump(0, 'theta'),
])
def test_theta_in_jump_parameters_refused(make):
    with pytest.raises(NoiseError, match='theta'):
        make()


@pytest.mark.parametrize('make', [
    lambda: N.Cumulants('Dn'),
    lambda: N.FinitePMF('12', [0.5, 0.5]),
    lambda: N.FinitePMF([1, 2], 'pq'),
    lambda: N.Hyperexponential('pq', [1, 2]),
    lambda: N.PhaseType('a', [[-1.0]]),
])
def test_string_sequences_refused(make):
    with pytest.raises(NoiseError, match='not the string'):
        make()


@pytest.mark.parametrize('make, name', [
    (lambda: N.Gaussian(var=-1), 'var'),
    (lambda: N.Poisson(rate=-2), 'rate'),
    (lambda: N.GammaProcess(-1, 1), 'shape_rate'),
    (lambda: N.InverseGaussianProcess(1, -1), 'shape'),
    (lambda: N.InverseGaussianProcess(-1, 1), 'mean_rate'),
    (lambda: N.Cumulants([-1]), 'kappa2'),
    (lambda: N.Gamma(-1, 1), 'shape'),
    (lambda: N.Bernoulli(1.5), 'p'),
    (lambda: N.Binomial(2.5, 0.3), 'n'),
    (lambda: N.Binomial(3, -0.1), 'p'),
    (lambda: N.FinitePMF([1, 2], [1.5, -0.5]), 'p1'),
])
def test_round1_domain_checks(make, name):
    with pytest.raises(NoiseError, match=name):
        make()


def test_uniform_degenerate_and_valid_reflected_laws():
    for lo, hi in ((0, 0), (1, 1), ('a', 'a')):
        with pytest.raises(NoiseError, match='lo != hi'):
            N.Uniform(lo, hi)
    # reflected (negative-jump) laws and symbolic parameters stay allowed
    assert N.CompoundPoisson(1, N.Exponential(-1)).cumulant(1) == -1
    assert N.CompoundPoisson(1, N.Gamma(2, -1)).cumulant(1) == -2
    assert N.GammaProcess(1, -1).cumulant(1) == -1
    N.Gaussian(var='s2'), N.Poisson(rate='r'), N.Bernoulli('p'), N.Uniform('lo', 'hi')

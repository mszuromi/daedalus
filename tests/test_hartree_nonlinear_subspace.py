"""
tests/test_hartree_nonlinear_subspace.py
========================================
The Hartree / Gaussian closure takes its expectations over the fields that
enter nonlinearly only (``daedalus/api/hartree.py``): a linear filter stage adds no
quadrature dimension.

* the nonlinear subspace is read off the drift and the noise (a field that
  looks linear but enters a product, or the noise, is counted);
* the subspace quadrature, the closed-form Gaussian moments and the old
  full-space tensor rule agree (rounding level) on a quadratic-rate model
  with K = 0, 3, 6 linear filter stages;
* K = 13 (15 fields, 3^15 points before) runs in seconds and equals an
  independent hand-written closure;
* a non-polynomial drift (Taylor polynomial) on 2 of 10 fields integrates on
  2 dimensions;
* a model whose nonlinear dimension is beyond the tensor rule is refused
  with the dimension and the fields named under ``expectations='quadrature'``
  and solved in closed form by default.
"""
import math
import os
import sys
import time

import numpy as np
import pytest
import scipy.linalg as sl

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from daedalus.api import hartree as H  # noqa: E402
from daedalus.api.hartree import hartree, HartreeUnsupported  # noqa: E402
from daedalus.api.noise import Gaussian  # noqa: E402
from daedalus.api.sde import SDE  # noqa: E402

PARS = dict(ts=0.5, D=0.3, mu=1.0, w=0.8, eps=0.5, h=0.4, a=2.0)


def rate_model(K):
    """A quadratic-rate model with K linear filter stages: white noise ->
    s1 -> ... -> sK -> v (cubic-stabilised, quadratic term h v^2) -> u, the
    rate v^2 / a read out through a linear stage."""
    m = SDE(f'quadratic rate with {K} filter stages')
    for p, d in PARS.items():
        m.parameter(p, default=d, domain='positive')
    for k in range(1, K + 1):
        m.field(f's{k}')
    m.field('v')
    m.field('u')
    for k in range(1, K + 1):
        prev = f's{k - 1}' if k > 1 else '0'
        m.drift(f's{k}', f'(-s{k} + {prev})/ts')
    m.drift('v', f'-mu*v + w*{f"s{K}" if K else "0"} + h*v^2 - eps*v^3')
    m.drift('u', '-u + v^2/a')
    m.noise('xi', Gaussian(var='2*D'), into={'s1' if K else 'v': 1})
    return m.build()


def _all_nonlinear(monkeypatch):
    """The old behaviour: every field in the quadrature."""
    monkeypatch.setattr(H.ActionSystem, 'nonlinear',
                        property(lambda self: list(range(len(self.fields)))))


def _close(a, b, tol=1e-11):
    return (np.max(np.abs(a.mean - b.mean)) < tol * (1 + np.max(np.abs(a.mean)))
            and np.max(np.abs(a.cov - b.cov)) < tol * np.max(np.abs(a.cov)))


# ── the subspace ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('K', [0, 3, 13])
def test_only_the_voltage_is_nonlinear(K):
    sysm = H.action_system(rate_model(K), {})
    names = [sysm.fields[i] for i in sysm.nonlinear]
    assert names == ['dv1']                # u and the filter stages are linear
    assert len(sysm.fields) == K + 2


def test_a_field_entering_a_product_or_the_noise_is_nonlinear():
    """b looks linear (its own drift is affine) but multiplies v in the drift
    of v; c enters the noise of v through a response-field product."""
    m = SDE('hidden nonlinearity')
    for p in ('D', 'mu'):
        m.parameter(p, default=1.0, domain='positive')
    m.parameter('g', default=0.6, domain='positive')
    for f in 'vbcz':
        m.field(f)
    m.drift('b', '1.0 - b').drift('c', '0.5 - c').drift('z', '-z + b')
    m.drift('v', '-mu*v + g*v*b - 0.3*v^3')
    m.noise('xi', Gaussian(var='2*D*(1 + c*c)'), into={'v': 1})
    m.noise('eta', Gaussian(var='0.2'), into={'b': 1})
    m.noise('zeta', Gaussian(var='0.1'), into={'c': 1})
    sysm = H.action_system(m.build(), {})
    names = {sysm.fields[i] for i in sysm.nonlinear}
    assert names == {'dv1', 'db1', 'dc1'}      # z stays linear
    # and the covariance between v and b matters, so the counting is needed
    r = hartree(m.build(), {}, warn=False)
    assert abs(r.cov[sysm.fields.index('dv1'), sysm.fields.index('db1')]) > 0


# ── agreement of the three ways to take the expectations ────────────────────

@pytest.mark.parametrize('K', [0, 3, 6])
def test_subspace_moments_and_full_space_agree(K, monkeypatch):
    m = rate_model(K)
    sub = hartree(m, {}, warn=False, expectations='quadrature')
    mom = hartree(m, {}, warn=False, expectations='moments')
    assert sub.converged and mom.converged
    assert sub.info['nonlinear_fields'] == ['dv1']
    assert sub.info['expectations'] == 'quadrature'
    assert mom.info['expectations'] == 'moments'
    assert sub.info['quadrature_nodes'] == 3        # one dimension, 3 nodes
    assert _close(sub, mom)
    _all_nonlinear(monkeypatch)
    full = hartree(m, {}, warn=False, expectations='quadrature')
    assert full.info['quadrature_nodes'] == 3 ** (K + 2)
    assert _close(sub, full, 1e-10) and _close(mom, full, 1e-10)


def test_existing_nonlinear_models_unchanged(monkeypatch):
    """Every field nonlinear: the subspace rule is the old rule, bit for bit."""
    import daedalus as dd
    m = dd.load_model('ou_quartic_coupled')[0]
    P = {'mu1': 1.0, 'mu2': 1.0, 'eps1': 1.0, 'eps2': 0.5, 'J1': 0.5,
         'J2': 0.5, 'D1': 1.0, 'D2': 1.0}
    new = hartree(m, P, warn=False)
    _all_nonlinear(monkeypatch)
    old = hartree(m, P, warn=False)
    assert np.array_equal(new.cov, old.cov) and np.array_equal(new.mean, old.mean)


# ── K = 13: 15 fields ───────────────────────────────────────────────────────

def _hand_closure(K, P=PARS):
    """The Gaussian closure of ``rate_model(K)`` written out by hand (the
    only nonlinear field is v: its Gaussian moments are closed forms)."""
    n = K + 2
    iv, iu = K, K + 1
    A0 = np.zeros((n, n))
    for k in range(K):
        A0[k, k] = -1 / P['ts']
        if k:
            A0[k, k - 1] = 1 / P['ts']
    Q = np.zeros((n, n))
    Q[0 if K else iv, 0 if K else iv] = 2 * P['D']
    if K:
        A0[iv, K - 1] = P['w']
    m = np.zeros(n)
    S = np.eye(n) * 0.1
    for _ in range(4000):
        mv, sv = m[iv], S[iv, iv]
        A = A0.copy()
        A[iv, iv] = -P['mu'] + 2 * P['h'] * mv - 3 * P['eps'] * (mv ** 2 + sv)
        A[iu, iu] = -1.0
        A[iu, iv] = 2 * mv / P['a']
        f = A0 @ m
        f[iv] += (-P['mu'] * mv + P['h'] * (mv ** 2 + sv)
                  - P['eps'] * (mv ** 3 + 3 * mv * sv))
        f[iv] += 0.0
        f[iu] = -m[iu] + (mv ** 2 + sv) / P['a']
        m_new = m - np.linalg.solve(A, f)
        S_new = sl.solve_continuous_lyapunov(A, -Q)
        done = np.max(np.abs(m_new - m)) + np.max(np.abs(S_new - S)) < 1e-14
        m = m + 0.5 * (m_new - m)
        S = S + 0.5 * (S_new - S)
        if done:
            break
    return m, S


def test_fifteen_fields_run_in_seconds_and_match_a_hand_closure():
    K = 13
    t0 = time.perf_counter()
    r = hartree(rate_model(K), {}, warn=False)
    assert time.perf_counter() - t0 < 120
    assert r.converged and len(r.fields) == 15
    assert r.info['nonlinear_fields'] == ['dv1']
    assert r.info['quadrature_nodes'] == 3
    m, S = _hand_closure(K)
    assert np.allclose(r.mean, m, rtol=1e-9, atol=1e-12)
    assert np.allclose(r.cov, S, rtol=1e-8, atol=1e-13)
    assert abs(r.mean[K + 1]) > 1e-4         # the rate has moved: the test bites


# ── non-polynomial drift: Taylor polynomial, quadrature on 2 dimensions ─────

def _exp_model(n_lin):
    m = SDE(f'exponential on 2 of {n_lin + 2}')
    m.parameter('D', default=0.2, domain='positive')
    m.field('x').field('y')
    for k in range(1, n_lin + 1):
        m.field(f'z{k}')
    m.drift('x', '-x - 0.3*(exp(x) - 1) + 0.2*y')
    m.drift('y', '-1.5*y - 0.2*(exp(y) - 1) + 0.1*x')
    for k in range(1, n_lin + 1):
        m.drift(f'z{k}', f'-{1 + 0.37 * k}*z{k} + {0.3 / k}*{"x" if k % 2 else "y"}')
    m.noise('xi', Gaussian(var='2*D'), into={'x': 1})
    m.noise('eta', Gaussian(var='D'), into={'y': 1})
    return m.build()


def test_non_polynomial_drift_integrates_on_the_nonlinear_two(monkeypatch):
    big = hartree(_exp_model(8), {}, warn=False, multistart=False)
    assert big.converged and len(big.fields) == 10
    assert big.info['nonlinear_fields'] == ['dx1', 'dy1']
    assert big.info['nonlinear_dim'] == 2 and big.info['quadrature_nodes'] < 100
    small = _exp_model(2)
    sub = hartree(small, {}, warn=False, multistart=False,
                  expectations='quadrature')
    _all_nonlinear(monkeypatch)
    brute = hartree(small, {}, warn=False, multistart=False,
                    expectations='quadrature')
    assert brute.info['quadrature_nodes'] == sub.info['quadrature_nodes'] ** 2
    assert _close(sub, brute, 1e-10)


# ── the point limit applies to the nonlinear dimension ──────────────────────

def _on_model(N):
    from daedalus.api.model import TemporalModelBuilder
    return (TemporalModelBuilder('O(N) quartic').population('pop', size=N)
            .physical_field('x', population='pop')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('eps', default=2.0, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .set_action_text(
                'sum(xt[i]*((Dt+mu)*x[i] + (eps/%d)*x[i]*sum(x[j]^2 for j in '
                'pop)) - D*xt[i]^2 for i in pop)' % N)
            .mf_residual(lhs='(Dt+mu)*x[i]',
                         rhs='-(eps/%d)*x[i]*sum(x[j]^2 for j in pop)' % N,
                         population='pop')
            .build())


def test_beyond_the_tensor_rule_is_refused_by_name_and_solved_in_closed_form():
    N = 14
    m = _on_model(N)
    with pytest.raises(HartreeUnsupported) as ei:
        hartree(m, {}, expectations='quadrature', warn=False)
    msg = str(ei.value)
    assert f"{N} nonlinear fields" in msg and "dx1" in msg and "dx14" in msg
    assert 'moments' in msg
    r = hartree(m, {}, warn=False, multistart=False)        # auto -> moments
    assert r.converged and r.info['expectations'] == 'moments'
    gg = 2.0 * (N + 2) / N
    s2 = (-1 + math.sqrt(1 + 4 * gg)) / (2 * gg)
    assert np.allclose(np.diag(r.cov), s2, rtol=1e-10)


def test_expectations_option_is_checked():
    with pytest.raises(ValueError, match='expectations'):
        hartree(rate_model(0), {}, expectations='sparse')


# ── the numeric entry point ─────────────────────────────────────────────────

def test_gaussian_closure_with_a_nonlinear_subset():
    """OU-quartic x plus a linear follower y (dy = (x - y) dt): the subset
    rule equals the full rule, and the refusal names the dimension."""
    mu, eps, D = 1.0, 0.7, 0.5

    def drift(X):
        return np.stack([-mu * X[:, 0] - eps * X[:, 0] ** 3,
                         X[:, 0] - X[:, 1]], axis=1)

    def jac(X):
        J = np.zeros((len(X), 2, 2))
        J[:, 0, 0] = -mu - 3 * eps * X[:, 0] ** 2
        J[:, 1, 0], J[:, 1, 1] = 1.0, -1.0
        return J

    def noise(X):
        Q = np.zeros((len(X), 2, 2))
        Q[:, 0, 0] = 2 * D
        return Q

    kw = dict(mean0=np.zeros(2), cov0=np.eye(2), degree=3,
              fields=['x', 'y'])
    full = H.gaussian_closure(drift, jac, noise, 2, **kw)
    sub = H.gaussian_closure(drift, jac, noise, 2, nonlinear=[0], **kw)
    assert sub.info['nonlinear_dim'] == 1 and full.info['nonlinear_dim'] == 2
    assert sub.info['quadrature_nodes'] == 3 and full.info['quadrature_nodes'] == 9
    assert np.allclose(sub.cov, full.cov, rtol=1e-11, atol=1e-14)
    with pytest.raises(HartreeUnsupported, match=r'3 nonlinear fields \(a, b, c\)'):
        old = H.MAX_QUADRATURE_POINTS
        H.MAX_QUADRATURE_POINTS = 10
        try:
            H._gh_rule(3, 3, names=['a', 'b', 'c'])
        finally:
            H.MAX_QUADRATURE_POINTS = old

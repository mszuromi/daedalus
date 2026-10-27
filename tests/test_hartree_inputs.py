"""
tests/test_hartree_inputs.py
============================
Exogenous input fields in the Gaussian closure and the screened series
(``api/hartree.py``): a point process, or a noise source exposed with
``as_field=``, is a field without dynamics (``n = noise``).  The closure
replaces it by its drive -- its rate as a mean (and, if the rate depends on
the fields, as a drift) and its second cumulant as white noise -- feeding the
equations and the kernel states it enters, so the input's noise drives the
kernel states directly.  The reference in every test is the model written
that way by hand.

* a Poisson input through an alpha kernel equals the hand-embedded model to
  rounding (Hartree; the screened series with the loop path's kernel route);
* a Gaussian input exposed as a field, direct and through a kernel, and a
  point process coupled directly;
* a rate that depends on the fields (a Hawkes-type feedback through a
  kernel, quadratic rate);
* a Gaussian-jittered Poisson input (the rational fit of the density) equals
  the analytic spectrum and variance of the exactly jittered input within the
  fit tolerance;
* what stays refused, with the reason.
"""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import daedalus as dd  # noqa: E402
from api.delay import Kernel  # noqa: E402
from api.hartree import hartree, HartreeUnsupported  # noqa: E402
from api.noise import Gaussian, Poisson  # noqa: E402
from api.sde import SDE  # noqa: E402

TAU = np.array([0.0, 0.5, 1.0])
P = dict(ts=0.5, lam=2.0, p=0.7, w=1.3, mu=1.0, eps=0.4, a=2.0, D=0.4)


def _close(a, b, tol):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tol * max(
        np.max(np.abs(b)), 1e-300)


def _poisson_alpha(hand):
    """A quadratic-rate model driven by a Poisson input through an alpha
    synapse (``hand``: the two synapse stages driven by the jumps)."""
    m = SDE('Poisson input, alpha synapse' + (' by hand' if hand else ''))
    for k, d in P.items():
        m.parameter(k, default=d, domain='positive')
    m.field('v').field('u')
    if hand:
        m.field('s1').field('s2')
        m.drift('s1', '-s1/ts').drift('s2', '(-s2 + s1)/ts')
        m.noise('N', Poisson(rate='lam', size='p/ts'), into={'s1': 1})
        m.drift('v', '-mu*v - eps*v^3 + w*s2')
    else:
        m.define_kernel('k', kernel=Kernel.alpha('ts'))
        m.point_process('n', 'lam', jump_size='p')
        m.drift('v', '-mu*v - eps*v^3 + w*Conv(k, n)')
    m.drift('u', '-u + v^2/a')
    return m.build()


def _fields(r, names):
    ix = [r.fields.index(f) for f in names]
    return ix


def _same_on(a, b, names, tol=1e-12):
    ia, ib = _fields(a, names), _fields(b, names)
    tau = np.array([-0.5, 0.0, 0.4, 1.5])
    om = np.array([0.0, 0.7, 3.0])
    return (_close(a.mean[ia], b.mean[ib], tol)
            and _close(a.cov[np.ix_(ia, ia)], b.cov[np.ix_(ib, ib)], tol)
            and _close(a.C_matrix(tau)[:, ia][:, :, ia],
                       b.C_matrix(tau)[:, ib][:, :, ib], tol)
            and _close(a.spectrum(om, ia[0], ia[1]),
                       b.spectrum(om, ib[0], ib[1]), tol))


# ── a point process through a kernel ────────────────────────────────────────

def test_poisson_input_through_an_alpha_kernel_equals_the_hand_model():
    a, b = hartree(_poisson_alpha(False), warn=False), hartree(
        _poisson_alpha(True), warn=False)
    assert a.converged and a.fields == ['dv1', 'du1']        # no input field
    assert a.info['kernel_states']['inputs'] == ['dn1']
    assert a.info['kernel_states']['states'] == 2
    assert _same_on(a, b, ['dv1', 'du1'], 1e-12)
    # the input's rate is the mean drive: <v> is well away from the saddle's 0
    assert a.mean[0] > 0.5


def test_screened_series_with_a_poisson_input_equals_the_hand_model():
    out = []
    for hand in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out.append(dd.run(_poisson_alpha(hand), dd.Config(
                k=2, max_ell=1, tau_grid=TAU,
                external_fields=[('v', 1), ('v', 1)],
                resum={'method': 'screened', 'observable': ['v', 'u']},
                verbose=False)))
    a, b = out
    assert a['resummation']['kappa'] == pytest.approx(
        b['resummation']['kappa'], rel=1e-9)
    assert _close(a['C_tau'], b['C_tau'], 1e-8)


def _single(kind, hand):
    m = SDE(f'{kind} {"by hand" if hand else ""}')
    for k, d in P.items():
        m.parameter(k, default=d, domain='positive')
    m.field('v')
    if kind == 'gaussian':
        if hand:
            m.drift('v', '-mu*v - eps*v^3')
            m.noise('xi', Gaussian(var='2*D'), into={'v': 'w'})
        else:
            m.noise('xi', Gaussian(var='2*D'), as_field='n')
            m.drift('v', '-mu*v - eps*v^3 + w*n')
    elif kind == 'gaussian kernel':
        m.define_kernel('k', kernel=Kernel.exponential('ts'))
        if hand:
            m.field('s')
            m.drift('s', '-s/ts').drift('v', '-mu*v - eps*v^3 + w*s')
            m.noise('xi', Gaussian(var='2*D/ts^2'), into={'s': 1})
        else:
            m.noise('xi', Gaussian(var='2*D'), as_field='n')
            m.drift('v', '-mu*v - eps*v^3 + w*Conv(k, n)')
    else:                                                   # a point process
        if hand:
            m.drift('v', '-mu*v - eps*v^3')
            m.noise('N', Poisson(rate='lam', size='p*w'), into={'v': 1})
        else:
            m.point_process('n', 'lam', jump_size='p')
            m.drift('v', '-mu*v - eps*v^3 + w*n')
    return m.build()


@pytest.mark.parametrize('kind', ['gaussian', 'gaussian kernel',
                                  'point process'])
def test_exposed_inputs_direct_and_through_a_kernel(kind):
    a, b = hartree(_single(kind, False), warn=False), hartree(
        _single(kind, True), warn=False)
    assert a.fields == ['dv1']
    ib = b.fields.index('dv1')
    assert _close(a.mean[0], b.mean[ib], 1e-12) or abs(a.mean[0]) < 1e-15
    assert a.cov[0, 0] == pytest.approx(b.cov[ib, ib], rel=1e-12)
    tau = np.array([0.0, 0.3, 2.0])
    assert np.allclose(a.C(tau), b.C_matrix(tau)[:, ib, ib], rtol=1e-11,
                       atol=1e-15)


def test_a_rate_that_depends_on_the_fields_feeds_back_through_the_kernel():
    """A Hawkes-type loop: the rate holds a linear and a quadratic term of v,
    so the input is its own drift (a nonlinear one) plus white noise."""
    rate = 'lam0 + lam1*v + lamq*v^2'

    def build(hand):
        m = SDE('feedback' + (' by hand' if hand else ''))
        for k, d in dict(ts=0.5, lam0=1.0, lam1=0.8, p=0.7, mu=1.2,
                         lamq=0.15).items():
            m.parameter(k, default=d, domain='positive')
        m.field('v')
        if hand:
            m.field('s1').field('s2')
            m.drift('s1', '-s1/ts').drift('s2', '(-s2 + s1)/ts')
            m.noise('N', Poisson(rate=rate, size='p/ts'), into={'s1': 1})
            m.drift('v', '-mu*v + 0.5*s2')
        else:
            m.define_kernel('k', kernel=Kernel.alpha('ts'))
            m.point_process('n', rate, jump_size='p')
            m.drift('v', '-mu*v + 0.5*Conv(k, n)')
        return m.build()
    a, b = hartree(build(False), warn=False), hartree(build(True), warn=False)
    assert a.converged and a.info['nonlinear_fields'] == ['dv1']
    ib = b.fields.index('dv1')
    assert a.mean[0] == pytest.approx(b.mean[ib], rel=1e-12)
    assert a.cov[0, 0] == pytest.approx(b.cov[ib, ib], rel=1e-12)
    assert a.mean[0] > 0.1                       # the loop moved the mean


# ── Gaussian jitter ─────────────────────────────────────────────────────────

def test_gaussian_jitter_equals_the_exactly_jittered_input_within_the_fit():
    """A Poisson input shifted by a Gaussian jitter of width sigma (the
    rational fit of the density) into a low-pass v: the spectrum of v is
    lam p^2 |w H_tau H_X|^2 exp(-sigma^2 w^2) exactly, and the closure's is
    that within the fit tolerance (1e-6 in the density)."""
    from scipy.integrate import quad
    tau, tauX, lam, p, w, sigma = 1.0, 0.5, 2.0, 0.7, 1.3, 0.3
    m = (SDE('jittered input')
         .parameter('tau', default=tau, domain='positive')
         .parameter('tauX', default=tauX, domain='positive')
         .parameter('lam', default=lam, domain='positive')
         .parameter('p', default=p).parameter('w', default=w)
         .parameter('sigma', default=sigma, domain='positive')
         .define_kernel('kX', kernel=Kernel.exponential('tauX'))
         .define_kernel('g', kernel=Kernel.gaussian('sigma', tol=1e-6))
         .point_process('n', 'lam', jump_size='p')
         .field('v').drift('v', '(-v + w*Conv(kX*g, n))/tau'))
    r = hartree(m.build(), warn=False)
    assert r.converged and r.fields == ['dv1']
    assert r.info['kernel_states']['inputs'] == ['dn1']
    S = lambda om: (lam * p * p * abs(w / (1 + 1j * om * tau)) ** 2
                    * abs(1 / (1 + 1j * om * tauX)) ** 2
                    * np.exp(-sigma ** 2 * om ** 2))
    om = np.array([0.0, 0.5, 2.0, 6.0])
    got = r.spectrum(om).real
    assert np.max(np.abs(got - [S(x) for x in om])) < 1e-5 * S(0.0)
    var = quad(S, -np.inf, np.inf, limit=400)[0] / (2 * np.pi)
    assert r.cov[0, 0] == pytest.approx(var, rel=1e-5)
    assert r.mean[0] == pytest.approx(w * lam * p, rel=1e-12)


# ── what stays refused ──────────────────────────────────────────────────────

def test_an_input_inside_a_nonlinear_term_is_refused_with_the_reason():
    m = (SDE('input times field')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('lam', default=2.0, domain='positive')
         .parameter('p', default=0.3)
         .point_process('n', 'lam', jump_size='p')
         .field('v').drift('v', '-mu*v + 0.5*n*v'))
    with pytest.raises(HartreeUnsupported, match='cannot be replaced by its '
                                                 'white noise'):
        hartree(m.build(), warn=False)


def test_an_input_whose_rate_holds_a_kernel_is_refused():
    m = (SDE('rate through a kernel')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('ts', default=0.5, domain='positive')
         .parameter('lam', default=2.0, domain='positive')
         .parameter('p', default=0.3)
         .define_kernel('k', kernel=Kernel.exponential('ts'))
         .field('v')
         .point_process('n', 'lam + 0.3*Conv(k, v)', jump_size='p')
         .drift('v', '-mu*v + 0.5*n'))
    with pytest.raises(HartreeUnsupported, match='kernel in the noise vertex'):
        hartree(m.build(), warn=False)


def test_a_state_dependent_rate_keeps_its_multiplicative_noise():
    """The input's second cumulant is p^2 lambda(v): the noise variance
    depends on the field.  The elimination keeps it as multiplicative noise
    averaged in the Gaussian (<lambda> = lambda(<v>) + lamq Var v), against
    (i) the hand-embedded model with that noise and (ii) the same model
    whose noise variance is frozen at the saddle's rate: the second differs
    visibly, the first does not."""
    pars = dict(ts=0.5, lam0=0.5, lam1=0.0001, lamq=3.0, p=0.8, mu=3.0)
    rate = 'lam0 + lam1*v + lamq*v^2'

    def build(kind):
        m = SDE(f'multiplicative {kind}')
        for k, d in pars.items():
            m.parameter(k, default=d, domain='positive')
        m.field('v')
        if kind == 'eliminated':
            m.define_kernel('k', kernel=Kernel.exponential('ts'))
            m.point_process('n', rate, jump_size='p')
            m.drift('v', '-mu*v + Conv(k, n)')
            return m.build()
        m.field('s').drift('v', '-mu*v + s')
        if kind == 'hand':
            m.drift('s', '-s/ts')
            m.noise('N', Poisson(rate=rate, size='p/ts'), into={'s': 1})
        else:                          # the noise variance frozen at lambda(v*)
            m.parameter('lamstar', default=frozen[0], domain='positive')
            m.drift('s', '-s/ts + (' + rate + ' - lamstar)*p/ts')
            m.noise('N', Poisson(rate='lamstar', size='p/ts'), into={'s': 1})
        return m.build()
    frozen = [0.0]
    a = hartree(build('eliminated'), warn=False)
    v_star = float(a.saddle[0])
    frozen[0] = (pars['lam0'] + pars['lam1'] * v_star
                 + pars['lamq'] * v_star ** 2)
    b = hartree(build('hand'), warn=False)
    c = hartree(build('frozen'), warn=False)
    ib, ic = b.fields.index('dv1'), c.fields.index('dv1')
    assert a.info['nonlinear_fields'] == ['dv1']
    assert a.cov[0, 0] == pytest.approx(b.cov[ib, ib], rel=1e-12)
    assert a.mean[0] == pytest.approx(b.mean[ib], rel=1e-12)
    # the multiplicative part moves the variance visibly
    assert abs(c.cov[ic, ic] / a.cov[0, 0] - 1) > 0.2        # 30 %
    # and the averaged noise is lambda(<v>) + lamq Var v, not lambda(v*)
    m_v, s_v = a.mean[0], a.cov[0, 0]
    lam_avg = (pars['lam0'] + pars['lam1'] * m_v
               + pars['lamq'] * (m_v ** 2 + s_v))
    assert abs(lam_avg / frozen[0] - 1) > 0.3

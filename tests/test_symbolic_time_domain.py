"""Tests for stage C2 of ``api/symbolic_out.py``: the tree-level covariance in
the time domain as a mode sum for any denominator degree, the stated
assumptions (Routh/Sturm sign conditions, distinct poles), and time-domain
kernel integrals.

Convention (``tree_modes``): C(tau) = (1/2 pi) int C0(omega) exp(-i omega tau)
d omega = sum over the terms of tau's side of coeff * tau^power *
exp(rate * tau), plus ``contact`` * delta(tau) (not in the terms).

Every oracle is compared with the pipeline's tree level (Phase J,
``compute_cumulants(max_ell=0)['C_tau']``) at numbers, and runs twice: with the
model's own symbol domains only ('plain') and after a FieldTheory of an
unrelated model has put positive domains on the same symbol names
('positive').  Sage symbol domains are global and sticky, so the helper
restores the previous state exactly.  Models are tracked ones or built here
with the SDE front-end; ``use_cache=False`` throughout.
"""
import json
import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from sage.all import SR, var, I, exp, sqrt, assumptions, forget, assume

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import daedalus as dd  # noqa: E402
from api import symbolic_out as so  # noqa: E402
from api.compute import compute_cumulants  # noqa: E402
from api.sde import SDE  # noqa: E402
from api.noise import Gaussian, Poisson  # noqa: E402

warnings.filterwarnings('ignore')

TAUS = np.array([0.3, 0.5, 1.0, 2.0, 5.0, -0.4, -1.0, -3.0])
RTOL = 1e-12
DOMAINS = ['plain', 'positive']


def _zero(expr):
    return bool(SR(expr).simplify_full() == 0)


def _max_rel(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))


def _pipeline(model, ext, par, taus=TAUS):
    res = compute_cumulants(model, k=2, max_ell=0, external_fields=ext,
                            parameters=par, tau_grid=taus, use_cache=False,
                            parallel=False, verbose=False)
    npar = {str(k): float(v) for k, v in res['num_params'].items()}
    return np.asarray(res['C_tau']), npar


# ── symbol-domain state ─────────────────────────────────────────────────────

def _clear_domain(name):
    """Reset the pynac domain of ``name``: a 'complex' redeclaration is the
    only way, and it adds a Maxima assumption that is forgotten here."""
    before = {str(a) for a in assumptions()}
    SR.var(name, domain='complex')
    for a in list(assumptions()):
        if str(a) not in before:
            forget(a)


@contextmanager
def domain_state(names, mode):
    """'plain': the names carry no domain; 'positive': a FieldTheory of an
    unrelated SDE model declares them positive.  Restores the previous
    domains and Maxima assumptions afterwards."""
    before = list(assumptions())
    was = {n: bool(SR.symbol(n).is_positive()) for n in names}
    try:
        for a in before:
            forget(a)
        for n in names:
            _clear_domain(n)
        if mode == 'positive':
            s = SDE('unrelated positive-domain model').field('q_unrel')
            for n in names:
                s.parameter(n, default=1.0, domain='positive')
            s.drift('q_unrel', f'-{names[0]}*q_unrel')
            s.noise('xi_unrel', Gaussian(var='2'), couples={'q_unrel': 1})
            so._expanded_theory(s.build(), 2, False, False)
            assert all(SR.symbol(n).is_positive() for n in names)
        else:
            assert not any(SR.symbol(n).is_positive() for n in names)
        yield
    finally:
        # forget everything, set the domains back (a positive domain adds
        # its own 'n > 0'), then re-assume what is missing: forgetting
        # 'n > 0' AFTER the redeclaration would drop the domain again
        for a in list(assumptions()):
            forget(a)
        for n in names:
            if was[n]:
                SR.var(n, domain='positive')
            else:
                _clear_domain(n)
        present = {str(a) for a in assumptions()}
        for a in before:
            if str(a) not in present:
                a.assume() if hasattr(a, 'assume') else assume(a)


# ── models ──────────────────────────────────────────────────────────────────

def _feedback_chain():
    """Three-stage linear filter chain with negative feedback: the
    covariance denominator has the irreducible cubic factors
    (s + a)^3 + k and (a - s)^3 + k."""
    return (SDE('feedback filter chain').parameter('a', default=1.0)
            .parameter('k', default=2.0).parameter('D', default=1.0)
            .field('x').field('y').field('z')
            .drift('x', '-a*x - k*z').drift('y', '-a*y + x')
            .drift('z', '-a*z + y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _rotation():
    """Underdamped (rotational) OU: eigenvalues -g +- i w0, complex pair."""
    return (SDE('underdamped OU').parameter('g', default=0.2)
            .parameter('w0', default=1.0).parameter('D', default=1.0)
            .field('x').field('y')
            .drift('x', '-g*x - w0*y').drift('y', 'w0*x - g*y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _double_pole_chain():
    """x' = -a x + xi, y' = -a y + x: C0_yy has (a^2 - s^2)^2, a double pole
    on each side."""
    return (SDE('equal-rate chain').parameter('a', default=1.3)
            .parameter('D', default=1.0).field('x').field('y')
            .drift('x', '-a*x').drift('y', '-a*y + x')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _two_rate_chain():
    return (SDE('two-rate chain').parameter('a', default=1.0)
            .parameter('b', default=2.0).parameter('D', default=1.0)
            .field('x').field('y').drift('x', '-a*x').drift('y', '-b*y + x')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _hawkes_1pop():
    """Linear Hawkes, one population, exponential kernel through the
    membrane: branching ratio b = a*w, mean rate a*Em/(1 - a*w)."""
    return (SDE('linear Hawkes 1-pop').parameter('Em', default=0.5)
            .parameter('tau', default=2.0).parameter('a', default=1.0)
            .parameter('w', default=0.4)
            .define_function('phi', args=['v'], expression='a*v')
            .noise('spikes', Poisson(rate='phi(v)'), expose='n')
            .field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + w*n')
            ).build()


# ═════════════════════════════════════════════════════════════════════════════
# Oracle 1: OU white noise
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def ou_ref():
    m, mod = dd.load_model('ou_quartic')
    par = dict(mod.DEFAULT_FUNDAMENTAL)
    ref, npar = _pipeline(m, [('x', 1), ('x', 1)], par)
    return m, npar, ref


@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_ou_white_noise(ou_ref, domains):
    m, npar, ref = ou_ref
    with domain_state(['mu', 'D'], domains):
        cov = so.tree_covariance(m, use_cache=False, mean_field={'xstar1': 0})
        ms = cov.modes('dx1', 'dx1', reference=npar)
        mu, D = var('mu'), var('D')
        assert len(ms.terms) == 2 and ms.contact == 0
        by_side = {t.side: t for t in ms.terms}
        assert _zero(by_side['positive'].coeff - D / mu)
        assert _zero(by_side['positive'].rate + mu)
        assert _zero(by_side['negative'].coeff - D / mu)
        assert _zero(by_side['negative'].rate - mu)
        assert all(t.power == 0 for t in ms.terms)
        tau = ms.tau
        assert _zero(ms.expression('positive') - D / mu * exp(-mu * tau))
        assert [str(a.expression) + a.relation for a in ms.assumptions] == \
            ['mu>']
        assert _max_rel(ms.evaluate(TAUS, npar), ref) < RTOL
        assert ms.assumptions[0].holds({'mu': 1.0, 'D': 1.0})
        with pytest.raises(so.AssumptionViolation, match='mu > 0'):
            ms.evaluate(TAUS, {'mu': -1.0, 'D': 1.0})


# ═════════════════════════════════════════════════════════════════════════════
# Oracle 2: underdamped oscillator (complex pair)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_underdamped_spectrum_2p39(domains):
    """The spectrum that returned 2.06 instead of 2.39 under positive domains
    (C1): 1/((w^2 - w0^2)^2 + g^2 w^2) at g = 0.2, w0 = 1."""
    names = ['g', 'w0']
    with domain_state(names, domains):
        w, g, w0 = var('omega'), var('g'), var('w0')
        C0 = 1 / ((w ** 2 - w0 ** 2) ** 2 + g ** 2 * w ** 2)
        ref = {'g': 0.2, 'w0': 1.0}
        ms = so.tree_modes(C0, w, reference=ref)
        assert len(ms.terms) == 4 and not ms.roots
        rates = sorted((complex(SR(t.rate).subs({g: 0.2, w0: 1.0}).n())
                        for t in ms.terms), key=lambda z: (z.real, z.imag))
        assert abs(rates[0] - complex(-0.1, -np.sqrt(0.99))) < 1e-14
        assert abs(rates[0].conjugate() - rates[1]) < 1e-14
        c = so.to_callable(C0, w, params=names)
        for t in (0.3, 1.7, 6.0, -2.2):
            q = so.numeric_inverse_transform(lambda x: c(x, **ref), t)
            got = complex(ms.evaluate(np.array([t]), ref)[0])
            assert abs(q.value - got) < 1e-9
        v03 = complex(ms.evaluate(np.array([0.3]), ref)[0]).real
        assert abs(v03 - 2.39) < 0.005           # not 2.06
        # the SR view agrees with the numeric view
        sr = complex(ms.expression('positive').subs(
            {g: 0.2, w0: 1.0, ms.tau: 0.3}).n(digits=30))
        assert abs(sr - v03) < 1e-13
        # and with the C1 explicit closed form
        r = so.inverse_transform(C0, w, reference=ref)
        f = so.to_callable(r['even'], r['tau'], params=names)
        assert abs(complex(f(0.3, **ref)[()]) - v03) < 1e-12
        conds = {str(a.expression) + a.relation for a in ms.assumptions}
        assert 'g>' in conds


@pytest.fixture(scope='module')
def rotation_ref():
    m = _rotation()
    par = {'g': 0.2, 'w0': 1.0, 'D': 1.0}
    refs = {ab: _pipeline(m, [(ab[0], 1), (ab[1], 1)], par)[0]
            for ab in (('x', 'x'), ('x', 'y'))}
    return m, par, refs


@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_underdamped_model_vs_pipeline(rotation_ref, domains):
    m, par, refs = rotation_ref
    with domain_state(['g', 'w0', 'D'], domains):
        cov = so.tree_covariance(m, use_cache=False,
                                 mean_field={'xstar1': 0, 'ystar1': 0})
        for (a, b), ref in refs.items():
            ms = cov.modes(f'd{a}1', f'd{b}1', reference=par)
            assert _max_rel(ms.evaluate(TAUS, par), ref) < RTOL
            num = ms.numeric_terms(par)
            pos = [t.rate for t in num if t.side == 'positive']
            assert any(abs(r.imag) > 0.5 for r in pos)       # oscillating
        conds = {str(x.expression) + x.relation for x in ms.assumptions}
        assert 'g>' in conds
        with pytest.raises(so.AssumptionViolation):
            ms.evaluate(TAUS, dict(par, g=-0.1))


# ═════════════════════════════════════════════════════════════════════════════
# Oracle 3: irreducible cubic denominator
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def chain_ref():
    m = _feedback_chain()
    par = {'a': 1.0, 'k': 2.0, 'D': 1.0}
    refs = {ab: _pipeline(m, [(ab[0], 1), (ab[1], 1)], par)[0]
            for ab in (('z', 'z'), ('x', 'z'), ('y', 'x'))}
    return m, par, refs


@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_cubic_denominator_vs_pipeline(chain_ref, domains):
    m, par, refs = chain_ref
    with domain_state(['a', 'k', 'D'], domains):
        cov = so.tree_covariance(m, use_cache=False, mean_field={
            'xstar1': 0, 'ystar1': 0, 'zstar1': 0})
        for (a, b), ref in refs.items():
            ms = cov.modes(f'd{a}1', f'd{b}1', reference=par)
            assert len(ms.roots) == 6 and len(ms.terms) == 6
            assert {r.side for r in ms.roots} == {'positive', 'negative'}
            assert _max_rel(ms.evaluate(TAUS, par), ref) < RTOL
        conds = {str(x.expression) + ' ' + x.relation
                 for x in ms.assumptions}
        assert {'a >', '8*a^3 - k >', 'a^3 + k >', 'k !='} <= conds
        # stability boundary k = 8 a^3 and the lower boundary a^3 + k = 0
        for bad in (dict(par, k=9.0), dict(par, k=-1.5)):
            with pytest.raises(so.AssumptionViolation):
                ms.evaluate(TAUS, bad)
        # inside the region at other values: modes still valid, and match
        # the pipeline there
        other = {'a': 1.4, 'k': 5.0, 'D': 0.7}
        ref2, _ = _pipeline(m, [('y', 1), ('x', 1)], other)
        assert _max_rel(ms.evaluate(TAUS, other), ref2) < RTOL
        # the SR view (RootOf symbols substituted by numeric roots) agrees
        md = ms._md
        pt = md.ctx.point(other)
        sub = {var(k): v for k, v in other.items()}
        for r in ms.roots:
            fam = md.families[int(str(r.symbol)[3])]
            z = md.ctx.numeric_roots(fam.p, pt)[r.index]
            sub[r.symbol] = complex(z)
        for side in so.SIDES:
            t = 0.8 if side == 'positive' else -0.8
            sr = complex(ms.expression(side).subs(sub).subs(
                {ms.tau: t}).n(digits=25))
            assert abs(sr - complex(ms.evaluate(np.array([t]), other)[0])) \
                < 1e-12


# ═════════════════════════════════════════════════════════════════════════════
# Oracle 4: double pole (confluent residues)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def double_ref():
    m = _double_pole_chain()
    par = {'a': 1.3, 'D': 1.0}
    refs = {ab: _pipeline(m, [(ab[0], 1), (ab[1], 1)], par)[0]
            for ab in (('y', 'y'), ('x', 'y'))}
    return m, par, refs


@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_double_pole_vs_pipeline(double_ref, domains):
    m, par, refs = double_ref
    with domain_state(['a', 'D'], domains):
        cov = so.tree_covariance(m, use_cache=False,
                                 mean_field={'xstar1': 0, 'ystar1': 0})
        ms = cov.modes('dy1', 'dy1', reference=par)
        assert sorted(t.power for t in ms.terms) == [0, 0, 1, 1]
        a, D, tau = var('a'), var('D'), ms.tau
        # C_yy(tau) = D (1 + a |tau|) e^{-a |tau|} / (2 a^3)
        assert _zero(ms.expression('positive')
                     - D * (1 + a * tau) * exp(-a * tau) / (2 * a ** 3))
        for (x, y), ref in refs.items():
            msx = cov.modes(f'd{x}1', f'd{y}1', reference=par)
            assert _max_rel(msx.evaluate(TAUS, par), ref) < RTOL


def test_coincident_poles_refused_then_exact_with_values():
    """Distinct rates a != b: at a = b the stated distinctness fails and the
    modes refuse; substituting the values makes the double pole structural
    and the result matches the pipeline."""
    m = _two_rate_chain()
    cov = so.tree_covariance(m, use_cache=False,
                             mean_field={'xstar1': 0, 'ystar1': 0})
    ms = cov.modes('dy1', 'dy1', reference={'a': 1.0, 'b': 2.0, 'D': 1.0})
    kinds = {x.kind for x in ms.assumptions}
    assert 'distinct' in kinds
    with pytest.raises(so.AssumptionViolation, match='distinct'):
        ms.evaluate(TAUS, {'a': 1.0, 'b': 1.0, 'D': 1.0})
    with pytest.raises(so.AssumptionViolation, match='values='):
        cov.modes('dy1', 'dy1', reference={'a': 1.0, 'b': 1.0, 'D': 1.0})
    eq = cov.modes('dy1', 'dy1', values={'a': 1, 'b': 1},
                   reference={'D': 1.0})
    assert max(t.power for t in eq.terms) == 1
    ref, _ = _pipeline(m, [('y', 1), ('y', 1)], {'a': 1.0, 'b': 1.0,
                                                 'D': 1.0})
    assert _max_rel(eq.evaluate(TAUS, {'D': 1.0}), ref) < RTOL
    # near (not at) the coincidence the generic modes are still accurate
    near = {'a': 1.0, 'b': 1.0 + 1e-6, 'D': 1.0}
    refn, _ = _pipeline(m, [('y', 1), ('y', 1)], near)
    assert _max_rel(ms.evaluate(TAUS, near), refn) < 1e-8


# ═════════════════════════════════════════════════════════════════════════════
# Oracle 5: linear Hawkes (contact = mean rate)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def hawkes1_ref():
    m = _hawkes_1pop()
    par = {'Em': 0.5, 'tau': 2.0, 'a': 1.0, 'w': 0.4}
    ref, npar = _pipeline(m, [('n', 1), ('n', 1)], par)
    return m, par, ref, npar


@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_linear_hawkes_1pop_closed_form(hawkes1_ref, domains):
    """C_nn(tau) = lam b (2 - b) / (2 (1 - b) tau_m) e^{-(1-b)|tau|/tau_m}
    with b = a w, lam = a Em / (1 - b); contact = lam."""
    m, par, ref, npar = hawkes1_ref
    with domain_state(['Em', 'tau', 'a', 'w'], domains):
        Em, tm, a, w = var('Em'), var('tau'), var('a'), var('w')
        lam = a * Em / (1 - a * w)
        cov = so.tree_covariance(m, use_cache=False,
                                 mean_field={'vstar1': Em / (1 - a * w)})
        ms = cov.modes('dn1', 'dn1', reference=par)
        b = a * w
        tau = ms.tau
        assert str(tau) == 'tau_lag'           # 'tau' is a model parameter
        closed = lam * b * (2 - b) / (2 * (1 - b) * tm) \
            * exp(-(1 - b) * tau / tm)
        assert _zero(ms.expression('positive') - closed)
        assert _zero(ms.expression('negative') - closed.subs({tau: -tau}))
        assert _zero(ms.contact - lam)
        assert abs(ms.contact_value(par) - npar['nstar1']) < 1e-12
        assert _max_rel(ms.evaluate(TAUS, par), ref) < RTOL
        conds = {str(x.expression) + ' ' + x.relation for x in ms.assumptions}
        assert '-a*w + 1 >' in conds


@pytest.fixture(scope='module')
def hawkes2_ref():
    m, mod = dd.load_model('linear_hawkes')
    par = dict(mod.DEFAULT_FUNDAMENTAL)
    refs = {}
    for ab in (('n1', 'n2'), ('n1', 'n1')):
        ext = [(ab[0][0], int(ab[0][1])), (ab[1][0], int(ab[1][1]))]
        refs[ab] = _pipeline(m, ext, par)
    return m, refs


@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_linear_hawkes_2pop_vs_pipeline(hawkes2_ref, domains):
    m, refs = hawkes2_ref
    names = ['a', 'tau', 'tau_g', 'vstar1', 'vstar2', 'w11', 'w12', 'w21',
             'w22']
    with domain_state(names, domains):
        cov = so.tree_covariance(m, use_cache=False)
        for (x, y), (ref, npar) in refs.items():
            ms = cov.modes('d' + x, 'd' + y, reference=npar)
            assert len(ms.roots) == 8                # two quartic factors
            assert _max_rel(ms.evaluate(TAUS, npar), ref) < RTOL
            if x == y:
                assert ms.contact == var('a') * var('vstar1')
                assert abs(ms.contact_value(npar) - npar['nstar1']) < 1e-12
            else:
                assert ms.contact == 0


# ═════════════════════════════════════════════════════════════════════════════
# Time-domain kernel integrals
# ═════════════════════════════════════════════════════════════════════════════

def _rand_points(names, n, seed, lo=0.3, hi=2.0):
    rng = np.random.default_rng(seed)
    return [{k: float(rng.uniform(lo, hi)) for k in names} for _ in range(n)]


def test_kernel_ou_two_sided_closed_form_equals_fourier():
    w, mu, D = var('omega'), var('mu'), var('D')
    Ap, tp, Am, tm = var('A_p'), var('tau_p'), var('A_m'), var('tau_m')
    C0 = 2 * D / (w ** 2 + mu ** 2)
    L = so.Kernel.two_sided_exponential(Ap, tp, -Am, tm)
    ki = so.integrate_against(C0, L, w)
    expect = D / mu * (Ap * tp / (1 + mu * tp) - Am * tm / (1 + mu * tm))
    assert _zero(ki.smooth - expect) and ki.contact == 0
    # Fourier route (C1, explicit residues): int dw/2pi L~(w) C0(w)
    pf = so.pair_with_kernel(C0, L.fourier(w).subs({w: -w}), w,
                             reference={'mu': 1.0, 'tau_p': 1.0,
                                        'tau_m': 1.0})
    assert pf.exact and _zero(pf.expression - ki.smooth)
    kinds = {(str(a.expression), a.kind) for a in ki.assumptions}
    assert {('tau_p', 'kernel'), ('tau_m', 'kernel'),
            ('mu', 'stability')} <= kinds


@pytest.mark.parametrize('domains', DOMAINS)
def test_kernel_cubic_two_sided_time_equals_fourier(chain_ref, domains):
    m, par, _ = chain_ref
    with domain_state(['a', 'k', 'D', 'A_p', 'tau_p', 'A_m', 'tau_m'],
                      domains):
        cov = so.tree_covariance(m, use_cache=False, mean_field={
            'xstar1': 0, 'ystar1': 0, 'zstar1': 0})
        Ap, tp, Am, tm = var('A_p'), var('tau_p'), var('A_m'), var('tau_m')
        L = so.Kernel.two_sided_exponential(Ap, tp, -Am, tm)
        w = cov.omega
        for a, b in (('dz1', 'dz1'), ('dx1', 'dz1')):
            ms = cov.modes(a, b, reference=par)
            ki = ms.integrate_against(L)
            assert not ki.roots                  # traces: no RootOf left
            C0 = cov.entry(a, b)
            pf = so.pair_with_kernel(C0, L.fourier(w).subs({w: -w}), w,
                                     any_degree=True, reference=par)
            assert pf.exact and _zero(pf.expression - ki.smooth)
            # the pre-C2 route declines the cubic, its quadrature agrees;
            # the default route (item 40, C2 decision (a)) is exact
            vals = dict(par, A_p=1.0, tau_p=0.7, A_m=0.6, tau_m=1.3)
            q = so.pair_with_kernel(C0, L.fourier(w).subs({w: -w}), w,
                                    values=vals, any_degree=False)
            assert not q.exact and 'degree' in q.reason
            kv = ms.integrate_against(L, values=vals).value
            assert abs(q.value - kv) < 1e-9
            qd = so.pair_with_kernel(C0, L.fourier(w).subs({w: -w}), w,
                                     values=vals)
            assert qd.exact and abs(qd.value - kv) < 1e-12
            # random points inside the region: both expressions, 1e-12
            free = sorted({str(v) for v in SR(ki.smooth).variables()})
            for pt in _rand_points(free, 4, seed=7):
                pt['k'] = 2.0 * pt['a'] ** 3        # inside 8a^3 > k
                sub = {var(k): v for k, v in pt.items()}
                x1 = complex(ki.smooth.subs(sub).n(digits=30))
                x2 = complex(pf.expression.subs(sub).n(digits=30))
                x3 = ms.integrate_against(L, values=pt).value
                assert abs(x1 - x2) <= 1e-12 * abs(x2)
                assert abs(x1 - x3) <= 1e-12 * abs(x1)


def test_kernel_poly_exp_on_double_pole_vs_fourier_and_quadrature(
        double_ref):
    m, par, _ = double_ref
    cov = so.tree_covariance(m, use_cache=False,
                             mean_field={'xstar1': 0, 'ystar1': 0})
    w = cov.omega
    c, T = var('c'), var('T')
    # L(s) = c s e^{-s/T} (s > 0)  +  e^{s/T}/T (s < 0)
    L = so.Kernel.exponential(c, -1 / T, 'positive', power=1) + \
        so.Kernel.exponential(1 / T, 1 / T, 'negative')
    ms = cov.modes('dx1', 'dy1', reference=par)
    ki = ms.integrate_against(L)
    pf = so.pair_with_kernel(cov.entry('dx1', 'dy1'),
                             L.fourier(w).subs({w: -w}), w,
                             reference=dict(par, T=1.0, c=1.0))
    assert pf.exact and _zero(pf.expression - ki.smooth)
    vals = dict(par, c=0.8, T=0.6)
    from scipy.integrate import quad
    f = lambda s: complex(ms.evaluate(np.array([s]), par)[0]).real
    Lf = lambda s: 0.8 * s * np.exp(-s / 0.6) if s > 0 else \
        np.exp(s / 0.6) / 0.6
    td = quad(lambda s: Lf(s) * f(s), 0, np.inf, epsabs=1e-13)[0] + \
        quad(lambda s: Lf(s) * f(s), -np.inf, 0, epsabs=1e-13)[0]
    assert abs(ms.integrate_against(L, values=vals).value - td) < 1e-10


def test_kernel_contact_rules_hawkes(hawkes1_ref):
    m, par, _, npar = hawkes1_ref
    cov = so.tree_covariance(m, use_cache=False, mean_field='solve',
                             parameters=par)
    ms = cov.modes('dn1', 'dn1', reference=par)
    Ap, tp, Am, tm = var('A_p'), var('tau_p'), var('A_m'), var('tau_m')
    L = so.Kernel.two_sided_exponential(Ap, tp, -Am, tm)
    with pytest.raises(so.SymbolicOutError, match='jumps'):
        ms.integrate_against(L)
    p3 = {k: par[k] for k in ('tau', 'a', 'w')}     # Em left with vstar1
    # Em is baked into the solved saddle: the same value is fine, another
    # one is refused (it would silently change nothing otherwise)
    ms.evaluate(TAUS, par)
    with pytest.raises(so.SymbolicOutError, match='mean-field solve'):
        ms.evaluate(TAUS, dict(par, Em=0.9))
    vals = dict(p3, A_p=1.0, tau_p=0.7, A_m=0.6, tau_m=1.3)
    ex = ms.integrate_against(L, values=vals, contact='exclude')
    mid = ms.integrate_against(L, values=vals, contact='mid')
    pos = ms.integrate_against(L, values=vals, contact='positive')
    lam = npar['nstar1']
    assert abs(mid.value - ex.value - lam * (1.0 - 0.6) / 2) < 1e-12
    assert abs(pos.value - ex.value - lam * 1.0) < 1e-12
    # smooth part == Fourier route on the smooth C0
    w = cov.omega
    C0s = cov.entry('dn1', 'dn1') - ms.contact
    pf = so.pair_with_kernel(C0s, L.fourier(w).subs({w: -w}), w,
                             values=vals, any_degree=True)
    assert pf.exact and abs(pf.value - ex.value) < 1e-12
    # a kernel continuous at 0 includes the contact without asking
    sym = so.Kernel.exponential(1, -1, 'positive') + \
        so.Kernel.exponential(1, 1, 'negative')
    k2 = ms.integrate_against(sym, values=p3)
    k2x = ms.integrate_against(sym, values=p3, contact='exclude')
    assert abs(k2.value - k2x.value - lam) < 1e-12
    # a growing kernel is refused
    with pytest.raises(so.SymbolicOutError, match='grows'):
        ms.integrate_against(so.Kernel.exponential(1, 1, 'positive'),
                             values=p3)


# ═════════════════════════════════════════════════════════════════════════════
# Stated assumptions: Routh/Sturm vs numeric roots
# ═════════════════════════════════════════════════════════════════════════════

def _split_numeric(coeffs):
    r = np.roots(coeffs[::-1])
    return int(np.sum(r.real < 0)), float(np.min(np.abs(r.real)))


@pytest.mark.parametrize('seed', range(6))
def test_routh_count_matches_numeric_roots(seed):
    """Random integer polynomials of degree 1..6: the stated split of each
    irreducible factor equals the numeric left-half-plane count (or the
    factor is refused because a root sits on the imaginary axis)."""
    from api.symbolic_out import _Bridge, _AlgCtx, _Family, _split_factor
    rng = np.random.default_rng(seed)
    w = SR.var('omega')
    n_checked = 0
    for d in range(1, 7):
        cs = [int(x) for x in rng.integers(-5, 6, size=d + 1)]
        cs[-1] = cs[-1] or 1
        cs[0] = cs[0] or 2
        s = I * w
        poly = sum(c * s ** i for i, c in enumerate(cs))
        b = _Bridge([1 / poly], w)
        ctx = _AlgCtx(b)
        pt = ctx.point({})
        f = b.to_field(1 / poly)
        for g, _ in f.denominator().factor():
            if b.deg_s(g) == 0:
                continue
            p = ctx.from_R(g)
            gc = [float(ctx.eval_K(c, pt).real()) for c in p.list()]
            n_neg, gap = _split_numeric(gc)
            fam = _Family(0, p, g, 1, [])
            try:
                _split_factor(fam, ctx, pt, 'p')
            except so.ClosedFormUnavailable:
                assert gap < 1e-7, (cs, gc)
                continue
            assert fam.n_neg == n_neg, (cs, gc)
            n_checked += 1
    assert n_checked >= 4


def test_even_factor_sturm_and_any_degree_pairing():
    """1/(w^6 + 5 w^2 + 3): an even, irreducible sextic in s (Routh array
    identically singular): Sturm split 3/3, exact pairing = quadrature."""
    w = var('omega')
    C0 = 1 / (w ** 6 + 5 * w ** 2 + 3)
    p = so.pair_with_kernel(C0, SR(1), w, any_degree=True, fallback=False,
                            values={})
    from scipy.integrate import quad
    ref6 = quad(lambda x: 1 / (x ** 6 + 5 * x ** 2 + 3), -np.inf, np.inf,
                epsabs=1e-14)[0] / (2 * np.pi)
    assert p.exact and abs(p.value - ref6) < 1e-12
    ms = so.tree_modes(C0, w)
    assert [f.method for f in ms._md.families] == ['sturm-even']
    assert ms._md.families[0].n_neg == 3
    c = so.to_callable(C0, w, params=[])
    for t in (0.5, 2.0, -1.0):
        q = so.numeric_inverse_transform(lambda x: c(x), t)
        assert abs(q.value - complex(ms.evaluate(np.array([t]), {})[0])) \
            < 1e-9
    # a pole on the real omega axis is refused, not mis-assigned
    with pytest.raises(so.ClosedFormUnavailable, match='real omega axis'):
        so.tree_modes(1 / ((w ** 2 - 1) * (w ** 2 + 4)), w)


def test_mixed_quadratic_factor_split_and_sign_of_leading_coeff():
    """Golden-ratio quartic: factors s^2 +- s - 1 each have one root per
    side; the assignment needs the sign of the leading coefficient, which
    is a stated condition when it is symbolic."""
    w, c = var('omega'), var('c')
    C1 = 1 / (w ** 4 + 3 * w ** 2 + 1)
    ms = so.tree_modes(C1, w)
    assert [f.n_neg for f in ms._md.families] == [1, 1]
    cal = so.to_callable(C1, w, params=[])
    for t in (0.4, 2.5, -0.7):
        q = so.numeric_inverse_transform(lambda x: cal(x), t)
        assert abs(q.value - complex(ms.evaluate(np.array([t]), {})[0])) \
            < 1e-9
    # symbolic scale: 1/((c s^2 + s - 1)(c s^2 - s - 1)) in s = i w
    s = I * w
    C2 = 1 / ((c * s ** 2 + s - 1) * (c * s ** 2 - s - 1))
    ms2 = so.tree_modes(C2, w, reference={'c': 1.0})
    assert any(str(a.expression) == 'c' and a.relation == '>'
               for a in ms2.assumptions)
    with pytest.raises(so.AssumptionViolation):
        ms2.evaluate(np.array([1.0]), {'c': -0.5})


def test_sampling_is_only_a_cross_check():
    """The C1 refusal case 1/((i w + mu - 1)(w^2 + 4)) at mu = 1.2: the
    stated route gives modes valid for mu > 1 instead of refusing."""
    w, mu = var('omega'), var('mu')
    C0 = 1 / ((I * w + mu - 1) * (w ** 2 + 4))
    ms = so.tree_modes(C0, w, reference={'mu': 1.2})
    assert any(str(a.expression) == 'mu - 1' for a in ms.assumptions)
    cal = so.to_callable(C0, w, params=['mu'])
    for m_, t in ((1.2, 0.7), (2.5, -0.5), (1.05, 1.1)):
        q = so.numeric_inverse_transform(lambda x: cal(x, mu=m_), t)
        assert abs(q.value - complex(ms.evaluate(np.array([t]),
                                                 {'mu': m_})[0])) < 1e-9
    with pytest.raises(so.AssumptionViolation, match='mu - 1 > 0'):
        ms.evaluate(np.array([0.7]), {'mu': 0.8})


# ═════════════════════════════════════════════════════════════════════════════
# Output format, exports, C1 open items
# ═════════════════════════════════════════════════════════════════════════════

def test_mode_fields_and_json(chain_ref):
    m, par, _ = chain_ref
    cov = so.tree_covariance(m, use_cache=False, mean_field={
        'xstar1': 0, 'ystar1': 0, 'zstar1': 0})
    ms = cov.modes('dz1', 'dz1', reference=par)
    import dataclasses
    assert [f.name for f in dataclasses.fields(so.Mode)] == \
        ['coeff', 'rate', 'power', 'side']
    num = ms.numeric_terms(par)
    assert all(isinstance(t.coeff, complex) and isinstance(t.rate, complex)
               and t.side in so.SIDES for t in num)
    d = json.loads(json.dumps(ms.to_dicts()))
    assert len(d['terms']) == 6 and len(d['roots']) == 6
    assert {'expression', 'relation', 'kind', 'origin'} <= \
        set(d['assumptions'][0])
    # tau = 0 uses the pipeline's Ito left limit by default
    v0 = ms.evaluate(np.array([0.0]), par)[0]
    assert abs(v0 - ms.evaluate(np.array([-1e-14]), par)[0]) < 1e-12


def test_G_ft_none_branch_gives_the_same_C0():
    m, mod = dd.load_model('ou_quartic_colored')
    res = compute_cumulants(m, k=2, max_ell=0,
                            external_fields=[('x', 1), ('x', 1)],
                            parameters=dict(mod.DEFAULT_FUNDAMENTAL),
                            tau_grid=np.array([0.5]), use_cache=False,
                            verbose=False, parallel=False)
    a = so.tree_covariance_from_result(res, model=m, mean_field='result')
    stripped = dict(res, propagator=dict(res['propagator'], G_ft=None))
    b = so.tree_covariance_from_result(stripped, model=m,
                                       mean_field='result')
    assert _zero(a.entry('dx1', 'dx1') - b.entry('dx1', 'dx1'))
    assert _zero(a.entry('dx1', 'dxi1') - b.entry('dx1', 'dxi1'))


def test_mean_field_symbols_follow_the_model_declarations():
    from api.symbolic_out import _mf_symbols_only, _is_saddle_symbol
    model = {'parameters': [{'name': 'msad', 'mean_field': True},
                            {'name': 'kstar_gain'}, {'name': 'mu'}]}
    npar = {SR.var('msad1'): 0.25, SR.var('kstar_gain'): 3.0,
            SR.var('mu'): 1.0}
    assert _mf_symbols_only(model, npar) == {'msad1': 0.25}
    assert not _is_saddle_symbol('msadx', model)
    # models without declarations: the '<base>star' naming rule
    assert _is_saddle_symbol('xstar1', None)
    assert not _is_saddle_symbol('kstar_gain', None)
    # 'solve' substitutes exactly the declared saddles (here nstar1, vstar1)
    m = _hawkes_1pop()
    cov = so.tree_covariance(m, use_cache=False, mean_field='solve',
                             parameters={'Em': 0.5, 'tau': 2.0, 'a': 1.0,
                                         'w': 0.4})
    assert set(cov.mean_field_values) == {'nstar1', 'vstar1'}
    assert abs(cov.mean_field_values['vstar1'] - 0.5 / 0.6) < 1e-12
    assert 'vstar1' not in cov.parameters


def test_split_cubic_and_repeated_cubic_vs_quadrature():
    """Adversarial: a cubic factor with roots on both sides (the kernel
    integral keeps RootOf terms), and a squared cubic factor (confluent
    residues on RootOf poles); both against direct quadrature."""
    w, b = var('omega'), var('b')
    s = I * w
    # s^3 - b s^2 - s + 1: at b = 2 roots ~ -0.80, 0.55, 2.25 (split 1/2)
    C0 = 1 / ((s ** 3 - b * s ** 2 - s + 1) * (s ** 2 - 4) * (s ** 2 - 9))
    ref = {'b': 2.0}
    ms = so.tree_modes(C0, w, reference=ref)
    split = [f for f in ms._md.families if 0 < f.n_neg < f.deg]
    assert len(split) == 1 and split[0].deg == 3 and split[0].n_neg == 1
    cal = so.to_callable(C0, w, params=['b'])
    for t in (0.6, 2.0, -0.9, -3.0):
        q = so.numeric_inverse_transform(lambda x: cal(x, b=2.0), t)
        assert abs(q.value - complex(ms.evaluate(np.array([t]), ref)[0])) \
            < 1e-9
    L = so.Kernel.exponential(1, -1, 'positive') + \
        so.Kernel.exponential(-0.5, 2, 'negative')
    ki = ms.integrate_against(L, values=ref)
    assert ki.roots and all(r.symbol in SR(ki.smooth).variables()
                            for r in ki.roots)
    from scipy.integrate import quad
    f = lambda x: complex(ms.evaluate(np.array([x]), ref)[0]).real
    td = quad(lambda x: np.exp(-x) * f(x), 0, np.inf, epsabs=1e-13)[0] + \
        quad(lambda x: -0.5 * np.exp(2 * x) * f(x), -np.inf, 0,
             epsabs=1e-13)[0]
    assert abs(ki.value - td) < 1e-10
    # squared cubic: (s + a)^3 + 2 and its mirror, each twice
    a = var('a')
    P = ((s + a) ** 3 + 2) * ((a - s) ** 3 + 2)
    C2 = 1 / P ** 2
    ms2 = so.tree_modes(C2, w, reference={'a': 1.0})
    assert {f.mult for f in ms2._md.families} == {2}
    assert max(t.power for t in ms2.terms) == 1 and ms2.roots
    cal2 = so.to_callable(C2, w, params=['a'])
    for t in (0.4, 1.5, -2.0):
        q = so.numeric_inverse_transform(lambda x: cal2(x, a=1.0), t)
        assert abs(q.value - complex(ms2.evaluate(np.array([t]),
                                                  {'a': 1.0})[0])) < 1e-9


# ═════════════════════════════════════════════════════════════════════════════
# Second-order oscillator vs the pipeline, tau = 0, review regressions
# ═════════════════════════════════════════════════════════════════════════════

def _oscillator():
    """x'' + g x' + w0^2 x = xi, <xi xi> = 2 D delta: C0_xx is the C1
    spectrum 2D/((w^2 - w0^2)^2 + g^2 w^2) (the 2.39-not-2.06 case at
    g = 0.2, w0 = 1, 2D = 1, tau = 0.3)."""
    from api.model import TemporalModelBuilder
    return (TemporalModelBuilder('damped oscillator', n_populations=1)
            .physical_field('x').physical_field('v')
            .parameter('w0', default=1.0).parameter('g', default=0.2)
            .parameter('D', default=0.5)
            .set_action_text('sum(xt[i]*(Dt*x[i] - v[i]) + vt[i]*((Dt + g)'
                             '*v[i] + w0^2*x[i]) - D*vt[i]^2 for i in pop)')
            .set_mf_equation('xstar', '0').set_mf_equation('vstar', '0')
            .build())


@pytest.fixture(scope='module')
def oscillator_ref():
    m = _oscillator()
    par = {'w0': 1.0, 'g': 0.2, 'D': 0.5}
    taus = np.append(TAUS, 0.0)
    refs = {ab: _pipeline(m, [(ab[0], 1), (ab[1], 1)], par, taus)[0]
            for ab in (('x', 'x'), ('x', 'v'))}
    return m, par, taus, refs


@pytest.mark.parametrize('domains', DOMAINS)
def test_oracle_second_order_oscillator_2p39_vs_pipeline(oscillator_ref,
                                                         domains):
    m, par, taus, refs = oscillator_ref
    with domain_state(['g', 'w0', 'D'], domains):
        cov = so.tree_covariance(m, use_cache=False,
                                 mean_field={'xstar1': 0, 'vstar1': 0})
        for (a, b), ref in refs.items():
            ms = cov.modes(f'd{a}1', f'd{b}1', reference=par)
            got = ms.evaluate(taus, par)
            # tau != 0 to 1e-12; tau = 0: the pipeline evaluates the Ito
            # left limit at tau = -1e-6, so O(1e-6) relative there
            assert _max_rel(got[:-1], ref[:-1]) < RTOL
            assert abs(got[-1] - ref[-1]) < 1e-5 * np.max(np.abs(ref))
            if a == b:
                assert abs(got[0].real - 2.3905379) < 1e-6   # tau = 0.3
                assert abs(got[-1].real - 2.5) < 1e-12       # 1/(2 g w0^2)


def test_review_near_coincident_poles_keep_precision():
    """Mode coefficients ~1/gap cancel; evaluate sums at 256 bits.
    Reference: the C1 explicit closed form at 50 digits."""
    from sage.all import QQ
    w, a = var('omega'), var('a')
    C0 = 1 / ((w ** 2 + 1) * (w ** 2 + 1 + a))
    ms = so.tree_modes(C0, w)
    r = so.inverse_transform(C0, w, reference={'a': 1.0})
    for eps in (1e-12, 1e-16):
        got = ms.evaluate(np.array([-2.0, 0.5]), {'a': eps})
        for t, g in zip((-2.0, 0.5), got):
            br = r['positive'] if t > 0 else r['negative']
            ref = complex(br.subs({a: QQ(eps), r['tau']: QQ(t)}).n(
                digits=50))
            assert abs(g - ref) < 1e-13 * abs(ref)


def test_review_values_for_substituted_or_unknown_names_raise():
    w, mu, D = var('omega'), var('mu'), var('D')
    ms = so.tree_modes(2 * D / (w ** 2 + mu ** 2), w, values={'mu': 1})
    with pytest.raises(so.SymbolicOutError, match='entered the construction'):
        ms.evaluate([0.5], {'mu': 2.0, 'D': 1.0})
    # a name that does not occur in C cannot change it: ignored
    assert ms.evaluate([0.5], {'D': 1.0, 'zzz': 3})[0] == \
        ms.evaluate([0.5], {'D': 1.0})[0]
    same = ms.evaluate([0.5], {'mu': 1.0, 'D': 1.0})[0]
    assert abs(same - np.exp(-0.5)) < 1e-15


def test_review_kernel_rate_equal_to_a_root_on_the_other_side():
    """Split quadratic s^2 + b s - c (roots 1 and -2 at b=1, c=2) and a
    kernel rate -r on s > 0 with r = 2: the other-side root equals -r; the
    value must not be NaN (it was) and must match the Fourier route."""
    w, b, c, r = var('omega'), var('b'), var('c'), var('r')
    s = I * w
    C0 = 1 / ((s ** 2 + b * s - c) * (s ** 2 - b * s - c))
    ref = {'b': 1.0, 'c': 2.0}
    ms = so.tree_modes(C0, w, reference=ref)
    L = so.Kernel.exponential(1, -r, 'positive')
    for rv in (2.0, 2.0 + 1e-9, 0.7):
        v = dict(ref, r=rv)
        ki = ms.integrate_against(L, values=v)
        pf = so.pair_with_kernel(C0, L.fourier(w).subs({w: -w}), w,
                                 values=v, any_degree=True)
        assert np.isfinite(ki.value) and abs(ki.value - pf.value) < 1e-12


def test_review_user_names_only_and_root_names_do_not_collide():
    w, a, b = var('omega'), var('a'), var('b')
    with pytest.raises(so.AssumptionViolation) as ei:
        so.tree_modes(1 / ((w ** 2 + a ** 2) * (w ** 2 + b ** 2)), w,
                      reference={'a': 1.0, 'b': 1.0})
    assert '__c1p' not in str(ei.value) and 'a - b' in str(ei.value)
    rho, k = var('rho0_0'), var('k')
    s = I * w
    C0 = rho / (((s + 1) ** 3 + k) * ((1 - s) ** 3 + k))
    ms = so.tree_modes(C0, w, reference={'k': 2.0, 'rho0_0': 1.5})
    assert all(str(r.symbol) != 'rho0_0' for r in ms.roots)
    assert '__c1p' not in str(ms.to_dicts())


def test_review_integrate_against_refuses_outside_the_region():
    w, mu, D = var('omega'), var('mu'), var('D')
    ms = so.tree_modes(2 * D / (w ** 2 + mu ** 2), w, reference={'mu': 1.0})
    L = so.Kernel.exponential(1, -1, 'positive')
    with pytest.raises(so.AssumptionViolation, match='mu > 0'):
        ms.integrate_against(L, values={'mu': -1.0, 'D': 1.0})
    ok = ms.integrate_against(L, values={'mu': 2.0, 'D': 1.0})
    assert abs(ok.value - 0.5 / 3.0) < 1e-15        # (D/mu)/(mu + 1)


def test_review_domain_state_restores_domains():
    from sage.all import sqrt as ssqrt
    nm = 'g_restore_probe'
    SR.var(nm, domain='positive')
    try:
        for mode in DOMAINS:
            with domain_state([nm], mode):
                pass
            x = SR.symbol(nm)
            assert x.is_positive() and ssqrt(x ** 2) == x
        nm2 = 'h_restore_probe'
        with domain_state([nm2], 'positive'):
            assert SR.symbol(nm2).is_positive()
        assert not SR.symbol(nm2).is_positive()
    finally:
        for a in list(assumptions()):
            if str(a).split(' ')[0] == nm:
                forget(a)
        _clear_domain(nm)

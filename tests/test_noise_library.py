"""The noise-source library (api/noise.py): every CGF against textbook cumulants, K(0) = 0, the
sign/normalisation convention of the hand-written actions, and the error cases."""
from math import comb, factorial

import pytest

pytest.importorskip('sage.all')
from sage.all import SR, exp, function                                  # noqa: E402

from api.noise import (THETA, CGF, Bernoulli, Binomial, CompoundPoisson,  # noqa: E402
                       Cumulants, Dirac, Exponential, FinitePMF, Gamma, GammaProcess,
                       Gaussian, GaussianJump, InverseGaussianProcess, Laplace, NoiseError,
                       Poisson, Uniform, expr_to_text, parse_expr)

V = SR.var
lam, a, b, s, k, p, mu, sig, lo, hi, D = (V(n) for n in
                                          ('lam', 'a', 'b', 's', 'k', 'p', 'mu', 'sig', 'lo', 'hi', 'D'))
M = range(1, 5)


def _eq(x, y):
    return (SR(x) - SR(y)).simplify_full().is_zero()


def _rising(x, m):
    out = SR(1)
    for r in range(m):
        out *= x + r
    return out


# (source, textbook cumulant rate of order m)
CASES = {
    'gaussian': (Gaussian(var='v0', mean='m0'), lambda m: {1: V('m0'), 2: V('v0')}.get(m, 0)),
    'poisson': (Poisson(rate='lam', size='s'), lambda m: lam * s ** m),
    'cp_dirac': (CompoundPoisson('lam', Dirac('a')), lambda m: lam * a ** m),
    'cp_exponential': (CompoundPoisson('lam', Exponential('b')), lambda m: lam * factorial(m) * b ** m),
    'cp_gamma': (CompoundPoisson('lam', Gamma('k', 's')), lambda m: lam * s ** m * _rising(k, m)),
    'cp_gaussian': (CompoundPoisson('lam', GaussianJump('mu', 'sig')),
                    lambda m: lam * {1: mu, 2: mu ** 2 + sig ** 2, 3: mu ** 3 + 3 * mu * sig ** 2,
                                     4: mu ** 4 + 6 * mu ** 2 * sig ** 2 + 3 * sig ** 4}[m]),
    'cp_laplace': (CompoundPoisson('lam', Laplace('b')),
                   lambda m: lam * {1: 0, 2: 2 * b ** 2, 3: 0, 4: 24 * b ** 4}[m]),
    'cp_uniform': (CompoundPoisson('lam', Uniform('lo', 'hi')),
                   lambda m: lam * (hi ** (m + 1) - lo ** (m + 1)) / ((m + 1) * (hi - lo))),
    'cp_bernoulli': (CompoundPoisson('lam', Bernoulli('p')), lambda m: lam * p),
    'cp_binomial': (CompoundPoisson('lam', Binomial(5, 'p')),
                    lambda m: lam * sum(comb(5, j) * p ** j * (1 - p) ** (5 - j) * j ** m
                                        for j in range(6))),
    'cp_finite_pmf': (CompoundPoisson('lam', FinitePMF([1, -2, 'a'], ['p', '1/2 - p', '1/2'])),
                      lambda m: lam * (p + (SR(1) / 2 - p) * (-2) ** m + a ** m / 2)),
    'gamma_process': (GammaProcess('k', 's'), lambda m: k * factorial(m - 1) * s ** m),
    'inverse_gaussian': (InverseGaussianProcess('mu', 'lam'),
                         lambda m: {1: mu, 2: mu ** 3 / lam, 3: 3 * mu ** 5 / lam ** 2,
                                    4: 15 * mu ** 7 / lam ** 3}[m]),
    'cumulants': (Cumulants(['k2', 'k3', 'k4'], mean='k1'), lambda m: V(f'k{m}')),
    'cgf': (CGF('lam*(cosh(a*theta) - 1)'), lambda m: lam * a ** m if m % 2 == 0 else 0),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_cumulants_match_textbook(name):
    src, ref = CASES[name]
    for m in M:
        assert _eq(src.cumulant(m), ref(m)), (name, m, src.cumulant(m), ref(m))
    assert _eq(src.mean(), ref(1))


@pytest.mark.parametrize('name', sorted(CASES))
def test_cgf_vanishes_at_zero(name):
    src, _ = CASES[name]
    K = src.cgf()
    K0 = K.subs({THETA: 0}) if src.regular else K.limit(theta=0)
    assert _eq(K0, 0)
    # The action form (Taylor polynomial for a removable singularity) vanishes at 0 as well.
    assert _eq(src.action_cgf(6).subs({THETA: 0}), 0)


def test_uniform_action_form_is_the_taylor_polynomial():
    src = CompoundPoisson('lam', Uniform('lo', 'hi'))
    assert not src.regular
    poly = src.action_cgf(5)
    assert poly.is_polynomial(THETA) and poly.degree(THETA) == 5
    for m in range(1, 6):
        assert _eq(poly.diff(THETA, m).subs({THETA: 0}), src.cumulant(m))


def test_normalisation_matches_hand_written_actions():
    """S contains -K(xt): Gaussian var = 2D gives -D*xt^2, Poisson rate phi gives -(exp(nt)-1)*phi."""
    xt, nt = V('xt'), V('nt')
    phi = function('phi')(V('v'))
    assert _eq(Gaussian(var='2*D').cgf(xt), D * xt ** 2)
    assert _eq(Poisson(rate='phi(v)').cgf(nt), (exp(nt) - 1) * phi)
    assert _eq(CompoundPoisson('lam', Dirac('a')).cgf(xt), Poisson(rate='lam', size='a').cgf(xt))
    assert expr_to_text(-Gaussian(var='2*D').cgf(V('xt__i'))) == '-D*xt[i]^2'


def test_field_dependent_parameters_and_index_notation():
    src = Gaussian(var='s0 + s1*x')
    assert _eq(src.cumulant(2), V('s0') + V('s1') * V('x'))
    assert src.free_names() == {'s0', 's1', 'x'}
    hawkes = Poisson(rate='phi[i](v[i])')
    assert hawkes.free_names() == {'phi', 'v'}
    assert expr_to_text(hawkes.cgf(V('nt__i'))) == '(exp(nt[i]) - 1)*phi[i](v[i])'
    rate = Poisson(rate='a*exp(v)')
    assert _eq(rate.cumulant(3), a * exp(V('v')))
    # round trip through the action-text printer
    e = parse_expr('-(exp(xt[i])-1)*phi[i](v[i]) - D*xt[i]^2 + w[i,j]/tau')
    assert _eq(parse_expr(expr_to_text(e)), e)


def test_validate_rejects_undeclared_symbols():
    src = Poisson(rate='a*exp(v)')
    src.validate({'a', 'v'})
    with pytest.raises(NoiseError, match='undeclared'):
        src.validate({'a'})
    with pytest.raises(NoiseError, match='undeclared'):
        CompoundPoisson('lam', Exponential('b')).validate({'lam'})


@pytest.mark.parametrize('make, match', [
    (lambda: CGF('theta^2/2 + 1'), 'K\\(0\\)'),
    (lambda: CGF('lam'), 'depend on theta'),
    (lambda: CGF('t*theta^2'), 'time'),
    (lambda: Gaussian(var='2*D*t'), 'time'),
    (lambda: Poisson(rate='lam', size='theta'), 'theta'),
    (lambda: CompoundPoisson('lam', Gaussian(var=1)), 'jump distribution'),
    (lambda: CompoundPoisson('lam', Dirac('a*t')), 'time'),
    (lambda: FinitePMF([1, 2], [0.5, 0.6]), 'sum to'),
    (lambda: FinitePMF([1, 2], [1]), 'equal length'),
    (lambda: Cumulants([]), 'kappa2'),
    (lambda: Gaussian(var='sum(D[j] for j in pop)'), 'population sums'),
    (lambda: Gaussian(var='2*'), 'cannot parse'),
])
def test_error_cases(make, match):
    with pytest.raises(NoiseError, match=match):
        make()

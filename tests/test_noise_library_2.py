"""Noise library, round 2 (item 48): the finite-moment Levy sources / jump laws / composites of api/noise.py
and their samplers (api/noise_samplers.py).

A  every CGF: cumulant rates (orders 1-4) against independent closed forms, action round trip, domain errors;
B  every sampler: sample cumulants / jump moments against the exact values (fixed seeds, batch standard errors);
C  one end-to-end check per family: OU driven by the source, tree k = 2 and k = 3 against the exact stationary
   cumulants kappa_n/(n mu) and against simulation;
F  the reserved-name fix (a declared parameter ``e`` was read as Euler's number inside noise parameters)."""
from math import comb, factorial, gamma, lgamma, cos, tan, sqrt, exp, log

import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR                                                          # noqa: E402

import api.compute as AC                                                         # noqa: E402
import api.noise as N                                                            # noqa: E402
from api.noise import NoiseError                                                 # noqa: E402
from api.noise_samplers import euler_ou_cumulant, simulate_ou_paths              # noqa: E402
from api.sde import SDE, SDEError                                                # noqa: E402
from tests.test_sde_frontend import _same, _to_sr                                # noqa: E402

M4 = (1, 2, 3, 4)


def _f(src, m):
    return float(SR(src.cumulant(m)))


# ── closed-form cumulant rates (independent derivations) ───────────────────────────────────────

def _vg(mu, s, nu, d):
    return {1: d + mu, 2: s ** 2 + nu * mu ** 2, 3: 2 * nu ** 2 * mu ** 3 + 3 * s ** 2 * nu * mu,
            4: 3 * s ** 4 * nu + 12 * s ** 2 * mu ** 2 * nu ** 2 + 6 * nu ** 3 * mu ** 4}


def _nig(a, b, d, mu):
    g = sqrt(a * a - b * b)
    return {1: mu + d * b / g, 2: d * a * a / g ** 3, 3: 3 * d * a * a * b / g ** 5,
            4: 3 * d * a * a * (a * a + 4 * b * b) / g ** 7}


def _cgmy(C, G, M, Y):
    return {m: C * gamma(m - Y) * (M ** (Y - m) + (-1) ** m * G ** (Y - m)) for m in M4}


def _meixner(a, b, d, mu):
    t, sec2 = tan(b / 2), 1 / cos(b / 2) ** 2
    return {1: mu + d * a * t, 2: d * a ** 2 / 2 * sec2, 3: d * a ** 3 / 2 * sec2 * t,
            4: d * a ** 4 / 4 * (2 * sec2 * t * t + sec2 ** 2)}


def _nbp(r, p):
    return {1: r * p / (1 - p), 2: r * p / (1 - p) ** 2, 3: r * p * (1 + p) / (1 - p) ** 3,
            4: r * p * (1 + 4 * p + p * p) / (1 - p) ** 4}


SOURCES = {
    'vg': (lambda: N.VarianceGamma(0.1, 0.3, 0.5, drift=0.05), _vg(0.1, 0.3, 0.5, 0.05)),
    'nig': (lambda: N.NormalInverseGaussian(2, 0.5, 1.2, mu=0.1), _nig(2, 0.5, 1.2, 0.1)),
    'cgmy_half': (lambda: N.CGMY(1, 2, 3, '1/2'), _cgmy(1, 2, 3, 0.5)),
    'cgmy_neg': (lambda: N.CGMY(1, 2, 3, '-1/2'), _cgmy(1, 2, 3, -0.5)),
    'cgmy_32': (lambda: N.CGMY(1, 2, 3, '3/2'), _cgmy(1, 2, 3, 1.5)),
    'ts_sub': (lambda: N.TemperedStableSubordinator('1/2', 1, 2),
               {m: gamma(m - 0.5) * 2 ** (0.5 - m) for m in M4}),
    'meixner': (lambda: N.Meixner(1, 0.3, 1, mu=0.1), _meixner(1, 0.3, 1, 0.1)),
    'meixner_neg': (lambda: N.Meixner('3/2', -0.5, 2), _meixner(1.5, -0.5, 2, 0)),
    'bilateral_gamma': (lambda: N.BilateralGamma(1, 2, 3, 4),
                        {m: factorial(m - 1) * (1 / 2 ** m + (-1) ** m * 3 / 4 ** m) for m in M4}),
    'skellam': (lambda: N.Skellam(1, 2), {m: 1 + (-1) ** m * 2 for m in M4}),
    'neg_binomial_process': (lambda: N.NegativeBinomialProcess(2, '1/3'), _nbp(2, 1 / 3)),
    'merton': (lambda: N.merton_jump_diffusion(0.1, 0.3, 2, 0.2, 0.4),
               {1: 0.1 + 2 * 0.2, 2: 0.09 + 2 * (0.04 + 0.16),
                3: 2 * (0.2 ** 3 + 3 * 0.2 * 0.16), 4: 2 * (0.2 ** 4 + 6 * 0.04 * 0.16 + 3 * 0.16 ** 2)}),
    'kou': (lambda: N.kou_jump_diffusion(0.0, 0.2, 3, 0.4, 5, 4),
            {1: 3 * (0.4 / 5 - 0.6 / 4), 2: 0.04 + 3 * (2 * 0.4 / 25 + 2 * 0.6 / 16),
             3: 3 * (6 * 0.4 / 125 - 6 * 0.6 / 64), 4: 3 * (24 * 0.4 / 625 + 24 * 0.6 / 256)}),
}


@pytest.mark.parametrize('name', sorted(SOURCES))
def test_source_cumulants_match_closed_forms(name):
    make, ref = SOURCES[name]
    src = make()
    for m in M4:
        assert _f(src, m) == pytest.approx(ref[m], rel=1e-12, abs=1e-13), (name, m)


def test_symbolic_cumulants():
    """exact symbolic identities for the algebraic families."""
    mu, s, nu, d = (SR.var(n) for n in ('mu', 's', 'nu', 'd'))
    vg = N.VarianceGamma('mu', 's', 'nu', drift='d')
    for m, ref in _vg(mu, s, nu, d).items():
        assert _same(vg.cumulant(m), ref)
    a, b, dl = (SR.var(n) for n in ('a', 'b', 'dl'))
    sk = N.Skellam('a', 'b')
    assert all(_same(sk.cumulant(m), a + (-1) ** m * b) for m in M4)
    bg = N.BilateralGamma('a', 'b', 'dl', 'mu')
    assert all(_same(bg.cumulant(m), factorial(m - 1) * (a / b ** m + (-1) ** m * dl / mu ** m)) for m in M4)


def test_cgmy_y_inside_gamma_neighbourhood_is_continuous():
    """CGMY(Y -> 0) -> bilateral gamma with alpha = C, lam = M, G (the Y = 0 limit is refused, Y near 0 is fine)."""
    c = N.CGMY(1, 2, 3, 1e-3)
    b = N.BilateralGamma(1, 3, 1, 2)
    for m in M4:
        assert _f(c, m) == pytest.approx(_f(b, m), rel=5e-3)


# jump laws: raw moments (independent)
def _pmf_moments(pmf, support):
    return {m: sum(k ** m * pmf(k) for k in support) for m in M4}


def _asym(b1, b2):
    return {m: sum(comb(m, j) * b1 ** j * factorial(j) * (-b2) ** (m - j) * factorial(m - j)
                   for j in range(m + 1)) for m in M4}


_PH_ALPHA, _PH_T = [0.5, 0.3], [[-2.0, 1.0], [0.5, -3.0]]


def _ph_moments():
    T = np.array(_PH_T)
    inv = np.linalg.inv(-T)
    al = np.array(_PH_ALPHA)
    return {m: factorial(m) * float(al @ np.linalg.matrix_power(inv, m) @ np.ones(2)) for m in M4}


def _nb_pmf(r, p):
    return lambda k: exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1) + r * log(1 - p) + k * log(p))


JUMPS = {
    'geometric': (lambda: N.Geometric('1/3'), _pmf_moments(lambda k: (1 / 3) * (2 / 3) ** (k - 1), range(1, 400))),
    'poisson_jump': (lambda: N.PoissonJump(2), _pmf_moments(lambda k: exp(-2) * 2 ** k / factorial(k), range(0, 80))),
    'neg_binomial': (lambda: N.NegativeBinomial(2, '1/3'), _pmf_moments(_nb_pmf(2, 1 / 3), range(0, 300))),
    'logarithmic': (lambda: N.Logarithmic('1/3'),
                    _pmf_moments(lambda k: -(1 / 3) ** k / (k * log(2 / 3)), range(1, 300))),
    'double_exponential': (lambda: N.DoubleExponential('2/5', 5, 4),
                           {m: factorial(m) * (0.4 / 5 ** m + 0.6 * (-1) ** m / 4 ** m) for m in M4}),
    'asymmetric_laplace': (lambda: N.AsymmetricLaplace(1, '1/2'), _asym(1.0, 0.5)),
    'hyperexponential': (lambda: N.Hyperexponential(['1/4', '3/4'], [1, '1/3']),
                         {m: factorial(m) * (0.25 + 0.75 / 3 ** m) for m in M4}),
    'phase_type': (lambda: N.PhaseType(_PH_ALPHA, _PH_T), _ph_moments()),
    'chi_squared': (lambda: N.ChiSquared(3), {m: float(np.prod([3 + 2 * j for j in range(m)])) for m in M4}),
    'log_normal': (lambda: N.LogNormal(0.1, '1/2', order=4),
                   {m: exp(m * 0.1 + m * m * 0.125) for m in M4}),
    'weibull': (lambda: N.Weibull(1.5, 2, order=4), {m: 1.5 ** m * gamma(1 + m / 2) for m in M4}),
}


@pytest.mark.parametrize('name', sorted(JUMPS))
def test_jump_moments_match(name):
    make, ref = JUMPS[name]
    cp = N.CompoundPoisson(1, make())               # cumulant rate of order m = rate * E[J^m]
    for m in M4:
        assert _f(cp, m) == pytest.approx(ref[m], rel=1e-9, abs=1e-12), (name, m)


def test_truncated_series_is_refused_beyond_its_order():
    cp = N.CompoundPoisson(1, N.LogNormal(0, 1, order=3))
    assert cp.cumulant(3) > 0
    with pytest.raises(NoiseError, match='truncated'):
        cp.cumulant(4)
    with pytest.raises(NoiseError, match='truncated'):
        cp.cgf_series(5)


def test_phase_type_special_cases_and_atom():
    """Erlang(2, rate 3) = Gamma(2, 1/3); an exponential with sub-stochastic alpha has an atom at 0."""
    er = N.PhaseType([1, 0], [[-3, 3], [0, -3]])
    g = N.Gamma(2, '1/3')
    th = SR.var('theta')
    assert _same(er.mgf(), g.mgf())
    atom = N.PhaseType(['1/2'], [[-2]])
    assert _same(atom.mgf(), SR(1) / 2 + SR(1) / 2 * 2 / (2 - th))


# ── action round trip ──────────────────────────────────────────────────────────────────────────

class _Idx:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, key):
        return SR.var(f'{self.name}_{int(key)}')


@pytest.mark.parametrize('name', sorted(SOURCES))
def test_action_round_trip(name):
    src = SOURCES[name][0]()
    s = SDE('rt').parameter('mu', default=1.0, domain='positive').field('x')
    text = s.drift('x', '-mu*x').noise('L', src, couples={'x': 1}).action_text()
    got = _to_sr(text, indexed=('x', 'xt'), scalars=('mu',), i=0)
    x0, xt0, Dt, mu = SR.var('x_0'), SR.var('xt_0'), SR.var('Dt'), SR.var('mu')
    want = xt0 * (Dt * x0 + mu * x0) - src.cgf(theta=xt0)
    for pt in ({'x_0': 0.3, 'xt_0': 0.05, 'Dt': 0.7, 'mu': 1.3}, {'x_0': -0.2, 'xt_0': -0.02, 'Dt': 0.1, 'mu': 0.6}):
        assert abs(complex(got.subs({SR.var(k): v for k, v in pt.items()}))
                   - complex(want.subs({SR.var(k): v for k, v in pt.items()}))) < 1e-12


def test_jump_law_action_round_trip():
    src = N.CompoundPoisson('lam', N.PhaseType([1], [[-2]]))
    s = SDE('rt2').parameter('mu', default=1.0, domain='positive').parameter('lam', default=2.0, domain='positive')
    text = s.field('x').drift('x', '-mu*x').noise('L', src, couples={'x': 1}).action_text()
    got = _to_sr(text, indexed=('x', 'xt'), scalars=('mu', 'lam'), i=0)
    xt0 = SR.var('xt_0')
    sub = {SR.var('x_0'): 0.4, xt0: 0.3, SR.var('Dt'): 0.2, SR.var('mu'): 1.1, SR.var('lam'): 2.0}
    want = xt0 * (SR.var('Dt') * SR.var('x_0') + SR.var('mu') * SR.var('x_0')) - 2.0 * (2.0 / (2.0 - xt0) - 1)
    assert abs(complex(got.subs(sub)) - complex(want.subs(sub))) < 1e-12


@pytest.mark.parametrize('make, match', [
    (lambda: N.VarianceGamma(0, 1, -1), 'nu'),
    (lambda: N.VarianceGamma(0, 0, 1), 'sigma'),
    (lambda: N.NormalInverseGaussian(1, 2, 1), r'\|beta\| < alpha'),
    (lambda: N.NormalInverseGaussian(1, 0, 0), 'delta'),
    (lambda: N.CGMY(1, 1, 1, 0), 'Y'),
    (lambda: N.CGMY(1, 1, 1, 1), 'Y'),
    (lambda: N.CGMY(1, 1, 1, 2), 'Y'),
    (lambda: N.CGMY(-1, 1, 1, '1/2'), 'C'),
    (lambda: N.TemperedStableSubordinator(1.5, 1, 1), 'alpha'),
    (lambda: N.Meixner(1, 4, 1), r'\|beta\| < pi'),
    (lambda: N.BilateralGamma(1, -2, 1, 1), 'lam_p'),
    (lambda: N.Skellam(-1, 1), 'rate_plus'),
    (lambda: N.NegativeBinomialProcess(1, 1.5), 'p'),
    (lambda: N.Geometric(0), 'p'),
    (lambda: N.Logarithmic(1), 'p'),
    (lambda: N.DoubleExponential(1.5, 1, 1), 'p'),
    (lambda: N.Hyperexponential([0.5, 0.4], [1, 2]), 'sum to'),
    (lambda: N.PhaseType([1], [[1]]), 'negative'),
    (lambda: N.PhaseType([1, 0], [[-1, 2], [0, -1]]), 'sums to'),
    (lambda: N.PhaseType([1, 1], [[-1, 0], [0, -1]]), 'exceeds 1'),
    (lambda: N.LogNormal(0, -1), 'sigma'),
    (lambda: N.Weibull(1, 0), 'shape'),
    (lambda: N.LogNormal(0, 1, order=0), 'order'),
    (lambda: N.SumSource(N.Skellam(1, 1)), 'at least two'),
])
def test_domain_errors(make, match):
    with pytest.raises(NoiseError, match=match):
        make()


def test_symbolic_parameters_pass_domain_checks():
    assert N.CGMY('C', 'G', 'M', 'Y').free_names() == {'C', 'G', 'M', 'Y'}      # symbolic: domain not checked
    with pytest.raises(NoiseError, match='undeclared'):
        N.VarianceGamma('mu', 's', 'nu').validate(['mu'])


# ── F: reserved names ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('bad', ['e', 'pi', 'I', 'exp', 'Dt', 'theta', 'sum'])
def test_reserved_declared_names_are_refused(bad):
    with pytest.raises(SDEError, match='reserved'):
        SDE('r').parameter(bad, default=1.0)
    with pytest.raises(SDEError, match='reserved'):
        SDE('r').define_function(bad, ['v'], 'v')
    with pytest.raises(SDEError, match='reserved'):
        SDE('r').field(bad)
    with pytest.raises(SDEError, match='reserved'):
        SDE('r').noise('n', N.Poisson(1), expose=bad)


def test_population_labels_are_not_symbols():
    for ok in ('I', 'E', 'e', 'pop'):
        SDE('r').population(ok)                            # loop-set labels, never Sage symbols
    for bad in ('Dt', 'for', 'lambda'):
        with pytest.raises(SDEError, match='reserved'):
            SDE('r').population(bad)


def test_parameter_named_like_euler_was_a_silent_wrong_action():
    """the bug: ``Poisson(rate='e')`` printed exp(1) while the drift's ``e`` was the parameter."""
    s = SDE('t').parameter('a', default=2.0, domain='positive').field('x')
    s.drift('x', '-x + a').noise('xi', N.Poisson(rate='a', size='a'), couples={'x': 1})
    assert 'exp(1)' not in s.action_text()
    assert str(N.parse_expr('e')) == 'e'                      # Sage's constant stays what a bare ``e`` means in a noise parameter


# ── B: samplers ────────────────────────────────────────────────────────────────────────────────

def _batch_check(x, exact, nb=40, nsig=5.0, label=''):
    """unbiased k-statistics per batch; |mean - exact| <= nsig * batch standard error."""
    from scipy.stats import kstat
    xb = np.array_split(x, nb)
    for m in M4:
        ks = np.array([kstat(b, m) for b in xb])
        se = ks.std(ddof=1) / sqrt(nb)
        assert abs(ks.mean() - exact[m]) <= nsig * se + 1e-12, (label, m, ks.mean(), exact[m], se)


def _sampling_cases():
    return {k: v[0] for k, v in SOURCES.items()
            if k not in ('cgmy_32',)}


@pytest.mark.parametrize('name', sorted(_sampling_cases()))
@pytest.mark.parametrize('dt', [1.0, 0.1])
def test_source_sampler_cumulants(name, dt):
    src = _sampling_cases()[name]()
    rng = np.random.default_rng(20261015)
    x = src.sample(dt, 200_000, rng)
    _batch_check(x, {m: _f(src, m) * dt for m in M4}, label=(name, dt))


def test_cgmy_series_sampler_small_and_slow_regime():
    """1 < Y < 2 is a controlled approximation (Gaussian small jumps); orders 1-2 are matched exactly by
    construction, orders 3-4 to about (eps max(M,G))^(4-Y)."""
    src = N.CGMY(1, 2, 3, '3/2')
    rng = np.random.default_rng(7)
    dt = 0.1
    x = src.sample(dt, 20_000, rng)
    _batch_check(x, {m: _f(src, m) * dt for m in M4}, nb=20, nsig=5.0, label='cgmy32')


@pytest.mark.parametrize('name', sorted(JUMPS))
def test_jump_sampler_moments(name):
    law = JUMPS[name][0]()
    rng = np.random.default_rng(31)
    j = law.sample(rng, 400_000)
    for m in M4:
        raw = j ** m
        xb = np.array_split(raw, 40)
        means = np.array([b.mean() for b in xb])
        se = means.std(ddof=1) / sqrt(40)
        assert abs(means.mean() - JUMPS[name][1][m]) <= 5 * se + 1e-12, (name, m)


def test_sampler_edge_cases():
    rng = np.random.default_rng(0)
    with pytest.raises(NoiseError, match='no sampler'):
        N.CGF('theta^2').sample(1.0, 4, rng)
    with pytest.raises(NoiseError, match='no sampler'):
        N.Cumulants([1.0]).sample(1.0, 4, rng)
    with pytest.raises(NoiseError, match='dt'):
        N.Skellam(1, 1).sample(0.0, 4, rng)
    with pytest.raises(NoiseError, match='values'):
        N.Poisson('a', 1).sample(1.0, 4, rng)
    # zero events -> exact zeros; field-dependent rate through values
    assert (N.CompoundPoisson(0, N.Dirac(1)).sample(1.0, 5, rng) == 0).all()
    a = np.array([0.0, 50.0])
    x = N.Poisson('a', 1).sample(1.0, 2, rng, values={'a': a})
    assert x[0] == 0 and x[1] > 20


def test_meixner_inversion_table_is_accurate():
    """deterministic check of the inversion table (no sampling noise): moments of the tabulated law."""
    from api.noise_samplers import _meixner_table
    for alpha, beta, d in ((1.0, 0.3, 1.0), (1.0, -0.5, 0.05), (2.0, 0.9, 7.0)):
        cdf, z = _meixner_table(alpha, beta, d)
        pdf_mass = np.diff(cdf)
        zm = 0.5 * (z[1:] + z[:-1])
        mean = (pdf_mass * zm).sum()
        var = (pdf_mass * (zm - mean) ** 2).sum()
        assert mean == pytest.approx(d * tan(beta / 2), rel=1e-5)
        assert var == pytest.approx(d / (2 * cos(beta / 2) ** 2), rel=1e-5)


def test_telegraph_noise_is_a_state_dependent_poisson_source():
    """dichotomous noise s in {+a, -a}: flips at rate g = a Poisson event with the state-dependent jump -2 s.
    The source and its action are expressible; a simulator must apply the jump per event (counts from
    Poisson(rate, 1).sample, ``s *= (-1)^n``).  The perturbative pipeline cannot use it (docs/noise_library.md)."""
    g, a = 1.5, 0.8
    src = N.Poisson(rate='g', size='-2*s')
    th, ss = SR.var('th'), SR.var('s')
    assert _same(src.cgf(theta=th), SR.var('g') * ((-2 * ss * th).exp() - 1))
    count = N.Poisson(rate='g', size=1)
    rng = np.random.default_rng(5)
    n, dt, steps = 40000, 0.05, 10      # tau = 0.5: the expected value 0.14 >> its 5 sigma (0.016)
    s = rng.choice([-a, a], n)
    s0 = s.copy()
    for _ in range(steps):
        s = s * (-1.0) ** count.sample(dt, n, rng, values={'g': g})
    assert set(np.unique(np.abs(s))) == {a}
    tau = steps * dt
    assert (s * s0).mean() == pytest.approx(a * a * exp(-2 * g * tau), abs=5 * a * a / sqrt(n))


# ── C: end-to-end per family ───────────────────────────────────────────────────────────────────

MU = 1.3

E2E = {
    'vg': lambda: N.VarianceGamma(0.1, 0.3, 0.5),
    'nig': lambda: N.NormalInverseGaussian(2, 0.5, 1.2),
    'cgmy': lambda: N.CGMY(1, 2, 3, '1/2'),
    'ts_sub': lambda: N.TemperedStableSubordinator('1/2', 1, 2),
    'meixner': lambda: N.Meixner(1, 0.3, 1),
    'bilateral_gamma': lambda: N.BilateralGamma(1, 2, 3, 4),
    'skellam': lambda: N.Skellam(1, 2),
    'neg_binomial_process': lambda: N.NegativeBinomialProcess(2, '1/3'),
    'merton': lambda: N.merton_jump_diffusion(0.1, 0.3, 2, 0.2, 0.4),
    'cp_phase_type': lambda: N.CompoundPoisson(2, N.PhaseType(_PH_ALPHA, _PH_T)),
    'cp_log_normal': lambda: N.CompoundPoisson(2, N.LogNormal(0.1, '1/2', order=8)),
}


def _ou_model(src):
    return (SDE('ou').parameter('mu', default=MU, domain='positive').field('x')
            .drift('x', '-mu*x').noise('L', src, couples={'x': 1}).stability_analysis(True).build())


def _tree(model, k):
    res = AC.compute_cumulants(model, k=k, max_ell=0, external_fields=[('dx', 1)] * k, parameters={'mu': MU},
                               tau_grid=np.array([0.0, 1.0]), use_cache=False, parallel=False, verbose=False)
    if k == 2:
        return complex(res['C_tau_by_ell'][0][0]).real
    return complex(res['total_C_by_ell'][0](*([0.0] * k))).real


@pytest.mark.parametrize('name', sorted(E2E))
def test_ou_tree_k2_k3_match_exact_stationary_cumulants(name):
    src = E2E[name]()
    model = _ou_model(src)
    for k in (2, 3):
        want = _f(src, k) / (k * MU)
        assert _tree(model, k) == pytest.approx(want, rel=1e-11), (name, k)


@pytest.mark.parametrize('name', ['vg', 'nig', 'meixner', 'bilateral_gamma', 'neg_binomial_process', 'cp_phase_type'])
def test_ou_simulation_matches_euler_exact_and_theory(name):
    """simulation (exact Levy increments, Euler OU step) vs the exact stationary cumulants of the Euler chain,
    which differ from the continuous kappa_n/(n mu) by O(mu dt) (asserted)."""
    src = E2E[name]()
    dt, rng = 0.02, np.random.default_rng(11)
    x = simulate_ou_paths(src, MU, T=dt, dt=dt, n_paths=60_000, rng=rng)[:, -1]
    ref = {m: euler_ou_cumulant(_f(src, m), m, MU, dt) for m in (2, 3)}
    from scipy.stats import kstat
    xb = np.array_split(x, 30)
    for m in (2, 3):
        ks = np.array([kstat(b, m) for b in xb])
        assert abs(ks.mean() - ref[m]) <= 5 * ks.std(ddof=1) / sqrt(30), (name, m)
        cont = _f(src, m) / (m * MU)
        assert abs(ref[m] / cont - 1) < 2 * MU * dt


# ── composites ─────────────────────────────────────────────────────────────────────────────────

def test_levy_driven_ou_builder():
    s = N.levy_driven_ou(N.VarianceGamma(0.1, 0.3, 0.5), rate='mu', rate_default=MU)
    assert 'mu' in s.action_text()
    model = s.stability_analysis(True).build()
    src = N.VarianceGamma(0.1, 0.3, 0.5)
    assert _tree(model, 2) == pytest.approx(_f(src, 2) / (2 * MU), rel=1e-12)


def test_sum_operator_and_flattening():
    a, b, c = N.Skellam(1, 2), N.Gaussian(1), N.Poisson(1)
    s = a + b + c
    assert isinstance(s, N.SumSource) and len(s.parts) == 3
    assert _same(s.cgf(), a.cgf() + b.cgf() + c.cgf())


def test_config_accepts_new_source_types():
    from api.config_run import SDE_NOISE_TYPES
    assert {'VarianceGamma', 'CGMY', 'Meixner', 'NormalInverseGaussian'} <= set(SDE_NOISE_TYPES)


def test_field_dependent_parameters_through_values():
    """array-valued parameters: compound-Poisson jump scale per entry; scalar-only samplers loop per entry."""
    rng = np.random.default_rng(3)
    m = np.concatenate([np.full(20_000, 0.5), np.full(20_000, 2.0)])
    x = N.CompoundPoisson(4, N.Exponential('m')).sample(1.0, 40_000, rng, values={'m': m})
    assert x[:20_000].mean() == pytest.approx(4 * 0.5, rel=0.03)
    assert x[20_000:].mean() == pytest.approx(4 * 2.0, rel=0.03)
    mv = N.Meixner('a', 0.3, 1).sample(0.5, 3, rng, values={'a': np.array([0.5, 1.0, 2.0])})
    assert mv.shape == (3,) and np.isfinite(mv).all()
    ts = N.TemperedStableSubordinator('1/2', 'c', 2).sample(0.5, 3, rng, values={'c': np.array([0.5, 1.0, 2.0])})
    assert (ts > 0).all()


def test_names_that_merely_contain_reserved_words_are_fine():
    s = (SDE('ok').parameter('e0', default=1.0, domain='positive').parameter('pie', default=1.0, domain='positive')
         .parameter('Imu', default=1.0, domain='positive').field('x'))
    s.drift('x', '-pie*x + e0').noise('xi', N.Poisson(rate='e0', size='Imu'), couples={'x': 1})
    assert 'exp(1)' not in s.action_text()


# ── review findings (independent reviewer, regression tests) ───────────────────────────────────

def test_truncated_series_guard_covers_sums_and_the_action():
    ln = N.CompoundPoisson(1, N.LogNormal(0, '1/2', order=3))
    summed = N.Gaussian(var=1) + ln
    with pytest.raises(NoiseError, match='truncated'):
        summed.cumulant(5)
    with pytest.raises(NoiseError, match='truncated'):
        ln.action_cgf(8)
    s = SDE('tr').parameter('mu', default=1.0, domain='positive').field('x').drift('x', '-mu*x')
    s.noise('L', ln, couples={'x': 1})
    with pytest.raises(NoiseError, match='truncated'):
        s.action_text()                                    # the builder's default series order is 8 > 3
    assert 'exp(' in s.series_order(3).action_text()


def test_k0_tolerance_is_relative_to_the_rates():
    big = N.CompoundPoisson(1e8, N.DoubleExponential(0.7, 3.3, 2.1))
    assert _f(big, 1) == pytest.approx(1e8 * (0.7 / 3.3 - 0.3 / 2.1), rel=1e-12)
    with pytest.raises(NoiseError, match='K\\(0\\)'):
        N.CGF('lam*exp(a*theta)')


def test_population_I_and_reserved_expose_names():
    s = SDE('p').population('I', size=2).field('x', population='I').parameter('mu', default=1.0, domain='positive')
    assert 'for i in I' in s.drift('x', '-mu*x').noise('L', N.Poisson('mu', 1), couples={'x': 1}).action_text()
    with pytest.raises(SDEError, match='reserved'):
        SDE('q').field('x').noise('n', N.Poisson(1), expose='e')
    with pytest.raises(SDEError, match='reserved'):
        SDE('q').parameter('lambda', default=1.0)


def test_phase_type_requires_an_absorbing_chain():
    with pytest.raises(NoiseError, match='singular'):
        N.PhaseType([1, 0], [[-1, 1], [1, -1]])


def test_cgmy_series_sampler_is_chunked():
    src = N.CGMY(2, 100, 0.5, '3/2')
    rng = np.random.default_rng(2)
    x = src.sample(0.1, 6_000, rng)
    assert x.shape == (6_000,)
    assert x.mean() == pytest.approx(_f(src, 1) * 0.1, abs=5 * sqrt(_f(src, 2) * 0.1 / 6_000))

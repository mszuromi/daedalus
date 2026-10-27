"""Public exact-oracle suite: the package against references that share no code with it.

Every model is built inline (or is a tracked model file), every reference comes from
``tests/oracles_public/refs.py`` (mpmath, 40-80 digits, no package import):

  * linear Gaussian SDEs (non-normal 3-field drift, correlated noise, a Jordan block, a 1e-7 pole
    split): Lyapunov equation + matrix exponential;
  * Levy-driven linear SDEs (compound Poisson + Gaussian), k = 2..4: the closed form
    kappa_k int prod_i [e^{A(t_i - s)} b] ds;
  * linear Hawkes (2 populations, exponential kernels): mean (I - W)^-1 mu and the smooth
    two-point function from the Markov embedding of the kernels driven by shot noise;
  * 1-D polynomial drifts loop by loop at unequal times (the backward generator in x / sqrt(D)
    with a Cauchy contour in sqrt(D)), including the equal-time Boltzmann series 1, -3, 24, -297;
  * the delayed OU process (exact BVP, method of steps); Erlang kernels (chain embedding);
  * the switch-on transient from t_min (Van Loan), Gaussian and Levy;
  * colored (OU) noise through its Markov embedding, one loop = the exact O(eps) mass shift;
  * the 1-D Allen-Cahn tree (erfc closed form) and one loop (tadpole mass shift), tau = 0 and 1.

Default engine (state space) everywhere; a few cells repeat on the pole engine
(``DAEDALUS_PHASE_J_ENGINE=poles``).  k = 2 at tau = 0 is the exact Ito left limit (the default).
Tolerance: 1e-12 relative to the largest reference value of the cell (the audit measured
<= 8.5e-14 on every route).  Contributions that must vanish (loops of a linear model, k >= 3 of a
Gaussian one) are checked against the same scale.
"""
import math

import mpmath as mp
import numpy as np
import pytest

import daedalus.api.noise as N
from daedalus.api.compute import compute_cumulants
from daedalus.api.delay import Kernel
from daedalus.api.sde import SDE
from tests.oracles_public import refs

RTOL = 1e-12


# ----------------------------------------------------------------------------- helpers

def _linear_sde(name, A, srcs, names):
    """dx = A x dt + sum_q b_q dL_q; srcs = [(b, NoiseSource), ...]."""
    s = SDE(name)
    for n in names:
        s = s.field(n)
    for i, n in enumerate(names):
        terms = [f'({A[i][j]!r})*{names[j]}' for j in range(len(names)) if A[i][j] != 0]
        s = s.drift(n, ' + '.join(terms) if terms else '0')
    for q, (b, src) in enumerate(srcs):
        s = s.noise(f'xi{q}', src, couples={names[i]: b[i] for i in range(len(names)) if b[i] != 0})
    return s.build()


def _run(model, k, ell, ext, times, params=None, **kw):
    """{ell: [complex value at each time tuple]} (k = 2: times are (0, tau) pairs)."""
    taus = sorted({t[1] - t[0] for t in times}) if k == 2 else [0.0, 1.0]
    res = compute_cumulants(model, k=k, max_ell=ell, external_fields=ext, parameters=params,
                            tau_grid=np.array(taus, float), parallel=False, verbose=False, **kw)
    out = {}
    for l in range(ell + 1):
        if k == 2:
            arr = np.asarray(res['C_tau_by_ell'][l])
            out[l] = [complex(arr[taus.index(t[1] - t[0])]) for t in times]
        else:
            out[l] = [complex(res['total_C_by_ell'][l](*t)) for t in times]
    return out


def _check(vals, ref, scale, what, rtol=RTOL):
    vals = np.asarray(vals, complex)
    ref = np.asarray([complex(r) for r in ref], complex)
    err = np.max(np.abs(vals - ref)) / scale
    assert err <= rtol, f'{what}: max |pkg - ref| / scale = {err:.3e} > {rtol:g}\npkg {vals}\nref {ref}'


def _vanish(vals, scale, what, rtol=RTOL):
    _check(vals, [0.0] * len(vals), scale, what + ' (must vanish)', rtol)


def _q(B, n):
    return [[sum(v * b[i] * b[j] for b, v in B) for j in range(n)] for i in range(n)]


@pytest.fixture
def poles(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')


# ----------------------------------------------------------------------------- the references themselves

def test_references_reproduce_textbook_closed_forms():
    """The references against closed forms they do not use internally (40 digits)."""
    bad = [(n, mp.nstr(abs(e), 3)) for n, e in _reference_self_checks() if abs(e) > 1e-25]
    assert not bad, bad


@refs._prec
def _reference_self_checks():
    worst = []
    v = refs.linear_k2([[-1.3]], [[2 * 0.4]], 0, 0, 0.7)
    worst.append(('OU', v / (mp.mpf(0.4) / mp.mpf(1.3) * mp.exp(-mp.mpf(1.3) * mp.mpf(0.7))) - 1))
    A = [[-1.0, 0.8], [-0.3, -0.6]]
    src = [([1, 0], {2: 0.4}), ([0.3, 1], {2: 0.1})]
    b, k2 = mp.mpf(0.3), mp.mpf(0.1)
    Q = [[mp.mpf(0.4) + k2 * b * b, k2 * b], [k2 * b, k2]]
    v2 = refs.linear_k2(A, Q, 0, 1, 0.9)
    worst.append(('Levy closed form k=2 vs Lyapunov', refs.linear_kpt(A, src, [0, 1], [0.0, 0.9]) / v2 - 1))
    with mp.workdps(80):     # Van Loan cancels ~e^{1.6 t} digits; the transient left at t = 40 is e^{-64}
        t1 = mp.mpf(40)          # (exact 40.9 - 40: the double 40.9 - 40 is 0.9 - 1.4e-15)
        worst.append(('switch-on -> stationary',
                      refs.switchon_k2(A, Q, 0, 1, t1, t1 + mp.mpf(0.9), 0.0) / v2 - 1))
    worst.append(('switch-on closed form vs Van Loan',
                  refs.linear_kpt(A, src, [0, 1], [1.0, 1.9], t_min=0.5)
                  / refs.switchon_k2(A, Q, 0, 1, 1.0, 1.9, 0.5) - 1))
    worst.append(('Levy k=3 equal time', refs.linear_kpt([[-0.8]], [([1], {3: 2.0})], [0, 0, 0], [0, 0, 0])
                  / (mp.mpf(2) / (3 * mp.mpf(0.8))) - 1))
    worst.append(('delayed OU, c = 0', refs.delayed_ou(1.2, 0.0, 0.9, 0.15, 1.4)
                  / (mp.mpf(0.15) / mp.mpf(1.2) * mp.exp(-mp.mpf(1.2) * mp.mpf(1.4))) - 1))
    P = refs.Poly1D(1.0, {3: -0.2}, pmax=6)
    lt = P.loop_term(0.7, [0.0, 0.8], [0, 1])
    worst.append(('poly tree', lt[0] / (mp.mpf(0.7) * mp.exp(-mp.mpf(0.8))) - 1))
    worst.append(('poly Hartree loop', lt[1] / (-3 * mp.mpf(0.2) * mp.mpf(0.7) ** 2 * (1 + mp.mpf(0.8))
                                                * mp.exp(-mp.mpf(0.8))) - 1))
    lt2 = refs.Poly1D(1.0, {3: -0.02}, pmax=8).loop_term(1.0, [0.0, 0.0], [0, 1, 2, 3])
    for l, c in ((1, -3), (2, 24), (3, -297)):
        worst.append((f'Boltzmann loop {l}', lt2[l] / (c * mp.mpf(0.02) ** l) - 1))
    worst.append(('odd Taylor coefficients', lt2['_odd_max']))
    lam = refs.hawkes_linear([0.7], [[0.4]], [[1.3]], 0, 0, 0, mean=True)
    worst.append(('Hawkes mean', lam / (mp.mpf(0.7) / mp.mpf(0.6)) - 1))
    w, tk = mp.mpf(0.4), mp.mpf(1.3)
    ex = lam * w * (2 - w) / (2 * (1 - w)) / tk * mp.exp(-(1 - w) * mp.mpf(0.9) / tk)
    worst.append(('Hawkes 1 population', refs.hawkes_linear([0.7], [[0.4]], [[1.3]], 0, 0, 0.9) / ex - 1))
    return worst


# ----------------------------------------------------------------------------- linear Gaussian (Lyapunov)

OU3_A = [[-1.0, 0.6, 0.0], [-0.4, -0.8, 0.3], [0.2, 0.0, -1.5]]
OU3_B = [([1, 0, 0], 0.6), ([0, 1, 0.5], 0.2)]
TAUS = [-1.3, -0.4, 0.0, 0.5, 1.7]


def _ou3():
    return _linear_sde('oracle ou3', OU3_A, [(b, N.Gaussian(var=v)) for b, v in OU3_B], ['x', 'y', 'z'])


@pytest.mark.parametrize('i,j', [(0, 0), (0, 1), (1, 0), (2, 1)])
def test_nonnormal_three_field_ou_two_point_is_lyapunov(i, j):
    names = ['x', 'y', 'z']
    Q = _q(OU3_B, 3)
    tt = [(0.0, t) for t in TAUS]
    vals = _run(_ou3(), 2, 1, [(names[i], 1), (names[j], 1)], tt)
    ref = [refs.linear_k2(OU3_A, Q, i, j, t) for t in TAUS]
    scale = abs(complex(refs.linear_k2(OU3_A, Q, 0, 0, 0)))
    _check(vals[0], ref, scale, f'C_{names[i]}{names[j]} tree')
    _vanish(vals[1], scale, f'C_{names[i]}{names[j]} one loop')


def test_nonnormal_three_field_ou_two_point_on_the_pole_engine(poles):
    Q = _q(OU3_B, 3)
    tt = [(0.0, t) for t in TAUS]
    vals = _run(_ou3(), 2, 0, [('x', 1), ('z', 1)], tt)
    ref = [refs.linear_k2(OU3_A, Q, 0, 2, t) for t in TAUS]
    _check(vals[0], ref, abs(complex(refs.linear_k2(OU3_A, Q, 0, 0, 0))), 'C_xz tree (poles)')


def test_gaussian_three_point_function_vanishes():
    tt = [(0.0, 0.4, -0.3), (0.2, 0.2, 0.2)]
    vals = _run(_ou3(), 3, 0, [('x', 1), ('y', 1), ('z', 1)], tt)
    _vanish(vals[0], abs(complex(refs.linear_k2(OU3_A, _q(OU3_B, 3), 0, 0, 0))), 'C_xyz')


JOR_B = [([1, 0], 0.5), ([0, 1], 0.3)]
CASES_2x2 = {'jordan': [[-1.0, 1.0], [0.0, -1.0]],              # a double pole
             'split1e-7': [[-1.0, 1.0], [0.0, -1.0000001]],     # two poles 1e-7 apart
             'oscillatory': [[-0.5, 3.0], [-3.0, -0.5000001]]}


@pytest.mark.parametrize('case', sorted(CASES_2x2))
@pytest.mark.parametrize('engine', ['statespace', 'poles'])
def test_repeated_and_near_degenerate_poles_two_point(case, engine, monkeypatch):
    if engine == 'poles':
        monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    A = CASES_2x2[case]
    m = _linear_sde('oracle ' + case, A, [(b, N.Gaussian(var=v)) for b, v in JOR_B], ['x', 'y'])
    Q = _q(JOR_B, 2)
    tt = [(0.0, t) for t in TAUS]
    scale = abs(complex(refs.linear_k2(A, Q, 0, 0, 0)))
    for (i, j) in ((0, 0), (0, 1)):
        vals = _run(m, 2, 0, [('xy'[i], 1), ('xy'[j], 1)], tt)
        ref = [refs.linear_k2(A, Q, i, j, t) for t in TAUS]
        _check(vals[0], ref, scale, f'{case} C_{"xy"[i]}{"xy"[j]} ({engine})')


# ----------------------------------------------------------------------------- Levy-driven linear SDEs

LEVY = dict(mu=0.8, rate=1.7, mean=0.6, var=0.3)
TIMES = {2: [(0.0, t) for t in (-0.7, 0.0, 0.4, 1.3)],
         3: [(0.0, 0.0, 0.0), (0.0, 0.3, 1.1), (0.0, -0.5, 0.8)],
         4: [(0.0, 0.0, 0.0, 0.0), (0.0, 0.2, 0.7, 1.5), (0.0, -0.4, 0.3, 0.9)]}


def _levy_kappa():
    L = LEVY
    return {n: L['rate'] * math.factorial(n) * L['mean'] ** n for n in (2, 3, 4)}


def _levy_src():
    return N.CompoundPoisson(LEVY['rate'], N.Exponential(LEVY['mean']))


@pytest.mark.parametrize('k', [2, 3, 4])
def test_levy_ou_cumulants_match_the_closed_form(k):
    """Scalar OU driven by compound Poisson (exponential marks) + Gaussian noise."""
    m = _linear_sde('oracle levy1', [[-LEVY['mu']]], [([1], _levy_src()), ([1], N.Gaussian(var=LEVY['var']))],
                    ['x'])
    sources = [([1], _levy_kappa()), ([1], {2: LEVY['var']})]
    tt = TIMES[k]
    ell = 1 if k == 2 else 0
    vals = _run(m, k, ell, [('x', 1)] * k, tt)
    ref = [refs.linear_kpt([[-LEVY['mu']]], sources, [0] * k, t) for t in tt]
    scale = max(abs(complex(r)) for r in ref)
    _check(vals[0], ref, scale, f'Levy OU k={k}')
    if ell:
        _vanish(vals[1], scale, 'Levy OU one loop')


def test_two_field_levy_three_point_matches_the_closed_form():
    A = [[-1.0, 0.5], [-0.3, -0.7]]
    m = _linear_sde('oracle levy2', A, [([1, 0.4], _levy_src()), ([0, 1], N.Gaussian(var=LEVY['var']))],
                    ['x', 'y'])
    sources = [([1, 0.4], _levy_kappa()), ([0, 1], {2: LEVY['var']})]
    tt = TIMES[3]
    vals = _run(m, 3, 0, [('x', 1), ('y', 1), ('x', 1)], tt)
    ref = [refs.linear_kpt(A, sources, [0, 1, 0], t) for t in tt]
    _check(vals[0], ref, max(abs(complex(r)) for r in ref), 'Levy 2-field C_xyx')


def test_levy_three_point_with_poles_1e7_apart():
    """kappa_3 through two poles 1e-7 apart: the reference runs at 80 digits."""
    A = CASES_2x2['split1e-7']
    m = _linear_sde('oracle nd7 levy', A, [([0.3, 1], _levy_src())], ['x', 'y'])
    tt = TIMES[3]
    vals = _run(m, 3, 0, [('x', 1), ('y', 1), ('y', 1)], tt)
    ref = [refs.linear_kpt_jordan(A, [([0.3, 1], _levy_kappa())], [0, 1, 1], t) for t in tt]
    _check(vals[0], ref, max(abs(complex(r)) for r in ref), 'near-degenerate Levy C_xyy')


# ----------------------------------------------------------------------------- linear Hawkes

HAWKES2 = dict(mu=[0.7, 1.1], W=[[0.2, 0.5], [0.1, 0.3]], TK=[[0.5, 1.2], [2.0, 0.8]])


def _hawkes2():
    H = HAWKES2
    return (SDE('oracle hawkes2').population('E', size=2)
            .parameter('mu', default=list(H['mu']), indexed_by=['E'], domain='positive')
            .parameter('w', default=H['W'], indexed_by=['E', 'E'])
            .parameter('tk', default=H['TK'], indexed_by=['E', 'E'], domain='positive')
            .define_kernel('g', kernel=Kernel.exponential('tk[i,j]'), indexed_by=['E', 'E'])
            .point_process('n', 'mu + sum(w[i,j]*Conv(g[i,j], n[j]) for j in E)', population='E')
            .build())


def test_linear_hawkes_rates_are_the_linear_solve():
    H = HAWKES2
    res = compute_cumulants(_hawkes2(), k=1, max_ell=0, external_fields=[('n', 1)],
                            tau_grid=np.array([0.0, 1.0]), parallel=False, verbose=False)
    mf = res['mf_values']
    key = next(k_ for k_ in mf if str(k_).startswith('n'))
    got = [complex(v) for v in mf[key]]
    ref = [refs.hawkes_linear(H['mu'], H['W'], H['TK'], i, i, 0, mean=True) for i in range(2)]
    _check(got, ref, max(abs(complex(r)) for r in ref), 'Hawkes rates')


@pytest.mark.parametrize('i,j', [(0, 0), (0, 1), (1, 0)])
def test_linear_hawkes_two_point_matches_the_markov_embedding(i, j):
    H = HAWKES2
    taus = [-2.0, -0.5, 0.3, 1.5]
    tt = [(0.0, t) for t in taus]
    vals = _run(_hawkes2(), 2, 1, [('n', i + 1), ('n', j + 1)], tt)
    ref = [refs.hawkes_linear(H['mu'], H['W'], H['TK'], i, j, t) for t in taus]
    scale = max(abs(complex(r)) for r in ref)
    _check(vals[0], ref, scale, f'Hawkes C_{i+1}{j+1}')
    _vanish(vals[1], scale, f'Hawkes C_{i+1}{j+1} one loop')


# ----------------------------------------------------------------------------- 1-D polynomial drift, loop by loop

POLY = {'quartic': dict(mu=1.0, f={3: -0.2}, D=0.7, drift='-mu*x - eps*x^3', P={'mu': 1.0, 'eps': 0.2, 'D': 0.7}),
        'cubic_quartic': dict(mu=1.2, f={2: -0.5, 3: -0.3}, D=0.4, drift='-mu*x - g*x^2 - eps*x^3',
                              P={'mu': 1.2, 'g': 0.5, 'eps': 0.3, 'D': 0.4})}
_POLY_REF = {}


def _poly_model(name):
    d = POLY[name]
    s = SDE('oracle poly ' + name)
    for p, v in d['P'].items():
        s = s.parameter(p, default=v)
    return s.field('x').drift('x', d['drift']).noise('xi', N.Gaussian(var='2*D'), couples={'x': 1}).build()


def _poly_ref(name, times, ells):
    """Taylor order p = k - 2 + 2 ell is the ell-loop term; a degree-(k + p) monomial basis is exact."""
    d = POLY[name]
    k = len(times)
    pmax = k - 2 + 2 * max(ells) + 1
    key = (name, k, pmax)
    if key not in _POLY_REF:
        _POLY_REF[key] = refs.Poly1D(d['mu'], d['f'], pmax=pmax, N=k + pmax + 2, Mpts=24)
    return _POLY_REF[key].loop_term(d['D'], list(times), ells)


@pytest.mark.parametrize('name,k,ell', [('quartic', 2, 2), ('cubic_quartic', 1, 2), ('cubic_quartic', 2, 2)])
def test_polynomial_drift_loop_by_loop_at_unequal_times(name, k, ell):
    tt = {1: [(0.0,)], 2: [(0.0, t) for t in (-0.7, 0.0, 0.4, 1.3)]}[k]
    vals = _run(_poly_model(name), k, ell, [('x', 1)] * k, tt, params=POLY[name]['P'])
    refl = [_poly_ref(name, t, list(range(ell + 1))) for t in tt]
    scale = max(max(abs(complex(r[l])) for r in refl) for l in range(ell + 1))
    for l in range(ell + 1):
        _check(vals[l], [r[l] for r in refl], scale, f'{name} k={k} ell={l}')


def test_equal_time_variance_is_the_boltzmann_loop_series():
    """dx = (-x - eps x^3) dt + sqrt(2) dW: <x^2> = 1 - 3 eps + 24 eps^2 - 297 eps^3 + O(eps^4)."""
    eps = 0.02
    m = (SDE('oracle boltzmann').parameter('eps', default=eps).field('x').drift('x', '-x - eps*x^3')
         .noise('xi', N.Gaussian(var=2), couples={'x': 1}).build())
    vals = _run(m, 2, 3, [('x', 1), ('x', 1)], [(0.0, 0.0)])
    for l, c in enumerate((1, -3, 24, -297)):
        _check(vals[l], [c * eps ** l], 1.0, f'Boltzmann loop {l}')


# ----------------------------------------------------------------------------- delays and kernels

DELAY = {'negative': dict(a=1.2, c=-0.7, d=0.9, D=0.15), 'positive': dict(a=1.5, c=0.6, d=0.7, D=0.2)}


@pytest.mark.parametrize('case', sorted(DELAY))
def test_delayed_ou_two_point_is_the_exact_bvp(case):
    P = DELAY[case]
    m = (SDE('oracle delay ' + case).parameter('a', default=P['a']).parameter('c', default=P['c'])
         .parameter('d', default=P['d'], domain='positive').parameter('D', default=P['D'])
         .field('x').drift('x', '-a*x + c*x(t - d)')
         .noise('xi', N.Gaussian(var='2*D'), couples={'x': 1}).build())
    taus = [t for t in (-1.5, -0.3, 0.0, 0.3, 0.9, 1.0, 1.7) if abs(t) <= 2 * P['d']]
    vals = _run(m, 2, 0, [('x', 1), ('x', 1)], [(0.0, t) for t in taus])
    ref = [refs.delayed_ou(P['a'], P['c'], P['d'], P['D'], t) for t in taus]
    _check(vals[0], ref, abs(complex(ref[taus.index(0.0)])), f'delayed OU ({case})')


ERL = dict(a=1.3, c=-0.6, m=0.8, D=0.25)


def _erlang_chain(n):
    """Drift of (x, y_1..y_n): x' = -a x + c y_n, y_1' = r (x - y_1), y_i' = r (y_{i-1} - y_i), r = n/m."""
    E = ERL
    r = n / E['m']
    A = [[0.0] * (n + 1) for _ in range(n + 1)]
    A[0][0] = -E['a']; A[0][n] = E['c']
    A[1][0] = r; A[1][1] = -r
    for i in range(2, n + 1):
        A[i][i - 1] = r; A[i][i] = -r
    return A


def _erlang_model(n, noise):
    E = ERL
    return (SDE(f'oracle erlang{n}').parameter('a', default=E['a']).parameter('c', default=E['c'])
            .parameter('m', default=E['m'], domain='positive').parameter('D', default=E['D'])
            .define_kernel('g', kernel=Kernel.gamma(n, 'm')).field('x').drift('x', '-a*x + c*Conv(g, x)')
            .noise('xi', noise, couples={'x': 1}).build())


@pytest.mark.parametrize('n', [2, 3])
def test_erlang_kernel_two_point_is_the_chain_lyapunov(n):
    A = _erlang_chain(n)
    Q = [[0.0] * (n + 1) for _ in range(n + 1)]
    Q[0][0] = 2 * ERL['D']
    vals = _run(_erlang_model(n, N.Gaussian(var='2*D')), 2, 0, [('x', 1), ('x', 1)], [(0.0, t) for t in TAUS])
    ref = [refs.linear_k2(A, Q, 0, 0, t) for t in TAUS]
    _check(vals[0], ref, abs(complex(ref[TAUS.index(0.0)])), f'Erlang-{n} kernel')


def test_erlang_kernel_levy_three_point():
    A = _erlang_chain(3)
    tt = TIMES[3]
    vals = _run(_erlang_model(3, _levy_src()), 3, 0, [('x', 1)] * 3, tt)
    ref = [refs.linear_kpt(A, [([1, 0, 0, 0], _levy_kappa())], [0, 0, 0], t) for t in tt]
    _check(vals[0], ref, max(abs(complex(r)) for r in ref), 'Erlang-3 kernel, Levy k=3')


# ----------------------------------------------------------------------------- switch-on transients

def test_switch_on_two_point_is_van_loan():
    tmin = 0.5
    Q = _q(OU3_B, 3)
    times = [(0.7, 0.7), (1.0, 2.2), (2.0, 1.3), (3.5, 3.9)]
    scale = abs(complex(refs.linear_k2(OU3_A, Q, 0, 0, 0)))
    for (i, j) in ((0, 0), (0, 1), (2, 1)):
        r = compute_cumulants(_ou3(), k=2, max_ell=0, external_fields=[('xyz'[i], 1), ('xyz'[j], 1)],
                              t_min=tmin, times=np.array(times), parallel=False, verbose=False)
        vals = [complex(v) for v in np.asarray(r['C_times_by_ell'][0])]
        ref = [refs.switchon_k2(OU3_A, Q, i, j, a, b, tmin) for a, b in times]
        _check(vals, ref, scale, f'switch-on C_{"xyz"[i]}{"xyz"[j]}')


def test_switch_on_levy_three_point_closed_form():
    tmin = 0.5
    m = _linear_sde('oracle switch-on levy', [[-LEVY['mu']]], [([1], _levy_src())], ['x'])
    times = [(0.9, 0.9, 0.9), (0.8, 1.4, 2.6), (1.5, 0.7, 1.1)]
    r = compute_cumulants(m, k=3, max_ell=0, external_fields=[('x', 1)] * 3, t_min=tmin,
                          times=np.array(times), parallel=False, verbose=False)
    vals = [complex(v) for v in np.asarray(r['C_times_by_ell'][0])]
    ref = [refs.linear_kpt([[-LEVY['mu']]], [([1], _levy_kappa())], [0, 0, 0], t, t_min=tmin) for t in times]
    _check(vals, ref, max(abs(complex(x)) for x in ref), 'switch-on Levy k=3')


# ----------------------------------------------------------------------------- colored noise (tracked model)

def test_colored_noise_tree_and_one_loop_mass_shift():
    """Tracked ou_quartic_colored: dx = (-mu x - eps x^3) dt + eta, eta an OU noise with
    <eta(0) eta(t)> = (D/tauc) e^{-|t|/tauc}.  Tree = the 2-field Lyapunov solution; one loop =
    the exact O(eps) mass shift mu -> mu + 3 eps Var0, i.e. 3 eps Var0 dC0/dmu (Hartree)."""
    import daedalus as dd
    P = dict(mu=1.0, eps=0.1, D=0.5, tauc=0.7)
    m, _ = dd.load_model('ou_quartic_colored')
    tt = [(0.0, t) for t in TAUS]
    vals = _run(m, 2, 1, [('x', 1), ('x', 1)], tt, params=dict(P))
    mu, D, tc, e = (mp.mpf(str(P[x])) for x in ('mu', 'D', 'tauc', 'eps'))

    def C0(mm, t):
        return refs.linear_k2([[-mm, 1.0], [0.0, -1 / tc]], [[0.0, 0.0], [0.0, 4 * D / tc ** 2]], 0, 0, t)
    var0 = C0(mu, 0)
    _check(vals[0], [C0(mu, t) for t in TAUS], abs(complex(var0)), 'colored tree')
    with mp.workdps(40):
        loop = [3 * e * var0 * mp.diff(lambda mm: C0(mm, t), mu) for t in TAUS]
    _check(vals[1], loop, abs(complex(var0)), 'colored one loop')


# ----------------------------------------------------------------------------- spatial Allen-Cahn d = 1 (tracked model)

AC = dict(mu=1.5, D=0.5, T=0.8, lam=0.15)
AC_X = [0.0, 1.0, 2.5]


@pytest.fixture(scope='module')
def allen_cahn_one_loop():
    import daedalus as dd
    m, _ = dd.load_model('allen_cahn_1d_subcritical_infinite')
    r = compute_cumulants(m, k=2, max_ell=1, parameters=dict(AC), external_fields=[('phi', 1), ('phi', 2)],
                          tau_max=1.0, tau_step=1.0, spatial_grid=np.array(AC_X), parallel=False, verbose=False)
    tg = list(np.asarray(r['tau_grid'], float))
    cum = r['C_tau_x_by_order']
    by = {l: np.asarray(cum[l]) - (np.asarray(cum[l - 1]) if l > 0 else 0) for l in sorted(cum)}
    return tg, by


def _ac_ref(tau):
    return [refs.allen_cahn_1d(AC['mu'], AC['D'], AC['T'], AC['lam'], x, tau) for x in AC_X]


@pytest.mark.parametrize('tau', [0.0, 1.0])
def test_allen_cahn_d1_tree_is_the_erfc_closed_form(allen_cahn_one_loop, tau):
    """(d_t + mu - D d_x^2) phi = -lam phi^3 + eta on the line: tree = the erfc closed form
    (quadrature-checked inside the reference)."""
    tg, by = allen_cahn_one_loop
    ref = _ac_ref(tau)
    _check(by[0][tg.index(tau)], [a for a, _ in ref], abs(complex(_ac_ref(0.0)[0][0])), f'tree tau={tau}')


def test_allen_cahn_d1_one_loop_is_the_tadpole_mass_shift(allen_cahn_one_loop):
    """One loop = the tadpole mass shift mu -> mu + 3 lam Var0 (exact at this order), tau = 1."""
    tg, by = allen_cahn_one_loop
    ref = _ac_ref(1.0)
    _check(by[1][tg.index(1.0)], [b for _, b in ref], max(abs(complex(b)) for _, b in ref), 'one loop tau=1',
           rtol=1e-10)


def test_allen_cahn_d1_one_loop_at_equal_times(allen_cahn_one_loop):
    """The same mass shift at tau = 0 (2e-6..7e-6 off before the exact equal-time loop rule; 3.7e-13 now,
    so 1e-11 leaves a platform margin)."""
    tg, by = allen_cahn_one_loop
    ref = _ac_ref(0.0)
    _check(by[1][tg.index(0.0)], [b for _, b in ref], max(abs(complex(b)) for _, b in ref), 'one loop tau=0',
           rtol=1e-11)

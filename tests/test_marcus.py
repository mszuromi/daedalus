"""The Marcus rule for jump and non-Gaussian sources (``SDE(...).interpretation('marcus')``, api/sde.py).

A jump ``z`` of a source coupled through ``G = sum_i b_i d/dx_i`` moves the state along the flow of ``G``,
``x -> exp(z G) x``; the Ito-ordered action carries ``sum_k kappa_k/k! D_k``, ``D_k = e^{-xt.x} G^k e^{xt.x}``.

1  Gaussian-only models: 'marcus' gives exactly the 'stratonovich' action, equations and notes.
2  Constant (additive) couplings: 'marcus' gives exactly the Ito action, equations and notes.
3  Scalar affine compound Poisson ``(a + b x) <> dL``: the exact Marcus map, and the tree and one-loop covariance
   against the exact moment hierarchy, graded by loops (<= 1e-11).
4  Conductance shot noise ``-w (V - Es) <> dN``: exact moments as in 3; the Marcus map never crosses ``Es`` while
   the Ito jump does for ``w > 1`` (simulator); a Monte Carlo check of the Marcus moments.
5  Rotation ``g = (y, -x)``: the Marcus jump part of the generator annihilates ``r^2`` (Ito does not); the
   simulator's Marcus jumps keep ``r^2`` to round-off.
6  Refusals.
7  A superlinear coupling: the truncation advisory fires at the saddle (warning, result, G6).
8  Affine couplings take the exact route (closed-form flow, mgf coefficients): it reproduces the series route of
   3/4 to 1e-13; the geometric exponential-jump model (series divergent at high degree) converges to ell = 3 with
   no advisory; two fields against quadrature; mgf-pole refusals naming the degree; the rounding advisory.
9  The advisory counts only the vertices a (k, max_ell) run can use.

Loop grading (3, 4).  The action ``S = xt.(Dt x) - sum_j eta_j(x) xt^j`` is the generator
``L = sum_j eta_j(x) d^j`` (normal ordered).  Scaling ``eta_j -> hbar^(j-1) eta_j`` makes ``S/hbar`` the loop
expansion, so the ``ell``-loop part of an ``E``-point cumulant is its ``hbar^(E-1+ell)`` coefficient.  For an
affine coupling and a linear drift ``eta_j`` has degree ``<= j``, so the hierarchy
``E[L_hbar x^n] = 0`` closes and is solved order by order in ``hbar`` below, from the exact (untruncated) Marcus
coefficients ``eta_j = lam E[(phi_J(x) - x)^j]/j!``; the action's truncation at ``SDE.series_order`` is below
the advisory's 1e-10 per vertex degree for the parameters used (checked).
"""
import math
import re
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR                                                     # noqa: E402

import api.compute as AC                                                    # noqa: E402
from api.analyze import Policy, analyze                                     # noqa: E402
from api.model import SpatialModelBuilder                                   # noqa: E402
from api.noise import (CGF, CompoundPoisson, Cumulants, Dirac, Exponential,  # noqa: E402
                       GammaProcess, Gaussian, LogNormal, NoiseError, Poisson, Uniform,
                       parse_expr)
from api.pp_simulate import _marcus_jump                                    # noqa: E402
from api.sde import SDE, SDEError                                           # noqa: E402

TAU = np.array([0.0, 0.7])


def _same_build(a, b):
    """The built model dicts agree except for the interpretation stamp (callables compared by presence)."""
    ma, mb = a.build(), b.build()
    ma.pop('interpretation', None)
    mb.pop('interpretation', None)
    assert set(ma) == set(mb)
    for key in ma:
        if callable(ma[key]) or callable(mb[key]):
            assert callable(ma[key]) and callable(mb[key]), key
        else:
            assert repr(ma[key]) == repr(mb[key]), key


def _identical(a, b):
    assert a.action_text() == b.action_text()
    assert a.equations() == b.equations()
    assert a.notes == b.notes


# ── 1: Gaussian-only models: Marcus = Stratonovich ──────────────────────────────────────────

def _strat_models(interp):
    """The Stratonovich test models of tests/test_sde_frontend.py, test_leftovers_39.py and
    test_sde_nonpolynomial_drift.py, plus a mean + state-dependent variance case."""
    out = {}
    out['ou_quartic'] = (SDE('OU quartic').parameter('mu', default=1.0, domain='positive')
                         .parameter('eps', default=0.02, domain='positive').parameter('D', default=1.0)
                         .field('x').drift('x', '-mu*x - eps*x^3')
                         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    out['linear_mult'] = (SDE('Stratonovich multiplicative').parameter('mu', default=1.0, domain='positive')
                          .parameter('s0', default=0.6, domain='positive')
                          .parameter('s1', default=0.2, domain='positive').field('x')
                          .drift('x', '-mu*x').noise('W', Gaussian(var=1), couples={'x': 's0 + s1*x'})
                          .stability_analysis(True))
    out['two_field'] = (SDE('strat 2d').parameter('mu').parameter('tau').field('x').field('y')
                        .drift('x', '-mu*x').drift('y', '-mu*y')
                        .noise('W', Gaussian(var=1), couples={'x': 'y', 'y': 'x'}))
    out['tau_lhs'] = (SDE('strat tau').parameter('tau').parameter('c0').parameter('c1').field('x')
                      .equation('x', lhs='(tau*Dt + 1)*x', rhs='0')
                      .noise('W', Gaussian(var=1), couples={'x': 'c0 + c1*x'}))
    for lhs in ('Dt*x', '2*Dt*x', '(Dt + 1)*x'):
        out[f'lhs {lhs}'] = (SDE('s').parameter('D', default=0.1, domain='positive').field('x')
                             .noise('xi', Gaussian(var='2*D'), couples={'x': '(1 + 0.5*x^2)'})
                             .equation('x', lhs=lhs, rhs='-x'))
    out['two_lhs'] = (SDE('s').parameter('D', default=0.1, domain='positive').field('x').field('y')
                      .noise('xi', Gaussian(var='2*D'), couples={'x': '(1 + 0.5*x^2)'})
                      .equation('x', lhs='Dt*x + Dt*y', rhs='-x')
                      .noise('eta', Gaussian(var='2*D'), couples={'y': 1})
                      .equation('y', lhs='(Dt + 1)*y', rhs='0'))
    out['nonpolynomial'] = (SDE('strat').parameter('eps', default=0.05, domain='positive').field('x')
                            .drift('x', '-x/(1+eps*x^2)')
                            .noise('xix', Gaussian(var='2'), couples={'x': '1 + x/5'}).stability_analysis(True))
    out['mean_and_var'] = (SDE('mv').parameter('mu', default=1.0).parameter('c', default=0.3)
                           .parameter('s', default=0.4).field('x').drift('x', '-mu*x')
                           .noise('W', Gaussian(mean='c*x', var='s^2*(1 + x^2)'), couples={'x': 'x'}))
    for s in out.values():
        s.interpretation(interp)
    return out


@pytest.mark.parametrize('name', list(_strat_models('ito')))
def test_gaussian_only_marcus_is_stratonovich(name):
    m, s = _strat_models('marcus')[name], _strat_models('stratonovich')[name]
    _identical(m, s)
    _same_build(m, s)
    assert not m._marcus_terms


@pytest.mark.parametrize('lhs', ['(1+x^2)*Dt*x', 'x*Dt*x'])
def test_gaussian_only_marcus_refuses_like_stratonovich(lhs):
    for interp in ('stratonovich', 'marcus'):
        s = (SDE('s').parameter('D', default=0.1).field('x').interpretation(interp)
             .noise('xi', Gaussian(var='2*D'), couples={'x': '(1 + 0.5*x^2)'}).equation('x', lhs=lhs, rhs='-x'))
        with pytest.raises(SDEError, match=f'{interp}: cannot read a constant Dt coefficient'):
            s.equations()


# ── 2: constant couplings: Marcus = Ito ─────────────────────────────────────────────────────

def _additive_models(interp):
    def base(name):
        return (SDE(name).parameter('mu', default=1.0, domain='positive').parameter('lam', default=1.5)
                .parameter('w', default=0.3).parameter('m', default=0.5).field('x').drift('x', '-mu*x'))
    out = {
        'poisson': base('p').noise('N', Poisson('lam'), couples={'x': 'w'}),
        'cp_state_rate': base('cp').noise('L', CompoundPoisson('lam*(1 + x^2)', Exponential('m')),
                                          couples={'x': 1}),
        'gamma': base('g').noise('G', GammaProcess('lam', 'm'), couples={'x': '-w'}),
        'uniform': base('u').noise('U', CompoundPoisson('lam', Uniform(0, 'm')), couples={'x': 1}).series_order(5),
        'jump_diffusion': base('jd').noise('J', Gaussian(var='w') + CompoundPoisson('lam', Exponential('m')),
                                           couples={'x': 'w'}),
        'two_fields': (base('2f').field('y').drift('y', '-mu*y')
                       .noise('N', Poisson('lam'), couples={'x': 1, 'y': '-w'})),
        'mixed': (base('mix').noise('N', Poisson('lam'), couples={'x': 'w'})
                  .noise('W', Gaussian(var='m'), couples={'x': 1})),
        'point_process': (SDE('pp').parameter('tau', default=1.0, domain='positive').parameter('w', default=0.2)
                          .parameter('a', default=1.0, domain='positive').field('v')
                          .point_process('n', 'a')
                          .equation('v', lhs='(tau*Dt + 1)*v', rhs='w*n')),
    }
    for s in out.values():
        s.interpretation(interp)
    return out


@pytest.mark.parametrize('name', list(_additive_models('ito')))
def test_additive_jumps_marcus_is_ito(name):
    m, i = _additive_models('marcus')[name], _additive_models('ito')[name]
    _identical(m, i)
    _same_build(m, i)
    assert not m._marcus_terms


# ── the exact loop-graded moment hierarchy (scalar affine coupling, linear drift) ──────────

def _poly_mul(p, q):
    return np.convolve(p, q)


def _eta(j, lam, Eq, d0, d1, alpha):
    """Coefficients (in x) of eta_j = lam E[q^j] (d0 + d1 x)^j / j!  (+ the drift alpha for j = 1)."""
    base = np.array([1.0])
    for _ in range(j):
        base = _poly_mul(base, np.array([d0, d1]))
    out = lam * Eq(j) / math.factorial(j) * base
    if j == 1:
        out = out + np.array(alpha)
    return out


def _graded_moments(lam, Eq, d0, d1, alpha, nmax, rmax):
    """m[n][r]: the hbar^r coefficient of E[x^n] for L_hbar = sum_j hbar^(j-1) eta_j(x) d^j (see the module
    docstring).  The Marcus jump is written ``phi_J(x) - x = q_J (d0 + d1 x)`` and ``Eq(j) = E[q_J^j]``;
    ``alpha`` is the linear drift ``alpha[0] + alpha[1] x``."""
    eta = {j: _eta(j, lam, Eq, d0, d1, alpha) for j in range(1, nmax + 1)}
    m = np.zeros((nmax + 1, rmax + 1))
    m[0, 0] = 1.0
    beta = eta[1][1]
    for r in range(rmax + 1):
        for n in range(1, nmax + 1):
            acc = 0.0
            for j in range(1, n + 1):
                rr = r - (j - 1)
                if rr < 0:
                    continue
                poly = _poly_mul(eta[j], np.eye(n - j + 1)[n - j])         # eta_j x^(n-j)
                fac = math.factorial(n) / math.factorial(n - j)
                for p, c in enumerate(poly):
                    if j == 1 and p == n:
                        continue                                          # the unknown
                    acc += fac * c * m[p, rr]
            m[n, r] = -acc / (n * beta)
    return m


def _graded_cumulant2(m, rmax):
    """hbar-series of Var = m2 - m1^2."""
    m1sq = np.convolve(m[1], m[1])[:rmax + 1]
    return m[2] - m1sq


def _pipeline_cov(model, par, max_ell=1):
    res = AC.compute_cumulants(model, k=2, max_ell=max_ell, external_fields=[('dx', 1)] * 2, parameters=par,
                               tau_grid=TAU, use_cache=False, parallel=False, verbose=False)
    return {int(l): np.real(np.asarray(a, dtype=complex)) for l, a in res['C_tau_by_ell'].items()}, res


# ── 3: scalar affine compound Poisson ───────────────────────────────────────────────────────

def _affine(interp, order=16):
    return (SDE('affine compound Poisson').parameter('mu', default=1.0, domain='positive')
            .parameter('lam', default=1.5, domain='positive').parameter('a', default=0.4)
            .parameter('b', default=0.15).parameter('mJ', default=0.2, domain='positive')
            .field('x').drift('x', '-mu*x')
            .noise('L', CompoundPoisson('lam', Exponential('mJ')), couples={'x': 'a + b*x'})
            .interpretation(interp).series_order(order).stability_analysis(True))


def _exp_Eq(b, mJ):
    """E[(e^(b J) - 1)^j] for J ~ Exp(mean mJ): binomial sum of the mgf 1/(1 - mJ s)."""
    return lambda j: sum(math.comb(j, i) * (-1) ** (j - i) / (1 - mJ * i * b) for i in range(j + 1))


def test_affine_marcus_map_is_exact():
    """The simulator's Marcus jump: closed-form flow and the ODE fallback against x e^{bz} + (a/b)(e^{bz} - 1)."""
    rng = np.random.default_rng(3)
    a, b = 0.4, -0.7
    x0 = rng.normal(size=64)
    z = rng.exponential(1.3, size=64)
    z[::5] = 0.0
    exact = x0 * np.exp(b * z) + a / b * np.expm1(b * z)
    code = compile('a + b*x[i]', '<c>', 'eval')
    acode = [[compile('b', '<A>', 'eval')]]
    for A in (acode, None):
        state = {'x': x0.copy()[None, :]}

        def ev(c, i):
            return eval(c, {'a': a, 'b': b, 'x': state['x'], 'i': i})
        _marcus_jump({'fields': ['x'], 'A': A}, {'x': code}, {'x': [(1.0, 0.0)]}, 0, z, state, ev, len(z))
        assert np.max(np.abs(state['x'][0] - exact)) < (1e-14 if A is not None else 1e-10)


@pytest.mark.parametrize('b', [0.15, -0.15])
def test_affine_compound_poisson_against_moment_hierarchy(b):
    mu, lam, a, mJ = 1.0, 1.5, 0.4, 0.2
    par = {'mu': mu, 'lam': lam, 'a': a, 'b': b, 'mJ': mJ}
    Eq = _exp_Eq(b, mJ)
    # phi_J(x) - x = (e^{bJ} - 1)(x + a/b) = q (a/b + x): d0 = a/b, d1 = 1
    m = _graded_moments(lam, Eq, a / b, 1.0, [0.0, -mu], nmax=2, rmax=2)
    var = _graded_cumulant2(m, 2)
    beta = -mu + lam * Eq(1)
    got, res = _pipeline_cov(_affine('marcus').build(), par)
    assert res['mf_values']['xstar'][0] == pytest.approx(m[1, 0], rel=1e-12)
    assert abs(m[1, 1]) < 1e-15 and abs(m[1, 2]) < 1e-15                    # the mean has no loop corrections
    decay = np.exp(beta * TAU)
    for ell in (0, 1):
        ref = var[1 + ell] * decay
        assert np.max(np.abs(got[ell] - ref)) <= 1e-11 * abs(ref[0]), (ell, got[ell], ref)
    assert all(r['status'] == 'ok' for r in res['marcus_truncation'])       # below 1e-10 per degree
    assert res['analysis'].has('G6_MARCUS_TRUNCATION') and not res['analysis'].has('G6_MARCUS_TRUNCATION_LOW')
    # Ito differs at the tree level (the jump is (a + b x_pre) J, not the flow)
    got_ito, _ = _pipeline_cov(_affine('ito').build(), par, max_ell=0)
    assert abs(got_ito[0][0] - var[1]) > 1e-3 * abs(var[1])


# ── 4: conductance shot noise ───────────────────────────────────────────────────────────────

def _conductance(interp, w=0.3, order=20):
    return (SDE('conductance shot noise').parameter('tau', default=1.0, domain='positive')
            .parameter('EL', default=0.0).parameter('Es', default=1.0)
            .parameter('lam', default=2.0, domain='positive').parameter('w', default=w, domain='positive')
            .field('x').drift('x', '-(x - EL)/tau')
            .noise('N', Poisson('lam'), couples={'x': '-w*(x - Es)'})
            .interpretation(interp).series_order(order).stability_analysis(True))


def _conductance_moments(tau, EL, Es, lam, w, nmax, rmax):
    q = math.expm1(-w)                      # V -> Es + (V - Es) e^{-w}:  phi(x) - x = q (x - Es)
    return _graded_moments(lam, lambda j: q ** j, -Es, 1.0, [EL / tau, -1.0 / tau], nmax, rmax)


def test_conductance_shot_noise_against_exact_moments():
    tau, EL, Es, lam, w = 1.0, 0.0, 1.0, 2.0, 0.3
    par = {'tau': tau, 'EL': EL, 'Es': Es, 'lam': lam, 'w': w}
    m = _conductance_moments(tau, EL, Es, lam, w, 2, 2)
    var = _graded_cumulant2(m, 2)
    got, res = _pipeline_cov(_conductance('marcus').build(), par)
    assert res['mf_values']['xstar'][0] == pytest.approx(m[1, 0], rel=1e-12)
    decay = np.exp((-1 / tau + lam * math.expm1(-w)) * TAU)
    for ell in (0, 1):
        ref = var[1 + ell] * decay
        assert np.max(np.abs(got[ell] - ref)) <= 1e-11 * abs(ref[0]), (ell, got[ell], ref)
    assert all(r['status'] == 'ok' for r in res['marcus_truncation'])


def _sim(s, seed=7, **kw):
    kw = {'T': 40.0, 'dt': 0.01, 'n_paths': 64, 'burn_in': 2.0, **kw}
    return s.simulate(seed=seed, **kw)['fields']['x'][:, 0, :]


def test_conductance_marcus_never_crosses_reversal_potential():
    """w = 1.5 > 1: the Ito jump (V - Es) -> (1 - w)(V - Es) overshoots Es; the Marcus flow
    (V - Es) -> e^{-w} (V - Es) never reaches it (paths start at EL < Es)."""
    marcus = _sim(_conductance('marcus', w=1.5))
    ito = _sim(_conductance('ito', w=1.5))
    assert marcus.max() < 1.0 and np.isfinite(marcus).all()
    assert ito.max() > 1.0


def test_conductance_monte_carlo_matches_marcus_moments():
    tau, EL, Es, lam, w = 1.0, 0.0, 1.0, 2.0, 0.5
    m = _conductance_moments(tau, EL, Es, lam, w, 2, 40)
    full = np.linalg.solve(*_hbar1_system(tau, EL, Es, lam, w))
    # the loop series summed at hbar = 1 is the exact moment (a cross-check of the graded hierarchy)
    assert m[1].sum() == pytest.approx(full[0], rel=1e-12) and m[2].sum() == pytest.approx(full[1], rel=1e-10)
    x = _sim(_conductance('marcus', w=w), T=60.0, dt=0.002, n_paths=96, seed=11)
    per_path_mean, per_path_sq = x.mean(0), (x ** 2).mean(0)
    se1 = per_path_mean.std(ddof=1) / math.sqrt(x.shape[1])
    se2 = per_path_sq.std(ddof=1) / math.sqrt(x.shape[1])
    # the Euler drift is first order in dt: allow 0.5 % on top of 5 standard errors
    assert abs(per_path_mean.mean() - full[0]) < 5 * se1 + 5e-3 * abs(full[0])
    assert abs(per_path_sq.mean() - full[1]) < 5 * se2 + 5e-3 * abs(full[1])
    # and the Ito moments are far outside that band
    q_ito = -w
    mean_ito = (EL / tau - lam * q_ito * Es) / (1 / tau - lam * q_ito)
    assert abs(mean_ito - full[0]) > 10 * se1


def _hbar1_system(tau, EL, Es, lam, w):
    """E[L x] = E[L x^2] = 0 at hbar = 1 for the exact Marcus generator (an independent 2 x 2 solve)."""
    q = math.expm1(-w)
    # L x = (EL - x)/tau + lam q (x - Es)
    # L x^2 = 2x (EL - x)/tau + lam [ (x + q(x - Es))^2 - x^2 ]
    #       = 2x (EL - x)/tau + lam [ 2 q x (x - Es) + q^2 (x - Es)^2 ]
    A = np.array([[-1 / tau + lam * q, 0.0],
                  [2 * EL / tau - 2 * lam * q * Es - 2 * lam * q * q * Es, -2 / tau + 2 * lam * q + lam * q * q]])
    rhs = -np.array([EL / tau - lam * q * Es, lam * q * q * Es * Es])
    return A, rhs


# ── 5: the non-commuting rotation g = (y, -x) ───────────────────────────────────────────────

def _rotation(interp, mu='mu'):
    return (SDE('rotation').parameter('mu', default=1.0).parameter('lam', default=1.0)
            .parameter('z', default=0.7).field('x').field('y')
            .drift('x', f'-{mu}*x').drift('y', f'-{mu}*y')
            .noise('L', CompoundPoisson('lam', Dirac('z')), couples={'x': 'y', 'y': '-x'})
            .interpretation(interp).series_order(7))


def _generator_on_r2(s):
    """L r^2 for the generator of the action (Dt -> 0): H = -S0 = sum_alpha eta_alpha xt^a yt^b and
    L r^2 = 2 x eta_(1,0) + 2 y eta_(0,1) + 2 eta_(2,0) + 2 eta_(0,2)."""
    text = s.action_text()
    inner = re.sub(r'\s*for i in pop\)\s*$', '', re.sub(r'^\s*sum\(', '', text.strip()))
    x, y, xt, yt = (SR.var(n) for n in ('x__i', 'y__i', 'xt__i', 'yt__i'))
    s_ = SR.var('s_')
    S0 = parse_expr(inner).subs({SR.var('Dt'): 0}).subs({xt: s_ * xt, yt: s_ * yt})
    H = -S0.taylor(s_, 0, 2).subs({s_: 1}).expand()      # the Ito action keeps K in closed form
    eta = lambda p, q: H.coefficient(xt, p).coefficient(yt, q).subs({xt: 0, yt: 0})   # noqa: E731
    return (2 * x * eta(1, 0) + 2 * y * eta(0, 1) + 2 * eta(2, 0) + 2 * eta(0, 2)).expand()


def test_rotation_marcus_preserves_r2_in_the_generator():
    x, y = SR.var('x__i'), SR.var('y__i')
    drift_part = -2 * SR.var('mu') * (x ** 2 + y ** 2)
    marcus = (_generator_on_r2(_rotation('marcus')) - drift_part).expand()
    ito = (_generator_on_r2(_rotation('ito')) - drift_part).expand()
    assert marcus.is_zero(), marcus
    assert (ito - SR.var('lam') * SR.var('z') ** 2 * (x ** 2 + y ** 2)).expand().is_zero(), ito


def test_rotation_marcus_jump_map_keeps_r2():
    """The simulator's Marcus jump for g = (y, -x) (closed-form flow: a rotation by z) keeps r^2 to round-off
    over 50 jumps; the Ito jump (x, y) -> (x + z y, y - z x) multiplies it by 1 + z^2."""
    rng = np.random.default_rng(2)
    th = rng.uniform(0, 2 * np.pi, 200)
    state = {'x': np.cos(th)[None, :].copy(), 'y': np.sin(th)[None, :].copy()}
    cc = {'x': compile('y[i]', '<cx>', 'eval'), 'y': compile('-x[i]', '<cy>', 'eval')}
    A = [[compile('0', '<A>', 'eval'), compile('1', '<A>', 'eval')],
         [compile('-1', '<A>', 'eval'), compile('0', '<A>', 'eval')]]

    def ev(c, i):
        return eval(c, {'x': state['x'], 'y': state['y'], 'i': i})
    for flow in ({'fields': ['x', 'y'], 'A': A}, {'fields': ['x', 'y'], 'A': None}):
        for _ in range(50 if flow['A'] is not None else 5):
            _marcus_jump(flow, cc, {'x': [(1.0, 0.0)], 'y': [(1.0, 0.0)]}, 0,
                         rng.exponential(0.8, 200), state, ev, 200)
        assert np.max(np.abs(state['x'][0] ** 2 + state['y'][0] ** 2 - 1.0)) < 1e-10
    xi, yi, z = np.cos(th), np.sin(th), 0.7
    assert np.allclose((xi + z * yi) ** 2 + (yi - z * xi) ** 2, 1 + z * z)


def test_rotation_simulator_from_an_offset_start():
    """End to end through SDE.simulate: a constant drift moves the paths off the origin (a rotation drives
    nothing from 0); with mu = 0 each step changes r^2 only by the drift kick 2 x dt + dt^2, the Marcus jumps
    rotating it."""
    s = (SDE('rotation offset').parameter('lam', default=1.0).parameter('z', default=0.5)
         .parameter('mu', default=0.0).field('x').field('y')
         .drift('x', '-mu*x + 1').drift('y', '-mu*y')
         .noise('L', CompoundPoisson('lam', Dirac('z')), couples={'x': 'y', 'y': '-x'})
         .interpretation('marcus').series_order(7))
    rec = s.simulate(T=4.0, dt=0.01, n_paths=8, burn_in=0.0, seed=1)
    x, y = rec['fields']['x'][:, 0, :], rec['fields']['y'][:, 0, :]
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    # between steps r^2 changes only by the drift kick 2 x dt + dt^2 (the jumps rotate)
    r2 = x ** 2 + y ** 2
    kick = 2 * x[:-1] * 0.01 + 0.01 ** 2
    assert np.max(np.abs(np.diff(r2, axis=0) - kick)) < 1e-12


# ── 6: refusals ─────────────────────────────────────────────────────────────────────────────

def _jbase():
    return (SDE('ref').parameter('mu', default=1.0).parameter('lam', default=1.0).parameter('b', default=0.1)
            .field('x').drift('x', '-mu*x'))


def test_refuses_moment_series_jump_laws_beyond_their_order():
    s = (_jbase().noise('L', CompoundPoisson('lam', LogNormal(0, 1, order=4)), couples={'x': '1 + b*x'})
         .interpretation('marcus'))
    with pytest.raises(NoiseError, match='truncated at order 4.*Marcus series'):
        s.action_text()
    s.series_order(4)                       # within the declared moments: accepted (no next term to size)
    assert 'xt[i]^4' in s.action_text()
    assert s.build()['marcus_series']['L']['next'] is None


def test_refuses_exposed_sources():
    s = _jbase().noise('p', Poisson('lam'), expose='n').interpretation('marcus')
    with pytest.raises(SDEError, match="marcus.*exposed source 'p'"):
        s.action_text()


@pytest.mark.parametrize('coupling', ['b*sum(x[j] for j in E)', 'b*x[0]', 'b*x[ j ]'])
def test_refuses_joint_jumps_across_units(coupling):
    s = (SDE('pop').population('E', size=2).parameter('lam', default=1.0).parameter('b', default=0.1)
         .field('x', population='E').drift('x', '-x')
         .noise('L', Poisson('lam'), couples={'x': coupling}, population='E')
         .interpretation('marcus'))
    with pytest.raises(SDEError, match='joint'):
        s.action_text()


def test_refuses_state_dependent_point_process_coupling():
    def pp(rhs, size='1'):
        return (SDE('pp').parameter('w', default=0.2).parameter('Es', default=1.0).parameter('a', default=1.0)
                .field('v').point_process('n', 'a', size=size).drift('v', rhs).interpretation('marcus'))
    with pytest.raises(SDEError, match='state-dependent coefficient'):
        pp('-v + w*(Es - v)*n').action_text()
    with pytest.raises(SDEError, match='state-dependent size'):
        (SDE('pp').parameter('w', default=0.2).parameter('a', default=1.0).field('v').field('u')
         .drift('u', '-u').point_process('n', 'a', size='1 + u^2').drift('v', '-v + w*n')
         .interpretation('marcus').action_text())
    pp('-v + w*n').action_text()            # a constant coupling: accepted (it is the Ito model, test 2)


def test_stratonovich_with_jumps_points_to_marcus():
    with pytest.raises(SDEError, match=r"interpretation\('marcus'\)"):
        _jbase().noise('p', Poisson('lam'), couples={'x': 1}).interpretation('stratonovich').action_text()


def test_refuses_state_dependent_cumulant_sources():
    s = _jbase().noise('C', Cumulants([0, 'lam*(1 + x^2)', 'b']), couples={'x': '1 + b*x'}).interpretation('marcus')
    with pytest.raises(SDEError, match='Gaussian part cannot be told'):
        s.action_text()
    s = _jbase().noise('C', CGF('lam*(exp(x*theta) - 1)'), couples={'x': 1}).interpretation('marcus')
    with pytest.raises(SDEError, match='Gaussian part cannot be told'):
        s.action_text()


def test_marcus_policy_and_config():
    sp = (SpatialModelBuilder('spatial').physical_field('phi', spatial_dim=1)
          .parameter('mu', default=1.0).parameter('D', default=1.0).parameter('T', default=1.0, domain='positive')
          .set_action_text('phit*(Dt*phi + mu*phi - D*Lap(phi)) - T*phit^2')
          .equation(lhs='(Dt + mu)*phi', rhs='0').build())
    rep = analyze(sp, None, Policy(interpretation='marcus'), use_cache=False)
    assert rep.has('G0_POLICY_INVALID')
    with pytest.raises(SDEError, match="'ito', 'stratonovich' or 'marcus'"):
        SDE('x').interpretation('levy')


def test_config_accepts_marcus_and_series_order(tmp_path):
    from api.config_run import load_config
    text = """\
sde:
  name: affine inline
  parameters:
    mu: {default: 1.0, domain: positive}
    lam: 1.5
    b: 0.1
  fields: [x]
  drift: {x: "-mu*x"}
  noise:
    L: {type: Poisson, rate: lam, couples: {x: "1 + b*x"}}
  interpretation: marcus
  series_order: 6
question:
  k: 2
  max_ell: 1
  tau_grid: [0.0]
"""
    p = tmp_path / 'm.yaml'
    p.write_text(text)
    resolved, model, _, _ = load_config(str(p))
    assert model['interpretation'] == 'marcus' and model['marcus_series']['L']['order'] == 6
    assert resolved['sde']['interpretation'] == 'marcus' and resolved['sde']['series_order'] == 6


# ── 7: a superlinear coupling: the truncation advisory ──────────────────────────────────────

def _superlinear(order):
    return (SDE('superlinear').parameter('mu', default=2.0, domain='positive').parameter('lam', default=1.0)
            .parameter('eps', default=0.2, domain='positive').field('x').drift('x', '-mu*x + 1')
            .noise('N', Poisson('lam'), couples={'x': 'eps*x^2'})
            .interpretation('marcus').series_order(order).stability_analysis(True))


def test_superlinear_coupling_truncation_advisory():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = AC.compute_cumulants(_superlinear(8).build(), k=2, max_ell=0, external_fields=[('dx', 1)] * 2,
                                   tau_grid=np.array([0.0]), use_cache=False, parallel=False, verbose=False)
    rows = res['marcus_truncation']
    assert rows[0]['status'] == 'above' and rows[0]['relative'] > 1e-10
    assert any(issubclass(x.category, AC.MarcusTruncationWarning) for x in w)
    assert res['analysis'].has('G6_MARCUS_TRUNCATION_LOW')
    # the flow of eps x^2 is x/(1 - eps z x): its Marcus drift lam eps x^2/(1 - eps x) is the saddle's
    xs = res['mf_values']['xstar'][0]
    # residual of the exact (untruncated) saddle equation is of the size of the omitted terms
    resid = -2 * xs + 1 + 0.2 * xs ** 2 / (1 - 0.2 * xs)
    assert abs(resid) < 1e-6 and abs(resid) > 1e-12
    # a higher order shrinks the omitted term (x^2 coupling: D_k has degree 2k in x, so stay low)
    res11 = AC.compute_cumulants(_superlinear(11).build(), k=2, max_ell=0, external_fields=[('dx', 1)] * 2,
                                 tau_grid=np.array([0.0]), use_cache=False, parallel=False, verbose=False)
    assert res11['marcus_truncation'][0]['relative'] < 1e-2 * rows[0]['relative']


# ── 8: affine couplings: the exact (closed-form flow) route ────────────────────────────────

def _geometric(route='auto', b=-0.3, order=10):
    """dx = (a - mu x) dt + s dW + b x <> dL, L compound Poisson with Exp(th) jumps: the Marcus map x -> x e^{bz}.
    On the series route the coefficient of xt^n x^n is ~ sum_k (n b th)^k, divergent for n > 1/(|b| th)."""
    return (SDE('geometric jumps').parameter('a', default=0.5).parameter('mu', default=1.0, domain='positive')
            .parameter('s', default=0.3).parameter('lam', default=1.0, domain='positive').parameter('b', default=b)
            .parameter('th', default=0.5, domain='positive').field('x').drift('x', 'a - mu*x')
            .noise('W', Gaussian(var='s^2'), couples={'x': 1})
            .noise('L', CompoundPoisson('lam', Exponential('th')), couples={'x': 'b*x'})
            .interpretation('marcus').marcus_route(route).series_order(order).stability_analysis(True))


def _geometric_exact(a, mu, s, lam, b, th, tau):
    """The linear moment hierarchy closes: c_k = lam (E[e^{k b z}] - 1), m = a/(mu - c1),
    M2 = (2 a m + s^2)/(2 mu - c2), C(tau) = (M2 - m^2) e^{-(mu - c1) tau}."""
    c1, c2 = lam * (1 / (1 - b * th) - 1), lam * (1 / (1 - 2 * b * th) - 1)
    m = a / (mu - c1)
    return m, ((2 * a * m + s * s) / (2 * mu - c2) - m * m) * np.exp(-(mu - c1) * tau)


def test_exact_route_builds_the_closed_form_coefficients():
    """Exponential jumps, coupling b x: P_n = lam E[(e^{bz} - 1)^n]/b^n = lam n! th^n/prod_{j<=n} (1 - j b th),
    one fraction (no alternating sum), so no cancellation at any b."""
    rec = _geometric(order=4).build()['marcus_series']['L']
    assert rec['route'] == 'exact' and rec['mgf_bases']
    b, th, lam = SR.var('b'), SR.var('th'), SR.var('lam')
    for n in range(1, 5):
        want = lam * math.factorial(n) * th ** n
        for j in range(1, n + 1):
            want /= (1 - j * b * th)
        assert (parse_expr(rec['coefficients'][str(n)]) - want).simplify_full().is_zero(), n
    # the affine conductance coupling with Poisson spikes: lam (e^{-w} - 1)^n (Es - x)^n xt^n / n!
    text = _conductance('marcus', order=3).action_text()
    assert '(exp(-w) - 1)^3' in text and 'factorial' not in text


@pytest.mark.parametrize('which', ['affine+', 'affine-', 'conductance'])
def test_exact_route_reproduces_the_series_route(which):
    """PR #86's affine models (series kept to orders 16/20, truncation below 1e-10 per degree there): the exact
    route gives the same mean and loop-graded covariance to 1e-13."""
    if which == 'conductance':
        build = lambda route: _conductance('marcus').marcus_route(route).build()                 # noqa: E731
        par = {'tau': 1.0, 'EL': 0.0, 'Es': 1.0, 'lam': 2.0, 'w': 0.3}
    else:
        build = lambda route: _affine('marcus').marcus_route(route).build()                      # noqa: E731
        par = {'mu': 1.0, 'lam': 1.5, 'a': 0.4, 'b': 0.15 if which == 'affine+' else -0.15, 'mJ': 0.2}
    ex, res_ex = _pipeline_cov(build('auto'), par)
    se, res_se = _pipeline_cov(build('series'), par)
    assert res_ex['marcus_truncation'][0]['route'] == 'exact'
    assert res_se['marcus_truncation'][0]['route'] == 'series'
    assert res_ex['mf_values']['xstar'][0] == pytest.approx(res_se['mf_values']['xstar'][0], rel=1e-13, abs=0)
    for ell in (0, 1):
        assert np.max(np.abs(ex[ell] - se[ell])) <= 1e-13 * np.max(np.abs(se[ell])), (ell, ex[ell], se[ell])


def test_geometric_exponential_jumps_exact_route_converges_without_warning():
    """The coordinator's model (scratch check of PR #86): on the exact route no advisory fires, the saddle is the
    exact mean, and the loop series to ell = 3 approaches the exact covariance with the ell >= 4 tail (3.1e-8)."""
    a, mu, s, lam, b, th = 0.5, 1.0, 0.3, 1.0, -0.3, 0.5
    tau = np.array([0.0, 0.4, 1.3])
    m, exact = _geometric_exact(a, mu, s, lam, b, th, tau)
    par = dict(a=a, mu=mu, s=s, lam=lam, b=b, th=th)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = AC.compute_cumulants(_geometric().build(), k=2, max_ell=3, external_fields=[('dx', 1)] * 2,
                                   parameters=par, tau_grid=tau, use_cache=False, parallel=False, verbose=False)
    assert not [x for x in w if issubclass(x.category, AC.MarcusTruncationWarning)]
    row = res['marcus_truncation'][0]
    assert row['route'] == 'exact' and row['status'] == 'ok' and row['relative'] < 1e-14
    assert row['mgf_degree_limit'] is None                     # b < 0: e^{bz} <= 1, every mgf value exists
    assert res['analysis'].has('G6_MARCUS_TRUNCATION') and not res['analysis'].has('G6_MARCUS_TRUNCATION_LOW')
    assert res['mf_values']['xstar'][0] == pytest.approx(m, rel=1e-14)
    by = {int(l): np.real(np.asarray(v, dtype=complex)) for l, v in res['C_tau_by_ell'].items()}
    acc, errs = np.zeros_like(tau), []
    for ell in range(4):
        acc = acc + by[ell]
        errs.append(np.max(np.abs(acc / exact - 1)))
    assert all(e1 < 0.05 * e0 for e0, e1 in zip(errs, errs[1:])), errs     # geometric convergence
    assert 1e-8 < errs[3] < 5e-8, errs                                       # the ell >= 4 tail


def test_geometric_series_route_flags_its_divergent_vertices():
    """The same model on the series route: the high-degree vertex coefficients diverge with the order, and the
    advisory says so (degrees up to k + max_ell = 5 are usable at max_ell = 3)."""
    res = {'mf_values': {'xstar': [0.4423]}, 'config': {'fundamental': {'a': 0.5, 'mu': 1.0, 's': 0.3, 'lam': 1.0,
                                                                         'b': -0.3, 'th': 0.5}}}
    row = AC._marcus_truncation(_geometric('series').build(), res, 2, 3)[0]
    assert row['route'] == 'series' and row['status'] == 'above' and row['relative_by_degree'][5] > 1e-2
    ex = AC._marcus_truncation(_geometric('auto').build(), res, 2, 3)[0]
    assert ex['route'] == 'exact' and ex['status'] == 'ok'


def _noise_coefficients(s, resp, point, nmax):
    """sum_n s^n H_n at a numeric point: the s^n coefficients (n >= 2) of -S(Dt = 0) with the responses scaled by s
    (the xt-linear drift only enters at n = 1)."""
    text = s.action_text()
    inner = re.sub(r'\s*for i in pop\)\s*$', '', re.sub(r'^\s*sum\(', '', text.strip()))
    sv = SR.var('s_')
    S0 = parse_expr(inner).subs({SR.var('Dt'): 0}).subs({SR.var(r): sv * SR.var(r) for r in resp})
    S0 = S0.subs({SR.var(k): v for k, v in point.items()}).expand()
    return {n: float(-S0.coefficient(sv, n)) for n in range(2, nmax + 1)}


def test_exact_route_two_fields_against_quadrature():
    """A diagonal vector field with a moving and a translated component and a non-unit Dt coefficient:
    G = (a1 + b1 x) d/dx + (c2/tau) d/dy.  H_n = lam E[(xt (phi_z(x) - x) + tau yt c2 z/tau)^n]/n! with
    phi_z(x) - x = (x + a1/b1)(e^{b1 z} - 1), against mpmath quadrature over the Exp(th) jump density."""
    import mpmath as mp
    mp.mp.dps = 30
    s = (SDE('two fields').parameter('lam', default=1.2).parameter('a1', default=0.3).parameter('b1', default=-0.4)
         .parameter('c2', default=0.7).parameter('tau', default=2.0).parameter('th', default=0.6)
         .field('x').field('y').drift('x', '-x').equation('y', lhs='(tau*Dt + 1)*y', rhs='0')
         .noise('L', CompoundPoisson('lam', Exponential('th')), couples={'x': 'a1 + b1*x', 'y': 'c2'})
         .interpretation('marcus').series_order(5))
    assert s.build()['marcus_series']['L']['route'] == 'exact'
    vals = {'lam': 1.2, 'a1': 0.3, 'b1': -0.4, 'c2': 0.7, 'tau': 2.0, 'th': 0.6}
    x, xt, yt = 0.9, 0.8, -0.5
    got = _noise_coefficients(s, ['xt__i', 'yt__i'], {**vals, 'x__i': x, 'y__i': 0.2, 'xt__i': xt, 'yt__i': yt}, 5)
    u = xt * (x + vals['a1'] / vals['b1'])
    for n, g in got.items():
        f = lambda z: (u * mp.expm1(vals['b1'] * z) + yt * vals['c2'] * z) ** n * mp.exp(-z / vals['th']) / vals['th']  # noqa: E731,E501
        want = vals['lam'] * float(mp.quad(f, [0, mp.inf])) / math.factorial(n)
        assert g == pytest.approx(want, rel=1e-13, abs=1e-15), (n, g, want)


def test_exact_route_refuses_where_the_mgf_does_not_exist():
    """b > 0 with exponential jumps: E[e^{n b z}] = 1/(1 - n b th) exists only for n b th < 1.  Degree n's
    vertices need it; refused, naming the degree, when k + max_ell reaches it, else recorded."""
    tau = np.array([0.0])
    par = dict(a=0.5, mu=1.0, s=0.3, lam=1.0, b=0.6, th=0.5)       # b th = 0.3: degrees >= 4 diverge
    m = _geometric(b=0.6).build()
    with pytest.raises(ValueError, match=r'degree 4 -- .*mgf does not exist'):
        AC.compute_cumulants(m, k=2, max_ell=2, external_fields=[('dx', 1)] * 2, parameters=par,
                             tau_grid=tau, use_cache=False, parallel=False, verbose=False)
    res = AC.compute_cumulants(m, k=2, max_ell=1, external_fields=[('dx', 1)] * 2, parameters=par,
                               tau_grid=tau, use_cache=False, parallel=False, verbose=False)
    assert res['marcus_truncation'][0]['mgf_degree_limit'] == 4
    with pytest.raises(ValueError, match=r'degree 1 -- .*mean drift itself is infinite'):
        AC.compute_cumulants(m, k=2, max_ell=0, external_fields=[('dx', 1)] * 2,
                             parameters={**par, 'b': 2.5}, tau_grid=tau, use_cache=False, parallel=False,
                             verbose=False)


def test_exact_route_rounding_advisory_and_series_remedy():
    """A Gaussian jump law has no one-fraction form: at a small slope the alternating mgf sum cancels.  The
    rounding advisory fires and names the series route, which is accurate there."""
    from api.noise import GaussianJump

    def model(route):
        return (SDE('small slope').parameter('mu', default=1.0, domain='positive').parameter('lam', default=1.5)
                .parameter('a', default=0.4).parameter('b', default=1e-4).field('x').drift('x', '-mu*x')
                .noise('L', CompoundPoisson('lam', GaussianJump(0.1, 0.3)), couples={'x': 'a + b*x'})
                .interpretation('marcus').marcus_route(route).stability_analysis(True))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _, res = _pipeline_cov(model('auto').build(), {}, max_ell=0)
    msg = [str(x.message) for x in w if issubclass(x.category, AC.MarcusTruncationWarning)]
    assert msg and 'rounding' in msg[0] and "marcus_route('series')" in msg[0]
    assert res['marcus_truncation'][0]['status'] == 'above'
    assert res['analysis'].has('G6_MARCUS_TRUNCATION_LOW')
    _, res = _pipeline_cov(model('series').build(), {}, max_ell=0)
    assert res['marcus_truncation'][0]['status'] == 'ok'


def test_marcus_route_validation_and_config(tmp_path):
    with pytest.raises(SDEError, match="'auto' or 'series'"):
        SDE('x').marcus_route('exact')
    from api.config_run import load_config
    p = tmp_path / 'm.yaml'
    p.write_text("""\
sde:
  name: route
  parameters: {mu: {default: 1.0, domain: positive}, lam: 1.5, b: 0.1}
  fields: [x]
  drift: {x: "-mu*x"}
  noise:
    L: {type: Poisson, rate: lam, couples: {x: "1 + b*x"}}
  interpretation: marcus
  marcus_route: series
question: {k: 2, max_ell: 1, tau_grid: [0.0]}
""")
    resolved, model, _, _ = load_config(str(p))
    assert resolved['sde']['marcus_route'] == 'series' and model['marcus_series']['L']['route'] == 'series'
    p.write_text(p.read_text().replace('  marcus_route: series\n', ''))
    assert load_config(str(p))[1]['marcus_series']['L']['route'] == 'exact'


# ── 9: the advisory counts only the vertices a run can use ──────────────────────────────────

def test_advisory_weights_vertices_by_what_k_and_max_ell_can_use():
    """A vertex xt^r dx^q (about the saddle) can enter a k-point max_ell-loop diagram only if r + q <= k + 2 max_ell
    and r <= k + max_ell.  A large omitted xt^4 term is no concern at k = 2, max_ell = 1; an omitted xt^2 x^6 term
    (degree 8 about the saddle x* = 0) only from max_ell = 3."""
    def model(nxt):
        return {'marcus_series': {'L': {'route': 'series', 'order': 4, 'responses': ['xt'], 'fields': ['x'],
                                        'kept': 'xt__i*x__i + xt__i^2 + xt__i^4 + xt__i^2*x__i^6', 'next': nxt}}}
    res = {'mf_values': {'xstar': [0.0]}, 'config': {'fundamental': {}}}
    r4 = model('1e-3*xt__i^4')
    assert AC._marcus_truncation(r4, res, 2, 1)[0]['status'] == 'ok'
    assert AC._marcus_truncation(r4, res, 2, 2)[0]['relative_by_degree'][4] == pytest.approx(1e-3)
    d8 = model('1e-3*xt__i^2*x__i^6')
    for ell in (1, 2):
        assert AC._marcus_truncation(d8, res, 2, ell)[0]['status'] == 'ok', ell
    assert AC._marcus_truncation(d8, res, 2, 3)[0]['relative_by_degree'][2] == pytest.approx(1e-3)
    # off x* = 0 the degree-8 monomial also feeds the usable xt^2 dx^q vertices, which count
    res1 = {'mf_values': {'xstar': [1.0]}, 'config': {'fundamental': {}}}
    assert AC._marcus_truncation(d8, res1, 2, 1)[0]['status'] == 'above'


def test_rotation_series_advisory_tracks_max_ell():
    """The non-commuting rotation stays on the series route; at order 15 its usable vertices are below 1e-10
    at max_ell = 1 but not at max_ell = 2 (the xt^3 vertices join)."""
    rot = _rotation('marcus').series_order(15).build()
    assert rot['marcus_series']['L']['route'] == 'series'
    res = {'mf_values': {'xstar': [0.0], 'ystar': [0.0]}, 'config': {'fundamental': {'mu': 1.0, 'lam': 1.0, 'z': 0.7}}}
    assert AC._marcus_truncation(rot, res, 2, 1)[0]['status'] == 'ok'
    assert AC._marcus_truncation(rot, res, 2, 2)[0]['status'] == 'above'


def test_mgf_check_sees_even_order_poles():
    """Gamma(2) jumps: K = lam((1 - th theta)^(-2) - 1), whose expanded base th^2 theta^2 - 2 th theta + 1 touches 0
    at the pole without changing sign; NegativeBinomial(r = 2): (1 - p e^theta)^(-2).  Both are seen (the
    unexpanded bases and the real roots of polynomial bases), so a b past the pole is refused, not computed."""
    from api.noise import Gamma, NegativeBinomial
    c = 0.37
    quartic = ' + '.join(f'({math.comb(4, j) * (-c) ** j!r})*theta^{j}' for j in range(5))
    cases = ((CompoundPoisson('lam', Gamma(2, 'th')), {'b': 1.5, 'th': 0.5}, 2),     # n b th >= 1 from n = 2
             (CompoundPoisson('lam', Gamma(2, 'th')), {'b': 10.0, 'th': 0.5}, 1),
             (CompoundPoisson('lam', NegativeBinomial(2, 0.5)), {'b': 0.4}, 2),     # e^{n b} p >= 1: n b >= log 2
             # the same laws typed in expanded: a double root in u = e^theta, a 4-fold root in theta at 1/c
             (CGF('1/(4 - 4*exp(theta) + exp(2*theta)) - 1'), {'b': 0.3}, 3),
             (CGF(f'1/({quartic}) - 1'), {'b': 0.7}, 4))                            # 4 b c >= 1
    for src, vals, n0 in cases:
        s = (SDE('even pole').parameter('a', default=0.5).parameter('mu', default=1.0, domain='positive')
             .parameter('lam', default=1.0).parameter('b', default=vals['b']).parameter('th', default=0.5)
             .field('x').drift('x', 'a - mu*x')
             .noise('L', src, couples={'x': 'b*x'}).interpretation('marcus').series_order(6))
        rec = s.build()['marcus_series']['L']
        assert rec['route'] == 'exact'
        assert AC._marcus_mgf_degree(rec, vals)[0] == n0, (src, vals)
    m = s.build()
    with pytest.raises(ValueError, match=r'degree 4 -- .*mgf does not exist'):
        AC.compute_cumulants(m, k=2, max_ell=2, external_fields=[('dx', 1)] * 2, tau_grid=np.array([0.0]),
                             use_cache=False, parallel=False, verbose=False)


def test_exact_route_needs_k_plus_max_ell_response_legs():
    """The exact term is kept to SDE.series_order response legs and a run uses up to k + max_ell of them: order 5
    serves k = 2, max_ell = 3 (the moment series needs k + 2 max_ell = 8); order 4 is refused, by name."""
    par = dict(a=0.5, mu=1.0, s=0.3, lam=1.0, b=-0.3, th=0.5)
    m5 = _geometric(order=5).build()
    assert 'noise_series_order' not in m5 and m5['marcus_exact_order'] == {'order': 5, 'sources': ['L']}
    with pytest.raises(ValueError, match=r'exact noise term to 4 response legs.*series_order\(5\)'):
        AC.compute_cumulants(_geometric(order=4).build(), k=2, max_ell=3, external_fields=[('dx', 1)] * 2,
                             parameters=par, tau_grid=np.array([0.0]), use_cache=False, parallel=False,
                             verbose=False)
    ser = _geometric('series', order=5).build()
    assert ser['noise_series_order']['order'] == 5

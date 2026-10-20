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

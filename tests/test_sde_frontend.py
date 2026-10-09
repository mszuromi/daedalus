"""SDE front-end (api/sde.py): the action and mean-field equations derived from declared fields, drift and noise
sources, checked against the tracked hand-written models (A, C), physics oracles through the full pipeline (B) and the
rule errors (D).  The ell = 2 pipeline runs are marked slow."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR, function, preparse                                 # noqa: E402
import sage.all as sage_all                                                   # noqa: E402

import api.compute as AC                                                      # noqa: E402
import api.model as api_model                                                 # noqa: E402
import tests._ito_moments as im                                               # noqa: E402
from api._mean_field_dae import linear_stability, solve_mean_field_dae        # noqa: E402
from api.noise import (CompoundPoisson, Dirac, Gaussian, NoiseError, Poisson,  # noqa: E402
                       Uniform)
from api.sde import SDE, SDEError                                             # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / 'models'
T_STATIONARY = 150.0


# ── an independent evaluator of action / equation text (no api.model_compiler) ──────────────

class _Idx:
    """``name[i]`` / ``name[i, j]`` -> a fresh symbol ``name_i_j``; ``name(...)`` -> a formal function."""

    def __init__(self, name):
        self.name = name

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return SR.var(self.name + '_' + '_'.join(str(int(k)) for k in key))

    def __call__(self, *args):
        return function(self.name)(*args)


def _indexed_call(name, *key):
    return function(name + '_' + '_'.join(str(int(k)) for k in key))


def _to_sr(text, *, n=1, pops=('pop',), indexed=(), scalars=(), i=None):
    """Evaluate Sage-syntax model text to SR; names in ``indexed`` are indexable, ``scalars`` are plain symbols."""
    ns = dict(sage_all.__dict__)
    for p in pops:
        ns[p] = range(n)
    for name in indexed:
        ns[name] = _Idx(name)
    for name in scalars:
        ns[name] = SR.var(name)
    ns['Dt'] = SR.var('Dt')
    ns['Conv'] = lambda k, x: k * x
    ns['_call'] = _indexed_call
    if i is not None:
        ns['i'] = i
    text = re.sub(r'\b(\w+)\[([^\]]+)\]\s*\(', r"_call('\1', \2)(", ' '.join(text.split()))
    return SR(eval(preparse(text), ns))


def _same(a, b):
    return (SR(a) - SR(b)).expand().simplify_full().is_zero()


def _hand_builder(model_file, monkeypatch):
    """The tracked model's TemporalModelBuilder just before .build() (reads the hand-written text)."""
    spec = importlib.util.spec_from_file_location('hand_' + Path(model_file).stem.replace('.', '_'),
                                                  MODELS / model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with monkeypatch.context() as mp:
        mp.setattr(api_model._BaseModelBuilder, 'build', lambda self: self)
        b = mod.build()
    return b, mod


def _residuals(eqs, **kw):
    out = []
    for e in eqs:
        lhs = e['lhs_text'] if 'lhs_text' in e else e['lhs']
        rhs = e['rhs_text'] if 'rhs_text' in e else e['rhs']
        out.append(_to_sr(f'({lhs}) - ({rhs})', i=0, **kw))
    return out


# ── SDE declarations of the tracked models ──────────────────────────────────────────────────

def _ou(name, rhs, defaults, drift_form=True, stability=None):
    s = SDE(name).population('pop', size=1).field('x', population='pop', description='variable')
    for p, (d, dom) in defaults.items():
        s.parameter(p, default=d, domain=dom)
    if drift_form:
        s.drift('x', f'-mu*x + {rhs}')
    else:
        s.equation('x', lhs='(Dt+mu)*x[i]', rhs=rhs)
    s.noise('xi', Gaussian(var='2*D'), couples={'x': 1})
    if stability is not None:
        s.stability_analysis(stability)
    return s


def sde_ou_quartic(drift_form=True):
    return _ou('OU Quartic (white noise)', '-eps*x^3',
               {'mu': (1.0, 'positive'), 'eps': (0.02, 'positive'), 'D': (1.0, 'positive')}, drift_form)


def sde_ou_sextic(drift_form=True):
    return _ou('OU Sextic', '-eps*x^3 - gamma*x^5',
               {'mu': (1.0, 'positive'), 'eps': (0.05, 'positive'), 'D': (1.0, 'positive'),
                'gamma': (0.05, 'positive')}, drift_form, stability=True)


def sde_ou_double_well(drift_form=True):
    s = _ou('OU Quartic Double Well', '-eps*x^3',
            {'mu': (0.1, 'real'), 'eps': (0.1, 'positive'), 'D': (1.0, 'positive')}, drift_form, stability=True)
    s._fields['x']['domain'] = 'real'
    return s


def sde_linear_hawkes():
    return (SDE('Linear Hawkes 2-pop', n_populations=2)
            .parameter('E', indexed=True, default=[0.78, 0.81])
            .parameter('tau', default=10.0, domain='positive')
            .parameter('a', default=1.0)
            .parameter('tau_g', default=2.5, domain='positive')
            .parameter('w', indexed='matrix', default=[[0.30, 0.25], [0.30, 0.35]])
            .define_function('phi', args=['v'], expression='a * v', latex=r'\varphi')
            .define_kernel('g', freq_image='1 / (1 + I*omega*tau_g)', latex_name='g')
            .noise('spikes', Poisson(rate='phi(v)'), expose='n')
            .field('v')
            .equation('v', lhs='(tau*Dt + 1)*v', rhs='E[i] + sum(w[i, j]*g*n[j] for j in pop)'))


def sde_exp_hawkes(raw_rate=True):
    s = (SDE('exp Hawkes 1-pop (SDE)')
         .parameter('Em', default=-1.5).parameter('tau', default=2.0, domain='positive')
         .parameter('taug', default=1.0, domain='positive').parameter('a', default=1.0, domain='positive')
         .parameter('w', default=0.4)
         .define_kernel('g', time_expr='exp(-t/taug)/taug*heaviside(t)'))
    if raw_rate:
        s.noise('spikes', Poisson(rate='a*exp(v)'), expose='n')
    else:
        s.define_function('phi', args=['v'], expression='a*exp(v)')
        s.noise('spikes', Poisson(rate='phi(v)'), expose='n')
    return s.field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + w*g*n').stability_analysis(True)


OU_CASES = {
    'ou_quartic.model.py': (sde_ou_quartic, ('mu', 'eps', 'D')),
    'ou_sextic.model.py': (sde_ou_sextic, ('mu', 'eps', 'D', 'gamma')),
    'ou_quartic_double_well.model.py': (sde_ou_double_well, ('mu', 'eps', 'D')),
}


# ── A: action round trip ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('model_file', sorted(OU_CASES))
@pytest.mark.parametrize('drift_form', [True, False])
def test_action_and_equations_round_trip_ou(model_file, drift_form, monkeypatch):
    make, scalars = OU_CASES[model_file]
    hand, _ = _hand_builder(model_file, monkeypatch)
    sde = make(drift_form)
    kw = dict(indexed=('x', 'xt'), scalars=scalars)
    assert _same(_to_sr(sde.action_text(), **kw), _to_sr(hand._action_text, **kw))
    assert not _same(_to_sr(sde.action_text(), **kw), _to_sr(hand._action_text.replace('D*', '2*D*'), **kw))
    got, ref = _residuals(sde.equations(), **kw), _residuals(hand._equations, **kw)
    assert len(got) == len(ref) == 1 and _same(got[0], ref[0])


def test_action_round_trip_two_field_cross_correlated_gaussian():
    """Two coupled fields, an own Gaussian source each and one common source feeding both (cross-correlation)."""
    ref = ('sum(xt[i]*((Dt+mu1)*x[i] - J1*y[i]) + yt[i]*((Dt+mu2)*y[i] - J2*x[i] + eps*x[i]^2)'
           ' - (D1 + r)*xt[i]^2 - (D2 + r)*yt[i]^2 - 2*r*xt[i]*yt[i] for i in pop)')
    sde = (SDE('two-field').field('x').field('y')
           .equation('x', lhs='(Dt+mu1)*x', rhs='J1*y')
           .equation('y', lhs='(Dt+mu2)*y', rhs='J2*x - eps*x^2')
           .noise('own_x', Gaussian(var='2*D1'), couples={'x': 1})
           .noise('own_y', Gaussian(var='2*D2'), couples={'y': 1})
           .noise('common', Gaussian(var='2*r'), couples={'x': 1, 'y': 1}))
    for p in ('mu1', 'mu2', 'J1', 'J2', 'eps', 'D1', 'D2', 'r'):
        sde.parameter(p)
    kw = dict(indexed=('x', 'xt', 'y', 'yt'), scalars=('mu1', 'mu2', 'J1', 'J2', 'eps', 'D1', 'D2', 'r'))
    assert _same(_to_sr(sde.action_text(), **kw), _to_sr(ref, **kw))
    ref_eqs = [{'lhs': '(Dt+mu1)*x[i]', 'rhs': 'J1*y[i]'}, {'lhs': '(Dt+mu2)*y[i]', 'rhs': 'J2*x[i] - eps*x[i]^2'}]
    for g, r in zip(_residuals(sde.equations(), **kw), _residuals(ref_eqs, **kw)):
        assert _same(g, r)


def test_action_round_trip_linear_hawkes(monkeypatch):
    hand, _ = _hand_builder('linear_hawkes.model.py', monkeypatch)
    kw = dict(n=2, pops=('pop', 'pop_all'), indexed=('n', 'nt', 'v', 'vt', 'E', 'w', 'phi'),
              scalars=('tau', 'a', 'tau_g', 'g'))
    assert _same(_to_sr(sde_linear_hawkes().action_text(), **kw), _to_sr(hand._action_text, **kw))


def test_exposed_hawkes_action_is_the_hand_written_form():
    hand = 'sum(nt[i]*n[i] - (exp(nt[i]) - 1)*a*exp(v[i]) + vt[i]*((tau*Dt + 1)*v[i] - Em - w*g*n[i]) for i in pop)'
    kw = dict(indexed=('n', 'nt', 'v', 'vt'), scalars=('a', 'tau', 'Em', 'w', 'g'))
    assert _same(_to_sr(sde_exp_hawkes().action_text(), **kw), _to_sr(hand, **kw))
    eqs = sde_exp_hawkes().equations()
    assert [e['lhs'] for e in eqs] == ['n[i]', '(tau*Dt + 1)*v[i]']
    assert _same(_to_sr(eqs[0]['rhs'], i=0, **kw), _to_sr('a*exp(v[0])', **kw))
    assert _same(_to_sr(eqs[1]['rhs'], i=0, **kw), _to_sr('Em + w*n[0]', **kw))   # kernel integral = 1


def test_to_builder_continues_by_hand_and_show_is_copyable(capsys):
    sde = sde_ou_quartic()
    b = sde.to_builder()
    assert isinstance(b, api_model.TemporalModelBuilder)
    assert b._action_text == sde.action_text()
    b.set_action_text(b._action_text + ' + 0')                    # the builder stays editable
    text = sde.show()
    assert 'set_action_text' in capsys.readouterr().out
    assert sde.action_text() in text and ".equation(lhs='Dt*x[i] + mu*x[i]'" in text


# ── C: mean field and linear stability ──────────────────────────────────────────────────────

@pytest.mark.parametrize('model_file', sorted(OU_CASES))
def test_mean_field_and_stability_match_hand_written(model_file):
    make, _ = OU_CASES[model_file]
    spec = importlib.util.spec_from_file_location('mf_' + model_file.split('.')[0], MODELS / model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hand, gen = mod.build(), make().build()
    assert gen['stability_analysis'] == hand['stability_analysis']
    for par in (mod.DEFAULT_FUNDAMENTAL or {'mu': 1.0, 'eps': 0.05, 'D': 1.0, 'gamma': 0.05},
                {'mu': -1.0, 'eps': 0.1, 'D': 1.0, 'gamma': 0.02}):
        mh = solve_mean_field_dae(hand, par, verbose=False)
        mg = solve_mean_field_dae(gen, par, verbose=False)
        rh = sorted(float(r['values']['xstar'][0]) for r in mh['mf_all_roots'])
        rg = sorted(float(r['values']['xstar'][0]) for r in mg['mf_all_roots'])
        assert rg == pytest.approx(rh, abs=1e-10)
        for rec in mh['mf_all_roots']:
            root = rec['values']
            sh, sg = linear_stability(hand, par, root), linear_stability(gen, par, root)
            assert sh['stable'] == sg['stable']
            assert np.allclose(np.sort_complex(sh['eigenvalues_finite']),
                               np.sort_complex(sg['eigenvalues_finite']))


# ── B: physics oracles through the full pipeline ────────────────────────────────────────────

def _cov_by_ell(model, max_ell, par, ext=('dx', 1)):
    res = AC.compute_cumulants(model, k=2, max_ell=max_ell, external_fields=[ext, ext], parameters=par,
                               tau_grid=np.array([0.0, 1.0]), use_cache=False, parallel=False, verbose=False)
    return {int(l): complex(a[0]).real for l, a in res['C_tau_by_ell'].items()}, res


def _equal_time(model, k, par, ell=0, ext=('dx', 1)):
    res = AC.compute_cumulants(model, k=k, max_ell=ell, external_fields=[ext] * k, parameters=par,
                               tau_grid=np.array([0.0, 1.0]), use_cache=False, parallel=False, verbose=False)
    if k == 2:
        return complex(res['C_tau_by_ell'][ell][0]).real
    return complex(res['total_C_by_ell'][ell](*([0.0] * k))).real


@pytest.mark.parametrize('max_ell', [1, pytest.param(2, marks=pytest.mark.slow)])
def test_ou_quartic_from_sde_boltzmann_coefficients(max_ell):
    """<x x> at equal time = sum_ell c_ell eps^ell with c = 1, -3, 24 (mu = D = 1)."""
    eps = 0.02
    got, _ = _cov_by_ell(sde_ou_quartic().build(), max_ell, {'mu': 1.0, 'eps': eps, 'D': 1.0})
    assert got[0] == pytest.approx(1.0, rel=1e-6, abs=0.0)              # Ito tau = 0 nudge
    for ell, c in [(1, -3.0), (2, 24.0)][:max_ell]:
        assert got[ell] / eps ** ell == pytest.approx(c, rel=1e-9, abs=0.0)


def _poisson_ou(exposed):
    s = SDE('Poisson-driven OU' + (' (exposed)' if exposed else ''))
    for p, d in (('mu', 1.0), ('lam', 3.0), ('a', 0.7)):
        s.parameter(p, default=d, domain='positive')
    s.field('x')
    if exposed:
        s.noise('eta', CompoundPoisson('lam', Dirac('a')), expose='n', domain='positive')
        s.drift('x', '-mu*x + n')
    else:
        s.drift('x', '-mu*x').noise('eta', CompoundPoisson('lam', Dirac('a')), couples={'x': 1})
    return s.stability_analysis(True).build()


@pytest.mark.parametrize('exposed', [False, True])
def test_poisson_driven_ou_cumulants(exposed):
    """kappa_m = lam a^m / (m mu) at ell = 0; the exposed and integrated-out forms agree."""
    par = {'mu': 1.0, 'lam': 3.0, 'a': 0.7}
    model = _poisson_ou(exposed)
    for k, ref in ((2, 0.735), (3, 0.343), (4, 0.180075)):
        tol = 1e-6 if k == 2 else 1e-9                                   # k = 2 carries the Ito tau = 0 nudge
        assert _equal_time(model, k, par) == pytest.approx(ref, rel=tol, abs=0.0)


def _mult(m=None):
    return (SDE('multiplicative linear variance').parameter('mu', default=1.0, domain='positive')
            .parameter('m', default=2.0, domain='positive').parameter('s0', default=0.5, domain='positive')
            .parameter('s1', default=0.3, domain='positive').field('x')
            .drift('x', '-mu*(x - m)').noise('W', Gaussian(var='s0 + s1*x'), couples={'x': 1})
            .stability_analysis(True).build())


@pytest.mark.parametrize('max_ell', [1, pytest.param(2, marks=pytest.mark.slow)])
def test_multiplicative_linear_variance_closes(max_ell):
    par = {'mu': 1.0, 'm': 2.0, 's0': 0.5, 's1': 0.3}
    got, res = _cov_by_ell(_mult(), max_ell, par)
    assert res['mf_values']['xstar'][0] == pytest.approx(2.0, abs=1e-12)
    assert got[0] == pytest.approx((0.5 + 0.3 * 2.0) / 2.0, rel=1e-6, abs=0.0)
    for ell in range(1, max_ell + 1):
        assert abs(got[ell]) < 1e-10


def test_multiplicative_third_cumulant_against_moment_hierarchy():
    mu, m, s0, s1 = 1.0, 2.0, 0.5, 0.3
    sde = im.PolySDE(D=1, n_tags=1)
    sde.add_linear_drift(np.array([[-mu]]))
    sde.diffusion.append((0, 0, s0 + s1 * m, (0,), (0,)))
    sde.diffusion.append((0, 0, s1, (1,), (1,)))                       # S(X) = s0 + s1 (m + X), tagged by s1
    x1, x3 = im.unit(1, 0), im.unit(1, 0, 0, 0)
    targets = [(x1, (t,)) for t in range(3)] + [(x3, (t,)) for t in range(3)]
    out = im.solve(sde, targets, [T_STATIONARY])
    mean = sum(out[(x1, (t,))][0] for t in range(3))
    third = sum(out[(x3, (t,))][0] for t in range(3))
    assert abs(mean) < 1e-12                                            # moments about the saddle = cumulants
    got = _equal_time(_mult(), 3, {'mu': mu, 'm': m, 's0': s0, 's1': s1})
    assert got == pytest.approx(third, rel=1e-8, abs=0.0)
    assert got == pytest.approx(s1 * (s0 + s1 * m) / (2 * mu ** 2), rel=1e-8)


def _strat_mult(stratonovich=True):
    s = (SDE('Stratonovich multiplicative').parameter('mu', default=1.0, domain='positive')
         .parameter('s0', default=0.6, domain='positive').parameter('s1', default=0.2, domain='positive')
         .field('x'))
    if stratonovich:
        s.drift('x', '-mu*x').noise('W', Gaussian(var=1), couples={'x': 's0 + s1*x'}).interpretation('stratonovich')
    else:
        s.drift('x', '-mu*x + s1*(s0 + s1*x)/2').noise('W', Gaussian(var=1), couples={'x': 's0 + s1*x'})
    return s.stability_analysis(True)


def test_stratonovich_additive_equals_ito():
    ito = sde_ou_quartic()
    strat = sde_ou_quartic().interpretation('stratonovich')
    assert strat.action_text() == ito.action_text() and strat.equations() == ito.equations()


def test_stratonovich_multiplicative_matches_ito_with_drift_correction():
    kw = dict(indexed=('x', 'xt'), scalars=('mu', 's0', 's1'))
    assert _same(_to_sr(_strat_mult(True).action_text(), **kw), _to_sr(_strat_mult(False).action_text(), **kw))


def test_stratonovich_multi_field_known_answer():
    """dX = B X o dW with B = [[0, 1], [1, 0]]: the Ito drift correction is B^2 X / 2 = X / 2; with an lhs
    (tau*Dt + 1) the coefficient enters as c/tau and the correction in lhs units is c c' / (2 tau)."""
    s = (SDE('strat 2d').parameter('mu').parameter('tau').field('x').field('y')
         .drift('x', '-mu*x').drift('y', '-mu*y')
         .noise('W', Gaussian(var=1), couples={'x': 'y', 'y': 'x'}).interpretation('stratonovich'))
    r = (SDE('ito 2d').parameter('mu').parameter('tau').field('x').field('y')
         .drift('x', '-mu*x + x/2').drift('y', '-mu*y + y/2')
         .noise('W', Gaussian(var=1), couples={'x': 'y', 'y': 'x'}))
    kw = dict(indexed=('x', 'xt', 'y', 'yt'), scalars=('mu', 'tau'))
    assert _same(_to_sr(s.action_text(), **kw), _to_sr(r.action_text(), **kw))
    s1 = (SDE('strat tau').parameter('tau').parameter('c0').parameter('c1').field('x')
          .equation('x', lhs='(tau*Dt + 1)*x', rhs='0')
          .noise('W', Gaussian(var=1), couples={'x': 'c0 + c1*x'}).interpretation('stratonovich'))
    r1 = (SDE('ito tau').parameter('tau').parameter('c0').parameter('c1').field('x')
          .equation('x', lhs='(tau*Dt + 1)*x', rhs='c1*(c0 + c1*x)/(2*tau)')
          .noise('W', Gaussian(var=1), couples={'x': 'c0 + c1*x'}))
    kw = dict(indexed=('x', 'xt'), scalars=('tau', 'c0', 'c1'))
    assert _same(_to_sr(s1.action_text(), **kw), _to_sr(r1.action_text(), **kw))


def test_stratonovich_multiplicative_against_moment_hierarchy():
    """Ito form: drift -(mu - s1^2/2) X about the saddle, diffusion (A + s1 X)^2 with A = s0 + s1 x*;
    a diagram with ell loops carries s1^(2 ell) from the diffusion vertices, so ell = 0, 1 <-> tags 0, 2."""
    mu, s0, s1 = 1.0, 0.6, 0.2
    par = {'mu': mu, 's0': s0, 's1': s1}
    got, res = _cov_by_ell(_strat_mult(True).build(), 1, par)
    xs = s0 * s1 / 2 / (mu - s1 ** 2 / 2)
    assert res['mf_values']['xstar'][0] == pytest.approx(xs, rel=1e-12)
    A = s0 + s1 * xs
    sde = im.PolySDE(D=1, n_tags=1)
    sde.add_linear_drift(np.array([[-(mu - s1 ** 2 / 2)]]))
    sde.diffusion += [(0, 0, A ** 2, (0,), (0,)), (0, 0, 2 * A * s1, (1,), (1,)), (0, 0, s1 ** 2, (2,), (2,))]
    xx = im.unit(1, 0, 0)
    out = im.solve(sde, [(xx, (0,)), (xx, (1,)), (xx, (2,))], [T_STATIONARY])
    assert abs(out[(xx, (1,))][0]) < 1e-14
    assert got[0] == pytest.approx(out[(xx, (0,))][0], rel=2e-6, abs=0.0)
    # rel 2e-6 at both orders: the Ito tau = 0 nudge (tau = -1e-6) moves C by ~ (mu - s1^2/2) * 1e-6 relative,
    # as in tests/test_phase_j_moment_oracle.py.
    assert got[1] == pytest.approx(out[(xx, (2,))][0], rel=2e-6, abs=0.0)


HAWKES_PAR = {'Em': -1.5, 'tau': 2.0, 'taug': 1.0, 'a': 1.0, 'w': 0.4}
HAWKES_REF = {0: 7.283331497e-3, 1: 7.036444021e-5, 2: 3.797794840e-6}


@pytest.mark.parametrize('max_ell, raw_rate', [(1, True), (1, False),
                                               pytest.param(2, True, marks=pytest.mark.slow)])
def test_exposed_exponential_hawkes(max_ell, raw_rate):
    got, res = _cov_by_ell(sde_exp_hawkes(raw_rate).build(), max_ell, HAWKES_PAR, ext=('dv', 1))
    assert res['mf_values']['vstar'][0] == pytest.approx(-1.401510052, abs=1e-9)
    assert res['mf_values']['nstar'][0] == pytest.approx(0.246224871, abs=1e-9)
    assert got[0] == pytest.approx(HAWKES_REF[0], rel=1e-6, abs=0.0)
    assert got[1] == pytest.approx(HAWKES_REF[1], rel=1e-8, abs=0.0)
    if max_ell >= 2:
        assert got[2] == pytest.approx(HAWKES_REF[2], rel=1e-7, abs=0.0)


def test_exposed_exponential_hawkes_rate_one_loop():
    res = AC.compute_cumulants(sde_exp_hawkes().build(), k=1, max_ell=1, external_fields=[('dn', 1)],
                               parameters=HAWKES_PAR, tau_grid=np.array([0.0]), use_cache=False,
                               parallel=False, verbose=False)
    assert complex(res['C_tau_by_ell'][1][0]).real == pytest.approx(9.946297e-4, rel=1e-6)


def test_linear_hawkes_matches_hand_written_and_loops_vanish(monkeypatch):
    _, mod = _hand_builder('linear_hawkes.model.py', monkeypatch)
    par = mod.DEFAULT_FUNDAMENTAL
    ext = [('dn', 1), ('dn', 2)]
    gen = AC.compute_cumulants(sde_linear_hawkes().build(), k=2, max_ell=1, external_fields=ext, parameters=par,
                               tau_grid=np.array([0.0, 2.0]), use_cache=False, parallel=False, verbose=False)
    hand = AC.compute_cumulants(mod.build(), k=2, max_ell=0, external_fields=ext, parameters=par,
                                tau_grid=np.array([0.0, 2.0]), use_cache=False, parallel=False, verbose=False)
    for f in ('nstar', 'vstar'):
        assert np.allclose(gen['mf_values'][f], hand['mf_values'][f], rtol=1e-10, atol=0.0)
    assert np.allclose(gen['C_tau_by_ell'][0], hand['C_tau_by_ell'][0], rtol=1e-9, atol=0.0)
    assert np.max(np.abs(gen['C_tau_by_ell'][1])) < 1e-12


def test_population_model_with_indexed_parameters():
    """Heterogeneous network: two-population E/I-style exposed Hawkes with matrix weights and per-pair kernels."""
    s = (SDE('E pop Hawkes').population('E', size=2)
         .parameter('Em', default=[-1.5, -1.4], indexed_by=['E'])
         .parameter('tau', default=[2.0, 3.0], indexed_by=['E'], domain='positive')
         .parameter('taug', default=[[1.0, 1.5], [1.2, 0.8]], indexed_by=['E', 'E'], domain='positive')
         .parameter('w', default=[[0.1, 0.2], [0.15, 0.1]], indexed_by=['E', 'E'])
         .parameter('a', default=1.0, domain='positive')
         .define_function('phi', args=['v'], expression='a*exp(v)', population='E')
         .define_kernel('g', time_expr='exp(-t/taug[i,j])/taug[i,j]*heaviside(t)', indexed_by=['E', 'E'])
         .noise('spikes', Poisson(rate='phi(v)'), expose='n', population='E')
         .field('v', population='E')
         .equation('v', lhs='(tau[i]*Dt + 1)*v', rhs='Em[i] + sum(w[i,j]*g[i,j]*n[j] for j in E)'))
    hand = ('sum(nt[i]*n[i] - (exp(nt[i]) - 1)*phi[i](v[i]) + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i] '
            '- sum(w[i,j]*g[i,j]*n[j] for j in E)) for i in E)')
    kw = dict(n=2, pops=('E',), indexed=('n', 'nt', 'v', 'vt', 'tau', 'Em', 'w', 'g', 'phi'), scalars=('a',))
    assert _same(_to_sr(s.action_text(), **kw), _to_sr(hand, **kw))
    eqs = s.equations()
    assert eqs[1]['rhs'] == 'Em[i] + sum(w[i,j]*(1)*n[j] for j in E)' and eqs[1]['population'] == 'E'
    par = {'Em': [-1.5, -1.4], 'tau': [2.0, 3.0], 'taug': [[1.0, 1.5], [1.2, 0.8]],
           'w': [[0.1, 0.2], [0.15, 0.1]], 'a': 1.0}
    # np.errstate: an exp overflow at a far seed would emit a numpy warning inside the DAE solver's eval with empty
    # __builtins__, which raises KeyError('__import__') (pre-existing; see the A1 report).
    with np.errstate(over='ignore', invalid='ignore'):
        mf = solve_mean_field_dae(s.build(), par, verbose=False)
    v, n = np.asarray(mf['mf_values']['vstar']), np.asarray(mf['mf_values']['nstar'])
    assert np.allclose(n, np.exp(v), atol=1e-10)
    assert np.allclose(v, np.array([-1.5, -1.4]) + np.array([[0.1, 0.2], [0.15, 0.1]]) @ n, atol=1e-10)
    with np.errstate(over='ignore', invalid='ignore'):
        res = AC.compute_cumulants(s.build(), k=2, max_ell=0, external_fields=[('dv', 1), ('dv', 2)],
                                   parameters=par, tau_grid=np.array([0.0, 1.0]), use_cache=False,
                                   parallel=False, verbose=False)
    assert np.allclose(res['mf_values']['vstar'], v, atol=1e-10)
    assert np.all(np.isfinite(np.asarray(res['C_tau_by_ell'][0])))


# ── D: rule errors and conventions ──────────────────────────────────────────────────────────

def test_gaussian_mean_equals_explicit_drift():
    a = (SDE('mean form').parameter('mu').parameter('c').field('x').drift('x', '-mu*x')
         .noise('eta', Gaussian(mean='c*x^2', var='c^2*x^4'), couples={'x': 1}))
    b = (SDE('explicit').parameter('mu').parameter('c').field('x').drift('x', '-mu*x + c*x^2')
         .noise('eta', Gaussian(var='c^2*x^4'), couples={'x': 1}))
    kw = dict(indexed=('x', 'xt'), scalars=('mu', 'c'))
    assert _same(_to_sr(a.action_text(), **kw), _to_sr(b.action_text(), **kw))
    for g, r in zip(_residuals(a.equations(), **kw), _residuals(b.equations(), **kw)):
        assert _same(g, r)
    assert any('moved into the drift' in n for n in a.notes)


def test_uniform_jump_action_is_a_polynomial():
    s = (SDE('uniform').parameter('mu').parameter('lam').parameter('h').field('x').drift('x', '-mu*x')
         .noise('eta', CompoundPoisson('lam', Uniform(0, 'h')), couples={'x': 1}).series_order(4))
    assert 'exp' not in s.action_text() and 'xt[i]^4' in s.action_text()
    assert s.equations()[0]['rhs'] == '(1/2)*h*lam'


def _base():
    return SDE('err').parameter('mu').parameter('lam').field('x')


@pytest.mark.parametrize('make, match', [
    (lambda: _base().set_action_text('xt*Dt*x'), 'extra_action'),
    (lambda: _base().drift('x', '-mu*x').noise('p', Poisson('lam'), couples={'x': 1})
     .interpretation('stratonovich').action_text(), 'stratonovich'),
    (lambda: _base().drift('x', '-mu*x').noise('p', Gaussian(var=1), expose='n')
     .interpretation('stratonovich').action_text(), 'exposed'),
    (lambda: _base().drift('x', '-mu*x + bogus').noise('W', Gaussian(var=1), couples={'x': 1}).action_text(),
     'undeclared'),
    (lambda: _base().drift('x', '-mu*x').noise('W', Gaussian(var='2*Q'), couples={'x': 1}).action_text(),
     'undeclared'),
    (lambda: _base().drift('x', '-mu*x').noise('W', Gaussian(var=1), couples={'x': 'Q'}).action_text(),
     'undeclared'),
    (lambda: _base().noise('W', Gaussian(var=1), couples={'x': 1}).action_text(), 'no equation'),
    (lambda: _base().drift('x', '-mu*x').noise('W', Gaussian(var=1), couples={'y': 1}).action_text(),
     'undeclared field'),
    (lambda: _base().equation('x', lhs='x', rhs='mu'), 'must contain the time derivative'),
    (lambda: _base().equation('x', lhs='Dt*x', rhs='Dt*mu'), 'Dt found in rhs'),
    (lambda: _base().drift('x', '-mu*x + sin(t)'), 'explicit time'),
    (lambda: _base().drift('x', '-mu*x').noise('W', Dirac(1), couples={'x': 1}), 'not processes'),
    (lambda: _base().drift('x', '-mu*x').noise('W', Gaussian(var=1)), 'couples='),
    (lambda: _base().drift('x', '-x^3').noise('W', Gaussian(var=1), couples={'x': 1}).equations(),
     'linear term'),
    (lambda: _base().parameter('theta'), 'reserved'),
    (lambda: SDE('two pops').population('A', size=1).population('B', size=1)
     .parameter('mu').field('x', population='A').field('y', population='B')
     .drift('x', '-mu*x').drift('y', '-mu*y')
     .noise('W', Gaussian(var=1), couples={'x': 1, 'y': 1}).action_text(), 'ONE'),
])
def test_rule_errors(make, match):
    with pytest.raises((SDEError, NoiseError), match=match):
        make()

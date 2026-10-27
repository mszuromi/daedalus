"""SDE front end with drifts that have no syntactic field-independent linear term (daedalus/api/sde.py ``_normalise_lhs``).

A rational or exponential drift (``-x/(1+eps*x^2)``, ``-x*exp(-eps*x^2)``), or a drift whose linear coefficient holds
a field, used to be refused: the mean-field equation of ``Dt*x = drift`` needs ``lhs|_{Dt=0} != 0``, and the front end
only knew how to move a syntactic linear term onto the lhs.  The mean-field lhs is now ``Dt*x + c*x`` and its rhs
``drift + c*x`` (c = 1), the residual being unchanged.

A  the cases build and run ``compute_cumulants(k=2, max_ell=1)``;
B  saddle, propagator, action and the k=2 tree + one-loop C(tau) equal the hand-written ``(Dt + c)*x`` model, for two c;
C  the analyzer's stability uses the true slope of the drift at the saddle, not c;
D  the tree is the linear-noise variance and the one loop moves it toward the exact stationary variance;
E  every SDE that built on main gives the same equations and notes as before;
F  the remaining refusal names the workaround form.
"""
import numpy as np
import pytest

pytest.importorskip('sage.all')
from scipy.integrate import quad                                              # noqa: E402

import daedalus.api.compute as AC                                                       # noqa: E402
from daedalus.api.analyze import analyze                                                # noqa: E402
from daedalus.api.noise import Gaussian                                                 # noqa: E402
from daedalus.api.sde import SDE, SDEError                                              # noqa: E402

TAU = np.array([0.0, 1.0])


# ── models: ``c`` None = the SDE front end (drift); a number = the hand-written (Dt + c)*x workaround ─────────────

def _finish(s, noise_fields):
    for name, var in noise_fields:
        s.noise(name, Gaussian(var=var), couples={name[2:]: 1})
    return s.stability_analysis(True)


def _put(s, field, drift, c):
    if c is None:
        return s.drift(field, drift)
    return s.equation(field, lhs=f'(Dt+{c})*{field}', rhs=f'({drift}) + {c}*{field}')


def rational(c=None):
    s = SDE('rational').parameter('eps', default=0.05, domain='positive').field('x')
    return _finish(_put(s, 'x', '-x/(1+eps*x^2)', c), [('xix', '2')])


def exponential(c=None):
    s = SDE('exponential').parameter('eps', default=0.05, domain='positive').field('x')
    return _finish(_put(s, 'x', '-x*exp(-eps*x^2)', c), [('xix', '2')])


def shifted(c=None):                      # saddle at m != 0
    s = (SDE('shifted').parameter('eps', default=0.05, domain='positive').parameter('m', default=0.7)
         .field('x'))
    return _finish(_put(s, 'x', '-(x-m)/(1+eps*(x-m)^2)', c), [('xix', '2')])


def two_field(c=None):                    # one rational drift, one linear, coupled
    s = (SDE('two-field').parameter('eps', default=0.05, domain='positive')
         .parameter('g', default=0.4).field('x').field('y'))
    _put(s, 'x', '-x/(1+eps*x^2) + g*y', c)
    s.drift('y', '-y - g*x')
    return _finish(s, [('xix', '2'), ('xiy', '1')])


def sgd_logistic(c=None):                 # gradient of the logistic loss, soft label y0, no L2 term
    s = (SDE('sgd logistic').parameter('x0', default=1.5, domain='positive')
         .parameter('y0', default=0.7, domain='positive').parameter('s', default=0.2, domain='positive')
         .field('w'))
    return _finish(_put(s, 'w', '-(1/(1+exp(-w*x0)) - y0)*x0', c), [('xiw', '2*s')])


def sgd_logistic_l2(c=None):              # the same with an L2 term: the linear term exists but is not alone
    s = (SDE('sgd logistic l2').parameter('x0', default=1.5, domain='positive')
         .parameter('y0', default=0.7, domain='positive').parameter('lam', default=0.1, domain='positive')
         .parameter('s', default=0.2, domain='positive').field('w'))
    return _finish(_put(s, 'w', '-(1/(1+exp(-w*x0)) - y0)*x0 - lam*w', c), [('xiw', '2*s')])


def stratonovich(c=None):                  # multiplicative noise, Ito drift correction added to the rhs
    s = SDE('strat').parameter('eps', default=0.05, domain='positive').field('x').interpretation('stratonovich')
    _put(s, 'x', '-x/(1+eps*x^2)', c)
    s.noise('xix', Gaussian(var='2'), couples={'x': '1 + x/5'})
    return s.stability_analysis(True)


CASES = {
    'rational': (rational, {'eps': 0.05}, 'dx'),
    'exponential': (exponential, {'eps': 0.05}, 'dx'),
    'shifted': (shifted, {'eps': 0.05, 'm': 0.7}, 'dx'),
    'two_field': (two_field, {'eps': 0.05, 'g': 0.4}, 'dx'),
    'stratonovich': (stratonovich, {'eps': 0.05}, 'dx'),
    'sgd_logistic': (sgd_logistic, {'x0': 1.5, 'y0': 0.7, 's': 0.2}, 'dw'),
    'sgd_logistic_l2': (sgd_logistic_l2, {'x0': 1.5, 'y0': 0.7, 'lam': 0.1, 's': 0.2}, 'dw'),
}


def _run(sde, par, ext, ell=1):
    return AC.compute_cumulants(sde.build(), k=2, max_ell=ell, external_fields=[(ext, 1)] * 2, parameters=par,
                                tau_grid=TAU, use_cache=True, parallel=False, verbose=False)


def _same(a, b):
    return all(np.array_equal(a['C_tau_by_ell'][l], b['C_tau_by_ell'][l]) for l in a['C_tau_by_ell'])


# ── A + B ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('case', list(CASES))
def test_builds_and_matches_the_hand_written_workaround_for_two_c(case):
    make, par, ext = CASES[case]
    gen, hand1, hand2 = make(), make(1), make('5/2')
    _run(gen, par, ext)                                                          # warm caches, discarded
    r_gen, r1, r2 = _run(gen, par, ext), _run(hand1, par, ext), _run(hand2, par, ext)
    assert sorted(r_gen['C_tau_by_ell']) == [0, 1]
    notes = ' '.join(gen.notes)
    if case == 'sgd_logistic_l2':
        # the syntactic linear term -lam*w exists: the old path (c = lam) runs, not the c = 1 fallback, so the
        # (Dt + 1)*w model differs in solver rounding only
        assert 'moved to the lhs' in notes and 'added w[i] to both sides' not in notes
        assert np.asarray(r1['mf_values']['wstar']) == pytest.approx(np.asarray(r_gen['mf_values']['wstar']),
                                                                      rel=1e-12)
    else:
        assert 'added' in notes and ('added x[i] to both sides' in notes or 'added w[i] to both sides' in notes)
        # the saddle and the tree + one-loop C(tau) are bit-identical to the (Dt + 1)*x model
        assert r_gen['mf_values'] == r1['mf_values']
        assert _same(r_gen, r1)
        assert str(r_gen['propagator']['G_ft']) == str(r1['propagator']['G_ft'])
    # ... and independent of c: a different c moves nothing beyond 1e-12 relative
    for key, v in r_gen['mf_values'].items():
        assert np.asarray(r2['mf_values'][key]) == pytest.approx(np.asarray(v), rel=1e-12, abs=1e-14)
    for l in (0, 1):
        a, b = np.asarray(r_gen['C_tau_by_ell'][l]), np.asarray(r2['C_tau_by_ell'][l])
        assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))
    # propagator: G_ft is the same matrix of rational functions
    assert str(r_gen['propagator']['G_ft']) == str(r2['propagator']['G_ft'])


@pytest.mark.parametrize('case', list(CASES))
def test_the_action_is_the_workaround_action_up_to_the_cancelled_term(case):
    """The fix leaves the action alone (as the linear-term path always did): it is the workaround's action with
    the added c*x cancelled, equal to it as an expression, not as text."""
    from tests.test_sde_frontend import _to_sr, _same as same_expr
    make, par, _ = CASES[case]
    fields = ('x', 'y', 'w')
    kw = dict(indexed=fields + tuple(f + 't' for f in fields), scalars=tuple(par) + ('s', 'lam'))
    gen, hand = make().action_text(), make('5/2').action_text()
    assert gen != hand
    assert same_expr(_to_sr(gen, **kw), _to_sr(hand, **kw))


# ── C: stability uses the true slope of the drift ───────────────────────────────────────────

def _codes(sde):
    return {f.code for f in analyze(sde.build()).findings}


def test_unstable_saddle_is_flagged_from_the_true_slope():
    s = SDE('anti-restoring').field('x')
    s.drift('x', 'x/(1+x^2)').noise('xi', Gaussian(var='2'), couples={'x': 1}).stability_analysis(True)
    assert 'G2_UNSTABLE_SADDLE' in _codes(s)                                      # true slope +1 at x* = 0


def test_restoring_rational_drift_is_not_flagged():
    assert 'G2_UNSTABLE_SADDLE' not in _codes(rational())
    assert 'G2_UNSTABLE_SADDLE' not in _codes(sgd_logistic())


def test_stability_is_independent_of_c():
    from daedalus.api._mean_field_dae import linear_stability, solve_mean_field_dae
    par = {'eps': 0.05}
    eig = []
    for c in (None, 1, 5.0):
        m = rational(c).build()
        root = solve_mean_field_dae(m, par, verbose=False)['mf_all_roots'][0]['values']
        eig.append(complex(linear_stability(m, par, root)['eigenvalues_finite'][0]))
    assert eig == pytest.approx([-1 + 0j] * 3, abs=1e-12)


def test_pure_cubic_builds_and_the_analyzer_flags_the_massless_saddle():
    """-x^3 was refused at build time ('linear term'); it builds now and the analyzer says what is wrong with it."""
    s = SDE('q').field('x').drift('x', '-x^3').noise('W', Gaussian(var=1), couples={'x': 1})
    s.stability_analysis(False)
    s.equations()
    assert 'added x[i] to both sides' in ' '.join(s.notes)
    assert 'G2_MARGINAL_MODE' in _codes(s)                                      # true slope 0 at x* = 0


# ── D: oracle, the exact stationary density ─────────────────────────────────────────────────────

def _exact_variance(logp):
    """Var of p(x) ~ exp(logp(x)) on the real line (symmetric, so the mean is 0)."""
    z = quad(lambda x: np.exp(logp(x)), -np.inf, np.inf, epsabs=0, epsrel=1e-13)[0]
    return quad(lambda x: x * x * np.exp(logp(x)), -np.inf, np.inf, epsabs=0, epsrel=1e-13)[0] / z


@pytest.mark.parametrize('case, eps, logp', [
    ('rational', 0.02, lambda e: (lambda x: -np.log1p(e * x * x) / (2 * e))),
    ('exponential', 0.02, lambda e: (lambda x: (np.exp(-e * x * x) - 1.0) / (2 * e))),
])
def test_one_loop_moves_the_variance_toward_the_exact_value(case, eps, logp):
    make, par, ext = CASES[case]
    par = dict(par, eps=eps)
    res = _run(make(), par, ext)
    tree = complex(res['C_tau_by_ell'][0][0]).real
    loop = complex(res['C_tau_by_ell'][1][0]).real
    exact = _exact_variance(logp(eps))
    assert tree == pytest.approx(1.0, rel=1e-12)                                   # the linear-noise value (Ito tau = 0)
    assert loop == pytest.approx(3 * eps, rel=1e-12)                               # first-order coefficient
    assert abs(exact - (tree + loop)) < 0.2 * abs(exact - tree)                   # one loop closes > 80 % of the gap


# ── E: no regressions, models that built on main give the same equations and notes ──────────────────────

MAIN_30BDA90 = {
    '-mu*x - eps*x^3': ('Dt*x[i] + mu*x[i]', '-eps*x[i]^3'),
    '-(1 + x^2)*x': ('Dt*x[i] + x[i]', '-x[i]^3'),
    '-mu*(x-m)': ('Dt*x[i] + mu*x[i]', 'm*mu'),
    '-x*(1+x)': ('Dt*x[i] + x[i]', '-x[i]^2'),
    '-mu*x + eps*x^2': ('Dt*x[i] + mu*x[i]', 'eps*x[i]^2'),
    '-(x-m)^3 - x': ('3*m^2*x[i] + Dt*x[i] + x[i]', 'm^3 + 3*m*x[i]^2 - x[i]^3'),
    '-mu*x': ('Dt*x[i] + mu*x[i]', '0'),
}


@pytest.mark.parametrize('drift', list(MAIN_30BDA90))
def test_models_that_built_before_are_unchanged(drift):
    s = (SDE('t').parameter('eps', default=0.1, domain='positive').parameter('mu', default=1.0, domain='positive')
         .parameter('m', default=0.5).field('x').drift('x', drift)
         .noise('xi', Gaussian(var='2'), couples={'x': 1}))
    (eq,) = s.equations()
    assert (eq['lhs'], eq['rhs']) == MAIN_30BDA90[drift]
    assert not any('added' in n for n in s.notes)


def test_a_field_dependent_linear_coefficient_builds():
    """x*y has a field in its linear coefficient: refused on main, c*x is added now."""
    s = (SDE('t').field('x').field('y').drift('x', '-x*(1+y^2)').drift('y', '-y')
         .noise('a', Gaussian(var='2'), couples={'x': 1}).noise('b', Gaussian(var='2'), couples={'y': 1}))
    eqs = s.equations()
    assert eqs[0]['lhs'] == 'Dt*x[i] + x[i]' and eqs[0]['rhs'].endswith('+ x[i]')
    assert any('added x[i] to both sides' in n for n in s.notes)
    s.build()


# ── F: the remaining refusals name the workaround ────────────────────────────────────────────

@pytest.mark.parametrize('drift', ['mu', '0', '-y'])
def test_a_drift_that_does_not_depend_on_the_field_is_still_refused(drift):
    s = (SDE('t').parameter('mu', default=1.0).field('x').field('y').drift('x', drift).drift('y', '-y')
         .noise('a', Gaussian(var='2'), couples={'x': 1, 'y': 1}))
    with pytest.raises(SDEError, match=r"does not depend on x.*\(Dt \+ 1\)\*x"):
        s.equations()


def test_an_rhs_that_cannot_be_split_is_still_refused_with_the_workaround():
    s = (SDE('t').population('E', size=2).field('r', population='E').parameter('W', indexed='matrix')
         .drift('r', '-r + sum(W[i,j]*tanh(r[j]) for j in E)').noise('a', Gaussian(var='2'), couples={'r': 1}))
    with pytest.raises(SDEError, match=r"cannot be split symbolically.*\(Dt \+ 1\)\*r"):
        s.equations()


def test_unreadable_lhs_names_the_workaround(monkeypatch):
    s = SDE('t').field('x').equation('x', lhs='Dt*x', rhs='-x').noise('xi', Gaussian(var='2'), couples={'x': 1})
    import daedalus.api.sde as mod
    real = mod.parse_expr

    def boom(text, what=''):
        if what.startswith('lhs of'):
            raise mod.NoiseError('unparseable')
        return real(text, what=what)
    monkeypatch.setattr(mod, 'parse_expr', boom)
    with pytest.raises(SDEError, match=r"cannot be read symbolically.*\(Dt \+ 1\)\*x"):
        s.equations()

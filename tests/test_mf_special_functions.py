"""Item 69: the mean-field solve evaluates every function the expression parser accepts in a drift.

The saddle solve compiles the equation text once and evaluates it in a numpy namespace; that namespace lacked
``erf``, ``arctan`` and ``heaviside`` (and the Sage namespace of the linear stability lacked them too), so a drift
using one stopped with ``NameError: name 'erf' is not defined`` (Sage 10.8 and 10.9).  Both tables now carry every
function ``daedalus.api.noise._MATH_FUNCS`` accepts, and a call of any other name is refused up front with the list of
supported functions.  (The vertex expansion evaluates the text in ``sage.all`` and was never affected.)

A  one-field drift per parser function: the saddle equals an independent ``scipy.optimize.brentq`` root of
   ``x - g(x)`` (``g`` written with numpy / scipy, not through the package), and the linearization eigenvalue is
   ``-1 + g'(x*)`` (central difference of that same ``g``);
B  heaviside: Theta(0) = 0 (the mean-field table, Phase J's Ito value, the point-process simulator), a saddle on the
   jump is refused with a clear message, the step off the saddle has slope 0;
C  a call of an unsupported name is refused before any evaluation (``UnsupportedMeanFieldFunctionError``), by both
   the saddle solve and the linear stability;
D  a k=2 tree run (``compute_cumulants``, ``max_ell=0``) finishes for erf, arctan, heaviside and the composition
   ``erf(arctan(x))`` and equals the linear-noise result ``D / (2 lambda) exp(-lambda |tau|)``.
"""
import math

import numpy as np
import pytest

pytest.importorskip('sage.all')
from scipy.optimize import brentq                                             # noqa: E402
from scipy.special import erf                                                 # noqa: E402

import daedalus.api.compute as AC                                                       # noqa: E402
from daedalus.api import _mean_field_dae as MF                                         # noqa: E402
from daedalus.api.noise import Gaussian                                                # noqa: E402
from daedalus.api.sde import SDE                                                       # noqa: E402

A = 0.5                      # the parameter ``a`` of every drift
VAR = 0.1                    # noise strength of the tree runs


def _heav(x):
    return np.heaviside(x, 0.0)


# name -> (drift text g(x), the same g in numpy / scipy, bracket for brentq of x - g(x)).  Each x - g(x) has ONE
# root in its bracket, and the saddle solve must find that one.
CASES = {
    'erf': ('a + erf(x)/3', lambda x: A + erf(x) / 3, (-5, 5)),
    'arctan': ('a + arctan(x)', lambda x: A + np.arctan(x), (-1, 10)),
    'heaviside_below': ('0.3 + 0.2*heaviside(x - 0.6)', lambda x: 0.3 + 0.2 * _heav(x - 0.6), (-2, 2)),
    'heaviside_above': ('0.3 + 0.2*heaviside(x - 0.1)', lambda x: 0.3 + 0.2 * _heav(x - 0.1), (-2, 2)),
    'erf_of_arctan': ('a + erf(arctan(x))/2', lambda x: A + erf(np.arctan(x)) / 2, (-5, 5)),
    'abs': ('0.4*abs(x) + a', lambda x: 0.4 * np.abs(x) + A, (-5, 5)),
    'tanh': ('a + tanh(x)', lambda x: A + np.tanh(x), (-1, 10)),
    'sinh': ('a - sinh(x)/4', lambda x: A - np.sinh(x) / 4, (-5, 5)),
    'cosh': ('0.2*cosh(x) + 0.1', lambda x: 0.2 * np.cosh(x) + 0.1, (-5, 1)),
    'sin': ('a + sin(x)/2', lambda x: A + np.sin(x) / 2, (-5, 5)),
    'cos': ('a + cos(x)/2', lambda x: A + np.cos(x) / 2, (-5, 5)),
    'tan': ('a + tan(x)/4', lambda x: A + np.tan(x) / 4, (-1.2, 1.2)),
    'exp': ('a*exp(-x)', lambda x: A * np.exp(-x), (-5, 5)),
    'log': ('a*log(1 + x*x) + 0.1', lambda x: A * np.log(1 + x * x) + 0.1, (-5, 5)),
    'sqrt': ('a + sqrt(1 + x*x)/4', lambda x: A + np.sqrt(1 + x * x) / 4, (-5, 5)),
}


def _model(drift, name='msf'):
    s = SDE(name).parameter('a', default=A).field('x')
    s.drift('x', f'-x + {drift}')
    return s.noise('xi', Gaussian(var=str(VAR)), couples={'x': 1}).stability_analysis(True)


def _slope(g, x, h=1e-6):
    return (g(x + h) - g(x - h)) / (2 * h)


# ── A: saddle and linearization against an independent root / derivative ─────────────────────────────────

@pytest.mark.parametrize('case', list(CASES))
def test_saddle_and_stability_match_an_independent_root(case):
    text, g, bracket = CASES[case]
    model = _model(text).build()
    r = MF.solve_mean_field_dae(model, {'a': A}, verbose=False)
    ref = brentq(lambda x: x - g(x), *bracket, xtol=1e-14, rtol=1e-14)
    got = r['mf_values']['xstar'][0]
    assert got == pytest.approx(ref, rel=1e-9, abs=1e-12)
    ls = MF.linear_stability(model, {'a': A}, r['mf_values'], verbose=False)
    assert ls['stable']
    assert ls['eigenvalues_finite'][0].real == pytest.approx(-1 + _slope(g, ref), abs=1e-6)


# ── B: Theta(0) ─────────────────────────────────────────────────────────────────────────────────────────────

def test_theta_at_zero_is_zero_everywhere():
    """Theta(0) = 0: the Ito value of Phase J and of the equal-time k = 2 limit tau -> 0^-.  The mean-field table,
    the point-process simulator and numpy-with-0 agree; Sage leaves ``heaviside(0)`` unevaluated (it fixes no
    other value)."""
    from daedalus.api import pp_simulate
    assert MF._MATH_NS['heaviside'](0.0) == 0.0
    assert MF._MATH_NS['heaviside'](np.array([-1.0, 0.0, 1e-300, 2.0])).tolist() == [0.0, 0.0, 1.0, 1.0]
    assert pp_simulate._NP['heaviside'](0.0) == 0.0
    from sage.all import heaviside
    assert heaviside(0).operator() is not None                                   # unevaluated, not 1/2 or 1


def test_a_saddle_on_the_jump_is_refused_with_a_clear_message():
    """``x - 1 - (x - 1)*Theta(x - 1)`` is zero at the jump whatever Theta(0) is, so the root x = 1 is found; the
    expansion about it needs the slope of the step (a Dirac delta at 0).  It stopped deep in the propagator with
    ``TypeError: unable to simplify to complex approximation``."""
    model = _model('1 + heaviside(x - 1)*(x - 1)', 'tie').build()
    with pytest.raises(MF.UnsupportedMeanFieldFunctionError, match=r'exactly on the jump of heaviside\(x\[i\] - 1\)'):
        MF.solve_mean_field_dae(model, {'a': A}, verbose=False)


def test_the_step_off_the_saddle_has_slope_zero():
    model = _model(CASES['heaviside_below'][0]).build()
    r = MF.solve_mean_field_dae(model, {'a': A}, verbose=False)
    ls = MF.linear_stability(model, {'a': A}, r['mf_values'], verbose=False)
    assert ls['eigenvalues_finite'][0].real == pytest.approx(-1.0, abs=1e-12)


# ── C: an unsupported call is refused before any evaluation ───────────────────────────────────────────────────

@pytest.mark.parametrize('call', ['erfc(x)', 'arcsinh(x)', 'foo(x)'])
def test_an_unsupported_function_is_refused_up_front(call):
    model = _model('a + erf(x)/3').build()
    model['equations'][0]['rhs_text'] = f'a + {call}'
    with pytest.raises(MF.UnsupportedMeanFieldFunctionError) as e:
        MF.solve_mean_field_dae(model, {'a': A}, verbose=False)
    assert not isinstance(e.value, NameError)
    msg = str(e.value)
    assert call.split('(')[0] in msg and 'erf' in msg and 'arctan' in msg and 'heaviside' in msg
    with pytest.raises(MF.UnsupportedMeanFieldFunctionError):
        MF.linear_stability(model, {'a': A}, {'xstar': [1.0]}, verbose=False)


@pytest.mark.parametrize('rhs, g', [
    ('a + erf(x)/3', lambda x: A + erf(x) / 3),
    ('a + arctan(x)', lambda x: A + np.arctan(x)),
    ('0.3 + 0.2*heaviside(x - 0.1)', lambda x: 0.3 + 0.2 * _heav(x - 0.1)),
    ('a + sign(x)/4', lambda x: A + np.sign(x) / 4),
    ('a + abs(x)/4', lambda x: A + np.abs(x) / 4),
])
def test_scalar_mode_bare_field_argument(rhs, g):
    """A population-less model writes the bare field (``erf(x)``): the linearization wraps it in a size-1
    field wrapper that Sage's functions refuse unless unwrapped (``erf(x)`` is not ``exp(-x)``, an expression)."""
    from daedalus.api.model import TemporalModelBuilder
    m = (TemporalModelBuilder('bare').physical_field('x').parameter('a', default=A, domain='positive')
         .stability_analysis(True).set_action_text('xt*((Dt + 1)*x)')
         .mf_residual(lhs='(Dt+1)*x', rhs=rhs).build())
    r = MF.solve_mean_field_dae(m, {'a': A}, verbose=False)
    ref = brentq(lambda x: x - g(x), -5, 5, xtol=1e-14, rtol=1e-14)
    assert r['mf_values']['xstar'][0] == pytest.approx(ref, rel=1e-9)
    ls = MF.linear_stability(m, {'a': A}, r['mf_values'], verbose=False)
    assert ls['eigenvalues_finite'][0].real == pytest.approx(-1 + _slope(g, ref), abs=1e-6)


def test_the_constant_e_and_sign_in_an_equation_text():
    """A bare ``e`` (Euler's number, as the parser reads it) and ``sign`` (Sage ``sgn``) evaluate in both tables."""
    model = _model('a + erf(x)/3').build()
    model['equations'][0]['rhs_text'] = 'a + e**(-x)/4 + 0.1*sign(x)'
    g = lambda x: A + np.exp(-x) / 4 + 0.1 * np.sign(x)                     # noqa: E731
    r = MF.solve_mean_field_dae(model, {'a': A}, verbose=False)
    ref = brentq(lambda x: x - g(x), 0.01, 5, xtol=1e-14, rtol=1e-14)
    assert r['mf_values']['xstar'][0] == pytest.approx(ref, rel=1e-9)
    ls = MF.linear_stability(model, {'a': A}, r['mf_values'], verbose=False)
    assert ls['eigenvalues_finite'][0].real == pytest.approx(-1 + _slope(g, ref), abs=1e-6)


def test_both_tables_carry_the_same_functions():
    """The numeric table and the Sage table of the linearization are the same set of names."""
    from daedalus.api.noise import _MATH_FUNCS
    parser = set(_MATH_FUNCS)                      # what the parser accepts in a drift
    assert parser <= set(MF._MATH_NS), parser - set(MF._MATH_NS)
    assert {'gamma', 'factorial'} <= set(MF._SPECIAL_NS)


# ── D: a k = 2 tree run ──────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('case', ['erf', 'arctan', 'heaviside_below', 'heaviside_above', 'erf_of_arctan'])
def test_k2_tree_runs_and_is_the_linear_noise_result(case):
    text, g, bracket = CASES[case]
    ref = brentq(lambda x: x - g(x), *bracket, xtol=1e-14, rtol=1e-14)
    lam = 1 - _slope(g, ref)
    tau = np.array([0.0, 0.7, 1.5])
    res = AC.compute_cumulants(_model(text, f'tree_{case}').build(), k=2, max_ell=0, external_fields=[('dx', 1)] * 2,
                               parameters={'a': A}, tau_grid=tau, use_cache=False, parallel=False, verbose=False)
    assert res['mf_values']['xstar'][0] == pytest.approx(ref, rel=1e-9)
    c = np.asarray(res['C_tau_by_ell'][0]).real
    want = VAR / (2 * lam) * np.exp(-lam * np.abs(tau))
    assert c == pytest.approx(want, rel=1e-6)


def test_coupled_composed_case_against_fsolve_and_lyapunov():
    """Two coupled fields, ``erf(arctan(y))`` and ``arctan(erf(x))`` composed and an active step: the saddle is
    scipy's ``fsolve`` root and the k = 2 tree is the linear-noise (Lyapunov) covariance of the same drift."""
    from scipy.linalg import expm, solve_continuous_lyapunov
    from scipy.optimize import fsolve
    a, b, dx, dy = 0.4, 0.8, 0.10, 0.06
    s = SDE('adv').parameter('a', default=a).parameter('b', default=b).field('x').field('y')
    s.drift('x', '-x + a + erf(arctan(y))')
    s.drift('y', '-y + b*arctan(erf(x)) + 0.3*heaviside(x - 0.9)')
    s.noise('xix', Gaussian(var=str(dx)), couples={'x': 1}).noise('xiy', Gaussian(var=str(dy)), couples={'y': 1})
    model = s.stability_analysis(True).build()
    f = lambda v: [-v[0] + a + erf(np.arctan(v[1])),                              # noqa: E731
                   -v[1] + b * np.arctan(erf(v[0])) + 0.3 * _heav(v[0] - 0.9)]
    v = fsolve(f, [0.5, 0.5], xtol=1e-14)
    assert v[0] > 0.9                                                             # the step is active
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1e-6
        J[:, j] = (np.array(f(v + e)) - np.array(f(v - e))) / 2e-6
    P = solve_continuous_lyapunov(J, -np.diag([dx, dy]))
    tau = np.array([0.0, 0.8])
    kw = dict(k=2, max_ell=0, parameters={'a': a, 'b': b}, tau_grid=tau, use_cache=False, parallel=False,
              verbose=False)
    xx = AC.compute_cumulants(model, external_fields=[('dx', 1), ('dx', 1)], **kw)
    xy = AC.compute_cumulants(model, external_fields=[('dx', 1), ('dy', 1)], **kw)
    assert xx['mf_values']['xstar'][0] == pytest.approx(v[0], rel=1e-9)
    assert xx['mf_values']['ystar'][0] == pytest.approx(v[1], rel=1e-9)
    cxx = np.asarray(xx['C_tau_by_ell'][0]).real
    assert cxx[0] == pytest.approx(P[0, 0], rel=1e-7)
    assert cxx[1] == pytest.approx((expm(J * tau[1]) @ P)[0, 0], rel=1e-7)
    assert np.asarray(xy['C_tau_by_ell'][0]).real[0] == pytest.approx(P[0, 1], rel=1e-7)

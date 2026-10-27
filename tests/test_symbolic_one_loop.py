"""
tests/test_symbolic_one_loop.py
===============================
Stage C3 (``daedalus/api/symbolic_loop.py``): equal-time cumulants and kernel
pairings at loop level as RATIONAL functions of the parameters, from the
state-space stationary sweep over ``F = QQ(params)``.

* OU quartic: one and two loops equal the exact Boltzmann O(eps), O(eps^2)
  coefficients symbolically; the saddle kept symbolic or given in closed
  form;
* colour-noise OU at the exact double pole (no distinct-pole assumption):
  the closed form, and the pipeline at a second parameter point;
* the recharge oscillator with red multiplicative noise (a public inline
  twin of the Levine-Jin type model), k = 3 equal-time one loop == the
  pipeline at two parameter points;
* kernel pairings == S4's exact ``Cumulant.integrate_against``; the
  rational triples == ``C_tau`` and their τ = 0 limit == the equal-time
  value; shot noise (δ edges); refusals by name; exports.

Inline public models or tracked ones only.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip('sage.all')

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))

import daedalus as dd                                                # noqa: E402
from sage.all import QQ, SR, factorial, var                          # noqa: E402
from daedalus.api import compute_cumulants                                    # noqa: E402
from daedalus.api import symbolic_loop as SL                                  # noqa: E402
from daedalus.api.cumulant import Cumulant                                    # noqa: E402
from daedalus.api.noise import Poisson                                        # noqa: E402
from daedalus.api.sde import SDE                                              # noqa: E402
from daedalus.api.symbolic_out import PairingKernel as Kernel                                  # noqa: E402
import daedalus.engine.integration.time_domain.statespace as SS               # noqa: E402


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SSM = _load('test_statespace_integrator.py', 'ss_models_c3')
ZM = _load('test_zero_mode_pruning.py', 'zm_models_c3')


@pytest.fixture(autouse=True)
def _default_engine(monkeypatch):
    monkeypatch.delenv(SS.ENGINE_ENV, raising=False)


def _cc(model, k, ell, ext, params=None, taus=(0.0,)):
    return compute_cumulants(model, k=k, max_ell=ell, external_fields=ext,
                             parameters=params, tau_grid=list(taus),
                             parallel=False, verbose=False, use_cache=False)


def _sr_equal(a, b):
    return bool((SR(a) - SR(b)).simplify_rational() == 0)


@pytest.fixture(scope='module')
def ouq_model():
    return dd.load_model('ou_quartic')[0]


# ═════════════════════════════════════════════════════════════════════════════
# A. OU quartic vs the exact Boltzmann series
# ═════════════════════════════════════════════════════════════════════════════

def _boltzmann_coefficients(order):
    """<x^2> of p(x) ∝ exp(-(mu x^2/2 + eps x^4/4)/D) to O(eps^order),
    exactly (Gaussian moments)."""
    mu, eps, D = var('mu eps D')

    def m(n):                           # <x^n>_0, variance D/mu
        if n % 2:
            return SR(0)
        return QQ(factorial(n)) / QQ(2 ** (n // 2) * factorial(n // 2)) \
            * (D / mu) ** (n // 2)
    g = -eps / (4 * D)
    J = order + 2
    N = sum(g ** j / factorial(j) * m(2 + 4 * j) for j in range(J))
    Z = sum(g ** j / factorial(j) * m(4 * j) for j in range(J))
    ser = (N / Z).series(eps, order + 1).truncate()
    return [ser.coefficient(eps, j) * eps ** j for j in range(order + 1)]


def test_ou_quartic_one_and_two_loops_equal_boltzmann(ouq_model):
    r = SL.symbolic_cumulant(ouq_model, 2, 2, external_fields=[('dx', 1)] * 2)
    assert set(r.names) == {'mu', 'eps', 'D'}
    ref = _boltzmann_coefficients(2)
    for ell in (0, 1, 2):
        assert _sr_equal(r.expression(ell), ref[ell]), (ell, r.expression(ell))
    assert str(r.by_ell[1]) == '(-3*eps*D^2)/mu^3'
    assert max(r.check().values()) <= 1e-12
    assert r.kind == 'equal_time' and r.stats['max_block'] == 1


def test_ou_quartic_saddle_modes(ouq_model):
    ext = [('dx', 1)] * 2
    res = _cc(ouq_model, 2, 1, ext)
    num = SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext,
                               result=res)
    sym = SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext,
                               result=res, saddle='symbol')
    given = SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext,
                                 result=res, saddle={'xstar1': 0})
    assert 'xstar1' in sym.names and 'xstar1' not in num.names
    # the symbolic saddle carries the propagator's 3 eps x*^2 shift
    assert _sr_equal(sym.expression(0), SR('D/(mu + 3*eps*xstar1^2)'))
    for ell in (0, 1):
        assert _sr_equal(sym.expression(ell).subs(xstar1=0),
                         num.expression(ell))
        assert given.by_ell[ell] == num.by_ell[ell]
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext,
                             result=res, saddle={'xstar1': 'mu'})
    assert e.value.reason == 'saddle_mismatch'
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext,
                             result=res, saddle='solve')
    assert e.value.reason == 'saddle_unsolved'


def test_numeric_parameters_and_exports(ouq_model):
    ext = [('dx', 1)] * 2
    r = SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext,
                             symbols=['eps'], parameters={'mu': 0.5, 'D': 0.1})
    assert r.names == ('eps',)
    # mu = 1/2, D = 1/10 enter as the simplest rationals of their doubles:
    # -3 eps D^2 / mu^3 = -6/25 eps
    assert r.by_ell[1] == r.field.F(QQ(-6) / 25) * r.field.F.gen(0)
    assert any('numeric parameters' in n for n in r.notes)
    f = r.callable(1)
    assert abs(f(eps=0.3) - (-6 / 25 * 0.3)) <= 1e-15
    assert 'eps' in r.latex(1)
    assert str(r.sympy(1)) == '-6*eps/25'
    js = json.loads(r.to_json(1))
    # integer coefficients without a common factor
    assert js == {'params': ['eps'], 'numerator': '-6*eps',
                  'denominator': '25'}
    R = r.field.R
    assert R(js['numerator']) / R(js['denominator']) == r.by_ell[1]
    assert abs(r.evaluate({'eps': 0.3}, ell=1) - (-6 / 25 * 0.3)) <= 1e-15
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext,
                             symbols=['nope'])
    assert e.value.reason == 'unbound_symbol'


# ═════════════════════════════════════════════════════════════════════════════
# A. colour-noise OU at the exact double pole
# ═════════════════════════════════════════════════════════════════════════════

def test_colour_noise_double_pole_closed_form_and_second_point():
    """x' = -mu x + e - eps x^3, e' = -e/tc + noise: at mu = 1/tc the
    propagator has a double pole; the rational result needs no
    distinct-pole assumption and holds at every parameter point."""
    m = SSM.ou_colour(1.0)
    ext = [('x', 1)] * 2
    r = SL.symbolic_cumulant(m, 2, 1, external_fields=ext)
    assert _sr_equal(r.expression(0), SR('2*D/(mu*(1 + mu*tc))'))
    assert _sr_equal(r.expression(1), SR(
        '-12*eps*D^2*(1 + 2*mu*tc)/(mu^3*(1 + mu*tc)^3)'))
    assert max(r.check().values()) <= 1e-12          # mu = tc = 1: double pole
    # another point (distinct poles): the pipeline agrees with the formula
    pt = {'mu': 0.7, 'eps': 0.05, 'D': 0.3, 'tc': 2.5}
    res2 = _cc(m, 2, 1, ext, params=pt)
    for ell in (0, 1):
        ref = complex(res2['phase_j_by_ell'][ell]['total_C'](0.0, 0.0))
        assert abs(r.evaluate(pt, ell=ell) - ref) <= 1e-12 * abs(ref), ell


# ═════════════════════════════════════════════════════════════════════════════
# A. k = 3 equal-time one loop: recharge oscillator, red multiplicative noise
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def recharge():
    m = ZM._oscillator()
    r = SL.symbolic_cumulant(m, 3, 1, external_fields=[('T', 1)] * 3)
    return m, r


def test_recharge_oscillator_k3_one_loop_equals_pipeline(recharge):
    m, r = recharge
    assert set(r.names) == set(ZM.PARS)
    assert max(r.check().values()) <= 1e-12
    assert r.stats['by_ell'][1]['subsets'] >= 10
    # a second parameter point: the pipeline vs the rational function
    pt = dict(lam=0.3, om=0.21, sig=0.4, r=1.1, B=0.7)
    res2 = _cc(m, 3, 1, [('T', 1)] * 3, params=pt)
    for ell in (0, 1):
        ref = complex(res2['phase_j_by_ell'][ell]['total_C'](0.0, 0.0, 0.0))
        assert abs(r.evaluate(pt, ell=ell) - ref) <= 1e-12 * abs(ref), ell
    # the tree third cumulant is odd in the multiplicative coupling B
    assert r.numerator(0).degree(r.field.R.gen(r.names.index('B'))) >= 1


def test_k3_symbol_subset_is_the_full_result_restricted(recharge):
    m, full = recharge
    part = SL.symbolic_cumulant(m, 3, 1, external_fields=[('T', 1)] * 3,
                                symbols=['B', 'sig'], ells=[1])
    pt = {n: full.reference[n] for n in full.names}
    pt.update({'B': 0.9, 'sig': 0.15})
    assert abs(part.evaluate({'B': 0.9, 'sig': 0.15}, ell=1)
               - full.evaluate(pt, ell=1)) <= 1e-13 * abs(
                   full.evaluate(pt, ell=1))
    assert part.size(1)['terms'] < full.size(1)['terms']


# ═════════════════════════════════════════════════════════════════════════════
# A. kernel pairings == S4's exact integrate_against
# ═════════════════════════════════════════════════════════════════════════════

def test_pairings_equal_s4_integrate_against(ouq_model):
    m = SSM.ou_colour(1.0)
    ext = [('x', 1)] * 2
    res = _cc(m, 2, 1, ext, taus=np.linspace(-4, 4, 9))
    c = Cumulant.from_result(res)
    kernels = [Kernel.exponential(1, -0.5),
               Kernel.two_sided_exponential(0.7, 2.0, 1.3, 0.5),
               Kernel.exponential(2, -1.5, power=2)
               + Kernel.exponential(1, 0.25, side='negative', power=1)]
    for K in kernels:
        r = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res,
                                 kernel=K)
        assert r.kind == 'pairing'
        for ell in (0, 1):
            ref = complex(c.integrate_against(K, ell=ell, method='exact',
                                              contact='exclude').smooth)
            assert abs(r.evaluate(ell=ell) - ref) <= 1e-12 * abs(ref), ell
    # symbolic kernel parameters join the field
    K = Kernel.exponential(SR.var('a'), -SR.var('lam_k'))
    r = SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=[('dx', 1)] * 2,
                             kernel=K, kernel_values={'a': 1.0, 'lam_k': 0.5})
    assert {'a', 'lam_k'} <= set(r.names) and r.pipeline
    assert _sr_equal(r.expression(0), SR('D*a/(mu*(mu + lam_k))'))
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=[('dx', 1)] * 2,
                             kernel=K)
    assert e.value.reason == 'unbound_symbol'


def test_rational_triples_equal_c_tau_and_equal_time():
    m = SSM.ou_colour(1.0)
    ext = [('x', 1)] * 2
    taus = np.linspace(-5, 5, 41)
    res = _cc(m, 2, 1, ext, taus=taus)
    tr = SL.symbolic_triple(m, 1, external_fields=ext, result=res)
    eq = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res)
    tt = taus       # C_tau's τ = 0 point: the exact Itô left limit (side '-')
    for ell in (0, 1):
        ref = res['C_tau_by_ell'][ell]
        assert np.max(np.abs(tr.C(tt, ell) - ref)) <= 1e-13 * np.max(
            np.abs(ref))
        assert tr.at_zero(ell, '-') == eq.by_ell[ell]
        G, x0, cv = tr.dense(ell, '+')
        assert G.nrows() == tr.n_states(ell, '+')
    # ∫_0^∞ e^{-z u} C(u) du = c (z - G)^{-1} x0 == the pairing
    K = Kernel.exponential(1, -2)
    pr = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res,
                              kernel=K)
    assert tr.laplace(2, 1, '+') == pr.by_ell[1]


# ═════════════════════════════════════════════════════════════════════════════
# shot noise (δ edges), refusals
# ═════════════════════════════════════════════════════════════════════════════

def _quad_hawkes():
    return (SDE('c3 quadratic Hawkes').parameter('Em', default=0.5)
            .parameter('tau', default=2.0).parameter('a', default=1.0)
            .parameter('b', default=0.2).parameter('w', default=0.3)
            .define_function('phi', args=['v'], expression='a*v + b*v^2')
            .noise('spikes', Poisson(rate='phi(v)'), expose='n')
            .field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + w*n')
            ).build()


def test_shot_noise_delta_edges_symbolic_saddle():
    """Poisson noise: δ parts of the propagator enter the subset prefactors
    symbolically; the saddle (a root of a quadratic) stays a symbol."""
    m = _quad_hawkes()
    r = SL.symbolic_cumulant(m, 2, 1, external_fields=[('n', 1)] * 2,
                             saddle='symbol')
    assert max(r.check().values()) <= 1e-12
    assert any(n.endswith('star1') for n in r.names)
    assert any(e.d != 0 for e in r._ctx['propagator'].entries.values())
    assert r.stats['delta_subsets'] > 0


def test_refusals_by_name():
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(SSM.alpha_kernel_model(0.4), 2, 1,
                             external_fields=[('x', 1)] * 2)
    assert e.value.reason == 'kernel_edge'
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(SSM.ou_erlang(8), 2, 1,
                             external_fields=[('x', 1)] * 2)
    assert e.value.reason == 'too_large'
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(dd.load_model('ou_quartic')[0], 3, 0,
                             external_fields=[('dx', 1)] * 3,
                             kernel=Kernel.exponential(1, -1))
    assert e.value.reason == 'k_unsupported'


def test_exports():
    from daedalus.api import symbolic_out as so
    for name in ('symbolic_cumulant', 'symbolic_triple', 'SymbolicCumulant',
                 'SymbolicTriple', 'SymbolicLoopRefused', 'SaddleWarning'):
        assert getattr(dd, name) is getattr(SL, name), name
        assert getattr(so, name) is getattr(SL, name), name
        assert name in dir(so)


def _two_field_correlated():
    """x' = -mu1 x - eps1 x^3 + J1 y + xi, y' = -mu2 y - eps2 y^3 + J2 x +
    eta, white noises with correlation rho (amplitude sqrt(D1 D2))."""
    from daedalus.api.model import TemporalModelBuilder
    b = TemporalModelBuilder('c3 two coupled fields, correlated noise')
    b.population('P', size=1)
    b.physical_field('x', population='P').physical_field('y', population='P')
    for p, d in (('mu1', 1.0), ('mu2', 1.3), ('eps1', 0.05), ('eps2', 0.04),
                 ('J1', 0.5), ('J2', 0.3), ('D1', 1.0), ('D2', 0.7),
                 ('rho', 0.4)):
        b.parameter(p, default=d)
    b.set_mf_equation('xstar', '0')
    b.set_mf_equation('ystar', '0')
    b.set_action_text('sum(xt[i]*((Dt+mu1)*x[i] + eps1*x[i]^3 - J1*y[i]) '
                      '+ yt[i]*((Dt+mu2)*y[i] + eps2*y[i]^3 - J2*x[i]) '
                      '- D1*xt[i]^2 - D2*yt[i]^2 '
                      '- 2*rho*sqrt(D1*D2)*xt[i]*yt[i] for i in P)')
    return b.build()


def test_correlated_noise_two_fields_and_square_roots():
    """Two coupled fields with correlated noise: rational in the couplings
    and the correlation; the noise amplitude enters as sqrt(D1 D2), refused
    by name (with the parameters to keep numeric) when D1, D2 are symbolic."""
    m = _two_field_correlated()
    ext = [('x', 1), ('y', 1)]
    res = _cc(m, 2, 1, ext)
    r = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res,
                             symbols=['J1', 'J2', 'rho'])
    assert max(r.check().values()) <= 1e-12
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res)
    assert e.value.reason == 'non_rational'
    assert 'non-integer powers' in str(e.value) and 'D1' in str(e.value)


def test_loop_level_modes_through_c2(ouq_model):
    """τ != 0 symbolically: C2's tree_modes on the triple's rational
    spectrum.  OU quartic one loop: -3 eps D^2/mu^3 (1 + mu|τ|) e^{-mu|τ|}."""
    ext = [('dx', 1)] * 2
    taus = np.linspace(-5, 5, 21)
    res = _cc(ouq_model, 2, 1, ext, taus=taus)
    tr = SL.symbolic_triple(ouq_model, 1, external_fields=ext, result=res)
    ms = tr.modes(1)
    want = {('positive', 0): '-3*D^2*eps/mu^3',
            ('positive', 1): '-3*D^2*eps/mu^2',
            ('negative', 0): '-3*D^2*eps/mu^3',
            ('negative', 1): '3*D^2*eps/mu^2'}
    got = {(mt.side, mt.power): mt.coeff for mt in ms.terms}
    assert set(got) == set(want)
    for key, c in want.items():
        assert _sr_equal(got[key], SR(c)), key
    assert all(_sr_equal(abs(SR(mt.rate)), SR('mu')) for mt in ms.terms)
    tt = taus       # C_tau's τ = 0 point: the exact Itô left limit (side '-')
    for ell in (0, 1):
        v = np.array([complex(x) for x in tr.modes(ell).evaluate(tt)])
        ref = res['C_tau_by_ell'][ell]
        assert np.max(np.abs(v - ref)) <= 1e-13 * np.max(np.abs(ref))
    # the spectrum at ω = 0 is ∫ C = L₊(0) + L₋(0) (the triple's resolvents)
    om = SR.var('omega')
    s0 = tr.spectrum(1).subs({om: 0})
    l0 = tr.laplace(0, 1, '+') + tr.laplace(0, 1, '-')
    assert _sr_equal(s0, tr.field.to_sr(l0))
    assert _sr_equal(s0, SR('-12*eps*D^2/mu^4'))


def test_mixed_fields_orient_the_lag_axis():
    """C_xe(τ) = κ(x(0), e(τ)) is not even in τ: the triples, their τ = 0
    limits and the modes must keep the orientation."""
    m = SSM.ou_colour(2.0)
    ext = [('x', 1), ('e', 1)]
    taus = np.linspace(-4, 4, 17)
    res = _cc(m, 2, 1, ext, taus=taus)
    tr = SL.symbolic_triple(m, 1, external_fields=ext, result=res)
    eq = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res)
    tt = taus       # C_tau's τ = 0 point: the exact Itô left limit (side '-')
    for ell in (0, 1):
        ref = res['C_tau_by_ell'][ell]
        assert abs(ref[2] - ref[-3]) > 1e-3 * np.max(np.abs(ref))   # odd part
        sc = np.max(np.abs(ref))
        assert np.max(np.abs(tr.C(tt, ell) - ref)) <= 1e-13 * sc
        v = np.array([complex(x) for x in tr.modes(ell).evaluate(tt)])
        assert np.max(np.abs(v - ref)) <= 1e-13 * sc
        assert tr.at_zero(ell, '-') == eq.by_ell[ell]
        # continuous at 0 (no shot noise): both limits are the equal-time value
        assert tr.at_zero(ell, '+') == eq.by_ell[ell]
    assert max(tr.check().values()) <= 1e-12


def _driven_quartic():
    from daedalus.api.noise import Gaussian
    return (SDE('c3 driven OU quartic').parameter('h', default=0.5)
            .parameter('mu', default=1.0).parameter('eps', default=0.1)
            .parameter('D', default=0.2).field('x')
            .drift('x', 'h - mu*x - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def test_saddle_auto_follows_a_moving_saddle():
    """The saddle is a root of a cubic that moves with every parameter:
    'auto' keeps it symbolic and the result holds at another point (with the
    pipeline's saddle there); 'numeric' warns when evaluated off the
    reference."""
    m = _driven_quartic()
    ext = [('x', 1)] * 2
    res = _cc(m, 2, 1, ext)
    r = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res)
    assert 'xstar1' in r.names and r.saddle_moves is False
    pt = {'h': 0.9, 'mu': 0.6, 'eps': 0.2, 'D': 0.3}
    res2 = _cc(m, 2, 1, ext, params=pt)
    pt['xstar1'] = float([v for k_, v in res2['num_params'].items()
                          if str(k_) == 'xstar1'][0])
    for ell in (0, 1):
        ref = complex(res2['phase_j_by_ell'][ell]['total_C'](0.0, 0.0))
        assert abs(r.evaluate(pt, ell=ell) - ref) <= 1e-12 * abs(ref)
    num = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res,
                               saddle='numeric')
    assert num.saddle_moves is True and 'xstar1' not in num.names
    assert any(n.startswith('WARNING') for n in num.notes)
    with pytest.warns(SL.SaddleWarning):
        num.evaluate({'h': 0.9}, ell=1)
    with pytest.warns(SL.SaddleWarning):
        num.callable(1)(h=0.9, mu=1.0, eps=0.1, D=0.2)
    # OU quartic: x* = 0 for every parameter, 'auto' is 'numeric'
    r0 = SL.symbolic_cumulant(dd.load_model('ou_quartic')[0], 2, 1,
                              external_fields=[('dx', 1)] * 2)
    assert r0.saddle_moves is False and 'xstar1' not in r0.names


def test_arguments_are_never_silently_ignored(ouq_model):
    import pickle
    ext = [('dx', 1)] * 2
    res = _cc(ouq_model, 2, 1, ext)
    with pytest.raises(ValueError, match='outside'):
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext, result=res,
                             ells=[3])
    with pytest.raises(ValueError, match='max_ell'):
        SL.symbolic_cumulant(ouq_model, 2, 2, external_fields=ext, result=res)
    with pytest.raises(ValueError, match='external_fields'):
        SL.symbolic_cumulant(SSM.ou_colour(1.0), 2, 0,
                             external_fields=[('x', 1), ('e', 1)],
                             result=_cc(SSM.ou_colour(1.0), 2, 0,
                                        [('x', 1)] * 2))
    with pytest.raises(ValueError, match='parameters'):
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext, result=res,
                             parameters={'mu': 3.0})
    # a growing kernel term: the integral diverges (S4 refuses it too)
    with pytest.raises(ValueError, match='grows'):
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext, result=res,
                             kernel=Kernel.exponential(1, 2.0))
    with pytest.raises(ValueError, match='model parameters'):
        SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext, result=res,
                             kernel=Kernel.exponential(1, -SR.var('mu')),
                             kernel_values={'mu': 2.0})
    r = SL.symbolic_cumulant(ouq_model, 2, 1, external_fields=ext, result=res,
                             symbols='eps')
    assert r.names == ('eps',)
    with pytest.raises(ValueError, match='entered as numbers'):
        r.evaluate({'mu': 5.0})
    with pytest.raises(ValueError, match='not variables'):
        r.evaluate({'Eps': 5.0})
    # check() is a real check
    r.pipeline[1] = r.pipeline[1] * (1 + 1e-9)
    with pytest.raises(AssertionError):
        r.check()
    # pickling (the live reference closures are dropped)
    exc = SL.SymbolicLoopRefused('too_large', 'x')
    assert pickle.loads(pickle.dumps(exc)).reason == 'too_large'
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.by_ell == r.by_ell and r2._ctx is None


def _hidden_field(g):
    """z = x + (g - 1/10) w: at g = 1/10 the w part of z vanishes, and a
    reference there would lose it."""
    from daedalus.api.model import TemporalModelBuilder
    b = TemporalModelBuilder(f'c3 hidden field {g}')
    b.population('P', size=1)
    for f in ('x', 'w', 'z'):
        b.physical_field(f, population='P')
    for p, d in (('mu', 1.0), ('eps', 0.1), ('D', 1.0), ('g', g),
                 ('nu', 2.0)):
        b.parameter(p, default=d)
    for f in ('x', 'w', 'z'):
        b.set_mf_equation(f + 'star', '0')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 '
                      '+ wt[i]*((Dt+nu)*w[i]) - wt[i]^2 '
                      '+ zt[i]*(z[i] - x[i] - (g - 1/10)*w[i]) for i in P)')
    return b.build()


def test_reference_points_that_hide_terms_are_refused():
    ext = [('z', 1)] * 2
    r = SL.symbolic_cumulant(_hidden_field(0.3), 2, 1, external_fields=ext)
    assert _sr_equal(r.expression(0), SR('D/mu + (g - 1/10)^2/nu'))
    with pytest.raises(SL.SymbolicLoopRefused) as e:
        SL.symbolic_cumulant(_hidden_field(0.1), 2, 1, external_fields=ext)
    assert e.value.reason == 'degenerate_reference'


def test_exact_zero_cumulant_and_the_diagram_check(monkeypatch):
    """<p q> = 0 exactly (q' = -w p, stationarity): symbolic 0 against the
    route's rounding residue passes; a 1e-9 error in one diagram does not."""
    m = SSM.oscillator_red_noise()
    ext = [('p', 1), ('q', 1)]
    res = _cc(m, 2, 1, ext)
    r = SL.symbolic_cumulant(m, 2, 1, external_fields=ext, result=res)
    assert all(v == 0 for v in r.by_ell.values())
    orig = SL._diagram_value

    def off(*a, **kw):
        val, n = orig(*a, **kw)
        return val * (1 + QQ(1) / 10 ** 9) + QQ(1) / 10 ** 9, n
    monkeypatch.setattr(SL, '_diagram_value', off)
    with pytest.raises(AssertionError, match='diagram'):
        SL.symbolic_cumulant(SSM.ou_colour(1.0), 2, 1,
                             external_fields=[('x', 1)] * 2)


def test_symbolic_saddle_needs_the_saddle_of_the_point():
    m = _driven_quartic()
    r = SL.symbolic_cumulant(m, 2, 1, external_fields=[('x', 1)] * 2)
    assert 'xstar1' in r.names
    with pytest.warns(SL.SaddleWarning, match='saddle'):
        r.evaluate({'h': 0.9}, ell=1)

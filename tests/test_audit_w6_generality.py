"""
tests/test_audit_w6_generality.py
=================================
Model-agnostic machinery (audit wave W6): no field, parameter or saddle
spelling decides what the framework does.

* GEN-01: ``ModelBuilder(..., n_populations=N)`` without ``.population(...)``
  is the explicit population ``pop`` of size N: an exact relabeling of a
  two-population linear Hawkes model (fields, drive, kernel) gives the same
  numbers, and its saddle is the closed form ``n* = (I - a W)^-1 a E``.  A
  mean-field check that cannot run says why (the solver's error is kept).
* GEN-02: every model goes through the one multi-root mean-field solver: a
  ``.set_mf_equation`` model with two roots reports both and
  ``fixed_point_index`` selects either (closed form of the quadratic).
* GEN-03/GEN-04: saddle = the builder's ``mean_field`` flag: a real parameter
  ``tstar`` is a parameter; a saddle named ``rate_mf`` (no ``star``) is a
  saddle for every consumer (results, accessors, means, labels, symbolic
  route) and matches its ``xstar``-named twin.
* GEN-07: field names ending in digits (``x1``, ``x0``/``x1``) are
  addressable and equal their letter-named twins; the symbol names are
  unambiguous (``dx1`` pop 1 vs ``dx`` pop 11).
* GEN-17: derived symbols (``phi0_1``, ``phi2_1``) are not free parameters
  of the symbolic loop route.
* GEN-05/GEN-06: the Hawkes / GTaS templates: a template-built linear Hawkes
  model equals the text-built one (renamed, any call order), no invented
  parameter values, the delta-kernel specialization survives any call order,
  and the GTaS cumulants are those of the cluster process (leg-order
  identity, Campbell integrals of the mixed-cell kappa3).
* W6 add-ons: the analyzer extracts a spatial drift (``GradX``) at a non-zero
  saddle; a two-population spatial model with a nonlinear coupling solves
  both populations (OU closed form).

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_audit_w6_generality.py -q
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, var, integrate, dirac_delta, oo    # noqa: E402

import daedalus as dd                                      # noqa: E402
from daedalus.api import compute_cumulants                          # noqa: E402
from daedalus.api.model import ModelBuilder                         # noqa: E402
from daedalus.api.sde import SDE                                    # noqa: E402
from daedalus.api.noise import Gaussian                             # noqa: E402

RUN = dict(parallel=False, verbose=False, use_cache=False)
TAUS = np.array([-1.0, 0.0, 0.5, 3.0])
E0, W0, A0 = [0.78, 0.81], [[0.30, 0.25], [0.30, 0.35]], 1.0


# ── GEN-01 ────────────────────────────────────────────────────────────
def _legacy_hawkes(n='n', v='v', E='E', g='g'):
    return (ModelBuilder(f'w6 lh {n}{v}{E}{g}', n_populations=2)
            .physical_field(n).physical_field(v)
            .parameter(E, indexed=True, default=E0)
            .parameter('tau', default=10.0, domain='positive')
            .parameter('a', default=A0)
            .parameter('tau_g', default=2.5, domain='positive')
            .parameter('w', indexed='matrix', default=W0)
            .define_function('phi', args=['v'], expression='a * v')
            .define_kernel(g, freq_image='1 / (1 + I*omega*tau_g)')
            .set_action_text(
                f'sum({n}t[i]*{n}[i] - (exp({n}t[i]) - 1)*phi[i]({v}[i])'
                f' + {v}t[i]*((tau*Dt + 1)*{v}[i] - {E}[i]'
                f' - sum(w[i, j]*{g}*{n}[j] for j in pop)) for i in pop)')
            .set_mf_equation(f'{v}star',
                             f'{E}[i] + sum(w[i, j] * {g} * {n}star[j] for j in pop)')
            .set_mf_equation(f'{n}star', f'phi({v}star[i])')
            .build())


def _C(model, ext, **kw):
    r = compute_cumulants(model, k=kw.pop('k', 2), max_ell=kw.pop('max_ell', 0),
                          external_fields=ext, tau_grid=kw.pop('taus', TAUS),
                          **RUN, **kw)
    return np.array([complex(x) for x in r['C_tau']]), r


def test_gen01_legacy_population_is_explicit_and_rename_invariant():
    m = _legacy_hawkes()
    assert m['populations'] == [{'name': 'pop', 'size': 2,
                                 'description': 'n_populations=2 index set'}]
    ref, r = _C(m, [('n', 1), ('n', 2)])
    # independent reference: the linear saddle in closed form
    n_exact = np.linalg.solve(np.eye(2) - A0 * np.array(W0), A0 * np.array(E0))
    assert np.allclose(r['mf_values']['nstar'], n_exact, rtol=4e-16, atol=0)
    for kw in (dict(n='r', v='u', E='b', g='k'), dict(E='b'), dict(g='k')):
        c, _ = _C(_legacy_hawkes(**kw), [(kw.get('n', 'n'), 1),
                                         (kw.get('n', 'n'), 2)])
        assert np.max(np.abs(c - ref) / np.abs(ref)) < 1e-15, kw


def test_gen01_mean_field_check_names_the_solver_failure():
    # x* = x*^2 + 1 has no real root: the numerical sector check cannot run,
    # and the error says why instead of a bare 'N/A'
    b = (ModelBuilder('w6 no real saddle')
         .physical_field('x', domain='real')
         .parameter('D', default=0.1, domain='positive')
         .set_action_text('xt*((Dt + 1)*x - x^2 - 1) - D*xt^2')
         .set_mf_equation('xstar', 'xstar^2 + 1'))
    with pytest.raises(Exception) as ei:
        _C(b.build(), [('x', 1), ('x', 1)], preflight=False)
    assert 'mean-field solve' in str(ei.value)


# ── GEN-02 ────────────────────────────────────────────────────────────
def _two_root(b_=3.0, c_=1.0):
    return (ModelBuilder(f'w6 two roots {b_} {c_}')
            .physical_field('x', domain='real')
            .parameter('b', default=b_).parameter('c', default=c_)
            .parameter('D', default=0.05, domain='positive')
            .set_action_text('xt*((Dt + b)*x - x^2 - c) - D*xt^2')
            .set_mf_equation('xstar', '(xstar^2 + c)/b')
            .build())


def test_gen02_set_mf_equation_models_get_every_root():
    b_, c_ = 3.0, 1.0
    roots = sorted([(b_ - math.sqrt(b_**2 - 4*c_)) / 2,
                    (b_ + math.sqrt(b_**2 - 4*c_)) / 2])
    m = _two_root(b_, c_)
    _, r0 = _C(m, [('x', 1), ('x', 1)], preflight=False)
    got = sorted(rr['values']['xstar'][0] for rr in r0['mf_all_roots'])
    assert np.allclose(got, roots, rtol=4e-16, atol=0)
    assert r0['mf_values']['xstar'][0] == pytest.approx(roots[0], rel=4e-16)
    # the other (unstable) root through the solver's front door
    from daedalus.engine.core.field_theory import FieldTheory
    from daedalus.api._mean_field import solve_mean_field
    ft = FieldTheory(m)
    ft.expand()
    r1 = solve_mean_field(ft, m, {}, verbose=False, fixed_point_index=1)
    assert r1['mf_values']['xstar'][0] == pytest.approx(roots[1], rel=4e-16)


def test_gen02_units_rescaled_root_found():
    # the same quadratic in field units 1e-6 (x -> 1e-6 x): the old solver's
    # unit-scale guesses could not reach it; the root rescales exactly
    m = _two_root(3.0e-6, 1.0e-12)
    _, r = _C(m, [('x', 1), ('x', 1)], preflight=False)
    lo = (3.0 - math.sqrt(5.0)) / 2 * 1e-6
    assert r['mf_values']['xstar'][0] == pytest.approx(lo, rel=1e-14)


# ── GEN-03 / GEN-04 ───────────────────────────────────────────────────
def _ou(tname):
    return (SDE(f'w6 ou {tname}').parameter(tname, default=1.0, domain='positive')
            .parameter('eps', default=0.02).parameter('D', default=1.0, domain='positive')
            .field('x').drift('x', f'-x/{tname} - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def test_gen03_real_parameter_named_tstar_is_a_parameter():
    from daedalus.api.symbolic_out import _is_saddle_symbol
    m = _ou('tstar')
    assert 'tstar' in dd.param_names(m)
    assert dd.fundamental_from_model(m)['tstar'] == 1.0
    assert not _is_saddle_symbol('tstar', m)
    assert _is_saddle_symbol('xstar1', m)
    c_s, _ = _C(m, [('x', 1), ('x', 1)], parameters={'tstar': 0.8, 'eps': 0.02, 'D': 1.0})
    c_d, _ = _C(_ou('tdec'), [('x', 1), ('x', 1)],
                parameters={'tdec': 0.8, 'eps': 0.02, 'D': 1.0})
    assert np.max(np.abs(c_s - c_d)) == 0.0


def _named_saddle(sname):
    b = ModelBuilder(f'w6 saddle {sname}').physical_field('x', auto_saddle=False)
    b.parameter(sname, indexed=True, mean_field=True, natural_name='x')
    return (b.parameter('mu', default=1.0, domain='positive')
            .parameter('h', default=0.4).parameter('D', default=0.1, domain='positive')
            .set_action_text('xt*((Dt + mu)*x + 0.2*x^3 - h) - D*xt^2')
            .mf_residual(lhs='(Dt + mu)*x', rhs='h - 0.2*x^3')
            .build())


def test_gen04_saddle_named_without_star_reaches_every_consumer():
    from daedalus.api.naming import saddle_for_field, saddle_label, field_for_saddle
    from daedalus.api.symbolic_out import _is_saddle_symbol
    ref_m, odd_m = _named_saddle('xstar'), _named_saddle('rate_mf')
    c_ref, r_ref = _C(ref_m, [('x', 1), ('x', 1)], max_ell=1)
    c_odd, r_odd = _C(odd_m, [('x', 1), ('x', 1)], max_ell=1)
    assert np.max(np.abs(c_odd - c_ref)) == 0.0
    assert r_odd['mf_values']['rate_mf'] == r_ref['mf_values']['xstar']
    # accessor, external mean, naming service, labels, symbolic route
    assert r_odd['mf']['x', 1] == r_ref['mf']['x', 1]
    assert dd._external_mean(odd_m, r_odd, ('x', 1)) == r_ref['mf_values']['xstar'][0]
    assert saddle_for_field(odd_m, 'dx') == 'rate_mf'
    assert field_for_saddle(odd_m, 'rate_mf') == 'x'
    assert saddle_label(odd_m, 'rate_mf') == 'x*'
    assert _is_saddle_symbol('rate_mf1', odd_m) and not _is_saddle_symbol('mu', odd_m)
    assert 'rate_mf' not in dd.param_names(odd_m)


def test_gen04_access_has_no_builtin_letter_table():
    from daedalus.api.access import normalize_external_fields, MeanField
    assert normalize_external_fields([('n', 1)]) == [('n', 1)]
    conv = {'fluctuation_fields': {'r': 'dr'}, 'mean_field_saddles': {'r': 'rstar'}}
    assert normalize_external_fields([('r', 2)], conv) == [('dr', 2)]
    assert MeanField({'rstar': [0.5]}, conv)['r', 1] == 0.5


# ── GEN-07 ────────────────────────────────────────────────────────────
def test_gen07_field_symbol_names_round_trip():
    from daedalus.engine.core.vertices import field_symbol_name, _parse_field_name
    for base in ('dn', 'dx1', 'dx', 'x_', 'a1_', 'dy12'):
        for idx in (1, 2, 11):
            assert _parse_field_name(field_symbol_name(base, idx)) == (base, idx)
        assert _parse_field_name(field_symbol_name(base))[0] == base
    assert field_symbol_name('dx1', 1) != field_symbol_name('dx', 11)
    assert field_symbol_name('dn', 2) == 'dn2'       # unchanged for letters


def _digits(a, b):
    return (SDE('w6 dig ' + a + b).parameter('mu', default=1.0).parameter('DD', default=0.5)
            .field(a).field(b).drift(a, f'-mu*{a} + 0.3*{b}').drift(b, f'-2*{b} + 0.1*{a}^2')
            .noise('xi', Gaussian(var='2*DD'), couples={a: 1})
            .noise('eta', Gaussian(var='DD'), couples={b: 1}).build())


def test_gen07_digit_field_names_equal_letter_twin():
    ref, _ = _C(_digits('xa', 'xb'), [('xa', 1), ('xb', 1)], max_ell=1)
    for a, b in (('x0', 'x1'), ('x1', 'x2'), ('u1', 'v1')):
        c, _ = _C(_digits(a, b), [(a, 1), (b, 1)], max_ell=1)
        assert np.max(np.abs(c - ref)) == 0.0, (a, b)


# ── GEN-17 ────────────────────────────────────────────────────────────
def test_gen17_derived_symbols_are_not_free_parameters():
    from daedalus.api.symbolic_loop import _saddle_names
    from daedalus.api.naming import is_parameter_symbol
    m = _legacy_hawkes()
    nump = {SR.var(s): 1.0 for s in ('E1', 'E2', 'w12', 'tau', 'a', 'nstar1',
                                     'vstar2', 'phi0_1', 'phi2_1')}
    sad = set(_saddle_names(m, nump))
    assert sad == {'nstar1', 'vstar2', 'phi0_1', 'phi2_1'}
    assert is_parameter_symbol('w12', m) and not is_parameter_symbol('phi1_2', m)


# ── GEN-05 / GEN-06: templates ────────────────────────────────────────
def _tmpl(order='ak', kernel='exp', names=None):
    from daedalus.api.model_templates import HawkesAction, ExpSynapticKernel, DeltaKernel
    nm = dict(E='E', w='w', tau='tau', a='a', g='g', tau_g='tau_g')
    nm.update(names or {})
    b = (ModelBuilder('w6 tmpl ' + ' '.join(nm.values()) + order + kernel,
                      n_populations=2)
         .physical_field('dn', natural_name='n').physical_field('dv', natural_name='v')
         .parameter(nm['E'], indexed=True, default=E0)
         .parameter(nm['w'], indexed='matrix', default=W0)
         .parameter(nm['tau'], default=10.0, domain='positive')
         .parameter(nm['a'], default=A0)
         .parameter(nm['tau_g'], default=2.5, domain='positive')
         .kernel(nm['g']))
    act = HawkesAction(phi='linear', gain_param=nm['a'], synaptic_kernel=nm['g'],
                       recurrent_weight=nm['w'], external_drive=nm['E'],
                       membrane_timescale=nm['tau'])
    ker = (ExpSynapticKernel(kernel_name=nm['g'], timescale_param=nm['tau_g'])
           if kernel == 'exp' else DeltaKernel(kernel_name=nm['g']))
    if order == 'ak':
        return b.use_action_template(act).use_synaptic_kernel(ker)
    return b.use_synaptic_kernel(ker).use_action_template(act)


def test_gen06_template_equals_text_model_renamed_any_order():
    ref, _ = _C(dd.load_model('linear_hawkes')[0], [('n', 1), ('n', 2)])
    for kw in ({}, dict(order='ka'),
               dict(names=dict(E='b', w='J', tau='tm', a='gain', g='k', tau_g='tk'))):
        c, _ = _C(_tmpl(**kw).build(), [('n', 1), ('n', 2)])
        assert np.max(np.abs(c - ref) / np.abs(ref)) < 1e-15, kw


def test_gen06_delta_kernel_specialization_any_order():
    from types import SimpleNamespace
    for order in ('ak', 'ka'):
        m = _tmpl(order=order, kernel='delta').build()
        ns = SimpleNamespace(pop=[0, 1], _taylor_order=3, g=SR.var('g'),
                             delta_D=SR.var('delta_D'))
        assert ns.g in m['specializations'](ns), order


def test_gen06_gtas_parameters_are_not_invented():
    from daedalus.api.model_templates import GTaSNoise
    m = _tmpl().add_gtas_noise(GTaSNoise()).build()
    decl = {p['name']: p for p in m['parameters']}
    for name in ('w_X', 'lambda_X', 'p_part', 'mu_shift_diff', 'sigma_shift_diff_sq'):
        assert decl[name].get('default') is None, name
    with pytest.raises(ValueError, match='lambda_X'):
        _C(m, [('n', 1), ('n', 2)])


def test_gen05_gtas_cumulants_are_the_cluster_process():
    from types import SimpleNamespace
    from daedalus.api.model_templates import GTaSNoise
    lam, p, mu, s2 = 1.7, 0.6, 0.5, 0.2
    ns = SimpleNamespace(lam=SR(lam), pp=SR(p), md=SR(mu), sd=SR(s2))
    blk = GTaSNoise(mother_rate='lam', participation='pp', mu_diff='md',
                    sigma_diff_sq='sd').correlated_noises_block()['X']
    k2, k3, k4 = (blk['cumulants'][n] for n in (2, 3, 4))
    t, t1, t2, t3 = var('t t1 t2 t3')
    for tv in (-1.0, 0.3, 0.5):                       # k_ij(tau) = k_ji(-tau)
        assert float(k2(ns, 0, 1, t).subs(t=tv)) == pytest.approx(
            float(k2(ns, 1, 0, t).subs(t=-tv)), rel=1e-15)
    # framework convention tau = t_first - t_second: cell 1 fires mu after
    # cell 0, so k_01 peaks at tau = -mu
    assert float(k2(ns, 0, 1, t).subs(t=-mu)) == pytest.approx(
        lam * p**2 / math.sqrt(2 * math.pi * s2), rel=1e-15)
    assert float(k2(ns, 0, 0, t).coefficient(dirac_delta(t))) == lam * p
    # Campbell: the integral of a mixed-cell kappa3 is lambda p^2 (cells
    # {0, 1}); the Monte Carlo of the audit gave 29.6 +- 2.2 per T = 50
    for cells in ((0, 0, 1), (0, 1, 1), (1, 0, 0)):
        e = k3(ns, *cells, t1, t2)
        val = integrate(integrate(e, t1, -oo, oo), t2, -oo, oo)
        assert float(val) == pytest.approx(lam * p**2, rel=1e-12), cells
    e4 = k4(ns, 0, 0, 1, 1, t1, t2, t3)
    v4 = integrate(integrate(integrate(e4, t1, -oo, oo), t2, -oo, oo), t3, -oo, oo)
    assert float(v4) == pytest.approx(lam * p**2, rel=1e-12)
    # independent cells (no shift law): mixed-cell cumulants vanish
    ind = GTaSNoise(mother_rate='lam', participation='pp', mu_diff=None,
                    sigma_diff_sq=None).correlated_noises_block()['X']['cumulants']
    assert ind[3](ns, 0, 0, 1, t1, t2) == 0 and ind[2](ns, 0, 1, t) == 0


# ── W6 add-ons ────────────────────────────────────────────────────────
def test_analyzer_extracts_spatial_drift_at_nonzero_saddle():
    from daedalus.api.analyze import analyze
    m = (ModelBuilder('w6 advected nz', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .parameter('h', default=0.5).parameter('v', default=0.7, domain='real')
         .mf_residual(lhs='(Dt + mu - D*Laplacian)*phi', rhs='h')
         .set_action_text('phit*(Dt(phi) + mu*phi - D*Lap(phi) + v*Dx(phi, 0)'
                          ' - h) - T*phit^2').operator_ir()
         .boundary('infinite').initial('stationary').build())
    codes = [f.code for f in analyze(m, use_cache=False).findings]
    assert 'G3_SPATIAL_FORM' not in codes, codes


def _ou_closed(x, tau, mu, D, T):
    from scipy.special import erfc
    x, tau = abs(float(x)), abs(float(tau))
    a = math.sqrt(mu / D)
    if tau == 0.0:
        return T / (2.0 * math.sqrt(mu * D)) * math.exp(-a * x)
    s, r = math.sqrt(mu * tau), x / (2.0 * math.sqrt(D * tau))
    return T / (4.0 * math.sqrt(mu * D)) * (
        math.exp(-a * x) * erfc(s - r) + math.exp(a * x) * erfc(s + r))


def test_two_population_spatial_nonlinear_solves_both_populations():
    m = (ModelBuilder('w6 two-pop spatial nl', n_populations=2)
         .physical_field('p', spatial_dim=1)
         .parameter('mu', default=[1.0, 4.0], indexed='vector', domain='positive')
         .parameter('DD', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .parameter('g', default=0.3)
         .set_action_text('sum(pt[i]*((Dt+mu[i]-DD*Laplacian)*p[i] + g*p[i]^3)'
                          ' - T*pt[i]^2 for i in pop)')
         .mf_residual(lhs='(Dt + mu[i] - DD*Laplacian)*p[i]', rhs='-g*p[i]^3')
         .boundary('infinite').initial('stationary').build())
    taus, xs = [0.0, 0.5], [0.0, 0.7]
    for pop, mu in ((1, 1.0), (2, 4.0)):
        r = compute_cumulants(m, k=2, max_ell=0, external_fields=[('p', pop)] * 2,
                              tau_grid=np.array(taus), spatial_grid=np.array(xs),
                              spatial_parallel=False, **RUN)
        assert r['mf_values']['pstar'] == [0.0, 0.0]
        C = np.real(np.asarray(r['C_tau_x']))
        ref = np.array([[_ou_closed(x, t, mu, 1.0, 1.0) for x in xs] for t in taus])
        assert np.max(np.abs(C - ref)) < 1e-13, (pop, C, ref)


# ── order-2 noise kernels in their own leg orientation (found with GEN-05) ─
@pytest.mark.parametrize('engine', ['statespace', 'poles'])
def test_gtas_cross_covariance_is_the_cluster_process_in_both_orders(
        monkeypatch, engine):
    """<m1(t) m2(t+tau)> of the GTaS noise field equals lambda p^2 N(tau;
    mu, s2) (cell 2 fires mu after cell 1) and the reverse order its mirror:
    the asymmetric cross kernel is no longer symmetrized."""
    from daedalus.api.model_templates import GTaSNoise
    lam, p, mu, s2 = 1.7, 0.6, 0.5, 0.2
    m = (_tmpl().parameter('w_X', default=0.25)
         .parameter('lambda_X', default=lam, domain='positive')
         .parameter('p_part', default=p, domain='positive')
         .parameter('mu_shift_diff', default=mu)
         .parameter('sigma_shift_diff_sq', default=s2, domain='positive')
         .add_gtas_noise(GTaSNoise()).build())
    taus = np.array([-1.0, 0.0, 1.0])
    g = lambda t, c: lam * p**2 * math.exp(-(t - c)**2 / (2 * s2)) / math.sqrt(2 * math.pi * s2)
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', engine)
    c12, r = _C(m, [('m', 1), ('m', 2)], taus=taus)
    c21, _ = _C(m, [('m', 2), ('m', 1)], taus=taus)
    assert np.allclose(c12.real, [g(t, mu) for t in taus], rtol=1e-13, atol=0)
    assert np.allclose(c21.real, [g(t, -mu) for t in taus], rtol=1e-13, atol=0)
    assert r['mf_values']['mstar'] == [pytest.approx(lam * p)] * 2
    # the state-space engine declines noise-kernel diagrams: both engine
    # settings evaluate them on the shared route fixed here
    assert r['phase_j_engine']['engine'] == engine


def test_asymmetric_cross_field_cgf_row_matches_quadrature():
    """A cross-field row (xt, yt) with a shifted-Gaussian kernel K(t_x - t_y)
    (its swapped order is reconstructed as K(-tau)): <x(0) y(tau)> equals the
    double integral of the two OU responses against K."""
    from scipy.integrate import dblquad
    a, b, c, m0, s2 = 1.0, 1.7, 0.3, 0.6, 0.15
    m = (ModelBuilder('w6 xcgf asym').physical_field('x').physical_field('y')
         .parameter('a', default=a).parameter('b', default=b).parameter('c', default=c)
         .parameter('m0', default=m0).parameter('s2', default=s2)
         .set_action_text('xt*((Dt + a)*x) + yt*((Dt + b)*y)')
         .mf_residual(lhs='(Dt + a)*x', rhs='0').mf_residual(lhs='(Dt + b)*y', rhs='0')
         .declare_cgf_term('Nx', response_legs=['xt', 'xt'], order=2, coefficient='1')
         .declare_cgf_term('Ny', response_legs=['yt', 'yt'], order=2, coefficient='0.8')
         .declare_cgf_term('Nxy', response_legs=['xt', 'yt'], order=2, coefficient='c',
                           kernel='exp(-(tau - m0)^2/(2*s2))/sqrt(2*pi*s2)',
                           markovianize=False)
         .build())
    taus = np.array([-1.0, 0.0, 0.6, 1.5])
    C, _ = _C(m, [('x', 1), ('y', 1)], taus=taus)
    K = lambda u: c * math.exp(-(u - m0)**2 / (2 * s2)) / math.sqrt(2 * math.pi * s2)
    for t, v in zip(taus, C.real):
        ref, _ = dblquad(lambda u, s: math.exp(a * s - b * (t - u)) * K(s - u),
                         -40, 0, lambda s: -40, lambda s: t, epsabs=1e-13, epsrel=1e-11)
        assert v == pytest.approx(ref, rel=1e-9)


# ── LOUD-02 ───────────────────────────────────────────────────────────
def test_loud02_cgmy_symbolic_exponent_mean_is_closed_form():
    """CGMY with a parameter-named Y: its mean carries Gamma functions (Sage
    prints gamma(1 - Y) as factorial(-Y)); the saddle solve evaluates them
    (it raised NameError 'factorial')."""
    from daedalus.api.noise import CGMY
    C_, G_, M_, Y_ = 0.2, 3.0, 2.0, 0.5
    s = SDE('w6 cgmy sym').parameter('mu', default=1.0)
    for nm, v in dict(cC=C_, cG=G_, cM=M_, cY=Y_).items():
        s = s.parameter(nm, default=v)
    m = (s.field('x').drift('x', '-mu*x')
         .noise('xi', CGMY('cC', 'cG', 'cM', 'cY'), couples={'x': 1}).build())
    c, r = _C(m, [('x', 1), ('x', 1)], taus=np.array([0.0]))
    mean = C_ * math.gamma(1 - Y_) * (M_**(Y_ - 1) - G_**(Y_ - 1))
    k2 = C_ * math.gamma(2 - Y_) * (M_**(Y_ - 2) + G_**(Y_ - 2))
    assert r['mf_values']['xstar'][0] == pytest.approx(mean, rel=1e-15)
    assert c[0].real == pytest.approx(k2 / 2, rel=1e-14)


def test_loud02_printed_factorial_is_sages_and_gamma_stays_a_parameter_name():
    """``factorial(...)`` / ``gamma(...)`` calls (Sage prints Gamma functions that way) parse as Sage's
    own functions: a formal function of that name broke Sage 10.9's Maxima conversion. A bare
    ``gamma`` is still a symbol, so a parameter may be named gamma."""
    from sage.all import SR, factorial
    from daedalus.api.noise import parse_expr
    e = parse_expr('cC*factorial(-cY) + gamma*x')
    assert not SR(e).is_zero()                        # Sage 10.9: RuntimeError (operator mismatch)
    assert e.subs(cC=1, cY=SR(1) / 2, gamma=0) == factorial(-SR(1) / 2)
    assert SR.var('gamma') in e.variables()
    # ... but not a user function name: its calls would mean Sage's function on some paths only
    # (a user function gamma(v) in a rate failed the saddle check on Sage 10.8 already).
    from daedalus.api.sde import SDE, SDEError
    for name in ('gamma', 'factorial'):
        with pytest.raises(SDEError, match='reserved'):
            SDE('w6 reserved fn').field('v').define_function(name, args=['v'], expression='v')


def test_gen02_deeply_negative_compound_saddle_falls_back_to_iteration_saddles():
    """A voltage saddle at -100 lies outside the default seed box of the full
    saddle system; the solver falls back to the iteration saddles alone (the
    compound one substituted), as the former iteration solver did."""
    from daedalus.engine.core.field_theory import FieldTheory
    from daedalus.api._mean_field import solve_mean_field
    E, w = -100.0, 0.3
    m = (ModelBuilder('w6 deep negative saddle')
         .population('P', size=2)
         .physical_field('n', population='P').physical_field('v', population='P')
         .parameter('E', default=[E, E], indexed_by=['P'])
         .parameter('w', default=[[w, 0.1], [0.1, w]], indexed_by=['P', 'P'])
         .parameter('tau', default=1.0, domain='positive')
         .define_function('phi', args=['v'], expression='1/(1+exp(-v))', population='P')
         .set_action_text('sum(nt[i]*n[i] - (exp(nt[i]) - 1)*phi[i](v[i]) + vt[i]*((tau*Dt + 1)*v[i]'
                          ' - E[i] - sum(w[i, j]*n[j] for j in P)) for i in P)')
         .set_mf_equation('vstar', 'E[i] + sum(w[i, j] * nstar[j] for j in P)')
         .set_mf_equation('nstar', 'phi[i](vstar[i])').build())
    ft = FieldTheory(m)
    ft.expand()
    r = solve_mean_field(ft, m, {}, verbose=False)
    n_ref = 0.0
    for _ in range(50):                       # independent fixed-point iteration
        v_ref = E + (w + 0.1) * n_ref
        n_ref = 1.0 / (1.0 + math.exp(-v_ref))
    assert r['mf_values']['nstar'] == [pytest.approx(n_ref, rel=1e-14)] * 2
    assert r['mf_values']['vstar'] == [pytest.approx(v_ref, rel=1e-15)] * 2


def test_gen04_custom_saddle_survives_save_and_load(tmp_path):
    from daedalus.api.model_serialize import load_spec_from_file, render_model_file
    src = '''from daedalus.api.model import ModelBuilder


def build():
    return (ModelBuilder('w6 rt saddle')
        .physical_field('x', auto_saddle=False)
        .parameter('rate_mf', indexed=True, mean_field=True, natural_name='x')
        .parameter('mu', default=1.0, domain='positive')
        .parameter('h', default=0.4)
        .parameter('D', default=0.1, domain='positive')
        .set_action_text('xt*((Dt + mu)*x + 0.2*x^3 - h) - D*xt^2')
        .mf_residual(lhs='(Dt + mu)*x', rhs='h - 0.2*x^3')
        .build())
'''
    f = tmp_path / 'w6_rt.model.py'
    f.write_text(src)
    txt = render_model_file(load_spec_from_file(str(f)))
    ns = {}
    exec(compile(txt, 'rendered', 'exec'), ns)
    m = ns['build']()
    sad = [p['name'] for p in m['parameters'] if p.get('mean_field')]
    assert sad == ['rate_mf'], sad
    c1, _ = _C(m, [('x', 1), ('x', 1)])
    c0, _ = _C(_named_saddle('rate_mf'), [('x', 1), ('x', 1)])
    assert np.max(np.abs(c1 - c0)) == 0.0


def test_permuted_cumulant_kernel_reads_the_rows_own_leg_order():
    from daedalus.engine.core.field_theory import _PermutedKernel
    calls = []

    def k3(ns, i, j, k, t1, t2):
        calls.append((i, j, k))
        return SR(1)
    _PermutedKernel(k3, (2, 0, 1))(None, 'a', 'b', 'c', var('t1'), var('t2'))
    assert calls == [('b', 'c', 'a')]          # position k holds row leg perm[k]
    t = var('t')
    k2 = lambda ns, i, j, tau: (i - j) * SR(var('c2')) * dirac_delta(tau) + (i + 2 * j) * tau
    out = _PermutedKernel(k2, (1, 0))(None, 1, 5, t)
    assert (out - k2(None, 5, 1, -t).subs({dirac_delta(-t): dirac_delta(t)})).expand() == 0
    assert out.coefficient(dirac_delta(t)) == 4 * var('c2')

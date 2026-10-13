"""D3 regime advisories (gate G5) and the D4 report table of ``api/analyze.py``.

A  pole statistics: decay-rate range, fast/slow ratio, close and repeated poles, with their remedies, on synthetic
   models (two decoupled OU fields; a Jordan block).
B  ``g_eff`` (definition: ``api.analyze._g_eff``): the closed form ``3 eps D / mu^2`` on the OU quartic model, monotone
   in the coupling and in the noise, zero for a linear model, a warning only when loops are requested, ``strict``
   semantics unchanged.
C  the Taylor-series check for non-polynomial responses: the estimator against a response with a known radius of
   convergence, monotone in the noise, a warning where the truncation is unreliable, nothing for polynomial models.
D  the numbers do not move: a run with the analyzer on equals a run with it off, bit for bit.
E  the report table: golden text on two models, at most 100 columns on every tracked temporal and spatial model,
   printed by ``dd.run(verbose=True)``.
"""
import importlib.util
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sage.all')

import api.compute as C                                                       # noqa: E402
from api import analyze as A                                                  # noqa: E402
from api.analyze import FINDING_CODES, TABLE_WIDTH, analyze                   # noqa: E402
from api.noise import Gaussian                                                # noqa: E402
from api.sde import SDE                                                       # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / 'models'


def _load(name):
    spec = importlib.util.spec_from_file_location(f'g5_{name}', MODELS / f'{name}.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(), mod


def _ou_quartic(**par):
    model, mod = _load('ou_quartic')
    return model, dict(mod.DEFAULT_FUNDAMENTAL, **par)


def _geff(eps=0.02, D=1.0, mu=1.0, max_ell=1, **kw):
    model, par = _ou_quartic(eps=eps, D=D, mu=mu)
    return analyze(model, question={'k': 2, 'max_ell': max_ell, 'parameters': par}, **kw)


def _two_ou(a, b):
    """Two decoupled OU fields with rates a and b, unit noise: the pole statistics are exact."""
    s = SDE('G5 two OU')
    s.parameter('a', default=a, domain='positive')
    s.parameter('b', default=b, domain='positive')
    return (s.field('x').field('y').drift('x', '-a*x').drift('y', '-b*y')
            .noise('Wx', Gaussian(var='2'), couples={'x': 1}).noise('Wy', Gaussian(var='2'), couples={'y': 1})
            .stability_analysis(True).build())


def _jordan():
    """dx = (-x + y) dt + noise, dy = -y dt + noise: a Jordan block, the repeated pole -i."""
    s = SDE('G5 Jordan block')
    s.parameter('a', default=1.0, domain='positive')
    return (s.field('x').field('y').drift('x', '-a*x + y').drift('y', '-a*y')
            .noise('Wx', Gaussian(var='2'), couples={'x': 1}).noise('Wy', Gaussian(var='2'), couples={'y': 1})
            .stability_analysis(True).build())


def _nl_ou(drift, pars, D):
    """Scalar OU-type SDE with a non-polynomial drift (Taylor expanded about the saddle)."""
    s = SDE('G5 nonlinear OU')
    for p, d in dict(pars, D=D).items():
        s.parameter(p, default=d, domain='positive')
    return (s.field('x').drift('x', drift).noise('W', Gaussian(var='2*D'), couples={'x': 1})
            .stability_analysis(True).build())


SIGMOID = ('-mu*x + a/(1+exp(-(x-c)))', dict(mu=1.0, a=2.0, c=0.5))
RATIONAL = ('-mu*x + a/(1+x/L)', dict(mu=1.0, a=2.0, L=3.0))
_TAYLOR_POLICY = {'allow_taylor_truncation': True}


# ── A: pole statistics ────────────────────────────────────────────────────────────────────────

def test_pole_rate_range_and_stiff_ratio_with_remedy():
    r = analyze(_two_ou(1.0, 5000.0), question={'k': 2, 'max_ell': 0})
    t = r.traits
    assert t.pole_rate_min == pytest.approx(1.0) and t.pole_rate_max == pytest.approx(5000.0, rel=1e-9)
    assert t.fast_slow_ratio == pytest.approx(5000.0, rel=1e-9) and t.n_poles == 2
    f = [f for f in r.findings if f.code == 'G3_STIFF'][0]
    assert f.level == 'info' and f.remedy and 'tau grid' in f.remedy
    assert not _two_ou_codes(2.0, 5.0) & {'G3_STIFF', 'G3_CLOSE_POLES'}


def _two_ou_codes(a, b):
    return set(analyze(_two_ou(a, b), question={'k': 2, 'max_ell': 0}).codes)


def test_close_pole_pair_has_a_remedy():
    r = analyze(_two_ou(1.0, 1.01), question={'k': 2, 'max_ell': 0})
    assert r.traits.n_close_pole_pairs == 1 and r.traits.pole_cluster_status == 'harmless'
    f = [f for f in r.findings if f.code == 'G3_CLOSE_POLES'][0]
    assert f.level == 'info' and f.remedy
    # not close beyond the linkage (5 %)
    assert 'G3_CLOSE_POLES' not in _two_ou_codes(1.0, 1.2)


def test_repeated_pole_is_reported():
    r = analyze(_jordan(), question={'k': 2, 'max_ell': 0})
    assert r.traits.repeated_poles is True
    f = [f for f in r.findings if f.code == 'G3_REPEATED_POLE'][0]
    assert f.level == 'info' and f.remedy
    assert r.traits.pole_rate_min == pytest.approx(1.0) and r.traits.pole_rate_max == pytest.approx(1.0)


# ── B: the loop parameter estimate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('eps,D,mu', [(0.005, 1.0, 1.0), (0.02, 1.0, 1.0), (0.1, 0.5, 1.0), (0.05, 2.0, 2.0),
                                      (0.3, 0.1, 0.5)])
def test_geff_matches_the_closed_form_on_the_ou_quartic(eps, D, mu):
    """x' = -mu x - eps x^3 + noise(2D): sigma^2 = D/mu and g_eff = 3 eps sigma^2 / mu = 3 eps D / mu^2 (the
    measured |one-loop / tree| at tau = 0 is exactly this; reports/24_report.md).  The margin is the
    quadrature error of the variance integral."""
    g = _geff(eps, D, mu).traits.loop_parameter_estimate
    assert g == pytest.approx(3 * eps * D / mu ** 2, rel=2e-3)


def test_geff_is_monotone_in_the_coupling_and_the_noise():
    by_eps = [_geff(e).traits.loop_parameter_estimate for e in (0.001, 0.01, 0.1, 1.0, 10.0)]
    by_D = [_geff(0.02, D).traits.loop_parameter_estimate for D in (0.01, 0.1, 1.0, 10.0)]
    assert by_eps == sorted(by_eps) and len(set(by_eps)) == 5
    assert by_D == sorted(by_D) and len(set(by_D)) == 4
    by_mu = [_geff(0.02, 1.0, mu).traits.loop_parameter_estimate for mu in (0.25, 1.0, 4.0)]
    assert by_mu == sorted(by_mu, reverse=True)                     # stiffer = more perturbative


def test_geff_is_zero_for_a_linear_model_and_has_no_warning():
    r = analyze(_two_ou(1.0, 3.0), question={'k': 2, 'max_ell': 1})
    assert r.traits.loop_parameter_estimate == 0.0 and not r.has('G5_LOOP_PARAMETER')
    assert r.traits.tree_std == pytest.approx((1.0, 0.57735), rel=2e-3)


@pytest.mark.parametrize('w0,g', [(50.0, 0.5), (5.0, 1.0), (1.0, 3.0)])
def test_tree_std_resolves_a_narrow_resonance(w0, g):
    """A damped oscillator (x' = v, v' = -w0^2 x - 2 g v + noise(2)): sigma_x^2 = 1/(2 g w0^2), sigma_v^2 = 1/(2 g).
    The frequency grid adds nodes around the resonance (Q = w0/(2g) up to 50 here): within 1.5 %."""
    s = SDE('G5 oscillator')
    for n, v in (('w0', w0), ('g', g), ('D', 1.0)):
        s.parameter(n, default=v, domain='positive')
    osc = (s.field('x').field('v').drift('x', 'v - 1e-4*x').drift('v', '-w0^2*x - 2*g*v')
           .noise('W', Gaussian(var='2*D'), couples={'v': 1}).stability_analysis(True).build())
    std = analyze(osc, question={'k': 2, 'max_ell': 1}).traits.tree_std
    assert std == pytest.approx((math.sqrt(1 / (2 * g * w0 ** 2)), math.sqrt(1 / (2 * g))), rel=0.015)


def test_geff_warning_only_when_loops_are_requested_and_g_is_large():
    big = _geff(0.5)                                                # g ~ 1.5
    assert big.traits.loop_parameter_estimate > A._GEFF_WARN and big.has('G5_LOOP_PARAMETER')
    f = [f for f in big.findings if f.code == 'G5_LOOP_PARAMETER'][0]
    assert f.level == 'warning' and f.remedy and 'xt1 dx1^3' in f.message.replace('r_', '')
    assert not _geff(0.5, max_ell=0).has('G5_LOOP_PARAMETER')       # tree level: the series is not used
    assert not _geff(0.02).has('G5_LOOP_PARAMETER')
    assert big.ok                                                   # a warning, never an error


def test_strict_semantics_are_unchanged_by_the_new_warnings():
    """strict promotes warnings to errors (D1 semantics): the large-g warning then stops the analysis."""
    r = _geff(0.5, policy={'mode': 'strict'})
    assert not r.ok and 'G5_LOOP_PARAMETER' in [f.code for f in r.errors]
    assert _geff(0.02, policy={'mode': 'strict'}).ok
    # and Policy.downgrade-free permissive mode leaves the finding a warning
    assert 'G5_LOOP_PARAMETER' in [f.code for f in _geff(0.5).warnings]


def test_geff_detail_names_the_dominant_vertex():
    r = _geff(0.1)
    assert r.traits.loop_parameter_detail.startswith('r_xt1 dx1^3:')


def test_geff_state_dependent_noise_and_quadratic_vertices_contribute():
    """Multiplicative noise (r^2 x vertex) and a quadratic drift vertex (second order in the vertex) both count."""
    s = SDE('G5 multiplicative')
    s.parameter('mu', default=1.0, domain='positive')
    s.parameter('s1', default=0.5, domain='positive')
    mult = (s.field('x').drift('x', '-mu*(x - 2)').noise('W', Gaussian(var='1 + s1*x'), couples={'x': 1})
            .stability_analysis(True).build())
    g_mult = analyze(mult, question={'k': 2, 'max_ell': 1}, policy={'interpretation': 'ito'}).traits.loop_parameter_estimate
    assert g_mult is not None and g_mult > 0
    quad = _nl_ou('-mu*x + b*x^2 + 1', dict(mu=1.0, b=0.1), 0.5)
    q = analyze(quad, question={'k': 2, 'max_ell': 1}).traits
    assert q.loop_parameter_estimate > 0
    quad2 = _nl_ou('-mu*x + b*x^2 + 1', dict(mu=1.0, b=0.2), 0.5)
    assert analyze(quad2, question={'k': 2, 'max_ell': 1}).traits.loop_parameter_estimate > q.loop_parameter_estimate


def test_fields_with_a_delta_part_have_no_tree_variance_and_g_eff_is_a_lower_bound():
    """A delta part of G (spike-count fields) makes the equal-time variance infinite: the cutoff would set the number,
    so tree_std is NaN there and the vertices touching those fields are left out (g_eff is a lower bound)."""
    model, mod = _load('linear_hawkes')
    r = analyze(model, question={'k': 2, 'max_ell': 1})
    std = np.asarray(r.traits.tree_std)
    assert r.has('G3_DELTA_COMPONENT') and np.isnan(std).sum() == 2 and np.isfinite(std).sum() == 2
    assert r.traits.loop_parameter_complete is True                  # nothing it could estimate was left out
    model, mod = _load('dendritic_linear_gate')
    r = analyze(model, question={'k': 2, 'max_ell': 1, 'parameters': getattr(mod, 'DEFAULT_FUNDAMENTAL', None) or None})
    assert r.traits.loop_parameter_complete is False and not r.has('G5_LOOP_PARAMETER')
    assert 'g_eff >= ' in r.table() and any('lower bound' in f.message for f in r.findings if f.code == 'G5_ADVISORY_SKIPPED')


def test_noiseless_nonpolynomial_model_has_no_internal_error():
    s = SDE('G5 noiseless')
    s.parameter('a', default=1.0, domain='positive')
    m = s.field('x').drift('x', '-a*x - x**3 + 1/(1+exp(-x))').build()
    r = analyze(m, question={'k': 2, 'max_ell': 1})
    assert 'INTERNAL_ERROR' not in r.codes and r.has('G4_NO_NOISE') and r.ok
    assert r.traits.loop_parameter_estimate is None and r.traits.taylor_radius_ratio is None


def test_table_header_is_truncated_to_100_columns():
    s = SDE('G5 ' + 'a very long model name ' * 6)
    for n, v in (('mu', 1.0), ('eps', 0.5), ('D', 1.0)):
        s.parameter(n, default=v, domain='positive')
    m = s.field('x').drift('x', '-mu*x - eps*x^3').noise('W', Gaussian(var='2*D'), couples={'x': 1}) \
        .stability_analysis(True).build()
    txt = analyze(m, question={'k': 2, 'max_ell': 1}).table()
    _lines_ok(txt)
    assert txt.splitlines()[0].count('...') == 1 and txt.splitlines()[0].endswith(']')


def test_g_eff_is_not_estimated_at_tree_level_and_the_table_says_why():
    """max_ell = 0 expands to order 2: the vertices are not in the expansion, so g_eff would read 0 and look
    harmless; it is left out (and no warning) unless the Taylor order is raised."""
    r = _geff(0.5, max_ell=0)
    assert r.traits.loop_parameter_estimate is None and not r.has('G5_LOOP_PARAMETER')
    assert 'tree level: the vertices are not expanded' in ' '.join(r.table().split())
    model, par = _ou_quartic(eps=0.5)
    r = analyze(model, question={'k': 2, 'max_ell': 0, 'parameters': par, 'taylor_order': 4})
    assert r.traits.loop_parameter_estimate == pytest.approx(1.5, rel=2e-3)


def test_taylor_order_in_the_question_reaches_the_report():
    drift, pars = SIGMOID
    r = analyze(_nl_ou(drift, pars, 0.1), question={'k': 2, 'max_ell': 1, 'taylor_order': 8}, policy=_TAYLOR_POLICY)
    assert r.traits.taylor_order == 8


def test_no_g5_findings_without_a_usable_tree():
    """Spatial models and colored noise are skipped quietly (spatial) or with an info (colored kernel)."""
    sp, mod = _load('edwards_wilkinson_1d')
    r = analyze(sp, question={'parameters': getattr(mod, 'DEFAULT_FUNDAMENTAL', None) or None})
    assert r.traits.loop_parameter_estimate is None and not [f for f in r.findings if f.gate == 'G5']


# ── C: the Taylor-series check ────────────────────────────────────────────────────────────────

def _taylor(family, D, max_ell=1):
    drift, pars = family
    return analyze(_nl_ou(drift, pars, D), question={'k': 2, 'max_ell': max_ell}, policy=_TAYLOR_POLICY)


def test_taylor_estimator_against_a_known_radius_of_convergence():
    """a/(1 + x/L): singularity at x = -L, so R = x* + L; the estimate q must be s/R with s = 3 sigma.  The
    successive-term ratio is exact for a pole (geometric series): within 10 %."""
    r = _taylor(RATIONAL, 0.1)
    t = r.traits
    xstar = dict(t.saddle)['xstar'][0]
    R = xstar + RATIONAL[1]['L']
    expected = 3 * t.tree_std[0] / R
    assert t.taylor_radius_ratio == pytest.approx(expected, rel=0.1)


def test_taylor_estimate_for_a_complex_pair_is_within_a_factor_two():
    """The logistic function has a complex-conjugate pair of singularities at distance R = |x* - c + i pi|; the
    ratio estimate q underestimates s/R by about 2 (documented in _taylor_ratio), hence the 0.3 threshold."""
    t = _taylor(SIGMOID, 0.1).traits
    xstar = dict(t.saddle)['xstar'][0]
    R = math.hypot(xstar - SIGMOID[1]['c'], math.pi)
    true = 3 * t.tree_std[0] / R
    assert 0.35 * true < t.taylor_radius_ratio < 1.1 * true


@pytest.mark.parametrize('family', [SIGMOID, RATIONAL])
def test_taylor_ratio_is_monotone_in_the_noise_and_warns_where_unreliable(family):
    ratios = []
    for D in (0.005, 0.05, 0.5, 2.0):
        r = _taylor(family, D)
        ratios.append(r.traits.taylor_radius_ratio)
        assert r.has('G5_TAYLOR_RADIUS') == (r.traits.taylor_radius_ratio > A._TAYLOR_RATIO_WARN)
    assert ratios == sorted(ratios) and len(set(ratios)) == 4
    assert ratios[0] < A._TAYLOR_RATIO_WARN < ratios[-1]


def test_taylor_warning_has_level_remedy_and_names_the_scale():
    r = _taylor(SIGMOID, 2.0)
    f = [f for f in r.findings if f.code == 'G5_TAYLOR_RADIUS'][0]
    assert f.level == 'warning' and f.remedy and 'unreliable' in f.message and "'dx1'" in f.message
    assert FINDING_CODES['G5_TAYLOR_RADIUS'] == ('G5', 'warning')


def test_taylor_check_is_absent_for_polynomial_models():
    r = _geff(0.02)
    assert r.traits.polynomial and r.traits.taylor_radius_ratio is None and not r.has('G5_TAYLOR_RADIUS')


# ── D: no number moves ────────────────────────────────────────────────────────────────────────

def test_the_analyzer_does_not_change_the_numbers():
    model, par = _ou_quartic(eps=0.05)
    kw = dict(k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)], parameters=par, tau_grid=np.array([0.0, 1.0, 2.5]),
              parallel=False, verbose=False)
    on = C.compute_cumulants(model, **kw)
    off = C.compute_cumulants(model, preflight=False, **kw)
    for ell in on['C_tau_by_ell']:
        assert np.array_equal(np.asarray(on['C_tau_by_ell'][ell]), np.asarray(off['C_tau_by_ell'][ell]))
    t = on['analysis'].traits
    assert t.loop_parameter_estimate == pytest.approx(0.15, rel=0.02) and off.get('analysis') is None
    assert on['analysis'].diagnostics['n_diagrams'] > 0
    # the wired and the standalone estimate agree (same definition, same inputs)
    stand = _geff(0.05).traits.loop_parameter_estimate
    assert t.loop_parameter_estimate == pytest.approx(stand, rel=1e-6)


# ── E: the report table ───────────────────────────────────────────────────────────────────────

GOLDEN_OU = """\
Model report: OU Quartic (white noise)                                    [OK; 0 warning(s), 0 info]
----------------------------------------------------------------------------------------------------
question   k=2  max_ell=1  external=[('x', 1), ('x', 1)]
route      unrouted
model      temporal; 1 physical / 1 response component(s); 4 parameter(s); Taylor order 4;
           time-homogeneous yes
vertices   polynomial yes; species [(1, 2), (1, 3)]; max degree 4; state-dependent noise no
noise      white; Gaussian yes; cross-correlated no
saddle     1 found, 1 stable; selected #0 stable yes; marginal modes 0
poles      n=1; rate min/max 1 / 1; fast/slow 1; repeated no; close pairs 0; delta part no
regime     g_eff 0.06 (small, warn > 0.2)
cost       analysis 12.3 ms
----------------------------------------------------------------------------------------------------
findings   none"""

GOLDEN_SIGMOID = """\
Model report: G5 nonlinear OU                                             [OK; 1 warning(s), 1 info]
----------------------------------------------------------------------------------------------------
question   k=2  max_ell=1
route      unrouted
model      temporal; 1 physical / 1 response component(s); 5 parameter(s); Taylor order 4;
           time-homogeneous yes
vertices   polynomial no; species [(1, 2), (1, 3)]; max degree 4; state-dependent noise no
noise      white; Gaussian yes; cross-correlated no
saddle     1 found, 1 stable; selected #0 stable yes; marginal modes 0
poles      n=1; rate min/max 0.595 / 0.595; fast/slow 1; repeated no; close pairs 0; delta part no
regime     g_eff 0.135 (small, warn > 0.2); Taylor term ratio at 3 std 0.448
cost       analysis 12.3 ms
----------------------------------------------------------------------------------------------------
findings   error 0 / warning 1 / info 1
  W        G5_TAYLOR_RADIUS: Taylor truncation (order 4) of the response in 'dx1' is unreliable a...
  I        G1_NONPOLYNOMIAL: the action is not polynomial in the fields: 1/(e^(c - dx1 - xstar1)..."""


def _fixed_time(rep):
    """The only non-deterministic entry of the table is the analyzer's own wall time."""
    return replace(rep, traits=replace(rep.traits, analysis_time_s=0.0123))


def _lines_ok(text):
    assert all(len(ln) <= TABLE_WIDTH for ln in text.splitlines()), [ln for ln in text.splitlines()
                                                                      if len(ln) > TABLE_WIDTH]


def test_table_golden_text_ou_quartic():
    rep = _geff(0.02, max_ell=1)
    rep = replace(rep, question=replace(rep.question, external_fields=(('x', 1), ('x', 1))))
    txt = _fixed_time(rep).table()
    assert txt == str(_fixed_time(rep))
    _lines_ok(txt)
    assert txt == GOLDEN_OU


def test_table_golden_text_flags_the_advisories():
    txt = _fixed_time(_taylor(SIGMOID, 0.5)).table()
    _lines_ok(txt)
    assert txt == GOLDEN_SIGMOID


@pytest.mark.parametrize('name', ['ou_quartic', 'ou_quartic_two_dim', 'multipopulation_test',
                                  'allen_cahn_1d_subcritical_infinite', 'reaction_diffusion_2d', 'linear_hawkes',
                                  'dendritic_linear_gate'])
def test_table_fits_100_columns_on_tracked_models(name):
    model, mod = _load(name)
    rep = analyze(model, question={'parameters': getattr(mod, 'DEFAULT_FUNDAMENTAL', None) or None})
    txt = rep.table()
    _lines_ok(txt)
    assert txt.splitlines()[0].startswith('Model report:') and '\nroute ' in txt and '\nfindings ' in txt


def test_dd_run_verbose_prints_the_table(capsys):
    import daedalus as dd
    model, mod = dd.load_model('ou_quartic')
    res = dd.run(model, dd.Config(k=2, max_ell=0, tau_grid=(-1, 1, 3), verbose=True, parallel=False), mod)
    out = capsys.readouterr().out
    assert 'Model report: OU Quartic' in out and 'phase_j' in out and 'g_eff' in out
    assert res['analysis'].table() in out
    dd.run(model, dd.Config(k=2, max_ell=0, tau_grid=(-1, 1, 3), verbose=False, parallel=False), mod)
    assert 'Model report:' not in capsys.readouterr().out


def test_table_of_a_failed_analysis_and_of_a_wired_run():
    bad = analyze({'not': 'a model'})
    assert 'ERROR' in bad.table().splitlines()[0]
    model, par = _ou_quartic()
    res = C.compute_cumulants(model, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)], parameters=par,
                              tau_grid=np.array([0.0]), parallel=False, verbose=False)
    t = res['analysis'].table()
    assert 'phase_j_' in t and 'diagrams' in t and 'accuracy estimate' in t

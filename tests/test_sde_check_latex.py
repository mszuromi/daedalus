"""SDE.check(), SDE.latex() and the caret parse errors of the SDE front end (api/sde.py, api/sde_latex.py).

Presentation and diagnostics only: the derived action, equations and built model are unchanged (tests/
test_declared_operators.py keeps the byte-identical snapshot of the whole catalogue); here the reports read them.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('sage.all')

from api.noise import Gaussian, NoiseError, Poisson                           # noqa: E402
from api.operators import Laplacian, TimeDerivative                           # noqa: E402
from api.sde import SDE, SDEError, SDEReport                                  # noqa: E402

from tests import _sde_model_catalogue as C                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _allen_cahn():
    return (SDE('Allen-Cahn', space=1)
            .operator('dt', TimeDerivative()).operator('lap', Laplacian())
            .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
            .parameter('lam', default=0.1, domain='positive').parameter('T', default=1.0, domain='positive')
            .field('phi')
            .equation('phi', lhs='dt(phi)', rhs='D*lap(phi) - mu*phi - lam*phi^3')
            .noise('eta', Gaussian(var='2*T', space='white'), into={'phi': 1}))


def _rows(report):
    return {(r['response_legs'], r['field_legs'], r['monomial']): r for r in report['vertices']['rows']}


# ── check() ─────────────────────────────────────────────────────────────────────────────────────

def test_check_ou_quartic(capsys):
    m = C.ou_quartic()
    action, eqs = m.action_text(), m.equations()
    r = m.check()
    out = capsys.readouterr().out
    assert isinstance(r, SDEReport) and out.strip() == str(r) == r.text()
    assert r['action']['text'] == action and r['mean_field'] == [dict(e, field='x') for e in eqs]
    assert m.action_text() == action and m.equations() == eqs                 # read, not changed
    decl = r['declarations']
    assert [f['name'] for f in decl['fields']] == ['x'] and decl['fields'][0]['response'] == 'xt'
    assert [p['name'] for p in decl['parameters']] == ['mu', 'eps', 'D']
    assert decl['operators'] == {'Dt': 'TimeDerivative'}
    kinds = [(t['kind'], t.get('field') or t.get('source')) for t in r['action']['terms']]
    assert kinds == [('dynamics', 'x'), ('noise', 'xi')]
    noise = r['action']['terms'][1]
    assert noise['source_type']['type'] == 'Gaussian' and noise['source_type']['params']['var'] == '2*D'
    assert noise['text'] == '-(D*xt[i]^2)'
    assert r['corrections'] == []
    # the vertex table: eps*x^3 gives (1 response, 2 field) 3*eps*xstar and (1, 3) eps
    rows = _rows(r)
    assert rows[(1, 2, 'xt1*dx1^2')]['coefficient'] == '3*eps*xstar1'
    assert rows[(1, 3, 'xt1*dx1^3')]['coefficient'] == 'eps'
    assert rows[(1, 3, 'xt1*dx1^3')]['degree'] == 4 and len(rows) == 2
    assert rows[(1, 2, 'xt1*dx1^2')]['loops'] == [1] and rows[(1, 2, 'xt1*dx1^2')]['min_loop'][1] == 1
    assert rows[(1, 2, 'xt1*dx1^2')]['min_loop'][3] == 0
    p = r['propagator']
    assert p['class'] == 'rational' and p['n_poles'] == 1 and p['stable'] is True
    assert p['noise_matrix'] == {'xt1^2': '-D'} and 'Dt' in p['linear_operator']['xt1*dx1']
    assert not [w for w in r['warnings'] if w['level'] != 'info']
    assert 'Vertices (Taylor order 4' in out and '3*eps*xstar1' in out


def test_check_runs_no_phase_j(monkeypatch):
    import api.compute

    def boom(*a, **k):
        raise AssertionError('check() must not integrate')
    monkeypatch.setattr(api.compute, 'compute_cumulants', boom)
    assert C.ou_quartic().check(verbose=False)['vertices'] is not None


def test_check_quiet_prints_nothing(capsys):
    C.poisson_driven().check(verbose=False)
    assert capsys.readouterr().out == ''


def test_check_stratonovich_drift_correction():
    m = C.stratonovich_multiplicative()
    r = m.check(verbose=False)
    (c,) = r['corrections']
    assert (c['kind'], c['rule'], c['source'], c['field']) == ('ito_drift', 'stratonovich', 'W', 'x')
    assert c['text'] == '(1/2)*s1^2*x[i] + (1/2)*s0*s1'
    assert "drift of 'x' += " in c['meaning']
    # the correction is in the action's drift and the multiplicative noise is a (2, 1) vertex
    assert '(1/2)*s1^2*x[i]' in r['action']['terms'][0]['text']
    rows = _rows(r)
    assert rows[(2, 1, 'xt1^2*dx1')]['coefficient'] == '-s1^2*xstar1 - s0*s1'
    assert rows[(2, 1, 'xt1^2*dx1')]['kind'] == 'interaction' and r['interpretation'] == 'stratonovich'
    assert not any(w['code'] == 'G0_CONVENTION_MISSING' for w in r['warnings'])


def test_check_two_field_stratonovich_sources_are_labelled():
    m = (SDE('two noises').parameter('mu', default=1.0).parameter('s', default=0.3).field('x')
         .drift('x', '-mu*x')
         .noise('A', Gaussian(var=1), couples={'x': '1 + s*x'})
         .noise('B', Gaussian(var=1), couples={'x': 's*x'})
         .interpretation('stratonovich'))
    r = m.check(verbose=False)
    assert sorted(c['source'] for c in r['corrections']) == ['A', 'B']
    assert all(c['rule'] == 'stratonovich' and c['field'] == 'x' for c in r['corrections'])


def test_check_marcus_exact_jump_model():
    r = C.marcus_affine().check(verbose=False)
    (c,) = r['corrections']
    assert (c['kind'], c['rule'], c['source'], c['route'], c['order']) == ('marcus_noise', 'marcus', 'L',
                                                                          'exact', 16)
    assert [w['code'] for w in r['warnings'] if w['code'].startswith('MARCUS')] == ['MARCUS_EXACT_ORDER']
    t = [t for t in r['action']['terms'] if t['kind'] == 'noise'][0]
    assert t['source'] == 'L' and t['source_type']['type'] == 'CompoundPoisson'
    mf = [w for w in r['warnings'] if w['code'] == 'MEAN_FIELD_ONLY']
    assert len(mf) == 1 and mf[0]['source'] == 'L' and mf[0]['field'] == 'x'
    rows = _rows(r)
    assert rows[(3, 0, 'xt1^3')]['kind'] == 'noise cumulant'
    assert rows[(2, 1, 'xt1^2*dx1')]['loops'] == [1]


def test_check_marcus_series_warns_about_the_truncation():
    r = C.marcus_rotation().check(verbose=False)
    (w,) = [w for w in r['warnings'] if w['code'] == 'MARCUS_SERIES_TRUNCATED']
    assert w['level'] == 'warning' and w['source'] == 'L' and 'order 7' in w['message']
    assert r['corrections'][0]['route'] == 'series' and r['corrections'][0]['order'] == 7
    assert len([w for w in r['warnings'] if w['code'] == 'MEAN_FIELD_ONLY']) == 2     # x and y
    assert r['propagator']['n_poles'] == 2


def test_check_population_model():
    r = C.linear_hawkes().check(verbose=False)
    decl = r['declarations']
    assert [f['name'] for f in decl['fields']] == ['n', 'v'] and decl['fields'][0]['exposed_source'] == 'spikes'
    assert decl['functions'] == [{'name': 'phi', 'args': ['v'], 'expression': 'a * v'}]
    assert decl['kernels'][0]['name'] == 'g' and 'freq_image' in decl['kernels'][0]
    assert decl['populations'][0]['size'] == 2
    assert [t['kind'] for t in r['action']['terms']] == ['dynamics', 'noise']
    assert r['action']['terms'][1]['source'] == 'spikes' and r['action']['terms'][1]['exposed_as'] == 'n'
    rows = _rows(r)
    assert rows[(2, 1, 'nt1^2*dv1')]['coefficient'] == '-(1/2)*a'
    assert rows[(3, 0, 'nt2^3')]['kind'] == 'noise cumulant'
    assert r['propagator']['n_poles'] == 4 and r['propagator']['class'] == 'rational'
    assert set(r['propagator']['saddle']) == {'nstar', 'vstar'}
    assert [e['field'] for e in r['mean_field']] == ['n', 'v']


def test_check_spde_through_the_bridge():
    m = _allen_cahn()
    r = m.check(verbose=False)
    assert r['space'] == 1 and r['declarations']['operators'] == {'dt': 'TimeDerivative', 'lap': 'Laplacian'}
    assert 'Laplacian' in r['action']['terms'][0]['text']
    assert r['mean_field'][0]['field'] == 'phi' and r['mean_field'][0]['population'] is None
    rows = _rows(r)
    assert rows[(1, 2, 'phit1*dphi1^2')]['coefficient'] == '3*lam*phistar1'
    assert rows[(1, 3, 'phit1*dphi1^3')]['coefficient'] == 'lam'
    p = r['propagator']
    assert p['class'] == 'spatial' and p['spatial'] is True and 'Laplacian' in p['linear_operator']['phit1*dphi1']
    assert 'space dimension 1' in str(r)


def test_check_reports_a_missing_parameter_instead_of_raising():
    m = (SDE('no defaults').parameter('mu').field('x').drift('x', '-mu*x')
         .noise('xi', Gaussian(var=1), couples={'x': 1}))
    r = m.check(verbose=False)
    assert r['mean_field'] and r['action']['text'] == m.action_text()
    (w,) = [w for w in r['warnings'] if w['level'] == 'error']
    assert w['code'] == 'G0_PARAMETERS_MISSING' and 'mu' in w['message']
    assert r['propagator']['class'] is None and r['vertices']['rows'] == []
    r2 = m.check(verbose=False, params={'mu': 2.0})
    assert r2['propagator']['class'] == 'rational' and r2['propagator']['stable'] is True
    assert not [w for w in r2['warnings'] if w['level'] == 'error']


def test_check_params_override_the_saddle():
    r = C.ou_quartic().check(verbose=False, params={'mu': 2.0})
    assert r['propagator']['stable'] is True and r['mean_field']


def test_check_and_latex_leave_every_catalogue_model_byte_identical():
    for name, make in C.CATALOGUE.items():
        m = make()
        a, e = m.action_text(), m.equations()
        m.latex()
        assert (m.action_text(), m.equations()) == (a, e), name


# ── latex() ─────────────────────────────────────────────────────────────────────────────────────

OU_SDE = (r'\begin{aligned}' '\n'
          r'\mathrm{d}x &= \left(-\varepsilon x^{3} - \mu x\right)\,\mathrm{d}t + \sqrt{2 \, D}\,\mathrm{d}W_{\xi}' '\n'
          r'\end{aligned}')
OU_ACTION = (r'\begin{aligned}' '\n'
             r'S &= \int \mathrm{d}t\,\Big[ \tilde{x}\left(\partial_t x - \left(-\varepsilon x^{3} - \mu x\right)\right) \\' '\n'
             r'&\qquad - D \tilde{x}^{2} \Big]' '\n'
             r'\end{aligned}')
OU_MF = (r'\begin{aligned}' '\n'
         r'\mu x_{\ast} &= -\varepsilon x_{\ast}^{3}' '\n'
         r'\end{aligned}')


def test_latex_ou_quartic():
    L = C.ou_quartic().latex()
    assert list(L) == ['sde', 'noise', 'action', 'mean_field'] and all(isinstance(v, str) for v in L.values())
    assert L['sde'] == OU_SDE
    assert L['action'] == OU_ACTION
    assert L['mean_field'] == OU_MF
    assert r'\langle \dot{W}_{\xi}(t)\,\dot{W}_{\xi}(t^{\prime}) \rangle &= 2 \, D\,\delta(t - t^{\prime})' in L['noise']


def test_latex_declared_operators_and_spde():
    L = _allen_cahn().latex()
    assert L['sde'] == (r'\begin{aligned}' '\n'
                        r'\mathrm{d}\phi &= \left(-\lambda \phi^{3} - \mu \phi + D \nabla^{2} \phi\right)'
                        r'\,\mathrm{d}t + \sqrt{2 \, T}\,\mathrm{d}W_{\eta}' '\n'
                        r'\end{aligned}')
    assert r'\int \mathrm{d}t\,\mathrm{d}^{1}x' in L['action'] and r'\partial_t \phi' in L['action']
    assert L['mean_field'] == (r'\begin{aligned}' '\n' r'\mu \phi_{\ast} &= -\lambda \phi_{\ast}^{3}' '\n'
                               r'\end{aligned}')
    assert r'\delta(x - x^{\prime})\,\delta(t - t^{\prime})' in L['noise']
    # renaming the time derivative and the call style change nothing in the typeset model
    other = (SDE('Allen-Cahn', space=1).operator('Δ', Laplacian())
             .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
             .parameter('lam', default=0.1, domain='positive').parameter('T', default=1.0, domain='positive')
             .field('phi')
             .equation('phi', lhs='Dt*phi', rhs='(D*Δ - mu)*phi - lam*phi^3')
             .noise('eta', Gaussian(var='2*T', space='white'), into={'phi': 1}))
    assert other.latex() == L


def test_latex_population_model_with_a_function_and_a_kernel():
    L = C.linear_hawkes().latex()
    assert L['sde'] == (r'\begin{aligned}' '\n'
                        r'\tau \partial_t v_{i} + v_{i} &= E_{i} + \sum_{j=1}^{N}\left[g w_{ij} n_{j}\right]' '\n'
                        r'\end{aligned}')
    assert r'\varphi\left(v\right)' in L['noise'] and r'\text{Poisson}' in L['noise']
    assert r'\sum_{i=1}^{N}\left[' in L['action'] and r'\varphi_{i}\left(v_{i}\right)' in L['action']
    assert L['mean_field'].count(r'\\') == 1 and r'n_{\ast,i} &= \varphi_{i}\left(v_{\ast,i}\right)' in L['mean_field']


def test_latex_stratonovich_marcus_delay_and_noise_kinds():
    assert (r'\left(s_{1} x + s_{0}\right)\,\circ\,\mathrm{d}W_{W}'
            in C.stratonovich_multiplicative().latex()['sde'])                       # not read as Ito
    assert r'\circ\,\dot{W}_{W}' in C.stratonovich_tau_lhs().latex()['sde']
    assert r'\diamond\,\mathrm{d}N_{L}' in C.marcus_rotation().latex()['sde']
    assert r'\circ' not in C.ou_quartic().latex()['sde'] and r'\diamond' not in C.poisson_driven().latex()['sde']
    jump = C.marcus_affine().latex()
    assert r'\mathrm{d}N_{L}' in jump['sde'] and r'\text{CompoundPoisson}' in jump['noise']
    assert (r'\text{jump}=\text{Uniform}(\mathrm{lo}=0, \mathrm{hi}=m)'
            in C.uniform_jumps().latex()['noise'])
    assert r'\mathrm{d}L_{G}' in C.gamma_process().latex()['sde']
    assert r'c x(t - d)' in C.delay_ou().latex()['sde']


def _spde_with_lap_of_square(rhs='D*lap(u) - mu*u - lap(u^2)', lhs='dt(u)'):
    return (SDE('m', space=1).operator('dt', TimeDerivative()).operator('lap', Laplacian())
            .parameter('D', default=1.0, domain='positive').parameter('mu', default=1.0, domain='positive')
            .field('u').equation('u', lhs=lhs, rhs=rhs)
            .noise('eta', Gaussian(var=2, space='white'), into={'u': 1}))


def test_latex_laplacian_of_an_expression_is_grouped_and_never_falls_back():
    L = _spde_with_lap_of_square().latex()
    assert L['sde'] == (r'\begin{aligned}' '\n'
                        r'\partial_t u &= -\mu u + D \nabla^2 u - \nabla^2 \left(u^{2}\right) + \sqrt{2}\,\dot{W}_{\eta}' '\n'
                        r'\end{aligned}')
    assert r'- \nabla^2 \left(u^{2}\right)\right)\right)' in L['action']
    assert not any('texttt' in v for v in L.values())
    # product-style operators on a model that also has an operator on an expression
    mixed = _spde_with_lap_of_square(lhs='(dt + mu)*u', rhs='D*lap(u) - lap(u^2)').latex()
    assert not any('texttt' in v for v in mixed.values())
    assert r'\mu u + \partial_t u' in mixed['sde'] or r'\partial_t u + \mu u' in mixed['sde']


def test_latex_without_equations_or_zero_rhs_is_well_formed():
    assert C.readme_hawkes().latex()['sde'] == ''                       # a model of point processes only
    m = (SDE('z').parameter('tau', default=2.0).parameter('c', default=0.5).field('x')
         .equation('x', lhs='(tau*Dt + 1)*x', rhs='0').noise('W', Gaussian(var=1), couples={'x': 'c'}))
    sde = m.latex()['sde']
    assert '= 0 +' not in sde and r'&= c\,\dot{W}_{W}' in sde


def test_latex_floats_print_as_written():
    m = (SDE('f').parameter('mu', default=1.0).field('x').drift('x', '-0.3*mu*x + 1e-3 + 0.01/x')
         .noise('xi', Gaussian(var=2), couples={'x': 1}))
    sde = m.latex()['sde']
    assert '0.3 \\mu x' in sde and '0.001' in sde and r'\frac{0.01}{x}' in sde and '0000' not in sde


def test_check_shows_the_jump_law_of_a_compound_poisson_source():
    r = C.uniform_jumps().check(verbose=False)
    t = [t for t in r['action']['terms'] if t['kind'] == 'noise'][0]
    assert t['source_type']['jump'] == {'type': 'Uniform', 'params': {'lo': '0', 'hi': 'm'}}
    assert 'jump=Uniform(lo=0, hi=m)' in str(r)


@pytest.mark.parametrize('kw', [{'k': 0}, {'k': -1}, {'k': 2.7}, {'max_ell': -1}, {'max_ell': 1.5}, {'k': True}])
def test_check_refuses_a_nonsense_question(kw):
    with pytest.raises(SDEError, match='must be an integer'):
        C.ou_quartic().check(verbose=False, **kw)


def test_latex_never_raises_and_never_falls_back_on_the_catalogue():
    for name, make in C.CATALOGUE.items():
        L = make().latex()
        assert not any('texttt' in v for v in L.values()), name
        for v in L.values():
            assert v == '' or v.startswith(r'\begin{aligned}'), name
            assert v.count('{') == v.count('}'), name


def test_latex_is_deterministic_across_processes():
    code = ('import sys; sys.path.insert(0, %r)\n'
            'from tests import _sde_model_catalogue as C\n'
            'from sage.all import var; var("zeta alpha")\n'
            'for n in ("linear_hawkes", "ou_quartic_drift", "marcus_rotation"):\n'
            '    L = C.CATALOGUE[n]().latex()\n'
            '    print(n, L)\n') % str(ROOT)
    outs = []
    for seed in ('1', '2'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=ROOT,
                              timeout=600)
        assert done.returncode == 0, done.stderr[-2000:]
        outs.append(done.stdout.splitlines()[-3:])
    assert outs[0] == outs[1]
    assert outs[0][1].startswith('ou_quartic_drift') and r'\\mathrm{d}x &= \\left(-\\varepsilon x^{3}' in outs[0][1]


# ── caret parse errors ──────────────────────────────────────────────────────────────────────────

def _base():
    return (SDE('t').operator('dt', TimeDerivative())
            .parameter('mu', default=1.0).parameter('eps', default=0.1).parameter('D', default=1.0).field('x'))


def _caret_lines(exc):
    lines = str(exc).splitlines()
    k = next(i for i, l in enumerate(lines) if l.strip() == '^')
    return lines[k - 1][4:], lines[k].index('^') - 4, lines


def test_unbalanced_parenthesis_has_a_caret_under_the_failing_character():
    with pytest.raises(SDEError) as ei:
        _base().drift('x', '-mu*x + )eps*x^3')
    text, col, lines = _caret_lines(ei.value)
    assert lines[0].startswith("equation('x') rhs: cannot parse the expression: unmatched ')'")
    assert text == '-mu*x + )eps*x^3' and text[col] == ')'
    assert any('hint:' in l for l in lines)


@pytest.mark.parametrize('expr, char, word', [
    ('-mu*x +', '+', 'ends after the operator'),
    ('mu x', 'x', 'missing operator'),
    ('2x', 'x', 'runs into a name'),
    ('exp(-x', '(', 'never closed'),
    ('mu*x $ 3', '$', 'unexpected character'),
    ('mu**/x', '/', 'two operators in a row'),
    ('-lambda*x', 'l', 'Python keyword'),
    ('x = 3', '=', 'assignment'),
    ('2(x)', '(', 'missing operator'),
    ('sum(w[i,j]*x[j] j in E)', 'i', "needs 'for'"),
    ('sum(w[i,j]*x[j] for j E)', 'E', "needs 'in'"),
    ('Conv(g x)', 'x', 'missing operator'),
])
def test_typical_syntax_mistakes_point_at_the_problem(expr, char, word):
    with pytest.raises(SDEError, match=word) as ei:
        _base().drift('x', expr)
    text, col, _ = _caret_lines(ei.value)
    assert text == expr and text[col] == char


def test_errors_name_the_field_or_the_noise_they_belong_to():
    with pytest.raises(SDEError, match=r"equation\('x'\) lhs"):
        _base().equation('x', lhs='dt(x', rhs='-mu*x')
    with pytest.raises(SDEError, match=r"noise\('xi'\) coefficient for 'x'"):
        _base().noise('xi', Gaussian(var='2*D'), couples={'x': '2 D'})
    with pytest.raises(SDEError, match=r"define_function\('phi'\) expression"):
        _base().define_function('phi', ['v'], 'a*v +')
    with pytest.raises(SDEError, match=r"point_process\('n'\) rate"):
        _base().point_process('n', 'exp(')


def test_undeclared_name_has_a_caret_and_a_suggestion():
    with pytest.raises(SDEError) as ei:
        _base().drift('x', '-mu*x - epsilon*x^3').noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build()
    text, col, lines = _caret_lines(ei.value)
    assert "equation('x') rhs: undeclared symbol 'epsilon'" in lines[0]
    assert text[col:col + 7] == 'epsilon' and "did you mean 'eps'?" in str(ei.value)
    with pytest.raises(SDEError) as ei:
        _base().drift('x', '-mu*x - sinn(x)').noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build()
    assert 'define_function' in str(ei.value) and _caret_lines(ei.value)[0][_caret_lines(ei.value)[1]:][:4] == 'sinn'


def test_undeclared_name_in_a_noise_parameter_names_the_noise_and_the_parameter():
    with pytest.raises(SDEError) as ei:
        _base().drift('x', '-mu*x').noise('xi', Gaussian(var='2*Dd'), couples={'x': 1}).build()
    text, col, lines = _caret_lines(ei.value)
    assert lines[0].startswith("noise('xi') Gaussian(var=...): undeclared symbol 'Dd'")
    assert text == '2*Dd' and text[col:] == 'Dd' and "did you mean 'D'?" in str(ei.value)


def test_operator_misuse_is_pointed_at():
    with pytest.raises(SDEError, match='argument of a function') as ei:
        (_base().equation('x', lhs='dt(x)', rhs='-mu*exp(dt)')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    text, col, _ = _caret_lines(ei.value)
    assert text == '-mu*exp(dt)' and text[col:col + 2] == 'dt'
    with pytest.raises(SDEError, match='operators act in the equations only') as ei:
        (_base().equation('x', lhs='dt(x)', rhs='-mu*x')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 'dt*2'}).build())
    text, col, _ = _caret_lines(ei.value)
    assert text == 'dt*2' and col == 0 and 'hint' in str(ei.value)
    sp = (SDE('s', space=1).operator('dt', TimeDerivative()).operator('lap', Laplacian())
          .parameter('mu', default=1.0).parameter('D', default=1.0).parameter('T', default=1.0).field('u'))
    with pytest.raises(SDEError, match=r"operator 'lap' is the argument of the function 'exp'") as ei:
        (sp.equation('u', lhs='dt(u)', rhs='D*exp(lap) - mu*u')
         .noise('eta', Gaussian(var='2*T', space='white'), into={'u': 1}).build())
    text, col, _ = _caret_lines(ei.value)
    assert text == 'D*exp(lap) - mu*u' and text[col:col + 3] == 'lap'
    assert 'lap(u)' in str(ei.value)


def test_noise_source_parse_errors_have_carets_and_keep_their_type():
    with pytest.raises(NoiseError, match='cannot parse') as ei:
        Gaussian(var='2*')
    text, col, _ = _caret_lines(ei.value)
    assert text == '2*' and text[col] == '*'


def test_error_types_are_unchanged():
    assert issubclass(SDEError, NoiseError) and issubclass(SDEError, ValueError)
    for bad in (lambda: _base().drift('x', '-mu*x +'),
                lambda: _base().noise('xi', Gaussian(var='2*D'), couples={'x': '1 +'}),
                lambda: _base().define_kernel('g', time_expr='exp(-t')):
        with pytest.raises(NoiseError):
            bad()
        with pytest.raises(ValueError):
            bad()


def test_valid_models_are_not_refused_by_the_syntax_check():
    # population sums, indices, calls, powers and sage constants all read as expressions
    for ok in ('-mu*x + 2^3 - x^-1', 'sum(w[i,j]*Conv(g[i,j], n[j]) for j in E)', 'exp(-x/2)*I', '1e-3*x',
               'phi[i](v[i])', 'x(t - d)', 'Dt*x', '(tau*dt + 1)*x'):
        from api.noise import expression_syntax_problem
        assert expression_syntax_problem(ok) is None, ok

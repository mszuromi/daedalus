"""The README's Python snippets, executed as written (audit W5: a README example that breaks or
returns an incomplete result should fail a test, not a reader).

The blocks are read from README.md itself, so editing a snippet changes what runs here.  Only
the placeholder repository path is substituted.  The Hawkes snippet's last line (a 200-path
simulation) is left out here: its sample paths are not checked by anything in the snippet.
"""
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _blocks():
    text = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    return re.findall(r'```python\n(.*?)```', text, re.S)


def _block(marker):
    hits = [b for b in _blocks() if marker in b]
    assert len(hits) == 1, f'README: {len(hits)} python blocks contain {marker!r}'
    return (hits[0].replace("'/path/to/daedalus/notebooks'", repr(os.path.join(ROOT, 'notebooks')))
            .replace("'/path/to/daedalus'", repr(ROOT)))


def _exec(code, name):
    ns = {}
    exec(compile(code, f'README.md ({name})', 'exec'), ns)
    return ns


def test_readme_python_blocks_are_the_ones_tested_here():
    """A new README snippet must get a test (or be listed here)."""
    markers = ['dd.run(model, dd.Config(', "SDE('Hawkes')", "SDE('Allen-Cahn', space=1)",
               "SDE('OU quartic')", 'spectra={', 'dd.build_cache(', 'dd.master_equation(', 'dd.JumpProcess(',
               'dd.covariance_matrix(', "SDE('Mixed noise')", 'dd.hartree(']
    blocks = _blocks()
    assert len(blocks) == len(markers), [b.splitlines()[0] for b in blocks]
    for b in blocks:
        assert sum(m in b for m in markers) == 1, b


def test_readme_script_example_runs_complete():
    ns = _exec(_block('dd.run(model, dd.Config('), 'script example')
    res = ns['res']
    assert isinstance(res, dict) and res.get('status', 'ok') != 'incomplete', res.get('status')


def test_readme_sde_hawkes_example_is_ok():
    code = _block("SDE('Hawkes')")
    head, sep, _sim = code.partition('rec = m.simulate(')
    assert sep, 'the Hawkes snippet no longer ends with the simulator line'
    res = _exec(head, 'Hawkes example')['res']
    assert res['status'] == 'ok', res['status']
    assert not res.get('phase_j_incomplete')


def test_readme_spde_example_is_the_one_loop_allen_cahn():
    """The SPDE in the declared-operator style runs tree + one loop; its equal-time variance at x = 0 is the tree
    value T/(2 sqrt(mu D)) = 1/2 lowered by the phi^4 tadpole."""
    res = _exec(_block("SDE('Allen-Cahn', space=1)"), 'SPDE example')['res']
    assert res['status'] == 'ok', res['status']
    C = res['C_tau_x']
    it0, ix0 = list(res['tau_grid']).index(0.0), list(res['spatial_grid']).index(0.0)
    assert res['C_tau_x_by_ell'][0][it0, ix0] == pytest.approx(0.5, rel=1e-12)
    assert 0.4 < C[it0, ix0] < 0.5


def test_readme_build_cache_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _exec(_block('dd.build_cache('), 'build_cache dry run')
    assert list(tmp_path.iterdir()) == []


def test_readme_spectra_example_is_ok():
    ns = _exec(_block('spectra={'), 'spectra example')
    sp = ns['sp']
    assert sp['status'] == 'ok' and set(sp['method_by_ell'].values()) == {'exact'}
    assert sp['S'].shape == (201,) and sp['integral'].shape == (2,)
    assert abs(sp['S0'] - 2 * sp['integral_inf']) <= 1e-13 * abs(sp['S0'])


def test_readme_master_equation_example_is_the_poisson_law():
    ns = _exec(_block('dd.master_equation('), 'master-equation example')
    import numpy as np
    np.testing.assert_allclose(ns['k4'].ravel(), 20.0, rtol=1e-9)             # lam / mu
    np.testing.assert_allclose(ns['cc'][:, 0, 0], 20.0 * np.exp(-np.array([0.0, 1.0, 2.0])), rtol=1e-9)


def test_readme_jump_process_example_converges_at_one_loop():
    ns = _exec(_block('dd.JumpProcess('), 'jump-process example')
    err = ns['rel_err']
    assert set(err) == {'kappa_1(A)', 'kappa_1(B)', 'kappa_2(A,A)', 'kappa_2(A,B)', 'kappa_2(B,B)'}
    assert max(err.values()) < 1e-3


def test_readme_check_and_latex_example(capsys):
    ns = _exec(_block("SDE('OU quartic')"), 'check example')
    out = capsys.readouterr().out
    rows = ns['report']['vertices']['rows']
    assert [(r['response_legs'], r['field_legs'], r['coefficient']) for r in rows] == [
        (1, 2, '3*eps*xstar1'), (1, 3, 'eps')]
    assert '     1      2       3  interaction           1  xt1*dx1^2 -> 3*eps*xstar1' in out
    assert set(ns['tex']) == {'sde', 'noise', 'action', 'mean_field'}
    assert r'\sqrt{2 \, D}\,\mathrm{d}W_{\xi}' in ns['tex']['sde']
    from api.noise import Gaussian
    from api.sde import SDE, SDEError
    bad = (SDE('OU quartic').parameter('mu', default=1.0).parameter('eps', default=0.02)
           .parameter('D', default=1.0).field('x').drift('x', '-mu*x - epsilon*x^3')
           .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    with pytest.raises(SDEError) as ei:
        bad.build()
    shown = '\n'.join(re.search(r'```text\n(equation\(.*?)```', open(os.path.join(ROOT, 'README.md'),
                                                                       encoding='utf-8').read(), re.S)
                      .group(1).splitlines())
    assert str(ei.value) == shown.rstrip()


def test_readme_covariance_example_is_ok():
    ns = _exec(_block('dd.covariance_matrix('), 'covariance example')
    cm, d = ns['cm'], ns['d']
    assert cm['status'] == 'ok' and cm['Sigma'].shape == (3, 2, 2)
    assert cm['shared_stages']['misses'] == 1
    assert d['status'] == 'ok' and float(d['error'].max()) < 1e-5


def test_readme_resummation_example():
    """g = 1: Hartree within 8 % of the exact variance 0.46792, screened order 2 within 2.5 %."""
    ns = _exec(_block('dd.hartree('), 'resummation example')
    h, r = ns['h'], ns['r']
    import math
    from scipy.integrate import quad
    w = lambda x: math.exp(-(x * x / 2 + x ** 4 / 4))
    exact = quad(lambda x: x * x * w(x), -12, 12)[0] / quad(w, -12, 12)[0]      # 0.46792
    assert abs(h.cov[0, 0] / exact - 1) < 0.08
    assert abs(float(r['C_tau'][0].real) / exact - 1) < 0.025
    assert r['resummation']['kappa_source'].startswith('pms order 1')
    assert set(r['C_tau_by_order']) == {0, 1, 2}


def test_readme_paper_style_example():
    ns = _exec(_block("SDE('Mixed noise')"), 'paper-style example')
    assert [(c['source'], c['rule'], c['text']) for c in ns['fix']] == [('b', 'stratonovich', '(1/2)*g^2*x[i]')]
    assert ns['text'] == 'dx = (-mu*x) dt + s dW_a + g*x ∘ dW_b'

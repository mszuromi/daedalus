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
               "SDE('OU quartic')", 'spectra={', 'dd.build_cache(']
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

"""Tests for the YAML entry point (``api/config_run.py``, ``dd run config.yaml``).

A: a run through the config reproduces ``compute_cumulants`` called directly
   (``np.array_equal``) for k = 2 (OU quartic, ell <= 1), k = 1, and a
   multi-population model (the cross-population correlator of ``linear_hawkes``).
B: every validation error carries the config line number.
C: ``--dry-run`` computes nothing; outputs and ``run_info.json`` exist after a
   run; relative paths resolve against the config file's directory.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'notebooks'))

import yaml  # noqa: E402
import daedalus as dd  # noqa: E402
import api  # noqa: E402
from api import config_run  # noqa: E402
from api.config_run import ConfigError, load_config, parse_config, run_config  # noqa: E402


def _write(tmp_path, text, name='cfg.yaml'):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _direct(model_name, k, max_ell, ext, tau, params=None):
    """compute_cumulants called directly, with what dd.run would pass."""
    model, mod = dd.load_model(model_name)
    fund = dd.fundamental_from_model(model)
    fund.update(getattr(mod, 'DEFAULT_FUNDAMENTAL', {}) or {})
    fund.update(params or {})
    return api.compute_cumulants(
        model=model, k=k, max_ell=max_ell, fundamental=fund,
        external_fields=ext, tau_grid=np.asarray(tau, dtype=float),
        parallel=False, verbose=False)


def _equal_totals(a, b):
    assert np.array_equal(np.asarray(a['tau_grid']), np.asarray(b['tau_grid']))
    assert np.array_equal(np.asarray(a['C_tau']), np.asarray(b['C_tau']))
    assert set(a['C_tau_by_ell']) == set(b['C_tau_by_ell'])
    for ell in a['C_tau_by_ell']:
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][ell]),
                              np.asarray(b['C_tau_by_ell'][ell]))


# ── A: the config reproduces compute_cumulants ───────────────────────────────

OU_K2 = """\
model: ou_quartic
parameters: {mu: 1.0, eps: 1e-2}
question:
  k: 2
  max_ell: 1
  tau_grid: {start: -4, stop: 4, num: 9}
outputs:
  npz: out/c.npz
  csv: out/c.csv
  plot: out/c.png
"""


@pytest.fixture(scope='module')
def ou_k2_run(tmp_path_factory):
    """One k=2, ell<=1 run through the config, with the outputs (also used by C)."""
    d = tmp_path_factory.mktemp('ou_k2')
    path = _write(d, OU_K2)
    # discarded warm-up: a cold run and a warm run differ at ~1e-16
    _direct('ou_quartic', 2, 1, [('x', 1), ('x', 1)], np.linspace(-4, 4, 9),
            {'mu': 1.0, 'eps': 1e-2})
    other = tmp_path_factory.mktemp('elsewhere')
    old = os.getcwd()
    os.chdir(other)                 # relative paths must NOT follow the cwd
    try:
        result, resolved = run_config(path)
    finally:
        os.chdir(old)
    return d, other, result, resolved


def test_a_ou_quartic_k2_ell1_matches_direct(ou_k2_run):
    _, _, result, resolved = ou_k2_run
    assert resolved['question']['external_fields'] == [['x', 1], ['x', 1]]
    direct = _direct('ou_quartic', 2, 1, [('x', 1), ('x', 1)],
                     np.linspace(-4, 4, 9), {'mu': 1.0, 'eps': 1e-2})
    _equal_totals(result, direct)
    assert set(direct['C_tau_by_ell']) == {0, 1}


def test_a_k1_matches_direct(tmp_path):
    path = _write(tmp_path, "model: ou_quartic\nquestion: {k: 1, max_ell: 1}\n")
    _direct('ou_quartic', 1, 1, [('x', 1)], np.arange(-8, 8.25, 0.5))   # warm
    result, resolved = run_config(path)
    assert resolved['question']['k'] == 1
    direct = _direct('ou_quartic', 1, 1, [('x', 1)], np.arange(-8, 8.25, 0.5))
    _equal_totals(result, direct)


def test_a_multi_population_matches_direct(tmp_path):
    path = _write(tmp_path, """\
model: linear_hawkes
question:
  k: 2
  max_ell: 0
  external_fields: [[n, 1], [n, 2]]
  tau_grid: {start: -10, stop: 10, num: 5}
""")
    tau = np.linspace(-10, 10, 5)
    _direct('linear_hawkes', 2, 0, [('n', 1), ('n', 2)], tau)           # warm
    result, resolved = run_config(path)
    assert resolved['question']['external_fields'] == [['n', 1], ['n', 2]]
    direct = _direct('linear_hawkes', 2, 0, [('n', 1), ('n', 2)], tau)
    _equal_totals(result, direct)


def test_taylor_order_and_use_cache_reach_compute_cumulants(tmp_path, monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def fake(*args, **kw):
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(api, 'compute_cumulants', fake)
    base = "model: ou_quartic\nquestion: {k: 2, max_ell: 0%s}\n"
    for text, expect in [
            (base % ', taylor_order: 3' + "policy: {use_cache: false}\n",
             {'taylor_order': 3, 'use_cache': False}),
            (base % '', {})]:
        with pytest.raises(Stop):
            run_config(_write(tmp_path, text))
        kw = seen[-1]
        assert kw.get('taylor_order') == expect.get('taylor_order')
        assert kw.get('use_cache', True) == expect.get('use_cache', True)
        assert kw['parallel'] is False and kw['verbose'] is False


# ── B: every validation error has the line number ────────────────────────────

BAD = [
    ('unknown top-level key',
     "model: ou_quartic\n\nmodle_typo: 1\n", 3, "unknown key 'modle_typo'"),
    ('unknown question key',
     "model: ou_quartic\nquestion:\n  k: 2\n  taylor: 4\n", 4, "unknown key 'taylor'"),
    ('unknown policy key',
     "model: ou_quartic\npolicy:\n  verbose: false\n  workers: 2\n", 4, "unknown key 'workers'"),
    ('unknown outputs key',
     "model: ou_quartic\noutputs:\n  pdf: a.pdf\n", 3, "unknown key 'pdf'"),
    ('missing model',
     "# a comment\nparameters: {mu: 1.0}\n", 2, "missing required key 'model'"),
    ('unknown model',
     "\nmodel: no_such_model\n", 2, "unknown model"),
    ('external field not a field',
     "model: ou_quartic\nquestion:\n  external_fields:\n    - [x, 1]\n    - [y, 1]\n",
     5, "external field 'y'"),
    ('external field population out of range',
     "model: ou_quartic\nquestion:\n  external_fields: [[x, 2]]\n", 3, "out of range"),
    ('k disagrees with external_fields',
     "model: ou_quartic\nquestion:\n  k: 3\n  external_fields: [[x, 1], [x, 1]]\n",
     4, "k=3"),
    ('tau_grid not increasing',
     "model: ou_quartic\nquestion:\n  tau_grid:\n    - 0.0\n    - 1.0\n    - 0.5\n",
     6, "strictly increasing"),
    ('tau_grid start >= stop',
     "model: ou_quartic\nquestion:\n  tau_grid: {start: 2, stop: -2, num: 5}\n",
     3, "start < stop"),
    ('tau_grid missing num',
     "model: ou_quartic\nquestion:\n  tau_grid: {start: -2, stop: 2}\n", 3, "missing: num"),
    ('tau_grid num too small',
     "model: ou_quartic\nquestion:\n  tau_grid: {start: -2, stop: 2, num: 1}\n",
     3, "tau_grid.num"),
    ('non-numeric parameter',
     "model: ou_quartic\nparameters:\n  mu: 1.0\n  eps: small\n", 4, "must be a number"),
    ('boolean parameter',
     "model: ou_quartic\nparameters:\n  mu: true\n", 3, "must be a number"),
    ('non-numeric entry in a parameter list',
     "model: ou_quartic\nparameters:\n  mu: [1.0, x]\n", 3, "must be a number"),
    ('unknown parameter',
     "model: ou_quartic\nparameters:\n  mu: 1.0\n  epsilon: 0.1\n", 4,
     "unknown parameter 'epsilon'"),
    ('k not an integer',
     "model: ou_quartic\nquestion:\n  k: two\n", 3, "question.k must be an integer"),
    ('policy not a boolean',
     "model: ou_quartic\npolicy:\n  parallel: 1\n", 3, "true or false"),
    ('duplicate key',
     "model: ou_quartic\nquestion: {k: 2}\nmodel: ou_sextic\n", 3, "duplicate key"),
    ('YAML syntax error',
     "model: ou_quartic\nquestion: [k: 2\npolicy: {}\n", 3, "invalid YAML"),
    ('points on a temporal model',
     "model: ou_quartic\nquestion:\n  k: 3\n  points: [[[0, 0], [1, 0]]]\n", 4,
     "question.points applies only"),
]


@pytest.mark.parametrize('label,text,line,fragment', BAD,
                         ids=[b[0] for b in BAD])
def test_b_validation_error_has_line_number(tmp_path, label, text, line, fragment):
    path = _write(tmp_path, text)
    with pytest.raises(ConfigError) as ei:
        load_config(path)
    msg = str(ei.value)
    assert ei.value.line == line
    assert f'cfg.yaml: line {line}:' in msg
    assert fragment in msg


def test_b_scientific_notation_without_a_dot_is_a_number(tmp_path):
    raw = parse_config(_write(tmp_path, "model: ou_quartic\n"
                                        "parameters: {eps: 1e-3, mu: [1, 2.5e1]}\n"))
    assert raw['parameters']['eps'][0] == 0.001
    assert raw['parameters']['mu'][0] == [1, 25.0]


def test_b_cli_reports_a_config_error_with_status_2(tmp_path, capsys):
    path = _write(tmp_path, "model: ou_quartic\nbogus: 1\n")
    with pytest.raises(SystemExit) as ei:
        dd.main(['run', path])
    assert ei.value.code == 2
    assert 'line 2' in capsys.readouterr().err


# ── C: dry run, outputs, relative paths ──────────────────────────────────────

def test_c_dry_run_computes_nothing_and_writes_nothing(tmp_path, monkeypatch, capsys):
    def boom(*a, **kw):
        raise AssertionError('--dry-run must not compute')

    monkeypatch.setattr(api, 'compute_cumulants', boom)
    monkeypatch.setattr(dd, 'run', boom)
    path = _write(tmp_path, OU_K2)
    dd.main(['run', path, '--dry-run'])
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown['question']['k'] == 2
    assert shown['question']['external_fields'] == [['x', 1], ['x', 1]]
    assert shown['parameters'] == {'mu': 1.0, 'eps': 0.01, 'D': 1.0}
    assert shown['outputs']['npz'] == str(tmp_path / 'out' / 'c.npz')
    assert sorted(os.listdir(tmp_path)) == ['cfg.yaml']       # nothing written


def test_c_dry_run_still_validates_the_external_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'compute_cumulants',
                        lambda *a, **kw: pytest.fail('computed'))
    path = _write(tmp_path, "model: ou_quartic\nquestion:\n  external_fields: [[z, 1]]\n")
    with pytest.raises(SystemExit):
        dd.main(['run', path, '--dry-run'])


def test_c_outputs_and_run_info_exist_next_to_the_config(ou_k2_run):
    d, other, result, resolved = ou_k2_run
    out = d / 'out'
    for name in ('c.npz', 'c.csv', 'c.png', 'config_resolved.yaml', 'run_info.json'):
        assert (out / name).is_file(), name
    # the cwd got no output (the engine's own cache directories aside)
    assert set(os.listdir(other)) <= {'saved_models', 'saved_prediagrams'}
    info = json.loads((out / 'run_info.json').read_text())
    assert info['daedalus_version'] == dd.__version__
    assert info['wall_time_s'] > 0
    assert info['resolved'] == json.loads(json.dumps(resolved))
    assert 'git_commit' in info
    assert set(info['outputs']) == {'npz', 'csv', 'plot'}
    # the saved NPZ is the existing writer's output for this result
    z = np.load(out / 'c.npz')
    assert np.array_equal(z['C_total'], np.asarray(result['C_tau']))


def test_c_config_resolved_yaml_is_itself_a_valid_config(ou_k2_run):
    d, _, _, resolved = ou_k2_run
    again, _, _, _ = load_config(str(d / 'out' / 'config_resolved.yaml'))
    assert again == resolved


def test_c_model_path_resolves_against_the_config_directory(tmp_path):
    src = os.path.join(ROOT, 'models', 'ou_quartic.model.py')
    sub = tmp_path / 'my_models'
    sub.mkdir()
    (sub / 'copy.model.py').write_text(open(src).read())
    path = _write(tmp_path, "model: my_models/copy.model.py\n")
    resolved, model, _, _ = load_config(path)
    assert resolved['model'] == str(sub / 'copy.model.py')
    assert dd.field_names(model) == ['x']


def test_c_python_dash_m_daedalus(tmp_path):
    path = _write(tmp_path, "model: ou_quartic\nquestion: {k: 2, max_ell: 0}\n")
    env = dict(os.environ, PYTHONHASHSEED='0')
    done = subprocess.run([sys.executable, '-m', 'daedalus', 'run', path,
                           '--dry-run'], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert yaml.safe_load(done.stdout)['question']['max_ell'] == 0


def test_d_readme_example_config_validates(tmp_path):
    text = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    section = text.split('## Running from a config file', 1)[1]
    block = section.split('```yaml\n', 1)[1].split('```', 1)[0]
    resolved, _, _, _ = load_config(_write(tmp_path, block))
    assert resolved['question']['taylor_order'] == 4
    assert resolved['question']['tau_grid'][0] == -6.0
    assert resolved['outputs']['npz'] == str(tmp_path / 'results' / 'ou.npz')

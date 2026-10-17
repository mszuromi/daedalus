"""Follow-ups of the YAML entry point (``api/config_run.py``), on top of ``test_config_run.py``.

A  the ``daedalus`` console script (``[project.scripts] daedalus = "daedalus:main"``) and ``--dry-run`` printing the analyzer
   report table next to the resolved config (as YAML comments, so the output is still a valid config).
B  keys: spatial ``chi_grid`` (alias ``spatial_grid``; a mapping is a range, a list is literal points, a 3-element
   list that looks like a range raises, as in ``dd.Config``), ``output`` (directory or file stem), and the parameter
   list shape check, all with line-numbered errors.
C  end to end, the paths that were wired but untested: spatial k=2 with a ``chi_grid``, spatial ``points`` (k=3),
   ``report_pdf`` (skipped without LaTeX) and temporal k=3.  Each run equals the direct call, bit for bit (same
   process, same working directory, warm caches: the engine caches are cwd-relative).
"""
import json
import os
import shutil
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
from api import config_run  # noqa: E402
from api.config_run import ConfigError, load_config, run_config  # noqa: E402


def _write(tmp_path, text, name='cfg.yaml'):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _bad(tmp_path, text, line, fragment):
    with pytest.raises(ConfigError) as e:
        load_config(_write(tmp_path, text))
    assert e.value.line == line, str(e.value)
    assert fragment in str(e.value), str(e.value)


# ── A: the dd script and the dry-run report ───────────────────────────────────────────────────

def test_a_console_script_entry_point_is_declared_and_importable():
    """``pip install -e .`` would create ``daedalus`` from this entry; installing into a throw-away environment is not
    cheap on Sage, so the declaration is parsed and the target is checked to be the callable ``main(argv)``."""
    import tomllib
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as fh:
        meta = tomllib.load(fh)
    assert 'dd' not in meta['project']['scripts']                   # renamed: dd shadows coreutils dd
    target = meta['project']['scripts']['daedalus']
    assert target == 'daedalus:main'
    module, func = target.split(':')
    fn = getattr(__import__(module), func)
    assert callable(fn) and fn is dd.main


def test_a_main_handles_run_and_exit_status(tmp_path, capsys):
    ok = _write(tmp_path, "model: ou_quartic\nquestion: {k: 2, max_ell: 0}\n")
    assert dd.main(['run', ok, '--dry-run']) is None                  # returns on success (no SystemExit)
    capsys.readouterr()
    bad = _write(tmp_path, "model: ou_quartic\nquestion: {taylor: 3}\n", 'bad.yaml')
    with pytest.raises(SystemExit) as e:
        dd.main(['run', bad, '--dry-run'])
    assert e.value.code == 2 and 'bad.yaml: line 2' in capsys.readouterr().err


def test_a_dry_run_prints_the_resolved_config_and_the_report_table(tmp_path, monkeypatch, capsys):
    def boom(*a, **kw):
        raise AssertionError('--dry-run must not compute')
    import api
    monkeypatch.setattr(api, 'compute_cumulants', boom)
    path = _write(tmp_path, "model: ou_quartic\nparameters: {eps: 0.5}\nquestion: {k: 2, max_ell: 1}\n")
    dd.main(['run', path, '--dry-run'])
    out = capsys.readouterr().out
    shown = yaml.safe_load(out)                                       # report lines are YAML comments
    assert shown['parameters']['eps'] == 0.5 and shown['question']['max_ell'] == 1
    rep = [ln[2:] for ln in out.splitlines() if ln.startswith('# ')]
    table = '\n'.join(rep)
    assert 'Model report: OU Quartic' in table and 'route      phase_j_per_diagram (predicted)' in table
    assert 'G5_LOOP_PARAMETER' in table                               # eps = 0.5: g_eff = 1.5 > 0.2
    assert all(len(ln) <= 102 for ln in out.splitlines() if ln.startswith('# '))
    assert sorted(os.listdir(tmp_path)) == ['cfg.yaml']               # nothing written


def test_a_dry_run_report_names_the_route_a_spatial_model_would_take(tmp_path, capsys):
    path = _write(tmp_path, "model: reaction_diffusion_2d\nquestion: {k: 2, max_ell: 1}\n")
    dd.main(['run', path, '--dry-run'])
    out = capsys.readouterr().out
    assert 'route      spatial_integrator (predicted)' in out.replace('# ', '')


def test_a_dry_run_without_the_analyzer_says_so(tmp_path, capsys):
    path = _write(tmp_path, "model: ou_quartic\npolicy: {preflight: false}\n")
    dd.main(['run', path, '--dry-run'])
    out = capsys.readouterr().out
    assert '# no analyzer (policy.preflight is false)' in out and 'Model report' not in out


def test_a_dry_run_report_follows_the_policy(tmp_path, capsys):
    """strict never promotes the G5 advisories: the report keeps the large-g warning a warning."""
    path = _write(tmp_path, "model: ou_quartic\nparameters: {eps: 0.5}\nquestion: {k: 2, max_ell: 1}\npolicy: {mode: strict}\n")
    dd.main(['run', path, '--dry-run'])
    out = capsys.readouterr().out
    assert '[OK; 1 warning(s)' in out and 'G5_LOOP_PARAMETER' in out and 'ERROR' not in out


# ── B: keys ───────────────────────────────────────────────────────────────────────────────────

SP = "model: reaction_diffusion_2d\nquestion:\n  k: 2\n  max_ell: 1\n"


def test_b_chi_grid_range_literal_and_alias(tmp_path):
    r = load_config(_write(tmp_path, SP + "  chi_grid: {start: 0, stop: 2, num: 3}\n"))[0]
    assert r['question']['chi_grid'] == [0.0, 1.0, 2.0]
    r = load_config(_write(tmp_path, SP + "  spatial_grid: [0.0]\n"))[0]
    assert r['question']['chi_grid'] == [0.0] and 'spatial_grid' not in r['question']
    r = load_config(_write(tmp_path, SP + "  chi_grid: [0.0, 0.5, 1.0, 3.0]\n"))[0]
    assert r['question']['chi_grid'] == [0.0, 0.5, 1.0, 3.0]
    # the resolved config is itself a valid config
    again = _write(tmp_path, config_run._dump_yaml(r), 'again.yaml')
    assert load_config(again)[0]['question']['chi_grid'] == [0.0, 0.5, 1.0, 3.0]


def test_b_three_point_grids_round_trip_through_values(tmp_path):
    """A 3-point range resolves to [0.0, 1.0, 2.0], which the range-slip rule would reject if it were written
    back as a list: the resolved config writes {values: [...]}, and a user can write an uneven 3-point grid so."""
    text = SP + "  tau_grid: {start: 0, stop: 2, num: 3}\n  chi_grid: {values: [0.0, 1.0, 4.0]}\n"
    r = load_config(_write(tmp_path, text))[0]
    assert r['question']['tau_grid'] == [0.0, 1.0, 2.0] and r['question']['chi_grid'] == [0.0, 1.0, 4.0]
    dumped = config_run._dump_yaml(r)
    assert 'values' in dumped
    again = load_config(_write(tmp_path, dumped, 'again.yaml'))[0]
    assert again['question']['tau_grid'] == [0.0, 1.0, 2.0] and again['question']['chi_grid'] == [0.0, 1.0, 4.0]
    _bad(tmp_path, SP + "  tau_grid: {values: [0, 1], start: 0}\n", 5, "'values' cannot be combined")
    _bad(tmp_path, SP + "  chi_grid: [0.0, 1.0, 4.0]\n", 5, 'values: [0.0, 1.0, 4.0]')


def test_b_resolved_config_of_spatial_models_with_one_population_is_valid(tmp_path):
    """Spatial legs are slots, not populations: the model's recommended [[phi, 1], [phi, 2]] must be accepted
    on its way back in."""
    for name in ('edwards_wilkinson_1d', 'allen_cahn_1d_subcritical_infinite', 'linear_diffusion_test'):
        r = load_config(_write(tmp_path, f"model: {name}\n"))[0]
        again = load_config(_write(tmp_path, config_run._dump_yaml(r), 'again.yaml'))[0]
        assert again['question'] == r['question'], name


def test_b_chi_grid_three_element_list_raises_like_dd_config(tmp_path):
    with pytest.raises(ValueError):
        dd.Config(k=2, chi_grid=[-5, 5, 51])                          # the rule being mirrored
    _bad(tmp_path, SP + "  chi_grid: [-5, 5, 51]\n", 5, '3-element LIST')
    _bad(tmp_path, SP + "  spatial_grid: [-5, 5, 51]\n", 5, 'question.spatial_grid is a 3-element LIST')
    _bad(tmp_path, "model: ou_quartic\nquestion:\n  tau_grid: [-5, 5, 51]\n", 3, '3-element LIST')
    r = load_config(_write(tmp_path, SP + "  chi_grid: [0.0, 1.5, 0.5]\n"))[0]      # last entry < 2: literal points
    assert r['question']['chi_grid'] == [0.0, 1.5, 0.5]


def test_b_spatial_grids_are_never_silently_ignored(tmp_path):
    ac = "model: allen_cahn_1d_subcritical_infinite\nquestion:\n"
    k3 = "  k: 3\n  external_fields: [[phi, 1], [phi, 1], [phi, 1]]\n"
    _bad(tmp_path, ac + k3, 3, 'needs question.points')
    _bad(tmp_path, ac + k3 + "  chi_grid: {start: 0, stop: 10, num: 200}\n  tau_grid: {start: 0, stop: 5, num: 200}\n",
         5, 'more than 20000')
    _bad(tmp_path, ac + k3 + "  points: [[[0.5, 0.2], [1.0, 0.4]]]\n  chi_grid: [0.0, 1.0]\n", 6,
         'question.chi_grid is ignored')
    _bad(tmp_path, ac + k3 + "  points: [[[0.5, 0.2], [1.0, 0.4]]]\n  tau_grid: [0.0, 1.0]\n", 6,
         'question.tau_grid is ignored')
    _bad(tmp_path, ac + "  k: 1\n  chi_grid: [0.0, 1.0]\n", 4, 'question.chi_grid is ignored')
    ok = load_config(_write(tmp_path, ac + k3 + "  chi_grid: [0.0, 1.0]\n  tau_grid: [0.0, 0.5]\n"))[0]
    assert ok['question']['chi_grid'] == [0.0, 1.0]


def test_b_dry_run_report_uses_the_requested_taylor_order(tmp_path, capsys):
    dd.main(['run', _write(tmp_path, "model: ou_quartic\nquestion: {k: 2, max_ell: 1, taylor_order: 6}\n"), '--dry-run'])
    assert 'Taylor order 6' in capsys.readouterr().out


def test_b_output_decision_is_fixed_before_the_run_creates_directories(tmp_path):
    """output: results (a stem, the directory does not exist yet) with outputs.npz inside results/: the run creates
    results/, which must not turn the stem into a directory half way (provenance files are prefixed throughout)."""
    path = _write(tmp_path, "model: ou_quartic\nquestion: {k: 2, max_ell: 0, tau_grid: [0.0, 1.0]}\n"
                  "output: results\noutputs: {npz: results/x.npz}\n")
    result, resolved = run_config(path)
    assert resolved['outputs']['csv'] == str(tmp_path / 'results.csv')
    assert sorted(os.listdir(tmp_path / 'results')) == ['results.config_resolved.yaml', 'results.run_info.json', 'x.npz']


def test_b_chi_grid_errors_have_lines(tmp_path):
    _bad(tmp_path, SP + "  chi_grid: {start: 0, stop: 2}\n", 5, 'missing: num')
    _bad(tmp_path, SP + "  chi_grid: {start: 2, stop: 0, num: 4}\n", 5, 'start < stop')
    _bad(tmp_path, SP + "  chi_grid: [0.0]\n  spatial_grid: [1.0]\n", 6, "not both (chi_grid is on line 5)")
    _bad(tmp_path, SP + "  chi_grid: abc\n", 5, 'chi_grid must be a non-empty list')
    _bad(tmp_path, "model: ou_quartic\nquestion:\n  chi_grid: [0.0, 1.0]\n", 3, 'applies only to a spatial model')


def test_b_output_directory_and_stem(tmp_path):
    base = "model: ou_quartic\nquestion: {k: 2, max_ell: 0}\n"
    r = load_config(_write(tmp_path, base + "output: results/\n"))[0]
    assert r['outputs'] == {'npz': str(tmp_path / 'results' / 'cumulants.npz'),
                            'csv': str(tmp_path / 'results' / 'cumulants.csv'),
                            'plot': str(tmp_path / 'results' / 'cumulants.png')}
    r = load_config(_write(tmp_path, base + "output: runs/ou1\n"))[0]
    assert r['outputs']['npz'] == str(tmp_path / 'runs' / 'ou1.npz') and r['outputs']['plot'].endswith('ou1.png')
    # an existing directory without a trailing slash is a directory; explicit 'outputs' entries win per kind
    (tmp_path / 'there').mkdir()
    r = load_config(_write(tmp_path, base + "output: there\noutputs: {csv: elsewhere.csv}\n"))[0]
    assert r['outputs']['npz'] == str(tmp_path / 'there' / 'cumulants.npz')
    assert r['outputs']['csv'] == str(tmp_path / 'elsewhere.csv')
    _bad(tmp_path, base + "output: out/run.npz\n", 3, "directory or a file stem without an extension")
    _bad(tmp_path, base + "output: [a, b]\n", 3, "'output' must be a non-empty string")


def test_b_parameter_list_shapes_are_validated_with_lines(tmp_path):
    mp = "model: multipopulation_test\nparameters:\n"
    ok = load_config(_write(tmp_path, mp + "  tauE: [10, 9.5]\n  wEE: [[0.25, 0.22], [0.21, 0.19]]\n"))[0]
    assert ok['parameters']['tauE'] == [10, 9.5]
    _bad(tmp_path, mp + "  tauE: [10, 9.5, 8]\n", 3, "parameter 'tauE' has shape (2,) (indexed by E); got shape (3,)")
    _bad(tmp_path, mp + "  tauE: [10, 9.5]\n  wEE: [0.25, 0.22]\n", 4, "parameter 'wEE' has shape (2, 2) (indexed by E, E)")
    _bad(tmp_path, mp + "  wEE: [[0.25, 0.22], [0.21]]\n", 3, 'nested lists must be rectangular')
    _bad(tmp_path, "model: ou_quartic\nparameters:\n  mu: [1.0, 2.0]\n", 3,
         "parameter 'mu' is a scalar parameter; got a list of shape (2,)")
    # a scalar for an indexed parameter fails deep in the run (a float is not subscriptable): stop it here
    _bad(tmp_path, mp + "  tauE: 10\n", 3, "parameter 'tauE' has shape (2,) (indexed by E); got a scalar")
    _bad(tmp_path, "model: toy_quartic_double_well\nparameters:\n  mu: -1.0\n", 3, 'got a scalar')


# ── C: end to end, equal to the direct call ───────────────────────────────────────────────────

def _equal(a, b, keys):
    for key in keys:
        assert key in a and key in b, key
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


#: a d=1 spatial model for the runs (the d=2 model's one loop has a tadpole,
#: UV-divergent at d>=2, and its tree at x=tau=0 is the UV-divergent equal
#: point -- both refused since audit wave 2)
SP1 = "model: reaction_diffusion_quadratic_1d\nquestion:\n  k: 2\n  max_ell: 1\n"


def test_c_spatial_k2_with_chi_grid_matches_direct(tmp_path):
    path = _write(tmp_path, SP1 + "  tau_grid: [0.0, 0.5]\n  chi_grid: {start: 0, stop: 2, num: 3}\n"
                  "output: out/rd\n")
    model, mod = dd.load_model('reaction_diffusion_quadratic_1d')
    cfg = dd.Config(k=2, max_ell=1, tau_grid=np.array([0.0, 0.5]), chi_grid=np.array([0.0, 1.0, 2.0]),
                    parallel=False, verbose=False)
    dd.run(model, cfg, mod)                                           # warm
    result, resolved = run_config(path)
    direct = dd.run(model, cfg, mod)
    _equal(result, direct, ['C_tau_x', 'spatial_grid', 'tau_grid'])
    assert np.asarray(result['C_tau_x']).shape == (2, 3)
    out = tmp_path / 'out'
    assert sorted(os.listdir(out)) == ['rd.config_resolved.yaml', 'rd.csv', 'rd.npz', 'rd.png', 'rd.run_info.json']
    info = json.load(open(out / 'rd.run_info.json'))
    assert info['route'] == 'spatial_integrator' and info['resolved']['question']['chi_grid'] == [0.0, 1.0, 2.0]
    # two stems in one directory keep their own provenance files
    run_config(_write(tmp_path, SP1 + "  tau_grid: [0.0]\n  chi_grid: [0.0, 1.0]\noutput: out/rd2\n", 'cfg2.yaml'))
    assert 'rd2.run_info.json' in os.listdir(out) and 'rd.run_info.json' in os.listdir(out)


def test_c_spatial_points_k3_matches_direct(tmp_path):
    path = _write(tmp_path, "model: reaction_diffusion_quadratic_1d\nquestion:\n  k: 3\n  max_ell: 0\n"
                  "  points: [[[0.5, 0.2], [1.0, 0.4]], [[0.0, 0.0], [1.0, 0.0]]]\n")
    model, mod = dd.load_model('reaction_diffusion_quadratic_1d')
    pts = [[[0.5, 0.2], [1.0, 0.4]], [[0.0, 0.0], [1.0, 0.0]]]
    cfg = dd.Config(k=3, max_ell=0, spatial_points=pts, parallel=False, verbose=False)
    dd.run(model, cfg, mod)
    result, resolved = run_config(path)
    direct = dd.run(model, cfg, mod)
    _equal(result, direct, ['C_kpoint'])
    assert np.asarray(result['C_kpoint']).shape == (2,) and np.all(np.isfinite(result['C_kpoint']))
    assert resolved['question']['points'] == pts
    assert np.any(np.asarray(result['C_kpoint']) != 0)                # the quadratic vertex gives a skewed field


TEMPORAL_K3 = '''\
from api.model import TemporalModelBuilder


def build():
    return (TemporalModelBuilder('Quadratic OU')
            .population('pop', size=1)
            .physical_field('x', population='pop', description='variable')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('b', default=0.2, domain='positive')
            .parameter('c', default=1.0, domain='positive')
            .parameter('D', default=0.5, domain='positive')
            .set_action_text('sum(xt[i]*((Dt+mu)*x[i] - b*x[i]^2 - c) - D*xt[i]^2 for i in pop)')
            .equation(lhs='(Dt+mu)*x[i]', rhs='b*x[i]^2 + c', population='pop')
            .stability_analysis(True)
            .build())
'''


def test_c_temporal_k3_matches_direct(tmp_path):
    (tmp_path / 'quad_ou.model.py').write_text(TEMPORAL_K3)
    path = _write(tmp_path, "model: quad_ou.model.py\nquestion:\n  k: 3\n  max_ell: 0\n"
                  "  tau_grid: {start: -1, stop: 1, num: 3}\noutputs: {npz: o/k3.npz, csv: o/k3.csv}\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location('models.quad_ou', tmp_path / 'quad_ou.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build()
    cfg = dd.Config(k=3, max_ell=0, external_fields=[('x', 1)] * 3, tau_grid=np.linspace(-1, 1, 3), parallel=False,
                    verbose=False)
    dd.run(model, cfg, mod)
    result, resolved = run_config(path)
    direct = dd.run(model, cfg, mod)
    _equal(result, direct, ['C_tau', 'tau_grid'])
    assert set(result['C_tau_slices']) == {1, 2} and np.all(np.isfinite(result['C_tau']))
    assert np.any(np.abs(np.asarray(result['C_tau'])) > 1e-6)         # a skewed stationary state: kappa_3 != 0
    assert resolved['question']['k'] == 3 and (tmp_path / 'o' / 'k3.npz').is_file()


@pytest.mark.skipif(shutil.which('pdflatex') is None and shutil.which('latexmk') is None,
                    reason='report_pdf needs a LaTeX installation')
def test_c_report_pdf_is_written_and_recorded(tmp_path):
    path = _write(tmp_path, "model: ou_quartic\nquestion: {k: 2, max_ell: 0, tau_grid: {start: -2, stop: 2, num: 5}}\n"
                  "outputs: {report_pdf: rep/ou.pdf, csv: rep/ou.csv}\n")
    result, resolved = run_config(path)
    pdf = tmp_path / 'rep' / 'ou.pdf'
    assert pdf.is_file() and pdf.stat().st_size > 1000 and pdf.read_bytes()[:4] == b'%PDF'
    info = json.load(open(tmp_path / 'rep' / 'run_info.json'))
    assert info['outputs']['report_pdf'] == str(pdf)


# ── D: inline SDE blocks (config v2) ──────────────────────────────────────────────────────────

SDE_CFG = """\
sde:
  name: OU quartic inline
  parameters:
    mu: {default: 1.0, domain: positive}
    eps: {default: 0.02, domain: positive}
    D: {default: 1.0, domain: positive}
  fields: [x]
  drift: {x: "-mu*x - eps*x^3"}
  noise:
    xi: {type: Gaussian, var: "2*D", couples: {x: 1}}
  stability_analysis: true
question:
  k: 2
  max_ell: 1
  tau_grid: [0.0, 0.5, 1.0, 2.0]
"""


def _python_sde():
    from api.noise import Gaussian
    from api.sde import SDE
    s = SDE('OU quartic inline')
    s.parameter('mu', default=1.0, domain='positive')
    s.parameter('eps', default=0.02, domain='positive')
    s.parameter('D', default=1.0, domain='positive')
    return (s.field('x').drift('x', '-mu*x - eps*x^3').noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .stability_analysis(True).build())


def test_d_inline_sde_matches_the_python_built_model(tmp_path):
    path = _write(tmp_path, SDE_CFG)
    model = _python_sde()
    cfg = dd.Config(k=2, max_ell=1, tau_grid=np.array([0.0, 0.5, 1.0, 2.0]), parallel=False, verbose=False)
    dd.run(model, cfg)                                                # warm
    result, resolved = run_config(path)
    direct = dd.run(model, cfg)
    _equal(result, direct, ['C_tau', 'tau_grid'])
    assert set(result['C_tau_by_ell']) == {0, 1} and result['route']['backend'].startswith('phase_j')
    assert 'model' not in resolved and resolved['sde']['fields'] == ['x']
    # the resolved config is itself a valid config (same model, same numbers)
    again = load_config(_write(tmp_path, yaml.safe_dump(resolved), 'again.yaml'))[0]
    assert again['sde'] == resolved['sde'] and again['parameters'] == resolved['parameters']


def test_d_inline_sde_expressions_may_be_numbers_and_unsupported_keys_are_errors(tmp_path):
    eq = SDE_CFG.replace('  drift: {x: "-mu*x - eps*x^3"}\n',
                         '  equations: {x: {lhs: "(Dt+mu)*x", rhs: "-eps*x^3 + 0"}}\n')
    r = load_config(_write(tmp_path, eq))[0]
    assert r['sde']['equations']['x'] == {'lhs': '(Dt+mu)*x', 'rhs': '-eps*x^3 + 0'}
    num = load_config(_write(tmp_path, eq.replace('"-eps*x^3 + 0"', '0'), 'n.yaml'))[0]
    assert num['sde']['equations']['x']['rhs'] == '0'
    _bad(tmp_path, SDE_CFG.replace('eps: {default: 0.02, domain: positive}',
                                   'eps: {default: 0.02, indexed: true}'), 5, "unknown key 'indexed'")


def test_d_inline_sde_dry_run_and_other_noise_types(tmp_path, capsys):
    text = SDE_CFG.replace('    xi: {type: Gaussian, var: "2*D", couples: {x: 1}}\n',
                           '    xi: {type: Gaussian, var: "2*D", couples: {x: 1}}\n'
                           '    shot: {type: CompoundPoisson, rate: 3.0, jump: {type: Dirac, a: 0.1}, couples: {x: 1}}\n')
    dd.main(['run', _write(tmp_path, text), '--dry-run'])
    out = capsys.readouterr().out
    assert yaml.safe_load(out)['sde']['noise']['shot']['jump'] == {'type': 'Dirac', 'a': 0.1}
    assert 'Model report: OU quartic inline' in out and 'non-Gaussian' not in out.split('findings')[0]


@pytest.mark.parametrize('label,text,line,fragment', [
    ('both model and sde', "model: ou_quartic\n" + SDE_CFG, 2, "give either 'model' or 'sde'"),
    ('neither', "question: {k: 2}\n", 1, "missing required key 'model' (or an inline 'sde' block)"),
    ('unknown sde key', SDE_CFG.replace('  fields: [x]\n', '  fields: [x]\n  drfit: {}\n'), 8, "unknown key 'drfit'"),
    ('no noise', SDE_CFG.replace('  noise:\n    xi: {type: Gaussian, var: "2*D", couples: {x: 1}}\n', ''), 2,
     "'sde' needs 'noise'"),
    ('bad noise type', SDE_CFG.replace('type: Gaussian', 'type: Gausian'), 10, "'type' must be one of"),
    ('no couples', SDE_CFG.replace(', couples: {x: 1}', ''), 10, "'couples' must be a mapping"),
    ('undeclared field in drift', SDE_CFG.replace('drift: {x: "-mu*x - eps*x^3"}', 'drift: {x: "-mu*x - eps*y^3"}'), 8,
     'y'),
    ('drift for an undeclared field', SDE_CFG.replace('drift: {x: ', 'drift: {z: '), 8, "declare the field first"),
    ('bad jump law', SDE_CFG.replace('xi: {type: Gaussian, var: "2*D", couples: {x: 1}}',
                                     'xi: {type: CompoundPoisson, rate: 2, jump: {type: Gaussian, mu: 1}, couples: {x: 1}}'),
     10, 'jump law type must be one of'),
    ('bad interpretation', SDE_CFG.replace('  stability_analysis: true\n', '  interpretation: strat\n'), 11,
     'sde.interpretation must be ito or stratonovich'),
])
def test_d_inline_sde_errors_have_lines(tmp_path, label, text, line, fragment):
    _bad(tmp_path, text, line, fragment)

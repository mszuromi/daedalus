"""The single ``daedalus`` namespace: the deprecated old top-level names
(``api``, ``engine``, ``simulations``), what the installed package ships, and
where ``dd.load_model`` looks for model files.

The old names must give the SAME module objects as ``daedalus.<name>`` (a
second copy would split module state such as Phase J's counters and break
``except`` / ``isinstance``), warn once each, and never shadow a user's own
package of the same name.  Each case runs in a fresh interpreter so the import
state and the once-per-name warnings start clean.
"""
from __future__ import annotations

import fnmatch
import os
import pickle
import subprocess
import sys
import textwrap
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code, cwd=None, env_extra=None, path=(ROOT,)):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(path)
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(code)],
                         capture_output=True, text=True, timeout=300,
                         cwd=cwd or ROOT, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_old_names_are_the_same_modules_and_warn_once():
    out = _run('''
        import sys, warnings
        warnings.simplefilter('always')
        with warnings.catch_warnings(record=True) as rec:
            import daedalus
            import engine.integration.time_domain.final_integral as fi_old
            import engine.core.vertices
            from api import compute_cumulants
            from api.compute import PhaseJIncompleteError
            import simulations.cumulant_estimator as ce
        import daedalus.engine.integration.time_domain.final_integral as fi
        import daedalus.api as A, daedalus.api.compute as C
        assert fi_old is fi
        assert sys.modules['engine'] is sys.modules['daedalus.engine']
        assert sys.modules['engine.core.vertices'] is sys.modules['daedalus.engine.core.vertices']
        assert compute_cumulants is A.compute_cumulants
        assert PhaseJIncompleteError is C.PhaseJIncompleteError
        assert ce is sys.modules['daedalus.simulations.cumulant_estimator']
        # a plain module is not turned into a package by the alias
        assert not hasattr(sys.modules['daedalus.engine.core.vertices'], '__path__')
        dep = [w for w in rec if issubclass(w.category, DeprecationWarning)
               and 'old top-level name' in str(w.message)]
        assert sorted(str(w.message).split('`')[1] for w in dep) == ['api', 'engine', 'simulations'], dep
        assert all(w.filename == '<string>' for w in dep), [w.filename for w in dep]
        print('ok')
    ''')
    assert out.strip() == 'ok'


def test_checkout_forwarding_package_without_importing_daedalus_first():
    out = _run('''
        import sys, warnings
        warnings.simplefilter('always')
        with warnings.catch_warnings(record=True) as rec:
            import api
            import engine.errors
        import daedalus.api, daedalus.engine.errors
        assert api is daedalus.api and engine.errors is daedalus.engine.errors
        assert len([w for w in rec if issubclass(w.category, DeprecationWarning)]) == 2
        print('ok')
    ''')
    assert out.strip() == 'ok'


def test_missing_old_submodule_is_an_import_error():
    _run('''
        import daedalus
        try:
            import engine.no_such_module
        except ModuleNotFoundError as e:
            assert e.name in ('engine.no_such_module', 'daedalus.engine.no_such_module'), e.name
        else:
            raise AssertionError('imported a module that does not exist')
    ''')


def test_a_users_own_api_package_wins(tmp_path):
    pkg = tmp_path / 'api'
    pkg.mkdir()
    (pkg / '__init__.py').write_text('MINE = True\n')
    (pkg / 'compute.py').write_text('LOCAL = 1\n')
    out = _run('''
        import sys, daedalus, api, api.compute
        assert getattr(api, 'MINE', False), api
        assert api.compute.LOCAL == 1
        assert sys.modules['api'] is api and 'daedalus.api' not in sys.modules
        print('ok')
    ''', cwd=str(tmp_path), path=(str(tmp_path), ROOT))
    assert out.strip() == 'ok'


def test_alias_keeps_the_modules_own_spec_and_reload_works():
    out = _run('''
        import importlib, sys
        import daedalus
        import engine.errors
        import daedalus.engine.errors as E
        assert E.__spec__.name == 'daedalus.engine.errors', E.__spec__
        assert E.__name__ == 'daedalus.engine.errors'
        assert importlib.reload(E) is E and E.__name__ == 'daedalus.engine.errors'
        assert sys.modules['engine.errors'] is E
        print('ok')
    ''')
    assert out.strip() == 'ok'


def test_a_path_into_the_package_never_loads_a_second_copy():
    """``<repo>/daedalus`` on sys.path makes ``daedalus/api`` look like a
    top-level ``api``: the alias must still win (no second copy)."""
    out = _run('''
        import os, sys, pickle
        import daedalus
        sys.path.insert(0, os.path.dirname(daedalus.__file__))
        import api, engine.errors
        assert api is sys.modules['daedalus.api']
        assert engine.errors is sys.modules['daedalus.engine.errors']
        print('ok')
    ''')
    assert out.strip() == 'ok'


def test_a_bare_folder_named_engine_does_not_hide_the_alias(tmp_path):
    (tmp_path / 'engine').mkdir()                     # e.g. a results folder
    out = _run('''
        import sys, daedalus
        import engine.errors
        assert engine.errors is sys.modules['daedalus.engine.errors']
        print('ok')
    ''', cwd=str(tmp_path), path=('', ROOT))
    assert out.strip() == 'ok'


def test_a_package_next_to_an_installed_daedalus_wins(monkeypatch, tmp_path):
    """Installed (no checkout), a package beside ``daedalus`` in the same
    site-packages is another distribution's, not our forwarding package."""
    from importlib.machinery import ModuleSpec
    from daedalus import _compat
    site_init = os.path.join(_compat._CHECKOUT, 'api', '__init__.py')
    spec = ModuleSpec('api', None, origin=site_init)
    monkeypatch.setattr(_compat, '_in_checkout', lambda: False)
    assert _compat._yields_to(spec, 'api')
    monkeypatch.setattr(_compat, '_in_checkout', lambda: True)
    assert not _compat._yields_to(spec, 'api')


def test_a_local_module_in_the_checkouts_old_folder_still_imports(tmp_path):
    """An untracked local file in ``<repo>/simulations/`` keeps importing
    under its old name (the folder now only forwards)."""
    (tmp_path / 'pyproject.toml').write_text('')
    (tmp_path / 'simulations').mkdir()
    (tmp_path / 'simulations' / 'my_local_sim.py').write_text(
        'from simulations.cumulant_estimator import estimate_kpoint_slices\nOK = 1\n')
    out = _run(f'''
        import sys
        import daedalus
        from daedalus import _compat
        _compat._CHECKOUT = {str(tmp_path)!r}
        import simulations.my_local_sim as m
        assert m.OK == 1 and m.estimate_kpoint_slices is sys.modules[
            'daedalus.simulations.cumulant_estimator'].estimate_kpoint_slices
        try:
            import simulations.not_here
        except ModuleNotFoundError:
            print('ok')
    ''')
    assert out.strip() == 'ok'


def test_find_spec_probe_does_not_use_up_the_warning():
    out = _run('''
        import importlib.util, warnings
        import daedalus
        importlib.util.find_spec('engine')          # e.g. an IDE completer
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            import engine
        assert sum(issubclass(w.category, DeprecationWarning) for w in rec) == 1
        print('ok')
    ''')
    assert out.strip() == 'ok'


def test_old_pickle_names_load_after_import_daedalus():
    import daedalus                                      # noqa: F401  (installs the alias)
    from daedalus.engine.errors import DaedalusError
    blob = pickle.dumps(DaedalusError, protocol=0)
    old = blob.replace(b'daedalus.engine.errors', b'engine.errors')
    assert old != blob
    assert pickle.loads(old) is DaedalusError


def test_forwarding_packages_hold_only_init():
    """The checkout's api/ engine/ simulations/ are forwarding stubs only:
    no tracked module may hide in them."""
    try:
        out = subprocess.run(['git', '-C', ROOT, 'ls-files', 'api', 'engine', 'simulations'],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip('not a git checkout')
    assert sorted(out.split()) == ['api/__init__.py', 'engine/__init__.py',
                                   'simulations/__init__.py']


def _pyproject():
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        return tomllib.load(f)


def test_distribution_ships_only_the_daedalus_package():
    cfg = _pyproject()
    assert cfg['project']['name'] == 'daedalus-msrjd'
    assert cfg['tool']['setuptools']['packages']['find']['include'] == ['daedalus*']
    assert cfg['project']['scripts']['daedalus'] == 'daedalus:main'


def test_every_data_file_in_the_package_is_shipped():
    """Each tracked non-Python file under daedalus/ matches a package-data
    pattern, so the wheel carries it (a new data file must be listed)."""
    try:
        out = subprocess.run(['git', '-C', ROOT, 'ls-files', 'daedalus'],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip('not a git checkout')
    data = _pyproject()['tool']['setuptools']['package-data']
    files = [f for f in out.split() if not f.endswith('.py') or f.endswith('.model.py')]
    assert len(files) > 40, files
    missing = []
    for f in files:
        ok = False
        for pkg, pats in data.items():
            base = pkg.replace('.', '/') + '/'
            if f.startswith(base) and any(fnmatch.fnmatch(f[len(base):], p) for p in pats):
                ok = True
        if not ok:
            missing.append(f)
    assert not missing, missing


def test_spawn_worker_import_is_light():
    """The Phase J worker imports the ``daedalus`` package first: that must
    load neither Sage nor matplotlib.pyplot."""
    out = _run('''
        import sys
        import daedalus.engine._spawn_worker as w
        w._bootstrap()
        print(any(m == 'sage' or m.startswith('sage.') for m in sys.modules),
              'matplotlib.pyplot' in sys.modules)
    ''', cwd='/')
    assert out.split() == ['False', 'False']


def test_python_dash_m_daedalus():
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, '-m', 'daedalus', 'run', '--help'],
                         capture_output=True, text=True, timeout=300, cwd='/', env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'config' in out.stdout.lower()


# ── where load_model looks ─────────────────────────────────────────────

_TOY = '# a model file (only its location matters here): {tag}\n'


def test_models_path_env_and_explicit_paths(tmp_path, monkeypatch):
    import daedalus as dd
    shipped = dd.list_models()
    assert 'ou_quartic' in shipped and len(shipped) >= 20
    assert os.path.isfile(os.path.join(dd.MODELS_DIR, 'ou_quartic.model.py'))
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    (a / 'mine.model.py').write_text(_TOY.format(tag='a'))
    (b / 'mine.model.py').write_text(_TOY.format(tag='b'))
    monkeypatch.setenv(dd.MODELS_PATH_ENV, os.pathsep.join([str(a), str(b)]))
    assert dd.model_dirs()[:2] == [str(a), str(b)]
    assert 'mine' in dd.list_models() and 'ou_quartic' in dd.list_models()
    with pytest.warns(UserWarning, match='more than one model directory'):
        assert dd._model_path('mine') == str(a / 'mine.model.py')
    assert dd._model_path(str(b / 'mine.model.py')) == str(b / 'mine.model.py')
    assert dd._model_path('ou_quartic') == os.path.join(dd.MODELS_DIR, 'ou_quartic.model.py')

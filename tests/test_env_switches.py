"""
The one environment-switch parser (``daedalus.engine._env``, audit W7 RED-05) and the
retired switches (RED-02).
"""
import math
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.engine import _env as E                                    # noqa: E402


@pytest.mark.parametrize('raw,want', [
    ('1', True), ('true', True), (' YES ', True), ('On', True),
    ('0', False), ('false', False), ('no', False), ('OFF', False)])
def test_flag_words(raw, want):
    assert E.flag('X', not want, {'X': raw}) is want


@pytest.mark.parametrize('default', [True, False])
def test_flag_unset_or_empty_is_the_default(default):
    assert E.flag('X', default, {}) is default
    assert E.flag('X', default, {'X': '  '}) is default


@pytest.mark.parametrize('raw', ['2', 'enable', 'nope', 'O'])
def test_flag_garbage_raises_naming_the_switch(raw):
    with pytest.raises(ValueError, match='X='):
        E.flag('X', True, {'X': raw})


def test_choice():
    assert E.choice('X', ('a', 'b'), 'a', {'X': ' B '}) == 'b'
    assert E.choice('X', ('a', 'b'), 'a', {}) == 'a'
    with pytest.raises(ValueError, match="expected one of 'a', 'b'"):
        E.choice('X', ('a', 'b'), 'a', {'X': 'c'})


def test_integer_and_number():
    assert E.integer('N', 3, environ={'N': '7'}) == 7
    assert E.integer('N', 3, environ={}) == 3
    with pytest.raises(ValueError):
        E.integer('N', 3, environ={'N': '7.5'})
    with pytest.raises(ValueError, match='>= 1'):
        E.integer('N', 3, minimum=1, environ={'N': '0'})
    assert E.number('Q', 1.0, environ={'Q': '2.5'}) == 2.5
    for bad in ('nan', 'inf', 'x'):
        with pytest.raises(ValueError):
            E.number('Q', 1.0, environ={'Q': bad})
    with pytest.raises(ValueError, match='positive'):
        E.number('Q', 1.0, positive=True, environ={'Q': '0'})


@pytest.mark.parametrize('raw,want', [
    ('2', 2.0), ('0.5', 0.5), ('off', math.inf), ('None', math.inf),
    ('inf', math.inf), ('unlimited', math.inf)])
def test_budget_gb(raw, want):
    assert E.budget_gb('B', {'B': raw}) == want


@pytest.mark.parametrize('raw', ['0', '-1', 'lots'])
def test_budget_gb_refuses_zero_negative_and_garbage(raw):
    with pytest.raises(ValueError, match='B='):
        E.budget_gb('B', {'B': raw})


def test_budget_unset_is_the_callers_default():
    assert E.budget_gb('B', {}) is None


def test_retired_warns_once_and_reports_set():
    E._RETIRED_SEEN.discard('W7_TEST_RETIRED')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert E.retired('W7_TEST_RETIRED', 'gone', {'W7_TEST_RETIRED': '1'})
        assert E.retired('W7_TEST_RETIRED', 'gone', {'W7_TEST_RETIRED': '0'})
        assert not E.retired('W7_TEST_RETIRED', 'gone', {})
    msgs = [str(x.message) for x in w if issubclass(x.category, FutureWarning)]
    assert msgs == ['W7_TEST_RETIRED is retired and ignored: gone']


# ── the converted readers (RED-05) ──────────────────────────────────────────

def test_one_engine_parser(monkeypatch):
    from daedalus.engine.integration.time_domain import statespace as SS
    for raw, want in (('', 'statespace'), ('default', 'statespace'),
                      (' StateSpace ', 'statespace'), ('poles', 'poles')):
        assert E.phase_j_engine({'DAEDALUS_PHASE_J_ENGINE': raw}) == want
        assert SS.engine_requested({'DAEDALUS_PHASE_J_ENGINE': raw}) is (
            want == 'statespace')
    # every reader refuses a typo (four of five used to read it as default)
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'pole')
    from daedalus.api import compute as C
    from daedalus.engine.integration.time_domain import pipeline as P
    with pytest.raises(ValueError, match='DAEDALUS_PHASE_J_ENGINE'):
        C._ss_route_prediction({}, {}, ())
    with pytest.raises(ValueError, match='DAEDALUS_PHASE_J_ENGINE'):
        P._ss_grid([(0.0, 0.0)])
    with pytest.raises(ValueError, match='DAEDALUS_PHASE_J_ENGINE'):
        SS.engine_requested()


def test_fallback_preflight_tau0_are_validated(monkeypatch):
    from daedalus.engine.integration.time_domain import statespace as SS
    from daedalus.api import compute as C
    assert SS.fallback_enabled({}) is True
    assert SS.fallback_enabled({'DAEDALUS_PHASE_J_SS_FALLBACK': 'off'}) is False
    with pytest.raises(ValueError, match='DAEDALUS_PHASE_J_SS_FALLBACK'):
        SS.fallback_enabled({'DAEDALUS_PHASE_J_SS_FALLBACK': 'maybe'})
    monkeypatch.setenv('DAEDALUS_PREFLIGHT', 'no')
    assert C._preflight_default() is False
    monkeypatch.setenv('DAEDALUS_PREFLIGHT', 'garbage')
    with pytest.raises(ValueError, match='DAEDALUS_PREFLIGHT'):
        C._preflight_default()
    monkeypatch.setenv('DAEDALUS_TAU0', 'NUDGE')
    assert C._tau0_mode() == 'nudge'


@pytest.mark.parametrize('raw', ['0', '-1', 'lots'])
def test_spatial_budget_refuses_zero_and_negative(monkeypatch, raw):
    """``SPATIAL_MEM_BUDGET_GB=0`` used to refuse every spatial loop and
    ``off`` raised; now 0 / negative raise naming the switch and ``off`` is
    no limit, as for ``PHASE_J_MEM_BUDGET_GB``."""
    from daedalus.engine.integration.spatial import full_integrator as F
    monkeypatch.setenv('SPATIAL_MEM_BUDGET_GB', raw)
    with pytest.raises(ValueError, match='SPATIAL_MEM_BUDGET_GB'):
        F._spatial_mem_budget()


def test_spatial_budget_off_and_explicit(monkeypatch):
    from daedalus.engine.integration.spatial import full_integrator as F
    monkeypatch.setenv('SPATIAL_MEM_BUDGET_GB', 'off')
    assert F._spatial_mem_budget()[0] == math.inf
    monkeypatch.setenv('SPATIAL_MEM_BUDGET_GB', '2.5')
    assert F._spatial_mem_budget() == (2.5e9, 'SPATIAL_MEM_BUDGET_GB=2.5')


def _import_in_child(env, code):
    import subprocess
    e = dict(os.environ, PYTHONHASHSEED='0', **env)
    root = os.path.join(os.path.dirname(__file__), '..')
    return subprocess.run([sys.executable, '-W', 'default', '-c', code],
                          cwd=root, env=e, capture_output=True, text=True,
                          timeout=600)


def test_cert_backend_typo_raises_at_import():
    r = _import_in_child({'DAEDALUS_CERT_BACKEND': 'nauty'},
                         'import daedalus.engine.enumeration.loop_diagram_enumeration')
    assert r.returncode != 0 and 'DAEDALUS_CERT_BACKEND' in r.stderr


def test_retired_enumeration_switches_only_warn():
    code = ('import daedalus.engine.enumeration.loop_diagram_enumeration as L; '
            'print(L.EDGE_GENERATOR, L.ORIENTATION_CHECKER, '
            'L.DELTA_ORBIT_DEDUP)')
    r = _import_in_child({'DAEDALUS_EDGE_GENERATOR': 'multiset',
                          'DAEDALUS_ORIENTATION_CHECKER': 'sage',
                          'DAEDALUS_DELTA_ORBIT_DEDUP': '1'}, code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ['canonical', 'integer', 'auto']
    for name in ('DAEDALUS_EDGE_GENERATOR', 'DAEDALUS_ORIENTATION_CHECKER',
                 'DAEDALUS_DELTA_ORBIT_DEDUP'):
        assert f'{name} is retired and ignored' in r.stderr

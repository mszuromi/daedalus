"""
tests/test_save_version_stamp.py
================================
Provenance stamp on saved results (0.2.0, CHANGELOG.md).

``api.save.save_npz`` / ``save_csv`` record ``daedalus_version`` and
``phase_j_convention``; ``api.save.load_npz`` warns once per file and
process (``StaleResultWarning``) when the stamp is missing, older than
0.2.0, or names the pre-0.2.0 Phase J convention.  Also pins that the three
places stating the package version agree, and that ``CITATION.cff``
describes a release: the package version once CHANGELOG.md dates it, the
previous release while its entry is "(unreleased)".

Model-free: every result dict here is hand-built, except in the last test,
which checks that ``compute_cumulants`` records the convention.  It runs
the public one-population spike-reset configuration of
``tests/test_phase_j_theta0.py`` in that file's private empty cwd, and
shares its result cache.

Run:  sage -python -m pytest tests/test_save_version_stamp.py -q
"""
from __future__ import annotations

import os
import re
import sys
import tomllib
import warnings

import numpy as np
import pytest

import api
import api.save as save
import engine.integration.time_domain.final_integral as FI

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── helpers ───────────────────────────────────────────────────────────

def _minimal_result():
    grid = np.array([-1.0, 0.0, 1.0])
    curve = np.array([0.5 + 0j, 1.0 + 0j, 0.5 + 0j])
    loop = np.array([0.01 + 0j, -0.02 + 0j, 0.01 + 0j])
    return {
        'config': {'k': 2, 'max_ell': 1, 'fundamental': {'mu': 0.1, 'D': 1.0},
                   'external_fields': [('dx', 1), ('dx', 1)],
                   'model_name': 'stamp_test'},
        'tau_grid': grid,
        'C_tau': curve + loop,
        'C_tau_by_ell': {0: curve, 1: loop},
        'mf_values': {'xstar': [0.0]},
    }


def _write_npz(path, **stamp):
    """An npz with the save_npz schema but an arbitrary (or no) stamp,
    standing in for a file written by an older version."""
    payload = {'k': np.array([2]), 'max_ell': np.array([0]),
               'tau_grid': np.array([0.0, 1.0]),
               'C_total': np.array([1.0 + 0j, 0.5 + 0j])}
    for key, val in stamp.items():
        payload[key] = np.asarray([val])
    np.savez(path, **payload)
    return str(path)


def _stale_warnings(record):
    return [w for w in record if issubclass(w.category, save.StaleResultWarning)]


def _load_recording(path, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        data = save.load_npz(path, **kw)
    return data, _stale_warnings(rec)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """Fresh warn-once registry and the default Phase J mode per test."""
    monkeypatch.setattr(save, '_STALE_WARNED', set())
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'ito')


# ── version strings agree ─────────────────────────────────────────────

def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding='utf-8') as f:
        return f.read()


def test_version_strings_in_sync():
    with open(os.path.join(_ROOT, 'pyproject.toml'), 'rb') as f:
        v_pyproject = tomllib.load(f)['project']['version']
    v_dd = re.search(r"^__version__\s*=\s*'([^']+)'", _read('daedalus.py'), re.M)
    v_api_src = re.search(r"^__version__\s*=\s*'([^']+)'",
                          _read(os.path.join('api', '__init__.py')), re.M)
    assert v_dd and v_api_src
    versions = {'pyproject.toml': v_pyproject,
                'daedalus.py': v_dd.group(1),
                'api/__init__.py': v_api_src.group(1),
                'api.__version__': api.__version__}
    assert len(set(versions.values())) == 1, versions
    assert save._parse_version(api.__version__) >= save._STAMP_MIN_VERSION


def _changelog_releases():
    """``[(version, date-or-None)]`` of the CHANGELOG entries, newest first
    (``## 0.2.0 (unreleased)`` -> None, ``## 0.2.0 (2026-10-01)`` -> date)."""
    out = []
    for m in re.finditer(r'^## (\S+)(?: \(([^)]*)\))?', _read('CHANGELOG.md'),
                         re.M):
        when = m.group(2)
        out.append((m.group(1), None if when in (None, 'unreleased') else when))
    return out


def test_citation_describes_a_release():
    """CITATION.cff names a released version with its release date: the
    package version once the CHANGELOG dates it, otherwise an older one
    (the last release; bump version and date-released together when the
    package version is released)."""
    cff = _read('CITATION.cff')
    v_cff = re.search(r'^version:\s*"([^"]+)"', cff, re.M)
    date = re.search(r'^date-released:\s*"(\d{4}-\d{2}-\d{2})"', cff, re.M)
    assert v_cff and date
    v_cff = v_cff.group(1)
    releases = _changelog_releases()
    assert releases and releases[0][0] == api.__version__
    if releases[0][1] is None:                     # package version unreleased
        assert (save._parse_version(v_cff)
                < save._parse_version(api.__version__)), v_cff
    else:
        assert v_cff == api.__version__
        assert date.group(1) == releases[0][1]


def test_version_lookup_without_the_package_imported(monkeypatch):
    # Standalone use (numpy-only tests load api/save.py by path): the
    # version comes from the api/__init__.py source, not an import.
    expected = api.__version__
    monkeypatch.delitem(sys.modules, 'api')
    assert save._daedalus_version() == expected


def test_changelog_has_an_entry_for_this_version():
    assert re.search(rf'^## {re.escape(api.__version__)}\b',
                     _read('CHANGELOG.md'), re.M)


# ── stamp written on save ─────────────────────────────────────────────

def test_npz_round_trip_carries_current_stamp(tmp_path):
    res = _minimal_result()
    out = save.save_npz(res, str(tmp_path / 'r.npz'))
    with np.load(out) as z:
        assert z['daedalus_version'].tolist() == [api.__version__]
        assert z['phase_j_convention'].tolist() == [save.PHASE_J_CONVENTION]
    data, stale = _load_recording(out)
    assert stale == []
    assert save.PHASE_J_CONVENTION == 'theta0_ito_const_rows'
    assert str(data['daedalus_version'][0]) == api.__version__
    np.testing.assert_array_equal(data['C_total'], res['C_tau'])
    np.testing.assert_array_equal(data['C_1_loop'], res['C_tau_by_ell'][1])
    np.testing.assert_array_equal(data['tau_grid'], res['tau_grid'])


def test_extra_cannot_override_the_stamp(tmp_path):
    out = save.save_npz(_minimal_result(), str(tmp_path / 'r.npz'),
                        extra={'daedalus_version': ['0.0.1'],
                               'phase_j_convention': ['theta0_legacy_clip'],
                               'C_sim_mean': [1.0, 2.0, 3.0]})
    data, stale = _load_recording(out)
    assert stale == []
    assert data['daedalus_version'].tolist() == [api.__version__]
    assert data['phase_j_convention'].tolist() == [save.PHASE_J_CONVENTION]
    np.testing.assert_array_equal(data['C_sim_mean'], [1.0, 2.0, 3.0])


def test_csv_header_carries_the_stamp(tmp_path):
    out = save.save_csv(_minimal_result(), str(tmp_path / 'r.csv'))
    with open(out, encoding='utf-8') as f:
        header = [ln.rstrip('\n') for ln in f if ln.startswith('#')]
    assert f'# daedalus_version: {api.__version__}' in header
    assert f'# phase_j_convention: {save.PHASE_J_CONVENTION}' in header


def test_legacy_mode_is_stamped_and_warns_on_load(tmp_path, monkeypatch):
    # A result dict without a compute-time record (hand-built): the rule in
    # force when the file is written.
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'legacy_clip')
    out = save.save_npz(_minimal_result(), str(tmp_path / 'legacy.npz'))
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'ito')
    data, stale = _load_recording(out)
    assert data['phase_j_convention'].tolist() == ['theta0_legacy_clip']
    assert len(stale) == 1
    assert 'legacy' in str(stale[0].message)


@pytest.mark.parametrize('computed, saved_under', [
    ('theta0_legacy_clip', 'ito'),
    ('theta0_ito_const_rows', 'legacy_clip'),
])
def test_stamp_is_the_convention_the_numbers_were_computed_with(
        tmp_path, monkeypatch, computed, saved_under):
    # compute_cumulants records the rule in result['config'] when it
    # computes; changing the flags before saving must not relabel the file.
    res = _minimal_result()
    res['config']['phase_j_convention'] = computed
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', saved_under)
    out = save.save_npz(res, str(tmp_path / 'r.npz'))
    csv = save.save_csv(res, str(tmp_path / 'r.csv'))
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'ito')
    data, stale = _load_recording(out)
    assert data['phase_j_convention'].tolist() == [computed]
    assert len(stale) == (1 if computed == 'theta0_legacy_clip' else 0)
    with open(csv, encoding='utf-8') as f:
        assert f'# phase_j_convention: {computed}\n' in f.read()


def test_convention_follows_the_environment_when_phase_j_not_loaded(
        monkeypatch):
    # Without final_integral in sys.modules the environment decides.
    monkeypatch.delitem(sys.modules, save._FI_MODULE)
    monkeypatch.delenv('DAEDALUS_PHASE_J_LEGACY', raising=False)
    monkeypatch.delenv('DAEDALUS_PHASE_J_THETA0_CONST_ROW', raising=False)
    assert save._phase_j_convention() == 'theta0_ito_const_rows'
    monkeypatch.setenv('DAEDALUS_PHASE_J_LEGACY', '1')
    assert save._phase_j_convention() == 'theta0_legacy_clip'


@pytest.mark.parametrize('env', [
    {},
    {'DAEDALUS_PHASE_J_LEGACY': '1'},
    {'DAEDALUS_PHASE_J_LEGACY': 'true', 'DAEDALUS_PHASE_J_THETA0_CONST_ROW': 'ito'},
    {'DAEDALUS_PHASE_J_LEGACY': '0', 'DAEDALUS_PHASE_J_THETA0_CONST_ROW': 'legacy_clip'},
    {'DAEDALUS_PHASE_J_LEGACY': 'off'},
    {'DAEDALUS_PHASE_J_THETA0_CONST_ROW': ' ITO '},
    {'DAEDALUS_PHASE_J_THETA0_CONST_ROW': ''},
])
def test_env_mirror_matches_final_integral(env):
    assert (save._theta0_mode_from_env(env)
            == FI._initial_phase_j_flags(env)['THETA0_CONST_ROW_MODE'])


# ── warning on load ───────────────────────────────────────────────────

def test_missing_stamp_warns_once_per_file(tmp_path):
    old = _write_npz(tmp_path / 'old.npz')
    other = _write_npz(tmp_path / 'other.npz')
    with pytest.warns(save.StaleResultWarning,
                      match='no daedalus_version stamp') as rec:
        data = save.load_npz(old)
    assert len(_stale_warnings(rec)) == 1
    msg = str(_stale_warnings(rec)[0].message)
    assert 'recomputed' in msg and 'instantaneous' in msg
    # both 0.2.0 changes are named: Θ(0) = 0 and the external-time tie order
    assert 'Θ(0) = 0' in msg and 'tie order' in msg and 'k ≥ 3' in msg
    np.testing.assert_array_equal(data['C_total'], [1.0 + 0j, 0.5 + 0j])
    # the same file again: silent; a different stale file: warns
    assert _load_recording(old)[1] == []
    assert len(_load_recording(other)[1]) == 1


@pytest.mark.parametrize('stamp, match', [
    ({'daedalus_version': '0.1.0',
      'phase_j_convention': 'theta0_ito_const_rows'}, 'written by daedalus 0.1.0'),
    ({'daedalus_version': '0.1.9'}, 'written by daedalus 0.1.9'),
    ({'daedalus_version': '0.2.0'}, 'no phase_j_convention stamp'),
    ({'daedalus_version': 'dev',
      'phase_j_convention': 'theta0_ito_const_rows'}, 'not a version'),
    ({'daedalus_version': '0.2.0',
      'phase_j_convention': 'theta0_legacy_clip'}, 'pre-0.2.0 Phase J convention'),
])
def test_old_or_incomplete_stamp_warns(tmp_path, stamp, match):
    path = _write_npz(tmp_path / 'r.npz', **stamp)
    with pytest.warns(save.StaleResultWarning, match=match):
        save.load_npz(path)


@pytest.mark.parametrize('version', ['0.2.0', '0.2.1', '0.3.0rc1', '1.0'])
def test_current_or_newer_stamp_is_silent(tmp_path, version):
    path = _write_npz(tmp_path / 'r.npz', daedalus_version=version,
                      phase_j_convention='theta0_ito_const_rows')
    assert _load_recording(path)[1] == []


def test_warn_stale_false_skips_the_check(tmp_path):
    path = _write_npz(tmp_path / 'old.npz')
    data, stale = _load_recording(path, warn_stale=False)
    assert stale == [] and 'C_total' in data


@pytest.mark.parametrize('text, expected', [
    ('0.2.0', (0, 2, 0)), ('0.2', (0, 2, 0)), ('v1.4.7', (1, 4, 7)),
    ('0.3.0rc1', (0, 3, 0)), ('0.10.2+local', (0, 10, 2)), ('dev', None),
    ('', None),
])
def test_parse_version(text, expected):
    assert save._parse_version(text) == expected


# ── public surface ────────────────────────────────────────────────────

def test_load_npz_is_exported():
    import daedalus as dd
    assert api.load_npz is save.load_npz
    assert dd.load_npz is save.load_npz
    assert 'load_npz' in api.__all__


# ── compute_cumulants records the convention (one public model) ───────
#
# Last in this file on purpose: the module-scoped ``private_cwd`` fixture
# stays in force until the module ends.  The fixtures and the memoised
# runner come from tests/test_phase_j_theta0.py.

from tests.test_phase_j_theta0 import (                    # noqa: E402,F401
    _spike1, private_cwd, spike1_model)


def test_compute_records_the_convention_it_used(spike1_model):
    """``compute_cumulants`` stamps ``config['phase_j_convention']`` with the
    rule the numbers were computed with (``api.save`` copies it)."""
    for mode, label in (('ito', save.PHASE_J_CONVENTION),
                        ('legacy_clip', 'theta0_legacy_clip')):
        res, _c = _spike1(spike1_model, 2, 1, False, mode=mode)
        assert res['config']['phase_j_convention'] == label
        assert save._result_stamp(res)['phase_j_convention'] == label

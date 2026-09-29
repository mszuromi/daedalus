"""
tests/test_phase_j_legacy_baseline.py
=====================================
Pre-change (M0.3) Phase J baseline guard (``docs/integration_speedup_plan.md``
§3.1 "Rollback and flag hygiene", §4.3, §4.4).

The data files:

* ``tests/fixtures/phase_j_legacy_baseline.npz`` (and its readable ``.json``
  twin) -- the model-zoo baseline written by
  ``tests/tools/phase_j_zoo_baseline.py`` on the pre-M1 code (PYTHONHASHSEED=0);
* ``tests/phase_j_refactor_fixtures/legacy/*.npz`` -- byte-for-byte copies of
  the four frozen Phase J fixtures, taken before M1.

Neither is ever refrozen.  For now (M0) this file checks that:

1. the baseline loads, is self-consistent and contains no local-only model;
2. the legacy fixture copies are byte-identical to their M0 hashes;
3. a few cheap zoo entries reproduce on the current code to rtol 1e-13 (plan
   §4.4: hash randomisation causes 1-ulp jitter, so ``==`` is not used in-repo),
   and so do their nquad / zero-normal counters.

TODO(M1): this becomes the legacy-umbrella test.  Every entry must then
reproduce under ``DAEDALUS_PHASE_J_LEGACY=1`` (both kernel backends at M9),
and with the new defaults only entries whose baseline ``m1_expected`` is
'unchanged' may be compared directly.  Entries marked ``# M1 moves`` below
must be switched to the legacy flag at M1.

Cost: about 20 s (three small models, warm ``saved_models/`` cache).
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.tools import phase_j_zoo_baseline as Z   # noqa: E402

_LEGACY_DIR = os.path.join(os.path.dirname(__file__),
                           'phase_j_refactor_fixtures', 'legacy')
_LEGACY_SHA256 = {
    'spike_reset_k1_ell1.npz':
        '7814492635b59703f4209771e8b0a555817dcf18f8ab65273b9fc2d809eaf079',
    'spike_reset_k2_ell0.npz':
        '50e50379c3f8e73778b01ab5f278acbb4839b905bbe2eca62d40b75bfa6f205d',
    'spike_reset_k2_ell1.npz':
        '2fafe8fad0107bceb25cbf40872667fc3e8757c56d834713af7fca3c106d6ec7',
    'quad_exp_k2_ell0.npz':
        'ac2d0c0ec008ac4428b6f5f539af79e1ede530751080d9d6f9b1295d0844993e',
}

# Cheap entries re-run in-process.  Keep the default-suite cost <= ~60 s.
_REPRODUCE = [
    'ou_quartic-k2-l3',          # M1: unchanged (0 nquad, no zero-normal rows)
    'spike_reset-k1-l1',         # M1: unchanged (measured bit-identical)
    # M1 moves: TODO(M1) legacy flag.  Also NOT bit-reproducible run to run
    # (1-2 ulp, measured at M0.3: max rel 7.7e-16), hence rtol 1e-13.
    'spike_reset-k2-l1',
]

_COUNTER_KEYS = ('nquad_calls', 'polytope_m0_direct', 'scipy_nquad_called_m1',
                 'scipy_nquad_called_m2', 'scipy_nquad_called_mge3',
                 'nquad_fallback_by_reason', 'zero_normal_rows_seen',
                 'polygon_attempted', 'poset_attempted')


@pytest.fixture(scope='module')
def baseline():
    if not os.path.exists(Z.TRACKED_NPZ):
        pytest.fail(f'missing baseline {Z.TRACKED_NPZ}: regenerate with '
                    '`PYTHONHASHSEED=0 sage -python -m '
                    'tests.tools.phase_j_zoo_baseline --run tracked '
                    '--assemble` on the PRE-M1 code only')
    return Z.load_baseline(Z.TRACKED_NPZ)


def test_legacy_fixture_copies_are_frozen():
    """The four pre-M1 fixture copies are byte-identical to their M0 hashes."""
    for fname, sha in _LEGACY_SHA256.items():
        path = os.path.join(_LEGACY_DIR, fname)
        assert os.path.exists(path), f'missing legacy fixture {fname}'
        with open(path, 'rb') as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        assert got == sha, (f'{fname} changed: legacy fixtures are never '
                            'refrozen')


def test_baseline_loads_and_is_consistent(baseline):
    meta, arrays = baseline
    assert meta['schema_version'] == Z.SCHEMA_VERSION
    assert meta['destination'] == 'tracked'
    entries = meta['entries']
    tracked_zoo = [e['name'] for e in Z.all_entries(include_local=False)
                   if Z.destination(e) == 'tracked']
    # every tracked zoo entry is recorded (possibly as TIMEOUT/ERR)
    missing = sorted(set(tracked_zoo) - set(entries))
    assert not missing, f'zoo entries absent from the baseline: {missing}'
    # nothing local-only leaked into the tracked file: neither a zoo entry
    # routed to the local file nor an entry of a local extra-zoo file
    # (``DAEDALUS_ZOO_EXTRA``; none on a fresh clone)
    local = {e['name'] for e in Z.all_entries(include_local=True)
             if Z.destination(e) == 'local'}
    leaked = sorted((set(entries) | set(arrays)) & local)
    assert not leaked, f'local-only entries in the tracked baseline: {leaked}'
    for name, s in entries.items():
        # every model behind a tracked entry is tracked (or new, not ignored)
        cfg = s.get('config') or {}
        path = Z.model_path(cfg) if (cfg.get('model') or
                                     cfg.get('model_file')) else None
        if path is not None:
            assert Z.git_route(path) in ('tracked', 'new'), (name, path)
        if s['status'] == 'ok':
            assert s['pythonhashseed'] == '0', name
            assert s['model_git'] in ('tracked', 'new'), name
            assert name in arrays and 'total' in arrays[name], name
            assert np.all(np.isfinite(arrays[name]['total'])), name
        if s['status'] == 'census_only':
            assert name not in arrays, 'census-only values must not be stored'
    # the pre-M1 code state is recorded
    assert meta['code']['git_head']
    assert len(meta['code']['sha256_final_integral']) == 64


@pytest.mark.parametrize('name', _REPRODUCE)
def test_cheap_entries_reproduce(baseline, name):
    """Pre-M1: the current code reproduces the baseline to rtol 1e-13."""
    meta, arrays = baseline
    s = meta['entries'].get(name)
    if s is None or s['status'] != 'ok':
        pytest.fail(f'{name}: no ok baseline entry ({s and s["status"]})')
    entry = Z.entry_by_name(name, include_local=False)
    rec = Z.run_entry(entry, with_provenance=False)
    assert rec['status'] == 'ok', rec.get('status')
    assert rec['params_effective'] == s['params_effective'], name
    # The MODEL must be the one baselined (its file may change elsewhere).
    assert rec['model_spec_signature'] == s['model_spec_signature'], (
        f'{name}: model definition changed since the baseline')
    cmp = Z.compare_values(arrays[name], Z.result_arrays(rec))
    bad = {k: v for k, v in cmp.items() if not v[2]}
    assert not bad, f'{name}: (max_abs, max_rel, ok) per array: {bad}'
    # Control flow is deterministic: counters reproduce exactly.
    for key in _COUNTER_KEYS:
        assert rec['counters'][key] == s['counters'][key], (name, key)

"""
tests/test_shipped_prediagrams.py
=================================
The prediagram cells shipped with the package
(``engine/enumeration/shipped_prediagrams/``) must (i) exist for every cell in
``SHIPPED_CELLS`` with the Table I prediagram count, (ii) agree, up to
isomorphism, with a fresh enumeration by the current code, so the files cannot
drift from the algorithm, and (iii) be found by ``load_prediagrams`` from any
working directory when no local cache exists -- the fresh-clone situation.

Run:  sage -python -m pytest tests/test_shipped_prediagrams.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.enumeration.loop_diagram_enumeration as L
from engine.enumeration import prediagram_cache as pdc

TABLE_I = {(1, 1): 1, (1, 2): 15, (1, 3): 434, (1, 4): 22332,
           (2, 0): 1, (2, 1): 9, (2, 2): 283, (2, 3): 14928, (2, 4): 1152032,
           (3, 0): 3, (3, 1): 80, (3, 2): 4496, (3, 3): 358983,
           (4, 0): 13, (4, 1): 755, (4, 2): 65956,
           (5, 0): 69, (5, 1): 7412, (5, 2): 922728,
           (6, 0): 448, (6, 1): 75253}


def test_every_shipped_cell_exists_with_the_table_i_count():
    assert set(pdc.SHIPPED_CELLS) == set(TABLE_I)
    for (k, ell), n in TABLE_I.items():
        assert pdc.shipped_exists(k, ell), (k, ell)
        certs = pdc.load_shipped_certs(k, ell)
        assert len(certs) == n, (k, ell, len(certs))
        assert all(isinstance(b, bytes) for b in certs)


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1), (2, 2), (1, 3), (3, 2), (2, 3)])
def test_shipped_cells_match_a_fresh_enumeration(k, ell):
    fresh = L.stream_prediagram_certs(k, ell, n_procs=1)
    assert pdc.recanonicalize(pdc.load_shipped_certs(k, ell)) == fresh


def test_load_prediagrams_finds_shipped_files_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'auto')
    monkeypatch.chdir(tmp_path)                       # no local cache anywhere here
    root = str(tmp_path / 'saved_prediagrams')
    recs, source = pdc.load_prediagrams(root, 3, 1)
    assert source == 'shipped' and len(recs) == 80
    assert all(rec[2] == [0, 1, 2] for rec in recs)   # leaves-first convention kept
    assert os.path.isfile(pdc.v2_path(root, 3, 1))    # copied into the local root ...
    recs2, source2 = pdc.load_prediagrams(root, 3, 1)
    assert source2 == 'v2' and len(recs2) == 80       # ... so the next run is a local hit
    recs0, source0 = pdc.load_prediagrams(root, 1, 0)  # not shipped: computed, written
    assert source0 == 'computed' and recs0 == []
    assert os.path.isfile(pdc.v2_path(root, 1, 0))


def test_rebuild_writes_an_identical_set(tmp_path, monkeypatch):
    monkeypatch.setattr(pdc, 'SHIPPED_DIR', str(tmp_path))
    path, n = pdc.write_shipped_cell(2, 2, n_procs=1)
    assert n == 283 and os.path.isfile(path)
    assert pdc.load_shipped_certs(2, 2) == L.stream_prediagram_certs(2, 2, n_procs=1)


# ── the manifest ────────────────────────────────────────────────────────────

def test_manifest_lists_every_file_and_every_entry_checks_out():
    import json
    man = pdc.load_manifest()
    assert man['convention'] == pdc.CONVENTION
    entries = man['files']
    names = {e['name'] for e in entries}
    on_disk = {n for n in os.listdir(pdc.SHIPPED_DIR) if n != pdc.MANIFEST_NAME}
    assert names == on_disk, (names ^ on_disk)            # lists every file, no stray
    assert {(e['k'], e['ell']) for e in entries} == set(pdc.SHIPPED_CELLS)
    for e in entries:
        path = os.path.join(pdc.SHIPPED_DIR, e['name'])
        assert os.path.getsize(path) == e['size'], e['name']
        assert pdc._sha256(path) == e['sha256'], e['name']
        assert e['count'] == TABLE_I[(e['k'], e['ell'])], e['name']
        assert e['size'] <= 5 * 1024 * 1024, e['name']    # the tracked-file size rule
        assert e['convention'] == pdc.CONVENTION


def test_manifest_entries_load_with_the_stated_count():
    for e in pdc.load_manifest()['files']:
        certs, stamp = pdc.read_cert_file(
            os.path.join(pdc.SHIPPED_DIR, e['name']), e['k'], e['ell'])
        assert len(certs) == e['count'] and stamp['count'] == e['count']
        assert pdc.shipped_exists(e['k'], e['ell'])


def test_xz_shipped_cell_loads_through_load_prediagrams(tmp_path, monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'auto')
    assert pdc.shipped_path(6, 1).endswith('.pkl.xz')
    recs, source = pdc.load_prediagrams(str(tmp_path / 'c'), 6, 1)
    assert source == 'shipped' and len(recs) == 75253
    assert all(rec[2] == list(range(6)) for rec in recs[:200])


def test_manifest_rebuild_is_reproducible(tmp_path):
    import shutil
    d = tmp_path / 'ship'
    d.mkdir()
    for n in ('prediagrams_v2_k2_l2.pkl', 'prediagrams_v2_k3_l1.pkl'):
        shutil.copy(os.path.join(pdc.SHIPPED_DIR, n), d / n)
    got = pdc.build_manifest(str(d))
    want = [e for e in pdc.load_manifest()['files']
            if e['name'] in ('prediagrams_v2_k2_l2.pkl', 'prediagrams_v2_k3_l1.pkl')]
    assert got == want

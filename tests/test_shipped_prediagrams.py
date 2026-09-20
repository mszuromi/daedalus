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
           (2, 0): 1, (2, 1): 9, (2, 2): 283, (2, 3): 14928,
           (3, 0): 3, (3, 1): 80, (3, 2): 4496,
           (4, 0): 13, (4, 1): 755, (4, 2): 65956,
           (5, 0): 69, (5, 1): 7412, (6, 0): 448}


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

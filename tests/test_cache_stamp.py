"""
tests/test_cache_stamp.py
=========================
The v2 prediagram cert files carry a version stamp.  A stamped file is
checked on load (convention, format, edge pack, cell, class count); an
unstamped legacy file (a bare set of certs) is accepted; a stamp from another
convention is refused with a message naming both stamps.

Run:  sage -python -m pytest tests/test_cache_stamp.py -q
"""
from __future__ import annotations

import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.enumeration.loop_diagram_enumeration as L
from engine.enumeration import prediagram_cache as pdc


def test_every_shipped_file_is_stamped_and_consistent():
    for k, ell in pdc.SHIPPED_CELLS:
        certs, stamp = pdc.read_cert_file(pdc.shipped_path(k, ell), k, ell)
        assert stamp is not None, (k, ell)
        assert stamp['convention'] == pdc.CONVENTION == 'prediagrams_v1'
        assert stamp['edge_pack'] == pdc.EDGE_PACK
        assert (stamp['k'], stamp['ell'], stamp['count']) == (k, ell, len(certs))
        assert stamp['cert_backend'] in ('sage', 'fast')
        assert stamp['sage_version']


def test_writer_stamps_and_round_trips(tmp_path):
    certs = L.stream_prediagram_certs(2, 1, n_procs=1)
    root = str(tmp_path)
    path = pdc.save_v2_certs(root, 2, 1, certs)
    got, stamp = pdc.read_cert_file(path, 2, 1)
    assert got == set(certs)
    assert stamp['cert_backend'] == L.CERT_BACKEND and stamp['count'] == 9
    assert pdc.load_v2_certs(root, 2, 1) == set(certs)
    # an explicit backend is recorded verbatim
    pdc.save_v2_certs(root, 2, 1, certs, cert_backend='sage')
    assert pdc.read_cert_file(path)[1]['cert_backend'] == 'sage'


def test_legacy_unstamped_file_is_accepted(tmp_path):
    certs = L.stream_prediagram_certs(2, 1, n_procs=1)
    path = tmp_path / 'legacy.pkl'
    with open(path, 'wb') as f:
        pickle.dump(set(certs), f)
    got, stamp = pdc.read_cert_file(str(path), 2, 1)
    assert got == set(certs) and stamp is None


def test_convention_mismatch_is_refused_naming_both_stamps(tmp_path):
    certs = L.stream_prediagram_certs(2, 1, n_procs=1)
    stamp = pdc.make_stamp(2, 1, certs)
    stamp['convention'] = 'prediagrams_v9'
    path = tmp_path / 'bad.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'stamp': stamp, 'certs': sorted(certs)}, f)
    with pytest.raises(ValueError) as e:
        pdc.read_cert_file(str(path), 2, 1)
    msg = str(e.value)
    assert 'prediagrams_v9' in msg and 'prediagrams_v1' in msg and 'Delete' in msg


@pytest.mark.parametrize('field,value', [('edge_pack', 99), ('format', 99)])
def test_format_or_edge_pack_mismatch_is_refused(tmp_path, field, value):
    certs = L.stream_prediagram_certs(2, 1, n_procs=1)
    stamp = pdc.make_stamp(2, 1, certs)
    stamp[field] = value
    path = tmp_path / 'bad.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'stamp': stamp, 'certs': sorted(certs)}, f)
    with pytest.raises(ValueError, match='does not match'):
        pdc.read_cert_file(str(path), 2, 1)


def test_wrong_cell_and_wrong_count_are_refused(tmp_path):
    certs = L.stream_prediagram_certs(2, 1, n_procs=1)
    path = pdc.write_cert_file(str(tmp_path / 'a.pkl'), 2, 1, certs) and str(tmp_path / 'a.pkl')
    with pytest.raises(ValueError, match='stamped for cell'):
        pdc.read_cert_file(path, 3, 1)
    stamp = pdc.make_stamp(2, 1, certs)
    stamp['count'] += 1
    bad = tmp_path / 'b.pkl'
    with open(bad, 'wb') as f:
        pickle.dump({'stamp': stamp, 'certs': sorted(certs)}, f)
    with pytest.raises(ValueError, match='classes'):
        pdc.read_cert_file(str(bad), 2, 1)


@pytest.mark.parametrize('k,ell', [(2, 0), (2, 1), (2, 2), (3, 1)])
def test_stamped_shipped_equals_fresh_enumeration(k, ell):
    fresh = L.stream_prediagram_certs(k, ell, n_procs=1)
    assert pdc.recanonicalize(pdc.load_shipped_certs(k, ell)) == pdc.recanonicalize(fresh)


def test_load_prediagrams_refuses_a_mismatched_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'auto')
    root = str(tmp_path / 'c')
    certs = L.stream_prediagram_certs(2, 1, n_procs=1)
    stamp = pdc.make_stamp(2, 1, certs)
    stamp['convention'] = 'prediagrams_v9'
    p = pdc.v2_path(root, 2, 1)
    os.makedirs(os.path.dirname(p))
    with open(p, 'wb') as f:
        pickle.dump({'stamp': stamp, 'certs': sorted(certs)}, f)
    with pytest.raises(ValueError, match='prediagrams_v9'):
        pdc.load_prediagrams(root, 2, 1)


def test_shipped_copy_into_local_root_keeps_the_backend(tmp_path, monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'auto')
    root = str(tmp_path / 'c')
    recs, src = pdc.load_prediagrams(root, 2, 1)
    assert src == 'shipped' and len(recs) == 9
    stamp = pdc.read_cert_file(pdc.v2_path(root, 2, 1), 2, 1)[1]
    assert stamp['cert_backend'] == pdc.shipped_stamp(2, 1)['cert_backend']

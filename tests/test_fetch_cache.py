"""
tests/test_fetch_cache.py
=========================
``fetch_cache`` downloads prediagram cells that are not tracked in git, only
when called, only from an explicit base URL, and refuses anything whose size,
SHA-256, stamp or class count disagrees with the manifest.  No network: the
"server" is a temporary ``file://`` directory holding a fake cell.

Run:  sage -python -m pytest tests/test_fetch_cache.py -q
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.enumeration import prediagram_cache as pdc

CELL = (9, 9)                      # not a shipped cell
FAKE_CERTS = {b'\x03\x00\x01\x00\x02', b'\x04\x00\x01\x00\x02\x01\x03', b'\x02\x00\x01'}


def _make_server(tmp_path, cell=CELL, certs=FAKE_CERTS, xz=True):
    server = tmp_path / 'server'
    server.mkdir()
    name = f'prediagrams_v2_k{cell[0]}_l{cell[1]}.pkl' + ('.xz' if xz else '')
    path = str(server / name)
    pdc.write_cert_file(path, cell[0], cell[1], certs, 'sage')
    data = open(path, 'rb').read()
    manifest = {'files': [{'name': name, 'k': cell[0], 'ell': cell[1],
                           'count': len(certs), 'size': len(data),
                           'sha256': hashlib.sha256(data).hexdigest(),
                           'core': True, 'url': ''}]}
    return server, path, manifest


def _url(server):
    return 'file://' + str(server)


def test_fetch_writes_a_verified_file_that_load_finds(tmp_path):
    server, _, man = _make_server(tmp_path)
    dest = str(tmp_path / 'cache')
    rep = pdc.fetch_cache('core', dest=dest, base_url=_url(server), manifest=man,
                          verbose=False)
    assert rep == {CELL: 'fetched'}
    assert pdc.v2_exists(dest, *CELL)
    assert pdc.load_v2_certs(dest, *CELL) == FAKE_CERTS
    assert not [n for n in os.listdir(os.path.join(dest, pdc.V2_SUBDIR)) if n.endswith('.part')]
    # second call: nothing to do, nothing downloaded
    rep2 = pdc.fetch_cache([CELL], dest=dest, base_url='file:///nonexistent', manifest=man,
                           verbose=False)
    assert rep2 == {CELL: 'present'}


def test_corrupted_copy_is_refused_and_leaves_nothing(tmp_path):
    server, path, man = _make_server(tmp_path)
    data = bytearray(open(path, 'rb').read())
    data[len(data) // 2] ^= 0xFF                      # same size, different bytes
    open(path, 'wb').write(bytes(data))
    dest = str(tmp_path / 'cache')
    with pytest.raises(ValueError, match='SHA-256 mismatch|corrupt|LZMA|lzma'):
        pdc.fetch_cache('core', dest=dest, base_url=_url(server), manifest=man, verbose=False)
    outdir = os.path.join(dest, pdc.V2_SUBDIR)
    assert not os.path.isdir(outdir) or os.listdir(outdir) == []
    assert not pdc.v2_exists(dest, *CELL)


def test_truncated_copy_is_refused(tmp_path):
    server, path, man = _make_server(tmp_path)
    data = open(path, 'rb').read()
    open(path, 'wb').write(data[:-10])
    with pytest.raises(ValueError, match='bytes'):
        pdc.fetch_cache('core', dest=str(tmp_path / 'c'), base_url=_url(server),
                        manifest=man, verbose=False)


def test_right_hash_but_wrong_convention_is_refused(tmp_path):
    server, path, man = _make_server(tmp_path)
    stamp = pdc.make_stamp(*CELL, FAKE_CERTS)
    stamp['convention'] = 'prediagrams_v9'
    import lzma
    with lzma.open(path, 'wb') as f:
        pickle.dump({'stamp': stamp, 'certs': sorted(FAKE_CERTS)}, f)
    data = open(path, 'rb').read()
    man['files'][0].update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    dest = str(tmp_path / 'c')
    with pytest.raises(ValueError, match='prediagrams_v9'):
        pdc.fetch_cache('core', dest=dest, base_url=_url(server), manifest=man, verbose=False)
    assert not pdc.v2_exists(dest, *CELL)


def test_wrong_class_count_is_refused(tmp_path):
    server, _, man = _make_server(tmp_path)
    man['files'][0]['count'] += 1
    with pytest.raises(ValueError, match='classes'):
        pdc.fetch_cache('core', dest=str(tmp_path / 'c'), base_url=_url(server),
                        manifest=man, verbose=False)


def test_no_base_url_is_an_error_and_env_var_is_used(tmp_path, monkeypatch):
    server, _, man = _make_server(tmp_path)
    monkeypatch.delenv(pdc.FETCH_URL_ENV, raising=False)
    with pytest.raises(ValueError, match=pdc.FETCH_URL_ENV):
        pdc.fetch_cache('core', dest=str(tmp_path / 'c'), manifest=man, verbose=False)
    monkeypatch.setenv(pdc.FETCH_URL_ENV, _url(server))
    assert pdc.fetch_cache('core', dest=str(tmp_path / 'c'), manifest=man,
                           verbose=False) == {CELL: 'fetched'}


def test_unknown_cell_and_missing_hash_are_errors(tmp_path):
    server, _, man = _make_server(tmp_path)
    with pytest.raises(ValueError, match='no manifest entry'):
        pdc.fetch_cache([(1, 1)], dest=str(tmp_path / 'c'), base_url=_url(server), manifest=man)
    man['files'][0]['sha256'] = ''
    with pytest.raises(ValueError, match='no sha256'):
        pdc.fetch_cache('core', dest=str(tmp_path / 'c'), base_url=_url(server),
                        manifest=man, verbose=False)


def test_shipped_cells_are_skipped(tmp_path):
    man = {'files': [{'name': 'prediagrams_v2_k2_l2.pkl', 'k': 2, 'ell': 2, 'count': 283,
                      'size': 1, 'sha256': 'x', 'core': True}]}
    assert pdc.fetch_cache('core', dest=str(tmp_path), manifest=man,
                           verbose=False) == {(2, 2): 'shipped'}


def test_load_prediagrams_never_fetches(tmp_path, monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise AssertionError('network access attempted')
    monkeypatch.setattr(urllib.request, 'urlopen', boom)
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'auto')
    recs, src = pdc.load_prediagrams(str(tmp_path / 'c'), 2, 1)
    assert src == 'shipped' and len(recs) == 9


def test_the_real_fetch_manifest_is_wellformed():
    man = pdc.load_fetch_manifest()
    assert man['convention'] == pdc.CONVENTION
    cells = {(e['k'], e['ell']) for e in man['files']}
    assert cells.isdisjoint(pdc.SHIPPED_CELLS)             # only cells NOT in git
    for e in man['files']:
        assert len(e['sha256']) == 64 and e['count'] > 0 and e['size'] > 5 * 1024 * 1024
        assert 'url' in e and e['url'] == ''                # placeholder, base URL decides

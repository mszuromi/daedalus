"""``dd.build_cache``: the cost report that precedes a build, and the build."""
import os

import pytest

pytest.importorskip("sage.all")
from daedalus.engine.enumeration import prediagram_cache as pdc            # noqa: E402


def _tree(path):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs)


def test_dry_run_reports_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / 'cache'
    rep = pdc.build_cache([(2, 2), (4, 3)], dest=str(dest), dry_run=True,
                          calibrate=False)
    out = capsys.readouterr().out
    assert not dest.exists() and _tree(tmp_path) == []
    assert rep[(2, 2)]['status'] == 'shipped'
    assert rep[(2, 2)]['est_seconds'] is None
    assert 'nothing to build' in out
    r = rep[(4, 3)]                     # not shipped; its count is in the fetch manifest
    assert r['status'] == 'missing' and r['n_classes_how'] == 'known'
    assert r['n_classes'] == 7_257_435
    assert r['est_seconds'] > 60
    assert r['est_load_bytes'] > 20 * r['est_build_bytes']      # kB per class >> bytes per class
    assert '(4,3) missing' in out and '7,257,435 classes' in out


def test_dry_run_calibrates_in_memory_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rep = pdc.build_cache((2, 5), dry_run=True, verbose=False)
    assert rep[(2, 5)]['status'] == 'missing'
    assert rep[(2, 5)]['n_classes_how'] == 'extrapolated'
    assert _tree(tmp_path) == []


def test_local_cells_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(pdc, 'shipped_exists', lambda k, ell: False)
    pdc.save_v2_certs(str(tmp_path), 2, 2, pdc.load_shipped_certs(2, 2))
    rep = pdc.build_cache([(2, 2)], dest=str(tmp_path), dry_run=True,
                          calibrate=False, verbose=False)
    assert rep[(2, 2)]['status'] == 'local'


@pytest.mark.parametrize('k,ell,tol', [(2, 3, 1.15), (3, 3, 1.15), (4, 3, 1.15),
                                       (5, 2, 1.2), (6, 2, 1.15)])
def test_class_count_extrapolation_against_the_shipped_count(k, ell, tol):
    counts = {c: n for c, n in pdc.known_class_counts().items() if c != (k, ell)}
    true = pdc.known_class_counts()[(k, ell)]
    n, how = pdc.estimate_class_count(k, ell, counts)
    assert how == 'extrapolated'
    assert true / tol <= n <= true * tol, (n, true)


def test_estimate_count_needs_two_cells():
    assert pdc.estimate_class_count(2, 0, {}) == (None, None)
    assert pdc.estimate_class_count(2, 3, {(2, 2): 283}) == (None, None)
    assert pdc.estimate_class_count(2, 3, {(2, 1): 9, (2, 2): 283})[1] == 'extrapolated'


def test_time_estimate_is_within_a_factor_3_of_a_measured_cell(tmp_path):
    dest = tmp_path / 'cache'
    rep = pdc.build_cache([(2, 3)], dest=str(dest), force=True, verbose=False)
    r = rep[(2, 3)]
    assert r['action'] == 'built' and r['n_built'] == r['n_classes'] == 14928
    assert r['est_seconds'] / 3 <= r['seconds'] <= r['est_seconds'] * 3, \
        (r['est_seconds'], r['seconds'])
    assert os.path.isfile(pdc.v2_path(str(dest), 2, 3))
    assert len(pdc.load_v2_certs(str(dest), 2, 3)) == 14928     # stamp + count checked


def test_missing_cell_is_built_into_dest_and_then_found(tmp_path, monkeypatch):
    monkeypatch.setattr(pdc, 'shipped_exists', lambda k, ell: False)
    monkeypatch.setattr(pdc, 'load_shipped_certs',
                        lambda k, ell: (_ for _ in ()).throw(AssertionError('shipped read')))
    dest = str(tmp_path / 'cache')
    rep = pdc.build_cache([(2, 2)], dest=dest, calibrate=False, verbose=False)
    assert rep[(2, 2)]['action'] == 'built' and rep[(2, 2)]['n_built'] == 283
    assert pdc.build_cache([(2, 2)], dest=dest, calibrate=False,
                           verbose=False)[(2, 2)]['status'] == 'local'
    recs, src = pdc.load_prediagrams(dest, 2, 2)
    assert (len(recs), src) == (283, 'v2')


def test_build_refused_when_the_estimate_exceeds_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(pdc, 'shipped_exists', lambda k, ell: False)
    monkeypatch.setattr(pdc, '_physical_memory', lambda: 1_000_000)
    dest = tmp_path / 'cache'
    with pytest.raises(ValueError, match='force=True'):
        pdc.build_cache([(2, 3)], dest=str(dest), calibrate=False, verbose=False)
    assert not dest.exists()


def test_exposed_as_dd_build_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import daedalus as dd
    rep = dd.build_cache([(2, 2)], dry_run=True, calibrate=False, verbose=False)
    assert rep[(2, 2)]['status'] == 'shipped' and _tree(tmp_path) == []

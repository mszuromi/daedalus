"""Memory guards of audit wave W8: a clean refusal before the allocation,
never a swap storm.

MEM-01: ``load_prediagrams`` rebuilds ~11 kB of Sage graphs per class; the
shipped (2,4) / (5,2) cells need 12.9 / 10.3 GB as records and were loaded
under any budget (12 GB in 35 s under a 4 GB budget).
"""
import pytest

pytest.importorskip('sage.all')

from engine.enumeration import prediagram_cache as pc
from engine.integration.time_domain import final_integral as fi


def _no_records(*a, **kw):
    raise AssertionError('records were built before the refusal')


@pytest.mark.parametrize('use_cache', [True, False])
def test_prediagram_load_refused_before_records(tmp_path, monkeypatch,
                                                use_cache):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '0.5')
    monkeypatch.setattr(pc, 'records_from_certs', _no_records)
    with pytest.raises(fi.PhaseJResourceError) as ei:
        pc.load_prediagrams(str(tmp_path), 4, 2, use_cache=use_cache)
    msg = str(ei.value)
    assert '65,956 classes' in msg and 'PHASE_J_MEM_BUDGET_GB' in msg
    assert ei.value.estimate['classes'] == 65956
    assert ei.value.estimate['bytes'] == 65956 * pc._LOAD_BYTES_PER_CLASS
    assert not list(tmp_path.iterdir())          # nothing read or written


def test_prediagram_load_within_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '0.5')
    recs, src = pc.load_prediagrams(str(tmp_path), 3, 2, use_cache=True)
    assert len(recs) == 4496 and src == 'shipped'
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')
    pc._load_guard(6, 2, None)                   # no limit: no refusal


def test_prediagram_guard_counts_and_estimates(monkeypatch):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1')
    # known cells: exact; (2,4) = 1,152,032 classes = 12.9 GB
    with pytest.raises(fi.PhaseJResourceError, match=r'1,152,032 classes'):
        pc._load_guard(2, 4, None)
    pc._load_guard(4, 2, None)                   # 0.74 GB fits 1 GB
    # an extrapolated count is refused only beyond twice the budget
    monkeypatch.setattr(pc, 'estimate_class_count',
                        lambda k, ell, counts=None: (150_000, 'extrapolated'))
    pc._load_guard(9, 9, None)                   # 1.68 GB <= 2 x 1 GB
    monkeypatch.setattr(pc, 'estimate_class_count',
                        lambda k, ell, counts=None: (200_000, 'extrapolated'))
    with pytest.raises(fi.PhaseJResourceError, match='estimated'):
        pc._load_guard(9, 9, None)               # 2.24 GB > 2 x 1 GB
    # the counted check after the certs are in hand is exact
    with pytest.raises(fi.PhaseJResourceError, match='counted'):
        pc._load_guard(9, 9, 100_000)

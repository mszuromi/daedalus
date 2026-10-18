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


# ── MEM-11 / MEM-02: the real-growth guard ─────────────────────────────────

def test_growth_guard_refuses_growth_over_budget(monkeypatch):
    fp = [1.0e9]
    monkeypatch.setattr(fi, 'process_footprint_bytes', lambda: fp[0])
    with fi.run_growth_scope(what='test run', budget=0.5e9, rule='set'):
        fi.growth_guard('start')                 # no growth
        fp[0] = 1.4e9
        fi.growth_guard('a bit later')           # 0.4 GB <= 0.5 GB
        fp[0] = 1.6e9
        with pytest.raises(fi.PhaseJResourceError) as ei:
            fi.growth_guard('diagram #7 of 9')
        assert ei.value.estimate['growth_bytes'] == pytest.approx(0.6e9)
        assert 'diagram #7 of 9' in str(ei.value)
        assert 'test run' in str(ei.value)
    fi.growth_guard('outside any scope')         # no scope: no check


def test_growth_guard_nested_scopes_share_the_baseline(monkeypatch):
    fp = [2.0e9]
    monkeypatch.setattr(fi, 'process_footprint_bytes', lambda: fp[0])
    with fi.run_growth_scope(budget=1e9, rule='set'):
        fp[0] = 2.6e9
        with fi.run_growth_scope(budget=1e9, rule='set'):   # inner: no reset
            fp[0] = 3.2e9
            with pytest.raises(fi.PhaseJResourceError):
                fi.growth_guard('inner')
    with fi.run_growth_scope(budget=None, rule='off'):     # no limit
        fp[0] = 100e9
        fi.growth_guard('no limit')


def test_growth_guard_stops_compute_cumulants_in_typing(monkeypatch):
    """A run whose footprint grows past the budget is stopped with a
    refusal, here at the first typing checkpoint (fake footprint)."""
    import numpy as np
    from api.compute import compute_cumulants
    from api.model import TemporalTheoryBuilder
    model = (TemporalTheoryBuilder('growth guard ou').population('pop', size=1)
             .physical_field('x', population='pop')
             .parameter('eps', default=0.1)
             .set_action_text('sum(xt[i]*(Dt*x[i] + x[i] + eps*x[i]^3) '
                              '- xt[i]^2/2 for i in pop)')
             .set_mf_equation('xstar', '0').build())
    calls = [0]

    def fake_fp():
        calls[0] += 1
        return 1e9 if calls[0] == 1 else 1e12   # the baseline, then +999 GB
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1')   # no default rule read
    monkeypatch.setattr(fi, 'process_footprint_bytes', fake_fp)
    with pytest.raises(fi.PhaseJResourceError, match='grown by'):
        compute_cumulants(model, k=2, max_ell=1,
                          external_fields=[('dx', 1), ('dx', 1)],
                          tau_grid=np.array([0.0]), parallel=False,
                          verbose=False, use_cache=False)


# ── item 57 Q4: forked workers share the remaining budget ────────────────

def test_forked_workers_get_a_share_of_the_budget():
    from engine.integration.time_domain import pipeline as pl
    assert pl._worker_share(4) == {}             # no scope: no initializer
    with fi.phase_j_memory_scope(budget=1000, rule='set', use_env=False):
        with fi._PLAN_MEM_LOCK:
            fi._PLAN_MEM['used'] = 200
        kw = pl._worker_share(4)
        assert kw['initargs'] == (200,)          # (1000 - 200) / 4
        kw['initializer'](*kw['initargs'])       # what each child runs
        assert fi._PLAN_MEM['budget'] == 400     # its own 200 + share

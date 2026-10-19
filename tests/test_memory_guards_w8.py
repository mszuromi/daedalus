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
    # the machine sentence of the Phase J pre-pass refusal
    ram = fi._ram_bytes()
    if ram:
        assert f'This machine has {ram / 1e9:.1f} GB of RAM' in msg
    assert f"the estimate is {ei.value.estimate['bytes'] / 1e9:.2f} GB" in msg
    assert ei.value.estimate['budget_rule'] == 'PHASE_J_MEM_BUDGET_GB=0.5'
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


# ── MEM-13: the point-process simulator's output ───────────────────────────

def test_pp_simulate_refuses_an_oversized_output():
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))
    from test_point_process_front_end import pp_spike_only
    s = pp_spike_only(1)
    t0 = time.perf_counter()
    with pytest.raises(MemoryError, match=r'8\.00 GB'):
        s.simulate(T=1e4, dt=1e-3, n_paths=100, burn_in=0.0, max_bytes=4e9)
    assert time.perf_counter() - t0 < 5          # refused, not simulated
    out = s.simulate(T=2.0, dt=1e-3, n_paths=3, burn_in=0.0, seed=1,
                     max_bytes=1e6)               # 2000 x 3 x 8 B = 48 kB
    assert sum(v.nbytes for v in out['counts'].values()) == 2000 * 3 * 8
    with pytest.raises(MemoryError):
        s.simulate(T=2.0, dt=1e-3, n_paths=3, burn_in=0.0, max_bytes=1e4)


# ── MEM-05: the pre-pass's cheap branch and stateless delta lines ─────────

def test_prepass_cheap_branch_counts_only_subsets_built():
    from engine.integration.time_domain import statespace as SS
    # two stateless delta lines (n = 0: always contracted) and two with states
    args = (4, 0b1001, ((1, 2, 0, True), (2, 3, 0, True), (1, 3, 1, True),
                        (0, 1, 1, True)), 8)
    SS._EST_CACHE.clear()
    cheap = SS._layout_estimate(*args, n_tau=200, n_perm=2)
    old = SS.PREPASS_CHEAP_BYTES
    try:
        SS.PREPASS_CHEAP_BYTES = 0
        SS._EST_CACHE.clear()
        enum = SS._layout_estimate(*args, n_tau=200, n_perm=2)
    finally:
        SS.PREPASS_CHEAP_BYTES = old
        SS._EST_CACHE.clear()
    # still an upper bound of the enumerated estimate, no longer 4x it
    assert enum[0] <= cheap[0] <= 1.01 * enum[0], (cheap, enum)


# ── MEM-06: an edge whose residues are all exactly zero ───────────────────

def test_all_zero_residue_edge_builds_no_plan(monkeypatch):
    from engine.integration.time_domain.final_integral import EdgeModeSum
    lams = [complex(-(j + 1), 0.3 * j) for j in range(6)]
    zero = EdgeModeSum(0, 0, 0j, tuple((0j, lam) for lam in lams))
    full = EdgeModeSum(0, 0, 0j, tuple((complex(0.5 + j, 0.1), lam)
                                       for j, lam in enumerate(lams)))
    rows = [([1.0], [0.0], 0.0)] * 3
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1e-4')   # 100 kB < 216 tuples
    plan = fi._build_modesum_plan([zero, full, full], rows, 1, 1)
    assert len(plan['pole_tuples']) == 0 and plan['n_tuples_full'] == 0
    # the unpruned plan of the same edges: 216 tuples, every coefficient 0
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')
    ref = fi._build_modesum_plan([zero, full, full], rows, 1, 1, prune=False)
    assert len(ref['pole_tuples']) == 216
    assert all(C == 0 for (C, _lams) in ref['pole_tuples'])


# ── MEM-03: the dense forms cached on matrix-exponential blocks ───────────

def test_me_form_cache_is_capped_and_released(monkeypatch):
    import gc
    import numpy as np
    from api import me_output as me

    class Owner:
        pass
    form = (np.zeros((100, 100), dtype=complex), np.zeros(100, dtype=complex))
    nb = sum(a.nbytes for a in form)                  # 161,600 bytes
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', str(3 * nb / 1e9))
    base = me._FORM_CACHE['bytes']
    owners = [Owner() for _ in range(3)]
    admitted = [me._form_cache_admit(o, form) for o in owners]
    assert admitted == [True, False, False]           # cap = 1.5 forms
    assert me._FORM_CACHE['bytes'] == base + nb
    del owners
    gc.collect()
    assert me._FORM_CACHE['bytes'] == base            # released with the owner
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')
    assert me._form_cache_admit(Owner(), form)        # no limit


# ── MEM-07: the dense map between moving segments of a chain output ───────

def test_chain_dense_map_refused_before_allocation(monkeypatch):
    from engine.integration.time_domain import statespace as SS
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1')
    SS._check_dense_map(2000, 2000)                  # 64 MB: fine
    with pytest.raises(fi.PhaseJResourceError, match='6.40 GB'):
        SS._check_dense_map(20000, 20000)            # was charged 1.9 MB


# ── MEM-09: SideME.triple's dense form; the decomposition cache in bytes ──

def test_dense_triple_and_decomposition_cache_are_bounded(monkeypatch):
    import numpy as np
    from api import me_output as me
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1')
    me._check_dense_bytes(16 * 2000 ** 2, 'small')          # 64 MB
    with pytest.raises(fi.PhaseJResourceError, match='6.40 GB'):
        me._check_dense_bytes(16 * 20000 ** 2, 'the triple')  # MAX_DENSE_TOTAL
    S = np.zeros((4000, 4000), dtype=complex)                # 256 MB each
    me._DECOMP.clear()
    me._DECOMP_BYTES[0] = 0
    for i in range(3):
        me._decomp_put(('k', i), (None, S, S, [], 0.0))     # 512 MB per entry
    assert me._DECOMP_BYTES[0] <= 0.5e9 + 1 and len(me._DECOMP) == 0
    small = np.zeros((10, 10), dtype=complex)
    me._decomp_put(('s', 0), (None, small, small, [], 0.0))
    assert len(me._DECOMP) == 1
    me._DECOMP.clear()
    me._DECOMP_BYTES[0] = 0


# ── RED-11: a compressed shipped cell is copied compressed ────────────────

def test_shipped_xz_cell_copied_compressed(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')
    src = pc.shipped_path(3, 3)
    assert src.endswith('.xz')
    monkeypatch.setattr(pc, 'records_from_certs', lambda certs: list(certs))
    recs, how = pc.load_prediagrams(str(tmp_path), 3, 3, use_cache=True)
    assert how == 'shipped' and len(recs) == 358983
    local = pc.v2_path(str(tmp_path), 3, 3)
    assert local.endswith('.xz')
    assert os.path.getsize(local) == os.path.getsize(src)
    recs2, how2 = pc.load_prediagrams(str(tmp_path), 3, 3, use_cache=True)
    assert how2 == 'v2' and sorted(recs2) == sorted(recs)


# ── MEM-10: simulator buffers and the 2-D equal-time estimators ───────────

def test_simulators_bounded():
    import numpy as np
    from simulations import spatial_field_2d_sim as S2
    from simulations.coupled_rd_1d_sim import simulate_coupled_rd_1d
    rng = np.random.default_rng(3)
    snaps = rng.standard_normal((150, 8, 8))           # > one chunk
    meta = {'L': 4.0, 'N': 8, 'k_max': 3.0}
    F = np.fft.fft2(snaps, axes=(1, 2))
    S_ref = np.mean(np.abs(F) ** 2, axis=0) * (16.0 / 8 ** 4)
    _k, _k2, S = S2.structure_factor_2d(snaps, meta)
    assert np.allclose(S, S_ref, rtol=1e-13, atol=0)
    with pytest.raises(MemoryError, match='max_bytes'):
        simulate_coupled_rd_1d(np.eye(2), [1.0, 1.0], np.eye(2), t_run=1e5,
                               max_bytes=1e8)

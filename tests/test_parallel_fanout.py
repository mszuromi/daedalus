"""Parallel evaluation of a Phase J τ batch (W8p S1): diagram-only
partitioning on threads (every platform) and fork (Linux only).

The contract: ``total_C_batch`` / ``eval_per_diagram_batch`` with
``parallel=True`` return the SAME BYTES as ``parallel=False`` for any worker
count and method; worker threads lose no ``_RUNTIME_COUNTERS`` increment;
``DAEDALUS_PARALLEL=0`` and a switch-on scope keep the batch serial; macOS
never forks.

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_parallel_fanout.py -q
"""
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                            # noqa: E402
import engine.fork_safety as fs                                  # noqa: E402
from api.compute import compute_cumulants                        # noqa: E402
from engine.integration.time_domain import final_integral as FI  # noqa: E402
from engine.integration.time_domain import pipeline as P         # noqa: E402

LINUX_FORK_ONLY = pytest.mark.skipif(
    sys.platform == 'darwin',
    reason='macOS never forks (engine.fork_safety); Linux CI covers fork')

POINTS = [(0.0, t) for t in (-2.31, -0.7, 0.0, 0.43, 1.9, 3.3)]


@pytest.fixture(scope='module')
def td():
    """ou_quartic k=2 to two loops: several diagrams, all on the state-space
    route (the default engine)."""
    model, _ = dd.load_model('ou_quartic')
    res = compute_cumulants(model, k=2, max_ell=2,
                            external_fields=[('dx', 1), ('dx', 1)],
                            parameters={'mu': 1.0, 'eps': 0.02, 'D': 1.0},
                            tau_grid=[0.0, 0.5], parallel=False,
                            use_cache=False, verbose=False)
    tds = [t for _, t in sorted(res['phase_j_by_ell'].items()) if t]
    td = max(tds, key=lambda t: sum(1 for g in t['groups']
                                    if g.get('contribution') is not None))
    assert sum(1 for g in td['groups'] if g.get('contribution')) >= 3
    return td


@pytest.fixture
def every_diagram_threads(monkeypatch):
    """Send every diagram to the thread pool (the real gate keeps these
    small diagrams on the main thread)."""
    monkeypatch.setattr(P, 'THREAD_BLOCK_MIN', 0)
    monkeypatch.delenv('DAEDALUS_PARALLEL', raising=False)


def _bytes(x):
    return np.asarray(x, dtype=complex).tobytes()


@pytest.mark.parametrize('n_workers', [1, 2, 4])
def test_threads_bit_identical_to_serial(td, every_diagram_threads, n_workers):
    serial = td['total_C_batch'](POINTS, parallel=False)
    serial_pd, loops = td['eval_per_diagram_batch'](POINTS, parallel=False)
    before = FI._RUNTIME_COUNTERS.get('parallel_thread_batches', 0)
    got = td['total_C_batch'](POINTS, parallel=True, n_workers=n_workers,
                              start_method='threads')
    got_pd, loops2 = td['eval_per_diagram_batch'](
        POINTS, parallel=True, n_workers=n_workers, start_method='threads')
    assert _bytes(got) == _bytes(serial)
    assert _bytes(got_pd) == _bytes(serial_pd) and loops2 == loops
    ran = FI._RUNTIME_COUNTERS.get('parallel_thread_batches', 0) - before
    assert ran == (2 if n_workers > 1 else 0)     # one worker: serial


def test_thread_workers_lose_no_counter_increment(td, every_diagram_threads):
    """The same batch counts the same integer events serially and on 4
    threads (the overlay merges every worker's increments)."""
    pts = [(0.0, t) for t in (-1.1, 0.27, 2.2)]

    def run(**kw):
        snap = {k: v for k, v in FI._RUNTIME_COUNTERS.items()
                if isinstance(v, int) and not isinstance(v, bool)}
        td['total_C_batch'](pts, **kw)
        return {k: v - snap.get(k, 0) for k, v in FI._RUNTIME_COUNTERS.items()
                if isinstance(v, int) and not isinstance(v, bool)
                and v - snap.get(k, 0) and not k.startswith('parallel_')}
    td['total_C_batch'](pts, parallel=False)          # warm
    serial = run(parallel=False)
    threaded = run(parallel=True, n_workers=4, start_method='threads')
    assert serial and threaded == serial


def test_fan_out_on_plain_callables_sums_in_diagram_order(monkeypatch):
    """Unit level: arbitrary callables, a counter bumped by every call."""
    monkeypatch.setattr(P, 'THREAD_BLOCK_MIN', 0)
    monkeypatch.delenv('DAEDALUS_PARALLEL', raising=False)
    key = 'test_fanout_calls'

    def make(c):
        def f(t0, t1):
            FI._RUNTIME_COUNTERS[key] = FI._RUNTIME_COUNTERS.get(key, 0) + 1
            return c * (1.0 + 1e-17 * t1) + 0.1 * t1
        return f
    fns = [make(c) for c in (1e16, -1e16, 3.0, 1e-3, 7.25)]
    base = FI._RUNTIME_COUNTERS.get(key, 0)
    vals = P._fan_out(fns, POINTS, True, 3, 'threads')
    assert FI._RUNTIME_COUNTERS[key] - base == len(fns) * len(POINTS)
    want = [[complex(f(*p)) for p in POINTS] for f in fns]
    assert vals == want


def test_a_worker_error_reaches_the_caller(monkeypatch):
    monkeypatch.setattr(P, 'THREAD_BLOCK_MIN', 0)

    def ok(*t):
        return 1.0

    def bad(*t):
        raise FloatingPointError('diagram 1 failed')
    with pytest.raises(FloatingPointError, match='diagram 1'):
        P._fan_out([ok, bad, ok], POINTS, True, 2, 'threads')


def test_small_diagrams_stay_on_the_main_thread(td, monkeypatch):
    """The real gate: these diagrams' blocks are far below
    ``THREAD_BLOCK_MIN``, so no pool starts and the batch runs serially."""
    monkeypatch.delenv('DAEDALUS_PARALLEL', raising=False)
    fns = [g['contribution'] for g in td['groups'] if g.get('contribution')]
    assert max(P._diagram_block(f) for f in fns) < P.THREAD_BLOCK_MIN
    assert P._fan_out(fns, POINTS, True, 4, 'threads') is None


def test_kill_switch_and_unknown_method(td, every_diagram_threads,
                                        monkeypatch):
    fns = [g['contribution'] for g in td['groups'] if g.get('contribution')]
    monkeypatch.setenv('DAEDALUS_PARALLEL', '0')
    assert P._fan_out(fns, POINTS, True, 4, 'threads') is None
    monkeypatch.delenv('DAEDALUS_PARALLEL')
    with pytest.raises(ValueError, match='start_method'):
        P._fan_out(fns, POINTS, True, 4, 'spawn')


def test_switch_on_scope_stays_serial(td, every_diagram_threads):
    from engine.integration.time_domain import statespace as SS
    fns = [g['contribution'] for g in td['groups'] if g.get('contribution')]
    with SS.switch_on_scope(-5.0):
        assert P._fan_out(fns, POINTS, True, 4, 'threads') is None


def test_macos_explicit_fork_warns_and_never_forks(td, every_diagram_threads,
                                                    monkeypatch):
    import multiprocessing as mp
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(fs, '_WARNED', set())

    def boom(*a, **k):
        raise AssertionError('forked on macOS')
    monkeypatch.setattr(mp, 'get_context', boom)
    serial = td['total_C_batch'](POINTS, parallel=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        got = td['total_C_batch'](POINTS, parallel=True, n_workers=2,
                                  start_method='fork')
        auto = td['total_C_batch'](POINTS, parallel=True, n_workers=2)
    assert _bytes(got) == _bytes(serial) == _bytes(auto)
    msgs = [str(r.message) for r in rec if 'disabled on macOS' in str(r.message)]
    assert len(msgs) == 1                 # the explicit 'fork' only, once


@LINUX_FORK_ONLY
@pytest.mark.parametrize('n_workers', [2, 3])
def test_fork_bit_identical_and_counters_returned(td, n_workers, monkeypatch):
    monkeypatch.delenv('DAEDALUS_PARALLEL', raising=False)
    fns = [g['contribution'] for g in td['groups'] if g.get('contribution')]
    if len(fns) < 2 * n_workers:
        pytest.skip('too few diagrams for the fork path at this worker count')
    pts = [(0.0, t) for t in (-0.9, 0.31, 1.7)]
    td['total_C_batch'](pts, parallel=False)                    # warm
    snap = FI._RUNTIME_COUNTERS.get('ss_evals', 0)
    serial = td['total_C_batch'](pts, parallel=False)
    n_serial = FI._RUNTIME_COUNTERS.get('ss_evals', 0) - snap
    before = FI._RUNTIME_COUNTERS.get('parallel_fork_batches', 0)
    snap = FI._RUNTIME_COUNTERS.get('ss_evals', 0)
    got = td['total_C_batch'](pts, parallel=True, n_workers=n_workers,
                              start_method='fork')
    assert FI._RUNTIME_COUNTERS['parallel_fork_batches'] == before + 1
    assert _bytes(got) == _bytes(serial)
    # the workers' counters came back with their values
    assert FI._RUNTIME_COUNTERS.get('ss_evals', 0) - snap == n_serial

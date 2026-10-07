"""M4 (P5): the ``_chain_with_intermediate_uppers`` memo is a pure speed-up.

Every value with ``USE_CHAIN_UPPERS_MEMO`` on must be bit-identical
(``np.array_equal``, not a tolerance) to the memo off.  The captured calls in
``tests/fixtures/chain_uppers_captured.npz`` are distinct calls of the
function recorded from quad_exp, spike_reset, ou_quartic_two_dim and
ou_quartic (k=4) runs, with their memo-off values.

Run with::

    PYTHONHASHSEED=0 python -m pytest tests/test_chain_uppers_memo.py -v
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.integration.time_domain.final_integral as FI

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures',
                       'chain_uppers_captured.npz')


def _load_calls():
    z = np.load(FIXTURE)
    calls = []
    for i in range(len(z['lens'])):
        m = int(z['lens'][i])
        alphas = [complex(a) for a in z['alphas'][i, :m]]
        upp = {int(p): float(v) for p, v in zip(z['up_pos'][i], z['up_val'][i])
               if p >= 0}
        ref = None if z['value_is_none'][i] else complex(z['value'][i])
        calls.append((alphas, float(z['L'][i]), upp, float(z['U'][i]), ref))
    return calls


CALLS = _load_calls()


def _same(a, b):
    """Bit-for-bit equality of two results (a complex or ``None``)."""
    if a is None or b is None:
        return a is b
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _run(calls=CALLS):
    return [FI._chain_with_intermediate_uppers(a, L, dict(u), U)
            for a, L, u, U, _ in calls]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Cold tables and zeroed counters in, the same out; flags restored."""
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', True)
    FI._chain_simplex_memo_clear()
    FI._reset_runtime_counters()
    yield
    FI._chain_simplex_memo_clear()
    FI._reset_runtime_counters()


def test_fixture_covers_the_interesting_calls():
    """The fixture is small and has chains of several lengths, calls with
    intermediate uppers, complex alphas and (at least) a stored value."""
    assert os.path.getsize(FIXTURE) < 1_000_000
    assert len(CALLS) >= 50
    assert len({len(a) for a, *_ in CALLS}) >= 2
    assert any(u for _, _, u, _, _ in CALLS)
    assert any(abs(x.imag) > 0 for a, *_ in CALLS for x in a)


def test_memo_on_vs_off_array_equal_on_captured_calls():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
        off = _run()
        assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 0
        assert not FI._chain_uppers_memo
    FI._chain_simplex_memo_clear()
    on_cold = _run()
    on_warm = _run()
    for (a, L, u, U, ref), o, c, w in zip(CALLS, off, on_cold, on_warm):
        assert _same(o, ref), 'memo-off run must reproduce the capture'
        assert _same(o, c) and _same(o, w)
    n = len(CALLS)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] <= n
    assert (FI._RUNTIME_COUNTERS['chain_uppers_memo_hits']
            >= n)                         # the whole second pass hits


def test_a_hit_is_the_stored_value_not_a_recomputation():
    a, L, u, U, _ = next(c for c in CALLS if c[2])
    v1 = FI._chain_with_intermediate_uppers(a, L, dict(u), U)
    misses = FI._RUNTIME_COUNTERS['chain_uppers_memo_misses']
    v2 = FI._chain_with_intermediate_uppers(list(a), L, dict(u), U)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == misses
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 1
    assert v1 is v2 or _same(v1, v2)


def test_dict_order_of_the_uppers_does_not_matter():
    a = [-0.3 + 0.1j, -0.5 + 0.0j, -0.7 - 0.2j, -0.2 + 0.0j]
    fwd = {0: -1.0, 1: -0.5, 2: -0.25}
    rev = {2: -0.25, 1: -0.5, 0: -1.0}
    assert list(fwd) != list(rev)
    v1 = FI._chain_with_intermediate_uppers(a, -5.0, fwd, 0.0)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 1
    v2 = FI._chain_with_intermediate_uppers(a, -5.0, rev, 0.0)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 1   # a hit
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 1
    assert _same(v1, v2)
    assert len(FI._chain_uppers_memo) == 1


def test_different_arguments_do_not_alias():
    a = [-0.3 + 0.1j, -0.5 + 0.0j, -0.7 - 0.2j]
    base = FI._chain_with_intermediate_uppers(a, -5.0, {0: -1.0}, 0.0)
    others = [
        FI._chain_with_intermediate_uppers(a, -5.0, {0: -1.5}, 0.0),
        FI._chain_with_intermediate_uppers(a, -5.0, {1: -1.0}, 0.0),
        FI._chain_with_intermediate_uppers(a, -4.0, {0: -1.0}, 0.0),
        FI._chain_with_intermediate_uppers(a, -5.0, {0: -1.0}, 0.5),
        # complex alphas key exactly: one ulp apart is another key
        FI._chain_with_intermediate_uppers(
            [np.nextafter(a[0].real, 0) + 0.1j] + a[1:], -5.0, {0: -1.0}, 0.0),
    ]
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 0
    assert len(FI._chain_uppers_memo) == 6
    assert not any(_same(base, o) for o in others)


def test_a_none_valued_upper_is_not_dropped_from_the_key():
    """The uncached function treats ``{k: None}`` as absent for the
    effective uppers but as a direct constraint for the cut positions, so
    ``{0: None}`` and ``{}`` may differ and must not share a key."""
    a = [-0.3 + 0.0j, -0.5 + 0.0j, -0.7 + 0.0j]
    k_none = FI._chain_uppers_key_suffix(-5.0, {0: None, 1: -1.0}, 0.0)
    k_abs = FI._chain_uppers_key_suffix(-5.0, {1: -1.0}, 0.0)
    assert k_none != k_abs
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
        ref = FI._chain_with_intermediate_uppers(
            a, -5.0, {0: None, 1: -1.0}, 0.0)
    got = FI._chain_with_intermediate_uppers(
        a, -5.0, {0: None, 1: -1.0}, 0.0)
    assert _same(ref, got)


# ─── flag toggles ──────────────────────────────────────────────────
@pytest.mark.parametrize('flag, value', [
    ('USE_NUMBA_CHAIN_SIMPLEX', False),
    ('USE_CHAIN_SIMPLEX_PRECISION_FIX', True),
    ('_CHAIN_SIMPLEX_CANCEL_THRESHOLD', 0.5),
    ('USE_POSET_MPMATH_ACCUMULATION', True),
    ('USE_POSET_CAP_MATCH_SCIPY', True),
])
def test_a_flag_toggle_misses_the_memo(monkeypatch, flag, value):
    a = [-0.3 + 0.1j, -0.5 + 0.0j, -0.7 - 0.2j]
    args = (a, -5.0, {0: -1.0}, 0.0)
    FI._chain_with_intermediate_uppers(*args)
    FI._chain_with_intermediate_uppers(*args)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 1
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 1
    monkeypatch.setattr(FI, flag, value)
    FI._chain_with_intermediate_uppers(*args)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 2
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 1
    monkeypatch.undo()
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', True)
    FI._chain_with_intermediate_uppers(*args)           # back: a hit again
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 2
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 2


def test_precision_flag_toggle_keys_the_inner_tables_too(monkeypatch):
    """The fast table used to key on the arguments only, so a toggle of the
    precision gate (or numba) was answered from a stale entry.  Each state
    must get the value that state computes, in either order of toggling."""
    al = [0.0 - 0.329j, 0.0 - 0.351j, -0.4 + 0.0j]     # close-pole chain
    args = (al, -5.0, 0.0)
    states = (False, True, False, True)
    for fix in states:
        monkeypatch.setattr(FI, 'USE_CHAIN_SIMPLEX_PRECISION_FIX', fix)
        cached = FI._exp_over_chain_simplex_fast(*args)
        assert _same(cached, FI._exp_over_chain_simplex_fast_uncached(*args))
    assert len(FI._chain_simplex_memo_fast) == 2        # one entry per state


def test_the_flag_off_bypasses_the_table(monkeypatch):
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
    a = [-0.3 + 0.0j, -0.5 + 0.0j]
    FI._chain_with_intermediate_uppers(a, -5.0, {}, 0.0)
    FI._chain_with_intermediate_uppers(a, -5.0, {}, 0.0)
    assert not FI._chain_uppers_memo
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 0
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 0


def test_the_flag_is_read_at_call_time(monkeypatch):
    a = [-0.3 + 0.0j, -0.5 + 0.0j]
    FI._chain_with_intermediate_uppers(a, -5.0, {}, 0.0)
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
    FI._chain_with_intermediate_uppers(a, -5.0, {}, 0.0)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 0
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', True)
    FI._chain_with_intermediate_uppers(a, -5.0, {}, 0.0)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 1


def test_a_bad_flag_value_raises(monkeypatch):
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', 'yes')
    with pytest.raises(ValueError, match='USE_CHAIN_UPPERS_MEMO'):
        FI._chain_with_intermediate_uppers([-0.3 + 0j], -5.0, {}, 0.0)


@pytest.mark.parametrize('text, expected', [
    ('', True), ('1', True), ('on', True), ('TRUE', True),
    ('0', False), ('off', False), ('No', False), ('false', False)])
def test_the_env_variable(text, expected):
    env = {} if text == '' else {'DAEDALUS_CHAIN_UPPERS_MEMO': text}
    assert FI._initial_chain_uppers_memo_flag(env) is expected


def test_the_env_variable_rejects_garbage():
    with pytest.raises(ValueError, match='DAEDALUS_CHAIN_UPPERS_MEMO'):
        FI._initial_chain_uppers_memo_flag(
            {'DAEDALUS_CHAIN_UPPERS_MEMO': 'maybe'})


def test_the_umbrella_does_not_touch_the_memo_flag():
    """A pure speed switch: ``DAEDALUS_PHASE_J_LEGACY`` restores numbers."""
    assert FI._initial_chain_uppers_memo_flag(
        {'DAEDALUS_PHASE_J_LEGACY': '1'}) is True
    assert 'USE_CHAIN_UPPERS_MEMO' not in FI._initial_phase_j_flags({})


def test_an_unhashable_argument_skips_the_memo(monkeypatch):
    """A ``TypeError`` while building/hashing the key sends the call straight
    to the uncached function, without touching the table or the counters.
    (The uncached function is stubbed: the inner per-simplex tables, which
    predate M4, do their own lookup of an unhashable alpha.)"""
    class Odd(complex):
        __hash__ = None
    seen = []
    monkeypatch.setattr(
        FI, '_chain_with_intermediate_uppers_uncached',
        lambda a, L, u, U: seen.append((list(a), L, dict(u), U)) or 'direct')
    a = [Odd(-0.3), -0.5 + 0.0j]
    assert FI._chain_with_intermediate_uppers(a, -5.0, {1: -1.0}, 0.0) == 'direct'
    assert len(seen) == 1 and seen[0][1:] == (-5.0, {1: -1.0}, 0.0)
    assert not FI._chain_uppers_memo
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 0
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 0
    # an unkeyable L / upper value takes the same road
    assert FI._chain_with_intermediate_uppers(
        [-0.3 + 0j], np.array([1.0, 2.0]), {}, 0.0) == 'direct'
    assert FI._chain_with_intermediate_uppers(
        [-0.3 + 0j], -5.0, {0: np.array([1.0, 2.0])}, 0.0) == 'direct'
    assert not FI._chain_uppers_memo


def test_clear_drops_the_new_table_too():
    FI._chain_with_intermediate_uppers([-0.3 + 0j, -0.5 + 0j], -5.0, {}, 0.0)
    assert FI._chain_uppers_memo
    FI._chain_simplex_memo_clear()
    assert not FI._chain_uppers_memo
    assert not FI._chain_simplex_memo_fast
    assert not FI._chain_simplex_memo_poly


def test_counters_are_zeroed_by_reset():
    FI._chain_with_intermediate_uppers([-0.3 + 0j], -5.0, {}, 0.0)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 1
    FI._reset_runtime_counters()
    for k in ('hits', 'misses', 'evictions'):
        assert FI._RUNTIME_COUNTERS[f'chain_uppers_memo_{k}'] == 0


# ─── the size cap ──────────────────────────────────────────────────
def test_behaviour_at_the_cap(monkeypatch):
    """At the cap the table is cleared once (a locked ``clear()``) and the
    new entry goes in: the size never exceeds the cap, results are
    unchanged, and an entry from before the clear is recomputed."""
    monkeypatch.setattr(FI, '_CHAIN_UPPERS_MEMO_MAX', 5)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
        ref = _run(CALLS[:40])
    FI._chain_simplex_memo_clear()
    got = []
    for a, L, u, U, _ in CALLS[:40]:
        got.append(FI._chain_with_intermediate_uppers(a, L, dict(u), U))
        assert len(FI._chain_uppers_memo) <= 5
    assert all(_same(r, g) for r, g in zip(ref, got))
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_evictions'] >= 1
    # the first call was evicted: asking again is a miss, with the same value
    misses = FI._RUNTIME_COUNTERS['chain_uppers_memo_misses']
    a, L, u, U, _ = CALLS[0]
    again = FI._chain_with_intermediate_uppers(a, L, dict(u), U)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == misses + 1
    assert _same(again, ref[0])


def test_cap_of_one_never_grows(monkeypatch):
    monkeypatch.setattr(FI, '_CHAIN_UPPERS_MEMO_MAX', 1)
    for a, L, u, U, _ in CALLS[:10]:
        FI._chain_with_intermediate_uppers(a, L, dict(u), U)
        assert len(FI._chain_uppers_memo) == 1


# ─── threads ───────────────────────────────────────────────────────
def test_threaded_eviction_matches_the_serial_run(monkeypatch):
    """8 threads x the captured arguments with the cap forced to 100, so the
    table is cleared mid-run: no exception, every result array_equal to the
    serial (memo-off) run."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
        serial = _run()
    FI._chain_simplex_memo_clear()
    FI._reset_runtime_counters()
    monkeypatch.setattr(FI, '_CHAIN_UPPERS_MEMO_MAX', 100)
    assert len(CALLS) > 100                 # the cap really binds
    barrier = threading.Barrier(8)

    def work(seed):
        order = np.random.RandomState(seed).permutation(len(CALLS))
        barrier.wait()
        out = {}
        for i in order:
            a, L, u, U, _ = CALLS[i]
            out[int(i)] = FI._chain_with_intermediate_uppers(
                list(a), L, dict(u), U)
            assert len(FI._chain_uppers_memo) <= 100
        return out

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(work, range(8)))      # re-raises any exception
    for out in results:
        assert len(out) == len(CALLS)
        for i, v in out.items():
            assert _same(v, serial[i]), i
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_evictions'] >= 1
    assert len(FI._chain_uppers_memo) <= 100
    c = FI._RUNTIME_COUNTERS
    assert c['chain_uppers_memo_hits'] + c['chain_uppers_memo_misses'] \
        == 8 * len(CALLS)                   # the counters are exact (locked)


def test_threaded_flag_toggle_mid_run_never_returns_a_stale_value():
    """One thread flips the precision gate while others call: whenever the
    flag state did not move during a call, the memoised result equals what
    the uncached function gives (the numba and mpmath states agree to
    rounding only, so a stale entry would show up as a bit difference)."""
    calls = CALLS[:60]
    stop = threading.Event()
    errors = []
    saved = FI.USE_CHAIN_SIMPLEX_PRECISION_FIX

    def flipper():
        v = False
        while not stop.is_set():
            v = not v
            FI.USE_CHAIN_SIMPLEX_PRECISION_FIX = v

    def caller(seed):
        order = np.random.RandomState(seed).permutation(len(calls))
        for i in order:
            a, L, u, U = calls[i][:4]
            tag = FI._chain_uppers_kernel_tag()
            got = FI._chain_with_intermediate_uppers(list(a), L, dict(u), U)
            ref = FI._chain_with_intermediate_uppers_uncached(
                list(a), L, dict(u), U)
            if tag == FI._chain_uppers_kernel_tag() and not _same(got, ref):
                errors.append((int(i), tag))

    t = threading.Thread(target=flipper)
    t.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(caller, range(4)))
    finally:
        stop.set()
        t.join()
        FI.USE_CHAIN_SIMPLEX_PRECISION_FIX = saved
    assert not errors


# ─── the plan-loop key hoist ───────────────────────────────────────
def test_the_hoisted_suffix_is_the_unhoisted_key():
    a, L, u, U, _ = next(c for c in CALLS if len(c[2]) >= 2)
    suffix = FI._chain_uppers_key_suffix(L, u, U)
    v1 = FI._chain_with_intermediate_uppers(a, L, dict(u), U,
                                            _key_suffix=suffix)
    v2 = FI._chain_with_intermediate_uppers(a, L, dict(reversed(list(
        u.items()))), U)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 1
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 1
    assert _same(v1, v2)


# ─── end to end ────────────────────────────────────────────────────
def test_two_tau_end_to_end_array_equal(monkeypatch):
    """ou_quartic_two_dim k=2 ell=2 at two taus: memo on == memo off, bit
    for bit (totals and per-ell), and the memo is actually used."""
    import tests.tools.phase_j_zoo_baseline as Z
    entry = Z._E('t-m4-ou2', model='ou_quartic_two_dim', k=2, max_ell=2,
                 ext=[('x', 1), ('x', 1)], params=None,
                 tau_grid=[0.0, 1.5])
    out = {}
    for mode in (False, True):
        monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', mode)
        FI._chain_simplex_memo_clear()
        rec = Z.run_entry(entry, with_provenance=False)
        assert rec['status'] == 'ok', rec.get('status')
        out[mode] = (Z.result_arrays(rec), rec['counters'])
    off, on = out[False][0], out[True][0]
    assert set(off) == set(on) and 'total' in off
    for k in off:
        assert np.array_equal(off[k], on[k]), k
    assert out[False][1]['chain_uppers_memo_misses'] == 0
    assert out[True][1]['chain_uppers_memo_misses'] > 0
    assert out[True][1]['chain_uppers_memo_hits'] > 0

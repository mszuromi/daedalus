"""M6: the opt-in ``CHAIN_UPPERS_KERNEL`` switch of
``_chain_with_intermediate_uppers`` and the ``USE_PHASE_J_DD_KERNELS`` flag.

``'legacy'`` (default) is today's evaluator, untouched; ``'transfer'`` routes
the function to ``expdd.chain_with_uppers``.  The two agree to rounding (not
bit for bit), so the selector is part of the M4 memo key and a toggle must
miss the memo.

Run with::

    PYTHONHASHSEED=0 python -m pytest tests/test_chain_uppers_kernel.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.integration.time_domain.final_integral as FI
from tests.test_chain_uppers_memo import CALLS, _run, _same


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Cold tables and zeroed counters in, the same out; flags restored."""
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', True)
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'legacy')
    FI._chain_simplex_memo_clear()
    FI._reset_runtime_counters()
    yield
    FI._chain_simplex_memo_clear()
    FI._reset_runtime_counters()


def _uncached(kernel, monkeypatch):
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', kernel)
    return [FI._chain_with_intermediate_uppers_uncached(a, L, dict(u), U)
            for a, L, u, U, _ in CALLS]


# ─── defaults ──────────────────────────────────────────────────────
def test_defaults_are_legacy_and_off():
    assert FI._initial_chain_uppers_kernel({}) == 'legacy'
    assert FI._initial_phase_j_dd_kernels_flag({}) is False


def test_the_legacy_kernel_does_not_import_expdd():
    """With the default the new module is never even loaded."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import engine.integration.time_domain.final_integral as FI\n"
            "from tests.test_chain_uppers_memo import CALLS\n"
            "for a, L, u, U, _ in CALLS[:40]:\n"
            "    FI._chain_with_intermediate_uppers(a, L, dict(u), U)\n"
            "assert FI.CHAIN_UPPERS_KERNEL == 'legacy'\n"
            "assert 'engine.integration.time_domain.expdd' not in sys.modules\n"
            % os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    env = {k: v for k, v in os.environ.items()
           if k != 'DAEDALUS_CHAIN_UPPERS_KERNEL'}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, env=dict(env, PYTHONHASHSEED='0'),
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]


# ─── the transfer kernel ───────────────────────────────────────────
def test_transfer_vs_legacy_on_the_captured_calls(monkeypatch):
    """The 320 captured calls: <= 1e-11 relative (M6 criterion C)."""
    leg = _uncached('legacy', monkeypatch)
    tra = _uncached('transfer', monkeypatch)
    worst = max(abs(t - l) / abs(l) for t, l in zip(tra, leg))
    assert worst <= 1e-11, worst


def test_legacy_matches_the_stored_values(monkeypatch):
    """The legacy evaluator still gives the M4 fixture's memo-off values
    (stored on another platform, so to rounding, not bit for bit)."""
    for got, (*_, ref) in zip(_uncached('legacy', monkeypatch), CALLS):
        assert abs(got - ref) <= 1e-13 * abs(ref)


def test_transfer_is_expdd(monkeypatch):
    from engine.integration.time_domain import expdd
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    for a, L, u, U, _ in CALLS[::10]:
        assert _same(FI._chain_with_intermediate_uppers_uncached(a, L, u, U),
                     expdd.chain_with_uppers(a, L, u, U))
    assert FI._chain_with_intermediate_uppers_uncached([], -5.0, {}, 0.0) == 1
    assert FI._chain_with_intermediate_uppers_uncached(
        [-0.3 + 0j, -0.5 + 0j], -5.0, {0: -5.0}, 0.0) == 0


def test_transfer_overflow_is_none(monkeypatch):
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    assert FI._chain_with_intermediate_uppers_uncached(
        [10.0 + 0j] * 3, 0.0, {0: 30.0}, 40.0) is None


def test_transfer_divergent_semi_infinite_is_none(monkeypatch):
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    assert FI._chain_with_intermediate_uppers_uncached(
        [1.0 + 0j, -2.0 + 0j], -np.inf, {0: -1.0}, 0.0) is None


def test_transfer_underflow_is_none(monkeypatch):
    """expdd's explicit FloatingPointError (state underflow) is the
    function's "no finite value" signal, ``None``."""
    from engine.integration.time_domain import expdd

    def underflow(*args):
        raise FloatingPointError('expdd: the chain state underflowed')
    monkeypatch.setattr(expdd, 'chain_with_uppers', underflow)
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    assert FI._chain_with_intermediate_uppers_uncached(
        [-0.3 + 0j, -0.5 + 0j], -5.0, {0: -1.0}, 0.0) is None


# ─── the memo ──────────────────────────────────────────────────────
def test_toggling_the_kernel_misses_the_memo(monkeypatch):
    a = [-0.3 + 0.1j, -0.5 + 0.0j, -0.7 - 0.2j]
    args = (a, -5.0, {0: -1.0}, 0.0)
    v_leg = FI._chain_with_intermediate_uppers(*args)
    FI._chain_with_intermediate_uppers(*args)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 1
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 1
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    v_tra = FI._chain_with_intermediate_uppers(*args)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 2
    assert _same(v_tra, FI._chain_with_intermediate_uppers_uncached(*args))
    assert abs(v_tra - v_leg) <= 1e-13 * abs(v_leg)
    FI._chain_with_intermediate_uppers(*args)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 2
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'legacy')
    assert _same(FI._chain_with_intermediate_uppers(*args), v_leg)
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_misses'] == 2
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == 3


def test_the_kernel_is_in_the_tag(monkeypatch):
    t0 = FI._chain_uppers_kernel_tag()
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    assert FI._chain_uppers_kernel_tag() != t0
    assert 'transfer' in FI._chain_uppers_kernel_tag()


def test_memo_on_vs_off_with_the_transfer_kernel(monkeypatch):
    """The memo stays a pure speed-up under the transfer kernel too."""
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
    off = _run()
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', True)
    on = _run()
    again = _run()
    assert all(_same(a, b) for a, b in zip(off, on))
    assert all(_same(a, b) for a, b in zip(on, again))
    assert FI._RUNTIME_COUNTERS['chain_uppers_memo_hits'] == len(CALLS)


def test_legacy_results_survive_a_round_trip_through_transfer(monkeypatch):
    """legacy -> transfer -> legacy in one process: the legacy values are
    bit-identical before and after (memo on)."""
    first = _run()
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    _run()
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'legacy')
    FI._chain_simplex_memo_clear()
    assert all(_same(a, b) for a, b in zip(first, _run()))


# ─── validation and the environment ────────────────────────────────
def test_the_kernel_is_read_at_call_time(monkeypatch):
    a = [-0.3 + 0.1j, -0.5 + 0.0j]
    leg = FI._chain_with_intermediate_uppers_uncached(a, -5.0, {}, 0.0)
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'transfer')
    tra = FI._chain_with_intermediate_uppers_uncached(a, -5.0, {}, 0.0)
    from engine.integration.time_domain import expdd
    assert _same(tra, expdd.chain_with_uppers(a, -5.0, {}, 0.0))
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', 'legacy')
    assert _same(FI._chain_with_intermediate_uppers_uncached(a, -5.0, {}, 0.0),
                 leg)


@pytest.mark.parametrize('bad', ['Transfer', 'expdd', '', None, 1, ['legacy']])
def test_a_bad_kernel_value_raises(monkeypatch, bad):
    monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', bad)
    with pytest.raises(ValueError, match='CHAIN_UPPERS_KERNEL'):
        FI._chain_with_intermediate_uppers([-0.3 + 0j], -5.0, {}, 0.0)
    assert not FI._chain_uppers_memo


@pytest.mark.parametrize('text, expected', [
    ('', 'legacy'), ('legacy', 'legacy'), (' Legacy ', 'legacy'),
    ('transfer', 'transfer'), ('TRANSFER', 'transfer')])
def test_the_kernel_env_variable(text, expected):
    env = {} if text == '' else {'DAEDALUS_CHAIN_UPPERS_KERNEL': text}
    assert FI._initial_chain_uppers_kernel(env) == expected


def test_the_kernel_env_variable_rejects_garbage():
    with pytest.raises(ValueError, match='DAEDALUS_CHAIN_UPPERS_KERNEL'):
        FI._initial_chain_uppers_kernel(
            {'DAEDALUS_CHAIN_UPPERS_KERNEL': 'expm'})


@pytest.mark.parametrize('text, expected', [
    ('', False), ('0', False), ('off', False), ('No', False),
    ('1', True), ('on', True), ('TRUE', True), ('yes', True)])
def test_the_dd_flag_env_variable(text, expected):
    env = {} if text == '' else {'DAEDALUS_PHASE_J_DD_KERNELS': text}
    assert FI._initial_phase_j_dd_kernels_flag(env) is expected


def test_the_dd_flag_validation(monkeypatch):
    with pytest.raises(ValueError, match='DAEDALUS_PHASE_J_DD_KERNELS'):
        FI._initial_phase_j_dd_kernels_flag(
            {'DAEDALUS_PHASE_J_DD_KERNELS': 'maybe'})
    assert FI._phase_j_dd_kernels_on() is False
    monkeypatch.setattr(FI, 'USE_PHASE_J_DD_KERNELS', True)
    assert FI._phase_j_dd_kernels_on() is True
    monkeypatch.setattr(FI, 'USE_PHASE_J_DD_KERNELS', 'yes')
    with pytest.raises(ValueError, match='USE_PHASE_J_DD_KERNELS'):
        FI._phase_j_dd_kernels_on()


def test_the_umbrella_touches_neither():
    assert FI._initial_chain_uppers_kernel(
        {'DAEDALUS_PHASE_J_LEGACY': '1'}) == 'legacy'
    assert FI._initial_phase_j_dd_kernels_flag(
        {'DAEDALUS_PHASE_J_LEGACY': '1'}) is False
    flags = FI._initial_phase_j_flags({})
    assert 'CHAIN_UPPERS_KERNEL' not in flags
    assert 'USE_PHASE_J_DD_KERNELS' not in flags


# ─── end to end ────────────────────────────────────────────────────
def test_two_tau_end_to_end_transfer_vs_legacy(monkeypatch):
    """ou_quartic k=2 ell=2 at two taus, memo on: legacy before and after a
    transfer run is bit-identical; transfer agrees to 1e-11 and actually
    runs (the chain-uppers memo misses under it)."""
    import tests.tools.phase_j_zoo_baseline as Z
    entry = Z._E('t-m6-ou', model='ou_quartic', k=2, max_ell=2,
                 ext=[('dx', 1), ('dx', 1)], params=Z.P_OU,
                 tau_grid=[0.0, 1.5])
    out = []
    for kernel in ('legacy', 'transfer', 'legacy'):
        monkeypatch.setattr(FI, 'CHAIN_UPPERS_KERNEL', kernel)
        rec = Z.run_entry(entry, with_provenance=False)
        assert rec['status'] == 'ok', rec.get('status')
        out.append((Z.result_arrays(rec), rec['counters']))
    (leg, _), (tra, ctr), (leg2, _) = out
    assert ctr['chain_uppers_memo_misses'] > 0
    assert set(leg) == set(tra) and 'total' in leg
    for k in leg:
        assert np.array_equal(leg[k], leg2[k]), k
        scale = np.max(np.abs(leg[k]))
        assert np.max(np.abs(tra[k] - leg[k])) <= 1e-11 * scale, k

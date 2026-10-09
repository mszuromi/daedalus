"""M10 follow-up: the δ-edge elimination of ``integrate_grouped_diagram``
(``grouped_integral._grouped_delta_solve``) by the same exact rational linear
algebra as the per-diagram path (``USE_SETUP_DELTA_SOLVE``), with the same
validation mode (``VALIDATE_DELTA``).

The lever must not move any number: lever off / on / validate give
``np.array_equal`` results, the substitutions are the same polynomials and SR
trees as ``sage_solve``'s, and the M2a token replay
(``_delta_solve_leaves_residual``) still agrees with the solve it replays.

Run with::

    PYTHONHASHSEED=0 python -m pytest tests/test_grouped_delta_solve.py -v
"""
from __future__ import annotations

import contextlib
import io
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import api.compute as AC
import daedalus as dd
import engine.integration.time_domain.final_integral as FI
import engine.integration.time_domain.grouped_integral as GI
from sage.all import SR, pi, sin, cos

from tests.test_phase_j_delta_solve import _random_system, _U, _T, _P
from tests.test_phase_j_structural_zeros import (
    CYC, FD_TREE, _grouped, _prop, _td)

_P_SPIKE = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
            'w': [[0.55, 0.65], [0.7, 0.8]]}


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    """Lever on, validation off, umbrella / env switches unset."""
    for env in ('DAEDALUS_PHASE_J_LEGACY_SETUP',
                'DAEDALUS_PHASE_J_VALIDATE_DELTA'):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', True)
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', False)
    FI._reset_runtime_counters()
    yield
    FI._reset_runtime_counters()


def _dts(edge_info):
    return [e['dt_sym'] for e in edge_info]


# ── randomized systems: exact grouped solve == sage_solve ─────────────────
@pytest.mark.parametrize('seed', range(8))
def test_random_systems_match_sage_solve_exactly(seed):
    rng = random.Random(2000 + seed)
    n_checked = 0
    for _ in range(25):
        edge_info, delta, _smooth, ivars = _random_system(rng)
        dts = _dts(edge_info)
        fast = GI._grouped_delta_solve_exact(dts, delta, ivars)
        legacy = GI._grouped_delta_solve_legacy(dts, delta, ivars)
        # raises on any difference (substitutions as polynomials and as SR
        # trees, residuals, their zero verdicts)
        GI._validate_grouped_delta(fast, legacy, f'seed {seed}')
        if fast is None:
            continue
        for k, v in fast[0].items():
            assert v.is_trivially_equal(legacy[0][k]), (k, v, legacy[0][k])
        assert (GI._has_nontrivial_equality_exact(fast[2])
                == GI._has_nontrivial_equality_legacy(legacy[2]))
        n_checked += 1
    assert n_checked > 0
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fallback'] == 0
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fast'] > 0
    assert FI._RUNTIME_COUNTERS['setup_delta_validated'] > 0


def test_no_chain_resolution_like_the_legacy_grouped_solve():
    """The grouped solve applies earlier solutions to each new equation but
    does not rewrite earlier substitutions; the exact one does the same."""
    u1, u2, u3 = _U[:3]
    dts = [u1 - u2, u2 - u3 - _T[0]]
    fast = GI._grouped_delta_solve_exact(dts, [0, 1], [u1, u2, u3])
    legacy = GI._grouped_delta_solve_legacy(dts, [0, 1], [u1, u2, u3])
    GI._validate_grouped_delta(fast, legacy, 'chain')
    assert fast[0][u1].is_trivially_equal(u2)       # not rewritten in u3


# ── exact zeros and residual equalities ────────────────────────────────────
def test_residual_verdicts_are_exact():
    t1, t2 = _T
    u1 = _U[0]
    dts = [u1 - t1, t1 - t2, u1 - t1]
    fast = GI._grouped_delta_solve_exact(dts, [0, 1, 2], [u1])
    legacy = GI._grouped_delta_solve_legacy(dts, [0, 1, 2], [u1])
    GI._validate_grouped_delta(fast, legacy, 'residual')
    assert len(fast[2]) == 2
    assert FI._residual_is_zero_exact(fast[2][0]) is False
    assert FI._residual_is_zero_exact(fast[2][1]) is True
    assert GI._has_nontrivial_equality_exact(fast[2]) is True
    assert GI._has_nontrivial_equality_exact(fast[2][1:]) is False
    assert GI._has_nontrivial_equality_exact([SR(0)]) is False
    assert GI._has_nontrivial_equality_exact([]) is False


def test_nonlinear_residual_keeps_the_sr_zero_test():
    t1 = _T[0]
    zero = sin(t1) ** 2 + cos(t1) ** 2 - 1          # not a linear form
    assert FI._residual_is_zero_exact(zero) is None
    assert GI._has_nontrivial_equality_exact([zero]) \
        == GI._has_nontrivial_equality_legacy([zero]) is False
    assert GI._has_nontrivial_equality_exact([sin(t1)]) is True


# ── fallbacks ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize('eq,why', [
    (1.5 * _U[0] - _T[0], 'float coefficient'),
    (_P * _U[0] - _T[0], 'symbolic coefficient'),
    (_U[0] ** 2 - _T[0], 'nonlinear'),
    (_U[0] * _U[1] - _T[0], 'product of variables'),
    (_U[0] - pi, 'non-rational constant'),
])
def test_fallback_goes_to_sage_solve_unchanged(eq, why):
    dts = [eq, _U[0] - _T[1]]
    ivars = [_U[0], _U[1]]
    fast = GI._grouped_delta_solve_exact(dts, [0], ivars)
    legacy = GI._grouped_delta_solve_legacy(dts, [0], ivars)
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fallback'] == 1, why
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fast'] == 0, why
    assert (fast is None) == (legacy is None)
    for k in fast[0]:
        assert fast[0][k].is_trivially_equal(legacy[0][k])


def test_zero_coefficient_falls_back_and_stays_equal():
    # u1 cancels after the first substitution: the second equation then has
    # a zero coefficient on u1 and no variable left to solve for
    u1, u2 = _U[:2]
    dts = [u1 - _T[0], u1 - _T[0] + u2]
    fast = GI._grouped_delta_solve_exact(dts, [0, 1], [u1, u2])
    legacy = GI._grouped_delta_solve_legacy(dts, [0, 1], [u1, u2])
    GI._validate_grouped_delta(fast, legacy, 'zero coefficient')


# ── flag, umbrella, validate mode ──────────────────────────────────────────
def _sys():
    u1, u2 = _U[:2]
    return [u1 - _T[0], u2 - u1 - _T[1], _T[0] - _T[1]], [0, 1, 2], [u1, u2]


def test_flag_off_uses_the_legacy_solve(monkeypatch):
    dts, delta, ivars = _sys()
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', False)
    GI._grouped_delta_solve(dts, delta, ivars)
    c = FI._RUNTIME_COUNTERS
    assert c['setup_delta_solve_fast'] == c['setup_delta_solve_fallback'] == 0
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', True)
    GI._grouped_delta_solve(dts, delta, ivars)
    assert c['setup_delta_solve_fast'] == 2


def test_umbrella_forces_the_lever_off(monkeypatch):
    dts, delta, ivars = _sys()
    monkeypatch.setenv('DAEDALUS_PHASE_J_LEGACY_SETUP', '1')
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '1')
    GI._grouped_delta_solve(dts, delta, ivars)
    GI._has_nontrivial_equality([_T[0] - _T[1]])
    c = FI._RUNTIME_COUNTERS
    assert c['setup_delta_solve_fast'] == c['setup_delta_solve_fallback'] == 0
    assert c['setup_delta_validated'] == 0


def test_flag_is_read_at_call_time_and_validated(monkeypatch):
    dts, delta, ivars = _sys()
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', 'yes')
    with pytest.raises(ValueError):
        GI._grouped_delta_solve(dts, delta, ivars)
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', True)
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', 'yes')
    with pytest.raises(ValueError):
        GI._grouped_delta_solve(dts, delta, ivars)


def test_validate_mode_counts_and_agrees(monkeypatch):
    dts, delta, ivars = _sys()
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '1')
    off = GI._grouped_delta_solve_legacy(dts, delta, ivars)
    on = GI._grouped_delta_solve(dts, delta, ivars)
    assert FI._RUNTIME_COUNTERS['setup_delta_validated'] == 2
    assert [str(v) for v in on[1]] == [str(v) for v in off[1]]


@pytest.mark.parametrize('how', ['shift', 'scale'])
def test_validate_mode_catches_a_wrong_fast_result(monkeypatch, how):
    """Positive control: a deliberately wrong exact solution is caught."""
    orig = FI._delta_eliminate_exact

    def wrong(eq, var):
        r = orig(eq, var)
        if r is None:
            return None
        return r + SR(1) / 3 if how == 'shift' else r * 2
    monkeypatch.setattr(FI, '_delta_eliminate_exact', wrong)
    dts, delta, ivars = _sys()
    GI._grouped_delta_solve(dts, delta, ivars)         # validation off: silent
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '1')
    with pytest.raises(FI.DeltaSolveValidationError,
                       match=r'grouped δ-solve mismatch \(subset δ edges'):
        GI._grouped_delta_solve(dts, delta, ivars,
                                where='subset δ edges [0, 1, 2]')


def test_validate_mode_catches_structural_disagreements():
    u1, u2 = _U[:2]
    dts = [u1 - _T[0], u2 - u1]
    legacy = GI._grouped_delta_solve_legacy(dts, [0], [u1, u2])
    subs, rem, ext = GI._grouped_delta_solve_exact(dts, [0], [u1, u2])
    with pytest.raises(FI.DeltaSolveValidationError, match='remaining'):
        GI._validate_grouped_delta((subs, [], ext), legacy, 'x')
    with pytest.raises(FI.DeltaSolveValidationError, match='feasibility'):
        GI._validate_grouped_delta(None, legacy, 'x')
    with pytest.raises(FI.DeltaSolveValidationError, match='residual'):
        GI._validate_grouped_delta((subs, rem, [_T[0] - _T[1]]),
                                   (legacy[0], legacy[1], [_T[0] - _T[1] + 1]),
                                   'x')
    other = (_T[0] + 1) ** 2 - _T[0] ** 2 - _T[0] - 1   # same poly, other tree
    with pytest.raises(FI.DeltaSolveValidationError, match='SR tree'):
        GI._validate_grouped_delta(({u1: other}, rem, ext), legacy, 'x')


# ── a real grouped run ─────────────────────────────────────────────────────
def _spike_grouped(*, on, validate=False, record=None):
    FI.USE_SETUP_DELTA_SOLVE = on
    FI.VALIDATE_DELTA = validate
    FI._reset_runtime_counters()
    model = dd.load_model('single_population_spike_reset_test')[0]
    with contextlib.redirect_stdout(io.StringIO()):
        res = AC.compute_cumulants(
            model, k=1, max_ell=1, external_fields=[('n', 1)],
            parameters=_P_SPIKE, tau_grid=np.array([0.0]), use_cache=True,
            parallel=False, verbose=False, use_grouped_phase_j=True)
    vals = [np.asarray(res['C_tau'], complex)]
    vals += [np.asarray(a, complex)
             for _, a in sorted(res['C_tau_by_ell'].items())]
    return np.concatenate(vals), dict(FI._RUNTIME_COUNTERS)


def test_grouped_run_on_off_validate_array_equal(monkeypatch):
    # warm the disk cache with one discarded run, then toggle in one process
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', True)
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', False)
    _spike_grouped(on=False)
    off, c_off = _spike_grouped(on=False)
    assert c_off['setup_delta_solve_fast'] == 0
    assert c_off['setup_delta_solve_fallback'] == 0
    on, c_on = _spike_grouped(on=True)
    assert c_on['setup_delta_solve_fast'] > 0
    assert c_on['setup_delta_solve_fallback'] == 0
    assert c_on['setup_delta_validated'] == 0
    val, c_val = _spike_grouped(on=True, validate=True)
    assert c_val['setup_delta_validated'] == c_val['setup_delta_solve_fast']
    assert np.array_equal(off, on) and np.array_equal(off, val)


# ── the M2a token replay still agrees with the solve it replays ───────────
def _replay_vs_solve(monkeypatch, tds, cps, pd, replays):
    """Run a grouped build (``STRUCTURAL_ZEROS`` on) recording, for every
    subset the M2a replay is consulted on, its verdict and the group's
    equations (read from the caller's frame)."""
    orig = GI._delta_solve_leaves_residual

    def spy(delta_edges, *tokens):
        f = sys._getframe(1).f_locals
        verdict = orig(delta_edges, *tokens)
        replays.append((list(delta_edges),
                        [ei['dt_sym'] for ei in f['ref_ei']],
                        list(f['integration_vars_grouped']), verdict))
        return verdict
    monkeypatch.setattr(GI, '_delta_solve_leaves_residual', spy)
    monkeypatch.setattr(FI, 'STRUCTURAL_ZEROS', True)
    return _grouped(tds, cps, pd)


def test_replay_verdict_equals_the_solve_verdict(monkeypatch):
    pd, _ = _prop()
    a = _td(FD_TREE, (3, 4), (0, 1, 2))
    b = _td(FD_TREE, (3, 4), (0, 1, 2), prediagram=a.prediagram)
    monkeypatch.setattr(FI, 'USE_POLYGON_M2_INTEGRATOR', False)
    c1 = _td(CYC, (3, 4), (0, 1, 2))
    c2 = _td(CYC, (3, 4), (0, 1, 2), prediagram=c1.prediagram)
    replays = []
    for tds in ([a, b], [c1, c2]):
        _replay_vs_solve(monkeypatch, tds, [1.0, 0.5], pd, replays)
    assert len(replays) >= 2, 'the M2a replay was never consulted'
    for delta_edges, dts, ivars, verdict in replays:
        legacy = GI._grouped_delta_solve_legacy(dts, delta_edges, ivars)
        fast = GI._grouped_delta_solve_exact(dts, delta_edges, ivars)
        assert legacy is not None and fast is not None
        assert verdict == GI._has_nontrivial_equality_legacy(legacy[2]), \
            delta_edges
        assert verdict == GI._has_nontrivial_equality_exact(fast[2]), \
            delta_edges


@pytest.mark.parametrize('tree', ['FD_TREE', 'CYC'])
def test_hand_built_groups_on_off_validate_bit_identical(monkeypatch, tree):
    pd, _ = _prop()
    monkeypatch.setattr(FI, 'USE_POLYGON_M2_INTEGRATOR', False)
    spec = {'FD_TREE': FD_TREE, 'CYC': CYC}[tree]
    a = _td(spec, (3, 4), (0, 1, 2))
    b = _td(spec, (3, 4), (0, 1, 2), prediagram=a.prediagram)
    out = {}
    for tag, on, val in (('off', False, False), ('on', True, False),
                         ('val', True, True)):
        monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', on)
        monkeypatch.setattr(FI, 'VALIDATE_DELTA', val)
        out[tag] = _grouped([a, b], [1.0, 0.5], pd)
    assert out['off'][2]['setup_delta_solve_fast'] == 0
    assert out['on'][2]['setup_delta_solve_fast'] > 0
    assert out['val'][2]['setup_delta_validated'] \
        == out['val'][2]['setup_delta_solve_fast']
    for tag in ('on', 'val'):
        assert np.array_equal(np.asarray(out['off'][1]),
                              np.asarray(out[tag][1]))
        assert out['off'][3] == out[tag][3]         # same subset statuses


@pytest.mark.parametrize('seed', range(4))
def test_replay_equals_solve_on_random_token_graphs(seed):
    """Both verdicts: random δ-edge sets over integration, external-time and
    origin vertices; the token replay equals the solve's residual verdict."""
    rng = random.Random(3000 + seed)
    toks = {0: ('zero',), 1: ('ext', 't1'), 2: ('ext', 't2'),
            3: ('int', 3), 4: ('int', 4), 5: ('int', 5)}
    sym = {0: SR(0), 1: _T[0], 2: _T[1], 3: _U[0], 4: _U[1], 5: _U[2]}
    int_order = [toks[3], toks[4], toks[5]]
    seen = set()
    for _ in range(60):
        ends = [tuple(rng.sample(range(6), 2))
                for _ in range(rng.randint(1, 5))]
        delta = list(range(len(ends)))
        dts = [sym[v] - sym[u] for u, v in ends]
        verdict = GI._delta_solve_leaves_residual(delta, ends, toks,
                                                  int_order)
        solved = GI._grouped_delta_solve_legacy(
            dts, delta, [_U[0], _U[1], _U[2]])
        fast = GI._grouped_delta_solve_exact(
            dts, delta, [_U[0], _U[1], _U[2]])
        GI._validate_grouped_delta(fast, solved, f'seed {seed}')
        assert verdict == GI._has_nontrivial_equality_legacy(solved[2])
        assert verdict == GI._has_nontrivial_equality_exact(fast[2])
        seen.add(verdict)
    assert seen == {True, False}

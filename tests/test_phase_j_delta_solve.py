"""M10 (L4a): the δ-edge elimination of ``integrate_diagram`` by exact
rational linear algebra (``USE_SETUP_DELTA_SOLVE``) instead of Maxima
(``sage_solve``) and SR ``==`` tests.

The lever must not move any number: the substitutions are the same
polynomials (and the same SR trees), every constraint row is ``==`` equal,
exact zeros stay exactly ``0.0``, and every result is ``np.array_equal`` to
the lever off.  ``VALIDATE_DELTA`` / ``DAEDALUS_PHASE_J_VALIDATE_DELTA=1``
runs both solves and raises ``DeltaSolveValidationError`` on any difference.

Run with::

    PYTHONHASHSEED=0 python -m pytest tests/test_phase_j_delta_solve.py -v
"""
from __future__ import annotations

import contextlib
import io
import os
import random
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import api.compute as AC
import daedalus as dd
import engine.integration.time_domain.final_integral as FI
from sage.all import QQ, SR, pi, solve as sage_solve

from tests.test_phase_j_setup_fastpath import _delta_values, _pair_tree

_U = [SR.var(f'm10_u{i}') for i in range(1, 5)]      # integration variables
_T = [SR.var(f'm10_t{i}') for i in range(1, 3)]      # external times
_P = SR.var('m10_p')                                 # a model parameter


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    """Lever on, validation off, both umbrellas / env switches unset."""
    for env in ('DAEDALUS_PHASE_J_LEGACY_SETUP',
                'DAEDALUS_PHASE_J_VALIDATE_DELTA'):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', True)
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', False)
    FI._reset_runtime_counters()
    yield
    FI._reset_runtime_counters()


# ── helpers ────────────────────────────────────────────────────────────────
def _rand_form(rng, variables, lo=-5, hi=5):
    """A random linear form with integer coefficients in [lo, hi] (and a
    rational constant)."""
    expr = SR(QQ(rng.randint(lo, hi)) / rng.choice((1, 2, 3)))
    for v in variables:
        expr = expr + rng.randint(lo, hi) * v
    return expr


def _fraction_form(expr):
    """{name: Fraction} + Fraction constant of a rational linear SR form
    (independent of the engine helper: via ``.coefficient``)."""
    names = {str(v): v for v in expr.variables()}
    coef = {n: Fraction(str(expr.coefficient(v, 1))) for n, v in names.items()}
    const = Fraction(str(expr.subs({v: 0 for v in names.values()})))
    return coef, const


def _fraction_rows(dt_syms, delta_edges, smooth_edges, int_vars, ext_vars):
    """Reference elimination in ``fractions.Fraction`` (same order as the
    engine: first remaining integration variable present, by name), then
    the ``(a_int, a_ext, c0)`` rows of the smooth edges as floats."""
    forms = [_fraction_form(SR(e)) for e in dt_syms]
    subs = {}                       # name -> (coef dict, const)
    remaining = [str(v) for v in int_vars]

    def apply(form):
        coef, const = dict(form[0]), form[1]
        for n in list(coef):
            if n in subs:
                c = coef.pop(n)
                s_coef, s_const = subs[n]
                for m, q in s_coef.items():
                    coef[m] = coef.get(m, 0) + c * q
                const += c * s_const
        return {n: q for n, q in coef.items() if q != 0}, const

    for i in delta_edges:
        coef, const = apply(forms[i])
        x = next((n for n in remaining if n in coef), None)
        if x is None:
            continue
        a = coef.pop(x)
        rhs = ({n: -q / a for n, q in coef.items()}, -const / a)
        for k in list(subs):                     # chain-resolve
            k_coef, k_const = subs[k]
            if x in k_coef:
                c = k_coef.pop(x)
                for m, q in rhs[0].items():
                    k_coef[m] = k_coef.get(m, 0) + c * q
                subs[k] = ({m: q for m, q in k_coef.items() if q != 0},
                           k_const + c * rhs[1])
        subs[x] = rhs
        remaining.remove(x)
    rows = []
    for i in smooth_edges:
        coef, const = apply(forms[i])
        rows.append(([float(coef.get(n, 0)) for n in remaining],
                     [float(coef.get(str(s), 0)) for s in ext_vars],
                     float(const)))
    return rows


def _random_system(rng):
    """2-4 integration variables, 1-3 δ equations, a few smooth rows (some
    built to cancel EXACTLY after the elimination)."""
    n_int = rng.randint(2, 4)
    ivars = _U[:n_int]
    n_eq = rng.randint(1, 3)
    dts = []
    for _ in range(n_eq):
        vs = rng.sample(ivars + _T, rng.randint(2, n_int + 2))
        dts.append(_rand_form(rng, vs))
    smooth = []
    for _ in range(rng.randint(1, 3)):
        smooth.append(_rand_form(rng, rng.sample(ivars + _T, 2)))
    # exact cancellations: a rational multiple of a δ equation (its int-var
    # part vanishes after the elimination, so a_int must be exactly 0.0 and
    # the whole row exactly zero), and the same plus pure external terms
    q = QQ(rng.choice((-3, -1, 1, 2, 5))) / rng.choice((1, 3, 7))
    j = rng.randrange(n_eq)
    smooth.append(q * dts[j])
    smooth.append(q * dts[j] + rng.randint(1, 5) * _T[0] - _T[1]
                  + SR(1) / 3)
    edge_info = [{'dt_sym': e} for e in dts + smooth]
    delta = list(range(n_eq))
    smooth_idx = list(range(n_eq, n_eq + len(smooth)))
    return edge_info, delta, smooth_idx, ivars


# ── randomized systems: exact solve == sage_solve ─────────────────────────
@pytest.mark.parametrize('seed', range(8))
def test_random_systems_match_sage_solve_exactly(seed):
    rng = random.Random(1000 + seed)
    n_checked = n_zero_rows = 0
    for _ in range(25):
        edge_info, delta, smooth, ivars = _random_system(rng)
        fast = FI._delta_solve_exact(edge_info, delta, ivars)
        legacy = FI._delta_solve_legacy(edge_info, delta, ivars)
        # raises on any difference (substitution, residuals, rows)
        FI._validate_delta_subset(fast, legacy, edge_info, smooth, _T,
                                  f'seed {seed}')
        if fast is None:
            continue
        subs, rem, _ = fast
        for k, v in subs.items():
            assert v.is_trivially_equal(legacy[0][k]), (k, v, legacy[0][k])
        rows = FI._delta_constraint_rows(edge_info, smooth, subs, rem, _T)
        ref = _fraction_rows([e['dt_sym'] for e in edge_info], delta, smooth,
                             ivars, _T)
        assert rows == ref
        # the exact-cancellation rows: every int-var coefficient is the
        # float 0.0 (never a 1e-17 residue); when every δ equation was used
        # for an elimination (no residual) the pure multiple is all zeros
        pure, mixed = rows[-2], rows[-1]
        assert all(type(x) is float and x == 0.0 for x in pure[0] + mixed[0])
        if not fast[2]:
            assert pure == ([0.0] * len(rem), [0.0, 0.0], 0.0)
            n_zero_rows += 1
        n_checked += 1
    assert n_checked > 0 and n_zero_rows > 0
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fallback'] == 0
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fast'] > 0


def test_chain_of_substitutions_is_resolved_like_sage():
    u1, u2, u3 = _U[:3]
    t1, t2 = _T
    # u1 = u2 first, then u2 = (t1 + u3)/3 (u1 must chain), then u3 = t2 - 1/2
    dts = [u1 - u2, 3 * u2 - t1 - u3, 2 * u3 - 2 * t2 + 1,
           u1 - u3 + t1, u1 - t1 / 3 - u3 / 3]
    ei = [{'dt_sym': e} for e in dts]
    fast = FI._delta_solve_exact(ei, [0, 1, 2], [u1, u2, u3])
    legacy = FI._delta_solve_legacy(ei, [0, 1, 2], [u1, u2, u3])
    FI._validate_delta_subset(fast, legacy, ei, [3, 4], _T, 'chain')
    subs = fast[0]
    assert fast[1] == []
    assert subs[u1].is_trivially_equal(subs[u2])
    assert not {str(v) for k in subs for v in subs[k].variables()} & \
        {'m10_u1', 'm10_u2', 'm10_u3'}
    rows = FI._delta_constraint_rows(ei, [3, 4], subs, [], _T)
    # u1 - t1/3 - u3/3 with u1 = (t1 + u3)/3: exactly zero
    assert rows[1] == ([], [0.0, 0.0], 0.0)


def test_residual_equalities_and_their_verdicts():
    u1 = _U[0]
    t1, t2 = _T
    # the second δ edge has no integration variable left: a residual
    # external-time equality (nonzero), the third one cancels exactly
    ei = [{'dt_sym': u1 - t1}, {'dt_sym': t1 - t2}, {'dt_sym': u1 - t1}]
    fast = FI._delta_solve_exact(ei, [0, 1, 2], [u1])
    legacy = FI._delta_solve_legacy(ei, [0, 1, 2], [u1])
    FI._validate_delta_subset(fast, legacy, ei, [], _T, 'residual')
    ext = fast[2]
    assert len(ext) == 2
    assert FI._residual_is_zero_exact(ext[0]) is False
    assert FI._residual_is_zero_exact(ext[1]) is True
    assert FI._residual_is_zero_exact(1.5 * t1) is None    # keeps SR test


# ── fallbacks ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize('eq,why', [
    (1.5 * _U[0] - _T[0], 'float coefficient'),
    (_P * _U[0] - _T[0], 'symbolic coefficient'),
    (_U[0] ** 2 - _T[0], 'nonlinear'),
    (_U[0] * _U[1] - _T[0], 'product of variables'),
    (_U[0] - pi, 'non-rational constant'),
    (_U[0] - 0.5 * _T[0], 'float coefficient elsewhere'),
])
def test_fallback_goes_to_sage_solve_unchanged(eq, why):
    assert FI._delta_eliminate_exact(eq, _U[0]) is None, why
    ei = [{'dt_sym': eq}, {'dt_sym': _U[0] - _T[1]}]
    fast = FI._delta_solve_exact(ei, [0], [_U[0], _U[1]])
    legacy = FI._delta_solve_legacy(ei, [0], [_U[0], _U[1]])
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fallback'] == 1
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fast'] == 0
    assert (fast is None) == (legacy is None)
    for k in fast[0]:
        assert fast[0][k].is_trivially_equal(legacy[0][k])


def test_a_repeated_variable_is_removed_once_like_list_remove():
    t1, t2 = _U[:2]
    ei = [{'dt_sym': t1 - _T[0]}, {'dt_sym': t1 - t2}]
    fast = FI._delta_solve_exact(ei, [0], [t1, t1, t2])
    legacy = FI._delta_solve_legacy(ei, [0], [t1, t1, t2])
    assert [str(v) for v in fast[1]] == [str(v) for v in legacy[1]]
    FI._validate_delta_subset(fast, legacy, ei, [1], _T, 'repeat')


def test_validate_mode_requires_the_same_sr_tree():
    u1 = _U[0]
    ei = [{'dt_sym': u1 - _T[0]}]
    legacy = FI._delta_solve_legacy(ei, [0], [u1])
    # an equal polynomial held unexpanded (another SR tree)
    other = (_T[0] + 1) ** 2 - _T[0] ** 2 - _T[0] - 1
    assert (other - _T[0]).expand().is_trivial_zero()
    assert not other.is_trivially_equal(_T[0])
    with pytest.raises(FI.DeltaSolveValidationError, match='SR tree'):
        FI._validate_delta_subset(({u1: other}, [], []), legacy, ei, [],
                                  _T, 'tree')


def test_zero_coefficient_falls_back():
    # the variable does not occur (zero coefficient): no exact solution
    assert FI._delta_eliminate_exact(_T[0] + 2, _U[0]) is None
    assert FI._delta_eliminate_exact(_U[1] - _U[1] + _T[0], _U[0]) is None


def test_linear_form_parser():
    lf = FI._exact_linear_form(3 * (_U[0] - _U[1] / 2) + SR(7) / 3)
    terms, const = lf
    assert const == QQ(7) / 3
    assert {n: c for n, (_, c) in terms.items()} == {'m10_u1': 3,
                                                     'm10_u2': QQ(-3) / 2}
    assert FI._exact_linear_form(SR(0)) == ({}, 0)
    for bad in (SR(1.5) * _U[0], SR(2.0), _U[0] * _U[1], _U[0] ** 2,
                SR(1j) * _U[0], pi * _U[0]):
        assert FI._exact_linear_form(bad) is None, bad
    # the exact solution is the same SR tree as sage_solve's
    eq = 3 * _U[0] - 2 * _U[1] + _T[0] / 5 - SR(4) / 7
    mine = FI._delta_eliminate_exact(eq, _U[0])
    ref = sage_solve(eq == 0, _U[0], solution_dict=True)[0][_U[0]]
    assert mine.is_trivially_equal(ref)


# ── integrate_diagram: flag, umbrella, validate mode ───────────────────────
_PTS = [(0.0, 0.7, 1.3), (0.0, -0.4, 0.9), (0.0, 0.0, 0.0)]


def _star_values():
    return _delta_values(FI.integrate_diagram(**_pair_tree()), _PTS)


def test_star_tree_on_off_validate_array_equal(monkeypatch):
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', False)
    off = _star_values()
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fast'] == 0
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', True)
    on = _star_values()
    n_fast = FI._RUNTIME_COUNTERS['setup_delta_solve_fast']
    assert n_fast > 0
    assert FI._RUNTIME_COUNTERS['setup_delta_solve_fallback'] == 0
    assert FI._RUNTIME_COUNTERS['setup_delta_validated'] == 0
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '1')
    val = _star_values()
    assert FI._RUNTIME_COUNTERS['setup_delta_validated'] == n_fast
    assert np.array_equal(off, on) and np.array_equal(off, val)


def test_umbrella_forces_the_lever_off(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PHASE_J_LEGACY_SETUP', '1')
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '1')
    _star_values()
    c = FI._RUNTIME_COUNTERS
    assert c['setup_delta_solve_fast'] == c['setup_delta_solve_fallback'] == 0
    assert c['setup_delta_validated'] == 0
    assert FI._setup_lever_on('USE_SETUP_DELTA_SOLVE') is False
    monkeypatch.delenv('DAEDALUS_PHASE_J_LEGACY_SETUP')
    assert FI._setup_lever_on('USE_SETUP_DELTA_SOLVE') is True


def test_flags_are_validated_and_read_at_call_time(monkeypatch):
    assert FI._validate_delta_on() is False
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', True)
    assert FI._validate_delta_on() is True
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', False)
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '1')
    assert FI._validate_delta_on() is True
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '0')
    assert FI._validate_delta_on() is False
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', 'yes')
    with pytest.raises(ValueError):
        FI._validate_delta_on()
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', False)
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', 'yes')
    with pytest.raises(ValueError):
        FI._setup_lever_on('USE_SETUP_DELTA_SOLVE')
    assert FI._initial_setup_flag('DAEDALUS_SETUP_DELTA_SOLVE', {}) is True
    assert FI._initial_setup_flag('DAEDALUS_SETUP_DELTA_SOLVE',
                                  {'DAEDALUS_SETUP_DELTA_SOLVE': '0'}) is False


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
    _star_values()                                  # validation off: silent
    monkeypatch.setenv('DAEDALUS_PHASE_J_VALIDATE_DELTA', '1')
    with pytest.raises(FI.DeltaSolveValidationError,
                       match=r'diagram serial .* subset .* δ edges'):
        _star_values()


def test_validate_mode_catches_wrong_rows_with_equal_substitutions():
    """Rows are compared too: equal substitutions but a wrong remaining
    variable list are caught."""
    u1, u2 = _U[:2]
    ei = [{'dt_sym': u1 - _T[0]}, {'dt_sym': u2 - u1}]
    legacy = FI._delta_solve_legacy(ei, [0], [u1, u2])
    subs, rem, ext = FI._delta_solve_exact(ei, [0], [u1, u2])
    with pytest.raises(FI.DeltaSolveValidationError, match='remaining'):
        FI._validate_delta_subset((subs, [], ext), legacy, ei, [1], _T, 'x')
    with pytest.raises(FI.DeltaSolveValidationError, match='feasibility'):
        FI._validate_delta_subset(None, legacy, ei, [1], _T, 'x')


# ── a real model: the fast path fires on spike_reset ℓ=1 ──────────────────
_P_SPIKE = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
            'w': [[0.55, 0.65], [0.7, 0.8]]}


def _spike_reset_k1(monkeypatch, *, on, validate=False):
    monkeypatch.setattr(FI, 'USE_SETUP_DELTA_SOLVE', on)
    monkeypatch.setattr(FI, 'VALIDATE_DELTA', validate)
    FI._reset_runtime_counters()
    model = dd.load_model('single_population_spike_reset_test')[0]
    with contextlib.redirect_stdout(io.StringIO()):
        res = AC.compute_cumulants(
            model, k=1, max_ell=1, external_fields=[('n', 1)],
            parameters=_P_SPIKE, tau_grid=np.array([0.0]), use_cache=True,
            parallel=False, verbose=False)
    vals = [np.asarray(res['C_tau'], complex)]
    vals += [np.asarray(a, complex)
             for _, a in sorted(res['C_tau_by_ell'].items())]
    vals = np.concatenate(vals)
    return vals, dict(FI._RUNTIME_COUNTERS)


def test_fast_path_fires_on_spike_reset_ell1(monkeypatch):
    off, c_off = _spike_reset_k1(monkeypatch, on=False)
    assert c_off['setup_delta_solve_fast'] == 0
    on, c_on = _spike_reset_k1(monkeypatch, on=True, validate=True)
    assert c_on['setup_delta_solve_fast'] > 0
    assert c_on['setup_delta_solve_fallback'] == 0
    assert c_on['setup_delta_validated'] == c_on['setup_delta_solve_fast']
    assert np.array_equal(off, on)

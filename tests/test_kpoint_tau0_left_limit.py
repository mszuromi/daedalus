"""``dd.run`` k>=3 slices: every point is the left limit of its own curve.

``daedalus.run`` turns a temporal k>=3 cumulant into k-1 slices: slice j
sweeps leg j over ``tau_grid``, and the other non-anchor legs (the pinned
legs) sit at ``Config.kpoint_base_lags`` (default 0, evaluated at the Itô
left limit 0- = -_ITO_EPS).  Before this change every leg with |t| <= 1e-12
went to -_ITO_EPS, so at tau = 0 the swept leg and the pinned legs at base 0
landed on the SAME time.  The engine's leg-order tie rule then decided the
value, and for legs of different fields of a model with instantaneous (delta)
propagator parts the point was the limit of the plotted curve from neither
side (single_population_spike_reset_test, <n1 n2 n1> tree, slice 1:
tau = 0 point +4.96369e-2, left limit -3.32555e-1, right limit -2.34321e-1).
Now ``daedalus._kpoint_slice_times`` places the swept leg one more _ITO_EPS
below every leg it coincides with (the anchor, or a pinned leg's base lag or
evaluation time), so the point is the left limit of its slice, like the k=2
grid's tau = 0 point (sampled at tau = -_ITO_EPS).

What is checked
---------------
1. Model-free: the times ``_kpoint_slice_times`` produces (default base lags
   at k = 3 and 4, crossings of nonzero base lags, the anchor with nonzero
   base lags, the repeated step, the 1e-12 coincidence tolerance, pinned
   legs that stay tied, the next-float step where _ITO_EPS is below the
   float spacing of the time, |t| >= 2**34), and that every point that
   coincides with no other leg gets exactly the times the pre-change code
   used.
2. k = 2 never goes through the slice code, and ``dd.run``'s k = 2 output is
   bit-identical, in process, to the pre-change ``daedalus.py`` (git
   revision ``_PRE_CHANGE_REV``; skipped without git or that revision).
3. Public models with delta parts: the tau = 0 point of the k = 3 slices 1
   and 2 and of the k = 4 slices equals the left limit of the SAME slice,
   i.e. the Lagrange extrapolation of the slice from tau = -1e-4, -1e-5,
   -3e-6 (swept leg below every other leg) to the time at which the point
   is evaluated, i.e. the point lies on the slice's left branch.  (The left
   limit at tau = 0 itself differs from it by the curve's change over that
   O(2 _ITO_EPS) offset; CHANGELOG 0.2.0.)  The right limit is extrapolated
   from +1e-4, +1e-5, +3e-6; where it differs from the left limit (a
   genuine jump) the check can tell the two sides apart.  With nonzero
   ``kpoint_base_lags`` the point where the swept leg crosses a pinned
   leg's lag is the left limit too.
4. OU (no delta parts, no constant rows): the cumulant is continuous at the
   tau = 0 point, and with the default base lags it is flat to first order
   there, so the point moves by O(_ITO_EPS**2) only.  Measured:
   ``ou_quartic`` k = 4 (mu = 1, eps = 0.02, D = 1) 2.50e-12 relative at
   tree level and 1.07e-12 at one loop; ``ou_quartic`` k = 3 is identically
   0 (symmetric saddle), before and after.  (A crossing of a nonzero base
   lag moves by O(_ITO_EPS): <= 5.0e-7 per loop order, 6.5e-7 for the
   total, for base lags [0.5, 0.5, 0].)
5. (slow) An exact tie (the full grid's diagonals, the coincident
   non-swept legs of k >= 4 slices) is the tie-order limit only as
   accurately as it is evaluated: multipopulation_test at (0, 0.4, 0.4)
   sends 16 regions to the scipy quadrature fallback and was 2.9e-5
   relative off with its default tolerance (a strict expected failure
   until the fallback was hardened, M2b; now the limit to 6.3e-15).  (The
   other route, the poset lower-bound inheritance error at a tie, made the
   default k = 4 slices of single_population_spike_reset_test 1.0-8.5 %
   off at tau = 0.5 until that error was fixed (M3, CHANGELOG 0.2.0); not
   re-measured since.  Not tested here: the build plus one such point
   takes 4 to 5 minutes, its limit about 15 more.)
6. Moment outputs (``Config.output = 'central_moment'``): the cumulant
   blocks of 3 or more legs are evaluated at the times of slice 1 with the
   default base lags.  The k = 3 central moment equals slice 1
   bit-for-bit, including its tau = 0 left limit (linear delta spikes
   <n1 n2 n1>; before, the moment's tau = 0 point was the tie-order value
   at (0, 0, 0), 3.0 % off); the k = 4 central moment of ``ou_quartic``
   minus its pair terms equals slice 1 to rounding (the old block times
   (0, tau, 0, 0) are 6.1e-7 relative off at tau = +-0.5).

Tolerances.  Left limit vs the point: 1e-10 relative for the analytic
models (measured <= 1.3e-13), 1e-9 for single_population_spike_reset_test
(its k >= 3 tree level is partly evaluated by scipy.nquad; measured
<= 5.6e-13).  A genuine jump: the two one-sided limits differ by > 1e-2
relative.

Runtime (measured): the default part builds linear delta spikes k = 2 and
3 (tree), and ``ou_quartic`` k = 2 and 4, in a private cwd: 17 s of test
time, 22 s wall, cold (29-38 s wall on a heavily loaded machine); the two
moment tests of item 6 add about 3 s.  The slow
part (spike reset and multipopulation k = 3, linear delta spikes k = 4, the
grouped Phase J path) took ~7.5 min under heavy load, 6.4 min of it
multipopulation_test; the tie test of item 5 adds 35 s when it follows
the multipopulation slice test (it reuses that build) and ~6 min alone.

Run:  sage -python -m pytest tests/test_kpoint_tau0_left_limit.py -q
      sage -python -m pytest tests/test_kpoint_tau0_left_limit.py -q -m slow
"""
import math
import os
import random
import subprocess
import sys
import types

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

import api                                                  # noqa: E402
import daedalus as dd                                       # noqa: E402
from api.compute import _ITO_EPS                            # noqa: E402

EPS = _ITO_EPS

#: The last revision whose ``daedalus.py`` used the old k>=3 nudge (every
#: non-anchor leg with |t| <= 1e-12 at -1e-6), for the k = 2 bit-identity
#: check.
_PRE_CHANGE_REV = '8aad964aa2f503882b578fe2b6701c6fd17ac945'

# Sample offsets of the one-sided limits (all > 2 _ITO_EPS away, so the
# swept leg stays below / above every leg at the tie).
_XS_LEFT = (-1e-4, -1e-5, -3e-6)
_XS_RIGHT = (1e-4, 1e-5, 3e-6)

_RTOL_ANALYTIC = 1e-10
_RTOL_NQUAD = 1e-9
_JUMP = 1e-2

# Parameters of the CHANGELOG 0.2.0 tables (tests/tools/phase_j_zoo_baseline.py)
_P_SPIKE = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
            'w': [[0.55, 0.65], [0.7, 0.8]]}
_P_LD = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0], 'w': [[0.0, 0.25], [0.2, 0.0]]}
_P_MP = {'tauE': [10.0, 9.5], 'tauI': [8.0, 7.0], 'EmE': [0.7, 0.72],
         'EmI': [0.4, 0.42], 'aE': [0.37, 0.41], 'aI': [0.23, 0.28],
         'wEE': [[0.25, 0.22], [0.21, 0.19]],
         'wEI': [[0.12, 0.15], [0.13, 0.10]],
         'wIE': [[0.19, 0.17], [0.15, 0.12]],
         'wII': [[0.12, 0.13], [0.14, 0.15]],
         'taugEE': [[4.0, 4.0], [3.0, 3.0]], 'taugEI': [[2.0, 1.0], [1.0, 3.0]],
         'taugIE': [[5.0, 6.0], [2.0, 1.0]],
         'taugII': [[1.5, 1.2], [1.1, 1.0]]}
_P_OU = {'mu': 1.0, 'eps': 0.02, 'D': 1.0}

# model name -> (parameters, (field, index) of the two alternating legs)
_MODELS = {
    'ld': ('single_population_linear_delta_spikes_test', _P_LD,
           (('n', 1), ('n', 2))),
    'spike': ('single_population_spike_reset_test', _P_SPIKE,
              (('n', 1), ('n', 2))),
    'mp': ('multipopulation_test', _P_MP, (('nE', 1), ('nE', 2))),
}


def _ext(which, k):
    """<a b a ...>: legs alternate between the model's two fields."""
    a, b = _MODELS[which][2]
    return [a if i % 2 == 0 else b for i in range(k)]


def _old_times(base, j, tau):
    """The pre-change mapping: every non-anchor leg with |t| <= 1e-12 at
    -1e-6, every other leg as given (the swept leg at tau)."""
    v = [float(b) for b in base]
    v[j - 1] = float(tau)
    return [0.0] + [t if abs(t) > 1e-12 else -1e-6 for t in v]


def _lagrange(xs, ys, x):
    out = 0.0
    for i, xi in enumerate(xs):
        w = 1.0
        for m, xm in enumerate(xs):
            if m != i:
                w *= (x - xm) / (xi - xm)
        out += w * ys[i]
    return out


# ═══════════════════════════════════════════════════════════════════════
# 1. The times of a slice point (model-free)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('k', [3, 4, 5])
def test_default_base_tau0_puts_the_swept_leg_below_the_pinned_legs(k):
    base = [0.0] * (k - 1)
    for j in range(1, k):
        t = dd._kpoint_slice_times(base, j, 0.0, EPS)
        want = [0.0] + [-EPS] * (k - 1)
        want[j] = -2 * EPS
        assert t == want, (k, j, t)
        assert all(type(x) is float for x in t)
    # the example of the docs, exactly
    assert dd._kpoint_slice_times([0.0, 0.0], 1, 0.0, EPS) == [0.0, -2e-6, -1e-6]


def test_crossing_a_base_lag_and_the_anchor():
    base = [0.5, 0.3]
    f = dd._kpoint_slice_times
    # swept leg meets the pinned leg's lag: one _ITO_EPS below it
    assert f(base, 1, 0.3, EPS) == [0.0, 0.3 - EPS, 0.3]
    assert f(base, 2, 0.5, EPS) == [0.0, 0.5, 0.5 - EPS]
    # swept leg meets only the anchor: -_ITO_EPS, exactly as before
    assert f(base, 1, 0.0, EPS) == [0.0, -EPS, 0.3] == _old_times(base, 1, 0.0)
    assert f(base, 2, 0.0, EPS) == [0.0, 0.5, -EPS] == _old_times(base, 2, 0.0)
    # the swept leg's own base lag plays no part
    assert f([0.0, 0.3], 2, 0.0, EPS) == [0.0, -EPS, -2 * EPS]
    assert f([0.0, 0.3], 2, 0.3, EPS) == [0.0, -EPS, 0.3]
    # k = 4, one pinned lag 0 and one nonzero
    assert f([0.0, 0.3, 0.9], 3, 0.0, EPS) == [0.0, -EPS, 0.3, -2 * EPS]
    assert f([0.0, 0.3, 0.9], 3, 0.3, EPS) == [0.0, -EPS, 0.3, 0.3 - EPS]
    assert f([0.0, 0.3, 0.9], 1, 0.3, EPS) == [0.0, 0.3 - EPS, 0.3, 0.9]


def test_the_swept_leg_never_lands_on_another_leg():
    f = dd._kpoint_slice_times
    # the first step (-_ITO_EPS below the anchor) meets a pinned leg whose
    # base lag is -_ITO_EPS: the step is repeated
    assert f([-EPS, 0.0], 2, 0.0, EPS) == [0.0, -EPS, -2 * EPS]
    assert f([0.0, -EPS, 0.3], 3, 0.0, EPS) == [0.0, -EPS, -EPS, -2 * EPS]
    # the swept value coincides with a pinned leg's EVALUATION time
    assert f([0.0, 0.0], 1, -EPS, EPS) == [0.0, -2 * EPS, -EPS]


def test_the_swept_leg_never_ties_where_eps_is_below_the_float_spacing():
    """For |t| >= 2**34 (~1.7e10), t - _ITO_EPS rounds back to t; the step is
    then the next float below the leg met, so the swept leg still never ties
    (it used to, e.g. at 1.8e10, 2e10 and -1e11)."""
    f = dd._kpoint_slice_times
    for base, j, tau in (([0.0, 1.8e10], 1, 1.8e10), ([2e10, 0.0], 2, 2e10),
                         ([-1e11, 0.0], 2, -1e11),
                         ([0.0, 2.0 ** 52], 1, 2.0 ** 52)):
        t = f(base, j, tau, EPS)
        assert t[j] == math.nextafter(tau, -math.inf), (base, j, t)
        assert len(set(t)) == len(t), t
        assert all(type(x) is float for x in t)
    # below 2**34 the ordinary _ITO_EPS step applies, as before
    assert f([0.0, 1.7e10], 1, 1.7e10, EPS) == [0.0, 1.7e10 - EPS, 1.7e10]
    # fuzz over magnitudes from 0 to 1e16: never a tie, always below the
    # legs met, the non-swept legs never moved
    rng = random.Random(20260929)
    mags = [0.0, 1e-13, 1e-12, 2e-12, 5e-7, 1e-6, 2e-6, 0.3, 1.0, 1e9,
            1.7e10, 1.8e10, 2e10, 1e11, 1e15, 2.0 ** 52, 1e16]
    for _ in range(20000):
        k = rng.choice([3, 4, 5])
        base = [rng.choice(mags) * rng.choice([1, -1]) for _ in range(k - 1)]
        j = rng.randint(1, k - 1)
        tau = rng.choice([0.0] + base + [b - EPS for b in base]
                         + [rng.choice(mags) * rng.choice([1, -1])])
        t = f(base, j, tau, EPS)
        pinned = [i for i in range(1, k) if i != j]
        assert [t[i] for i in pinned] == [
            dd._kpoint_pinned_time(base[i - 1], EPS) for i in pinned]
        assert all(abs(t[j] - t[i]) > 1e-12 for i in range(k) if i != j), (
            base, j, tau, t)
        if t[j] != tau:
            assert t[j] < tau, (base, j, tau, t)


def test_coincidence_tolerance_is_1e_12():
    f = dd._kpoint_slice_times
    for tau in (1e-12, -1e-12, 1e-13, -0.0):
        assert f([0.0, 0.0], 1, tau, EPS) == [0.0, -2 * EPS, -EPS], tau
    assert f([0.0, 0.0], 1, 2e-12, EPS) == [0.0, 2e-12, -EPS]
    assert f([0.3, 0.0], 2, 0.3 + 5e-13, EPS) == [0.0, 0.3, 0.3 - EPS]
    assert f([0.0, 0.3], 1, 0.3 + 5e-13, EPS) == [0.0, 0.3 - EPS, 0.3]
    assert f([0.0, 0.3], 1, 0.3 + 2e-12, EPS) == [0.0, 0.3 + 2e-12, 0.3]


def test_invalid_slice_index_raises():
    for j in (0, 3):
        with pytest.raises(ValueError, match='slice index'):
            dd._kpoint_slice_times([0.0, 0.0], j, 0.0, EPS)


def test_points_that_meet_no_other_leg_are_evaluated_as_before():
    """Only a point where the swept leg meets another leg moves; every other
    point gets exactly (==, as Python floats) the pre-change times."""
    grids = [np.linspace(-2.0, 2.0, 41), np.linspace(-10.0, 10.0, 21),
             np.array([0.0, 0.3, 0.7, 1e-13, -1e-13, 0.5, 2e-12])]
    bases = [[0.0, 0.0], [0.5, 0.5], [0.3, 0.7], [0.0, 0.3], [1.0, 0.0],
             [0.0] * 3, [0.3, 0.3, 0.7], [0.5, 0.0, 1.0]]
    n_same = n_moved = 0
    for base in bases:
        for grid in grids:
            for j in range(1, len(base) + 1):
                for tau in list(grid) + list(base):
                    t = float(tau)
                    new = dd._kpoint_slice_times(base, j, tau, EPS)
                    old = _old_times(base, j, tau)
                    pinned = [i for i in range(1, len(base) + 1) if i != j]
                    # the legs the swept leg meets: the anchor, and pinned
                    # legs by base lag or by evaluation time
                    met = ([0] if abs(t) <= 1e-12 else []) + [
                        i for i in pinned if abs(t - base[i - 1]) <= 1e-12
                        or abs(t - old[i]) <= 1e-12]
                    assert [new[i] for i in pinned] == [old[i] for i in pinned]
                    if met:
                        n_moved += 1
                        # strictly earlier than every leg it met, tied with none
                        assert new[j] < min(new[i] for i in met), (base, j, t)
                        assert all(abs(new[j] - new[i]) > 1e-12
                                   for i in range(len(new)) if i != j)
                    else:
                        n_same += 1
                        assert new == old, (base, j, tau)
                        assert all(type(x) is float for x in new)
    assert n_same > 1000 and n_moved > 100, (n_same, n_moved)


def test_pinned_legs_that_tie_stay_tied_along_the_whole_slice():
    """k = 4, default base lags: the two pinned legs of each slice sit at
    -_ITO_EPS at every point (their tie is left to the engine's leg-order
    rule, the same at every point, so it adds no jump along the curve)."""
    for j in (1, 2, 3):
        pinned = [i for i in (1, 2, 3) if i != j]
        for tau in np.linspace(-2.0, 2.0, 41):
            t = dd._kpoint_slice_times([0.0] * 3, j, tau, EPS)
            assert [t[i] for i in pinned] == [-EPS, -EPS]
            assert t[j] != -EPS


# ═══════════════════════════════════════════════════════════════════════
# 2. k = 2 is untouched
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def private_cwd(tmp_path_factory):
    """A private empty cwd for the model caches (fresh-clone state)."""
    prev = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('kpoint_tau0_cache'))
    try:
        yield
    finally:
        os.chdir(prev)


def _pre_change_daedalus():
    """The pre-change ``daedalus.py`` exec'd into a fresh module object (same
    process, same ``api``), or None without git / that revision."""
    try:
        src = subprocess.run(
            ['git', '-C', _REPO, 'show', f'{_PRE_CHANGE_REV}:daedalus.py'],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if src.returncode != 0 or '_kpoint_slice_times' in src.stdout:
        return None
    name = 'daedalus_pre_kpoint_nudge'
    mod = types.ModuleType(name)
    mod.__file__ = dd.__file__
    sys.modules[name] = mod           # @dataclass resolves its module by name
    try:
        exec(compile(src.stdout, dd.__file__, 'exec'), mod.__dict__)
    finally:
        del sys.modules[name]
    return mod


def test_k2_never_reaches_the_slice_code(private_cwd, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError('k = 2 must not use the k>=3 slice times')
    monkeypatch.setattr(dd, '_kpoint_slice_times', _boom)
    monkeypatch.setattr(dd, '_kpoint_pinned_time', _boom)
    model, mod = dd.load_model('ou_quartic')
    res = dd.run(model, dd.Config(k=2, max_ell=0, parameters=_P_OU,
                                  tau_grid=np.array([-1.0, 0.0, 1.0])), mod)
    assert '_kpoint_slice_times' not in res and 'C_tau_slices' not in res
    # the k = 2 grid samples its own tau = 0 point at -_ITO_EPS
    fn = res['phase_j_by_ell'][0]['total_C']
    assert complex(res['C_tau'][1]) == complex(fn(0.0, -EPS))


# ═══════════════════════════════════════════════════════════════════════
# 3. Models with delta parts: the tau = 0 point is the left limit
# ═══════════════════════════════════════════════════════════════════════

_RUNS = {}


def _run(which, k, grid, base=None, grouped=False, monkeypatch=None):
    """``dd.run`` of the <a b a ...> tree-level slices, cached per config."""
    key = (which, k, tuple(grid), None if base is None else tuple(base),
           grouped)
    if key not in _RUNS:
        name, params, _legs = _MODELS[which]
        model, mod = dd.load_model(name)
        if grouped:
            orig = api.compute_cumulants
            monkeypatch.setattr(
                api, 'compute_cumulants',
                lambda **kw: orig(**{**kw, 'use_grouped_phase_j': True}))
        _RUNS[key] = dd.run(model, dd.Config(
            k=k, max_ell=0, external_fields=_ext(which, k), parameters=params,
            tau_grid=np.asarray(grid, dtype=float),
            kpoint_base_lags=base), mod)
    return _RUNS[key]


def _limit_grid(*points, right=True):
    """tau grid: each point with its left (and right) samples."""
    out = []
    for p in points:
        out += [p + x for x in _XS_LEFT] + [p]
        if right:
            out += [p + x for x in reversed(_XS_RIGHT)]
    return out


def _check_left_limit(res, j, i0, rtol, jump=True):
    """The point i0 of slice j (whose left samples are i0-3..i0-1 and right
    samples, if any, i0+1..i0+3) is the left limit of the slice; returns
    (value, left limit, right limit or None)."""
    tau = np.asarray(res['tau_grid'], dtype=float)
    vals = np.real(np.asarray(res['C_tau_slices'][j]))
    times = res['_kpoint_slice_times'][j]
    assert np.all(np.isfinite(vals))
    has_right = i0 + 3 < tau.size
    x0 = float(times[i0][j])                  # where the swept leg really is
    assert x0 < float(tau[i0]), (j, x0)
    # the samples are the plain slice: swept leg at tau, below / above the tie
    samples = list(range(i0 - 3, i0))
    if has_right:
        samples += list(range(i0 + 1, i0 + 4))
    for i in samples:
        assert float(times[i][j]) == float(tau[i])
    left = _lagrange(tau[i0 - 3:i0], vals[i0 - 3:i0], x0)
    right = (_lagrange(tau[i0 + 1:i0 + 4], vals[i0 + 1:i0 + 4],
                       float(tau[i0])) if has_right else None)
    v = float(vals[i0])
    assert abs(v - left) <= rtol * abs(left), (
        f'slice {j} point {tau[i0]}: {v!r} vs left limit {left!r} '
        f'(rel {abs(v - left) / abs(left):.2e})')
    if jump:
        assert has_right and abs(right - left) > _JUMP * abs(left), (
            j, left, right)
    # the point is the cumulant at the recorded times
    fn = res['total_C']
    assert complex(fn(*[float(x) for x in times[i0]])).real == v
    return v, left, right


def test_k3_linear_delta_tau0_is_the_left_limit_of_both_slices(private_cwd):
    res = _run('ld', 3, _limit_grid(0.0))
    fn = res['total_C']
    for j in (1, 2):
        v, left, right = _check_left_limit(res, j, 3, _RTOL_ANALYTIC)
        old = complex(fn(*_old_times([0.0, 0.0], j, 0.0))).real
        if j == 1:
            # the old point was decided by the tie order: n2 (leg 1) after
            # n1 (leg 2) -- the right-hand side, not this curve's left limit
            assert abs(old - left) > _JUMP * abs(left)
        else:
            # leg 2 was already the earlier leg of the tie: O(_ITO_EPS) change
            assert abs(old - v) <= 1e-6 * abs(v)
    # the canonical single slice is slice 1
    assert np.array_equal(res['C_tau'], res['C_tau_slices'][1])


def test_k3_linear_delta_base_lag_crossings_are_left_limits(private_cwd):
    """kpoint_base_lags = [0.5, 0.3]: slice 1 crosses the pinned leg 2 at
    tau = 0.3, slice 2 the pinned leg 1 at tau = 0.5 (different fields: a
    jump), and both cross the anchor at tau = 0."""
    base = [0.5, 0.3]
    res = _run('ld', 3, _limit_grid(0.0, 0.3, 0.5), base=base)
    tau = list(np.asarray(res['tau_grid']))
    for j, cross in ((1, 0.3), (2, 0.5)):
        i0 = tau.index(cross)
        times = res['_kpoint_slice_times'][j][i0]
        assert list(times) == dd._kpoint_slice_times(base, j, cross, EPS)
        assert times[j] == cross - EPS
        _check_left_limit(res, j, i0, _RTOL_ANALYTIC)
        # tau = 0: the swept leg meets only the anchor (-_ITO_EPS, unchanged)
        i0 = tau.index(0.0)
        assert list(res['_kpoint_slice_times'][j][i0]) == _old_times(base, j, 0.0)
        _check_left_limit(res, j, i0, _RTOL_ANALYTIC, jump=False)


@pytest.mark.slow
@pytest.mark.parametrize('which, rtol', [('spike', _RTOL_NQUAD),
                                         ('mp', _RTOL_ANALYTIC)],
                         ids=['spike_reset', 'multipopulation'])
def test_k3_tau0_is_the_left_limit_of_both_slices(private_cwd, which, rtol):
    # multipopulation_test: jumps of only 4.6e-6 (slice 1) and 5.6e-7
    # (slice 2) relative at tau = 0 (CHANGELOG k = 3 table), so no jump
    # check and no right samples (~9 s per evaluation).  The point is
    # checked against the left branch at the swept leg's time (-2e-6); the
    # left limit at tau = 0 itself is 1.8e-6 / 2.4e-6 relative away, the
    # O(2 _ITO_EPS) offset of the nudge.
    spike = which == 'spike'
    res = _run(which, 3, _limit_grid(0.0, right=spike))
    for j in (1, 2):
        _check_left_limit(res, j, 3, rtol, jump=spike)
    if which == 'spike':
        # the M1 review's example: the old point +4.96369e-2 was neither limit
        v, left, right = _check_left_limit(res, 1, 3, rtol)
        old = complex(res['total_C'](0.0, -EPS, -EPS)).real
        # the tabulated values (CHANGELOG k = 3 table), to their 6 digits
        assert abs(left - (-3.32555e-1)) <= 1e-5 * 3.32555e-1
        assert abs(right - (-2.34321e-1)) <= 1e-5 * 2.34321e-1
        assert abs(old - 4.96369e-2) < 1e-6


@pytest.mark.slow
def test_k3_exact_tie_on_the_full_grid_diagonal_is_the_limit(private_cwd):
    """The full grid's diagonals (and the coincident non-swept legs of the
    k >= 4 slices) keep the engine's tie order: at an exact tie the value is
    the one-sided limit in which the later-listed leg approaches from below
    -- as accurately as the tie is evaluated.
    multipopulation_test <nE1 nE2 nE1>, tree, at (0, 0.4, 0.4) (a point of
    the full grid's diagonal whenever 0.4 is on its axis) sends 16 two-time
    regions to the scipy quadrature fallback (polygon triangle guard; none
    at (0, 0.4, 0.4 - 1e-7)).  Before the hardened fallback (M2b) the tie
    was +1.6859737e-6 against the limit +1.6859242e-6 (2.9e-5 relative,
    twelve times the jump between the two one-sided limits, 2.4e-6; this
    test was a strict expected failure); with it, +1.68592416791e-6, the
    limit (Lagrange from 0.4 - 1e-5, 1e-6, 1e-7) to 6.3e-15 relative
    (measured in process against the pre-M2b engine).  At
    (0, 0.7, 0.7) nothing reaches the fallback and the tie equals the limit
    to 8.6e-16.  Reuses the multipopulation build of the slice test above
    when both run (~4 evaluations, ~9-13 s each)."""
    res = _run('mp', 3, _limit_grid(0.0, right=False))
    fn = res['total_C']

    def f(*t):
        return complex(fn(*t)).real

    tie = f(0.0, 0.4, 0.4)
    xs = (-1e-5, -1e-6, -1e-7)               # leg 2 approaching from below
    lim = _lagrange(xs, [f(0.0, 0.4, 0.4 + x) for x in xs], 0.0)
    assert abs(tie - lim) <= 1e-8 * abs(lim), (
        f'tie {tie!r} vs limit {lim!r} (rel {abs(tie - lim) / abs(lim):.2e})')


@pytest.mark.slow
def test_k4_linear_delta_tau0_is_the_left_limit_of_every_slice(private_cwd):
    """<n1 n2 n1 n2>, default base lags: the swept leg at -2 _ITO_EPS below
    the two pinned legs (tied at -_ITO_EPS, by the engine's leg order)."""
    res = _run('ld', 4, _limit_grid(0.0))
    fn = res['total_C']
    for j in (1, 2, 3):
        v, left, right = _check_left_limit(res, j, 3, _RTOL_ANALYTIC)
        old = complex(fn(*_old_times([0.0] * 3, j, 0.0))).real
        # the tie order put the swept leg after the later-listed pinned legs
        # (slices 1, 2); slice 3's swept leg was already the earliest
        moved = abs(old - v) > _JUMP * abs(v)
        assert moved == (j in (1, 2)), (j, old, v)


@pytest.mark.slow
@pytest.mark.parametrize('which, rtol', [('ld', _RTOL_ANALYTIC),
                                         ('spike', _RTOL_NQUAD)],
                         ids=['linear_delta_spikes', 'spike_reset'])
def test_k3_tau0_left_limit_on_the_grouped_path(private_cwd, monkeypatch,
                                                which, rtol):
    res = _run(which, 3, _limit_grid(0.0), grouped=True,
               monkeypatch=monkeypatch)
    for j in (1, 2):
        _check_left_limit(res, j, 3, rtol)


# ═══════════════════════════════════════════════════════════════════════
# 4. OU: no delta parts, continuous at tau = 0, moves at O(_ITO_EPS**2)
# ═══════════════════════════════════════════════════════════════════════

def test_ou_quartic_tau0_moves_only_at_the_ito_eps_level(private_cwd):
    model, mod = dd.load_model('ou_quartic')
    res = dd.run(model, dd.Config(
        k=4, max_ell=1, external_fields=[('dx', 1)] * 4, parameters=_P_OU,
        tau_grid=np.asarray(_limit_grid(0.0))), mod)
    for j in (1, 2, 3):
        for ell, fn in res['total_C_by_ell'].items():
            v = float(np.real(res['C_tau_slices_by_ell'][j][ell][3]))
            old = complex(fn(*_old_times([0.0] * 3, j, 0.0))).real
            # measured: 2.50e-12 (tree), 1.07e-12 (one loop) relative
            assert 0 < abs(v - old) <= 1e-10 * abs(old), (j, ell, v, old)
        _check_left_limit(res, j, 3, _RTOL_ANALYTIC, jump=False)
    # k = 3 of the symmetric ou_quartic is identically 0, before and after
    r3 = dd.run(model, dd.Config(
        k=3, max_ell=1, external_fields=[('dx', 1)] * 3, parameters=_P_OU,
        tau_grid=np.array([-0.5, 0.0, 0.5])), mod)
    for j in (1, 2):
        assert np.all(np.asarray(r3['C_tau_slices'][j]) == 0)


# ═══════════════════════════════════════════════════════════════════════
# 5. k = 2 against the pre-change module (last: the linear delta spikes
#    propagator is then already in the private cache)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('name, ext, params, max_ell', [
    ('single_population_linear_delta_spikes_test', [('n', 1), ('n', 2)],
     _P_LD, 0),
    ('ou_quartic', [('dx', 1), ('dx', 1)], _P_OU, 1),
], ids=['linear_delta_spikes', 'ou_quartic'])
def test_k2_run_is_bit_identical_to_the_pre_change_module(
        private_cwd, name, ext, params, max_ell):
    old = _pre_change_daedalus()
    if old is None:
        pytest.skip(f'git revision {_PRE_CHANGE_REV[:7]} not available')
    model, mod = dd.load_model(name)
    grid = np.array([-2.0, -0.5, 0.0, 1e-13, 0.5, 2.0])

    def cfg(module):
        return module.Config(k=2, max_ell=max_ell, external_fields=ext,
                             parameters=params, tau_grid=grid)

    dd.run(model, cfg(dd), mod)                             # warm the caches
    r_old = old.run(model, cfg(old), mod)
    r_new = dd.run(model, cfg(dd), mod)
    assert np.array_equal(r_old['tau_grid'], r_new['tau_grid'])
    assert np.array_equal(r_old['C_tau'], r_new['C_tau'])
    assert sorted(r_old['C_tau_by_ell']) == sorted(r_new['C_tau_by_ell'])
    for ell in r_old['C_tau_by_ell']:
        assert np.array_equal(r_old['C_tau_by_ell'][ell],
                              r_new['C_tau_by_ell'][ell]), ell
    assert sorted(r_old) == sorted(r_new)                   # no new keys
    for key in ('k', 'max_ell', 'external_fields'):
        assert r_old['_resolved'][key] == r_new['_resolved'][key]


# ═══════════════════════════════════════════════════════════════════════
# 6. Moment outputs (Config.output = 'central_moment'): the cumulant
#    blocks of 3 or more legs are evaluated at the times of slice 1
# ═══════════════════════════════════════════════════════════════════════

def test_k3_central_moment_is_slice_1_including_its_tau0_left_limit(
        private_cwd):
    """The k = 3 central moment IS kappa_3, so it equals slice 1 (``C_tau``)
    bit-for-bit, also at tau = 0, where slice 1 is its left limit and the
    old moment point (0, 0, 0) was the tie-order value (3.0 % higher for
    <n1 n2 n1> of the linear delta spikes)."""
    model, mod = dd.load_model('single_population_linear_delta_spikes_test')
    grid = np.array([-0.5, 0.0, 0.5])
    res = dd.run(model, dd.Config(
        k=3, max_ell=0, external_fields=_ext('ld', 3), parameters=_P_LD,
        tau_grid=grid, output='central_moment'), mod)
    assert res['output_kind'] == 'central_moment'
    assert np.array_equal(res['moment'], res['C_tau'])
    tie = complex(res['total_C'](0.0, 0.0, 0.0)).real
    m0 = float(np.real(res['moment'][1]))
    assert abs(m0 - tie) > _JUMP * abs(m0), (m0, tie)
    assert complex(res['total_C'](0.0, -2 * EPS, -EPS)).real == m0


def test_k4_central_moment_kappa4_block_is_slice_1(private_cwd):
    """k = 4, one field: the central moment minus its pair terms
    (3 kappa_2(tau) kappa_2(0), loop budget shared) is slice 1 to rounding
    (measured <= 5.4e-15 relative); the old block times (0, tau, 0, 0)
    differ from it by 6.1e-7 relative at tau = +-0.5."""
    model, mod = dd.load_model('ou_quartic')
    grid = np.array([-0.5, 0.0, 0.5])
    ext = [('dx', 1)] * 4
    r4 = dd.run(model, dd.Config(
        k=4, max_ell=1, external_fields=ext, parameters=_P_OU,
        tau_grid=grid, output='central_moment'), mod)
    r2 = dd.run(model, dd.Config(
        k=2, max_ell=1, external_fields=ext[:2], parameters=_P_OU,
        tau_grid=grid), mod)
    k2 = {e: np.real(np.asarray(r2['C_tau_by_ell'][e])) for e in (0, 1)}
    pairs = 3.0 * (k2[0] * k2[0][1] + k2[0] * k2[1][1] + k2[1] * k2[0][1])
    k4 = np.real(np.asarray(r4['moment'])) - pairs
    c = np.real(np.asarray(r4['C_tau']))
    assert np.allclose(k4, c, rtol=1e-13, atol=0), (k4, c)
    old = np.array([complex(r4['total_C'](0.0, float(t), 0.0, 0.0)).real
                    for t in (-0.5, 0.5)])
    assert np.all(np.abs(old - c[[0, 2]]) > 1e-8 * np.abs(c[[0, 2]]))

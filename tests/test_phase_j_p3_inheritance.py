"""P3: the poset lower-bound inheritance rule and the exact DBM route
(Phase J milestone M3, ``final_integral.USE_DBM_FALLBACK``).

The m≥3 poset path integrates every variable from ONE shared scalar lower
bound L.  A variable without a scalar lower of its own is bounded below by L
only through a predecessor with one in the transitive closure of the strict
order (s_v > s_u > L).  Before M3 every variable inherited L regardless, so
a variable with no such predecessor -- one that extends below L, down to the
domain cap -- had that part of the region cut off.  Since M3 the poset path
refuses such a poset (``'poset_lower_not_inherited'``) and the exact DBM
route (``_integrate_subset_dbm``) integrates it on the scipy fallback's box.

* the i907 subset: a model-free m=3 subset (poset s_2 < s_1 < s_0,
  0 < s_0 < 1) taken from the public spike-reset model's k=2 ell=1
  one-loop.  Its exact raw value (prefactor included) is
  2.60362113647539e-07; the pre-M3 poset path gave 2.7212e-09 at any cap;
* the inheritance rule on hand-made posets (transitive, diamond, a
  successor that does not bound below, flag off, 'legacy_clip');
* a tied external time: a constant row comparing two legs' times at an
  exact tie is decided by the tie order (a later-listed leg is
  infinitesimally earlier), and the DBM value at the tie is the one-sided
  limit;
* end to end (``single_population_spike_reset_test``, the fixture
  ``spike_reset_k2_ell1``): every P3 subset of the one-loop at (0, 1) is
  answered by the DBM route (none reaches the scipy fallback), the
  one-loop moves to the corrected value at τ = 1 and 3 (flag on vs off in
  one process), grouped equals per-diagram, every m≥3 region forced
  through the DBM gives the same total (non-P3 bails take the route too),
  and the value at the exact tie (0, 0) is the left limit, with the tie
  context reaching the DBM.

Values of the end-to-end checks are current values, not validated
(CHANGELOG 0.2.0).  Runtime: ~5 s model-free, ~1 min end to end (the
model is built in a private cwd, so its caches start empty).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.integration.time_domain.final_integral as FI  # noqa: E402
import engine.integration.time_domain.grouped_integral as GI  # noqa: E402


@pytest.fixture(autouse=True)
def _ito_mode(monkeypatch):
    """The default Θ(0) mode, whatever the environment says
    (``DAEDALUS_PHASE_J_LEGACY``); a test that needs another sets it."""
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'ito')

# ═══════════════════════════════════════════════════════════════════════
# The i907 subset (model-free numbers from the public spike-reset model)
# ═══════════════════════════════════════════════════════════════════════
# Rows (a_int, a_ext, c0): a_int·s + a_ext·t + c0 > 0, t = FREE.
# s_0 < t, s_1 < s_0, s_2 < s_1 (twice), s_0 > 0: s_0 has a scalar lower
# (0), s_1 and s_2 have no lower-bounded predecessor.
I907_PREF = 0.18095710274066346 + 0j
I907_FREE = [1.0]
I907_ROWS = [
    ((-1.0, 0.0, 0.0), (1.0,), 0.0),
    ((1.0, -1.0, 0.0), (0.0,), 0.0),
    ((0.0, 1.0, -1.0), (0.0,), 0.0),
    ((0.0, 1.0, -1.0), (0.0,), 0.0),
    ((1.0, 0.0, 0.0), (0.0,), 0.0),
]
_LAM_A = -0.16642039481632437 - 9.020562075079397e-17j
_LAM_B = -0.19513338438151706 + 1.6653345369377348e-16j
I907_EDGE_MODES = [
    ((0.011914296190551303 - 7.077124554334512e-17j, _LAM_A),
     (0.0436412593650042 + 1.2724666782500463e-16j, _LAM_B)),
    ((0.011914296190551303 - 7.077124554334512e-17j, _LAM_A),
     (0.0436412593650042 + 1.2724666782500463e-16j, _LAM_B)),
    ((-0.0034500477044512893 + 1.4620425898143496e-16j, _LAM_A),
     (-0.05774784777981831 - 2.449313607062822e-16j, _LAM_B)),
    ((-0.0017250238522253125 + 1.112128288665055e-16j, _LAM_A),
     (-0.02887392388990958 - 1.063873307066642e-16j, _LAM_B)),
    ((-0.01848280146218943 - 1.9231051876152722e-16j, _LAM_A),
     (0.040705023684411534 + 3.0265297022750724e-16j, _LAM_B)),
]
I907_EXACT = 2.60362113647539e-07       # raw subset integral, prefactor in
I907_PRE_M3 = 2.721217908753063e-09     # the pre-M3 poset path (any cap)


def _i907_modes():
    return [FI.EdgeModeSum(ri=-1, pi=-1, delta_coeff=0j, modes=md)
            for md in I907_EDGE_MODES]


def test_i907_poset_path_refuses_and_dbm_is_exact(monkeypatch):
    """A1: the poset path bails on the inheritance rule and the DBM route
    gives the exact value (rtol 1e-9), with and without a plan."""
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', True)
    modes = _i907_modes()
    FI._reset_runtime_counters()
    assert FI._integrate_nd_polytope_poset_modesum(
        modes, I907_PREF, I907_ROWS, I907_FREE, 3) is None
    assert FI._pop_bail_reason() == 'poset_lower_not_inherited'
    assert FI._RUNTIME_COUNTERS['poset_lower_not_inherited'] == 1
    plan = FI._build_modesum_plan(modes, I907_ROWS, 3, 1)
    for kw in ({}, {'plan': plan}):
        v = FI._integrate_subset_dbm(modes, I907_PREF, I907_ROWS, I907_FREE,
                                     3, poset_bail_reason=(
                                         'poset_lower_not_inherited'), **kw)
        assert abs(v - I907_EXACT) <= 1e-9 * I907_EXACT, (kw, v)
        assert abs(v.imag) <= 1e-12 * I907_EXACT
    assert FI._RUNTIME_COUNTERS['dbm_answered_p3'] == 2


def test_i907_exact_value_by_independent_quadrature():
    """The embedded exact value, re-derived by tplquad of the raw
    integrand over s_2 < s_1 < s_0, 0 < s_0 < 1 (lower limits -200, the
    DBM's box; the integrand decays like exp(-0.166 Δt) into the past)."""
    from scipy import integrate
    rows = [(np.array(a), np.array(e), c) for a, e, c in I907_ROWS]

    def f(s2, s1, s0):
        s = np.array([s0, s1, s2])
        v = I907_PREF
        for (a, e, c), md in zip(rows, I907_EDGE_MODES):
            dt = a @ s + e @ np.array(I907_FREE) + c
            v *= sum(C * np.exp(lam * dt) for C, lam in md)
        return v.real

    val, _err = integrate.tplquad(
        f, 0.0, 1.0, lambda s0: -200.0, lambda s0: s0,
        lambda s0, s1: -200.0, lambda s0, s1: s1,
        epsabs=1e-20, epsrel=1e-11)
    assert abs(val - I907_EXACT) <= 1e-9 * I907_EXACT, val


def test_i907_flag_off_is_the_pre_m3_value(monkeypatch):
    """``USE_DBM_FALLBACK = False`` restores the pre-M3 inheritance (the
    rollback value; wrong by a factor ~96 here), at any cap."""
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', False)
    for cap in (None, 50.0, 1000.0):
        v = FI._integrate_nd_polytope_poset_modesum(
            _i907_modes(), I907_PREF, I907_ROWS, I907_FREE, 3, bbox_cap=cap)
        assert v == pytest.approx(I907_PRE_M3, rel=1e-12, abs=0)


def test_i907_legacy_theta0_mode_keeps_the_pre_m3_route(monkeypatch):
    """'legacy_clip' reproduces the pre-M1 numbers bit-for-bit, so the M3
    changes do not apply in it (whatever ``USE_DBM_FALLBACK`` says)."""
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', True)
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'legacy_clip')
    assert not FI._dbm_route_on()
    v = FI._integrate_nd_polytope_poset_modesum(
        _i907_modes(), I907_PREF, I907_ROWS, I907_FREE, 3)
    assert v == pytest.approx(I907_PRE_M3, rel=1e-12, abs=0)


# ═══════════════════════════════════════════════════════════════════════
# The flag
# ═══════════════════════════════════════════════════════════════════════

def test_flag_initialisation_from_the_environment():
    f = FI._initial_phase_j_flags
    assert f({})['USE_DBM_FALLBACK'] is True
    for v in ('0', 'false', 'no', 'off', ' OFF '):
        assert f({'DAEDALUS_PHASE_J_DBM': v})['USE_DBM_FALLBACK'] is False
    for v in ('', '1', 'true', 'yes', 'on'):
        assert f({'DAEDALUS_PHASE_J_DBM': v})['USE_DBM_FALLBACK'] is True
    # the umbrella wins over the per-flag variable
    assert f({'DAEDALUS_PHASE_J_LEGACY': '1',
              'DAEDALUS_PHASE_J_DBM': '1'})['USE_DBM_FALLBACK'] is False
    with pytest.raises(ValueError):
        f({'DAEDALUS_PHASE_J_DBM': 'maybe'})
    if not any(os.environ.get(v) for v in (
            'DAEDALUS_PHASE_J_LEGACY', 'DAEDALUS_PHASE_J_DBM',
            'DAEDALUS_PHASE_J_THETA0_CONST_ROW')):
        assert FI.USE_DBM_FALLBACK is True and FI._dbm_route_on()


def test_invalid_flag_attribute_raises(monkeypatch):
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', 'yes')
    with pytest.raises(ValueError):
        FI._dbm_route_on()
    with pytest.raises(ValueError):
        FI._integrate_nd_polytope_poset_modesum(
            _i907_modes(), I907_PREF, I907_ROWS, I907_FREE, 3)


# ═══════════════════════════════════════════════════════════════════════
# The inheritance rule
# ═══════════════════════════════════════════════════════════════════════

def _poset(m, edges, lowers):
    return FI._CausalPoset(m=m, edges=tuple(edges),
                           scalar_lowers=tuple(lowers), scalar_uppers=())


@pytest.mark.parametrize('m, edges, lowers, inherited', [
    # chain s_0 < s_1 < s_2 < s_3, lower on the bottom: all inherit
    (4, [(0, 1), (1, 2), (2, 3)], [(0, 0.0)], True),
    # the same chain, lower on s_1: s_0 is below it and does not
    (4, [(0, 1), (1, 2), (2, 3)], [(1, 0.0)], False),
    # transitive: s_3 > s_2 > s_1 > s_0 > L through two plain hops
    (4, [(2, 3), (1, 2), (0, 1)], [(0, -1.0)], True),
    # diamond: s_3 > s_1 > s_0, s_3 > s_2 (no lower), s_2 alone below
    (4, [(0, 1), (1, 3), (2, 3)], [(0, 0.0)], False),
    # diamond with the lower on both feet
    (4, [(0, 1), (1, 3), (2, 3)], [(0, 0.0), (2, 0.0)], True),
    # an isolated variable without a lower
    (3, [(0, 1)], [(0, 0.0)], False),
    # every variable has its own lower (equal): nothing to inherit
    (3, [], [(0, 2.0), (1, 2.0), (2, 2.0)], True),
    # the i907 shape: s_2 < s_1 < s_0 > 0
    (3, [(1, 0), (2, 1)], [(0, 0.0)], False),
])
def test_inheritance_rule(m, edges, lowers, inherited, monkeypatch):
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', True)
    p = _poset(m, edges, lowers)
    L, ok = FI._causal_poset_consistent_scalar_lower(p)
    assert ok is inherited
    if inherited:
        assert L == max(c for _v, c in lowers)
    else:
        assert L is None
        assert FI._poset_scalar_lower_verdict(p)[2] == 'not_inherited'
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', False)
    L, ok = FI._causal_poset_consistent_scalar_lower(p)
    assert ok is True and L == max(c for _v, c in lowers)


def test_no_lower_at_all_and_unequal_lowers_are_unchanged(monkeypatch):
    """No scalar lower: (None, True) (the caller's cap); unequal lowers:
    'inconsistent', before the inheritance rule is asked."""
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', True)
    assert FI._poset_scalar_lower_verdict(_poset(3, [], [])) == (
        None, True, None)
    assert FI._poset_scalar_lower_verdict(
        _poset(3, [], [(0, 0.0), (1, 1.0)])) == (None, False, 'inconsistent')


# ═══════════════════════════════════════════════════════════════════════
# A tied external time (model-free)
# ═══════════════════════════════════════════════════════════════════════
# Legs 0 (origin, t_0) and 1 (free value t = t_1 - t_0).  The P3 shape
# 0 < s_0 < t, s_2 < s_1 < s_0, plus a constant row Δt = ±(t_1 - t_0)
# (both endpoints merged into the legs' times by δ edges).  At t_1 == t_0
# the tie order makes leg 1 infinitesimally EARLIER: Θ(t_1 - t_0) = 0
# (EMPTY), Θ(t_0 - t_1) = 1 (DROP) -- the left limit t → 0⁻ of a strip that
# closes as t → 0.  We put the tie at t = 0.8 by shifting the box: rows
# with t_0 = 0, leg 1 at t, and the constant row compares legs 1 and 2
# (both at t, leg 2 the free value t_2 - t_0) so the region stays open.

_TIE_BASE = [
    ((-1.0, 0.0, 0.0), (1.0, 0.0), 0.0),    # s_0 < t_1
    ((1.0, 0.0, 0.0), (0.0, 0.0), 0.0),     # s_0 > 0
    ((1.0, -1.0, 0.0), (0.0, 0.0), 0.0),    # s_1 < s_0
    ((0.0, 1.0, -1.0), (0.0, 0.0), 0.0),    # s_2 < s_1
]
_TIE_LAM = (-0.4 + 0.15j, -0.25 - 0.1j)


def _tie_modes(n):
    return [FI.EdgeModeSum(ri=-1, pi=-1, delta_coeff=0j,
                           modes=((0.7 + 0.2j, _TIE_LAM[0]),
                                  (-0.3 + 0.1j, _TIE_LAM[1])))
            for _ in range(n)]


@pytest.mark.parametrize('sign', [+1.0, -1.0])
def test_tied_external_times_take_the_tie_order(sign, monkeypatch):
    """Δt = sign·(t_2 - t_1) at t_1 == t_2 == 0.8 (legs 1, 2 tied): the DBM
    decides the constant row by the tie order and equals the one-sided
    limit in which leg 2 (the later-listed) sits infinitesimally below
    leg 1; the scipy fallback gives the same verdict."""
    monkeypatch.setattr(FI, 'USE_DBM_FALLBACK', True)
    row = ((0.0, 0.0, 0.0), (-sign, sign), 0.0)
    rows = _TIE_BASE + [row]
    modes = _tie_modes(len(rows))
    t = 0.8
    tie = FI._TieContext(times=(0.0, t, t), free_legs=(1, 2), origin_leg=0)
    at_tie = FI._integrate_subset_dbm(modes, 1.0, rows, [t, t], 3,
                                      tie_ctx=tie)
    # the one-sided limit: leg 2 at t - h, h -> 0+ (Richardson, linear)
    lim = []
    for h in (1e-6, 2e-6):
        ctx = FI._TieContext(times=(0.0, t, t - h), free_legs=(1, 2),
                             origin_leg=0)
        lim.append(FI._integrate_subset_dbm(modes, 1.0, rows, [t, t - h], 3,
                                            tie_ctx=ctx))
    left = 2 * lim[0] - lim[1]
    if sign > 0:                 # Θ(t_2 - t_1): leg 2 below leg 1 -> EMPTY
        assert at_tie == 0 and lim[0] == 0
    else:                        # Θ(t_1 - t_2) holds: the row is dropped
        # from the geometry; its edge contributes Σ_α C_α exp(λ_α · 0)
        no_row = FI._integrate_subset_dbm(_tie_modes(4), 1.0, _TIE_BASE,
                                          [t, t], 3)
        edge0 = sum(C for C, _l in modes[-1].modes)
        assert at_tie == pytest.approx(no_row * edge0, rel=1e-13, abs=0)
        assert abs(at_tie - left) <= 1e-10 * abs(at_tie)
    # the hardened scipy fallback decides the row the same way
    fe = FI._build_fast_subset_evaluator_from_modes(1.0 + 0j, modes, rows, 3)
    resolved = [(list(a), c0 + sum(x * f for x, f in zip(e, [t, t])))
                for (a, e, c0) in rows]
    nq = FI._integrate_polytope(fe, resolved, [t, t], 3, raw_rows=rows,
                                tie_ctx=tie)
    assert abs(nq - at_tie) <= 1e-8 * max(abs(at_tie), 1e-300) + 1e-14


# ═══════════════════════════════════════════════════════════════════════
# End to end: single_population_spike_reset_test, k = 2, ell = 1
# ═══════════════════════════════════════════════════════════════════════
# The parameters of the frozen fixture ``spike_reset_k2_ell1`` (and of the
# zoo entry ``spike_reset-k2-l1``).
_SPIKE = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
          'w': [[0.55, 0.65], [0.7, 0.8]]}
# One-loop term at the raw point (0, 1).  Pre-M3 (the flag off): measured on
# the unmodified code.  M3: the current value, not validated -- the model
# has no exact solution; it equals the pre-M3 value plus the sum over the 96
# P3-affected regions of (tight quadrature - pre-M3 value), measured
# independently (M3 brief: -3.8660e-3 to 5 digits).
# τ: (pre-M3, M3, independent).
_ONE_LOOP = {
    1.0: (-6.3683627449738905e-03, -3.865960659466003e-03, -3.8660e-03),
    3.0: (-1.2953050283558442e-03, -1.0261461422474269e-03, -1.0261e-03),
}


@pytest.fixture(scope='module')
def spike_k2(tmp_path_factory):
    """{grouped: raw one-loop callable}, built once per path in a private
    cwd (every cache root is cwd-relative, so the caches start empty).  The
    raw Phase J callable (``phase_j_by_ell[1]['total_C']``, as
    ``tests/tools/phase_j_subset_diff.py`` uses it) evaluates exactly the
    times it is given; the API's ``total_C_by_ell`` moves a k = 2 tie to
    τ = -1e-6."""
    import daedalus as dd
    from api import compute_cumulants
    prev = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('phase_j_p3_cache'))
    try:
        model = dd.load_model('single_population_spike_reset_test')[0]
        out = {}
        for grouped in (False, True):
            res = compute_cumulants(
                model, k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)],
                parameters=_SPIKE, tau_grid=np.array([1.0]), use_cache=True,
                parallel=False, verbose=False, use_grouped_phase_j=grouped)
            out[grouped] = res['phase_j_by_ell'][1]['total_C']
        yield out
    finally:
        os.chdir(prev)


def _eval(fn, pt, flag):
    saved = FI.USE_DBM_FALLBACK
    FI.USE_DBM_FALLBACK = flag
    FI._reset_runtime_counters()
    try:
        v = complex(fn(*pt))
    finally:
        FI.USE_DBM_FALLBACK = saved
    c = {k: (dict(v_) if isinstance(v_, dict) else v_)
         for k, v_ in FI._RUNTIME_COUNTERS.items()}
    return v, c


@pytest.mark.parametrize('tau', sorted(_ONE_LOOP))
def test_spike_reset_p3_regions_take_the_dbm_route(spike_k2, tau):
    """Every P3 refusal of the one-loop at (0, τ) is answered by the DBM
    route (none reaches the scipy fallback; no m≥3 region does), and the
    one-loop moves from the pre-M3 value to the corrected one.  The flag is
    read at call time: the same callable gives the pre-M3 value with it
    off."""
    pre_m3, m3, independent = _ONE_LOOP[tau]
    fn = spike_k2[False]
    v_on, c_on = _eval(fn, (0.0, tau), True)
    assert c_on['poset_lower_not_inherited'] > 0
    assert c_on['dbm_answered_p3'] == c_on['poset_lower_not_inherited']
    assert c_on['scipy_nquad_called_mge3'] == 0
    assert abs(v_on.real - m3) <= 1e-9 * abs(m3)
    assert abs(v_on.real - independent) <= 1e-7
    v_off, c_off = _eval(fn, (0.0, tau), False)
    assert c_off['poset_lower_not_inherited'] == 0
    assert c_off['dbm_attempted'] == 0
    assert abs(v_off.real - pre_m3) <= 1e-12 * abs(pre_m3)


@pytest.mark.parametrize('tau', sorted(_ONE_LOOP))
def test_spike_reset_grouped_equals_per_diagram(spike_k2, tau):
    """Grouped Phase J shares the inheritance rule and the DBM route: the
    two paths agree to 1e-12 relative, and the grouped one refuses and
    re-routes its P3 regions too."""
    v_pd, _c = _eval(spike_k2[False], (0.0, tau), True)
    v_gr, c_gr = _eval(spike_k2[True], (0.0, tau), True)
    assert c_gr['poset_lower_not_inherited'] > 0
    assert c_gr['dbm_answered_p3'] == c_gr['poset_lower_not_inherited']
    assert abs(v_gr - v_pd) <= 1e-12 * abs(v_pd), (v_gr, v_pd)


@pytest.mark.parametrize('grouped', [False, True])
def test_spike_reset_every_m3_region_through_the_dbm(spike_k2, grouped,
                                                     monkeypatch):
    """The poset path forced to bail on every m≥3 region with a non-P3
    reason (``'poset_no_extension'``): the DBM route takes all of them
    (per-diagram at (0, 1): 520 regions, m = 3 and 4, 32 of them split
    into two elimination cases, against at most one case on every P3
    region), none is declined or reaches the scipy fallback, none is
    counted as P3, and the total equals the normal one (poset where valid,
    DBM on P3) to 1e-12 relative (measured ~1e-16)."""
    fn = spike_k2[grouped]
    v_ref, c_ref = _eval(fn, (0.0, 1.0), True)
    forced = (lambda *a, **k: FI._bail('poset_no_extension'))
    monkeypatch.setattr(FI, '_integrate_nd_polytope_poset_modesum', forced)
    monkeypatch.setattr(GI, '_integrate_nd_polytope_poset_modesum', forced)
    v, c = _eval(fn, (0.0, 1.0), True)
    assert c['dbm_attempted'] == c['dbm_answered'] > c_ref['dbm_answered']
    assert c['dbm_answered_p3'] == 0 and c['poset_lower_not_inherited'] == 0
    assert c['scipy_nquad_called_mge3'] == 0
    assert c['nquad_calls'] == c_ref['nquad_calls']        # m ≤ 2 only
    assert abs(v - v_ref) <= 1e-12 * abs(v_ref), (v, v_ref)


def test_spike_reset_exact_tie_is_the_left_limit(spike_k2, monkeypatch):
    """At the exact tie (0, 0) the tie order makes leg 1 infinitesimally
    earlier: the one-loop is the left limit τ → 0⁻ (quadratic Richardson
    extrapolation from τ = -1e-4, -1e-5, -1e-6), with P3 regions answered
    by the DBM route at the tie (cf. ``tests/test_phase_j_ties.py``).
    Every one of them is exactly 0 there (measured: all EMPTY), and the
    Itô rule alone would give the same, so the comparison cannot see a tie
    context lost on the way to the DBM: a spy checks that every DBM call
    receives one."""
    fn = spike_k2[False]
    real, ctxs = FI._integrate_subset_dbm, []

    def spy(*a, **k):
        ctxs.append(k.get('tie_ctx'))
        return real(*a, **k)
    monkeypatch.setattr(FI, '_integrate_subset_dbm', spy)
    v_tie, c_tie = _eval(fn, (0.0, 0.0), True)
    monkeypatch.setattr(FI, '_integrate_subset_dbm', real)
    assert c_tie['poset_lower_not_inherited'] > 0
    assert c_tie['dbm_answered_p3'] == c_tie['poset_lower_not_inherited']
    assert len(ctxs) == c_tie['dbm_attempted'] > 0
    assert all(isinstance(t, FI._TieContext) for t in ctxs)
    hs = (1e-4, 1e-5, 1e-6)
    vals = [_eval(fn, (0.0, -h), True)[0] for h in hs]
    # f(h) = f0 + a h + b h^2 through the three nudges
    A = np.array([[1.0, -h, h * h] for h in hs])
    f0 = np.linalg.solve(A, np.array(vals))[0]
    assert abs(v_tie - f0) <= 1e-9 * abs(f0), (v_tie, f0)

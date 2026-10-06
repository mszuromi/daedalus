"""The exact DBM route (``engine/integration/time_domain/dbm_integral.py``,
Phase J milestone M3) and its ``final_integral`` wrapper
``_integrate_subset_dbm``.

* a plain chain equals the closed-form chain simplex
  ``_exp_over_chain_simplex``;
* shifted orderings, unequal scalar lowers, a strip between two shifted
  order rows and a variable bounded only by the domain cap, against tight
  nested scipy quadrature with hand-derived exact bounds;
* an order cycle, a ``Δt ≡ 0`` pair of rows and a constant row decided
  EMPTY by Θ(0) = 0 give exactly 0; a constant row decided DROP leaves the
  geometry;
* log-domain safety: a term whose ``exp(β·L)`` underflows on its own but
  whose product with the outer variable's factor is O(1) is kept; a
  genuinely astronomical integral is reported as an overflow (never a
  silent inf / NaN);
* rows outside the DBM's scope (a 3-term row) are declined.

Model-free and fast (< 5 s).
"""
import math
import os
import sys

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.integration.time_domain.final_integral as FI  # noqa: E402
from engine.integration.time_domain import dbm_integral as DBM  # noqa: E402


def _expsum(rows, m, alphas, cap=200.0, C=1.0, E=0.0):
    return DBM.integrate_exp_sum(rows, m, [(C, E, alphas)], cap)


def _ref3(alphas, s2_lim, s1_lim, s0_lim):
    """Tight tplquad of exp(Σ α_v s_v) (real and imaginary parts); bounds
    outermost first: s2 in s2_lim, s1 in s1_lim(s2), s0 in s0_lim(s1, s2)."""
    a0, a1, a2 = alphas

    def part(fn):
        val, _err = integrate.tplquad(
            lambda s0, s1, s2: fn(np.exp(a0 * s0 + a1 * s1 + a2 * s2)),
            s2_lim[0], s2_lim[1],
            lambda s2: s1_lim(s2)[0], lambda s2: s1_lim(s2)[1],
            lambda s2, s1: s0_lim(s1, s2)[0],
            lambda s2, s1: s0_lim(s1, s2)[1],
            epsabs=1e-15, epsrel=1e-12)
        return val
    return complex(part(np.real), part(np.imag))


ALPHAS = (0.7 - 0.3j, -0.45 + 0.2j, 0.15 + 0.4j)


# ── the chain simplex ────────────────────────────────────────────────

@pytest.mark.parametrize('alphas, L, U', [
    ((0.3 + 0.1j, -0.2 + 0.05j, 0.45 - 0.2j), -1.5, 2.0),
    ((0.9, 0.4, -1.1), 0.0, 3.0),
    ((0.2 - 0.7j, 0.1 + 0.3j, -0.25 + 0.0j, 0.05 - 0.1j), -2.0, 1.0),
])
def test_plain_chain_equals_chain_simplex(alphas, L, U):
    """L < s_0 < s_1 < ... < s_{N-1} < U: the DBM equals
    ``_exp_over_chain_simplex`` (alphas innermost first)."""
    m = len(alphas)
    rows = [(tuple(1.0 if i == 0 else 0.0 for i in range(m)), -L)]
    for v in range(1, m):
        a = [0.0] * m
        a[v], a[v - 1] = 1.0, -1.0                 # s_v - s_{v-1} > 0
        rows.append((tuple(a), 0.0))
    rows.append((tuple(-1.0 if i == m - 1 else 0.0 for i in range(m)), U))
    res = _expsum(rows, m, alphas)
    ref = FI._exp_over_chain_simplex(list(alphas), L, U)
    assert res.status == DBM.STATUS_OK
    assert abs(res.value - ref) <= 1e-13 * abs(ref), (res.value, ref)
    # an equal lower bound on EVERY variable changes nothing
    res2 = _expsum(rows + [(tuple(1.0 if i == v else 0.0 for i in range(m)),
                            -L) for v in range(1, m)], m, alphas)
    assert abs(res2.value - ref) <= 1e-13 * abs(ref)


def test_coefficient_scaled_rows_are_normalised():
    """``2 s_0 + 2 > 0`` is ``s_0 > -1``; ``3 (s_1 - s_0) > 0`` an order."""
    rows = [((2.0, 0.0), 2.0), ((-3.0, 3.0), 0.0), ((0.0, -1.0), 1.5)]
    unit = [((1.0, 0.0), 1.0), ((-1.0, 1.0), 0.0), ((0.0, -1.0), 1.5)]
    al = (0.4 - 0.2j, -0.3 + 0.1j)
    assert _expsum(rows, 2, al).value == pytest.approx(
        _expsum(unit, 2, al).value, rel=1e-14)


# ── shifted orderings, unequal lowers, the cap ───────────────────────

def test_shifted_chain_against_quadrature():
    """s_0 > -1, s_1 > s_0 + 0.4, s_2 > s_1 - 0.25, s_2 < 1.5."""
    rows = [((1.0, 0.0, 0.0), 1.0),
            ((-1.0, 1.0, 0.0), -0.4),
            ((0.0, -1.0, 1.0), 0.25),
            ((0.0, 0.0, -1.0), 1.5)]
    res = _expsum(rows, 3, ALPHAS)
    ref = _ref3(ALPHAS, (-0.85, 1.5),
                lambda s2: (-0.6, s2 + 0.25),
                lambda s1, s2: (-1.0, s1 - 0.4))
    assert res.status == DBM.STATUS_OK
    assert abs(res.value - ref) <= 1e-11 * abs(ref), (res.value, ref)


def test_unequal_lowers_against_quadrature():
    """s_0 > -1, s_1 > 0.5 (unequal lowers), s_2 > s_0, s_2 > s_1,
    s_2 < 2."""
    rows = [((1.0, 0.0, 0.0), 1.0),
            ((0.0, 1.0, 0.0), -0.5),
            ((-1.0, 0.0, 1.0), 0.0),
            ((0.0, -1.0, 1.0), 0.0),
            ((0.0, 0.0, -1.0), 2.0)]
    res = _expsum(rows, 3, ALPHAS)
    ref = _ref3(ALPHAS, (0.5, 2.0),
                lambda s2: (0.5, s2),
                lambda s1, s2: (-1.0, s2))
    assert abs(res.value - ref) <= 1e-11 * abs(ref), (res.value, ref)


def test_variable_bounded_only_by_the_cap():
    """The P3 shape: 0 < s_0 < 1, s_1 < s_0, s_2 < s_1 and nothing else
    below s_1, s_2: they extend down to -cap (not to s_0's lower 0)."""
    al = (0.1 + 0.05j, 0.3 - 0.1j, 0.25 + 0.2j)   # decays into the past
    rows = [((1.0, 0.0, 0.0), 0.0), ((-1.0, 0.0, 0.0), 1.0),
            ((1.0, -1.0, 0.0), 0.0), ((0.0, 1.0, -1.0), 0.0)]
    for cap in (20.0, 60.0):
        res = _expsum(rows, 3, al, cap=cap)
        # s_0 outermost here: s_0 in (0, 1), s_1 in (-cap, s_0),
        # s_2 in (-cap, s_1)
        a0, a1, a2 = al

        def part(fn):
            return integrate.tplquad(
                lambda s2, s1, s0: fn(np.exp(a0 * s0 + a1 * s1 + a2 * s2)),
                0.0, 1.0, lambda s0: -cap, lambda s0: s0,
                lambda s0, s1: -cap, lambda s0, s1: s1,
                epsabs=1e-15, epsrel=1e-12)[0]
        ref = complex(part(np.real), part(np.imag))
        assert abs(res.value - ref) <= 1e-11 * abs(ref), (cap, res.value, ref)
    # the single-L chain simplex (every variable from 0) is a different,
    # smaller region
    wrong = FI._exp_over_chain_simplex([al[2], al[1], al[0]], 0.0, 1.0)
    assert abs(wrong - res.value) > 0.1 * abs(res.value)


def test_strip_between_shifted_order_rows():
    """s_1 > s_0 + 0.3 and s_0 > s_1 - 0.5 (a strip of width 0.2), in
    -1 < s_0 < 1."""
    rows = [((-1.0, 1.0), -0.3), ((1.0, -1.0), 0.5),
            ((1.0, 0.0), 1.0), ((-1.0, 0.0), 1.0)]
    al = (0.4 + 0.1j, -0.2 + 0.3j)
    res = _expsum(rows, 2, al)

    def part(fn):
        return integrate.dblquad(
            lambda s1, s0: fn(np.exp(al[0] * s0 + al[1] * s1)),
            -1.0, 1.0, lambda s0: s0 + 0.3, lambda s0: s0 + 0.5,
            epsabs=1e-15, epsrel=1e-12)[0]
    ref = complex(part(np.real), part(np.imag))
    assert abs(res.value - ref) <= 1e-12 * abs(ref)


# ── empty and measure-zero regions ───────────────────────────────────

@pytest.mark.parametrize('rows', [
    # an unshifted 3-cycle: s_1 > s_0, s_2 > s_1, s_0 > s_2
    [((-1.0, 1.0, 0.0), 0.0), ((0.0, -1.0, 1.0), 0.0),
     ((1.0, 0.0, -1.0), 0.0)],
    # a shifted cycle whose shifts sum to 0
    [((-1.0, 1.0, 0.0), -0.4), ((0.0, -1.0, 1.0), 0.15),
     ((1.0, 0.0, -1.0), 0.25)],
    # Δt ≡ 0: s_1 - s_0 + 0.3 > 0 and s_0 - s_1 - 0.3 > 0
    [((-1.0, 1.0, 0.0), 0.3), ((1.0, -1.0, 0.0), -0.3),
     ((0.0, 0.0, 1.0), 1.0)],
    # crossed scalar bounds: s_2 > 1 and s_2 < 0.5
    [((0.0, 0.0, 1.0), -1.0), ((0.0, 0.0, -1.0), 0.5)],
])
def test_cycle_and_zero_width_regions_are_exactly_zero(rows):
    res = _expsum(rows, 3, ALPHAS)
    assert res.status == DBM.STATUS_EMPTY and res.value == 0


def _w_edges(lams, n):
    return [FI.EdgeModeSum(ri=-1, pi=-1, delta_coeff=0j,
                           modes=tuple((1.0 + 0j, complex(l)) for l in lams))
            for _ in range(n)]


BASE3 = [((-1.0, 1.0, 0.0), (0.0,), 0.0),           # s_1 > s_0
         ((0.0, -1.0, 1.0), (0.0,), 0.0),           # s_2 > s_1
         ((0.0, 0.0, -1.0), (1.0,), 0.0),           # s_2 < t
         ((1.0, 0.0, 0.0), (0.0,), 2.0)]            # s_0 > -2
BASE3_ROWS = [(a, c0 + e[0] * 1.0) for a, e, c0 in BASE3]   # at t = 1


def test_wrapper_constant_rows_theta0():
    """A Δt ≡ c row: c > 0 is dropped from the geometry (its edge still
    contributes exp(λ c)); c == 0 is Θ(0) = 0 (EMPTY: exactly 0, no
    integration); c < 0 is EMPTY."""
    lam = -0.3 + 0.1j
    free = [1.5]
    base = FI._integrate_subset_dbm(_w_edges([lam], 4), 1.0, BASE3, free, 3)
    for c0, expect in ((0.7, base * complex(np.exp(lam * 0.7))),
                       (0.0, 0.0), (-0.2, 0.0)):
        rows = BASE3 + [((0.0, 0.0, 0.0), (0.0,), c0)]
        FI._reset_runtime_counters()
        v = FI._integrate_subset_dbm(_w_edges([lam], 5), 1.0, rows, free, 3)
        assert v == pytest.approx(expect, rel=1e-14, abs=0), c0
        if c0 <= 0.0:
            assert v == 0
            assert FI._RUNTIME_COUNTERS['dbm_empty'] == 1


def test_thin_strip_is_integrated_not_dropped():
    """No tolerance (the engine's rule): a strip between external times
    1e-12 apart is a genuine region, ∫_t^{t+1e-12} exp(s/2) ds."""
    t = 0.3
    rows = [((1.0,), -t), ((-1.0,), t + 1e-12)]
    res = DBM.integrate_exp_sum(rows, 1, [(1.0, 0.0, (0.5,))], 200.0)
    width = (t + 1e-12) - t                     # the float width
    assert res.status == DBM.STATUS_OK
    assert res.value.real == pytest.approx(width * math.exp(t / 2), rel=1e-3)


def test_too_many_cases_is_declined(monkeypatch):
    monkeypatch.setattr(DBM, 'MAX_CASES', 0)
    res = _expsum(BASE3_ROWS, 3, ALPHAS)
    assert res.status == DBM.STATUS_TOO_MANY_CASES and res.value is None
    FI._reset_runtime_counters()
    assert FI._integrate_subset_dbm(_w_edges([-0.5], 4), 1.0, BASE3,
                                    [1.0], 3) is None
    assert FI._pop_bail_reason() == 'dbm_too_many_cases'
    assert FI._RUNTIME_COUNTERS['dbm_declined_other'] == 1


# ── log-domain safety ────────────────────────────────────────────────

def test_lower_cap_factor_that_underflows_alone_is_kept():
    """∫_{-200}^{0} ds_1 ∫_{-200}^{s_1} ds_0 exp(4 s_0 - 4 s_1)
    = 50 - (1 - e^{-800}) / 16.  The s_0 lower-bound piece carries
    exp(-800), which only the outer factor exp(+800) at s_1 = -200
    brings back to O(1): a plain float product loses it (50.0)."""
    rows = [((1.0, 0.0), 200.0), ((-1.0, 1.0), 0.0), ((0.0, -1.0), 0.0)]
    res = _expsum(rows, 2, (4.0, -4.0), cap=200.0)
    assert res.status == DBM.STATUS_OK
    assert res.value == pytest.approx(50.0 - 1.0 / 16.0, rel=1e-14, abs=0)


def test_large_log_scale_seed_is_safe():
    """A seed exp(E) with Re E = 650 times a factor exp(-4 s) that is
    ~exp(-660) on the region: the product is O(1)."""
    rows = [((1.0,), -165.0), ((-1.0,), 166.0)]          # 165 < s < 166
    res = DBM.integrate_exp_sum(rows, 1, [(1.0, 650.0, (-4.0,))], 200.0)
    ref = (math.exp(650 - 4 * 165) - math.exp(650 - 4 * 166)) / 4.0
    assert res.status == DBM.STATUS_OK
    assert res.value.real == pytest.approx(ref, rel=1e-12)


def test_astronomical_integral_is_an_overflow_not_inf():
    """exp(-5 s_0) on s_0 > -200 is ~exp(1000): reported as an overflow,
    and the wrapper declines (the caller goes on to the fallback)."""
    rows = [((-1.0, 1.0, 0.0), 0.0), ((0.0, -1.0, 1.0), 0.0),
            ((0.0, 0.0, -1.0), 0.0)]
    res = _expsum(rows, 3, (-5.0, 0.0, 0.0))
    assert res.status == DBM.STATUS_OVERFLOW and res.value is None
    edges = [FI.EdgeModeSum(ri=-1, pi=-1, delta_coeff=0j,
                            modes=((1.0 + 0j, 5.0 + 0j),))]
    cdata = [((-1.0, 0.0, 0.0), (0.0,), 0.0),    # Δt = -s_0 (λ = 5)
             ((-1.0, 1.0, 0.0), (0.0,), 0.0),
             ((0.0, -1.0, 1.0), (0.0,), 0.0),
             ((0.0, 0.0, -1.0), (0.0,), 0.0)]
    edges = edges + _w_edges([0.0], 3)
    FI._reset_runtime_counters()
    assert FI._integrate_subset_dbm(edges, 1.0, cdata, [0.0], 3) is None
    assert FI._pop_bail_reason() == 'dbm_overflow'
    assert FI._RUNTIME_COUNTERS['dbm_declined_overflow'] == 1


# ── scope ────────────────────────────────────────────────────────────

def test_three_term_row_is_declined():
    rows = BASE3 + [((1.0, 1.0, -1.0), (0.0,), 0.5)]   # not a difference row
    assert DBM.difference_bounds([(a, c) for a, _e, c in rows], 3, 200.0) \
        is None
    FI._reset_runtime_counters()
    v = FI._integrate_subset_dbm(_w_edges([-0.5], 5), 1.0, rows, [1.0], 3)
    assert v is None and FI._pop_bail_reason() == 'dbm_not_difference_rows'
    assert FI._RUNTIME_COUNTERS['dbm_declined_rows'] == 1


def test_wrapper_matches_the_poset_path_where_that_is_valid():
    """A chain with a shared, inherited lower bound: the poset path's
    single-L chain simplex is exact there, and the DBM agrees (the box is
    not active)."""
    modes = [FI.EdgeModeSum(ri=-1, pi=-1, delta_coeff=0j,
                            modes=((0.6 - 0.1j, -0.35 + 0.2j),
                                   (0.2 + 0.3j, -0.9 + 0.0j)))
             for _ in range(4)]
    free = [2.5]
    poset = FI._integrate_nd_polytope_poset_modesum(
        modes, 0.8 + 0.1j, BASE3, free, 3)
    plan = FI._build_modesum_plan(modes, BASE3, 3, 1)
    for kw in ({}, {'plan': plan}):
        dbm = FI._integrate_subset_dbm(modes, 0.8 + 0.1j, BASE3, free, 3,
                                       **kw)
        assert abs(dbm - poset) <= 1e-13 * abs(poset), (kw, dbm, poset)

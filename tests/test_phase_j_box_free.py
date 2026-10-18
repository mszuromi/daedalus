"""M7 (F5 + F6): box-free exact Phase J on the divided-difference kernels.

Kernel level (``engine/integration/time_domain/boxfree.py``), each against a
reference that shares no code with it:

* the m = 1 interval against mpmath closed forms at 40 digits;
* the 2D region (exact vertices + recession rays, ideal-vertex fan) against the
  divergence-theorem SWEEP formula of the M7 plan (§3.1 L4), implemented here
  in 50-digit mpmath, and against brute-force mpmath quadrature on a box for
  bounded regions;
* the open-top chain (mirror image) against mpmath.

Engine level (``final_integral``; since audit W7 the box-free routes are the
only ones, their switches ``USE_PHASE_J_DD_KERNELS`` / ``SEMIINF_CHAIN`` /
``UNBOUNDED_2D`` retired): the polygon / interval / poset / DBM paths in
both the plan and the no-plan branch against references, the one-sided
final-exponent check, the finite lower bound ``PHASE_J_T_MIN``, and
end-to-end models against exact oracles.  Where a test used to compute the
retired boxed route live as a "before" value, that value is a frozen literal
(captured at the last commit with the boxed routes, before audit W7).
"""
import cmath
import itertools
import math
import random

import mpmath as mp
import numpy as np
import pytest

from engine.integration.time_domain import boxfree as bf


# ═══════════════════════════════════════════════════════════════════════
# Independent references
# ═══════════════════════════════════════════════════════════════════════

def _mp_interval(alpha, gamma, L, U, dps=40):
    with mp.workdps(dps):
        a = mp.mpc(alpha)
        g = mp.mpc(gamma)
        if L == -math.inf:
            return complex(mp.exp(a * U + g) / a)
        if U == math.inf:
            return complex(-mp.exp(a * L + g) / a)
        if a == 0:
            return complex((mp.mpf(U) - mp.mpf(L)) * mp.exp(g))
        return complex((mp.exp(a * U + g) - mp.exp(a * L + g)) / a)


def _sweep_mp(cons, alpha, gamma=0j, dps=50):
    """∫ exp(α·s + γ) over {a·s + c > 0} by the divergence theorem with a
    direction d in the recession cone (every facet weight >= 0), at
    ``dps`` digits.  ``cons``: ((a0, a1), c) pairs; the region must be
    pointed and the integral convergent."""
    with mp.workdps(dps):
        rows = []
        for (a, c) in cons:
            n = mp.sqrt(mp.mpf(a[0]) ** 2 + mp.mpf(a[1]) ** 2)
            r = ((mp.mpf(a[0]) / n, mp.mpf(a[1]) / n), mp.mpf(c) / n)
            if not any(abs(r[0][0] - q[0][0]) < 1e-30
                       and abs(r[0][1] - q[0][1]) < 1e-30
                       and abs(r[1] - q[1]) < 1e-30 for q in rows):
                rows.append(r)
        al = (mp.mpc(alpha[0]), mp.mpc(alpha[1]))
        A = [a for a, _ in rows]
        tol = mp.mpf('1e-30')
        rays = []
        for a in A:
            for u in ((-a[1], a[0]), (a[1], -a[0])):
                if all(b[0] * u[0] + b[1] * u[1] >= -tol for b in A) and not any(
                        abs(u[0] - r[0]) < 1e-20 and abs(u[1] - r[1]) < 1e-20
                        for r in rays):
                    rays.append(u)
        d = ((sum(r[0] for r in rays), sum(r[1] for r in rays)) if rays
             else (mp.mpf(1), mp.mpf('0.37')))
        ad = al[0] * d[0] + al[1] * d[1]
        total = mp.mpc(0)
        for i, (a, c) in enumerate(rows):
            w = d[0] * a[0] + d[1] * a[1]
            if abs(w) < tol:
                continue
            p0 = (-c * a[0], -c * a[1])
            u = (-a[1], a[0])
            lo, hi = -mp.inf, mp.inf
            empty = False
            for j, (b, cb) in enumerate(rows):
                if j == i:
                    continue
                bu = b[0] * u[0] + b[1] * u[1]
                rhs = -(b[0] * p0[0] + b[1] * p0[1] + cb)
                if abs(bu) < tol:
                    if rhs > tol:
                        empty = True
                        break
                    continue
                t = rhs / bu
                if bu > 0:
                    lo = max(lo, t)
                else:
                    hi = min(hi, t)
            if empty or lo >= hi:
                continue
            kappa = al[0] * u[0] + al[1] * u[1]
            e0 = al[0] * p0[0] + al[1] * p0[1]
            if hi == mp.inf:
                F = mp.exp(e0 + kappa * lo) / (-kappa)
            elif lo == -mp.inf:
                F = mp.exp(e0 + kappa * hi) / kappa
            else:
                F = (mp.exp(e0 + kappa * hi) - mp.exp(e0 + kappa * lo)) / kappa
            total += w * F
        return complex(total / (-ad) * mp.exp(mp.mpc(gamma)))


def _box_quad_mp(cons, alpha, gamma, B, dps=30):
    """Brute-force mpmath quadrature of exp(α·s + γ) over {a·s + c > 0}
    ∩ [-B, B]^2: inner s1 in closed form, outer s0 by mp.quad with
    breakpoints at every vertex abscissa."""
    with mp.workdps(dps):
        a1c = mp.mpc(alpha[1])

        def inner(x):
            lo, hi = -B, B
            if not -B < x < B:
                return mp.mpc(0)
            for (a, c) in cons:
                if a[1] == 0:
                    if a[0] * x + c <= 0:
                        return mp.mpc(0)
                    continue
                t = -(a[0] * x + c) / a[1]
                if a[1] > 0:
                    lo = max(lo, t)
                else:
                    hi = min(hi, t)
            if lo >= hi:
                return mp.mpc(0)
            return (mp.exp(mp.mpc(alpha[0]) * x + mp.mpc(gamma))
                    * (mp.exp(a1c * hi) - mp.exp(a1c * lo)) / a1c)
        pts = {-B, B}
        for (a, c) in cons:
            for (b, d) in cons:
                det = a[0] * b[1] - a[1] * b[0]
                if det != 0:
                    x = (-c * b[1] + d * a[1]) / det
                    if -B < x < B:
                        pts.add(x)
            if a[1] == 0 and a[0] != 0 and -B < -c / a[0] < B:
                pts.add(-c / a[0])
        return complex(mp.quad(inner, sorted(pts)))


_NORMALS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1), (1, 1),
            (-1, -1), (2, -1), (1, -3)]


def _random_system(rng, n_lo=2, n_hi=6, integer_c=False):
    cons = []
    for _ in range(rng.randint(n_lo, n_hi)):
        a = rng.choice(_NORMALS)
        c = (float(rng.choice([-2, -1, 0, 1, 2, 3])) if integer_c
             else rng.uniform(-3.0, 3.0))
        cons.append((a, c))
    return cons


def _decaying_alpha(rng, lo=0.05, hi=12.0):
    """Exponent coefficients that decay toward the past (Re > 0)."""
    return (complex(rng.uniform(lo, hi), rng.uniform(-3, 3)),
            complex(rng.uniform(lo, hi), rng.uniform(-3, 3)))


# ═══════════════════════════════════════════════════════════════════════
# Kernel level: interval
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('alpha,gamma,L,U', [
    (0.7 - 0.3j, 0.2 + 0.1j, -3.0, 2.5),
    (-12.0 + 4j, 1.0, -40.0, 3.0),
    (1e-9 + 1e-10j, -0.5j, -7.0, 11.0),         # tiny α: no 1/α cancellation
    (0j, 0.3, -1.0, 1.0),                        # α exactly 0
    (2.5 + 1j, -3.0, -math.inf, 4.0),
    (-0.15 + 2j, 0.0, -6.0, math.inf),
    (36.0 - 5j, 500.0, -200.0, -13.0),           # large exponents, finite value
])
def test_interval_matches_mpmath(alpha, gamma, L, U):
    st, v = bf.interval(alpha, gamma, L, U)
    assert st == bf.OK
    ref = _mp_interval(alpha, gamma, L, U)
    assert abs(v - ref) <= 1e-13 * abs(ref)


def test_interval_small_alpha_beats_the_legacy_difference_quotient():
    """(e^{αU} - e^{αL})/α cancels for |α|(U-L) << 1; the divided difference
    does not."""
    alpha, gamma, L, U = 3e-12 + 1e-12j, 0j, -2.0, 5.0
    st, v = bf.interval(alpha, gamma, L, U)
    ref = _mp_interval(alpha, gamma, L, U)
    legacy = (cmath.exp(alpha * U + gamma) - cmath.exp(alpha * L + gamma)) / alpha
    assert st == bf.OK and abs(v - ref) <= 1e-15 * abs(ref)
    assert abs(legacy - ref) > 1e-8 * abs(ref)        # the old form loses digits


@pytest.mark.parametrize('alpha,L,U', [
    (-1.0, -math.inf, 3.0),          # grows toward -inf
    (0j, -math.inf, 3.0),            # flat
    (0.5, 1.0, math.inf),            # grows toward +inf
    (0.5, -math.inf, math.inf),      # a line
])
def test_interval_divergent_is_reported_never_a_number(alpha, L, U):
    st, v = bf.interval(alpha, 0j, L, U)
    assert st == bf.DIVERGENT and v is None


def test_interval_overflow_is_reported_on_the_final_exponent_only():
    # the integral itself is ~e^800: overflow
    st, v = bf.interval(1.0, 800.0, -1.0, 0.0)
    assert st == bf.OVERFLOW and v is None
    # a huge γ cancelled by the integration range: a finite value
    st, v = bf.interval(10.0, 2000.0, -260.0, -199.0)
    ref = _mp_interval(10.0, 2000.0, -260.0, -199.0)
    assert st == bf.OK and abs(v - ref) <= 1e-12 * abs(ref)


def test_interval_empty_range_is_exactly_zero():
    assert bf.interval(1.0, 0.0, 2.0, 2.0) == (bf.OK, 0j)
    assert bf.interval(1.0, 0.0, 3.0, 2.0) == (bf.OK, 0j)


# ═══════════════════════════════════════════════════════════════════════
# Kernel level: 2D regions
# ═══════════════════════════════════════════════════════════════════════

def _region(cons):
    return bf.region_2d([(a[0], a[1], c) for (a, c) in cons])


def test_region_kinds_on_hand_made_systems():
    # bounded triangle
    r = _region([((1, 0), 0.0), ((0, 1), 0.0), ((-1, -1), 1.0)])
    assert r.kind == 'bounded' and len(r.vertices) == 3
    # quadrant toward the past: one vertex, two rays
    r = _region([((-1, 0), 1.0), ((0, -1), 2.0)])
    assert r.kind == 'unbounded' and len(r.vertices) == 1
    assert sorted(r.rays) == [(-1.0, 0.0), (0.0, -1.0)]
    # half-strip: two vertices, one ray direction
    r = _region([((1, 0), 0.0), ((-1, 0), 1.0), ((0, -1), 0.0)])
    assert r.kind == 'unbounded' and len(r.rays) == 1
    # a strip (contains a line), a half-plane, nothing
    assert _region([((1, 0), 0.0), ((-1, 0), 1.0)]).kind == 'lineality'
    assert _region([((1, -1), 0.5)]).kind == 'lineality'
    assert bf.region_2d([]).kind == 'lineality'
    # empty: crossed bounds; the 2-cycle s1 > s0, s0 > s1 (a segment);
    # a single point; a ray
    assert _region([((1, 0), -2.0), ((-1, 0), 1.0), ((0, 1), 0.0)]).kind == 'empty'
    assert _region([((-1, 1), 0.0), ((1, -1), 0.0), ((0, -1), 1.0)]).kind == 'empty'
    assert _region([((1, 0), 0.0), ((-1, 0), 0.0), ((0, 1), 0.0),
                    ((0, -1), 0.0)]).kind == 'empty'
    assert _region([((1, -1), 0.0), ((-1, 1), 0.0), ((0, -1), 0.0)]).kind == 'empty'
    # three boundary lines through one point, no opposite pair: a point
    assert _region([((-1, -1), 0.0), ((1, 0), 0.0), ((0, 1), 0.0)]).kind == 'empty'


@pytest.mark.parametrize('seed', [3, 11])
def test_region_integral_matches_the_sweep_oracle(seed):
    """Unbounded regions: the ideal-vertex fan against the 50-digit sweep
    formula (an independent algorithm: facet sums, no triangulation)."""
    rng = random.Random(seed)
    n = 0
    worst = 0.0
    for _ in range(400):
        cons = _random_system(rng)
        alpha = _decaying_alpha(rng)
        gamma = complex(rng.uniform(-2, 2), rng.uniform(-1, 1))
        reg = _region(cons)
        st, v = bf.region_integral(reg, alpha[0], alpha[1], gamma)
        if reg.kind != 'unbounded' or st != bf.OK:
            continue
        ref = _sweep_mp(cons, alpha, gamma)
        worst = max(worst, abs(v - ref) / abs(ref))
        n += 1
    assert n > 50
    assert worst <= 1e-12


def test_region_integral_on_degenerate_systems():
    """Small-integer constants: triple intersections, duplicate and opposite
    rows, zero-area regions.  Bounded and empty ones against brute-force
    mpmath on a box, unbounded ones against the sweep oracle."""
    rng = random.Random(5)
    n = 0
    worst = 0.0
    for _ in range(300):
        cons = _random_system(rng, integer_c=True)
        alpha = _decaying_alpha(rng, 0.2, 5.0)
        gamma = complex(rng.uniform(-2, 2), rng.uniform(-1, 1))
        reg = _region(cons)
        st, v = bf.region_integral(reg, alpha[0], alpha[1], gamma)
        if st == bf.DIVERGENT:
            assert reg.kind in ('unbounded', 'lineality')
            continue
        if reg.kind in ('bounded', 'empty'):
            ref = _box_quad_mp(cons, alpha, gamma, 60)
        else:
            ref = _sweep_mp(cons, alpha, gamma)
        n += 1
        if reg.kind == 'empty':
            assert v == 0 and abs(ref) < 1e-25
            continue
        worst = max(worst, abs(v - ref) / abs(ref))
    assert n > 100
    assert worst <= 1e-12


def test_region_divergent_rays_and_overflow():
    quadrant = _region([((-1, 0), 1.0), ((0, -1), 2.0)])   # rays (-1,0), (0,-1)
    # Re α0 < 0: grows along (-1, 0)
    assert bf.region_integral(quadrant, -0.1, 1.0, 0j) == (bf.DIVERGENT, None)
    assert bf.region_integral(quadrant, 0j, 1.0, 0j) == (bf.DIVERGENT, None)
    lin = _region([((1, 0), 0.0), ((-1, 0), 1.0)])
    assert bf.region_integral(lin, 1.0, 1.0, 0j) == (bf.DIVERGENT, None)
    st, v = bf.region_integral(quadrant, 1.0, 1.0, 800.0)   # the value is e^803
    assert st == bf.OVERFLOW and v is None


def test_region_far_from_the_origin_has_no_bilateral_bail():
    """A bounded triangle 1000 time units in the past: every vertex exponent
    is ~-12000 (the legacy bilateral guard bails at |Re| > 600); the value is
    representable once γ brings it back."""
    cons = [((1, 0), 1000.0), ((0, 1), 1000.0), ((-1, -1), -1999.0)]
    alpha = (6.0 + 1j, 6.0 - 2j)
    reg = _region(cons)
    st, v = bf.region_integral(reg, alpha[0], alpha[1], 12000.0)
    ref = _box_quad_mp(cons, alpha, 12000.0, 1100, dps=40)
    assert st == bf.OK and abs(v - ref) <= 1e-11 * abs(ref)


# ═══════════════════════════════════════════════════════════════════════
# Kernel level: chains with an open top (mirror image)
# ═══════════════════════════════════════════════════════════════════════

def _mp_chain(alphas, L, U, dps=40):
    """∫_{L<s_1<…<s_m<U} exp(Σ α_k s_k) by iterated closed forms in mpmath
    (exact for distinct prefix sums; L = -inf or U = +inf allowed)."""
    with mp.workdps(dps):
        al = [mp.mpc(a) for a in alphas]
        m = len(al)
        if U == math.inf:
            # mirror: t_j = -s_{m+1-j}
            return _mp_chain([-a for a in reversed(alphas)], -math.inf, -L, dps)
        # integrate s_1 first: F_k(x) = ∫_{L<s_1<..<s_k<x}; F_k = Σ c e^{p x}
        terms = [(mp.mpc(1), mp.mpc(0))]          # F_0 = 1
        for k in range(m):
            new = []
            for (c, p) in terms:
                q = p + al[k]
                new.append((c / q, q))
                if L != -math.inf:
                    new.append((-c * mp.exp(q * L) / q, mp.mpc(0)))
            # merge constants
            terms = new
        return complex(sum(c * mp.exp(p * U) for (c, p) in terms))


@pytest.mark.parametrize('alphas,L', [
    ([-1.3 + 0.2j, -0.4, -2.1 - 1j], 0.5),
    ([-0.2, -0.35 + 1j, -1.0, -0.6], -2.0),
    ([-5.0 + 3j, -0.05, -0.7], 4.0),
])
def test_open_top_chain_is_the_mirror_of_a_semi_infinite_chain(alphas, L):
    st, ls, mant = bf.chain_log(alphas, L, {}, math.inf)
    assert st == bf.OK
    v = mant * cmath.exp(ls)
    ref = _mp_chain(alphas, L, math.inf)
    assert abs(v - ref) <= 1e-13 * abs(ref)


def test_open_top_chain_with_intermediate_uppers_is_unsupported():
    st, _ls, _m = bf.chain_log([-1.0, -2.0, -0.5], 0.0, {0: 3.0}, math.inf)
    assert st == bf.UNSUPPORTED
    # a None upper is absent, as in the legacy convention
    st, _ls, _m = bf.chain_log([-1.0, -2.0, -0.5], 0.0, {0: None}, math.inf)
    assert st == bf.OK


def test_chain_both_sides_open_is_divergent():
    st, _ls, _m = bf.chain_log([1.0, -1.0], -math.inf, {}, math.inf)
    assert st == bf.DIVERGENT


def test_exp_times_has_no_intermediate_overflow():
    assert bf.exp_times(800.0 + 0j, 1e-200)[0] == bf.OK
    st, v = bf.exp_times(800.0 + 1j, 1e-200)
    ref = complex(mp.mpf('1e-200') * mp.exp(mp.mpc(800.0, 1.0)))
    assert abs(v - ref) <= 1e-12 * abs(ref)
    assert bf.exp_times(705.0 + 0j, 1.0) == (bf.OVERFLOW, None)
    assert bf.exp_times(-2000.0 + 0j, 1.0) == (bf.OK, 0j)
    assert bf.exp_times(3.0 + 0j, 0j) == (bf.OK, 0j)


# ═══════════════════════════════════════════════════════════════════════
# Engine level (final_integral with the M7 flags)
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def FI():
    pytest.importorskip('sage.all')
    import engine.integration.time_domain.final_integral as fi
    return fi


def _set_flags(monkeypatch, fi, t_min=-math.inf):
    monkeypatch.setattr(fi, 'PHASE_J_T_MIN', t_min)


def _ems(fi, modes):
    return fi.EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=tuple(modes))


def _tuples(modes):
    for combo in itertools.product(*modes):
        C = 1.0 + 0j
        for (c, _l) in combo:
            C *= c
        yield C, tuple(l for (_c, l) in combo)


def _poly_ref(rows, modes, free, pref=1.0):
    """Σ_tuples pref·C ∫ exp(Σ λ_e Δt_e) over the m=2 region of ``rows``
    ((a_int, a_ext, c0) per edge), by the 50-digit sweep oracle."""
    cons = []
    for (a, e, c0) in rows:
        cons.append(((a[0], a[1]), c0 + sum(x * t for x, t in zip(e, free))))
    total = 0j
    for C, lams in _tuples(modes):
        al = (sum(l * r[0][0] for l, r in zip(lams, rows)),
              sum(l * r[0][1] for l, r in zip(lams, rows)))
        g = sum(l * (r[2] + sum(x * t for x, t in zip(r[1], free)))
                for l, r in zip(lams, rows))
        total += pref * C * _sweep_mp(cons, al, g)
    return total


# A wedge toward the past: s1 < s0 < t (vertex (t, t), rays (0,-1), (-1,-1)).
_WEDGE_ROWS = [((-1.0, 0.0), (1.0,), 0.0),     # t - s0
               ((1.0, -1.0), (0.0,), 0.0),     # s0 - s1
               ((0.0, -1.0), (1.0,), 0.3)]     # t + 0.3 - s1 (implied)
_WEDGE_FAST = [[(1.0, -1.3 + 0.4j), (0.5, -2.0)],
               [(0.8 - 0.2j, -0.9)],
               [(1.0, -0.6 + 1j)]]
_WEDGE_SLOW = [[(1.0, -0.03)], [(0.7, -1.0 + 0.5j)], [(1.0, -0.03 + 0.2j)]]


def _polygon(fi, rows, modes, free, plan=False):
    ems = [_ems(fi, md) for md in modes]
    if plan:
        p = fi._build_modesum_plan(ems, rows, 2, len(free))
        return fi._integrate_2d_polygon_modesum(
            ems, 1.0, rows, list(free), pole_tuples=p['pole_tuples'], plan=p)
    return fi._integrate_2d_polygon_modesum(ems, 1.0, rows, list(free))


@pytest.mark.parametrize('plan', [False, True])
def test_polygon_routes_agree_on_a_fast_wedge(FI, plan):
    free = (0.7,)
    ref = _poly_ref(_WEDGE_ROWS, _WEDGE_FAST, free)
    v = _polygon(FI, _WEDGE_ROWS, _WEDGE_FAST, free, plan)
    assert abs(v - ref) <= 1e-12 * abs(ref)


# The value of the slow wedge on the retired boxed DD route (±200 box,
# ``UNBOUNDED_2D='box'``; frozen at audit W7).
_SLOW_WEDGE_BOXED = (-0.953395515862667 + 2.4916442533467866j)


def test_polygon_box_discriminator_on_a_slow_wedge(FI, monkeypatch):
    """Decay rate 0.06 along a ray: the ±200 box truncated visibly
    (e^{-12}; frozen value of the retired boxed route); the exact route
    does not."""
    free = (0.7,)
    ref = _poly_ref(_WEDGE_ROWS, _WEDGE_SLOW, free)
    _set_flags(monkeypatch, FI)
    exact = _polygon(FI, _WEDGE_ROWS, _WEDGE_SLOW, free)
    assert abs(exact - ref) <= 1e-12 * abs(ref)
    assert abs(_SLOW_WEDGE_BOXED - ref) > 1e-7 * abs(ref)
    FI._reset_runtime_counters()
    _set_flags(monkeypatch, FI)
    _polygon(FI, _WEDGE_ROWS, _WEDGE_SLOW, free, plan=True)
    assert FI._RUNTIME_COUNTERS['polygon_exact_unbounded'] == 1


def test_polygon_dd_needs_no_bilateral_guard(FI, monkeypatch):
    """Fast modes: the retired legacy triangle route bailed here on its
    guard (|Re p| > 600 at the far corners of the ±200 box:
    'polygon_triangle_guard'); the DD route returns the value."""
    free = (0.0,)
    modes = [[(1.0, -12.0)], [(1.0, -12.0 + 2j)], [(1.0, -12.0)]]
    ref = _poly_ref(_WEDGE_ROWS, modes, free)
    _set_flags(monkeypatch, FI)
    v = _polygon(FI, _WEDGE_ROWS, modes, free)
    assert abs(v - ref) <= 1e-12 * abs(ref)


def test_polygon_dd_overflow_and_divergence_are_explicit(FI, monkeypatch):
    free = (0.0,)
    # a growing mode on a large bounded region: the integral itself is ~e^800
    rows = [((1.0, 0.0), (0.0,), 100.0), ((0.0, 1.0), (0.0,), 100.0),
            ((-1.0, 0.0), (1.0,), 0.0), ((0.0, -1.0), (1.0,), 0.0)]
    modes = [[(1.0, 4.0)], [(1.0, 4.0)], [(1.0, 0.0)], [(1.0, 0.0)]]
    _set_flags(monkeypatch, FI)
    FI._reset_runtime_counters()
    FI._pop_bail_reason()
    assert _polygon(FI, rows, modes, free) is None
    assert FI._pop_bail_reason() == 'polygon_dd_overflow'
    assert FI._RUNTIME_COUNTERS['dd_overflow'] == 1
    # growing along the ray (-1, -1) of the exact region: Δt1 and Δt3 grow
    # there, at the net rate 1.5 - 1.0 > 0
    modes = [[(1.0, 1.5)], [(1.0, -1.0)], [(1.0, -1.0)]]
    _set_flags(monkeypatch, FI)
    FI._reset_runtime_counters()
    assert _polygon(FI, _WEDGE_ROWS, modes, (0.0,)) is None
    assert FI._pop_bail_reason() == 'polygon_divergent'
    assert FI._RUNTIME_COUNTERS['divergent_subsets'] == 1


def test_polygon_constant_rows_follow_theta0_on_the_exact_route(FI,
                                                                monkeypatch):
    """A constant row c_eff = 0 (Θ(0) = 0) empties the region; c_eff > 0
    drops it from the geometry but keeps exp(λ c) in γ."""
    _set_flags(monkeypatch, FI)
    free = (0.7,)
    zero = ((0.0, 0.0), (0.0,), 0.0)
    assert _polygon(FI, _WEDGE_ROWS + [zero], _WEDGE_FAST + [[(1.0, -1.0)]],
                    free) == 0
    shifted = ((0.0, 0.0), (0.0,), 0.3)
    v = _polygon(FI, _WEDGE_ROWS + [shifted],
                 _WEDGE_FAST + [[(2.0, -1.5)]], free)
    base = _polygon(FI, _WEDGE_ROWS, _WEDGE_FAST, free)
    assert abs(v - 2.0 * cmath.exp(-1.5 * 0.3) * base) <= 1e-13 * abs(v)


# ── m = 1 ─────────────────────────────────────────────────────────────
_INTERVAL_ROWS = [((-1.0,), (1.0,), 0.0),     # t - s > 0
                  ((1.0,), (-1.0,), 2.0)]     # s - (t - 2) > 0


def _interval(fi, rows, modes, free, plan=False):
    ems = [_ems(fi, md) for md in modes]
    p = fi._build_modesum_plan(ems, rows, 1, len(free)) if plan else None
    return fi._integrate_1d_polytope_modesum(
        ems, 0.7 - 0.1j, rows, list(free),
        pole_tuples=None if p is None else p['pole_tuples'], plan=p)


# The small-α interval on the retired legacy (difference-quotient) route
# (``USE_PHASE_J_DD_KERNELS=0``; frozen at audit W7, plan and no plan).
_SMALL_ALPHA_LEGACY = (0.18945542355945375 - 0.027065060508493395j)


@pytest.mark.parametrize('plan', [False, True])
def test_interval_routes_agree_and_the_small_alpha_case(FI, plan):
    free = (0.4,)
    modes = [[(1.0, -1.3 + 0.2j), (0.4, -0.5)], [(0.9, -0.8 + 1j)]]
    v = _interval(FI, _INTERVAL_ROWS, modes, free, plan)
    ref = 0j
    for C, (l1, l2) in _tuples(modes):
        ref += (0.7 - 0.1j) * C * _mp_interval(
            -l1 + l2, l1 * 0.4 + l2 * (-0.4 + 2.0), 0.4 - 2.0, 0.4)
    assert abs(v - ref) <= 1e-13 * abs(ref)
    # α = -λ1 + λ2 = 1e-12: the legacy difference quotient cancelled
    modes = [[(1.0, -1.0)], [(1.0, -1.0 + 1e-12)]]
    ref = (0.7 - 0.1j) * _mp_interval(1e-12, -0.4 + (-1.0 + 1e-12) * 1.6,
                                      -1.6, 0.4)
    v_dd = _interval(FI, _INTERVAL_ROWS, modes, free, plan)
    assert abs(v_dd - ref) <= 1e-14 * abs(ref)
    assert abs(_SMALL_ALPHA_LEGACY - ref) > 1e-10 * abs(ref)


def test_semi_infinite_interval_routes_agree_with_the_exact_integral(
        FI, monkeypatch):
    """The semi-infinite side (only the t - s row): -inf < s < t (ported
    from the retired grouped m = 1 test)."""
    free = (0.4,)
    rows1 = _INTERVAL_ROWS[:1]
    modes = [[(1.0, -0.7 + 0.3j), (0.5, -2.0)]]
    ref = (0.7 - 0.1j) * sum(
        C * _mp_interval(-lam, lam * 0.4, -math.inf, 0.4)
        for C, lam in modes[0])
    _set_flags(monkeypatch, FI)
    v = _interval(FI, rows1, modes, free)
    assert abs(v - ref) <= 1e-14 * abs(ref), (v, ref)


# ── m >= 3 ────────────────────────────────────────────────────────────
# The chain s0 < s1 < s2 < t: Δt = s1-s0, s2-s1, t-s2 are independent
# positive increments, so the semi-infinite integral is Π C_e / (-λ_e).
_CHAIN_ROWS = [((-1.0, 1.0, 0.0), (0.0,), 0.0),
               ((0.0, -1.0, 1.0), (0.0,), 0.0),
               ((0.0, 0.0, -1.0), (1.0,), 0.0)]


def _poset(fi, rows, modes, free, plan=False, m=3):
    ems = [_ems(fi, md) for md in modes]
    if plan:
        p = fi._build_modesum_plan(ems, rows, m, len(free))
        return fi._integrate_nd_polytope_poset_modesum(
            ems, 1.0, rows, list(free), m, pole_tuples=p['pole_tuples'],
            plan=p)
    return fi._integrate_nd_polytope_poset_modesum(ems, 1.0, rows,
                                                   list(free), m)


def _increments_exact(modes):
    tot = 0j
    for C, lams in _tuples(modes):
        tot += C * np.prod([1.0 / (-l) for l in lams])
    return tot


@pytest.mark.parametrize('plan', [False, True])
def test_semi_infinite_chain_is_exact(FI, monkeypatch, plan):
    free = (0.7,)
    fast = [[(1.0, -2.0 + 1j), (0.3, -1.1)], [(0.8, -1.5)], [(1.0, -3.0)]]
    ref = _increments_exact(fast)
    _set_flags(monkeypatch, FI)
    v = _poset(FI, _CHAIN_ROWS, fast, free, plan)
    assert abs(v - ref) <= 1e-13 * abs(ref)
    # slow: the retired margin-50 lower bound truncated (e^{-0.05·50} ~ 8 %;
    # frozen value of the boxed DD route, ``SEMIINF_CHAIN=0``, audit W7)
    slow = [[(1.0, -0.05)], [(0.8, -1.5)], [(1.0, -3.0)]]
    ref = _increments_exact(slow)
    FI._reset_runtime_counters()
    v = _poset(FI, _CHAIN_ROWS, slow, free, plan)
    assert abs(v - ref) <= 1e-14 * abs(ref)
    assert FI._RUNTIME_COUNTERS['poset_semiinf_chains'] == 1
    margin50 = 3.259076886184265 + 0j
    assert abs(margin50 - ref) > 1e-2 * abs(ref)


def test_semi_infinite_chain_far_lower_limit(FI, monkeypatch):
    """L = -inf against the chain on the FINITE lower bound L = -60 (the
    kept finite-L closed form, here through PHASE_J_T_MIN), whose truncation
    e^{-κ·60}·(...) is below rounding for κ >= 1, and against mpmath."""
    free = (0.7,)
    modes = [[(1.0, -1.0 + 1j)], [(0.8, -1.5)], [(1.0, -3.0 - 0.5j)]]
    _set_flags(monkeypatch, FI)
    v_inf = _poset(FI, _CHAIN_ROWS, modes, free)
    _set_flags(monkeypatch, FI, t_min=-60.0)
    v_60 = _poset(FI, _CHAIN_ROWS, modes, free)
    ref = _increments_exact(modes)
    assert abs(v_inf - ref) <= 1e-14 * abs(ref)
    assert abs(v_60 - v_inf) <= 1e-13 * abs(ref)


# The finite-L routes retired in audit W7 on the slow chain below, their
# lower bound earliest_ext - POSET_PHYSICAL_MARGIN (frozen; plan and no plan
# gave the same values): the legacy chain-simplex kernels and the DD kernels
# on the legacy box.
_MARGIN50_LEGACY = (5.879734548079079 + 14.986364525794217j)
_MARGIN50_DD_BOX = (5.879734548079085 + 14.986364525794233j)


@pytest.mark.parametrize('plan', [False, True])
def test_finite_lower_bound_reproduces_the_legacy_margin(FI, monkeypatch,
                                                         plan):
    """PHASE_J_T_MIN = earliest_ext - POSET_PHYSICAL_MARGIN on the box-free
    route gives the value of the retired finite-L routes, whose lower bound
    was exactly that (bit for bit the DD-kernel one when it was live)."""
    free = (0.7,)
    slow = [[(1.0, -0.05 + 0.1j)], [(0.8, -1.5)], [(1.0, -0.3)]]
    L = min(0.0, free[0]) - FI.POSET_PHYSICAL_MARGIN
    _set_flags(monkeypatch, FI, t_min=L)
    v_tmin = _poset(FI, _CHAIN_ROWS, slow, free, plan)
    assert abs(v_tmin - _MARGIN50_DD_BOX) <= 1e-14 * abs(_MARGIN50_DD_BOX)
    assert abs(v_tmin - _MARGIN50_LEGACY) <= 1e-12 * abs(_MARGIN50_LEGACY)


def test_open_top_chain_in_the_poset(FI, monkeypatch):
    """t < s0 < s1 < s2 with no upper: the chain top is +inf on the box-free
    route (mirror form); the increments are again independent."""
    rows = [((1.0, 0.0, 0.0), (-1.0,), 0.0),       # s0 - t
            ((-1.0, 1.0, 0.0), (0.0,), 0.0),
            ((0.0, -1.0, 1.0), (0.0,), 0.0)]
    modes = [[(1.0, -2.0)], [(0.8, -0.7 + 1j)], [(1.0, -0.4)]]
    free = (0.7,)
    ref = _increments_exact(modes)
    _set_flags(monkeypatch, FI)
    v = _poset(FI, rows, modes, free)
    assert abs(v - ref) <= 1e-14 * abs(ref)


def test_t_min_is_validated_and_never_ignored(FI, monkeypatch):
    _set_flags(monkeypatch, FI, t_min=-5.0)
    assert FI._t_min() == -5.0
    with pytest.raises(ValueError, match='PHASE_J_T_MIN'):
        FI._integrate_polytope(lambda *a: 1.0, [([1.0], 0.0)], [], 1)
    monkeypatch.setattr(FI, 'PHASE_J_T_MIN', float('nan'))
    with pytest.raises(ValueError, match='PHASE_J_T_MIN'):
        FI._t_min()


# ── DBM ───────────────────────────────────────────────────────────────
# s1 < s0 < t and s1 < s2 < t: two linear extensions, no scalar lower.
_DBM_ROWS = [((-1.0, 0.0, 0.0), (1.0,), 0.0),
             ((1.0, -1.0, 0.0), (0.0,), 0.0),
             ((0.0, -1.0, 1.0), (0.0,), 0.0),
             ((0.0, 0.0, -1.0), (1.0,), 0.0)]


def _dbm(fi, modes, free, cap=None, plan=False):
    ems = [_ems(fi, md) for md in modes]
    p = fi._build_modesum_plan(ems, _DBM_ROWS, 3, len(free)) if plan else None
    return fi._integrate_subset_dbm(
        ems, 1.0, _DBM_ROWS, list(free), 3,
        pole_tuples=None if p is None else p['pole_tuples'], plan=p, cap=cap)


@pytest.mark.parametrize('plan', [False, True])
def test_dbm_box_free_and_slow_mode_box(FI, monkeypatch, plan):
    free = (0.7,)
    fast = [[(1.0, -1.0 + 1j)], [(0.8, -1.5)], [(1.0, -0.9)], [(0.6, -2.0)]]
    _set_flags(monkeypatch, FI)
    far = _dbm(FI, fast, free, cap=3000.0, plan=plan)
    FI._reset_runtime_counters()
    v = _dbm(FI, fast, free, plan=plan)
    assert FI._RUNTIME_COUNTERS['dbm_box_free'] == 1
    assert abs(v - far) <= 1e-13 * abs(far)
    # the poset route (two extensions, semi-infinite chains) agrees
    p = _poset(FI, _DBM_ROWS, fast, free, plan)
    assert abs(p - v) <= 1e-13 * abs(v)
    # slow modes on the ray (0, -1, 0) (s1 -> -inf: Δt2 and Δt3 grow, net
    # rate 0.02): the ±200 box of the pre-0.2.0 route (``cap=200.0``, its
    # default) truncates; the box-free route does not.  (The retired boxed
    # DD route's ±max(200, 40/κ_min) agreed with the box-free value to
    # 1e-13; an explicit cap at that distance, 40/0.01 = 4000, is the same
    # independent reference.)
    slow = [[(1.0, -1.0)], [(0.8, -0.01)], [(1.0, -0.01 + 0.3j)], [(0.6, -2.0)]]
    v_free = _dbm(FI, slow, free, plan=plan)
    v_200 = _dbm(FI, slow, free, cap=200.0, plan=plan)
    v_far = _dbm(FI, slow, free, cap=4000.0, plan=plan)
    assert abs(v_far - v_free) <= 1e-13 * abs(v_free)
    assert abs(v_200 - v_free) > 1e-3 * abs(v_free)


def test_dbm_box_free_divergence_is_reported(FI, monkeypatch):
    _set_flags(monkeypatch, FI)
    # the ray (0, -1, 0) grows at the net rate 1.0 - 0.5 > 0
    grow = [[(1.0, -1.0)], [(0.8, 1.0)], [(1.0, -0.5)], [(0.6, -2.0)]]
    FI._reset_runtime_counters()
    FI._pop_bail_reason()
    assert _dbm(FI, grow, (0.7,)) is None
    assert FI._pop_bail_reason() == 'dbm_divergent'
    assert FI._RUNTIME_COUNTERS['dbm_declined_divergent'] == 1


# ── end to end ────────────────────────────────────────────────────────
def _zoo_run(entry):
    import tests.tools.phase_j_zoo_baseline as Z
    rec = Z.run_entry(entry, with_provenance=False)
    assert rec['status'] == 'ok', rec.get('status')
    return Z.result_arrays(rec), rec['counters']


def test_ou_quartic_routes(FI, monkeypatch):
    """Pure-pole model on the box-free routes (ell <= 2, k = 2): 0 scipy
    calls, every m >= 3 subset on the semi-infinite DD chain (the pole
    engine's routes: S3 makes the state-space route the default).  (Before
    audit W7 retired them, the legacy routes gave the same numbers to
    1e-12 here.)"""
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    import tests.tools.phase_j_zoo_baseline as Z
    entry = Z._E('t-m7-ou', model='ou_quartic', k=2, max_ell=2,
                 ext=[('dx', 1), ('dx', 1)], params=Z.P_OU,
                 tau_grid=[0.0, 0.5, 2.0])
    _set_flags(monkeypatch, FI)
    on, ctr = _zoo_run(entry)
    assert ctr['nquad_calls'] == 0
    assert ctr['dd_poset'] > 0 and ctr['poset_semiinf_chains'] > 0
    # main's numbers (dca8804, frozen; the legacy routes agreed to 1e-12)
    from tests.test_phase_j_structural_zeros import _frozen_close
    _frozen_close('ou_routes|', {k: v for k, v in on.items()})


# ═══════════════════════════════════════════════════════════════════════
# Review round 1 (M7): regressions
# ═══════════════════════════════════════════════════════════════════════

def test_region_geometry_is_exact():
    # an unbounded strip 1e-13 wide is a region (here divergent along
    # (1, 1)), not "empty" by a tolerance
    thin = bf.region_2d([(1.0, -1.0, 1e-13), (-1.0, 1.0, 0.0), (1.0, 0.0, 0.5)])
    assert thin.kind == 'unbounded'
    assert bf.region_integral(thin, 1.0, 1.0, 0j)[0] == bf.DIVERGENT
    # ... and convergent the other way, with the strip's tiny area
    st, v = bf.region_integral(thin, -1.0, -2.0, 0j)
    ref = _sweep_mp([((1, -1), 1e-13), ((-1, 1), 0.0), ((1, 0), 0.5)],
                    (-1.0, -2.0))
    assert st == bf.OK and abs(v - ref) <= 1e-12 * abs(ref)
    # exactly zero width with scaled opposite normals: measure zero
    z = bf.region_2d([(1.0, -1.0, -5.461280594519856),
                      (-1.5, 1.5, 8.191920891779784)])
    assert z.kind == 'empty' and z.reason == 'measure_zero'
    # two vertices 1e-13 apart are kept apart
    r = bf.region_2d([(-1.0, 0.0, 1.0), (0.0, -1.0, 1.0), (-1.0, 1.0, 1e-13)])
    assert len(r.vertices) == 2
    with pytest.raises(ValueError):
        bf.region_2d([(1.0, 0.0, float('nan'))])


def test_region_overflow_is_decided_on_the_sum_of_the_pieces():
    """Oscillatory exponent on the unit square: each fan piece is ~e^2.7
    larger than the integral, whose log is 699 (representable)."""
    sq = bf.region_2d([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                       (-1.0, 0.0, 1.0), (0.0, -1.0, 1.0)])
    al, be = 0.01 + 60j, 0.01 - 60j
    with mp.workdps(40):
        a, b = mp.mpc(al), mp.mpc(be)
        unit = ((mp.exp(a) - 1) / a) * ((mp.exp(b) - 1) / b)
        gamma = 699.0 - float(mp.log(abs(unit)))
        ref = complex(unit * mp.exp(gamma))
    st, v = bf.region_integral(sq, al, be, gamma)
    assert st == bf.OK and abs(v - ref) <= 1e-10 * abs(ref)
    assert bf.region_integral(sq, al, be, gamma + 2.0)[0] == bf.OVERFLOW


@pytest.mark.parametrize('seed', [3, 11])
def test_divergent_verdicts_match_an_independent_ray_test(seed):
    """Every DIVERGENT of the kernel is a real one: some recession ray
    (computed here independently, in exact arithmetic) has Re(α·r) >= 0, or
    the region contains a line."""
    from fractions import Fraction as F
    rng = random.Random(seed)
    n_div = 0
    for _ in range(400):
        cons = _random_system(rng)
        alpha = (complex(rng.uniform(-2, 6), rng.uniform(-3, 3)),
                 complex(rng.uniform(-2, 6), rng.uniform(-3, 3)))
        reg = _region(cons)
        st, _v = bf.region_integral(reg, alpha[0], alpha[1], 0j)
        A = [(F(a[0]), F(a[1])) for (a, _c) in cons]
        rays = [u for (a0, a1) in A for u in ((-a1, a0), (a1, -a0))
                if all(b0 * u[0] + b1 * u[1] >= 0 for (b0, b1) in A)]
        line = any((-u[0], -u[1]) in rays for u in rays)
        grows = any(float(alpha[0].real * float(u[0])
                          + alpha[1].real * float(u[1])) >= 0 for u in rays)
        if reg.kind in ('bounded', 'empty'):
            assert st == bf.OK
        elif st == bf.DIVERGENT:
            n_div += 1
            assert line or grows
        else:
            assert st == bf.OK and not line and not grows
    assert n_div > 20


def test_interval_semi_infinite_with_slow_decay_is_finite():
    for a in (0.5, 0.05, 1e-6):
        st, v = bf.interval(a + 0.3j, 0.1j, -math.inf, 2.0)
        ref = _mp_interval(a + 0.3j, 0.1j, -math.inf, 2.0)
        assert st == bf.OK and abs(v - ref) <= 1e-12 * abs(ref)
    # a residue below the caller's margin is not decay
    assert bf.interval(1e-17, 0j, -math.inf, 2.0, tol=1e-14)[0] == bf.DIVERGENT


def test_chain_log_open_top_corner_cases():
    al = [-1.0, -2.0, -0.5]
    # an upper at or below L: exactly 0
    assert bf.chain_log(al, 0.0, {0: -1.0}, math.inf)[:1] == (bf.OK,)
    st, ls, mant = bf.chain_log(al, 0.0, {0: -1.0}, math.inf)
    assert mant == 0
    # an upper on the last position closes the top
    st, ls, mant = bf.chain_log(al, 0.0, {2: 3.0}, math.inf)
    st2, ls2, mant2 = bf.chain_log(al, 0.0, {}, 3.0)
    assert st == st2 == bf.OK and mant * cmath.exp(ls) == mant2 * cmath.exp(ls2)
    with pytest.raises(ValueError):
        bf.chain_log(al, 0.0, {1: float('nan')}, math.inf)
    assert bf.chain_log([], -math.inf, {}, math.inf) == (bf.OK, 0.0, 1.0 + 0j)
    # a suffix sum within the margin is not decay
    assert bf.chain_log([-1.0, 1e-17], 0.0, {}, math.inf, tol=1e-14)[0] \
        == bf.DIVERGENT


def test_poset_chain_far_from_the_origin(FI, monkeypatch):
    """γ is folded before the chain's own overflow verdict: the chain
    s0 < s1 < s2 < t has a huge exponent at t = 80 or 200 (its own log is
    ~800), the integral is the same at every t."""
    modes = [[(1.0, -1.0 + 0.5j)], [(1.0, -2.0)], [(1.0, -10.0)]]
    ref = _increments_exact(modes)
    _set_flags(monkeypatch, FI)
    for t in (80.0, 200.0, -400.0, -1000.0):
        for plan in (False, True):
            v = _poset(FI, _CHAIN_ROWS, modes, (t,), plan)
            assert v is not None, (t, FI._pop_bail_reason())
            assert abs(v - ref) <= 1e-12 * abs(ref), t


def test_t_min_on_every_route_against_references(FI, monkeypatch):
    """A finite PHASE_J_T_MIN (all switches on) bounds every time from
    below, on the m=1, m=2 (exact), m>=3 and DBM routes."""
    _set_flags(monkeypatch, FI, t_min=-1.0)
    # m = 1: -1 < s < t
    modes1 = [[(1.0, -0.7 + 0.2j)], [(1.0, -0.3)]]
    rows1 = [((-1.0,), (1.0,), 0.0), ((-1.0,), (1.0,), 0.5)]
    v1 = FI._integrate_1d_polytope_modesum(
        [_ems(FI, m) for m in modes1], 1.0, rows1, [0.4])
    lam = -0.7 + 0.2j - 0.3
    ref1 = _mp_interval(-lam, lam * 0.4 + (-0.3) * 0.5, -1.0, 0.4)
    assert abs(v1 - ref1) <= 1e-13 * abs(ref1)
    # m = 2: the wedge s1 < s0 < t, cut at -1
    v2 = _polygon(FI, _WEDGE_ROWS, _WEDGE_FAST, (0.7,))
    cut = [((a[0], a[1]), c + sum(x * 0.7 for x in e))
           for (a, e, c) in _WEDGE_ROWS] + [((1, 0), 1.0), ((0, 1), 1.0)]
    ref2 = 0j
    for C, lams in _tuples(_WEDGE_FAST):
        al = (sum(l * r[0][0] for l, r in zip(lams, _WEDGE_ROWS)),
              sum(l * r[0][1] for l, r in zip(lams, _WEDGE_ROWS)))
        g = sum(l * (r[2] + 0.7 * r[1][0]) for l, r in zip(lams, _WEDGE_ROWS))
        ref2 += C * _box_quad_mp(cut, al, g, 5)
    assert abs(v2 - ref2) <= 1e-12 * abs(ref2)
    # m >= 3 and the DBM: s1 < s0 < t, s1 < s2 < t, all > -1, two routes
    modes = [[(1.0, -1.0 + 1j)], [(0.8, -1.5)], [(1.0, -0.9)], [(0.6, -2.0)]]
    p = _poset(FI, _DBM_ROWS, modes, (0.7,))
    d = _dbm(FI, modes, (0.7,))
    d_cap = _dbm(FI, modes, (0.7,), cap=3000.0)     # explicit cap: rows kept
    assert abs(p - d) <= 1e-13 * abs(d) and abs(d_cap - d) <= 1e-13 * abs(d)
    _set_flags(monkeypatch, FI, t_min=-2.0)
    assert abs(_dbm(FI, modes, (0.7,)) - d) > 1e-3 * abs(d)


def test_t_min_is_absolute_time_in_every_wick_frame(FI, monkeypatch):
    """contribution() moves the origin leg of each permutation to 0; the
    bound follows it.  The same region in two frames gives one value."""
    _set_flags(monkeypatch, FI, t_min=-1.0)
    modes1 = [[(1.0, -0.7)]]
    rows1 = [((-1.0,), (1.0,), 0.0)]                  # s < t_free
    ems = [_ems(FI, m) for m in modes1]
    absolute = FI._integrate_1d_polytope_modesum(ems, 1.0, rows1, [0.4])
    # the same evaluation seen from a permutation whose origin leg is at 3
    tie = FI._TieContext((3.0, 3.4), (1,), 0)
    _set_flags(monkeypatch, FI, t_min=2.0)
    shifted = FI._integrate_1d_polytope_modesum(ems, 1.0, rows1, [0.4],
                                                tie_ctx=tie)
    assert shifted == absolute
    with pytest.raises(ValueError, match='before the switch-on'):
        _set_flags(monkeypatch, FI, t_min=3.2)
        FI._integrate_1d_polytope_modesum(ems, 1.0, rows1, [0.4],
                                          tie_ctx=tie)


def test_switch_on_transient_of_the_ou_tree(FI, monkeypatch):
    """End to end: the OU process started at 0 at T = -3 has
    Cov(x(0), x(2)) = e^{-2} (1 - e^{-6}) at tree level (μ = D = 1)."""
    import tests.tools.phase_j_zoo_baseline as Z
    entry = Z._E('t-m7-switch-on', model='ou_quartic', k=2, max_ell=0,
                 ext=[('dx', 1), ('dx', 1)], params=Z.P_OU, tau_grid=[2.0])
    _set_flags(monkeypatch, FI, t_min=-3.0)
    arr, _ctr = _zoo_run(entry)
    assert abs(arr['ell0'][0] - math.exp(-2.0) * (1 - math.exp(-6.0))) \
        <= 1e-13
    _set_flags(monkeypatch, FI)
    arr, _ctr = _zoo_run(entry)
    assert abs(arr['ell0'][0] - math.exp(-2.0)) <= 1e-13


def test_dbm_translation_invariant_subset_with_t_min_is_divergent(FI,
                                                                  monkeypatch):
    """Two times tied only to each other (s1 > s0 > T): the integrand is
    constant along (1, 1) up to a rounding residue, which must not pass as
    decay."""
    _set_flags(monkeypatch, FI, t_min=-5.0)
    rows = [((-1.0, 1.0), (0.0,), 0.0)]               # s1 - s0 > 0
    for lam in (-0.3 + 0.1j, -0.1, -7.0 + 2j):
        modes = [[(1.0, lam)]]
        ems = [_ems(FI, m) for m in modes]
        FI._pop_bail_reason()
        v = FI._integrate_subset_dbm(ems, 1.0, rows, [0.0], 2)
        assert v is None and FI._pop_bail_reason() == 'dbm_divergent'


def test_dbm_cancelled_term_on_an_open_side(FI):
    import engine.integration.time_domain.dbm_integral as DBM
    rows = [((1.0, 0.0), 1.0), ((-1.0, 1.0), 0.0)]     # s0 > -1, s1 > s0
    seeds = [(1.0, 0j, (0j, 0j)), (-1.0, 0j, (0j, 0j)), (1.0, 0j, (0j, -1.0))]
    res = DBM.integrate_exp_sum(rows, 2, seeds, math.inf)
    assert res.status == DBM.STATUS_OK and abs(res.value - math.e) <= 1e-13


def test_kernel_input_validation():
    with pytest.raises(ValueError):
        bf.interval(-0.5, 0j, -math.inf, 0.0, tol=-1.0)
    with pytest.raises(ValueError):
        bf.interval(1e300, 0j, -1e10, 1.0)          # α·T is not finite
    with pytest.raises(ValueError):
        bf.region_2d([(1.0, 0.0, 1e300), (0.0, 1.0, 1e300), (-1.0, -1.0, 1.0)])
    quadrant = _region([((-1, 0), 1.0), ((0, -1), 2.0)])
    with pytest.raises(ValueError):
        bf.region_integral(quadrant, 1.0, 1.0, 0j, tol=-1.0)
    tri = _region([((1, 0), 0.0), ((0, 1), 0.0), ((-1, -1), 1.0)])
    assert bf.region_integral(tri, 1e300, 0.0, 0j)[0] == bf.OVERFLOW
    with pytest.raises(ValueError):
        bf.chain_log([-1.0], 0.0, {}, 1.0, tol=float('nan'))


def test_dbm_default_margin_with_generator_seeds():
    """The default open-side margin is taken from the seeds also when they
    come as a generator (they are walked twice)."""
    import engine.integration.time_domain.dbm_integral as DBM
    rows = [((-1.0, 1.0), 0.0), ((1.0, 0.0), 5.0), ((0.0, 1.0), 5.0)]
    lam = -0.3 + 0.1j
    seeds = [(1.0, 0j, (-lam, lam + 1e-16))]   # the translation residue
    as_list = DBM.integrate_exp_sum(rows, 2, seeds, math.inf)
    as_gen = DBM.integrate_exp_sum(rows, 2, (x for x in seeds), math.inf)
    assert as_list.status == as_gen.status == DBM.STATUS_DIVERGENT


def test_t_min_at_the_fallback_explains_itself(FI, monkeypatch):
    """A finite PHASE_J_T_MIN that reaches the scipy fallback (which cannot
    honour it) raises with the region, the external times, the DBM's
    ill-conditioned declines and the remedy (M7 report: spike reset with T
    within ~2 of the earliest external time)."""
    _set_flags(monkeypatch, FI, t_min=-1.0)
    FI._reset_runtime_counters()
    with pytest.raises(ValueError,
                       match=r'PHASE_J_T_MIN \(= -1\.0\).*cannot honour it'
                             r'.*m = 1.*external times \(0\.5,\).*'
                             r'ill-conditioned 0 time.*further back'):
        FI._integrate_polytope(lambda s, t: 1.0, [([1.0], 0.0)], [0.5], 1)


def test_dd_error_accounting(FI, monkeypatch):
    """M7 report: stiff mode ratios lose accuracy with an OK status.  The
    box-free subset sums now carry the DBM's conditioning test as
    bookkeeping (counters ``dd_ill_conditioned`` / ``dd_err_ratio_max``),
    the value unchanged.  Half line s < t: slow modes (rate 1e-7) with
    nearly opposite residues make two ~1e7 terms cancel to O(1e-2) next to a
    fast mode (rate 100): flagged; the fast mode alone: not."""
    _set_flags(monkeypatch, FI)
    half_line = [((-1.0,), (1.0,), 0.0)]

    def run(modes):
        FI._reset_runtime_counters()
        ems = [_ems(FI, modes)]
        v = FI._integrate_1d_polytope_modesum(ems, 1.0, half_line, [0.4])
        c = FI._RUNTIME_COUNTERS
        return v, c['dd_ill_conditioned'], c['dd_err_ratio_max']

    v, n, r = run([(1.0, -100.0)])
    assert v is not None and n == 0 and r < 1e-3
    v, n, r = run([(1.0, -1e-7), (-1.0, -1e-7 * (1 + 1e-9)), (1.0, -100.0)])
    assert v is not None and n == 1 and r > 1e3
    # the rule itself: COND_ERR_FACTOR·mag vs max(COND_RTOL·|v|, COND_ATOL·scale)
    FI._reset_runtime_counters()
    FI._dd_accuracy(1.0, 1e4, 1.0)                  # 1e-11 vs 1e-10: fine
    assert FI._RUNTIME_COUNTERS['dd_ill_conditioned'] == 0
    assert abs(FI._RUNTIME_COUNTERS['dd_err_ratio_max'] - 0.1) < 1e-12
    FI._dd_accuracy(1.0, 1e6, 1.0)                  # 1e-9 vs 1e-10: flagged
    assert FI._RUNTIME_COUNTERS['dd_ill_conditioned'] == 1

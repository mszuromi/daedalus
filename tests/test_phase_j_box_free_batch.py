"""M7b: the batched box-free kernels (``boxfree.region_integral_batch`` /
``region_modesum`` / ``group_fold`` / ``chain_fold``) and the batch memo of
the poset DD route (``final_integral._chain_dd_batch``).

Each batched kernel against the scalar routine (or the scalar loop of
``final_integral``) it replaces, on random inputs with repeated and close
nodes; the hand-over to the scalar loop whenever a term leaves the plain
case; and the batch memo's keys (exact, same object on a hit).  The batched
values agree with the scalar ones to rounding (compiled complex products
may be contracted to FMAs), not bit for bit.
"""
import cmath
import math
import random

import numpy as np
import pytest

from engine.integration.time_domain import boxfree as bf
from tests.test_phase_j_box_free import (_CHAIN_ROWS, _ROUTES, _WEDGE_FAST,
                                         _WEDGE_ROWS, _ems, _set_flags)


# ═══════════════════════════════════════════════════════════════════════
# Kernels
# ═══════════════════════════════════════════════════════════════════════

def _random_region(rng):
    while True:
        rows = [(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3, 3))
                for _ in range(rng.randint(2, 5))]
        reg = bf.region_2d(rows)
        if reg.kind in ('bounded', 'unbounded'):
            return reg


def _random_exponents(rng, n):
    """``n`` (alpha, beta, gamma): random, with repeated and close alphas
    (the divided differences' degenerate nodes) and a few overflowing γ."""
    al = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n)]
    be = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n)]
    ga = [complex(rng.uniform(-40, 40), rng.uniform(-5, 5)) for _ in range(n)]
    for t in range(0, n, 5):
        al[t] = be[t]                                  # repeated
        if t + 1 < n:
            al[t + 1] = al[t] + 1e-9                   # close
            be[t + 1] = be[t] - 1e-11j
    for t in range(3, n, 17):
        ga[t] = complex(850.0, 0.3)                    # overflow
    return al, be, ga


def _scalar(region, a, b, g, tol):
    try:
        return bf.region_integral(region, a, b, g, tol)
    except ValueError:
        return -1, None


def test_region_integral_batch_matches_the_scalar_routine():
    rng = random.Random(7)
    n_checked = 0
    statuses = set()
    while n_checked < 1000:
        reg = _random_region(rng)
        al, be, ga = _random_exponents(rng, 50)
        tol = 1e-12
        st, val = bf.region_integral_batch(reg, al, be, ga, tol)
        for t in range(len(al)):
            s0, v0 = _scalar(reg, al[t], be[t], ga[t], tol)
            assert st[t] == s0, (t, s0, st[t])
            statuses.add(s0)
            if s0 == bf.OK:
                assert abs(val[t] - v0) <= 1e-12 * abs(v0), (t, v0, val[t])
            n_checked += 1
    assert {bf.OK, bf.OVERFLOW, bf.DIVERGENT} <= statuses


def test_region_integral_batch_flags_the_inputs_the_scalar_routine_rejects():
    reg = bf.region_2d([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                        (-1.0, -1.0, 1.0)])           # a triangle
    st, val = bf.region_integral_batch(reg, [complex('nan'), 0.3],
                                       [0.1, 0.2], [0j, 0j])
    assert list(st) == [-1, bf.OK]
    with pytest.raises(ValueError):
        bf.region_integral(reg, complex('nan'), 0.1, 0j)
    for kind, status in (('empty', bf.OK), ('lineality', bf.DIVERGENT)):
        st, val = bf.region_integral_batch(bf.Region2D(kind), [1j], [1j],
                                           [0j])
        assert list(st) == [status] and val[0] == 0


def test_region_modesum_is_the_ordered_sum_or_hands_over():
    rng = random.Random(11)
    n_handed = 0
    for _ in range(40):
        reg = _random_region(rng)
        n = 30
        # decaying along every ray: alpha, beta with large negative parts
        al = [complex(-rng.uniform(4, 6), rng.uniform(-1, 1)) for _ in range(n)]
        be = [complex(-rng.uniform(4, 6), rng.uniform(-1, 1)) for _ in range(n)]
        ga = [complex(rng.uniform(-5, 5), rng.uniform(-1, 1)) for _ in range(n)]
        co = [complex(rng.uniform(0.5, 1), rng.uniform(0, 0.1))
              for _ in range(n)]
        co[4] = 0j
        sts = [_scalar(reg, a, b, g, 0.0) for a, b, g in zip(al, be, ga)]
        if any(s != bf.OK for (s, _v) in sts):
            continue
        ref = 0j
        for c, (_s, v) in zip(co, sts):
            if c != 0:
                ref += c * v
        got = bf.region_modesum(reg, co, al, be, ga)
        assert abs(got - ref) <= 1e-12 * sum(
            abs(c * v) for c, (_s, v) in zip(co, sts)), (got, ref)
        if reg.kind == 'unbounded':
            # a divergent tuple (growing along a ray) with a ZERO coefficient
            # is never evaluated; with a nonzero one the scalar loop decides
            rx, ry = reg.rays[0]
            al2, be2, co2 = list(al), list(be), list(co)
            al2[4], be2[4] = complex(5 * rx, 0.0), complex(5 * ry, 0.0)
            assert bf.region_modesum(reg, co2, al2, be2, ga) is not None
            co2[4] = 1.0
            assert bf.region_modesum(reg, co2, al2, be2, ga) is None
            n_handed += 1
        ga2 = list(ga)
        ga2[2] = complex(2000.0, 0.0)
        sts2 = [_scalar(reg, a, b, g, 0.0) for a, b, g in zip(al, be, ga2)]
        plain = all(s == bf.OK for c, (s, _v) in zip(co, sts2) if c != 0)
        assert (bf.region_modesum(reg, co, al, be, ga2) is not None) == plain
        n_handed += not plain
    assert n_handed >= 10


def _group_fold_ref(coef, gamma, groups):
    tc, gmax = [], []
    for idxs in groups:
        mem = [(coef[t], gamma[t]) for t in idxs if coef[t] != 0]
        if not mem:
            tc.append(0j)
            gmax.append(0.0)
            continue
        g_max = max(g.real for (_c, g) in mem)
        s = 0j
        for (c, g) in mem:
            s += c * cmath.exp(g - g_max)
        tc.append(s)
        gmax.append(g_max)
    return tc, gmax


def test_group_fold_matches_the_scalar_fold():
    rng = random.Random(5)
    T = 1000
    coef = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(T)]
    gamma = [complex(rng.uniform(-60, 60), rng.uniform(-3, 3))
             for _ in range(T)]
    for t in range(0, T, 13):
        coef[t] = 0j
    for t in range(1, T, 7):
        gamma[t] = gamma[t - 1] + 1e-12                 # close exponents
    perm = list(range(T))
    rng.shuffle(perm)
    groups, i = [], 0
    while i < T:
        k = rng.randint(1, 6)
        groups.append(sorted(perm[i:i + k]))
        i += k
    groups.append([0])                                  # all-zero group
    member_idx = [t for g in groups for t in g]
    member_off = np.concatenate([[0], np.cumsum([len(g) for g in groups])])
    tc, gmax = bf.group_fold(coef, gamma, member_idx, member_off)
    tc0, gmax0 = _group_fold_ref(coef, gamma, groups)
    assert list(gmax) == gmax0
    for a, b in zip(tc, tc0):
        assert (a == 0) == (b == 0)
        assert abs(a - b) <= 1e-14 * max(abs(b), 1e-300)


def _chain_fold_ref(tc, gmax, ST, LS, MANT):
    total = 0j
    for g in range(len(tc)):
        if tc[g] == 0:
            continue
        for e in range(len(ST)):
            if ST[e][g] not in (bf.OK, bf.OVERFLOW):
                return None
            zr = gmax[g] + LS[e][g]
            if not (-bf.LOG_OVERFLOW <= zr <= bf.LOG_OVERFLOW
                    and abs(MANT[e][g]) < 1e150 and abs(tc[g]) < 1e150):
                return None
            total += (tc[g] * MANT[e][g]) * math.exp(zr)
    return total


def test_chain_fold_matches_the_scalar_fold_and_hands_over():
    rng = random.Random(9)
    for _ in range(25):
        G, E = rng.randint(1, 60), rng.randint(1, 5)
        tc = [complex(rng.uniform(0.1, 1), rng.uniform(-1, 1))
              for _ in range(G)]
        tc[0] = 0j
        gmax = [rng.uniform(-30, 30) for _ in range(G)]
        ST = [[bf.OK] * G for _ in range(E)]
        LS = [[rng.uniform(-30, 30) for _ in range(G)] for _ in range(E)]
        MANT = [[complex(rng.uniform(0.5, 1), rng.uniform(0, 0.2))
                 for _ in range(G)] for _ in range(E)]
        ref = _chain_fold_ref(tc, gmax, ST, LS, MANT)
        got = bf.chain_fold(tc, gmax, ST, LS, MANT)
        scale = sum(abs(tc[g] * MANT[e][g]) * math.exp(gmax[g] + LS[e][g])
                    for g in range(G) for e in range(E))
        assert abs(got - ref) <= 1e-14 * scale
        # a divergent chain in an INACTIVE group is never looked at
        ST[E - 1][0] = bf.DIVERGENT
        assert bf.chain_fold(tc, gmax, ST, LS, MANT) == got
        if G > 1:
            ST[E - 1][G - 1] = bf.DIVERGENT
            assert bf.chain_fold(tc, gmax, ST, LS, MANT) is None
            ST[E - 1][G - 1] = bf.OVERFLOW              # folded: in range
            assert bf.chain_fold(tc, gmax, ST, LS, MANT) is not None
            LS[0][G - 1] = 800.0                        # out of range
            assert bf.chain_fold(tc, gmax, ST, LS, MANT) is None


# ═══════════════════════════════════════════════════════════════════════
# Engine: the batched plan branches against the scalar loops
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def FI():
    pytest.importorskip('sage.all')
    import engine.integration.time_domain.final_integral as fi
    fi._chain_simplex_memo_clear()
    yield fi
    fi._chain_simplex_memo_clear()


def _plan(fi, rows, modes, free, m):
    ems = [_ems(fi, md) for md in modes]
    return ems, fi._build_modesum_plan(ems, rows, m, len(free))


def _poly_plan(fi, rows, modes, free):
    ems, p = _plan(fi, rows, modes, free, 2)
    return fi._integrate_2d_polygon_modesum(
        ems, 0.9 - 0.1j, rows, list(free), pole_tuples=p['pole_tuples'],
        plan=p)


def _poset_plan(fi, rows, modes, free, m=3):
    ems, p = _plan(fi, rows, modes, free, m)
    return fi._integrate_nd_polytope_poset_modesum(
        ems, 0.9 - 0.1j, rows, list(free), m, pole_tuples=p['pole_tuples'],
        plan=p)


_CTR = ('dd_polygon', 'polygon_exact_unbounded', 'dd_poset',
        'poset_semiinf_chains', 'divergent_subsets', 'dd_overflow',
        'dd_unsupported', 'nquad_calls')


def _run(fi, fn, *args):
    fi._reset_runtime_counters()
    fi._chain_simplex_memo_clear()
    v = fn(fi, *args)
    return v, {k: fi._RUNTIME_COUNTERS[k] for k in _CTR}


def _spy(monkeypatch, obj, name):
    calls = []
    orig = getattr(obj, name)

    def w(*a, **k):
        r = orig(*a, **k)
        calls.append(r)
        return r
    monkeypatch.setattr(obj, name, w)
    return calls


# The wedge of the M7 tests with 3 x 2 x 2 = 12 pole tuples (>= _DD_BATCH_MIN).
_WEDGE_MANY = [[(1.0, -1.3 + 0.4j), (0.5, -2.0), (0.2, -1.7)],
               [(0.8 - 0.2j, -0.9), (0.3, -1.2)],
               [(1.0, -0.6 + 1j), (0.4, -0.8)]]


def test_polygon_batch_agrees_with_the_scalar_loop(FI, monkeypatch):
    _set_flags(monkeypatch, FI, **_ROUTES['box_free'])
    calls = _spy(monkeypatch, bf, 'region_modesum')
    v, c = _run(FI, _poly_plan, _WEDGE_ROWS, _WEDGE_MANY, (0.7,))
    assert len(calls) == 1 and calls[0] is not None
    monkeypatch.setattr(bf, 'region_modesum', lambda *a, **k: None)
    v0, c0 = _run(FI, _poly_plan, _WEDGE_ROWS, _WEDGE_MANY, (0.7,))
    assert c == c0 and c['polygon_exact_unbounded'] == 1
    assert abs(v - v0) <= 1e-14 * abs(v0)


def test_polygon_batch_hands_a_divergent_subset_to_the_scalar_loop(
        FI, monkeypatch):
    _set_flags(monkeypatch, FI, **_ROUTES['box_free'])
    growing = [[(1.0, 2.0)] + _WEDGE_MANY[0][1:]] + _WEDGE_MANY[1:]
    calls = _spy(monkeypatch, bf, 'region_modesum')
    v, c = _run(FI, _poly_plan, _WEDGE_ROWS, growing, (0.7,))
    assert calls == [None]
    assert v is None and c['divergent_subsets'] == 1


# Chain s0 < s1 < s2 < t with an intermediate upper s0 < t - 0.4 (cuts).
_CHAIN_UPPER_ROWS = _CHAIN_ROWS + [((-1.0, 0.0, 0.0), (1.0,), -0.4)]
_CHAIN_MODES = [[(1.0, -2.0 + 1j), (0.3, -1.1), (0.2, -1.1 + 1e-9)],
                [(0.8, -1.5), (0.4, -0.7 - 0.3j)],
                [(1.0, -3.0), (0.5, -2.0)],
                [(1.0, -0.5)]]


@pytest.mark.parametrize('rows, modes', [
    (_CHAIN_ROWS, _CHAIN_MODES[:3]),
    (_CHAIN_UPPER_ROWS, _CHAIN_MODES),
])
def test_poset_batch_agrees_with_the_scalar_loop(FI, monkeypatch, rows,
                                                 modes):
    _set_flags(monkeypatch, FI, **_ROUTES['box_free'])
    calls = _spy(monkeypatch, FI, '_poset_dd_batch')
    v, c = _run(FI, _poset_plan, rows, modes, (0.7,))
    assert len(calls) == 1 and calls[0] is not None
    monkeypatch.setattr(FI, '_poset_dd_batch', lambda *a, **k: None)
    v0, c0 = _run(FI, _poset_plan, rows, modes, (0.7,))
    assert c == c0 and c['dd_poset'] == 1 and c['poset_semiinf_chains'] == 1
    assert abs(v - v0) <= 1e-14 * abs(v0)


def test_poset_batch_hands_a_divergent_chain_to_the_scalar_loop(
        FI, monkeypatch):
    _set_flags(monkeypatch, FI, **_ROUTES['box_free'])
    growing = [[(1.0, 0.5), (0.5, 0.6)], [(0.8, -0.1), (0.4, -0.2)],
               [(1.0, -3.0), (0.5, -2.5)]]                  # 8 groups
    calls = _spy(monkeypatch, FI, '_poset_dd_batch')
    batches = _spy(monkeypatch, FI, '_chain_dd_batch')
    v, c = _run(FI, _poset_plan, _CHAIN_ROWS, growing, (0.7,))
    assert calls == [None] and len(batches) == 1       # handed over
    assert v is None and c['divergent_subsets'] == 1


def test_small_plans_stay_on_the_scalar_loops(FI, monkeypatch):
    _set_flags(monkeypatch, FI, **_ROUTES['box_free'])
    assert FI._DD_BATCH_MIN == 8
    poly = _spy(monkeypatch, bf, 'region_modesum')
    batches = _spy(monkeypatch, FI, '_chain_dd_batch')
    _run(FI, _poly_plan, _WEDGE_ROWS, _WEDGE_FAST, (0.7,))     # 2 tuples
    _run(FI, _poset_plan, _CHAIN_ROWS,
         [[(1.0, -2.0 + 1j), (0.3, -1.1)], [(0.8, -1.5)], [(1.0, -3.0)]],
         (0.7,))                                                # 2 groups
    assert poly == [] and batches == []


def _batch_args(fi, modes, sigma=(0, 1, 2), L=-math.inf, upp=None, U=0.7,
                tol=0.0):
    ems, p = _plan(fi, _CHAIN_ROWS, modes, (0.7,), 3)
    upp = {} if upp is None else upp
    key = fi._chain_uppers_key_suffix(L, upp, U)
    return p, fi._m7b_plan_arrays(p), 3, list(sigma), L, upp, U, key, tol


def test_chain_batch_memo_keys_are_exact(FI, monkeypatch):
    _set_flags(monkeypatch, FI, **_ROUTES['box_free'])
    modes = _CHAIN_MODES[:3]
    a = FI._chain_dd_batch(*_batch_args(FI, modes))
    # same rows, a fresh plan: the very same object
    assert FI._chain_dd_batch(*_batch_args(FI, modes)) is a
    G = len(a[0])
    # one value per group, those of the per-chain route
    p = _batch_args(FI, modes)[0]
    for g, (alphas, _i) in enumerate(p['tuple_groups']):
        st, ls, mant = FI._bf().chain_log(list(alphas), -math.inf, {}, 0.7)
        assert (a[0][g], a[1][g], a[2][g]) == (st, ls, mant)
    # a different upper, extension, tol or plan alphas: a different entry
    others = [FI._chain_dd_batch(*_batch_args(FI, modes, U=0.71)),
              FI._chain_dd_batch(*_batch_args(FI, modes, upp={0: 0.2})),
              FI._chain_dd_batch(*_batch_args(FI, modes, sigma=(1, 0, 2))),
              FI._chain_dd_batch(*_batch_args(FI, modes, tol=1e-12))]
    ulp = [list(md) for md in modes]
    c0, l0 = ulp[0][0]
    ulp[0][0] = (c0, complex(math.nextafter(l0.real, 0.0), l0.imag))
    others.append(FI._chain_dd_batch(*_batch_args(FI, ulp)))
    assert all(o is not a and len(o[0]) == G for o in others)
    # dropped with the chain memo
    FI._chain_simplex_memo_clear()
    assert FI._chain_dd_batch(*_batch_args(FI, modes)) is not a


def test_poset_batch_is_off_with_the_chain_memo(FI, monkeypatch):
    _set_flags(monkeypatch, FI, **_ROUTES['box_free'])
    monkeypatch.setattr(FI, 'USE_CHAIN_UPPERS_MEMO', False)
    calls = _spy(monkeypatch, FI, '_poset_dd_batch')
    v, c = _run(FI, _poset_plan, _CHAIN_ROWS, _CHAIN_MODES[:3], (0.7,))
    assert calls == [] and c['dd_poset'] == 1
    assert not FI._chain_dd_batch_memo

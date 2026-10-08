"""M6: the divided-difference kernels of ``engine/integration/time_domain/expdd.py``.

Accuracy is measured against mpmath at 80 digits (``mp.expm`` of the
lower-bidiagonal matrix, whose ``(n, 0)`` entry is ``exp[z_0..z_n]``), with
the acceptance rule of the M6 brief: an error is accepted when it is
``<= 1e-11`` relative to the value, or ``<= 1e-14`` relative to the
absolute-value integral (the same integral with every exponent replaced by
its real part) when the value itself cancels.

The batteries port the prototype checks (``cfx_accuracy`` sections 1-6 and
``cs_bench`` A-C, reduced) and add the three gaps the prototype never
measured: oscillatory nodes (``|Im z| >> |Re z|``), long chains
(``n = 8..16``) and nodes of mixed magnitude (``1e-6..1e3``).

``DAEDALUS_EXPDD_TEST_SEED`` (default 0) reseeds every random battery.

Run with::

    PYTHONHASHSEED=0 python -m pytest tests/test_expdd.py -v
    NUMBA_DISABLE_JIT=1 PYTHONHASHSEED=0 python -m pytest tests/test_expdd.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.integration.time_domain.expdd as X
import engine.integration.time_domain.final_integral as FI

ROOT = os.path.join(os.path.dirname(__file__), '..')
SEED = int(os.environ.get('DAEDALUS_EXPDD_TEST_SEED', '0'))
DPS = 80
REL_TOL = 1e-11          # relative to the value
ABS_TOL = 1e-14          # relative to the absolute-value integral


def _rng(tag):
    """An independent generator per battery (``tag`` keeps them apart)."""
    return np.random.default_rng([SEED, tag])


# ─── mpmath references ─────────────────────────────────────────────────
def _mpc(z):
    z = complex(z)
    return mp.mpc(z.real, z.imag)


def mp_dd(nodes):
    """``exp[nodes]`` at the working precision (nodes are mpc or complex)."""
    n = len(nodes) - 1
    A = mp.zeros(n + 1, n + 1)
    for i in range(n + 1):
        A[i, i] = nodes[i] if isinstance(nodes[i], mp.mpc) else _mpc(nodes[i])
        if i:
            A[i, i - 1] = 1
    return mp.expm(A)[n, 0]


def mp_dd_partial_fractions(nodes, dps):
    """Independent reference for DISTINCT nodes: sum_i e^{z_i} / prod_{j!=i}
    (z_i - z_j), at ``dps`` digits."""
    with mp.workdps(dps):
        z = [_mpc(x) for x in nodes]
        tot = mp.mpc(0)
        for i, zi in enumerate(z):
            den = mp.mpc(1)
            for j, zj in enumerate(z):
                if j != i:
                    den *= zi - zj
            tot += mp.exp(zi) / den
        return complex(tot)


def mp_chain(alphas, L, U):
    """Chain simplex: ``T^m exp[v]`` with the nodes formed in mpmath from the
    double inputs (so the reference is exact for those inputs)."""
    with mp.workdps(DPS):
        m = len(alphas)
        if m == 0:
            return mp.mpc(1)
        L_, U_ = mp.mpf(L), mp.mpf(U)
        T = U_ - L_
        a = [_mpc(x) for x in alphas]
        P = [mp.mpc(0)]
        for x in a:
            P.append(P[-1] + x)
        S = P[-1]
        return T ** m * mp_dd([U_ * S - T * p for p in P])


def mp_uppers(alphas, L, upp, U_top):
    """Chain with intermediate uppers: product of mpmath segment ``expm``
    propagators with projections (the prototype reference)."""
    with mp.workdps(DPS):
        m = len(alphas)
        eff = [U_top] * m
        run = U_top
        for k in range(m - 1, -1, -1):
            if upp.get(k) is not None:
                run = min(run, float(upp[k]))
            eff[k] = run
        if any(e <= L for e in eff):
            return mp.mpc(0)
        top = eff[-1]
        cuts = {}
        for k in range(m):
            if eff[k] < top:
                cuts[eff[k]] = max(cuts.get(eff[k], 0), k + 1)
        a = [_mpc(x) for x in alphas]
        beta = [mp.fsum(a[j:]) for j in range(m)] + [mp.mpc(0)]
        w = mp.matrix(m + 1, 1)
        w[0] = mp.exp(beta[0] * mp.mpf(L))
        t = mp.mpf(L)
        for tn in sorted(cuts) + [top]:
            dt = mp.mpf(tn) - t
            # expm(dt (diag(beta) + N)) = G expm(dt diag(beta) + N) G^-1 with
            # G = diag(dt^j): mp.expm's norm-relative tolerance would lose the
            # tiny dt^(i-j) entries of a short segment (M6 verification, F6)
            A = mp.matrix(m + 1, m + 1)
            for j in range(m + 1):
                A[j, j] = beta[j] * dt
                if j:
                    A[j, j - 1] = 1
            E = mp.expm(A)
            for i in range(m + 1):
                for j in range(i):
                    E[i, j] *= dt ** (i - j)
            w = E * w
            t = mp.mpf(tn)
            if tn in cuts:
                for j in range(cuts[tn]):
                    w[j] = 0
        return w[m]


def _real(xs):
    return [complex(x).real for x in xs]


# ─── the acceptance rule ───────────────────────────────────────────────
def _err(got, ref, absref):
    """``(err/|ref|, err/absref)``; ``got`` complex, refs mpmath."""
    err = abs((got if isinstance(got, mp.mpc) else _mpc(got)) - ref)
    rel = float(err / abs(ref)) if ref != 0 else (0.0 if err == 0 else math.inf)
    return rel, float(err / abs(absref))


def _accept(rel, rabs):
    return rel <= REL_TOL or rabs <= ABS_TOL


class _Worst:
    """Collects the battery's errors; reports every failing case."""

    def __init__(self, label):
        self.label, self.fails, self.max_rel, self.max_abs, self.n = (
            label, [], 0.0, 0.0, 0)

    def add(self, rel, rabs, case):
        self.n += 1
        self.max_rel = max(self.max_rel, rel)
        self.max_abs = max(self.max_abs, rabs)
        if not _accept(rel, rabs):
            self.fails.append((rel, rabs, case))

    def check(self):
        assert self.n > 0, f'{self.label}: empty battery'
        assert not self.fails, (
            f'{self.label}: {len(self.fails)}/{self.n} cases fail the M6 rule; '
            f'worst {max(self.fails, key=lambda f: f[1])}')


def _check_dd(battery, z):
    """Compare mantissas (log domain): the reference is scaled by e^{-c}."""
    with mp.workdps(DPS):
        c, mant = X.log_dd_exp(z)
        s = mp.exp(-mp.mpf(c))
        ref = mp_dd(z) * s
        absref = mp_dd(_real(z)) * s
        battery.add(*_err(mant, ref, absref), case=list(z))
        return mant, ref


def _mpval(result):
    """``(status, log_scale, mant)`` -> the value in mpmath (no underflow)."""
    st, ls, mant = result
    assert st == X.OK, result
    return mp.exp(mp.mpf(ls)) * _mpc(mant)


def _check_chain(battery, alphas, L, U):
    with mp.workdps(DPS):
        battery.add(*_err(_mpval(X.chain_simplex_log(alphas, L, U)),
                          mp_chain(alphas, L, U),
                          mp_chain(_real(alphas), L, U)),
                    case=(list(alphas), L, U))


# ─── node generators (cfx_accuracy section 1) ──────────────────────────
def _gen_nodes(rng, n):
    kind = rng.integers(6)
    base = complex(rng.normal() * rng.choice([0.1, 1, 30, 300]),
                   rng.normal() * rng.choice([0, 1, 30]))
    if kind == 0:      # generic, one scale per case
        s = rng.choice([0.01, 1, 10, 300, 3000])
        return [complex(rng.normal() * s, rng.normal() * s)
                for _ in range(n + 1)]
    if kind == 1:      # exact repeats of two values
        pool = [complex(rng.normal() * 20, rng.normal() * 5) for _ in range(2)]
        return [pool[rng.integers(2)] for _ in range(n + 1)]
    if kind == 2:      # near-degenerate cluster
        d = 10.0 ** rng.uniform(-12, -1)
        return [base + d * complex(rng.normal(), rng.normal())
                for _ in range(n + 1)]
    if kind == 3:      # large real negative spread (fast poles, big box)
        return [complex(-rng.uniform(0, 5000), 0) for _ in range(n + 1)]
    if kind == 4:      # oscillatory
        return [complex(-rng.uniform(0, 5), rng.uniform(-200, 200))
                for _ in range(n + 1)]
    return [0j] + [complex(-rng.uniform(0, 1e4), rng.normal())
                   * rng.choice([1e-8, 1]) for _ in range(n)]


# The three gaps of the M6 brief.
def _gen_oscillatory(rng, n):
    """Re z in [-1, 0], |Im z| in [10, 1000] (random sign)."""
    return list(-rng.uniform(0, 1, n + 1)
                + 1j * rng.uniform(10, 1000, n + 1)
                * rng.choice([-1, 1], n + 1))


def _gen_long(rng, n, kind):
    if kind == 'cluster':          # spread << 1: centroid Taylor branch
        base = complex(rng.normal() * 5, rng.normal() * 5)
        return list(base + 1e-3 * (rng.normal(size=n + 1)
                                   + 1j * rng.normal(size=n + 1)))
    if kind == 'near-cluster':     # spread of a few units: expm branch
        base = complex(rng.normal() * 5, rng.normal() * 5)
        return list(base + 2.0 * (rng.normal(size=n + 1)
                                  + 1j * rng.normal(size=n + 1)))
    return list(-rng.uniform(0, 50, n + 1)            # spread
                + 1j * rng.uniform(-20, 20, n + 1))


def _gen_mixed(rng, n, real):
    """|z| log-uniform in [1e-6, 1e3]."""
    mag = 10.0 ** rng.uniform(-6, 3, n + 1)
    if real:
        return list(-mag * rng.choice([1.0, -1e-3], n + 1) + 0j)
    return list(mag * np.exp(1j * rng.uniform(0, 2 * np.pi, n + 1)))


# ═══ (1) divided differences ══════════════════════════════════════════
@pytest.mark.parametrize('n', range(1, 8))
def test_dd_random_battery(n):
    """cfx_accuracy (1): exp divided differences of every node family."""
    rng = _rng(100 + n)
    b = _Worst(f'dd n={n}')
    for _ in range(40 if n <= 3 else 16):
        _check_dd(b, _gen_nodes(rng, n))
    b.check()


def test_dd_reference_is_independent():
    """The mp.expm reference agrees with the partial-fraction formula at 300
    digits on distinct, well separated nodes."""
    rng = _rng(1)
    for n in (2, 4, 7):
        for _ in range(4):
            z = list(rng.normal(size=n + 1) * 5 + 1j * rng.normal(size=n + 1))
            with mp.workdps(DPS):
                ref = complex(mp_dd(z))
            pf = mp_dd_partial_fractions(z, 300)
            assert abs(ref - pf) <= 1e-30 * abs(pf)


def test_dd_small_orders_and_exact_values():
    assert X.dd_exp([2.0]) == pytest.approx(math.exp(2.0), rel=1e-15)
    assert X.dd_exp([0.0, 0.0]) == 1.0                    # exp[0,0] = e^0
    assert abs(X.dd_exp([0.0, 0.0, 0.0]) - 0.5) < 1e-16
    for n in range(1, 17):
        c, mant = X.log_dd_exp([1.5 + 0.25j] * (n + 1))  # e^z / n!
        val = mant * math.exp(c)
        ref = complex(mp.exp(_mpc(1.5 + 0.25j)) / mp.factorial(n))
        assert abs(val - ref) <= 1e-15 * abs(ref), n


def test_dd_hermite_genocchi_bound():
    """``|mant| <= 1/n!`` (up to rounding) on every family."""
    rng = _rng(2)
    for n in range(1, 13):
        for _ in range(20):
            z = _gen_nodes(rng, n)
            c, mant = X.log_dd_exp(z)
            assert c == max(complex(x).real for x in z)
            assert abs(mant) <= (1 + 1e-12) / math.factorial(n)


def test_dd_is_symmetric_in_the_nodes():
    rng = _rng(3)
    for n in (2, 3, 6):
        z = _gen_nodes(rng, n)
        c0, m0 = X.log_dd_exp(z)
        for _ in range(3):
            c1, m1 = X.log_dd_exp(list(rng.permutation(z)))
            assert c1 == c0
            assert abs(m1 - m0) <= 1e-13 / math.factorial(n)


# ═══ (2) unit triangle ═════════════════════════════════════════════════
def _triangle_pq(rng):
    kind = rng.integers(5)
    if kind == 0:
        return (complex(rng.normal() * 50, rng.normal() * 50),
                complex(rng.normal() * 50, rng.normal() * 50))
    if kind == 1:
        p = complex(rng.normal() * 20, rng.normal() * 5)
        return p, p + 10 ** rng.uniform(-9, -3) * complex(rng.normal(),
                                                           rng.normal())
    if kind == 2:
        return (10 ** rng.uniform(-9, -4) * complex(rng.normal(), rng.normal()),
                complex(rng.normal() * 20, rng.normal()))
    if kind == 3:
        return complex(-rng.uniform(0, 590), 0), complex(-rng.uniform(0, 590), 0)
    return (complex(rng.uniform(-590, 590), rng.normal()),
            complex(rng.uniform(-590, 590), rng.normal()))


def test_unit_triangle_battery():
    """cfx_accuracy (2): J(p, q) = exp[0, p, q] on 400 cases (the prototype
    measured 2.1e-15)."""
    rng = _rng(20)
    b = _Worst('unit triangle')
    with mp.workdps(DPS):
        for _ in range(400):
            p, q = _triangle_pq(rng)
            got = X.unit_triangle(p, q)
            b.add(*_err(got, mp_dd([0j, p, q]),
                        mp_dd([0.0, p.real, q.real])), case=(p, q))
    b.check()
    assert b.max_rel <= 1e-13, b.max_rel


def test_unit_triangle_matches_the_legacy_closed_form_where_it_is_safe():
    for p, q in ((0.3 + 0.1j, -1.2 + 2j), (-2.0, -3.5), (1j, 2j)):
        assert X.unit_triangle(p, q) == pytest.approx(
            FI._exp_over_unit_triangle(p, q), rel=1e-13)


# ═══ (3) triangles on big boxes ════════════════════════════════════════
def _mp_triangle(V, al, be, ga=0j, real=False):
    with mp.workdps(DPS):
        x = [mp.mpf(v[0]) for v in V]
        y = [mp.mpf(v[1]) for v in V]
        a, b_, g = _mpc(al), _mpc(be), _mpc(ga)
        if real:
            a, b_, g = mp.mpf(a.real), mp.mpf(b_.real), mp.mpf(g.real)
        det = abs((x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]))
        return det * mp_dd([a * x[i] + b_ * y[i] + g for i in range(3)])


def test_triangle_big_boxes_never_bail():
    """cfx_accuracy (3): fast poles on boxes up to 2000 wide.  The legacy
    routine bails (``None``) above |Re| 600; expdd only on a true overflow."""
    rng = _rng(30)
    b = _Worst('triangle big boxes')
    legacy_bails = 0
    for _ in range(300):
        cap = rng.choice([6, 12, 200, 2000])
        V = [tuple(rng.uniform(-cap, 0.5, 2)) for _ in range(3)]
        al = complex(rng.choice([12, 24, 0.6, 36]) * rng.choice([1, -1, 0.5]),
                     rng.normal() * 0.3)
        be = complex(rng.choice([12, 24, 0.6, 36]) * rng.choice([1, -1, 0.5]),
                     rng.normal() * 0.3)
        ref = _mp_triangle(V, al, be)
        if ref != 0 and mp.log(abs(ref)) > X.LOG_OVERFLOW:
            assert X.triangle(*V, al, be) is None
            continue
        legacy_bails += FI._exp_over_triangle(*V, al, be) is None
        with mp.workdps(DPS):
            b.add(*_err(_mpval(X.triangle_log(*V, al, be)), ref,
                        _mp_triangle(V, al, be, real=True)), case=(V, al, be))
    b.check()
    assert legacy_bails > 0          # the battery does reach the old guard


def test_triangle_gamma_and_degenerate():
    V = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    assert X.triangle(*V, 0.3, -0.2, 2 + 1j) == pytest.approx(
        complex(_mp_triangle(V, 0.3, -0.2, 2 + 1j)), rel=1e-14)
    assert X.triangle((0, 0), (1, 1), (2, 2), 1.0, 2.0) == 0j


# ═══ (4) chain simplex, finite L ═══════════════════════════════════════
_CHAIN_FAMILIES = {
    'generic': lambda rng, m: [complex(rng.normal() * 3, rng.normal())
                               for _ in range(m)],
    'single-pole degenerate (OU)': lambda rng, m: [
        complex(rng.choice([-1, 1, 2, -2, 0]) * 1.0, 0) for _ in range(m)],
    'close pairs 1e-3': lambda rng, m: (lambda b: [
        b[i // 2] * (1 if i % 2 else -1) + (1e-3 * rng.normal() if i % 2 else 0)
        for i in range(m)])([complex(rng.normal(), rng.normal())
                             for _ in range(m)]),
    'fast poles (12..36)': lambda rng, m: [
        complex(rng.choice([12, 24, -12, 36]), 0) for _ in range(m)],
}


@pytest.mark.parametrize('family', sorted(_CHAIN_FAMILIES))
def test_chain_simplex_families(family):
    """cfx_accuracy (4): the prototype measured close pairs at 2.7e-14."""
    rng = _rng(40 + sorted(_CHAIN_FAMILIES).index(family))
    b = _Worst(f'chain {family}')
    for _ in range(40):
        m = int(rng.integers(2, 7))
        al = _CHAIN_FAMILIES[family](rng, m)
        L = -float(rng.choice([5, 50, 200]))
        U = float(rng.uniform(-0.5, 2))
        st, ls, mant = X.chain_simplex_log(al, L, U)
        if st == X.OVERFLOW:
            with mp.workdps(DPS):
                assert mp.log(abs(mp_chain(al, L, U))) > X.LOG_OVERFLOW - 1e-9
            continue
        _check_chain(b, al, L, U)
    b.check()


def test_chain_simplex_edge_cases():
    assert X.chain_simplex([], -1.0, 2.0) == 1.0
    assert X.chain_simplex([1.0, 2.0], 1.0, 1.0) == 0j
    assert X.chain_simplex([1.0, 2.0], 2.0, 1.0) == 0j
    # m = 1: (e^{aU} - e^{aL}) / a
    a, L, U = -0.7 + 0.4j, -3.0, 1.25
    ref = (np.exp(a * U) - np.exp(a * L)) / a
    assert X.chain_simplex([a], L, U) == pytest.approx(ref, rel=1e-14)


def test_chain_simplex_agrees_with_the_transfer_core():
    """The dd form (T^m exp[v]) and the transfer core (no cuts) are two
    evaluations of the same integral."""
    rng = _rng(41)
    for _ in range(30):
        m = int(rng.integers(1, 9))
        al = list(rng.normal(size=m) * 2 + 1j * rng.normal(size=m))
        L, U = -float(rng.uniform(1, 50)), float(rng.uniform(0, 2))
        st, sig, mant = X._chain_transfer_core(
            np.asarray(al, dtype=np.complex128), L, U,
            np.zeros(0), np.zeros(0, dtype=np.int64))
        assert st == X.OK
        _, ls, m1 = X.chain_simplex_log(al, L, U)
        _, lr, mr = X.chain_simplex_log(_real(al), L, U)
        a = m1 * math.exp(ls - sig)              # both in units of e^sig
        absref = abs(mr) * math.exp(lr - sig)
        assert abs(a - mant) <= max(1e-12 * abs(mant), 1e-14 * absref)


# ═══ (5) semi-infinite chains ══════════════════════════════════════════
def test_chain_semi_infinite_vs_far_L():
    """cfx_accuracy (5): L = -inf vs mpmath at an L far enough that
    e^{-800} is below double resolution (prototype: 8.9e-16)."""
    rng = _rng(50)
    b = _Worst('L=-inf')
    while b.n < 60:
        m = int(rng.integers(1, 7))
        al = [complex(rng.uniform(0.1, 30), rng.normal())
              * (1 if k == 0 else rng.choice([1, 1, -0.3])) for k in range(m)]
        P = np.cumsum(al)
        if any(p.real <= 0.05 for p in P):
            continue
        U = float(rng.uniform(-1, 1))
        L = -800.0 / min(p.real for p in P)
        with mp.workdps(DPS):
            b.add(*_err(_mpval(X.chain_simplex_log(al, -math.inf, U)),
                        mp_chain(al, L, U),
                        mp_chain(_real(al), L, U)), case=(al, U))
    b.check()
    # regression guard below the M6 rule: what is left is the rounding of the
    # exponent S*U in double, eps*|S U| ~ 2e-14 at |S U| ~ 180 (the mpmath
    # reference forms it exactly)
    assert b.max_rel <= 1e-13, b.max_rel


def test_chain_semi_infinite_divergence_is_explicit():
    for al in ([1.0, -2.0], [-0.5], [0.0, 1.0], [1.0, -1.0 + 1j]):
        st, ls, mant = X.chain_simplex_log(al, -math.inf, 0.0)
        assert st == X.DIVERGENT and mant == 0
        with pytest.raises(X.DivergentIntegralError):
            X.chain_simplex(al, -math.inf, 0.0)
        with pytest.raises(X.DivergentIntegralError):
            X.chain_with_uppers(al + [1.0], -math.inf, {0: -1.0}, 0.0)


def test_semi_infinite_tiny_prefix_sums_are_values():
    """Prefix sums down to subnormal: no overflow inside, and the value (e.g.
    e^-1000 * 1e397 ~ e^-86 for [1e-200, 0, 1000]) is returned, not 0."""
    for al, U in (([1e-200, 0.0, 1000.0], -1.0), ([1e-320, 1000.0], -1.0),
                  ([1e-160, 1e-160, 1e-160], 0.0)):
        got = X.chain_simplex(al, -math.inf, U)
        with mp.workdps(DPS):
            P = [mp.mpf(0)]
            for a in al:
                P.append(P[-1] + mp.mpf(a))
            ref = mp.exp(P[-1] * U)
            for p in P[1:]:
                ref /= p
            if mp.log(abs(ref)) > X.LOG_OVERFLOW:
                assert got is None
                continue
            assert got is not None and got != 0
            assert abs(_mpc(got) - ref) <= 1e-13 * abs(ref), (al, got, ref)


def test_chain_semi_infinite_extreme_prefix_sums():
    """Tiny and huge prefix sums: the start vector is renormalised, so the
    value stays exact instead of overflowing 1/prod(P_j)."""
    al = [1e-30] * 12
    st, ls, mant = X.chain_simplex_log(al, -math.inf, 0.0)
    assert st == X.OVERFLOW                 # 1/prod(P) ~ 1e360 > e^700
    al = [1e-25] + [0.0] * 5
    ref = mp.mpf(1) / mp.mpf(1e-25) ** 6       # prod_j P_j = (1e-25)^6
    st, ls, mant = X.chain_simplex_log(al, -math.inf, 0.0)
    assert st == X.OK
    assert abs(ls + math.log(abs(mant)) - float(mp.log(ref))) < 1e-13


# ═══ (6) chains with intermediate uppers ═══════════════════════════════
def _random_uppers_case(rng):
    m = int(rng.integers(2, 6))
    al = [complex(rng.normal() * 2, rng.normal()) for _ in range(m)]
    L = -float(rng.choice([3, 10]))
    U_top = float(rng.uniform(0, 2))
    upp = {k: float(rng.uniform(-2, 1.5)) for k in range(m - 1)
           if rng.random() < 0.4}
    return al, L, upp, U_top


def test_uppers_vs_legacy_cut_enumeration():
    """cfx_accuracy (6): transfer matrix vs the production cut enumeration on
    300 cases (prototype: 1.2e-12)."""
    rng = _rng(60)
    worst = 0.0
    for _ in range(300):
        al, L, upp, U_top = _random_uppers_case(rng)
        o = FI._chain_with_intermediate_uppers_uncached(al, L, upp, U_top)
        nw = X.chain_with_uppers(al, L, upp, U_top)
        if o is None:
            continue
        worst = max(worst, abs(nw - o) / max(abs(o), 1e-14))
    assert worst <= 1e-11, worst


def test_uppers_vs_mpmath():
    rng = _rng(61)
    b = _Worst('uppers vs mpmath')
    for _ in range(60):
        al, L, upp, U_top = _random_uppers_case(rng)
        with mp.workdps(DPS):
            b.add(*_err(_mpval(X.chain_with_uppers_log(al, L, upp, U_top)),
                        mp_uppers(al, L, upp, U_top),
                        mp_uppers(_real(al), L, upp, U_top)),
                  case=(al, L, upp, U_top))
    b.check()


def test_uppers_nested_quadrature():
    """Independent of every closed form: m = 2 and 3 with one intermediate
    upper by nested mpmath quadrature."""
    rng = _rng(62)

    def inner(a, L, x):                 # int_L^x e^{a s} ds, exactly
        return (mp.exp(a * x) - mp.exp(a * L)) / a

    with mp.workdps(20):
        for _ in range(2):
            al = [_mpc(complex(rng.normal(), rng.normal() * 0.5))
                  for _ in range(2)]
            L, U_top, u0 = -2.0, 1.0, float(rng.uniform(-1, 0.5))
            ref = mp.quad(lambda s2: mp.exp(al[1] * s2)
                          * inner(al[0], L, min(s2, u0)), [L, u0, U_top])
            got = X.chain_with_uppers([complex(a) for a in al], L, {0: u0},
                                      U_top)
            assert abs(got - complex(ref)) <= 1e-12 * abs(complex(ref))
        al = [_mpc(complex(rng.normal(), rng.normal() * 0.5)) for _ in range(3)]
        L, U_top, u1 = -2.0, 1.0, float(rng.uniform(-1, 0.5))
        ref = mp.quad(lambda s3: mp.exp(al[2] * s3) * mp.quad(
            lambda s2: mp.exp(al[1] * s2) * inner(al[0], L, s2),
            [L, min(s3, u1)]), [L, u1, U_top])
        got = X.chain_with_uppers([complex(a) for a in al], L, {1: u1}, U_top)
        assert abs(got - complex(ref)) <= 1e-12 * abs(complex(ref))


def test_uppers_semi_infinite_vs_far_L():
    """cfx_accuracy (6, last): L = -inf with uppers vs L = -60 (every prefix
    sum decays at rate >= 1, so the far-L remainder is ~e^-60)."""
    rng = _rng(63)
    worst = 0.0
    for _ in range(100):
        m = int(rng.integers(2, 6))
        al = [complex(rng.uniform(1, 20), rng.normal()) for _ in range(m)]
        upp = {k: float(rng.uniform(-1, 1)) for k in range(m - 1)
               if rng.random() < 0.5}
        a = X.chain_with_uppers(al, -math.inf, upp, 1.5)
        b = X.chain_with_uppers(al, -60.0, upp, 1.5)
        worst = max(worst, abs(a - b) / abs(b))
    assert worst <= 1e-12, worst


def test_uppers_tiny_segments_long_chains():
    """Verification finding F1: many variables forced into a window of a few
    ulps (or extreme scales).  The plain propagator's dt^(i-j) factors
    underflowed and the projection left an exact 0 (or, with too few Taylor
    terms, a wrong value); the graded frame gets the value."""
    cases = [([700 / 21] * 21, 1.0, {19: 1.0 + 4 * 2.0 ** -52}, 1.5),
             ([700 / 41] * 41, 1.0, {39: 1.0 + 1e-8}, 1.5),
             ([1e68] * 5, 1e-66, {3: 1e-66 + 1e-70}, 1e-66 + 2e-69)]
    for al, L, u, U in cases:
        with mp.workdps(DPS):
            ref = mp_uppers(al, L, u, U)
            got = _mpval(X.chain_with_uppers_log(al, L, u, U))
            assert abs(got - ref) <= 1e-11 * abs(ref), (len(al), got, ref)


def test_long_divided_differences_beyond_sixteen_nodes():
    """n = 17..30 near a cluster (expm branch, few squarings): every Taylor
    term an entry far below the diagonal needs is kept."""
    rng = _rng(64)
    b = _Worst('dd n=17..30')
    for n in (17, 21, 30):
        for _ in range(2):
            _check_dd(b, list(0.4 * (rng.normal(size=n + 1)
                                     + 1j * rng.normal(size=n + 1))))
    b.check()
    assert b.max_rel <= 1e-13, b.max_rel


def test_nearly_collinear_triangle():
    """Verification finding F4: the determinant is exact, so a nearly
    collinear triangle does not inherit its cancellation error."""
    V = [(3.5046916201295613, 10.357800930024487),
         (4.5696843641100875, 9.85002695630001),
         (5.634677108224429, 9.342252982275832)]
    a = 13.93291430571367 + 0.40721448348643774j
    be = -24.51226320276745 + 0.6298220620922315j
    g = 22.449500945465502 + 0.106044467798903j
    ref = _mp_triangle(V, a, be, g)
    assert abs(_mpc(X.triangle(*V, a, be, g)) - ref) <= 1e-13 * abs(ref)


def test_a_nan_upper_raises():
    """Verification finding F5: ``min(u, nan)`` would ignore it silently."""
    with pytest.raises(ValueError, match='NaN upper'):
        X.chain_with_uppers([1.0, 1.0], 0.0, {0: math.nan}, 1.0)


def _captured_calls():
    from tests.test_chain_uppers_memo import CALLS
    return CALLS


def test_uppers_semantics_on_the_captured_calls():
    """The 320 real calls of ``_chain_with_intermediate_uppers`` (M4 fixture,
    legacy memo-off values): transfer agrees to <= 1e-11 relative."""
    worst = 0.0
    for a, L, u, U, ref in _captured_calls():
        got = X.chain_with_uppers(a, L, dict(u), U)
        assert got is not None
        worst = max(worst, abs(got - ref) / abs(ref))
    assert worst <= 1e-11, worst


def test_uppers_argument_conventions():
    a = [-0.3 + 0.1j, -0.5 + 0.0j, -0.7 - 0.2j]
    f = X.chain_with_uppers
    # empty chain, missing / None uppers
    assert f([], -5.0, {0: -1.0}, 0.0) == 1.0
    assert f(a, -5.0, None, 0.0) == f(a, -5.0, {}, 0.0)
    assert f(a, -5.0, {0: None, 1: -1.0}, 0.0) == f(a, -5.0, {1: -1.0}, 0.0)
    # keys outside the chain are ignored
    assert f(a, -5.0, {7: -4.0, -1: -4.0}, 0.0) == f(a, -5.0, {}, 0.0)
    # an upper at or below L (a tie included) empties the domain
    assert f(a, -5.0, {1: -5.0}, 0.0) == 0j
    assert f(a, -5.0, {0: -6.0}, 0.0) == 0j
    assert f(a, -5.0, {}, -5.0) == 0j
    # an upper on the top position lowers the top; one above it is inert
    assert f(a, -5.0, {2: -1.0}, 0.0) == pytest.approx(
        X.chain_simplex(a, -5.0, -1.0), rel=1e-14)
    assert f(a, -5.0, {1: 3.0}, 0.0) == f(a, -5.0, {}, 0.0)
    # the same conventions as the legacy routine and the mpmath reference,
    # case by case ({0: -4.999}: legacy is off by 1e-12 there, expdd 1e-16)
    for upp in ({0: None, 1: -1.0}, {0: -2.0, 1: -2.0}, {0: -1.0, 1: -2.0},
                {1: -5.0}, {2: -1.0, 0: -3.0}, {0: -4.999}):
        got = f(a, -5.0, upp, 0.0)
        leg = FI._chain_with_intermediate_uppers_uncached(a, -5.0, upp, 0.0)
        assert got == pytest.approx(leg, rel=1e-11, abs=1e-300), upp
        with mp.workdps(DPS):
            ref = complex(mp_uppers(a, -5.0, upp, 0.0))
        assert got == pytest.approx(ref, rel=1e-14, abs=1e-300), upp


# ═══ cs_bench: captured degenerate calls, synthetic block sums ═════════
def _degenerate_calls():
    z = np.load(os.path.join(os.path.dirname(__file__), 'fixtures',
                             'expdd_degenerate_chain_calls.npz'))
    return [([complex(x) for x in z['alphas'][i, :z['lens'][i]]],
             float(z['L'][i]), float(z['U'][i]), int(z['source'][i]))
            for i in range(len(z['lens']))]


def test_captured_degenerate_chain_calls():
    """cs_bench [A]: 200 real degenerate chain calls (ou_quartic_two_dim k=2
    l=2, ou_quartic k=4 l=2; every block sum of the alphas that vanishes is a
    repeated node)."""
    calls = _degenerate_calls()
    assert len(calls) == 200 and {s for *_, s in calls} == {0, 1}
    b = _Worst('captured degenerate chains')
    for al, L, U, _ in calls:
        _check_chain(b, al, L, U)
    b.check()
    assert b.max_rel <= 1e-12, b.max_rel


def _block_sum_cases():
    out = []
    for d in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1):
        out += [
            ([1.0, -1.0 + d, 1.0, -1.0, 1.0], -50.0, 0.0),
            ([-1.0, -1.0 + d, -1.0 - d, -1.0 + 2 * d], -50.0, 0.0),
            ([0.5 + 2j, -2j + d, 0.3 + 1j, -1j - 0.3 + d, -0.5], -50.0, 0.0),
            ([3.0, -3.0 + d] * 3, -2.0, 0.0),
            ([5.0, -5.0 + d, 5.0, -5.0], -200.0, 0.0),
        ]
    return out


def test_block_sum_near_degenerate_chains():
    """cs_bench [B]: alphas with a block sum = delta, delta from 0 to 0.1."""
    b = _Worst('block sums')
    for al, L, U in _block_sum_cases():
        _check_chain(b, al, L, U)
    b.check()


def test_random_non_degenerate_chains():
    """cs_bench [C]: random complex alphas m = 3..7, L in [-50, -1]."""
    rng = _rng(70)
    b = _Worst('random chains')
    for _ in range(100):
        m = int(rng.integers(3, 8))
        al = list(rng.normal(size=m) * 2 + 1j * rng.normal(size=m))
        _check_chain(b, al, -float(rng.uniform(1, 50)), float(rng.uniform(0, 2)))
    b.check()


# ═══ the three gaps the prototype never measured ═══════════════════════
@pytest.mark.parametrize('n', [1, 2, 3, 5, 8])
def test_gap_oscillatory_nodes(n):
    """Re z in [-1, 0], |Im z| in [10, 1000]: the value cancels to
    ~|Im|^-n while the absolute-value integral stays ~1/n!."""
    rng = _rng(200 + n)
    b = _Worst(f'oscillatory n={n}')
    for _ in range(15):
        _check_dd(b, _gen_oscillatory(rng, n))
    b.check()


def test_gap_oscillatory_chains():
    """Chains whose alphas oscillate fast (|Im alpha| in [10, 1000])."""
    rng = _rng(210)
    b = _Worst('oscillatory chains')
    for _ in range(20):
        m = int(rng.integers(2, 9))
        al = list(-rng.uniform(0, 1, m) + 1j * rng.uniform(10, 1000, m)
                  * rng.choice([-1, 1], m))
        _check_chain(b, al, -float(rng.uniform(1, 5)), float(rng.uniform(0, 1)))
    b.check()


@pytest.mark.parametrize('n', [8, 12, 16])
@pytest.mark.parametrize('kind', ['cluster', 'near-cluster', 'spread'])
def test_gap_long_divided_differences(n, kind):
    rng = _rng(300 + n + 1000 * ['cluster', 'near-cluster', 'spread'].index(kind))
    b = _Worst(f'long {kind} n={n}')
    for _ in range(4 if n == 16 else 6):
        _check_dd(b, _gen_long(rng, n, kind))
    b.check()


def test_gap_long_oscillatory_divided_differences():
    rng = _rng(320)
    b = _Worst('long oscillatory')
    for n in (8, 12, 16):
        for _ in range(3):
            _check_dd(b, _gen_oscillatory(rng, n))
    b.check()


def test_gap_long_chains():
    """Chains of m = 8..16: OU-like degenerate (clustered nodes) and spread
    complex alphas."""
    rng = _rng(330)
    b = _Worst('long chains')
    for m in (8, 11, 16):
        for _ in range(2):
            ou = [complex(rng.choice([-1.0, 1.0]), 0) for _ in range(m)]
            _check_chain(b, ou, -float(rng.uniform(5, 30)), 0.5)
            sp = list(rng.normal(size=m) * 3 + 1j * rng.normal(size=m))
            _check_chain(b, sp, -float(rng.uniform(1, 10)), 0.5)
    b.check()


@pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 12])
@pytest.mark.parametrize('real', [False, True], ids=['complex', 'real'])
def test_gap_mixed_magnitudes(n, real):
    rng = _rng(400 + n + 100 * real)
    b = _Worst(f'mixed magnitudes n={n} real={real}')
    for _ in range(12 if n <= 5 else 6):
        _check_dd(b, _gen_mixed(rng, n, real))
    b.check()


# ═══ overflow and invalid input are explicit ═══════════════════════════
def test_true_overflow_is_reported_not_inf():
    assert X.dd_exp([800.0, 0.0]) is None
    c, mant = X.log_dd_exp([800.0, 0.0])
    assert c == 800.0 and math.isfinite(abs(mant))
    assert X.triangle((0, 0), (1, 0), (0, 1), 720.0, 0.0) is None
    assert X.chain_simplex([10.0, 10.0], 0.0, 40.0) is None
    st, ls, mant = X.chain_with_uppers_log([10.0, 10.0, 10.0], 0.0,
                                           {0: 30.0}, 40.0)
    assert st == X.OVERFLOW


def test_a_large_scale_with_a_small_mantissa_is_a_value():
    """log|value| decides, not the scale: exp[705, 0, 0, 0] ~ e^705/705^3
    has c = 705 > LOG_OVERFLOW but is a finite value."""
    c, mant = X.log_dd_exp([705.0, 0.0, 0.0, 0.0])
    assert c > X.LOG_OVERFLOW
    got = X.dd_exp([705.0, 0.0, 0.0, 0.0])
    assert got is not None and math.isfinite(abs(got))
    with mp.workdps(DPS):
        assert got == pytest.approx(complex(mp_dd([705.0, 0.0, 0.0, 0.0])),
                                    rel=1e-13)
    V = ((0.0, 0.0), (1e-5, 0.0), (0.0, 1e-5))      # |det| = 1e-10
    got = X.triangle(*V, 0.0, 0.0, 705.0)
    assert got == pytest.approx(complex(_mp_triangle(V, 0, 0, 705.0)), rel=1e-13)
    # beyond the legacy guard (|Re| > 600) but representable
    V = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    got = X.triangle(*V, 650.0, -30.0)
    assert got == pytest.approx(complex(_mp_triangle(V, 650.0, -30.0)), rel=1e-13)
    assert FI._exp_over_triangle(*V, 650.0, -30.0) is None


def test_overflow_boundary_is_consistent_with_mpmath():
    """Near e^700: every finite return matches mpmath; every OVERFLOW is a
    reference above the threshold."""
    rng = _rng(80)
    for _ in range(60):
        n = int(rng.integers(1, 6))
        z = list(rng.uniform(690, 710, n + 1)
                 + 1j * rng.normal(size=n + 1) * 10)
        got = X.dd_exp(z)
        with mp.workdps(DPS):
            ref = mp_dd(z)
            if got is None:
                assert mp.log(abs(ref)) > X.LOG_OVERFLOW - 1e-9
            else:
                assert math.isfinite(abs(got))
                assert abs(_mpc(got) - ref) <= 1e-12 * abs(ref)


def test_finish_scales_in_steps_and_never_returns_non_finite():
    """A subnormal mantissa with a log scale above 1400 is a finite value
    (no intermediate overflow); a non-finite intermediate raises."""
    for ls, mant in ((1400.0, 5e-320 + 0j), (705.0, 1e-5 - 2e-5j),
                     (-800.0, 0.3 + 0j), (2100.0, 0j),
                     (-779.0, 2.0 ** 1000 * (0.6 - 0.3j)),   # e^-86: a value
                     (-1100.0, 2.0 ** 500 + 0j), (-2000.0, 1e300 + 0j)):
        st = X._status(ls, mant)
        got = X._finish(st, ls, mant)
        with mp.workdps(DPS):
            ref = mp.exp(mp.mpf(ls)) * _mpc(mant)
            if st == X.OVERFLOW:
                assert got is None and mp.log(abs(ref)) > X.LOG_OVERFLOW
                continue
            assert math.isfinite(abs(got))
            assert abs(_mpc(got) - ref) <= 1e-14 * abs(ref) + 1e-320
    for ls, mant in ((math.nan, 1 + 0j), (0.0, complex(math.nan, 0)),
                     (math.inf, 1 + 0j), (0.0, complex(0, math.inf))):
        with pytest.raises(FloatingPointError):
            X._status(ls, mant)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf,
                                 complex(0, math.nan)])
def test_non_finite_input_raises(bad):
    with pytest.raises(ValueError):
        X.log_dd_exp([0.0, bad])
    with pytest.raises(ValueError):
        X.chain_simplex([1.0, bad], -1.0, 0.0)
    with pytest.raises(ValueError):
        X.triangle((0, 0), (1, 0), (0, 1), bad, 0.0)


def test_bad_limits_raise():
    with pytest.raises(ValueError):
        X.chain_simplex([1.0], -1.0, math.inf)
    with pytest.raises(ValueError):
        X.chain_simplex([1.0], math.nan, 0.0)
    with pytest.raises(ValueError):
        X.log_dd_exp([])


# ═══ (B) backends ══════════════════════════════════════════════════════
def _backend_battery():
    """The inputs of the batteries above, regenerated (no references)."""
    out = []
    rng = _rng(900)
    for n in range(1, 8):
        out += [('dd', _gen_nodes(rng, n)) for _ in range(12)]
    for n in (1, 3, 5, 8, 12, 16):
        out.append(('dd', _gen_oscillatory(rng, n)))
        out.append(('dd', _gen_mixed(rng, n, False)))
        for kind in ('cluster', 'near-cluster', 'spread'):
            out.append(('dd', _gen_long(rng, n, kind)))
    for _ in range(30):
        out.append(('tri', _triangle_pq(rng)))
    for fam in sorted(_CHAIN_FAMILIES):
        for _ in range(8):
            m = int(rng.integers(2, 7))
            out.append(('chain', (_CHAIN_FAMILIES[fam](rng, m),
                                  -float(rng.choice([5, 50])), 0.5)))
    out += [('chain', c) for c in _block_sum_cases()]
    out += [('chain', (al, L, U)) for al, L, U, _ in _degenerate_calls()[::5]]
    for _ in range(40):
        out.append(('uppers', _random_uppers_case(rng)))
    out += [('uppers', (a, L, u, U)) for a, L, u, U, _ in _captured_calls()[::8]]
    return out


def _evaluate(battery):
    """Per case ``(mant, log_scale, |absolute-value integral| / e^log_scale)``
    as JSON-friendly tuples (log domain, so nothing over- or underflows)."""
    res = []
    for kind, args in battery:
        if kind in ('dd', 'tri'):
            z = args if kind == 'dd' else (0j,) + tuple(args)   # J = exp[0,p,q]
            c, mant = X.log_dd_exp(z)
            cr, mr = X.log_dd_exp(_real(z))
        else:
            if kind == 'chain':
                al, L, U = args
                st, c, mant = X.chain_simplex_log(al, L, U)
                _, cr, mr = X.chain_simplex_log(_real(al), L, U)
            else:
                al, L, u, U = args
                st, c, mant = X.chain_with_uppers_log(al, L, u, U)
                _, cr, mr = X.chain_with_uppers_log(_real(al), L, u, U)
            if st != X.OK:
                res.append((None, st, 0.0))
                continue
        res.append(((mant.real, mant.imag), c, abs(mr) * math.exp(cr - c)))
    return res


_SUBPROCESS = r'''
import json, os, sys
sys.path.insert(0, {root!r})
import tests.test_expdd as T
assert T.X.BACKEND == {backend!r}, T.X.BACKEND
json.dump(T._evaluate(T._backend_battery()), sys.stdout)
'''


def _run_backend(env_extra, backend, block_numba=False):
    code = _SUBPROCESS.format(root=os.path.abspath(ROOT), backend=backend)
    if block_numba:
        code = "import sys; sys.modules['numba'] = None\n" + code
    env = dict(os.environ, PYTHONHASHSEED='0', **env_extra)
    out = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout)


def _compare(a, b, rel=1e-13):
    """Agreement to ``rel`` of the value, or ``rel * 1e-3`` of the
    absolute-value integral for a cancelling value; returns the worst
    value-relative difference."""
    worst = 0.0
    assert len(a) == len(b)
    for i, ((va, ca, sa), (vb, cb, sb)) in enumerate(zip(a, b)):
        if va is None or vb is None:             # a status: must agree
            assert (va, ca) == (vb, cb), i
            continue
        za = complex(*va)
        zb = complex(*vb) * math.exp(cb - ca)    # in units of e^ca
        d = abs(za - zb)
        if za != 0:
            worst = max(worst, d / abs(za))
        assert d <= rel * abs(za) or d <= rel * 1e-3 * sa, (i, za, zb)
    return worst


@pytest.mark.skipif(X.BACKEND != 'numba', reason='numba backend not active')
def test_numba_and_pure_python_agree():
    """(B) The same battery under NUMBA_DISABLE_JIT=1 agrees with the
    compiled kernels to 1e-13 (not bit for bit: FMA and libm)."""
    here = _evaluate(_backend_battery())
    py = _run_backend({'NUMBA_DISABLE_JIT': '1'}, 'python')
    _compare(here, py)


def test_module_works_without_numba():
    """With numba unimportable the module selects the pure-Python cores, and
    they agree with this process's backend."""
    here = _evaluate(_backend_battery())
    _compare(here, _run_backend({}, 'python', block_numba=True))

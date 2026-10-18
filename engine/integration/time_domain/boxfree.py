r"""Box-free exact Phase J integrals on the divided-difference kernels (M7).

``final_integral`` routes every Phase J closed form here (the only route
since audit W7 retired the boxed pre-0.2.0 routes and their switches
``USE_PHASE_J_DD_KERNELS``, ``SEMIINF_CHAIN``, ``UNBOUNDED_2D``).  Everything
is evaluated in the log
domain through ``expdd`` and folded with the constant part γ of the exponent
before any ``exp`` is taken, so the only failure left is a result that
truly exceeds the double range: ``log|value| > LOG_OVERFLOW`` gives the
status ``OVERFLOW`` (one one-sided test on the FINAL exponent), never a
silent ``inf``/``nan`` and never a bail on a large intermediate exponent.

Every routine returns ``(status, value)`` with ``status`` one of

* ``OK``         -- ``value`` is the integral (a complex; ``0j`` for an
                    empty or measure-zero region);
* ``OVERFLOW``   -- the integral itself exceeds ``e^LOG_OVERFLOW``;
* ``DIVERGENT``  -- an open direction does not decay (``Re(α·r) >= -tol`` on
                    a recession ray ``r``, a lineality direction, or a
                    semi-infinite chain prefix sum with ``Re <= tol``).  What
                    to do then is the divergence policy (M8), not this
                    module's.  ``tol`` (``>= 0``, given by the caller from
                    the scale of the exponents it summed) keeps a rounding
                    residue of an exponent that is exactly 0 (a direction in
                    which the integrand is constant) from passing as decay;
* ``UNSUPPORTED`` -- a shape this module does not take (the caller goes on
                    to its next route).

Interval (m = 1)
----------------
``∫_L^U exp(α s + γ) ds = (U - L) exp[αL + γ, αU + γ]`` (a first divided
difference, anchored at the end with the larger real exponent, so a tiny
``α`` costs no accuracy); ``L = -inf`` gives ``e^{αU+γ}/α`` (needs
``Re α > 0``), ``U = +inf`` gives ``-e^{αL+γ}/α`` (needs ``Re α < 0``).

Polygon (m = 2), possibly unbounded
-----------------------------------
``region_2d(rows)`` builds the region ``{s : a·s + c > 0 for every row}``
exactly, in rational arithmetic on the rows' floats (no tolerance, as every
emptiness test of Phase J): its finite vertices (pairwise intersections of
the boundary lines that satisfy every row) and, when it is unbounded, the
extreme rays of its recession cone ``{r : a·r >= 0 for every row}``.  A
genuinely thin region, however thin, is integrated (or found divergent); a
region of measure zero is empty.  Only the vertices are rounded, once, to
floats for the integration.  Ordered by angle around an
interior point, the finite vertices and the ideal vertices (the rays) form a
convex projective polygon; its fan from a finite vertex ``x0`` has three kinds
of pieces, each a closed form (``f(x) = α·x + γ``):

* triangle ``(x0, u, w)``:      ``|det(u - x0, w - x0)| exp[f(x0), f(u), f(w)]``;
* half-strip ``conv(x0, u) + R+ r``:
                                ``|det(u - x0, r)| exp[f(x0), f(u)] / (-α·r)``;
* wedge ``x0 + cone(r1, r2)``:  ``|det(r1, r2)| e^{f(x0)} / ((-α·r1)(-α·r2))``.

The ray formulas need ``Re(α·r) < 0`` on every ray used (``DIVERGENT``
otherwise).  The pieces are summed in the log domain and the overflow test
is made once, on the sum.  The divergence-theorem sweep formula of the M7 plan is kept as an
independent test oracle (``tests/test_phase_j_box_free.py``), not here.

Chain (m >= 3)
--------------
``chain_log`` is ``expdd.chain_with_uppers_log`` (finite or ``L = -inf``),
plus the mirror image for an open top (``U = +inf``):
``∫_{L<s_1<…<s_m<∞} e^{Σ α_k s_k} = ∫_{-∞<t_1<…<t_m<-L} e^{Σ (-α_{m+1-j}) t_j}``
(``UNSUPPORTED`` when that chain also carries an intermediate upper below
the open top; an upper at or below ``L`` gives 0, and an upper on the last
position just closes the top).  The chain's overflow status is that of the
chain alone; the caller folds γ and decides on the folded value.
"""
import cmath
import math
from fractions import Fraction
from functools import cmp_to_key

import numpy as np

from engine.integration.time_domain import expdd

OK = expdd.OK
OVERFLOW = expdd.OVERFLOW
DIVERGENT = expdd.DIVERGENT
UNSUPPORTED = 4
#: a chain state that underflowed below the double range (``expdd`` raises
#: ``FloatingPointError``); the caller treats it like ``UNSUPPORTED``
UNDERFLOW = 5

LOG_OVERFLOW = expdd.LOG_OVERFLOW

#: ``|α|`` below which an interval or ray exponent counts as 0 on an
#: infinite side (divergent), as the pre-0.2.0 interval routines' ``1e-15``
ALPHA_ZERO = 1e-15

_INF = math.inf


def _finite_complex(name, z):
    z = complex(z)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise ValueError(f'boxfree: {name} must be finite, got {z!r}')
    return z


# ───────────────────────────────────────────────────────────────── values

def exp_times(z, mant=1.0):
    """``(status, mant * e^z)`` for complex ``z`` and ``mant`` without an
    intermediate overflow; ``OVERFLOW`` iff ``log|value| > LOG_OVERFLOW``.
    A value below the double range underflows to 0 (benign).  A non-finite
    input raises ``ValueError``."""
    z = _finite_complex('exponent', z)
    mant = _finite_complex('mantissa', mant)
    if mant == 0:
        return OK, 0j
    zr = z.real
    lv = zr + math.log(abs(mant))
    if lv > LOG_OVERFLOW:
        return OVERFLOW, None
    if -LOG_OVERFLOW <= zr <= LOG_OVERFLOW:
        return OK, mant * cmath.exp(z)
    if lv < -2.0 * LOG_OVERFLOW:
        return OK, 0j
    return OK, (mant / abs(mant)) * cmath.exp(complex(lv, z.imag))


def _core(nodes):
    """``(c, mant)`` of ``exp[nodes]`` (``c = max Re``), on the compiled
    core of ``expdd``."""
    c, mant = expdd._log_dd_core(np.asarray(nodes, dtype=np.complex128))
    return float(c), complex(mant)


# ─────────────────────────────────────────────────────────────── interval

def _check_tol(tol):
    tol = float(tol)
    if not (tol >= 0.0 and math.isfinite(tol)):
        raise ValueError(f'boxfree: tol must be finite and >= 0, got {tol!r}')
    return tol


def interval(alpha, gamma, L, U, tol=0.0):
    """``(status, ∫_L^U exp(alpha s + gamma) ds)``; ``L`` may be ``-inf`` and
    ``U`` ``+inf`` (not both; an infinite side needs ``|Re alpha| > tol``
    with the decaying sign).  A NaN or misplaced infinite input raises
    ``ValueError``."""
    alpha = _finite_complex('alpha', alpha)
    gamma = _finite_complex('gamma', gamma)
    tol = _check_tol(tol)
    L = float(L)
    U = float(U)
    if math.isnan(L) or math.isnan(U) or L == _INF or U == -_INF:
        raise ValueError(f'boxfree.interval: bad limits ({L!r}, {U!r})')
    if not L < U:
        return OK, 0j
    L_inf = L == -_INF
    U_inf = U == _INF
    if L_inf and U_inf:
        return DIVERGENT, None
    if L_inf:
        if not alpha.real > tol or abs(alpha) < ALPHA_ZERO:
            return DIVERGENT, None
        return exp_times(alpha * U + gamma - cmath.log(alpha))
    if U_inf:
        if not alpha.real < -tol or abs(alpha) < ALPHA_ZERO:
            return DIVERGENT, None
        return exp_times(alpha * L + gamma - cmath.log(-alpha))
    T = U - L
    if not math.isfinite(T):
        raise ValueError(f'boxfree.interval: width {L!r}..{U!r} overflows')
    eL = alpha * L + gamma
    eU = alpha * U + gamma
    aT = alpha * T
    for z in (eL, eU, aT):
        _finite_complex('exponent', z)
    # anchor at the end with the larger real exponent; the gap is α·T
    if eL.real >= eU.real:
        c, mant = _core((0j, aT))
        anchor = eL
    else:
        c, mant = _core((-aT, 0j))
        anchor = eU
    return exp_times(anchor + (c + math.log(T)), mant)


# ──────────────────────────────────────────────────────────── 2D regions

class Region2D:
    """The exact region ``{s : a·s + c > 0 for every row}`` in 2D.

    ``kind``: ``'empty'`` (empty or of measure zero), ``'bounded'``,
    ``'unbounded'`` or ``'lineality'`` (contains a line: no closed form,
    ``DIVERGENT``).  ``reason`` (for 'empty'): ``'opposite'`` (two exactly
    opposite rows that do not overlap), ``'measure_zero'`` (a segment, a
    point or a ray), ``'infeasible'`` (no point at all).  ``vertices``: the
    finite vertices (floats), ``rays``: the extreme rays of the recession
    cone (unit vectors), ``pieces``: the fan of the projective polygon (see
    the module docstring), each one of ``('tri', x0, u, w, det)``,
    ``('strip', x0, u, r, det)``, ``('wedge', x0, r1, r2, det)`` with
    ``det`` the (positive) Jacobian."""

    __slots__ = ('kind', 'vertices', 'rays', 'pieces', 'reason', '_packed')

    def __init__(self, kind, vertices=(), rays=(), pieces=(), reason=None):
        self.kind = kind
        self.vertices = tuple(vertices)
        self.rays = tuple(rays)
        self.pieces = tuple(pieces)
        self.reason = reason
        self._packed = None

    def __repr__(self):
        return (f'Region2D({self.kind!r}, vertices={self.vertices!r}, '
                f'rays={self.rays!r}, n_pieces={len(self.pieces)}, '
                f'reason={self.reason!r})')


def _abs_det_exact(ax, ay, bx, by):
    """``|ax*by - ay*bx|`` from exact rationals, rounded once (a nearly
    collinear pair would otherwise carry the float cancellation)."""
    d = Fraction(ax) * Fraction(by) - Fraction(ay) * Fraction(bx)
    return abs(float(d))


# one copy (audit W7, RED-07): exactly opposite normals whose half-planes
# do not overlap, exact comparisons only
from engine.integration.time_domain.final_integral import (  # noqa: E402
    _opposite_rows_empty)


def _cross(u, w):
    return u[0] * w[1] - u[1] * w[0]


def region_2d(rows):
    """The ``Region2D`` of ``rows`` (``(a0, a1, c)`` triples of finite floats
    with nonzero normals: constant rows are the caller's, decided by the
    Θ(0) rule).  Every decision is exact (rational arithmetic)."""
    rows = [(float(a0), float(a1), float(c)) for (a0, a1, c) in rows]
    for r in rows:
        if not all(math.isfinite(x) for x in r):
            raise ValueError(f'boxfree.region_2d: non-finite row {r!r}')
        if r[0] == 0.0 and r[1] == 0.0:
            raise ValueError('boxfree.region_2d: a zero normal')
    try:
        return _region_2d(rows)
    except OverflowError as exc:
        raise ValueError(f'boxfree.region_2d: a vertex or Jacobian beyond '
                         f'the double range ({exc})') from None


def _region_2d(rows):
    if not rows:
        return Region2D('lineality')
    if _opposite_rows_empty(rows):
        return Region2D('empty', reason='opposite')
    R = [(Fraction(a0), Fraction(a1), Fraction(c)) for (a0, a1, c) in rows]
    b = (R[0][0], R[0][1])
    if all(_cross((a0, a1), b) == 0 for (a0, a1, _c) in R):
        # every normal parallel to b: a 1D problem along b (a_k = λ_k b)
        lo = hi = None
        for (a0, a1, c) in R:
            lam = a0 / b[0] if b[0] != 0 else a1 / b[1]
            w = -c / lam               # b·s > w (λ > 0) or b·s < w (λ < 0)
            if lam > 0:
                lo = w if lo is None or w > lo else lo
            else:
                hi = w if hi is None or w < hi else hi
        if lo is None or hi is None or lo < hi:
            return Region2D('lineality')
        return Region2D('empty', reason='measure_zero' if lo == hi
                        else 'infeasible')
    # finite vertices: feasible pairwise intersections of boundary lines
    verts = []
    n = len(R)
    for i in range(n):
        a0, a1, ci = R[i]
        for j in range(i + 1, n):
            c0, c1, cj = R[j]
            det = a0 * c1 - a1 * c0
            if det == 0:
                continue
            x = (-ci * c1 + cj * a1) / det
            y = (-a0 * cj + c0 * ci) / det
            if all(p0 * x + p1 * y + pc >= 0 for (p0, p1, pc) in R):
                if (x, y) not in verts:
                    verts.append((x, y))
    if not verts:
        # rank-2 normals: a nonempty region is pointed and has a vertex
        return Region2D('empty', reason='infeasible')
    # extreme rays of the recession cone (exact; unnormalised directions)
    cands = []
    for (a0, a1, _c) in R:
        for u in ((-a1, a0), (a1, -a0)):
            if all(p0 * u[0] + p1 * u[1] >= 0 for (p0, p1, _pc) in R):
                if not any(_cross(u, w) == 0 and u[0] * w[0] + u[1] * w[1] > 0
                           for w in cands):
                    cands.append(u)
    rays = []
    if cands:
        # the cone is pointed (rank-2 normals): its extreme rays are the
        # candidates with every other one on one side
        r_lo = next(u for u in cands if all(_cross(u, w) >= 0 for w in cands))
        r_hi = next(u for u in cands if all(_cross(u, w) <= 0 for w in cands))
        rays = [r_lo] if _cross(r_lo, r_hi) == 0 else [r_lo, r_hi]
    # positive area?  (the vertices plus a point along each ray)
    pts = verts + [(verts[0][0] + r[0], verts[0][1] + r[1]) for r in rays]
    p0 = pts[0]
    if not any(_cross((p[0] - p0[0], p[1] - p0[1]),
                      (q[0] - p0[0], q[1] - p0[1])) != 0
               for p in pts for q in pts):
        return Region2D('empty', reason='measure_zero')
    # cyclic boundary order by angle around an interior point q (exact):
    # the centroid of the vertices pushed along every ray
    k = len(verts)
    qx = sum(v[0] for v in verts) / k + sum(r[0] for r in rays)
    qy = sum(v[1] for v in verts) / k + sum(r[1] for r in rays)
    items = [('v', v, (v[0] - qx, v[1] - qy)) for v in verts]
    items += [('r', r, r) for r in rays]

    def half(d):
        return 0 if (d[1] > 0 or (d[1] == 0 and d[0] > 0)) else 1

    def cmp(a, b):
        ha, hb = half(a[2]), half(b[2])
        if ha != hb:
            return ha - hb
        cr = _cross(a[2], b[2])
        return -1 if cr > 0 else (1 if cr < 0 else 0)

    items.sort(key=cmp_to_key(cmp))
    k0 = next(i for i, it in enumerate(items) if it[0] == 'v')
    items = items[k0:] + items[:k0]

    def fv(v):
        return (float(v[0]), float(v[1]))

    def fr(r):
        x, y = float(r[0]), float(r[1])
        h = math.hypot(x, y)
        return (x / h, y / h)

    x0 = items[0][1]
    pieces = []
    for i in range(1, len(items) - 1):
        (ki, pi, _di), (kj, pj, _dj) = items[i], items[i + 1]
        if ki == 'v' and kj == 'v':
            det = abs(float(_cross((pi[0] - x0[0], pi[1] - x0[1]),
                                   (pj[0] - x0[0], pj[1] - x0[1]))))
            if det > 0.0:
                pieces.append(('tri', fv(x0), fv(pi), fv(pj), det))
        elif ki == 'v' or kj == 'v':
            u, r = (pi, pj) if ki == 'v' else (pj, pi)
            rf = fr(r)
            det = abs(float(_cross((u[0] - x0[0], u[1] - x0[1]), r)))
            det /= math.hypot(float(r[0]), float(r[1]))
            if det > 0.0:
                pieces.append(('strip', fv(x0), fv(u), rf, det))
        else:
            r1, r2 = fr(pi), fr(pj)
            det = abs(float(_cross(pi, pj))) / (
                math.hypot(float(pi[0]), float(pi[1]))
                * math.hypot(float(pj[0]), float(pj[1])))
            if det > 0.0:
                pieces.append(('wedge', fv(x0), r1, r2, det))
    if not pieces:
        return Region2D('empty', reason='measure_zero')
    return Region2D('unbounded' if rays else 'bounded',
                    [fv(v) for v in verts], [fr(r) for r in rays], pieces)


def region_integral(region, alpha, beta, gamma, tol=0.0):
    """``(status, ∫_region exp(alpha s0 + beta s1 + gamma))`` over a
    ``Region2D``; every ray needs ``Re(α·r) < -tol``.  The fan pieces are
    summed in the log domain and the overflow test is made once, on the
    sum."""
    kind = region.kind
    if kind == 'empty':
        return OK, 0j
    if kind == 'lineality':
        return DIVERGENT, None
    alpha = _finite_complex('alpha', alpha)
    beta = _finite_complex('beta', beta)
    gamma = _finite_complex('gamma', gamma)
    tol = _check_tol(tol)
    for (rx, ry) in region.rays:
        ar = alpha * rx + beta * ry
        if not ar.real < -tol or abs(ar) < ALPHA_ZERO:
            return DIVERGENT, None
    terms = [_piece_log(piece, alpha, beta, gamma) for piece in region.pieces]
    for (ls, m) in terms:
        if not math.isfinite(ls):
            raise ValueError('boxfree.region_integral: a non-finite exponent')
        if m == 0 and ls > LOG_OVERFLOW:
            # the divided difference underflowed under a huge anchor: the
            # value cannot be told apart from an overflow
            return OVERFLOW, None
    terms = [(ls, m) for (ls, m) in terms if m != 0]
    if not terms:
        return OK, 0j
    top = max(ls for (ls, _m) in terms)
    acc = 0j
    for (ls, m) in terms:
        acc += m * math.exp(ls - top)
    if acc == 0:
        return OK, 0j
    return exp_times(complex(top, 0.0), acc)


def _anchored_nodes(pts, alpha, beta, gamma):
    """``(anchor, nodes)`` for the exponents ``f(p) = α·p + γ`` at ``pts``:
    the anchor is ``f`` at the point with the largest real exponent, formed
    directly, and the nodes are the gaps to it, formed from the edge vectors
    out of that point (two large exponents are never subtracted)."""
    rel = [alpha * (p[0] - pts[0][0]) + beta * (p[1] - pts[0][1])
           for p in pts]
    k = max(range(len(pts)), key=lambda i: rel[i].real)
    pk = pts[k]
    nodes = [alpha * (p[0] - pk[0]) + beta * (p[1] - pk[1]) for p in pts]
    nodes[k] = 0j
    return alpha * pk[0] + beta * pk[1] + gamma, nodes


def _piece_log(piece, alpha, beta, gamma):
    """``(log_scale, mant)`` with the piece's integral ``mant · e^log_scale``
    (``log_scale`` real; the phase is in ``mant``)."""
    tag = piece[0]
    x0 = piece[1]
    if tag == 'tri':
        _t, _x0, u, w, det = piece
        anchor, nodes = _anchored_nodes((x0, u, w), alpha, beta, gamma)
        c, mant = _core(nodes)
        return (anchor.real + c + math.log(det),
                mant * cmath.exp(1j * anchor.imag))
    if tag == 'strip':
        _t, _x0, u, r, det = piece
        ar = alpha * r[0] + beta * r[1]
        anchor, nodes = _anchored_nodes((x0, u), alpha, beta, gamma)
        c, mant = _core(nodes)
        lg = cmath.log(-ar)
        return (anchor.real + c + math.log(det) - lg.real,
                mant * cmath.exp(1j * (anchor.imag - lg.imag)))
    _t, _x0, r1, r2, det = piece
    f0 = alpha * x0[0] + beta * x0[1] + gamma
    l1 = cmath.log(-(alpha * r1[0] + beta * r1[1]))
    l2 = cmath.log(-(alpha * r2[0] + beta * r2[1]))
    return (f0.real + math.log(det) - l1.real - l2.real,
            cmath.exp(1j * (f0.imag - l1.imag - l2.imag)))


# ──────────────────────────────────────────────────────────────── chains

def chain_log(alphas, L, upper_per_position, U, tol=0.0):
    """``(status, log_scale, mant)`` of the chain integral (the conventions
    of the pre-0.2.0 ``final_integral._chain_with_intermediate_uppers``: keys outside ``0..m-1`` and ``None`` uppers are absent, an
    effective upper at or below ``L`` gives 0, an empty chain is 1); ``L``
    may be ``-inf``, ``U`` may be ``+inf``.  With an open top the chain may
    carry no finite upper below the top (``UNSUPPORTED``); an upper on the
    last position just closes the top.  An open side needs every prefix sum
    (``L = -inf``: ``Re > tol``) or suffix sum (``U = +inf``: ``Re < -tol``)
    to decay.  The status is the chain's own: an ``OVERFLOW`` still carries a
    valid ``(log_scale, mant)``."""
    m = len(alphas)
    tol = _check_tol(tol)
    L = float(L)
    U = float(U)
    if math.isnan(L) or math.isnan(U) or L == _INF or U == -_INF:
        raise ValueError(f'boxfree.chain_log: bad limits ({L!r}, {U!r})')
    if m == 0:
        return OK, 0.0, 1.0 + 0j
    al = [complex(a) for a in alphas]
    if L == -_INF:
        p = 0j
        for a in al:
            p += a
            if not p.real > tol:
                return DIVERGENT, 0.0, 0j
    try:
        if U != _INF:
            return expdd.chain_with_uppers_log(alphas, L, upper_per_position,
                                               U)
        uppers = upper_per_position or {}
        running = _INF
        eff = [_INF] * m
        for k in range(m - 1, -1, -1):
            u = uppers.get(k)
            if u is not None:
                u = float(u)
                if math.isnan(u):
                    raise ValueError(f'boxfree.chain_log: NaN upper at {k}')
                running = min(running, u)
            eff[k] = running
        if any(e <= L for e in eff):
            return OK, 0.0, 0j
        if eff[m - 1] != _INF:
            return expdd.chain_with_uppers_log(alphas, L, uppers, eff[m - 1])
        if any(e != _INF for e in eff):
            return UNSUPPORTED, 0.0, 0j
        if L == -_INF:
            return DIVERGENT, 0.0, 0j
        p = 0j
        for a in reversed(al):
            p += a
            if not p.real < -tol:
                return DIVERGENT, 0.0, 0j
        mirrored = [-a for a in reversed(al)]
        return expdd.chain_simplex_log(mirrored, -_INF, -L)
    except FloatingPointError:
        return UNDERFLOW, 0.0, 0j


# ─────────────────────────────────────────────────────── batched (M7b)
# The routes above take one pole tuple per call; ``final_integral`` runs
# them over every tuple of a subset (up to ~10^5 per region), which costs
# Python-level work per tuple.  The kernels below take all tuples of one
# region (one chain-fold) at once on the compiled backend of ``expdd``
# (plain Python without numba).  They compute what the scalar routines
# compute, in the same order, and they only ever answer the plain case:
# every term OK and in range.  Anything else (a non-OK status, an overflow
# branch, a non-finite input that the scalar routine would raise on) is
# reported as "not taken" and the caller reruns the scalar loop, so
# statuses, bail reasons and exceptions are those of the scalar routines.
# The values agree with the scalar loop to rounding, not bit for bit
# (compiled complex products may be contracted to FMAs).

#: ``_region_value`` status: the scalar routine would raise ``ValueError``
#: (a non-finite input or exponent)
_RAISES = -1

# Compiled WITHOUT numba's on-disk cache: these kernels call
# ``expdd._log_dd_core`` and freeze expdd's constants, and numba's cache only
# re-checks the file that defines a function -- after an edit to expdd.py a
# cached kernel here would keep running the old compiled code (found in
# review; the scalar route would pick up the edit).  The cost is one
# compilation per process.
if getattr(expdd, '_numba', None) is not None and expdd.BACKEND == 'numba':
    _jit = expdd._numba.njit(cache=False)
else:                                     # pragma: no cover - plain Python
    def _jit(f):
        return f
_log_dd_core = expdd._log_dd_core


def _pack_region(region):
    """``(codes, P, rays)`` of a ``Region2D`` for the compiled kernels
    (cached on the region): ``codes[i]`` 0/1/2 for a triangle / half-strip
    / wedge, ``P[i] = (x0, a, b, log det)`` flattened (``a, b`` = ``u, w``,
    ``u, r`` or ``r1, r2``), ``rays`` the unit rays."""
    packed = region._packed
    if packed is None:
        n = len(region.pieces)
        codes = np.empty(n, dtype=np.int64)
        P = np.empty((n, 7), dtype=np.float64)
        for i, (tag, x0, a, b, det) in enumerate(region.pieces):
            codes[i] = {'tri': 0, 'strip': 1, 'wedge': 2}[tag]
            P[i] = (x0[0], x0[1], a[0], a[1], b[0], b[1], math.log(det))
        rays = np.array([tuple(r) for r in region.rays],
                        dtype=np.float64).reshape(-1, 2)
        packed = (codes, P, rays)
        region._packed = packed
    return packed


@_jit
def _piece_log_c(code, p, alpha, beta, gamma):
    """``_piece_log`` on a packed piece."""
    x0x = p[0]
    x0y = p[1]
    if code == 2:
        f0 = alpha * x0x + beta * x0y + gamma
        l1 = cmath.log(-(alpha * p[2] + beta * p[3]))
        l2 = cmath.log(-(alpha * p[4] + beta * p[5]))
        return (f0.real + p[6] - l1.real - l2.real,
                cmath.exp(1j * (f0.imag - l1.imag - l2.imag)))
    npts = 3 if code == 0 else 2
    px = np.empty(npts, dtype=np.float64)
    py = np.empty(npts, dtype=np.float64)
    px[0] = x0x
    py[0] = x0y
    px[1] = p[2]
    py[1] = p[3]
    if code == 0:
        px[2] = p[4]
        py[2] = p[5]
    # _anchored_nodes: the first point with the largest real exponent
    k = 0
    best = 0.0
    for i in range(npts):
        rel = alpha * (px[i] - px[0]) + beta * (py[i] - py[0])
        if i == 0 or rel.real > best:
            best = rel.real
            k = i
    nodes = np.empty(npts, dtype=np.complex128)
    for i in range(npts):
        nodes[i] = alpha * (px[i] - px[k]) + beta * (py[i] - py[k])
    nodes[k] = 0j
    anchor = alpha * px[k] + beta * py[k] + gamma
    c, mant = _log_dd_core(nodes)
    if code == 0:
        return (anchor.real + c + p[6], mant * cmath.exp(1j * anchor.imag))
    ar = alpha * p[4] + beta * p[5]
    lg = cmath.log(-ar)
    return (anchor.real + c + p[6] - lg.real,
            mant * cmath.exp(1j * (anchor.imag - lg.imag)))


@_jit
def _finite_c(z):
    return math.isfinite(z.real) and math.isfinite(z.imag)


@_jit
def _region_value(codes, P, rays, alpha, beta, gamma, tol):
    """``region_integral`` of a packed nonempty region: ``(status,
    value)``, status ``_RAISES`` where the scalar routine raises."""
    if not (_finite_c(alpha) and _finite_c(beta) and _finite_c(gamma)):
        return _RAISES, 0j
    for i in range(rays.shape[0]):
        ar = alpha * rays[i, 0] + beta * rays[i, 1]
        if not ar.real < -tol or abs(ar) < ALPHA_ZERO:
            return DIVERGENT, 0j
    n = codes.shape[0]
    ls = np.empty(n, dtype=np.float64)
    ms = np.empty(n, dtype=np.complex128)
    for i in range(n):
        ls[i], ms[i] = _piece_log_c(codes[i], P[i], alpha, beta, gamma)
    for i in range(n):
        if not math.isfinite(ls[i]):
            return _RAISES, 0j
        if ms[i] == 0 and ls[i] > LOG_OVERFLOW:
            return OVERFLOW, 0j
    top = 0.0
    found = False
    for i in range(n):
        if ms[i] != 0 and (not found or ls[i] > top):
            top = ls[i]
            found = True
    if not found:
        return OK, 0j
    acc = 0j
    for i in range(n):
        if ms[i] != 0:
            acc += ms[i] * math.exp(ls[i] - top)
    if acc == 0:
        return OK, 0j
    # exp_times(complex(top, 0.0), acc)
    if not _finite_c(acc):
        return _RAISES, 0j
    lv = top + math.log(abs(acc))
    if lv > LOG_OVERFLOW:
        return OVERFLOW, 0j
    if -LOG_OVERFLOW <= top <= LOG_OVERFLOW:
        return OK, acc * cmath.exp(complex(top, 0.0))
    if lv < -2.0 * LOG_OVERFLOW:
        return OK, 0j
    return OK, (acc / abs(acc)) * cmath.exp(complex(lv, 0.0))


@_jit
def _region_batch_core(codes, P, rays, alphas, betas, gammas, tol):
    n = alphas.shape[0]
    st = np.empty(n, dtype=np.int64)
    val = np.empty(n, dtype=np.complex128)
    for t in range(n):
        st[t], val[t] = _region_value(codes, P, rays, alphas[t], betas[t],
                                      gammas[t], tol)
    return st, val


@_jit
def _region_modesum_core(codes, P, rays, coef, alphas, betas, gammas, tol):
    total = 0j
    mag = 0.0
    for t in range(coef.shape[0]):
        c = coef[t]
        if c == 0:
            continue
        st, v = _region_value(codes, P, rays, alphas[t], betas[t],
                              gammas[t], tol)
        if st != OK:
            return False, 0j, 0.0
        total += c * v
        mag += abs(c * v)
    return True, total, mag


def _as_c128(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.complex128).ravel())


def region_integral_batch(region, alphas, betas, gammas, tol=0.0):
    """``region_integral`` for many ``(alpha, beta, gamma)`` at once:
    ``(status, value)`` arrays.  ``status`` is that of ``region_integral``,
    or ``-1`` where it raises ``ValueError``; ``value`` is 0 where the
    status is not ``OK``."""
    tol = _check_tol(tol)
    alphas, betas, gammas = (_as_c128(alphas), _as_c128(betas),
                             _as_c128(gammas))
    n = alphas.shape[0]
    if not betas.shape[0] == gammas.shape[0] == n:
        raise ValueError('boxfree.region_integral_batch: length mismatch')
    if region.kind == 'empty':
        return np.full(n, OK, dtype=np.int64), np.zeros(n, np.complex128)
    if region.kind == 'lineality':
        return (np.full(n, DIVERGENT, dtype=np.int64),
                np.zeros(n, np.complex128))
    codes, P, rays = _pack_region(region)
    return _region_batch_core(codes, P, rays, alphas, betas, gammas, tol)


def region_modesum(region, coefs, alphas, betas, gammas, tol=0.0):
    """``Σ_t coefs[t] · region_integral(region, alphas[t], betas[t],
    gammas[t])`` over the tuples with ``coefs[t] != 0``, summed in order,
    when every such term is ``OK``; ``None`` otherwise (the caller runs its
    scalar loop, which yields the status, bail or exception).  ``region``
    must be nonempty (not 'empty' or 'lineality')."""
    tol = _check_tol(tol)
    if region.kind not in ('bounded', 'unbounded'):
        raise ValueError(f'boxfree.region_modesum: region kind '
                         f'{region.kind!r}')
    r = region_modesum_mag(region, coefs, alphas, betas, gammas, tol)
    return None if r is None else r[0]


def region_modesum_mag(region, coefs, alphas, betas, gammas, tol=0.0):
    """``region_modesum`` and the sum of the terms' absolute values:
    ``(value, Σ|term|)``, or ``None`` (see there)."""
    tol = _check_tol(tol)
    if region.kind not in ('bounded', 'unbounded'):
        raise ValueError(f'boxfree.region_modesum: region kind '
                         f'{region.kind!r}')
    codes, P, rays = _pack_region(region)
    ok, total, mag = _region_modesum_core(
        codes, P, rays, _as_c128(coefs), _as_c128(alphas), _as_c128(betas),
        _as_c128(gammas), tol)
    return (complex(total), float(mag)) if ok else None


@_jit
def _group_fold_core(coef, gamma, member_idx, member_off):
    """Per group ``g`` (members ``member_idx[member_off[g]:member_off[g+1]]``
    in tuple order): ``g_max`` = the largest ``Re γ`` over the members with
    ``coef != 0`` and ``tc = Σ coef e^{γ - g_max}``; ``tc = 0`` marks a group
    with no such member."""
    G = member_off.shape[0] - 1
    tc = np.zeros(G, dtype=np.complex128)
    gmax = np.zeros(G, dtype=np.float64)
    tcabs = np.zeros(G, dtype=np.float64)
    for g in range(G):
        found = False
        best = 0.0
        for i in range(member_off[g], member_off[g + 1]):
            t = member_idx[i]
            if coef[t] != 0 and (not found or gamma[t].real > best):
                best = gamma[t].real
                found = True
        if not found:
            continue
        acc = 0j
        aabs = 0.0
        for i in range(member_off[g], member_off[g + 1]):
            t = member_idx[i]
            if coef[t] != 0:
                x = coef[t] * cmath.exp(gamma[t] - best)
                acc += x
                aabs += abs(x)
        tc[g] = acc
        gmax[g] = best
        tcabs[g] = aabs
    return tc, gmax, tcabs


def group_fold(coefs, gammas, member_idx, member_off):
    """``(tc, g_max)`` per group of pole tuples, as the poset DD route folds
    them (see ``_group_fold_core``); ``coefs``, ``gammas`` per tuple."""
    return group_fold_mag(coefs, gammas, member_idx, member_off)[:2]


def group_fold_mag(coefs, gammas, member_idx, member_off):
    """``group_fold`` and, per group, ``Σ |coef e^{γ - g_max}|`` over its
    members (the magnitude the fold cancels)."""
    return _group_fold_core(_as_c128(coefs), _as_c128(gammas),
                            np.ascontiguousarray(member_idx, dtype=np.int64),
                            np.ascontiguousarray(member_off, dtype=np.int64))


@_jit
def _chain_fold_core(tc, gmax, ST, LS, MANT, tcabs):
    total = 0j
    mag = 0.0
    G = tc.shape[0]
    E = ST.shape[0]
    for g in range(G):
        c = tc[g]
        if c == 0:
            continue
        zg = gmax[g]
        for e in range(E):
            st = ST[e, g]
            if st != OK and st != OVERFLOW:
                return False, 0j, 0.0
            zr = zg + LS[e, g]
            mant = MANT[e, g]
            if (-LOG_OVERFLOW <= zr <= LOG_OVERFLOW and abs(mant) < 1e150
                    and abs(c) < 1e150):
                v = (c * mant) * math.exp(zr)
                total += v
                mag += tcabs[g] * abs(mant) * math.exp(zr)
            else:
                return False, 0j, 0.0
    return True, total, mag


def chain_fold(tc, gmax, ST, LS, MANT):
    """``Σ_g Σ_e tc[g] · MANT[e, g] · e^{g_max[g] + LS[e, g]}`` over the
    groups with ``tc != 0`` (groups outer, extensions inner, summed in that
    order) when every chain status is ``OK`` or ``OVERFLOW`` and every term
    takes the in-range branch of the poset DD route; ``None`` otherwise.
    ``ST``, ``LS``, ``MANT``: ``(E, G)`` chain results ``(status,
    log_scale, mant)`` per extension and group."""
    r = chain_fold_mag(tc, gmax, ST, LS, MANT)
    return None if r is None else r[0]


def chain_fold_mag(tc, gmax, ST, LS, MANT, tcabs=None):
    """``chain_fold`` and the sum of the terms' absolute values:
    ``(value, Σ|term|)``, or ``None`` (see there).  ``tcabs`` (per group,
    ``Σ |member|`` of its fold, ``group_fold_mag``; default ``|tc|``) makes
    the magnitude count the cancellation inside each group's fold too."""
    tc = _as_c128(tc)
    tcabs = (np.abs(tc) if tcabs is None
             else np.ascontiguousarray(tcabs, dtype=np.float64))
    ok, total, mag = _chain_fold_core(
        tc, np.ascontiguousarray(gmax, dtype=np.float64),
        np.ascontiguousarray(ST, dtype=np.int64),
        np.ascontiguousarray(LS, dtype=np.float64),
        np.ascontiguousarray(MANT, dtype=np.complex128), tcabs)
    return (complex(total), float(mag)) if ok else None

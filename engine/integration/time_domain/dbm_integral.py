r"""Exact integral of an exponential-polynomial integrand over a
difference-constraint polytope (the "DBM route", Phase J milestone M3).

Region
------
Every row of the region is a strict linear inequality ``a·s + c > 0`` in the
integration variables ``s_0 … s_{m-1}`` whose normal ``a`` is a
difference-bound row:

* one nonzero coefficient ``a_i`` -- a scalar bound
  (``s_i > -c / a_i`` for ``a_i > 0``, ``s_i < c / |a_i|`` for ``a_i < 0``);
* two nonzero coefficients ``a_i = -a_j`` -- a shifted order row
  ``s_j - s_i < c / a_i`` (``a_i > 0``).

A direction that the rows leave open is closed at the box: ``s_v >= -cap``
where the rows give ``s_v`` no lower bound (directly or through other
variables), ``s_v <= cap`` where they give it no upper bound.  A finite bound
from the rows is never clipped, as in the scipy fallback's legacy routines
(``_integrate_nd_polytope``).  The box itself is absolute (±cap from the
time origin), not relative to the region, as in those routines: a region
with an open direction whose finite bound on the other side lies near or
beyond the box (e.g. bounded above by an external time near or below -cap,
with a time open below) is truncated by it, or emptied (exactly 0,
``STATUS_EMPTY``).  A zero node ``Z`` (``s_Z ≡ 0``, index ``m``) turns the
scalar bounds into differences too.  ``D[u][v]`` is the tightest known upper bound
on ``s_v - s_u`` (``math.inf``: none).  Strict and non-strict inequalities
differ by a set of measure zero, which the integral does not see.

Facts used
----------
* A difference-bound matrix (DBM) is closed under projection: the
  Floyd-Warshall closure followed by deleting a variable's row and column is
  exact Fourier-Motzkin elimination of that variable.
* After the closure ``D[v][v]`` is the weight of the lightest cycle through
  ``v``.  A cycle of weight ``<= 0`` means the region is empty or has measure
  zero, so its integral is exactly 0.  The test is exact, with no tolerance
  (the engine's rule: a genuinely thin region, e.g. a strip between external
  times 1e-12 apart, is integrated; a sliver left by rounding contributes at
  rounding level).  This covers a directed
  order cycle (no linear extension), a ``Δt ≡ 0`` pair of opposite rows and
  crossed scalar bounds, with no special casing.  (Θ(0) = 0 for constant
  rows is the caller's business: a row with a zero normal never reaches this
  module.)
* Case split.  The innermost variable ``s_k`` lies between the largest of
  its lower bounds and the smallest of its upper bounds.  For every pair
  ``(a, b)`` of non-redundant bounds the case "``s_a + lo`` is the largest
  lower and ``s_b + hi`` the smallest upper" is again a difference
  constraint system; its cases tile the region up to sets of measure zero.
  In each case ``s_k`` is integrated in closed form between
  ``s_a + lo`` and ``s_b + hi`` and eliminated.

Integrand
---------
A sum of terms ``C · exp(E + Σ_v β_v s_v) · Π_v s_v^{n_v}`` with a complex
coefficient ``C`` and a complex LOG-SCALE constant ``E``.  Integrating
``s_k`` over ``[s_node + c_lo, s_node' + c_hi]`` moves ``β_k`` onto the
bound's node and adds ``β_k · c`` to ``E``; ``exp`` is taken once, at the
very end, on a fully integrated term.  No intermediate ``exp(β·L)`` is formed
in isolation, so a lower cap far in the past cannot overflow on its own.  A
final term of magnitude ``> exp(LOG_OVERFLOW)`` is reported as an overflow
(the integral of a bounded integrand over the box is never that large, so
such a term could only cancel catastrophically).

A coefficient with ``|β_k| · cap <= BETA_ZERO_SPAN`` is treated as 0 (the
polynomial antiderivative): dropping ``exp(β_k s_k)`` on ``|s_k| <= cap``
changes the term by at most ``BETA_ZERO_SPAN`` relative (rounding residues of
sums of poles are ~1e-16).  Any larger ``β_k`` takes the exponential
antiderivative, whose ``1/β_k`` factors cancel when ``|β_k|`` times the width
of the interval is small (close poles, thin regions).

Conditioning.  Every term carries the absolute values of everything merged
into it, so ``magnitude`` is the Σ|term| of the full expansion and the
rounding error of the value is about ``COND_ERR_FACTOR · magnitude``.  When
that exceeds ``max(COND_RTOL · |value|, COND_ATOL · scale)`` (``scale``: the
integrand's scale, given by the caller, e.g. Σ|pole-tuple coefficient|) the
result is ``STATUS_ILL_CONDITIONED``: the value is returned for information
only, and the caller integrates the region another way.

The prototype this is ported from is ``poset_dbm_integrator.py`` (M3 brief).
Differences: all pole tuples are integrated in one pass through one case
tree (the case split depends only on the region), terms that share a
monomial are merged in the log domain, redundant bounds are pruned without a
tolerance (an exactly tied pair keeps its smaller index, so pruning can never
drop every bound), and the final sum is compensated (``math.fsum``).
"""
import cmath
import math

import numpy as _np

INF = math.inf
#: A closed-DBM cycle of weight <= this is empty / measure zero (integral 0).
DEGENERATE_TOL = 0.0
#: ``|β| · cap <=`` this is integrated as a polynomial (β treated as 0).
BETA_ZERO_SPAN = 1e-10
#: Rounding error estimate = COND_ERR_FACTOR × magnitude (≈ 4.5 ulp).
COND_ERR_FACTOR = 1e-15
#: Accepted error: max(COND_RTOL × |value|, COND_ATOL × scale).
COND_RTOL = 1e-10
COND_ATOL = 1e-14
#: A final term with ``Re E + log|C|`` above this is an overflow.
LOG_OVERFLOW = 700.0
#: A coefficient within this of 0 is a zero coefficient of a row normal.
_COEF_ATOL = 1e-12
#: More elimination cases than this: give up (the caller falls back).
MAX_CASES = 200000
#: Retry of an ill-conditioned closed form: an interval with two constant
#: bounds whose width w has ``|β| · w <=`` this is integrated by Gauss-Legendre
#: (no antiderivative difference F(b) - F(a) to cancel).
THIN_SPAN = 1.0
#: Nodes and weights of the 16-point Gauss-Legendre rule on [-1, 1]: exact for
#: polynomials of degree <= 31; for x^n exp(β x) with |β| w <= 1 the error is
#: far below rounding.
_GL_X, _GL_W = (tuple(float(v) for v in a)
                for a in _np.polynomial.legendre.leggauss(16))

STATUS_OK = 'ok'
STATUS_EMPTY = 'empty'                 # measure-zero / empty region: 0
STATUS_NOT_DBM = 'not_difference_rows'  # a row is not a difference row
STATUS_OVERFLOW = 'overflow'           # a final term beyond LOG_OVERFLOW
STATUS_TOO_MANY_CASES = 'too_many_cases'  # more than MAX_CASES cases
STATUS_ILL_CONDITIONED = 'ill_conditioned'  # rounding error above tolerance


class _TooManyCases(Exception):
    pass


class DBMResult:
    """Outcome of ``integrate_exp_sum``.

    ``value``     complex integral (``None`` unless ``status`` is ``'ok'``,
                  ``'empty'`` or ``'ill_conditioned'``; for the last it is
                  for information only);
    ``status``    one of ``STATUS_*``;
    ``magnitude`` Σ of the absolute values of every term of the full
                  expansion (merges included): the cancellation scale; the
                  value's rounding error is about 1e-16 times it (0.0 for
                  an empty region);
    ``n_cases``   leaf cases of the elimination (0 for an empty region);
    ``error_ratio`` the rounding error estimate over the accepted error
                  (``> 1``: ``STATUS_ILL_CONDITIONED``; 0.0 when not
                  computed);
    ``thin_retry`` True when the value comes from the thin-interval retry
                  (see ``integrate_exp_sum``).
    """
    __slots__ = ('value', 'status', 'magnitude', 'n_cases', 'error_ratio',
                 'thin_retry')

    def __init__(self, value, status, magnitude=0.0, n_cases=0,
                 error_ratio=0.0, thin_retry=False):
        self.value = value
        self.status = status
        self.magnitude = magnitude
        self.n_cases = n_cases
        self.error_ratio = error_ratio
        self.thin_retry = thin_retry

    def __repr__(self):
        return (f'DBMResult(value={self.value!r}, status={self.status!r}, '
                f'magnitude={self.magnitude!r}, n_cases={self.n_cases}, '
                f'error_ratio={self.error_ratio!r}, '
                f'thin_retry={self.thin_retry})')


# ───────────────────────────────────────────────────────────── the region

def difference_bounds(rows, m, cap=None):
    r"""The DBM of ``{s : a·s + c > 0 ∀ rows}``.

    ``rows``: iterable of ``(a, c)`` with ``a`` a length-``m`` sequence of
    floats.  Returns an ``(m+1) × (m+1)`` list of lists (node ``m`` is the
    zero node), or ``None`` when a row is not a difference row (a zero
    normal, a non-opposite pair, or three or more nonzero coefficients).
    Without ``cap`` the matrix holds the rows only and is not closed (its
    diagonal is ``INF``).  With ``cap`` it is the CLOSED matrix of the region
    whose open directions are closed at ±cap (module docstring); a
    degenerate rows-only closure is returned as it is.
    """
    n = m + 1
    Z = m
    D = [[INF] * n for _ in range(n)]
    for (a, c) in rows:
        c = float(c)
        nz = []
        for i in range(m):
            ai = float(a[i])
            if abs(ai) > _COEF_ATOL:
                nz.append((i, ai))
        if len(nz) == 1:
            (i, ai) = nz[0]
            if ai > 0.0:            # s_i > -c/ai:  s_Z - s_i < c/ai
                w = c / ai
                if w < D[i][Z]:
                    D[i][Z] = w
            else:                   # s_i < c/|ai|:  s_i - s_Z < c/|ai|
                w = c / -ai
                if w < D[Z][i]:
                    D[Z][i] = w
            continue
        if len(nz) == 2:
            (i, ai), (j, aj) = nz
            if ai != -aj:
                return None
            up, lo, a_up = (i, j, ai) if ai > 0.0 else (j, i, aj)
            # a_up (s_up - s_lo) + c > 0:  s_lo - s_up < c / a_up
            w = c / a_up
            if w < D[up][lo]:
                D[up][lo] = w
            continue
        return None
    if cap is None:
        return D
    cap = float(cap)
    D = close(D)
    if is_degenerate(D):
        return D
    open_dir = False
    for v in range(m):
        if D[v][Z] == INF:
            D[v][Z] = cap           # s_v >= -cap (no lower bound from rows)
            open_dir = True
        if D[Z][v] == INF:
            D[Z][v] = cap           # s_v <= cap (no upper bound from rows)
            open_dir = True
    return close(D) if open_dir else D


def close(D):
    """Floyd-Warshall closure of ``D`` (a new matrix).  ``D[i][i]`` of the
    result is the weight of the lightest cycle through ``i`` (``INF`` when
    the input diagonal is ``INF`` and there is none)."""
    n = len(D)
    D = [row[:] for row in D]
    for k in range(n):
        Dk = D[k]
        for i in range(n):
            dik = D[i][k]
            if dik == INF:
                continue
            Di = D[i]
            for j in range(n):
                v = dik + Dk[j]
                if v < Di[j]:
                    Di[j] = v
    return D


def is_degenerate(D, tol=DEGENERATE_TOL):
    """True if the closed DBM has a cycle of weight <= ``tol`` (the region
    is empty or of measure zero)."""
    return any(D[i][i] <= tol for i in range(len(D)))


# ───────────────────────────────────────────────────────────── elimination

_FACT = [1]
for _i in range(1, 64):
    _FACT.append(_FACT[-1] * _i)


def _binom_shift(p, c):
    """``(x + c)^p = Σ_q coef[q] x^q``: the list ``coef`` (length p+1)."""
    return [math.comb(p, q) * (c ** (p - q)) for q in range(p + 1)]


def _rescaled(A, d):
    """``A·exp(d)`` for a magnitude ``A`` >= 0, ``inf`` past the float
    range (a magnitude may only overstate the rounding error)."""
    if A == 0:
        return 0.0
    if d > LOG_OVERFLOW:
        return INF
    return A * math.exp(d)


def _add_term(acc, key, C, E, A):
    """Merge ``C·exp(E)`` into ``acc[key] = (C0, E0, A0)`` in the log domain
    (the larger real exponent is kept as the base, so no factor overflows).
    ``A`` (>= |C|, on the same scale ``exp(Re E)``) accumulates the absolute
    values of everything merged into the term, so that the cancellation of
    the full expansion stays measurable (``DBMResult.magnitude``).  A term
    whose coefficient has cancelled to exactly 0 still carries its ``A``;
    merging it never moves the base of a nonzero term (that could change
    or underflow the coefficient), and a nonzero term replaces a zero one
    on its own base."""
    old = acc.get(key)
    if old is None:
        acc[key] = (C, E, A)
        return
    C0, E0, A0 = old
    if C == 0 and C0 != 0:
        acc[key] = (C0, E0, A0 + _rescaled(A, E.real - E0.real))
    elif C0 == 0 and C != 0:
        acc[key] = (C, E, A + _rescaled(A0, E0.real - E.real))
    elif E == E0:
        acc[key] = (C0 + C, E0, A0 + A)
    elif E.real > E0.real:
        f = cmath.exp(E0 - E)
        acc[key] = (C + C0 * f, E, A + A0 * abs(f))
    else:
        f = cmath.exp(E - E0)
        acc[key] = (C0 + C * f, E0, A0 + A * abs(f))


def _integrate_var(terms, k, lo, hi, Z, eps, thin=False):
    """Integrate every term of ``terms`` (a dict ``(beta, mono) -> (C, E,
    A)``, see ``_add_term``) over ``s_k ∈ [s_lo_node + lo_c, s_hi_node +
    hi_c]``; ``lo`` / ``hi`` are ``(node, c)``.  Returns the new dict
    (``s_k`` eliminated).  ``thin``: a term on an interval with two constant
    bounds and ``|β_k| · width <= THIN_SPAN`` is integrated by Gauss-Legendre
    (the retry of ``integrate_exp_sum``)."""
    out = {}
    if thin and lo[0] == Z and hi[0] == Z:
        a_, b_ = lo[1], hi[1]
        w = b_ - a_
        rest = {}
        for (beta, mono), (C, E, A) in terms.items():
            bk = beta[k]
            if abs(bk) * w > THIN_SPAN:
                rest[(beta, mono)] = (C, E, A)
                continue
            nk = mono[k]
            # ∫_a^b x^n e^{β x} dx = e^{β b} Σ_i w_i x_i^n e^{β (x_i - b)}
            J = 0j
            mag = 0.0
            for xg, wg in zip(_GL_X, _GL_W):
                x = 0.5 * (a_ + b_) + 0.5 * w * xg
                v = 0.5 * w * wg * (x ** nk) * cmath.exp(bk * (x - b_))
                J += v
                mag += abs(v)
            nb = list(beta)
            nb[k] = 0j
            nm = list(mono)
            nm[k] = 0
            _add_term(out, (tuple(nb), tuple(nm)), C * J, E + bk * b_,
                      A * mag)
        if not rest:
            return out
        for key, (C, E, A) in _integrate_var(rest, k, lo, hi, Z,
                                             eps).items():
            _add_term(out, key, C, E, A)
        return out
    for (beta, mono), (C, E, A) in terms.items():
        bk = beta[k]
        nk = mono[k]
        base_beta = list(beta)
        base_beta[k] = 0j
        base_mono = list(mono)
        base_mono[k] = 0
        if abs(bk) >= eps:
            # ∫ x^n e^{b x} dx = e^{b x} Σ_j (-1)^j n!/(n-j)! x^{n-j} / b^{j+1}
            fn = _FACT[nk]
            pieces = [((-1) ** j * (fn // _FACT[nk - j]) / bk ** (j + 1),
                       nk - j) for j in range(nk + 1)]
            carries = True
        else:
            pieces = [(1.0 / (nk + 1), nk + 1)]
            carries = False
        for sign, (node, c) in ((1.0, hi), (-1.0, lo)):
            Enew = E + bk * c if carries else E
            if node == Z:
                nb = tuple(base_beta)
                nm = tuple(base_mono)
                key = (nb, nm)
                tot = 0j
                mag = 0.0
                for (pc, p) in pieces:
                    x = pc * (c ** p)           # s_Z ≡ 0: only q = 0
                    tot += x
                    mag += abs(x)
                if mag != 0:            # tot == 0 keeps its magnitude
                    _add_term(out, key, sign * C * tot, Enew, A * mag)
                continue
            nb = list(base_beta)
            if carries:
                nb[node] = nb[node] + bk
            nb = tuple(nb)
            for (pc, p) in pieces:
                shift = _binom_shift(p, c)
                for q, sc in enumerate(shift):
                    if sc == 0:
                        continue
                    nm = list(base_mono)
                    nm[node] += q
                    kappa = pc * sc
                    _add_term(out, (nb, tuple(nm)), sign * C * kappa, Enew,
                              A * abs(kappa))
    return out


def _bounds_of(D, k, active, Z):
    """The non-redundant lower and upper bound nodes of ``s_k``.

    Lower: ``s_k >= s_a - D[k][a]``; upper: ``s_k <= s_b + D[b][k]``.  A
    bound implied by another through the closed DBM is dropped; of an exactly
    tied (mutually implied) pair the smaller node index is kept, so the lists
    are never emptied by pruning."""
    others = [a for a in active if a != k] + [Z]
    lows = [a for a in others if D[k][a] < INF]
    ups = [b for b in others if D[b][k] < INF]

    def _prune_lows(cands):
        keep = []
        for a in cands:
            red = False
            for a2 in cands:
                if a2 == a:
                    continue
                if D[k][a2] + D[a2][a] <= D[k][a]:
                    if a < a2 and D[k][a] + D[a][a2] <= D[k][a2]:
                        continue
                    red = True
                    break
            if not red:
                keep.append(a)
        return keep or list(cands)

    def _prune_ups(cands):
        keep = []
        for b in cands:
            red = False
            for b2 in cands:
                if b2 == b:
                    continue
                if D[b][b2] + D[b2][k] <= D[b][k]:
                    if b < b2 and D[b2][b] + D[b][k] <= D[b2][k]:
                        continue
                    red = True
                    break
            if not red:
                keep.append(b)
        return keep or list(cands)

    return _prune_lows(lows), _prune_ups(ups)


def _eliminate(terms, D, active, Z, eps, leaves, stats, thin=False):
    """Recursive case-split elimination of the variables ``active`` from
    ``terms`` over the closed, non-degenerate DBM ``D``.  Fully integrated
    terms are appended to ``leaves`` as ``(C, E)``."""
    if not terms:
        return
    if not active:
        stats['cases'] += 1
        for (_beta, _mono), (C, E, A) in terms.items():
            leaves.append((C, E, A))
        return
    n = len(D)
    best = None
    for k in active:
        lows, ups = _bounds_of(D, k, active, Z)
        if not lows or not ups:
            # The box gives every variable a finite bound to Z: unreachable.
            raise ValueError(f'DBM variable {k} is unbounded')
        cost = len(lows) * len(ups)
        if best is None or cost < best[0]:
            best = (cost, k, lows, ups)
    _, k, lows, ups = best
    rest = [a for a in active if a != k]
    for a in lows:
        lo_c = -D[k][a]
        for b in ups:
            stats['visits'] += 1
            if stats['visits'] > MAX_CASES:
                raise _TooManyCases()
            hi_c = D[b][k]
            Dc = [row[:] for row in D]
            for a2 in lows:
                if a2 != a:
                    # s_a + lo_c >= s_a2 + lo_c2:  s_a2 - s_a <= lo_c - lo_c2
                    w = lo_c + D[k][a2]
                    if w < Dc[a][a2]:
                        Dc[a][a2] = w
            for b2 in ups:
                if b2 != b:
                    # s_b + hi_c <= s_b2 + hi_c2:  s_b - s_b2 <= hi_c2 - hi_c
                    w = D[b2][k] - hi_c
                    if w < Dc[b2][b]:
                        Dc[b2][b] = w
            # a nonempty interval: s_a + lo_c <= s_b + hi_c
            w = hi_c - lo_c
            if w < Dc[b][a]:
                Dc[b][a] = w
            Dc = close(Dc)
            if is_degenerate(Dc):
                continue
            for i in range(n):
                Dc[k][i] = INF
                Dc[i][k] = INF
            new_terms = _integrate_var(terms, k, (a, lo_c), (b, hi_c), Z, eps,
                                       thin)
            _eliminate(new_terms, Dc, rest, Z, eps, leaves, stats, thin)


def _sum_leaves(leaves):
    """``(Σ C·exp(E), Σ A·exp(Re E))`` with compensated summation (``A``:
    the absolute magnitude a leaf has accumulated, see ``_add_term``), or
    ``(None, max log-magnitude)`` when a term exceeds ``LOG_OVERFLOW``."""
    re_parts = []
    im_parts = []
    mag = 0.0
    lg_max = -INF
    for (C, E, A) in leaves:
        # every leaf's magnitude counts, a cancelled (C == 0) one included:
        # its rounding error is that of the terms that cancelled
        mag += A * math.exp(E.real) if E.real <= LOG_OVERFLOW else INF
        if C == 0:
            continue
        lg = E.real + math.log(abs(C))
        if lg > lg_max:
            lg_max = lg
        if lg > LOG_OVERFLOW:
            continue
        if E.real > LOG_OVERFLOW:
            x = cmath.exp(E + cmath.log(C))
        else:
            x = C * cmath.exp(E)
        re_parts.append(x.real)
        im_parts.append(x.imag)
    if lg_max > LOG_OVERFLOW:
        return None, lg_max
    return complex(math.fsum(re_parts), math.fsum(im_parts)), mag


# ───────────────────────────────────────────────────────────── driver

def integrate_exp_sum(rows, m, seeds, cap, *, scale=None, eps=None):
    r"""``Σ_t C_t ∫_{region} exp(E_t + Σ_v β_tv s_v) ds``.

    ``rows``  ``(a, c)`` pairs, ``a·s + c > 0`` (see ``difference_bounds``);
    ``m``     the number of integration variables (>= 1);
    ``seeds`` iterable of ``(C, E, beta)`` with ``beta`` a length-``m``
              sequence of complex exponent coefficients;
    ``cap``   the box half-width (open directions closed at ±cap);
    ``scale`` the integrand's scale for the conditioning test's absolute
              floor (default: Σ_t |C_t| exp(Re E_t));
    ``eps``   ``|β|`` below which β is treated as 0 (default
              ``BETA_ZERO_SPAN / cap``).

    A closed form that fails the conditioning test is retried once with every
    interval between two constant bounds that is thin for its exponent
    (``|β| · width <= THIN_SPAN``) integrated by Gauss-Legendre; the retry is
    returned only if it passes the same test (``thin_retry`` set), else the
    first verdict.

    Returns a ``DBMResult``.
    """
    D = difference_bounds(rows, m, cap)
    if D is None:
        return DBMResult(None, STATUS_NOT_DBM)
    n = m + 1
    Z = m
    if is_degenerate(D):
        return DBMResult(0j, STATUS_EMPTY)
    if eps is None:
        eps = BETA_ZERO_SPAN / max(float(cap), 1.0)
    terms = {}
    zero_mono = (0,) * n
    seed_scale = 0.0
    for (C, E, beta) in seeds:
        C = complex(C)
        if C == 0:
            continue
        E = complex(E)
        seed_scale += abs(C) * math.exp(min(E.real, LOG_OVERFLOW))
        key = (tuple(complex(b) for b in beta) + (0j,), zero_mono)
        _add_term(terms, key, C, E, abs(C))
    if scale is None:
        scale = seed_scale
    res = _run(terms, D, m, Z, eps, scale, thin=False)
    if res.status != STATUS_ILL_CONDITIONED:
        return res
    # Retry once with thin constant-bound intervals integrated by
    # Gauss-Legendre: a thin region (an external-time nudge, a tie) makes
    # F(b) - F(a) cancel there.  Taken only if it passes the same test, so
    # every value accepted on the first pass is unchanged.
    res2 = _run(terms, D, m, Z, eps, scale, thin=True)
    if res2.status == STATUS_OK:
        res2.thin_retry = True
        return res2
    return res


def _run(terms, D, m, Z, eps, scale, thin):
    """One elimination pass of ``integrate_exp_sum`` and its conditioning
    verdict."""
    leaves = []
    stats = {'cases': 0, 'visits': 0}
    try:
        _eliminate(terms, D, list(range(m)), Z, eps, leaves, stats, thin)
    except _TooManyCases:
        return DBMResult(None, STATUS_TOO_MANY_CASES, 0.0, stats['cases'])
    total, mag = _sum_leaves(leaves)
    if total is None:
        return DBMResult(None, STATUS_OVERFLOW, mag, stats['cases'])
    accepted = max(COND_RTOL * abs(total), COND_ATOL * float(scale))
    err = COND_ERR_FACTOR * mag
    ratio = err / accepted if accepted > 0 else (0.0 if err == 0 else INF)
    if ratio > 1.0:
        return DBMResult(total, STATUS_ILL_CONDITIONED, mag, stats['cases'],
                         ratio)
    return DBMResult(total, STATUS_OK, mag, stats['cases'], ratio)

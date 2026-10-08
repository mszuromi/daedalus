r"""Divided-difference kernels for the exponential integrals of Phase J (M6).

Every closed-form integral Phase J evaluates is an integral of an exponential
of an affine form over a simplex, and by the Hermite-Genocchi formula such an
integral is a divided difference of ``exp``:

    ∫_{unit n-simplex} exp(Σ_i t_i z_i) dt  =  exp[z_0, …, z_n]
                                               (t_0 = 1 − Σ_{i≥1} t_i).

So

* triangle   ``∫∫_T exp(α x + β y + γ) dA = |det| · exp[a_0, a_1, a_2]``
  with ``a_i`` the exponent at vertex ``i``;
* chain      ``∫_{L<s_1<…<s_m<U} exp(Σ_k α_k s_k) ds = T^m · exp[v_0, …, v_m]``
  with ``T = U − L``, prefix sums ``P_j = α_1 + … + α_j``, ``S = P_m`` and
  ``v_j = L P_j + U (S − P_j) = U S − T P_j``: the exponent at the simplex
  vertex with the first ``j`` variables at ``L`` and the rest at ``U``.
  Coincident nodes (a vanishing block sum of the alphas, the "degenerate
  poles" of the legacy routines) are just repeated nodes: a divided
  difference is a smooth function of its nodes, so no polynomial special
  case is needed and a near-coincidence costs no accuracy.

Log domain
----------
``log_dd_exp(z)`` returns ``(c, mant)`` with ``exp[z] = e^c · mant``,
``c = max Re z_i`` and ``|mant| ≤ 1/n!`` (the Hermite-Genocchi bound for the
shifted nodes, all with ``Re ≤ 0``).  Nothing can overflow inside a kernel;
underflow is benign.  The only failure left is a result that truly exceeds
the double range, ``log|value| > LOG_OVERFLOW``; it is reported as the status
``OVERFLOW`` (or ``None`` from the value-returning wrappers), never as a
silent ``inf``/``nan``.

Evaluation of ``exp[w_0..w_n]`` for shifted nodes ``w``
-------------------------------------------------------
* ``n = 1``: ``e^a · (e^{b−a} − 1)/(b − a)`` anchored at the node with the
  larger real part, so the ``expm1`` argument has ``Re ≤ 0`` (a series for
  ``|b − a| < 1e-4``);
* tight cluster (every pairwise distance below 1): a Taylor expansion about
  the centroid, ``exp[w] = e^{w̄} Σ_k h_k(w − w̄)/(n + k)!`` with ``h_k`` the
  complete homogeneous symmetric polynomials;
* ``n = 2``: the Newton form over the most distant pair ``(i, j)``,
  ``exp[w_i, w_k, w_j] = (exp[w_i, w_k] − exp[w_k, w_j]) / (w_i − w_j)``;
* ``n ≥ 3``: the ``(n, 0)`` entry of ``expm(diag(w) + subdiag(1))`` by scaling
  and squaring, with the diagonal and the first subdiagonal recomputed
  exactly after every squaring (Al-Mohy & Higham 2009, triangular case; the
  subdiagonal ``exp[a, b]`` is anchored at the larger real part, which keeps
  the ``expm1`` argument at ``Re ≤ 0`` and rules out an ``inf · 0`` NaN).

Chains with intermediate scalar uppers (transfer matrix)
---------------------------------------------------------
With tail sums ``β_j = α_{j+1} + … + α_m`` (``β_m = 0``) and the state
``w_j(t) = e^{β_j t} ∫_{L<s_1<…<s_j<t} Π_{i≤j} e^{α_i s_i}`` one has
``w' = A w`` with ``A = diag(β_0..β_m) + subdiag(1)``, ``w(L) = e^{β_0 L} e_0``
and the integral is ``w_m(U)``.  An upper ``s_{k+1} ≤ U_k`` (chain position
``k``) is a projection: at ``t = U_k`` at least ``k + 1`` variables must be
placed, so components ``0..k`` are zeroed there.  The integral is a product
of at most ``q + 1`` segment propagators (``q`` = number of effective upper
breakpoints), with no cut-tuple enumeration.  Each segment is propagated
with the shift ``c = max Re β_j`` over the live components and a running
log scale.

Semi-infinite chains (``L = −∞``)
---------------------------------
Started from the dominant eigenvector: ``w(t) = e^{β_0 t} v`` with ``v_0 = 1``,
``v_j = v_{j−1} / P_j``.  This is the integral iff ``Re P_j > 0`` for every
``j``; otherwise the integral diverges and the kernels report the status
``DIVERGENT`` (``DivergentIntegralError`` from the value-returning wrappers),
never a finite number.

Backends
--------
The cores are written once in a numba-compatible subset of Python.  They are
compiled with ``numba.njit(cache=True)`` when numba imports and run as plain
Python otherwise (also under ``NUMBA_DISABLE_JIT=1``).  ``BACKEND`` says which
one is active.  The two agree to rounding (FMA contraction and libm
differences), not bit for bit.

Nothing in Phase J calls this module by default (M6).  The only opt-in is
``final_integral.CHAIN_UPPERS_KERNEL = 'transfer'``, which routes
``_chain_with_intermediate_uppers`` to ``chain_with_uppers``.
"""
import cmath
import math
from fractions import Fraction

import numpy as np

try:
    import numba as _numba
except ImportError:          # pragma: no cover - numba is a pinned dependency
    _numba = None

if _numba is not None:
    _jit = _numba.njit(cache=True)
    BACKEND = 'python' if _numba.config.DISABLE_JIT else 'numba'
else:                        # pragma: no cover
    def _jit(f):
        return f
    BACKEND = 'python'

# Status codes of the ``*_log`` kernels.
OK = 0
OVERFLOW = 1
DIVERGENT = 2
# internal: the chain state underflowed to 0 although the integral is not 0
# (surfaced as FloatingPointError by the Python-level API)
_UNDERFLOW = 3

# A result with log|value| above this is reported as an overflow (headroom of
# ~e^9 below the double range for the caller's own sums and prefactors).
LOG_OVERFLOW = 700.0

# Scaling-and-squaring: the scaled matrix has 1-norm <= 1/2.  Entry (i, j) of
# its exponential starts at the Taylor term k = i - j, so an n x n matrix takes
# n - 1 + _EXPM_TAYLOR_TERMS terms: 16 beyond the first leave a relative
# truncation error ~ 0.5^16 / (17 * 16!) on every entry (a fixed 16 terms
# would drop the entries more than 16 below the diagonal: 2.6e-3 at n = 21).
_EXPM_TAYLOR_TERMS = 16
# Centroid Taylor expansion for clusters of pairwise spread < 1: the k-th
# term is bounded by 1/(n! k!), so 30 terms are far below double rounding.
_CLUSTER_SPREAD = 1.0
_CLUSTER_TERMS = 30
# Below this |z| the (e^z - 1)/z series (to z^4) is used.
_PHI1_SERIES = 1e-4
# The chain state is rescaled (by an exact power of two, so the mantissa is
# not rounded and the log scale moves by k*ln2) only when its largest
# component leaves [2^-_RESCALE_EXP, 2^_RESCALE_EXP].  An unconditional
# renormalisation would add eps*|log max| to the log scale at every step.
_RESCALE_EXP = 500
_LN2 = math.log(2.0)


class DivergentIntegralError(ValueError):
    """A semi-infinite (``L = -inf``) chain whose integral diverges (some
    prefix sum of the alphas has ``Re P_j <= 0``)."""


# ───────────────────────────────────────────────────────────────────────
# Cores (numba-compiled when available)
# ───────────────────────────────────────────────────────────────────────
@_jit
def _phi1(z):
    """``(e^z - 1)/z`` for complex ``z`` with ``Re z <= 0``, without
    cancellation (``Re(e^z - 1) = expm1(x) cos y - 2 sin^2(y/2)``)."""
    if abs(z) < _PHI1_SERIES:
        return 1.0 + z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0
                                                      + z / 120.0)))
    x = z.real
    y = z.imag
    sy = math.sin(0.5 * y)
    re = math.expm1(x) * math.cos(y) - 2.0 * sy * sy
    im = math.exp(x) * math.sin(y)
    return complex(re, im) / z


@_jit
def _dd2(a, b):
    """``exp[a, b]``, anchored at the node with the larger real part."""
    if a.real >= b.real:
        return cmath.exp(a) * _phi1(b - a)
    return cmath.exp(b) * _phi1(a - b)


@_jit
def _expm_bidiag(d, h, f):
    """``expm(diag(h d) + f * subdiag(1))``, lower triangular ``n x n``
    (``f = h``: the plain ``expm(h (diag(d) + subdiag(1)))``; the chain
    transfer passes a graded ``f``, see ``_chain_transfer_core``).

    Scaling and squaring with a Taylor series; the diagonal and the first
    subdiagonal are set to their exact values (``e^{h d_j}`` and
    ``f exp[h d_{j-1}, h d_j]``) before the first squaring and after every
    one, which stops the 2^s error amplification on the entries that drive
    the accuracy of all the others.  Meant for ``Re d_j <= 0`` (shifted
    nodes), where every entry is bounded."""
    n = d.shape[0]
    nrm = 0.0
    for j in range(n):
        a = abs(d[j])
        if a > nrm:
            nrm = a
    nrm *= h
    if n > 1:
        nrm += f
    s = 0
    if nrm > 0.5:
        s = int(math.ceil(math.log2(nrm / 0.5)))
    hs = h * 0.5 ** s
    fs = f * 0.5 ** s
    # Taylor series of M = diag(hs d) + fs subdiag(1); M is bidiagonal, so
    # P @ M costs O(n^2): (P M)[i, j] = P[i, j] M[j, j] + P[i, j+1] M[j+1, j].
    E = np.zeros((n, n), dtype=np.complex128)
    P = np.zeros((n, n), dtype=np.complex128)
    Q = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        E[j, j] = 1.0
        P[j, j] = 1.0
    for k in range(1, n + _EXPM_TAYLOR_TERMS):
        for i in range(n):
            for j in range(i + 1):
                acc = P[i, j] * (d[j] * hs)
                if j + 1 <= i:
                    acc += P[i, j + 1] * fs
                Q[i, j] = acc / k
        for i in range(n):
            for j in range(i + 1):
                P[i, j] = Q[i, j]
                E[i, j] += Q[i, j]
    hh = hs
    ff = fs
    for it in range(s + 1):
        if it > 0:
            # square: E <- E @ E (lower triangular)
            for i in range(n):
                for j in range(i + 1):
                    acc = 0.0 + 0.0j
                    for l in range(j, i + 1):
                        acc += E[i, l] * E[l, j]
                    Q[i, j] = acc
            for i in range(n):
                for j in range(i + 1):
                    E[i, j] = Q[i, j]
            hh = hh * 2.0
            ff = ff * 2.0
        for j in range(n):
            E[j, j] = cmath.exp(d[j] * hh)
        for j in range(1, n):
            E[j, j - 1] = ff * _dd2(d[j - 1] * hh, d[j] * hh)
    return E


@_jit
def _dd_cluster(w):
    """``exp[w]`` for a tight cluster: Taylor expansion about the centroid,
    ``e^{cen} * sum_k h_k(w - cen) / (n + k)!``."""
    n1 = w.shape[0]
    n = n1 - 1
    cen = 0.0 + 0.0j
    for i in range(n1):
        cen += w[i]
    cen = cen / n1
    h = np.zeros(_CLUSTER_TERMS + 1, dtype=np.complex128)
    h[0] = 1.0
    for i in range(n1):
        u = w[i] - cen
        for k in range(1, _CLUSTER_TERMS + 1):
            h[k] += u * h[k - 1]
    f = 1.0
    for i in range(2, n + 1):
        f /= i
    tot = 0.0 + 0.0j
    for k in range(_CLUSTER_TERMS + 1):
        tot += h[k] * f
        f /= (n + k + 1)
    return cmath.exp(cen) * tot


@_jit
def _log_dd_core(z):
    """``(c, mant)`` with ``exp[z_0..z_n] = e^c * mant``, ``c = max Re z``."""
    n1 = z.shape[0]
    n = n1 - 1
    c = z[0].real
    for i in range(1, n1):
        if z[i].real > c:
            c = z[i].real
    w = np.empty(n1, dtype=np.complex128)
    for i in range(n1):
        w[i] = z[i] - c
    if n == 0:
        return c, cmath.exp(w[0])
    if n == 1:
        return c, _dd2(w[0], w[1])
    spread = 0.0
    bi = 0
    bj = 1
    for i in range(n1):
        for j in range(i):
            g = abs(w[i] - w[j])
            if g > spread:
                spread = g
                bi = i
                bj = j
    if spread < _CLUSTER_SPREAD:
        return c, _dd_cluster(w)
    if n == 2:
        bk = 3 - bi - bj
        d1 = _dd2(w[bi], w[bk])
        d2 = _dd2(w[bk], w[bj])
        return c, (d1 - d2) / (w[bi] - w[bj])
    E = _expm_bidiag(w, 1.0, 1.0)
    return c, E[n, 0]


@_jit
def _cldexp(z, k):
    """``z * 2^k`` for complex ``z``, exact unless it under- or overflows."""
    k = int(k)          # a numpy integer in the pure-Python backend
    return complex(math.ldexp(z.real, k), math.ldexp(z.imag, k))


@_jit
def _regrade(v, lo, hi, F0, G):
    """Multiply ``v[j]`` (``lo <= j < hi``) by ``2^(F0 + (j - lo) G - X)``,
    exactly (ldexp), and return ``X`` (an int): 0 when every result stays
    within ``[2^-_RESCALE_EXP, 2^_RESCALE_EXP]`` in modulus at the top,
    otherwise the exponent that brings the largest result to ``[1/2, 1)``
    (smaller ones may underflow: they are negligible next to it).  The
    caller adds ``X ln 2`` to its log scale."""
    top = 0
    found = False
    for j in range(lo, hi):
        a = abs(v[j])
        if a != 0.0:
            e = math.frexp(a)[1] + F0 + (j - lo) * G
            if not found or e > top:
                top = e
                found = True
    if not found:
        return 0
    X = 0
    if top > _RESCALE_EXP or top < -_RESCALE_EXP:
        X = top
    for j in range(lo, hi):
        v[j] = _cldexp(v[j], F0 + (j - lo) * G - X)
    return X


@_jit
def _dominant_start(alphas, t0):
    """State ``w(t0)`` of a semi-infinite chain, as ``(status, sigma, v)``
    with ``w = e^sigma * v``.  ``status`` is ``DIVERGENT`` unless every
    prefix sum has a positive real part.

    ``v_j = v_{j-1} / P_j`` is formed with ``P_j`` scaled to modulus
    ``[1/2, 1)`` by an exact power of two, and the power is applied to
    ``v_j`` or, when that would leave ``[.., 2^_RESCALE_EXP]``, the frame
    moves (the earlier components shrink by that power, possibly to 0: they
    are negligible next to ``v_j``).  So no step overflows, even for a
    subnormal ``P_j``, and the log scale moves only when the range needs
    it."""
    m = alphas.shape[0]
    v = np.zeros(m + 1, dtype=np.complex128)
    S = 0.0 + 0.0j
    for j in range(m):
        S += alphas[j]
    p = 0.0 + 0.0j
    sigma = (S * t0).real
    v[0] = cmath.exp(1j * (S * t0).imag)
    for j in range(1, m + 1):
        p += alphas[j - 1]
        if not p.real > 0.0:
            return DIVERGENT, 0.0, v
        ep = math.frexp(abs(p))[1]
        x = v[j - 1] / _cldexp(p, -ep)          # true v_j = x * 2^-ep
        ax = abs(x)
        if ax == 0.0:
            continue                             # v_{j-1} underflowed: 0
        ex = math.frexp(ax)[1]
        if ex - ep <= _RESCALE_EXP:
            v[j] = _cldexp(x, -ep)
        else:                                    # move the frame up
            shift = ex - ep
            for i in range(j):
                v[i] = _cldexp(v[i], -shift)
            v[j] = _cldexp(x, -ep - shift)
            sigma += shift * _LN2
    return OK, sigma, v


@_jit
def _chain_transfer_core(alphas, L, U, cut_times, cut_state):
    """Chain ``L < s_1 < ... < s_m < U`` with projections: at
    ``cut_times[i]`` (increasing, inside ``(L, U)``) the state components
    ``< cut_state[i]`` are zeroed.  ``L`` may be ``-inf``.

    Returns ``(status, sigma, mant)`` with the integral ``e^sigma * mant``
    (``mant = 0`` and ``sigma = 0`` for an exact zero).

    Graded frame: the state is ``w_j = e^sigma 2^(eg (j - lo)) wh_j`` for
    ``j >= lo`` (components below ``lo`` are zero).  A segment of length
    ``dt = f 2^e`` (``f`` in [1/2, 1)) is propagated with grading
    ``2^eg = 2^e``, i.e. by ``expm(diag(dt d) + f subdiag(1))``: the
    ``dt^(i-j)`` factors of the plain propagator, which underflow for a tiny
    ``dt`` and a long chain, live in the exact powers of two instead.  The
    frame is re-anchored at ``lo`` after every projection.  If the surviving
    state still underflows to 0, the status is ``_UNDERFLOW`` (never a
    silent 0)."""
    m = alphas.shape[0]
    n = m + 1
    ncut = cut_times.shape[0]
    beta = np.zeros(n, dtype=np.complex128)
    acc = 0.0 + 0.0j
    for j in range(m - 1, -1, -1):
        acc += alphas[j]
        beta[j] = acc
    if L == -np.inf:
        t = U
        if ncut > 0:
            t = cut_times[0]
        st, sigma, wh = _dominant_start(alphas, t)
        if st != OK:
            return st, 0.0, 0.0 + 0.0j
    else:
        t = L
        wh = np.zeros(n, dtype=np.complex128)
        sigma = (beta[0] * L).real
        wh[0] = cmath.exp(1j * (beta[0] * L).imag)
    lo = 0
    eg = 0
    for seg in range(ncut + 1):
        t_next = U
        if seg < ncut:
            t_next = cut_times[seg]
        dt = t_next - t
        if dt > 0.0:
            fr, e = math.frexp(dt)
            sigma += _regrade(wh, lo, n, 0, eg - e) * _LN2
            eg = e
            c = beta[lo].real
            for j in range(lo + 1, n):
                if beta[j].real > c:
                    c = beta[j].real
            nl = n - lo
            d = np.empty(nl, dtype=np.complex128)
            for j in range(nl):
                d[j] = beta[lo + j] - c
            E = _expm_bidiag(d, dt, fr)
            new = np.zeros(n, dtype=np.complex128)
            for i in range(nl):
                a2 = 0.0 + 0.0j
                for j in range(i + 1):
                    a2 += E[i, j] * wh[lo + j]
                new[lo + i] = a2
            for i in range(n):
                wh[i] = new[i]
            sigma += c * dt + _regrade(wh, lo, n, 0, 0) * _LN2
            t = t_next
        if seg < ncut:
            k = cut_state[seg]
            if k > lo:
                for j in range(lo, k):
                    wh[j] = 0.0
                sigma += _regrade(wh, k, n, eg * (k - lo), 0) * _LN2
                lo = k
        mx = 0.0
        for j in range(lo, n):
            if abs(wh[j]) > mx:
                mx = abs(wh[j])
        if mx == 0.0:
            return _UNDERFLOW, 0.0, 0.0 + 0.0j
    sigma += _regrade(wh, m, n, eg * (m - lo), 0) * _LN2
    return OK, sigma, wh[m]


# ───────────────────────────────────────────────────────────────────────
# Python-level API
# ───────────────────────────────────────────────────────────────────────
def _as_nodes(z):
    arr = np.ascontiguousarray(np.asarray(z, dtype=np.complex128).ravel())
    if arr.shape[0] == 0:
        raise ValueError('expdd: a divided difference needs at least one node')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'expdd: non-finite node in {z!r}')
    return arr


def _check_finite(name, x):
    if not math.isfinite(x):
        raise ValueError(f'expdd: {name} must be finite, got {x!r}')


def _finish(status, log_scale, mant):
    """``(status, log_scale, mant)`` -> the value, ``None`` on overflow,
    ``DivergentIntegralError`` on divergence.  Never ``inf``/``nan``."""
    if status == DIVERGENT:
        raise DivergentIntegralError(
            'expdd: semi-infinite chain with a prefix sum Re P_j <= 0')
    if status == OVERFLOW:
        return None
    v = complex(mant)
    if v == 0:
        return 0j
    # |v| e^r <= e^LOG_OVERFLOW (status OK).  Bring r into
    # [-LOG_OVERFLOW, LOG_OVERFLOW] by moving it into v in steps of at most
    # LOG_OVERFLOW: from above |v| stays <= 1 (no intermediate overflow even
    # for a subnormal mantissa); from below v only shrinks, so a large
    # mantissa with a very negative scale (a representable value) is not
    # flushed to 0 by exp(r) underflowing on its own.
    r = log_scale
    while r > LOG_OVERFLOW:
        step = min(r - LOG_OVERFLOW, LOG_OVERFLOW)
        v = v * math.exp(step)
        r -= step
    while r < -LOG_OVERFLOW and v != 0:
        step = max(r + LOG_OVERFLOW, -LOG_OVERFLOW)
        v = v * math.exp(step)
        r -= step
    return v * math.exp(r)


def _status(log_scale, mant):
    """``OVERFLOW`` iff ``log|e^log_scale * mant| > LOG_OVERFLOW``; a
    non-finite intermediate (which no finite input should produce) raises
    ``FloatingPointError`` instead of passing as a value."""
    if not (math.isfinite(log_scale) and math.isfinite(mant.real)
            and math.isfinite(mant.imag)):
        raise FloatingPointError(
            f'expdd: non-finite intermediate (log_scale={log_scale!r}, '
            f'mant={mant!r})')
    if mant == 0:
        return OK
    if log_scale + math.log(abs(mant)) > LOG_OVERFLOW:
        return OVERFLOW
    return OK


def log_dd_exp(z):
    """``(c, mant)`` with ``exp[z_0, ..., z_n] = e^c * mant``.

    ``c = max_i Re z_i`` (a float), ``mant`` complex with
    ``|mant| <= 1/n!`` up to rounding.  ``z`` is a sequence of finite complex
    nodes (repeats allowed); a non-finite node raises ``ValueError``."""
    c, mant = _log_dd_core(_as_nodes(z))
    c, mant = float(c), complex(mant)
    _status(c, mant)                     # raises on a non-finite result
    return c, mant


def dd_exp(z):
    """``exp[z_0, ..., z_n]`` = the integral of ``exp(sum t_i z_i)`` over the
    unit n-simplex, or ``None`` if ``log|value| > LOG_OVERFLOW``."""
    c, mant = log_dd_exp(z)
    return _finish(_status(c, mant), c, mant)


def unit_triangle(p, q):
    """``J(p, q) = int_0^1 du int_0^{1-u} dw exp(p u + q w) = exp[0, p, q]``
    (the quantity of ``final_integral._exp_over_unit_triangle``); ``None``
    on overflow."""
    return dd_exp((0j, p, q))


def triangle_log(v0, v1, v2, alpha, beta, gamma=0j):
    """``(status, log_scale, mant)`` for
    ``int_T exp(alpha x + beta y + gamma) dA``, ``T = (v0, v1, v2)``.

    The vertex with the largest real exponent is the anchor (a scalar
    factor) and the nodes are the gaps to it, formed from the edge vectors
    out of it, so two large exponents never cancel in the log scale.  A
    zero-area triangle gives an exact zero."""
    xs = (float(v0[0]), float(v1[0]), float(v2[0]))
    ys = (float(v0[1]), float(v1[1]), float(v2[1]))
    for val in xs + ys:
        _check_finite('vertex', val)
    alpha, beta, gamma = complex(alpha), complex(beta), complex(gamma)
    # the determinant exactly (rationals), rounded once: a nearly collinear
    # triangle would otherwise carry the cancellation error of the float
    # formula straight into the result
    X0, X1, X2 = (Fraction(x) for x in xs)
    Y0, Y1, Y2 = (Fraction(y) for y in ys)
    det = float((X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0))
    if det == 0.0:
        return OK, 0.0, 0j
    a = [alpha * xs[i] + beta * ys[i] + gamma for i in range(3)]
    for z in a:
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            raise ValueError(f'expdd: non-finite vertex exponent {z!r}')
    # anchor at the vertex with the largest real exponent; the gaps come from
    # the edge vectors out of it (no cancellation of two large exponents)
    k = max(range(3), key=lambda i: a[i].real)
    nodes = [alpha * (xs[i] - xs[k]) + beta * (ys[i] - ys[k]) for i in range(3)]
    nodes[k] = 0j
    c, mant = log_dd_exp(nodes)
    ls = c + a[k].real + math.log(abs(det))
    mant = mant * cmath.exp(1j * a[k].imag)
    return _status(ls, mant), ls, mant


def triangle(v0, v1, v2, alpha, beta, gamma=0j):
    """``int_T exp(alpha x + beta y + gamma) dA``; ``None`` only if the
    result itself overflows (no bail on a large exponent, unlike the legacy
    ``final_integral._exp_over_triangle``)."""
    return _finish(*triangle_log(v0, v1, v2, alpha, beta, gamma))


def _validate_chain(alphas, L, U):
    al = np.ascontiguousarray(np.asarray(alphas, dtype=np.complex128).ravel())
    if not np.all(np.isfinite(al)):
        raise ValueError(f'expdd: non-finite alpha in {alphas!r}')
    L = float(L)
    U = float(U)
    if math.isnan(L) or L == math.inf:
        raise ValueError(f'expdd: lower limit must be finite or -inf, got {L!r}')
    _check_finite('upper limit', U)
    return al, L, U


def chain_simplex_log(alphas, L, U):
    """``(status, log_scale, mant)`` for
    ``int_{L<s_1<...<s_m<U} exp(sum_k alpha_k s_k) ds`` (alphas innermost
    first).  ``L`` may be ``-inf`` (status ``DIVERGENT`` unless every prefix
    sum has ``Re P_j > 0``).  ``m = 0`` gives 1, ``U <= L`` gives 0."""
    al, L, U = _validate_chain(alphas, L, U)
    m = al.shape[0]
    if m == 0:
        return OK, 0.0, 1.0 + 0j
    if U <= L:
        return OK, 0.0, 0j
    if L == -math.inf:
        return _semi_infinite_chain(al, U)
    T = U - L
    # Vertex exponents v_j = L P_j + U (S - P_j); anchor at the largest real
    # part, formed directly (L P_* + U * tail sum), and take the gaps
    # v_j - v_* = -T (P_j - P_*) from block sums of the alphas.  Shifting by
    # a different vertex would cancel two large exponents (e.g. U S against
    # -T P_m when U >> |L|) and lose eps*|U S| in the log scale.
    P = [0j]
    for j in range(m):
        P.append(P[-1] + complex(al[j]))
    S = P[-1]
    js = max(range(m + 1), key=lambda j: (L * P[j] + U * (S - P[j])).real)
    tail = 0j
    for j in range(js, m):
        tail += complex(al[j])
    anchor = L * P[js] + U * tail
    nodes = np.zeros(m + 1, dtype=np.complex128)
    B = 0j
    for j in range(js + 1, m + 1):
        B += complex(al[j - 1])
        nodes[j] = -T * B
    B = 0j
    for j in range(js - 1, -1, -1):
        B += complex(al[j])
        nodes[j] = T * B
    c, mant = _log_dd_core(nodes)
    ls = float(c) + anchor.real + m * math.log(T)
    mant = complex(mant) * cmath.exp(1j * anchor.imag)
    return _status(ls, mant), ls, mant


def _semi_infinite_chain(al, U):
    """``L = -inf``: ``e^{S U} / prod_j P_j``, the scale of the product kept
    as an exact power of two (folded back into the mantissa when it fits)."""
    S = 0j
    for a in al:
        S += complex(a)
    mant = cmath.exp(1j * (S * U).imag)
    e2 = 0
    p = 0j
    for a in al:
        p += complex(a)
        if not p.real > 0.0:
            return DIVERGENT, 0.0, 0j
        ep = math.frexp(abs(p))[1]
        mant = mant / complex(math.ldexp(p.real, -ep), math.ldexp(p.imag, -ep))
        e2 -= ep
        em = math.frexp(abs(mant))[1]            # keep |mant| near 1
        mant = complex(math.ldexp(mant.real, -em), math.ldexp(mant.imag, -em))
        e2 += em
    # |mant| is in [1/2, 1): fold as much of 2^e2 into it as keeps it a
    # normal double (exact); only the rest goes to the log scale
    fold = min(max(e2, -1000), 1000)
    mant = complex(math.ldexp(mant.real, fold), math.ldexp(mant.imag, fold))
    ls = (S * U).real + (e2 - fold) * _LN2
    return _status(ls, mant), ls, mant


def chain_simplex(alphas, L, U):
    """``int_{L<s_1<...<s_m<U} exp(sum_k alpha_k s_k) ds``; ``None`` on
    overflow, ``DivergentIntegralError`` for a divergent ``L = -inf``
    chain."""
    return _finish(*chain_simplex_log(alphas, L, U))


def chain_with_uppers_log(alphas_chain, L, upper_per_position, U_chain_top):
    """``(status, log_scale, mant)`` for the chain integral with scalar
    uppers at arbitrary positions -- the quantity of
    ``final_integral._chain_with_intermediate_uppers_uncached``, with the
    same argument conventions:

    * ``upper_per_position[k]`` bounds chain position ``k`` (0-indexed);
      keys outside ``0..m-1`` are ignored and a ``None`` value is absent;
    * the effective upper of position ``k`` is the minimum of its own upper,
      those of every later position, and ``U_chain_top``; if any effective
      upper is ``<= L`` the integral is 0 (so a tie at ``L`` is dropped,
      Theta(0) = 0);
    * an empty chain is 1.

    Each effective upper below the top is a breakpoint where the state must
    already hold ``k + 1`` placed variables (the largest such ``k`` for a
    shared breakpoint).  ``L`` may be ``-inf``."""
    m = len(alphas_chain)
    if m == 0:
        return OK, 0.0, 1.0 + 0j
    if upper_per_position is None:
        upper_per_position = {}
    effective_upper = [U_chain_top] * m
    running_min = U_chain_top
    for k in range(m - 1, -1, -1):
        u_k = upper_per_position.get(k)
        if u_k is not None:
            u_k = float(u_k)
            if math.isnan(u_k):
                raise ValueError(f'expdd: NaN upper at chain position {k}')
            running_min = min(running_min, u_k)
        effective_upper[k] = running_min
    if any(effective_upper[k] <= L for k in range(m)):
        return OK, 0.0, 0j
    top = float(effective_upper[m - 1])
    cuts = {}
    for k in range(m):
        e = float(effective_upper[k])
        if e < top:
            cuts[e] = max(cuts.get(e, 0), k + 1)
    if not cuts:
        return chain_simplex_log(alphas_chain, L, top)
    al, L, top = _validate_chain(alphas_chain, L, top)
    times = sorted(cuts)
    for t in times:
        _check_finite('upper', t)
    st, sigma, mant = _chain_transfer_core(
        al, L, top, np.asarray(times, dtype=np.float64),
        np.asarray([cuts[t] for t in times], dtype=np.int64))
    if st == _UNDERFLOW:
        raise FloatingPointError(
            'expdd: the chain state underflowed below double range')
    if st != OK:
        return st, 0.0, 0j
    mant = complex(mant)
    return _status(sigma, mant), float(sigma), mant


def chain_with_uppers(alphas_chain, L, upper_per_position, U_chain_top):
    """Value of ``chain_with_uppers_log``: a complex, ``None`` on overflow,
    ``DivergentIntegralError`` for a divergent ``L = -inf`` chain."""
    return _finish(*chain_with_uppers_log(
        alphas_chain, L, upper_per_position, U_chain_top))

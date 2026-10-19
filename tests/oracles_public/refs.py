"""Exact references for the public oracle suite (mpmath only; no package code is imported).

Used by tests/test_public_oracles.py.  Every function is independent of Daedalus: a test that
compares the package against one of these compares two unrelated computations.

Convention everywhere: C_ab(tau) = <a(0) b(tau)>; k-point cumulants at absolute times t_1..t_k.

  linear_k2(A, Q, i, j, tau)            stationary linear SDE dx = A x dt + dW_Q: Lyapunov (Kronecker solve) + expm.
                                         Works for Jordan blocks (no eigendecomposition).
  linear_kpt(A, sources, idx, times, t_min=None)
                                         k-point cumulant of a linear SDE driven by independent Levy sources
                                         (b vector, {n: kappa_n}): kappa_k int prod_i [e^{A(t_i-s)} b]_{a_i} ds,
                                         eigen closed form; t_min = switch-on time (fluctuations zero at t_min).
  switchon_k2(A, Q, i, j, t1, t2, t_min) Van Loan.
  hawkes_linear(mu, W, TK, i, j, tau)   linear Hawkes, exponential kernels (unit mass): exact smooth C_ij(tau) via the
                                         Markov embedding y_ij = g_ij * n_j driven by white shot noise diag(lambda).
  delayed_ou(a, c, d, D, tau)           scalar x' = -a x + c x(t-d) + sqrt(2D) xi: exact BVP (method of steps),
                                         |tau| <= 2d.
  poly1d_cumulant(f, D, times, ell)     1-D gradient SDE dx = f(x) dt + sqrt(2D) dW, f(0) = 0: the ell-loop part of
                                         the k-point cumulant (coefficient of D^{k-1+ell}) via the backward generator
                                         on monomials in y = x/sqrt(D), Taylor coefficients by a Cauchy contour in
                                         s = sqrt(D) (truncation provably exact at the orders used).
  allen_cahn_1d(mu, D, T, lam, x, tau)  spatial d=1 infinite domain tree + exact one-loop (tadpole = mass shift).
"""
import functools
import itertools

import mpmath as mp

DPS = 40


def _prec(fn):
    """Run ``fn`` at DPS digits without touching the global mpmath precision (the package uses
    mpmath too: a module-level ``mp.mp.dps = ...`` would leak into every later test)."""
    @functools.wraps(fn)
    def wrapped(*args, **kw):
        with mp.workdps(max(DPS, mp.mp.dps)):
            return fn(*args, **kw)
    return wrapped


def M(a):
    return mp.matrix([[mp.mpf(x) if not isinstance(x, mp.mpc) else x for x in row] for row in a])


def expm(A, t):
    return mp.expm(A * t)


# ---------------------------------------------------------------- linear Gaussian, k = 2

@_prec
def lyap(A, Q):
    """Sigma with A S + S A^T + Q = 0 (Kronecker solve, any A with no eigenvalue pair summing to 0)."""
    n = A.rows
    K = mp.zeros(n * n, n * n)
    for i in range(n):
        for j in range(n):
            r = i * n + j
            for k in range(n):
                K[r, k * n + j] += A[i, k]       # (A S)_ij
                K[r, i * n + k] += A[j, k]       # (S A^T)_ij
    q = mp.matrix([-Q[i, j] for i in range(n) for j in range(n)])
    s = mp.lu_solve(K, q)
    return mp.matrix([[s[i * n + j] for j in range(n)] for i in range(n)])


@_prec
def linear_k2(A, Q, i, j, tau):
    """<x_i(0) x_j(tau)> stationary (0-based i, j)."""
    A, Q = M(A), M(Q)
    S = lyap(A, Q)
    tau = mp.mpf(tau)
    if tau >= 0:
        C = S * expm(A.T, tau)
    else:
        C = expm(A, -tau) * S
    return C[i, j]


@_prec
def switchon_k2(A, Q, i, j, t1, t2, t_min):
    """<x_i(t1) x_j(t2)>, x(t_min) = 0 (Van Loan for Sigma(t))."""
    A, Q = M(A), M(Q)
    n = A.rows

    def Sig(t):
        h = mp.mpf(t) - mp.mpf(t_min)
        B = mp.zeros(2 * n, 2 * n)
        for a in range(n):
            for b in range(n):
                B[a, b] = -A[a, b]; B[a, n + b] = Q[a, b]; B[n + a, n + b] = A[b, a]
        E = expm(B, h)
        F = mp.matrix([[E[n + a, n + b] for b in range(n)] for a in range(n)])
        G = mp.matrix([[E[a, n + b] for b in range(n)] for a in range(n)])
        return F.T * G
    t1, t2 = mp.mpf(t1), mp.mpf(t2)
    if t2 >= t1:
        C = Sig(t1) * expm(A.T, t2 - t1)
    else:
        C = expm(A, t1 - t2) * Sig(t2)
    return C[i, j]


# ---------------------------------------------------------------- linear, Levy sources, any k

@_prec
def linear_kpt(A, sources, idx, times, t_min=None):
    """k-point joint cumulant of x_{idx[0]}(times[0]), ... for dx = A x dt + sum_s b_s dL_s,
    L_s independent Levy with cumulant rates kappa_n (dict n -> value).  Eigen closed form
    (A must be diagonalizable; checked).  t_min: fluctuations start at 0 there."""
    A = M(A)
    k = len(idx)
    ev, V = mp.eig(A)
    Vi = mp.inverse(V)
    cond = mp.mnorm(V, 1) * mp.mnorm(Vi, 1)
    if cond > mp.mpf(10) ** 12:
        raise ValueError(f'linear_kpt: eigenbasis ill conditioned ({mp.nstr(cond, 3)})')
    times = [mp.mpf(t) for t in times]
    t0 = min(times)
    total = mp.mpc(0)
    n = A.rows
    for b, kap in sources:
        kk = kap.get(k, 0)
        if kk == 0:
            continue
        u = Vi * mp.matrix([mp.mpf(x) for x in b])
        for ms in itertools.product(range(n), repeat=k):
            coef = mp.mpc(1)
            lam = mp.mpc(0)
            ex = mp.mpc(0)
            for a, m_, t in zip(idx, ms, times):
                coef *= V[a, m_] * u[m_]
                lam += ev[m_]
                ex += ev[m_] * (t - t0)
            if coef == 0:
                continue
            if t_min is None:
                val = mp.exp(ex) / (-lam)
            else:
                # int_{t_min}^{t0} e^{sum lam_m (t_i - s)} ds
                val = (mp.exp(ex) - mp.exp(ex + lam * (t0 - mp.mpf(t_min)))) / (-lam)
            total += kk * coef * val
    return total


def linear_kpt_jordan(A, sources, idx, times, t_min=None):
    """linear_kpt at 80 digits (near-degenerate eigenvalues: the eigen closed form cancels ~1/split digits)."""
    with mp.workdps(80):
        return +_kpt_nocond(A, sources, idx, times, t_min)


def _kpt_nocond(A, sources, idx, times, t_min):
    A = M(A); k = len(idx)
    ev, V = mp.eig(A); Vi = mp.inverse(V)
    times = [mp.mpf(t) for t in times]; t0 = min(times); n = A.rows; total = mp.mpc(0)
    for b, kap in sources:
        kk = kap.get(k, 0)
        if kk == 0:
            continue
        u = Vi * mp.matrix([mp.mpf(x) for x in b])
        for ms in itertools.product(range(n), repeat=k):
            coef = mp.mpc(1); lam = mp.mpc(0); ex = mp.mpc(0)
            for a, m_, t in zip(idx, ms, times):
                coef *= V[a, m_] * u[m_]; lam += ev[m_]; ex += ev[m_] * (t - t0)
            total += kk * coef * mp.exp(ex) / (-lam)
    return total


# ---------------------------------------------------------------- linear Hawkes (exponential kernels)

@_prec
def hawkes_linear(mu, W, TK, i, j, tau, mean=False):
    """Smooth <n_i(0) n_j(tau)> (tau != 0) of the linear Hawkes process with rate_a = mu_a + sum_b W_ab (g_ab * n_b),
    g_ab(t) = e^{-t/TK_ab}/TK_ab.  mean=True returns lambda_i."""
    P = len(mu)
    Wm = M(W)
    lam = mp.lu_solve(mp.eye(P) - Wm, mp.matrix([mp.mpf(x) for x in mu]))
    if mean:
        return lam[i]
    pairs = [(a, b) for a in range(P) for b in range(P) if W[a][b] != 0]
    ns = len(pairs)
    Mm = mp.zeros(ns, ns); B = mp.zeros(ns, P); Cm = mp.zeros(P, ns)
    pos = {p: r for r, p in enumerate(pairs)}
    for r, (a, b) in enumerate(pairs):
        t = mp.mpf(TK[a][b])
        # tau_ab y_ab' = -y_ab + n_b,  n_b = sum_c W_bc y_bc + xi_b
        Mm[r, r] += -1 / t
        for c in range(P):
            if (b, c) in pos:
                Mm[r, pos[(b, c)]] += mp.mpf(W[b][c]) / t
        B[r, b] += 1 / t
    for a in range(P):
        for c in range(P):
            if (a, c) in pos:
                Cm[a, pos[(a, c)]] = mp.mpf(W[a][c])
    Lam = mp.diag([lam[a] for a in range(P)])
    S = lyap(Mm, B * Lam * B.T)
    t = mp.mpf(tau)
    if t < 0:
        i, j, t = j, i, -t
    E = expm(Mm.T, t)
    C = Cm * S * E * Cm.T + Lam * B.T * E * Cm.T
    return C[i, j]


# ---------------------------------------------------------------- delayed OU (exact)

@_prec
def delayed_ou(a, c, d, D, tau):
    a, c, d, D = (mp.mpf(x) for x in (a, c, d, D))
    t = abs(mp.mpf(tau))
    if t > 2 * d:
        raise ValueError('delayed_ou: |tau| <= 2d')
    Mm = mp.matrix([[-a, c], [-c, a]])
    E = expm(Mm, d)
    # unknowns u0 = C(0), v0 = C(d): v0 = (E[u0,v0])_0 ; -a u0 + c v0 = -D
    A2 = mp.matrix([[E[0, 0], E[0, 1] - 1], [-a, c]])
    u0, v0 = mp.lu_solve(A2, mp.matrix([0, -D]))
    res = (E * mp.matrix([u0, v0]))[1] - u0          # the other boundary condition (consistency)
    if abs(res) > mp.mpf(10) ** (-25) * abs(u0):
        raise ValueError(f'delayed_ou BVP inconsistent: {res}')
    if t <= d:
        return (expm(Mm, t) * mp.matrix([u0, v0]))[0]
    M3 = mp.matrix([[-a, c, 0], [0, -a, c], [0, -c, a]])
    w0 = (E * mp.matrix([u0, v0]))[0]                 # C(d)
    return (expm(M3, t - d) * mp.matrix([w0, u0, v0]))[0]


# ---------------------------------------------------------------- 1-D polynomial drift, loop by loop

def _gauss_moment(m, mu):
    if m % 2:
        return mp.mpf(0)
    return mp.fac2(m - 1) / mp.mpf(mu) ** (m // 2) if m > 0 else mp.mpf(1)


class Poly1D:
    """dx = f(x) dt + sqrt(2D) dW with f(x) = -mu x + sum_{n>=2} f_n x^n.
    In y = x/s (s = sqrt(D)): backward generator L(s) = d^2/dy^2 + sum_n f_n s^{n-1} y^n d/dy,
    stationary density ~ exp(-mu y^2/2) exp(R), R = sum_{n>=2} f_n s^{n-1} y^{n+1}/(n+1).
    Order s^p raises the polynomial degree by at most p, so a monomial basis of degree N >= k + p_max
    is exact through order p_max."""

    @_prec
    def __init__(self, mu, fcoef, pmax=10, N=None, r='0.05', Mpts=40):
        self.mu = mp.mpf(mu)
        self.f = {int(n): mp.mpf(v) for n, v in fcoef.items()}
        self.pmax = pmax
        self.N = N if N is not None else 4 + pmax + 6
        self.r = mp.mpf(r); self.Mpts = Mpts
        self._wcache = {}

    def _L(self, s):
        N = self.N
        L = mp.zeros(N + 1, N + 1)   # column n = image of y^n
        for n in range(N + 1):
            if n >= 2:
                L[n - 2, n] += n * (n - 1)
            L[n, n] += -self.mu * n
            for m_, fm in self.f.items():
                if n + m_ - 1 <= N:
                    L[n + m_ - 1, n] += fm * s ** (m_ - 1) * n
        return L

    def _weight_series(self, s):
        """E_s[y^m] for m = 0..N, as truncated (order pmax+2) series evaluated at s."""
        J = self.pmax + 2
        # exp(R) = sum_j R^j / j!, R = sum c_q s^q y^{q+2}, q = n-1
        terms = {(0, 0): mp.mpf(1)}          # (s power, y power) -> coef
        Rterms = {(n - 1, n + 1): fm / (n + 1) for n, fm in self.f.items()}
        acc = dict(terms); cur = dict(terms)
        for j in range(1, J + 1):
            nxt = {}
            for (p1, y1), c1 in cur.items():
                for (p2, y2), c2 in Rterms.items():
                    if p1 + p2 <= J:
                        nxt[(p1 + p2, y1 + y2)] = nxt.get((p1 + p2, y1 + y2), 0) + c1 * c2 / j
            cur = nxt
            for kk, v in cur.items():
                acc[kk] = acc.get(kk, 0) + v
        Z = sum(c * s ** p * _gauss_moment(yp, self.mu) for (p, yp), c in acc.items())
        mom = [sum(c * s ** p * _gauss_moment(yp + m, self.mu) for (p, yp), c in acc.items()) / Z
               for m in range(self.N + 1)]
        return mom

    def _moment(self, s, times, Ls, Pcache):
        """E[y(t_1) ... y(t_k)] (times sorted ascending)."""
        N = self.N
        vec = mp.zeros(N + 1, 1); vec[0] = 1          # the constant polynomial 1
        ts = sorted(times)
        for idx in range(len(ts) - 1, -1, -1):
            # multiply by y
            nv = mp.zeros(N + 1, 1)
            for n in range(N):
                nv[n + 1] = vec[n]
            vec = nv
            if idx > 0:
                dt = ts[idx] - ts[idx - 1]
                if dt != 0:
                    if dt not in Pcache:
                        Pcache[dt] = mp.expm(Ls * dt)
                    vec = Pcache[dt] * vec
        mom = self._wcur
        return sum(vec[n] * mom[n] for n in range(N + 1))

    def _cumulant_at(self, s, times):
        Ls = self._L(s)
        self._wcur = self._weight_series(s)
        Pc = {}
        k = len(times)
        idxs = list(range(k))
        memo = {}

        def mom(block):
            key = tuple(sorted(block))
            if key not in memo:
                memo[key] = self._moment(s, [times[i] for i in key], Ls, Pc)
            return memo[key]
        tot = mp.mpc(0)
        for part in _set_partitions(idxs):
            b = len(part)
            term = (-1) ** (b - 1) * mp.factorial(b - 1)
            for blk in part:
                term *= mom(blk)
            tot += term
        return tot

    @_prec
    def coefficients(self, times):
        """Taylor coefficients a_p (p = 0..pmax) of the y-cumulant in s."""
        times = [mp.mpf(t) for t in times]
        vals = []
        for j in range(self.Mpts):
            s = self.r * mp.expjpi(2 * mp.mpf(j) / self.Mpts)
            vals.append((s, self._cumulant_at(s, times)))
        return [sum(v * s ** (-p) for s, v in vals) / self.Mpts for p in range(self.pmax + 1)]

    @_prec
    def loop_term(self, D, times, ells):
        """{ell: the ell-loop part of the x-cumulant at noise D} (x-cumulant = s^k * y-cumulant,
        ell-loop <-> D^{k-1+ell} <-> p = k-2+2 ell)."""
        k = len(times)
        a = self.coefficients(times)
        s = mp.sqrt(mp.mpf(D))
        out = {}
        for l in ells:
            p = k - 2 + 2 * l
            out[l] = mp.mpf(0) if p < 0 else a[p] * s ** (p + k)
        out['_odd_max'] = max([abs(a[p]) for p in range(len(a)) if (p - k) % 2 == 1] or [0])
        return out


def _set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


# ---------------------------------------------------------------- spatial Allen-Cahn d = 1 (infinite)

@_prec
def allen_cahn_1d(mu, D, T, lam, x, tau):
    """(tree, one-loop) of <phi(0,0) phi(x,tau)> for (Dt + mu - D Lap) phi = -lam phi^3 + eta, <eta eta> = 2T delta delta."""
    mu, D, T, lam = (mp.mpf(v) for v in (mu, D, T, lam))
    x, tau = abs(mp.mpf(x)), abs(mp.mpf(tau))

    def C0(m):
        # closed form of int dq/2pi e^{iqx} T e^{-(m+Dq^2)tau}/(m+Dq^2)
        kap = mp.sqrt(m / D)
        if tau == 0:
            return T / (2 * mp.sqrt(m * D)) * mp.exp(-kap * x)
        a, b = mp.sqrt(m * tau), x / (2 * mp.sqrt(D * tau))
        return T / (4 * mp.sqrt(m * D)) * (mp.exp(-kap * x) * mp.erfc(a - b) + mp.exp(kap * x) * mp.erfc(a + b))

    def dC0(m):
        return mp.diff(C0, m)
    # independent check of the closed form by quadrature (tau > 0 only: Gaussian decay)
    if tau > mp.mpf('0.2'):
        f = lambda q: mp.cos(q * x) * T / (mu + D * q * q) * mp.exp(-(mu + D * q * q) * tau)
        qv = mp.quad(f, [0, 1, 4, mp.inf]) / mp.pi
        if abs(qv / C0(mu) - 1) > mp.mpf(10) ** -25:
            raise ValueError('allen_cahn_1d closed form vs quadrature')
    var0 = T / (2 * mp.sqrt(mu * D))
    return C0(mu), 3 * lam * var0 * dC0(mu)

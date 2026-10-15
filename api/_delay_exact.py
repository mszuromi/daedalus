r"""
Exact tree of a delayed linear system: the covariance boundary-value problem
============================================================================

The tree-level two-point function of a model is the stationary covariance of
its linearization, ``u = G y`` with ``G = K^{-1}`` and white forcing ``y``
(``E[y(t) y(s)^T] = N delta(t - s)``, ``N`` the noise matrix at the saddle).
For a kernel matrix with ONE delay,

    K(s) = K0(s) + K1(s) e^{-s d},       K0, K1 rational,

this module computes that covariance exactly (up to rounding), without the
Pade approximation the loops use.

Realization.  ``G0 = K0^{-1} = C0 (sI - A0)^{-1} B0 + D0`` (``api._realization``,
exact over QQ then rounded once) and ``K1 = C1 (sI - A1)^{-1} B1 + D1`` (one
chain per entry).  With ``w(t) = u(t - d)``, ``K u = y`` reads
``u = G0 (y - K1 w)``.  When ``D0 D1 = 0`` (no instantaneous path through
the delay back into ``u``: otherwise refused, ``'neutral'``), the state
``z = (x0, x1)`` obeys the delay equation

    z' = A z + A_d z(t - d) + B y(t) + B_d y(t - d),     u = C_z z + D0 y,

    A = [[A0, -B0 C1], [0, A1]],   B_w = [[-B0 D1], [B1]],
    A_d = B_w C_z,   B_d = B_w D0,   B = [[B0], [0]],   C_z = [C0, -D0 C1].

Covariance.  ``C(tau) = E[z(t + tau) z(t)^T]`` and the response
``R(tau) = e^{A tau} B`` on ``(0, d)``, ``R(d+) = R(d-) + B_d``,
``R' = A R + A_d R(tau - d)`` beyond.  On ``s in (0, d)`` with
``U(s) = C(s)``, ``V(s) = C(s - d)``:

    U' = A U + A_d V + B_d N (e^{A(d-s)} B)^T,
    V' = -V A^T - U A_d^T - e^{A s} B N B_d^T,

a linear ODE (the exponentials are carried as extra states), with the
conditions ``V(d) = U(0)``, ``U(d) = V(0)^T``, ``U(0) = U(0)^T`` and the
equal-time balance

    A U0 + U0 A^T + A_d V0 + V0^T A_d^T + B_d N B^T e^{A^T d} + e^{A d} B N B_d^T
        + B N B^T + B_d N B_d^T = 0.

Beyond ``d`` (method of steps) ``C' = A C + A_d C(tau - d)``: the history
blocks ``C(s + j d)``, ``R(s + j d)`` are stacked with ``U, V`` into one
lower block-triangular linear ODE on ``[0, d]``, so every value is one
matrix exponential of the stack (no step-size error).  The field covariance
is ``E[u(t + tau) u(t)^T] = C_z C(tau) C_z^T + C_z R(tau) N D0^T``
(``tau > 0``; the ``D0 N D0^T delta(tau)`` part is the delta record).

The scalar delayed OU (``A = -a, A_d = c, B = 1, B_d = 0``) reduces to the
2 x 2 problem of item 29's reference (spectral cross-check 1.6e-10).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sl

__all__ = ['ExactTree', 'ExactTreeUnavailable', 'delayed_state_space',
           'DelayedStateSpace']


class ExactTreeUnavailable(ValueError):
    """The exact tree does not apply (several delays, a neutral term, a
    singular instantaneous part, ...); the Pade tree is used instead."""


class DelayedStateSpace:
    """``z' = A z + A_d z(t-d) + B y + B_d y(t-d)``, ``u = C_z z + D0 y``."""

    def __init__(self, A, Ad, B, Bd, Cz, D0, d, N):
        self.A, self.Ad, self.B, self.Bd = A, Ad, B, Bd
        self.Cz, self.D0, self.d, self.N = Cz, D0, float(d), N


def _split_entries(K_ft, omega, num_params):
    """``(K0, K1, d)``: SR matrices with ``K = K0 + K1 exp(-i omega d)``."""
    from sage.all import SR, matrix, exp
    from api.delay import delay_factors, _exp_delay
    nf = K_ft.nrows()
    E = SR.var('_delayE_')
    dval = None
    K0 = [[SR(0)] * nf for _ in range(nf)]
    K1 = [[SR(0)] * nf for _ in range(nf)]
    for i in range(nf):
        for j in range(nf):
            e = SR(K_ft[i, j])
            if e.is_trivial_zero():
                continue
            e = e.subs(num_params)
            ds = delay_factors(e, omega)
            for X in ds:
                xv = float(SR(X).subs(num_params))
                if dval is None:
                    dval = xv
                elif abs(xv - dval) > 1e-13 * max(1.0, abs(dval)):
                    raise ExactTreeUnavailable(
                        f'several delays ({dval:g}, {xv:g}): the exact tree covers '
                        'one delay')

            def rebuild(x):
                x = SR(x)
                hit = _exp_delay(x, omega)
                if hit is not None:
                    return exp(hit[0]) * E
                ops = x.operands()
                return x if not ops else x.operator()(*[rebuild(o) for o in ops])

            ee = SR(rebuild(e))
            if not ee.has(E):
                K0[i][j] = ee
                continue
            num, den = ee.rational_simplify().numerator_denominator()
            if den.has(E):
                raise ExactTreeUnavailable(f'K[{i},{j}]: the delay factor sits in a '
                                           'denominator')
            num = num.expand()
            if num.degree(E) > 1:
                raise ExactTreeUnavailable(f'K[{i},{j}] is not linear in the delay factor')
            K0[i][j] = (num.coefficient(E, 0) / den)
            K1[i][j] = (num.coefficient(E, 1) / den)
    if dval is None:
        raise ExactTreeUnavailable('no delay factor in K(omega)')
    if dval <= 0:
        raise ExactTreeUnavailable(f'non-positive delay {dval}')
    return matrix(SR, K0), matrix(SR, K1), dval


def _realize_K1(K1, omega):
    """``(A1, B1, C1, D1)`` of the rational matrix ``K1(s)`` (a chain per entry;
    ``B1``: states x nf (input = delayed fields), ``C1``: nf x states)."""
    from api import _realization as R
    nf = K1.nrows()
    entries, F, P, fname = R._entries_to_field(K1, omega, {})
    cplx = fname != 'QQ'
    blocks = []
    D1 = np.zeros((nf, nf), dtype=complex if cplx else float)
    for (i, j), x in sorted(entries.items()):
        q, r, den = R._split_entry(x, P)
        if q.degree() >= 1:
            raise ExactTreeUnavailable(
                f'K1[{i},{j}] grows with s: a delayed time derivative (neutral system)')
        D1[i, j] = complex(q.constant_coefficient()) if cplx else float(q.constant_coefficient())
        if r != 0:
            Ab, b, c, _ = R._realize_proper(r, den, F)
            blocks.append((i, j, R._to_numpy(Ab, cplx), R._to_numpy(b, cplx),
                           R._to_numpy(c, cplx)))
    n1 = sum(b[2].shape[0] for b in blocks)
    A1 = np.zeros((n1, n1), dtype=D1.dtype)
    B1 = np.zeros((n1, nf), dtype=D1.dtype)
    C1 = np.zeros((nf, n1), dtype=D1.dtype)
    off = 0
    for i, j, Ab, b, c in blocks:
        m = Ab.shape[0]
        A1[off:off + m, off:off + m] = Ab
        B1[off:off + m, j] = b[:, 0]
        C1[i, off:off + m] = c[0, :]
        off += m
    return A1, B1, C1, D1


def delayed_state_space(K_ft, omega, num_params, N):
    """The ``DelayedStateSpace`` of ``K_ft`` at the parameter point."""
    from api._realization import realize_K, RealizationRefused
    K0, K1, d = _split_entries(K_ft, omega, num_params)
    try:
        r0 = realize_K(K0, omega, {})
    except RealizationRefused as exc:
        raise ExactTreeUnavailable(f'K0 = K at e^(-sd) = 0: {exc}') from exc
    A0, B0, C0, D0 = (np.asarray(r0.A), np.asarray(r0.B), np.asarray(r0.C),
                      np.asarray(r0.D))
    try:
        A1, B1, C1, D1 = _realize_K1(K1, omega)
    except Exception as exc:
        if isinstance(exc, ExactTreeUnavailable):
            raise
        raise ExactTreeUnavailable(f'K1 (the delayed part): {exc}') from exc
    cplx = any(np.iscomplexobj(M) for M in (A0, B0, C0, D0, A1, B1, C1, D1))
    dt = complex if cplx else float
    DD = D0 @ D1
    if np.max(np.abs(DD), initial=0.0) > 1e-12 * max(1.0, np.max(np.abs(D0), initial=0.0)
                                                     * np.max(np.abs(D1), initial=0.0)):
        raise ExactTreeUnavailable('an instantaneous path through the delay feeds back '
                                   'into the fields (D0 D1 != 0)')
    n0, n1, nf = A0.shape[0], A1.shape[0], D0.shape[0]
    A = np.zeros((n0 + n1, n0 + n1), dtype=dt)
    A[:n0, :n0] = A0
    A[:n0, n0:] = -B0 @ C1
    A[n0:, n0:] = A1
    Bw = np.vstack([-B0 @ D1, B1]).astype(dt)
    Cz = np.hstack([C0, -D0 @ C1]).astype(dt)
    B = np.vstack([B0, np.zeros((n1, nf))]).astype(dt)
    Ad = Bw @ Cz
    Bd = Bw @ D0
    if not cplx:
        A, Ad, B, Bd, Cz = (np.real(M) for M in (A, Ad, B, Bd, Cz))
        D0 = np.real(D0)
    return DelayedStateSpace(A, Ad, B, Bd, Cz, D0, d, np.asarray(N, dtype=float))


class _Stack:
    """The stacked linear ODE on ``s in [0, d]``: blocks ``U, V, W1, W2`` and
    ``(C_j, R_j)`` for ``j = 1..K`` (``C_j(s) = C(s + j d)``)."""

    def __init__(self, ss, K):
        self.ss = ss
        n = ss.A.shape[0]
        m = ss.B.shape[1]
        self.n, self.m, self.K = n, m, K
        shapes = [('U', (n, n)), ('V', (n, n)), ('W1', (n, m)), ('W2', (n, m))]
        for j in range(1, K + 1):
            shapes += [(f'C{j}', (n, n)), (f'R{j}', (n, m))]
        self.shapes = shapes
        self.off = {}
        o = 0
        for name, sh in shapes:
            self.off[name] = (o, sh)
            o += sh[0] * sh[1]
        self.dim = o
        self.L = self._build()

    def block(self, Y, name):
        o, sh = self.off[name]
        return Y[o:o + sh[0] * sh[1]].reshape(sh)

    def size_upto(self, k):
        """Dimension of the leading sub-stack up to history block ``k``."""
        if k == 0:
            o, sh = self.off['W2']
        else:
            o, sh = self.off[f'R{k}']
        return o + sh[0] * sh[1]

    def deriv(self, Y):
        ss = self.ss
        A, Ad, Bd, N = ss.A, ss.Ad, ss.Bd, ss.N
        U, V = self.block(Y, 'U'), self.block(Y, 'V')
        W1, W2 = self.block(Y, 'W1'), self.block(Y, 'W2')
        out = [A @ U + Ad @ V + Bd @ N @ W1.T,
               -V @ A.T - U @ Ad.T - W2 @ N @ Bd.T,
               -A @ W1, A @ W2]
        prevC, prevR = U, W2
        for j in range(1, self.K + 1):
            Cj, Rj = self.block(Y, f'C{j}'), self.block(Y, f'R{j}')
            out += [A @ Cj + Ad @ prevC, A @ Rj + Ad @ prevR]
            prevC, prevR = Cj, Rj
        return np.concatenate([x.ravel() for x in out])

    def _build(self):
        D = self.dim
        L = np.zeros((D, D))
        e = np.zeros(D)
        for i in range(D):
            e[i] = 1.0
            L[:, i] = self.deriv(e)
            e[i] = 0.0
        return L


def _expm_apply(L, Y0, s):
    if L.shape[0] <= 600:
        return sl.expm(L * s) @ Y0
    from scipy.sparse.linalg import expm_multiply
    return expm_multiply(L * s, Y0)


class ExactTree:
    """The exact k = 2 tree of a delayed model for one external pair."""

    method = 'bvp'

    def __init__(self, ss, ia, ib, tau_max=None):
        self.ss = ss
        self.ia, self.ib = int(ia), int(ib)
        self._stack = None
        self._Y0 = None
        self._K = 0
        self._ensure(0.0 if tau_max is None else float(tau_max))

    @classmethod
    def from_context(cls, ctx, external_fields):
        """From the analyzer's state of the TRUE model (``analyze_with_context``)."""
        from api.analyze import _noise_matrix
        from api.access import normalize_external_fields
        from engine.diagrams.type_assignment import build_field_index_map
        if ctx is None or ctx.K_ft is None or ctx.num_params is None or ctx.sectors is None:
            raise ExactTreeUnavailable('the analyzer did not reach the linear operator '
                                       'and the saddle')
        if ctx.t.get('noise_class') not in (None, 'white', 'none'):
            raise ExactTreeUnavailable(f"noise class {ctx.t.get('noise_class')!r}: the exact "
                                       'tree needs white noise')
        N, _, unresolved = _noise_matrix(ctx)
        if unresolved:
            raise ExactTreeUnavailable(f'noise matrix keeps symbols {sorted(unresolved)}')
        ext = normalize_external_fields(list(external_fields),
                                        naming_convention=ctx.model.get('naming_convention'))
        if len(ext) != 2:
            raise ExactTreeUnavailable('k = 2 only')
        _, phys_idx = build_field_index_map(ctx.ring_names, ctx.n_tilde)
        ia, ib = phys_idx[tuple(ext[0])], phys_idx[tuple(ext[1])]
        ss = delayed_state_space(ctx.K_ft, ctx.omega, ctx.num_params, N)
        return cls(ss, ia, ib)

    # ── the boundary-value problem and the method of steps ──
    def _base_conditions(self, st):
        ss = self.ss
        A, Ad, B, Bd, N, d = ss.A, ss.Ad, ss.B, ss.Bd, ss.N, ss.d
        n = st.n
        D0 = st.size_upto(0)
        Phi = sl.expm(st.L[:D0, :D0] * d)
        eAd = sl.expm(A * d)
        W1_0, W2_0 = eAd @ B, B

        def resid(x):
            U0 = x[:n * n].reshape(n, n)
            V0 = x[n * n:].reshape(n, n)
            Y0 = np.concatenate([U0.ravel(), V0.ravel(), W1_0.ravel(), W2_0.ravel()])
            Yd = Phi @ Y0
            Ud = Yd[:n * n].reshape(n, n)
            Vd = Yd[n * n:2 * n * n].reshape(n, n)
            lyap = (A @ U0 + U0 @ A.T + Ad @ V0 + V0.T @ Ad.T
                    + Bd @ N @ B.T @ eAd.T + eAd @ B @ N @ Bd.T
                    + B @ N @ B.T + Bd @ N @ Bd.T)
            return np.concatenate([lyap.ravel(), (Vd - U0).ravel(), (Ud - V0.T).ravel(),
                                   (U0 - U0.T).ravel()])

        nx = 2 * n * n
        r0 = resid(np.zeros(nx))
        M = np.empty((r0.size, nx))
        e = np.zeros(nx)
        for i in range(nx):
            e[i] = 1.0
            M[:, i] = resid(e) - r0
            e[i] = 0.0
        x, *_ = np.linalg.lstsq(M, -r0, rcond=None)
        res = np.linalg.norm(M @ x + r0) / max(np.linalg.norm(r0), 1e-300)
        if not np.isfinite(res) or res > 1e-9:
            raise ExactTreeUnavailable(f'the covariance boundary-value problem has no '
                                       f'consistent solution (residual {res:.2g}); is the '
                                       'saddle stable?')
        sv = np.linalg.svd(M, compute_uv=False)
        if sv[-1] <= 1e-13 * sv[0]:
            raise ExactTreeUnavailable('the covariance boundary-value problem is singular '
                                       '(marginal mode?)')
        U0 = x[:n * n].reshape(n, n)
        V0 = x[n * n:].reshape(n, n)
        self.residual = float(res)
        return np.concatenate([U0.ravel(), V0.ravel(), W1_0.ravel(), W2_0.ravel()])

    def _ensure(self, tau_max):
        K = max(0, int(math.floor(abs(tau_max) / self.ss.d)))
        if self._stack is not None and K <= self._K:
            return
        st = _Stack(self.ss, K)
        base = self._base_conditions(st) if self._Y0 is None else self._Y0[:st.size_upto(0)]
        Y0 = np.zeros(st.dim)
        Y0[:base.size] = base
        d = self.ss.d
        for j in range(1, K + 1):
            Dj = st.size_upto(j)
            Yd = _expm_apply(st.L[:Dj, :Dj], Y0[:Dj], d)
            prevC = st.block(Yd, 'U' if j == 1 else f'C{j - 1}')
            prevR = st.block(Yd, 'W2' if j == 1 else f'R{j - 1}')
            oc, shc = st.off[f'C{j}']
            orr, shr = st.off[f'R{j}']
            Y0[oc:oc + prevC.size] = prevC.ravel()
            Y0[orr:orr + prevR.size] = (prevR + (self.ss.Bd if j == 1 else 0.0)).ravel()
        self._stack, self._Y0, self._K = st, Y0, K

    def _blocks_at(self, tau):
        """``(C(tau), R(tau))`` of the state, ``tau >= 0``."""
        d = self.ss.d
        self._ensure(tau)
        st = self._stack
        k = int(math.floor(tau / d))
        s = tau - k * d
        if s < 0:
            s = 0.0
        Dk = st.size_upto(k)
        Y = _expm_apply(st.L[:Dk, :Dk], self._Y0[:Dk], s)
        if k == 0:
            return st.block(Y, 'U'), st.block(Y, 'W2')
        return st.block(Y, f'C{k}'), st.block(Y, f'R{k}')

    def field_covariance(self, tau):
        """``E[u(t + tau) u(t)^T]`` for ``tau > 0`` (smooth part)."""
        ss = self.ss
        Cs, Rs = self._blocks_at(float(tau))
        return ss.Cz @ Cs @ ss.Cz.T + ss.Cz @ Rs @ ss.N @ ss.D0.T

    def value(self, tau):
        """The pipeline's convention: ``<u_a(t) u_b(t + tau)>`` for the
        external pair ``(a, b)``; at ``tau = 0`` the Ito left limit."""
        tau = float(tau)
        a, b = self.ia, self.ib
        if tau > 0:
            return float(np.real(self.field_covariance(tau)[b, a]))
        if tau < 0:
            return float(np.real(self.field_covariance(-tau)[a, b]))
        # tau = 0: the left limit tau -> 0- of <u_a(t) u_b(t + tau)>
        return float(np.real(self.field_covariance(0.0)[a, b]))

    def values(self, taus):
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        if taus.size:
            self._ensure(float(np.max(np.abs(taus))))
        return np.array([self.value(t) for t in taus])

    def callable(self):
        """``f(t0, t1)``: the pipeline's k = 2 callable convention (leg 0 at
        ``t0``, leg 1 at ``t1``, ``tau = t1 - t0``; equal times are evaluated
        at ``tau = -_ITO_EPS``, the pipeline's Ito left limit)."""
        from api.compute import _ITO_EPS

        def f(t0, t1):
            tau = float(np.real(t1)) - float(np.real(t0))
            return self.value(-_ITO_EPS if abs(tau) <= 1e-12 else tau)
        return f

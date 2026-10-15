"""
api.me_output
=============
C0 output (state-space stage S4): the smooth part of a computed cumulant as
**matrix-exponential triples**, with the exponential-mode list derived from
them on request.

The state-space route (``engine/integration/time_domain/statespace.py``)
evaluates every Phase J subset integral by a downset sweep.  Along a ray of
the external times (k = 2: the lag ``τ`` on one side of 0) the order of the
external times is fixed and only the LAST segment of the sweep moves, so
each subset is

    I(u) = const · Π_r g_r(β_r u) · c · exp(G β u) · x0,        u = |τ| > 0,

with ``G`` the segment's block generator (a block lower-triangular system of
Kronecker sums over the downsets of the subset's causal poset), ``x0`` the
state that enters it (the stationary state fired through the first external
time) and ``c`` the contraction of the last external time; ``g_r`` are the
rows that join two external legs directly (their own SISO realizations,
Kronecker-summed into the triple).  Summed with the prefactors, the Wick
permutations and the diagrams exactly as ``compute_cumulants`` sums the same
subsets, one loop order on one side is a block-diagonal triple
``C(u) = c e^{G u} x0`` (:class:`SideME`, blocks :class:`MEBlock`).

From the triple, without poles:

* ``C(u)``: the segment's own Taylor action (the arithmetic of ``C_tau``);
* pairings ``∫ L(τ) C(τ) dτ`` with exponential and polynomial × exponential
  kernels: ``q! c (zI − G)^{−(q+1)} x0``, one triangular resolvent solve per
  block (the segment in its per-edge triangular basis is upper triangular
  with the exact eigenvalue sums on the diagonal); rational kernels (a SISO
  realization ``c_L e^{A_L u} b_L``) by one Sylvester solve per block;
* the spectrum ``S(ω) = c (iω − G)^{−1} x0`` (τ > 0) plus the mirror
  ``c (−iω − G)^{−1} x0`` (τ < 0) plus the δ floor;
* the mode list ``Σ coeff · τ^n e^{λτ}``: eigenvalue clusters of the
  triangular generator, decoupled by Sylvester solves, ``τ^n`` terms for
  repeated eigenvalues (Jordan chains); a :class:`MEConditioningWarning`
  when the decomposition or the cancellation between modes is
  ill-conditioned (the triple itself is not affected).

Conventions (k = 2) are those of ``C_tau``: ``C(τ) = κ(x_1(0), x_2(τ))``;
side ``'+'`` is τ > 0, ``'-'`` is τ < 0; a :class:`~api.cumulant.ModeTerm`
``(coeff, rate, power, side)`` is ``coeff · τ**power · exp(rate · τ)`` in the
signed τ (as item 23's ``ModeSum``).  At τ = 0 the default is the Itô left
limit (side '-', exactly); ``C_tau``'s τ = 0 point is sampled at
``τ = −api.compute._ITO_EPS`` instead.

Where the triple is not available the object says why (``reasons``) and
nothing is guessed: a subset on the pole route (the state-space route
declined it), a pole-cluster deformation run, grouped Phase J, a finite
``PHASE_J_T_MIN``, k ≠ 2 (see :func:`slice_me` for the k >= 3 slices).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, NamedTuple, Optional

import numpy as np
import scipy.linalg as sl

__all__ = ['MEBlock', 'SideME', 'CumulantME', 'MEUnavailable',
           'MEConditioningWarning', 'MEFallbackWarning', 'RationalKernel',
           'Pairing', 'from_result', 'slice_me', 'kernel_value',
           'kernel_at_zero', 'quad_pairing']

#: eigenvalues of one block closer than this (relative to the largest
#: |eigenvalue|) form one cluster (a Jordan chain: τ^n terms)
CLUSTER_RTOL = 1e-9
#: modes of different blocks are merged when their rates agree to this
#: (relative); the merged rate is the cluster mean
MERGE_RTOL = 1e-12
#: warn when Σ|terms| exceeds |C| by this factor at some u in the check grid,
#: or when a decoupling transformation has a norm above it
COND_WARN = 1e6
#: largest block the triangular / dense forms are built for
MAX_BLOCK = 2000


class MEUnavailable(ValueError):
    """The matrix-exponential triple of this result is not available (the
    message lists the reasons)."""


class MEFallbackWarning(UserWarning):
    """``Cumulant.spectrum`` / ``integrate_against`` with ``method='auto'``
    had no matrix-exponential triples for the requested loop orders and
    integrated the Phase J callables numerically instead."""


class MEConditioningWarning(UserWarning):
    """The mode list derived from a triple is ill-conditioned (nearly
    coincident eigenvalue clusters, or modes that cancel); the triple, its
    values and its resolvent pairings are not affected."""


# ─── one block ──────────────────────────────────────────────────────────────

class MEBlock:
    """One subset of one diagram in one Wick permutation, on one side:
    ``weight · const · Π_r g_r(β_r u) · c e^{G β u} x0`` (``u = |τ|``).

    Attributes: ``diagram_index`` (index into the loop order's diagrams, as
    ``DeltaComponent.diagram_index``), ``subset_index``, ``perm_index``,
    ``weight`` (prefactor / Wick compensation × the constant rows), ``n``
    (state count of the dense triple).  :meth:`triple` gives the dense
    ``(G, x0, c)`` with the weight folded into ``c``."""

    __slots__ = ('weight', 'seg', 'beta', 'x0', 'c', 'rows', 'dtype',
                 'diagram_index', 'subset_index', 'perm_index', '_tri',
                 '_triple')

    def __init__(self, ray, weight, dtype, diagram_index, subset_index,
                 perm_index):
        self.weight = complex(weight) * complex(ray['const'])
        self.seg = ray['seg']
        self.beta = float(ray['beta'])
        self.x0 = ray['x0']
        self.c = ray['c']
        self.rows = list(ray['rows'])
        self.dtype = dtype
        self.diagram_index = diagram_index
        self.subset_index = subset_index
        self.perm_index = perm_index
        self._tri = None
        self._triple = None

    @property
    def n(self):
        n = self.seg.D if self.seg is not None else 1
        for _b, ent in self.rows:
            n *= ent.n
        return n

    @property
    def constant(self):
        """True for a block that does not move with u (no segment, no row):
        a non-decaying constant (its resolvent is singular at z = 0)."""
        return self.seg is None and not self.rows

    # values
    def values(self, u):
        """The block at the sorted array ``u >= 0`` (the segment's Taylor
        action stepped from one length to the next, as ``C_tau``'s τ
        batches; the rows by their own impulses)."""
        u = np.asarray(u, dtype=float)
        out = np.zeros(u.shape, dtype=complex)
        if self.weight == 0:
            return out
        order = np.argsort(u, kind='stable')
        dt = self.dtype
        v = self.x0
        cur = 0.0
        for i in order:
            ui = float(u[i])
            if self.seg is not None:
                L = self.beta * ui
                if L > cur:
                    v = self.seg.action(v, L - cur, dt)
                    cur = L
                val = complex(self.c @ v)
            else:
                val = 1.0 + 0j
            for b, ent in self.rows:
                val *= ent.impulse(b * ui)
            out[i] = self.weight * val
        return out

    # dense forms
    def triple(self):
        """``(G, x0, c)`` dense, in the segment's stacked basis (Kronecker
        product with the rows' realizations), weight folded into ``c``:
        the block is ``c @ expm(G u) @ x0``."""
        if self._triple is None:
            if self.n > MAX_BLOCK:
                raise MEUnavailable(f'a block of {self.n} states is beyond '
                                    f'the dense forms (MAX_BLOCK={MAX_BLOCK})')
            if self.seg is not None:
                G = self.beta * np.asarray(self.seg.dense(self.dtype)[0])
                x, c = self.x0, self.c
            else:
                G = np.zeros((1, 1))
                x, c = np.ones(1), np.ones(1)
            for b, ent in self.rows:
                G = _kron_sum(G, b * ent.a)
                x = np.kron(x, ent.b)
                c = np.kron(c, ent.c)
            self._triple = (G, np.asarray(x), self.weight * np.asarray(c))
        return self._triple

    def tri(self):
        """``(T, xt, ct)``: the same block in the per-edge triangular basis,
        ``T`` upper triangular with the exact eigenvalue sums on its
        diagonal (``_Segment.tri``); the weight is folded into ``ct``."""
        if self._tri is None:
            if self.n > MAX_BLOCK:
                raise MEUnavailable(f'a block of {self.n} states is beyond '
                                    f'the dense forms (MAX_BLOCK={MAX_BLOCK})')
            if self.seg is not None:
                seg = self.seg
                Gt, roffs = seg.tri()
                T = self.beta * np.asarray(Gt)
                xt = np.zeros(seg.D, dtype=complex)
                ct = np.zeros(seg.D, dtype=complex)
                for k in range(len(seg.blocks)):
                    o, sz = seg.offs[k], seg.sizes[k]
                    x = self.x0[o:o + sz].reshape(seg.shapes[k])
                    c = self.c[o:o + sz].reshape(seg.shapes[k])
                    for ax, e in enumerate(seg.ents[k]):
                        x = e.to_basis(x, ax)
                        c = _c_to_basis(e, c, ax)
                    xt[roffs[k]:roffs[k] + sz] = np.ravel(x)
                    ct[roffs[k]:roffs[k] + sz] = np.ravel(c)
            else:
                T = np.zeros((1, 1))
                xt, ct = np.ones(1, dtype=complex), np.ones(1, dtype=complex)
            for b, ent in self.rows:
                T = _kron_sum(T, b * np.asarray(ent.T))
                xt = np.kron(xt, ent.bt)
                ct = np.kron(ct, ent.ct)
            self._tri = (np.asarray(T, dtype=complex), xt,
                         self.weight * ct)
        return self._tri

    # resolvents
    def resolvent(self, z, power=1):
        """``c (z I − G)^{−power} x0`` (one triangular solve per power)."""
        T, xt, ct = self.tri()
        if self.constant and z == 0:
            raise ZeroDivisionError('a constant block has no resolvent at 0')
        A = z * np.eye(T.shape[0]) - T
        y = xt
        for _ in range(int(power)):
            y = sl.solve_triangular(A, y, lower=False, check_finite=False)
        return complex(ct @ y)

    def sylvester(self, A_L, b_L, c_L):
        """``∫_0^∞ c_L e^{A_L u} b_L · c e^{G u} x0 du`` for a Hurwitz SISO
        kernel realization: ``c Y c_L^T`` with ``T Y + Y A_L^T =
        −xt b_L^T``."""
        T, xt, ct = self.tri()
        rhs = -np.outer(xt, b_L)
        Y = sl.solve_sylvester(T, np.asarray(A_L).T, rhs)
        return complex(ct @ Y @ np.asarray(c_L))

    def __repr__(self):
        return (f'MEBlock(diagram={self.diagram_index}, '
                f'subset={self.subset_index}, perm={self.perm_index}, '
                f'n={self.n}, weight={self.weight:.6g})')


def _kron_sum(A, B):
    return (np.kron(A, np.eye(B.shape[0], dtype=B.dtype))
            + np.kron(np.eye(A.shape[0], dtype=A.dtype), B))


def _c_to_basis(e, c, ax):
    """The contraction ``c`` in an edge's triangular basis: ``c . x =
    c_t . x_t`` with ``x = from_basis(x_t)``."""
    if e.kind == 'lower':
        return np.flip(c, axis=ax)
    if e.kind == 'schur':
        return np.moveaxis(np.tensordot(e.U.T, c, axes=(1, ax)), 0, ax)
    return c


# ─── one side of one loop order ─────────────────────────────────────────────

class SideME:
    """One loop order on one side (``'+'``: τ > 0, ``'-'``: τ < 0) as the
    block-diagonal triple ``C(u) = Σ_blocks c_b e^{G_b u} x0_b``,
    ``u = |τ|``."""

    def __init__(self, side, blocks, ell):
        self.side = side
        self.blocks = [b for b in blocks if b.weight != 0]
        self.ell = ell

    @property
    def n_states(self):
        return sum(b.n for b in self.blocks)

    def __call__(self, u):
        """``C(u)``, ``u = |τ| >= 0`` (array)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if np.any(u < 0) or not np.all(np.isfinite(u)):
            raise ValueError('u = |τ| must be finite and >= 0')
        out = np.zeros(u.shape, dtype=complex)
        for b in self.blocks:
            out += b.values(u)
        return out

    def triple(self):
        """The dense block-diagonal ``(G, x0, c)`` (weights in ``c``)."""
        if not self.blocks:
            return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
        parts = [b.triple() for b in self.blocks]
        G = sl.block_diag(*[p[0] for p in parts])
        x = np.concatenate([p[1] for p in parts])
        c = np.concatenate([p[2] for p in parts])
        return G, x, c

    def resolvent(self, z, power=1):
        """``c (z I − G)^{−power} x0`` summed over the blocks."""
        return sum((b.resolvent(z, power) for b in self.blocks), 0j)

    def laplace(self, z, power=0):
        """``∫_0^∞ u^power e^{−z u} C(u) du = power! c (zI − G)^{−power−1}
        x0`` (``Re z`` above the rightmost eigenvalue: always for
        ``Re z >= 0`` with Hurwitz blocks)."""
        q = int(power)
        return math.factorial(q) * self.resolvent(z, q + 1)

    def sylvester(self, A_L, b_L, c_L):
        return sum((b.sylvester(A_L, b_L, c_L) for b in self.blocks), 0j)

    def modes(self, cluster_rtol=CLUSTER_RTOL, merge_rtol=MERGE_RTOL,
              warn=True):
        """``[(coeff, lam, n)]``: ``C(u) = Σ coeff u^n e^{lam u}`` (u =
        |τ|), merged over the blocks; with ``warn``, a
        :class:`MEConditioningWarning` when the decomposition is
        ill-conditioned."""
        terms = []
        cond = 0.0
        for b in self.blocks:
            t, cnd = _block_modes(b, cluster_rtol)
            terms.extend(t)
            cond = max(cond, cnd)
        terms = _merge_terms(terms, merge_rtol)
        if warn:
            _check_conditioning(self, terms, cond)
        return terms

    def __repr__(self):
        return (f'SideME(ell={self.ell}, side={self.side!r}, '
                f'blocks={len(self.blocks)}, states={self.n_states})')


def _block_modes(b, rtol):
    """Mode terms ``(coeff, lam, n)`` of one block and the largest norm of
    its decoupling transformations."""
    T, xt, ct = b.tri()
    n = T.shape[0]
    d = np.diag(T).copy()
    scale = max(1.0, float(np.max(np.abs(d)))) if n else 1.0
    labels = _clusters(d, rtol * scale)
    K = int(labels.max()) + 1 if n else 0
    centers = [d[labels == k].mean() for k in range(K)]
    T = np.array(T, dtype=complex, copy=True)
    Q = np.eye(n, dtype=complex)
    starts, p = [], 0
    from scipy.linalg.lapack import ztrsen, ztrsyl
    for k in range(K):
        starts.append(p)
        if k == K - 1:
            break
        sub = T[p:, p:]
        dd = np.diag(sub)
        lab = np.array([int(np.argmin([abs(x - c) for c in centers]))
                        for x in dd])
        sel = (lab == k).astype(np.int32)
        if not sel.any():
            continue
        ts, qs, _w, m, _s, _sep, info = ztrsen(sel, sub, np.eye(
            sub.shape[0], dtype=complex), job='N')
        if info != 0:
            raise MEUnavailable(f'eigenvalue reordering failed (info={info})')
        T[p:, p:] = ts
        T[:p, p:] = T[:p, p:] @ qs
        Q[:, p:] = Q[:, p:] @ qs
        p += int(m)
    starts.append(n)
    x = Q.conj().T @ xt
    c = ct @ Q
    cond = 0.0
    # decouple: [[A, C], [0, B]] -> diag(A, B) with A X - X B = -C
    for k in range(K - 1):
        r0, r1 = starts[k], starts[k + 1]
        if r1 == r0 or r1 == n:
            continue
        A, B, C = T[r0:r1, r0:r1], T[r1:, r1:], T[r0:r1, r1:]
        X, scale_, info = ztrsyl(A, B, -C, trana='N', tranb='N', isgn=-1)
        if info < 0:
            raise MEUnavailable(f'Sylvester decoupling failed (info={info})')
        X = X / scale_
        cond = max(cond, float(np.abs(X).sum(axis=1).max(initial=0.0)))
        x[r0:r1] = x[r0:r1] - X @ x[r1:]
        c[r1:] = c[r1:] + c[r0:r1] @ X
        T[r0:r1, r1:] = 0.0
    out = []
    for k in range(K):
        r0, r1 = starts[k], starts[k + 1]
        if r1 == r0:
            continue
        A = T[r0:r1, r0:r1]
        lam = complex(np.mean(np.diag(A)))
        N = A - lam * np.eye(r1 - r0)
        y = x[r0:r1].copy()
        cc = c[r0:r1]
        for j in range(r1 - r0):
            coeff = complex(cc @ y) / math.factorial(j)
            if coeff != 0:
                out.append((coeff, lam, j))
            y = N @ y
    return out, cond


def _clusters(d, tol):
    """Labels 0..K-1 of the connected components of ``|d_i - d_j| <= tol``,
    numbered by first appearance."""
    n = len(d)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if n:
        D = np.abs(d[:, None] - d[None, :]) <= tol
        ii, jj = np.nonzero(np.triu(D, 1))
        for i, j in zip(ii, jj):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    lab, out = {}, np.zeros(n, dtype=int)
    for i, r in enumerate(roots):
        out[i] = lab.setdefault(r, len(lab))
    return out


def _merge_terms(terms, rtol):
    if not terms:
        return []
    lams = np.array([t[1] for t in terms])
    scale = max(1.0, float(np.max(np.abs(lams))))
    lab = _clusters(lams, rtol * scale)
    acc = {}
    cen = {}
    for (c, lam, j), k in zip(terms, lab):
        acc[(k, j)] = acc.get((k, j), 0j) + c
        cen.setdefault(k, []).append(lam)
    out = []
    for (k, j), c in acc.items():
        if c != 0:
            out.append((c, complex(np.mean(cen[k])), j))
    out.sort(key=lambda t: (-t[1].real, t[1].imag, t[2]))
    return out


def _check_conditioning(side, terms, cond):
    if cond > COND_WARN:
        warnings.warn(f'ell={side.ell} side {side.side!r}: the mode '
                      f'decomposition needed a transformation of norm '
                      f'{cond:.3g} (nearly coincident eigenvalue clusters); '
                      f'the mode coefficients are ill-conditioned (the '
                      f'triple is not)', MEConditioningWarning, stacklevel=3)
        return
    if not terms:
        return
    u = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    mag = np.zeros(u.shape)
    val = np.zeros(u.shape, dtype=complex)
    for c, lam, j in terms:
        t = c * u ** j * np.exp(lam * u)
        mag += np.abs(t)
        val += t
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(val) > 0, mag / np.abs(val), np.inf)
    r = float(np.max(ratio[mag > 1e-300], initial=0.0))
    if r > COND_WARN:
        warnings.warn(f'ell={side.ell} side {side.side!r}: the modes cancel '
                      f'(Σ|terms| / |C| up to {r:.3g}): summing them in '
                      f'double precision loses about {math.log10(r):.0f} '
                      f'digits; evaluate the triple instead',
                      MEConditioningWarning, stacklevel=3)


# ─── kernels ────────────────────────────────────────────────────────────────

class RationalKernel:
    """A kernel with a rational Laplace transform on one side::

        side '+':  ∫_0^∞ L(s) e^{−p s} ds = num(p) / den(p)
        side '-':  ∫_0^∞ L(−s) e^{−p s} ds = num(p) / den(p)

    (strictly proper, ``den`` Hurwitz).  ``num`` / ``den``: coefficient
    lists, highest power first (``numpy.polyval`` order).  Realized as the
    controllable canonical form ``L(±s) = c_L e^{A_L s} b_L``; paired with a
    triple by one Sylvester solve per block (no poles)."""

    def __init__(self, num, den, side='+'):
        num = np.atleast_1d(np.asarray(num, dtype=complex))
        den = np.atleast_1d(np.asarray(den, dtype=complex))
        den = np.trim_zeros(den, 'f')
        num = np.trim_zeros(num, 'f') if np.any(num) else np.zeros(1)
        if side not in ('+', '-'):
            raise ValueError("side must be '+' or '-'")
        if len(den) < 1 or den[0] == 0:
            raise ValueError('den must have a nonzero leading coefficient')
        if len(num) >= len(den):
            raise ValueError('num/den must be strictly proper (a δ or its '
                             'derivatives are not a kernel term here)')
        num, den = num / den[0], den / den[0]
        n = len(den) - 1
        A = np.zeros((n, n), dtype=complex)
        A[0, :] = -den[1:]
        if n > 1:
            A[1:, :-1] = np.eye(n - 1)
        b = np.zeros(n, dtype=complex)
        b[0] = 1.0
        c = np.zeros(n, dtype=complex)
        c[n - len(num):] = num
        if np.any(np.real(np.linalg.eigvals(A)) >= 0):
            raise ValueError('den is not Hurwitz: the kernel does not decay')
        if np.all(np.isreal(A)) and np.all(np.isreal(c)):
            A, b, c = A.real, b.real, c.real
        self.side, self.num, self.den = side, num, den
        self.A, self.b, self.c = A, b, c

    def at_zero(self):
        """``L(0±)`` on its side (``c_L b_L``)."""
        return complex(self.c @ self.b)

    def __call__(self, s):
        """``L(s)`` on the kernel's side (0 on the other)."""
        s = float(s)
        u = s if self.side == '+' else -s
        if u < 0:
            return 0j
        return complex(self.c @ sl.expm(self.A * u) @ self.b)

    def __repr__(self):
        return (f'RationalKernel(num={self.num.tolist()}, '
                f'den={self.den.tolist()}, side={self.side!r})')


class Pairing(NamedTuple):
    """``∫ L(τ) C(τ) dτ``: ``value`` = ``smooth`` + ``contact`` (the δ
    components), ``method`` ('exact' / 'quad'), ``abs_error`` (the
    quadrature's estimate; ``None`` for exact)."""
    value: complex
    smooth: complex
    contact: complex
    method: str
    abs_error: Optional[float] = None

    def __complex__(self):
        return complex(self.value)

    def __float__(self):
        return float(np.real(self.value))


def _as_list(kernel):
    return list(kernel) if isinstance(kernel, (list, tuple)) else [kernel]


def kernel_value(kernel, s, values=None):
    """``L(s)`` of a ``symbolic_out.Kernel``, a :class:`RationalKernel` or
    a list of them (s != 0)."""
    s = float(s)
    tot = 0j
    for k_ in _as_list(kernel):
        if isinstance(k_, RationalKernel):
            tot += k_(s)
            continue
        for a, r, q, side in _kernel_terms(k_, values):
            if (side == '+') == (s > 0):
                tot += a * s ** q * np.exp(r * s)
    return tot


def kernel_at_zero(kernel, values=None):
    """``(L(0+), L(0-))``."""
    lp = lm = 0j
    for k_ in _as_list(kernel):
        if isinstance(k_, RationalKernel):
            if k_.side == '+':
                lp += k_.at_zero()
            else:
                lm += k_.at_zero()
            continue
        for a, _r, q, side in _kernel_terms(k_, values):
            if q == 0:
                if side == '+':
                    lp += a
                else:
                    lm += a
    return lp, lm


def quad_pairing(C, kernel, values, epsabs=1e-12, epsrel=1e-10, limit=400):
    """``∫ L(τ) C(τ) dτ`` by adaptive quadrature of ``C`` (a callable of
    the signed τ) on (−∞, 0) and (0, ∞): ``(value, abs_error)``."""
    from scipy.integrate import quad
    tot, err = 0j, 0.0
    for sg in (1.0, -1.0):
        def f(u, sg=sg):
            return kernel_value(kernel, sg * u, values) * C(sg * u)
        kw = dict(epsabs=epsabs, epsrel=epsrel, limit=limit)
        re, e1 = quad(lambda u: f(u).real, 0.0, np.inf, **kw)
        im, e2 = quad(lambda u: f(u).imag, 0.0, np.inf, **kw)
        tot += re + 1j * im
        err += e1 + e2
    return tot, err


def _kernel_terms(kernel, values):
    """``[(a, rate, power, side)]`` numeric terms of a ``symbolic_out.Kernel``
    (``side`` '+' / '-')."""
    out = []
    for t in kernel.terms:
        a = _num(t.coeff, values)
        r = _num(t.rate, values)
        side = '+' if t.side == 'positive' else '-'
        out.append((a, r, int(t.power), side))
    return out


def _num(x, values):
    try:
        return complex(x)
    except (TypeError, ValueError):
        pass
    from sage.all import SR
    e = SR(x)
    if values:
        sub = {v: SR(values[str(v)]) for v in e.variables()
               if str(v) in values}
        e = e.subs(sub)
    try:
        return complex(e.n())
    except (TypeError, ValueError):
        raise ValueError(f'kernel term {x} has free symbols '
                         f'{[str(v) for v in e.variables()]}: pass values=')


# ─── the whole result ───────────────────────────────────────────────────────

class CumulantME:
    """The matrix-exponential form of a k = 2 result's smooth part:
    ``sides[ell]['+' | '-']`` = :class:`SideME`.  ``complete`` is False when
    some smooth term has no triple (``reasons`` says which; the methods then
    raise :class:`MEUnavailable` for that loop order)."""

    def __init__(self, k, sides, reasons, counts):
        self.k = k
        self.sides = sides
        self.reasons = reasons            # {ell: [str]}
        self.counts = counts

    @property
    def complete(self):
        return not any(self.reasons.values())

    @property
    def orders(self):
        return sorted(self.sides)

    def _ells(self, ell):
        ells = self.orders if ell is None else (
            [ell] if np.ndim(ell) == 0 else list(ell))
        bad = [e for e in ells if self.reasons.get(e)]
        if bad:
            raise MEUnavailable('; '.join(
                f'ell={e}: ' + '; '.join(self.reasons[e]) for e in bad))
        missing = [e for e in ells if e not in self.sides]
        if missing:
            raise ValueError(f'loop order(s) {missing} not in the result '
                             f'(orders: {self.orders})')
        return ells

    def __call__(self, tau, ell=None, at_zero='negative'):
        """``C(τ)`` (signed ``τ``, array) summed over ``ell``.  At ``τ = 0``
        the limit from ``at_zero`` ('negative': the Itô left limit, exact;
        'positive')."""
        if at_zero not in ('negative', 'positive'):
            raise ValueError("at_zero must be 'negative' or 'positive'")
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        pos = (tau > 0) | ((tau == 0) & (at_zero == 'positive'))
        out = np.zeros(tau.shape, dtype=complex)
        for e in self._ells(ell):
            if pos.any():
                out[pos] += self.sides[e]['+'](tau[pos])
            if (~pos).any():
                out[~pos] += self.sides[e]['-'](-tau[~pos])
        return out

    def modes(self, ell=None, **kw):
        """``{ell: [ModeTerm(coeff, rate, power, side)]}`` in the signed τ
        (see :meth:`SideME.modes` for the options)."""
        from api.cumulant import ModeTerm
        out = {}
        for e in self._ells(ell):
            terms = []
            for c, lam, j in self.sides[e]['+'].modes(**kw):
                terms.append(ModeTerm(c, lam, j, '+'))
            for c, lam, j in self.sides[e]['-'].modes(**kw):
                # u = −τ: c u^j e^{λu} = c (−1)^j τ^j e^{−λτ}
                terms.append(ModeTerm(c * (-1) ** j, -lam, j, '-'))
            out[e] = terms
        return out

    def integrate_smooth(self, kernel, ell=None, values=None):
        """``∫ L(τ) C_smooth(τ) dτ`` (no δ part) for a
        ``symbolic_out.Kernel`` (exponential / polynomial × exponential
        terms), a :class:`RationalKernel`, or a list of them."""
        if isinstance(kernel, (list, tuple)):
            return sum((self.integrate_smooth(k_, ell, values)
                        for k_ in kernel), 0j)
        tot = 0j
        for e in self._ells(ell):
            if isinstance(kernel, RationalKernel):
                tot += self.sides[e][kernel.side].sylvester(
                    kernel.A, kernel.b, kernel.c)
                continue
            for a, r, q, side in _kernel_terms(kernel, values):
                if a == 0:
                    continue
                if side == '+':
                    if r.real > 0:
                        raise ValueError(f'kernel term rate {r} grows for s > 0')
                    # ∫_0^∞ a u^q e^{r u} C_+(u) du = a q! c (−r − G)^{−q−1} x0
                    tot += a * self.sides[e]['+'].laplace(-r, q)
                else:
                    if r.real < 0:
                        raise ValueError(f'kernel term rate {r} grows for s < 0')
                    # s = −u: a (−u)^q e^{−r u}
                    tot += a * (-1) ** q * self.sides[e]['-'].laplace(r, q)
        return tot

    def spectrum_smooth(self, omega, ell=None):
        """``∫ e^{−iωτ} C_smooth(τ) dτ`` = ``c (iω − G_+)^{−1} x0 + c (−iω −
        G_−)^{−1} x0`` (array over ``omega``)."""
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        out = np.zeros(omega.shape, dtype=complex)
        for e in self._ells(ell):
            for i, w in enumerate(omega):
                out[i] += (self.sides[e]['+'].resolvent(1j * w)
                           + self.sides[e]['-'].resolvent(-1j * w))
        return out

    def __repr__(self):
        st = ', '.join(f'{e}: {len(s["+"].blocks)}+{len(s["-"].blocks)} '
                       f'blocks' for e, s in sorted(self.sides.items()))
        return (f'CumulantME(k={self.k}, {st}, complete={self.complete})')


def _groups_of(pj):
    return [g for g in (pj.get('groups') or []) if isinstance(g, dict)]


def from_result(result) -> CumulantME:
    """Build the triples of a k = 2 ``compute_cumulants`` / ``dd.run``
    result from its live Phase J closures (nothing is integrated again: the
    stationary states and segments are the route's cached ones).  Raises
    :class:`MEUnavailable` for results that cannot have one at all (k != 2,
    spatial, a loaded file, moment output); per-loop-order gaps are in
    ``reasons``."""
    from engine.integration.time_domain import final_integral as FI
    cfg = result.get('config') or {}
    k = int(cfg.get('k', 0))
    if k != 2:
        raise MEUnavailable(f'triples on the lag axis are defined for k = 2 '
                            f'(got k = {k}); for k >= 3 use slice_me')
    if 'C_tau_x' in result or cfg.get('spatial'):
        raise MEUnavailable('spatial results have no state-space triples')
    if cfg.get('output', 'cumulant') not in (None, 'cumulant'):
        raise MEUnavailable(f"output={cfg.get('output')!r}: triples are "
                            f"built for the connected cumulant")
    pjs = result.get('phase_j_by_ell')
    if not pjs:
        raise MEUnavailable('the result has no live Phase J closures (a '
                            'loaded file or a non-Phase-J result)')
    global_reasons = []
    eng = result.get('phase_j_engine') or {}
    if eng.get('pole_deformation') == 'used':
        global_reasons.append('the pole-cluster deformation ran (the value '
                              'is an average over deformed propagators)')
    if eng.get('engine') == 'poles':
        global_reasons.append('DAEDALUS_PHASE_J_ENGINE=poles')
    if cfg.get('use_grouped_phase_j'):
        global_reasons.append('grouped Phase J')
    try:
        if float(FI.PHASE_J_T_MIN) != float('-inf'):
            global_reasons.append('finite PHASE_J_T_MIN')
    except (TypeError, ValueError):
        global_reasons.append('PHASE_J_T_MIN not a float')
    sides, reasons = {}, {}
    counts = {'blocks': 0, 'zero': 0, 'pole_route': 0, 'pruned': 0}
    for ell, pj in sorted(pjs.items()):
        ell = int(ell)
        rs = list(global_reasons)
        blocks = {'+': [], '-': []}
        if pj:
            sk = list(pj.get('skipped_kernel_ids') or [])
            if sk:
                rs.append(f'{len(sk)} diagram(s) failed in Phase J')
            for gi, g in enumerate(_groups_of(pj)):
                fn = g.get('contribution')
                if fn is None:
                    continue
                parts = getattr(fn, 'phase_j_parts', None)
                if parts is None:
                    rs.append(f'diagram {g.get("kernel_id", gi)}: its '
                              f'contribution exposes no subset structure')
                    continue
                _blocks_of(parts, g.get('kernel_id', gi), blocks, rs,
                           counts, FI)
        if rs:
            # collapse repeated reasons
            seen = {}
            for r in rs:
                seen[r] = seen.get(r, 0) + 1
            rs = [r if n == 1 else f'{r} (x{n})' for r, n in seen.items()]
        sides[ell] = {s: SideME(s, blocks[s], ell) for s in ('+', '-')}
        reasons[ell] = rs
    return CumulantME(2, sides, reasons, counts)


def _blocks_of(parts, diagram_index, blocks, rs, counts, FI):
    perms = parts['perms']
    comp = parts['compensation']
    org = parts['origin_leaf_idx']
    fidx = parts['free_ext_idx']
    kk = parts['k']
    for side, sg in (('+', 1.0), ('-', -1.0)):
        times = (0.0, sg)
        for pidx, perm in enumerate(perms):
            permuted = [times[perm[j]] for j in range(kk)]
            if org is not None:
                t0 = permuted[org]
                permuted = [t - t0 for t in permuted]
            fu = [float(permuted[j]) for j in fidx]
            tie = FI._TieContext(times, tuple(perm[j] for j in fidx),
                                 None if org is None else perm[org])
            for si, cfn in enumerate(parts['subsets']):
                sub = getattr(cfn, 'ss_subset', None)
                if sub is None:
                    counts['pole_route'] += 1
                    rs.append('a subset is on the pole route (the '
                              'state-space route declined it)')
                    continue
                if getattr(cfn, 'ss_theta0_empty', False):
                    counts['pruned'] += 1
                    continue
                from engine.integration.time_domain import statespace as SS
                try:
                    ray = sub.me_ray(fu, tie)
                except SS.SSIneligible as exc:
                    rs.append(f'a subset is not one matrix exponential on '
                              f'the lag axis ({exc.reason})')
                    continue
                except SS._late_errors() as exc:
                    rs.append(f'the state-space route refused a subset '
                              f'({SS._late_reason(exc)})')
                    continue
                if ray is None:
                    counts['zero'] += 1
                    continue
                w = complex(cfn.ss_prefactor) / comp
                blocks[side].append(MEBlock(ray, w, sub.model.dtype,
                                            diagram_index, si, pidx))
                counts['blocks'] += 1


def slice_me(*_a, **_k):
    """k >= 3 slices (placeholder; see the module docstring)."""
    raise MEUnavailable('k >= 3 slices are not built yet')

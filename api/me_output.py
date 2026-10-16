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

__all__ = ['MEBlock', 'SideME', 'CumulantME', 'SliceME', 'MEUnavailable',
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
#: warn when a cluster's truncated Jordan expansion is off by more than this
TRUNC_WARN = 1e-12
#: dense forms of blocks up to this many states are cached on the block
CACHE_MAX = 512
#: SideME.triple() forms the dense block-diagonal triple up to this size
MAX_DENSE_TOTAL = 20000


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
        for _r0, _r1, ent in self.rows:
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
                    try:
                        v = self.seg.action(v, L - cur, dt)
                    except _late() as exc:
                        raise MEUnavailable(
                            f'the state-space route refused the segment at '
                            f'|tau| = {ui:g} ({exc})') from exc
                    cur = L
                val = complex(self.c @ v)
            else:
                val = 1.0 + 0j
            for r0, r1, ent in self.rows:
                val *= ent.impulse(r0 + r1 * ui)
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
            for r0, r1, ent in self.rows:
                G = _kron_sum(G, r1 * ent.a)
                x = np.kron(x, _row_start(ent, r0, 'orig'))
                c = np.kron(c, ent.c)
            trip = (G, np.asarray(x), self.weight * np.asarray(c))
            if self.n > CACHE_MAX:
                return trip
            self._triple = trip
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
            for r0, r1, ent in self.rows:
                T = _kron_sum(T, r1 * np.asarray(ent.T))
                xt = np.kron(xt, _row_start(ent, r0, 'tri'))
                ct = np.kron(ct, ent.ct)
            tri = (np.asarray(T, dtype=complex), xt, self.weight * ct)
            if self.n > CACHE_MAX:
                return tri              # large: rebuilt on demand
            self._tri = tri
        return self._tri

    # resolvents
    def resolvent(self, z, power=1):
        """``c (z I − G)^{−power} x0`` (one triangular solve per power; a
        block beyond the dense forms without constant rows: forward block
        substitution on the segment, :meth:`_Segment.resolve`)."""
        if self.constant and z == 0:
            raise ZeroDivisionError('a constant block has no resolvent at 0')
        if (self.n > MAX_BLOCK and self.seg is not None and not self.rows
                and self.beta != 0):
            # G = beta G_seg: (z − beta G_seg)^{-1} = (z/beta − G_seg)^{-1}/beta
            if self.weight == 0:
                return 0j
            zb = complex(z) / self.beta
            y = np.asarray(self.x0, dtype=complex)
            for _ in range(int(power)):
                y = self.seg.resolve(y, zb) / self.beta
            return complex(self.weight * (np.asarray(self.c) @ y))
        T, xt, ct = self.tri()
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
        # all complex: with a real A_L scipy takes its REAL Schur form (2 x 2
        # blocks for complex poles) and hands it to ztrsyl, which needs a
        # triangular matrix (review: wrong values for oscillatory kernels)
        A = np.asarray(A_L, dtype=complex)
        rhs = -np.outer(xt, np.asarray(b_L, dtype=complex))
        Y = sl.solve_sylvester(np.asarray(T, dtype=complex), A.T, rhs)
        return complex(ct @ Y @ np.asarray(c_L, dtype=complex))

    def __repr__(self):
        return (f'MEBlock(diagram={self.diagram_index}, '
                f'subset={self.subset_index}, perm={self.perm_index}, '
                f'n={self.n}, weight={self.weight:.6g})')


class MEChainBlock:
    """One subset on a FINITE chamber where several segments move (a k >= 3
    slice whose swept leg lies between two other legs; item 54):
    ``weight · const · Π_r g_r(ρ0 + ρ1 s) · c e^{G_M L_M(s)} M_{M-1} …
    M_1 e^{G_1 L_1(s)} x0`` with ``L_j(s) = L0_j + β_j s``, ``0 <= s <=
    u_max`` (``SubsetSS.me_chain``).

    It is one matrix exponential of the Kronecker product of the moving
    segments: the first one acts forwards on ``x0``, the last one
    transposed on ``c`` (``β_M G_Mᵀ``), a middle one as ``I ⊗ β_j G_j`` on
    its own propagator ``vec(e^{G_j L_j})``, and the fixed maps ``M_j`` are
    the contraction.  The values are evaluated segment by segment (the
    route's own Taylor actions); :meth:`triple` / :meth:`tri` give the
    Kronecker form (the mode list).  A shrinking segment contributes
    growing exponentials, so there is no Laplace transform: a finite
    chamber has values and modes only."""

    __slots__ = ('weight', 'chain', 'rows', 'dtype', 'u_max',
                 'diagram_index', 'subset_index', 'perm_index', '_tri',
                 '_triple')

    def __init__(self, ray, weight, dtype, diagram_index, subset_index,
                 perm_index, u_max):
        self.weight = complex(weight) * complex(ray['const'])
        self.chain = ray['chain']
        self.rows = list(ray['rows'])
        self.dtype = dtype
        self.u_max = float(u_max)
        self.diagram_index = diagram_index
        self.subset_index = subset_index
        self.perm_index = perm_index
        self._tri = None
        self._triple = None

    @property
    def n(self):
        ch = self.chain
        n = 1
        if ch is not None:
            Ds = [s.D for s in ch['segs']]
            n = Ds[0] if len(Ds) == 1 else Ds[0] * Ds[-1]
            for D in Ds[1:-1]:
                n *= D * D
        for _r0, _r1, ent in self.rows:
            n *= ent.n
        return n

    @property
    def constant(self):
        return self.chain is None and not self.rows

    def values(self, u):
        u = np.asarray(u, dtype=float)
        out = np.zeros(u.shape, dtype=complex)
        if self.weight == 0:
            return out
        ch, dt = self.chain, self.dtype
        for i, ui in np.ndenumerate(u):
            ui = float(ui)
            if ch is not None:
                v = ch['x0']
                try:
                    for j, seg in enumerate(ch['segs']):
                        if j:
                            v = ch['maps'][j - 1] @ v
                        L = max(ch['L0s'][j] + ch['betas'][j] * ui, 0.0)
                        if L > 0:
                            v = seg.action(v, L, dt)
                except _late() as exc:
                    raise MEUnavailable(
                        f'the state-space route refused a segment at s = '
                        f'{ui:g} ({exc})') from exc
                val = complex(ch['c'] @ v)
            else:
                val = 1.0 + 0j
            for r0, r1, ent in self.rows:
                val *= ent.impulse(max(r0 + r1 * ui, 0.0))
            out[i] = self.weight * val
        return out

    def _forms(self, basis):
        """``(G, x0, c)`` of the Kronecker form in the original stacked
        bases (``'orig'``) or the per-edge triangular ones (``'tri'``: G
        upper triangular with the exact eigenvalue sums on the diagonal)."""
        if self.n > MAX_BLOCK:
            raise MEUnavailable(f'a block of {self.n} states is beyond '
                                f'the dense forms (MAX_BLOCK={MAX_BLOCK})')
        from api._realization import expm_upper_triangular
        ch = self.chain
        if ch is None:
            G = np.zeros((1, 1))
            x, c = np.ones(1), np.ones(1)
        else:
            segs, Ls, bs = ch['segs'], ch['L0s'], ch['betas']
            Ms = list(ch['maps'])
            if basis == 'tri':
                bases = [_seg_basis(s) for s in segs]
                gens = [b[0] for b in bases]
                x_in = bases[0][1] @ ch['x0']
                c_out = bases[-1][2].T @ ch['c']
                # M_j between triangular bases: P_{j+1}^{-1} M_j P_j
                Ms = [bases[j + 1][1] @ Ms[j] @ bases[j][2]
                      for j in range(len(Ms))]

                def ex(A, L):
                    return expm_upper_triangular(A, L)
            else:
                gens = [np.asarray(s.dense(self.dtype)[0]) for s in segs]
                x_in, c_out = ch['x0'], ch['c']

                def ex(A, L):
                    return sl.expm(A * L)
            M = len(segs)
            first = ex(gens[0], Ls[0]) @ x_in
            if M == 1:
                G, x, c = bs[0] * gens[0], first, c_out
            else:
                # last factor: w = e^{G_Mᵀ (L0 + β s)} c, index reversed in
                # the triangular basis (J T_Mᵀ J is upper triangular)
                GT = gens[-1].T
                if basis == 'tri':
                    GT = GT[::-1, ::-1]
                    w0 = (ex(gens[-1], Ls[-1]).T @ c_out)[::-1]
                    Ms[-1] = Ms[-1][::-1, :]
                else:
                    w0 = ex(gens[-1], Ls[-1]).T @ c_out
                G, x = bs[-1] * GT, w0
                for j in range(M - 2, 0, -1):
                    nj = gens[j].shape[0]
                    G = _kron_sum(G, np.kron(bs[j] * gens[j], np.eye(nj)))
                    x = np.kron(x, np.ravel(ex(gens[j], Ls[j])))
                G = _kron_sum(G, bs[0] * gens[0])
                x = np.kron(x, first)
                # contraction w_p M_{M-1}[p, r] E[r, q] M_{M-2}[q, r'] ... y_t
                T = np.asarray(Ms[-1])
                for j in range(M - 2, 0, -1):
                    T = np.multiply.outer(T, np.asarray(Ms[j - 1]))
                c = np.ravel(T)
        for r0, r1, ent in self.rows:
            if basis == 'tri':
                G = _kron_sum(G, r1 * np.asarray(ent.T))
                x = np.kron(x, _row_start(ent, r0, 'tri'))
                c = np.kron(c, ent.ct)
            else:
                G = _kron_sum(G, r1 * ent.a)
                x = np.kron(x, _row_start(ent, r0, 'orig'))
                c = np.kron(c, ent.c)
        return (np.asarray(G), np.asarray(x), self.weight * np.asarray(c))

    def triple(self):
        """``(G, x0, c)`` dense (weight in ``c``): the block is ``c @
        expm(G s) @ x0`` for ``0 <= s <= u_max``."""
        if self._triple is None:
            trip = self._forms('orig')
            if self.n > CACHE_MAX:
                return trip
            self._triple = trip
        return self._triple

    def tri(self):
        if self._tri is None:
            T, x, c = self._forms('tri')
            tri = (np.asarray(T, dtype=complex), x.astype(complex),
                   c.astype(complex))
            if self.n > CACHE_MAX:
                return tri
            self._tri = tri
        return self._tri

    def resolvent(self, z, power=1):
        raise MEUnavailable('a middle chamber is a finite interval of the '
                            'swept time: values and modes only (no Laplace '
                            'transform)')

    def sylvester(self, A_L, b_L, c_L):
        self.resolvent(0.0)

    def __repr__(self):
        M = 0 if self.chain is None else len(self.chain['segs'])
        return (f'MEChainBlock(diagram={self.diagram_index}, '
                f'subset={self.subset_index}, perm={self.perm_index}, '
                f'moving={M}, n={self.n}, weight={self.weight:.6g})')


def _seg_basis(seg):
    """``(T, Pinv, P)`` of a segment: ``T`` its generator in the per-edge
    triangular basis (``_Segment.tri``, upper triangular), ``Pinv`` the
    change of basis from the stacked state to it, ``P`` its inverse."""
    Gt, roffs = seg.tri()
    D = seg.D
    Pinv = np.zeros((D, D), dtype=complex)
    P = np.zeros((D, D), dtype=complex)
    for k in range(len(seg.blocks)):
        o, sz, r = seg.offs[k], seg.sizes[k], roffs[k]
        sh = seg.shapes[k]
        X = np.eye(sz, dtype=complex).reshape(sh + (sz,))
        Y = X.copy()
        for ax, e in enumerate(seg.ents[k]):
            X = e.to_basis(X, ax)
            Y = e.from_basis(Y, ax)
        Pinv[r:r + sz, o:o + sz] = X.reshape(sz, sz)
        P[o:o + sz, r:r + sz] = Y.reshape(sz, sz)
    return np.asarray(Gt, dtype=complex), Pinv, P


def _row_start(ent, r0, basis):
    """``e^{a r0} b`` of a row entry, in its original or triangular basis
    (``r0 >= 0``; the triangular basis by the reset-diagonal Pade)."""
    if basis == 'tri':
        if r0 == 0:
            return np.asarray(ent.bt)
        from api._realization import expm_upper_triangular
        return expm_upper_triangular(np.asarray(ent.T), r0) @ ent.bt
    if r0 == 0:
        return np.asarray(ent.b)
    from api._realization import expm_upper_triangular
    v = expm_upper_triangular(np.asarray(ent.T), r0) @ ent.bt
    return ent.from_basis(v, 0)


def _late():
    from engine.integration.time_domain import statespace as SS
    return SS._late_errors()


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

    def __init__(self, side, blocks, ell, u_max=math.inf):
        self.side = side
        self.blocks = [b for b in blocks if b.weight != 0]
        self.ell = ell
        # the largest u of the side: a finite chamber of a k >= 3 slice
        self.u_max = float(u_max)

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
        if self.n_states > MAX_DENSE_TOTAL:
            raise MEUnavailable(f'{self.n_states} states in all: the dense '
                                f'block-diagonal triple is not formed '
                                f'(MAX_DENSE_TOTAL={MAX_DENSE_TOTAL}); use '
                                f'the blocks (.blocks[i].triple())')
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
        cond = trunc = 0.0
        for b in self.blocks:
            t, cnd, tr = _block_modes(b, cluster_rtol)
            terms.extend(t)
            cond, trunc = max(cond, cnd), max(trunc, tr)
        terms = _merge_terms(terms, merge_rtol)
        if warn:
            if trunc > TRUNC_WARN:
                warnings.warn(
                    f'ell={self.ell} side {self.side!r}: an eigenvalue '
                    f'cluster of spread delta was represented as one '
                    f'Jordan chain; its truncated terms are off by about '
                    f'{trunc:.2g} (relative) over five decay times: lower '
                    f'cluster_rtol (now {cluster_rtol:g})',
                    MEConditioningWarning, stacklevel=2)
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
    # a chain block's diagonal holds sums of growing and decaying rates:
    # equal rates cancel to rounding (λ - λ = O(eps |λ|), not 0), which a
    # purely relative test never merges
    atol = (64 * np.finfo(float).eps * float(np.max(np.abs(d), initial=0.0))
            if isinstance(b, MEChainBlock) else 0.0)
    labels = _clusters(d, rtol, atol)
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
        if info == 1:                   # eigenvalues perturbed: near-equal
            cond = math.inf
        X = X / scale_
        cond = max(cond, float(np.abs(X).sum(axis=1).max(initial=0.0)))
        x[r0:r1] = x[r0:r1] - X @ x[r1:]
        c[r1:] = c[r1:] + c[r0:r1] @ X
        T[r0:r1, r1:] = 0.0
    out = []
    worst_trunc = 0.0
    for k in range(K):
        r0, r1 = starts[k], starts[k + 1]
        if r1 == r0:
            continue
        A = T[r0:r1, r0:r1]
        lam = complex(np.mean(np.diag(A)))
        N = A - lam * np.eye(r1 - r0)
        # exp(N u) is truncated at u^(m-1): exact when the cluster's
        # eigenvalues are equal; their spread delta costs ~ (delta h)^m / m!
        # over the decay horizon h = 5 / |Re lam| (charged to ``cond``)
        delta = float(np.max(np.abs(np.diag(A) - lam)))
        if delta > 0:
            h = min(5.0 / max(abs(lam.real), 1e-300),
                    getattr(b, 'u_max', math.inf))
            m_ = r1 - r0
            trunc = (delta * h) ** m_ / math.factorial(m_)
            worst_trunc = max(worst_trunc, trunc)
        y = x[r0:r1].copy()
        cc = c[r0:r1]
        for j in range(r1 - r0):
            coeff = complex(cc @ y) / math.factorial(j)
            if coeff != 0:
                out.append((coeff, lam, j))
            y = N @ y
    return out, cond, worst_trunc


def _clusters(d, rtol, atol=0.0):
    """Labels 0..K-1 of the connected components of ``|d_i - d_j| <= rtol
    max(|d_i|, |d_j|) + atol`` (pairwise relative: a slow pair in a stiff
    block is judged on its own scale), numbered by first appearance."""
    n = len(d)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if n:
        mag = np.maximum(np.abs(d)[:, None], np.abs(d)[None, :])
        D = np.abs(d[:, None] - d[None, :]) <= rtol * mag + atol
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
    lab = _clusters(lams, rtol)
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
    # Σ|terms| against the size of C over a grid of u (relative to the
    # largest |C|, so a zero crossing of a cross-correlation is no alarm)
    lam_min = min((abs(t[1].real) for t in terms if t[1].real < 0),
                  default=1.0)
    u = np.linspace(0.0, min(5.0 / max(lam_min, 1e-12),
                             getattr(side, 'u_max', math.inf)), 41)
    mag = np.zeros(u.shape)
    val = np.zeros(u.shape, dtype=complex)
    for c, lam, j in terms:
        t = c * u ** j * np.exp(lam * u)
        mag += np.abs(t)
        val += t
    top = float(np.max(np.abs(val)))
    r = float(np.max(mag)) / top if top > 0 else (
        np.inf if np.max(mag) > 0 else 0.0)
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


# ─── the exact delayed tree (api.delay, k = 2) ──────────────────────────────

class ExactDelayTreeSide:
    """One side of the EXACT delayed tree (``api._delay_exact.ExactTree``,
    installed by ``api.delay`` as the k = 2 tree level) with the interface of
    :class:`SideME` where it has one (item 54): values, Laplace transforms
    (so the pairings ``integrate_against``) and the spectrum.  It is not one
    matrix exponential (a delay system has infinitely many characteristic
    roots), so it has no triple and no finite mode list.

    Laplace transform.  For ``τ > 0`` the state covariance obeys ``C' = A C
    + A_d C(τ − d) + 1_{τ<d} B_d N W1(τ)ᵀ`` with ``C(τ − d) = V(τ)`` on
    ``(0, d)`` (the boundary-value segment: ``U, V, W1`` are blocks of its
    stack ``Y(s) = e^{L s} Y0``), so, with ``Δ(z) = zI − A − A_d e^{−zd}``,

        Δ(z) C̃(z) = U(0) + A_d ∫_0^d e^{−zs} V(s) ds
                     + B_d N ∫_0^d e^{−zs} W1(s)ᵀ ds,
        Δ(z) R̃(z) = B + B_d e^{−zd}                      (the response),

    the history segment on ``[0, d]`` plus the exact tail through ``Δ``
    (every delay interval beyond ``d``).  The segment integrals are one
    augmented matrix exponential (``∫ s^k e^{(L − z)s} ds Y0``, a Jordan
    chain of integrators: no cancellation).  ``∫ u^q e^{−zu} C(u) du`` is
    the ``ε^q`` coefficient of the same equations at ``z + ε`` (truncated
    power series: one LU of ``Δ(z)``).  The field covariance is ``C_z C
    C_zᵀ + C_z R N D0ᵀ``; side ``'+'`` takes its ``[b, a]`` entry, side
    ``'-'`` (``u = −τ``) its ``[a, b]`` entry (``ExactTree.value``).
    Valid for ``Re z`` to the right of every characteristic root (always
    for ``Re z >= 0`` on a stable saddle).

    Spectrum (:func:`exact_delay_spectrum`): ``H(iω) N H(iω)ᴴ − D0 N D0ᵀ``
    with the true delayed image ``H(s) = C_z Δ(s)^{-1} (B + B_d e^{−sd}) +
    D0`` (the ``D0 N D0ᵀ`` δ is a δ component, not the smooth part)."""

    blocks = ()
    n_states = 0

    def __init__(self, tree, side, ell=0):
        self.tree = tree
        self.side = side
        self.ell = ell
        self.u_max = math.inf
        self._base = None

    # values
    def __call__(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if np.any(u < 0) or not np.all(np.isfinite(u)):
            raise ValueError('u = |τ| must be finite and >= 0')
        t = self.tree
        a, b = t.ia, t.ib
        out = np.zeros(u.shape, dtype=complex)
        try:
            if u.size:
                t._ensure(float(np.max(u)))
            for i, ui in np.ndenumerate(u):
                F = t.field_covariance(float(ui))
                out[i] = F[b, a] if self.side == '+' else F[a, b]
        except Exception as exc:
            if type(exc).__name__ != 'ExactTreeUnavailable':
                raise
            raise MEUnavailable(f'the exact delayed tree at |tau| = '
                                f'{float(np.max(u)):g}: {exc}') from exc
        return out

    def _segment(self):
        """``(L, Y0, stack)`` of the boundary-value segment on ``[0, d]``."""
        if self._base is None:
            t = self.tree
            st = t._stack
            D0 = st.size_upto(0)
            self._base = (np.asarray(st.L[:D0, :D0]),
                          np.asarray(t._Y0[:D0]), st)
        return self._base

    def _moments(self, z, q):
        """``[∫_0^d s^k e^{−zs} Y(s) ds for k = 0..q]`` (stack vectors)."""
        L, Y0, _st = self._segment()
        D = L.shape[0]
        nb = q + 1
        # states (X_0..X_q, Z_0..Z_q): X_k = s^k/k! e^{(L-z)s} Y0,
        # Z_k' = X_k (Z_k(d) = ∫ s^k/k! e^{(L-z)s} Y0 ds)
        n = 2 * nb * D
        Gm = np.zeros((n, n), dtype=complex)
        Mz = L - z * np.eye(D)
        for k in range(nb):
            Gm[k * D:(k + 1) * D, k * D:(k + 1) * D] = Mz
            if k:
                Gm[k * D:(k + 1) * D, (k - 1) * D:k * D] = np.eye(D)
            o = (nb + k) * D
            Gm[o:o + D, k * D:(k + 1) * D] = np.eye(D)
        x0 = np.zeros(n, dtype=complex)
        x0[:D] = Y0
        d = self.tree.ss.d
        y = sl.expm(Gm * d) @ x0
        return [math.factorial(k) * y[(nb + k) * D:(nb + k + 1) * D]
                for k in range(nb)]

    def _series(self, z, q):
        """``[X_k]``, ``[R_k]``, k = 0..q: the ``ε^k`` coefficients of
        ``C̃(z + ε)`` and ``R̃(z + ε)``."""
        ss = self.tree.ss
        A, Ad, B, Bd, N, d = ss.A, ss.Ad, ss.B, ss.Bd, ss.N, ss.d
        n = A.shape[0]
        _L, _Y0, st = self._segment()
        mom = self._moments(z, q)
        ez = np.exp(-z * d)
        Dl = [z * np.eye(n) - A - Ad * ez]
        for k in range(1, q + 1):
            Dl.append((np.eye(n) if k == 1 else 0.0)
                      - Ad * ez * (-d) ** k / math.factorial(k))
        FC, FR = [], []
        for k in range(q + 1):
            fk = (-1) ** k / math.factorial(k)
            V = st.block(mom[k], 'V')
            W1 = st.block(mom[k], 'W1')
            F = fk * (Ad @ V + Bd @ N @ W1.T)
            if k == 0:
                F = F + st.block(_Y0, 'U')
            FC.append(F)
            FR.append((B if k == 0 else 0.0)
                      + Bd * ez * (-d) ** k / math.factorial(k))
        lu = sl.lu_factor(Dl[0])
        X, R = [], []
        for k in range(q + 1):
            rc, rr = FC[k].astype(complex), np.asarray(FR[k], dtype=complex)
            for j in range(1, k + 1):
                rc = rc - Dl[j] @ X[k - j]
                rr = rr - Dl[j] @ R[k - j]
            X.append(sl.lu_solve(lu, rc))
            R.append(sl.lu_solve(lu, rr))
        return X, R

    def laplace(self, z, power=0):
        """``∫_0^∞ u^power e^{−z u} C(u) du`` on this side."""
        q = int(power)
        z = complex(z)
        ss = self.tree.ss
        X, R = self._series(z, q)
        F = ss.Cz @ X[q] @ ss.Cz.T + ss.Cz @ R[q] @ ss.N @ ss.D0.T
        a, b = self.tree.ia, self.tree.ib
        v = F[b, a] if self.side == '+' else F[a, b]
        return complex((-1) ** q * math.factorial(q) * v)

    def resolvent(self, z, power=1):
        """``laplace(z, power − 1) / (power − 1)!`` (the :class:`SideME`
        resolvent ``c (zI − G)^{−power} x0``)."""
        p = int(power) - 1
        return self.laplace(z, p) / math.factorial(p)

    def pair_rational(self, kernel):
        """``∫_0^∞ L(±u) C(u) du`` for a :class:`RationalKernel` on this
        side: its partial fractions ``r / (p − λ)^{j+1}`` are ``r u^j / j!
        e^{λ u}``, each a Laplace transform (checked against the kernel's
        realization; :class:`MEUnavailable` when they disagree)."""
        from scipy.signal import residue
        r, p, _k = residue(kernel.num, kernel.den)
        terms, prev, j = [], None, 0
        for ri, pi in zip(r, p):
            j = j + 1 if (prev is not None and pi == prev) else 0
            prev = pi
            terms.append((complex(ri), complex(pi), j))
        us = np.linspace(0.0, 5.0 / max(min(-x.real for _, x, _ in terms),
                                        1e-12), 9)
        ref = np.array([kernel.c @ sl.expm(kernel.A * u) @ kernel.b
                        for u in us])
        got = np.array([sum(c * u ** j / math.factorial(j) * np.exp(lam * u)
                            for c, lam, j in terms) for u in us])
        if np.max(np.abs(got - ref)) > 1e-10 * max(np.max(np.abs(ref)),
                                                   1e-300):
            raise MEUnavailable('the partial fractions of the rational '
                                'kernel are ill-conditioned (repeated or '
                                'nearly repeated poles): no exact pairing '
                                'with the exact delayed tree')
        return sum((c / math.factorial(j) * self.laplace(-lam, j)
                    for c, lam, j in terms), 0j)

    def triple(self):
        raise MEUnavailable(_NO_TRIPLE_DELAY)

    def modes(self, *a, **kw):
        raise MEUnavailable(_NO_TRIPLE_DELAY)

    def sylvester(self, A_L, b_L, c_L):
        raise MEUnavailable('the exact delayed tree pairs a rational kernel '
                            'through its partial fractions (pair_rational)')

    def __repr__(self):
        return (f'ExactDelayTreeSide(side={self.side!r}, d={self.tree.ss.d:g}, '
                f'states={self.tree.ss.A.shape[0]})')


_NO_TRIPLE_DELAY = ('the k = 2 tree is the exact delayed covariance (a delay '
                    'system: infinitely many characteristic roots, no finite '
                    'triple or mode list; values, pairings and the spectrum '
                    "are exact; delay={'tree': 'pade'} gives the Pade "
                    "tree's triples)")


def exact_delay_spectrum(tree, omega):
    """``∫ e^{−iωτ} C_smooth(τ) dτ`` of the exact delayed tree: ``[H(iω) N
    H(iω)ᴴ − D0 N D0ᵀ][b, a]`` with the true delayed image (no Pade, no
    quadrature)."""
    ss = tree.ss
    A, Ad, B, Bd, N, d = ss.A, ss.Ad, ss.B, ss.Bd, ss.N, ss.d
    n = A.shape[0]
    a, b = tree.ia, tree.ib
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.zeros(omega.shape, dtype=complex)
    floor = ss.D0 @ N @ ss.D0.T
    for i, w in enumerate(omega):
        s = 1j * w
        e = np.exp(-s * d)
        H = ss.Cz @ np.linalg.solve(s * np.eye(n) - A - Ad * e, B + Bd * e) \
            + ss.D0
        S = H @ N @ H.conj().T - floor
        out[i] = S[b, a]
    return out


def _exact_tree_object(result):
    """The ``ExactTree`` behind an exact delayed tree (``api.delay``
    attaches it to the tree's callable), or ``None``."""
    f0 = (result.get('total_C_by_ell') or {}).get(0)
    return getattr(f0, 'exact_tree', None)


# ─── the whole result ───────────────────────────────────────────────────────

class CumulantME:
    """The matrix-exponential form of a k = 2 result's smooth part:
    ``sides[ell]['+' | '-']`` = :class:`SideME`.  ``complete`` is False when
    some smooth term has no triple (``reasons`` says which; the methods then
    raise :class:`MEUnavailable` for that loop order)."""

    def __init__(self, k, sides, reasons, counts, mode_reasons=None):
        self.k = k
        self.sides = sides
        self.reasons = reasons            # {ell: [str]}
        self.counts = counts
        # orders with exact values, pairings and spectra but no finite mode
        # list (the exact delayed tree): {ell: [str]}
        self.mode_reasons = dict(mode_reasons or {})

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
            if self.mode_reasons.get(e):
                raise MEUnavailable(f'ell={e}: ' + '; '.join(
                    self.mode_reasons[e]))
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
                sd = self.sides[e][kernel.side]
                if isinstance(sd, ExactDelayTreeSide):
                    tot += sd.pair_rational(kernel)
                else:
                    tot += sd.sylvester(kernel.A, kernel.b, kernel.c)
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
            sp = self.sides[e]['+']
            if isinstance(sp, ExactDelayTreeSide):
                out += exact_delay_spectrum(sp.tree, omega)
                continue
            for i, w in enumerate(omega):
                out[i] += (self.sides[e]['+'].resolvent(1j * w)
                           + self.sides[e]['-'].resolvent(-1j * w))
        return out

    def __repr__(self):
        st = ', '.join(
            f'{e}: exact delayed tree' if isinstance(s['+'], ExactDelayTreeSide)
            else f'{e}: {len(s["+"].blocks)}+{len(s["-"].blocks)} blocks'
            for e, s in sorted(self.sides.items()))
        return (f'CumulantME(k={self.k}, {st}, complete={self.complete})')


def _groups_of(pj):
    return [g for g in (pj.get('groups') or []) if isinstance(g, dict)]


def _stamped_flags(cfg):
    """``{name: value}`` of the run's ``config['phase_j_flags']`` stamp."""
    out = {}
    for part in str(cfg.get('phase_j_flags') or '').split(';'):
        if '=' in part:
            k_, v_ = part.split('=', 1)
            out[k_.strip()] = v_.strip()
    return out


def _global_reasons(result, FI):
    """Reasons that hold for every loop order of a result."""
    cfg = result.get('config') or {}
    out = []
    # the triples are built now, with the live Phase J flags: they must be
    # the run's (``config['phase_j_flags']``)
    st = _stamped_flags(cfg)
    for name in ('PHASE_J_T_MIN', 'THETA0_CONST_ROW_MODE'):
        if name in st and str(getattr(FI, name, '<absent>')) != st[name]:
            out.append(f'{name} changed since the run ({st[name]} -> '
                       f'{getattr(FI, name, "<absent>")})')
    if st.get('THETA0_CONST_ROW_MODE') == 'legacy_clip':
        out.append("THETA0_CONST_ROW_MODE='legacy_clip'")
    if int((result.get('phase_j_engine') or {}).get('fallback_evals') or 0):
        out.append('the state-space route fell back on the pole route for '
                   'some evaluations of the run (C_tau there is the pole '
                   "route's; result['phase_j_engine'])")
    eng = result.get('phase_j_engine') or {}
    if eng.get('pole_deformation') == 'used':
        out.append('the pole-cluster deformation ran (the value is an average '
                   'over deformed propagators)')
    if eng.get('engine') == 'poles':
        out.append('DAEDALUS_PHASE_J_ENGINE=poles')
    if cfg.get('use_grouped_phase_j'):
        out.append('grouped Phase J')
    try:
        if float(FI.PHASE_J_T_MIN) != float('-inf'):
            out.append('finite PHASE_J_T_MIN')
    except (TypeError, ValueError):
        out.append('PHASE_J_T_MIN not a float')
    return out


def _exact_delay_tree(result):
    """True when ``api.delay`` installed the exact delayed tree in place of
    the Pade tree (k = 2): the tree's closures are then not ``C_tau``'s."""
    dl = result.get('delay') if isinstance(result, dict) else None
    return isinstance(dl, dict) and dl.get('tree') == 'exact'


def _refuse_switch_on(result):
    """A switch-on result (``t_min``) is a function of the absolute times,
    not of a lag: no triples (its closures would give stationary ones)."""
    if isinstance(result, dict) and result.get('t_min') is not None:
        raise MEUnavailable(
            "a switch-on result (t_min) has no lag-axis triples: use "
            "result['C_times'] / result['total_C'] (absolute times)")


def from_result(result) -> CumulantME:
    """Build the triples of a k = 2 ``compute_cumulants`` / ``dd.run``
    result from its live Phase J closures (nothing is integrated again: the
    stationary states and segments are the route's cached ones).  Raises
    :class:`MEUnavailable` for results that cannot have one at all (k != 2,
    spatial, a loaded file, moment output); per-loop-order gaps are in
    ``reasons``."""
    from engine.integration.time_domain import final_integral as FI
    cfg = result.get('config') or {}
    _refuse_switch_on(result)
    k = int(cfg.get('k', 0))
    if k != 2:
        raise MEUnavailable(f'triples on the lag axis are defined for k = 2 '
                            f'(got k = {k}); for k >= 3 use slice_me')
    if 'C_tau_x' in result or cfg.get('spatial'):
        raise MEUnavailable('spatial results have no state-space triples')
    pjs = result.get('phase_j_by_ell')
    if not pjs:
        raise MEUnavailable('the result has no live Phase J closures (a '
                            'loaded file or a non-Phase-J result)')
    global_reasons = _global_reasons(result, FI)
    sides, reasons, mode_reasons = {}, {}, {}
    counts = {'blocks': 0, 'zero': 0, 'pole_route': 0, 'pruned': 0}
    for ell, pj in sorted(pjs.items()):
        ell = int(ell)
        rs = list(global_reasons)
        blocks = {'+': [], '-': []}
        exact_tree = ell == 0 and _exact_delay_tree(result)
        tree_obj = _exact_tree_object(result) if exact_tree else None
        if tree_obj is not None:
            # item 54: the exact delayed tree's own values, Laplace
            # transforms and spectrum (not from the Pade copy's closures)
            sides[ell] = {s: ExactDelayTreeSide(tree_obj, s, ell)
                          for s in ('+', '-')}
            reasons[ell] = []
            mode_reasons[ell] = [_NO_TRIPLE_DELAY]
            continue
        if pj and not exact_tree:
            sk = list(pj.get('skipped_kernel_ids') or [])
            if sk:
                rs.append(f'{len(sk)} diagram(s) failed in Phase J')
            for gi, g in enumerate(_groups_of(pj)):
                fn = g.get('contribution')
                if fn is None:
                    continue
                parts = _parts_list(fn)
                if parts is None:
                    rs.append(f'diagram {g.get("kernel_id", gi)}: its '
                              f'contribution exposes no subset structure')
                    continue
                _blocks_of(parts, g.get('kernel_id', gi), blocks, rs,
                           counts, FI)
        if exact_tree:
            # the delay driver replaced the tree by the exact delayed
            # covariance (api.delay); the closures are the Pade copy's
            rs.append('the k = 2 tree is the exact delayed covariance (method '
                      'of steps over the delay: not one matrix exponential; '
                      "delay={'tree': 'pade'} gives the Pade tree's triples)")
        if rs:
            # collapse repeated reasons
            seen = {}
            for r in rs:
                seen[r] = seen.get(r, 0) + 1
            rs = [r if n == 1 else f'{r} (x{n})' for r, n in seen.items()]
        sides[ell] = {s: SideME(s, blocks[s], ell) for s in ('+', '-')}
        reasons[ell] = rs
    return CumulantME(2, sides, reasons, counts, mode_reasons)


def _parts_list(fn):
    """``[(weight, phase_j_parts)]`` of a diagram's contribution, or
    ``None``: one entry, or one per kernel-slot placement of a diagram with
    conductance vertices (``_average_conv_placements``: the contribution is
    the weighted sum of the placements' own contributions)."""
    parts = getattr(fn, 'phase_j_parts', None)
    if parts is not None:
        return [(1.0, parts)]
    places = getattr(fn, 'phase_j_placements', None)
    if places is None:
        return None
    out = []
    for w, f in places:
        sub = _parts_list(f)
        if sub is None:
            return None
        out.extend((w * w2, p) for w2, p in sub)
    return out


def _blocks_of(parts_list, diagram_index, blocks, rs, counts, FI):
    """The k = 2 blocks of one diagram's contribution, both sides."""
    for scale, parts in parts_list:
        for side, sg in (('+', 1.0), ('-', -1.0)):
            line = ((0.0, 0.0), (0.0, sg), 0.0, 1.0)   # T(u) = T0 + u e
            _line_blocks(parts, diagram_index, line, blocks[side], rs, counts,
                         FI, scale)


def _line_blocks(parts, diagram_index, line, out, rs, counts, FI, scale=1.0):
    """Append the :class:`MEBlock` of every subset and Wick permutation of
    one contribution on the line of caller times ``T(t) = T0 + t * e``,
    chamber containing ``t_ref``, anchored at ``t_anchor`` (``line = (T0, e,
    t_anchor, t_ref)``, or ``(T0, e, t_anchor, t_ref, t_end)`` for a finite
    chamber: a subset where several segments move, or a constant row
    shrinks, is then one :class:`MEChainBlock`)."""
    from engine.integration.time_domain import statespace as SS
    T0, e, t_anchor, t_ref = line[:4]
    t_end = line[4] if len(line) > 4 else math.inf
    perms = parts['perms']
    comp = parts['compensation']
    org = parts['origin_leaf_idx']
    fidx = parts['free_ext_idx']
    kk = parts['k']
    T_ref = tuple(a + t_ref * b for a, b in zip(T0, e))

    def free(T):
        def f(perm):
            p_ = [T[perm[j]] for j in range(kk)]
            if org is not None:
                o = p_[org]
                p_ = [t - o for t in p_]
            return [float(p_[j]) for j in fidx]
        return f
    for pidx, perm in enumerate(perms):
        fb = free(T0)(perm)
        # the direction exactly (integers): a difference of two times would
        # carry rounding into the slopes (spurious moving segments)
        eo = 0.0 if org is None else float(e[perm[org]])
        fd = [float(e[perm[j]]) - eo for j in fidx]
        tie = FI._TieContext(T_ref, tuple(perm[j] for j in fidx),
                             None if org is None else perm[org])
        for si, cfn in enumerate(parts['subsets']):
            sub = getattr(cfn, 'ss_subset', None)
            if sub is None:
                counts['pole_route'] += 1
                rs.append('a subset is on the pole route (the state-space '
                          'route declined it)')
                continue
            if getattr(cfn, 'ss_theta0_empty', False):
                counts['pruned'] += 1
                continue
            chain = False
            try:
                try:
                    ray = sub.me_line(fb, fd, t_anchor, t_ref, tie)
                except SS.SSIneligible as exc:
                    if exc.reason not in ('several_lags', 'row_crosses_zero') \
                            or not math.isfinite(t_end):
                        raise
                    # a finite chamber (item 54): several moving segments
                    ray = sub.me_chain(fb, fd, t_anchor, t_ref, t_end, tie)
                    chain = True
            except SS.SSIneligible as exc:
                rs.append(f'a subset is not one matrix exponential on this '
                          f'line ({exc.reason})')
                continue
            except SS._late_errors() as exc:
                rs.append(f'the state-space route refused a subset '
                          f'({SS._late_reason(exc)})')
                continue
            if ray is None:
                counts['zero'] += 1
                continue
            w = scale * complex(cfn.ss_prefactor) / comp
            if chain:
                out.append(MEChainBlock(ray, w, sub.model.dtype,
                                        diagram_index, si, pidx,
                                        abs(t_end - t_anchor)))
                counts['chain_blocks'] = counts.get('chain_blocks', 0) + 1
            else:
                out.append(MEBlock(ray, w, sub.model.dtype, diagram_index,
                                   si, pidx))
            counts['blocks'] += 1


# ─── k >= 3: the slices of dd.run ───────────────────────────────────────────

class SliceME:
    """One k >= 3 slice of ``dd.run`` (leg ``j`` swept over τ, the other
    non-anchor legs at their pinned times) as matrix-exponential triples,
    one per CHAMBER of τ (an interval between the other legs' times, where
    the order of the external times is fixed):
    ``chambers = [(lo, hi, anchor, sigma, {ell: SideME})]``, ``C(τ) =
    Σ_blocks c e^{G s} x0`` with ``s = sigma (τ − anchor) >= 0``.  The two
    outer chambers (the swept leg first or last) have one moving segment and
    are always built; a chamber where the swept leg lies between two legs
    moves two segments and has no triple (``reasons``).  At a chamber
    boundary the value is the left limit (``dd.run``'s convention)."""

    def __init__(self, j, pinned, chambers, reasons, counts):
        self.j = j
        self.pinned = pinned
        self.chambers = chambers
        self.reasons = reasons           # {(lo, hi): {ell: [str]}}
        self.counts = counts

    def chamber_of(self, tau):
        for ch in self.chambers:
            lo, hi = ch[0], ch[1]
            if lo < tau <= hi:
                return ch
        raise ValueError(f'tau={tau} is in no chamber')

    def __call__(self, tau, ell=None):
        """``C`` along the slice at the swept leg's times ``tau`` (array),
        summed over ``ell``."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.zeros(tau.shape, dtype=complex)
        for i, t in enumerate(tau):
            lo, hi, a, sg, sides = self.chamber_of(float(t))
            bad = {e: r for e, r in self.reasons[(lo, hi)].items() if r
                   and (ell is None or e == ell)}
            if bad:
                raise MEUnavailable(f'chamber ({lo}, {hi}): ' + '; '.join(
                    f'ell={e}: ' + '; '.join(r) for e, r in bad.items()))
            ells = sorted(sides) if ell is None else [ell]
            for e in ells:
                out[i] += sides[e](np.array([sg * (t - a)]))[0]
        return out

    def modes(self, ell=None, **kw):
        """``{(lo, hi): {ell: [ModeTerm]}}`` in the swept leg's time τ (the
        side label is the chamber ``'(lo, hi)'``), for the chambers that
        have triples."""
        from api.cumulant import ModeTerm
        out = {}
        for lo, hi, a, sg, sides in self.chambers:
            if any(self.reasons[(lo, hi)].values()):
                continue
            lab = f'({lo:g}, {hi:g})'
            per = {}
            for e in (sorted(sides) if ell is None else [ell]):
                terms = []
                for c, lam, n in sides[e].modes(**kw):
                    # s = sg (τ − a): c s^n e^{λ s} expanded in τ
                    for m in range(n + 1):
                        cm = (c * sg ** n * math.comb(n, m) * (-a) ** (n - m)
                              * np.exp(-lam * sg * a))
                        if not np.isfinite(cm):
                            warnings.warn(
                                f'slice {self.j}, chamber {lab}: a mode '
                                f'coefficient overflows when re-expanded '
                                f'about tau = 0 (anchor {a:g}, rate {lam:.3g});'
                                f' use the triple or the chamber-local '
                                f'modes (sides[ell].modes())',
                                MEConditioningWarning, stacklevel=2)
                        if cm != 0:
                            terms.append(ModeTerm(cm, lam * sg, m, lab))
                per[e] = terms
            out[(lo, hi)] = per
        return out

    def __repr__(self):
        return (f'SliceME(j={self.j}, chambers='
                f'{[(c[0], c[1]) for c in self.chambers]})')


def slice_me(result, j=1, base=None, eps=None) -> SliceME:
    """Triples of the k >= 3 slice ``j`` of a ``compute_cumulants`` /
    ``dd.run`` result: leg ``j`` swept, the other non-anchor legs at
    ``base`` (default: ``result['_kpoint_base']`` or 0; a base lag of 0 is
    pinned at ``-eps``, the Itô left limit, as ``dd.run`` does)."""
    from engine.integration.time_domain import final_integral as FI
    cfg = result.get('config') or {}
    _refuse_switch_on(result)
    k = int(cfg.get('k', 0))
    if k < 3:
        raise MEUnavailable('slice_me is for k >= 3 (k = 2: from_result)')
    if not 1 <= j < k:
        raise ValueError(f'slice index j={j} must be in 1..{k - 1}')
    if eps is None:
        from api.compute import _ITO_EPS as eps
    if base is None:
        base = result.get('_kpoint_base') or [0.0] * (k - 1)
    base = [float(b) for b in base]
    if len(base) != k - 1:
        raise ValueError(f'base must have k-1 = {k - 1} entries')
    pinned = {leg: (b if abs(b) > 1e-12 else -eps)
              for leg, b in zip(range(1, k), base) if leg != j}
    T0 = [0.0] * k
    for leg, t in pinned.items():
        T0[leg] = t
    e = [0.0] * k
    e[j] = 1.0
    bounds = sorted(set([0.0] + list(pinned.values())))
    cham = [(-math.inf, bounds[0], bounds[0], -1.0, bounds[0] - 1.0,
             -math.inf)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        cham.append((lo, hi, lo, 1.0, 0.5 * (lo + hi), hi))
    cham.append((bounds[-1], math.inf, bounds[-1], 1.0, bounds[-1] + 1.0,
                 math.inf))
    pjs = result.get('phase_j_by_ell') or {}
    if not pjs:
        raise MEUnavailable('the result has no live Phase J closures')
    g_rs = _global_reasons(result, FI)
    counts = {'blocks': 0, 'zero': 0, 'pole_route': 0, 'pruned': 0}
    chambers, reasons = [], {}
    for lo, hi, a, sg, t_ref, t_end in cham:
        sides, rs_ch = {}, {}
        for ell, pj in sorted(pjs.items()):
            ell = int(ell)
            rs, blocks = list(g_rs), []
            for gi, g in enumerate(_groups_of(pj or {})):
                fn = g.get('contribution')
                if fn is None:
                    continue
                plist = _parts_list(fn)
                if plist is None:
                    rs.append('a contribution exposes no subset structure')
                    continue
                for scale, parts in plist:
                    _line_blocks(parts, g.get('kernel_id', gi),
                                 (tuple(T0), tuple(e), a, t_ref, t_end),
                                 blocks, rs, counts, FI, scale)
            if pj and pj.get('skipped_kernel_ids'):
                rs.append('diagrams failed in Phase J')
            sides[ell] = SideME(f'({lo:g}, {hi:g})', blocks, ell,
                                u_max=abs(t_end - a))
            rs_ch[ell] = sorted(set(rs))
        chambers.append((lo, hi, a, sg, sides))
        reasons[(lo, hi)] = rs_ch
    return SliceME(j, pinned, chambers, reasons, counts)


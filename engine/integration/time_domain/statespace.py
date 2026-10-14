r"""
State-space subset integrator for Phase J (stage S2, opt-in)
============================================================

Evaluates one Phase J subset integral

    I(f) = P · ∫_R  Π_{e ∈ E} g_e(Δt_e)  ds,      g_e(t) = c_e e^{a_e t} b_e  (t > 0),

without poles, residues or pole tuples.  Each smooth edge ``e`` carries the
minimal SISO realization ``(a_e, b_e, c_e)`` of ITS propagator entry
``G[pi_e, ri_e]`` (``api._realization``, S1: ``Realization.entries``), so its
state axis has ``n_e = entries[(pi, ri)].n_states`` components, never the
global state dimension.  ``R`` is the subset's region: every row is
``Δt_e = a_int·s + a_ext·f + c0 > 0`` with ``a_int`` one head (+1) and/or one
tail (-1) internal vertex; heads/tails without a vertex are the external
times ("constants", affine in the free external times ``f``).  Rows without
an internal vertex are scalar factors ``g_e(c_eff)`` decided by Phase J's own
Θ(0) / tie rule (``final_integral._const_row_verdict``).  ``P`` is the
subset prefactor.

Opt in with ``DAEDALUS_PHASE_J_ENGINE=statespace`` (read when a diagram is
integrated).  Unset (or ``poles`` / ``default``) nothing here runs and every
number is the pole engine's.  A subset is taken here only when every row is a
propagator edge of the model's realization (``subset_maker``: no noise-kernel
boxes, no conductance pseudo-edges, no dressed lines, a numeric prefactor, a
Hurwitz realization, rows of the shape above, every vertex followed by an
external time); everything else stays on the pole routes.

The downset sweep
-----------------
Time runs upward.  The state at time ``t`` is indexed by the set ``I`` of
internal vertices already placed ("fired") -- a downset of the subset's
causal poset -- and is a tensor with one axis per ACTIVE edge (tail passed,
head not):

    X_I(t) = ∫_{R_I(t)} Π_{e done} g_e(Δt_e) ⊗_{e active} e^{a_e (t - t_e^tail)} b_e  ds_I,

``R_I(t)``: the times of the vertices of ``I``, all ``< t``, consistent with
the rows among ``I`` and the external times already passed.

1. **Generator.**  Between external times ``X`` solves the linear system
   ``X_I' = M_I X_I + Σ_{v ∈ max I, v placeable} F_v X_{I \ v}`` with
   ``M_I = ⊕_{e active} a_e`` (a Kronecker sum, applied axis by axis, never
   formed) and the firing map ``F_v``: contract the axes of the edges with
   head ``v`` by ``c_e``, open axes for the edges with tail ``v`` by ``b_e``.
2. **Stationary start** (``t_min = -inf``): before the first external time
   the problem is translation invariant, so ``X_I`` is constant there and
   ``X_∅ = 1``, ``X_J = (-M_J)^{-1} Σ_v F_v X_{J \ v}`` (resolvents of
   Kronecker sums; with two active edges this is the Lyapunov equation).
   These states do not depend on the external times: computed once per
   subset and cached.  A finite ``final_integral.PHASE_J_T_MIN`` is a finite
   first segment from ``X = {∅: 1}`` instead.
3. **External time** ``c``: contract the edges whose head is ``c``, open the
   edges whose tail is ``c``, drop the downsets that miss a vertex required
   before ``c``.  Equal external times form one event.
4. **Segment** ``(c_j, c_{j+1})``: ``X ← exp(G L) X``, ``L = c_{j+1} - c_j``,
   ``G`` block lower triangular over the downsets reachable in the segment
   (diagonal ``M_I``, sub-diagonal ``F_v``).
5. The answer is ``P · Π(scalar factors) · X_V`` (all vertices placed, no
   active edge) after the last external time.

**Proof sketch (the poset generalisation of Van Loan / Carbonell-Jiménez-
Pedroso).**  (a) Differentiating ``X_I(t)`` in its upper limit ``t`` gives
two terms: the open factors ``e^{a_e (t - t_tail)}`` give ``M_I X_I``; the
moving face ``max_{v∈I} s_v = t`` of ``R_I(t)`` splits (up to a null set)
by WHICH vertex is the latest, necessarily a maximal element ``v`` of ``I``;
on that face the edges into ``v`` complete with ``Δt = t - t_tail``, i.e.
``c_e^T`` applied to the open axis of ``X_{I∖v}``, and the edges out of ``v``
open with ``e^{a_e·0} b_e = b_e``: that is ``F_v X_{I∖v}(t)``, present only
while ``v``'s window (its external-time rows) admits ``t``.  So the stacked
state solves the linear ODE ``X' = G X`` on each segment, and the solution
is ``exp(G L) X``.  (b) ``G`` is block triangular with nilpotent
off-diagonal part (each firing adds a vertex), so the Dyson series of
``exp(G L)`` terminates; its block ``(J, I)`` is the sum over all saturated
chains ``I = I_0 ⊂ I_1 ⊂ ... ⊂ I_k = J`` of the nested integrals
``∫_{0<u_1<...<u_k<L} e^{M_J (L-u_k)} F ... F e^{M_I u_1}``.  A chain of the
downset lattice from ``∅`` to ``V`` is exactly a linear extension of the
poset, so every vertex order compatible with the rows is integrated once and
none is enumerated.  For a chain poset the lattice is a chain and this is
Van Loan's (1978) block-triangular exponential / Carbonell et al.'s (2008)
multiple integrals; with scalar edges it is Opitz's divided-difference
formula that Phase J's chain kernels evaluate per pole tuple (Phase J =
diagonalise every edge, then run this sweep once per pole tuple).  (c)
Before the first external time the region and integrand are invariant under
a common shift of all times, so ``X_I(t)`` is constant; the ODE then gives
``0 = M_I X_I + Σ F_v X_{I∖v}``; ``M_I`` is invertible because every
active edge is Hurwitz and a nonempty downset has at least one active edge
(its maximal vertex has an out-edge: ``subset_maker`` requires every vertex
to be followed by an external time).  Absolute convergence: every open
direction is damped by an active Hurwitz edge.  (d) At an external time the
rows with that head/tail change the integrand only by completing/opening
factors, and a vertex whose window ends there must already be placed.  ∎

Numerics
--------
* **Resolvents** ``(-M)^{-1} r`` in the per-edge Schur basis (exact
  identity / reversal for triangular ``a_e``, e.g. Erlang chains; complex
  Schur otherwise): the Kronecker sum of upper-triangular factors is upper
  triangular, solved by a vectorised back substitution (Jacobi sweeps on the
  nilpotent strictly-upper part: exact after ``1 + Σ(n_e - 1)`` sweeps and
  operation-for-operation the back substitution), then mapped back and
  refined once on the residual in the original basis (S1: a plain Schur-basis
  solve loses up to ~3e-13 on graded realizations).  Divisions are by sums
  of eigenvalues of Hurwitz factors only: no eigenvalue gap enters.
* **Segments and impulses** ``exp(G L) x`` / ``c e^{a t} b`` by the
  Al-Mohy-Higham (2011) truncated-Taylor action with the trace shift: no
  squaring, so neither the ``eps/gap`` loss on near-repeated eigenvalues nor
  the loss of scaling-and-squaring ``expm`` on decaying matrices (measured
  here: scipy's ``expm`` of a 2 x 2 propagator block is 2e-14 / 9e-14 off
  relative to its norm at ``t = 3 / 30``; the action 4e-16 / 1.5e-15).  The
  generator is dense up to ``OP_DENSE_MAX`` and matrix-free (per-block axis
  products and einsum firings) beyond.
* **τ batches** (``SubsetSS.evaluate_many``, ``tau_grid_prefetch``): points
  that share the order of the external times reuse the stationary state and
  the first event; the last segment is stepped from one length to the next.

Memory: before a subset's stationary states are allocated their size (the
product of the per-edge dimensions over each downset's active edges, summed)
is charged against ``PHASE_J_MEM_BUDGET_GB`` through item 35's running total
(``final_integral.phase_j_memory_scope``); over budget raises
``PhaseJResourceError`` before anything is allocated.
"""
from __future__ import annotations

import itertools
import math
import os
import threading
from collections import OrderedDict

import numpy as np
import scipy.linalg as sl

__all__ = [
    'ENGINE_ENV', 'engine_requested', 'subset_maker', 'SubsetSS',
    'realization_for', 'tau_grid_prefetch', 'SSIneligible',
]

ENGINE_ENV = 'DAEDALUS_PHASE_J_ENGINE'
_ENGINE_ON = ('statespace',)
_ENGINE_OFF = ('', 'poles', 'default')

#: dense generator (Taylor action with dense mat-vecs) up to this dimension;
#: matrix-free beyond
OP_DENSE_MAX = 2000
#: refinement steps of a resolvent solve done in a unitary Schur basis
REFINE_STEPS = 2
#: a subset with more stationary downsets than this is declined
MAX_DOWNSETS = 50000
#: bytes of dense segment generators kept across calls (LRU)
DENSE_CACHE_BYTES = 256 * 2 ** 20
#: at most this many smooth edges (einsum labels)
MAX_EDGES = 50


class SSIneligible(Exception):
    """A subset (or model) the state-space route does not take; ``reason``
    is the counter key."""

    def __init__(self, reason, detail=''):
        self.reason = reason
        super().__init__(f'{reason}: {detail}' if detail else reason)


def engine_requested(environ=None):
    """True when ``DAEDALUS_PHASE_J_ENGINE`` selects the state-space route.
    Unset / ``poles`` / ``default``: False.  Any other value raises."""
    env = os.environ if environ is None else environ
    raw = str(env.get(ENGINE_ENV, '')).strip().lower()
    if raw in _ENGINE_ON:
        return True
    if raw in _ENGINE_OFF:
        return False
    raise ValueError(f'{ENGINE_ENV}={raw!r}: expected one of '
                     f'{_ENGINE_ON + _ENGINE_OFF[1:]} (or unset)')


# ─── counters (in final_integral._RUNTIME_COUNTERS, created on import) ────

def _FI():
    from engine.integration.time_domain import final_integral as FI
    return FI


_COUNTER_KEYS = (
    'ss_subsets_built', 'ss_subsets_declined', 'ss_evals',
    'ss_stationary_builds', 'ss_resolvent_solves', 'ss_refinements',
    'ss_segments_taylor', 'ss_taylor_matvecs',
    'ss_batch_points', 'ss_prefetch_hits',
    'ss_realizations_built', 'ss_realization_refused',
)


def _counters():
    C = _FI()._RUNTIME_COUNTERS
    if 'ss_evals' not in C:
        for k_ in _COUNTER_KEYS:
            C.setdefault(k_, 0)
        C.setdefault('ss_declined_by_reason', {})
    return C


def _count(key, n=1):
    C = _counters()
    C[key] = C.get(key, 0) + n


def _decline(reason):
    C = _counters()
    C['ss_subsets_declined'] = C.get('ss_subsets_declined', 0) + 1
    d = C.setdefault('ss_declined_by_reason', {})
    d[reason] = d.get(reason, 0) + 1
    return None


# ─── realization cache ────────────────────────────────────────────────────

_REAL_LOCK = threading.RLock()
_REAL_CACHE = OrderedDict()
_REAL_CACHE_MAX = 8


def _np_key(num_params):
    if not num_params:
        return ()
    try:
        return tuple(sorted((str(k), repr(v)) for k, v in num_params.items()))
    except Exception:
        return ('id', id(num_params))


def realization_for(propagator_data, num_params):
    """``(_ModelOps, None)`` or ``(None, reason)`` for this propagator at
    ``num_params``.  Cached by the identity of ``K_ft`` (shared by the
    pole-cluster deformation copies) and the parameter values; the cache
    holds strong references, so an id cannot be recycled while cached."""
    K = propagator_data.get('K_ft') if hasattr(propagator_data, 'get') else None
    key = (id(K), id(propagator_data.get('omega')) if K is not None else None,
           _np_key(num_params))
    with _REAL_LOCK:
        hit = _REAL_CACHE.get(key)
        if hit is not None and hit[0] is K:
            _REAL_CACHE.move_to_end(key)
            return hit[1], hit[2]
    from api._realization import try_realize
    try:
        real, exc = try_realize(propagator_data, num_params or {})
    except Exception as e:                      # never let S1 break a run
        real, exc = None, e
    if real is None:
        reason = getattr(exc, 'reason', None) or 'realization_error'
        ops, why = None, f'realization_{reason}'
        _count('ss_realization_refused')
    elif not (real.hurwitz is True or (real.hurwitz is None and
                                       (real.lyapunov_margin or 0) > 0)):
        ops, why = None, 'not_hurwitz'
        _count('ss_realization_refused')
    else:
        ops, why = _ModelOps(real), None
        _count('ss_realizations_built')
    with _REAL_LOCK:
        _REAL_CACHE[key] = (K, ops, why)
        _REAL_CACHE.move_to_end(key)
        while len(_REAL_CACHE) > _REAL_CACHE_MAX:
            _REAL_CACHE.popitem(last=False)
    return ops, why


def _clear_caches():
    """Drop the realization and dense-generator caches (tests)."""
    with _REAL_LOCK:
        _REAL_CACHE.clear()
    with _DENSE_LOCK:
        _DENSE_CACHE.clear()
        _DENSE_BYTES[0] = 0


# ─── per-entry operators ──────────────────────────────────────────────────

def _mode(x, M, ax):
    """``M`` applied along axis ``ax`` of tensor ``x``."""
    return np.moveaxis(np.tensordot(M, x, axes=(1, ax)), 0, ax)


class _EntryOps:
    """One propagator entry ``g(t) = c e^{a t} b``: the realization and its
    triangular (Schur) basis.  ``kind``: 'zero' (no smooth part), 'scalar'
    (n = 1), 'upper' / 'lower' (triangular ``a``: the basis change is the
    identity / the index reversal, exact), 'schur' (complex Schur, unitary).
    """
    __slots__ = ('n', 'a', 'b', 'c', 'kind', 'T', 'U', 'Uh', 'dT', 'Nup',
                 'trace', 'norm1', 'cnorm', 'bnorm', 'lam', '_tay')

    def __init__(self, ent, dtype):
        self.n = n = int(ent.n_states)
        self.a = np.asarray(ent.a, dtype=dtype)
        self.b = np.asarray(ent.b, dtype=dtype).reshape(n)
        self.c = np.asarray(ent.c, dtype=dtype).reshape(n)
        self.trace = complex(np.trace(self.a)) if n else 0j
        self.norm1 = float(np.abs(self.a).sum(axis=0).max()) if n else 0.0
        self.cnorm = float(np.abs(self.c).max()) if n else 0.0
        self.bnorm = float(np.abs(self.b).sum()) if n else 0.0
        self.U = self.Uh = None
        self.lam = None
        self._tay = None
        if n == 0:
            self.kind = 'zero'
            self.T = self.dT = self.Nup = None
            return
        if n == 1:
            self.kind = 'scalar'
            self.lam = self.a[0, 0]
            self.T = self.a
            self.dT = self.a.diagonal().copy()
            self.Nup = None
            return
        a = self.a
        if not np.any(np.tril(a, -1)):
            self.kind, T = 'upper', a
        elif not np.any(np.triu(a, 1)):
            self.kind, T = 'lower', a[::-1, ::-1]
        else:
            self.kind = 'schur'
            T, U = sl.schur(np.asarray(a, dtype=complex), output='complex')
            self.U, self.Uh = U, U.conj().T
        self.T = T
        self.dT = T.diagonal().copy()
        self.Nup = np.triu(T, 1)

    def to_basis(self, x, ax):
        if self.kind == 'lower':
            return np.flip(x, axis=ax)
        if self.kind == 'schur':
            return _mode(x, self.Uh, ax)
        return x

    def from_basis(self, x, ax):
        if self.kind == 'lower':
            return np.flip(x, axis=ax)
        if self.kind == 'schur':
            return _mode(x, self.U, ax)
        return x

    def impulse(self, t):
        """``c e^{a t} b`` (``t >= 0``) by the trace-shifted Taylor action
        (scipy's scaling-and-squaring ``expm`` loses up to ~1e-13 relative
        on decaying exponentials; the action does not square)."""
        if self.n == 0 or t < 0:
            return 0.0
        if self._tay is None:
            mu = self.trace / self.n
            if not np.iscomplexobj(self.a):
                mu = float(np.real(mu))
            A = self.a - mu * np.eye(self.n, dtype=self.a.dtype)
            self._tay = (A, mu, _onenorm_powers(A))
        A, mu, dn = self._tay
        return (self.c @ _taylor_action(lambda y: A @ y, mu, dn, self.b,
                                        float(t))).item()


class _ModelOps:
    """The realization of one model at one parameter point, as per-entry
    operators, plus the resolvent data per active-edge pattern (shared by
    every subset of the model)."""

    def __init__(self, real):
        self.real = real
        self.is_real = not np.iscomplexobj(real.A)
        self.dtype = float if self.is_real else complex
        self.entries = {}
        for key, ent in real.entries.items():
            self.entries[key] = _EntryOps(ent, self.dtype)
        self._solve_data = {}
        self._lock = threading.RLock()

    def entry(self, pi, ri):
        return self.entries.get((int(pi), int(ri)))

    def solve_data(self, ents):
        """Resolvent data for a tensor whose axes carry the entries ``ents``
        (a tuple of ``_EntryOps``)."""
        key = tuple(id(e) for e in ents)
        with self._lock:
            sd = self._solve_data.get(key)
            if sd is None:
                sd = _SolveData(ents, self.is_real)
                self._solve_data[key] = sd
            return sd


class _SolveData:
    """``(⊕ a_e) y = -r`` for one active-edge pattern."""

    def __init__(self, ents, is_real):
        self.ents = ents
        self.shape = tuple(e.n for e in ents)
        self.is_real = is_real
        lam = np.zeros(self.shape, dtype=complex)
        for ax, e in enumerate(ents):
            sh = [1] * len(ents)
            sh[ax] = e.n
            lam = lam + e.dT.reshape(sh)
        if is_real and not any(e.kind == 'schur' for e in ents):
            lam = lam.real
        self.lam = lam
        self.nontriv = [(ax, e) for ax, e in enumerate(ents) if e.n > 1]
        self.sweeps = 1 + sum(e.n - 1 for e in ents)
        self.unitary = any(e.kind == 'schur' for e in ents)
        self.scalar = sum(e.lam for e in ents if e.n == 1)

    def apply_M(self, y):
        """``(⊕ a_e) y`` in the original basis."""
        out = self.scalar * y if self.scalar != 0 else np.zeros_like(y)
        for ax, e in self.nontriv:
            out = out + _mode(y, e.a, ax)
        return out

    def _solve_once(self, r):
        rt = r
        for ax, e in self.nontriv:
            rt = e.to_basis(rt, ax)
        y = -rt / self.lam
        for _ in range(self.sweeps - 1):
            acc = -rt
            for ax, e in self.nontriv:
                acc = acc - _mode(y, e.Nup, ax)
            y = acc / self.lam
        for ax, e in self.nontriv:
            y = e.from_basis(y, ax)
        if self.is_real and np.iscomplexobj(y):
            y = y.real
        return y

    def solve_neg(self, r):
        """``y`` with ``(⊕ a_e) y = -r``, i.e. ``y = (-M)^{-1} r``."""
        _count('ss_resolvent_solves')
        y = self._solve_once(r)
        if not self.unitary or REFINE_STEPS <= 0:
            return y
        res = -r - self.apply_M(y)
        best, rbest = y, float(np.abs(res).max())
        for _ in range(REFINE_STEPS):
            if rbest == 0.0:
                break
            y = best + self._solve_once(-res)
            res = -r - self.apply_M(y)
            rn = float(np.abs(res).max())
            _count('ss_refinements')
            if not rn < rbest:
                break
            best, rbest = y, rn
        return best


# ─── dense segment generators (LRU, byte capped) ──────────────────────────

_DENSE_LOCK = threading.RLock()
_DENSE_CACHE = OrderedDict()
_DENSE_BYTES = [0]


def _dense_get(key):
    with _DENSE_LOCK:
        hit = _DENSE_CACHE.get(key)
        if hit is not None:
            _DENSE_CACHE.move_to_end(key)
        return hit


def _dense_put(key, val, nbytes):
    with _DENSE_LOCK:
        if nbytes > DENSE_CACHE_BYTES:
            return
        _DENSE_CACHE[key] = (val, nbytes)
        _DENSE_BYTES[0] += nbytes
        while _DENSE_BYTES[0] > DENSE_CACHE_BYTES and _DENSE_CACHE:
            _k, (_v, nb) = _DENSE_CACHE.popitem(last=False)
            _DENSE_BYTES[0] -= nb


# ─── Taylor action (Al-Mohy & Higham 2011) ────────────────────────────────

_THETA = {1: 2.29e-16, 2: 2.58e-08, 3: 1.39e-05, 4: 3.4e-04, 5: 2.4e-03,
          6: 9.07e-03, 7: 2.38e-02, 8: 5.0e-02, 9: 8.96e-02, 10: 1.44e-01,
          11: 2.14e-01, 12: 3.0e-01, 13: 4.0e-01, 14: 5.14e-01, 15: 6.41e-01,
          16: 7.81e-01, 17: 9.31e-01, 18: 1.09, 19: 1.26, 20: 1.44,
          21: 1.62, 22: 1.82, 23: 2.01, 24: 2.22, 25: 2.43, 26: 2.64,
          27: 2.86, 28: 3.08, 29: 3.31, 30: 3.54, 35: 4.7, 40: 6.0,
          45: 7.2, 50: 8.5, 55: 9.9}
_M_MAX = 55
_P_MAX = 8                      # largest p with p(p-1) <= m_max + 1
_TOL = 2.0 ** -53


def _taylor_params(dnorms, t):
    """(m*, s) of Al-Mohy & Higham's fragment (3.1) for ``t A``; ``dnorms[p]``
    = ``||A^p||_1^{1/p}`` (an upper bound is safe), p = 1 .. P_MAX + 1."""
    a1 = t * dnorms[1]
    if a1 == 0.0:
        return 0, 1
    ell = 2
    if a1 <= 2 * ell * _P_MAX * (_P_MAX + 3) * _THETA[_M_MAX] / _M_MAX:
        best = None
        for m, th in _THETA.items():
            s = int(math.ceil(a1 / th))
            if best is None or m * s < best[0] * best[1]:
                best = (m, s)
        return best
    best = None
    for p in range(2, _P_MAX + 1):
        alpha = t * max(dnorms[p], dnorms[p + 1])
        for m in range(p * (p - 1) - 1, _M_MAX + 1):
            th = _THETA.get(m)
            if th is None:
                continue
            s = max(1, int(math.ceil(alpha / th)))
            if best is None or m * s < best[0] * best[1]:
                best = (m, s)
    return best


def _taylor_action(apply_A, mu, dnorms, x, t):
    """``exp(t (A + mu I)) x`` by the truncated Taylor series of ``t A`` in
    ``s`` steps (``A`` already trace shifted)."""
    m, s = _taylor_params(dnorms, t)
    if m == 0:
        return x * np.exp(t * mu) if mu != 0 else x.copy()
    _count('ss_segments_taylor')
    F = x
    b = x
    eta = np.exp(t * mu / s)
    nmv = 0
    for _ in range(s):
        c1 = float(np.abs(b).max())
        for j in range(1, m + 1):
            b = apply_A(b) * (t / (s * j))
            nmv += 1
            c2 = float(np.abs(b).max())
            F = F + b
            if c1 + c2 <= _TOL * float(np.abs(F).max()):
                break
            c1 = c2
        F = eta * F
        b = F
    _count('ss_taylor_matvecs', nmv)
    return F


def _onenorm_powers(A, pmax=_P_MAX + 1):
    """``d_p = ||A^p||_1^{1/p}``, p = 1..pmax: exact for a small dense ``A``,
    1-norm estimates (scipy ``onenormest``) for larger ones."""
    d = {1: float(np.abs(A).sum(axis=0).max()) if A.size else 0.0}
    n = A.shape[0]
    if n <= 300:
        P = A
        for p in range(2, pmax + 1):
            P = P @ A
            d[p] = float(np.abs(P).sum(axis=0).max()) ** (1.0 / p)
        return d
    from scipy.sparse.linalg import LinearOperator, onenormest
    AT = A.conj().T
    for p in range(2, pmax + 1):
        def mv(v, p=p):
            for _ in range(p):
                v = A @ v
            return v

        def rmv(v, p=p):
            for _ in range(p):
                v = AT @ v
            return v
        op = LinearOperator((n, n), matvec=mv, rmatvec=rmv,
                            matmat=mv, rmatmat=rmv, dtype=A.dtype)
        d[p] = float(onenormest(op)) ** (1.0 / p)
    return d


# ─── geometry ─────────────────────────────────────────────────────────────

class _Segment:
    """The block system of one finite segment: blocks = downsets reachable
    in it, each with its active edges; firings between them."""

    def __init__(self, sub, blocks, acts, trans, key):
        self.sub = sub
        self.blocks = blocks
        self.index = {I: k for k, I in enumerate(blocks)}
        self.acts = acts
        self.ents = [tuple(sub.edge_ops[e] for e in act) for act in acts]
        self.shapes = [tuple(e.n for e in en) for en in self.ents]
        self.sizes = [int(np.prod(sh, dtype=np.int64)) for sh in self.shapes]
        offs, o = [], 0
        for sz in self.sizes:
            offs.append(o)
            o += sz
        self.offs = offs
        self.D = o
        self.trans = trans          # (kI, kJ, sublist operands builder)
        self.key = key
        self._sd = [sub.model.solve_data(en) for en in self.ents]
        self._dnorms_free = None

    # stacking
    def stack(self, xd, dtype):
        v = np.zeros(self.D, dtype=dtype)
        for I, x in xd.items():
            k = self.index[I]
            v[self.offs[k]:self.offs[k] + self.sizes[k]] = np.ravel(x)
        return v

    def unstack(self, v):
        return {I: v[self.offs[k]:self.offs[k] + self.sizes[k]].reshape(
                    self.shapes[k]) for k, I in enumerate(self.blocks)}

    # matrix-free apply of G
    def apply(self, v):
        out = np.zeros_like(v)
        for k in range(len(self.blocks)):
            o, sz = self.offs[k], self.sizes[k]
            x = v[o:o + sz].reshape(self.shapes[k])
            out[o:o + sz] += np.ravel(self._sd[k].apply_M(x))
        for (kI, kJ, fire) in self.trans:
            o, sz = self.offs[kI], self.sizes[kI]
            x = v[o:o + sz].reshape(self.shapes[kI])
            oJ, szJ = self.offs[kJ], self.sizes[kJ]
            out[oJ:oJ + szJ] += np.ravel(fire(x))
        return out

    def trace(self):
        tr = 0j
        for k, en in enumerate(self.ents):
            for e in en:
                tr += e.trace * (self.sizes[k] / e.n)
        return tr

    def dense(self, dtype):
        key = ('G', self.key)
        hit = _dense_get(key)
        if hit is not None:
            return hit[0]
        D = self.D
        G = np.zeros((D, D), dtype=dtype)
        for k in range(len(self.blocks)):
            o, sz = self.offs[k], self.sizes[k]
            G[o:o + sz, o:o + sz] = _kron_sum_dense(self.ents[k], dtype)
        for (kI, kJ, fire) in self.trans:
            o, sz = self.offs[kI], self.sizes[kI]
            oJ, szJ = self.offs[kJ], self.sizes[kJ]
            basis = np.eye(sz, dtype=dtype).reshape(self.shapes[kI] + (sz,))
            G[oJ:oJ + szJ, o:o + sz] = fire(basis, batch=True).reshape(szJ, sz)
        mu = np.trace(G) / D if D else 0.0
        if self.sub.model.is_real:
            mu = float(np.real(mu))
        A = G - mu * np.eye(D, dtype=dtype)
        dn = _onenorm_powers(A)
        val = (G, A, mu, dn)
        _dense_put(key, val, 2 * G.nbytes)
        return val

    def free_norms(self):
        """1-norm bounds for the matrix-free operator (trace shifted)."""
        if self._dnorms_free is None:
            mu = self.trace() / self.D if self.D else 0.0
            if self.sub.model.is_real:
                mu = float(np.real(mu))
            col = [0.0] * len(self.blocks)
            for k, en in enumerate(self.ents):
                col[k] = sum(e.norm1 for e in en) + abs(mu)
            for (kI, kJ, fire) in self.trans:
                col[kI] += fire.norm1
            nrm = max(col) if col else 0.0
            self._dnorms_free = (mu, {p: nrm for p in range(1, _P_MAX + 2)})
        return self._dnorms_free

    def propagate(self, xd, L, dtype):
        """``exp(G L)`` applied to the state dict ``xd`` (missing blocks: 0)."""
        v = self.stack(xd, dtype)
        if L == 0.0 or self.D == 0:
            return self.unstack(v)
        out = self.action(v, L, dtype)
        return self.unstack(out)

    def action(self, v, L, dtype):
        """``exp(G L) v``: the trace-shifted truncated-Taylor action (dense
        ``G`` up to ``OP_DENSE_MAX``, matrix-free beyond).  No squaring:
        scipy's ``expm`` loses up to ~1e-13 relative on decaying blocks."""
        if self.D <= OP_DENSE_MAX:
            G, A, mu, dn = self.dense(dtype)
            return _taylor_action(lambda y: A @ y, mu, dn, v, L)
        mu, dn = self.free_norms()
        if mu != 0:
            return _taylor_action(lambda y: self.apply(y) - mu * y, mu, dn,
                                  v, L)
        return _taylor_action(self.apply, 0.0, dn, v, L)


def _kron_sum_dense(ents, dtype):
    shape = tuple(e.n for e in ents)
    D = int(np.prod(shape, dtype=np.int64))
    M = np.zeros((D, D), dtype=dtype)
    for ax, e in enumerate(ents):
        left = int(np.prod(shape[:ax], dtype=np.int64))
        right = int(np.prod(shape[ax + 1:], dtype=np.int64))
        M = M + np.kron(np.kron(np.eye(left, dtype=dtype), e.a),
                        np.eye(right, dtype=dtype))
    return M


class _Fire:
    """Contract the edges ``ends`` (by ``c_e``) and open the edges
    ``starts`` (by ``b_e``) of a tensor whose axes are ``act_in``; the result
    has axes ``act_out`` (all lists of edge ids, increasing)."""
    __slots__ = ('args', 'out', 'act_in', 'norm1')

    def __init__(self, sub, act_in, ends, starts, act_out):
        args = []
        for e in ends:
            args += [sub.edge_ops[e].c, [e]]
        for e in starts:
            args += [sub.edge_ops[e].b, [e]]
        self.args = args
        self.act_in = list(act_in)
        self.out = list(act_out)
        n1 = 1.0
        for e in ends:
            n1 *= sub.edge_ops[e].cnorm
        for e in starts:
            n1 *= sub.edge_ops[e].bnorm
        self.norm1 = n1

    def __call__(self, x, batch=False):
        if batch:
            z = MAX_EDGES + 1
            return np.einsum(x, self.act_in + [z], *self.args,
                             self.out + [z])
        return np.einsum(x, self.act_in, *self.args, self.out)


_SUBSET_SERIAL = itertools.count()


class SubsetSS:
    """One subset's geometry and its state-space evaluation (no prefactor).

    ``ents``: the ``_EntryOps`` of the smooth rows (``None`` entries are
    not allowed); ``cdata``: the rows ``(a_int, a_ext, c0)``; ``m``: number
    of internal times; ``n_free``: number of free external times.  Raises
    ``SSIneligible`` for a shape it does not take.
    """

    def __init__(self, model, ents, cdata, m, n_free):
        self.model = model
        self.m = int(m)
        self.n_free = int(n_free)
        if len(cdata) > MAX_EDGES:
            raise SSIneligible('too_many_edges', str(len(cdata)))
        self.const_rows = []            # (row idx, a_int, a_ext, c0, ent)
        edges = []                      # (tail, head, ent, row idx)
        slots = {}
        slot_list = []

        def slot(a_ext, c0):
            key = tuple(0.0 if x == 0 else float(x) for x in a_ext) + (
                0.0 if c0 == 0 else float(c0),)
            k_ = slots.get(key)
            if k_ is None:
                k_ = slots[key] = len(slot_list)
                slot_list.append(key)
            return k_

        for idx, ((a_int, a_ext, c0), ent) in enumerate(zip(cdata, ents)):
            a_int = [float(x) for x in a_int]
            a_ext = [float(x) for x in a_ext]
            c0 = float(c0)
            if len(a_int) != self.m or len(a_ext) != self.n_free:
                raise SSIneligible('row_shape', 'row length')
            nz = [(i, a) for i, a in enumerate(a_int) if a != 0.0]
            if not nz:
                self.const_rows.append((idx, a_int, a_ext, c0, ent))
                continue
            heads = [i for i, a in nz if a == 1.0]
            tails = [i for i, a in nz if a == -1.0]
            if (len(heads) + len(tails) != len(nz) or len(heads) > 1
                    or len(tails) > 1):
                raise SSIneligible('row_shape', f'a_int={a_int}')
            if heads and tails:
                if any(x != 0.0 for x in a_ext) or c0 != 0.0:
                    raise SSIneligible('offset_row', f'{a_ext}, {c0}')
                edges.append((('v', tails[0]), ('v', heads[0]), ent, idx))
            elif heads:                 # Δt = s_v + (a·f + c0): tail -(a·f+c0)
                edges.append((('c', slot([-x for x in a_ext], -c0)),
                              ('v', heads[0]), ent, idx))
            else:                       # Δt = -s_u + (a·f + c0): head a·f+c0
                edges.append((('v', tails[0]),
                              ('c', slot(a_ext, c0)), ent, idx))
        self.edges = edges
        self.edge_ops = [e[2] for e in edges]
        self.slots = slot_list
        m_ = self.m
        self.in_edges = [[] for _ in range(m_)]
        self.out_edges = [[] for _ in range(m_)]
        self.preds = [set() for _ in range(m_)]
        self.succs = [set() for _ in range(m_)]
        self.lo_slots = [set() for _ in range(m_)]
        self.hi_slots = [set() for _ in range(m_)]
        self.slot_heads = [[] for _ in slot_list]   # edges ending at a slot
        self.slot_tails = [[] for _ in slot_list]   # edges starting at a slot
        for k_, (t, h, _ent, _i) in enumerate(edges):
            if t[0] == 'v':
                self.out_edges[t[1]].append(k_)
            else:
                self.slot_tails[t[1]].append(k_)
            if h[0] == 'v':
                self.in_edges[h[1]].append(k_)
            else:
                self.slot_heads[h[1]].append(k_)
            if t[0] == 'v' and h[0] == 'v':
                if t[1] == h[1]:
                    raise SSIneligible('row_shape', 'self loop')
                self.preds[h[1]].add(t[1])
                self.succs[t[1]].add(h[1])
            elif t[0] == 'c':
                self.lo_slots[h[1]].add(t[1])
            else:
                self.hi_slots[t[1]].add(h[1])
        self.full = frozenset(range(m_))
        self.serial = next(_SUBSET_SERIAL)
        self._stat = None
        self._patterns = {}
        self._lock = threading.RLock()
        self._prefetch = {}
        # a directed cycle of internal rows (s_u < s_v < ... < s_u, every
        # internal row has zero offset): the region is empty, the value 0
        order = self._topo_order()
        self.empty = order is None
        if self.empty:
            self.free0 = frozenset()
            self._stat_downsets = [frozenset()]
            return
        # every vertex followed by an external time
        reach = [False] * m_
        for v in reversed(order):
            reach[v] = bool(self.hi_slots[v]) or any(reach[w]
                                                     for w in self.succs[v])
        if m_ and not all(reach):
            raise SSIneligible('vertex_after_last_time')
        if m_ and not slot_list:
            raise SSIneligible('no_external_time')
        # F0: vertices with no external-time tail in their ancestry
        free0 = [False] * m_
        for v in order:
            free0[v] = (not self.lo_slots[v]) and all(free0[u]
                                                      for u in self.preds[v])
        self.free0 = frozenset(v for v in range(m_) if free0[v])
        self._stat_downsets = self._downsets(self.free0)

    # ── structure helpers ─────────────────────────────────────────────
    def _topo_order(self):
        indeg = [len(p) for p in self.preds]
        stack = [v for v in range(self.m) if indeg[v] == 0]
        out = []
        while stack:
            v = stack.pop()
            out.append(v)
            for w in sorted(self.succs[v]):
                indeg[w] -= 1
                if indeg[w] == 0:
                    stack.append(w)
        if len(out) != self.m:
            return None
        return out

    def _closure(self, starts, placeable):
        """Downsets reachable from ``starts`` by placing vertices of
        ``placeable`` (whose predecessors are placed); sorted."""
        seen = set(starts)
        frontier = list(starts)
        while frontier:
            nxt = []
            for I in frontier:
                for v in placeable:
                    if v in I or not self.preds[v] <= I:
                        continue
                    J = I | {v}
                    if J not in seen:
                        seen.add(J)
                        nxt.append(J)
                        if len(seen) > MAX_DOWNSETS:
                            raise SSIneligible('too_many_downsets')
            frontier = nxt
        return sorted(seen, key=lambda I: (len(I), sorted(I)))

    def _downsets(self, verts):
        return self._closure([frozenset()], sorted(verts))

    def _act(self, I, passed):
        """Active edges (increasing ids) of downset ``I`` once the slots in
        ``passed`` are behind (``None``: the stationary segment)."""
        out = []
        for k_, (t, h, _e, _i) in enumerate(self.edges):
            if t[0] == 'v':
                t_done = t[1] in I
            else:
                t_done = passed is not None and t[1] in passed
            if not t_done:
                continue
            if h[0] == 'v':
                h_done = h[1] in I
            else:
                h_done = passed is not None and h[1] in passed
            if not h_done:
                out.append(k_)
        return tuple(out)

    def _maximal(self, J):
        return [v for v in sorted(J) if not (self.succs[v] & J)]

    def state_bytes(self):
        """Bytes of the cached stationary states (the memory estimate)."""
        item = 8 if self.model.is_real else 16
        tot = 0
        for J in self._stat_downsets:
            sz = 1
            for e in self._act(J, None):
                sz *= self.edge_ops[e].n
            tot += sz
        return tot * item

    def max_width(self):
        """(largest number of active edges, largest block size) over the
        stationary downsets."""
        w = b = 0
        for J in self._stat_downsets:
            act = self._act(J, None)
            sz = 1
            for e in act:
                sz *= self.edge_ops[e].n
            w, b = max(w, len(act)), max(b, sz)
        return w, b

    # ── stationary states (cached) ────────────────────────────────────
    def stationary(self):
        with self._lock:
            if self._stat is not None:
                return self._stat
            _count('ss_stationary_builds')
            dt = self.model.dtype
            x = {frozenset(): np.ones((), dtype=dt)}
            acts = {frozenset(): ()}
            for J in self._stat_downsets:
                if not J:
                    continue
                actJ = self._act(J, None)
                acts[J] = actJ
                r = None
                for v in self._maximal(J):
                    I = J - {v}
                    xI = x.get(I)
                    if xI is None:
                        continue
                    f = _Fire(self, acts[I], self.in_edges[v],
                              self.out_edges[v], actJ)
                    term = f(xI)
                    r = term if r is None else r + term
                if r is None:
                    continue
                sd = self.model.solve_data(tuple(self.edge_ops[e]
                                                 for e in actJ))
                x[J] = sd.solve_neg(r)
            self._stat = x
            return x

    # ── per-order programs ────────────────────────────────────────────
    def _program(self, pattern, finite_start):
        key = (pattern, finite_start)
        with self._lock:
            prog = self._patterns.get(key)
            if prog is not None:
                return prog
        prog = self._build_program(pattern, finite_start)
        with self._lock:
            self._patterns[key] = prog
        return prog

    def _build_program(self, pattern, finite_start):
        steps = []
        if finite_start:
            blocks = self._downsets(self.free0)
            seg = self._segment(blocks, None, sorted(self.free0),
                                ('tmin', self.serial, pattern))
            steps.append(('seg', seg))
            states = list(blocks)
        else:
            states = list(self._stat_downsets)
        passed = set()
        for j, group in enumerate(pattern):
            before = frozenset(passed)
            passed |= set(group)
            after = frozenset(passed)
            req = frozenset(v for v in range(self.m)
                            if self.hi_slots[v] & after)
            ends = [k_ for s in group for k_ in self.slot_heads[s]]
            starts = [k_ for s in group for k_ in self.slot_tails[s]]
            ev = []
            kept = []
            for I in states:
                if not req <= I:
                    continue
                act_in = self._act(I, before)
                act_out = self._act(I, after)
                e_in = sorted(e for e in ends if e in act_in)
                f = _Fire(self, act_in, e_in, sorted(starts), act_out)
                ev.append((I, f))
                kept.append(I)
            steps.append(('event', ev))
            if j + 1 == len(pattern):
                states = kept
                break
            placeable = [v for v in range(self.m)
                         if self.lo_slots[v] <= after
                         and not (self.hi_slots[v] & after)]
            blocks = self._closure(kept, placeable)
            seg = self._segment(blocks, after, placeable,
                                (self.serial, pattern, finite_start, j))
            steps.append(('seg', seg))
            states = blocks
        if self.full not in states:
            steps.append(('empty', None))
        return steps

    def _segment(self, blocks, passed, placeable, key):
        acts = [self._act(I, passed) for I in blocks]
        index = {I: k for k, I in enumerate(blocks)}
        trans = []
        pl = set(placeable)
        for kI, I in enumerate(blocks):
            for v in sorted(pl - I):
                if not self.preds[v] <= I:
                    continue
                J = I | {v}
                kJ = index.get(J)
                if kJ is None:
                    continue
                f = _Fire(self, acts[kI], sorted(self.in_edges[v]),
                          sorted(self.out_edges[v]), acts[kJ])
                trans.append((kI, kJ, f))
        return _Segment(self, blocks, acts, trans, key)

    # ── evaluation ────────────────────────────────────────────────────
    def _slot_values(self, free_vals):
        vals = []
        for key in self.slots:
            v = key[-1]
            for a, f in zip(key[:-1], free_vals):
                if a != 0.0:
                    v = v + a * float(f)
            vals.append(v)
        return vals

    def _pattern(self, vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        groups, gv = [], []
        for i in order:
            if gv and vals[i] == gv[-1]:
                groups[-1].append(i)
            else:
                groups.append([i])
                gv.append(vals[i])
        return tuple(tuple(sorted(g)) for g in groups), gv

    def scalar_factor(self, free_vals, tie_ctx):
        """Product of the constant rows' factors; 0 when one is EMPTY."""
        FI = _FI()
        scal = 1.0
        for (_idx, a_int, a_ext, c0, ent) in self.const_rows:
            verdict = FI._const_row_verdict(a_int, a_ext, c0, free_vals,
                                            'edge', tie_ctx)
            if verdict == 'EMPTY':
                return 0.0
            c_eff = float(c0) + sum(float(a_ext[j]) * float(free_vals[j])
                                    for j in range(len(a_ext)))
            scal = scal * ent.impulse(c_eff)
            if scal == 0:
                return 0.0
        return scal

    def evaluate(self, free_vals, tie_ctx=None, t_min=float('-inf')):
        """The subset integral (WITHOUT prefactor) at ``free_vals``."""
        if self.empty or any(e.kind == 'zero' for e in self.edge_ops) or any(
                r[4].kind == 'zero' for r in self.const_rows):
            return 0.0
        scal = self.scalar_factor(free_vals, tie_ctx)
        if scal == 0:
            return 0.0
        if self.m == 0:
            return scal
        vals = self._slot_values(free_vals)
        if any(v != v for v in vals):
            raise ValueError('state-space route: NaN external time')
        pattern, gv = self._pattern(vals)
        finite = t_min != float('-inf')
        prog = self._program(pattern, finite)
        x = self._run(prog, gv, finite, t_min)
        return scal * x

    def _run(self, prog, gv, finite, t_min, upto=None):
        """Run a program; ``gv``: the event times.  With ``upto``: return
        the state dict before step ``upto``."""
        dt = self.model.dtype
        x = ({frozenset(): np.ones((), dtype=dt)} if finite
             else self.stationary())
        j = 0
        prev = t_min if finite else None
        for i, (kind, data) in enumerate(prog):
            if upto is not None and i == upto:
                return x
            if kind == 'seg':
                L = gv[j] - prev
                if L < 0:
                    raise ValueError('state-space route: external times '
                                     'before PHASE_J_T_MIN')
                x = data.propagate(x, L, dt)
            elif kind == 'event':
                nx = {}
                for I, f in data:
                    xI = x.get(I)
                    if xI is not None:
                        nx[I] = f(xI)
                x = nx
                prev = gv[j]
                j += 1
            else:
                return 0.0
        v = x.get(self.full)
        return 0.0 if v is None else v.item()

    def evaluate_many(self, points, t_min=float('-inf')):
        """Values (without prefactor) at many ``(free_vals, tie_ctx)``
        points.  Points sharing the order of the external times share the
        stationary state and every step before the last segment; along the
        last segment they are stepped from one length to the next (sorted).
        """
        out = [None] * len(points)
        groups = {}
        finite = t_min != float('-inf')
        for i, (fv, tc) in enumerate(points):
            if self.empty or any(e.kind == 'zero' for e in self.edge_ops) or any(
                    r[4].kind == 'zero' for r in self.const_rows):
                out[i] = 0.0
                continue
            scal = self.scalar_factor(fv, tc)
            if scal == 0 or self.m == 0:
                out[i] = scal if self.m == 0 else 0.0
                continue
            vals = self._slot_values(fv)
            pattern, gv = self._pattern(vals)
            groups.setdefault(pattern, []).append((i, gv, scal))
        dt = self.model.dtype
        for pattern, items in groups.items():
            prog = self._program(pattern, finite)
            _count('ss_batch_points', len(items))
            segs = [i for i, (k_, _d) in enumerate(prog) if k_ == 'seg']
            if (finite or not segs or prog[-1][0] != 'event'
                    or segs[-1] != len(prog) - 2 or len(items) < 2):
                for (i, gv, scal) in items:
                    out[i] = scal * self._run(prog, gv, finite, t_min)
                continue
            last = segs[-1]
            # the state before the last segment depends only on the lengths
            # of the earlier segments (translation invariance): group by them
            pre = {}
            for (i, gv, scal) in items:
                lens = tuple(gv[q + 1] - gv[q] for q in range(len(gv) - 1))
                pre.setdefault(lens[:-1], []).append((i, gv, scal, lens[-1]))
            seg = prog[last][1]
            ev_last = [(I, f) for I, f in prog[last + 1][1]
                       if I == self.full]
            for _k, its in pre.items():
                x0 = self._run(prog, its[0][1], False, t_min, upto=last)
                its = sorted(its, key=lambda it: it[3])
                v = seg.stack(x0, dt)
                cur = 0.0
                for (i, gv, scal, L_last) in its:
                    if L_last > cur:
                        v = seg.action(v, L_last - cur, dt)
                        cur = L_last
                    nx = 0.0
                    if ev_last:
                        I, f = ev_last[0]
                        k = seg.index.get(I)
                        if k is not None:
                            xI = v[seg.offs[k]:seg.offs[k] + seg.sizes[k]]
                            nx = f(xI.reshape(seg.shapes[k])).item()
                    out[i] = scal * nx
        return out


# ─── τ-grid prefetch (batching through the per-point API) ─────────────────

_PREFETCH = threading.local()


_PREFETCH_TOKENS = itertools.count(1)


class tau_grid_prefetch:
    """Within this context, the first call of a state-space subset with a
    given Wick permutation (its tie context's leg mapping) evaluates the
    whole registered grid of raw external-time tuples at once
    (``SubsetSS.evaluate_many``) and later calls of the grid are served
    from that memo.  ``points``: the k-tuples of raw external times exactly
    as they will be passed to ``contribution(*times)``.  The values differ
    from per-point evaluation only by rounding (the last segment is
    stepped from one length to the next).  Per thread; nests."""

    def __init__(self, points):
        self.points = [tuple(float(x) for x in p) for p in points]

    def __enter__(self):
        self._prev = getattr(_PREFETCH, 'grid', None)
        _PREFETCH.grid = (self.points, next(_PREFETCH_TOKENS))
        return self

    def __exit__(self, *exc):
        _PREFETCH.grid = self._prev
        return False


def _prefetched(sub, free_vals, tie_ctx, t_min):
    grid = getattr(_PREFETCH, 'grid', None)
    if grid is None or tie_ctx is None or t_min != float('-inf'):
        return None
    points, token = grid
    try:
        times = tuple(float(x) for x in tie_ctx.times)
    except (TypeError, ValueError):
        return None
    mapping = (tuple(tie_ctx.free_legs), tie_ctx.origin_leg)
    with sub._lock:
        memos = sub._prefetch.get(token)
        if memos is None:
            memos = {}
            sub._prefetch = {token: memos}          # one grid at a time
        memo = memos.get(mapping)
    if memo is None:
        pts = []
        for T in points:
            if len(T) != len(times):
                return None
            o = 0.0 if tie_ctx.origin_leg is None else T[tie_ctx.origin_leg]
            fv = [T[leg] - o for leg in tie_ctx.free_legs]
            pts.append((fv, type(tie_ctx)(T, tie_ctx.free_legs,
                                          tie_ctx.origin_leg)))
        vals = sub.evaluate_many(pts)
        memo = {}
        for T, (fv, _tc), v in zip(points, pts, vals):
            memo[(T, tuple(fv))] = v
        with sub._lock:
            memos[mapping] = memo
    v = memo.get((times, tuple(float(x) for x in free_vals)))
    if v is not None:
        _count('ss_prefetch_hits')
    return v


# ─── the dispatch point's maker ───────────────────────────────────────────

def subset_maker(*, propagator_data, num_params, pi_ri, plain_edges,
                 prefactor, cdata, m, row_kinds, n_free, modes=None):
    """Called by ``integrate_diagram`` for each subset when
    ``DAEDALUS_PHASE_J_ENGINE`` is set.  Returns ``None`` when the route is
    off or the subset is not eligible (counted by reason in
    ``ss_declined_by_reason``); otherwise a function with the signature of
    ``integrate_diagram``'s ``_make_subset_contrib`` that builds the subset
    closure ``_contrib(free_vals, _hook_ctx=None, _tie_ctx=None)``."""
    if not engine_requested():
        return None
    FI = _FI()
    if FI._theta0_legacy():
        return _decline('theta0_legacy')
    if not plain_edges:
        return _decline('non_propagator_rows')
    if prefactor is None:
        return _decline('symbolic_prefactor')
    if row_kinds is not None and any(k_ != 'edge' for k_ in row_kinds):
        return _decline('non_edge_rows')
    if len(cdata) != len(pi_ri):
        return _decline('row_count')
    model, why = realization_for(propagator_data, num_params)
    if model is None:
        return _decline(why)
    ents = []
    for (pi, ri) in pi_ri:
        e = model.entry(pi, ri)
        if e is None:
            return _decline('entry_missing')
        ents.append(e)
    try:
        sub = SubsetSS(model, ents, cdata, m, n_free)
    except SSIneligible as exc:
        return _decline(exc.reason)
    _charge(sub)
    _count('ss_subsets_built')
    pref = complex(prefactor)
    edge_modes = modes            # the dispatch passes its own (None) modes

    def make(fc, cdata_, m_val, modes=None, pref_c=None, pole_tuples=None,
             plan=None, hook_meta=None, row_kinds=None, theta0_empty=False):
        def _contrib(free_vals, _hook_ctx=None, _tie_ctx=None):
            if theta0_empty and not FI._theta0_legacy():
                FI._RUNTIME_COUNTERS['theta0_subsets_pruned'] += 1
                return 0.0 + 0.0j
            if FI._theta0_legacy():
                raise ValueError(
                    'the state-space Phase J route implements the Itô '
                    'Θ(0) rule only; unset DAEDALUS_PHASE_J_ENGINE to use '
                    "THETA0_CONST_ROW_MODE='legacy_clip'")
            t_min = _t_min_frame(free_vals, _tie_ctx)
            val = _prefetched(sub, free_vals, _tie_ctx, t_min)
            if val is None:
                val = sub.evaluate(free_vals, _tie_ctx, t_min)
            val = complex(pref * val)
            _count('ss_evals')
            hook = FI._SUBSET_HOOK
            if hook is not None:
                FI._emit_subset_hook(
                    hook, hook_meta, _hook_ctx, free_vals, m_val, cdata_,
                    path='ss', evaluator='statespace.SubsetSS', branch='ss',
                    value=val, bail_reason=None, attempted='ss',
                    modes=edge_modes, prefactor=pref, plan=sub,
                    pole_tuples=None, integrand=None,
                    row_kinds=row_kinds, tie_ctx=_tie_ctx)
            return val
        _contrib.ss_subset = sub
        return _contrib
    return make


def _t_min_frame(free_vals, tie_ctx):
    """``final_integral.PHASE_J_T_MIN`` in the frame of one evaluation (as
    ``final_integral._t_min_frame``, without its box-free-route
    requirement, which concerns the pole routes only)."""
    FI = _FI()
    raw = FI.PHASE_J_T_MIN
    try:
        t = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'final_integral.PHASE_J_T_MIN={raw!r}: expected '
                         f'a float') from None
    if t == float('-inf'):
        return t
    if not math.isfinite(t):
        raise ValueError(f'final_integral.PHASE_J_T_MIN={t!r}: expected '
                         f'-inf or a finite float')
    t0 = 0.0
    if tie_ctx is not None and tie_ctx.origin_leg is not None:
        t0 = float(tie_ctx.times[tie_ctx.origin_leg])
    t_rel = t - t0
    if t_rel > min([0.0] + [float(x) for x in free_vals]):
        raise ValueError(
            f'final_integral.PHASE_J_T_MIN={t!r} is after an external time '
            f'(the cumulant is asked for before the switch-on)')
    return t_rel


def _charge(sub):
    """Charge the subset's stationary-state bytes against item 35's running
    budget (``PHASE_J_MEM_BUDGET_GB``) before anything is allocated."""
    FI = _FI()
    nbytes = sub.state_bytes()
    with FI._PLAN_MEM_LOCK:
        scoped = FI._PLAN_MEM['active'] > 0
        budget = (FI._PLAN_MEM['budget'] if scoped
                  else FI.phase_j_mem_budget_bytes())
        used = FI._PLAN_MEM['used'] if scoped else 0
        if budget is not None and used + nbytes > budget:
            w, b = sub.max_width()
            run = f' of {FI._PLAN_MEM["what"]}' if (
                scoped and FI._PLAN_MEM['what']) else ''
            raise FI.PhaseJResourceError(
                f'Phase J memory budget exceeded{run}: the state-space '
                f'route would hold {FI._fmt_gb(nbytes)} of stationary states '
                f'for one subset ({len(sub._stat_downsets)} downsets, '
                f'widest cut {w} edges, largest block {b:,} entries) on top '
                f'of {FI._fmt_gb(used)}, over the budget of '
                f'{FI._fmt_gb(budget)}.  Nothing was allocated for it.  '
                f'Options: {FI._BUDGET_HOW}.',
                {'bytes': nbytes, 'bytes_so_far': used,
                 'budget_bytes': budget, 'width': w, 'block': b})
        if scoped:
            FI._PLAN_MEM['used'] += nbytes

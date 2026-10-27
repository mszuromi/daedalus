r"""
State-space realization of the propagator without roots
========================================================

``realize(prop, num_params)`` turns the propagator stage's Fourier kernel
``K_ft`` (``api/_propagator.py``: rows = response fields, columns = physical
fields, symbolic in ``omega``) into a state-space quadruple

    G(s) = K(s)^{-1} = C (s I - A)^{-1} B + D,          s = i omega,
    G(t) = Theta(t) C e^{A t} B + D delta(t),

without computing an eigenvalue or a numerical polynomial root.  The
state-space Phase J route (``engine/integration/time_domain/statespace.py``,
the default engine) integrates with it; the pole engine
(``DAEDALUS_PHASE_J_ENGINE=poles``) does not use it.

Construction (all exact, then rounded once)
-------------------------------------------
1. Every parameter value in ``num_params`` is replaced by the exact rational
   of its binary floating-point value, every other inexact number (a float
   literal in a kernel string, a constant such as ``e^{-13/10}``) by the
   exact rational of its correctly rounded double, ``omega -> -i s``, and
   every entry of ``K`` is converted to an element of ``F(s)`` with
   ``F = QQ`` (or the Gaussian rationals when a coefficient is complex).
   An entry that is not rational in ``s`` is refused (``non_rational``), a
   symbol left unbound is refused (``unbound_symbol``; ``spatial`` for the
   Laplacian / momentum).
2. Each entry is split ``K_ij = q0 + q1 s + r(s)/d(s)``, ``deg r < deg d``
   (a polynomial part of degree >= 2 is refused, ``improper_entry``).  The
   denominator is factored EXACTLY over ``F`` (algebra, no tolerance): linear
   factors carry the kernel's given rates, irreducible factors of higher
   degree stay whole.  ``r/d`` is realized as a cascade (chain) of blocks,
   one per factor (with multiplicity), each with unit DC gain
   (``z_k = f_k(0)/f_k(s) z_{k-1}``): a first-order block per rate, a
   tridiagonal Schwarz block per irreducible factor (from its Routh table,
   exactly similar to the companion matrix; a vanishing Routh pivot is
   avoided by an exact integer shift, the companion matrix is the last
   resort), with output coefficients from a Newton-type division
   ``r = c_N + f_N (c_{N-1} + f_{N-1}(...))`` -- the divided-difference
   basis, which, unlike partial fractions (residues), stays bounded when
   rates coincide or nearly coincide.  Erlang kernels become the Erlang
   chain (subdiagonal = rate), alpha kernels a 2-chain, exponential sums a
   chain of their rates, rational (e.g. Pade) images Schwarz blocks.
3. The loop is closed as a descriptor pencil in the variables ``(u, z)``
   (physical fields, kernel states):

       [s K1 + K0   C_R    ] [u]   [y]
       [ -B_R     s I - A_R] [z] = [0],       u = G(s) y,

   i.e. ``E = diag(K1, I)``.  If ``E`` is singular (algebraic fields, delta
   parts) the pencil is split exactly (pivot rows/columns of ``E`` and pivot
   bases of its left/right kernels) and the algebraic block is eliminated by
   an exact Schur complement.  Index 1 <=> that block is invertible; index
   > 1 is refused (``index_gt_1``; the propagator would carry delta'), a
   singular pencil (``det K(s) == 0``) is refused (``singular_pencil``).
4. Exact Kalman reduction (controllable, then observable subspace by exact
   Krylov ranks over ``F``: no rank tolerance) gives a minimal global
   realization and, for every entry ``G_ij``, its own minimal SISO
   realization (``entry_states[i, j]`` states).
5. Rounded once to double precision, then balanced by an exact power-of-two
   diagonal similarity (``scipy.linalg.matrix_balance``, Parlett-Reinsch):
   ``A_bal = T^{-1} A T``, ``B_bal = T^{-1} B``, ``C_bal = C T``.
6. Model reduction: every per-entry realization (and every
   conductance kernel, ``realize_kernel``) is balanced-truncated
   (``balanced_truncation``) to the fewest states whose H-infinity error
   bound ``2 sum(discarded Hankel singular values)`` is at most ``tol``
   times the largest Hankel singular value (itself at most the entry's
   H-infinity norm), ``tol = truncation_tol()``: ``TRUNCATION_TOL_DEFAULT``
   (``1e-12``, 1 % of the pipeline's default accuracy target ``1e-10``),
   ``DAEDALUS_REALIZATION_TRUNCATION`` (``0`` / ``off`` /
   ``exact``: none), or ``truncation_scope(tol)``; ``realize_K(...,
   truncation=)`` / ``realize_kernel(..., truncation=)`` override it.  An
   entry with nothing to discard is returned as it was (the same object:
   minimal, well-conditioned realizations keep every number bit for bit).
   The Gramian factors are computed directly (Hammarling; an
   eigendecomposition of a computed Gramian loses the small Hankel singular
   values of graded Pade entries), and every truncation is checked a
   posteriori on a frequency grid and retried with more states (then
   refused) when its sampled error exceeds the tolerance.  Exact Kalman
   minimality (step 4) removes only states that are exactly uncontrollable
   or unobservable; the Pade blocks of a delay inside a feedback loop leave
   many that are numerically negligible (Hankel singular values 1e-11 ..
   1e-17 of the largest at Pade-16), which is where the state-space route's
   cost goes.  The global ``A, B, C`` (Hurwitz test, exact delay tree,
   stationary covariance) are never truncated.  ``Realization.truncation``
   records the tolerance and, per entry, the states before / after, the
   Hankel singular values, the bound and the sampled error.

Hurwitz property of ``A`` (reported only, not used for routing): exact
Routh test on the exact characteristic polynomial (no roots), plus the
Lyapunov margin ``A X + X A^H = -I`` (``X > 0`` <=> Hurwitz; Bartels-Stewart
on a Schur form, no eigenvalue-gap sensitivity).

Why exact at the parameter point.  The state-space integrator needs numbers at ``num_params``;
doing the algebra over ``QQ`` and rounding only the final matrices makes
``D`` exact (it is compared exactly with the pipeline's ``D_delta``), makes
every rank decision (index, minimality) tolerance-free, and recovers the
kernel rates by exact factorization even when the frequency image was
expanded (e.g. ``1 + 2 tau s + tau^2 s^2``).  The algebra is generic over the
field ``F``; the symbolic route (``F = QQ(params)``, wanted later for a
symbolic one-loop stage) needs only a different conversion in step 1.

Interface contract (consumed by the state-space subset integrator)
---------------------------------------------------------------------
``Realization`` (frozen):

* ``A`` (N x N), ``B`` (N x nf), ``C`` (nf x N), ``D`` (nf x nf):
  float64 (complex128 only when the model has complex coefficients),
  balanced, read-only.  ``G[p, r] = C[p, :] (sI - A)^{-1} B[:, r] + D[p, r]``
  where ``p`` indexes ``phys_names`` (the columns of ``K_ft``) and ``r``
  ``resp_names`` (the rows of ``K_ft``) -- the ``(pi, ri)`` of Phase J's
  ``EdgeModeSum``.  ``N == 0`` means ``G`` is the constant ``D``.
* ``D`` is the delta(t) coefficient, ``D_delta`` of the pipeline; the
  equal-time (Ito) convention at ``t = 0`` is the caller's business:
  ``impulse(t)`` returns the smooth part ``C e^{At} B`` for ``t >= 0`` and
  ``0`` for ``t < 0``.
* ``entries[(p, r)]``: ``EntryRealization`` (``a``, ``b``, ``c``, ``d``,
  ``n_states``), the minimal SISO realization of ``G[p, r]``, balanced,
  then balanced-truncated to the tolerance in effect (step 6; within that
  relative H-infinity tolerance of the exact entry).  ``entry_states`` is
  the nf x nf array of their state counts.
* ``balance``: the diagonal ``T`` (``x_unbalanced = T x``).
* ``cls``: ``'A'`` first order (``K1`` invertible), ``'B'`` index-1
  descriptor, ``'C'`` rational kernels (``descriptor`` flags whether the
  pencil was singular too).  ``provenance``: one line per decision.
* ``hurwitz`` (True / False / None), ``lyapunov_margin`` and
  ``conditioning`` (norms of A before/after balancing, range of T, ...).
* ``exact``: the exact matrices over ``F`` (Sage) before rounding AND before
  balancing (``A_bal = T^{-1} A_exact T``, ``B_bal = T^{-1} B_exact``,
  ``C_bal = C_exact T``), for references and later symbolic work; excluded
  from comparisons.  The per-entry realizations carry their own balancing,
  not exposed.
* ``truncation``: the record of step 6 (``tol``, ``n_truncated``,
  ``states_before`` / ``states_after``, ``max_bound_rel``, ``note`` and
  ``entries[(p, r)]``: ``n``, ``r``, ``hsv``, ``bound_rel``,
  ``sampled_err_rel``, ``err_rel``, ``kept``); excluded from comparisons.
* ``phys_names`` / ``resp_names`` come from ``ring_gen_names`` (response
  fields first, then physical fields, ``nf`` each); the state basis after the
  Kalman reduction is not tied to fields.
* methods: ``transfer(s)``, ``transfer_omega(omega)``, ``impulse(t)``,
  ``transfer_entry``, ``impulse_entry``, ``stationary_covariance(Q)``,
  ``schur()`` (complex Schur basis ``A = U T U^H``: ``T, B U, C U``).

Numerical notes for consumers (measured on rational test models): do not
exponentiate through a dense Schur form of a graded realization (5e-13 on
the balanced Pade-24 loop) and do not trust scipy's ``expm`` on
near-repeated eigenvalues of a triangular matrix (1.4e-9 at a gap of 1e-8);
``expm_structured`` uses the reset-diagonal triangular Pade on a triangular
``A`` and scipy's ``expm`` (no Schur step) otherwise; its accuracy on a
non-triangular ``A`` depends on the basis, which is why irreducible factors
get Schwarz rather than companion blocks.  A plain Bartels-Stewart Lyapunov
solve loses up to 3e-13 on graded realizations; ``lyapunov_refined``
restores ~1e-15 by residual refinement.

A model without a realization raises ``RealizationRefused`` with a
machine-readable ``reason`` (``'non_rational'``, ``'unbound_symbol'``,
``'spatial'``, ``'improper_entry'``, ``'singular_pencil'``,
``'index_gt_1'``, ``'no_kernel'``, ``'ill_conditioned'``: exactly Hurwitz
but not numerically after rounding, i.e. graded beyond double precision);
the pole engine stays the fallback.
``try_realize`` returns ``(realization, None)`` or ``(None, refusal)``.

``verify_against_K`` evaluates ``K(s)^{-1}`` independently (Sage
``ComplexField``, ~50 digits, from the symbolic ``K_ft``) at random points
on and off the imaginary axis and reports the entrywise error of
``transfer(s)`` (and of the exact matrices) scaled by each entry's norm.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

import numpy as np
import scipy.linalg as sl
from engine.errors import DaedalusError

__all__ = [
    'Realization', 'EntryRealization', 'RealizationRefused', 'realize',
    'realize_K', 'try_realize', 'verify_against_K', 'compare_D',
    'expm_upper_triangular', 'expm_structured', 'lyapunov_refined',
    'balanced_truncation', 'hankel_singular_values', 'truncate_entry',
    'truncation_tol', 'truncation_scope', 'TRUNCATION_TOL_DEFAULT',
]

_REASONS = ('non_rational', 'unbound_symbol', 'spatial', 'improper_entry',
            'singular_pencil', 'index_gt_1', 'no_kernel', 'ill_conditioned',
            # realize_kernel
            'no_kernel_image', 'kernel_not_strictly_proper',
            'kernel_not_hurwitz', 'kernel_shifted_step')


class RealizationRefused(DaedalusError, ValueError):
    """The propagator has no (supported) finite state-space realization.

    ``reason`` is one of ``non_rational``, ``unbound_symbol``, ``spatial``,
    ``improper_entry``, ``singular_pencil``, ``index_gt_1``, ``no_kernel``,
    ``ill_conditioned`` (exactly Hurwitz, not numerically once rounded);
    for a conductance kernel (``realize_kernel``) also ``no_kernel_image``,
    ``kernel_not_strictly_proper``, ``kernel_not_hurwitz``,
    ``kernel_shifted_step``.
    """

    def __init__(self, reason, message):
        assert reason in _REASONS, reason
        self.reason = reason
        super().__init__(f'[{reason}] {message}')


def _phi1(z):
    """``(e^z - 1)/z`` for ``Re z <= 0`` without cancellation."""
    if abs(z) < 1e-3:
        return 1.0 + z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z / 120.0)))
    x, y = z.real, z.imag
    sy = np.sin(0.5 * y)
    return complex(np.expm1(x) * np.cos(y) - 2.0 * sy * sy, np.exp(x) * np.sin(y)) / z


def _dd2(a, b):
    """``(e^b - e^a)/(b - a)``, anchored at the node with the larger real part."""
    if a.real >= b.real:
        return np.exp(a) * _phi1(b - a)
    return np.exp(b) * _phi1(a - b)


_PADE13 = (64764752532480000., 32382376266240000., 7771770303897600.,
           1187353796428800., 129060195264000., 10559470521600.,
           670442572800., 33522128640., 1323241920., 40840800., 960960.,
           16380., 182., 1.)
_THETA13 = 5.371920351148152


def _pade13_expm_triangular(M):
    """Pade-13 scaling and squaring of an upper-triangular ``M`` with the
    diagonal and first superdiagonal reset exactly after every squaring
    (Higham 2005 scaling ``||M||_1 <= theta_13``; over-scaling is harmless
    here because the reset stops the squaring error growth)."""
    n = M.shape[0]
    cplx = np.iscomplexobj(M)
    dt = complex if cplx else float
    nrm1 = lambda X: float(np.abs(X).sum(axis=0).max())
    I_ = np.eye(n, dtype=dt)
    eta = nrm1(M)
    s = max(0, int(np.ceil(np.log2(eta / _THETA13)))) if eta > 0 else 0
    X = M * 0.5 ** s
    X2 = X @ X
    X4 = X2 @ X2
    X6 = X4 @ X2
    b = _PADE13
    U = X @ (X6 @ (b[13] * X6 + b[11] * X4 + b[9] * X2)
             + b[7] * X6 + b[5] * X4 + b[3] * X2 + b[1] * I_)
    V = (X6 @ (b[12] * X6 + b[10] * X4 + b[8] * X2)
         + b[6] * X6 + b[4] * X4 + b[2] * X2 + b[0] * I_)
    d = np.diag(M).astype(complex)
    sup = np.diag(M, 1).astype(complex)
    E = np.triu(sl.solve_triangular(np.triu(V - U), np.triu(V + U), lower=False)).astype(complex)
    for it in range(s + 1):
        if it:
            E = np.triu(E @ E)
        h = 0.5 ** (s - it)
        for j in range(n):
            E[j, j] = np.exp(d[j] * h)
        for j in range(n - 1):
            E[j, j + 1] = sup[j] * h * _dd2(d[j] * h, d[j + 1] * h)
    return E if cplx else E.real


def expm_upper_triangular(T, t=1.0):
    """``expm(T t)`` for upper-triangular ``T``: Pade-13 scaling and squaring
    with the diagonal and the first superdiagonal reset to their exact values
    (``e^{t_jj}``, ``t_{j,j+1} exp[t_jj, t_{j+1,j+1}]``) after every squaring
    (Al-Mohy & Higham 2009, triangular case).  Without the reset scipy's
    ``expm`` loses ~``eps / gap`` on near-repeated eigenvalues (1.4e-9 at a
    gap of 1e-8 in a 2x2 Jordan-like block)."""
    T = np.asarray(T)
    if T.shape[0] == 0:
        return np.zeros((0, 0), dtype=T.dtype)
    return _pade13_expm_triangular(np.triu(T) * t)


def expm_structured(A, t=1.0):
    """``expm(A t)`` on ``A`` itself (no Schur step, whose dense backward error
    hurts graded realizations: 5e-13 on the balanced Pade-24 loop):
    ``expm_upper_triangular`` when ``A`` is upper (or, transposed, lower)
    triangular (Jordan-like chains, near-repeated rates), else scipy's
    ``expm`` (Al-Mohy & Higham 2009 with its backward-error scaling)."""
    A = np.asarray(A)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=A.dtype)
    if not np.any(np.tril(A, -1)):
        return expm_upper_triangular(A, t)
    if not np.any(np.triu(A, 1)):
        return expm_upper_triangular(A.T, t).T
    return sl.expm(A * t)


def lyapunov_refined(A, Qm, max_steps=4):
    """``X`` with ``A X + X A^H + Qm = 0``: Bartels-Stewart (scipy) followed by
    iterative refinement on the double-precision residual, keeping the
    iterate with the smallest residual (refinement in working precision can
    drift once the residual is at rounding level).  The plain solve loses
    8e-14 / 3e-13 on the balanced Pade-8 / Pade-16 loops (the Schur step's
    dense backward error on a graded matrix); refinement restores ~1e-15.
    Assumes a stable ``A`` (callers check ``hurwitz``)."""
    A = np.asarray(A)
    if A.shape[0] == 0:
        return np.zeros((0, 0), dtype=A.dtype)
    X = sl.solve_continuous_lyapunov(A, -Qm)

    def resid(X):
        return A @ X + X @ A.conj().T + Qm
    R = resid(X)
    best, rbest = X, float(np.abs(R).max())
    for _ in range(max_steps):
        X = X + sl.solve_continuous_lyapunov(A, -R)
        R = resid(X)
        r = float(np.abs(R).max())
        if not r < rbest:
            break
        best, rbest = X, r
    return best


@dataclass(frozen=True)
class EntryRealization:
    """Minimal SISO realization of one entry: ``g(s) = c (sI - a)^{-1} b + d``."""
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: Any                      # float for real models, complex otherwise
    n_states: int

    def __post_init__(self):
        for nm in ('a', 'b', 'c'):
            getattr(self, nm).setflags(write=False)

    def transfer(self, s):
        if self.n_states == 0:
            return self.d
        x = np.linalg.solve(s * np.eye(self.n_states) - self.a, self.b)
        return complex(self.c @ x) + self.d

    def impulse(self, t):
        if t < 0 or self.n_states == 0:
            return 0.0
        v = (self.c @ expm_structured(self.a, t) @ self.b).item()
        return v.real if not np.iscomplexobj(self.a) else v


@dataclass(frozen=True)
class Realization:
    """``G(s) = C (sI - A)^{-1} B + D`` (module docstring: the contract)."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    cls: str
    descriptor: bool
    n_states: int
    entry_states: np.ndarray
    entries: dict
    phys_names: tuple
    resp_names: tuple
    balance: np.ndarray
    hurwitz: Optional[bool]
    lyapunov_margin: Optional[float]
    conditioning: dict
    provenance: tuple
    field_name: str
    exact: Any = field(default=None, repr=False, compare=False)
    truncation: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for nm in ('A', 'B', 'C', 'D', 'entry_states', 'balance'):
            arr = getattr(self, nm)
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)

    @property
    def nf(self):
        return self.D.shape[0]

    def transfer(self, s):
        """``C (sI - A)^{-1} B + D`` at complex ``s`` (nf x nf array)."""
        if self.n_states == 0:
            return np.array(self.D, dtype=complex)
        X = np.linalg.solve(s * np.eye(self.n_states) - self.A, self.B)
        return self.C @ X + self.D

    def transfer_omega(self, omega):
        """``G_ft(omega) = transfer(i omega)`` (the pipeline's convention)."""
        return self.transfer(1j * omega)

    def impulse(self, t):
        """Smooth part ``C e^{At} B`` for ``t >= 0``, zero for ``t < 0``
        (``expm_structured``)."""
        if t < 0 or self.n_states == 0:
            return np.zeros(self.D.shape, dtype=self.A.dtype if self.n_states else float)
        G = self.C @ expm_structured(self.A, t) @ self.B
        return G.real if not np.iscomplexobj(self.A) else G

    def transfer_entry(self, p, r, s):
        return self.entries[(p, r)].transfer(s)

    def impulse_entry(self, p, r, t):
        return self.entries[(p, r)].impulse(t)

    def stationary_covariance(self, Q):
        """Smooth equal-time covariance ``C X C^H`` of ``u = G * y`` for white
        sources ``<y y^H> = Q delta`` (``Q``: nf x nf in response order),
        ``A X + X A^H + B Q B^H = 0``.  The ``D Q D^H delta(0)`` part of a
        descriptor is not included.  Bartels-Stewart plus residual
        refinement (``lyapunov_refined``)."""
        if self.n_states == 0:
            return np.zeros(self.D.shape)
        if self.hurwitz is False or (self.hurwitz is None and not (self.lyapunov_margin or 0) > 0):
            raise ValueError('no stationary covariance: A is not Hurwitz (an '
                             'unstable or marginal saddle)')
        BQB = self.B @ np.asarray(Q) @ self.B.conj().T
        return self.C @ lyapunov_refined(self.A, BQB) @ self.C.conj().T

    def schur(self):
        """Complex Schur basis: ``(T, B_s, C_s)`` with ``A = U T U^H``,
        ``B_s = U^H B``, ``C_s = C U`` (``T`` upper triangular)."""
        T, U = sl.schur(np.asarray(self.A, dtype=complex), output='complex')
        return T, U.conj().T @ self.B, self.C @ U


# ─── balanced truncation (model reduction) ────────────────────────────────

#: Default relative H-infinity tolerance of the balanced truncation of every
#: per-entry realization and conductance kernel: states are discarded only
#: while ``2 * sum(discarded Hankel singular values) <= tol * sigma_1``.  1 %
#: of the pipeline's default accuracy target (``accuracy``, G6: 1e-10), so the
#: truncation never takes a visible share of it (the observable moved by
#: 0.01 .. 0.8 of the per-entry tolerance on the benchmarks: a
#: multi-population model at 1e-10 moved by 8e-11).
TRUNCATION_TOL_DEFAULT = 1e-12
#: ``DAEDALUS_REALIZATION_TRUNCATION``: a tolerance, or ``0`` / ``off`` /
#: ``exact`` for none (the exact minimal realizations, as before the balanced truncation).
TRUNCATION_ENV = 'DAEDALUS_REALIZATION_TRUNCATION'

# the innermost truncation_scope (process-wide, not thread-local: the
# state-space route builds conductance-kernel realizations lazily, possibly
# in a worker thread of the same run)
_TRUNC_LOCK = threading.RLock()
_TRUNC_STACK = []                # [(token, tol)]: the last entry is in effect
_ENV_WARNED = set()


def _env_truncation_tol(environ=None):
    env = os.environ if environ is None else environ
    v = str(env.get(TRUNCATION_ENV, '') or '').strip().lower()
    if v in ('', 'default'):
        return TRUNCATION_TOL_DEFAULT
    if v in ('0', 'off', 'exact', 'none', 'false', 'no'):
        return 0.0
    try:
        tol = float(v)
    except ValueError:
        raise ValueError(f'{TRUNCATION_ENV}={v!r}: expected a tolerance (e.g. 1e-12), '
                         "or 0 / 'off' / 'exact' for none") from None
    if not (tol >= 0.0 and np.isfinite(tol)):
        raise ValueError(f'{TRUNCATION_ENV}={v!r}: the tolerance must be finite and >= 0')
    return tol


def truncation_tol():
    """The relative H-infinity tolerance of the balanced truncation in effect:
    the innermost open ``truncation_scope`` (``None``: none), else
    ``DAEDALUS_REALIZATION_TRUNCATION``, else ``TRUNCATION_TOL_DEFAULT``;
    ``0`` means exact (no truncation).  A malformed environment value warns
    (once per value) and the default is used: a ``ValueError`` here would be
    swallowed by the state-space route's refusal handling and silently send
    every model to the pole route."""
    with _TRUNC_LOCK:
        tol = _TRUNC_STACK[-1][1] if _TRUNC_STACK else None
    if tol is not None:
        return tol
    try:
        return _env_truncation_tol()
    except ValueError as exc:
        v = os.environ.get(TRUNCATION_ENV)
        if v not in _ENV_WARNED:
            _ENV_WARNED.add(v)
            import warnings
            warnings.warn(f'{exc}; using the default {TRUNCATION_TOL_DEFAULT:g}',
                          UserWarning, stacklevel=2)
        return TRUNCATION_TOL_DEFAULT


def _clear_consumer_caches():
    """Drop the state-space route's realization cache (keyed by the model and
    the parameters, not by the tolerance), if that module is loaded."""
    import sys
    ss = sys.modules.get('engine.integration.time_domain.statespace')
    if ss is not None:
        ss._clear_caches()


class truncation_scope:
    """``with truncation_scope(tol):`` -- realizations built inside use the
    relative H-infinity tolerance ``tol`` (``0``: exact; ``None``: the
    environment / default).  Process-wide (the realizations of one run may
    be built in its worker threads): the innermost open scope is in effect,
    and a scope's exit removes only its own entry (interleaved exits of
    scopes opened in different threads leave the others in force).  The
    state-space route caches realizations by model and parameters, not by
    tolerance, so a change of the tolerance in effect clears that cache (on
    entry and on exit); a change of ``DAEDALUS_REALIZATION_TRUNCATION``
    after realizations were built does not (set it before the run, or clear
    ``engine.integration.time_domain.statespace._clear_caches()``)."""

    def __init__(self, tol):
        self.tol = None if tol is None else float(tol)
        if self.tol is not None and not (self.tol >= 0.0 and np.isfinite(self.tol)):
            raise ValueError(f'truncation tolerance must be finite and >= 0, got {tol!r}')
        self._token = object()

    def __enter__(self):
        with _TRUNC_LOCK:
            before = truncation_tol()
            _TRUNC_STACK.append((self._token, self.tol))
            changed = truncation_tol() != before
            if changed:
                _clear_consumer_caches()
        return self

    def __exit__(self, *exc):
        with _TRUNC_LOCK:
            before = truncation_tol()
            # its own entry (scopes of other threads may have been entered
            # after it and still be open)
            for i in range(len(_TRUNC_STACK) - 1, -1, -1):
                if _TRUNC_STACK[i][0] is self._token:
                    del _TRUNC_STACK[i]
                    break
            changed = truncation_tol() != before
            if changed:
                _clear_consumer_caches()
        return False


def _lyap_factor(a, b):
    """A real (for real ``a``, ``b``) or complex factor ``Z`` (n x n) with
    ``Z Z^H = P``, ``a P + P a^H + b b^H = 0`` (``a`` Hurwitz), computed
    directly (Hammarling 1982, on the complex Schur form) instead of from
    ``P``: the small singular values of ``Z`` keep their relative accuracy,
    which an eigendecomposition of a computed ``P`` loses below
    ``sqrt(eps) ||P||`` (seen: the Pade-16 entries of a two-neuron loop, whose
    truncation through ``eigh(P)`` failed the a posteriori check by 1e-7).
    For a real system ``P = Re(Z) Re(Z)^T + Im(Z) Im(Z)^T``, so a real
    triangular factor comes from the QR decomposition of ``[Re Z, Im Z]^T``."""
    T, Q = sl.schur(np.asarray(a, dtype=complex), output='complex')
    n = T.shape[0]
    bt = Q.conj().T @ np.asarray(b, dtype=complex).reshape(n)
    U = np.zeros((n, n), dtype=complex)
    for k in range(n - 1, -1, -1):
        al, be = T[k, k], bt[k]
        if be == 0:
            continue
        nu = abs(be) / np.sqrt(-2.0 * al.real)
        U[k, k] = nu
        if k:
            rhs = -(T[:k, k] * (nu * nu) + bt[:k] * np.conj(be)) / nu
            u = sl.solve_triangular(T[:k, :k] + np.conj(al) * np.eye(k), rhs)
            U[:k, k] = u
            bt[:k] = bt[:k] - u * (be / nu)
    Z = Q @ U
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        return Z
    Rq = np.linalg.qr(np.hstack([Z.real, Z.imag]).T, mode='r')
    return Rq.T


def _equalize_bc(b, c):
    """``b sigma, c / sigma`` with the power of two ``sigma`` closest to
    ``sqrt(|c| / |b|)`` (exact; the transfer function is unchanged)."""
    nb, nc = float(np.abs(b).sum()), float(np.abs(c).max())
    if not (nb > 0 and nc > 0):
        return b, c
    sig = 2.0 ** round(0.5 * np.log2(nc / nb))
    return b * sig, c / sig


def _prebalance_siso(a, b, c):
    """A diagonal (power-of-two) similarity of ``(a, b, c)`` that ungrades it
    before the Gramians: the entry as given, ``a`` balanced alone, and the
    augmented matrix ``[[a, b], [c, 0]]`` balanced (``scipy.linalg.
    matrix_balance``), each with ``b`` / ``c`` equalised; the candidate with
    the smallest ``max(|a|_1, |b|_1 |c|_inf)`` is kept (the per-entry
    realizations balance ``a`` alone, which leaves ``b`` and ``c`` graded in
    opposite directions on Pade entries: |b| ~ 1e12, |c| ~ 1e-12 at n = 16,
    where the Gramians then lose everything below ~1e-16 relative).  The
    transfer function is unchanged exactly."""
    n = a.shape[0]
    cands = [(a, *_equalize_bc(b, c))]
    aug = np.zeros((n + 1, n + 1), dtype=np.result_type(a, b, c))
    aug[:n, :n], aug[:n, n], aug[n, :n] = a, b, c
    if np.all(np.isfinite(aug)):
        _x, (sa, _p) = sl.matrix_balance(a, permute=False, separate=True)
        sa = np.asarray(sa, dtype=float)
        if not np.all(sa == 1.0):
            cands.append((a * (sa[None, :] / sa[:, None]), *_equalize_bc(b / sa, c * sa)))
        _x, (sg, _p) = sl.matrix_balance(aug, permute=False, separate=True)
        sg = np.asarray(sg, dtype=float)
        if not np.all(sg == 1.0):
            S, sn = sg[:n], sg[n]
            cands.append((a * (S[None, :] / S[:, None]),
                          *_equalize_bc(b * (sn / S), c * (S / sn))))

    def cost(abc):
        a_, b_, c_ = abc
        return max(float(np.abs(a_).sum(axis=0).max()),
                   float(np.abs(b_).sum()) * float(np.abs(c_).max()))
    return min(cands, key=cost)


def hankel_singular_values(a, b, c):
    """Hankel singular values of the stable SISO ``(a, b, c)``, descending:
    ``sqrt(eig(P Q))`` from the controllability / observability Gramians,
    as the singular values of ``L^H R`` (``P = R R^H``, ``Q = L L^H``, the
    factors computed directly: ``_lyap_factor``), after a diagonal power-of-two
    similarity (``_prebalance_siso``).  Returns ``(sigma, R, L, U, Vh, a_s,
    b_s, c_s)`` (``L^H R = U diag(sigma) Vh``; the factors belong to the
    similar realization ``(a_s, b_s, c_s)``)."""
    a, b, c = _prebalance_siso(np.asarray(a), np.asarray(b).reshape(-1),
                               np.asarray(c).reshape(-1))
    Rf = _lyap_factor(a, b)
    Lf = _lyap_factor(a.conj().T, c.conj())
    U, sv, Vh = np.linalg.svd(Lf.conj().T @ Rf)
    return sv, Rf, Lf, U, Vh, a, b, c


def _check_frequencies(lam):
    """Frequencies ``w >= 0`` for the a posteriori check of a truncation:
    log-spaced from (slowest decay rate) / 100 to 100 max|lambda|, plus nodes
    around every underdamped eigenvalue and ``w = 0``."""
    rates = np.abs(lam.real)
    lo = max(float(rates.min()) / 100.0, 1e-300)
    hi = 100.0 * float(np.max(np.abs(lam)))
    dec = max(1.0, np.log10(hi / lo))
    nodes = [np.zeros(1), np.geomspace(lo, hi, int(30 * dec) + 2)]
    for z in lam:
        if abs(z.imag) > 0.1 * abs(z.real):
            nodes.append(np.abs(abs(z.imag) + abs(z.real) * np.linspace(-3, 3, 13)))
    return np.unique(np.concatenate(nodes))


def _siso_transfer(a, b, c, s):
    """``c (s I - a)^{-1} b`` at the points ``s`` (vectorized)."""
    n = a.shape[0]
    M = s[:, None, None] * np.eye(n) - a
    x = np.linalg.solve(M, np.broadcast_to(b.reshape(n, 1), (s.size, n, 1)).astype(M.dtype))
    return (x[:, :, 0] @ c.reshape(n)).reshape(-1)


def balanced_truncation(a, b, c, tol, *, check=True):
    """Balanced truncation of the stable SISO realization ``g(s) = c (sI -
    a)^{-1} b`` (square-root method on directly computed Gramian factors:
    Moore 1981, Tombs & Postlethwaite 1987, Hammarling 1982) to the fewest
    states ``r`` with the H-infinity error bound

        || g - g_r ||_inf <= 2 * sum_{i > r} sigma_i <= tol * sigma_1

    (``sigma_i`` the Hankel singular values; ``sigma_1 <= ||g||_inf``, so the
    bound is also relative to the entry's H-infinity norm; the recorded bound
    adds ``4 n eps sigma_1`` for the rounding of the computed ``sigma_i``).
    The truncated
    realization is balanced, stable, real for a real ``(a, b, c)``.  States
    are kept (``r`` raised) where two Hankel singular values at the cut are
    equal to ``1e-8`` (BT's stability needs a strict gap).  ``check``: the
    error ``|g - g_r|`` is sampled on a frequency grid (``_check_frequencies``,
    a lower bound of the true H-infinity error); a truncation whose sampled
    error exceeds ``tol * sigma_1`` (a positive ``tol`` below
    ``TRUNCATION_TOL_MIN`` is raised to it; rounding: the computed Hankel singular
    values near the cut carry absolute errors ~ eps sigma_1) is retried with
    one more state, at most ``_TRUNC_RETRIES`` times, then rejected.  Returns
    ``(a_r, b_r, c_r, info)`` -- ``info``: ``n``, ``r``, ``hsv``, ``bound``
    (absolute) and ``bound_rel`` (``2 sum(discarded) / sigma_1``),
    ``sampled_err_rel``, ``err_rel`` (the larger of the two: the error this
    truncation is recorded with), ``kept`` (why nothing was discarded) --
    or ``(a, b, c, info)`` unchanged (the same arrays) when nothing can be
    discarded or the truncation is rejected."""
    a = np.asarray(a)
    n = a.shape[0]
    if tol > 0:
        tol = max(float(tol), TRUNCATION_TOL_MIN)
    info = {'n': n, 'r': n, 'tol': float(tol), 'hsv': (), 'bound': 0.0,
            'bound_rel': 0.0, 'sampled_err_rel': None, 'err_rel': 0.0, 'kept': None}
    if n < 2 or not tol > 0:
        info['kept'] = 'trivial' if n < 2 else 'tol_zero'
        return a, b, c, info
    lam = np.linalg.eigvals(a)
    if not np.all(lam.real < 0):
        info['kept'] = 'not_hurwitz'
        return a, b, c, info
    try:
        sv, Rf, Lf, U, Vh, ae, be, ce = hankel_singular_values(a, b, c)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        info['kept'] = f'gramians_failed ({type(exc).__name__})'
        return a, b, c, info
    info['hsv'] = tuple(float(x) for x in sv)
    s1 = float(sv[0]) if sv.size else 0.0
    if not (s1 > 0 and np.all(np.isfinite(sv))):
        info['kept'] = 'gramians_failed'
        return a, b, c, info
    real = not (np.iscomplexobj(a) or np.iscomplexobj(b) or np.iscomplexobj(c))
    tail = 2.0 * np.concatenate([np.cumsum(sv[::-1])[::-1], [0.0]])
    # the recorded bound adds the rounding of the computed sigma_i: kept <= tol
    rnd = 4.0 * n * np.finfo(float).eps * s1
    r = max(int(np.argmax(tail + rnd <= tol * s1)), 1)
    while r < n and sv[r - 1] - sv[r] <= 1e-8 * sv[r - 1]:
        r += 1
    if r >= n:
        info['kept'] = 'no_small_hsv'
        return a, b, c, info
    s = None
    if check:
        w = _check_frequencies(lam)
        s = 1j * w if real else 1j * np.concatenate([w, -w[w > 0]])
        g = _siso_transfer(ae, be, ce, s)
    why = None
    for _attempt in range(_TRUNC_RETRIES + 1):
        if r >= n:
            break
        srt = np.sqrt(sv[:r])
        T = (Rf @ Vh[:r].conj().T) / srt
        W = (Lf @ U[:, :r]) / srt
        ar, br, cr = W.conj().T @ ae @ T, W.conj().T @ be, ce @ T
        if real:
            ar, br, cr = ar.real, br.real, cr.real
        if not (np.all(np.isfinite(ar)) and np.all(np.linalg.eigvals(ar).real < 0)):
            why = 'truncated_not_hurwitz'
            r += 1
            continue
        err = 0.0
        if check:
            err = float(np.max(np.abs(g - _siso_transfer(ar, br, cr, s)))) / s1
            if not err <= tol * (1.0 + 1e-9):
                why = f'check_failed (sampled {err:.3g} > tol {tol:.3g})'
                r += 1
                continue
        # the computed Hankel singular values carry absolute errors of a few
        # eps sigma_1 (backward-stable SVD of L^H R): added to the bound
        bound = float(tail[r]) + rnd
        info.update(r=r, bound=bound, bound_rel=bound / s1,
                    sampled_err_rel=err if check else None,
                    err_rel=max(bound / s1, err), kept=None)
        return (np.ascontiguousarray(ar), np.ascontiguousarray(br),
                np.ascontiguousarray(cr), info)
    info.update(r=n, kept=why or 'no_small_hsv')
    return a, b, c, info


#: Retries (one more state each) of a truncation that fails its check.
_TRUNC_RETRIES = 3
#: Smallest relative tolerance honoured (a smaller positive one is raised to
#: it): the Hankel singular values, the bound and the sampled check carry
#: rounding errors of a few eps sigma_1.
TRUNCATION_TOL_MIN = 1e-13


def truncate_entry(ent, tol, *, check=True):
    """``(EntryRealization, info)``: ``ent`` balanced-truncated to the relative
    H-infinity tolerance ``tol`` (``balanced_truncation``), or ``ent`` itself
    (the same object) when no state can be discarded."""
    a_r, b_r, c_r, info = balanced_truncation(ent.a, ent.b, ent.c, tol, check=check)
    if info['r'] >= ent.n_states:
        return ent, info
    k = a_r.shape[0]
    return EntryRealization(a=np.ascontiguousarray(a_r), b=b_r.reshape(k),
                            c=c_r.reshape(k), d=ent.d, n_states=k), info


def _truncation_summary(tol, recs):
    """The record of a realization's truncation (``Realization.truncation``)."""
    cut = {k: v for k, v in recs.items() if v['r'] < v['n']}
    before = sum(v['n'] for v in recs.values())
    after = sum(v['r'] for v in recs.values())
    worst = max((v['bound_rel'] for v in cut.values()), default=0.0)
    note = (f'balanced truncation (relative H-infinity tolerance {tol:g}): '
            f'{len(cut)} entr{"y" if len(cut) == 1 else "ies"} truncated, '
            f'per-entry states {before} -> {after} in total; '
            f'largest bound 2*sum(discarded HSV)/sigma_1 = {worst:.3g}')
    return {'tol': float(tol), 'entries': recs, 'n_truncated': len(cut),
            'states_before': before, 'states_after': after,
            'max_bound_rel': worst, 'note': note}


# ─── exact helpers (Sage) ─────────────────────────────────────────────────

def _sage():
    import sage.all as S
    return S


def _gauss_field():
    S = _sage()
    return S.QuadraticField(-1, 'I')


def _exact_number(v, allow_complex=True):
    """Exact rational (or Gaussian rational) of a parameter value.  Floats are
    taken at their exact binary value; Sage exact rationals stay exact."""
    S = _sage()
    from sage.rings.rational import Rational
    from sage.rings.integer import Integer
    if isinstance(v, (int, Integer, Rational)) and not isinstance(v, bool):
        return S.QQ(v)
    try:
        sv = S.SR(v)
        if sv.is_numeric():
            po = sv.pyobject()
            if isinstance(po, (Rational, Integer, int)):
                return S.QQ(po)
    except Exception:
        pass
    c = complex(v)

    def q(x):
        fr = Fraction(float(x))
        return S.QQ(fr.numerator) / S.QQ(fr.denominator)
    if c.imag == 0.0:
        return q(c.real)
    if not allow_complex:
        raise ValueError(f'complex parameter value {v!r}')
    return S.SR(q(c.real)) + S.I * S.SR(q(c.imag))


def _sr_exact_number(v):
    """An exact SR number (rational, or Gaussian rational) as a field element;
    raises for anything else (a transcendental constant, a float)."""
    S = _sage()
    v = S.SR(v)
    if not v.is_numeric():
        raise ValueError('not an exact number')
    po = v.pyobject()
    par = getattr(po, 'parent', None)
    if par is None or not par().is_exact():
        raise ValueError('inexact number')
    try:
        return S.QQ(po)
    except (TypeError, ValueError):
        G = _gauss_field()
        return G(S.QQ(v.real_part())) + G(S.QQ(v.imag_part())) * G.gen()


def _exact_subs(num_params):
    S = _sage()
    return {S.SR(k): S.SR(_exact_number(v)) for k, v in (num_params or {}).items()}


def _round(x, prec=53):
    """Correctly rounded complex double of an exact field element."""
    S = _sage()
    try:
        return complex(float(S.RealField(prec)(x)), 0.0)
    except (TypeError, ValueError):
        z = S.ComplexField(prec)(x)
        return complex(float(z.real()), float(z.imag()))


def _to_numpy(M, cplx):
    if M.nrows() == 0 or M.ncols() == 0:
        return np.zeros((M.nrows(), M.ncols()), dtype=complex if cplx else float)
    arr = np.array([[_round(M[i, j]) for j in range(M.ncols())]
                    for i in range(M.nrows())], dtype=complex)
    return arr if cplx else arr.real.copy()


def _exactify(e):
    """Replace every inexact number in an SR expression by the exact rational
    of its binary value (a float literal in a kernel string, e.g. a Pade
    coefficient, means that double, not the simplest nearby rational)."""
    S = _sage()
    if not e.variables() and not e.is_numeric():
        # a constant that is not a plain number (e^(-13/10), sqrt(2), ...):
        # its correctly rounded double, as the float pipeline sees it (its
        # literals made exact first, so the 200-bit evaluation is not
        # truncated to 53 bits by a float operand)
        ops = e.operands()
        if ops:
            e = e.operator()(*[_exactify(o) for o in ops])
        try:
            z = S.ComplexField(200)(e)
        except (TypeError, ValueError):
            return e
        re = S.RealField(53)(z.real()).exact_rational()
        im = S.RealField(53)(z.imag()).exact_rational()
        return S.SR(re) if im == 0 else S.SR(re) + S.I * S.SR(im)
    if e.is_numeric():
        po = e.pyobject()
        par = getattr(po, 'parent', None)
        if isinstance(po, int) or (par is not None and par().is_exact()):
            return e
        z = complex(po)                                   # the double the pipeline uses
        re, im = _exact_number(z.real), _exact_number(z.imag)
        return S.SR(re) if im == 0 else S.SR(re) + S.I * S.SR(im)
    ops = e.operands()
    if not ops:
        return e
    return e.operator()(*[_exactify(o) for o in ops])


def _spatial_symbol(name):
    """The inert spatial operator of ``build_propagator`` (a spatial model is
    also caught from its propagator data in ``realize``)."""
    return name.startswith('Laplacian')


def _entries_to_field(K_ft, omega, subs):
    """Convert every entry of ``K_ft`` to ``F(s)``; return (entries, F, F[s], name of F)."""
    S = _sage()
    nf = K_ft.nrows()
    s = S.SR.var('ssreal')
    exprs = {}
    for i in range(nf):
        for j in range(K_ft.ncols()):
            e = S.SR(K_ft[i, j])
            if e.is_zero():
                continue
            try:
                e = _exactify(e.subs(subs)).subs({S.SR(omega): -S.I * s})
            except Exception as exc:                      # e.g. an unevaluated integral
                raise RealizationRefused(
                    'non_rational', f'K[{i},{j}] = {str(K_ft[i, j])[:120]} cannot be '
                    f'evaluated as a rational function of omega ({type(exc).__name__}: '
                    f'{str(exc)[:120]})') from exc
            free = [str(v) for v in e.variables() if str(v) != str(s)]
            if free:
                if any(_spatial_symbol(f) for f in free):
                    raise RealizationRefused(
                        'spatial', f'K[{i},{j}] carries the spatial symbol(s) '
                        f'{sorted(free)} (a per-momentum realization is not part of the realization)')
                raise RealizationRefused(
                    'unbound_symbol', f'K[{i},{j}] has symbols {sorted(free)} '
                    'without a value in num_params (or a kernel without a '
                    'frequency image)')
            exprs[(i, j)] = e
    last_err = None
    for fname in ('QQ', 'QQ[I]'):
        F = S.QQ if fname == 'QQ' else _gauss_field()
        P = S.PolynomialRing(F, str(s))
        Fs = P.fraction_field()
        out = {}
        try:
            for key, e in exprs.items():
                out[key] = _sr_to_frac(e, s, Fs, F)
            return out, F, P, fname
        except RealizationRefused:
            raise
        except Exception as exc:                          # try the next field
            last_err = (key, exc)
    key, exc = last_err
    raise RealizationRefused(
        'non_rational', f'K[{key[0]},{key[1]}] = {str(exprs[key])[:160]} is not a '
        f'rational function of omega ({type(exc).__name__}: {str(exc)[:120]})')


def _sr_to_frac(e, s, Fs, F):
    """Exact conversion of an SR rational expression in ``s`` to ``Fs``."""
    S = _sage()
    if F is S.QQ:
        return Fs(e)
    # Gaussian rationals: map I to the field generator through the
    # numerator/denominator coefficient lists.
    ee = e.rational_simplify()
    num, den = ee.numerator(), ee.denominator()
    if not (num.is_polynomial(s) and den.is_polynomial(s)):
        raise TypeError('not a rational function of s')
    P = Fs.ring()
    g = F.gen()

    def to_poly(pe):
        cs = pe.expand().coefficients(s, sparse=False)
        out = []
        for c in cs:
            c = S.SR(c)
            re, im = c.real_part(), c.imag_part()
            out.append(F(S.QQ(re)) + F(S.QQ(im)) * g)
        return P(out)
    return Fs(to_poly(num)) / Fs(to_poly(den))


def _split_entry(x, P):
    """``x = q + r/d`` with ``deg r < deg d``; return (q, r, d)."""
    num, den = x.numerator(), x.denominator()
    q, r = num.quo_rem(den)
    return q, r, den


def _routh_column(f):
    """First column ``c_1..c_{m+1}`` of the Routh array of a monic ``f``
    (exact), or None when a pivot vanishes."""
    a = list(reversed(f.list()))
    row0, row1 = a[0::2], a[1::2]
    col = [row0[0], row1[0]]
    while len(col) < f.degree() + 1:
        if row1[0] == 0:
            return None
        new = []
        for k in range(len(row0) - 1):
            nxt = row1[k + 1] if k + 1 < len(row1) else 0
            new.append(row0[k + 1] - row0[0] * nxt / row1[0])
        col.append(new[0])
        row0, row1 = row1, new
    return None if any(c == 0 for c in col) else col


def _schwarz_matrix(g, F):
    """Tridiagonal Schwarz matrix with characteristic polynomial ``g`` (monic),
    from its Routh column: ``b_1 = c_2`` on the last diagonal entry,
    ``b_k = c_{k+1} / c_{k-1}`` on the subdiagonal, ones above.  Pure field
    arithmetic (valid over QQ[I] too); None when a Routh pivot vanishes."""
    S_ = _sage()
    m = g.degree()
    col = _routh_column(g)
    if col is None:
        return None
    c = [F(1)] + list(col)
    S = S_.matrix(F, m, m)
    for i in range(m - 1):
        S[i, i + 1] = 1
    for i in range(1, m):
        S[i, i - 1] = -c[m - i + 2] / c[m - i]
    S[m - 1, m - 1] = -c[2]
    return S


_SCHWARZ_SHIFTS = (0, 1, -1, 2, -2, 3, -3)


def _schwarz_block(f, F):
    """Root-free tridiagonal (Schwarz) form similar to the companion matrix of
    the monic ``f``: ``S = T^{-1} Comp T`` with ``T`` exact (Krylov).  When a
    Routh pivot of ``f`` vanishes, the Schwarz matrix of ``g(x) = f(x + sigma)``
    for a small integer shift is used, ``S = S_g + sigma I`` (same
    characteristic polynomial, exactly).  Returns (S, T, sigma) or None
    (companion fallback).  Much closer to normal than the companion matrix:
    the balanced Pade-24 closed loop loses 3e-13 in a double-precision expm
    with companion blocks (2e-9 with complex coefficients)."""
    S_ = _sage()
    m = f.degree()
    x = f.parent().gen()
    S = None
    for sigma in _SCHWARZ_SHIFTS:
        g = f(x + sigma) if sigma else f
        Sg = _schwarz_matrix(g, F)
        if Sg is not None:
            S = Sg + sigma * S_.identity_matrix(F, m)
            break
    if S is None:
        return None
    a = f.list()
    Comp = S_.matrix(F, m, m)
    for i in range(m - 1):
        Comp[i, i + 1] = 1
    for i in range(m):
        Comp[m - 1, i] = -a[i]
    e = S_.matrix(F, m, 1)
    e[m - 1, 0] = 1
    Kc = e
    Ks = e
    vc, vs = e, e
    for _ in range(m - 1):
        vc, vs = Comp * vc, S * vs
        Kc, Ks = Kc.augment(vc), Ks.augment(vs)
    if Ks.det() == 0:                                     # (S, e_m) not controllable
        return None
    T = Kc * Ks.inverse()
    if Comp * T != T * S:                                 # never seen; stay safe
        return None
    return S, T, sigma


def _realize_proper(r, d, F):
    """Exact chain realization of the strictly proper ``r/d`` (no roots: the
    denominator is factored exactly over F).  Returns (A, b, c, blocks)."""
    S = _sage()
    lc = d.leading_coefficient()
    d = d / lc
    r = r / lc
    fac = list(d.factor())
    # deterministic order: by degree, then by the (exact) coefficients
    fac.sort(key=lambda fm: (fm[0].degree(), str(fm[0])))
    blocks = []
    for f, m in fac:
        blocks.extend([f.monic()] * int(m))
    # Newton-type division: r = c_N + f_N (c_{N-1} + f_{N-1} (... c_1))
    coefs = [None] * len(blocks)
    rem = r
    for k in range(len(blocks) - 1, -1, -1):
        rem, c_k = rem.quo_rem(blocks[k])
        coefs[k] = c_k
    if rem != 0:                                          # pragma: no cover
        raise AssertionError('Newton division left a quotient (deg r >= deg d)')
    # Block k realizes z_k = g_k / f_k(s) * z_{k-1} (z_0 = u) with the gain
    # g_k = f_k(0) (the rate for a first-order block; 1 if f_k(0) = 0): unit
    # DC gain, i.e. the natural Erlang chain (subdiagonal = rate).  The
    # Newton coefficients absorb the gains: c'_k = c_k / prod_{j<=k} g_j.
    # (Without the gains the chain is graded by rate^k and no diagonal
    # balancing recovers it: Erlang-16 lost 1e-10 in expm / Lyapunov.)
    N = sum(f.degree() for f in blocks)
    A = S.matrix(F, N, N)
    b = S.matrix(F, N, 1)
    c = S.matrix(F, 1, N)
    off = 0
    prev_first = None
    prev_cols = None
    prev_z = None
    schwarz_blocks = []
    gprod = F(1)
    for k, f in enumerate(blocks):
        m = f.degree()
        a = f.list()                                      # a[0..m], a[m] = 1
        g = a[0] if a[0] != 0 else F(1)
        gprod = gprod * g
        # block states w with companion coordinates Tb w: [z, s z, ..., s^{m-1} z]
        sch = _schwarz_block(f, F) if m >= 2 else None
        if sch is None:
            Ab = S.matrix(F, m, m)
            for i in range(m - 1):
                Ab[i, i + 1] = 1
            for i in range(m):
                Ab[m - 1, i] = -a[i]
            Tb = S.identity_matrix(F, m)
        else:
            Ab, Tb, _sigma = sch
            schwarz_blocks.append(m)
        Tbi = Tb.inverse()
        A[off:off + m, off:off + m] = Ab
        inp = Tbi[:, m - 1]                               # companion input e_m
        if prev_first is None:
            b[off:off + m, 0] = g * inp                   # the input drives block 1
        else:
            A[off:off + m, prev_cols] = g * inp * prev_z  # block k driven by z_{k-1}
        ck = coefs[k].list() + [F(0)] * m
        crow = S.matrix(F, 1, m, [ck[i] / gprod for i in range(m)])
        c[0, off:off + m] = crow * Tb
        prev_first = off
        prev_cols = list(range(off, off + m))
        prev_z = Tb[0:1, :]                               # z_k = (Tb w)_0
        off += m
    return A, b, c, (blocks, schwarz_blocks)


def _describe_blocks(blocks, schwarz=()):
    lin = [f for f in blocks if f.degree() == 1]
    hi = [f for f in blocks if f.degree() > 1]
    parts = []
    if lin:
        rates = {}
        for f in lin:
            rates[f] = rates.get(f, 0) + 1
        mult = sorted(rates.values(), reverse=True)
        parts.append(f'{len(lin)} first-order block(s) from {len(rates)} '
                     f'given rate(s)' + (f' (multiplicities {mult})' if max(mult) > 1 else ''))
    if hi:
        parts.append('irreducible block(s) of degree ' + ','.join(str(f.degree()) for f in hi)
                     + (f' (Schwarz form: {len(schwarz)}, companion: {len(hi) - len(schwarz)})'))
    return '; '.join(parts)


def _index1_reduce(E, Ap, Bp, Cp, F):
    """Exact index-1 reduction of ``G(s) = Cp (sE - Ap)^{-1} Bp``.
    Returns (A, B, C, D, descriptor_flag, note)."""
    S = _sage()
    n = E.nrows()
    r = E.rank()
    if r == n:
        Ei = E.inverse()
        return Ei * Ap, Ei * Bp, Cp, S.matrix(F, Cp.nrows(), Bp.ncols()), False, 'E invertible'
    # regularity of the pencil: det(s0 E - Ap) != 0 at some exact s0
    regular = False
    for s0 in (S.QQ(7) / 3, S.QQ(-11) / 5, S.QQ(13) / 2, S.QQ(3) / 17):
        if (s0 * E - Ap).det() != 0:
            regular = True
            break
    if not regular:                                       # decide exactly: det over F[x]
        Px = S.PolynomialRing(F, 'xpencil')
        xg = Px.gen()
        regular = (xg * E.change_ring(Px) - Ap.change_ring(Px)).det() != 0
    if not regular:
        raise RealizationRefused(
            'singular_pencil', 'det K(s) vanishes identically: some field is '
            'undriven / the free action is degenerate')
    pc = list(E.pivots())
    pr = list(E.pivot_rows())
    Nr = E.right_kernel_matrix(basis='pivot')             # rows span ker E
    Nl = E.left_kernel_matrix(basis='pivot')              # rows span ker E^T
    Q = S.matrix(F, n, n)
    for k, j in enumerate(pc):
        Q[j, k] = 1
    for k in range(Nr.nrows()):
        for i in range(n):
            Q[i, r + k] = Nr[k, i]
    P = S.matrix(F, n, n)
    for k, i in enumerate(pr):
        P[k, i] = 1
    for k in range(Nl.nrows()):
        for j in range(n):
            P[r + k, j] = Nl[k, j]
    if Q.det() == 0 or P.det() == 0:                      # pragma: no cover
        raise AssertionError('pivot completion of the kernel bases is singular')
    Et = P * E * Q
    At = P * Ap * Q
    Bt = P * Bp
    Ct = Cp * Q
    E11 = Et[:r, :r]
    assert Et[r:, :].is_zero() and Et[:, r:].is_zero()
    A11, A12, A21, A22 = At[:r, :r], At[:r, r:], At[r:, :r], At[r:, r:]
    if A22.det() == 0:
        raise RealizationRefused(
            'index_gt_1', f'descriptor of index > 1 (the {n - r}x{n - r} '
            'algebraic block of the pencil is singular): the propagator would '
            "carry delta'(t) or higher terms")
    A22i = A22.inverse()
    E11i = E11.inverse()
    A = E11i * (A11 - A12 * A22i * A21)
    B = E11i * (Bt[:r, :] - A12 * A22i * Bt[r:, :])
    C = Ct[:, :r] - Ct[:, r:] * A22i * A21
    D = -Ct[:, r:] * A22i * Bt[r:, :]
    return A, B, C, D, True, f'index-1 descriptor: {n - r} algebraic variable(s) eliminated'


#: Prime of the modular Krylov pivots (``_controllable_part_modular``).
_KRYLOV_PRIME = 8388593


def _controllable_part_modular(A, B):
    """``_controllable_part`` over ``QQ`` with the Krylov ranks taken modulo
    a prime, certified exactly: the Krylov columns independent mod p (a
    nonzero minor mod p is nonzero over ``QQ``) span a subspace ``W`` of the
    controllable one; ``W`` is A-invariant and contains ``B`` iff the exact
    solves ``V X = A V``, ``V Y = B`` are consistent, and then ``W`` IS the
    controllable subspace.  ``V`` is the echelon basis of ``W`` (unique for
    the subspace), so the result is the one of the exact Krylov ranks, bit
    for bit; their cost grows with the height of the Krylov vectors (80 s
    for 46 states holding a 16-state rational fit, against ~1 s here).
    Returns None (the caller runs the exact ranks) when the certificate
    fails or a denominator is divisible by p."""
    S = _sage()
    n = A.nrows()
    Fp = S.GF(_KRYLOV_PRIME)
    try:
        Ap, blk = A.change_ring(Fp), B.change_ring(Fp)
    except (ZeroDivisionError, TypeError, ArithmeticError):
        # a denominator divisible by p (Sage raises TypeError: "matrix
        # denominator not coprime to modulus")
        return None
    E, picked, power = S.matrix(Fp, n, 0), [], 0
    while True:
        grew = False
        for j in range(blk.ncols()):
            E2 = E.augment(blk.matrix_from_columns([j]))
            if E2.rank() > E.ncols():
                E, grew = E2, True
                picked.append((power, j))
        if not grew or E.ncols() == n:
            break
        blk, power = Ap * blk, power + 1
    if E.ncols() == n:
        return A, B, None
    if not picked:
        if B.is_zero():
            return S.matrix(S.QQ, 0, 0), S.matrix(S.QQ, 0, B.ncols()), S.matrix(S.QQ, n, 0)
        return None
    cols, blk_q = [], B
    for k in range(max(p for p, _ in picked) + 1):
        cols += [blk_q.column(j) for p, j in picked if p == k]
        blk_q = A * blk_q
    V = S.matrix(S.QQ, cols).transpose().column_space().basis_matrix().transpose()
    try:
        Ar = V.solve_right(A * V)
        Br = V.solve_right(B)
    except ValueError:                     # W not A-invariant / B outside W
        return None
    return Ar, Br, V


def _controllable_part(A, B, F):
    """Exact restriction to the controllable subspace (Krylov ranks over F;
    over ``QQ`` modulo a prime, certified exactly: ``_controllable_part_modular``)."""
    S = _sage()
    n = A.nrows()
    if n == 0 or B.ncols() == 0:
        return A, B, None
    if F is S.QQ:
        out = _controllable_part_modular(A, B)
        if out is not None:
            return out
    K = B
    blk = B
    rank = K.rank()
    while True:
        blk = A * blk
        K2 = K.augment(blk)
        r2 = K2.rank()
        if r2 == rank:
            break
        K, rank = K2, r2
        if rank == n:
            break
    if rank == n:
        return A, B, None
    if rank == 0:
        return S.matrix(F, 0, 0), S.matrix(F, 0, B.ncols()), S.matrix(F, n, 0)
    V = K.column_space().basis_matrix().transpose()      # n x rank, echelon basis
    Ar = V.solve_right(A * V)
    Br = V.solve_right(B)
    return Ar, Br, V


def _kalman_minimal(A, B, C, F):
    """Exact minimal realization (controllable, then observable part)."""
    S = _sage()
    if A.nrows() == 0:
        return A, B, C
    Ar, Br, V = _controllable_part(A, B, F)
    Cr = C if V is None else C * V
    if Ar.nrows() == 0:
        return Ar, Br, S.matrix(F, C.nrows(), 0)
    # observable part by duality
    At, Ct_, W = _controllable_part(Ar.transpose(), Cr.transpose(), F)
    if W is None:
        return Ar, Br, Cr
    Am = At.transpose()
    Cm = Ct_.transpose()
    Bm = (Br.transpose() * W).transpose() if W.ncols() else S.matrix(F, 0, Br.ncols())
    return Am, Bm, Cm


def _routh_real(a):
    """Routh test of a real polynomial given by its coefficients ``a``
    (``a0 s^n + a1 s^{n-1} + ...``, exact rationals): True iff every root
    has negative real part."""
    if a and a[0] < 0:
        a = [-x for x in a]
    if any(x <= 0 for x in a):
        return False
    row0 = a[0::2]
    row1 = a[1::2]
    while row1:
        if row1[0] <= 0:
            return False
        new = []
        for k in range(len(row0) - 1):
            nxt = row1[k + 1] if k + 1 < len(row1) else 0
            new.append(row0[k + 1] - row0[0] * nxt / row1[0])
        row0, row1 = row1, new
    return True


def _routh_hurwitz(A, F):
    """Exact Routh test of the characteristic polynomial: True iff every root
    has negative real part (no roots computed).  Over ``QQ[I]`` the test runs
    on ``p(s) conj(p)(s)``: real rational coefficients, and the roots of
    ``conj(p)`` are the conjugates of those of ``p`` (same real parts), so it
    is Hurwitz iff ``p`` is.  (Returned None for complex fields: a complex
    realization never got the a-posteriori check.)"""
    S = _sage()
    n = A.nrows()
    if n == 0:
        return True
    p = A.charpoly()
    if F is S.QQ:
        return _routh_real(list(reversed(p.list())))
    try:
        pbar = p.map_coefficients(lambda c: c.conjugate())
        q = p * pbar
        coeffs = []
        for c in reversed(q.list()):
            re, im = (list(c.list()) + [0, 0])[:2]
            if im != 0:
                return None
            coeffs.append(S.QQ(re))
    except (AttributeError, TypeError, ValueError, ArithmeticError):
        return None
    return _routh_real(coeffs)


def _balance(A, B, C):
    if A.shape[0] == 0:
        return A, B, C, np.ones(0)
    Ab, (sca, _perm) = sl.matrix_balance(A, permute=False, separate=True)
    return Ab, B / sca[:, None], C * sca[None, :], sca


def _lyapunov_margin(A):
    """min eig / max eig of X, A X + X A^H = -I (> 0 <=> Hurwitz)."""
    n = A.shape[0]
    if n == 0:
        return 1.0
    try:
        X = sl.solve_continuous_lyapunov(A, -np.eye(n))
        X = 0.5 * (X + X.conj().T)
        w = np.linalg.eigvalsh(X)
        if not np.all(np.isfinite(w)):
            return None
        return float(w[0] / max(abs(w[-1]), np.finfo(float).tiny))
    except Exception:
        return None


#: Balancing range ``max T / min T`` above which ``realize_K`` checks the
#: rounded realization a posteriori (high-order rational kernels: Pade-n,
#: long Erlang chains); ordinary models stay within [0.125, 8] (census).
GRADED_BALANCE_RANGE = 1e6


def _aposteriori_check(A, B, C, rtol=1e-2):
    """``None`` when the rounded realization's stationary output covariance
    (unit white inputs; Lyapunov, ``lyapunov_refined``) is positive
    semidefinite and its diagonal agrees with the frequency-domain integral
    of the realization's own transfer function ``C (i w - A)^-1 B`` (which
    stays accurate when the time-domain operations do not) to ``rtol`` of its
    largest entry; else the reason.  Trapezoid in ``ln w`` between the decay
    rates / 100 and ``100 max|eig A|``, resonance nodes around underdamped
    eigenvalues, flat below and ``w^-2`` tail above (about 1e-4 accurate: the
    check catches a broken realization, not rounding)."""
    n = A.shape[0]
    if n == 0:
        return None
    try:
        P = lyapunov_refined(A, B @ B.conj().T)
    except Exception as exc:                              # pragma: no cover
        return f'Lyapunov solve failed ({type(exc).__name__})'
    Y = C @ P @ C.conj().T
    Y = 0.5 * (Y + Y.conj().T)
    if not np.all(np.isfinite(Y)):
        return 'non-finite stationary covariance'
    ev = np.linalg.eigvalsh(Y)
    top = float(np.max(np.abs(ev))) if ev.size else 0.0
    if top == 0.0:
        return None
    if ev[0] < -1e-6 * top:
        return (f'the stationary output covariance is not positive (eigenvalue '
                f'{ev[0]:.3g} against {top:.3g})')
    lam = np.linalg.eigvals(A)
    rates = np.abs(lam.real)
    lo = max(float(rates.min()) / 100.0, 1e-300)
    hi = 100.0 * float(np.max(np.abs(lam)))
    dec = max(1.0, np.log10(hi / lo))
    nodes = [np.geomspace(lo, hi, int(40 * dec) + 2)]
    for z in lam:
        # every underdamped eigenvalue (a complex-coefficient system has
        # resonances at negative frequencies with no conjugate partner)
        if abs(z.imag) > abs(z.real):
            nodes.append(abs(z.imag) + abs(z.real) * np.sinh(np.linspace(-8, 8, 81)))
    w = np.unique(np.concatenate(nodes))
    w = w[(w >= lo) & (w <= hi)]
    I_ = np.eye(n)
    S = np.zeros((w.size, C.shape[0]))
    for sgn in (1.0, -1.0):
        for i0 in range(0, w.size, 256):
            ws = w[i0:i0 + 256]
            M = 1j * sgn * ws[:, None, None] * I_ - A
            G = C @ np.linalg.solve(M, np.broadcast_to(B, (ws.size,) + B.shape))
            S[i0:i0 + 256] += np.real(np.einsum('wij,wij->wi', G, G.conj()))
    fn = getattr(np, 'trapezoid', None) or np.trapz
    yf = (fn(S * w[:, None], np.log(w), axis=0) + S[0] * lo + S[-1] * hi) / (2 * np.pi)
    yd = np.real(np.diag(Y))
    err = float(np.max(np.abs(yd - yf)))
    if not np.isfinite(err) or err > rtol * float(np.max(np.abs(yd))):
        return (f'the stationary output covariance disagrees with the frequency-'
                f'domain integral of the transfer function by {err:.3g} '
                f'(largest variance {float(np.max(np.abs(yd))):.3g})')
    return None


def _aposteriori_error(A, B, C):
    """The G6 realization term: the relative disagreement of the
    rounded realization's stationary output variances (Lyapunov,
    ``lyapunov_refined``) with the same variances integrated over frequency
    from its transfer function by adaptive quadrature in ``ln w``
    (``scipy.integrate.quad``, relative tolerance 1e-12, breakpoints at every
    ``|lambda|`` and the resonances; the pieces below ``min|lambda|/100``
    and above ``100 max|lambda|`` in ``w``; the
    transfer function stays accurate when the time-domain operations do
    not): ``max_i |Y_ii - Yf_ii| / max_i |Y_ii|``.  ``None`` when it cannot be
    measured to 1e-8 (then nothing is added).  A realization that passes
    ``_aposteriori_check`` (rtol 1e-2: it refuses broken ones) carries this
    number into the state-space accuracy estimate (analyzer G6)."""
    import warnings as _w
    from scipy.integrate import quad
    n = A.shape[0]
    if n == 0:
        return 0.0
    try:
        P = lyapunov_refined(A, B @ B.conj().T)
        yd = np.real(np.diag(C @ P @ C.conj().T))
    except Exception:
        return None
    top = float(np.max(np.abs(yd))) if yd.size else 0.0
    if not (np.isfinite(top) and top > 0):
        return None
    lam = np.linalg.eigvals(A)
    mags = np.abs(lam)
    lo = max(float(np.min(np.abs(lam.real))) / 100.0, 1e-300)
    hi = 100.0 * float(np.max(mags))
    pts = sorted({float(np.log(x)) for x in np.concatenate([mags, np.abs(lam.imag)])
                  if lo < x < hi})
    I_ = np.eye(n)

    def dens(w, i):
        tot = 0.0
        for sgn in (1.0, -1.0):
            g = C[i] @ np.linalg.solve(1j * sgn * w * I_ - A, B)
            tot += float(np.real(np.vdot(g, g)))
        return tot

    lin_pts = sorted({float(x) for x in np.concatenate([mags, np.abs(lam.imag)])
                      if 0.0 < x < hi})

    def variance_log(i):
        v, e = quad(lambda u: dens(np.exp(u), i) * np.exp(u),
                    np.log(lo), np.log(hi), points=pts or None,
                    epsabs=0.0, epsrel=1e-12, limit=1000)
        v0, e0 = quad(dens, 0.0, lo, args=(i,), epsabs=0.0, epsrel=1e-12,
                      limit=200)
        return v + v0, e + e0

    def variance_lin(i):
        return quad(dens, 0.0, hi, args=(i,), points=lin_pts or None,
                    epsabs=0.0, epsrel=1e-12, limit=1000)

    # two independently split quadratures: a wrong Lyapunov solution shows
    # in both; a quadrature that misses a narrow peak (stiff real spectra)
    # rarely in both, so the smaller discrepancy is the measured error
    err = 0.0
    try:
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            for i in range(C.shape[0]):
                vt, et = quad(dens, hi, np.inf, args=(i,), epsabs=0.0,
                              epsrel=1e-12, limit=200)
                best = None
                for fn in (variance_log, variance_lin):
                    v, e = fn(i)
                    v, e = v + vt, e + et
                    if np.isfinite(v) and e <= 1e-8 * abs(v):
                        d = abs(yd[i] - v / (2 * np.pi))
                        best = d if best is None else min(best, d)
                if best is None:
                    return None
                err = max(err, best)
    except Exception:
        return None
    rel = err / top
    return float(rel) if np.isfinite(rel) else None


# ─── main entry points ────────────────────────────────────────────────────

def _time_scale(ents):
    """A power of two near the rate scale of the rational functions
    ``ents`` (elements of ``F(s)``): the median of ``|K0 / K1|`` over the
    entries with a first-order polynomial part (``K0 + K1 s``: the fields'
    own rates), else over every numerator and denominator of
    ``|c_k / c_n|^(1/(n - k))`` (each coefficient ratio of a polynomial of
    degree ``n``, which scales as a rate^(n-k)).  The
    realization is formed in ``sigma = s / omega0`` (``_rescale_s``) and
    mapped back exactly (``A = omega0 A~``, ``B = omega0 B~``): in other
    time units the coefficients of a companion / chain form carry different
    powers of the unit, and the rounded, balanced realization was graded
    (balancing does not undo it), so exponentials lost ~1e-9 at rates 1e-9
    and a posteriori checks fired at rates 1e6.  1 when nothing sets a
    scale."""
    import math
    ents = list(ents)
    # the fields' own first-order parts first (|K0 / K1| of each entry's
    # polynomial part: the rates the model is written in); a kernel's
    # coefficients (a Pade-16 delay's ~n/d poles) only when no entry has one
    logs = []
    for x in ents:
        try:
            q = x.numerator().quo_rem(x.denominator())[0]
        except (AttributeError, ArithmeticError, TypeError, ValueError):
            continue
        if q.degree() == 1:
            c0, c1 = abs(complex(q[0])), abs(complex(q[1]))
            if c0 > 0 and c1 > 0:
                logs.append(math.log2(c0 / c1))
    for x in ([] if logs else ents):
        for poly in (x.numerator(), x.denominator()):
            n = poly.degree()
            if n < 1:
                continue
            cs = poly.list()
            cn = abs(complex(cs[n]))
            if cn == 0:
                continue
            for k in range(n):
                ck = abs(complex(cs[k]))
                if ck > 0:
                    logs.append(math.log2(ck / cn) / (n - k))
    if not logs:
        return 1
    logs.sort()
    m = len(logs)
    med = logs[m // 2] if m % 2 else 0.5 * (logs[m // 2 - 1] + logs[m // 2])
    return 2 ** int(round(med))


def _field_scales(ents, nf, sweeps=8):
    """Powers of two ``(r, c)`` (row / column exponents) that equilibrate
    ``|K(sigma = 1)|`` (Ruiz iterations): ``K~ = diag(2^r) K diag(2^c)``
    has the fields in comparable units.  The realization of ``K~`` is mapped
    back exactly (``realize_K``); without it a model whose fields carry
    different powers of the time unit (a spike train next to a voltage) was
    'graded' at fast time units and the a posteriori check added a spurious
    5e-4..8e-4 to the accuracy estimate."""
    import math
    mag = {}
    for (i, j), x in ents.items():
        try:
            v = abs(complex(x(1)))
        except (ZeroDivisionError, ArithmeticError, TypeError, ValueError):
            v = 0.0
        if v > 0 and math.isfinite(v):
            mag[(i, j)] = math.log2(v)
    r = [0] * nf
    c = [0] * nf
    for _ in range(sweeps):
        changed = False
        for i in range(nf):
            row = [m + r[i] + c[j] for (ii, j), m in mag.items() if ii == i]
            if row:
                d = -int(round(max(row) / 2.0)) if abs(max(row)) >= 1 else 0
                if d:
                    r[i] += d
                    changed = True
        for j in range(nf):
            col = [m + r[i] + c[j] for (i, jj), m in mag.items() if jj == j]
            if col:
                d = -int(round(max(col) / 2.0)) if abs(max(col)) >= 1 else 0
                if d:
                    c[j] += d
                    changed = True
        if not changed:
            break
    return r, c


def _rescale_s(x, w0, P):
    """``x(w0 s)`` for ``x`` in ``F(s)`` (``w0`` a power of two: exact)."""
    if w0 == 1:
        return x
    sv = P.gen() * P.base_ring()(w0)
    return x.numerator()(sv) / x.denominator()(sv)


def realize_K(K_ft, omega, num_params, *, phys_names=None, resp_names=None,
              per_entry=True, minimal=True, balance=True, truncation=None):
    """Realization of ``G = K_ft^{-1}`` at ``num_params`` (module docstring).

    ``K_ft``: Sage matrix (rows = response fields, columns = physical fields),
    symbolic in ``omega``; ``num_params``: {symbol: value} as built by the
    mean-field stage.  ``truncation``: the relative H-infinity tolerance of
    the balanced truncation of the per-entry realizations (``None``: the
    one in effect, ``truncation_tol()``; ``0``: exact, none).  Raises
    ``RealizationRefused``.
    """
    S = _sage()
    nf = K_ft.nrows()
    if K_ft.ncols() != nf:
        raise RealizationRefused('singular_pencil', f'K_ft is {nf}x{K_ft.ncols()}, not square')
    subs = _exact_subs(num_params)
    ents, F, P, fname = _entries_to_field(K_ft, omega, subs)
    prov = [f'field {fname}; omega -> -i s; parameters at their exact binary values']
    # realized in sigma = s / w0 (unit free), mapped back below
    w0 = _time_scale(ents.values())
    if w0 != 1:
        ents = {k: _rescale_s(x, w0, P) for k, x in ents.items()}
        prov.append(f'realized in s / {w0} (the model rate scale, a power of two)')
    # and with the fields in comparable units (K~ = diag(2^fr) K diag(2^fc);
    # G = diag(2^fc) G~ diag(2^fr), mapped back below, exactly)
    fr, fc = _field_scales(ents, nf)
    if any(fr) or any(fc):
        ents = {(i, j): x * F(2) ** (fr[i] + fc[j]) for (i, j), x in ents.items()}
        prov.append(f'fields equilibrated by powers of two (rows {fr}, columns {fc})')
    rs = np.array([2.0 ** k for k in fr])          # G's input (column) scales
    cs = np.array([2.0 ** k for k in fc])          # G's output (row) scales

    K0 = S.matrix(F, nf, nf)
    K1 = S.matrix(F, nf, nf)
    blocks_A, blocks_b, blocks_c, owners = [], [], [], []
    n_kernel_entries = 0
    for (i, j), x in sorted(ents.items()):
        q, r, d = _split_entry(x, P)
        if q.degree() >= 2:
            raise RealizationRefused(
                'improper_entry', f'K[{i},{j}] has a polynomial part of degree '
                f'{q.degree()} in s (second- or higher-order time derivative)')
        ql = q.list() + [0, 0]
        K0[i, j] = ql[0]
        K1[i, j] = ql[1]
        if r != 0:
            Ae, be, ce, (blocks, sch) = _realize_proper(r, d, F)
            blocks_A.append(Ae)
            blocks_b.append(be)
            blocks_c.append(ce)
            owners.append((i, j))
            n_kernel_entries += 1
            prov.append(f'K[{i},{j}]: rational part, denominator degree {d.degree()} '
                        f'-> {Ae.nrows()} state(s): {_describe_blocks(blocks, sch)}')
    NR = sum(Ae.nrows() for Ae in blocks_A)
    n = nf + NR
    E = S.matrix(F, n, n)
    Ap = S.matrix(F, n, n)
    Bp = S.matrix(F, n, nf)
    Cp = S.matrix(F, nf, n)
    for i in range(nf):
        Bp[i, i] = 1
        Cp[i, i] = 1
        for j in range(nf):
            E[i, j] = K1[i, j]
            Ap[i, j] = -K0[i, j]
    off = nf
    for Ae, be, ce, (i, j) in zip(blocks_A, blocks_b, blocks_c, owners):
        m = Ae.nrows()
        for a in range(m):
            E[off + a, off + a] = 1
            Ap[off + a, j] += be[a, 0]                   # s z = A_R z + B_R u_j
            Ap[i, off + a] -= ce[0, a]                   # s K1 u = -K0 u - C_R z + y
            for b_ in range(m):
                Ap[off + a, off + b_] = Ae[a, b_]
        off += m
    A, B, C, D, descr, note = _index1_reduce(E, Ap, Bp, Cp, F)
    prov.append(f'pencil n={n} (nf={nf}, kernel states {NR}): {note}')
    n_pre = A.nrows()
    if minimal:
        A, B, C = _kalman_minimal(A, B, C, F)
        prov.append(f'exact Kalman reduction: {n_pre} -> {A.nrows()} state(s)')
    cls = 'C' if NR else ('B' if descr else 'A')
    hur = _routh_hurwitz(A, F)

    cplx = fname != 'QQ'
    An, Bn, Cn, Dn = (_to_numpy(M, cplx) for M in (A, B, C, D))
    norm_A_raw = float(np.linalg.norm(An, 1)) * w0 if An.size else 0.0
    if balance:
        An, Bn, Cn, sca = _balance(An, Bn, Cn)
    else:
        sca = np.ones(An.shape[0])
    margin = _lyapunov_margin(An)
    graded = bool(sca.size) and float(sca.max()) > GRADED_BALANCE_RANGE * float(
        max(sca.min(), np.finfo(float).tiny))
    bad_margin = not (margin is not None and np.isfinite(margin) and margin > 0)
    aposteriori = {'checked': False, 'err_rel': None}
    if hur is True and An.size and (bad_margin or graded):
        # exactly Hurwitz, but the rounded matrices are strongly graded (high-
        # order rational kernels: Pade-n, long chains) or not even numerically
        # Hurwitz: the transfer function stays right while exponentials and
        # Lyapunov solves may not.  Check the rounded realization a posteriori;
        # refuse it when the check fails, so the state-space Phase J route
        # falls back to the pole route instead of returning wrong numbers
        # (seen: Pade-24 of a delay on a 4-field cycle, |B| ~ 1e24, C ~ 1e13).
        why = _aposteriori_check(An, Bn, Cn)
        aposteriori = {'checked': True,
                       'err_rel': None if why is not None
                       else _aposteriori_error(An, Bn, Cn)}
        if why is not None:
            raise RealizationRefused(
                'ill_conditioned',
                f'the realization is exactly Hurwitz but its rounded form fails '
                f'the a posteriori check (N = {An.shape[0]}, Lyapunov margin '
                f'{margin}, balancing range {float(sca.max() / max(sca.min(), 1e-300)):.3g}, '
                f'|B| {float(np.abs(Bn).max()):.3g}, |C| {float(np.abs(Cn).max()):.3g}): {why}')

    # the checks above ran on the unit-free realization; back to the model's
    # time and field units (exact powers of two)
    if w0 != 1:
        An, Bn = An * w0, Bn * w0
    if An.size or Bn.size:
        Bn = Bn * rs[None, :]
        Cn = Cn * cs[:, None]
    Dn = Dn * cs[:, None] * rs[None, :]
    entries = {}
    est = np.zeros((nf, nf), dtype=int)
    trunc = None
    if per_entry:
        tol = truncation_tol() if truncation is None else float(truncation)
        recs = {}
        for p in range(nf):
            for r_ in range(nf):
                a_, b_, c_ = A, B[:, r_], C[p, :]
                if minimal:
                    a_, b_, c_ = _kalman_minimal(a_, b_, c_, F)
                an, bn, cn = (_to_numpy(M, cplx) for M in (a_, b_, c_))
                if balance and an.shape[0]:
                    an, bn, cn, _s = _balance(an, bn, cn)
                if w0 != 1:
                    an, bn = an * w0, bn * w0
                bn, cn = bn * rs[r_], cn * cs[p]
                dn = _round(D[p, r_]) * (cs[p] * rs[r_])
                dn = dn if cplx else dn.real
                k = an.shape[0]
                ent = EntryRealization(a=an, b=bn.reshape(k), c=cn.reshape(k),
                                       d=dn, n_states=k)
                ent, recs[(p, r_)] = truncate_entry(ent, tol)   # (as is: n < 2, tol 0)
                entries[(p, r_)] = ent
                est[p, r_] = ent.n_states
        prov.append(f'per-entry state counts (minimal{", then truncated" if tol > 0 else ""}): '
                    f'max {int(est.max()) if nf else 0}, total {int(est.sum())}')
        trunc = _truncation_summary(tol, recs)
        if trunc['n_truncated']:
            prov.append(trunc['note'])
    cond = {
        'norm_A_unbalanced': norm_A_raw,
        'norm_A': float(np.linalg.norm(An, 1)) if An.size else 0.0,
        'norm_B': float(np.linalg.norm(Bn, 1)) if Bn.size else 0.0,
        'norm_C': float(np.linalg.norm(Cn, 1)) if Cn.size else 0.0,
        'norm_D': float(np.linalg.norm(Dn, 1)) if Dn.size else 0.0,
        'balance_min': float(sca.min()) if sca.size else 1.0,
        'balance_max': float(sca.max()) if sca.size else 1.0,
        'n_pencil': n, 'n_kernel_states': NR, 'n_before_reduction': n_pre,
        # the a posteriori check of a graded / not numerically
        # Hurwitz rounded realization, and its measured relative error
        'aposteriori_checked': aposteriori['checked'],
        'aposteriori_err_rel': aposteriori['err_rel'],
        'graded': graded, 'bad_margin': bad_margin,
    }
    return Realization(
        A=An, B=Bn, C=Cn, D=Dn, cls=cls, descriptor=descr,
        n_states=An.shape[0], entry_states=est, entries=entries,
        phys_names=tuple(phys_names or range(nf)),
        resp_names=tuple(resp_names or range(nf)),
        balance=np.asarray(sca, dtype=float), hurwitz=hur,
        lyapunov_margin=margin, conditioning=cond, provenance=tuple(prov),
        field_name=fname,
        exact={'A': A * F(w0),
               'B': B * F(w0) * S.diagonal_matrix(F, [F(2) ** k for k in fr]),
               'C': S.diagonal_matrix(F, [F(2) ** k for k in fc]) * C,
               'D': (S.diagonal_matrix(F, [F(2) ** k for k in fc]) * D
                     * S.diagonal_matrix(F, [F(2) ** k for k in fr])),
               'field': F,
               'K0': S.diagonal_matrix(F, [F(2) ** -k for k in fr]) * K0
               * S.diagonal_matrix(F, [F(2) ** -k for k in fc]),
               'K1': S.diagonal_matrix(F, [F(2) ** -k for k in fr]) * K1
               * S.diagonal_matrix(F, [F(2) ** -k for k in fc]) / F(w0)},
        truncation=trunc)


def realize(prop, num_params, **kw):
    """Realization from the propagator dict of ``build_propagator`` (uses
    ``K_ft``, ``omega``, ``nf``, ``ring_gen_names``)."""
    if prop.get('K_ft') is None or prop.get('omega') is None:
        raise RealizationRefused('no_kernel', 'propagator data has no K_ft / omega')
    if prop.get('G_tx_sym') is not None or int(prop.get('spatial_dim') or 0) > 0:
        raise RealizationRefused('spatial', 'spatial propagator (per-momentum '
                                 'realization is not part of the realization)')
    nf = int(prop.get('nf') or prop['K_ft'].nrows())
    names = list(prop.get('ring_gen_names') or [])
    resp = names[:nf] if len(names) >= 2 * nf else None
    phys = names[nf:2 * nf] if len(names) >= 2 * nf else None
    return realize_K(prop['K_ft'], prop['omega'], num_params,
                     phys_names=phys, resp_names=resp, **kw)


def try_realize(prop, num_params, **kw):
    """``(Realization, None)`` or ``(None, RealizationRefused)``."""
    try:
        return realize(prop, num_params, **kw), None
    except RealizationRefused as exc:
        return None, exc


def realize_kernel(att, num_params=None, *, balance=True, truncation=None,
                   direct=False):
    """Minimal SISO realization of a causal conductance kernel
    ``g(t) = Theta(t) c e^{a t} b`` (the state-space route's kernel
    edges and dressed lines), as an ``EntryRealization`` with ``d = 0``.
    ``direct=True``: also a kernel with a direct term, ``g = d
    delta + Theta c e^{a t} b`` (image ``d + (strictly proper)``: the
    all-pass Pade (n, n) image of a bare delay), with ``d`` in the
    realization; a polynomial part of degree >= 1 is refused as before.

    ``att``: a ``ConvVertexType`` kernel attachment (``engine/core/
    vertices.py``) with ``kernel_ft_fn`` (``omega -> g^(omega)``, the
    codebase's ``g^(omega) = int g(t) e^{-i omega t} dt``) or ``kernel_td_fn``
    (``tau -> g(tau)``, Heaviside stripped; its Laplace transform is taken
    exactly).  Exactly as for the propagator (module docstring): the
    parameters at their exact binary values, ``omega -> -i s``, the image
    an element of ``F(s)``, realized by the root-free chain of
    ``_realize_proper`` (exact factorization: EQUAL or nearly equal rates
    give a Jordan-type chain, no residues, no rate gap enters), reduced by
    the exact Kalman decomposition, rounded once and balanced, then
    balanced-truncated to ``truncation`` (``truncate_entry``; ``None``: the
    tolerance in effect, ``truncation_tol()``; ``0``: none).  Refused
    (``RealizationRefused``): no image (``no_kernel_image``), a symbol
    without a value (``unbound_symbol``), not rational in ``s``
    (``non_rational``: e.g. a Gaussian kernel), a delta or polynomial part
    (``kernel_not_strictly_proper``), a non-Hurwitz chain
    (``kernel_not_hurwitz``), a step other than ``heaviside(t)`` in a time
    image (``kernel_shifted_step``: only the causal step at 0 is stripped).
    Returns ``(EntryRealization, field_name)``."""
    S = _sage()
    subs = _exact_subs(num_params)
    s = S.SR.var('ssreal')
    ft_fn = att.get('kernel_ft_fn') if hasattr(att, 'get') else None
    td_fn = att.get('kernel_td_fn') if hasattr(att, 'get') else None
    try:
        if ft_fn is not None:
            w = S.SR.var('wkernelreal')
            e = _exactify(S.SR(ft_fn(w)).subs(subs)).subs({w: -S.I * s})
        elif td_fn is not None:
            # a plain variable: a domain would leave a global Maxima
            # assumption behind (the Laplace transform integrates t > 0)
            t = S.SR.var('tkernelreal')
            g0 = S.SR(td_fn(t))
            w_ = S.SR.wild(0)
            for h in g0.find(S.heaviside(w_)):
                if not bool((h.operands()[0] - t).is_zero()):
                    raise RealizationRefused(
                        'kernel_shifted_step', f'kernel {att.get("symbol")}: '
                        f'{h} is not the causal step heaviside(t)')
            g = g0.substitute_function(S.heaviside, lambda _x: S.SR(1))
            g = _exactify(g.subs(subs))
            e = g.laplace(t, s)
        else:
            raise RealizationRefused('no_kernel_image',
                                     'the kernel has neither a time nor a '
                                     'frequency image')
    except RealizationRefused:
        raise
    except Exception as exc:
        raise RealizationRefused(
            'non_rational', f'kernel {att.get("symbol")}: its image cannot be '
            f'formed at the parameter point ({type(exc).__name__}: '
            f'{str(exc)[:120]})') from exc
    free = [str(v) for v in S.SR(e).variables() if str(v) != str(s)]
    if free:
        raise RealizationRefused(
            'unbound_symbol', f'kernel {att.get("symbol")}: symbols '
            f'{sorted(free)} without a value')
    x = F = P = fname = None
    last = None
    for fname in ('QQ', 'QQ[I]'):
        F = S.QQ if fname == 'QQ' else _gauss_field()
        P = S.PolynomialRing(F, str(s))
        try:
            x = _sr_to_frac(S.SR(e), s, P.fraction_field(), F)
            break
        except Exception as exc:
            last, x = exc, None
    if x is None:
        raise RealizationRefused(
            'non_rational', f'kernel {att.get("symbol")}: {str(e)[:120]} is '
            f'not a rational function of s ({type(last).__name__})')
    w0 = _time_scale([x])                  # realized in s / w0 (realize_K)
    x = _rescale_s(x, w0, P)
    q, r, d = _split_entry(x, P)
    cplx = fname != 'QQ'
    d0 = 0.0
    if q != 0:
        if not (direct and q.degree() == 0):
            raise RealizationRefused(
                'kernel_not_strictly_proper', f'kernel {att.get("symbol")} '
                'has a delta or polynomial part')
        # the direct term, rounded once (exact in F)
        d0 = _round(q[0]) if cplx else _round(q[0]).real
    if r == 0:
        z = np.zeros((0, 0), dtype=complex if cplx else float)
        return EntryRealization(a=z, b=np.zeros(0, z.dtype),
                                c=np.zeros(0, z.dtype), d=d0,
                                n_states=0), fname
    A, b, c, _blocks = _realize_proper(r, d, F)
    A, b, c = _kalman_minimal(A, b, c, F)
    hur = _routh_hurwitz(A, F)
    an, bn, cn = (_to_numpy(M, cplx) for M in (A, b, c))
    if hur is None:
        hur = bool(np.all(np.linalg.eigvals(an).real < 0)) if an.size else True
    if not hur:
        raise RealizationRefused(
            'kernel_not_hurwitz', f'kernel {att.get("symbol")} grows or does '
            'not decay (a rate with non-negative real part)')
    if balance and an.shape[0]:
        an, bn, cn, _s = _balance(an, bn, cn)
    k = an.shape[0]
    ent = EntryRealization(a=an, b=bn.reshape(k), c=cn.reshape(k), d=d0,
                           n_states=k)
    tol = truncation_tol() if truncation is None else float(truncation)
    if tol > 0 and k >= 2:
        ent, _info = truncate_entry(ent, tol)
    if w0 != 1 and ent.n_states:          # back to the model's time unit (exact)
        ent = EntryRealization(a=ent.a * w0, b=ent.b * w0, c=ent.c, d=ent.d,
                               n_states=ent.n_states)
    return ent, fname


# ─── checks (tests, census) ───────────────────────────────────────────────

def compare_D(real, D_delta, num_params):
    """Compare ``real.D`` with the pipeline's ``D_delta``: exact comparison of
    the exact ``D`` with ``D_delta`` at the exact parameter values, and the
    max relative difference of the rounded ``D``.  Returns a dict."""
    S = _sage()
    if D_delta is None:
        return {'exact_equal': None, 'max_rel': None}
    subs = _exact_subs(num_params)
    nf = real.nf
    De = real.exact['D'] if real.exact else None
    exact_equal = True
    ref = np.zeros((nf, nf), dtype=complex)
    for i in range(nf):
        for j in range(nf):
            v = S.SR(D_delta[i, j]).subs(subs)
            try:
                ref[i, j] = complex(S.ComplexField(80)(v))
            except Exception:
                return {'exact_equal': None, 'max_rel': None,
                        'note': f'D_delta[{i},{j}] not numeric: {str(v)[:80]}'}
            if exact_equal is not None and De is not None:
                # exact only when D_delta is exact at the exact parameters
                # (a transcendental constant is rounded at a different stage
                # in the realization: numeric comparison only)
                try:
                    q = De.base_ring()(_sr_exact_number(v))
                    exact_equal = exact_equal and bool(De[i, j] == q)
                except Exception:
                    exact_equal = None
    scale = max(np.abs(ref).max(), 1e-300)
    return {'exact_equal': exact_equal,
            'max_rel': float(np.abs(np.asarray(real.D) - ref).max() / scale)
            if np.abs(ref).max() > 0 else float(np.abs(real.D).max())}


def _sample_points(rho, n_points, seed):
    rng = np.random.default_rng(seed)
    half = n_points // 2
    mags = rho * 10.0 ** rng.uniform(-2, 2, n_points)
    pts = []
    for k in range(n_points):
        if k < half:
            pts.append(1j * mags[k] * (1 if rng.random() < 0.5 else -1))
        else:
            ang = rng.uniform(-0.45 * np.pi, 0.45 * np.pi)
            if rng.random() < 0.25:                      # some in the left half plane
                ang = np.pi - ang
            pts.append(mags[k] * np.exp(1j * ang))
    return pts


def verify_against_K(real, K_ft, omega, num_params, n_points=20, seed=0,
                     prec=170, points=None):
    """Entrywise error of ``real.transfer(s)`` against ``K(s)^{-1}`` computed
    independently from the symbolic ``K_ft`` in ``ComplexField(prec)``.

    Returns ``{'max_err': float, 'max_err_exact': float, 'per_entry': array,
    'points': list}``; the error of entry (p, r) is
    ``max_s |G~_pr(s) - G_pr(s)| / max_s |G_pr(s)|`` (entries that vanish at
    every point are scaled by ``max |G|``).  ``max_err_exact`` is the same
    for the exact (unrounded) matrices evaluated in ``ComplexField(prec)``.
    """
    S = _sage()
    CF = S.ComplexField(prec)
    subs = _exact_subs(num_params)
    nf = K_ft.nrows()
    Ks = [[_exactify(S.SR(K_ft[i, j]).subs(subs)) for j in range(nf)] for i in range(nf)]
    rho = max(float(np.linalg.norm(real.A, 1)) if real.n_states else 1.0, 1e-3)
    pts = list(points) if points is not None else _sample_points(rho, n_points, seed)
    ex = real.exact
    Ae = ex['A'].change_ring(CF) if ex and ex['A'].nrows() else None
    Be, Ce, De = ((ex[k].change_ring(CF)) for k in ('B', 'C', 'D')) if ex else (None,) * 3
    om = S.SR(omega)
    npts = len(pts)
    absref = np.zeros((npts, nf, nf))
    errf = np.zeros((npts, nf, nf))
    errx = np.zeros((npts, nf, nf))
    erre = np.zeros((npts, nf, nf))
    for k, s0 in enumerate(pts):
        sC = CF(s0.real, s0.imag)
        wC = -CF(0, 1) * sC
        Km = S.matrix(CF, nf, nf, [[CF(Ks[i][j].subs({om: wC})) if not Ks[i][j].is_zero() else CF(0)
                                    for j in range(nf)] for i in range(nf)])
        G = Km.inverse()                                  # the reference, ~prec bits
        got = real.transfer(s0)
        Gx = None
        if ex is not None:
            if Ae is None:
                Gx = De
            else:
                Gx = Ce * (sC * S.identity_matrix(CF, Ae.nrows()) - Ae).solve_right(Be) + De
        for i in range(nf):
            for j in range(nf):
                g = G[i, j]
                absref[k, i, j] = float(abs(g))
                errf[k, i, j] = float(abs(CF(got[i, j].real, got[i, j].imag) - g))
                if Gx is not None:
                    errx[k, i, j] = float(abs(Gx[i, j] - g))
                if real.entries:
                    v = real.entries[(i, j)].transfer(s0)
                    erre[k, i, j] = float(abs(CF(complex(v).real, complex(v).imag) - g))
    scale = absref.max(axis=0)
    gmax = max(absref.max(), 1e-300)
    # an entry of G that vanishes identically comes back from the pivoted
    # high-precision inverse as rounding noise (~2^-prec); scale it by max|G|
    scale = np.where(scale > gmax * 2.0 ** (20 - prec), scale, gmax)
    per = errf.max(axis=0) / scale
    out = {'max_err': float(per.max()), 'per_entry': per, 'points': pts}
    if ex is not None:
        out['max_err_exact'] = float((errx.max(axis=0) / scale).max())
    if real.entries:
        pe = erre.max(axis=0) / scale
        out['max_err_entries'] = float(pe.max())
        out['per_entry_siso'] = pe
    return out

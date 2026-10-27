"""api.master_equation -- exact master-equation oracle for lattice jump processes.

A continuous-time Markov chain on a (truncated) box of the integer lattice
``Z^d`` is described by a list of *transitions* ``(rate, jump)``: at state
``n`` the jump ``n -> n + jump`` happens at rate ``rate(x)``, where
``x = origin + h * n`` is the physical state and ``h`` the lattice step (the
inverse system size).  The generator ``Q`` is assembled as a sparse matrix and
solved by direct linear algebra, with no simulation::

    import daedalus as dd
    me = dd.master_equation(
        transitions=[('lam', +1), ('mu*x', -1)],          # immigration-death
        fields=['x'], params={'lam': 20.0, 'mu': 1.0},
        h=1.0, box=(0, 80))
    st = me.stationary()                  # p(n), residual, truncation diagnostics
    cm = me.cumulants(kmax=4)             # exact kappa_1 .. kappa_4 (full tensors)
    cc = me.correlation(np.linspace(0, 5, 11))   # C_jk(tau) = <dx_j(0) dx_k(tau)>

What is returned, and how it is checked
---------------------------------------
* ``stationary()`` solves ``Q^T p = 0`` by one sparse LU with a pinned row,
  followed by iterative refinement, on the unique closed communicating class
  (transient states get ``p = 0``; several closed classes are refused).  The
  relative residual ``|Q^T p|_1 / sum |Q_nn| p_n`` is reported.
* **Truncation.**  A jump that would leave the box is suppressed
  (``policy='reflect'``, the default) or wrapped (``'periodic'``).  A box edge
  where the rate vanishes anyway (a death rate at 0 particles) costs nothing.
  The diagnostic is the *suppressed flux* ``sum_n p_n s_n / sum_n p_n e_n``
  (``s_n`` the rate of blocked jumps at ``n``, ``e_n`` the total exit rate)
  and the *edge mass* ``sum p_n`` over the states that have a blocked jump.  A
  :class:`TruncationWarning` is given when the flux exceeds
  ``truncation_tol`` (default 1e-10); ``on_truncation='raise'`` makes it an
  error.
* ``cumulants()`` are central-moment sums over ``p`` (full symmetric tensors
  for ``k = 1..4``), carrying the residual and truncation numbers.
* ``correlation(taus)`` is ``C_jk(tau) = sum_n p_n f_j(n) [e^{tau Q} f_k](n)``,
  ``f = x - <x>``, ``tau >= 0`` (negative ``tau`` by ``C_jk(-tau) = C_kj(tau)``).
  ``method='direct'`` marches ``e^{Dt Q}`` over the grid with
  ``scipy.sparse.linalg.expm_multiply`` (round-off accurate; cost grows with
  ``tau_max * max exit rate``).  ``method='krylov'`` is shift-and-invert
  Arnoldi on ``(I - gamma Q)^{-1}`` with full re-orthogonalisation and one LU
  for the whole grid; the Krylov dimension grows in steps of 20 until two
  successive projections agree to ``krylov_tol``, and the reported
  ``krylov_error`` is that last change.  ``'auto'`` picks between them from
  the estimated work.
* ``integrated_correlation()`` is ``int_0^inf C_jk dtau`` from one Poisson
  solve ``Q u = -f`` (no time grid).

Diffusion (Kramers-Moyal) limit
-------------------------------
:meth:`MasterEquation.kramers_moyal` compares the jump process with the Ito
SDE ``dx = a(x) dt + sqrt(D(x)) dW`` of the same first two jump moments,
``a = h sum_k r_k j_k``, ``D = h^2 sum_k r_k j_k j_k^T``.  The diffusion is
solved on the sub-lattice ``h/m`` by a central-difference Fokker-Planck
generator written as a jump process, one pair of rates
``(m^2 S +- m J)/2`` per jump direction ``+-j`` (``S`` the sum and ``J`` the
difference of the rates of ``+j`` and ``-j``).  At ``m = 1`` it *is* the jump
process, and every observable is ``M(m) = M_KM + c_2 m^-2 + O(m^-4)``, so a
Richardson (Romberg) extrapolation in ``m^-2`` over ``refine=(1, 2, 4)``
gives ``M_KM`` to ``O(m^-6)``.  Note that the diffusion matches the jump
process in its first two jump cumulants only: ``kappa_3`` and ``kappa_4`` of
the SDE differ from those of the jump process at the order the jump process
itself makes them nonzero (immigration-death: ``kappa_4 = lambda/mu`` for the
jump process, ``3 lambda / (2 mu)`` for the diffusion).

Bridge to the SDE front end
---------------------------
With string rates, :meth:`MasterEquation.to_sde` builds the SDE-front-end
model whose noise is one state-dependent :class:`api.noise.Poisson` source of
size ``h`` per transition (``kind='jump'``) or the Gaussian diffusion of the
same two moments (``kind='diffusion'``), and
:meth:`MasterEquation.compare_with_daedalus` runs ``compute_cumulants`` on it
and tabulates, cumulant by cumulant, the exact value, the loop-order partial
sums and their errors.

What the loop expansion should and should not match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The loop expansion is an expansion in the jump size ``h`` about the
mean-field saddle.  Per cumulant ``kappa_k`` the tree level is ``O(h^{k-1})``
and each loop adds a relative factor ``O(h)``; at ``max_ell = L`` the relative
error against the exact master equation is ``O(h^{L+1})``.  It should
therefore *converge to* the oracle as ``h -> 0`` at fixed ``x`` (checked as
the error ratio between two system sizes), and it should *not* match at fixed
``h`` beyond that order, near a bifurcation (bistability, where the saddle is
not unique and the stationary law is not a single bump), when the box cuts the
law (lattice effects, boundaries such as ``x = 0`` for counts), or in the
exponentially small tails.  The lattice itself is exact in the oracle; the
expansion knows nothing of it.  For linear rates the tree level (linear
noise) is exact for ``kappa_1`` and ``kappa_2`` and all loop corrections
vanish.

The solver uses only numpy, scipy and sympy (rate strings); the bridge needs
the Sage-backed pipeline.
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from engine.errors import DaedalusError

__all__ = [
    'master_equation', 'MasterEquation', 'Transition',
    'StationaryResult', 'CumulantResult', 'CorrelationResult', 'KMResult',
    'ComparisonResult', 'MasterEquationError', 'TruncationWarning',
    'KrylovWarning',
]


class MasterEquationError(DaedalusError, ValueError):
    """A master-equation problem that is ill-posed or refused (bad box, a
    negative rate, several closed classes, a memory budget)."""


class TruncationWarning(UserWarning):
    """The state box cuts the stationary law: probability flux was
    suppressed at its edge above ``truncation_tol``."""


class KrylovWarning(UserWarning):
    """The Krylov projection of ``e^{tau Q}`` did not reach ``krylov_tol``
    within the dimension budget (the reported ``krylov_error`` is honest)."""


# ═══ transitions and rate compilation ═══════════════════════════════════════

@dataclass(frozen=True)
class Transition:
    """One jump channel: ``n -> n + jump`` at ``rate(x)``.

    ``rate`` is a string in the field names, the ``params`` and the lattice
    step ``h`` (``'k1*x*(x - h)/2'``; ``^`` is a power), or a callable
    ``rate(x_1, ..., x_d)`` of numpy arrays.  ``jump`` is a tuple of integers
    (an int for one field)."""
    rate: Any
    jump: tuple
    name: str = ''


def _as_jump(jump, d, where):
    if np.isscalar(jump):
        jump = (jump,)
    try:
        j = tuple(int(v) for v in jump)
    except (TypeError, ValueError):
        raise MasterEquationError(f'{where}: jump must be integers, got {jump!r}.')
    if any(float(a) != float(b) for a, b in zip(j, jump)):
        raise MasterEquationError(f'{where}: jump must be integers, got {jump!r}.')
    if len(j) != d:
        raise MasterEquationError(
            f'{where}: jump {jump!r} has {len(j)} components for {d} field(s).')
    if not any(j):
        raise MasterEquationError(f'{where}: the zero jump is not a transition.')
    return j


def _compile_rate(rate, fields, params, h, where):
    """Rate -> function of an ``(N, d)`` array of physical states."""
    d = len(fields)
    if callable(rate):
        def f(X):
            return np.asarray(rate(*[X[:, i] for i in range(d)]), dtype=float)
        return f
    if not isinstance(rate, str):
        if np.isscalar(rate):
            c = float(rate)
            return lambda X: np.full(X.shape[0], c)
        raise MasterEquationError(f'{where}: rate must be a string, a number or a callable.')
    import sympy
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            convert_xor)
    names = list(fields) + [k for k in params if k not in fields] + ['h']
    local = {n: sympy.Symbol(n) for n in names}
    try:
        expr = parse_expr(rate, local_dict=local,
                          transformations=standard_transformations + (convert_xor,))
    except Exception as e:                       # sympy raises many kinds
        raise MasterEquationError(f'{where}: cannot parse rate {rate!r}: {e}') from e
    free = {s.name for s in expr.free_symbols}
    unknown = free - set(names)
    if unknown:
        raise MasterEquationError(
            f'{where}: rate {rate!r} uses undeclared name(s) {sorted(unknown)}; '
            f'fields {list(fields)}, parameters {sorted(params)}, and h are known.')
    subs = {sympy.Symbol(k): float(v) for k, v in params.items() if k not in fields}
    subs[sympy.Symbol('h')] = float(h)
    expr = expr.subs(subs)
    syms = [sympy.Symbol(n) for n in fields]
    fn = sympy.lambdify(syms, expr, modules='numpy')

    def f(X):
        out = fn(*[X[:, i] for i in range(d)])
        return np.asarray(out, dtype=float)
    return f


# ═══ results ════════════════════════════════════════════════════════════════

@dataclass
class StationaryResult:
    """Stationary law on the box (``p`` has the box's shape)."""
    p: np.ndarray
    x: np.ndarray                    # (N, d) physical states, C order of ``p``
    residual: float                  # |Q^T p|_1 / sum |Q_nn| p_n
    truncation_flux: float           # suppressed-jump rate / total rate under p
    edge_mass: float                 # mass of states with a suppressed jump
    negative_mass: float             # mass clipped from negative LU round-off
    n_states: int
    closed_class_size: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True when the suppressed flux exceeded the object's ``truncation_tol``."""
        return bool(self.diagnostics.get('truncated', False))

    def summary(self) -> str:
        return (f'stationary: {self.n_states} states ({self.closed_class_size} recurrent), '
                f'residual {self.residual:.2e}, truncation flux {self.truncation_flux:.2e}, '
                f'edge mass {self.edge_mass:.2e}')


@dataclass
class CumulantResult:
    """Exact stationary cumulants.  ``cumulants[k]`` is the full symmetric
    tensor of order ``k`` (shape ``(d,)*k``), index order = ``fields``."""
    fields: tuple
    cumulants: dict
    mean: np.ndarray
    residual: float
    truncation_flux: float
    edge_mass: float
    h: float

    def __getitem__(self, k):
        return self.cumulants[k]

    def cumulant(self, *idx):
        """``kappa(x_a, x_b, ...)`` by field names or indices."""
        ix = tuple(self.fields.index(i) if isinstance(i, str) else int(i) for i in idx)
        return float(self.cumulants[len(ix)][ix])


@dataclass
class CorrelationResult:
    """``C[t, j, k] = <dx_j(0) dx_k(tau_t)>`` at ``tau = taus[t]``."""
    taus: np.ndarray
    C: np.ndarray
    method: str
    krylov_error: float              # worst relative change at the last Krylov step (0 for direct)
    krylov_dim: int
    gamma: Optional[float]
    residual: float
    truncation_flux: float
    diagnostics: dict = field(default_factory=dict)


@dataclass
class KMResult:
    """Kramers-Moyal (diffusion) limit by Richardson extrapolation in m^-2."""
    refine: tuple
    cumulants: dict                  # extrapolated {k: tensor}
    cumulants_by_m: dict             # {m: {k: tensor}}
    error_estimate: dict             # {k: tensor}: |full - drop-coarsest extrapolation|
    correlation: Optional[np.ndarray]
    taus: Optional[np.ndarray]
    truncation_flux: dict            # {m: flux}


@dataclass
class ComparisonResult:
    """Cumulant-by-cumulant comparison against ``compute_cumulants``.

    ``rows`` is a list of dicts: ``cumulant`` (e.g. ``'kappa_2(x,x)'``),
    ``exact``, ``loop`` (partial sums by loop order ``{ell: value}``) and
    ``error`` (``{ell: loop - exact}``), ``rel_error``."""
    rows: list
    h: float
    max_ell: int
    notes: list

    def table(self) -> str:
        L = self.max_ell
        head = f"{'cumulant':<22}{'exact':>16}" + ''.join(
            f"{'ell<=%d' % l:>16}{'rel.err':>11}" for l in range(L + 1))
        lines = [head]
        for r in self.rows:
            s = f"{r['cumulant']:<22}{r['exact']:>16.9e}"
            for l in range(L + 1):
                s += f"{r['loop'][l]:>16.9e}{r['rel_error'][l]:>11.2e}"
            lines.append(s)
        return '\n'.join(lines + [''] + [f'note: {n}' for n in self.notes])


# ═══ the solver object ══════════════════════════════════════════════════════

class MasterEquation:
    """Lattice jump process on a finite box; see the module docstring.

    Parameters
    ----------
    transitions : list of ``(rate, jump)`` / :class:`Transition`.
    fields : names of the state coordinates (default ``('x',)`` or
        ``x1, x2, ...``).
    params : dict of numeric parameters usable in string rates.
    h : lattice step, ``x = origin + h n`` (default 1).  ``h`` is also usable
        as a name inside string rates.
    origin : physical value of ``n = 0`` (scalar or per field).
    box : ``[(lo, hi), ...]`` inclusive integer index ranges per field
        (a single pair for one field), or ``'auto'`` (linear-noise box
        ``mean +- K sigma`` about the mean field from ``guess``).
    guess : physical starting point for the mean-field solve / the LU pin
        (default: box centre / ``origin``).
    lower, upper : physical bounds that clip an ``'auto'`` box.
    K : half-width in standard deviations of the ``'auto'`` box.
    policy : ``'reflect'`` (drop jumps that leave the box) or ``'periodic'``.
    truncation_tol : suppressed-flux level above which ``TruncationWarning``.
    on_truncation : ``'warn'`` | ``'raise'`` | ``'ignore'``.
    """

    def __init__(self, transitions, *, fields=None, params=None, h=1.0,
                 origin=0.0, box=None, guess=None, lower=None, upper=None,
                 K=9.0, policy='reflect', truncation_tol=1e-10,
                 on_truncation='warn', max_states=20_000_000):
        if not transitions:
            raise MasterEquationError('need at least one transition.')
        trs = list(transitions)
        first = trs[0]
        jv0 = first.jump if isinstance(first, Transition) else first[1]
        d = 1 if np.isscalar(jv0) else len(tuple(jv0))
        if fields is None:
            fields = ('x',) if d == 1 else tuple(f'x{i + 1}' for i in range(d))
        fields = tuple(fields)
        if len(fields) != d or len(set(fields)) != d:
            raise MasterEquationError(f'fields {fields!r} must be {d} distinct names.')
        if policy not in ('reflect', 'periodic'):
            raise MasterEquationError(f"policy must be 'reflect' or 'periodic', got {policy!r}.")
        if on_truncation not in ('warn', 'raise', 'ignore'):
            raise MasterEquationError("on_truncation must be 'warn', 'raise' or 'ignore'.")
        h = float(h)
        if not (h > 0 and math.isfinite(h)):
            raise MasterEquationError(f'h must be positive, got {h!r}.')
        self.fields = fields
        self.d = d
        self.params = {k: float(v) for k, v in (params or {}).items()}
        self.h = h
        self.origin = np.broadcast_to(np.asarray(origin, float), (d,)).copy()
        self.policy = policy
        self.truncation_tol = float(truncation_tol)
        self.on_truncation = on_truncation
        self.K = float(K)
        self._lower = lower
        self._upper = upper
        self._max_states = int(max_states)

        self.transitions: list[Transition] = []
        self._rate_fns = []
        for i, t in enumerate(trs):
            if isinstance(t, Transition):
                rate, jump, name = t.rate, t.jump, t.name
            else:
                if len(t) != 2:
                    raise MasterEquationError(
                        f'transition {i}: expected (rate, jump), got {t!r}.')
                rate, jump = t
                name = ''
            where = f'transition {i}'
            jump = _as_jump(jump, d, where)
            self.transitions.append(Transition(rate, jump, name or f't{i}'))
            self._rate_fns.append(_compile_rate(rate, fields, self.params, h, where))

        self._guess = None if guess is None else np.broadcast_to(
            np.asarray(guess, float), (d,)).copy()
        if isinstance(box, str):
            if box != 'auto':
                raise MasterEquationError("box must be [(lo, hi), ...] or 'auto'.")
            self._box = None
            self._auto = True
        else:
            self._auto = False
            self._box = self._check_box(box)
        self._Q = None
        self._stat = None
        self._lna = None

    # ── geometry ────────────────────────────────────────────────────────────

    def _check_box(self, box):
        if box is None:
            raise MasterEquationError("box is required (pairs of integer bounds, or 'auto').")
        b = np.asarray(box)
        if self.d == 1 and b.shape == (2,):
            b = b.reshape(1, 2)
        if b.shape != (self.d, 2):
            raise MasterEquationError(
                f'box must be {self.d} pair(s) (lo, hi) of integer indices, got {box!r}.')
        if np.any(b != np.round(b)):
            raise MasterEquationError(f'box bounds must be integers, got {box!r}.')
        b = b.astype(np.int64)
        if np.any(b[:, 1] < b[:, 0]):
            raise MasterEquationError(f'empty box {box!r}.')
        n = int(np.prod((b[:, 1] - b[:, 0] + 1).astype(object)))
        if n > self._max_states:
            raise MasterEquationError(
                f'box has {n} states, above max_states={self._max_states}.')
        return b

    @property
    def box(self):
        if self._box is None:
            self._box = self._auto_box()
        return self._box

    @property
    def shape(self):
        b = self.box
        return tuple(int(v) for v in (b[:, 1] - b[:, 0] + 1))

    @property
    def n_states(self):
        return int(np.prod(self.shape))

    def _x_of(self, n):
        return self.origin + self.h * np.asarray(n, float)

    def states(self):
        """Physical states, shape ``(N, d)``, in C order of the box."""
        b = self.box
        g = np.indices(self.shape).reshape(self.d, -1)
        return self.origin + self.h * (g.T + b[:, 0])

    def rates(self, X):
        """Rates of every transition at physical states ``X`` (``(N, d)``),
        shape ``(n_transitions, N)``."""
        X = np.atleast_2d(np.asarray(X, float))
        return np.stack([np.broadcast_to(f(X), (X.shape[0],)) for f in self._rate_fns])

    # ── mean field / linear noise ───────────────────────────────────────────

    def drift_diffusion(self, X):
        """Jump-moment drift ``a(x) = h sum r_k j_k`` (``(N, d)``) and
        diffusion ``D(x) = h^2 sum r_k j_k j_k^T`` (``(N, d, d)``)."""
        R = self.rates(X)
        J = np.array([t.jump for t in self.transitions], float)       # (T, d)
        a = self.h * (R.T @ J)
        D = self.h ** 2 * np.einsum('tn,ti,tj->nij', R, J, J)
        return a, D

    def lna(self, x0=None):
        """Mean-field fixed point and linear-noise covariance.

        Returns ``dict(mean, A, D, cov, stable)``: ``a(mean) = 0``,
        ``A = da/dx`` (central differences), ``D`` the diffusion matrix and
        ``cov`` the solution of ``A S + S A^T + D = 0`` (nan if ``A`` is not
        stable).  Exact for linear rates."""
        from scipy.optimize import fsolve
        from scipy.linalg import solve_continuous_lyapunov
        if x0 is None:
            if self._guess is not None:
                x0 = self._guess
            elif self._box is not None:
                x0 = self._x_of(self._box.mean(axis=1))
            else:
                x0 = self.origin
        x0 = np.broadcast_to(np.asarray(x0, float), (self.d,))
        drift = lambda x: self.drift_diffusion(x[None, :])[0][0]
        xs, info, ier, msg = fsolve(drift, x0, xtol=1e-12, full_output=True)
        flux = self.h * float(np.max(np.abs(self.rates(xs[None, :]))))
        if ier != 1 and np.max(np.abs(drift(xs))) > 1e-8 * max(flux, 1e-300):
            raise MasterEquationError(f'mean-field solve from {x0} did not converge: {msg}')
        A = np.zeros((self.d, self.d))
        for i in range(self.d):
            e = np.zeros(self.d)
            s = 1e-6 * max(1.0, abs(xs[i]))
            e[i] = s
            A[:, i] = (drift(xs + e) - drift(xs - e)) / (2 * s)
        D = self.drift_diffusion(xs[None, :])[1][0]
        ev = np.linalg.eigvals(A)
        stable = bool(np.all(ev.real < 0))
        cov = (solve_continuous_lyapunov(A, -D) if stable
               else np.full((self.d, self.d), np.nan))
        return dict(mean=xs, A=A, D=D, cov=cov, stable=stable)

    def _auto_box(self):
        lna = self.lna()
        if not lna['stable']:
            raise MasterEquationError(
                "box='auto': the mean-field fixed point is not linearly stable "
                f"(eigenvalues {np.linalg.eigvals(lna['A'])}); give an explicit box.")
        sig = np.sqrt(np.maximum(np.diag(lna['cov']), 0.0))
        lo = lna['mean'] - self.K * sig
        hi = lna['mean'] + self.K * sig
        if self._lower is not None:
            lo = np.maximum(lo, np.broadcast_to(np.asarray(self._lower, float), (self.d,)))
        if self._upper is not None:
            hi = np.minimum(hi, np.broadcast_to(np.asarray(self._upper, float), (self.d,)))
        nlo = np.floor((lo - self.origin) / self.h + 1e-9).astype(np.int64)
        nhi = np.ceil((hi - self.origin) / self.h - 1e-9).astype(np.int64)
        self._guess = lna['mean']
        return self._check_box(np.stack([nlo, nhi], 1))

    # ── generator ───────────────────────────────────────────────────────────

    def generator(self) -> sp.csr_matrix:
        """The sparse generator ``Q`` (``Q[n, m]`` = rate ``n -> m``, rows sum
        to zero) on the box, C ordering."""
        if self._Q is None:
            self._assemble()
        return self._Q

    def _assemble(self):
        shape = np.array(self.shape, np.int64)
        N = int(np.prod(shape))
        grid = np.indices(tuple(shape)).reshape(self.d, -1)
        X = self.origin + self.h * (grid.T + self.box[:, 0])
        strides = np.ones(self.d, np.int64)
        for i in range(self.d - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        rows, cols, vals = [], [], []
        exit_rate = np.zeros(N)
        blocked = np.zeros(N)
        for t, fn in zip(self.transitions, self._rate_fns):
            r = np.broadcast_to(np.asarray(fn(X), float), (N,)).copy()
            bad = ~np.isfinite(r)
            if bad.any():
                k = int(np.argmax(bad))
                raise MasterEquationError(
                    f'{t.name}: non-finite rate at x={X[k]} (and {int(bad.sum()) - 1} more states).')
            scale = max(float(np.max(np.abs(r))), 1e-300)
            neg = r < -1e-12 * scale
            if neg.any():
                k = int(np.argmin(r))
                raise MasterEquationError(
                    f'{t.name}: negative rate {r[k]:.3e} at x={X[k]} -- the rate must be '
                    f'>= 0 on the whole box (shrink the box, or make the rate vanish at its edge).')
            r[r < 0] = 0.0
            ng = grid + np.array(t.jump, np.int64)[:, None]
            if self.policy == 'periodic':
                ng = ng % shape[:, None]
                inside = np.ones(N, bool)
            else:
                inside = np.all((ng >= 0) & (ng < shape[:, None]), axis=0)
            dst = (ng * strides[:, None]).sum(axis=0)
            src = np.arange(N)
            active = r > 0
            ok = active & inside & (dst != src)
            blocked += np.where(active & ~inside, r, 0.0)
            rows.append(src[ok]); cols.append(dst[ok]); vals.append(r[ok])
            exit_rate[ok] += r[ok]
        rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(-exit_rate)
        Q = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N)).tocsr()
        self._Q = Q
        self._exit = exit_rate
        self._blocked = blocked
        self._X = X

    # ── stationary law ──────────────────────────────────────────────────────

    def _closed_class(self, Q):
        n = Q.shape[0]
        A = Q.copy()
        A.setdiag(0)
        A.eliminate_zeros()
        ncomp, lab = connected_components(A, directed=True, connection='strong')
        if ncomp == 1:
            return np.arange(n)
        coo = A.tocoo()
        cross = lab[coo.row] != lab[coo.col]
        open_comps = np.unique(lab[coo.row[cross]])
        closed = np.setdiff1d(np.arange(ncomp), open_comps)
        if len(closed) != 1:
            sizes = [int((lab == c).sum()) for c in closed]
            raise MasterEquationError(
                f'the chain has {len(closed)} closed communicating classes (sizes {sizes}): '
                f'the stationary law is not unique (a conservation law, or a box that '
                f'splits the state space).  Restrict the box to one class.')
        return np.flatnonzero(lab == closed[0])

    def _pin_candidates(self, keep):
        """Indices (into ``keep``) of states to pin at ``p = 1``, best first:
        the mean-field fixed point, the user's ``guess``, the box centre.  The
        pinned LU is only accurate when the pin carries non-negligible
        probability, so the mean-field state comes first."""
        grid = np.indices(self.shape).reshape(self.d, -1)[:, keep] + self.box[:, 0][:, None]

        def nearest(target):
            return int(np.argmin(((grid - np.asarray(target)[:, None]) ** 2).sum(axis=0)))

        cands = []
        try:
            cands.append(nearest((self.lna()['mean'] - self.origin) / self.h))
        except (MasterEquationError, FloatingPointError, ValueError):
            pass
        if self._guess is not None:
            cands.append(nearest((self._guess - self.origin) / self.h))
        cands.append(nearest(self.box.mean(axis=1)))
        return list(dict.fromkeys(cands))

    @staticmethod
    def _solve_pinned(Qw, c):
        """Stationary vector of ``Qw`` with state ``c`` pinned: returns
        ``(pk, negative_mass, residual)`` (``pk`` clipped and normalised)."""
        nk = Qw.shape[0]
        M = Qw.T.tocsr()
        mask = np.ones(nk); mask[c] = 0.0
        M = (sp.diags(mask) @ M + sp.csr_matrix(([1.0], ([c], [c])), shape=(nk, nk))).tocsc()
        rhs = np.zeros(nk); rhs[c] = 1.0
        lu = spla.splu(M, permc_spec='COLAMD')
        pk = lu.solve(rhs)
        for _ in range(2):                               # iterative refinement
            res = rhs - M @ pk
            pk = pk + lu.solve(res)
        if not np.all(np.isfinite(pk)) or np.abs(pk).sum() == 0.0:
            return None, np.inf, np.inf
        neg = float(np.abs(pk[pk < 0]).sum() / np.abs(pk).sum())
        pk = np.maximum(pk, 0.0)
        pk /= pk.sum()
        diag = np.abs(Qw.diagonal())
        resid = float(np.abs(Qw.T @ pk).sum() / max((diag * pk).sum(), 1e-300))
        return pk, neg, resid

    def stationary(self) -> StationaryResult:
        """Stationary law (cached).  See the module docstring."""
        if self._stat is not None:
            return self._stat
        Q = self.generator()
        N = Q.shape[0]
        keep = self._closed_class(Q)
        Qw = Q if len(keep) == N else Q[keep, :][:, keep]
        nk = len(keep)
        tried, tried_local, best = [], [], None
        queue = self._pin_candidates(keep)
        while queue:
            c = queue.pop(0)
            pk, neg, resid = self._solve_pinned(Qw, c)
            tried.append((int(keep[c]), neg, resid))
            tried_local.append(c)
            if pk is not None and neg < 1e-6 and resid < 1e-8:
                best = (c, pk, neg, resid)
                break
            if pk is not None and len(tried) <= 3:         # re-pin where the failed solve put its mass
                alt = int(np.argmax(pk))
                if alt not in tried_local and alt not in queue:
                    queue.append(alt)
        if best is None:
            raise MasterEquationError(
                'the stationary solve failed for every pin state tried (state, negative mass, '
                f'residual): {tried}.  The law is probably too wide or too stiff for one LU '
                'over this box; give a guess near the bulk, or a smaller box.')
        c, pk, neg, resid = best
        p = np.zeros(N)
        p[keep] = pk
        tot = float((self._exit * p).sum() + (self._blocked * p).sum())
        flux = float((self._blocked * p).sum() / tot) if tot > 0 else 0.0
        edge = float(p[self._blocked > 0].sum())
        self._keep = keep
        st = StationaryResult(
            p=p.reshape(self.shape), x=self._X, residual=resid, truncation_flux=flux,
            edge_mass=edge, negative_mass=neg, n_states=N, closed_class_size=int(nk),
            diagnostics=dict(pin=int(keep[c]), policy=self.policy,
                             truncated=bool(flux > self.truncation_tol),
                             box=self.box.tolist(), h=self.h))
        self._stat = st
        if flux > self.truncation_tol and self.on_truncation != 'ignore':
            msg = (f'the state box cuts the stationary law: suppressed jump flux {flux:.2e} '
                   f'(edge mass {edge:.2e}) exceeds truncation_tol={self.truncation_tol:.1e}; '
                   f'enlarge the box (box={self.box.tolist()}) before trusting the cumulants.')
            if self.on_truncation == 'raise':
                raise MasterEquationError(msg)
            warnings.warn(msg, TruncationWarning, stacklevel=2)
        return st

    # ── cumulants ───────────────────────────────────────────────────────────

    def cumulants(self, kmax: int = 4) -> CumulantResult:
        """Exact stationary cumulants of orders ``1..kmax`` (``kmax <= 4``)."""
        if not (1 <= kmax <= 4):
            raise MasterEquationError('kmax must be 1..4.')
        st = self.stationary()
        p = st.p.ravel()
        X = st.x
        mean = p @ X
        F = X - mean
        d = self.d

        def moment(ix):
            col = np.ones_like(p)
            for i in ix:
                col = col * F[:, i]
            return float(p @ col)

        cum = {1: mean.copy()}
        if kmax >= 2:
            cum[2] = np.array([[moment((a, b)) for b in range(d)] for a in range(d)])
        if kmax >= 3:
            k3 = np.zeros((d,) * 3)
            for ix in itertools.combinations_with_replacement(range(d), 3):
                v = moment(ix)
                for q in set(itertools.permutations(ix)):
                    k3[q] = v
            cum[3] = k3
        if kmax >= 4:
            k4 = np.zeros((d,) * 4)
            C2 = cum[2]
            for ix in itertools.combinations_with_replacement(range(d), 4):
                a, b, c, e = ix
                v = moment(ix) - (C2[a, b] * C2[c, e] + C2[a, c] * C2[b, e] + C2[a, e] * C2[b, c])
                for q in set(itertools.permutations(ix)):
                    k4[q] = v
            cum[4] = k4
        return CumulantResult(self.fields, cum, mean, st.residual, st.truncation_flux,
                              st.edge_mass, self.h)

    # ── lag correlations ────────────────────────────────────────────────────

    def _working(self):
        st = self.stationary()
        Q = self.generator()
        keep = self._keep
        Qw = Q if len(keep) == Q.shape[0] else Q[keep, :][:, keep]
        P = st.p.ravel()[keep]
        mean = P @ st.x[keep]
        F = st.x[keep] - mean
        return Qw.tocsr(), P, F

    def correlation(self, taus, *, method='auto', krylov_tol=1e-10, m_max=300,
                    gamma=None, max_memory_gb=4.0, direct_budget=3e9) -> CorrelationResult:
        """Lag correlations ``C[t, j, k] = <dx_j(0) dx_k(tau_t)>``.

        ``method`` is ``'direct'``, ``'krylov'`` or ``'auto'``; ``krylov_tol``
        the target change between successive Krylov projections; ``m_max``
        the largest Krylov dimension; ``gamma`` the shift-and-invert time
        (default: the geometric mean of the smallest and largest positive
        ``tau``); ``direct_budget`` the work (``nnz * d * matvecs``) above
        which ``'auto'`` switches to Krylov."""
        taus = np.atleast_1d(np.asarray(taus, float))
        if taus.ndim != 1 or not np.all(np.isfinite(taus)):
            raise MasterEquationError('taus must be a finite 1-D array.')
        st = self.stationary()
        Qw, P, F = self._working()
        d = self.d
        n = Qw.shape[0]
        absT = np.abs(taus)
        pos = np.unique(absT[absT > 0])
        norm1 = float(np.abs(Qw).sum(axis=0).max())
        if method == 'auto':
            work = Qw.nnz * d * (norm1 * (pos.max() if len(pos) else 0.0) + 10.0 * len(pos))
            method = 'direct' if work <= direct_budget else 'krylov'
        if method not in ('direct', 'krylov'):
            raise MasterEquationError("method must be 'direct', 'krylov' or 'auto'.")
        PF = P[:, None] * F
        C0 = F.T @ PF                                           # tau = 0
        Cpos = {}
        kerr, kdim, used_gamma = 0.0, 0, None
        if len(pos):
            if method == 'direct':
                Y = F.copy()
                t_prev = 0.0
                for t in pos:
                    Y = spla.expm_multiply(Qw * (t - t_prev), Y)
                    t_prev = t
                    Cpos[t] = PF.T @ Y
            else:
                Cmat, kerr, kdim, used_gamma = self._krylov_correlation(
                    Qw, PF, F, pos, krylov_tol, m_max, gamma, max_memory_gb)
                Cpos = {t: Cmat[i] for i, t in enumerate(pos)}
        C = np.empty((len(taus), d, d))
        for i, t in enumerate(taus):
            if t == 0:
                C[i] = C0
            elif t > 0:
                C[i] = Cpos[t]
            else:
                C[i] = Cpos[-t].T
        if method == 'krylov' and kerr > krylov_tol:
            warnings.warn(f'Krylov projection changed by {kerr:.2e} at dimension {kdim} '
                          f'(target {krylov_tol:.1e}); raise m_max or pass method="direct".',
                          KrylovWarning, stacklevel=2)
        return CorrelationResult(taus=taus, C=C, method=method, krylov_error=kerr,
                                 krylov_dim=kdim, gamma=used_gamma, residual=st.residual,
                                 truncation_flux=st.truncation_flux,
                                 diagnostics=dict(n_recurrent=n, norm1=norm1))

    def _krylov_correlation(self, Qw, PF, F, pos, tol, m_max, gamma, max_gb):
        n, d = F.shape
        gam = float(gamma) if gamma is not None else float(
            math.sqrt(pos.min() * pos.max()))
        lu = spla.splu((sp.identity(n, format='csc') - gam * Qw.tocsc()), permc_spec='COLAMD')
        m_cap = int(min(m_max, max_gb * 1e9 / (8.0 * n) - 2))
        if m_cap < 20:
            raise MasterEquationError(
                f'{n} states leave room for a Krylov dimension of only {m_cap} within '
                f'{max_gb} GB; raise max_memory_gb.')
        step = 20
        out = np.zeros((len(pos), d, d))
        worst, dim_used = 0.0, 0
        for k in range(d):
            v0 = F[:, k].copy()
            beta = float(np.linalg.norm(v0))
            if beta == 0.0:
                continue
            V = np.empty((n, m_cap + 1), order='F')
            H = np.zeros((m_cap + 1, m_cap))
            V[:, 0] = v0 / beta
            prev, err, m = None, np.inf, 0
            Wl = np.zeros((d, m_cap + 1))
            Wl[:, 0] = PF.T @ V[:, 0]
            while m < m_cap:
                top = min(m + step, m_cap)
                broke = False
                for j in range(m, top):
                    w = lu.solve(V[:, j])
                    for _ in range(2):                           # CGS2
                        hc = V[:, :j + 1].T @ w
                        w -= V[:, :j + 1] @ hc
                        H[:j + 1, j] += hc
                    nw = float(np.linalg.norm(w))
                    H[j + 1, j] = nw
                    if nw <= 1e-13 * max(1.0, abs(H[0, 0])):
                        broke = True
                        top = j + 1
                        break
                    V[:, j + 1] = w / nw
                    Wl[:, j + 1] = PF.T @ V[:, j + 1]
                m = top
                cur = self._krylov_eval(H[:m, :m], Wl[:, :m], beta, gam, pos)
                if broke:
                    err = 0.0
                    break
                if prev is not None:
                    # relative change per lag (floored at 1e-6 of the largest value, where
                    # the correlation has decayed into round-off)
                    scale = np.maximum(np.max(np.abs(cur), axis=1, keepdims=True),
                                       1e-6 * max(np.max(np.abs(cur)), 1e-300))
                    err = float(np.max(np.abs(cur - prev) / scale))
                    if err <= tol:
                        break
                prev = cur
            out[:, :, k] = cur
            worst = max(worst, err)
            dim_used = max(dim_used, m)
        return out, worst, dim_used, gam

    @staticmethod
    def _krylov_eval(Hm, W, beta, gam, taus):
        """``[W exp(tau Qm) e1 beta]`` for each ``tau``, ``Qm = (I - Hm^-1)/gam``;
        returns ``(ntau, d)``."""
        from scipy.linalg import expm
        m = Hm.shape[0]
        Qm = (np.eye(m) - np.linalg.inv(Hm)) / gam
        e1 = np.zeros(m); e1[0] = beta
        out = np.empty((len(taus), W.shape[0]))
        try:
            lam, R = np.linalg.eig(Qm)
            c = np.linalg.solve(R, e1)
            recon = np.linalg.norm(R @ (lam * c) - Qm @ e1) / max(np.linalg.norm(Qm @ e1), 1e-300)
            ok = (np.linalg.cond(R) < 1e8 and recon < 1e-10 and np.all(np.abs(lam.imag) <= 1e-8 * np.abs(lam)))
        except np.linalg.LinAlgError:
            ok = False
        if ok:
            WR = W @ R
            for i, t in enumerate(taus):
                out[i] = np.real(WR @ (np.exp(lam * t) * c))
        else:
            for i, t in enumerate(taus):
                out[i] = W @ (expm(Qm * t) @ e1)
        return out

    def integrated_correlation(self):
        """``int_0^inf C_jk(tau) dtau`` (``(d, d)``), from one Poisson solve
        ``Q u = -f`` with ``<u> = 0``; also ``C(0)`` for the correlation time."""
        Qw, P, F = self._working()
        n = Qw.shape[0]
        c = int(np.argmax(P))
        M = Qw.tolil()
        M[c, :] = 0
        M[c, c] = 1.0
        lu = spla.splu(M.tocsc(), permc_spec='COLAMD')
        U = np.empty_like(F)
        for k in range(self.d):
            rhs = -F[:, k].copy()
            rhs[c] = 0.0
            u = lu.solve(rhs)
            u = u + lu.solve(rhs - M @ u)
            U[:, k] = u - P @ u
        return (P[:, None] * F).T @ U

    # ── diffusion (Kramers-Moyal) limit ─────────────────────────────────────

    def _km_transitions(self, m):
        """Central-difference Fokker-Planck generator on the lattice ``h/m``,
        as transitions: per jump direction ``+-j`` the pair
        ``(m^2 S +- m J)/2`` (``S`` / ``J`` the sum / difference of the rates
        of ``+j`` and ``-j``)."""
        classes: dict = {}
        for t, fn in zip(self.transitions, self._rate_fns):
            j = t.jump
            sgn = 1
            for v in j:
                if v != 0:
                    sgn = 1 if v > 0 else -1
                    break
            key = tuple(sgn * v for v in j)
            classes.setdefault(key, []).append((sgn, fn))
        trs = []
        for key, members in classes.items():
            def S_J(X, members=members):
                S = 0.0
                J = 0.0
                for sgn, fn in members:
                    r = np.broadcast_to(np.asarray(fn(X), float), (X.shape[0],))
                    S = S + r
                    J = J + sgn * r
                return S, J
            def rp(*cols, S_J=S_J):
                S, J = S_J(np.stack(cols, 1))
                return (m * m * S + m * J) / 2.0
            def rm(*cols, S_J=S_J):
                S, J = S_J(np.stack(cols, 1))
                return (m * m * S - m * J) / 2.0
            trs.append(Transition(rp, key, f'km+{key}'))
            trs.append(Transition(rm, tuple(-v for v in key), f'km-{key}'))
        return trs

    def kramers_moyal(self, refine=(1, 2, 4), kmax=4, taus=None, box=None, **kw) -> KMResult:
        """Diffusion-limit cumulants (and correlations) by Richardson
        extrapolation in ``m^-2`` over sub-lattices ``h/m``, ``m`` in
        ``refine`` (see the module docstring).  The box keeps its physical
        extent unless ``box`` (integer index bounds at the lattice step ``h``) is
        given.  The jump process (``m = 1``) needs non-negative rates on the whole
        box, so a box reaching below a natural boundary (``x < 0`` for counts) is
        refused; start ``refine`` above 1 only when the rates stay positive there.  ``error_estimate`` is the difference between the extrapolation
        over all of ``refine`` and over ``refine[1:]``."""
        refine = tuple(int(m) for m in refine)
        if len(set(refine)) != len(refine) or min(refine) < 1:
            raise MasterEquationError('refine must be distinct integers >= 1.')
        box = self.box if box is None else self._check_box(box)
        by_m, flux, cors = {}, {}, {}
        for m in refine:
            me = MasterEquation(
                self._km_transitions(m), fields=self.fields, h=self.h / m,
                origin=self.origin, box=box * m, policy=self.policy,
                truncation_tol=self.truncation_tol, on_truncation=self.on_truncation,
                guess=self._guess, max_states=self._max_states)
            cr = me.cumulants(kmax)
            by_m[m] = cr.cumulants
            flux[m] = cr.truncation_flux
            if taus is not None:
                cors[m] = me.correlation(taus, **kw).C

        def extrap(vals_by_m, ms):
            s = np.array([1.0 / (m * m) for m in ms])
            stack = np.stack([vals_by_m[m] for m in ms])
            flat = stack.reshape(len(ms), -1)
            if len(ms) == 1:
                return vals_by_m[ms[0]].copy()
            # Lagrange interpolation at s = 0
            w = np.ones(len(ms))
            for i in range(len(ms)):
                for j in range(len(ms)):
                    if i != j:
                        w[i] *= (0.0 - s[j]) / (s[i] - s[j])
            return (w @ flat).reshape(stack.shape[1:])

        ms = sorted(refine)
        cum, err = {}, {}
        for k in range(1, kmax + 1):
            vals = {m: by_m[m][k] for m in ms}
            full = extrap(vals, ms)
            cum[k] = full
            err[k] = (np.abs(full - extrap(vals, ms[1:])) if len(ms) > 1
                      else np.full_like(full, np.nan))
        corr = extrap(cors, ms) if taus is not None else None
        return KMResult(refine=tuple(ms), cumulants=cum, cumulants_by_m=by_m,
                        error_estimate=err, correlation=corr,
                        taus=None if taus is None else np.asarray(taus, float),
                        truncation_flux=flux)

    # ── bridge to the SDE front end ─────────────────────────────────────────

    def _string_rates(self):
        out = []
        for t in self.transitions:
            if not isinstance(t.rate, str):
                if np.isscalar(t.rate) and not callable(t.rate):
                    out.append(repr(float(t.rate)))
                    continue
                raise MasterEquationError(
                    f'{t.name}: the SDE bridge needs string rates (got a callable).')
            out.append(t.rate)
        return out

    def to_sde(self, name='jump process', *, kind='jump', domain='real'):
        """The SDE-front-end builder for this process.

        ``kind='jump'``: one :class:`api.noise.Poisson` source per transition
        (rate ``r(x)``, size ``h``) coupled to field ``i`` with coefficient
        ``jump_i``, zero drift.  ``kind='diffusion'``: drift ``a(x)`` and one
        Gaussian source of variance ``h^2 r(x)`` per transition (the
        Kramers-Moyal SDE).  Parameters are the entries of ``params`` plus
        ``h`` (declared with ``domain``; ``'positive'`` where the value is
        positive and ``domain='real'`` is given)."""
        from api.sde import SDE
        from api.noise import Poisson, Gaussian
        if kind not in ('jump', 'diffusion'):
            raise MasterEquationError("kind must be 'jump' or 'diffusion'.")
        rates = self._string_rates()
        s = SDE(name)
        for k, v in {**self.params, 'h': self.h}.items():
            if k in self.fields:
                continue
            s.parameter(k, default=float(v), domain='positive' if v > 0 else domain)
        for f in self.fields:
            s.field(f)
        if kind == 'jump':
            for f in self.fields:
                s.drift(f, '0')
            for t, r in zip(self.transitions, rates):
                s.noise(t.name, Poisson(rate=r, size='h'),
                        into={f: j for f, j in zip(self.fields, t.jump) if j})
        else:
            for i, f in enumerate(self.fields):
                terms = [f'({t.jump[i]})*({r})' for t, r in zip(self.transitions, rates)
                         if t.jump[i]]
                s.drift(f, '0' if not terms else 'h*(' + ' + '.join(terms) + ')')
            for t, r in zip(self.transitions, rates):
                s.noise(t.name, Gaussian(var=f'h^2*({r})'),
                        into={f: j for f, j in zip(self.fields, t.jump) if j})
        return s

    def compare_with_daedalus(self, model=None, *, ks=(1, 2), max_ell=1, parameters=None,
                              kind='jump', config=None, verbose=False) -> ComparisonResult:
        """Run ``compute_cumulants`` on the SDE-front-end model and tabulate it
        against the exact cumulants, cumulant by cumulant.

        ``ks`` are the orders (1..3); ``max_ell`` the loop order.  ``kind`` is
        the model :meth:`to_sde` builds (``'jump'`` is the one the loop
        expansion is asymptotic for; ``'diffusion'`` compares with the
        Kramers-Moyal limit, ``kmax <= 2`` only).  The returned rows hold, per
        independent entry of ``kappa_k``: the exact value, the partial sums
        over loop orders ``0..max_ell`` and their relative errors.  The mean
        is ``x* + sum_ell C^(ell)`` (the run returns fluctuations about the
        saddle)."""
        import daedalus as dd
        if not set(ks) <= {1, 2, 3}:
            raise MasterEquationError('compare_with_daedalus supports ks within 1..3.')
        if model is None:
            model = self.to_sde(kind=kind).build()
        elif not isinstance(model, dict) and callable(getattr(model, 'build', None)):
            model = model.build()
        pars = dict(self.params)
        pars['h'] = self.h
        if 'h' not in (parameters or {}):
            from api.naming import free_parameter_names
            if 'h' not in set(free_parameter_names(model)):
                pars.pop('h')               # a model written in the system size has no 'h'
        pars.update(parameters or {})
        cr = self.cumulants(max(ks))
        notes = [
            'The loop expansion is asymptotic in the jump size h: at max_ell = L its relative '
            'error is O(h^(L+1)) at fixed x; it should converge to the exact value as h -> 0 '
            'and need not match at fixed h.',
            'It does not see the lattice, state-space boundaries (x = 0 for counts), or '
            'bistability; the oracle does (check truncation_flux and the mean-field saddle).',
        ]
        if cr.truncation_flux > self.truncation_tol:
            notes.append(f'exact values carry box truncation flux {cr.truncation_flux:.2e}.')
        rows = []
        for k in ks:
            for ix in itertools.combinations_with_replacement(range(self.d), k):
                ext = [('d' + self.fields[i], 1) for i in ix]
                res = dd.run(model, dd.Config(
                    k=k, max_ell=max_ell, external_fields=ext, parameters=pars,
                    tau_grid=np.array([0.0]), parallel=False, verbose=verbose,
                    **(config or {})))
                by = {int(l): float(np.real(np.atleast_1d(v)[0]))
                      for l, v in res['C_tau_by_ell'].items()}
                base = 0.0
                if k == 1:
                    base = float(np.asarray(res['mf_values'][self.fields[ix[0]] + 'star'])
                                 .ravel()[0])
                part, run = {}, base
                for l in range(max_ell + 1):
                    run += by.get(l, 0.0)
                    part[l] = run
                ex = float(cr.cumulants[k][ix])
                # a cumulant that is zero by symmetry is measured against the
                # largest one of its order, not itself
                scale = max(abs(ex), 1e-6 * float(np.abs(cr.cumulants[k]).max()), 1e-300)
                label = f'kappa_{k}(' + ','.join(self.fields[i] for i in ix) + ')'
                rows.append(dict(cumulant=label, exact=ex, loop=part,
                                 error={l: part[l] - ex for l in part},
                                 rel_error={l: abs(part[l] - ex) / scale for l in part}))
        return ComparisonResult(rows=rows, h=self.h, max_ell=max_ell, notes=notes)


def master_equation(transitions, **kwargs) -> MasterEquation:
    """Build a :class:`MasterEquation` (exposed as ``dd.master_equation``);
    see the class for the arguments."""
    return MasterEquation(transitions, **kwargs)

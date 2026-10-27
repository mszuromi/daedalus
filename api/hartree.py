"""
api.hartree
===========
Self-consistent Gaussian (Hartree) resummation of a temporal model
(``dd.hartree``, ``dd.Config(resum='hartree')``).

What it computes
----------------
For an Itô SDE ``dx = f(x) dt + noise`` with white noise of covariance
``Q(x) δ(t - t')`` (the MSR action ``S = xtᵀ (Dt x - f(x)) - ½ xtᵀ Q(x) xt
+ ...``), the self-consistent Gaussian approximation replaces the process by
the Gaussian one whose mean ``m`` and covariance ``Σ`` solve

    ⟨f⟩ = 0,          A Σ + Σ Aᵀ + ⟨Q⟩ = 0,        A = ⟨∂f/∂x⟩,

with every ``⟨·⟩`` taken in the Gaussian ``N(m, Σ)`` itself.  In diagram
language this is the Hartree(-Fock) truncation of the 2PI effective action:
the propagator is dressed by the tadpole (and the noise vertex by its
equal-time loop), each closed loop built from the dressed lines, and the
mean is shifted by the dressed one-point tadpoles; all daisy / cactus
diagrams are summed.  Response legs contracted inside one vertex vanish
(Itô, ``R(0) = 0``), so action terms with three or more response fields
(the higher jump cumulants of a Poisson source) do not enter.  By Stein's
lemma this is the cumulant-neglect (Gaussian) moment closure.  The lag
correlator is one matrix exponential, ``C(τ) = Σ e^{Aᵀτ}`` for ``τ >= 0``.

What it misses: every non-cactus diagram (from two loops on: the sunset),
every connected cumulant of order three or more (``κ_3 = κ_4 = 0``), the
multi-exponential shape of ``C(τ)`` and activated hopping between wells.
:mod:`api.resum` builds on it a screened loop series (optimized perturbation
theory) that adds those back order by order.

Expectations are taken over the *nonlinear* fields only: a field that enters
no monomial of degree two or more of the drift and none of the noise (a linear
filter stage, say) has an exactly known contribution (its mean), so it adds no
dimension.  Over the nonlinear fields the expectation is a tensor Gauss-Hermite
quadrature, exact for a polynomial drift and noise of the expansion's degree,
or -- for a polynomial system whose nonlinear dimension makes the tensor rule
large -- the closed-form Gaussian moments (Isserlis / Wick), which need no
quadrature at all.

Scope
-----
Temporal models whose action has a first-order ``Dt`` pencil
(``M Dt x``, ``M`` constant and invertible), white noise, no kernels
(``Conv``, delays), no non-local cumulant kernels and no time derivatives
inside vertices.  Colored noise enters through its Markovian embedding, which
``FieldTheory`` performs for ``declare_cgf_term`` exponential kernels.  A
drift that is not a polynomial in the fields is replaced by its Taylor
polynomial of ``taylor_order`` about the mean-field saddle, with an advisory.
Jump (Poisson) sources keep their first two jump moments: Hartree is then the
Gaussian closure of the master equation.

Entry points: :func:`hartree` (a model) and :func:`gaussian_closure`
(numeric drift / Jacobian / noise callables).
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sl
from scipy.optimize import root

from engine.errors import DaedalusError

__all__ = ['hartree', 'gaussian_closure', 'HartreeResult', 'HartreeError',
           'HartreeUnsupported', 'HartreeBistabilityWarning',
           'HartreeAdvisoryWarning', 'action_system']

#: tensor quadrature points above which the closure refuses (the rule has
#: ``(degree // 2 + 2) ** n_nonlinear`` points)
MAX_QUADRATURE_POINTS = 2_000_000

#: ``expectations='auto'`` uses the closed-form Gaussian moments of a
#: polynomial system once the tensor rule over its nonlinear fields would
#: have more points than this
AUTO_QUADRATURE_POINTS = 20_000

#: two closure solutions are the same when their means and covariances agree
#: to this fraction of the solutions' scale
_SAME_SOLUTION_RTOL = 1e-6

#: default Taylor order of a drift that is not a polynomial in the fields
DEFAULT_TAYLOR_ORDER = 6


class HartreeError(DaedalusError, RuntimeError):
    """The closure has no stable self-consistent solution from any start."""


class HartreeUnsupported(DaedalusError, NotImplementedError):
    """The model is outside the closure's scope (kernels, spatial, delays,
    derivative vertices, algebraic constraints, too many fields)."""


class HartreeBistabilityWarning(UserWarning):
    """The closure has several stable solutions: the model is multistable
    and a single Gaussian describes one basin only."""


class HartreeAdvisoryWarning(UserWarning):
    """A Hartree advisory other than bistability (e.g. a Taylor-truncated
    drift)."""


# ── result ──────────────────────────────────────────────────────────────────

@dataclass
class HartreeResult:
    """A converged self-consistent Gaussian.

    ``fields`` are the ring names of the physical fluctuation fields
    (``'dx1'``), ``legs`` the same as ``(field, index)`` tuples
    (``('dx', 1)``, the ``external_fields`` form).  ``mean`` is absolute
    (saddle plus the Hartree shift), ``cov`` the equal-time covariance ``Σ``,
    ``A`` the averaged Jacobian (the dressed inverse propagator is
    ``Dt - A``) and ``Q`` the averaged noise covariance.  ``J0`` and ``Q0``
    are the bare (tree-level) Jacobian and noise at the saddle.
    ``solutions`` lists every distinct stable solution the multi-start found
    (this one first); ``advisories`` the notes of the solve (``code``,
    ``message``).
    """
    fields: list
    mean: np.ndarray
    cov: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    saddle: np.ndarray
    converged: bool
    residual: float
    iterations: int
    legs: list = field(default_factory=list)
    naturals: list = field(default_factory=list)
    J0: np.ndarray = None
    Q0: np.ndarray = None
    solutions: list = field(default_factory=list)
    advisories: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    # ── indexing ──
    def index(self, leg):
        """The position of ``leg`` (an int, a ring name ``'dx1'`` or an
        ``external_fields`` tuple ``('dx', 1)``) among :attr:`fields`."""
        if isinstance(leg, (int, np.integer)):
            if not 0 <= int(leg) < len(self.fields):
                raise IndexError(f'hartree: field index {leg} out of range '
                                 f'(fields {self.fields})')
            return int(leg)
        if isinstance(leg, str) and leg in self.fields:
            return self.fields.index(leg)
        key = tuple(leg) if isinstance(leg, (list, tuple)) else (leg, 1)
        for n, lg in enumerate(self.legs):
            if lg == key or (lg[0], lg[1] or 1) == key:
                return n
        for n, (nat, lg) in enumerate(zip(self.naturals, self.legs)):
            if nat is not None and (nat, lg[1] or 1) == key:
                return n
        raise KeyError(f'hartree: {leg!r} is not a physical field leg of the '
                       f'closure (legs {self.legs})')

    # ── lag correlator ──
    def C_matrix(self, tau):
        """``Σ(τ)[n, i, j] = κ(x_i(t), x_j(t + τ_n))`` on an array of lags
        (either sign): ``Σ e^{Aᵀτ}`` for ``τ >= 0``, ``e^{A|τ|} Σ`` below."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.empty((tau.size, len(self.fields), len(self.fields)))
        for n, t in enumerate(tau.ravel()):
            if t >= 0:
                out[n] = self.cov @ sl.expm(self.A.T * t)
            else:
                out[n] = sl.expm(self.A * (-t)) @ self.cov
        return out

    def C(self, tau, i=0, j=0):
        """``κ(x_i(t), x_j(t + τ))`` on an array of lags (either sign)."""
        return self.C_matrix(tau)[:, self.index(i), self.index(j)]

    def covariance_matrix(self, tau_grid=None):
        """The covariance matrix over every pair of fields in the layout of
        ``dd.covariance_matrix`` (``Sigma[n, i, j]``)."""
        tau = (np.array([0.0]) if tau_grid is None
               else np.atleast_1d(np.asarray(tau_grid, dtype=float)))
        S = self.C_matrix(tau)
        return {'Sigma': S, 'tau_grid': tau, 'fields': self.field_legs(),
                'ring_fields': list(self.fields), 'legs': list(self.legs),
                'method': 'hartree', 'mean': self.mean.copy()}

    def field_legs(self):
        """The fields as ``(natural name, population)`` pairs, the
        ``dd.covariance_matrix`` form."""
        nat = self.naturals or [None] * len(self.legs)
        return [(n if n is not None else f, i if i is not None else 1)
                for n, (f, i) in zip(nat, self.legs)]

    # ── spectra ──
    def spectrum(self, omega, i=0, j=0):
        """``S_ij(ω) = ∫ C_ij(τ) e^{+iωτ} dτ`` (the ``api.spectra``
        convention, no ``1/2π``), entry ``(i, j)`` of ``H Q Hᴴ`` with
        ``H = (-iω - A)^{-1}``."""
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        i, j = self.index(i), self.index(j)
        n = self.A.shape[0]
        I = np.eye(n)
        out = np.empty(omega.shape, dtype=complex)
        for k, w in enumerate(omega.ravel()):
            out.flat[k] = -(self.cov @ np.linalg.inv(self.A.T + 1j * w * I)
                            + np.linalg.inv(self.A - 1j * w * I) @ self.cov
                            )[i, j]
        return out

    def spectra(self, omega=None, T=None, i=0, j=0):
        """The spectral and integrated outputs of ``C_ij``, with the keys and
        conventions of ``dd.spectra`` (one order, ``method='exact'``):
        ``S(ω)``, ``S0``, ``∫_0^T C``, ``∫_{-T}^0 C``, the one-sided
        integrals to infinity, ``C0`` and ``tau_c``."""
        from api.spectra import resolve_spectra_options, CONVENTIONS
        opts = resolve_spectra_options({'omega': omega, 'T': T},
                                       'HartreeResult.spectra')
        ii, jj = self.index(i), self.index(j)
        n = self.A.shape[0]
        A, S = self.A, self.cov
        inf_pos = float((S @ np.linalg.solve(-A.T, np.eye(n)))[ii, jj])
        inf_neg = float((np.linalg.solve(-A, np.eye(n)) @ S)[ii, jj])
        om = opts['omega']
        Ts = opts['T']
        out = {'omega': om, 'S': (None if om is None
                                  else self.spectrum(om, ii, jj)),
               'S0': complex(inf_pos + inf_neg), 'T': Ts,
               'integral_inf': complex(inf_pos),
               'integral_inf_neg': complex(inf_neg), 'delta0': 0j,
               'C0': float(S[ii, jj]), 'method': 'exact',
               'conventions': CONVENTIONS, 'notes': [], 'status': 'ok'}
        if Ts is not None:
            pos, neg = [], []
            Z = np.zeros((n, n))
            for t in np.atleast_1d(Ts):
                # ∫_0^T e^{Bu} du: the top-right block of exp([[B, I], [0, 0]] T)
                Ep = sl.expm(np.block([[A.T, np.eye(n)], [Z, Z]]) * t)[:n, n:]
                En = sl.expm(np.block([[A, np.eye(n)], [Z, Z]]) * t)[:n, n:]
                pos.append((S @ Ep)[ii, jj])
                neg.append((En @ S)[ii, jj])
            out['integral'] = np.asarray(pos, dtype=complex)
            out['integral_neg'] = np.asarray(neg, dtype=complex)
        out['tau_c'] = (inf_pos / out['C0'] if out['C0'] != 0 else np.nan)
        return out

    @property
    def rates(self):
        """Decay rates (``-eig(A)``) of the dressed propagator."""
        return -np.linalg.eigvals(self.A)

    @property
    def mean_shift(self):
        """The Hartree mean minus the mean-field saddle."""
        return self.mean - self.saddle


# ── quadrature and the closure ─────────────────────────────────────────────

def _gh_rule(n_dim, n_nodes, names=None):
    """Tensor Gauss-Hermite nodes ``z`` (shape ``(P, n_dim)``) and weights
    for the standard normal ``N(0, I)``."""
    if n_nodes ** n_dim > MAX_QUADRATURE_POINTS:
        which = (f' ({", ".join(map(str, names))})' if names else '')
        raise HartreeUnsupported(
            f'hartree: the tensor quadrature over the {n_dim} nonlinear '
            f'fields{which} would need {n_nodes}^{n_dim} = '
            f'{n_nodes ** n_dim} points (limit {MAX_QUADRATURE_POINTS}); '
            + ('expectations="moments" evaluates a polynomial model in '
               'closed form instead.' if names is not None else
               'pass nonlinear= (the fields the callables depend on '
               'nonlinearly) to shrink the rule.'))
    z1, w1 = np.polynomial.hermite_e.hermegauss(n_nodes)
    w1 = w1 / w1.sum()
    Z = np.array(list(itertools.product(z1, repeat=n_dim)))
    W = np.prod(np.array(list(itertools.product(w1, repeat=n_dim))), axis=1)
    return Z, W


class _MomentTable:
    """Central moments ``E[y^e]`` of ``y ~ N(0, S)`` by the Stein recursion
    ``E[y_i y^f] = sum_j S_ij f_j E[y^(f - 1_j)]``, memoised."""

    def __init__(self, S):
        self.S = np.asarray(S, dtype=float)
        self._memo = {}

    def __call__(self, e):
        hit = self._memo.get(e)
        if hit is not None:
            return hit
        if sum(e) % 2:
            out = 0.0
        elif not any(e):
            out = 1.0
        else:
            i = next(k for k, v in enumerate(e) if v)
            f = list(e)
            f[i] -= 1
            out = 0.0
            for j, fj in enumerate(f):
                if fj and self.S[i, j] != 0.0:
                    g = list(f)
                    g[j] -= 1
                    out += self.S[i, j] * fj * self(tuple(g))
        self._memo[e] = out
        return out


def _shifted_moment(e, m, table, memo):
    """``E[prod_i (m_i + y_i)^e_i]`` for ``y ~ N(0, S)`` (``table``)."""
    hit = memo.get(e)
    if hit is not None:
        return hit
    idx = [i for i, v in enumerate(e) if v]
    total = 0.0
    for ks in itertools.product(*[range(e[i] + 1) for i in idx]):
        k = [0] * len(e)
        c = 1.0
        for i, ki in zip(idx, ks):
            k[i] = ki
            c *= math.comb(e[i], ki) * m[i] ** (e[i] - ki)
        if c != 0.0:
            total += c * table(tuple(k))
    memo[e] = total
    return total


def _chol_psd(S):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return V * np.sqrt(np.clip(w, 0.0, None))


def _hurwitz(A):
    return bool(np.max(np.linalg.eigvals(A).real) < 0)


def gaussian_closure(drift, jac, noise, n, *, mean0, cov0=None, degree=3,
                     tol=1e-12, max_iter=200, damping=0.5, fields=None,
                     saddle=None, nonlinear=None, moments=None):
    """Solve the self-consistent Gaussian closure for numeric callables.

    ``drift(X) -> (P, n)``, ``jac(X) -> (P, n, n)`` (``∂f_a/∂x_b``) and
    ``noise(X) -> (P, n, n)`` act on a batch of points ``X`` of shape
    ``(P, n)`` in the same coordinates as ``mean0``.  ``degree`` is the
    largest polynomial degree to integrate exactly (the quadrature uses
    ``degree // 2 + 2`` nodes per dimension).

    ``nonlinear`` (indices; default every field) are the fields the
    quadrature runs over.  The callables are evaluated with every other field
    at its mean, which is exact when those fields enter the drift only
    affinely and the noise not at all (a linear filter stage).  The tensor
    rule then has ``(degree // 2 + 2) ** len(nonlinear)`` points.  ``moments``
    -- ``(mean, cov) -> (<f>, <df/dx>, <Q>)`` -- replaces the quadrature
    altogether (the closed-form Gaussian moments of a polynomial system).

    The iteration is a damped Newton step on the mean with the averaged
    Jacobian and a Lyapunov solve for the covariance, then a polish by
    ``scipy.optimize.root`` on the full residual.  Raises
    :class:`HartreeError` when the dressed drift stops being stable or no
    solution is found.
    """
    nl = (list(range(n)) if nonlinear is None
          else sorted({int(i) for i in nonlinear}))
    if moments is None:
        names = ([(fields or range(n))[i] for i in nl]
                 if nonlinear is not None else None)
        Z, W = _gh_rule(len(nl), max(2, degree // 2 + 2), names=names)
    else:
        Z, W = None, np.ones(1)
    mean = np.asarray(mean0, dtype=float).copy()
    if cov0 is None:
        cov0 = np.eye(n)
    cov = np.asarray(cov0, dtype=float).copy()

    def averages(m, S):
        if moments is not None:
            return moments(m, S)
        if len(nl) == n:
            X = m + Z @ _chol_psd(S).T
        else:
            X = np.repeat(m[None, :], len(W), axis=0)
            X[:, nl] = m[nl] + Z @ _chol_psd(S[np.ix_(nl, nl)]).T
        return (W @ drift(X), np.einsum('p,pab->ab', W, jac(X)),
                np.einsum('p,pab->ab', W, noise(X)))

    def lyap(A, Q):
        if not _hurwitz(A):
            raise HartreeError('hartree: the dressed drift is not stable '
                               '(eig(A) has a non-negative real part).')
        return sl.solve_continuous_lyapunov(A, -Q)

    # A start whose averaged drift is unstable (an unstable saddle, a small
    # guess) is widened until the fluctuations stabilise it.
    for _ in range(60):
        if _hurwitz(averages(mean, cov)[1]):
            break
        cov = 2.0 * cov

    it = 0
    res = np.inf
    for it in range(1, max_iter + 1):
        fbar, A, Q = averages(mean, cov)
        new_mean = mean - np.linalg.solve(A, fbar)
        new_cov = lyap(A, Q)
        step = (np.max(np.abs(new_mean - mean), initial=0.0)
                + np.max(np.abs(new_cov - cov)))
        mean = mean + damping * (new_mean - mean)
        cov = cov + damping * (new_cov - cov)
        cov = 0.5 * (cov + cov.T)
        res = step
        if not np.all(np.isfinite(cov)):
            raise HartreeError('hartree: the closure iteration diverged.')
        if step < tol * (1.0 + np.max(np.abs(cov))):
            break

    iu = np.triu_indices(n)

    def pack(m, S):
        return np.concatenate([m, S[iu]])

    def unpack(v):
        m = v[:n]
        S = np.zeros((n, n))
        S[iu] = v[n:]
        S = S + np.triu(S, 1).T
        return m, S

    def residual(v):
        m, S = unpack(v)
        fbar, A, Q = averages(m, S)
        L = A @ S + S @ A.T + Q
        return np.concatenate([fbar, L[iu]])

    sol = root(residual, pack(mean, cov), method='hybr', tol=1e-14)
    if sol.success or np.max(np.abs(sol.fun)) < np.max(np.abs(residual(
            pack(mean, cov)))):
        mean, cov = unpack(sol.x)
    fbar, A, Q = averages(mean, cov)
    final = float(np.max(np.abs(residual(pack(mean, cov)))))
    if not np.all(np.isfinite(cov)) or np.min(
            np.linalg.eigvalsh(0.5 * (cov + cov.T))) <= 0:
        raise HartreeError('hartree: the covariance is not positive '
                           'definite at the solution.')
    if not _hurwitz(A):
        raise HartreeError('hartree: no stable self-consistent solution '
                           'from this start.')
    scale = 1.0 + np.max(np.abs(cov)) + np.max(np.abs(Q))
    return HartreeResult(
        fields=list(fields or range(n)), mean=mean, cov=cov, A=A, Q=Q,
        saddle=(np.zeros(n) if saddle is None else np.asarray(saddle)),
        converged=final < 1e-9 * scale, residual=final, iterations=it,
        info={'quadrature_nodes': (len(W) if moments is None else 0),
              'nonlinear_dim': len(nl),
              'expectations': ('quadrature' if moments is None
                               else 'moments'),
              'fixed_point_step': res})


# ── model adapter ──────────────────────────────────────────────────────────

@dataclass
class _Poly:
    """A real polynomial in ``n`` variables: monomial exponents
    ``(m, n)`` and coefficients ``(m,)``, evaluated on a batch."""
    exps: np.ndarray
    coeffs: np.ndarray

    def __call__(self, X):
        if self.coeffs.size == 0:
            return np.zeros(len(X))
        return np.prod(X[:, None, :] ** self.exps[None, :, :],
                       axis=2) @ self.coeffs

    def d(self, b):
        e = self.exps[:, b]
        keep = e > 0
        exps = self.exps[keep].copy()
        exps[:, b] -= 1
        return _Poly(exps, self.coeffs[keep] * e[keep])


def _to_poly(expr, xs, what):
    from sage.all import SR, RDF, PolynomialRing
    R = PolynomialRing(RDF, [str(x) for x in xs])
    e = SR(expr).expand()
    try:
        p = e.polynomial(ring=R) if e != 0 else R.zero()
    except (TypeError, ValueError, AttributeError) as exc:
        extra = sorted(str(v) for v in SR(expr).variables()
                       if str(v) not in {str(x) for x in xs})
        raise HartreeUnsupported(
            f'hartree: the {what} is not a numeric polynomial in the fields'
            + (f' (unresolved symbols {", ".join(extra)})' if extra else '')
            + f': {exc}') from exc
    d = p.dict()
    n = len(xs)
    if not d:
        return _Poly(np.zeros((0, n), dtype=int), np.zeros(0))
    exps = np.array([list(k) if n > 1 else [int(k)] for k in d.keys()],
                    dtype=int).reshape(len(d), n)
    return _Poly(exps, np.array([float(v) for v in d.values()]))


@dataclass
class ActionSystem:
    """The drift and noise of a temporal model about its mean-field saddle,
    read off the expanded action (``FieldTheory``).

    ``fields``/``legs``: the physical fluctuation fields; ``f[a]`` and
    ``Q[a][b]`` polynomials in them (``x``-coordinates: ``f = M^{-1} g``,
    ``Q = M^{-1} Q_g M^{-T}`` for the action ``xtᵀ (M Dt dx - g) - ½ xtᵀ Q_g
    xt``); ``M`` the numeric ``Dt`` pencil; ``saddle`` the selected saddle
    (absolute), ``other_saddles`` the other mean-field roots as offsets from
    it; ``degree`` the polynomial degree; ``polynomial`` whether the drift and
    noise are exact polynomials (no Taylor truncation)."""
    fields: list
    legs: list
    naturals: list
    response: list
    f: list
    Q: list
    M: np.ndarray
    saddle: np.ndarray
    other_saddles: list
    degree: int
    polynomial: bool
    taylor_order: int
    mf: dict
    num_params: dict
    _terms: tuple = field(default=None, repr=False)

    @property
    def nonlinear(self):
        """Indices of the fields that enter some monomial of degree two or
        more of the drift, or any monomial of the noise: the dimensions the
        Gaussian expectations really have.  Every other field enters the
        drift only affinely and the noise not at all, so it contributes
        through its mean."""
        hit = set()
        for p in self.f:
            for e, c in zip(p.exps, p.coeffs):
                if c != 0 and e.sum() >= 2:
                    hit.update(np.flatnonzero(e).tolist())
        for row in self.Q:
            for p in row:
                for e, c in zip(p.exps, p.coeffs):
                    if c != 0 and e.sum() >= 1:
                        hit.update(np.flatnonzero(e).tolist())
        return sorted(hit)

    def expectations(self, m, S):
        """``(<f>, <df/dx>, <Q>)`` in ``N(m, S)`` by the closed-form Gaussian
        (Isserlis / Wick) moments of the polynomial drift and noise."""
        n = len(self.fields)
        if self._terms is None:
            self._terms = (
                [(a, p) for a, p in enumerate(self.f)],
                [(a, b, self.f[a].d(b)) for a in range(n) for b in range(n)],
                [(a, b, self.Q[a][b]) for a in range(n) for b in range(n)])
        table = _MomentTable(S)
        memo = {}

        def ev(p):
            return sum((c * _shifted_moment(tuple(int(v) for v in e), m,
                                            table, memo)
                        for e, c in zip(p.exps, p.coeffs)), 0.0)
        fbar = np.zeros(n)
        A = np.zeros((n, n))
        Q = np.zeros((n, n))
        for a, p in self._terms[0]:
            fbar[a] = ev(p)
        for a, b, p in self._terms[1]:
            A[a, b] = ev(p)
        for a, b, p in self._terms[2]:
            Q[a, b] = ev(p)
        return fbar, A, Q

    def drift(self, X):
        return np.stack([p(X) for p in self.f], axis=1)

    def jac(self, X):
        n = len(self.fields)
        return np.stack([np.stack([self.f[a].d(b)(X) for b in range(n)],
                                  axis=1) for a in range(n)], axis=1)

    def noise(self, X):
        n = len(self.fields)
        return np.stack([np.stack([self.Q[a][b](X) for b in range(n)],
                                  axis=1) for a in range(n)], axis=1)

    @property
    def J0(self):
        return self.jac(np.zeros((1, len(self.fields))))[0]

    @property
    def Q0(self):
        return self.noise(np.zeros((1, len(self.fields))))[0]


def refusal_reason(model):
    """Why the closure (and the resummations built on it) cannot treat
    ``model`` -- a message -- or ``None``.  Checked before any expansion."""
    if model.get('spatial') or model.get('operator_ir'):
        return ('spatial models are not supported (the spatial Hartree '
                'tadpole needs a mass counterterm for d >= 2)')
    if model.get('kernels'):
        return ('models with kernels (Conv, delays) are not supported: embed '
                'the kernel states as fields first')
    if model.get('iteration_saddles'):
        return ('models with iteration saddles are not supported')
    return None


def _phys_layout(model, ns):
    """The physical fields in ring order: ``[(ring name, field, index,
    saddle parameter name or None, natural name or None)]``."""
    from engine.core.vertices import field_symbol_name
    conv = model.get('naming_convention') or {}
    saddles = conv.get('mean_field_saddles') or {}
    fluct = {v: k for k, v in (conv.get('fluctuation_fields') or {}).items()}
    out = []
    for fspec in model['physical_fields']:
        name = fspec['name']
        natural = fspec.get('natural_name') or fluct.get(name)
        sad = saddles.get(natural) if natural else None
        arr = getattr(ns, name)
        if isinstance(arr, list):
            for i in range(len(arr)):
                out.append((field_symbol_name(name, i + 1), name, i + 1, sad,
                            natural))
        else:
            out.append((field_symbol_name(name), name, None, sad, natural))
    return out


def _saddle_vector(layout, values):
    vec = []
    for _ring, _f, idx, sad, _nat in layout:
        v = (values or {}).get(sad) if sad else None
        if v is None:
            vec.append(0.0)
        elif isinstance(v, (list, tuple, np.ndarray)):
            vec.append(float(np.real(v[(idx or 1) - 1])))
        else:
            vec.append(float(np.real(v)))
    return np.array(vec)


def _physical_degree(model, ft_cls, taylor_probe=2):
    """``(polynomial, degree)`` of the action in the physical fields, read
    from the un-expanded action (``None`` degree when it cannot be read)."""
    from sage.all import SR
    try:
        probe = ft_cls(model, taylor_order=taylor_probe)
        ns, _R, _nt = probe._build_namespace()
        S = SR(model['action'](ns)).expand()
        phys = list(ns._phys_sr_vars)
        if not all(S.is_polynomial(v) for v in phys):
            return False, None
        t = SR.var('_hartree_degree_t')
        return True, int(S.subs({v: t * v for v in phys}).expand().degree(t))
    except Exception:                    # noqa: BLE001 -- probe only
        return False, None


def action_system(model, parameters=None, *, fixed_point_index=0,
                  taylor_order=None, mf_dae_n_starts=64, mf_dae_seed_box=None):
    """The drift and noise of a temporal model about its mean-field saddle
    (an :class:`ActionSystem`), with every parameter substituted.  Raises
    :class:`HartreeUnsupported` outside the closure's scope."""
    from sage.all import SR, matrix
    from engine.core.field_theory import FieldTheory
    from api._mean_field import solve_mean_field, complete_parameters

    why = refusal_reason(model)
    if why is not None:
        raise HartreeUnsupported('hartree: ' + why + '.')
    polynomial, deg = _physical_degree(model, FieldTheory)
    if taylor_order is None:
        order = max(4, deg + 2) if polynomial and deg is not None \
            else DEFAULT_TAYLOR_ORDER
    else:
        order = int(taylor_order)
    exact = bool(polynomial and deg is not None and order >= deg + 2)
    ft = FieldTheory(model, taylor_order=order)
    ft.expand()
    if getattr(ft._ns, '_cumulant_kernels', None):
        raise HartreeUnsupported(
            'hartree: non-local noise cumulant kernels (correlated_noises with '
            'a smooth kernel) are not supported; declare the noise through '
            'its Markovian embedding.')
    fundamental = complete_parameters(model, parameters or {}, strict=False)
    mf = solve_mean_field(ft, model, fundamental, verbose=False,
                          fixed_point_index=fixed_point_index,
                          n_starts=mf_dae_n_starts, seed_box=mf_dae_seed_box)
    num = dict(mf['num_params'])

    names = list(ft._ns._ring_var_names)
    nt = ft._n_tilde
    resp, phys = names[:nt], names[nt:]
    if len(resp) != len(phys):
        raise HartreeUnsupported(
            'hartree: the action needs one response field per physical field.')
    Dt = SR.var('Dt')
    xs = [SR.var(p) for p in phys]
    n = len(phys)
    M = [[SR(0)] * n for _ in range(n)]
    g = [SR(0)] * n
    Qg = [[SR(0)] * n for _ in range(n)]
    degree = 1
    for (ntil, nphys), poly in ft._by_tp.items():
        if poly == 0 or ntil >= 3 or ntil == 0:
            continue        # Itô: R(0) = 0, no Hartree contribution
        for exps, coeff in poly.dict().items():
            c = SR(coeff)
            ex_r, ex_p = exps[:nt], exps[nt:]
            mono = SR(1)
            for x, e in zip(xs, ex_p):
                mono = mono * x ** e
            if ntil == 1:
                a = list(ex_r).index(1)
                cd = c.coefficient(Dt, 1)
                if c.coefficient(Dt, 2) != 0:
                    raise HartreeUnsupported(
                        'hartree: second time derivatives are not supported.')
                if cd != 0:
                    if sum(ex_p) != 1:
                        raise HartreeUnsupported(
                            'hartree: time derivatives inside vertices are '
                            'not supported.')
                    b = list(ex_p).index(1)
                    M[a][b] = M[a][b] + cd
                    c = c.subs({Dt: 0})
                g[a] = g[a] - c * mono
                degree = max(degree, sum(ex_p))
            else:
                if c.has(Dt):
                    raise HartreeUnsupported(
                        'hartree: time derivatives in the noise vertex are '
                        'not supported.')
                idx = [i for i, e in enumerate(ex_r) for _ in range(e)]
                a, b = idx
                if a == b:
                    Qg[a][a] = Qg[a][a] - 2 * c * mono
                else:
                    Qg[a][b] = Qg[a][b] - c * mono
                    Qg[b][a] = Qg[b][a] - c * mono
                degree = max(degree, sum(ex_p) + 1)
    try:
        Mn = np.array([[float(SR(M[a][b]).subs(num)) for b in range(n)]
                       for a in range(n)])
    except TypeError as exc:
        raise HartreeUnsupported(
            'hartree: the Dt pencil is not a numeric matrix (a field- or '
            'kernel-dependent time derivative coefficient).') from exc
    if abs(np.linalg.det(Mn)) < 1e-300 or np.linalg.cond(Mn) > 1e14:
        raise HartreeUnsupported(
            'hartree: singular Dt pencil (algebraic constraints are not '
            'supported).')
    Minv = np.linalg.inv(Mn)
    gp = [_to_poly(SR(gi).subs(num), xs, 'drift') for gi in g]
    Qp = [[_to_poly(SR(Qg[a][b]).subs(num), xs, 'noise') for b in range(n)]
          for a in range(n)]
    f = [_combine([(Minv[a, c], gp[c]) for c in range(n)], n)
         for a in range(n)]
    Qx = [[_combine([(Minv[a, c] * Minv[b, d], Qp[c][d]) for c in range(n)
                     for d in range(n)], n) for b in range(n)]
          for a in range(n)]
    layout = _phys_layout(model, ft._ns)
    saddle = _saddle_vector(layout, mf.get('saddle_values'))
    others = []
    for r in mf.get('mf_all_roots') or []:
        v = _saddle_vector(layout, r.get('values'))
        if np.max(np.abs(v - saddle), initial=0.0) > 1e-9 * (
                1 + np.max(np.abs(saddle), initial=0.0)):
            others.append({'offset': v - saddle,
                           'stable': bool(r.get('stable'))})
    return ActionSystem(
        fields=[r[0] for r in layout], legs=[(r[1], r[2]) for r in layout],
        naturals=[r[4] for r in layout], response=resp, f=f, Q=Qx, M=Mn, saddle=saddle, other_saddles=others,
        degree=int(degree), polynomial=exact, taylor_order=order, mf=mf,
        num_params=num)


def _combine(terms, n):
    """``Σ c_k p_k`` of polynomials, monomials merged."""
    acc = {}
    for c, p in terms:
        if c == 0:
            continue
        for e, v in zip(map(tuple, p.exps), p.coeffs):
            acc[e] = acc.get(e, 0.0) + c * v
    acc = {e: v for e, v in acc.items() if v != 0}
    if not acc:
        return _Poly(np.zeros((0, n), dtype=int), np.zeros(0))
    return _Poly(np.array(list(acc.keys()), dtype=int).reshape(len(acc), n),
                 np.array(list(acc.values())))


def _same(a, b):
    scale = 1.0 + max(np.max(np.abs(a.cov)), np.max(np.abs(b.cov)),
                      np.max(np.abs(a.mean)), np.max(np.abs(b.mean)))
    return (np.max(np.abs(a.mean - b.mean), initial=0.0)
            + np.max(np.abs(a.cov - b.cov))) < _SAME_SOLUTION_RTOL * scale


def _summary(r, label):
    return {'mean': r.mean.copy(), 'cov': r.cov.copy(),
            'rates': np.sort_complex(r.rates), 'start': label,
            'converged': r.converged}


def hartree(model, parameters=None, *, fixed_point_index=0, start=None,
            mean0=None, cov0=None, taylor_order=None, multistart=True,
            mf_dae_n_starts=64, mf_dae_seed_box=None, warn=True,
            expectations='auto', **kw):
    """Self-consistent Gaussian (Hartree) approximation of a temporal model
    (``dd.hartree``).

    Parameters
    ----------
    model : dict (or an unbuilt ``api.sde.SDE`` / builder)
    parameters : dict
        The model's parameters (omitted ones take their defaults).
    fixed_point_index : int
        The mean-field saddle the drift is expanded about (and the first
        start of the closure).
    start : dict, optional
        ``{'mean': absolute mean, 'cov': covariance}`` overriding the first
        start (``mean0`` / ``cov0`` are the same as keywords).
    taylor_order : int, optional
        Taylor order of a non-polynomial drift (default: exact for a
        polynomial action, else 6 with the advisory
        ``HARTREE_TAYLOR_TRUNCATED``).
    multistart : bool
        Also start from every other mean-field root (stable or not) and
        collect every distinct stable solution in ``.solutions``.  Several
        solutions give the advisory ``HARTREE_MULTIPLE_SOLUTIONS`` and a
        :class:`HartreeBistabilityWarning`: the model is multistable, the
        Gaussian describes one basin, and hopping between basins is missing.
    warn : bool
        Issue the advisories as warnings (they are always in
        ``.advisories``).
    expectations : {'auto', 'quadrature', 'moments'}
        How the Gaussian expectations are taken.  Only the fields that enter
        nonlinearly (``.info['nonlinear_fields']``) are integrated over: the
        others -- linear filter stages -- contribute through their mean.
        ``'quadrature'`` is the tensor Gauss-Hermite rule over those fields
        (refused above ``MAX_QUADRATURE_POINTS`` points); ``'moments'`` the
        closed-form Gaussian (Wick) moments of the polynomial drift and
        noise, with no quadrature and no size limit; ``'auto'`` (default)
        quadrature while the rule has at most ``AUTO_QUADRATURE_POINTS``
        points, the moments beyond.

    Returns
    -------
    HartreeResult
        The solution reached from the first start (the selected saddle, or
        ``start``); when that start does not converge, the first solution of
        the others, with the advisory ``HARTREE_START_FAILED``.
    """
    if not isinstance(model, dict) and callable(getattr(model, 'build', None)):
        model = model.build()
    if start is not None:
        mean0 = start.get('mean', mean0)
        cov0 = start.get('cov', cov0)
    sysm = action_system(model, parameters or {},
                         fixed_point_index=fixed_point_index,
                         taylor_order=taylor_order,
                         mf_dae_n_starts=mf_dae_n_starts,
                         mf_dae_seed_box=mf_dae_seed_box)
    n = len(sysm.fields)
    saddle = sysm.saddle
    deg = max(3, sysm.degree)
    if expectations not in ('auto', 'quadrature', 'moments'):
        raise ValueError(f"hartree: expectations must be 'auto', "
                         f"'quadrature' or 'moments', not {expectations!r}")
    nl = sysm.nonlinear
    points = max(2, deg // 2 + 2) ** len(nl)
    closed_form = (expectations == 'moments'
                   or (expectations == 'auto'
                       and points > AUTO_QUADRATURE_POINTS))
    closure_kw = ({'moments': sysm.expectations} if closed_form
                  else {'nonlinear': nl})

    def tree_cov(offset):
        A0 = sysm.jac(offset[None, :])[0]
        Q0 = sysm.noise(offset[None, :])[0]
        if _hurwitz(A0):
            try:
                S = sl.solve_continuous_lyapunov(A0, -Q0)
                if np.min(np.linalg.eigvalsh(0.5 * (S + S.T))) > 0:
                    return S
            except (np.linalg.LinAlgError, ValueError):
                pass
        return np.eye(n) * max(1e-3, float(np.max(np.abs(Q0))))

    starts = []
    m0 = (np.zeros(n) if mean0 is None
          else np.asarray(mean0, dtype=float).reshape(n) - saddle)
    starts.append(('start' if mean0 is not None else 'saddle', m0,
                   tree_cov(m0) if cov0 is None
                   else np.asarray(cov0, dtype=float).reshape(n, n)))
    if multistart:
        if mean0 is not None:
            starts.append(('saddle', np.zeros(n), tree_cov(np.zeros(n))))
        for k, o in enumerate(sysm.other_saddles):
            starts.append((f'mean-field root {k + 1}', o['offset'],
                           tree_cov(o['offset'])))
    found, failures = [], []
    first = None
    for label, m, c in starts:
        try:
            r = gaussian_closure(sysm.drift, sysm.jac, sysm.noise, n,
                                 mean0=m, cov0=c, degree=deg,
                                 fields=sysm.fields, saddle=saddle,
                                 **closure_kw, **kw)
        except (HartreeError, np.linalg.LinAlgError, ValueError) as exc:
            failures.append((label, str(exc)))
            continue
        if first is None and label == starts[0][0]:
            first = r
        if not any(_same(r, s) for s, _l in found):
            found.append((r, label))
    if not found:
        raise HartreeError(
            'hartree: no stable self-consistent Gaussian from any start ('
            + '; '.join(f'{lab}: {msg}' for lab, msg in failures) + ').')
    advisories = []
    if first is None:
        first = found[0][0]
        advisories.append({
            'code': 'HARTREE_START_FAILED',
            'message': (f'the closure did not converge from the first start '
                        f'({failures[0][1] if failures else "?"}); the '
                        f'solution from {found[0][1]} is returned')})
    # the returned solution first
    found.sort(key=lambda t: 0 if _same(t[0], first) else 1)
    for r, _l in found:
        r.mean = r.mean + saddle
    res = first
    res.legs = list(sysm.legs)
    res.naturals = list(sysm.naturals)
    res.J0, res.Q0 = sysm.J0, sysm.Q0
    res.solutions = [_summary(r, lab) for r, lab in found]
    if len(found) > 1:
        desc = '; '.join(
            f'mean {np.array2string(s["mean"], precision=4)}, variance '
            f'{np.array2string(np.diag(s["cov"]), precision=4)}'
            for s in res.solutions)
        advisories.append({
            'code': 'HARTREE_MULTIPLE_SOLUTIONS',
            'message': (f'the self-consistent Gaussian has {len(found)} '
                        f'stable solutions ({desc}): the model is likely '
                        f'multistable. One Gaussian describes one basin only; '
                        f'the hopping between basins (and its slow rate) is '
                        f'missing from every solution. Returned: the one '
                        f'reached from the {found[0][1]}.')})
    if mean0 is None and sysm.other_saddles:
        # the closure from the selected saddle settled nearer another
        # mean-field root (distances in the solution's own covariance)
        off = first.mean - saddle
        P = np.linalg.inv(first.cov)
        d0 = float(off @ P @ off)
        near = [o for o in sysm.other_saddles
                if float((off - o['offset']) @ P @ (off - o['offset'])) < d0]
        if near:
            advisories.append({
                'code': 'HARTREE_LEFT_SADDLE',
                'message': (f'the closure started at the selected saddle '
                            f'{np.array2string(saddle, precision=4)} settled '
                            f'at mean {np.array2string(first.mean, precision=4)}'
                            f', nearer the mean-field root '
                            f'{np.array2string(saddle + near[0]["offset"], precision=4)}'
                            f': the Gaussian describes that basin. The screened '
                            f'series built on it mixes the vertices of the '
                            f'selected saddle with this propagator; a '
                            f'non-polynomial drift is Taylor-expanded about '
                            f'the selected saddle')})
    if not sysm.polynomial:
        advisories.append({
            'code': 'HARTREE_TAYLOR_TRUNCATED',
            'message': (f'the drift or noise is not a polynomial in the '
                        f'fields; it is replaced by its Taylor polynomial of '
                        f'order {sysm.taylor_order} about the saddle (raise '
                        f'taylor_order to check)')})
    if not res.converged:
        advisories.append({
            'code': 'HARTREE_NOT_CONVERGED',
            'message': (f'the closure residual {res.residual:.2e} is above '
                        f'its tolerance')})
    res.advisories = advisories
    res.info.update({'taylor_order': sysm.taylor_order,
                     'polynomial': sysm.polynomial, 'degree': sysm.degree,
                     'pencil': sysm.M, 'starts': [s[0] for s in starts],
                     'nonlinear_fields': [sysm.fields[i] for i in nl],
                     'expectations': ('moments' if closed_form
                                      else 'quadrature'),
                     'failed_starts': failures,
                     'model': model.get('name', '<unnamed>')})
    if warn:
        for a in advisories:
            cls = (HartreeBistabilityWarning
                   if a['code'] == 'HARTREE_MULTIPLE_SOLUTIONS'
                   else HartreeAdvisoryWarning)
            warnings.warn(f"hartree [{a['code']}]: {a['message']}", cls,
                          stacklevel=2)
    return res


# ── dd.run(Config(resum='hartree')) ────────────────────────────────────────

def run_result(h, external_fields, k, tau_grid, *, spectra=None,
               covariance=None):
    """The result dict of ``dd.run`` with ``resum='hartree'``: ``C_tau`` of
    the requested pair (``k = 2``) or the mean shift (``k = 1``) on
    ``tau_grid``, the closure in ``resummation['hartree']``, optional
    ``spectra`` (the ``dd.spectra`` keys) or the covariance-matrix layout."""
    tau = np.atleast_1d(np.asarray(tau_grid, dtype=float))
    legs = [tuple(e) if isinstance(e, (list, tuple)) else e
            for e in external_fields]
    info = {'method': 'hartree', 'mean': h.mean.copy(), 'cov': h.cov.copy(),
            'A': h.A.copy(), 'Q': h.Q.copy(), 'rates': h.rates,
            'converged': h.converged, 'residual': h.residual,
            'solutions': h.solutions, 'advisories': h.advisories,
            'fields': list(h.fields), 'legs': list(h.legs)}
    if covariance is not None:
        sel = covariance.get('fields')
        cm = h.covariance_matrix(tau)
        if sel:
            ix = [_cov_index(h, f) for f in sel]
            cm['Sigma'] = cm['Sigma'][:, ix][:, :, ix]
            cm['fields'] = [cm['fields'][i] for i in ix]
            cm['ring_fields'] = [h.fields[i] for i in ix]
            cm['legs'] = [h.legs[i] for i in ix]
        cm['resummation'] = info
        cm['status'] = 'ok' if h.converged else 'not_converged'
        return cm
    out = {'tau_grid': tau, 'external_fields': legs, 'resummation': info,
           'status': 'ok' if h.converged else 'not_converged',
           'config': {'k': k, 'max_ell': 0, 'resum': 'hartree'}}
    if k == 1:
        i = h.index(legs[0])
        out['C_tau'] = np.full(tau.shape, h.mean_shift[i])
        out['mean'] = float(h.mean[i])
        return out
    i, j = h.index(legs[0]), h.index(legs[1])
    out['C_tau'] = h.C(tau, i, j)
    if spectra is not None:
        out['spectra'] = h.spectra(spectra.get('omega'), spectra.get('T'),
                                   i, j)
    return out


def _cov_index(h, f):
    """A ``covariance_matrix`` field selector (a natural or fluctuation name,
    a ring name or a leg tuple) as an index of ``h.fields``."""
    try:
        return h.index(f)
    except (KeyError, IndexError):
        pass
    raise KeyError(f'hartree: covariance field {f!r} is not one of '
                   f'{h.legs}')

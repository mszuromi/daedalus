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
(``M Dt x``, ``M`` constant and invertible), white noise, no non-local
cumulant kernels and no time derivatives inside vertices.  Kernels
(``Conv``) of the fields enter linearly: a rational kernel is embedded
exactly, before the closure, as linear state fields (the engine's
state-space realization of the kernel entry), and a hard delay through its
Pade approximant of an order chosen by the same search as the loop path
(``delay=`` options); the results are on the model's own fields, the kernel
states are in ``result.full``.  Colored noise enters through its Markovian embedding, which
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
import time
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
    #: models with kernels: the closure on the fields AND the kernel states
    #: (``mean``, ``cov``, ``A``, ``Q``, ``saddle``, ``J0``, ``Q0``,
    #: ``aux`` names); ``mean``/``cov``/``A``/``Q`` above are its block on the
    #: model's own fields.  ``None`` for a model without kernels.
    full: dict = None

    @property
    def _A_full(self):
        return self.A if self.full is None else self.full['A']

    @property
    def _cov_full(self):
        return self.cov if self.full is None else self.full['cov']

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
        nf = len(self.fields)
        A, S = self._A_full, self._cov_full
        out = np.empty((tau.size, nf, nf))
        for n, t in enumerate(tau.ravel()):
            if t >= 0:
                out[n] = (S @ sl.expm(A.T * t))[:nf, :nf]
            else:
                out[n] = (sl.expm(A * (-t)) @ S)[:nf, :nf]
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
        A, S = self._A_full, self._cov_full
        n = A.shape[0]
        I = np.eye(n)
        out = np.empty(omega.shape, dtype=complex)
        for k, w in enumerate(omega.ravel()):
            out.flat[k] = -(S @ np.linalg.inv(A.T + 1j * w * I)
                            + np.linalg.inv(A - 1j * w * I) @ S)[i, j]
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
        A, S = self._A_full, self._cov_full
        n = A.shape[0]
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
        """Decay rates (``-eig(A)``) of the dressed propagator (with the
        kernel states, for a model with kernels)."""
        return -np.linalg.eigvals(self._A_full)

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
    w_cov = np.linalg.eigvalsh(0.5 * (cov + cov.T)) if np.all(
        np.isfinite(cov)) else None
    # (a stiff kernel state can be numerically uncontrollable: a rounding-size
    # negative eigenvalue is a zero one)
    if w_cov is None or w_cov.min() < -1e-13 * max(w_cov.max(), 0.0) \
            or (w_cov.min() <= 0 and w_cov.max() <= 0):
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
    #: names of the kernel states appended after ``fields`` (a model with
    #: kernels), and what they realize
    aux: list = field(default_factory=list)
    kernel_info: dict = None
    _terms: tuple = field(default=None, repr=False)

    @property
    def n(self):
        """The dimension of the closure: the fields and the kernel states."""
        return len(self.f)

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
        n = self.n
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
        n = self.n
        return np.stack([np.stack([self.f[a].d(b)(X) for b in range(n)],
                                  axis=1) for a in range(n)], axis=1)

    def noise(self, X):
        n = self.n
        return np.stack([np.stack([self.Q[a][b](X) for b in range(n)],
                                  axis=1) for a in range(n)], axis=1)

    @property
    def J0(self):
        return self.jac(np.zeros((1, self.n)))[0]

    @property
    def Q0(self):
        return self.noise(np.zeros((1, self.n)))[0]


def refusal_reason(model):
    """Why the closure (and the resummations built on it) cannot treat
    ``model`` -- a message -- or ``None``.  Checked before any expansion."""
    if model.get('spatial') or model.get('operator_ir'):
        return ('spatial models are not supported (the spatial Hartree '
                'tadpole needs a mass counterterm for d >= 2)')
    if model.get('kernels') and model.get('kernel_ft_image') is None:
        return ('models with kernels but no frequency image of them are not '
                'supported (the kernel states are realized from the image)')
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


def _kernel_images(model, ns):
    """``{kernel symbol: frequency image}`` of a model's kernels (``{}`` for
    none): the hook every front end fills."""
    from sage.all import SR
    hook = model.get('kernel_ft_image')
    if hook is None or not model.get('kernels'):
        return {}
    return dict(hook(ns, SR.var('omega')) or {})


def _kernel_pencil(klin, n, images, num):
    """The linear part of a model with kernels, as a pencil plus kernel
    states: ``K(s) = K1 s + K0 + sum_blocks c (s - A_R)^-1 b`` for each entry
    ``(i, j)`` of ``klin`` (the coefficients of the linear monomials
    ``xt_i x_j``, with ``Dt`` and the kernel symbols).  This is the engine's
    own realization (:func:`api._realization._realize_proper`: an exact
    chain of first-order blocks per rate and Schwarz blocks per irreducible
    factor, no roots); the closure then runs on the fields and these states.

    Returns ``(K1, K0, blocks)``, ``blocks`` a list of ``(i, j, A, b, c)``
    (numpy; the states obey ``Dt z = A z + b x_j`` and feed row ``i`` of
    ``K1 Dt x + K0 x + c z = ...``)."""
    from sage.all import SR, matrix
    from api import _realization as R
    omega, Dt = SR.var('omega'), SR.var('Dt')
    K = matrix(SR, n, n)
    for (i, j), c in klin.items():
        K[i, j] = SR(c).subs(images).subs({Dt: SR.I() * omega})
    try:
        ents, F, P, fname = R._entries_to_field(K, omega, R._exact_subs(num))
    except R.RealizationRefused as exc:
        raise HartreeUnsupported(
            f'hartree: a kernel is not a rational function of the frequency '
            f'({exc}); a hard delay is embedded through its Pade '
            f'approximant (the delay= options), any other kernel needs a '
            f'rational frequency image.') from exc
    if fname != 'QQ':
        raise HartreeUnsupported(
            'hartree: kernels with complex coefficients are not supported.')
    K1, K0 = np.zeros((n, n)), np.zeros((n, n))
    blocks = []
    for (i, j), x in sorted(ents.items()):
        q, r, d = R._split_entry(x, P)
        if q.degree() >= 2:
            raise HartreeUnsupported(
                f'hartree: the entry ({i}, {j}) has a second time derivative '
                f'(a polynomial part of degree {q.degree()} in the frequency).')
        ql = q.list() + [0, 0]
        K0[i, j], K1[i, j] = float(ql[0]), float(ql[1])
        if r != 0:
            Ae, be, ce, _desc = R._realize_proper(r, d, F)
            # the same diagonal (power of two) balancing the engine applies,
            # and a uniform one that equalizes the input and output vectors:
            # exact similarity transforms of the states, which keep graded
            # Pade blocks within the conditioning of the closure
            Ab, bb, cb, _sca = R._balance(R._to_numpy(Ae, False),
                                          R._to_numpy(be, False),
                                          R._to_numpy(ce, False))
            # and the engine's balanced truncation of the entry at its
            # tolerance (1e-12 of the entry's H-infinity norm by default):
            # a high-order Pade block keeps the states it needs
            Ab, b1, c1, _info = R.balanced_truncation(
                Ab, bb[:, 0], cb[0, :], R.truncation_tol())
            bb, cb = np.asarray(b1)[:, None], np.asarray(c1)[None, :]
            nb, nc = np.linalg.norm(bb), np.linalg.norm(cb)
            mu = math.sqrt(nc / nb) if nb > 0 and nc > 0 else 1.0
            blocks.append((i, j, Ab, bb[:, 0] * mu, cb[0, :] / mu))
    return K1, K0, blocks


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
    images = _kernel_images(model, ft._ns)
    knames = {str(k) for k in images}
    klin = {}
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
                if knames:
                    uses_k = bool(knames & {str(v) for v in c.variables()})
                    if sum(ex_p) == 1:
                        b = list(ex_p).index(1)
                        klin[(a, b)] = klin.get((a, b), SR(0)) + c
                        continue
                    if uses_k:
                        raise HartreeUnsupported(
                            'hartree: a kernel acts on a nonlinear function '
                            'of the fields in this action; only kernels of '
                            'the fields themselves are embedded (a '
                            'nonlinear argument is embedded by the model '
                            'builder).')
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
                if knames & {str(v) for v in c.variables()}:
                    raise HartreeUnsupported(
                        'hartree: a kernel in the noise vertex (a non-local '
                        'noise cumulant) is not embedded; declare the noise '
                        'through its Markovian embedding.')
                idx = [i for i, e in enumerate(ex_r) for _ in range(e)]
                a, b = idx
                if a == b:
                    Qg[a][a] = Qg[a][a] - 2 * c * mono
                else:
                    Qg[a][b] = Qg[a][b] - c * mono
                    Qg[b][a] = Qg[b][a] - c * mono
                degree = max(degree, sum(ex_p) + 1)
    n_phys = n
    aux, kernel_info = [], None
    if klin:
        K1, K0, blocks = _kernel_pencil(klin, n, images, num)
        m_aux = sum(b[2].shape[0] for b in blocks)
        zs = [SR.var(f'kernel_state_{k + 1}') for k in range(m_aux)]
        aux = [str(z) for z in zs]
        n = n_phys + m_aux
        xs = xs + zs
        M = [[SR(K1[a, b]) if a < n_phys and b < n_phys else SR(0)
              for b in range(n)] for a in range(n)]
        g = [g[a] - sum((SR(K0[a, b]) * xs[b] for b in range(n_phys)), SR(0))
             for a in range(n_phys)] + [SR(0)] * m_aux
        Qg = [[Qg[a][b] if a < n_phys and b < n_phys else SR(0)
               for b in range(n)] for a in range(n)]
        off = n_phys
        for i_, j_, Ae, be, ce in blocks:
            m_ = Ae.shape[0]
            for k_ in range(m_):
                M[off + k_][off + k_] = SR(1)
                g[i_] = g[i_] - SR(ce[k_]) * zs[off - n_phys + k_]
                g[off + k_] = (sum((SR(Ae[k_, l]) * zs[off - n_phys + l]
                                    for l in range(m_)), SR(0))
                               + SR(be[k_]) * xs[j_])
            off += m_
        kernel_info = {'kernels': sorted(knames), 'states': m_aux,
                       'entries': [(i_, j_, Ae.shape[0])
                                   for i_, j_, Ae, _be, _ce in blocks]}
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
            'supported; an exogenous input field, such as a point-process '
            'input that drives a kernel, has no dynamics of its own).')
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
        naturals=[r[4] for r in layout], response=resp, f=f, Q=Qx, M=Mn,
        saddle=saddle, other_saddles=others,
        degree=int(degree), polynomial=exact, taylor_order=order, mf=mf,
        num_params=num, aux=aux, kernel_info=kernel_info)


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


def _restrict(r, nph, sysm):
    """Move a closure solution on the fields and the kernel states to the
    model's own fields (``nph`` of them): the full solution stays in
    ``r.full``."""
    if r.A.shape[0] == nph:
        return
    r.full = {'mean': r.mean.copy(), 'cov': r.cov.copy(), 'A': r.A.copy(),
              'Q': r.Q.copy(), 'saddle': r.saddle.copy(),
              'aux': list(sysm.aux)}
    r.mean, r.cov = r.mean[:nph], r.cov[:nph, :nph]
    r.A, r.Q = r.A[:nph, :nph], r.Q[:nph, :nph]
    r.saddle = r.saddle[:nph]


# ── hard delays: the Pade order search ─────────────────────────────────────

def _delay_names(model):
    """The kernels of ``model`` that carry a delay factor (``{}`` for none)."""
    hook = model.get('kernel_ft_image')
    if hook is None or not model.get('kernels'):
        return {}
    from api.delay import kernel_delays, specs_may_delay
    specs = getattr(hook, '_kernel_specs', None)
    if specs is not None and not specs_may_delay(specs):
        return {}
    from engine.core.field_theory import FieldTheory
    ns, _R, _nt = FieldTheory(model, taylor_order=2)._build_namespace()
    return kernel_delays(model, ns)


def _delay_observable(r, omegas):
    """What two Pade orders are compared on: the mean, the covariance and
    the spectra of the model's own fields at ``omegas``.  These converge
    geometrically with the order; the lag correlator at a fixed lag converges
    only algebraically (the Pade approximant of a delay rings at its cutoff
    frequency), so it is reported (:func:`_delay_lag_change`) but not part of
    the target."""
    parts = [np.asarray(r.mean).ravel(), np.asarray(r.cov).ravel()]
    for i in range(len(r.fields)):
        parts.append(np.real(r.spectrum(omegas, i, i)))
    return np.concatenate(parts)


def _delay_scales(r):
    """Frequencies and lags in the units of the slowest relaxation of the
    closure ``r``."""
    rates = np.real(r.rates)
    slow = rates[rates > 1e-12]
    t_c = 1.0 / slow.min() if slow.size else 1.0
    return (np.array([0.3, 1.0, 3.0]) / t_c, t_c * np.array([0.1, 0.5, 1.0, 2.0]))


def _delay_lag_change(a, b, lags):
    """Largest change of the lag correlator between two orders, relative to
    its largest entry."""
    ca, cb = a.C_matrix(lags), b.C_matrix(lags)
    return float(np.max(np.abs(cb - ca)) / max(np.max(np.abs(cb)), 1e-300))


def _hartree_delayed(model, parameters, delay, kw):
    """The closure of a model with hard delays: every delay factor of the
    kernel images by its Pade approximant of order ``n``; ``n`` from the
    order search of the loop path (``api.delay``): ``n_start``, doubling up
    to ``n_max``, the first order that agrees with the one before to ``tol``
    (the mean, the covariance and the lag correlator, relative to the
    largest entry).  A target that is not met is refused."""
    from api import delay as D
    o = D.resolve_options(delay)
    warn = kw.get('warn', True)
    kw = dict(kw, warn=False)
    names = sorted(_delay_names(model))
    run = lambda n: _hartree(D.pade_model(model, int(n)), parameters, **kw)

    def finish(r, n, est, source, orders, ests):
        r.info['model'] = model.get('name', '<unnamed>')
        r.info['delay'] = {'order': int(n), 'estimate': est,
                           'target': o['tol'], 'order_source': source,
                           'orders': list(orders), 'estimates': list(ests),
                           'kernels': names,
                           'lag_correlator_change': r.info.get(
                               'delay_lag_change')}
        msg = (f'the hard delay of {", ".join(names)} is embedded through its '
               f'Pade approximant of order {int(n)}'
               + (f', estimated relative error {est:.2e} (target '
                  f'{o["tol"]:.1e})' if est is not None else
                  ' (order given: no error estimate)')
               + (f'; the lag correlator changed by {r.info["delay_lag_change"]:.1e} '
                  f'from the order before (it converges more slowly than the '
                  f'spectrum and the covariance)'
                  if r.info.get('delay_lag_change') is not None else ''))
        r.advisories = list(r.advisories) + [
            {'code': 'HARTREE_DELAY_PADE', 'message': msg}]
        if warn:
            for a in r.advisories:
                cls = (HartreeBistabilityWarning
                       if a['code'] == 'HARTREE_MULTIPLE_SOLUTIONS'
                       else HartreeAdvisoryWarning)
                warnings.warn(f"hartree [{a['code']}]: {a['message']}", cls,
                              stacklevel=3)
        return r
    if o['order'] != 'auto':
        return finish(run(o['order']), o['order'], None, 'given',
                      [o['order']], [])
    key = _delay_cache_key(model, parameters, kw, o)
    cached = D._order_cache_get(key).get('hartree')
    if cached is not None:
        r = run(cached['order'])
        return finish(r, cached['order'], cached['estimate'], 'replay',
                      cached['orders'], cached['estimates'])
    seq, n = [], o['n_start']
    while n < o['n_max']:
        seq.append(n)
        n *= 2
    seq.append(o['n_max'])
    t0 = time.perf_counter()
    prev, scales = None, None
    orders, ests = [], []
    for n in seq:
        r = run(n)
        orders.append(n)
        if prev is not None:
            scales = _delay_scales(prev) if scales is None else scales
            a = _delay_observable(prev, scales[0])
            b = _delay_observable(r, scales[0])
            est = float(np.max(np.abs(b - a)) / max(np.max(np.abs(b)), 1e-300))
            ests.append(est)
            r.info['delay_lag_change'] = _delay_lag_change(prev, r, scales[1])
            if est <= o['tol']:
                D._order_cache_put(key, 'hartree', {
                    'order': n, 'estimate': est, 'orders': list(orders),
                    'estimates': list(ests)})
                return finish(r, n, est, 'search', orders, ests)
            if time.perf_counter() - t0 > o['wall_max']:
                break
        prev = r
    raise HartreeError(
        f'hartree: the Pade order of the delay of {", ".join(names)} did not '
        f'reach the target: estimated relative error '
        f'{ests[-1] if ests else float("nan"):.2e} at order {orders[-1]} '
        f'(target {o["tol"]:.1e}; orders tried {orders}). Raise '
        f'delay={{"n_max": ...}} or relax delay={{"tol": ...}}, or give '
        f'delay={{"order": n}} to accept an order without an estimate.')


def _delay_cache_key(model, parameters, kw, o):
    if not o['order_cache'] or model.get('spec_signature') is None:
        return None
    from api._propagator import model_cache_signature
    return repr(('hartree', model_cache_signature(model), model.get('name'),
                 sorted((str(k), repr(v)) for k, v in (parameters or {}).items()),
                 sorted((str(k), repr(v)) for k, v in kw.items()
                        if k != 'warn'),
                 sorted((k, v) for k, v in o.items() if k != 'order_cache')))


def _same(a, b):
    scale = 1.0 + max(np.max(np.abs(a.cov)), np.max(np.abs(b.cov)),
                      np.max(np.abs(a.mean)), np.max(np.abs(b.mean)))
    return (np.max(np.abs(a.mean - b.mean), initial=0.0)
            + np.max(np.abs(a.cov - b.cov))) < _SAME_SOLUTION_RTOL * scale


def _summary(r, label):
    return {'mean': r.mean.copy(), 'cov': r.cov.copy(),
            'rates': np.sort_complex(r.rates), 'start': label,
            'converged': r.converged}


def hartree(model, parameters=None, *, delay=None, **kw):
    """Self-consistent Gaussian (Hartree) approximation of a temporal model
    (``dd.hartree``); see :func:`_hartree` for the other options.

    ``delay``: the options of a model with hard delays (the keys of
    ``dd.Config.delay``: ``order`` ``'auto'`` or an integer, ``tol``,
    ``n_start``, ``n_max``, ``wall_max``, ``order_cache``).  A model without
    a delay ignores it.  With delays the kernels are embedded through their
    Pade approximants; ``order='auto'`` runs the order search of the loop
    path (``n_start``, doubling to ``n_max``) until two orders agree to
    ``tol`` in the mean, the covariance and the lag correlator, and refuses
    (:class:`HartreeError`) when ``n_max`` or ``wall_max`` is reached first.
    ``result.info['delay']`` records the order, the estimated error and
    where the order came from (``'search'``, ``'replay'`` of a cached order,
    ``'given'``), and the advisory ``HARTREE_DELAY_PADE`` is always set.
    """
    if not isinstance(model, dict) and callable(getattr(model, 'build', None)):
        model = model.build()
    if _delay_names(model):
        return _hartree_delayed(model, parameters, delay, kw)
    return _hartree(model, parameters, **kw)


def _hartree(model, parameters=None, *, fixed_point_index=0, start=None,
             mean0=None, cov0=None, taylor_order=None, multistart=True,
             mf_dae_n_starts=64, mf_dae_seed_box=None, warn=True,
             expectations='auto', **kw):
    """The closure of a model without hard delays (see :func:`hartree`).

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
    nph = len(sysm.fields)
    n = sysm.n                       # the fields and the kernel states
    saddle_o = sysm.saddle
    saddle = np.concatenate([saddle_o, np.zeros(n - nph)])

    def pad(v):
        return np.concatenate([np.asarray(v, dtype=float), np.zeros(n - nph)])
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
          else pad(np.asarray(mean0, dtype=float).reshape(nph) - saddle_o))

    def start_cov(m):
        c = tree_cov(m)
        if cov0 is not None:
            c[:nph, :nph] = np.asarray(cov0, dtype=float).reshape(nph, nph)
        return c
    starts.append(('start' if mean0 is not None else 'saddle', m0,
                   start_cov(m0)))
    if multistart:
        if mean0 is not None:
            starts.append(('saddle', np.zeros(n), tree_cov(np.zeros(n))))
        for k, o in enumerate(sysm.other_saddles):
            starts.append((f'mean-field root {k + 1}', pad(o['offset']),
                           tree_cov(pad(o['offset']))))
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
        _restrict(r, nph, sysm)
    res = first
    res.legs = list(sysm.legs)
    res.naturals = list(sysm.naturals)
    res.J0, res.Q0 = sysm.J0[:nph, :nph], sysm.Q0[:nph, :nph]
    if res.full is not None:
        res.full['J0'], res.full['Q0'] = sysm.J0, sysm.Q0
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
        off = first.mean - saddle_o
        P = np.linalg.inv(first.cov)
        d0 = float(off @ P @ off)
        near = [o for o in sysm.other_saddles
                if float((off - o['offset']) @ P @ (off - o['offset'])) < d0]
        if near:
            advisories.append({
                'code': 'HARTREE_LEFT_SADDLE',
                'message': (f'the closure started at the selected saddle '
                            f'{np.array2string(saddle_o, precision=4)} settled '
                            f'at mean {np.array2string(first.mean, precision=4)}'
                            f', nearer the mean-field root '
                            f'{np.array2string(saddle_o + near[0]["offset"], precision=4)}'
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
                     'pencil': sysm.M[:nph, :nph],
                     'kernel_states': (None if sysm.kernel_info is None
                                       else dict(sysm.kernel_info,
                                                 names=list(sysm.aux))),
                     'starts': [s[0] for s in starts],
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

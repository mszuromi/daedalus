"""
api.hartree
===========
Self-consistent Gaussian (Hartree) resummation of a temporal model: a
research prototype, not wired into ``compute_cumulants``.

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
do not enter at this order.  The lag correlator is a single matrix
exponential, ``C(τ) = e^{Aτ} Σ`` for ``τ >= 0``.

Expectations are taken by tensor Gauss-Hermite quadrature, exact for a
polynomial drift and noise of the expansion's degree.

Scope (prototype)
-----------------
Temporal models whose action has a first-order ``Dt`` pencil, no kernels
(``Conv``) and no time derivatives inside vertices.  Colored noise enters
through its Markovian embedding, which ``FieldTheory`` already performs for
``declare_cgf_term`` exponential kernels.  A non-polynomial drift is
replaced by its Taylor polynomial of ``taylor_order`` about the mean-field
saddle.

Entry points: :func:`hartree` (a model dict) and :func:`gaussian_closure`
(numeric drift / Jacobian / noise callables).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sl
from scipy.optimize import root


@dataclass
class HartreeResult:
    """A converged self-consistent Gaussian.

    ``mean`` is absolute (saddle plus the Hartree shift); ``cov`` is the
    equal-time covariance ``Σ``; ``A`` the averaged Jacobian (the dressed
    inverse propagator is ``Dt - A``); ``Q`` the averaged noise covariance.
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
    info: dict = field(default_factory=dict)

    def C(self, tau, i=0, j=0):
        """``κ(x_i(t), x_j(t + τ))`` on an array of lags (either sign)."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.empty(tau.shape)
        for n, t in enumerate(tau):
            if t >= 0:
                out[n] = (self.cov @ sl.expm(self.A.T * t))[i, j]
            else:
                out[n] = (sl.expm(self.A * (-t)) @ self.cov)[i, j]
        return out

    def spectrum(self, omega, i=0, j=0):
        """``S_ij(ω)``, entry ``(i, j)`` of ``H Q Hᴴ`` with
        ``H = (iω - A)^{-1}`` (the matrix-spectrum convention of
        ``api.spectra``)."""
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        n = self.A.shape[0]
        out = np.empty(omega.shape, dtype=complex)
        for k, w in enumerate(omega):
            H = np.linalg.inv(1j * w * np.eye(n) - self.A)
            out[k] = (H @ self.Q @ H.conj().T)[i, j]
        return out

    @property
    def rates(self):
        """Decay rates (``-eig(A)``) of the dressed propagator."""
        return -np.linalg.eigvals(self.A)


def _gh_rule(n_dim, n_nodes):
    """Tensor Gauss-Hermite nodes ``z`` (shape ``(P, n_dim)``) and weights
    for the standard normal ``N(0, I)``."""
    z1, w1 = np.polynomial.hermite_e.hermegauss(n_nodes)
    w1 = w1 / w1.sum()
    Z = np.array(list(itertools.product(z1, repeat=n_dim)))
    W = np.prod(np.array(list(itertools.product(w1, repeat=n_dim))), axis=1)
    return Z, W


def _chol_psd(S):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return V * np.sqrt(np.clip(w, 0.0, None))


def gaussian_closure(drift, jac, noise, n, *, mean0, cov0=None, degree=3,
                     tol=1e-12, max_iter=200, damping=0.5, fields=None,
                     saddle=None):
    """Solve the self-consistent Gaussian closure for numeric callables.

    ``drift(X) -> (P, n)``, ``jac(X) -> (P, n, n)`` (``∂f_a/∂x_b``) and
    ``noise(X) -> (P, n, n)`` act on a batch of points ``X`` of shape
    ``(P, n)`` in the same coordinates as ``mean0``.  ``degree`` is the
    largest polynomial degree to integrate exactly (the quadrature uses
    ``degree // 2 + 2`` nodes per dimension).

    The iteration is a Newton step on the mean with the averaged Jacobian and
    a Lyapunov solve for the covariance, then a polish by ``scipy.optimize.root``
    on the full residual.  Raises ``RuntimeError`` when the dressed drift
    stops being stable or no solution is found.
    """
    Z, W = _gh_rule(n, max(2, degree // 2 + 2))
    mean = np.asarray(mean0, dtype=float).copy()
    if cov0 is None:
        cov0 = np.eye(n)
    cov = np.asarray(cov0, dtype=float).copy()

    def averages(m, S):
        X = m + Z @ _chol_psd(S).T
        return (W @ drift(X), np.einsum('p,pab->ab', W, jac(X)),
                np.einsum('p,pab->ab', W, noise(X)))

    def lyap(A, Q):
        if np.max(np.linalg.eigvals(A).real) >= 0:
            raise RuntimeError('Hartree closure: the dressed drift is not '
                               'stable (eig(A) has a non-negative real part).')
        return sl.solve_continuous_lyapunov(A, -Q)

    # A start whose averaged drift is unstable (an unstable saddle, a small
    # guess) is widened until the fluctuations stabilise it.
    for _ in range(60):
        if np.max(np.linalg.eigvals(averages(mean, cov)[1]).real) < 0:
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
    if np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T))) <= 0:
        raise RuntimeError('Hartree closure: the covariance is not positive '
                           'definite at the solution.')
    if np.max(np.linalg.eigvals(A).real) >= 0:
        raise RuntimeError('Hartree closure: no stable self-consistent '
                           'solution from this start.')
    scale = 1.0 + np.max(np.abs(cov)) + np.max(np.abs(Q))
    return HartreeResult(
        fields=list(fields or range(n)), mean=mean, cov=cov, A=A, Q=Q,
        saddle=(np.zeros(n) if saddle is None else np.asarray(saddle)),
        converged=final < 1e-9 * scale, residual=final, iterations=it,
        info={'quadrature_nodes': len(W), 'fixed_point_step': res})


# ── Model adapter ──────────────────────────────────────────────────────────

def drift_and_noise(model, parameters, *, fixed_point_index=0,
                    taylor_order=None):
    """The drift ``f`` and noise covariance ``Q`` of a temporal model, as Sage
    expressions in the fluctuation fields about the mean-field saddle.

    Returns ``(phys_names, f, Q, saddle_values, degree, ft, num_params)``:
    ``f[a]`` and ``Q[a][b]`` are expressions in the physical fluctuation
    variables ``phys_names`` with every parameter substituted.  The action is
    read as ``Σ_a xt_a (Σ_b M_ab Dt dx_b - g_a(dx)) - ½ xtᵀ Q xt + O(xt³)`` and
    ``f = M^{-1} g``.
    """
    from sage.all import SR, matrix
    from engine.core.field_theory import FieldTheory
    from api._mean_field import solve_mean_field, complete_parameters

    if model.get('kernels') or model.get('spatial'):
        raise NotImplementedError(
            'hartree: kernels and spatial models are not supported by the '
            'prototype (temporal, kernel-free actions only).')
    order = int(taylor_order or 4)
    ft = FieldTheory(model, taylor_order=order)
    ft.expand()
    fundamental = complete_parameters(model, parameters, strict=False)
    mf = solve_mean_field(ft, model, fundamental, verbose=False,
                          fixed_point_index=fixed_point_index)
    num = dict(mf['num_params'])

    names = list(ft._ns._ring_var_names)
    nt = ft._n_tilde
    resp, phys = names[:nt], names[nt:]
    if len(resp) != len(phys):
        raise NotImplementedError(
            'hartree: the action needs one response field per physical field.')
    R = ft._R
    Rv = R.gens()
    rv, pv = Rv[:nt], Rv[nt:]
    Dt = SR.var('Dt')
    xs = [SR.var(p) for p in phys]

    n = len(phys)
    M = [[SR(0)] * n for _ in range(n)]
    g = [SR(0)] * n
    Q = [[SR(0)] * n for _ in range(n)]
    degree = 1
    for (ntil, nphys), poly in ft._by_tp.items():
        if poly == 0:
            continue
        if ntil >= 3:
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
                if cd != 0:
                    if sum(ex_p) != 1 or c.coefficient(Dt, 2) != 0:
                        raise NotImplementedError(
                            'hartree: time derivatives inside vertices are '
                            'not supported by the prototype.')
                    b = list(ex_p).index(1)
                    M[a][b] = M[a][b] + cd
                    c = c.subs({Dt: 0})
                g[a] = g[a] - c * mono
                degree = max(degree, sum(ex_p))
            elif ntil == 2:
                idx = [i for i, e in enumerate(ex_r) for _ in range(e)]
                a, b = idx
                if a == b:
                    Q[a][a] = Q[a][a] - 2 * c * mono
                else:
                    Q[a][b] = Q[a][b] - c * mono
                    Q[b][a] = Q[b][a] - c * mono
                degree = max(degree, sum(ex_p) + 1)
    Mn = matrix(SR, M).subs(num)
    if Mn.det() == 0:
        raise NotImplementedError(
            'hartree: singular Dt pencil (algebraic constraints are not '
            'supported by the prototype).')
    gn = matrix(SR, n, 1, [gi.subs(num) for gi in g])
    f = list((Mn.inverse() * gn).list())
    Qn = [[Q[a][b].subs(num) for b in range(n)] for a in range(n)]
    saddle = []
    for p in phys:
        base = p[1:] if p.startswith('d') else p
        nm, idx = base.rstrip('0123456789'), base[len(base.rstrip(
            '0123456789')):]
        key = SR.var(nm + 'star' + idx)
        saddle.append(float(num.get(key, 0.0)))
    return phys, f, Qn, np.array(saddle), degree, ft, num


def _vectorize(exprs, xs):
    """Batch numpy callable for a nested list of Sage expressions."""
    from sage.all import fast_callable, RDF, SR
    flat = [SR(e) for e in exprs]
    fns = [fast_callable(e, vars=xs, domain=RDF) for e in flat]

    def call(X):
        return np.array([[fn(*row) for fn in fns] for row in X])
    return call


def hartree(model, parameters=None, *, fixed_point_index=0, mean0=None,
            cov0=None, taylor_order=None, **kw):
    """Self-consistent Gaussian (Hartree) approximation of a temporal model.

    ``parameters`` are the model's fundamental parameters (defaults filled).
    The expansion point is the mean-field saddle picked by
    ``fixed_point_index``; ``mean0`` (absolute) and ``cov0`` override the
    starting point of the iteration (default: the saddle and the tree-level
    covariance, or the identity when the saddle is unstable).  Returns a
    :class:`HartreeResult`.
    """
    from sage.all import SR, matrix
    phys, f, Q, saddle, degree, _ft, _num = drift_and_noise(
        model, parameters or {}, fixed_point_index=fixed_point_index,
        taylor_order=taylor_order)
    n = len(phys)
    xs = [SR.var(p) for p in phys]
    J = matrix(SR, n, n, [[SR(f[a]).derivative(xs[b]) for b in range(n)]
                         for a in range(n)])
    f_call = _vectorize(f, xs)
    J_call = _vectorize(J.list(), xs)
    Q_call = _vectorize([Q[a][b] for a in range(n) for b in range(n)], xs)

    def drift(X):
        return f_call(X)

    def jac(X):
        return J_call(X).reshape(len(X), n, n)

    def noise(X):
        return Q_call(X).reshape(len(X), n, n)

    m0 = (np.zeros(n) if mean0 is None
          else np.asarray(mean0, dtype=float) - saddle)
    if cov0 is None:
        A0 = jac(m0[None, :])[0]
        Q0 = noise(m0[None, :])[0]
        if np.max(np.linalg.eigvals(A0).real) < 0:
            cov0 = sl.solve_continuous_lyapunov(A0, -Q0)
        else:
            cov0 = np.eye(n) * max(1e-3, float(np.max(np.abs(Q0))))
    r = gaussian_closure(drift, jac, noise, n, mean0=m0, cov0=cov0,
                         degree=max(3, degree), fields=phys, saddle=saddle,
                         **kw)
    r.mean = r.mean + saddle
    return r

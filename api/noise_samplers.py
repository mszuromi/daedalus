"""Samplers for the noise library (:mod:`api.noise`).

``source.sample(dt, size, rng, values=None)`` draws ``size`` i.i.d. increments
of the Levy process over a step ``dt``; ``jump.sample(rng, size, values=None)``
draws jump sizes.  ``rng`` is a ``numpy.random.Generator``.  ``values`` binds
the names of field-dependent parameters (``{'x': array}``; bracket or flat
names) so a simulator can call a source once per step with the current state.

Exactness
---------
Exact increments (closed-form representation, any ``dt``):
Gaussian, Poisson, CompoundPoisson (Poisson number of i.i.d. jumps),
GammaProcess, InverseGaussianProcess, VarianceGamma (Brownian motion on a gamma
clock), NormalInverseGaussian (Brownian motion on an inverse-Gaussian clock),
BilateralGamma, Skellam, NegativeBinomialProcess, SumSource, the
TemperedStableSubordinator and CGMY with ``Y < 1`` (exponentially tilted
positive-stable rejection; the step is split so the acceptance probability is
at least ``exp(-1)``).

Controlled approximations (parameters documented at the function):
 * CGMY with ``1 < Y < 2``: large jumps (``> eps``) exactly, the small-jump
   part replaced by a Gaussian with matched mean and variance
   (Asmussen-Rosinski); the neglected cumulants are of order
   ``eps^(m-Y)`` (m >= 3), relative error about ``(eps*max(M, G))^(4-Y)`` in
   the fourth cumulant.  ``eps = CGMY_EPS / max(M, G)``.
 * Meixner: numerical inversion of the exact density on a fine grid
   (``MEIXNER_GRID`` points, ``z = d sinh(s)`` change of variable);
   relative error in the cumulants of order 1e-6 or better.

Field-dependent parameters are supported by every sampler (the scalar-only
ones loop over the entries).  ``Cumulants`` and ``CGF`` have no sampler (a
CGF does not determine a sampling scheme) and raise :class:`NoiseError`.
"""
from __future__ import annotations

import numpy as np

from api import noise as N
from api.noise import NoiseError, _numeric

__all__ = ['sample_source', 'sample_jump', 'simulate_ou_paths',
           'euler_ou_cumulant', 'CGMY_EPS', 'MEIXNER_GRID']

CGMY_EPS = 0.02
MEIXNER_GRID = 1 << 16


# ── parameter evaluation ───────────────────────────────────────────────

_LAMBDIFY_CACHE: dict = {}


def _lookup(values, name):
    if values is None:
        return None
    if name in values:
        return values[name]
    base = N._unflatten_name(name)
    return values.get(base)


def _evaluate(expr, values):
    """float / array value of a parameter expression."""
    x = _numeric(expr)
    if x is not None:
        return x
    from sage.all import SR
    expr = SR(expr)
    names = sorted(str(v) for v in expr.variables())
    missing = [n for n in names if _lookup(values, n) is None]
    if missing:
        raise NoiseError(
            f'sampling needs values for {missing} (field-dependent parameter '
            f'{N.expr_to_text(expr)!r}); pass values={{name: float or array}}.')
    key = (str(expr), tuple(names))
    fn = _LAMBDIFY_CACHE.get(key)
    if fn is None:
        import sympy
        fn = sympy.lambdify([sympy.Symbol(n) for n in names], expr._sympy_(), 'numpy')
        _LAMBDIFY_CACHE[key] = fn
    return fn(*[np.asarray(_lookup(values, n), dtype=float) for n in names])


def _params(obj, values):
    return {k: _evaluate(v, values) for k, v in obj.params.items()}


def _has_array(P):
    return any(np.ndim(v) > 0 for v in P.values())


def _row(P, i):
    return {k: (v[i] if np.ndim(v) > 0 else v) for k, v in P.items()}


def _loop_scalar(fn, P, size, *args):
    """Run a scalar-parameter sampler entry by entry when a parameter is an array."""
    out = np.empty(size)
    for i in range(size):
        out[i] = fn(_row(P, i), 1, *args)[0]
    return out


# ── primitive samplers ─────────────────────────────────────────────────

def _stable_positive(alpha, size, rng):
    """Positive alpha-stable (0<alpha<1) with ``E exp(-u S) = exp(-u^alpha)``
    (Kanter / Chambers-Mallows-Stuck)."""
    U = rng.uniform(0.0, np.pi, size)
    W = rng.exponential(1.0, size)
    return (np.sin(alpha * U) / np.sin(U) ** (1.0 / alpha)
            * (np.sin((1.0 - alpha) * U) / W) ** ((1.0 - alpha) / alpha))


def _tempered_stable_subordinator(alpha, c, lam, dt, size, rng):
    """Exact: ``E exp(-uX) = exp(-c|Gamma(-alpha)| dt ((lam+u)^alpha - lam^alpha))``.
    Tilted-stable rejection per sub-step of length ``tau`` (acceptance
    ``exp(-tau*kappa*lam^alpha) >= exp(-1)``); the sub-steps are summed."""
    from math import gamma, ceil
    kappa = c * gamma(1.0 - alpha) / alpha
    m = max(1, ceil(dt * kappa * lam ** alpha))
    tau = dt / m
    scale = (tau * kappa) ** (1.0 / alpha)
    total = np.zeros(size)
    for _ in range(m):
        out = np.empty(size)
        todo = np.arange(size)
        while todo.size:
            S = scale * _stable_positive(alpha, todo.size, rng)
            ok = rng.uniform(size=todo.size) < np.exp(-lam * S)
            out[todo[ok]] = S[ok]
            todo = todo[~ok]
        total += out
    return total


def _upper_gamma(a, x):
    """``Gamma(a, x)`` for real ``a`` (negative allowed) and ``x > 0``."""
    import mpmath
    return float(mpmath.gammainc(a, x))


def _cgmy(C, G, M, Y, dt, size, rng):
    from math import gamma
    if Y < 0:                                   # finite activity: compound Poisson
        out = np.zeros(size)
        for lam_, sign in ((M, 1.0), (G, -1.0)):
            rate = C * gamma(-Y) * lam_ ** Y
            n = rng.poisson(rate * dt, size)
            tot = int(n.sum())
            jumps = rng.gamma(-Y, 1.0 / lam_, tot)
            out += sign * np.bincount(np.repeat(np.arange(size), n), weights=jumps,
                                      minlength=size)
        return out
    if Y < 1:                                   # exact: difference of subordinators
        return (_tempered_stable_subordinator(Y, C, M, dt, size, rng)
                - _tempered_stable_subordinator(Y, C, G, dt, size, rng))
    # 1 < Y < 2: truncated series with Gaussian small jumps, chunked over paths
    # so that the jumps held at once stay below ~2e6
    eps = CGMY_EPS / max(M, G)
    per_path = dt * C * (M ** Y * _upper_gamma(-Y, M * eps) + G ** Y * _upper_gamma(-Y, G * eps))
    chunk = int(max(1, min(size, 2e6 / max(per_path, 1e-9))))
    if chunk < size:
        return np.concatenate([_cgmy_series(C, G, M, Y, dt, min(chunk, size - i), rng, eps)
                               for i in range(0, size, chunk)])
    return _cgmy_series(C, G, M, Y, dt, size, rng, eps)


def _cgmy_series(C, G, M, Y, dt, size, rng, eps):
    from math import gamma
    from scipy.special import gammainc
    large_mean = 0.0
    out = np.zeros(size)
    var_small = 0.0
    for lam_, sign in ((M, 1.0), (G, -1.0)):
        nu0 = C * lam_ ** Y * _upper_gamma(-Y, lam_ * eps)          # nu((eps, inf))
        nu1 = C * lam_ ** (Y - 1) * _upper_gamma(1 - Y, lam_ * eps)  # int x nu
        large_mean += sign * nu1
        var_small += C * lam_ ** (Y - 2) * gamma(2 - Y) * gammainc(2 - Y, lam_ * eps)
        n = rng.poisson(nu0 * dt, size)
        tot = int(n.sum())
        xs = np.empty(tot)
        todo = np.arange(tot)
        while todo.size:                       # density ~ x^(-1-Y) e^{-lam x} on (eps, inf)
            x = eps * rng.uniform(size=todo.size) ** (-1.0 / Y)
            ok = rng.uniform(size=todo.size) < np.exp(-lam_ * (x - eps))
            xs[todo[ok]] = x[ok]
            todo = todo[~ok]
        out += sign * np.bincount(np.repeat(np.arange(size), n), weights=xs,
                                  minlength=size)
    kappa1 = C * gamma(1 - Y) * (M ** (Y - 1) - G ** (Y - 1))
    small_mean = kappa1 - large_mean
    out += small_mean * dt + np.sqrt(var_small * dt) * rng.standard_normal(size)
    return out


_MEIXNER_CACHE: dict = {}


def _meixner_table(alpha, beta, d):
    """(cdf, z) tables of the standardized Meixner density
    ``f(z) = (2cos(b/2))^(2d)/(2 pi Gamma(2d)) e^(b z) |Gamma(d + i z)|^2``
    on a grid uniform in ``s`` with ``z = d sinh(s)``: resolves the
    Cauchy-like core of width ``d`` (small ``dt``) and the exponential
    tails with the same relative accuracy."""
    key = (round(beta, 14), round(d, 14))
    hit = _MEIXNER_CACHE.get(key)
    if hit is not None:
        return hit
    from scipy.special import loggamma
    mean = d * np.tan(beta / 2.0)
    sd = np.sqrt(d / (2.0 * np.cos(beta / 2.0) ** 2))
    width = 14.0 * sd + 60.0 / (np.pi - abs(beta))
    lo = min(mean - width, -1e-3 * width)
    hi = max(mean + width, 1e-3 * width)
    s = np.linspace(np.arcsinh(lo / d), np.arcsinh(hi / d), MEIXNER_GRID + 1)
    z = d * np.sinh(s)
    logf = (2 * d * np.log(2 * np.cos(beta / 2)) - np.log(2 * np.pi)
            - float(loggamma(2 * d).real) + beta * z
            + 2.0 * loggamma(d + 1j * z).real)
    g = np.exp(logf) * d * np.cosh(s)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(s))])
    cdf /= cdf[-1]
    out = (cdf, z)
    if len(_MEIXNER_CACHE) > 64:
        _MEIXNER_CACHE.clear()
    _MEIXNER_CACHE[key] = out
    return out


def _meixner(alpha, beta, delta, mu, dt, size, rng):
    d = delta * dt
    cdf, z = _meixner_table(alpha, beta, d)
    u = rng.uniform(size=size)
    return mu * dt + alpha * np.interp(u, cdf, z)


# ── source dispatch ────────────────────────────────────────────────────

def _compound(src, dt, size, rng, values):
    rate = _evaluate(src.params['rate'], values)
    n = rng.poisson(np.broadcast_to(np.asarray(rate, float) * dt, (size,)))
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(size)
    rep = None
    if values:
        rep = {}
        for k, v in values.items():
            a = np.asarray(v)
            rep[k] = np.repeat(a, n) if a.ndim > 0 and a.shape[0] == size else a
    jumps = sample_jump(src.jump, rng, tot, rep)
    return np.bincount(np.repeat(np.arange(size), n), weights=jumps, minlength=size)


def sample_source(src, dt, size, rng, values=None):
    """``size`` i.i.d. increments of ``src`` over ``dt`` (see module docstring)."""
    dt = float(dt)
    size = int(size)
    if dt <= 0:
        raise NoiseError('sample: dt must be positive.')
    if isinstance(src, N.SumSource):
        return sum(sample_source(q, dt, size, rng, values) for q in src.parts)
    if isinstance(src, N.CompoundPoisson):
        return _compound(src, dt, size, rng, values)
    P = _params(src, values)
    if isinstance(src, N.Gaussian):
        return P['mean'] * dt + np.sqrt(np.maximum(P['var'], 0.0) * dt) * rng.standard_normal(size)
    if isinstance(src, N.Poisson):
        return P['size'] * rng.poisson(np.broadcast_to(np.asarray(P['rate'], float) * dt, (size,)))
    if isinstance(src, N.GammaProcess):
        return rng.gamma(P['shape_rate'] * dt, P['scale'], size)
    if isinstance(src, N.InverseGaussianProcess):
        return rng.wald(P['mean_rate'] * dt, P['shape'] * dt ** 2, size)
    if isinstance(src, N.VarianceGamma):
        G = rng.gamma(dt / P['nu'], P['nu'], size)
        return (P['drift'] * dt + P['mu'] * G
                + P['sigma'] * np.sqrt(G) * rng.standard_normal(size))
    if isinstance(src, N.NormalInverseGaussian):
        a, b, d = P['alpha'], P['beta'], P['delta']
        gam = np.sqrt(a ** 2 - b ** 2)
        Y = rng.wald(d * dt / gam, (d * dt) ** 2, size)
        return P['mu'] * dt + b * Y + np.sqrt(Y) * rng.standard_normal(size)
    if isinstance(src, N.BilateralGamma):
        return (rng.gamma(P['alpha_p'] * dt, 1.0 / P['lam_p'], size)
                - rng.gamma(P['alpha_m'] * dt, 1.0 / P['lam_m'], size))
    if isinstance(src, N.Skellam):
        return (rng.poisson(np.broadcast_to(np.asarray(P['rate_plus'], float) * dt, (size,)))
                - rng.poisson(np.broadcast_to(np.asarray(P['rate_minus'], float) * dt, (size,)))
                ).astype(float)
    if isinstance(src, N.NegativeBinomialProcess):
        # N(dt) ~ NB(shape_rate*dt, p): Poisson(Gamma(r dt, p/(1-p)))
        lam = rng.gamma(P['shape_rate'] * dt, P['p'] / (1.0 - P['p']), size)
        return rng.poisson(lam).astype(float)
    if isinstance(src, N.TemperedStableSubordinator):
        if _has_array(P):
            return _loop_scalar(lambda q, n_: _tempered_stable_subordinator(
                q['alpha'], q['c'], q['lam'], dt, n_, rng), P, size)
        return _tempered_stable_subordinator(P['alpha'], P['c'], P['lam'], dt, size, rng)
    if isinstance(src, N.CGMY):
        if _has_array(P):
            return _loop_scalar(lambda q, n_: _cgmy(
                q['C'], q['G'], q['M'], q['Y'], dt, n_, rng), P, size)
        return _cgmy(P['C'], P['G'], P['M'], P['Y'], dt, size, rng)
    if isinstance(src, N.Meixner):
        if _has_array(P):
            return _loop_scalar(lambda q, n_: _meixner(
                q['alpha'], q['beta'], q['delta'], q['mu'], dt, n_, rng), P, size)
        return _meixner(P['alpha'], P['beta'], P['delta'], P['mu'], dt, size, rng)
    if isinstance(src, (N.Cumulants, N.CGF)):
        raise NoiseError(
            f'{type(src).__name__} has no sampler: a cumulant list / CGF does not '
            f'determine a sampling scheme; use a named source.')
    raise NoiseError(f'no sampler for {type(src).__name__}.')


# ── jump laws ──────────────────────────────────────────────────────────

def _phase_type(alpha, T, size, rng):
    n = len(alpha)
    T = np.asarray(T, float)
    alpha = np.asarray(alpha, float)
    exit_rate = -np.diag(T)
    absorb = -T.sum(axis=1)                              # t = -T 1
    probs = np.zeros((n, n + 1))
    for i in range(n):
        for j in range(n):
            if j != i:
                probs[i, j] = T[i, j] / exit_rate[i]
        probs[i, n] = max(absorb[i], 0.0) / exit_rate[i]
        probs[i] /= probs[i].sum()
    init = np.append(alpha, max(1.0 - alpha.sum(), 0.0))
    state = rng.choice(n + 1, size=size, p=init / init.sum())
    out = np.zeros(size)
    active = np.flatnonzero(state < n)
    cum = np.cumsum(probs, axis=1)
    for _ in range(10 ** 6):
        if not active.size:
            break
        s = state[active]
        out[active] += rng.exponential(1.0 / exit_rate[s])
        u = rng.uniform(size=active.size)
        state[active] = (u[:, None] > cum[s]).sum(axis=1)
        active = active[state[active] < n]
    else:
        raise NoiseError('PhaseType sampler: the chain did not absorb (is T a proper sub-generator?).')
    return out


def sample_jump(law, rng, size, values=None):
    """``size`` i.i.d. draws of the jump law ``law``."""
    size = int(size)
    P = _params(law, values)
    if isinstance(law, N.Dirac):
        return np.broadcast_to(np.asarray(P['a'], float), (size,)).copy()
    if isinstance(law, N.ChiSquared):
        return rng.chisquare(2.0 * P['shape'], size)
    if isinstance(law, N.Exponential):
        return rng.exponential(P['mean'], size)
    if isinstance(law, N.Gamma):
        return rng.gamma(P['shape'], P['scale'], size)
    if isinstance(law, N.GaussianJump):
        return rng.normal(P['mu'], P['sigma'], size)
    if isinstance(law, N.Laplace):
        return rng.laplace(0.0, P['b'], size)
    if isinstance(law, N.Uniform):
        return rng.uniform(P['lo'], P['hi'], size)
    if isinstance(law, N.Bernoulli):
        return (rng.uniform(size=size) < P['p']).astype(float)
    if isinstance(law, N.Binomial):
        return rng.binomial(np.round(P['n']).astype(int), P['p'], size).astype(float)
    if isinstance(law, N.FinitePMF):
        vals = np.array([_evaluate(law.params[f'a{j}'], values) for j in range(law._n)], float)
        pr = np.array([_evaluate(law.params[f'p{j}'], values) for j in range(law._n)], float)
        if vals.ndim > 1 or pr.ndim > 1:
            raise NoiseError('FinitePMF sampling needs scalar values and probabilities.')
        return rng.choice(vals, size=size, p=pr / pr.sum())
    if isinstance(law, N.Geometric):
        return rng.geometric(P['p'], size).astype(float)
    if isinstance(law, N.PoissonJump):
        return rng.poisson(P['mean'], size).astype(float)
    if isinstance(law, N.NegativeBinomial):
        return rng.negative_binomial(P['r'], 1.0 - P['p'], size).astype(float)
    if isinstance(law, N.Logarithmic):
        return rng.logseries(P['p'], size).astype(float)
    if isinstance(law, N.DoubleExponential):
        up = rng.uniform(size=size) < P['p']
        return np.where(up, rng.exponential(1.0, size) / P['eta_up'],
                        -rng.exponential(1.0, size) / P['eta_down'])
    if isinstance(law, N.AsymmetricLaplace):
        return (P['loc'] + P['scale_up'] * rng.exponential(1.0, size)
                - P['scale_down'] * rng.exponential(1.0, size))
    if isinstance(law, N.Hyperexponential):
        pr = np.array([P[f'p{j}'] for j in range(law._n)], float)
        mn = np.array([P[f'm{j}'] for j in range(law._n)], float)
        if pr.ndim > 1 or mn.ndim > 1:
            raise NoiseError('Hyperexponential sampling needs scalar parameters.')
        comp = rng.choice(law._n, size=size, p=pr / pr.sum())
        return rng.exponential(mn[comp])
    if isinstance(law, N.PhaseType):
        n = law._n
        if _has_array(P):
            raise NoiseError('PhaseType sampling needs scalar parameters.')
        return _phase_type([P[f'a{i}'] for i in range(n)],
                           [[P[f'T{i}_{j}'] for j in range(n)] for i in range(n)],
                           size, rng)
    if isinstance(law, N.LogNormal):
        return rng.lognormal(P['mu'], P['sigma'], size)
    if isinstance(law, N.Weibull):
        return P['scale'] * rng.weibull(P['shape'], size)
    raise NoiseError(f'no sampler for jump law {type(law).__name__}.')


# ── simulator hook: Levy-driven OU ─────────────────────────────────────

def simulate_ou_paths(source, mu, *, T, dt, n_paths, rng, burn=None, x0=0.0):
    """Euler simulation of ``d x = -mu*x dt + d L`` with ``L`` the Levy
    process ``source`` (its increments are drawn exactly).  Returns the
    ``(n_paths, n_steps)`` array of ``x`` after the burn-in ``burn``
    (default ``8/mu``) up to time ``T``.  The Euler discretization is
    exact for the linear recursion ``x' = (1 - mu dt) x + dL``; its
    stationary cumulants are :func:`euler_ou_cumulant`."""
    n = int(round(T / dt))
    burn_n = int(round((8.0 / mu if burn is None else burn) / dt))
    x = np.full(n_paths, float(x0))
    keep = np.empty((n_paths, n))
    a = 1.0 - mu * dt
    for step in range(burn_n + n):
        x = a * x + source.sample(dt, n_paths, rng)
        if step >= burn_n:
            keep[:, step - burn_n] = x
    return keep


def euler_ou_cumulant(kappa_rate, n, mu, dt):
    """Stationary n-th cumulant of the Euler chain ``x' = (1 - mu dt) x + dL``:
    ``kappa_n(dL)/(1 - (1 - mu dt)^n)`` with ``kappa_n(dL) = kappa_rate*dt``;
    tends to the continuous ``kappa_rate/(n mu)`` as ``dt -> 0``."""
    return kappa_rate * dt / (1.0 - (1.0 - mu * dt) ** n)

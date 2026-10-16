r"""
Euler-Maruyama for stochastic delay differential equations (ring buffer)
=======================================================================

    dx = f(x(t), {x(t - d_j)}_j) dt + B dW,      x(t) = x0 for t <= 0,

for ``n_fields`` components and ``n_paths`` independent paths at once
(numpy, vectorised over paths; no processes).  The history is a ring buffer
of the last ``max_lag + 1`` states; ``x(t - d)`` is read at ``lag = d/dt``
steps back (linear interpolation between neighbouring steps when ``d/dt`` is
not an integer).  ``B`` (``n_fields x n_noise``) is constant (additive
noise), the scheme is the explicit Euler-Maruyama step

    x_{k+1} = x_k + dt f(x_k, x_{k - lag}) + sqrt(dt) B xi_k.

Its weak order is 1: the stationary covariance is biased by ``O(dt)``.  For a
LINEAR scalar equation ``em_linear_autocovariance`` gives the exact
stationary autocovariance of the discrete scheme itself (a Yule-Walker
solve), so a test can separate the statistical error of a run from the
discretisation bias (which it computes exactly).

``autocovariance(rec, i, j, lag_steps)`` estimates
``E[x_i(t) x_j(t + tau)]`` (paths and time averaged, the path mean removed)
with a standard error from batch means over independent groups of paths.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ['simulate', 'autocovariance', 'em_linear_autocovariance']


def simulate(drift, delays, B, x0, *, dt, t_max, n_paths, burn_in=20.0,
             record_every=1, seed=0):
    """Run the scheme; returns ``{'t': (n_rec,), 'x': (n_rec, n_fields, n_paths),
    'dt_rec': dt * record_every}``.

    ``drift(x, xd)``: ``x`` the current state ``(n_fields, n_paths)``, ``xd`` a
    list of the delayed states (one per entry of ``delays``, same shape);
    returns the drift ``(n_fields, n_paths)``."""
    # Sage-preparsed literals (Integer / RealNumber) as Python numbers
    dt, t_max, burn_in = float(dt), float(t_max), float(burn_in)
    n_paths, record_every = int(n_paths), int(record_every)
    if seed is not None and hasattr(seed, '__index__') and np.ndim(seed) == 0:
        seed = int(seed.__index__())
    B = np.atleast_2d(np.asarray(B, dtype=float))
    nfld, nnoise = B.shape
    x = np.empty((nfld, n_paths))
    x[:] = np.asarray(x0, dtype=float).reshape(nfld, 1)
    delays = [float(d) for d in delays]
    if any(d <= 0 for d in delays):
        raise ValueError('delays must be positive')
    lags = [d / dt for d in delays]
    lo = [int(math.floor(l + 1e-9)) for l in lags]
    frac = [l - f for l, f in zip(lags, lo)]
    frac = [0.0 if abs(fr) < 1e-9 else fr for fr in frac]
    L = (max(lo) + 2) if lo else 1
    buf = np.empty((L, nfld, n_paths))
    buf[:] = x
    ptr = 0                                    # buf[ptr] = x_k
    rng = np.random.default_rng(seed)
    n_burn = int(round(burn_in / dt))
    n_steps = n_burn + int(round(t_max / dt))
    sq = math.sqrt(dt)
    out = []
    ts = []
    for k in range(n_steps):
        xd = []
        for l0, fr in zip(lo, frac):
            a = buf[(ptr - l0) % L]
            if fr:
                b = buf[(ptr - l0 - 1) % L]
                a = (1.0 - fr) * a + fr * b
            xd.append(a)
        x = x + dt * drift(x, xd) + sq * (B @ rng.standard_normal((nnoise, n_paths)))
        ptr = (ptr + 1) % L
        buf[ptr] = x
        if k + 1 >= n_burn and (k + 1 - n_burn) % record_every == 0:
            out.append(x.copy())
            ts.append((k + 1 - n_burn) * dt)
    return {'t': np.array(ts), 'x': np.array(out), 'dt_rec': dt * record_every}


def autocovariance(rec, i, j, lag_steps, n_batches=20):
    """``(estimate, standard error)`` of ``E[x_i(t) x_j(t + tau)]`` at
    ``tau = lag_steps * dt_rec`` (array of lags in recorded steps)."""
    X = rec['x'] if isinstance(rec, dict) else rec
    xi = X[:, i, :] - X[:, i, :].mean()
    xj = X[:, j, :] - X[:, j, :].mean()
    n_t, n_p = xi.shape
    lag_steps = np.atleast_1d(np.asarray(lag_steps, dtype=int))
    nb = max(2, min(int(n_batches), n_p))
    groups = np.array_split(np.arange(n_p), nb)
    est, se = [], []
    for h in lag_steps:
        if h >= n_t:
            raise ValueError(f'lag {h} exceeds the record length {n_t}')
        prod = xi[:n_t - h, :] * xj[h:, :]
        per_path = prod.mean(axis=0)
        b = np.array([per_path[g].mean() for g in groups])
        est.append(per_path.mean())
        se.append(b.std(ddof=1) / math.sqrt(nb))
    return np.array(est), np.array(se)


def em_linear_autocovariance(a, c, sigma2, dt, lag, n_lags):
    """Exact stationary autocovariance ``gamma(h)``, ``h = 0..n_lags``, of the
    Euler-Maruyama scheme of ``dx = (-a x + c x(t - lag dt)) dt + sqrt(sigma2) dW``:
    the AR process ``x_{k+1} = (1 - a dt) x_k + c dt x_{k-lag} + e_k``,
    ``Var e = sigma2 dt`` (Yule-Walker)."""
    lag = int(lag)
    p = lag + 1
    phi = np.zeros(p + 1)                      # phi[j], j = 1..p
    phi[1] += 1.0 - a * dt
    phi[p] += c * dt
    # unknowns gamma(0..p): gamma(h) = sum_j phi_j gamma(|h - j|) (h >= 1),
    # gamma(0) = sum_j phi_j gamma(j) + sigma2 dt
    M = np.zeros((p + 1, p + 1))
    rhs = np.zeros(p + 1)
    for h in range(p + 1):
        M[h, h] += 1.0
        for jj in range(1, p + 1):
            if phi[jj] != 0.0:
                M[h, abs(h - jj)] -= phi[jj]
        if h == 0:
            rhs[0] = sigma2 * dt
    g = list(np.linalg.solve(M, rhs))
    while len(g) <= n_lags:
        h = len(g)
        g.append(sum(phi[jj] * g[h - jj] for jj in range(1, p + 1) if phi[jj] != 0.0))
    return np.array(g[:n_lags + 1])

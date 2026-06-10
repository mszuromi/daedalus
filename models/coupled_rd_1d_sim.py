"""
models/coupled_rd_1d_sim.py
===========================
N-species coupled stochastic reaction-diffusion Langevin simulator on a
1D ring (periodic boundary) + the exact linear "box correlator" oracle.
This is the physics oracle for COUPLED-FIELD theory validation (the
spatial Dyson series for multi-species theories with cross-noise and
per-species — possibly unequal — diffusion).

Dynamics (fields phi_i(x,t), i = 0..N-1, periodic grid of n_x points,
spacing dx = L/n_x):

    d_t phi_i = -sum_j M_ij phi_j + D_i d_x^2 phi_i - g_i phi_i^3 + xi_i,
    <xi_i(x,t) xi_j(x',t')> = Nnoise_ij delta(x-x') delta(t-t').

Conventions match ``models/spatial_field_1d_sim.py`` (the scalar 1D
oracle): finite-difference Laplacian on the ring, spatial white noise
discretized as ``delta(x-x') -> delta_mm'/dx`` so each Euler step adds
``chol(Nnoise) @ randn(N, n_x) * sqrt(dt/dx)``, and estimators are
periodic translational averages computed via FFT.

Integration schemes
-------------------
``scheme='semi_implicit'`` (default): Crank-Nicolson on the FULL linear
operator A_lat(q) = M + diag(D)*omega_lat(q) (reaction + FD Laplacian),
with the noise and the explicit nonlinear forcing routed THROUGH the
implicit solve:

    (I + dt/2 A) phi_{n+1} = (I - dt/2 A) phi_n + dt*f_nl(phi_n) + eta_n,
    eta_n = chol(Nnoise) @ randn * sqrt(dt/dx).

This update has the EXACT continuous-time stationary covariance of the
lattice system at ANY stable dt: writing B = (I+dt/2 A)^{-1}(I-dt/2 A),
R = (I+dt/2 A)^{-1}, the discrete stationary covariance solves
Sigma = B Sigma B^T + R (Nnoise dt) R^T; conjugating by (I+dt/2 A) gives
(I+dt/2 A)Sigma(I+dt/2 A)^T = (I-dt/2 A)Sigma(I-dt/2 A)^T + dt*Nnoise,
whose O(dt^2) terms cancel IDENTICALLY, leaving the continuous Lyapunov
equation A Sigma + Sigma A^T = Nnoise.  (The lag propagator is the
Pade(1,1) approximant B^s ~ e^{-A tau}, accurate to O(dt^2) per unit
time — negligible for the slow modes that survive at finite tau.)
The explicit half of the Laplacian is applied via ``np.roll`` finite
differences; the implicit solve uses the exact Fourier diagonalization
of the SAME FD operator (eigenvalue -omega_lat(q), omega_lat(q) =
(2/dx^2)(1-cos(q dx))), so the two halves are operator-consistent.

``scheme='explicit'``: plain Euler-Maruyama (np.roll Laplacian); biased
at O(dt) in the stationary statistics — kept as a cross-check.

Stability: the diffusive Courant number ``dt*max(D)/dx^2`` must stay
below the explicit threshold (0.5 for FD diffusion); we assert it is
< 0.4 for BOTH schemes so that the semi-implicit run remains in the
regime where the Pade lag bias is negligible, and the explicit scheme is
actually stable.

Matched-cutoff principle (same as the scalar simulators / their tests):
the simulator realizes the LATTICE dispersion omega_lat(q), not the
continuum q^2 — so quantitative validation compares against
``coupled_box_correlator(..., dispersion='lattice')`` (exact classical
result for the discretized system, no approximation), while
``dispersion='continuum'`` (the default) is the ground truth the Dyson
series is tested against (they differ by O(dx) on C(0,0): ~4% at
L=20, n_x=128).

Run the validation tests::

    sage -python -m pytest tests/test_coupled_rd_sim.py -q
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

__all__ = ['simulate_coupled_rd_1d', 'coupled_box_correlator']


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as_dvec(D, n_fields):
    """Per-species diffusion vector from scalar OR length-N input."""
    D = np.asarray(D, dtype=float)
    if D.ndim == 0:
        return np.full(n_fields, float(D))
    if D.shape != (n_fields,):
        raise ValueError(f'D must be a scalar or length-{n_fields} array; '
                         f'got shape {D.shape}.')
    return D


def _noise_factor(Nnoise):
    """Cholesky factor of the (symmetric PSD) noise matrix; falls back to
    the symmetric eigen square root for semidefinite input."""
    Nnoise = np.asarray(Nnoise, dtype=float)
    if not np.allclose(Nnoise, Nnoise.T):
        raise ValueError('Nnoise must be symmetric.')
    try:
        return np.linalg.cholesky(Nnoise)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Nnoise)
        if np.min(w) < -1e-12 * max(1.0, np.max(np.abs(w))):
            raise ValueError('Nnoise must be positive semidefinite.')
        return V * np.sqrt(np.clip(w, 0.0, None))


def _lattice_omega(n_x, dx):
    """FD-Laplacian eigenvalues omega_lat(q) = (2/dx^2)(1-cos(q dx)) >= 0
    on the FULL fft mode grid (length n_x)."""
    m = np.arange(int(n_x))
    return (2.0 / dx ** 2) * (1.0 - np.cos(2.0 * np.pi * m / int(n_x)))


def _resolve_lags(lags, dt, target_sample_dt=0.04):
    """Pick the snapshot stride and per-lag snapshot offsets.

    Snapshots are stored every ``stride`` steps with ``stride`` chosen so
    that (a) every requested positive lag is an integer number of strides
    (lags are first snapped to multiples of dt) and (b) the sampling
    interval stays near ``target_sample_dt`` (denser sampling only adds
    correlated, redundant pairs).  Returns (stride, lag_steps, taus_eff).
    """
    lags = [float(t) for t in lags]
    if any(t < 0 for t in lags):
        raise ValueError('lags must be non-negative (C(-tau) = C(tau)^T '
                         'with x -> -x by stationarity).')
    s_steps = [int(round(t / dt)) for t in lags]          # lag in dt units
    stride = max(1, int(round(target_sample_dt / dt)))
    for s in s_steps:
        if s > 0:
            stride = math.gcd(stride, s)
    lag_steps = [s // stride for s in s_steps]
    taus_eff = [s * dt for s in s_steps]
    return stride, lag_steps, taus_eff


# --------------------------------------------------------------------------
# simulator
# --------------------------------------------------------------------------

def simulate_coupled_rd_1d(M, D, Nnoise, g=None, L=20.0, n_x=128, dt=2e-3,
                           t_burn=20.0, t_run=200.0, n_rep=4, seed=0,
                           lags=(0.0,), x_lags=None, scheme='semi_implicit'):
    """Simulate the N-species coupled reaction-diffusion Langevin system on
    a periodic 1D grid and estimate the cross-correlation matrix

        C_ij(x_lag, tau_lag) = <phi_i(x + x_lag, t + tau_lag) phi_j(x, t)>

    averaged over x (periodic translational average via FFT), over time
    after burn-in, and over ``n_rep`` independent replicas.

    Parameters
    ----------
    M : (N, N) array — reaction/relaxation matrix (must be stable).
    D : scalar or (N,) array — per-species diffusion (unequal D supported).
    Nnoise : (N, N) symmetric PSD noise matrix (cross-noise allowed);
        its Cholesky factor mixes the per-step Gaussian draws.
    g : None or (N,) array — cubic coefficients (-g_i phi_i^3); None = linear.
    L, n_x : box size and number of grid points (dx = L/n_x).
    dt : time step.  Must satisfy the diffusive Courant constraint
        ``dt*max(D)/dx^2 < 0.4`` (asserted).
    t_burn, t_run : burn-in and measurement window (per replica).
    n_rep, seed : replica count and base RNG seed.
    lags : non-negative time lags tau (snapped to multiples of the
        snapshot interval; the snapped values are returned in 'taus').
    x_lags : None -> all grid lags (C has trailing axis n_x and 'x_grid'
        is the full lag grid).  A scalar -> C has shape (n_tau, N, N).
        A sequence -> trailing axis len(x_lags) (values snapped to dx).
    scheme : 'semi_implicit' (Crank-Nicolson linear part; dt-exact
        stationary covariance — see module docstring) or 'explicit'
        (plain Euler-Maruyama, O(dt) biased).

    Returns
    -------
    dict with
      'C'      : (n_tau, N, N[, n_xlag]) replica-mean correlator,
      'C_err'  : same shape — replica-spread standard error (std/sqrt(n_rep)),
      'C_rep'  : (n_rep, n_tau, N, N[, n_xlag]) per-replica estimates,
      'x_grid' : spatial lags actually used,
      'taus'   : time lags actually used (snapped),
      'meta'   : dx, dt, sample_dt, courant, scheme, ...
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f'M must be square; got shape {M.shape}.')
    n_f = M.shape[0]
    n_x = int(n_x)
    n_rep = int(n_rep)
    seed = int(seed)
    dt = float(dt)
    Dvec = _as_dvec(D, n_f)
    cholN = _noise_factor(Nnoise)
    if cholN.shape != (n_f, n_f):
        raise ValueError('Nnoise must be (N, N) with N = M.shape[0].')
    gvec = None if g is None else np.broadcast_to(
        np.asarray(g, dtype=float), (n_f,)).copy()

    # Coerce scalar arguments to plain python types: under the Sage NOTEBOOK
    # kernel, numeric literals arrive as sage Integer/RealNumber, which break
    # round()/np arithmetic below (no __round__).  Harmless under sage -python.
    L, dt = float(L), float(dt)
    t_burn, t_run = float(t_burn), float(t_run)
    n_x, n_rep, seed = int(n_x), int(n_rep), int(seed)
    lags = tuple(float(t) for t in lags)

    dx = L / n_x
    # Diffusive Courant constraint: explicit FD diffusion is stable for
    # dt*D/dx^2 < 0.5; we keep BOTH schemes below 0.4 (margin + Pade
    # accuracy for the semi-implicit lag propagator).
    courant = dt * float(np.max(Dvec)) / dx ** 2
    assert courant < 0.4, (
        f'dt*max(D)/dx^2 = {courant:.3f} >= 0.4: reduce dt or refine grid '
        f'(dt={dt}, dx={dx:.4g}, max D={float(np.max(Dvec)):.4g}).')
    if scheme not in ('semi_implicit', 'explicit'):
        raise ValueError(f"scheme must be 'semi_implicit' or 'explicit'; "
                         f"got {scheme!r}.")

    stride, lag_steps, taus_eff = _resolve_lags(lags, dt)
    sample_dt = stride * dt
    n_burn = int(round(t_burn / dt))
    n_rec = max(max(lag_steps) + 2, int(round(t_run / dt)) // stride)
    n_steps = n_rec * stride

    # Per-rfft-mode implicit CN solve matrices (real, (Mr, N, N)):
    # Pinv = (I + dt/2 * (M + diag(D)*omega_lat(q)))^{-1}.
    Mr = n_x // 2 + 1
    omega_r = _lattice_omega(n_x, dx)[:Mr]
    if scheme == 'semi_implicit':
        A_q = M[None, :, :] + omega_r[:, None, None] * np.diag(Dvec)[None]
        Pinv = np.linalg.inv(np.eye(n_f)[None] + 0.5 * dt * A_q)

    rng = np.random.default_rng(seed)
    sqrt_dtdx = np.sqrt(dt / dx)
    phi = np.zeros((n_rep, n_f, n_x), dtype=np.float64)
    snaps = np.empty((n_rec, n_rep, n_f, n_x), dtype=np.float64)

    def _step(phi):
        # np.roll FD Laplacian (ring) + reaction drift, explicit side.
        lap = (np.roll(phi, 1, axis=-1) - 2.0 * phi
               + np.roll(phi, -1, axis=-1)) / dx ** 2
        drift = (-np.einsum('ij,rjx->rix', M, phi)
                 + Dvec[None, :, None] * lap)
        eta = sqrt_dtdx * np.einsum(
            'ij,rjx->rix', cholN, rng.standard_normal((n_rep, n_f, n_x)))
        if gvec is not None:
            f_nl = -gvec[None, :, None] * phi ** 3
        else:
            f_nl = 0.0
        if scheme == 'explicit':
            return phi + dt * (drift + f_nl) + eta
        # semi-implicit CN: rhs through the implicit solve (incl. noise).
        rhs = phi + 0.5 * dt * drift + dt * f_nl + eta
        ahat = np.fft.rfft(rhs, axis=-1)              # (n_rep, N, Mr)
        ahat = np.einsum('kij,rjk->rik', Pinv, ahat)
        return np.fft.irfft(ahat, n=n_x, axis=-1)

    for _ in range(n_burn):
        phi = _step(phi)
    ri = 0
    for step in range(n_steps):
        phi = _step(phi)
        if (step + 1) % stride == 0:
            snaps[ri] = phi
            ri += 1

    # ---- FFT cross-correlation estimator -------------------------------
    # C_ij(r) = ifft( fft(phi_i) * conj(fft(phi_j)) )[r] / n_x, averaged
    # over snapshots (pairs t, t+lag) and replicas.
    F = np.fft.fft(snaps, axis=-1)                    # (n_rec, n_rep, N, n_x)
    n_tau = len(lag_steps)
    C_rep = np.empty((n_rep, n_tau, n_f, n_f, n_x), dtype=np.float64)
    for k, ell in enumerate(lag_steps):
        n_pair = n_rec - ell
        S = np.einsum('trik,trjk->rijk', F[ell:n_rec],
                      np.conj(F[:n_pair]))            # (n_rep, N, N, n_x)
        C_rep[:, k] = np.fft.ifft(S, axis=-1).real / (n_x * n_pair)

    # ---- x-lag selection -------------------------------------------------
    scalar_x = (x_lags is not None and np.ndim(x_lags) == 0)
    if x_lags is None:
        x_grid = np.arange(n_x) * dx
    else:
        xl = np.atleast_1d(np.asarray(x_lags, dtype=float))
        idx = np.array([int(round(x / dx)) % n_x for x in xl])
        C_rep = C_rep[..., idx]
        x_grid = idx * dx
    C = C_rep.mean(axis=0)
    C_err = C_rep.std(axis=0, ddof=1) / np.sqrt(n_rep)
    if scalar_x:
        C, C_err, C_rep = C[..., 0], C_err[..., 0], C_rep[..., 0]
        x_grid = float(x_grid[0])

    meta = {'dx': dx, 'dt': dt, 'sample_dt': sample_dt, 'stride': stride,
            'L': L, 'n_x': n_x, 'n_fields': n_f, 'n_rec': n_rec,
            'n_rep': n_rep, 'scheme': scheme, 'courant': courant,
            'k_max': np.pi / dx, 'seed': seed}
    return {'C': C, 'C_err': C_err, 'C_rep': C_rep,
            'x_grid': x_grid, 'taus': np.asarray(taus_eff), 'meta': meta}


# --------------------------------------------------------------------------
# exact linear oracle on the periodic box (pure numpy/scipy — no sim)
# --------------------------------------------------------------------------

def coupled_box_correlator(M, Dvec, Nnoise, L, n_x, taus,
                           dispersion='continuum'):
    """Exact stationary two-point matrix of the LINEAR coupled system on the
    periodic box — classical result, no approximation:

        C_ij(x, tau) = (1/L) * sum_q [ expm(-A(q)|tau|) @ Sigma(q) ]_ij e^{iqx},

    summed over the box modes q = 2*pi*n/L, |n| <= n_x/2 (the n_x fft
    modes), where A(q) = M + diag(D)*omega(q) and Sigma(q) solves the
    Lyapunov equation A Sigma + Sigma A^T = Nnoise
    (scipy.linalg.solve_continuous_lyapunov).  Unequal per-species D is
    exact here (full matrix exponential per mode) — THIS is the ground
    truth the coupled Dyson series is validated against.

    dispersion : 'continuum' (default) -> omega(q) = q^2, the physical box
        oracle; 'lattice' -> omega(q) = (2/dx^2)(1-cos(q dx)), the exact
        statistics of the FD-discretized simulator (matched-cutoff
        comparison, same principle as the scalar-sim tests).

    Returns dict {'C': (n_tau, N, N, n_x), 'x_grid': (n_x,) lag grid,
    'taus': (n_tau,)} with the same layout/normalization as
    :func:`simulate_coupled_rd_1d` (C[..., r] = C(x = r*dx)).
    """
    L = float(L)
    n_x = int(n_x)
    taus = [float(t) for t in taus]
    M = np.asarray(M, dtype=float)
    n_f = M.shape[0]
    n_x = int(n_x)
    Dvec = _as_dvec(Dvec, n_f)
    Nnoise = np.asarray(Nnoise, dtype=float)
    dx = L / n_x
    q = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)       # 2*pi*n/L, |n|<=n_x/2
    if dispersion == 'continuum':
        omega = q ** 2
    elif dispersion == 'lattice':
        omega = _lattice_omega(n_x, dx)
    else:
        raise ValueError(f"dispersion must be 'continuum' or 'lattice'; "
                         f"got {dispersion!r}.")
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    # Per-mode C(q, tau) = expm(-A|tau|) @ Sigma  (tau >= 0).
    Cq = np.empty((len(taus), n_f, n_f, n_x), dtype=np.float64)
    for m in range(n_x):
        A = M + np.diag(Dvec) * omega[m]
        Sigma = solve_continuous_lyapunov(A, Nnoise)
        for k, tau in enumerate(taus):
            G = expm(-A * abs(float(tau)))
            Cm = G @ Sigma if tau >= 0 else Sigma @ G.T
            Cq[k, :, :, m] = Cm
    # C(x_r) = (1/L) sum_n Cq_n e^{i q_n r dx} = (n_x/L) * ifft over modes.
    C = (n_x / L) * np.fft.ifft(Cq, axis=-1).real
    return {'C': C, 'x_grid': np.arange(n_x) * dx, 'taus': taus}

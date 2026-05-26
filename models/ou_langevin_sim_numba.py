"""
Euler-Maruyama simulators for scalar Langevin SDEs (OU-type theories).

Distinct from the Poisson-Hawkes simulators in ``hawkes_sim_multipop_numba.py``:
no spike events, no synaptic filters — just a continuous stochastic ODE
``dx/dt = drift(x) + √(2D) η(t)`` integrated by Euler-Maruyama.

Output shape matches the Hawkes ``voltage_bins`` convention
(``np.ndarray (npop=1, n_bins)``) so the same downstream cumulant
estimator ``models.cumulant_estimator.compute_kpoint_slice`` can be
used with ``field_types=['dv', 'dv', ...]``.
"""
import numpy as np
import numba


@numba.njit
def sim_ou_quartic_numba(n_steps, dt_sim,
                         mu, eps, D,
                         x_init,
                         bin_size_steps, n_bins, seed):
    """
    Euler-Maruyama for the scalar quartic Langevin

        dx/dt = -mu·x - eps·x³ + √(2D)·η(t),     ⟨η η⟩ = δ(t − t')

    matching the action in ``theories/ou_quartic_double_well.theory.py``:

        S = ∫ dt  xt·((Dt + mu)·x + eps·x³) - D·xt²

    Parameters
    ----------
    n_steps : int
        Number of Euler-Maruyama steps.
    dt_sim : float
        Time-step (small relative to the relaxation time 1/mu).
    mu, eps, D : float
        Linear restoring force, cubic non-linearity, noise intensity.
    x_init : float
        Initial value of x.  Use the MF saddle (= 0) for stationary
        statistics; transient is forgotten on timescale 1/mu.
    bin_size_steps : int
        Number of Euler steps per output bin (``dt_bin = dt_sim ×
        bin_size_steps``).
    n_bins : int
        Number of output bins.  ``n_steps`` should be at least
        ``bin_size_steps × n_bins``.
    seed : int
        np.random.seed() value.

    Returns
    -------
    x_bins : np.ndarray (1, n_bins)
        Bin-averaged x trajectory.  Shape matches the Hawkes
        ``voltage_bins`` convention so ``compute_kpoint_slice`` with
        ``field_types=['dv']`` can consume it directly.

    Notes
    -----
    * Sign convention follows the theory file: positive ``mu`` =
      stable origin (sub-critical); ``mu = 0`` = pitchfork
      bifurcation; ``mu < 0`` = double-well wells at
      ``x = ±√(|mu|/eps)``.
    * Stationary variance at mu > 0 with eps = 0 (pure OU): D/mu.
      For finite eps the perturbative corrections shift this by
      ``O(eps · D² / mu³)``.
    """
    np.random.seed(seed)
    x = x_init
    x_bins = np.zeros((1, n_bins))
    accum = 0.0
    cur_bin = 0
    steps_in_bin = 0
    sqrt_2D_dt = np.sqrt(2.0 * D * dt_sim)
    for step in range(n_steps):
        if cur_bin < n_bins:
            accum += x
        # Euler-Maruyama step:
        # x(t+dt) = x(t) + dt·(-mu·x - eps·x³) + √(2D·dt)·N(0,1)
        x = (x
             + dt_sim * (-mu * x - eps * x * x * x)
             + sqrt_2D_dt * np.random.randn())
        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if cur_bin < n_bins:
                x_bins[0, cur_bin] = accum / bin_size_steps
                accum = 0.0
            cur_bin += 1
            steps_in_bin = 0
    return x_bins


@numba.njit
def sim_ou_sextic_numba(n_steps, dt_sim,
                        mu, eps, gamma, D,
                        x_init,
                        bin_size_steps, n_bins, seed):
    """
    Euler-Maruyama for the cubic+quintic Langevin

        dx/dt = -mu·x - eps·x³ - gamma·x⁵ + √(2D)·η(t)

    matching the action in ``theories/ou_sextic.theory.py``:

        S = ∫ dt  xt·((Dt+mu)·x + eps·x³ + gamma·x⁵) - D·xt²

    The corresponding equilibrium Boltzmann potential is

        U(x) = (mu/2)·x² + (eps/4)·x⁴ + (gamma/6)·x⁶

    which is bounded below (= stable, normalizable) when mu, eps,
    gamma > 0 — including the gamma > 0 case that the pure-quintic
    drift ``-gamma·x⁵`` alone would have made unstable were the
    cubic term absent (because eps > 0 added a x⁴/4 term to U that
    suppresses any odd-power instability).

    Parameters
    ----------
    n_steps, dt_sim, x_init, bin_size_steps, n_bins, seed
        Same as ``sim_ou_quartic_numba``.
    mu, eps, gamma, D : float
        Linear, cubic, quintic, and noise-intensity coefficients.

    Returns
    -------
    x_bins : np.ndarray (1, n_bins)
        Bin-averaged x trajectory.
    """
    np.random.seed(seed)
    x = x_init
    x_bins = np.zeros((1, n_bins))
    accum = 0.0
    cur_bin = 0
    steps_in_bin = 0
    sqrt_2D_dt = np.sqrt(2.0 * D * dt_sim)
    for step in range(n_steps):
        if cur_bin < n_bins:
            accum += x
        # Euler-Maruyama step:
        # x(t+dt) = x(t) + dt·(-mu·x - eps·x³ - gamma·x⁵)
        #         + √(2D·dt)·N(0,1)
        x2 = x * x
        x = (x
             + dt_sim * (-mu * x - eps * x2 * x - gamma * x2 * x2 * x)
             + sqrt_2D_dt * np.random.randn())
        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if cur_bin < n_bins:
                x_bins[0, cur_bin] = accum / bin_size_steps
                accum = 0.0
            cur_bin += 1
            steps_in_bin = 0
    return x_bins


@numba.njit
def sim_ou_quartic_two_dim_numba(
        n_steps, dt_sim,
        mu1, mu2, eps1, eps2, J1, J2, D1, D2,
        x_init, y_init,
        bin_size_steps, n_bins, seed):
    """
    Euler-Maruyama for the 2-component coupled quartic Langevin

        dx/dt = -mu1·x - eps1·x³ + J1·y + √(2 D1)·η_x(t)
        dy/dt = -mu2·y - eps2·y³ + J2·x + √(2 D2)·η_y(t)

    matching the action in ``theories/ou_quartic_two_dim.theory.py``:

        S = xt·((Dt + mu1)·x + eps1·x³ - J1·y) - D1·xt²
          + yt·((Dt + mu2)·y + eps2·y³ - J2·x) - D2·yt²

    Cross-coupling enters drift via J1 (on x) and J2 (on y).  Each
    field has its own quartic non-linearity and noise strength.

    Linear stability at the trivial saddle (0, 0) requires
    ``|J1·J2| < mu1·mu2``; the bifurcation lives on
    ``J1·J2 = mu1·mu2``.

    Returns
    -------
    bins : np.ndarray (2, n_bins)
        Bin-averaged (x, y) trajectories stacked along axis 0.
        ``bins[0]`` = x, ``bins[1]`` = y.  Shape matches the
        ``voltage_bins`` convention so ``compute_kpoint_slice`` with
        ``field_types=['dv', 'dv']`` can consume it directly.
    """
    np.random.seed(seed)
    x = x_init
    y = y_init
    bins = np.zeros((2, n_bins))
    accum_x = 0.0
    accum_y = 0.0
    cur_bin = 0
    steps_in_bin = 0
    sqrt_2D1_dt = np.sqrt(2.0 * D1 * dt_sim)
    sqrt_2D2_dt = np.sqrt(2.0 * D2 * dt_sim)
    for step in range(n_steps):
        if cur_bin < n_bins:
            accum_x += x
            accum_y += y
        drift_x = -mu1 * x - eps1 * x * x * x + J1 * y
        drift_y = -mu2 * y - eps2 * y * y * y + J2 * x
        x_new = x + dt_sim * drift_x + sqrt_2D1_dt * np.random.randn()
        y_new = y + dt_sim * drift_y + sqrt_2D2_dt * np.random.randn()
        x = x_new
        y = y_new
        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if cur_bin < n_bins:
                bins[0, cur_bin] = accum_x / bin_size_steps
                bins[1, cur_bin] = accum_y / bin_size_steps
                accum_x = 0.0
                accum_y = 0.0
            cur_bin += 1
            steps_in_bin = 0
    return bins

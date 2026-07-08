"""
Euler-Maruyama simulators for scalar Langevin SDEs (OU-type models).

Distinct from the Poisson-Hawkes simulators in ``hawkes_sim_multipop_numba.py``:
no spike events, no synaptic filters — just a continuous stochastic ODE
``dx/dt = drift(x) + √(2D) η(t)`` integrated by Euler-Maruyama.

Output shape matches the Hawkes ``voltage_bins`` convention
(``np.ndarray (npop=1, n_bins)``) so the same downstream cumulant
estimator ``simulations.cumulant_estimator.compute_kpoint_slice`` can be
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

    matching the action in ``models/ou_quartic_double_well.model.py``:

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
    * Sign convention follows the model file: positive ``mu`` =
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
def sim_ou_quartic_colored_numba(n_steps, dt_sim,
                                  mu, eps, D, tauc,
                                  x_init,
                                  bin_size_steps, n_bins, seed):
    """
    Scalar quartic Langevin driven by COLORED Gaussian noise.
    Matches the κ² block in ``models/ou_quartic_colored.model.py``.

    System
    ------
    Auxiliary OU noise process (stationary, Lorentzian autocorrelation):

        dξ/dt = -ξ / τc + (√(2 D)/τc) · η(t),      ⟨η η⟩ = δ(t − t')

    so that

        ⟨ξ(t) ξ(t')⟩ = (2 D / τc) · exp(-|t − t'|/τc)

    matching the framework's ``coefficient='2*D/tauc'`` ×
    ``kernel='exp(-abs(tau)/tauc)'``.

    Deterministic dynamics driven by ξ:

        dx/dt = -mu·x - eps·x³ + ξ(t)

    Discretization
    --------------
    Auxiliary OU process: **exact** discretization,
        ξ(t + dt) = decay · ξ(t) + σ · η,    σ² = (2D/τc)(1 − decay²),
        decay = exp(-dt/τc).
    Preserves the stationary variance for any dt/τc ratio.

    (x) dynamics: Euler step ``x += dt·(drift + ξ)``.  Valid in the
    regime dt ≪ τc.  In the white limit τc → 0 (held with dt < τc),
    ξ(t) becomes a white Gaussian with ⟨ξ ξ⟩ → 4D·δ(τ) — note the
    factor of 4D rather than 2D from the framework's coefficient
    convention; if you want to compare numerically to the ``2D``
    white-noise variant ``sim_ou_quartic_numba`` use ``D_white =
    2 · D_colored`` (or equivalently halve D when crossing over).

    Returns
    -------
    x_bins : np.ndarray (1, n_bins)
        Bin-averaged x trajectory.  Shape matches the Hawkes
        ``voltage_bins`` convention so ``compute_kpoint_slice`` with
        ``field_types=['dv']`` consumes it directly.
    """
    np.random.seed(seed)
    x = x_init
    xi = 0.0

    # Exact OU discretization for the auxiliary noise process.
    decay = np.exp(-dt_sim / tauc)
    var_factor = 1.0 - decay * decay
    sigma_xi = np.sqrt((2.0 * D / tauc) * var_factor)

    x_bins = np.zeros((1, n_bins))
    accum = 0.0
    cur_bin = 0
    steps_in_bin = 0

    for step in range(n_steps):
        if cur_bin < n_bins:
            accum += x

        eta = np.random.randn()
        xi_new = decay * xi + sigma_xi * eta

        drift = -mu * x - eps * x * x * x
        x_new = x + dt_sim * (drift + xi)

        x = x_new
        xi = xi_new

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

    matching the action in ``models/ou_sextic.model.py``:

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
def sim_ou_quartic_two_dim_color_corr_numba(
        n_steps, dt_sim,
        mu1, mu2, eps1, eps2, J1, J2, D1, D2,
        tauc, rho,
        x_init, y_init,
        bin_size_steps, n_bins, seed):
    """
    Coupled-quartic 2D Langevin driven by COLORED, CROSS-CORRELATED
    Gaussian noise.  Matches the κ² block in
    ``models/ou_quartic_two_dim_color_corr.model.py``.

    System
    ------
    Auxiliary OU noise process (stationary, Lorentzian autocorrelation):

        dξ_x/dt = -ξ_x / τc + (√(2 D1) / τc) · η_x(t)
        dξ_y/dt = -ξ_y / τc + (√(2 D2) / τc) · η_y(t)

    where the WHITE drivers (η_x, η_y) are jointly Gaussian with
    covariance ⟨η_α(t) η_β(t')⟩ = C_αβ · δ(t-t'), C = [[1, ρ], [ρ, 1]].
    By stationarity, the auxiliary process has

        ⟨ξ_x(t) ξ_x(t')⟩ = (D1 / τc) · exp(-|t-t'|/τc)
        ⟨ξ_y(t) ξ_y(t')⟩ = (D2 / τc) · exp(-|t-t'|/τc)
        ⟨ξ_x(t) ξ_y(t')⟩ = (ρ · √(D1 D2) / τc) · exp(-|t-t'|/τc)

    matching the framework's `Cxx`, `Cyy`, `Cxy` cumulants exactly.

    Deterministic dynamics driven by the OU noise:

        dx/dt = -μ_1 x - ε_1 x³ + J_1 y + ξ_x(t)
        dy/dt = -μ_2 y - ε_2 y³ + J_2 x + ξ_y(t)

    Discretization
    --------------
    Auxiliary OU process: **exact** discretization.  Each step uses the
    closed-form solution of the linear OU SDE, ξ(t + dt) = ξ(t)·e^{-dt/τc}
    + σ·η, with σ² = (D/τc) · (1 - e^{-2 dt/τc}).  This preserves the
    stationary variance for any dt/τc ratio and avoids the bias that
    Euler-Maruyama on the auxiliary process accumulates near small τc.

    (x, y) dynamics: Euler step `x += dt · (drift + ξ_x)`.  Valid in
    the regime dt ≪ τc (the noise is approximately constant over one
    step).  For dt ≳ τc the Euler-on-x step under-counts the noise
    increment's variance — the rigorous noise-integral variance over
    one step is `2 D dt - 2 D τc (1 - e^{-dt/τc})`, which agrees with
    Euler in dt ≪ τc and with white-noise increment `2 D dt` in
    dt ≫ τc.  For default τc = 2.0 and dt = 0.01, the regime is well
    within dt ≪ τc.

    Correlation sampling
    --------------------
    The joint white driver (η_x, η_y) is sampled via Cholesky
    factorization of C = [[1, ρ], [ρ, 1]]:

        u, v ← independent N(0, 1)
        η_x = u
        η_y = ρ · u + √(1 - ρ²) · v

    ρ ∈ [-1, 1] strictly; outside that range the matrix loses positive
    semi-definiteness (no valid Cholesky factor) and the sim returns
    NaN.  The square root takes max(0, 1-ρ²) defensively.

    White limit
    -----------
    As τc → 0 (with dt held fixed and dt < τc), the auxiliary process
    becomes white Gaussian with ⟨ξ_α(t) ξ_β(t')⟩ → 2 D_α C_αβ · δ(t-t'),
    recovering the existing ``sim_ou_quartic_two_dim_numba`` when ρ = 0.
    Use τc = 0.01-ish with dt = 0.001 for a numerical white-limit check.

    Returns
    -------
    bins : np.ndarray (2, n_bins)
        Bin-averaged (x, y) trajectories.  Same shape as the
        white-noise sim so notebook plumbing drops in unchanged.
    """
    np.random.seed(seed)
    x = x_init
    y = y_init
    xi_x = 0.0
    xi_y = 0.0

    # Exact OU discretization for the auxiliary process:
    # ξ(t + dt) = decay · ξ(t) + σ · η  with σ² = (D/τc)(1 - decay²)
    decay = np.exp(-dt_sim / tauc)
    var_factor = 1.0 - decay * decay
    sigma_x = np.sqrt((D1 / tauc) * var_factor)
    sigma_y = np.sqrt((D2 / tauc) * var_factor)

    # Cholesky factor of [[1, ρ], [ρ, 1]]: η_y = ρ·u + √(1-ρ²)·v.
    one_minus_rho_sq = 1.0 - rho * rho
    if one_minus_rho_sq < 0.0:
        one_minus_rho_sq = 0.0
    rho_perp = np.sqrt(one_minus_rho_sq)

    bins = np.zeros((2, n_bins))
    accum_x = 0.0
    accum_y = 0.0
    cur_bin = 0
    steps_in_bin = 0

    for step in range(n_steps):
        if cur_bin < n_bins:
            accum_x += x
            accum_y += y

        # Sample correlated unit Gaussians.
        u = np.random.randn()
        v = np.random.randn()
        eta_x = u
        eta_y = rho * u + rho_perp * v

        # Step the OU auxiliary noise (exact).
        xi_x_new = decay * xi_x + sigma_x * eta_x
        xi_y_new = decay * xi_y + sigma_y * eta_y

        # Step (x, y) with colored-noise forcing.
        drift_x = -mu1 * x - eps1 * x * x * x + J1 * y
        drift_y = -mu2 * y - eps2 * y * y * y + J2 * x
        x_new = x + dt_sim * (drift_x + xi_x)
        y_new = y + dt_sim * (drift_y + xi_y)

        x = x_new
        y = y_new
        xi_x = xi_x_new
        xi_y = xi_y_new

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


@numba.njit
def sim_ou_quartic_two_dim_corr_numba(
        n_steps, dt_sim,
        mu1, mu2, eps1, eps2, J1, J2, D1, D2, rho,
        x_init, y_init,
        bin_size_steps, n_bins, seed):
    """
    Euler-Maruyama for the 2-component coupled quartic Langevin with
    WHITE, CROSS-CORRELATED Gaussian noise.  Same dynamics as
    ``sim_ou_quartic_two_dim_numba`` but with non-zero noise
    correlation ρ between the η_x and η_y drivers — matches the κ²
    block in ``models/ou_quartic_two_dim_corr.model.py``.

    System
    ------

        dx/dt = -μ_1 x - ε_1 x³ + J_1 y + √(2 D_1) η_x(t)
        dy/dt = -μ_2 y - ε_2 y³ + J_2 x + √(2 D_2) η_y(t)

        ⟨η_α(t) η_β(t')⟩ = C_αβ · δ(t-t'),   C = [[1, ρ], [ρ, 1]]

    which gives the noise correlators

        ⟨ξ_x(t) ξ_x(t')⟩ = 2 D_1 · δ(t-t')
        ⟨ξ_y(t) ξ_y(t')⟩ = 2 D_2 · δ(t-t')
        ⟨ξ_x(t) ξ_y(t')⟩ = 2 ρ √(D_1 D_2) · δ(t-t')

    where ξ_α = √(2 D_α) · η_α is the increment driver.

    Convention note
    ---------------
    This sim uses the standard MSR-JD convention ⟨ξ ξ⟩ = 2D·δ(τ).
    The model file as currently written declares
    ``coefficient='D1'``, which corresponds to ⟨ξ ξ⟩ = D·δ(τ) (half-
    strength).  For a quantitative model-vs-sim comparison, either
    edit the model file to ``coefficient='2*D1'`` (and similarly for
    ``D2`` and ``Cxy``'s coefficient), OR rerun this sim with
    half-strength D values.  Speed-test comparisons are unaffected.

    Correlation sampling
    --------------------
    Cholesky factorization of C = [[1, ρ], [ρ, 1]]:
        u, v ← independent N(0, 1)
        η_x = u
        η_y = ρ · u + √(1 - ρ²) · v
    ρ ∈ [-1, 1] strictly; outside that the matrix loses positive
    semi-definiteness.  The square root takes max(0, 1-ρ²) defensively.

    Returns
    -------
    bins : np.ndarray (2, n_bins)
        Bin-averaged (x, y) trajectories stacked along axis 0.  Same
        layout as the independent white-noise sim so notebook plumbing
        drops in unchanged.
    """
    np.random.seed(seed)
    x = x_init
    y = y_init
    sqrt_2D1_dt = np.sqrt(2.0 * D1 * dt_sim)
    sqrt_2D2_dt = np.sqrt(2.0 * D2 * dt_sim)

    # Cholesky factor of [[1, ρ], [ρ, 1]]: η_y = ρ·u + √(1-ρ²)·v.
    one_minus_rho_sq = 1.0 - rho * rho
    if one_minus_rho_sq < 0.0:
        one_minus_rho_sq = 0.0
    rho_perp = np.sqrt(one_minus_rho_sq)

    bins = np.zeros((2, n_bins))
    accum_x = 0.0
    accum_y = 0.0
    cur_bin = 0
    steps_in_bin = 0

    for step in range(n_steps):
        if cur_bin < n_bins:
            accum_x += x
            accum_y += y

        u = np.random.randn()
        v = np.random.randn()
        eta_x = u
        eta_y = rho * u + rho_perp * v

        drift_x = -mu1 * x - eps1 * x * x * x + J1 * y
        drift_y = -mu2 * y - eps2 * y * y * y + J2 * x
        x_new = x + dt_sim * drift_x + sqrt_2D1_dt * eta_x
        y_new = y + dt_sim * drift_y + sqrt_2D2_dt * eta_y

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

    matching the action in ``models/ou_quartic_two_dim.model.py``:

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

"""
models/hawkes_sim_linear_expg_gtas_numba.py
=========================================
Numba-JIT Euler-step simulator for the LINEAR Hawkes 2-population
network driven by **two correlated external neurons** generated via a
GTaS (Generalized Thinning and Shift) process with **Bernoulli
participation** and **Gaussian shifts**, plus the existing
exponential synaptic filter ``g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)``.

Network architecture
--------------------
    external[1] ──J──▶ cortical[1]
    external[2] ──J──▶ cortical[2]            (one-to-one feedforward)

    cortical[1] ⇄ cortical[2]                 (recurrent linear Hawkes)

The cortical layer is identical to ``hawkes_sim_expg_numba``: 2
neurons with linear gain ``phi(v) = a v`` and synaptic filter
``g``.  The EXTERNAL layer is a 2-neuron GTaS process with its
own correlation structure (same as the quadratic variant).

GTaS process (Trousdale et al. 2013)
------------------------------------
* Mother Poisson process of rate ``lambda_X`` on the simulation
  interval.  Each event of the mother process is independently
  marked with a subset D ⊆ {1, 2} via Bernoulli participation:
  each cell joins with probability ``p_part``, independent across
  cells per event.  So:

      p_{∅}   = (1 - p_part)^2
      p_{1}   = p_{2} = p_part * (1 - p_part)
      p_{1,2} = p_part^2

* For each event with assigned subset D, draw a 2-dim shift vector
  ``Y ~ N(mu_shift, Sigma_shift)``.  For each i ∈ D, place a spike
  in external process i at time ``t_event + Y_i``.
* Marginal rate per external cell: ``lambda_X * p_part``
  (each external is marginally Poisson at this rate).

The cumulant structure of this process is fully determined by the
above parameters.  In particular the only nontrivial higher-order
cumulant for N=2 is the order-2 cross-cumulant:

    kappa^(2)_{12}(tau)  =  lambda_X * p_part^2 * Gauss(tau; mu_diff, sigma_diff^2)

with ``mu_diff = mu_shift[1] - mu_shift[0]`` and
``sigma_diff^2 = Sigma_shift[0,0] + Sigma_shift[1,1] - 2*Sigma_shift[0,1]``.

Coupling to cortical neurons
----------------------------
External spikes flow through the SAME exponential synaptic filter g
that recurrent connections use.  Each cortical neuron receives input
from EXACTLY ONE external (one-to-one):

    cortical i ← external (i)         (i.e. sigma(i) = i)

The synaptic strength is the scalar ``w_X``.

Continuous-time dynamics
------------------------
    dF_j     /dt =  -F_j     /tau_g + n_j(t)/tau_g            (recurrent filter)
    dF_X_i   /dt =  -F_X_i   /tau_g + n_X_i(t)/tau_g          (feedforward filter)
    dv_i     /dt =  (1/tau)*(-v_i + E_i + sum_j w_ij F_j + w_X * F_X_i)
    lambda_i (t) =  max(a * v_i, 0)
    n_i(t)       ~  Poisson(lambda_i(t) dt)

The external spike trains ``n_X_i(t)`` are PRE-GENERATED via the
GTaS sampling procedure described above, then replayed during the
Euler loop.

Usage (from a Sage notebook cell)
---------------------------------
    from models.hawkes_sim_linear_expg_gtas_numba import (
        sample_gtas_external_spikes,
        sim_hawkes_linear_expg_gtas_numba,
    )

    ext_spike_grid = sample_gtas_external_spikes(
        T_total=float(T_sim),  dt_sim=float(dt_sim),
        lambda_X=float(lambda_X),  p_part=float(p_part),
        mu_shift=mu_shift_arr,  sigma_shift=sigma_shift_arr,
        seed=int(seed_ext),
    )
    binned, voltage, totals, ext_binned = sim_hawkes_linear_expg_gtas_numba(
        int(n_steps), float(dt_sim), float(tau), float(tau_g),
        float(a), float(w_X), E_arr, W_arr, v_init.copy(),
        ext_spike_grid,
        int(bin_size_steps), int(n_bins), int(seed),
    )

Sage preparser note: cast every integer/float via ``int()`` / ``float()``
at the call site -- ``Integer(0)`` / ``RealLiteral`` cannot be typed by
Numba.
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# GTaS external spike sampler  (Python, not jitted -- runs once per sim)
# ---------------------------------------------------------------------------

def sample_gtas_external_spikes(T_total, dt_sim, lambda_X, p_part,
                                mu_shift, sigma_shift, seed):
    """Sample the 2-cell GTaS external process and return its events
    binned onto the simulation grid.

    Parameters
    ----------
    T_total : float
        Total simulation time (in same units as dt_sim).
    dt_sim : float
        Euler timestep — also the bin width for the returned grid.
    lambda_X : float
        Mother Poisson rate (events per unit time).
    p_part : float
        Bernoulli participation probability per cell, in [0, 1].
    mu_shift : np.ndarray (2,)
        Mean shift vector per cell (for joint events).
    sigma_shift : np.ndarray (2, 2)
        Covariance of the joint shift vector.  Used for events with
        |D| = 2 (both cells fire).  For solo events (|D| = 1) only
        the diagonal entry ``sigma_shift[i, i]`` is used.
    seed : int
        RNG seed.

    Returns
    -------
    ext_spike_grid : np.ndarray (2, n_steps)  (int32)
        Number of external spikes per cell per Euler timestep.  Each
        bin can contain >= 0 spikes (Gaussian shifts can pile two
        events into the same bin).

    Notes
    -----
    * The procedure is fully marginal: any pair (lambda_X, p_part)
      is acceptable; in particular ``p_part = 0`` produces zero
      external spikes (degenerate), and ``lambda_X = 0`` produces
      none either.
    * Events whose shifted time falls outside [0, T_total] are
      clipped (dropped) -- this corresponds to the natural finite-
      window assumption of the simulation.
    * For p_part > 0, the marginal external rate per cell is
      ``lambda_X * p_part``.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T_total / dt_sim))

    # 1. Sample mother Poisson events on [0, T_total].
    n_mother = rng.poisson(lambda_X * T_total)
    if n_mother == 0:
        return np.zeros((2, n_steps), dtype=np.int32)
    mother_times = np.sort(rng.uniform(0.0, T_total, size=n_mother))

    # 2. For each mother event, decide independently whether each
    #    cell participates (Bernoulli per cell).
    participation = rng.random((n_mother, 2)) < p_part   # bool

    # 3. Build daughter event times.  We need to handle three cases
    #    per mother event based on |D|:
    #      |D| = 0 : do nothing.
    #      |D| = 1 : single cell fires; shift drawn from N(mu_i, sigma_ii).
    #      |D| = 2 : both fire; shift drawn from N(mu, Sigma) (joint).
    daughter_times_per_cell = [[], []]
    for k in range(n_mother):
        t_event = mother_times[k]
        D_count = int(participation[k, 0]) + int(participation[k, 1])
        if D_count == 0:
            continue
        if D_count == 2:
            # Joint event with full Gaussian shift
            Y = rng.multivariate_normal(mu_shift, sigma_shift)
            daughter_times_per_cell[0].append(t_event + Y[0])
            daughter_times_per_cell[1].append(t_event + Y[1])
        else:
            # Solo event for whichever cell participated
            i = 0 if participation[k, 0] else 1
            std_i = float(np.sqrt(sigma_shift[i, i]))
            mu_i = float(mu_shift[i])
            Y_i = rng.normal(mu_i, std_i)
            daughter_times_per_cell[i].append(t_event + Y_i)

    # 4. Bin into the Euler grid; clip to [0, T_total).
    ext_spike_grid = np.zeros((2, n_steps), dtype=np.int32)
    for i in range(2):
        if not daughter_times_per_cell[i]:
            continue
        times = np.asarray(daughter_times_per_cell[i], dtype=np.float64)
        # Filter to in-window
        mask = (times >= 0.0) & (times < T_total)
        times = times[mask]
        if times.size == 0:
            continue
        bins = (times / dt_sim).astype(np.int64)
        # numpy.bincount is faster than a Python loop for large counts
        counts = np.bincount(bins, minlength=n_steps)
        # Truncate to n_steps in case rounding produced one extra bin
        ext_spike_grid[i, :n_steps] = counts[:n_steps].astype(np.int32)

    return ext_spike_grid


# ---------------------------------------------------------------------------
# Cortical Euler-step simulator  (Numba, hot path)
# ---------------------------------------------------------------------------

@numba.njit
def sim_hawkes_linear_expg_gtas_numba(n_steps, dt_sim, tau, tau_g, a, w_X,
                                    E, W, v_init,
                                    ext_spike_grid,
                                    bin_size_steps, n_bins, seed):
    """
    Euler-step simulator of the linear, exp-filtered Hawkes process
    with GTaS-correlated external feedforward inputs.

    Parameters
    ----------
    n_steps : int
        Total number of Euler timesteps.
    dt_sim : float
        Euler timestep.
    tau, tau_g : float
        Membrane and synaptic-filter time constants.
    a : float
        Quadratic transfer-function gain.
    w_X : float
        Feedforward weight (one-to-one external -> cortical).
    E : np.ndarray (npop,)
        Constant external drive per cortical population.
    W : np.ndarray (npop, npop)
        Recurrent synaptic weight matrix.
    v_init : np.ndarray (npop,)
        Initial cortical voltage.
    ext_spike_grid : np.ndarray (npop, n_steps), int32
        Pre-sampled external spike counts per timestep, from
        ``sample_gtas_external_spikes``.  Must have second dim
        >= n_steps.  Cell i feeds cortical i (one-to-one).
    bin_size_steps, n_bins : int
        Binning parameters for the returned spike-count arrays.
    seed : int
        RNG seed for the cortical Poisson draws.

    Returns
    -------
    binned_counts : np.ndarray (npop, n_bins), float64
        Cortical spike counts per bin.
    voltage_bins : np.ndarray (npop, n_bins), float64
        Mean cortical voltage per bin.
    total_spikes : np.ndarray (npop,), float64
        Total cortical spike count per population.
    ext_binned_counts : np.ndarray (npop, n_bins), float64
        External spike counts per bin (re-binned from the per-step
        input grid for convenience downstream).
    """
    np.random.seed(seed)
    v = v_init.copy()
    npop = len(E)
    F = np.zeros(npop)         # filtered recurrent input
    F_X = np.zeros(npop)       # filtered feedforward (external) input

    binned_counts = np.zeros((npop, n_bins))
    voltage_bins = np.zeros((npop, n_bins))
    voltage_accum = np.zeros(npop)
    total_spikes = np.zeros(npop)
    ext_binned_counts = np.zeros((npop, n_bins))

    dt_tau   = dt_sim / tau
    dt_tau_g = dt_sim / tau_g
    decay_F  = 1.0 - dt_tau_g
    inv_tau_g = 1.0 / tau_g
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(npop, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(npop):
                voltage_accum[i] += v[i]

        # Draw cortical Poisson spikes: lambda_i = max(a * v_i, 0)
        for i in range(npop):
            v_i = v[i]
            lam = a * v_i
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]
                ext_binned_counts[i, current_bin] += ext_spike_grid[i, step]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(npop):
                    voltage_bins[i, current_bin] = voltage_accum[i] / bin_size_steps
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Update recurrent filter F_j: decay + spike kick
        for j in range(npop):
            F[j] = decay_F * F[j] + inv_tau_g * spikes[j]
        # Update feedforward filter F_X_i: decay + external spike kick
        for i in range(npop):
            F_X[i] = decay_F * F_X[i] + inv_tau_g * ext_spike_grid[i, step]

        # Euler voltage update: -v + E + recurrent + feedforward
        for i in range(npop):
            drive = E[i]
            for j in range(npop):
                drive += W[i, j] * F[j]
            drive += w_X * F_X[i]
            v[i] += dt_tau * (-v[i] + drive)

    return binned_counts, voltage_bins, total_spikes, ext_binned_counts

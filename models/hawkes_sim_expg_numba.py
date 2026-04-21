"""
models/hawkes_sim_expg_numba.py
===============================
Numba-JIT compiled Euler-step simulator for the linear Hawkes process
with tunable gain `a` and exponential synaptic filter
`g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)`.

Continuous-time dynamics (MSR-JD tree of the model in
``models/hawkes_linear_expg.py``):

    dF_j/dt     = -F_j / tau_g + n_j(t) / tau_g            # filtered input
    dv_i/dt     = (1/tau) * (-v_i + E_i + sum_j w_ij F_j)  # membrane
    lambda_i(t) = max(a * v_i, 0)                          # Poisson rate
    n_i(t) ~ Poisson(lambda_i(t) dt)                       # spikes

Here ``F_j`` is the exponentially-filtered spike train from population j
(one per presynaptic population).  Each spike delivers a kick of
``1/tau_g`` to ``F_j``, which then decays with timescale tau_g.  The
stationary mean is ``<F_j> = n*_j`` (since g integrates to 1), so the
filtered drive contributes ``sum_j w_ij n*_j`` to v_i on average,
matching the mean-field relations ``n*_i = a v*_i`` and
``v*_i = E_i + sum_j w_ij n*_j``.

This file MUST be a plain .py file (not executed through SageMath's
preparser) because Numba cannot handle Sage ring elements.

Usage (from a Sage notebook cell):

    from models.hawkes_sim_expg_numba import sim_hawkes_expg_numba

    binned_counts, voltage_bins, total_spikes = sim_hawkes_expg_numba(
        int(n_steps), float(dt_sim), float(tau), float(tau_g), float(a),
        E_arr, W_arr, v_init.copy(),
        int(bin_size_steps), int(n_bins), int(seed),
    )

Cast every call-site integer or float with int()/float() first -- Sage's
preparser turns ``0`` into ``Integer(0)``, which Numba cannot type.
"""

import numpy as np
import numba


@numba.njit
def sim_hawkes_expg_numba(n_steps, dt_sim, tau, tau_g, a, E, W, v_init,
                          bin_size_steps, n_bins, seed):
    """
    Euler-step simulation of the linear-gain, exp-filtered Hawkes process.

    Parameters
    ----------
    n_steps : int
        Total number of Euler timesteps.
    dt_sim : float
        Euler timestep.
    tau : float
        Membrane time constant.
    tau_g : float
        Synaptic exponential filter timescale.
    a : float
        Linear transfer-function gain: lambda_i = max(a * v_i, 0).
    E : np.ndarray (npop,)
        External drive per population.
    W : np.ndarray (npop, npop)
        Synaptic weight matrix, W[i, j] = weight from pop j to pop i.
    v_init : np.ndarray (npop,)
        Initial voltage (typically the MF fixed point).
    bin_size_steps : int
        Number of Euler steps per spike-count bin.
    n_bins : int
        Total number of bins to fill.
    seed : int
        Random seed.

    Returns
    -------
    binned_counts : np.ndarray (npop, n_bins)
        Spike counts per bin per population.
    voltage_bins : np.ndarray (npop, n_bins)
        Mean voltage per bin per population (averaged over the
        ``bin_size_steps`` Euler steps in each bin).
    total_spikes : np.ndarray (npop,)
        Total spike count per population over the full run.
    """
    np.random.seed(seed)
    v = v_init.copy()
    npop = len(E)
    # F_j is the filtered spike train from population j, one per pop.
    F = np.zeros(npop)

    binned_counts = np.zeros((npop, n_bins))
    voltage_bins = np.zeros((npop, n_bins))
    voltage_accum = np.zeros(npop)
    total_spikes = np.zeros(npop)

    dt_tau   = dt_sim / tau
    dt_tau_g = dt_sim / tau_g
    decay_F  = 1.0 - dt_tau_g        # multiplicative decay per Euler step
    inv_tau_g = 1.0 / tau_g
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(npop, dtype=np.int64)

    for step in range(n_steps):
        # Accumulate voltage for the bin average (only if still in-range)
        if current_bin < n_bins:
            for i in range(npop):
                voltage_accum[i] += v[i]

        # Draw Poisson spikes: lambda_i = max(a * v_i, 0)
        for i in range(npop):
            lam = a * v[i]
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        # Finalize bin if the step count is up
        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(npop):
                    voltage_bins[i, current_bin] = voltage_accum[i] / bin_size_steps
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Update filter F_j: decay + spike kick (each spike delivers 1/tau_g)
        for j in range(npop):
            F[j] = decay_F * F[j] + inv_tau_g * spikes[j]

        # Euler voltage update:
        #   dv_i/dt = (1/tau) * (-v_i + E_i + sum_j W_ij F_j)
        for i in range(npop):
            drive = E[i]
            for j in range(npop):
                drive += W[i, j] * F[j]
            v[i] += dt_tau * (-v[i] + drive)

    return binned_counts, voltage_bins, total_spikes

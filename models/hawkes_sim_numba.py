"""
models/hawkes_sim_numba.py
==========================
Numba-JIT compiled Euler-step simulator for the linear Hawkes process.

This file MUST be a plain .py file (not executed through SageMath's
preparser) because SageMath's preparser converts integer literals like
``0`` to ``Integer(0)`` (a Sage ring element), which Numba's type
inference cannot handle. Importing this module from a Sage notebook
works fine — only the notebook cell source is preparsed, not .py imports.

Usage (from a SageMath notebook cell)::

    from models.hawkes_sim_numba import sim_hawkes_numba

    binned_counts, total_spikes = sim_hawkes_numba(
        int(n_steps), float(dt_sim), float(tau),
        E_arr, W_arr, v_init.copy(),
        int(bin_size_steps), int(n_bins), int(seed),
    )

All arguments at the call site must be plain Python int / float /
numpy arrays — cast any Sage Integer or RealNumber with int() / float()
before calling.
"""

import numpy as np
import numba


@numba.njit
def sim_hawkes_numba(n_steps, dt_sim, tau, E, W, v_init,
                     bin_size_steps, n_bins, seed):
    """
    Euler-step Hawkes simulation with Poisson spike draws.

    ~100× faster than a pure-Python loop (~15M steps/sec on Apple M1).

    Parameters
    ----------
    n_steps : int
        Total number of Euler timesteps.
    dt_sim : float
        Euler timestep.
    tau : float
        Membrane time constant.
    E : np.ndarray (npop,)
        External drive per population.
    W : np.ndarray (npop, npop)
        Synaptic weight matrix W[i, j] = weight from pop j to pop i.
    v_init : np.ndarray (npop,)
        Initial voltage (typically the MF fixed point).
    bin_size_steps : int
        Number of Euler steps per spike-count bin.
    n_bins : int
        Total number of bins to fill.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    binned_counts : np.ndarray (npop, n_bins)
        Spike counts per bin per population.
    total_spikes : np.ndarray (npop,)
        Total spike count per population over the whole simulation.
    """
    np.random.seed(seed)
    v = v_init.copy()
    npop = len(E)
    binned_counts = np.zeros((npop, n_bins))
    total_spikes = np.zeros(npop)
    dt_tau = dt_sim / tau
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(npop, dtype=np.int64)

    for step in range(n_steps):
        # Draw Poisson spikes: lambda_i = max(v_i, 0) * dt
        for i in range(npop):
            lam = max(v[i], 0.0) * dt_sim
            spikes[i] = np.random.poisson(lam)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            current_bin += 1
            steps_in_bin = 0

        # Euler voltage update:
        #   v_i += (dt/tau) * (-v_i + E_i) + (1/tau) * sum_j W_ij * spikes_j
        for i in range(npop):
            v[i] += dt_tau * (-v[i] + E[i])
            for j in range(npop):
                v[i] += W[i, j] * spikes[j] / tau

    return binned_counts, total_spikes

"""
models/hawkes_sim_multipop_numba.py
====================================
Numba-JIT Euler-step simulator for the **linear multipopulation Hawkes
process** with per-pair exponential synaptic filters.  Matches the
theory file ``theories/multipopulation_test.theory.py``.

Continuous-time dynamics (MSR-JD tree of the heterogeneous-population
model, two populations E and I of arbitrary size):

    dF_{ij}/dt   = (1/tau_g[i,j]) * (n_j(t) - F_{ij})       # per-pair filter
    dv_i/dt      = (1/tau_v[i]) * (-v_i + E_drive_i
                                   + sum_j W[i,j] * F_{ij})  # membrane
    lambda_i(t)  = max(a_i * v_i, 0)                          # linear rate
    n_i(t)       ~ Poisson(lambda_i(t) dt)                    # spikes

Conventions
-----------
Neurons are stacked into a flat 1-D index ``i ∈ {0, ..., N-1}`` where
``N = N_E + N_I``.  The first ``N_E`` indices are excitatory, the next
``N_I`` are inhibitory.  Inputs are:

- ``tau_v[i]``   — membrane time constant of neuron ``i`` (vector, len N).
- ``a_gain[i]``  — linear-rate gain ``λ_i = a_i v_i`` (vector, len N).
- ``E_drive[i]`` — external constant drive (vector, len N).
- ``W[i,j]``     — effective signed coupling from neuron ``j`` (pre) to
                   neuron ``i`` (post).  E presynaptic carries a ``+``,
                   I presynaptic carries a ``-`` (so I→E and I→I
                   already include the inhibitory sign).  Shape (N, N).
- ``tau_g[i,j]`` — per-pair filter timescale.  Shape (N, N).

The pair-specific filter ``F_{ij}`` makes this faithful to the
``taugEE[i,j]``, ``taugEI[i,j]``, ``taugIE[i,j]``, ``taugII[i,j]``
matrices in the heterogeneous theory — there's no single "synaptic
filter" shared across all connections.

Usage (from a Sage notebook cell):

    from models.hawkes_sim_multipop_numba import sim_hawkes_multipop_numba

    binned_counts, voltage_bins, total_spikes = sim_hawkes_multipop_numba(
        int(n_steps), float(dt_sim),
        tau_v_arr, a_gain_arr, E_drive_arr,
        W_arr, tau_g_arr,
        v_init.copy(),
        int(bin_size_steps), int(n_bins), int(seed),
    )

Cast every call-site integer or float with int()/float() first — Sage's
preparser turns ``0`` into ``Integer(0)``, which Numba cannot type.
"""

import numpy as np
import numba


@numba.njit
def sim_hawkes_multipop_numba(n_steps, dt_sim,
                              tau_v, a_gain, E_drive,
                              W, tau_g,
                              v_init,
                              bin_size_steps, n_bins, seed):
    """
    Euler-step simulation of the heterogeneous-population linear-rate
    Hawkes process with per-pair exponential filters.

    Parameters
    ----------
    n_steps : int
        Total number of Euler timesteps.
    dt_sim : float
        Euler timestep (in time units used everywhere — typically ms).
    tau_v : np.ndarray (N,)
        Membrane time constant per neuron.
    a_gain : np.ndarray (N,)
        Linear-rate gain per neuron: ``lambda_i = max(a_i v_i, 0)``.
    E_drive : np.ndarray (N,)
        External constant drive per neuron.
    W : np.ndarray (N, N)
        Effective signed coupling matrix.  W[i, j] is the post-i,
        pre-j synaptic strength, already sign-encoded (E pre: +,
        I pre: −).
    tau_g : np.ndarray (N, N)
        Per-pair synaptic filter timescale.  ``tau_g[i, j]`` is the
        timescale of the exponential filter on spikes from neuron
        ``j`` driving neuron ``i``.
    v_init : np.ndarray (N,)
        Initial voltage (typically the MF fixed point).
    bin_size_steps : int
        Number of Euler steps per spike-count bin.
    n_bins : int
        Total number of bins to fill.
    seed : int
        Random seed.

    Returns
    -------
    binned_counts : np.ndarray (N, n_bins)
        Spike counts per bin per neuron.  Treat each row as its own
        "population" when feeding into ``compute_kpoint_slice``.
    voltage_bins : np.ndarray (N, n_bins)
        Mean voltage per bin per neuron (averaged over the
        ``bin_size_steps`` Euler steps in each bin).
    total_spikes : np.ndarray (N,)
        Total spike count per neuron over the full run.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()
    # F[i, j]: per-pair exponential filter of spikes from j into i.
    F = np.zeros((N, N))

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Draw Poisson spikes: λ_i = max(a_i * v_i, 0)
        for i in range(N):
            lam = a_gain[i] * v[i]
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Update each F[i, j] filter independently.  Each pair has its
        # own decay timescale tau_g[i, j].
        for i in range(N):
            for j in range(N):
                tg = tau_g[i, j]
                # decay factor exp(-dt/tg) ≈ 1 - dt/tg for small dt/tg
                decay = 1.0 - dt_sim / tg
                F[i, j] = decay * F[i, j] + (1.0 / tg) * spikes[j]

        # Euler voltage update.  W[i, j] already carries the sign
        # (positive for excitatory presyn, negative for inhibitory).
        for i in range(N):
            drive = E_drive[i]
            for j in range(N):
                drive += W[i, j] * F[i, j]
            v[i] += dt_sim / tau_v[i] * (-v[i] + drive)

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_quad_reset_numba(n_steps, dt_sim,
                                         tau_v, a_gain, E_drive,
                                         W, tau_g,
                                         v_init,
                                         bin_size_steps, n_bins, seed):
    """
    Euler-step simulator for the **quadratic-rate hard-spike-reset**
    variant of the heterogeneous-population Hawkes process.  Matches
    ``theories/multipopulation_spike_reset_test.theory.py``:

      dF_{ij}/dt   = (1/tau_g[i,j]) * (n_j(t) - F_{ij})
      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] F_{ij}
                         - tau_v_i · v_i · n_i(t)            # hard reset
      lambda_i(t)  = max(a_i · v_i^2, 0)                    # QUADRATIC rate
      n_i(t)       ~ Poisson(lambda_i(t) dt)

    The two structural differences from
    ``sim_hawkes_multipop_numba`` are:

      (1) ``λ_i = a_i · v_i²`` (quadratic, was linear) — gives ``φ''(v*) ≠ 0``
          so 1-loop diagrams from the cubic vertex become physically
          observable.

      (2) Each own-population spike **fully resets** v_i to 0.  In the
          continuum limit this is the ``− v · n`` term in the Langevin
          EOM (with no ``1/τ`` factor — the τ in the action
          ``+τ_v · v · n`` cancels the ``1/τ_v`` from the LHS
          ``(τ_v Dt + 1)·v`` when going to ``dv/dt`` form).
          Discretely, when a spike happens in a single Euler step we
          set ``v_i ← 0`` (a multi-spike step is bounded at 0, not
          driven negative).

    All other plumbing (per-pair exp-filter ``F[i,j]``, signed coupling
    ``W[i,j]``, binning) is identical to the linear-rate sim — so
    ``build_sim_arrays`` and ``flat_index_of`` work without changes
    for either theory.

    Parameters are the same as ``sim_hawkes_multipop_numba``.  Returns
    are the same too: ``(binned_counts, voltage_bins, total_spikes)``.

    Sign convention.  Both phi-quadratic and spike-reset are *strictly
    self-coupling* effects — each population only resets its own
    voltage (``-v_i · n_i``, not ``-v_i · n_j`` with j≠i).  If your
    theory file declares cross-population reset terms, this simulator
    will under-count them and you'll see a sim/theory mismatch.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()
    F = np.zeros((N, N))

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Quadratic rate: λ_i = max(a_i v_i², 0).  The ``max`` is for
        # safety only — with a_i > 0 the v_i² rate is automatically ≥ 0.
        for i in range(N):
            v_i = v[i]
            lam = a_gain[i] * v_i * v_i
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Per-pair F[i,j] filter update (same as the linear sim).
        for i in range(N):
            for j in range(N):
                tg = tau_g[i, j]
                decay = 1.0 - dt_sim / tg
                F[i, j] = decay * F[i, j] + (1.0 / tg) * spikes[j]

        # Euler voltage update + hard spike reset.  Two-step:
        #   1. Drift over dt: v += (dt/τ_v) · (-v + E + Σ W F)
        #   2. Hard reset: if any spike in this step, v → 0 (clamped
        #      so multi-spike steps don't drive v negative).
        # In the continuum limit step (2) corresponds to the ``-v · n(t)``
        # term in dv/dt (the τ_v cancels between the action's
        # ``+τ_v · v · n`` and the LHS ``(τ_v Dt + 1)·v``).
        for i in range(N):
            drive = E_drive[i]
            for j in range(N):
                drive += W[i, j] * F[i, j]
            v[i] += dt_sim / tau_v[i] * (-v[i] + drive)
            if spikes[i] > 0:
                v[i] = 0.0

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_quad_numba(n_steps, dt_sim,
                                   tau_v, a_gain, E_drive,
                                   W, tau_g,
                                   v_init,
                                   bin_size_steps, n_bins, seed):
    """
    Euler-step simulator for **quadratic-rate, no-reset** Hawkes.

      dF_{ij}/dt   = (1/tau_g[i,j]) * (n_j(t) - F_{ij})
      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] F_{ij}
      lambda_i(t)  = max(a_i · v_i^2, 0)
      n_i(t)       ~ Poisson(lambda_i(t) dt)

    Matches ``theories/single_population_quad_exp_test.theory.py`` and
    any other quad-φ Hawkes theory without spike reset.  Identical to
    ``sim_hawkes_multipop_quad_reset_numba`` except the ``v → 0``
    hard-reset step is removed — voltage drifts continuously through
    spike events.

    Same call signature and return shape as the other multipop sim
    variants, so ``build_sim_arrays`` / ``flat_index_of`` work unchanged.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()
    F = np.zeros((N, N))

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        for i in range(N):
            v_i = v[i]
            lam = a_gain[i] * v_i * v_i
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        for i in range(N):
            for j in range(N):
                tg = tau_g[i, j]
                decay = 1.0 - dt_sim / tg
                F[i, j] = decay * F[i, j] + (1.0 / tg) * spikes[j]

        for i in range(N):
            drive = E_drive[i]
            for j in range(N):
                drive += W[i, j] * F[i, j]
            v[i] += dt_sim / tau_v[i] * (-v[i] + drive)

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_cubic_alpha_numba(n_steps, dt_sim,
                                          tau_v, a_gain, E_drive,
                                          W, tau_g,
                                          v_init,
                                          bin_size_steps, n_bins, seed):
    """
    Euler-step simulator for **cubic-rate, no-reset, alpha-kernel** Hawkes.

      dF_aux_{ij}/dt = (1/tau_g[i,j]) * (n_j(t) - F_aux_{ij})    # 1st exp stage
      dF_{ij}/dt     = (1/tau_g[i,j]) * (F_aux_{ij} - F_{ij})    # 2nd exp stage
      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] F_{ij}
      lambda_i(t)    = max(a_i · v_i^3, 0)                       # CUBIC rate
      n_i(t)         ~ Poisson(lambda_i(t) dt)

    Matches ``theories/single_population_cubic_alpha_test.theory.py``:
    no spike reset, cubic φ, alpha synaptic kernel
    α(t) = (t / τ_g²) · exp(-t / τ_g) · H(t).

    Math: the alpha kernel is the convolution of two identical exponential
    kernels with the same time constant.  The cascade of two first-order
    low-pass filters above realises this exactly — F_{ij}(t) = (α * n_j)(t)
    where α is the desired alpha shape.  ∫ α(t) dt = 1 so the MF gain
    is preserved (each filter has unit DC gain).

    Same call signature and return shape as the other multipop sim
    variants, so ``build_sim_arrays`` / ``flat_index_of`` work unchanged.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()
    F_aux = np.zeros((N, N))     # first-stage filter state
    F     = np.zeros((N, N))     # second-stage filter state (== α * n_j)

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        for i in range(N):
            v_i = v[i]
            lam = a_gain[i] * v_i * v_i * v_i      # cubic rate
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Two-stage cascade → alpha kernel.  Order:
        #   F (the alpha-filtered rate) is updated from OLD F_aux as a
        #     continuous-rate exponential filter:
        #       F ← F·(1 - dt/τ_g) + (dt/τ_g)·F_aux
        #   F_aux (the single-exp-filtered spike rate) is updated from
        #     this step's spike COUNTS (delta-distributed input — no dt
        #     factor, the (1/τ_g) coefficient absorbs the delta weight):
        #       F_aux ← F_aux·(1 - dt/τ_g) + (1/τ_g)·spikes
        # Matches the standard Hawkes-cascade discretization used in the
        # exponential-kernel variants (``sim_hawkes_multipop_quad_numba``
        # etc.).  Equilibrium: E[F_aux] → ν (mean rate), E[F] → ν, DC
        # gain 1 as required for ∫α(t)dt = 1.
        for i in range(N):
            for j in range(N):
                tg = tau_g[i, j]
                decay = 1.0 - dt_sim / tg
                F[i, j]     = decay * F[i, j]     + (dt_sim / tg) * F_aux[i, j]
                F_aux[i, j] = decay * F_aux[i, j] + (1.0    / tg) * spikes[j]

        for i in range(N):
            drive = E_drive[i]
            for j in range(N):
                drive += W[i, j] * F[i, j]
            v[i] += dt_sim / tau_v[i] * (-v[i] + drive)

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_linear_reset_numba(n_steps, dt_sim,
                                           tau_v, a_gain, E_drive,
                                           W, tau_g,
                                           v_init,
                                           bin_size_steps, n_bins, seed):
    """
    Euler-step simulator for the **linear-rate hard-spike-reset** variant
    of the heterogeneous-population Hawkes process.  Matches
    ``theories/single_population_spike_reset_test.theory.py`` (and any
    other linear-φ + hard-reset theory).

      dF_{ij}/dt   = (1/tau_g[i,j]) * (n_j(t) - F_{ij})
      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] F_{ij}
                         - tau_v_i · v_i · n_i(t)            # hard reset
      lambda_i(t)  = max(a_i · v_i, 0)                       # LINEAR rate
      n_i(t)       ~ Poisson(lambda_i(t) dt)

    Differences from ``sim_hawkes_multipop_quad_reset_numba``:
      * Rate is LINEAR (``λ = a·v``) instead of quadratic.  This kills
        the cubic vertex from φ''(v*) (since φ'' = 0 for linear φ).
      * The (1, 2)-bigrade vertex from the spike-reset term
        ``vt · τ · dv · dn`` is STILL present, so 1-loop diagrams from
        the reset alone are physically observable.  This makes the
        linear + reset theory a clean diagnostic for reset-induced
        loop corrections in isolation.

    All other plumbing (per-pair exp-filter, signed coupling, binning,
    hard reset to 0) is identical to the quad-rate variant.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()
    F = np.zeros((N, N))

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Linear rate: λ_i = max(a_i · v_i, 0)
        for i in range(N):
            lam = a_gain[i] * v[i]
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Per-pair F[i,j] filter update.
        for i in range(N):
            for j in range(N):
                tg = tau_g[i, j]
                decay = 1.0 - dt_sim / tg
                F[i, j] = decay * F[i, j] + (1.0 / tg) * spikes[j]

        # Euler voltage drift + hard reset.
        for i in range(N):
            drive = E_drive[i]
            for j in range(N):
                drive += W[i, j] * F[i, j]
            v[i] += dt_sim / tau_v[i] * (-v[i] + drive)
            if spikes[i] > 0:
                v[i] = 0.0

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_linear_conductance_numba(
        n_steps, dt_sim,
        tau_v, a_gain, E_drive,
        W, tau_g, E_rev,
        v_init,
        bin_size_steps, n_bins, seed):
    """
    Euler-step simulator for the **linear-rate CONDUCTANCE-synapse**
    variant of the heterogeneous-population Hawkes process.  Matches
    ``theories/single_population_conductance_synapse_test.theory.py``:

      dF_{ij}/dt   = (1/tau_g) * (n_j(t) - F_{ij})         # exp synapse filter
      tau_v_i · dv_i/dt = -v_i + E_drive_i
                         + sum_j W[i,j] * (E_rev − v_i) * F_{ij}   # conductance
      lambda_i(t)  = max(a_i · v_i, 0)                     # linear rate
      n_i(t)       ~ Poisson(lambda_i(t) · dt)             # spikes

    Conductance vs current synapse
    ------------------------------
    The factor ``(E_rev − v_i)`` is the synaptic driving force.  Synaptic
    input vanishes as ``v_i → E_rev`` (e.g., as the membrane approaches
    the synaptic reversal potential), bounding the response — a
    multiplicative ``v·F`` coupling absent from current-based variants.
    This adds a (1, 2)-bigrade interaction vertex (one xt + one dv +
    one Conv(g, n)) to the action; the per-pair exp filter contributes
    additional rational poles.

    No spike reset.  ``E_rev`` is scalar (single reversal potential
    shared across all post-synaptic targets).  Output buffers match
    the rest of the ``hawkes_sim_multipop`` family.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()
    F = np.zeros((N, N))

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Linear rate: λ_i = max(a_i · v_i, 0)
        for i in range(N):
            lam = a_gain[i] * v[i]
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Per-pair F[i,j] exp-filter update.
        for i in range(N):
            for j in range(N):
                tg = tau_g[i, j]
                decay = 1.0 - dt_sim / tg
                F[i, j] = decay * F[i, j] + (1.0 / tg) * spikes[j]

        # Euler voltage drift with CONDUCTANCE synaptic input.
        for i in range(N):
            syn_drive = 0.0
            for j in range(N):
                syn_drive += W[i, j] * (E_rev - v[i]) * F[i, j]
            v[i] += dt_sim / tau_v[i] * (-v[i] + E_drive[i] + syn_drive)

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_linear_softreset_numba(
        n_steps, dt_sim,
        tau_v, a_gain, E_drive,
        W, tau_g,
        v_init,
        bin_size_steps, n_bins, seed,
        reset_coeff,
):
    """
    Euler-step simulator for **linear-rate + SOFT spike-reset +
    exponential-synapse** Hawkes.  Matches
    ``theories/single_population_spike_reset_test.theory.py`` after its
    2026-05-18 update (reset term changed from full ``tau[i]*v[i]*n[i]``
    to the parameter-free ``reset_coeff*v[i]*n[i]``).

    Langevin form::

      dF_{ij}/dt        = (1/tau_g[i,j]) · (n_j(t) - F_{ij})
      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] · F_{ij}
                          - reset_coeff · v_i · n_i(t)        # SOFT reset
      lambda_i(t)        = max(a_i · v_i, 0)                  # LINEAR rate
      n_i(t)             ~ Poisson(lambda_i(t) dt)

    Per-spike voltage update (integrating across the δ-impulse)::

      Δv_i = -(reset_coeff/tau_v_i) · v_i(t-)
      ⇒ v_i(t+) = v_i(t-) · (1 - reset_coeff/tau_v_i)

    Multiple spikes in one Euler step compound: ``v_i *= rf^spikes_i``.

    Differences from ``sim_hawkes_multipop_linear_reset_numba`` (hard
    reset):
      * SOFT (multiplicative, fractional) reset instead of ``v ← 0``.
        At ``tau_v_i = 10`` and ``reset_coeff = 1.0`` each spike cuts
        ``v_i`` by 10% (factor 0.9).
      * Adds the ``reset_coeff`` argument (positional, after seed).

    All other plumbing (per-pair ``F[i,j]`` exponential filter, signed
    coupling, binning) is identical.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()
    F = np.zeros((N, N))

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    # Per-pop multiplicative reset factor (clamped non-negative).
    reset_factor = np.empty(N)
    for i in range(N):
        f = 1.0 - reset_coeff / tau_v[i]
        if f < 0.0:
            f = 0.0
        reset_factor[i] = f

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Linear rate sample.
        for i in range(N):
            lam = a_gain[i] * v[i]
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Per-pair F[i,j] exponential-filter update.
        for i in range(N):
            for j in range(N):
                tg = tau_g[i, j]
                decay = 1.0 - dt_sim / tg
                F[i, j] = decay * F[i, j] + (1.0 / tg) * spikes[j]

        # Drift + soft reset.
        for i in range(N):
            drive = E_drive[i]
            for j in range(N):
                drive += W[i, j] * F[i, j]
            v[i] += dt_sim / tau_v[i] * (-v[i] + drive)
            if spikes[i] > 0:
                rf = reset_factor[i]
                for _ in range(spikes[i]):
                    v[i] *= rf

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_quad_reset_delta_numba(n_steps, dt_sim,
                                                tau_v, a_gain, E_drive,
                                                W, tau_g,
                                                v_init,
                                                bin_size_steps, n_bins, seed):
    """
    Euler-step simulator for the **quadratic-rate + hard-spike-reset +
    delta synapse** Hawkes process.  Matches
    ``theories/single_population_quad_spike_reset_test.theory.py``:

      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] · n_j(t)
                         - tau_v_i · v_i · n_i(t)                # hard reset
      lambda_i(t)        = max(a_i · v_i^2, 0)                   # quadratic
      n_i(t)             ~ Poisson(lambda_i(t) dt)

    Differences from ``sim_hawkes_multipop_quad_reset_numba``:

      * Synaptic input is **instantaneous** (delta kernel ``g(t) = δ(t)``)
        instead of running through an exponential filter ``F[i,j]``.
        In each Euler step ``v_i`` gets a delta kick
        ``+ W[i,j] · spikes[j] / tau_v_i`` for every spike in population j
        (the ``1/tau_v_i`` is the coefficient that survives when the
        action's ``vt · (sum_j w · n_j)`` is read off into
        ``dv/dt = sum_j (w/tau_v) · n_j``).

      * ``tau_g`` is accepted in the signature only to keep the call
        shape identical to the exponential-synapse simulators
        (``build_sim_arrays`` always emits a ``tau_g`` array even when
        unused).  This simulator ignores it.

    Other behaviour matches the exponential-synapse quad-reset
    simulator: quadratic rate, hard v→0 reset on each own-population
    spike, same binning convention.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Quadratic rate sample.
        for i in range(N):
            v_i = v[i]
            lam = a_gain[i] * v_i * v_i
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Voltage update: drift over dt, then instantaneous delta-synapse
        # kicks (one per pre-population spike), then hard reset for
        # own-population spikes.  Order matters at finite dt — drift
        # first (uses ``v_i`` at step start), then accumulate
        # synaptic kicks from this step's spikes, then reset.
        for i in range(N):
            v[i] += dt_sim / tau_v[i] * (-v[i] + E_drive[i])
        for i in range(N):
            kick = 0.0
            for j in range(N):
                kick += W[i, j] * spikes[j]
            v[i] += kick / tau_v[i]
            if spikes[i] > 0:
                v[i] = 0.0

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_quad_softreset_delta_numba(
        n_steps, dt_sim,
        tau_v, a_gain, E_drive,
        W, tau_g,
        v_init,
        bin_size_steps, n_bins, seed,
        reset_coeff,
):
    """
    Euler-step simulator for **quadratic-rate + SOFT spike-reset +
    delta-synapse** Hawkes.  Matches
    ``theories/single_population_quad_spike_reset_test.theory.py``
    after its 2026-05-18 update (reset term changed from full
    ``tau[i]*v[i]*n[i]`` to the parameter-free ``reset_coeff*v[i]*n[i]``
    with ``reset_coeff = 0.5``).

    Langevin form::

      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] · n_j(t)
                          - reset_coeff · v_i · n_i(t)   # SOFT reset

    Per-spike voltage update by integrating across the impulse
    ``n_i(t) = δ(t - t_spike)``::

      Δv_i = -(reset_coeff/tau_v_i) · v_i(t-)
      ⇒ v_i(t+) = v_i(t-) · (1 - reset_coeff/tau_v_i)

    For multiple spikes in a single Euler step the multiplicative
    factor compounds: ``v_i *= (1 - reset_coeff/tau_v_i)^spikes_i``.

    Differences from ``sim_hawkes_multipop_quad_reset_delta_numba``:
      * SOFT (multiplicative, fractional) reset instead of hard reset
        to 0.  At ``tau_v_i = 10`` and ``reset_coeff = 0.5`` each spike
        cuts ``v_i`` by 5% (factor 0.95).
      * Adds the ``reset_coeff`` argument (positional, after seed).

    Other plumbing (instantaneous synapses, ``tau_g`` ignored,
    binning) is identical to the hard-reset variant.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    # Per-pop multiplicative reset factor (clamped non-negative to
    # avoid the unphysical regime where reset_coeff > tau_v drives v
    # past zero in one spike — that would require a sub-step Euler).
    reset_factor = np.empty(N)
    for i in range(N):
        f = 1.0 - reset_coeff / tau_v[i]
        if f < 0.0:
            f = 0.0
        reset_factor[i] = f

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Quadratic rate sample.
        for i in range(N):
            v_i = v[i]
            lam = a_gain[i] * v_i * v_i
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Drift, delta-synapse kicks, soft reset.  Order matches the
        # hard-reset variant: drift first (uses pre-step v), then this
        # step's spike-driven updates.
        for i in range(N):
            v[i] += dt_sim / tau_v[i] * (-v[i] + E_drive[i])
        for i in range(N):
            kick = 0.0
            for j in range(N):
                kick += W[i, j] * spikes[j]
            v[i] += kick / tau_v[i]
            if spikes[i] > 0:
                # Multiplicative soft reset compounding per spike.
                rf = reset_factor[i]
                for _ in range(spikes[i]):
                    v[i] *= rf

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_linear_softreset_delta_numba(
        n_steps, dt_sim,
        tau_v, a_gain, E_drive,
        W, tau_g,
        v_init,
        bin_size_steps, n_bins, seed,
        reset_coeff,
):
    """
    Euler-step simulator for **linear-rate + SOFT spike-reset +
    delta-synapse** Hawkes.  Matches
    ``theories/single_population_spike_reset_test.theory.py`` after its
    2026-05-18 update (kernel changed from per-pair exponential
    ``(1/τ_g)·exp(-t/τ_g)·θ(t)`` to scalar ``dirac_delta(t)``, and
    reset term changed from full ``tau[i]*v[i]*n[i]`` to the
    parameter-free ``reset_coeff*v[i]*n[i]``).

    Langevin form::

      tau_v_i · dv_i/dt = -v_i + E_i + sum_j W[i,j] · n_j(t)
                          - reset_coeff · v_i · n_i(t)        # SOFT reset
      lambda_i(t)        = max(a_i · v_i, 0)                  # LINEAR rate
      n_i(t)             ~ Poisson(lambda_i(t) dt)

    Synapses are instantaneous (delta kernel ``g(t) = δ(t)``); each
    spike in pop j kicks ``v_i`` by ``W[i,j] / τ_v_i``.  Per-spike
    soft reset multiplies ``v_i`` by ``(1 - reset_coeff/τ_v_i)``
    (compounding for multi-spike Euler steps).

    Differences from ``sim_hawkes_multipop_linear_softreset_numba``
    (the exp-synapse soft-reset variant):
      * Synapses are **instantaneous** (delta) — no ``F[i,j]`` filter
        state, no per-pair ``tau_g`` time constant.
      * ``tau_g`` is accepted in the signature only to keep
        ``build_sim_arrays`` happy; this simulator ignores it.

    Other plumbing (linear rate sampling, delta kicks, soft reset,
    binning) is identical to the quad-rate delta-synapse soft-reset
    variant.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    # Per-pop multiplicative reset factor (clamped non-negative).
    reset_factor = np.empty(N)
    for i in range(N):
        f = 1.0 - reset_coeff / tau_v[i]
        if f < 0.0:
            f = 0.0
        reset_factor[i] = f

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Linear rate sample.
        for i in range(N):
            lam = a_gain[i] * v[i]
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Drift, delta-synapse kicks, soft reset.
        for i in range(N):
            v[i] += dt_sim / tau_v[i] * (-v[i] + E_drive[i])
        for i in range(N):
            kick = 0.0
            for j in range(N):
                kick += W[i, j] * spikes[j]
            v[i] += kick / tau_v[i]
            if spikes[i] > 0:
                rf = reset_factor[i]
                for _ in range(spikes[i]):
                    v[i] *= rf

    return binned_counts, voltage_bins, total_spikes


@numba.njit
def sim_hawkes_multipop_linear_conductance_delta_numba(
        n_steps, dt_sim,
        tau_v, a_gain, E_drive,
        W, tau_g, E_rev,
        v_init,
        bin_size_steps, n_bins, seed):
    """
    Euler-step simulator for **linear-rate + CONDUCTANCE-synapse +
    delta-kernel** Hawkes.  Matches
    ``theories/single_population_conductance_synapse_test.theory.py``
    after switching its synapse kernel from
    ``(1/τ_g)·exp(-t/τ_g)·θ(t)`` to ``dirac_delta(t)``.

    Langevin form::

      tau_v_i · dv_i/dt = -v_i + E_drive_i
                          + sum_j W[i,j] · (E_rev − v_i) · n_j(t)
      lambda_i(t)        = max(a_i · v_i, 0)               # LINEAR rate
      n_i(t)             ~ Poisson(lambda_i(t) dt)

    Differences from ``sim_hawkes_multipop_linear_conductance_numba``
    (the exp-synapse variant):
      * Synapses are **instantaneous** (delta) — no ``F[i,j]`` filter
        state, no per-pair ``tau_g`` time constant.
      * Each spike at neuron j kicks neuron i by
        ``(W[i,j] / τ_v_i) · (E_rev − v_i^-)`` using the pre-step v
        (Itô interpretation; matches the action the framework reads).
      * ``tau_g`` is accepted in the signature only to keep
        ``build_sim_arrays`` and the existing notebook call sites
        happy; this simulator ignores it.

    Voltage drift uses the same pre-step ``v[i]`` in the
    ``(E_rev − v[i])`` driving-force factor as the exp-synapse variant,
    so the Itô discretization is consistent across kernels.
    """
    np.random.seed(seed)
    N = len(tau_v)
    v = v_init.copy()

    binned_counts = np.zeros((N, n_bins))
    voltage_bins = np.zeros((N, n_bins))
    voltage_accum = np.zeros(N)
    total_spikes = np.zeros(N)
    current_bin = 0
    steps_in_bin = 0
    spikes = np.zeros(N, dtype=np.int64)

    for step in range(n_steps):
        if current_bin < n_bins:
            for i in range(N):
                voltage_accum[i] += v[i]

        # Linear rate: λ_i = max(a_i · v_i, 0).
        for i in range(N):
            lam = a_gain[i] * v[i]
            if lam < 0.0:
                lam = 0.0
            spikes[i] = np.random.poisson(lam * dt_sim)
            total_spikes[i] += spikes[i]
            if current_bin < n_bins:
                binned_counts[i, current_bin] += spikes[i]

        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    voltage_bins[i, current_bin] = (voltage_accum[i] /
                                                    bin_size_steps)
                    voltage_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # Drift + delta-synapse conductance kicks.  Pre-step v[i] in
        # the (E_rev - v[i]) factor matches the framework's Itô
        # convention; consistent with the exp-synapse variant.
        for i in range(N):
            v_pre = v[i]
            syn_kick = 0.0
            for j in range(N):
                syn_kick += W[i, j] * (E_rev - v_pre) * spikes[j]
            v[i] += dt_sim / tau_v[i] * (-v_pre + E_drive[i]) \
                    + syn_kick / tau_v[i]

    return binned_counts, voltage_bins, total_spikes


def build_sim_arrays(model, fundamental, mf_values):
    """
    Helper: assemble the flat per-neuron / per-pair simulation arrays
    from a heterogeneous-pop ``model`` dict, a ``fundamental`` parameter
    dict, and the pipeline's mean-field saddle values.

    Parameters
    ----------
    model : dict
        The build()-output dict from a heterogeneous-pop theory file.
        Must declare ``model['populations']`` (a list of
        ``{'name': str, 'size': int}`` records).
    fundamental : dict
        Numerical parameter values keyed by name (matches the keys
        used in ``compute_cumulants``).  Required keys depend on the
        theory; ``tauE``, ``tauI``, ``aE``, ``aI``, ``EmE``, ``EmI``,
        ``wEE``, ``wEI``, ``wIE``, ``wII``, ``taugEE``, ``taugEI``,
        ``taugIE``, ``taugII`` for the multipop_test file.
    mf_values : dict
        ``{'nstar_name': [v1, v2, ...]}`` for each MF saddle, sized
        per-population.  Typically ``result['mf_values']`` from
        ``compute_cumulants``.

    Returns
    -------
    arrays : dict with keys
        ``N``         : int — total stacked neuron count
        ``tau_v``     : np.ndarray (N,)
        ``a_gain``    : np.ndarray (N,)
        ``E_drive``   : np.ndarray (N,)
        ``W``         : np.ndarray (N, N)
        ``tau_g``     : np.ndarray (N, N)
        ``v_init``    : np.ndarray (N,) — at the MF fixed point
        ``pop_offsets`` : dict {pop_name: (start_idx, size)} — flat-index
                          range owned by each population, for
                          downstream selection.

    Convention
    ----------
    The two populations are stacked in the order they appear in
    ``model['populations']``.  E goes before I in the multipop_test
    file.  Within each population, the order is the natural 0-based
    index over its size.
    """
    pops = model['populations']
    pop_offsets = {}
    offset = 0
    for p in pops:
        pop_offsets[p['name']] = (offset, int(p['size']))
        offset += int(p['size'])
    N = offset

    tau_v   = np.zeros(N)
    a_gain  = np.zeros(N)
    E_drive = np.zeros(N)
    W       = np.zeros((N, N))
    tau_g   = np.ones((N, N))   # default 1.0 to avoid div-by-zero for
                                # pairs with no coupling (W=0)
    v_init  = np.zeros(N)

    def _resolve(keys):
        """Return ``fundamental[key]`` for the first ``key`` in ``keys``
        present in ``fundamental``, else raise KeyError listing the
        tried names.

        We try the suffixed multipop convention first
        (``tauE`` / ``aE`` / ``EmE`` / ``wEE`` …) and fall back to the
        bare single-pop convention (``tau`` / ``a`` / ``Em`` / ``w``).
        """
        for key in keys:
            if key in fundamental:
                return fundamental[key]
        raise KeyError(
            f'no parameter named any of {keys} in fundamental dict; '
            f'available: {sorted(fundamental.keys())}'
        )

    # ── Per-neuron arrays (membrane τ, gain, drive, MF v) ─────────
    # Two naming conventions handled:
    #   * Multipop (`multipopulation_test`):  ``tau<P>``, ``a<P>``, ``Em<P>``.
    #   * Single-pop (`single_population_spike_reset_test`):  bare names
    #     ``tau``, ``a``, ``Em`` (no per-pop suffix when there's only
    #     one population to disambiguate).
    # ``mf_values`` always uses the auto-saddle convention
    # (``v<natural>star`` from the field's ``natural_name``).  Single-pop
    # theories with field name ``v`` give ``vstar``; multipop with
    # ``vE`` gives ``vEstar``.
    for p in pops:
        pname = p['name']
        start, size = pop_offsets[pname]
        tau_arr   = _resolve([f'tau{pname}', 'tau'])
        a_arr     = _resolve([f'a{pname}', 'a'])
        e_arr     = _resolve([f'Em{pname}', 'Em', 'E'])
        # MF saddle keys: try suffixed first (vEstar), then bare (vstar).
        if f'v{pname}star' in mf_values:
            v_mf_arr = mf_values[f'v{pname}star']
        elif 'vstar' in mf_values:
            v_mf_arr = mf_values['vstar']
        else:
            raise KeyError(
                f'No MF saddle key for population {pname!r}: tried '
                f'{f"v{pname}star"!r} and {"vstar"!r}; '
                f'available: {list(mf_values.keys())}'
            )
        for i in range(size):
            tau_v[start + i]   = float(tau_arr[i])
            a_gain[start + i]  = float(a_arr[i])
            E_drive[start + i] = float(e_arr[i])
            v_init[start + i]  = float(v_mf_arr[i])

    # ── Per-pair coupling W and filter τ_g ────────────────────────
    # For the multipop_test action:
    #   (τE Dt+1) vE_i = EmE_i + sum_j wEE[i,j] gEE[i,j] nE_j
    #                          - sum_j wEI[i,j] gEI[i,j] nI_j
    # So the sign is: + wEE, - wEI, + wIE, - wII.  We bake the sign
    # into W here so the simulator's voltage update is just sum_j
    # W[i,j] * F[i,j] with no extra logic.
    #
    # For single-pop theories the matrix params are bare (``w``,
    # ``taug``) and the only pair is (E, E) so the sign is + by default.
    sign_map = {
        # (post_pop, pre_pop): sign
        ('E', 'E'): +1.0,
        ('E', 'I'): -1.0,
        ('I', 'E'): +1.0,
        ('I', 'I'): -1.0,
    }
    for post in pops:
        for pre in pops:
            wname_pop   = f'w{post["name"]}{pre["name"]}'
            tgname_pop  = f'taug{post["name"]}{pre["name"]}'
            # Coupling matrix W — required.  ``taug`` may be absent for
            # theories with a delta kernel (the simulator that consumes
            # them ignores tau_g).
            if wname_pop in fundamental:
                ws = fundamental[wname_pop]
            elif 'w' in fundamental:
                ws = fundamental['w']
            else:
                continue
            if tgname_pop in fundamental:
                tgs = fundamental[tgname_pop]
            elif 'taug' in fundamental:
                tgs = fundamental['taug']
            else:
                tgs = None
            sign = sign_map.get((post['name'], pre['name']), +1.0)
            post_start, post_size = pop_offsets[post['name']]
            pre_start, pre_size   = pop_offsets[pre['name']]
            for i in range(post_size):
                for j in range(pre_size):
                    W[post_start + i, pre_start + j] = sign * float(ws[i][j])
                    if tgs is not None:
                        tau_g[post_start + i, pre_start + j] = float(tgs[i][j])

    return {
        'N':           N,
        'tau_v':       tau_v,
        'a_gain':      a_gain,
        'E_drive':     E_drive,
        'W':           W,
        'tau_g':       tau_g,
        'v_init':      v_init,
        'pop_offsets': pop_offsets,
    }


def flat_index_of(model, pop_offsets, pop_name, idx_one_based):
    """
    Map an external-field tuple ``(pop_name, 1-based idx)`` to the
    flat stacked-neuron index used by the simulator and the cumulant
    estimator.

    Example
    -------
    For a 2-pop model with E (size 2) before I (size 2):
        flat_index_of(model, off, 'E', 1) → 0
        flat_index_of(model, off, 'E', 2) → 1
        flat_index_of(model, off, 'I', 1) → 2
        flat_index_of(model, off, 'I', 2) → 3
    """
    if pop_name not in pop_offsets:
        # Strategy 1: peel a leading 'n' or 'dn' off (multipop suffix
        # convention: 'nE' → 'E', 'dnE' → 'E').
        stripped = None
        for prefix in ('dn', 'n'):
            if pop_name.startswith(prefix):
                candidate = pop_name[len(prefix):]
                if candidate and candidate in pop_offsets:
                    stripped = candidate
                    break
        if stripped is not None:
            pop_name = stripped
        else:
            # Strategy 2: look up the population from the field's
            # ``population`` annotation in the model.  This handles
            # single-pop theories where the field is just 'n' (no
            # population suffix to strip).
            field_pop = None
            for f in (model.get('physical_fields', []) +
                      model.get('response_fields', [])):
                natural = f.get('natural_name') or f['name']
                if natural == pop_name or f['name'] == pop_name:
                    field_pop = f.get('population')
                    break
            if field_pop and field_pop in pop_offsets:
                pop_name = field_pop
            elif len(pop_offsets) == 1:
                # Strategy 3: single-pop fallback — there's only one
                # population, so any field maps to it.
                pop_name = next(iter(pop_offsets))
            else:
                raise KeyError(
                    f'Cannot map field/population {pop_name!r} to a '
                    f'stacked-index offset; known offsets '
                    f'{list(pop_offsets.keys())}'
                )
    start, size = pop_offsets[pop_name]
    if not (1 <= idx_one_based <= size):
        raise IndexError(
            f'Index {idx_one_based} out of range for population '
            f'{pop_name!r} of size {size}'
        )
    return start + (idx_one_based - 1)

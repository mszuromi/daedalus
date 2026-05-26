"""
Soma + dendrite compartmental linear-rate simulator (Numba JIT).

Implements the stochastic dynamics underlying
``theories/single_pop_dendritic_linear.theory.py``:

  Soma:    nS_i(t) ~ Poisson( λ_S_i(t) · dt )
           λ_S_i(t) = max(0, ES_i
                          + Σ_j wSS[i,j] (gS_i ⊛ nS_j)(t)
                          + Σ_j wSD[i,j] (gS_i ⊛ nD_j)(t))

  Dendrite (conditional on somatic spike count k_S_i at step t):
           nD_i(t) ~ Binomial( k_S_i, p_D_i(t) )
           p_D_i(t) = clip( ED_i
                            + Σ_j wDS[i,j] (gD_i ⊛ nS_j)(t)
                            + Σ_j wDD[i,j] (gD_i ⊛ nD_j)(t),
                            0, 1 )

Both filters ``gS_i`` and ``gD_i`` are normalized exponentials:
  ``gS_i(t) = (1/tauS_i) e^{-t/tauS_i} Θ(t)``
  ``gD_i(t) = (1/tauD_i) e^{-t/tauD_i} Θ(t)``

The ``clip(p_D, 0, 1)`` is the cap-at-one convention from the paper's
Bernoulli-probability rectification.  Without it the linear
parameterisation can push ``p_D > 1`` in regions of high somatic
drive; the MSR-JD theory's saddle equations don't enforce this cap
(the user is aware — known finite-source-of-mismatch when the cap
fires; treated as a probe of how far the linear-approximation
regime extends).

Per-pair filters
----------------
There are four (N×N) filter banks — one for each (post-side, pre-spike)
combination:

  F_SS[i,j]:  gS_i ⊛ nS_j   driving somatic rate of i
  F_SD[i,j]:  gS_i ⊛ nD_j   (dendritic spikes from j → somatic side of i)
  F_DS[i,j]:  gD_i ⊛ nS_j   (somatic spikes from j → dendritic side of i)
  F_DD[i,j]:  gD_i ⊛ nD_j   driving dendritic prob of i

The kernel is indexed only by the post-synaptic compartment (i),
not the pair (i,j); so all four banks share a common decay timescale
per row (``tauS_i`` for the S-side, ``tauD_i`` for the D-side).
"""
from __future__ import annotations

import numpy as np
import numba


# ── Simulator ────────────────────────────────────────────────────────


@numba.njit
def sim_dendritic_linear_numba(n_steps, dt_sim,
                               wSS, wSD, wDS, wDD,
                               tauS, tauD,
                               ES, ED,
                               bin_size_steps, n_bins, seed,
                               cap_pD_at_one=True):
    """
    Euler-step simulation of the soma + dendrite compartmental linear
    Hawkes-Bernoulli process.

    Parameters
    ----------
    n_steps : int
        Total number of Euler timesteps.
    dt_sim : float
        Euler timestep (in time units used everywhere — typically ms).
    wSS, wSD, wDS, wDD : np.ndarray (N, N)
        Signed effective coupling matrices.  ``wXY[i, j]`` is the
        post-i-X, pre-j-Y synaptic strength.
    tauS, tauD : np.ndarray (N,)
        Membrane / Bernoulli-filter time constants for the somatic
        and dendritic compartments respectively.
    ES, ED : np.ndarray (N,)
        Resting drive (somatic rate constant; dendritic Bernoulli
        baseline probability).
    bin_size_steps : int
        Number of Euler steps per spike-count bin.
    n_bins : int
        Total number of bins to fill.
    seed : int
        Random seed.
    cap_pD_at_one : bool, default True
        If True, clip ``p_D`` to ``[0, 1]`` each step.  Set False to
        let the simulator throw down whatever Binomial-with-prob>1
        np.random.binomial allows (it will raise) — useful only as a
        sanity check on the linear regime.

    Returns
    -------
    binned_nS : np.ndarray (N, n_bins)
        Somatic spike counts per bin per neuron.
    binned_nD : np.ndarray (N, n_bins)
        Dendritic spike counts per bin per neuron.
    rate_S_bins : np.ndarray (N, n_bins)
        Bin-averaged somatic rate (diagnostic; pre-Poisson).
    prob_D_bins : np.ndarray (N, n_bins)
        Bin-averaged dendritic Bernoulli probability (diagnostic).
    total_nS : np.ndarray (N,)
        Total somatic spike count per neuron over the full run.
    total_nD : np.ndarray (N,)
        Total dendritic spike count per neuron over the full run.
    """
    np.random.seed(seed)
    N = len(tauS)

    # Filters (one per (post, pre-train) pair).
    F_SS = np.zeros((N, N))
    F_SD = np.zeros((N, N))
    F_DS = np.zeros((N, N))
    F_DD = np.zeros((N, N))

    binned_nS = np.zeros((N, n_bins))
    binned_nD = np.zeros((N, n_bins))
    rate_S_bins = np.zeros((N, n_bins))
    prob_D_bins = np.zeros((N, n_bins))
    rate_S_accum = np.zeros(N)
    prob_D_accum = np.zeros(N)

    total_nS = np.zeros(N)
    total_nD = np.zeros(N)

    spikes_S = np.zeros(N, dtype=np.int64)
    spikes_D = np.zeros(N, dtype=np.int64)

    current_bin = 0
    steps_in_bin = 0

    for step in range(n_steps):
        # 1) Per-neuron rates and probabilities, then spikes.
        for i in range(N):
            lam = ES[i]
            p   = ED[i]
            for j in range(N):
                lam += wSS[i, j] * F_SS[i, j] + wSD[i, j] * F_SD[i, j]
                p   += wDS[i, j] * F_DS[i, j] + wDD[i, j] * F_DD[i, j]
            if lam < 0.0:
                lam = 0.0
            if cap_pD_at_one:
                if p < 0.0:
                    p = 0.0
                if p > 1.0:
                    p = 1.0
            rate_S_accum[i] += lam
            prob_D_accum[i] += p

            # Soma: Poisson(λ·dt).
            kS = np.random.poisson(lam * dt_sim)
            spikes_S[i] = kS
            total_nS[i] += kS

            # Dendrite: Binomial(kS, p) — one Bernoulli per somatic spike.
            kD = 0
            for _k in range(kS):
                if np.random.rand() < p:
                    kD += 1
            spikes_D[i] = kD
            total_nD[i] += kD

            if current_bin < n_bins:
                binned_nS[i, current_bin] += kS
                binned_nD[i, current_bin] += kD

        # 2) Bin completion bookkeeping.
        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    rate_S_bins[i, current_bin] = (rate_S_accum[i]
                                                   / bin_size_steps)
                    prob_D_bins[i, current_bin] = (prob_D_accum[i]
                                                   / bin_size_steps)
                    rate_S_accum[i] = 0.0
                    prob_D_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # 3) Exponential filter updates.  Each row's S-side filters
        # share decay tauS[i]; each row's D-side share decay tauD[i].
        for i in range(N):
            decay_S = 1.0 - dt_sim / tauS[i]
            decay_D = 1.0 - dt_sim / tauD[i]
            inv_tauS = 1.0 / tauS[i]
            inv_tauD = 1.0 / tauD[i]
            for j in range(N):
                F_SS[i, j] = decay_S * F_SS[i, j] + inv_tauS * spikes_S[j]
                F_SD[i, j] = decay_S * F_SD[i, j] + inv_tauS * spikes_D[j]
                F_DS[i, j] = decay_D * F_DS[i, j] + inv_tauD * spikes_S[j]
                F_DD[i, j] = decay_D * F_DD[i, j] + inv_tauD * spikes_D[j]

    return (binned_nS, binned_nD,
            rate_S_bins, prob_D_bins,
            total_nS, total_nD)


# ── Build sim arrays from a theory model dict ─────────────────────────


def build_sim_arrays(model: dict, fundamental: dict, mf_values: dict):
    """
    Translate ``model`` (from ``theories/single_pop_dendritic_linear.
    theory.py``) + ``fundamental`` + ``mf_values`` into the numpy
    arrays ``sim_dendritic_linear_numba`` consumes.

    Returns a dict with keys ``N``, ``wSS``, ``wSD``, ``wDS``, ``wDD``,
    ``tauS``, ``tauD``, ``ES``, ``ED``, plus ``nS_mf``, ``nD_mf`` for
    the saddle-point rate / dendritic-rate per neuron.  The
    ``stack_offsets`` dict maps the field-flavour-suffix (``'nS'``,
    ``'nD'``) to the row-range it occupies in the stacked
    ``binned_counts`` array the cumulant estimator consumes.
    """
    populations = model.get('populations') or []
    if len(populations) != 1:
        raise ValueError(
            "build_sim_arrays: dendritic-linear simulator currently "
            "supports a single declared population (the theory's "
            f"population list has {len(populations)} entries)."
        )
    N = int(populations[0]['size'])

    def _np_param(name, shape):
        if name not in fundamental:
            raise KeyError(
                f"build_sim_arrays: parameter {name!r} missing from "
                f"`fundamental`.  Provide a value matching the "
                f"declared shape {shape}."
            )
        arr = np.asarray(fundamental[name], dtype=float)
        if arr.shape != shape:
            raise ValueError(
                f"build_sim_arrays: parameter {name!r} has shape "
                f"{arr.shape}; expected {shape}."
            )
        return arr

    wSS = _np_param('wSS', (N, N))
    wSD = _np_param('wSD', (N, N))
    wDS = _np_param('wDS', (N, N))
    wDD = _np_param('wDD', (N, N))
    tauS = _np_param('tauS', (N,))
    tauD = _np_param('tauD', (N,))
    ES   = _np_param('ES',   (N,))
    ED   = _np_param('ED',   (N,))

    nS_mf = np.asarray(mf_values.get('nSstar', [0.0] * N), dtype=float)
    nD_mf = np.asarray(mf_values.get('nDstar', [0.0] * N), dtype=float)

    # Row stacking convention: somatic spike trains first, then
    # dendritic.  Two flavours × N neurons = 2N rows in the
    # ``binned_counts`` array fed to ``compute_kpoint_slice``.
    stack_offsets = {
        'nS': (0, N),
        'nD': (N, N),
    }

    return {
        'N':              N,
        'wSS':            wSS,
        'wSD':            wSD,
        'wDS':            wDS,
        'wDD':            wDD,
        'tauS':           tauS,
        'tauD':           tauD,
        'ES':             ES,
        'ED':             ED,
        'nS_mf':          nS_mf,
        'nD_mf':          nD_mf,
        'stack_offsets':  stack_offsets,
    }


def flat_index_of(stack_offsets: dict, field_name: str, idx_one_based: int) -> int:
    """
    Map ``('nS', 1)`` / ``('nD', 2)`` to a flat row index into the
    stacked ``binned_counts`` array.

    ``stack_offsets`` is the dict returned by :func:`build_sim_arrays`
    under the key of the same name.  ``field_name`` should be one of
    ``'nS'`` or ``'nD'``; ``idx_one_based`` is 1-based neuron index
    (matching the framework's external-field convention).
    """
    if field_name not in stack_offsets:
        raise KeyError(
            f"flat_index_of: field {field_name!r} not in "
            f"stack_offsets {list(stack_offsets)}."
        )
    start, size = stack_offsets[field_name]
    i = int(idx_one_based) - 1
    if i < 0 or i >= size:
        raise IndexError(
            f"flat_index_of: neuron index {idx_one_based} out of range "
            f"for {field_name!r} (size {size})."
        )
    return start + i


def stack_binned_counts(binned_nS: np.ndarray,
                        binned_nD: np.ndarray) -> np.ndarray:
    """Vertically stack soma and dendrite spike-count bins into a single
    ``(2N, n_bins)`` array, matching the ``stack_offsets`` convention
    used by :func:`flat_index_of`."""
    return np.vstack([binned_nS, binned_nD])

"""
Two-compartment (soma + dendrite) point-process simulator (Numba JIT) for
``theories/dendritic_quad_soma_sigmoid.theory.py``.

Voltage formulation — each compartment carries a voltage that exponentially
low-pass-filters its synaptic input:

  Somatic voltage    tauS_i vS_i' = -(vS_i - ES_i) + Σ_j wSD_ij nD_j
  Dendritic voltage  tauD_i vD_i' = -(vD_i - ED_i) + Σ_j wDS_ij nS_j

  Soma     -- QUADRATIC rate, Poisson spikes:
           nS_i(t) ~ Poisson( aS_i vS_i(t)^2 · dt )
  Dendrite -- SIGMOID probability, Bernoulli gate (one trial per somatic spike):
           nD_i(t) ~ Binomial( nS_i(t), σ(vD_i(t)) ),   σ(x) = 1/(1+e^{-x})

The sigmoid keeps p_D in (0,1) by construction, so — unlike the linear dendritic
model — no probability clipping is needed.  A single dendritic spike kicks the
somatic voltage by wSD/tauS and decays over tauS (∫ = wSD), matching the
framework's unit-DC-gain propagator 1/(1+iωτS).

Output shape matches the Hawkes ``binned_counts`` convention so
``simulations.cumulant_estimator`` consumes it directly.
"""
from __future__ import annotations

import numpy as np
import numba


@numba.njit
def sim_dendritic_quad_sigmoid_numba(n_steps, dt_sim,
                                     aS, ES, ED, tauS, tauD,
                                     wSD, wDS,
                                     bin_size_steps, n_bins, seed):
    """
    Euler-step simulation of the quadratic-soma / sigmoid-probability-dendrite
    two-compartment process.

    Parameters
    ----------
    n_steps : int
        Number of Euler timesteps.
    dt_sim : float
        Euler timestep (dt ≪ tauS, tauD).
    aS : np.ndarray (N,)
        Somatic quadratic gain (rate = aS·vS²).
    ES, ED : np.ndarray (N,)
        Somatic / dendritic voltage baselines (ED is the sigmoid input bias).
    tauS, tauD : np.ndarray (N,)
        Somatic / dendritic voltage time constants.
    wSD, wDS : np.ndarray (N, N)
        wSD = dendrite→soma weights; wDS = soma→dendrite weights.
        (wSD[i,j] kicks soma i's voltage from dendrite spikes of j.)
    bin_size_steps, n_bins, seed : int
        Binning and RNG control.

    Returns
    -------
    binned_nS, binned_nD : np.ndarray (N, n_bins)
        Somatic / dendritic spike counts per bin.
    rateS_bins, probD_bins : np.ndarray (N, n_bins)
        Bin-averaged somatic rate and dendritic probability (diagnostics).
    total_nS, total_nD : np.ndarray (N,)
        Total spike counts per neuron (for empirical rates).
    """
    np.random.seed(seed)
    N = len(aS)

    vS = ES.copy()                 # start at the voltage baselines
    vD = ED.copy()

    binned_nS = np.zeros((N, n_bins))
    binned_nD = np.zeros((N, n_bins))
    rateS_bins = np.zeros((N, n_bins))
    probD_bins = np.zeros((N, n_bins))
    rateS_accum = np.zeros(N)
    probD_accum = np.zeros(N)
    total_nS = np.zeros(N)
    total_nD = np.zeros(N)

    kS = np.zeros(N, dtype=np.int64)
    kD = np.zeros(N, dtype=np.int64)

    current_bin = 0
    steps_in_bin = 0

    for step in range(n_steps):
        # 1) Rates / probabilities from the CURRENT voltages, then spikes.
        for i in range(N):
            lam = aS[i] * vS[i] * vS[i]          # quadratic soma rate (>= 0)
            if lam < 0.0:
                lam = 0.0
            p = 1.0 / (1.0 + np.exp(-vD[i]))     # sigmoid probability in (0,1)
            rateS_accum[i] += lam
            probD_accum[i] += p

            ks = np.random.poisson(lam * dt_sim)
            kS[i] = ks
            total_nS[i] += ks

            # Binomial(ks, p).  Guard against pathological blow-up (a
            # supercritical parameter choice can send ks astronomical and make
            # the per-trial loop hang): for large ks use a Gaussian approx.
            if ks > 200:
                kd = int(round(ks * p + np.sqrt(ks * p * (1.0 - p)) * np.random.randn()))
                if kd < 0:
                    kd = 0
                elif kd > ks:
                    kd = ks
            else:
                kd = 0
                for _k in range(ks):
                    if np.random.rand() < p:
                        kd += 1
            kD[i] = kd
            total_nD[i] += kd

            if current_bin < n_bins:
                binned_nS[i, current_bin] += ks
                binned_nD[i, current_bin] += kd

        # 2) Bin bookkeeping.
        steps_in_bin += 1
        if steps_in_bin >= bin_size_steps:
            if current_bin < n_bins:
                for i in range(N):
                    rateS_bins[i, current_bin] = rateS_accum[i] / bin_size_steps
                    probD_bins[i, current_bin] = probD_accum[i] / bin_size_steps
                    rateS_accum[i] = 0.0
                    probD_accum[i] = 0.0
            current_bin += 1
            steps_in_bin = 0

        # 3) Voltage updates: decay toward baseline + spike kicks.
        #    ΔvS = (dt/tauS)(ES - vS) + (1/tauS) Σ_j wSD[i,j] kD_j  (unit-area PSP).
        for i in range(N):
            dvS = (dt_sim / tauS[i]) * (ES[i] - vS[i])
            dvD = (dt_sim / tauD[i]) * (ED[i] - vD[i])
            for j in range(N):
                dvS += (wSD[i, j] / tauS[i]) * kD[j]
                dvD += (wDS[i, j] / tauD[i]) * kS[j]
            vS[i] += dvS
            vD[i] += dvD

    return (binned_nS, binned_nD,
            rateS_bins, probD_bins,
            total_nS, total_nD)


# ── Build sim arrays from a theory model dict ─────────────────────────


def build_sim_arrays(model: dict, fundamental: dict, mf_values: dict):
    """Translate ``model`` + ``fundamental`` + ``mf_values`` into the numpy
    arrays ``sim_dendritic_quad_sigmoid_numba`` consumes.  Returns a dict with
    ``N``, ``aS``, ``ES``, ``ED``, ``tauS``, ``tauD``, ``wSD``, ``wDS``, plus
    ``nS_mf``/``nD_mf`` and ``stack_offsets`` (``'nS'``/``'nD'`` row ranges)."""
    populations = model.get('populations') or []
    if len(populations) != 1:
        raise ValueError(
            "build_sim_arrays: this simulator supports a single declared "
            f"population; the theory has {len(populations)}."
        )
    N = int(populations[0]['size'])

    def _p(name, shape):
        if name not in fundamental:
            raise KeyError(f"build_sim_arrays: parameter {name!r} missing from `fundamental`.")
        arr = np.asarray(fundamental[name], dtype=float)
        if arr.shape != shape:
            raise ValueError(
                f"build_sim_arrays: parameter {name!r} has shape {arr.shape}; expected {shape}.")
        return arr

    aS = _p('aS', (N,)); ES = _p('ES', (N,)); ED = _p('ED', (N,))
    tauS = _p('tauS', (N,)); tauD = _p('tauD', (N,))
    wSD = _p('wSD', (N, N)); wDS = _p('wDS', (N, N))

    nS_mf = np.asarray(mf_values.get('nSstar', [0.0] * N), dtype=float)
    nD_mf = np.asarray(mf_values.get('nDstar', [0.0] * N), dtype=float)

    return {
        'N': N, 'aS': aS, 'ES': ES, 'ED': ED, 'tauS': tauS, 'tauD': tauD,
        'wSD': wSD, 'wDS': wDS, 'nS_mf': nS_mf, 'nD_mf': nD_mf,
        'stack_offsets': {'nS': (0, N), 'nD': (N, N)},
    }


def flat_index_of(stack_offsets: dict, field_name: str, idx_one_based: int) -> int:
    """Map ``('nS', 1)`` / ``('nD', 2)`` to a flat row index into the stacked
    ``binned_counts`` array."""
    if field_name not in stack_offsets:
        raise KeyError(f"flat_index_of: field {field_name!r} not in {list(stack_offsets)}.")
    start, size = stack_offsets[field_name]
    i = int(idx_one_based) - 1
    if i < 0 or i >= size:
        raise IndexError(
            f"flat_index_of: neuron index {idx_one_based} out of range for {field_name!r}.")
    return start + i


def stack_binned_counts(binned_nS: np.ndarray, binned_nD: np.ndarray) -> np.ndarray:
    """Stack soma + dendrite spike-count bins into a single ``(2N, n_bins)`` array."""
    return np.vstack([binned_nS, binned_nD])

"""
models/cumulant_estimator.py
============================
Properly normalized k-point cumulant estimator for binned point-process
(spike train) data. Handles the self-spike correction that makes the
smooth part of the cumulant independent of the bin width dt_bin.

The key insight: when two or more fields from the SAME population share
the SAME time bin, the ordinary product of their binned counts includes
"self-spike" contributions — the same spike event contributing to
multiple factors. For a Poisson process with mean count μ = λ·dt_bin,
the ordinary second moment E[n²] = μ + μ² diverges as μ ~ dt_bin while
the factorial second moment E[n(n−1)] = μ² is dt_bin-independent.

General rule: for m same-population fields at the same bin, replace
the m-th power n^m with the falling factorial n·(n−1)·…·(n−m+1).
Fields from different populations at the same bin need no correction
(different populations can't share a spike event). Fields at different
bins need no correction (no self-correlation across bins).

This correction is the standard "factorial cumulant" approach for
multi-point statistics of point processes (see e.g., Daley & Vere-Jones,
"An Introduction to the Theory of Point Processes", Ch. 5).
"""

import numpy as np
from collections import Counter


def falling_factorial_array(n_arr, m):
    """
    Compute n·(n−1)·…·(n−m+1) element-wise for an integer array.

    Parameters
    ----------
    n_arr : np.ndarray
        Array of non-negative integer counts.
    m : int
        Number of factors in the falling factorial (m >= 1).

    Returns
    -------
    np.ndarray (float)
    """
    result = np.ones_like(n_arr, dtype=float)
    for j in range(m):
        result *= (n_arr - j)
    return result


def compute_kpoint_slice(binned_counts, dt_bin, pop_indices, lag_bins,
                         max_lag_bins, n_fft=None):
    """
    Compute a 1-D slice of the k-point cumulant from binned spike-train
    data, with proper factorial corrections for same-population
    same-bin coincidences.

    The k fields are specified by `pop_indices` (one per external leg)
    and `lag_bins` (one per external leg — the lag IN BINS relative to
    the reference time). Exactly ONE of the lags should be `None`,
    indicating the "varying" axis whose cross-correlation lag will be
    swept to produce the 1-D output array. The other k−1 lags are
    FIXED integers.

    Parameters
    ----------
    binned_counts : np.ndarray (npop, n_bins)
        Raw (UN-subtracted) spike counts per bin per population.
    dt_bin : float
        Bin width in time units.
    pop_indices : list of int
        Population index for each of the k external legs (0-based).
    lag_bins : list of (int or None)
        Time lag in bins for each leg. Exactly ONE entry must be None
        (the "sweep" axis). Others are fixed integers.
    max_lag_bins : int
        Maximum lag to extract from the cross-correlation.
    n_fft : int or None
        FFT length for zero-padding (default: 2 × n_bins).

    Returns
    -------
    tau_grid : np.ndarray (2*max_lag_bins + 1,)
        Lag values in time units.
    C_slice : np.ndarray (2*max_lag_bins + 1,)
        Cumulant slice values (real).
    """
    k = len(pop_indices)
    assert len(lag_bins) == k
    sweep_idx = [i for i, lb in enumerate(lag_bins) if lb is None]
    assert len(sweep_idx) == 1, (
        f"Exactly one lag must be None (the sweep axis); got {sweep_idx}"
    )
    sweep_idx = sweep_idx[0]

    npop, n_bins = binned_counts.shape
    if n_fft is None:
        n_fft = 2 * n_bins

    # Mean-subtract the counts (work with counts, not rates, for
    # cleaner factorial handling; convert to rate units at the end).
    mean_counts = binned_counts.mean(axis=1)  # (npop,)
    delta_counts = binned_counts - mean_counts[:, None]  # (npop, n_bins)

    # ── Partition the k legs into "same-bin groups" ──
    # Two legs are in the same bin if they have the same fixed lag.
    # The sweep leg is in its own group (its lag varies).
    fixed_legs = [(i, lag_bins[i]) for i in range(k) if i != sweep_idx]
    sweep_pop = pop_indices[sweep_idx]

    # Group the FIXED legs by their lag value
    lag_to_legs = {}
    for (i, lag) in fixed_legs:
        lag_to_legs.setdefault(lag, []).append(i)

    # ── Build the "product" time series ──
    # For each group of fixed legs at the same lag, compute the
    # corrected product (using falling factorials for same-pop
    # coincidences within the group).
    product = np.ones(n_bins, dtype=float)

    for lag_val, leg_indices in lag_to_legs.items():
        # Shift the counts to align with this lag
        # (circular shift — the FFT cross-correlation handles wrap)
        shifted_counts = {}
        shifted_delta = {}
        for i in leg_indices:
            pop = pop_indices[i]
            if pop not in shifted_counts:
                shifted_counts[pop] = np.roll(binned_counts[pop], -lag_val)
                shifted_delta[pop] = np.roll(delta_counts[pop], -lag_val)

        # Count how many legs from each population are at this lag
        pop_multiplicities = Counter(pop_indices[i] for i in leg_indices)

        for pop, m in pop_multiplicities.items():
            n_arr = shifted_counts[pop]
            mean_n = float(mean_counts[pop])

            if m == 1:
                # Single field: ordinary centered count
                factor = (n_arr - mean_n) / dt_bin
            elif m == 2:
                # Two same-pop fields at same bin: factorial correction
                # Centered falling factorial:
                #   n(n-1) - 2·mean·n + mean·(mean+1)  ... wait, let me
                # just compute the mean-subtracted version directly.
                #
                # We want: E_corrected = n(n-1)/dt² such that
                #   ⟨n(n-1)/dt²⟩ = ⟨n⟩²/dt² = mean_rate² for Poisson
                #   (no 1/dt shot-noise divergence)
                #
                # Mean-subtracted: [n(n-1) - mean_n·(mean_n - 1)] / dt²
                # But we also want the CENTERED version (subtract the
                # mean of the corrected product). Simplest: compute the
                # corrected product, then subtract its own mean.
                raw = falling_factorial_array(n_arr, 2) / dt_bin**2
                factor = raw - raw.mean()
            elif m == 3:
                raw = falling_factorial_array(n_arr, 3) / dt_bin**3
                factor = raw - raw.mean()
            else:
                raw = falling_factorial_array(n_arr, m) / dt_bin**m
                factor = raw - raw.mean()

            product *= factor

    # ── Cross-correlate with the sweep leg ──
    sweep_rate = delta_counts[sweep_pop] / dt_bin  # mean-subtracted rate

    F_product = np.fft.fft(product, n=n_fft)
    F_sweep = np.fft.fft(sweep_rate, n=n_fft)
    # Note: ifft(F_product × conj(F_sweep))[lag] = Σ_t product(t) × sweep(t − lag)
    # We want Σ_t product(t) × sweep(t + lag), so we flip the lag axis.
    xcorr = np.fft.ifft(F_product * np.conj(F_sweep)).real / n_bins
    xcorr = xcorr[::-1]  # flip to get positive-lag convention

    # ── Extract the lag window ──
    tau_grid = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_bin
    C_slice = np.zeros(2 * max_lag_bins + 1)
    for idx_lag, lag in enumerate(range(-max_lag_bins, max_lag_bins + 1)):
        C_slice[idx_lag] = xcorr[lag % n_fft]

    return tau_grid, C_slice

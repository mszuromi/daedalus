"""
models/cumulant_estimator.py
============================
Estimator for lagged factorial-cumulant slices of binned point-process
(spike train) data.

What this computes
------------------
For k fields at specified time lags, this estimates the **factorial
cumulant density** — the connected k-point correlation with self-spike
(diagonal) contributions removed via falling-factorial corrections.

For k=2 this coincides with the ordinary covariance of rates (since the
only "diagonal" is the shot noise δ(τ), which the factorial correction
removes).

For k=3 the estimator computes

    ⟨F_grouped(t) · δrate_sweep(t + lag)⟩

where F_grouped is a centered, factorial-corrected product of the
fixed-lag fields. This is a well-defined, dt_bin-independent estimator
for the **smooth** (off-diagonal) part of the third-order cumulant
density. It deliberately removes self-spike shot-noise contributions
that would otherwise scale as 1/dt_bin per coincident same-population
pair.

This is the correct comparison target for Phase J's tree-level output,
which also computes the smooth connected correlator with δ components
separated.

What this does NOT compute
--------------------------
- The full raw k-point cumulant including all diagonal/contact terms.
  Those terms are distributions (δ functions) that can't be represented
  as smooth functions on a τ grid.
- Higher-order factorial cumulant corrections beyond grouping by
  coincident time bins. For k ≥ 4 with complex grouping patterns, the
  estimator structure may need re-derivation.

Implementation details
----------------------
- **Factorial correction**: for m same-population fields at the same
  bin, replaces n^m with n(n−1)...(n−m+1) (falling factorial). This
  removes self-spike contributions that scale as 1/dt_bin.
- **Lag-dependent normalization**: uses (n_bins − max_fixed_lag − |sweep_lag|)
  valid overlapping bins per lag, not the constant n_bins. This gives
  an unbiased finite-window estimator.
- **No circular wraparound**: fixed-lag fields are aligned via explicit
  array slicing (not np.roll), so no periodic boundary contamination.
- **Zero-padded FFT**: the sweep-axis cross-correlation uses n_fft ≥ 2×n_valid
  to ensure the FFT correlation is a linear (non-circular) correlation
  over the valid window.

Reference: Daley & Vere-Jones, "An Introduction to the Theory of Point
Processes", Ch. 5 (factorial moment measures).
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
    Compute a 1-D slice of the factorial cumulant density from binned
    spike-train data.

    Parameters
    ----------
    binned_counts : np.ndarray (npop, n_bins)
        Raw (UN-subtracted) spike counts per bin per population.
    dt_bin : float
        Effective bin width in time units (use dt_bin_eff, not nominal).
    pop_indices : list of int
        Population index for each of the k external legs (0-based).
    lag_bins : list of (int or None)
        Time lag in bins for each leg. Exactly ONE entry must be None
        (the "sweep" axis). Others are fixed integers.
    max_lag_bins : int
        Maximum sweep lag to extract.
    n_fft : int or None
        FFT length for zero-padding (default: next power of 2 ≥ 2×n_valid).

    Returns
    -------
    tau_grid : np.ndarray (2*max_lag_bins + 1,)
        Lag values in time units (= lag_index × dt_bin).
    C_slice : np.ndarray (2*max_lag_bins + 1,)
        Factorial cumulant density slice in Hz^k. The smooth (off-diagonal)
        part of the k-point cumulant with self-spike contributions removed.
    """
    k = len(pop_indices)
    assert len(lag_bins) == k
    sweep_idx = [i for i, lb in enumerate(lag_bins) if lb is None]
    assert len(sweep_idx) == 1, (
        f"Exactly one lag must be None (the sweep axis); got {sweep_idx}"
    )
    sweep_idx = sweep_idx[0]

    npop, n_bins = binned_counts.shape

    # ── Determine the valid (non-wrapped) window ──
    # Fixed lags shift some fields relative to the reference. The valid
    # overlap region is the intersection of all shifted arrays.
    fixed_lags = [lag_bins[i] for i in range(k) if i != sweep_idx]
    max_fixed_lag = max(abs(l) for l in fixed_lags) if fixed_lags else 0

    # Valid bins: those where ALL fixed-lag fields have data (no wraparound).
    # If the largest fixed lag is L, the valid window is bins [L, n_bins-1]
    # for positive lags, or [0, n_bins-1-L] for negative, etc.
    # General: valid_start = max(0, max(positive_lags)),
    #          valid_end = n_bins - max(0, max(negative_lags_abs))
    # But since lags can be positive or negative, compute the range:
    all_fixed_lags = [0] + list(fixed_lags)  # include reference at lag 0
    min_lag = min(all_fixed_lags)
    max_lag = max(all_fixed_lags)
    valid_start = max(0, -min_lag)       # if min_lag < 0, need to skip first |min_lag| bins
    valid_end = n_bins - max(0, max_lag)  # if max_lag > 0, need to skip last max_lag bins
    n_valid = valid_end - valid_start

    if n_valid <= 2 * max_lag_bins:
        raise ValueError(
            f"Valid overlap window ({n_valid} bins) is too small for "
            f"max_lag_bins={max_lag_bins}. Reduce max_lag_bins or use a "
            f"longer recording."
        )

    # ── Compute mean counts (from the valid window for consistency) ──
    mean_counts = np.zeros(npop)
    for p in range(npop):
        mean_counts[p] = binned_counts[p, valid_start:valid_end].mean()

    # ── Build the "product" time series on the VALID window ──
    # For each group of fixed legs at the same lag, compute the
    # factorial-corrected product using explicit slicing (no np.roll).
    fixed_legs = [(i, lag_bins[i]) for i in range(k) if i != sweep_idx]
    sweep_pop = pop_indices[sweep_idx]

    # Group fixed legs by their lag value
    lag_to_legs = {}
    for (i, lag) in fixed_legs:
        lag_to_legs.setdefault(lag, []).append(i)

    product = np.ones(n_valid, dtype=float)

    for lag_val, leg_indices in lag_to_legs.items():
        # Extract the aligned slice for this lag (no circular wrap)
        # Bin index t in the valid window maps to original bin (valid_start + t).
        # Field at lag_val maps to original bin (valid_start + t + lag_val).
        offset = valid_start + lag_val

        # Group by population multiplicity
        pop_multiplicities = Counter(pop_indices[i] for i in leg_indices)

        for pop, m in pop_multiplicities.items():
            # Extract the aligned count array for this population at this lag
            n_arr = binned_counts[pop, offset:offset + n_valid].copy()
            mean_n = float(mean_counts[pop])

            if m == 1:
                factor = (n_arr - mean_n) / dt_bin
            else:
                # Factorial correction: n(n-1)...(n-m+1) / dt^m, centered
                raw = falling_factorial_array(n_arr, m) / dt_bin**m
                factor = raw - raw.mean()

            product *= factor

    # ── Build the sweep time series on the valid window ──
    sweep_arr = binned_counts[sweep_pop, valid_start:valid_end].copy()
    sweep_rate = (sweep_arr - mean_counts[sweep_pop]) / dt_bin

    # ── Linear (non-circular) cross-correlation via zero-padded FFT ──
    if n_fft is None:
        n_fft = int(2**np.ceil(np.log2(2 * n_valid)))

    F_product = np.fft.fft(product, n=n_fft)
    F_sweep = np.fft.fft(sweep_rate, n=n_fft)

    # ifft(F_product × conj(F_sweep))[L] = Σ_t product(t) × sweep(t − L)
    # We want C(+lag) = Σ_t product(t) × sweep(t + lag), which is the
    # value at L = −lag in the raw output. Access as raw[(-lag) % n_fft].
    # Do NOT use [::-1] — that shifts the index by 1 bin.
    raw_xcorr = np.fft.ifft(F_product * np.conj(F_sweep)).real

    # ── Extract the lag window with lag-dependent normalization ──
    tau_grid = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_bin
    C_slice = np.zeros(2 * max_lag_bins + 1)

    for idx_lag, lag in enumerate(range(-max_lag_bins, max_lag_bins + 1)):
        # Number of valid overlapping terms at this sweep lag
        n_overlap = n_valid - abs(lag)
        if n_overlap <= 0:
            C_slice[idx_lag] = 0.0
            continue
        C_slice[idx_lag] = raw_xcorr[(-lag) % n_fft] / n_overlap

    return tau_grid, C_slice


def compute_kpoint_slice_direct(binned_counts, dt_bin, pop_indices, lag_bins,
                                max_lag_bins):
    """
    Direct (non-FFT) k-point factorial cumulant density estimator.

    Same interface and output as ``compute_kpoint_slice`` but uses an
    explicit double loop over time bins and sweep lags. Slower for large
    datasets (O(n_bins × n_lags) vs O(n_bins log n_bins) for FFT) but
    free of any FFT-related issues (circular wrap, zero-padding
    artifacts, spectral leakage). Use this to cross-check the FFT
    estimator.

    Parameters / Returns: same as ``compute_kpoint_slice``.
    """
    k = len(pop_indices)
    assert len(lag_bins) == k
    sweep_idx = [i for i, lb in enumerate(lag_bins) if lb is None]
    assert len(sweep_idx) == 1
    sweep_idx = sweep_idx[0]

    npop, n_bins = binned_counts.shape
    sweep_pop = pop_indices[sweep_idx]

    # Fixed legs and their lags
    fixed_legs = [(i, lag_bins[i]) for i in range(k) if i != sweep_idx]
    sweep_lag_of = lag_bins  # will use None for sweep

    # Group fixed legs by (lag, population) for factorial correction
    lag_to_legs = {}
    for (i, lag) in fixed_legs:
        lag_to_legs.setdefault(lag, []).append(i)

    # Mean counts (full array)
    mean_counts = binned_counts.mean(axis=1)  # (npop,)

    tau_grid = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_bin
    C_slice = np.zeros(2 * max_lag_bins + 1)

    for idx_lag, sweep_lag in enumerate(range(-max_lag_bins, max_lag_bins + 1)):
        # For each reference time t, ALL fields must be within [0, n_bins).
        # Determine the valid range of t.
        all_lags = [lag_bins[i] if i != sweep_idx else sweep_lag
                    for i in range(k)]
        min_lag = min(all_lags)
        max_lag_val = max(all_lags)
        t_start = max(0, -min_lag)
        t_end = n_bins - max(0, max_lag_val)
        n_overlap = t_end - t_start

        if n_overlap <= 0:
            C_slice[idx_lag] = 0.0
            continue

        # Build the product for each valid t
        accumulator = 0.0
        for t in range(t_start, t_end):
            val = 1.0

            # Fixed-leg groups (with factorial correction for same-bin same-pop)
            for lag_val, leg_indices in lag_to_legs.items():
                bin_idx = t + lag_val
                pop_mults = Counter(pop_indices[i] for i in leg_indices)
                for pop, m in pop_mults.items():
                    n = binned_counts[pop, bin_idx]
                    mean_n = mean_counts[pop]
                    if m == 1:
                        val *= (n - mean_n) / dt_bin
                    else:
                        # Factorial: n(n-1)...(n-m+1) / dt^m, centered
                        ff = 1.0
                        for j in range(m):
                            ff *= (n - j)
                        ff /= dt_bin**m
                        # Subtract the population mean of the factorial product
                        # (approximated by the full-array mean for efficiency)
                        ff_mean = falling_factorial_array(
                            binned_counts[pop], m
                        ).mean() / dt_bin**m
                        val *= (ff - ff_mean)

            # Sweep leg
            sweep_bin = t + sweep_lag
            n_s = binned_counts[sweep_pop, sweep_bin]
            val *= (n_s - mean_counts[sweep_pop]) / dt_bin

            accumulator += val

        C_slice[idx_lag] = accumulator / n_overlap

    return tau_grid, C_slice

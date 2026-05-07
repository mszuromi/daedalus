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
                         max_lag_bins, n_fft=None,
                         field_types=None, voltage_bins=None,
                         ext_binned_counts=None):
    """
    Compute a 1-D slice of the factorial cumulant density from binned
    point-process (and optionally voltage) data.

    Parameters
    ----------
    binned_counts : np.ndarray (npop, n_bins)
        Raw (UN-subtracted) cortical spike counts per bin per
        population.
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
    field_types : list of str or None
        Field base name for each external leg.  Supported values:

          * ``'dn'`` — cortical spike-train fluctuation (counts in
            ``binned_counts``).  Discrete shot-noise; same-bin spike
            multiplets get the factorial correction.
          * ``'dv'`` — cortical voltage fluctuation (smooth, in
            ``voltage_bins``).  Linear centering, no factorial.
          * ``'dm'`` — external GTaS rate fluctuation (spike-like, in
            ``ext_binned_counts``).  Same shot-noise treatment as
            ``'dn'``: discrete counts, dt_bin denominator, factorial
            correction on same-bin multiplets.

        If None, all legs default to ``'dn'`` (backward compatibility).
    voltage_bins : np.ndarray (npop, n_bins) or None
        Bin-averaged cortical voltage per population.  Required if any
        ``field_types`` entry is ``'dv'``.
    ext_binned_counts : np.ndarray (npop, n_bins) or None
        Binned external (GTaS) spike counts per population.  Required
        if any ``field_types`` entry is ``'dm'``.

    Returns
    -------
    tau_grid : np.ndarray (2*max_lag_bins + 1,)
        Lag values in time units (= lag_index × dt_bin).
    C_slice : np.ndarray (2*max_lag_bins + 1,)
        Factorial cumulant density slice.  For spike-only legs, units
        are Hz^k.  For mixed spike/voltage legs, units are Hz^(#dn+#dm)
        × V^(#dv).  The smooth (off-diagonal) part of the k-point
        cumulant with self-spike contributions removed.
    """
    k = len(pop_indices)
    assert len(lag_bins) == k
    sweep_idx = [i for i, lb in enumerate(lag_bins) if lb is None]
    assert len(sweep_idx) == 1, (
        f"Exactly one lag must be None (the sweep axis); got {sweep_idx}"
    )
    sweep_idx = sweep_idx[0]

    # Default field_types: all 'dn' (backward compatibility)
    if field_types is None:
        field_types = ['dn'] * k
    assert len(field_types) == k
    if any(ft == 'dv' for ft in field_types):
        if voltage_bins is None:
            raise ValueError(
                "field_types contains 'dv' but voltage_bins is None. "
                "Pass bin-averaged voltage traces from sim_hawkes_numba."
            )
    if any(ft == 'dm' for ft in field_types):
        if ext_binned_counts is None:
            raise ValueError(
                "field_types contains 'dm' but ext_binned_counts is None. "
                "Pass GTaS external spike counts from "
                "sim_hawkes_<...>_gtas_numba (the fourth return value, "
                "e.g. ``ext_binned_run``)."
            )

    def _data_for(leg_idx):
        """Return the binned data array for leg leg_idx based on its field type."""
        ft = field_types[leg_idx]
        if ft == 'dv':
            return voltage_bins
        if ft == 'dm':
            return ext_binned_counts
        return binned_counts  # 'dn' (default)

    def _is_spike_ft(ft):
        """True if the field is a discrete spike-counting field (gets
        factorial correction at coincident bins)."""
        return ft in ('dn', 'dm')

    def _centering_denom(leg_idx):
        """Return dt_bin for spike-counting fields (rate conversion),
        1.0 for voltage."""
        return dt_bin if _is_spike_ft(field_types[leg_idx]) else 1.0

    npop, n_bins = binned_counts.shape

    # ── Determine the valid (non-wrapped) window ──
    fixed_lags = [lag_bins[i] for i in range(k) if i != sweep_idx]

    all_fixed_lags = [0] + list(fixed_lags)  # include reference at lag 0
    min_lag = min(all_fixed_lags)
    max_lag = max(all_fixed_lags)
    valid_start = max(0, -min_lag)
    valid_end = n_bins - max(0, max_lag)
    n_valid = valid_end - valid_start

    if n_valid <= 2 * max_lag_bins:
        raise ValueError(
            f"Valid overlap window ({n_valid} bins) is too small for "
            f"max_lag_bins={max_lag_bins}. Reduce max_lag_bins or use a "
            f"longer recording."
        )

    # ── Compute mean for each (pop, field_type) combination ──
    # Means are computed from the valid window for consistency.
    mean_by_pop_ft = {}  # (pop, field_type) -> mean
    for leg_idx in range(k):
        pop = pop_indices[leg_idx]
        ft = field_types[leg_idx]
        key = (pop, ft)
        if key not in mean_by_pop_ft:
            data = _data_for(leg_idx)
            mean_by_pop_ft[key] = float(data[pop, valid_start:valid_end].mean())

    # Helper: pick the data array for a given (pop, field_type) key
    def _data_for_ft(ft):
        if ft == 'dv':
            return voltage_bins
        if ft == 'dm':
            return ext_binned_counts
        return binned_counts

    # ── Build the "product" time series on the VALID window ──
    fixed_legs = [(i, lag_bins[i]) for i in range(k) if i != sweep_idx]
    sweep_pop = pop_indices[sweep_idx]
    sweep_ft = field_types[sweep_idx]

    # Group fixed legs by their lag value
    lag_to_legs = {}
    for (i, lag) in fixed_legs:
        lag_to_legs.setdefault(lag, []).append(i)

    product = np.ones(n_valid, dtype=float)

    for lag_val, leg_indices in lag_to_legs.items():
        offset = valid_start + lag_val

        # Group by (population, field_type) — factorial correction
        # applies only within groups of spike fields at the same bin.
        # Voltage fields are always centered linearly.
        pop_ft_multiplicities = Counter(
            (pop_indices[i], field_types[i]) for i in leg_indices)

        for (pop, ft), m in pop_ft_multiplicities.items():
            data = _data_for_ft(ft)
            n_arr = data[pop, offset:offset + n_valid].copy()
            mean_n = mean_by_pop_ft[(pop, ft)]
            denom = dt_bin if _is_spike_ft(ft) else 1.0

            if m == 1 or ft == 'dv':
                # Voltage fields: linear centering (not factorial)
                factor = (n_arr - mean_n) / denom
                # Voltage at multiplicity > 1: raise to power (no factorial)
                if m > 1:
                    factor = factor ** m
            else:
                # Spike-counting field (dn or dm), m >= 2: factorial
                # correction to remove same-bin shot-noise multiplets.
                raw = falling_factorial_array(n_arr, m) / denom**m
                factor = raw - raw.mean()

            product *= factor

    # ── Build the sweep time series on the valid window ──
    sweep_data = _data_for_ft(sweep_ft)
    sweep_arr = sweep_data[sweep_pop, valid_start:valid_end].copy()
    sweep_mean = mean_by_pop_ft[(sweep_pop, sweep_ft)]
    sweep_denom = dt_bin if _is_spike_ft(sweep_ft) else 1.0
    sweep_rate = (sweep_arr - sweep_mean) / sweep_denom

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

    # ── Cumulant subtraction for k >= 4 ──────────────────────────
    # For k=4, the connected cumulant is
    #   κ_4(X1, X2, X3, X4) = m_4 - m_2(1,2)m_2(3,4)
    #                              - m_2(1,3)m_2(2,4)
    #                              - m_2(1,4)m_2(2,3)
    # where m_n are CENTERED moments.  The cross-correlation above
    # gave us m_4(t) at each sweep lag.  We must subtract the three
    # disconnected pair products.
    #
    # For k <= 3, the cumulant equals the central moment; nothing to
    # subtract.  For k > 4, additional partition-based subtractions
    # are needed (not implemented here).
    if k == 4:
        # Identify all pair contractions.  The 6 pairs for 4 legs:
        # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # For each, we need m_2(i, j) at the appropriate lag.
        # m_2(i, j) = ⟨δX_i(t + lag_i) δX_j(t + lag_j)⟩
        #           = (cross-correlation of fields i and j at lag = lag_j - lag_i)
        # If i or j is the sweep axis, the lag involves τ_sweep and
        # we get a TIME-DEPENDENT m_2 (function of τ_sweep).
        # Otherwise it's a constant.

        # Pair partitions for k=4: {(0,1)(2,3), (0,2)(1,3), (0,3)(1,2)}
        partitions = [((0, 1), (2, 3)),
                      ((0, 2), (1, 3)),
                      ((0, 3), (1, 2))]

        def _two_point(leg_a, leg_b, lag_value):
            """Smooth 2-point moment of legs a, b at their relative lag.

            For consistency with the factorial-corrected 4-point construction
            that builds the grouped observable, this estimator is itself
            factorial-corrected when the pair is {same pop, same ft = 'dn',
            same effective lag}.  Ordinary linear centering is used in every
            other case (different lag → no shot-noise coincidence to remove;
            voltage or cross-pop or cross-ft → factorial doesn't apply).

            By stationarity the result equals the 2-point cross-correlation
            at lag delta = lb - la.
            """
            la = lag_bins[leg_a] if leg_a != sweep_idx else lag_value
            lb = lag_bins[leg_b] if leg_b != sweep_idx else lag_value
            pa = pop_indices[leg_a]
            pb = pop_indices[leg_b]
            fta = field_types[leg_a]
            ftb = field_types[leg_b]

            # ── Factorial-corrected same-pop same-ft same-lag spike pair ──
            # Only discrete-spike fields ('dn' / 'dm') have shot noise;
            # only same-pop same-bin coincidences produce a contact
            # diagonal that needs removing.
            if (_is_spike_ft(fta) and _is_spike_ft(ftb)
                    and fta == ftb and pa == pb and la == lb):
                arr = _data_for_ft(fta)[pa, :].astype(float)
                fact_rate_sq = arr * (arr - 1.0) / (dt_bin ** 2)
                mean_rate = mean_by_pop_ft[(pa, fta)] / dt_bin
                return float(fact_rate_sq.mean() - mean_rate * mean_rate)

            # ── Ordinary covariance for all other cases ──
            data_a = _data_for_ft(fta)
            data_b = _data_for_ft(ftb)
            mean_a = mean_by_pop_ft[(pa, fta)]
            mean_b = mean_by_pop_ft[(pb, ftb)]
            denom_a = dt_bin if _is_spike_ft(fta) else 1.0
            denom_b = dt_bin if _is_spike_ft(ftb) else 1.0

            # Effective lag difference: shift b relative to a.
            delta = lb - la
            if delta >= 0:
                # arr_a from bin 0, arr_b from bin delta
                n_pts = n_bins - delta
                if n_pts <= 0:
                    return 0.0
                arr_a_raw = data_a[pa, :n_pts]
                arr_b_raw = data_b[pb, delta:delta + n_pts]
            else:
                # arr_a from bin |delta|, arr_b from bin 0
                shift = -delta
                n_pts = n_bins - shift
                if n_pts <= 0:
                    return 0.0
                arr_a_raw = data_a[pa, shift:shift + n_pts]
                arr_b_raw = data_b[pb, :n_pts]

            arr_a = (arr_a_raw - mean_a) / denom_a
            arr_b = (arr_b_raw - mean_b) / denom_b
            return float(np.mean(arr_a * arr_b))

        # Compute disconnected piece for each lag in the sweep
        disc = np.zeros(2 * max_lag_bins + 1)
        for idx_lag, lag in enumerate(range(-max_lag_bins, max_lag_bins + 1)):
            n_overlap = n_valid - abs(lag)
            if n_overlap <= 0:
                continue
            for (a, b), (c, d) in partitions:
                m_ab = _two_point(a, b, lag)
                m_cd = _two_point(c, d, lag)
                disc[idx_lag] += m_ab * m_cd

        C_slice = C_slice - disc

    elif k > 4:
        import warnings
        warnings.warn(
            f'compute_kpoint_slice: cumulant subtraction not implemented '
            f'for k={k} > 4. Returning the centered k-point moment instead '
            f'of the connected cumulant.', RuntimeWarning)

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

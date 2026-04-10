"""
models/cumulant_direct_numba.py
===============================
Numba-JIT direct (non-FFT) 3-point cumulant estimator.

Plain .py file (not preparsed by SageMath) so Numba can compile it.
"""

import numpy as np
import numba


@numba.njit
def _kpoint_slice_direct_k3(counts_a, counts_b, counts_c,
                            mean_a, mean_b, mean_c,
                            fixed_lag, sweep_lags, dt_bin,
                            same_pop_ab, same_pop_ac, same_pop_bc,
                            which_fixed):
    """
    Direct k=3 cumulant slice estimator with factorial correction.

    Parameters
    ----------
    counts_a, counts_b, counts_c : 1D int arrays
        Raw spike counts per bin for fields a, b, c.
    mean_a, mean_b, mean_c : float
        Mean counts per bin.
    fixed_lag : int
        The lag (in bins) of the NON-sweep fixed field.
    sweep_lags : 1D int array
        Array of sweep lag values to evaluate.
    dt_bin : float
        Bin width.
    same_pop_ab, same_pop_ac, same_pop_bc : bool
        Whether pairs of fields are from the same population.
    which_fixed : int
        0 = field b is swept (c at fixed_lag)
        1 = field c is swept (b at fixed_lag)

    Returns
    -------
    C_slice : 1D float array, same length as sweep_lags
    """
    n_bins = len(counts_a)
    n_lags = len(sweep_lags)
    C_slice = np.zeros(n_lags)

    for idx in range(n_lags):
        sweep_lag = sweep_lags[idx]

        # Determine all three lags: a at 0, b at lag_b, c at lag_c
        if which_fixed == 0:
            lag_b = sweep_lag
            lag_c = fixed_lag
        else:
            lag_b = fixed_lag
            lag_c = sweep_lag

        # Valid time range: all three bins within [0, n_bins)
        all_lags_arr = np.array([0, lag_b, lag_c])
        min_l = all_lags_arr.min()
        max_l = all_lags_arr.max()
        t_start = max(0, -min_l)
        t_end = n_bins - max(0, max_l)
        n_overlap = t_end - t_start

        if n_overlap <= 0:
            C_slice[idx] = 0.0
            continue

        accum = 0.0
        for t in range(t_start, t_end):
            ta = t
            tb = t + lag_b
            tc = t + lag_c

            na = counts_a[ta]
            nb = counts_b[tb]
            nc = counts_c[tc]

            # Check which fields share the same bin AND same population
            # and apply factorial correction accordingly.

            # Default: centered rate for each
            fa = (na - mean_a) / dt_bin
            fb = (nb - mean_b) / dt_bin
            fc = (nc - mean_c) / dt_bin

            # Same-bin same-pop corrections:
            # If a and b are same pop AND same bin (lag_b == 0):
            if same_pop_ab and lag_b == 0:
                # Replace fa × fb with centered n(n-1)/dt^2
                n = na  # = nb since same bin same pop
                ff2 = n * (n - 1) / (dt_bin * dt_bin)
                # Mean of n(n-1) for Poisson with mean μ: μ² (+ small correction)
                ff2_mean = mean_a * (mean_a - 1) / (dt_bin * dt_bin)
                # Now check if c is ALSO same bin same pop
                if same_pop_ac and lag_c == 0:
                    # All three same bin same pop: use n(n-1)(n-2)/dt^3
                    ff3 = n * (n - 1) * (n - 2) / (dt_bin * dt_bin * dt_bin)
                    ff3_mean = mean_a * (mean_a - 1) * (mean_a - 2) / (dt_bin * dt_bin * dt_bin)
                    accum += (ff3 - ff3_mean)
                else:
                    accum += (ff2 - ff2_mean) * fc
            elif same_pop_ac and lag_c == 0:
                # a and c same bin same pop (but not b)
                n = na
                ff2 = n * (n - 1) / (dt_bin * dt_bin)
                ff2_mean = mean_a * (mean_a - 1) / (dt_bin * dt_bin)
                accum += (ff2 - ff2_mean) * fb
            elif same_pop_bc and lag_b == lag_c:
                # b and c same bin same pop (but not a)
                n = nb
                ff2 = n * (n - 1) / (dt_bin * dt_bin)
                ff2_mean = mean_b * (mean_b - 1) / (dt_bin * dt_bin)
                accum += fa * (ff2 - ff2_mean)
            else:
                # No same-bin same-pop pairs: ordinary centered product
                accum += fa * fb * fc

        C_slice[idx] = accum / n_overlap

    return C_slice

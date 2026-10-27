"""
tests/test_audit_wave1_estimators.py
====================================
Regression tests for the Oct 2026 audit's fix wave 1, simulation
estimators used for validation (``daedalus.simulations.cumulant_estimator``).

Reference: a Cox process whose intensity is a Gaussian OU process.  Its
factorial cumulant densities are the intensity's cumulants: C2(tau) =
sigma^2 exp(-|tau| / tau_c) and kappa_3 = 0 at every lag, while the old
estimator returned kappa_3 + 2 mu C2 at a coincident pair and the shot
noise rate / dt_bin at a coincident sweep lag.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.simulations.cumulant_estimator import (                           # noqa: E402
    compute_kpoint_slice, compute_kpoint_slice_direct, estimate_kpoint_slices)

LAM0, SIG, TC, DT, NB, L = 50.0, 10.0, 0.2, 0.01, 2_000_000, 20


@pytest.fixture(scope='module')
def cox_counts():
    from scipy.signal import lfilter
    rng = np.random.default_rng(7)
    a = np.exp(-DT / TC)
    xi = rng.standard_normal(NB) * SIG * np.sqrt(1 - a * a)
    x = lfilter([1.0], [1.0, -a], xi, zi=[a * rng.standard_normal() * SIG])[0]
    lam = np.clip(LAM0 + x, 0, None)
    return rng.poisson(lam * DT)[None, :].astype(float)


def test_k2_coincident_sweep_lag_has_no_shot_noise(cox_counts):
    taus, c2 = compute_kpoint_slice(cox_counts, DT, [0, 0], [0, None], L)
    ex = SIG ** 2 * np.exp(-np.abs(taus) / TC)
    # the old tau = 0 point carried rate / dt_bin = 5000 = 50 sigma^2
    assert abs(c2[L] - ex[L]) < 0.15 * SIG ** 2
    assert np.max(np.abs(c2 - ex)) < 0.25 * SIG ** 2


@pytest.mark.parametrize('lags', [[0, 0, None], [0, 3, None]])
def test_k3_factorial_cumulant_of_a_gaussian_cox_process_vanishes(cox_counts, lags):
    _, c3 = compute_kpoint_slice(cox_counts, DT, [0, 0, 0], lags, L)
    # the old coincident-pair estimate was ~ 2 mu C2(tau) (and a rate/dt^2
    # spike where the sweep leg met a fixed leg)
    assert np.max(np.abs(c3)) < 0.25 * 2 * LAM0 * SIG ** 2


def test_fft_and_direct_estimators_agree(cox_counts):
    # same window and means: identical to rounding; a shifted fixed leg
    # (window / mean of slightly different bins): 4e-5 measured
    for pops, lags, sc, tol in (([0, 0], [0, None], SIG ** 2, 1e-10),
                                ([0, 0, 0], [0, 0, None], 2 * LAM0 * SIG ** 2, 1e-10),
                                ([0, 0, 0], [0, -2, None], 2 * LAM0 * SIG ** 2, 1e-3)):
        _, f = compute_kpoint_slice(cox_counts, DT, pops, lags, L)
        _, d = compute_kpoint_slice_direct(cox_counts, DT, pops, lags, L)
        assert np.max(np.abs(f - d)) < tol * sc


def test_unrecognised_field_type_is_refused(cox_counts):
    with pytest.raises(ValueError, match="field type 'x'"):
        compute_kpoint_slice(cox_counts, DT, [0, 0], [0, None], 5, field_types=['x', 'x'])
    # natural names keep working
    estimate_kpoint_slices(DT, [0, 0], ['n', 'nE'], [0], 5, binned_counts=cox_counts)

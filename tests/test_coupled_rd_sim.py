"""
tests/test_coupled_rd_sim.py
============================
Validation of the N-species coupled reaction-diffusion Langevin simulator
(``models/coupled_rd_1d_sim.py``) against the EXACT linear box oracle
``coupled_box_correlator`` — the ground truth the coupled-field spatial
Dyson series will be tested against.

Comparison principle (same as the scalar-sim tests): the simulator
realizes the finite-difference LATTICE dispersion, so the statistical
3*C_err agreement bar is checked against the matched-cutoff oracle
(``dispersion='lattice'``, exact classical result for the discretized
system — zero approximation); the physical CONTINUUM oracle (the Dyson
ground truth) is checked at the <= 7% level on C_ii(0,0), where the two
dispersions differ by O(dx) (~4-6% at L=20, n_x=128).

Run:  sage -python -m pytest tests/test_coupled_rd_sim.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.coupled_rd_1d_sim import (
    simulate_coupled_rd_1d, coupled_box_correlator,
)

# ---- shared parameters (the 2-species validation point) -------------------
M = np.array([[1.5, 0.5], [0.3, 1.2]])
NNOISE = np.array([[1.0, 0.2], [0.2, 0.8]])
L, N_X = 20.0, 128
DX = L / N_X
TAUS = (0.0, 0.4)
X_IDX = (0, 3, 8)              # x = 0, 3*dx, 8*dx
SIM_KW = dict(L=L, n_x=N_X, dt=2e-3, t_burn=20.0, t_run=200.0,
              n_rep=4, seed=0, lags=TAUS)


def _run_case(D):
    sim = simulate_coupled_rd_1d(M, D, NNOISE, g=None, **SIM_KW)
    lat = coupled_box_correlator(M, D, NNOISE, L, N_X, TAUS,
                                 dispersion='lattice')
    con = coupled_box_correlator(M, D, NNOISE, L, N_X, TAUS)  # continuum
    return sim, lat, con


def _assert_agreement(sim, lat, con):
    """3*C_err vs the matched (lattice) oracle at all requested (tau, x),
    all ij; <= 7% relative vs the CONTINUUM oracle on C_ii(0,0)."""
    for k in range(len(TAUS)):
        for r in X_IDX:
            diff = np.abs(sim['C'][k, :, :, r] - lat['C'][k, :, :, r])
            bar = 3.0 * sim['C_err'][k, :, :, r]
            assert np.all(diff <= bar), (
                f'tau={TAUS[k]}, x={r * DX:.3f}: |sim-oracle|={diff} '
                f'exceeds 3*C_err={bar}')
    for i in range(2):
        rel = abs(sim['C'][0, i, i, 0] - con['C'][0, i, i, 0]) \
            / con['C'][0, i, i, 0]
        assert rel <= 0.07, (
            f'C_{i}{i}(0,0): sim {sim["C"][0, i, i, 0]:.5f} vs continuum '
            f'{con["C"][0, i, i, 0]:.5f} ({rel:.1%} > 7%)')


@pytest.fixture(scope='module')
def case_equal_d():
    return _run_case(0.7)                       # scalar D


@pytest.fixture(scope='module')
def case_unequal_d():
    return _run_case([0.9, 0.3])                # per-species D


# ---------------------------------------------------------------------------
# 1. linear, equal (scalar) diffusion
# ---------------------------------------------------------------------------

def test_linear_equal_d_matches_box_oracle(case_equal_d):
    sim, lat, con = case_equal_d
    _assert_agreement(sim, lat, con)


def test_box_oracle_consistent_with_spectral_propagator(case_equal_d):
    """For SCALAR diffusion the continuum box oracle must reproduce the
    framework's free coupled 2-point (spectral_propagator.coupled_two_point)
    mode-summed over the box — analytic vs analytic, ~machine precision."""
    from msrjd.integration.spatial.spectral_propagator import (
        build_reference, coupled_two_point,
    )
    _sim, _lat, con = case_equal_d
    ref = build_reference(M, 0.7 * np.eye(2))
    q = 2.0 * np.pi * np.fft.fftfreq(N_X, d=DX)
    for k, tau in enumerate(TAUS):
        C0 = sum(coupled_two_point(ref, NNOISE, qn ** 2, tau) for qn in q) / L
        assert np.max(np.abs(C0 - con['C'][k, :, :, 0])) < 1e-10


def test_equal_time_correlator_is_symmetric(case_equal_d):
    """C(x=0, tau=0) is a covariance matrix: symmetric within error bars."""
    sim, _lat, _con = case_equal_d
    asym = abs(sim['C'][0, 0, 1, 0] - sim['C'][0, 1, 0, 0])
    err = 3.0 * np.hypot(sim['C_err'][0, 0, 1, 0], sim['C_err'][0, 1, 0, 0])
    assert asym <= err


def test_output_shapes_and_grids(case_equal_d):
    sim, _lat, con = case_equal_d
    assert sim['C'].shape == (2, 2, 2, N_X)
    assert sim['C_err'].shape == (2, 2, 2, N_X)
    assert sim['C_rep'].shape == (4, 2, 2, 2, N_X)
    assert con['C'].shape == (2, 2, 2, N_X)
    assert np.allclose(sim['x_grid'], np.arange(N_X) * DX)
    assert np.allclose(sim['taus'], TAUS)


# ---------------------------------------------------------------------------
# 2. linear, UNEQUAL per-species diffusion (the Dyson-validation target)
# ---------------------------------------------------------------------------

def test_linear_unequal_d_matches_box_oracle(case_unequal_d):
    sim, lat, con = case_unequal_d
    _assert_agreement(sim, lat, con)


def test_unequal_d_oracle_solves_lyapunov_per_mode():
    """The helper's per-mode covariance solves A Sigma + Sigma A^T = Nnoise
    with the FULL A(q) = M + diag(D) q^2 (unequal D exact, no scalar-D
    approximation): check via the tau=0, x-summed projection onto q=0."""
    Dvec = np.array([0.9, 0.3])
    out = coupled_box_correlator(M, Dvec, NNOISE, L, N_X, (0.0,))
    # sum_x C(x)*dx isolates the q=0 mode: = Sigma(q=0) solving M S + S M^T = N
    S0 = out['C'][0].sum(axis=-1) * DX
    res = M @ S0 + S0 @ M.T - NNOISE
    assert np.max(np.abs(res)) < 1e-10


# ---------------------------------------------------------------------------
# 3. nonlinear smoke: stabilizing cubic suppresses the variance
# ---------------------------------------------------------------------------

def test_cubic_nonlinearity_suppresses_variance():
    """g phi^3 (stabilizing) must pull C_ii(0,0) DOWN vs the linear run.
    Same seed => common random numbers, so the per-replica PAIRED
    difference isolates the nonlinear shift with tiny variance."""
    kw = dict(L=L, n_x=N_X, dt=2e-3, t_burn=20.0, t_run=100.0,
              n_rep=4, seed=0, lags=(0.0,), x_lags=0.0)
    lin = simulate_coupled_rd_1d(M, 0.7, NNOISE, g=None, **kw)
    nl = simulate_coupled_rd_1d(M, 0.7, NNOISE, g=(0.1, 0.1), **kw)
    assert lin['C'].shape == (1, 2, 2)          # scalar-x squeeze contract
    assert np.all(np.isfinite(nl['C']))
    for i in range(2):
        d_rep = lin['C_rep'][:, 0, i, i] - nl['C_rep'][:, 0, i, i]
        shift, se = d_rep.mean(), d_rep.std(ddof=1) / np.sqrt(len(d_rep))
        assert shift > 0.0, f'species {i}: cubic did not suppress variance'
        assert shift > 3.0 * se, (
            f'species {i}: suppression {shift:.5f} not beyond error '
            f'bars (3*SE = {3 * se:.5f})')

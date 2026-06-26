"""
tests/test_coupled_two_point.py
===============================
Dyson step 3a: the spectral-Lyapunov tree 2-point for COUPLED fields
(``spectral_propagator.lyapunov_covariance`` / ``coupled_two_point``).

A coupled theory's free 2-point is the matrix Lyapunov/FDT object
``C(q,τ)=e^{−A(q)|τ|}Σ(q)``, ``A(q)=M+D₀q²I``, ``A Σ+Σ Aᵀ=N`` — NOT the diagonal
independent-mode sum.  Validated against:
  * the Lyapunov residual (``A Σ + Σ Aᵀ − N ≈ 0``);
  * the diagonal independent-mode reproduction (so the coupled form is a strict
    generalization of pipeline_bridge._modes_C_q_tau);
  * the fluctuation-regression identity ``C(q,τ)=e^{−Aτ}Σ``, ``C(q,0)=Σ``,
    ``C(q,−τ)=C(q,τ)ᵀ``;
  * a direct **2-species OU Langevin simulation** (the physics oracle);
  * rejection of the unequal-diffusion case (needs the Dyson series).

Run:  sage -python -m pytest tests/test_coupled_two_point.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.linalg import expm                                   # noqa: E402

from engine.integration.spatial.spectral_propagator import (     # noqa: E402
    build_reference, lyapunov_covariance, coupled_two_point,
)


def test_lyapunov_residual():
    A = np.array([[1.5, 0.5], [0.3, 1.2]])
    N = np.array([[1.0, 0.2], [0.2, 0.8]])
    Sig = lyapunov_covariance(A, N)
    assert np.allclose(A @ Sig + Sig @ A.T, N, atol=1e-10)
    assert np.allclose(Sig, Sig.T, atol=1e-10)                  # covariance symmetric


@pytest.mark.parametrize('qsq,tau', [(0.0, 0.0), (1.3, 0.4), (3.0, 1.2)])
def test_diagonal_reproduces_independent_mode_sum(qsq, tau):
    """Diagonal M + scalar D + diagonal N=2·diag(κ) ⇒ C diagonal and
    C_ii(q,τ)=κ_i/(μ_i+D q²)·e^{−(μ_i+D q²)|τ|} (the _modes_C_q_tau form)."""
    mu = np.array([1.0, 2.0])
    D = 0.5
    kap = np.array([0.7, 0.4])
    ref = build_reference(np.diag(mu), D * np.eye(2))
    C = coupled_two_point(ref, np.diag(2 * kap), qsq, tau)
    lam = mu + D * qsq
    expected = np.diag(kap / lam * np.exp(-lam * abs(tau)))
    assert np.allclose(C, expected, atol=1e-10)


def test_regression_identity_and_symmetry():
    M = np.array([[1.4, 0.6], [0.2, 1.1]])
    N = np.array([[1.0, 0.3], [0.3, 0.9]])
    ref = build_reference(M, 0.7 * np.eye(2))
    qsq = 1.0
    A = M + 0.7 * qsq * np.eye(2)
    Sig = lyapunov_covariance(A, N)
    assert np.allclose(coupled_two_point(ref, N, qsq, 0.0), Sig, atol=1e-10)
    for tau in (0.5, 1.5):
        C = coupled_two_point(ref, N, qsq, tau)
        assert np.allclose(C, expm(-A * tau) @ Sig, atol=1e-10)
        # C(q,−τ) = C(q,τ)ᵀ
        assert np.allclose(coupled_two_point(ref, N, qsq, -tau), C.T, atol=1e-10)
    # genuinely coupled ⇒ nonzero cross-correlation
    assert abs(coupled_two_point(ref, N, qsq, 0.4)[0, 1]) > 1e-3


def test_coupled_two_point_vs_ou_simulation():
    """2-species OU Langevin sim (q=0): empirical C_ij(τ) matches the Lyapunov
    prediction e^{−Mτ}Σ, including the off-diagonal cross-correlation."""
    M = np.array([[1.5, 0.5], [0.3, 1.2]])
    N = np.array([[1.0, 0.2], [0.2, 0.8]])
    ref = build_reference(M, 1.0 * np.eye(2))            # D0 irrelevant at q=0
    Sig = lyapunov_covariance(M, N)

    rng = np.random.default_rng(20260609)
    L = np.linalg.cholesky(N)
    dt, n_steps, burn = 0.02, 200_000, 5_000
    sqdt = np.sqrt(dt)
    x = np.zeros(2)
    for _ in range(burn):
        x = x - (M @ x) * dt + L @ (rng.standard_normal(2) * sqdt)
    X = np.empty((n_steps, 2))
    for s in range(n_steps):
        x = x - (M @ x) * dt + L @ (rng.standard_normal(2) * sqdt)
        X[s] = x

    for tau in (0.0, 0.3, 0.8):
        lag = int(round(tau / dt))
        Cemp = (X.T @ X) / n_steps if lag == 0 else (X[lag:].T @ X[:-lag]) / (n_steps - lag)
        Cthy = coupled_two_point(ref, N, 0.0, tau)       # = e^{−Mτ}Σ
        assert np.allclose(Cthy, expm(-M * tau) @ Sig, atol=1e-12)
        assert np.allclose(Cemp, Cthy, atol=0.04), \
            f'τ={tau}: sim\n{Cemp}\n vs theory\n{Cthy}'


def test_unequal_diffusion_rejected():
    ref = build_reference(np.array([[1.0, 0.4], [0.3, 1.6]]),
                          np.diag([0.9, 0.3]))             # 𝒟̂ ≠ 0
    assert not ref.is_scalar_diffusion
    with pytest.raises(ValueError, match='scalar diffusion'):
        coupled_two_point(ref, np.eye(2), 1.0, 0.5)

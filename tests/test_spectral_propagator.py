"""
tests/test_spectral_propagator.py
=================================
Step 1 of the Dyson–Duhamel integration: validate the spectral coupled-field
reference propagator ``G₀`` (``msrjd.integration.spatial.spectral_propagator``).

  * ``split_reference_diffusion`` / ``spectral_projectors`` algebra
    (𝒟 = D₀I + 𝒟̂; Σ P_α = I, P_α P_β = δ P_α, Σ m_α P_α = M);
  * **diagonal reproduction**: diagonal M + scalar 𝒟 ⇒ G₀ is the per-field scalar
    heat kernel e^{−(μ_i + D|k|²)t} (what heat_kernel.py builds today);
  * **expm oracle**: coupled M + scalar 𝒟 (𝒟̂=0) ⇒ G₀ == e^{−(M+D|k|²)t} EXACTLY;
  * **Dyson gap**: coupled M + non-scalar 𝒟 (𝒟̂≠0) ⇒ G₀ (n=0) ≠ the true
    e^{−(M+𝒟|k|²)t} (the n≥1 Dyson corrections, added in step 3, close the gap);
  * defective M raises a clear error (resolvent form deferred).

Run:  sage -python -m pytest tests/test_spectral_propagator.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.linalg import expm                                  # noqa: E402

from msrjd.integration.spatial.spectral_propagator import (    # noqa: E402
    split_reference_diffusion, spectral_projectors, build_reference,
)


def test_split_reference_diffusion():
    D = np.array([[0.8, 0.1], [0.0, 1.2]])
    D0, Dhat = split_reference_diffusion(D)
    assert D0 == pytest.approx((0.8 + 1.2) / 2)         # trace/N
    assert np.allclose(Dhat, D - D0 * np.eye(2))
    # scalar diffusion ⇒ residual exactly zero
    D0s, Dhats = split_reference_diffusion(0.5 * np.eye(3))
    assert D0s == pytest.approx(0.5) and np.allclose(Dhats, 0.0)
    # explicit override
    D0o, _ = split_reference_diffusion(D, D0=0.3)
    assert D0o == pytest.approx(0.3)


def test_spectral_projector_algebra():
    M = np.array([[1.0, 0.3], [0.2, 1.7]])
    w, P = spectral_projectors(M)
    n = M.shape[0]
    assert np.allclose(sum(P), np.eye(n))                       # Σ P_α = I
    assert np.allclose(sum(m * Pa for m, Pa in zip(w, P)), M)   # Σ m_α P_α = M
    for a in range(n):
        for b in range(n):
            prod = P[a] @ P[b]
            assert np.allclose(prod, P[a] if a == b else np.zeros_like(prod))


@pytest.mark.parametrize('ksq,t', [(0.0, 0.7), (1.3, 0.4), (4.0, 1.1)])
def test_diagonal_reproduces_scalar_heat_kernel(ksq, t):
    """Diagonal M + scalar 𝒟 = D·I ⇒ G₀ diagonal = e^{−(μ_i + D·ksq)t}."""
    mu = np.array([1.0, 2.5])
    D = 0.5
    ref = build_reference(np.diag(mu), D * np.eye(2))
    assert ref.is_scalar_diffusion
    G0 = ref.G0(ksq, t)
    expected = np.diag(np.exp(-(mu + D * ksq) * t))
    assert np.allclose(G0, expected, atol=1e-12)


@pytest.mark.parametrize('ksq,t', [(0.0, 0.5), (2.0, 0.9)])
def test_reference_equals_expm_for_scalar_diffusion(ksq, t):
    """Coupled (non-diagonal) M + scalar 𝒟 (𝒟̂=0) ⇒ G₀ == e^{−(M+D·ksq·I)t}
    EXACTLY (n=0 is the full propagator; no Dyson series needed)."""
    M = np.array([[1.0, 0.3], [0.2, 1.5]])
    D = 0.7
    ref = build_reference(M, D * np.eye(2))
    assert ref.is_scalar_diffusion
    G0 = ref.G0(ksq, t)
    oracle = expm(-(M + D * ksq * np.eye(2)) * t)               # full matrix heat kernel
    assert np.allclose(G0, oracle, atol=1e-10)
    assert np.allclose(G0.imag, 0.0, atol=1e-10)                # real M ⇒ real G₀


def test_dhat_nonzero_is_n0_approximation():
    """Coupled M + NON-scalar 𝒟 (𝒟̂≠0): G₀ (n=0) ≠ the true e^{−(M+𝒟·ksq)t}.
    The gap is the n≥1 Dyson correction (step 3).  G₀ must still equal
    e^{−(M+D₀·ksq·I)t} (its own definition)."""
    M = np.array([[1.0, 0.4], [0.3, 1.6]])
    D = np.array([[0.9, 0.0], [0.0, 0.3]])                      # unequal ⇒ 𝒟̂≠0
    ref = build_reference(M, D)
    assert not ref.is_scalar_diffusion
    ksq, t = 2.0, 0.8
    G0 = ref.G0(ksq, t)
    # G₀ is exactly the scalar-D₀ reference propagator:
    assert np.allclose(G0, expm(-(M + ref.D0 * ksq * np.eye(2)) * t), atol=1e-10)
    # ...but NOT the true full (𝒟̂≠0) propagator — that needs the Dyson series:
    full = expm(-(M + D * ksq) * t)
    assert not np.allclose(G0, full, atol=1e-3)


def test_defective_matrix_raises():
    """A defective (non-diagonalizable) M is rejected with a clear message."""
    J = np.array([[2.0, 1.0], [0.0, 2.0]])                      # Jordan block
    with pytest.raises(ValueError, match='defective'):
        spectral_projectors(J)

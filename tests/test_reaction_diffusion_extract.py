"""
tests/test_reaction_diffusion_extract.py
=========================================
Dyson step 1, wiring increment 1: extract the coupled reaction matrix ``M`` and
diffusion matrix ``𝒟`` from the symbolic inverse propagator ``K_ft``
(``heat_kernel.reaction_diffusion_matrices``), and bridge to the spectral
reference propagator (``spectral_propagator.build_reference``).

  * diagonal K_ft ⇒ M/𝒟/V are diagonal and per-entry == extract_mass_diffusion
    (the current Tier-1 extraction — reproduction guarantee);
  * coupled K_ft ⇒ off-diagonal cross-reaction / cross-diffusion captured;
  * off-diagonal ω term and higher-derivative (k⁴) kernels rejected;
  * FULL step-1 chain: coupled K_ft with scalar diffusion → numeric M,𝒟 →
    G₀ == e^{−(M+D|k|²)t} (scipy expm oracle).

Run:  sage -python -m pytest tests/test_reaction_diffusion_extract.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, var, matrix, I as SR_I                 # noqa: E402
from scipy.linalg import expm                                   # noqa: E402

from engine.integration.spatial.heat_kernel import (             # noqa: E402
    reaction_diffusion_matrices, extract_mass_diffusion, SpatialPropagatorError,
)
from engine.integration.spatial.spectral_propagator import build_reference  # noqa: E402

omega = var('omega')
k = var('k')
Lap = var('Laplacian')
mu1, mu2, D1, D2, g, h = var('mu1 mu2 D1 D2 g h')


def _diag_kft():
    """Two independent Allen-Cahn modes: i·ω + μ_i − D_i·Lap."""
    return matrix(SR, [[SR_I * omega + mu1 - D1 * Lap, 0],
                       [0, SR_I * omega + mu2 - D2 * Lap]])


def _coupled_kft(cross_reaction, cross_diff):
    off = cross_reaction - cross_diff * Lap
    return matrix(SR, [[SR_I * omega + mu1 - D1 * Lap, off],
                       [off, SR_I * omega + mu2 - D2 * Lap]])


def test_diagonal_matches_extract_mass_diffusion():
    K = _diag_kft()
    M, D, V = reaction_diffusion_matrices(K, omega, k, Lap)
    for i in range(2):
        A, B, Vv = extract_mass_diffusion(K[i, i], omega, k, Lap)
        assert (M[i, i] - A).is_zero()
        assert (D[i, i] - B).is_zero()
        assert (V[i, i] - Vv).is_zero()
    assert M[0, 1].is_zero() and D[0, 1].is_zero() and V[0, 1].is_zero()
    assert M[1, 0].is_zero() and D[1, 0].is_zero()


def test_coupled_extraction():
    K = _coupled_kft(cross_reaction=g, cross_diff=h)
    M, D, V = reaction_diffusion_matrices(K, omega, k, Lap)
    assert (M[0, 0] - mu1).is_zero() and (M[1, 1] - mu2).is_zero()
    assert (M[0, 1] - g).is_zero() and (M[1, 0] - g).is_zero()    # cross reaction
    assert (D[0, 0] - D1).is_zero() and (D[1, 1] - D2).is_zero()
    assert (D[0, 1] - h).is_zero() and (D[1, 0] - h).is_zero()    # cross diffusion
    assert all(V[i, j].is_zero() for i in range(2) for j in range(2))


def test_offdiagonal_omega_rejected():
    K = matrix(SR, [[SR_I * omega + mu1, SR_I * omega + g],
                    [0, SR_I * omega + mu2]])
    with pytest.raises(SpatialPropagatorError, match='expected 0'):
        reaction_diffusion_matrices(K, omega, k, Lap)


def test_higher_derivative_rejected():
    K = matrix(SR, [[SR_I * omega + mu1 + D1 * Lap**2, 0],
                    [0, SR_I * omega + mu2]])                     # Lap² → k⁴
    with pytest.raises(SpatialPropagatorError, match='k-power'):
        reaction_diffusion_matrices(K, omega, k, Lap)


def test_kft_to_spectral_scalar_diffusion_matches_expm():
    """Full step-1 chain: coupled K_ft with SCALAR diffusion (D1=D2, no cross-
    diffusion ⇒ 𝒟̂=0) → symbolic M,𝒟 → numeric → spectral G₀ exactly equals
    e^{−(M + D|k|²)t} (the matrix-heat-kernel oracle)."""
    K = _coupled_kft(cross_reaction=g, cross_diff=0)
    M, D, _ = reaction_diffusion_matrices(K, omega, k, Lap)
    params = {mu1: 1.0, mu2: 1.5, D1: 0.7, D2: 0.7, g: 0.3}       # D1=D2 ⇒ scalar
    Mn = np.array([[float(M[i, j].subs(params)) for j in range(2)]
                   for i in range(2)])
    Dn = np.array([[float(D[i, j].subs(params)) for j in range(2)]
                   for i in range(2)])
    ref = build_reference(Mn, Dn)
    assert ref.is_scalar_diffusion
    ksq, t = 2.0, 0.6
    assert np.allclose(ref.G0(ksq, t), expm(-(Mn + Dn * ksq) * t), atol=1e-10)

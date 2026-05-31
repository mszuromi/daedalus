"""
tests/test_loop_parametric.py
=============================
The momentum-FIRST parametric loop integrator core (Stage C.5 pivot,
``msrjd.integration.spatial.loop_parametric``).

The whole point of the pivot: doing the loop-momentum integral analytically
(Schwinger + Gaussian) reproduces the direct ``∫dℓ`` with NO momentum-dependent
poles, so the ``m≥3`` close-pair slow path that blocks the time-first
integrator can never arise.  These tests pin that the parametric route matches
a brute-force ``∫dℓ`` for both bubble self-energies.

Run:  sage -python -m pytest tests/test_loop_parametric.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.spatial.loop_parametric import (
    gaussian_momentum_integral, sigma_R_kernel, sigma_K_kernel, symanzik_UF,
)

MU = D = T = 1.0


def _m(k):
    return MU + D * k * k


def test_gaussian_momentum_integral_matches_direct():
    """∫dℓ/2π exp(-D[ℓ²w1 + (q-ℓ)²w2]) — closed form (Symanzik) vs quad."""
    for q in (0.0, 0.7, 1.5):
        for (w1, w2) in ((0.3, 0.8), (1.0, 1.0), (2.0, 0.5)):
            closed = gaussian_momentum_integral([1.0, -1.0], [0.0, 1.0],
                                                [w1, w2], q, D, spatial_dim=1)
            f = lambda l: math.exp(-D * (l * l * w1 + (q - l) ** 2 * w2))
            ref, _ = integrate.quad(f, -np.inf, np.inf, limit=200)
            ref /= 2 * np.pi
            assert abs(closed - ref) <= 1e-9 * max(abs(ref), 1e-12)


def test_symanzik_UF_bubble_reduction():
    """The extracted Symanzik core reproduces the known 1-loop bubble polynomials
    (backend_C_math §6): edges a=[1,−1], b=[0,1] ⇒ U=w1+w2,
    F_reduced=w1w2/(w1+w2); and gaussian_momentum_integral is its thin wrapper.
    This is the L=1 reference the C0 matrix `symanzik_polynomials` must reduce to."""
    for (w1, w2) in ((0.3, 0.7), (1.0, 1.0), (2.0, 0.5)):
        U, F_reduced, pref = symanzik_UF([1.0, -1.0], [0.0, 1.0], [w1, w2], D)
        assert abs(U - (w1 + w2)) <= 1e-12
        assert abs(F_reduced - w1 * w2 / (w1 + w2)) <= 1e-12
        assert abs(pref - (4.0 * math.pi * D * U) ** -0.5) <= 1e-12
        # wrapper consistency
        for q in (0.0, 0.9):
            assert abs(gaussian_momentum_integral([1.0, -1.0], [0.0, 1.0],
                       [w1, w2], q, D) - pref * math.exp(-D * q * q * F_reduced)
                       ) <= 1e-14


def test_gaussian_momentum_integral_zero_U_raises():
    with pytest.raises(ValueError, match='U'):
        gaussian_momentum_integral([0.0, 0.0], [1.0, 1.0], [1.0, 1.0],
                                   0.5, D)


@pytest.mark.parametrize('q', [0.0, 0.8, 1.5, 3.0])
@pytest.mark.parametrize('t', [0.2, 0.6, 1.2])
def test_sigma_R_kernel_matches_direct(q, t):
    """Σ_R kernel ∫dℓ/2π G_R(ℓ,t) C(q-ℓ,t): parametric vs direct ∫dℓ."""
    got = sigma_R_kernel(q, t, MU, D, T)
    f = lambda l: math.exp(-_m(l) * t) * (T / _m(q - l)) * math.exp(-_m(q - l) * t)
    ref, _ = integrate.quad(f, -np.inf, np.inf, limit=200)
    ref /= 2 * np.pi
    assert abs(got - ref) <= 1e-7 * max(abs(ref), 1e-12)


@pytest.mark.parametrize('q', [0.0, 0.8, 1.5])
@pytest.mark.parametrize('t', [0.3, 0.8])
def test_sigma_K_kernel_matches_direct(q, t):
    """Σ_K kernel ∫dℓ/2π C(ℓ,t) C(q-ℓ,t): parametric (2-D Schwinger) vs direct."""
    got = sigma_K_kernel(q, t, MU, D, T)
    f = lambda l: (T / _m(l)) * math.exp(-_m(l) * abs(t)) * \
        (T / _m(q - l)) * math.exp(-_m(q - l) * abs(t))
    ref, _ = integrate.quad(f, -np.inf, np.inf, limit=200)
    ref /= 2 * np.pi
    assert abs(got - ref) <= 1e-7 * max(abs(ref), 1e-12)


def test_equal_time_sigma_R_is_phi2_0_q_independent():
    """Σ_R(q,0⁺) = ⟨φ²⟩₀ = T/(2√(μD)), q-independent."""
    phi2_0 = T / (2 * math.sqrt(MU * D))
    for q in (0.0, 1.0, 3.0):
        s = sigma_R_kernel(q, 1e-7, MU, D, T)
        assert abs(s - phi2_0) <= 1e-3 * phi2_0

"""
tests/test_spatial_reduce.py
============================
Backend C — C0 (graph→Symanzik) and C1 (L-loop momentum integral),
``engine.integration.spatial.spatial_reduce``.

Validates:
  * C0: the Symanzik polynomials U, Q_eff reproduce the hand formulas of
    docs/backend_C_math.md §6 — the 1-loop bubble and the 2-loop sunset.
  * C1: the momentum integral reduces to loop_parametric.gaussian_momentum_integral
    at L=1, and matches an INDEPENDENT brute-force ∫dℓ₁dℓ₂ (the W11 sunset oracle)
    at L=2.

Run:  sage -python -m pytest tests/test_spatial_reduce.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.integration.spatial.spatial_reduce import (
    symanzik_polynomials, symanzik_matrices, momentum_integral,
)
from engine.integration.spatial.loop_parametric import gaussian_momentum_integral

D = 1.3

# routing coefficients (a over loop momenta, b over the single external q):
_BUBBLE_A = [(1.0,), (-1.0,)]        # k = ℓ ; k = q − ℓ
_BUBBLE_B = [(0.0,), (1.0,)]
_SUNSET_A = [(1.0, 0.0), (0.0, 1.0), (-1.0, -1.0)]   # ℓ₁ ; ℓ₂ ; q−ℓ₁−ℓ₂
_SUNSET_B = [(0.0,), (0.0,), (1.0,)]


# ── C0 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('w1,w2', [(0.3, 0.7), (1.0, 1.0), (2.0, 0.5)])
def test_c0_bubble_symanzik(w1, w2):
    """1-loop bubble: U=w₁+w₂, Q_eff=w₁w₂/(w₁+w₂) (math §6)."""
    U, Qeff = symanzik_polynomials(_BUBBLE_A, _BUBBLE_B, [w1, w2])
    assert abs(U - (w1 + w2)) <= 1e-12
    assert Qeff.shape == (1, 1)
    assert abs(Qeff[0, 0] - w1 * w2 / (w1 + w2)) <= 1e-12


@pytest.mark.parametrize('w', [(0.3, 0.7, 0.5), (1.0, 1.0, 1.0), (2.0, 0.4, 1.3)])
def test_c0_sunset_symanzik(w):
    """2-loop sunset: U=w₁w₂+w₂w₃+w₃w₁, Q_eff=w₁w₂w₃/U (math §6)."""
    w1, w2, w3 = w
    U, Qeff = symanzik_polynomials(_SUNSET_A, _SUNSET_B, [w1, w2, w3])
    assert abs(U - (w1 * w2 + w2 * w3 + w3 * w1)) <= 1e-12
    assert abs(Qeff[0, 0] - (w1 * w2 * w3) / U) <= 1e-12


def test_c0_matrices_shapes():
    Lam, N, Q = symanzik_matrices(_SUNSET_A, _SUNSET_B, [0.5, 0.6, 0.7])
    assert Lam.shape == (2, 2) and N.shape == (2, 1) and Q.shape == (1, 1)
    assert np.allclose(Lam, Lam.T)                     # Lam symmetric


def test_c0_zero_weights_raises():
    with pytest.raises(ValueError, match='U'):
        symanzik_polynomials(_BUBBLE_A, _BUBBLE_B, [0.0, 0.0])


# ── C1 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('q', [0.0, 0.7, 1.5])
@pytest.mark.parametrize('w1,w2', [(0.3, 0.8), (1.0, 1.0), (2.0, 0.5)])
def test_c1_l1_matches_gaussian_momentum_integral(q, w1, w2):
    """C1 at L=1 reduces EXACTLY to the validated 1-loop core."""
    got = momentum_integral(_BUBBLE_A, _BUBBLE_B, [w1, w2], q, D, spatial_dim=1)
    ref = gaussian_momentum_integral([1.0, -1.0], [0.0, 1.0], [w1, w2], q, D,
                                     spatial_dim=1)
    assert abs(got - ref) <= 1e-12 * max(abs(ref), 1e-300)


@pytest.mark.parametrize('q', [0.0, 0.6, 1.4])
@pytest.mark.parametrize('w', [(0.5, 0.6, 0.7), (1.0, 1.0, 1.0)])
def test_c1_l2_sunset_matches_brute_force(q, w):
    """C1 at L=2 (sunset) vs an INDEPENDENT brute-force ∫dℓ₁dℓ₂/(2π)² of
    exp(−D[w₁ℓ₁² + w₂ℓ₂² + w₃(q−ℓ₁−ℓ₂)²]) — the W11 2-loop momentum oracle."""
    w1, w2, w3 = w
    got = momentum_integral(_SUNSET_A, _SUNSET_B, [w1, w2, w3], q, D,
                            spatial_dim=1)

    def integrand(l2, l1):
        return math.exp(-D * (w1 * l1 ** 2 + w2 * l2 ** 2
                              + w3 * (q - l1 - l2) ** 2))
    ref, _ = integrate.dblquad(integrand, -np.inf, np.inf,
                               lambda _l1: -np.inf, lambda _l1: np.inf)
    ref /= (2 * np.pi) ** 2
    assert abs(got - ref) <= 1e-6 * max(abs(ref), 1e-300)


# ── C1 in higher dimension (the capability backend B's 1-D ∫dℓ lacks) ──
@pytest.mark.parametrize('q', [0.0, 0.8, 1.6])
@pytest.mark.parametrize('w1,w2', [(0.4, 0.9), (1.0, 1.0)])
def test_c1_d2_matches_brute_force(q, w1, w2):
    """C1 at d=2 vs an INDEPENDENT brute-force ∫d²ℓ/(2π)² (external momentum along
    x: q⃗=(q,0)).  The d-dim loop integral is CLOSED-FORM (Symanzik U^{−d/2}), not
    numerical angular quadrature — this is the d>1 reach backend B's 1-D line
    integral does not have."""
    got = momentum_integral(_BUBBLE_A, _BUBBLE_B, [w1, w2], q, D, spatial_dim=2)

    def integrand(ly, lx):
        return math.exp(-D * (w1 * (lx ** 2 + ly ** 2)
                              + w2 * ((q - lx) ** 2 + ly ** 2)))
    ref, _ = integrate.dblquad(integrand, -np.inf, np.inf,
                               lambda _lx: -np.inf, lambda _lx: np.inf)
    ref /= (2 * np.pi) ** 2
    assert abs(got - ref) <= 1e-6 * max(abs(ref), 1e-300)


@pytest.mark.parametrize('d', [1, 2, 3, 4])
@pytest.mark.parametrize('q', [0.0, 1.1])
def test_c1_dimension_factorization(d, q):
    """For an isotropic model the d-dim Gaussian factorizes over spatial
    components: I_d(q) = I_1(q)·I_1(0)^{d−1} (the external momentum lives along one
    axis; the other d−1 axes see q=0).  Confirms the U^{−d/2}/(4πD)^{−Ld/2}
    exponent is right for ANY d, so d is a parameter, not a re-derivation."""
    w = [0.7, 1.3]
    got = momentum_integral(_BUBBLE_A, _BUBBLE_B, w, q, D, spatial_dim=d)
    i1q = momentum_integral(_BUBBLE_A, _BUBBLE_B, w, q, D, spatial_dim=1)
    i10 = momentum_integral(_BUBBLE_A, _BUBBLE_B, w, 0.0, D, spatial_dim=1)
    assert abs(got - i1q * i10 ** (d - 1)) <= 1e-12 * max(abs(got), 1e-300)

"""
tests/test_temporal_integrate.py
================================
Backend C — C2 (causal time-simplex), ``msrjd.integration.spatial.temporal_integrate``.

Validates the parametric self-energy assembly Σ(q,t) for 2-vertex diagrams:
  * 1-loop bubble Σ_R / Σ_K vs backend B (loop_parametric.sigma_R/K_kernel, itself
    pinned vs direct ∫dℓ) — the III.0 oracle;
  * 2-loop sunset at t=0 vs a direct ∫dℓ₁dℓ₂ of C(ℓ₁)C(ℓ₂)C(q−ℓ₁−ℓ₂).

Run:  sage -python -m pytest tests/test_temporal_integrate.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.spatial.temporal_integrate import (
    sigma_parametric, bubble_edges, sunset_edges, bubble_delta_equal_time_via_C,
)
from msrjd.integration.spatial.loop_parametric import (
    sigma_R_kernel, sigma_K_kernel,
)
from msrjd.integration.spatial.loop_dyson import bubble_delta_S

MU = D = T = 1.0


def _m(k):
    return MU + D * k * k


# ── 1-loop bubble: C2 parametric route vs backend B ───────────────
@pytest.mark.parametrize('q', [0.0, 0.8, 1.5])
@pytest.mark.parametrize('t', [0.2, 0.6, 1.2])
def test_c2_bubble_sigma_R_matches_backend_B(q, t):
    """Σ_R(q,t) via C2 (sigma_parametric, general momentum_integral core) equals
    backend B's sigma_R_kernel (the validated 1-loop reference)."""
    got = sigma_parametric(bubble_edges('R'), q, t, MU, D, T)
    ref = sigma_R_kernel(q, t, MU, D, T)
    assert abs(got - ref) <= 1e-6 * max(abs(ref), 1e-12)


@pytest.mark.parametrize('q', [0.0, 0.8, 1.5])
@pytest.mark.parametrize('t', [0.3, 0.8])
def test_c2_bubble_sigma_K_matches_backend_B(q, t):
    """Σ_K(q,t) via C2 (both edges correlation) equals backend B's sigma_K_kernel."""
    got = sigma_parametric(bubble_edges('C'), q, t, MU, D, T)
    ref = sigma_K_kernel(q, t, MU, D, T)
    assert abs(got - ref) <= 1e-5 * max(abs(ref), 1e-12)


# ── 2-loop sunset at t=0: C2 vs direct ∫dℓ₁dℓ₂ ────────────────────
@pytest.mark.parametrize('q,t', [(0.0, 0.5), (0.7, 0.5), (1.3, 1.0)])
def test_c2_sunset_matches_direct(q, t):
    """Σ_sunset(q,t) = ∫dℓ₁dℓ₂/(2π)² C(ℓ₁,t)C(ℓ₂,t)C(q−ℓ₁−ℓ₂,t) via C2
    (a 3-D Gauss–Laguerre Schwinger quadrature over the L=2 momentum integral)
    vs the direct double momentum integral.  Exercises the L=2 momentum reduction
    inside the causal time-simplex.  Validated at t>0 (smooth; the t→0 corner is
    the UV regime a cutoff regularizes — out of this milestone's scope)."""
    got = sigma_parametric(sunset_edges(), q, t, MU, D, T)

    def integrand(l2, l1):                       # C(ℓ,t) = (T/m_ℓ) e^{−m_ℓ t}
        m1, m2, m3 = _m(l1), _m(l2), _m(q - l1 - l2)
        return ((T / m1) * (T / m2) * (T / m3)
                * math.exp(-(m1 + m2 + m3) * t))
    ref, _ = integrate.dblquad(integrand, -np.inf, np.inf,
                               lambda _l1: -np.inf, lambda _l1: np.inf)
    ref /= (2 * np.pi) ** 2
    assert abs(got - ref) <= 5e-3 * max(abs(ref), 1e-12)


# ── C2 in higher dimension (d=2) — backend B (1-D ∫dℓ) cannot do this ──
@pytest.mark.parametrize('q,t', [(0.0, 0.4), (0.9, 0.4), (1.5, 0.7)])
def test_c2_bubble_d2_sigma_R_matches_brute_force(q, t):
    """The d=2 retarded self-energy Σ_R(q,t)=∫d²ℓ/(2π)² G_R(ℓ,t)C(q−ℓ,t) via C2
    (sigma_parametric with spatial_dim=2 — the analytic Symanzik momentum step in
    d=2) vs a direct brute-force ∫d²ℓ (q⃗=(q,0)).  Demonstrates the C-stack reaches
    d>1 for the self-energy with NO angular quadrature — the closed-form Symanzik
    reduction makes the d-dim loop a parameter flip."""
    got = sigma_parametric(bubble_edges('R'), q, t, MU, D, T, spatial_dim=2)

    def integrand(ly, lx):
        m_l = MU + D * (lx ** 2 + ly ** 2)
        m_ql = MU + D * ((q - lx) ** 2 + ly ** 2)
        return math.exp(-m_l * t) * (T / m_ql) * math.exp(-m_ql * t)
    ref, _ = integrate.dblquad(integrand, -np.inf, np.inf,
                               lambda _lx: -np.inf, lambda _lx: np.inf)
    ref /= (2 * np.pi) ** 2
    assert abs(got - ref) <= 1e-4 * max(abs(ref), 1e-12)


# ── C3-lite capstone: the full bubble δC(q,0) end-to-end through the C stack ──
@pytest.mark.parametrize('q', [0.0, 0.7, 1.5])
def test_c3lite_bubble_delta_matches_golden(q):
    """END-TO-END C0→C1→C2→C3-lite: the equal-time bubble δC(q,0) assembled from
    the new stack (Σ via sigma_parametric → Dyson collapse) reproduces the golden
    backend-B reference loop_dyson.bubble_delta_S.  Proves the stack composes into
    the validated physical correlator."""
    g = 0.2
    got = bubble_delta_equal_time_via_C(q, MU, D, T, g=g)
    ref = bubble_delta_S(q, MU, D, T, g=g)
    assert abs(got - ref) <= 3e-2 * max(abs(ref), 1e-12)

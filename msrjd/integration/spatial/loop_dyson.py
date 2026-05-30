"""
msrjd.integration.spatial.loop_dyson
====================================
1-loop Dyson assembly for the spatial bubble (Stage C.5) — turns the
momentum-first self-energies (``loop_parametric``) into the dressed
correlator correction.

For the ``φ̃φ²`` reaction-diffusion theory the 1-loop self-energy is a
**bubble** (momentum-DEPENDENT), with a retarded part ``Σ_R = G_R·C`` and a
Keldysh part ``Σ_K = C·C``.  The dressed correlation is the standard MSR Dyson

    δC(q,ω) = G_R⁰ Σ_R C⁰ + G_R⁰ Σ_K G_A⁰ + C⁰ Σ_A G_A⁰ ,

with ``G_R⁰=1/(m-iω)``, ``G_A⁰=1/(m+iω)``, ``C⁰=2T/(m²+ω²)``, ``m=μ+Dq²``,
``Σ_A=Σ_R*``.  The **equal-time** structure-factor correction ``δC(q,τ=0)``
has the closed convolution form (derived by inverse-FT at ``τ=0``)

    δC(q,0) = (T/m²) ∫₀^∞ Σ_R(q,u) e^{-mu} du
            + (1/m)  ∫₀^∞ Σ_K(q,u) e^{-mu} du ,

(the Keldysh double-time integral ``∫∫e^{-m(a₁+a₂)}Σ_K(|a₁-a₂|)`` collapses to a
1-D integral under ``(a₁,a₂)→(s=a₁+a₂, u=a₁-a₂)`` since ``Σ_K`` is even),
validated frequency-route == time-route (``tests/test_loop_dyson.py``).

The self-energy ``∫dℓ`` is **pole-free** (a momentum integral of a product of
exponentials/Lorentzians) whether done directly or by the parametric Symanzik
route — the ``m≥3`` close-pair bug lived ONLY in Phase J's time-polytope, which
this assembly bypasses.  So the bubble integrator is fast and robust at any q.

Normalization here is per ``Σ_R = ∫dℓ G_R·C``, ``Σ_K = ∫dℓ C·C`` with NO
coupling / combinatorial factor — the caller multiplies by ``M(Γ)·(coupling)``
(``g²`` for the bubble), pinned from the pipeline.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate


def _mk(k, mu, D):
    return mu + D * k * k


# ── self-energy time kernels (direct ∫dℓ; pole-free, fast) ─────────
def sigma_R_time(q, t, mu, D, T):
    """Retarded bubble ``∫dℓ/2π G_R(ℓ,t) C(q-ℓ,t)``  (t>0)."""
    if t <= 0:
        return 0.0
    f = lambda l: (math.exp(-_mk(l, mu, D) * t)
                   * (T / _mk(q - l, mu, D)) * math.exp(-_mk(q - l, mu, D) * t))
    v, _ = integrate.quad(f, -np.inf, np.inf, limit=120)
    return v / (2 * math.pi)


def sigma_K_time(q, t, mu, D, T):
    """Keldysh bubble ``∫dℓ/2π C(ℓ,t) C(q-ℓ,t)``  (even in t)."""
    at = abs(t)
    f = lambda l: ((T / _mk(l, mu, D)) * math.exp(-_mk(l, mu, D) * at)
                   * (T / _mk(q - l, mu, D)) * math.exp(-_mk(q - l, mu, D) * at))
    v, _ = integrate.quad(f, -np.inf, np.inf, limit=120)
    return v / (2 * math.pi)


# ── equal-time structure-factor correction (closed Dyson form) ────
def bubble_delta_S(q, mu, D, T):
    """``δC(q, τ=0)`` from the bubble self-energies, normalization 1.

    The bubble's contribution to the equal-time structure factor ``S(q)``,
    per unit ``M(Γ)·coupling²``.  Even in ``q``.
    """
    m = _mk(q, mu, D)
    t1f = lambda u: sigma_R_time(q, u, mu, D, T) * math.exp(-m * u)
    t1, _ = integrate.quad(t1f, 0, np.inf, limit=200)
    t1 *= T / (m * m)
    t2f = lambda u: sigma_K_time(q, u, mu, D, T) * math.exp(-m * u)
    t2, _ = integrate.quad(t2f, 0, np.inf, limit=200)
    t2 /= m
    return t1 + t2


def bubble_delta_phi2(mu, D, T, q_cut=40.0):
    """``δ⟨φ²⟩ = ∫dq/2π δC(q,0)`` from the bubble, normalization 1.

    ``q_cut`` bounds the (fast-decaying) momentum integral; the integrand
    ``δS(q)`` falls off ``~1/q⁴`` so a finite cut is accurate.
    """
    f = lambda q: bubble_delta_S(q, mu, D, T)
    v, _ = integrate.quad(f, 0.0, q_cut, limit=200)
    return 2.0 * v / (2 * math.pi)        # even in q → 2·∫₀

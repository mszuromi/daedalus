"""
msrjd.integration.spatial.loop_parametric
==========================================
Momentum-FIRST parametric loop integration for spatial (heat-kernel) field
theories — the general loop integrator's core (Stage C.5 pivot).

The time-first integrator (reuse Phase J + numerical ``∫dℓ``) is blocked by the
``m≥3`` close-pair precision slow path, which spatial loops trip GENERICALLY
(the loop momentum sweeps edge masses past one another).  The cure is to do the
momentum integral ANALYTICALLY first: with a Schwinger parameter on each
correlation edge every edge is ``e^{-(μ+D k_e²) w_e}``, so

    ∫ d^dℓ/(2π)^d  exp(-D Σ_e k_e² w_e)
       = exp(-D q² (W - V²/U)) / (4π D U)^{d/2},

a pure Gaussian with the Symanzik forms (for ONE loop momentum ℓ, edges
``k_e = a_e ℓ + b_e q``):

    U = Σ_e a_e² w_e ,   V = Σ_e a_e b_e w_e ,   W = Σ_e b_e² w_e .

There are **no momentum-dependent poles** — only a Schwinger/time integral of
the erf family remains — so the close-pair slow path can never arise.

Validated (``docs/spatial_spikes/stageC5_momentumfirst_spike.py`` and the
``Σ_K`` 2-D check): both bubble self-energies reproduce the direct ``∫dℓ`` to
~1e-12.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate


# ── the Symanzik / Gaussian momentum-integral core ────────────────
def symanzik_UF(a, b, w, D, spatial_dim=1):
    """Symanzik data for the ONE-loop, single-external-momentum heat-kernel
    Gaussian momentum integral (the reusable core that backend C's ``C0``
    generalizes to L loops via the matrix forms ``M,N,Q``).

    Edges parametrized ``k_e = a_e ℓ + b_e q`` with Schwinger weight ``w_e ≥ 0``::

        U = Σ_e a_e² w_e            (first Symanzik polynomial = det M at L=1)
        V = Σ_e a_e b_e w_e         (the M⁻¹-coupling, = N at L=1)
        W = Σ_e b_e² w_e            (= Q at L=1)
        F_reduced = W − V²/U        (= Q − Nᵀ M⁻¹ N; the coeff of q² in the
                                     exponent — "F/U" per external q² )

    Returns ``(U, F_reduced, prefactor)`` with ``prefactor = (4πDU)^{−d/2}``, so
    the full integral ``∫d^dℓ/(2π)^d exp(−D Σ_e w_e k_e²) = prefactor ·
    exp(−D q² F_reduced)``.  Raises on ``U ≤ 0`` (all weights zero — no
    loop-momentum damping).  See ``docs/backend_C_math.md`` §2.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(w, dtype=float)
    U = float(np.sum(a * a * w))
    if U <= 0.0:
        raise ValueError(f'Symanzik U={U} ≤ 0 — no loop-momentum damping '
                         f'(all Schwinger weights zero?).')
    V = float(np.sum(a * b * w))
    W = float(np.sum(b * b * w))
    F_reduced = W - V * V / U               # the second Symanzik form (per q²)
    pref = (4.0 * math.pi * D * U) ** (-0.5 * spatial_dim)
    return U, F_reduced, pref


def gaussian_momentum_integral(a, b, w, q, D, spatial_dim=1):
    """``∫ d^dℓ/(2π)^d exp(-D Σ_e (a_e ℓ + b_e q)² w_e)``  for ONE loop
    momentum ``ℓ`` (1-D ℓ; ``spatial_dim`` = d enters the power only because
    each spatial component contributes one Gaussian).

    a, b, w : equal-length sequences — per-edge ℓ-coefficient, q-coefficient,
              and Schwinger weight ``w_e ≥ 0``.
    Returns ``exp(-D q² (W - V²/U)) / (4π D U)^{d/2}``.  ``U → 0`` (all weights
    zero) is a degenerate request and raises.  Thin wrapper over
    :func:`symanzik_UF` (the reusable Symanzik core).
    """
    U, F_reduced, pref = symanzik_UF(a, b, w, D, spatial_dim=spatial_dim)
    return pref * math.exp(-D * q * q * F_reduced)


# ── bubble self-energies via the parametric core ──────────────────
# (the tree mass m_k = μ + D k² lives in loop_dyson._mk — single source)
# Σ_R kernel = ∫dℓ/2π G_R(ℓ,t) C(q-ℓ,t)   (one response + one correlation edge)
#   response edge:    k=ℓ   (a=1, b=0),  weight w = t          (G_R = e^{-m_ℓ t})
#   correlation edge: k=q-ℓ (a=-1, b=1), weight w = s ≥ t      (C   = T∫_t^∞ ds e^{-m s})
def sigma_R_kernel(q, t, mu, D, T):
    """``∫dℓ/2π G_R(ℓ,t) C(q-ℓ,t)`` for t>0, via the parametric core
    (NO momentum poles).  This is the response-self-energy bubble kernel
    (combinatorial/coupling prefactor applied by the caller)."""
    if t <= 0:
        t = 1e-12

    def integrand(s):
        # edges: response (a=1,b=0,w=t), correlation (a=-1,b=1,w=s)
        gauss = gaussian_momentum_integral([1.0, -1.0], [0.0, 1.0], [t, s],
                                           q, D, spatial_dim=1)
        return math.exp(-mu * (t + s)) * gauss
    val, _ = integrate.quad(integrand, t, np.inf, limit=200)
    return T * val


# Σ_K kernel = ∫dℓ/2π C(ℓ,t) C(q-ℓ,t)   (two correlation edges)
def sigma_K_kernel(q, t, mu, D, T):
    """``∫dℓ/2π C(ℓ,t) C(q-ℓ,t)`` via the parametric core (2-D Schwinger)."""
    at = abs(t) if t != 0 else 1e-12

    def integrand(s2, s1):
        gauss = gaussian_momentum_integral([1.0, -1.0], [0.0, 1.0], [s1, s2],
                                           q, D, spatial_dim=1)
        return math.exp(-mu * (s1 + s2)) * gauss
    val, _ = integrate.dblquad(integrand, at, np.inf,
                               lambda s1: at, lambda s1: np.inf)
    return T * T * val

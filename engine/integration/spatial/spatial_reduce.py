"""
engine.integration.spatial.spatial_reduce
=========================================
Backend C — **C0 (graph → Symanzik polynomials)** and **C1 (the L-loop Gaussian
momentum integral)**.  The general, topology-/dimension-agnostic core that
backend C is built on.  Design: ``docs/backend_C_design.md`` (C0, C1); math:
``docs/backend_C_math.md`` §2, §6.

The heat-kernel ``(k,t)`` representation makes the edge times Schwinger
parameters, so the L-loop momentum integral is a pure Gaussian:

    ∫ ∏_{i=1}^{L} dᵈℓ_i/(2π)ᵈ  exp[ −D Σ_e w_e k_e² ],   k_e = Σ_i a_{ei} ℓ_i + Σ_j b_{ej} q_j

with per-edge **routing coefficients** ``(a_e, b_e)`` from
``RoutingResult.edge_coeffs()``.  Collecting the loop momenta into the quadratic
form gives the Symanzik matrices

    Lam_{ii'} = Σ_e w_e a_{ei} a_{ei'}      (L×L)   — first Symanzik:  U = det Lam
    N_{ij}  = Σ_e w_e a_{ei} b_{ej}       (L×n_ext)
    Q_{jj'} = Σ_e w_e b_{ej} b_{ej'}      (n_ext×n_ext)

and the integral collapses (``docs/backend_C_math.md`` §2):

    I_mom = (4πD)^{−Ld/2} · U^{−d/2} · exp[ −D · qᵀ Q_eff q ],
        Q_eff = Q − Nᵀ Lam⁻¹ N        (the reduced external quadratic form; the
                                     second-Symanzik form F/U).

**Scope (C0/C1).** This is the *momentum reduction* only — exact and `d`-general
(`d` enters solely as the `−Ld/2`/`−d/2` exponents).  The residual causal-time
parameter integral is **C2** (``temporal_integrate``); the cutoff that makes the
parameter integral finite is a first-class input there.  For a smooth-Gaussian
cutoff the reduction stays closed-form (shift ``w_e → w_e + σ²/D``); a hard or
lattice cutoff keeps it finite but is handled numerically downstream.

Validated against the hand polynomials (``docs/backend_C_math.md`` §6):
the 1-loop bubble (``U=w₁+w₂``, ``Q_eff=w₁w₂/(w₁+w₂)``) and the 2-loop sunset
(``U=w₁w₂+w₂w₃+w₃w₁``, ``Q_eff=w₁w₂w₃/U``), and against
``loop_parametric.symanzik_UF`` / ``gaussian_momentum_integral`` at L=1.
"""
from __future__ import annotations

import math

import numpy as np


# ── Notation (code ↔ paper App. B) ───────────────────────────────────
#   Lam      Λ        loop / first-Symanzik matrix  Σ_e w_e a_e a_eᵀ
#   N, Q     N_rb,Q_ab   Symanzik cross / external blocks  (match paper)
#   U        U_Γ      first Symanzik polynomial = det Lam
#   Q_eff    𝓑(w)/D   reduced external form  Q − Nᵀ Lam⁻¹ N   (𝓑 = D·Q_eff)
#   a, b     B_er,C_eb   edge routing coefficients (plain B,C in paper)
#   D        D_0      scalar reference diffusion ;  w,q ↔ w_e,q_b
# ─────────────────────────────────────────────────────────────────────


# ── C0: graph → Symanzik polynomials (numeric) ────────────────────
def symanzik_matrices(a_list, b_list, weights):
    """Build the Symanzik matrices ``(Lam, N, Q)`` from per-loop-edge routing
    coefficients and Schwinger weights (numeric).

    a_list : sequence of length-``L`` loop-coefficient tuples, one per loop edge.
    b_list : sequence of length-``n_ext`` external-coefficient tuples (same edges).
    weights: sequence of edge Schwinger weights ``w_e ≥ 0`` (same order).

    Returns ``(Lam, N, Q)`` as ``np.ndarray`` of shapes ``(L,L)``, ``(L,n_ext)``,
    ``(n_ext,n_ext)``.
    """
    a = np.asarray(a_list, dtype=float)          # (E, L)
    b = np.asarray(b_list, dtype=float)          # (E, n_ext)
    w = np.asarray(weights, dtype=float)         # (E,)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] != w.shape[0]:
        raise ValueError('a_list, b_list, weights must be E×L, E×n_ext, E '
                         'with matching E (one row per loop edge).')
    aw = a * w[:, None]                           # (E, L)
    Lam = aw.T @ a                                  # (L, L)
    N = aw.T @ b                                  # (L, n_ext)
    Q = (b * w[:, None]).T @ b                    # (n_ext, n_ext)
    return Lam, N, Q


def symanzik_polynomials(a_list, b_list, weights):
    """C0: return ``(U, Q_eff)`` — the first Symanzik ``U = det Lam`` and the
    reduced external quadratic form ``Q_eff = Q − Nᵀ Lam⁻¹ N`` (an ``n_ext×n_ext``
    matrix; the exponent of the momentum integral is ``−D·qᵀ Q_eff q``).

    Numeric.  ``U ≤ 0`` (no loop-momentum damping — all weights zero) raises.
    For a single external momentum (``n_ext==1``) ``Q_eff`` is a 1×1 whose entry
    is the scalar ``F_reduced`` of :func:`loop_parametric.symanzik_UF`.
    """
    Lam, N, Q = symanzik_matrices(a_list, b_list, weights)
    U = float(np.linalg.det(Lam))
    if U <= 0.0:
        raise ValueError(f'Symanzik U={U} ≤ 0 — no loop-momentum damping '
                         f'(all Schwinger weights zero?).')
    Q_eff = Q - N.T @ np.linalg.solve(Lam, N)       # Q − Nᵀ Lam⁻¹ N
    return U, Q_eff


# ── C1: the L-loop Gaussian momentum integral ─────────────────────
def momentum_integral(a_list, b_list, weights, q, D, spatial_dim=1):
    """C1: evaluate ``∫ ∏ dᵈℓ_i/(2π)ᵈ exp[−D Σ_e w_e k_e²]`` =
    ``(4πD)^{−Ld/2} U^{−d/2} exp[−D qᵀ Q_eff q]`` for ``L`` loop momenta and
    ``n_ext`` external momenta.

    ``q`` : the external momentum/momenta — a scalar (``n_ext==1``) or a
            length-``n_ext`` sequence (for an isotropic theory, the components
            along one fixed axis suffice since ``Q_eff`` couples only ``qᵢ·qⱼ``).
    Returns a float.  Generalizes :func:`loop_parametric.gaussian_momentum_integral`
    (the ``L=1`` case) to any ``L`` and any ``d``.
    """
    L = len(a_list[0]) if len(a_list) else 0
    if L == 0:
        raise ValueError('momentum_integral needs ≥1 loop momentum (L≥1).')
    U, Q_eff = symanzik_polynomials(a_list, b_list, weights)
    qv = np.atleast_1d(np.asarray(q, dtype=float))
    if qv.shape[0] != Q_eff.shape[0]:
        raise ValueError(f'q has {qv.shape[0]} components but n_ext='
                         f'{Q_eff.shape[0]}.')
    quad = float(qv @ Q_eff @ qv)                 # qᵀ Q_eff q
    pref = (4.0 * math.pi * D) ** (-0.5 * spatial_dim * L) * U ** (-0.5 * spatial_dim)
    return pref * math.exp(-D * quad)

"""
msrjd.integration.spatial.generic_evaluator
============================================
Backend C — **Phase 2 of the generic spatial loop pipeline**
(``docs/spatial_generic_pipeline_plan.md``): evaluate one
:class:`~msrjd.integration.spatial.diagram_descriptor.CStackDiagram`'s
contribution to the correlator, by the momentum-first route

    enumerate → map (Phase 1) → Symanzik ∫dᵈℓ (C0/C1) → causal-chamber ∫dt (C2).

This is the **one** evaluator every diagram goes through.  There is no
bubble/tadpole branch: the loop momentum couples (or not) to ``q`` purely through
the Symanzik polynomials, which are built mechanically from the edge list.

Two entry points (built/validated incrementally):
  * :func:`loop_self_energy` — Phase 2a: the amputated 2-vertex self-energy of a
    diagram's loop edges, via the validated
    :func:`~msrjd.integration.spatial.temporal_integrate.sigma_parametric`.  Used
    to confirm the descriptor → C-stack edge mapping reproduces the hand-coded
    ``bubble_edges`` oracle.
  * :func:`evaluate_diagram` — Phase 2b: the FULL diagram value ``C_Γ(q,τ)``
    (external legs included), the strict-fixed-order momentum-first replacement
    for ``compute_correction_td``.  *(under construction)*

Normalization: **kinematic only** — couplings, noise amplitudes and the
combinatorial ``M(Γ)`` are the enumeration's ``scalar_prefactor`` and are applied
by the caller.  A bare ``C`` edge contributes its Schwinger ``e^{−m|Δt|}/m`` with
unit weight (no extra ``T``).
"""
from __future__ import annotations

import math

import numpy as np

from msrjd.integration.spatial.temporal_integrate import sigma_parametric


def loop_self_energy(descr, q, t, mu, D, T=1.0, spatial_dim=1, **quad):
    """Phase 2a — the amputated self-energy ``Σ(q,t)`` of ``descr``'s **loop**
    edges (the internal, non-external lines) via :func:`sigma_parametric`.

    Valid for a 2-vertex loop (a bubble): all loop edges span the same
    inter-vertex time ``t`` (one ordering chamber), which is exactly
    ``sigma_parametric``'s domain.  Returns ``Σ(q,t)`` (kinematic; the caller
    multiplies the enumeration ``M(Γ)·prefactor``).  Diagrams whose loop has a
    self-loop (a tadpole) or >2 internal vertices are handled by the full
    :func:`evaluate_diagram` (Phase 2b), not here.
    """
    le = descr.loop_edges()
    selfloops = [e for e in le if e.u == e.v]
    if selfloops:
        raise ValueError(
            "loop_self_energy is the 2-vertex (bubble) helper; this diagram has a "
            "self-loop (tadpole) — use evaluate_diagram (Phase 2b).")
    edges = [(tuple(float(x) for x in e.a),
              tuple(float(x) for x in e.b), e.kind) for e in le]
    return sigma_parametric(edges, q, t, mu, D, T, spatial_dim=spatial_dim, **quad)


# ── the generic single-mode Dyson convolution (external-leg dressing) ──
# This is the strict-fixed-order first-order correction for a self-energy
# inserted into the tree correlator of a SINGLE mode ``(A, B, N)``
# (``m = A + B q²``, ``G_R⁰(t)=θ(t)e^{−mt}``, ``C⁰(t)=(N/m)e^{−m|t|}``):
#
#   retarded:  δC_R(τ) = (G_R⁰⊛Σ_R⊛C⁰)(τ) + (C⁰⊛Σ_A⊛G_A⁰)(τ)
#   Keldysh :  δC_K(τ) = (G_R⁰⊛Σ_K⊛G_A⁰)(τ)
#
# Model-independent: it depends ONLY on the tree mode (A,B,N) and the self-energy
# grid ``σ(a)`` — no theory-specific constants.  (This is the same convolution the
# now-retired bespoke ``loop_dyson`` did for the φ̃φ² bubble, generalized to any
# self-energy and parametrized by the mode; the small-``a`` power-law sliver is
# kept so equal-time / derivative-vertex self-energies integrate accurately.)


def _sigma_grid(sig_of_u, m, taus, n_floor=2000, t_max_cap=60.0):
    """Sample ``σ(a)`` on an adaptive grid resolving the convolution kernels
    (decay ``1/m``) and the τ-reach.  ``sig_of_u(a)`` is the kinematic
    self-energy (callable; one scalar per ``a``)."""
    tau_max = max((abs(float(t)) for t in taus), default=0.0)
    t_max = min(t_max_cap, max(2.0 * tau_max + 12.0 / m, 12.0 / m))
    n_t = int(min(max(n_floor, t_max * m * 50.0), 8000.0))
    ag = np.linspace(t_max / n_t, t_max, n_t)
    sg = np.array([float(sig_of_u(float(a))) for a in ag])
    return ag, sg


def _dyson_retarded(ag, sR, m, N, taus):
    """``δC_R(τ) = Term1(τ)+Term1(−τ)``, ``Term1=(N/m)∫₀^∞ σ_R(a)K(τ−a)da`` with
    the closed inner kernel ``K`` (``G_R⁰⊛C⁰``) and the small-``a`` power-law
    sliver (an equal-time / derivative-vertex σ_R may diverge integrably as
    ``a→0⁺``)."""
    a1 = ag[0]

    def _slope(s):
        if len(s) > 1 and s[0] > 0.0 and s[1] > 0.0:
            return math.log(s[1] / s[0]) / math.log(ag[1] / ag[0])
        return 0.0
    p = max(_slope(sR), -0.95)
    sliv = sR[0] * a1 / (1.0 + p)
    aeff = a1 * (1.0 + p) / (2.0 + p)

    def _K(c):
        c = np.asarray(c, dtype=float)
        return np.where(c >= 0.0, np.exp(-m * np.abs(c)) * (np.abs(c) + 0.5 / m),
                        np.exp(-m * np.abs(c)) / (2.0 * m))

    out = np.empty(len(taus))
    for i, tau in enumerate(taus):
        t1 = (N / m) * (sliv * _K(tau - aeff) + np.trapz(sR * _K(tau - ag), ag))
        t1m = (N / m) * (sliv * _K(-tau - aeff) + np.trapz(sR * _K(-tau - ag), ag))
        out[i] = t1 + t1m
    return out


def _dyson_keldysh(ag, sK, m, taus):
    """``δC_K(τ) = (1/2m)∫ σ_K(|τ−d|) e^{−m|d|} dd`` (``G_R⁰⊛Σ_K⊛G_A⁰``); σ_K is
    even and finite at 0, so no sliver is needed."""
    ag0 = np.concatenate(([0.0], ag))
    sK0 = np.concatenate(([sK[0]], sK))
    dg = np.concatenate((-ag[::-1], ag0))
    out = np.empty(len(taus))
    for i, tau in enumerate(taus):
        sh = np.interp(np.abs(tau - dg), ag0, sK0, left=sK0[0], right=0.0)
        out[i] = (1.0 / (2.0 * m)) * np.trapz(sh * np.exp(-m * np.abs(dg)), dg)
    return out


def _kinematic_to_physical(descr):
    """The universal kinematic↔enumeration normalization for a diagram: the
    enumeration ``M(Γ)·prefactor`` uses the noise-source action coefficient, while
    the kinematic convolution uses a unit (T=1) correlation amplitude per C edge.
    They differ by ``2^{−n_C}`` (``n_C`` = number of correlation edges = number of
    contracted 2-point noise sources).  VERIFIED universal across the φ̃φ² Σ_R/Σ_K
    bubbles (n_C=2 → 1/4 = C_R/M_R = C_K/M_K) and the tadpole; re-checked at the
    2-loop sunset (n_C=3)."""
    n_C = sum(1 for e in descr.edges if e.kind == 'C')
    return 2.0 ** (-n_C)


def bubble_delta_C(descr, prefactor_val, q, taus, A, B, N, mu, D,
                   spatial_dim=1):
    """The full 1-loop **bubble** contribution ``δC_Γ(q,τ)`` of one diagram,
    the generic momentum-first way: ``σ_Γ`` from the Symanzik route
    (:func:`loop_self_energy`), dressed by the generic single-mode Dyson
    convolution, weighted by ``2^{−n_C}·(M(Γ)·prefactor)``.

    ``descr`` must be a non-tadpole (no self-loop) 2-vertex bubble; classified
    Σ_R (loop kinds ``{R,C}``) → retarded dressing, Σ_K (``{C,C}``) → Keldysh.
    ``prefactor_val`` is the enumeration ``M(Γ)·prefactor`` evaluated at the
    params (carries the couplings + noise amplitudes).  Returns an array over
    ``taus`` (kinematic σ uses T=1; the T's live in ``prefactor_val``)."""
    taus = np.asarray(taus, dtype=float)
    m = A + B * q * q
    loop_kinds = tuple(sorted(e.kind for e in descr.loop_edges()))
    sig = lambda u: loop_self_energy(descr, q, u, mu, D, T=1.0,
                                     spatial_dim=spatial_dim)
    ag, sg = _sigma_grid(sig, m, taus)
    if loop_kinds == ('C', 'R'):
        kin = _dyson_retarded(ag, sg, m, N, taus)
    elif loop_kinds == ('C', 'C'):
        kin = _dyson_keldysh(ag, sg, m, taus)
    else:
        raise NotImplementedError(
            f"bubble_delta_C: unsupported loop-edge kinds {loop_kinds}.")
    return _kinematic_to_physical(descr) * float(prefactor_val) * kin

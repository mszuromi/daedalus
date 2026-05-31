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

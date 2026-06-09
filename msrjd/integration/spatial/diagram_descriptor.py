"""
msrjd.integration.spatial.diagram_descriptor
=============================================
Backend C — **Phase 1 of the generic spatial loop pipeline**
(``docs/spatial_generic_pipeline_plan.md``): map an enumerated typed diagram onto
the C-stack's representation — internal interaction vertices (integrated times),
external legs (fixed times), and a flat edge list ``[(a, b, kind, (u, v))]`` ready
for the Symanzik momentum reduction (C0/C1) + causal-chamber time integral (C2).

This is the **one** mapping every diagram goes through — tree, bubble, tadpole,
sunset alike.  There is no bubble/tadpole branch: the distinction is whether the
edge momenta couple the loop ``ℓ`` to the external ``q`` (a property of the
Symanzik ``F``, downstream), not a property of this descriptor.

The representation (see ``docs/spatial_loop_diagram_inventory.md``): the shared
enumerator uses **all-``G_R`` propagators + explicit noise sources**.  A
correlation line ``C`` is two ``G_R`` edges meeting at a 2-point noise source
(``SourceType`` with two response legs, the ``⟨φ̃φ̃⟩`` insertion).  So the mapping:

  * **contract** each 2-point noise source: its two incident ``G_R`` edges →
    ONE ``C`` edge between the two *other* endpoints (integrating out the source
    time analytically gives ``C(Δt)=(T/m)e^{−m|Δt|}``).  A source whose two edges
    land on the *same* vertex → a ``C`` **self-loop** (``u == v``) — the tadpole;
  * every remaining ``G_R`` edge stays an ``R`` edge.  **External legs** (an edge
    touching a leaf) are ``R`` (or ``C``) edges with loop-coefficient ``a = 0``:
    they carry only ``±q`` so they drop out of the loop ``∫dᵈℓ`` and contribute a
    plain time factor — uniform with every other edge.

Edge sign convention is irrelevant to the Symanzik forms: flipping one edge's
``(a, b) → (−a, −b)`` leaves every ``Σ_e w_e a a``, ``Σ_e w_e a b``,
``Σ_e w_e b b`` invariant, so the ``C``-edge contraction may take either half's
coefficients.

Normalization: this descriptor is **kinematic only** — couplings, noise
amplitudes, and the combinatorial ``𝒮(Γ)`` all live in the enumeration's
``scalar_prefactor`` and are applied by the evaluator (Phase 2), NOT here.
"""
from __future__ import annotations

from dataclasses import dataclass

from msrjd.core.vertices import SourceType, VertexType
from msrjd.integration.spatial.momentum_routing import route_momenta


@dataclass(frozen=True)
class CEdge:
    """One C-stack edge.  ``a`` = loop-momentum coefficients (over the diagram's
    loop momenta), ``b`` = external-``q`` coefficients; ``kind`` ∈ {'R','C'};
    ``u, v`` are the endpoint vertex ids (``u == v`` ⇒ a self-loop, i.e. a
    tadpole loop).  ``external`` is True iff the edge touches an external leaf.

    ``fpairs`` (coupled-field extension, Dyson 3c): the propagator matrix
    indices of the underlying ``G_R`` half-edge(s), from
    ``td.propagator_indices`` — each entry is ``(resp_idx, phys_idx)``.
    For an ``R`` edge: ``((ri, pi),)``.  For a ``C`` edge:
    ``((ri_u, pi_u), (ri_v, pi_v))`` — the half attached to endpoint ``u``
    first, then the half attached to ``v``.  Empty tuple for hand-built
    descriptors; the single-field paths never read it."""
    a: tuple
    b: tuple
    kind: str
    u: int
    v: int
    external: bool
    fpairs: tuple = ()

    def couples_loop(self):
        """True iff this edge carries any loop momentum (``a`` not all zero) —
        i.e. it participates in the ``∫dᵈℓ``.  External legs have ``a = 0``."""
        return any(ai != 0 for ai in self.a)


@dataclass(frozen=True)
class CStackDiagram:
    """The C-stack view of a typed diagram.

    internal_vertices : tuple of interaction-vertex ids (their times are integrated)
    external_legs     : tuple of leaf vertex ids, in correlation-function order
                        (leg 0 → external time 0; leg j → external time τ_j)
    edges             : tuple of :class:`CEdge`
    n_loops           : number of loop momenta (== diagram loop_number)
    """
    internal_vertices: tuple
    external_legs: tuple
    edges: tuple
    n_loops: int

    def loop_edges(self):
        """The internal (non-external) edges — the self-energy loop."""
        return tuple(e for e in self.edges if not e.external)

    def is_tadpole_like(self):
        """True iff some edge is a self-loop (``u == v``) — a decoupled
        ``⟨φ²⟩``-type loop.  (Diagnostic only; the evaluator does not branch on it.)"""
        return any(e.u == e.v for e in self.edges)


def _is_two_point_noise_source(asg):
    """A ``SourceType`` (no physical legs) with exactly two response legs — the
    Gaussian 2-point noise insertion that represents a ``C`` line."""
    return isinstance(asg, SourceType) and len(asg.response_legs) == 2


def diagram_to_cstack(td) -> CStackDiagram:
    """Map a typed diagram to its :class:`CStackDiagram` (Phase 1).

    Pure structure — no integration, no couplings.  Raises ``NotImplementedError``
    for source types this milestone does not yet handle (non-Gaussian / >2-point
    noise sources, which contract to higher correlation vertices).
    """
    D, _G, leaves, _internal = td.prediagram
    leaf_set = set(leaves)
    va = td.vertex_assignments

    noise, interaction = set(), set()
    for v, asg in va.items():
        if v in leaf_set:
            continue
        if isinstance(asg, VertexType):
            interaction.add(v)
        elif _is_two_point_noise_source(asg):
            noise.add(v)
        elif isinstance(asg, SourceType):
            raise NotImplementedError(
                f"diagram_to_cstack: source vertex {v} has "
                f"{len(asg.response_legs)} response legs (bigrade {asg.bigrade}); "
                f"only 2-point Gaussian noise sources (→ a C line) are supported "
                f"in this milestone (non-Gaussian / >2-point noise is future work).")
        else:
            raise NotImplementedError(
                f"diagram_to_cstack: unrecognized vertex assignment {asg!r} at {v}.")

    rr = route_momenta(td)
    ec = rr.edge_coeffs()
    edges = list(D.edges())                                # (u, v, lbl) 3-tuples

    incident = {}
    for e in edges:
        u, v, _lbl = e
        incident.setdefault(u, []).append(e)
        incident.setdefault(v, []).append(e)

    out_edges, used = [], set()

    # 1) contract each 2-point noise source into one C edge
    for n in noise:
        inc = incident.get(n, [])
        if len(inc) != 2:
            raise NotImplementedError(
                f"diagram_to_cstack: noise source {n} has degree {len(inc)} "
                f"(expected 2 for a 2-point C line).")
        endpoints = []
        for e in inc:
            u, v, _lbl = e
            endpoints.append(v if u == n else u)
            used.add(e)
        a, b = ec[inc[0]]                                  # sign-invariant downstream
        ext = any(p in leaf_set for p in endpoints)
        pidx = getattr(td, 'propagator_indices', None) or {}
        fp = (pidx.get(inc[0]), pidx.get(inc[1]))
        out_edges.append(CEdge(a=tuple(a), b=tuple(b), kind='C',
                               u=int(endpoints[0]), v=int(endpoints[1]),
                               external=ext,
                               fpairs=(fp if None not in fp else ())))

    # 2) every remaining (G_R) edge → an R edge
    for e in edges:
        if e in used:
            continue
        u, v, _lbl = e
        a, b = ec[e]
        ext = (u in leaf_set) or (v in leaf_set)
        pr = (getattr(td, 'propagator_indices', None) or {}).get(e)
        out_edges.append(CEdge(a=tuple(a), b=tuple(b), kind='R',
                               u=int(u), v=int(v), external=ext,
                               fpairs=((tuple(pr),) if pr is not None else ())))

    return CStackDiagram(
        internal_vertices=tuple(sorted(interaction)),
        external_legs=tuple(int(l) for l in leaves),
        edges=tuple(out_edges),
        n_loops=int(rr.n_loops))

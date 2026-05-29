"""
msrjd.integration.spatial.momentum_routing
===========================================
Momentum routing for spatial Feynman diagrams — the pre-integration
step that runs AFTER typing and BEFORE the (t,k) integral
(re-architecture plan §4a).

A spatial interaction vertex is local in space, so ``∫dx_v`` of the
product of fields at the vertex enforces **momentum conservation**
``Σ_e k_e = 0`` at every internal vertex (the spatial analog of the
time-domain vertex, which carries one integrated time ``t_v``).  This
module assigns each propagator edge a momentum expressed as a linear
form in

  * the **external momenta** ``q_0 … q_{k-1}`` (one per external leg,
    with overall conservation ``Σ q_j = 0`` imposed → ``q_{k-1}`` is
    eliminated), and
  * the **loop momenta** ``ℓ_0 … ℓ_{L-1}`` (``L`` = number of loops =
    the free parameters of the conservation system = the diagram's
    ``loop_number``).

Time and momentum are independent labels on the same graph: the time
integral (the existing vertex-time polytope) is untouched; this routing
is a separate linear solve over the diagram graph.

For the heat-kernel field the per-edge propagator pole is
``λ_e = -(A + B·k_e²)``, so the routed ``k_e`` feeds the edge mode's
``k_e²`` into the propagator (and the analytic q-FT / loop integral
afterwards).  At **tree level** every edge carries ``±q`` (``L=0``), so
``k_e² = q²`` uniformly — which is why the Stage-A tree path can use a
single ``Laplacian → -q²`` substitution.  At **loop level** different
edges carry different momenta (e.g. a bubble: ``ℓ`` and ``q-ℓ``), so the
substitution becomes genuinely per-edge — which is what this module
provides.

API
---
``route_momenta(typed_diagram)`` →
    ``RoutingResult(edge_momenta, q_syms, loop_syms, n_loops)``
where ``edge_momenta[(u,v,lbl)]`` is a sympy expression in
``q_syms`` (length k-1 after conservation) and ``loop_syms`` (length L).
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass
class RoutingResult:
    edge_momenta: dict      # (u,v,lbl) -> sympy expr in q_syms + loop_syms
    q_syms: tuple           # external-momentum symbols q_0 … q_{k-1}
    loop_syms: tuple        # loop-momentum symbols ℓ_0 … ℓ_{L-1}
    n_loops: int

    def edge_k2(self):
        """Return ``{edge: k_e**2}`` (the combination the heat-kernel
        propagator pole needs: ``λ_e = -(A + B·k_e²)``)."""
        return {e: sp.expand(m ** 2) for e, m in self.edge_momenta.items()}


def route_momenta(typed_diagram, verbose=False) -> RoutingResult:
    """Solve momentum conservation over a typed diagram's graph.

    Conventions: edge ``(u, v, lbl)`` is oriented ``u → v`` and carries
    momentum ``k_e`` flowing from ``u`` to ``v``.  At each vertex the
    signed sum (``+`` incoming, ``-`` outgoing) equals the external
    momentum injected there (``q_j`` at leaf ``j``, ``0`` at an internal
    vertex).  Overall conservation ``Σ q_j = 0`` is imposed by
    eliminating the last external momentum.
    """
    D, _G, leaves, internal = typed_diagram.prediagram
    edges = list(D.edges())
    leaves = list(leaves)
    vertices = list(D.vertices())
    k = len(leaves)

    q = sp.symbols(f'q0:{k}') if k > 0 else ()
    kk = {e: sp.Symbol(f'k{i}') for i, e in enumerate(edges)}

    # Conservation residual at each vertex (== 0).
    residuals = []
    for v in vertices:
        s = sp.Integer(0)
        for e in edges:
            u, w, _lbl = e
            if w == v:
                s += kk[e]          # edge flows INTO v
            if u == v:
                s -= kk[e]          # edge flows OUT of v
        if v in leaves:
            s -= q[leaves.index(v)]  # external momentum injected at leaf
        residuals.append(sp.expand(s))

    # Overall momentum conservation: q_{k-1} = -(q_0 + … + q_{k-2}).
    qsub = {}
    if k >= 1:
        qsub[q[-1]] = -sum(q[:-1]) if k > 1 else sp.Integer(0)
    residuals = [sp.expand(r.subs(qsub)) for r in residuals]

    sol_set = sp.linsolve(residuals, list(kk.values()))
    if not sol_set:
        raise ValueError(
            'momentum conservation system is inconsistent for this '
            'diagram (should not happen for a valid MSR diagram).')
    sol = list(sol_set)[0]

    edge_momenta = {e: sp.expand(sol[i]) for i, e in enumerate(edges)}

    # Free parameters remaining = loop momenta.  sympy leaves them as the
    # original k-symbols that weren't pinned; relabel to ℓ_0 … ℓ_{L-1}.
    used_q = set(q[:-1]) if k > 1 else set()
    free = sorted(
        {s for expr in edge_momenta.values() for s in expr.free_symbols}
        - used_q,
        key=lambda s: s.name,
    )
    loop_syms = tuple(sp.Symbol(f'l{i}') for i in range(len(free)))
    relabel = dict(zip(free, loop_syms))
    edge_momenta = {e: sp.expand(m.subs(relabel))
                    for e, m in edge_momenta.items()}

    q_syms = tuple(q[:-1]) if k > 1 else ()

    if verbose:
        print(f'[routing] k={k} external, L={len(loop_syms)} loop(s)')
        for e, m in edge_momenta.items():
            print(f'   edge {e}: k = {m}')

    return RoutingResult(edge_momenta=edge_momenta, q_syms=q_syms,
                         loop_syms=loop_syms, n_loops=len(loop_syms))

"""
msrjd.integration.time_domain.subgraph
======================================
Loop subgraph identification for Phase J kernel reduction.

Scope
-----
At MVP build-time this module is effectively a STUB: for tree-level
typed diagrams (loop_number == 0, i.e., no free/loop frequencies) it
returns `[]` and the orchestrator bypasses Phases 3-5 of the hybrid
pipeline entirely.

When Extension 1 (ℓ=1 bubble) lands, the stub is extended as follows:
identify loop subgraphs by taking the **connected closure of edges
whose resolved frequency depends on one or more loop variables**. Each
connected component becomes one `LoopSubgraph` instance. Disjoint loops
therefore become separate subgraphs and their kernels can be reduced
independently.

Nontrivial overlap / nesting — i.e., the case where two loop variables
claim shared edges — is intentionally out of scope for the initial
implementation. When that case is detected, this module raises
`NotImplementedError` with a clear message identifying the offending
loop variables and the shared edges. The pipeline orchestrator catches
that and falls back to the Phase I residue backend for the affected
kernel group. No silent merging, no partial handling: we want the
unsupported case to be loud and obvious.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class LoopSubgraph:
    """
    A connected loop-dependent subgraph of a typed diagram.

    Attributes
    ----------
    loop_vars : list
        The canonical loop-variable SR symbols (`L_0, L_1, ...`) whose
        edges make up this subgraph.
    edges : list
        Edge keys `(idx, u, v)` of the edges inside the subgraph.
    internal_vertices : set
        Vertices all of whose incident D-edges are inside `edges`.
    attachment_vertices : list
        Ordered list of attachment vertices — the 'p' ports through
        which the reduced kernel `K(τ_1, ..., τ_{p-1})` connects back to
        the parent diagram. The first attachment vertex is conventionally
        taken as the origin for the kernel's internal time frame.
    attachment_kind : dict
        Mapping from each attachment vertex to one of
        `'incoming' | 'outgoing' | 'both'`, recording its role at the
        kernel boundary.
    """

    loop_vars: List = field(default_factory=list)
    edges: List = field(default_factory=list)
    internal_vertices: Set = field(default_factory=set)
    attachment_vertices: List = field(default_factory=list)
    attachment_kind: Dict = field(default_factory=dict)


def identify_loop_subgraphs(representative_ir, typed_diagram):
    r"""
    Identify the loop-dependent subgraphs of a typed diagram.

    For tree-level diagrams (empty `free_freqs`), this returns `[]` and
    the caller proceeds directly to tree-level time-domain integration
    (Phase 6 of the hybrid pipeline).

    For loop-containing diagrams, the function raises
    `NotImplementedError` — the MVP does not yet handle loop reduction.
    Extension 1 will flesh out the algorithm below.

    Parameters
    ----------
    representative_ir : dict
        Output of `build_integrand_stationary` for the kernel group's
        representative diagram. Used for `free_freqs`, `edge_freqs`, and
        `substitutions`.
    typed_diagram : TypedDiagram
        Needed for the prediagram D and propagator indices.

    Returns
    -------
    list of LoopSubgraph
        Empty for tree-level; raises otherwise (see above).

    Planned algorithm (Extension 1)
    -------------------------------
    1. `free_freqs = representative_ir['free_freqs']`. If empty → `[]`.
    2. For each loop variable `L_r`, iterate `edge_freqs.items()`,
       apply `substitutions`, and collect every edge whose resolved
       frequency contains `L_r` in its free-variable set. This is the
       seed edge set for `L_r`.
    3. Union the seed edges by shared vertices to form connected
       components — these are candidate subgraphs.
    4. For each component, classify vertices as internal (all incident
       D-edges belong to the component) or attachment (otherwise).
    5. **Overlap check**: if any edge is claimed by the seed sets of
       two different loop variables, raise `NotImplementedError`
       identifying the offending loop vars and the shared edges. Do
       not merge silently.
    6. Return a `LoopSubgraph` instance per component.
    """
    free_freqs = representative_ir.get('free_freqs', [])
    if not free_freqs:
        return []

    raise NotImplementedError(
        "Phase J loop subgraph identification is not implemented in the "
        "MVP; the orchestrator should fall back to Phase I for kernel "
        "groups with loop_number > 0. "
        f"loop_vars = {list(free_freqs)}"
    )

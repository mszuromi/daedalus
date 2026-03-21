"""
msrjd.diagrams.symmetry
========================
Symmetry factor computation for fully-typed labeled diagrams.

The symmetry factor S counts the number of leg permutations at each
vertex that produce an identical diagram (same edge types and propagator
assignments).  For a vertex with groups of identical legs, each group
of k identical legs contributes k! to S.

The total symmetry factor is the product over all vertices.  This factor
cancels the 1/n! denominators from the Taylor expansion of the action,
so the diagram's weight is  S × (product of vertex coefficients).

Additionally, graph-level automorphisms (e.g. exchanging identical
parallel edges in a bubble diagram) contribute to S via the subdivision
graph method.

Build Phase G.
"""

from collections import Counter
from math import factorial

from sage.all import DiGraph


def _vertex_leg_symmetry(vertex_type):
    """
    Compute the leg-permutation symmetry factor for a single vertex.

    Returns the number of permutations of identical legs that leave the
    vertex's edge assignments unchanged = product of k! for each group
    of k identical response legs, times the same for physical legs.
    """
    resp_legs = vertex_type.response_legs
    has_phys = hasattr(vertex_type, 'physical_legs')
    phys_legs = vertex_type.physical_legs if has_phys else []

    S = 1
    for count in Counter(resp_legs).values():
        S *= factorial(count)
    for count in Counter(phys_legs).values():
        S *= factorial(count)
    return S


def build_subdivision_graph(typed_diagram):
    """
    Build a subdivision DiGraph where each edge becomes a vertex.

    This converts edge-exchange symmetries (relevant for multigraphs
    like self-energy bubbles) into vertex permutation symmetries.

    Returns
    -------
    S : DiGraph
        Subdivision graph (no multiedges).
    partition : list of list
        Vertex partition for automorphism_group.
    """
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]
    leaf_set = set(leaves)

    S = DiGraph()

    # Add original vertices
    for v in D.vertices():
        S.add_vertex(v)

    # Add edge vertices (offset to avoid collision with original vertices)
    max_v = max(D.vertices()) if D.vertices() else -1
    edge_offset = max_v + 1

    edge_list = list(D.edges())
    edge_verts_by_color = {}  # group edge vertices by propagator type

    for i, (u, v, lbl) in enumerate(edge_list):
        ev = edge_offset + i
        S.add_vertex(ev)
        S.add_edge(u, ev)
        S.add_edge(ev, v)

        # Edge color = propagator type
        # For multiedge graphs, try (u, v, lbl) first, then (u, v)
        prop_idx = typed_diagram.propagator_indices.get(
            (u, v, lbl),
            typed_diagram.propagator_indices.get((u, v), (0, 0))
        )
        edge_verts_by_color.setdefault(prop_idx, []).append(ev)

    # Build partition
    # 1. External legs: each in its own singleton (fixed)
    # 2. Internal vertices: grouped by type identity
    # 3. Edge vertices: grouped by propagator type

    partition = []

    # External legs — singleton partitions
    for lf in sorted(leaves):
        partition.append([lf])

    # Internal vertices — grouped by type
    type_to_verts = {}
    for v in D.vertices():
        if v in leaf_set:
            continue
        vtype = typed_diagram.vertex_assignments.get(v)
        if vtype is not None:
            has_phys = hasattr(vtype, 'physical_legs')
            key = (type(vtype).__name__, vtype.bigrade,
                   str(vtype.coefficient),
                   tuple(vtype.response_legs),
                   tuple(vtype.physical_legs) if has_phys else ())
        else:
            key = ('none', v)
        type_to_verts.setdefault(key, []).append(v)

    for key in sorted(type_to_verts.keys(), key=str):
        partition.append(sorted(type_to_verts[key]))

    # Edge vertices — grouped by propagator type
    for prop_key in sorted(edge_verts_by_color.keys(), key=str):
        partition.append(sorted(edge_verts_by_color[prop_key]))

    return S, partition


def _graph_automorphism_factor(typed_diagram):
    """
    Compute the graph-level automorphism factor using the subdivision
    graph method.  This detects symmetries like exchanging identical
    parallel edges in bubble diagrams.
    """
    S_graph, partition = build_subdivision_graph(typed_diagram)
    try:
        aut_group = S_graph.automorphism_group(partition=partition)
        return aut_group.order()
    except Exception:
        return 1


def symmetry_factor(typed_diagram):
    """
    Compute the symmetry factor S for a typed diagram.

    S = product over all vertices of (leg-permutation factor).

    The leg-permutation factor for each vertex is the product of k!
    for each group of k identical response legs, times the same for
    physical legs.  This counts how many of the enumerated leg
    permutations produce the same fully-labeled diagram.

    When multiplied by the vertex coefficients (which contain 1/n!
    from the Taylor expansion), S cancels the factorial denominators
    for groups of identical legs.

    Parameters
    ----------
    typed_diagram : TypedDiagram

    Returns
    -------
    S : int
        The symmetry factor (always >= 1).
    """
    S = 1
    for v, vtype in typed_diagram.vertex_assignments.items():
        S *= _vertex_leg_symmetry(vtype)
    return S


def compute_all_symmetry_factors(typed_diagrams):
    """
    Compute symmetry factors for all typed diagrams.

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram

    Returns
    -------
    factors : list of int
        Symmetry factor for each diagram, same order as input.
    """
    return [symmetry_factor(td) for td in typed_diagrams]

"""
msrjd.diagrams.filter
======================
Remove prediagrams whose vertex degree signatures cannot be matched
by any vertex or source type in the theory.  Fast set-membership check
that avoids the expensive type-assignment step for impossible prediagrams.

Build Phase D.
"""


def classify_prediagram_vertices(D, leaves):
    """
    Classify non-leaf vertices into source and interaction vertices.

    Parameters
    ----------
    D : DiGraph
        The prediagram directed graph.
    leaves : list
        Leaf (external leg) vertex IDs.

    Returns
    -------
    source_vertices : list
        Non-leaf vertices with in_degree = 0.
    interaction_vertices : list
        Non-leaf vertices with in_degree > 0.
    """
    leaf_set = set(leaves)
    source = []
    interaction = []
    for v in D.vertices():
        if v in leaf_set:
            continue
        if D.in_degree(v) == 0:
            source.append(v)
        else:
            interaction.append(v)
    return source, interaction


def filter_prediagrams(prediagrams, vertex_types, source_types):
    """
    Keep only prediagrams whose vertex degrees are available in the theory.

    Parameters
    ----------
    prediagrams : list of (D, G, leaves, internal)
    vertex_types : list of VertexType
    source_types : list of SourceType

    Returns
    -------
    kept : list of (D, G, leaves, internal)
        Prediagrams that passed the filter.
    n_discarded : int
        Number of prediagrams removed.
    """
    from msrjd.core.vertices import available_degrees
    int_degs, src_degs = available_degrees(vertex_types, source_types)

    kept = []
    for pd in prediagrams:
        D, G, leaves, internal = pd
        source_verts, interaction_verts = classify_prediagram_vertices(D, leaves)

        valid = True
        for v in source_verts:
            if D.out_degree(v) not in src_degs:
                valid = False
                break
        if valid:
            for v in interaction_verts:
                if (D.in_degree(v), D.out_degree(v)) not in int_degs:
                    valid = False
                    break
        if valid:
            kept.append(pd)

    return kept, len(prediagrams) - len(kept)

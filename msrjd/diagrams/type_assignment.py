"""
msrjd.diagrams.type_assignment
===============================
Enumerate all valid field-type assignments on prediagram edges,
vertices, and external legs.  Constraint-satisfaction engine.

For each filtered prediagram, this module produces all valid TypedDiagram
objects — fully labeled Feynman diagrams ready for causality checks,
symmetry factor computation, and integration.

Edge convention:  each directed edge u -> v carries a propagator G_{ij}
where i is the response-field index contributed by vertex u (tail) and
j is the physical-field index contributed by vertex v (head).

Build Phase E.
"""

from itertools import permutations, product
from sage.all import SR


# ── Data structures ──────────────────────────────────────────────────────────

class TypedDiagram:
    """
    A fully typed Feynman diagram.

    Attributes
    ----------
    prediagram : tuple
        (D, G, leaves, internal) — the underlying prediagram.
    vertex_assignments : dict
        {vertex_id: VertexType or SourceType}.
    edge_types : dict
        {(u, v, label): (resp_leg, phys_leg)} where each leg is (field_base, pop_idx).
        Uses 3-tuple edge keys to support multi-edges.
    external_legs : dict
        {leaf_vertex: (field_base, pop_idx)}.
    propagator_indices : dict
        {(u, v, label): (resp_matrix_idx, phys_matrix_idx)} — row/col into G_ft.
    """

    __slots__ = ('prediagram', 'vertex_assignments', 'edge_types',
                 'external_legs', 'propagator_indices')

    def __init__(self, prediagram, vertex_assignments, edge_types,
                 external_legs, propagator_indices):
        self.prediagram         = prediagram
        self.vertex_assignments = vertex_assignments
        self.edge_types         = edge_types
        self.external_legs      = external_legs
        self.propagator_indices = propagator_indices

    # Pickle support for __slots__
    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s, v in state.items():
            object.__setattr__(self, s, v)

    def __repr__(self):
        D = self.prediagram[0]
        return (f'TypedDiagram(vertices={len(D.vertices())}, '
                f'edges={len(D.edges())}, '
                f'assignments={len(self.vertex_assignments)})')


# ── Index map builder ────────────────────────────────────────────────────────

def build_field_index_map(ring_var_names, n_tilde):
    """
    Build mappings from (field_base, pop_idx) to matrix row/col indices.

    Parameters
    ----------
    ring_var_names : list of str
        Ring generator names, e.g. ['vt1','vt2','nt1','nt2','dv1','dv2','dn1','dn2'].
    n_tilde : int
        Number of response-field generators.

    Returns
    -------
    resp_index : dict
        {(base, pop_idx): matrix_row} for response fields.
    phys_index : dict
        {(base, pop_idx): matrix_col} for physical fields.
    """
    from msrjd.core.vertices import _parse_field_name

    resp_index = {}
    phys_index = {}
    resp_counter = 0
    phys_counter = 0

    for i, name in enumerate(ring_var_names):
        base, pop_idx = _parse_field_name(name)
        if i < n_tilde:
            resp_index[(base, pop_idx)] = resp_counter
            resp_counter += 1
        else:
            phys_index[(base, pop_idx)] = phys_counter
            phys_counter += 1

    return resp_index, phys_index


# ── Core enumeration ─────────────────────────────────────────────────────────

def enumerate_typed_diagrams(prediagram, external_fields, vertex_types,
                             source_types, G_ft, resp_index, phys_index):
    """
    Enumerate all valid typed diagrams for a single prediagram.

    Parameters
    ----------
    prediagram : tuple
        (D, G, leaves, internal).
    external_fields : list of (field_base, pop_idx)
        The k fields in the correlation function, in order.
    vertex_types : list of VertexType
    source_types : list of SourceType
    G_ft : matrix or None
        Propagator matrix (used to check for zero entries).
        If None, propagator consistency check is skipped.
    resp_index : dict
        {(base, pop_idx): row} for response fields.
    phys_index : dict
        {(base, pop_idx): col} for physical fields.

    Yields
    ------
    TypedDiagram
    """
    D, G_graph, leaves, internal = prediagram
    leaf_set = set(leaves)
    edges = list(D.edges())  # 3-tuples (u, v, label) — preserves multi-edges

    # Classify non-leaf vertices
    source_verts = []
    interaction_verts = []
    for v in D.vertices():
        if v in leaf_set:
            continue
        if D.in_degree(v) == 0:
            source_verts.append(v)
        else:
            interaction_verts.append(v)
    ordered_internal = source_verts + interaction_verts

    # Precompute outgoing/incoming edge lists per vertex (with labels for multi-edges)
    out_edges_of = {}
    in_edges_of = {}
    for v in D.vertices():
        out_edges_of[v] = list(D.outgoing_edges(v))
        in_edges_of[v]  = list(D.incoming_edges(v))

    # Build candidate types for each non-leaf vertex
    candidates = {}
    for v in source_verts:
        od = D.out_degree(v)
        cands = [st for st in source_types if st.out_degree == od]
        if not cands:
            return
        candidates[v] = cands

    for v in interaction_verts:
        ind, od = D.in_degree(v), D.out_degree(v)
        cands = [vt for vt in vertex_types
                 if vt.in_degree == ind and vt.out_degree == od]
        if not cands:
            return
        candidates[v] = cands

    # Determine leaf direction constraints
    leaf_directions = {}
    for lf in leaves:
        has_out = D.out_degree(lf) > 0
        has_in  = D.in_degree(lf) > 0
        if has_out and not has_in:
            leaf_directions[lf] = 'resp'
        elif has_in and not has_out:
            leaf_directions[lf] = 'phys'
        else:
            leaf_directions[lf] = 'both'

    # External leg assignment: leaf i gets external_fields[i] (fixed, not permuted).
    # External legs are labeled — leg 0 is field 0, leg 1 is field 1, etc.
    # Permuting would generate diagrams for different correlators
    # (e.g. <dn2 dn1> instead of <dn1 dn2>).
    ext_assignment = {}
    valid_ext = True
    for leaf_idx in range(len(external_fields)):
        lf = leaves[leaf_idx]
        field = external_fields[leaf_idx]
        direction = leaf_directions[lf]

        if direction == 'resp' and field not in resp_index:
            valid_ext = False; break
        if direction == 'phys' and field not in phys_index:
            valid_ext = False; break
        if direction == 'both':
            if field not in resp_index and field not in phys_index:
                valid_ext = False; break

        ext_assignment[lf] = field

    if not valid_ext:
        return

    if not ordered_internal:
        # No internal vertices — just external legs connected by edges
        yield from _try_build_diagram_no_internal(
            prediagram, edges, ext_assignment, leaf_set, leaf_directions,
            G_ft, resp_index, phys_index,
        )
        return

    # Enumerate vertex type assignments (Cartesian product)
    candidate_lists = [candidates[v] for v in ordered_internal]

    for combo in product(*candidate_lists):
        vert_assignment = {ordered_internal[i]: combo[i]
                           for i in range(len(ordered_internal))}

        yield from _try_build_diagram(
            prediagram, edges, ext_assignment, vert_assignment,
            ordered_internal, leaf_set, leaf_directions,
            out_edges_of, in_edges_of,
            G_ft, resp_index, phys_index,
        )


def _try_build_diagram_no_internal(prediagram, edges, ext_assignment,
                                    leaf_set, leaf_directions,
                                    G_ft, resp_index, phys_index):
    """Handle the case where all vertices are external legs (no internal vertices)."""
    edge_types = {}
    prop_indices = {}

    for edge in edges:
        u, v = edge[0], edge[1]
        # u is tail → contributes response leg
        # v is head → contributes physical leg
        resp_field = ext_assignment.get(u)
        phys_field = ext_assignment.get(v)

        if resp_field is None or phys_field is None:
            return

        # Verify field direction compatibility
        if resp_field not in resp_index:
            return
        if phys_field not in phys_index:
            return

        # Check propagator
        ri = resp_index[resp_field]
        pi = phys_index[phys_field]
        if G_ft is not None:
            if bool(SR(G_ft[ri, pi]).is_zero()):
                return

        edge_types[edge] = (resp_field, phys_field)
        prop_indices[edge] = (ri, pi)

    yield TypedDiagram(prediagram, {}, edge_types,
                       dict(ext_assignment), prop_indices)


def _try_build_diagram(prediagram, edges, ext_assignment, vert_assignment,
                        ordered_internal, leaf_set, leaf_directions,
                        out_edges_of, in_edges_of,
                        G_ft, resp_index, phys_index):
    """
    Given fixed external and vertex-type assignments, enumerate all valid
    leg matchings and check propagator consistency.

    For each internal vertex, enumerate all bijections from the vertex
    type's legs to its incident edges.  Use backtracking with early
    propagator checks.
    """
    # Precompute leg matching options per internal vertex
    per_vertex_options = []
    for v in ordered_internal:
        vtype = vert_assignment[v]
        options = list(_leg_matchings(vtype, out_edges_of[v], in_edges_of[v]))
        if not options:
            return
        per_vertex_options.append(options)

    # Backtrack over vertices
    yield from _backtrack(
        prediagram, edges, ext_assignment, vert_assignment,
        ordered_internal, leaf_set, leaf_directions,
        per_vertex_options, G_ft, resp_index, phys_index,
        vertex_idx=0, assigned_resp={}, assigned_phys={},
    )


def _leg_matchings(vertex_type, out_edges, in_edges):
    """
    Enumerate all bijections from a vertex type's legs to edges.

    Each outgoing edge gets a response leg; each incoming edge gets a
    physical leg.

    Yields
    ------
    (resp_map, phys_map) where each is a dict {edge: leg}.
    """
    resp_legs = vertex_type.response_legs
    has_phys = hasattr(vertex_type, 'physical_legs')
    phys_legs = vertex_type.physical_legs if has_phys else []

    if len(resp_legs) != len(out_edges) or len(phys_legs) != len(in_edges):
        return

    resp_perms = set(permutations(range(len(resp_legs)))) if resp_legs else {()}
    phys_perms = set(permutations(range(len(phys_legs)))) if phys_legs else {()}

    for rp in resp_perms:
        resp_map = {}
        for i, edge in enumerate(out_edges):
            resp_map[edge] = resp_legs[rp[i]]
        for pp in phys_perms:
            phys_map = {}
            for i, edge in enumerate(in_edges):
                phys_map[edge] = phys_legs[pp[i]]
            yield resp_map, phys_map


def _backtrack(prediagram, edges, ext_assignment, vert_assignment,
               ordered_internal, leaf_set, leaf_directions,
               per_vertex_options, G_ft, resp_index, phys_index,
               vertex_idx, assigned_resp, assigned_phys):
    """
    Recursive backtracking: assign leg matchings at each vertex, checking
    propagator consistency for fully-determined edges as we go.

    assigned_resp : {edge: resp_leg}  — response leg assigned so far
    assigned_phys : {edge: phys_leg}  — physical leg assigned so far
    """
    if vertex_idx == len(ordered_internal):
        # All internal vertices assigned — resolve edges involving leaves
        edge_types = {}
        prop_indices = {}

        for edge in edges:
            u, v = edge[0], edge[1]
            # Determine resp leg (from tail u)
            if edge in assigned_resp:
                resp_leg = assigned_resp[edge]
            elif u in leaf_set:
                resp_leg = ext_assignment.get(u)
                if resp_leg is None or resp_leg not in resp_index:
                    return
            else:
                return  # should not happen

            # Determine phys leg (from head v)
            if edge in assigned_phys:
                phys_leg = assigned_phys[edge]
            elif v in leaf_set:
                phys_leg = ext_assignment.get(v)
                if phys_leg is None or phys_leg not in phys_index:
                    return
            else:
                return  # should not happen

            # Check propagator
            ri = resp_index.get(resp_leg)
            pi = phys_index.get(phys_leg)
            if ri is None or pi is None:
                return
            if G_ft is not None and bool(SR(G_ft[ri, pi]).is_zero()):
                return

            edge_types[edge] = (resp_leg, phys_leg)
            prop_indices[edge] = (ri, pi)

        yield TypedDiagram(prediagram, dict(vert_assignment), edge_types,
                           dict(ext_assignment), prop_indices)
        return

    for resp_map, phys_map in per_vertex_options[vertex_idx]:
        new_resp = dict(assigned_resp)
        new_phys = dict(assigned_phys)
        new_resp.update(resp_map)
        new_phys.update(phys_map)

        # Early propagator check: for edges where both sides are now known
        consistent = True
        for edge in list(resp_map.keys()) + list(phys_map.keys()):
            if edge in new_resp and edge in new_phys:
                resp_leg = new_resp[edge]
                phys_leg = new_phys[edge]
                ri = resp_index.get(resp_leg)
                pi = phys_index.get(phys_leg)
                if ri is None or pi is None:
                    consistent = False; break
                if G_ft is not None and bool(SR(G_ft[ri, pi]).is_zero()):
                    consistent = False; break
            # Also check edges where one side is from a leaf
            u, v = edge[0], edge[1]
            if edge in new_resp and v in leaf_set:
                phys_leg = ext_assignment.get(v)
                if phys_leg is not None and phys_leg in phys_index:
                    resp_leg = new_resp[edge]
                    ri = resp_index.get(resp_leg)
                    pi = phys_index.get(phys_leg)
                    if ri is None or pi is None:
                        consistent = False; break
                    if G_ft is not None and bool(SR(G_ft[ri, pi]).is_zero()):
                        consistent = False; break
            if edge in new_phys and u in leaf_set:
                resp_leg = ext_assignment.get(u)
                if resp_leg is not None and resp_leg in resp_index:
                    phys_leg = new_phys[edge]
                    ri = resp_index.get(resp_leg)
                    pi = phys_index.get(phys_leg)
                    if ri is None or pi is None:
                        consistent = False; break
                    if G_ft is not None and bool(SR(G_ft[ri, pi]).is_zero()):
                        consistent = False; break

        if consistent:
            yield from _backtrack(
                prediagram, edges, ext_assignment, vert_assignment,
                ordered_internal, leaf_set, leaf_directions,
                per_vertex_options, G_ft, resp_index, phys_index,
                vertex_idx + 1, new_resp, new_phys,
            )


# ── Convenience: enumerate across all prediagrams ────────────────────────────

def enumerate_all(prediagrams, external_fields, vertex_types, source_types,
                  G_ft, resp_index, phys_index):
    """
    Enumerate typed diagrams across all prediagrams.

    Returns
    -------
    list of TypedDiagram
    """
    results = []
    for pd in prediagrams:
        for td in enumerate_typed_diagrams(pd, external_fields, vertex_types,
                                           source_types, G_ft,
                                           resp_index, phys_index):
            results.append(td)
    return results

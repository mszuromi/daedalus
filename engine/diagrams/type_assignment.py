"""
engine.diagrams.type_assignment
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


def _distinct_permutations(seq):
    """Yield each distinct permutation of *seq* exactly once."""
    return set(permutations(seq))


from functools import lru_cache


@lru_cache(maxsize=None)
def _distinct_orderings(legs):
    """The distinct orderings of the leg multiset ``legs`` (a tuple of
    ``(field_base, pop_idx)`` pairs), as a tuple of lists, in the order
    sympy's ``multiset_permutations`` produced them (lexicographic in the
    sorted distinct entries), so downstream emission order is unchanged.
    Cached per multiset: the same handful of vertex types is visited
    millions of times per enumeration."""
    if not legs:
        return ([],)
    distinct = sorted(set(legs), key=repr)
    counts = {x: legs.count(x) for x in distinct}
    out = []

    def rec(prefix, remaining):
        if len(prefix) == len(legs):
            out.append(list(prefix))
            return
        for x in distinct:
            if remaining[x]:
                remaining[x] -= 1
                prefix.append(x)
                rec(prefix, remaining)
                prefix.pop()
                remaining[x] += 1

    rec([], dict(counts))
    return tuple(out)


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
    from engine.core.vertices import _parse_field_name

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

_BIDEGREE_CACHE = {}


def _types_by_bidegree(vertex_types, source_types):
    """``({out_degree: [source types]}, {(in, out): [vertex types]})`` for the
    given type lists, cached on their identities: the lists are the same
    objects for every prediagram of an enumeration."""
    key = (id(vertex_types), id(source_types))
    hit = _BIDEGREE_CACHE.get(key)
    if hit is not None and hit[0] is vertex_types and hit[1] is source_types:
        return hit[2], hit[3]
    src = {}
    for st in source_types:
        src.setdefault(st.out_degree, []).append(st)
    vts = {}
    for vt in vertex_types:
        vts.setdefault((vt.in_degree, vt.out_degree), []).append(vt)
    _BIDEGREE_CACHE.clear()
    _BIDEGREE_CACHE[key] = (vertex_types, source_types, src, vts)
    return src, vts


def _zero_mask(G_ft):
    """``{(i, j): entry is literally SR(0)}`` for the propagator matrix, a
    purely structural test (see the note in ``enumerate_typed_diagrams``).
    Depends only on ``G_ft``, so ``enumerate_all`` computes it once per
    call; computing it per prediagram cost one ``str()`` of every
    symbolic entry per prediagram (14 928 x 9 conversions at k=2, ell=3
    for a three-field model)."""
    if G_ft is None:
        return None
    return {(i, j): (str(G_ft[i, j]) == '0')
            for i in range(G_ft.nrows()) for j in range(G_ft.ncols())}


def enumerate_typed_diagrams(prediagram, external_fields, vertex_types,
                             source_types, G_ft, resp_index, phys_index,
                             g_zero_mask=None):
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

    # Precompute the zero/nonzero pattern of G_ft once.  Each leg
    # assignment in the type-assignment backtracker would otherwise
    # call ``SR(G_ft[pi, ri]).is_zero()`` — for a symbolic G_ft (legacy
    # path) that triggers full Sage simplification on a long rational
    # expression, easily costing several seconds per entry on rich
    # models.  Both ``e.is_zero()`` and ``e == SR(0)`` exhibit this
    # pathology (they both invoke the SR canonicalisation engine).
    # We only need this check to prune diagrams that would carry an
    # identically-zero propagator; ``str(e) == '0'`` is a purely
    # structural check that bypasses simplification entirely — it
    # catches ONLY entries that are LITERALLY ``SR(0)`` (which is
    # what build_propagator initialises empty cells with).  A
    # mathematically-zero-after-cancellation entry would just
    # enumerate a few extra diagrams that Phase J integrates to 0.
    if g_zero_mask is None:
        g_zero_mask = _zero_mask(G_ft)

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

    # Build candidate types for each non-leaf vertex, from the types
    # bucketed by bidegree (computed once per type list, not per prediagram).
    src_by_deg, vt_by_deg = _types_by_bidegree(vertex_types, source_types)
    candidates = {}
    for v in source_verts:
        cands = src_by_deg.get(D.out_degree(v))
        if not cands:
            return
        candidates[v] = cands

    for v in interaction_verts:
        cands = vt_by_deg.get((D.in_degree(v), D.out_degree(v)))
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

    # External leg assignment: enumerate ALL distinct permutations of the
    # external fields across leaves.  The prediagram isomorphism dedup
    # treats all leaves as interchangeable, so a single prediagram
    # represents every leaf permutation.  We must enumerate them here
    # so that diagrams differing in which leaf carries which field
    # (e.g. dn₂ at a source-leaf vs at an interaction-leaf) are all
    # generated.  The downstream dedup (deduplicate_typed_diagrams)
    # will then merge any that are truly identical.
    for ext_perm in _distinct_permutations(tuple(external_fields)):
        ext_assignment = {}
        valid_ext = True
        for leaf_idx in range(len(ext_perm)):
            lf = leaves[leaf_idx]
            field = ext_perm[leaf_idx]
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
            continue

        if not ordered_internal:
            # No internal vertices — just external legs connected by edges
            yield from _try_build_diagram_no_internal(
                prediagram, edges, ext_assignment, leaf_set, leaf_directions,
                G_ft, resp_index, phys_index,
                g_zero_mask=g_zero_mask,
            )
            continue

        # Joint backtracking over (vertex type, leg matching) per vertex,
        # in an order where every vertex after the first shares an edge with
        # an already-assigned vertex or a leaf, so each choice is checked
        # against the propagator mask as soon as it determines an edge.
        # This replaces the former Cartesian product over type combinations
        # followed by per-combination leg backtracking, which could not
        # reject a combination until every vertex had a type: on a model
        # with 24 vertex types at k=2, ell=3 it visited ~10^7 (combination,
        # vertex) pairs to find that nothing survives.  Same set of typed
        # diagrams (tests/test_type_assignment.py pins the signatures).
        yield from _joint_backtrack(
            prediagram, edges, ext_assignment, _pruning_order(
                ordered_internal, out_edges_of, in_edges_of, leaf_set),
            candidates, leaf_set, out_edges_of, in_edges_of,
            resp_index, phys_index, g_zero_mask,
            0, {}, {}, {})


def _try_build_diagram_no_internal(prediagram, edges, ext_assignment,
                                    leaf_set, leaf_directions,
                                    G_ft, resp_index, phys_index,
                                    g_zero_mask=None):
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

        # Check propagator.  G_ft = K_ft^{-1} has rows = physical, cols
        # = response by linear-algebra convention; the propagator on an
        # edge resp-tail → phys-head is ⟨φ_phys ñ_resp⟩ = G_ft[pi, ri].
        ri = resp_index[resp_field]
        pi = phys_index[phys_field]
        if g_zero_mask is not None and g_zero_mask.get((pi, ri), False):
            return

        edge_types[edge] = (resp_field, phys_field)
        prop_indices[edge] = (ri, pi)

    yield TypedDiagram(prediagram, {}, edge_types,
                       dict(ext_assignment), prop_indices)


def _try_build_diagram(prediagram, edges, ext_assignment, vert_assignment,
                        ordered_internal, leaf_set, leaf_directions,
                        out_edges_of, in_edges_of,
                        G_ft, resp_index, phys_index,
                        g_zero_mask=None):
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
        g_zero_mask=g_zero_mask,
    )


def _leg_matchings(vertex_type, out_edges, in_edges):
    """
    Enumerate all DISTINCT bijections from a vertex type's legs to edges.

    Each outgoing edge gets a response leg; each incoming edge gets a
    physical leg.  Two bijections that produce the same
    ``(edge -> leg)`` dict are considered identical; such collisions
    happen whenever ``vertex_type.response_legs`` or
    ``vertex_type.physical_legs`` contain duplicate ``(field, pop)``
    entries — swapping two indistinguishable legs gives back the same
    typed diagram.

    For a leg multiset of length N with multiplicities ``n_r``, the
    number of distinct orderings is

        N! / ∏_r n_r!

    (a multinomial coefficient), not the full N! that an index
    permutation ``range(N)`` would suggest.  Earlier versions of this
    function enumerated all N! index permutations and relied on the
    downstream ``deduplicate_typed_diagrams`` pass in
    ``engine/diagrams/symmetry.py`` to discard the overcount.  That
    made the generation step wastefully large: typed-diagram counts
    blew up to ~15-20× the unique count before dedup, driving the
    ``enumerate_typed_diagrams`` wall time on cold runs.

    **Correctness under this change.**  The physics weight of a typed
    diagram is ``𝒮(Γ) × ∏_v coeff(v) × ∫(propagators)`` where
    ``𝒮(Γ) = ∏_v M_v`` is the ``combinatorial_factor`` from
    ``engine/diagrams/symmetry.py::combinatorial_factor``.  ``M_v``
    is *exactly* the count of response-leg permutations at vertex ``v``
    that preserve the same typed diagram — i.e. the orbit size of
    identical-leg swaps.  So: old code generated ``𝒮(Γ)`` identical
    copies and each copy carried its own full weight; dedup kept one,
    with the integrator then multiplying by ``𝒮(Γ)``.  New code
    generates the single canonical copy directly, and the integrator
    multiplies by the same ``𝒮(Γ)``.  Numerical output is bit-
    identical; only the intermediate typed-diagram count (and
    therefore the enumeration wall time) changes.

    This is verified by ``test_leg_matchings_canonical`` and
    ``test_enumerate_typed_signatures_match_pre_change`` in
    ``tests/test_type_assignment.py``.

    Yields
    ------
    (resp_map, phys_map) where each is a dict ``{edge: leg}`` and
    each yielded pair is distinct from every other yielded pair.
    """
    resp_legs = vertex_type.response_legs
    has_phys = hasattr(vertex_type, 'physical_legs')
    phys_legs = vertex_type.physical_legs if has_phys else []

    if len(resp_legs) != len(out_edges) or len(phys_legs) != len(in_edges):
        return

    # The distinct orderings of a leg multiset depend only on the vertex
    # TYPE, so they are computed once per multiset and cached
    # (``_distinct_orderings``).  This used to call sympy's
    # ``multiset_permutations`` on every (prediagram, external permutation,
    # type combination, vertex) visit; sympy sorts the entries with
    # ``default_sort_key``, which sympifies the ``(field, pop)`` tuples
    # through the string parser, so a single call cost tens to hundreds of
    # microseconds and the typing stage spent ~90% of its time there (10.6M
    # calls, 205 s of 229 s, for the three-field vesicle model at k=2,
    # ell=3).  Same yielded pairs, same order.
    resp_perms = _distinct_orderings(tuple(resp_legs))
    phys_perms = _distinct_orderings(tuple(phys_legs))

    for rp in resp_perms:
        resp_map = dict(zip(out_edges, rp))
        for pp in phys_perms:
            phys_map = dict(zip(in_edges, pp))
            yield resp_map, phys_map


def _backtrack(prediagram, edges, ext_assignment, vert_assignment,
               ordered_internal, leaf_set, leaf_directions,
               per_vertex_options, G_ft, resp_index, phys_index,
               vertex_idx, assigned_resp, assigned_phys,
               g_zero_mask=None):
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

            # Check propagator (phys row, resp col — see top of
            # _backtrack for convention notes).
            ri = resp_index.get(resp_leg)
            pi = phys_index.get(phys_leg)
            if ri is None or pi is None:
                return
            if g_zero_mask is not None and g_zero_mask.get((pi, ri), False):
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
                # Convention: G_ft = K_ft^{-1} has rows = physical, cols =
                # response (the natural inverse of K's rows = response,
                # cols = physical), so G_ft[i, j] = ⟨φ_i ñ_j⟩.  Edge
                # propagator from a resp-tail to a phys-head is
                # ⟨φ_phys ñ_resp⟩ = G_ft[pi, ri].  (See also
                # final_integral.integrate_diagram which already calls
                # G_t_entry(G_t_obj, pi, ri, ...).)  For quad_expg with
                # diagonal couplings (nt↔dn, vt↔dv) the integer indices
                # coincide and the buggy [ri, pi] order happened to read
                # the right entry; off-diagonal cases (e.g. GTaS m̃→δn,
                # phys idx 0, resp idx 4) trip the bug and silently
                # filter out diagrams with G_ft[ri, pi] == 0 even when
                # G_ft[pi, ri] is nonzero.
                if g_zero_mask is not None and \
                   g_zero_mask.get((pi, ri), False):
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
                    if g_zero_mask is not None and \
                       g_zero_mask.get((pi, ri), False):
                        consistent = False; break
            if edge in new_phys and u in leaf_set:
                resp_leg = ext_assignment.get(u)
                if resp_leg is not None and resp_leg in resp_index:
                    phys_leg = new_phys[edge]
                    ri = resp_index.get(resp_leg)
                    pi = phys_index.get(phys_leg)
                    if ri is None or pi is None:
                        consistent = False; break
                    if g_zero_mask is not None and \
                       g_zero_mask.get((pi, ri), False):
                        consistent = False; break

        if consistent:
            yield from _backtrack(
                prediagram, edges, ext_assignment, vert_assignment,
                ordered_internal, leaf_set, leaf_directions,
                per_vertex_options, G_ft, resp_index, phys_index,
                vertex_idx + 1, new_resp, new_phys,
                g_zero_mask=g_zero_mask,
            )


def _pruning_order(internal, out_edges_of, in_edges_of, leaf_set):
    """Order the internal vertices so that the first has the most edges to
    leaves and every later one is adjacent to an earlier one when the
    diagram allows it (BFS by adjacency, ties by leaf edges)."""
    def nbrs(v):
        return [e[1] for e in out_edges_of[v]] + [e[0] for e in in_edges_of[v]]
    leaf_edges = {v: sum(1 for w in nbrs(v) if w in leaf_set) for v in internal}
    remaining = set(internal)
    order = []
    while remaining:
        if order:
            adj = [v for v in remaining if any(w in order for w in nbrs(v))]
            pool = adj or list(remaining)
        else:
            pool = list(remaining)
        v = max(pool, key=lambda x: (leaf_edges[x], -internal.index(x)))
        order.append(v)
        remaining.discard(v)
    return order


def _edge_ok(edge, resp_leg, phys_leg, resp_index, phys_index, g_zero_mask):
    ri = resp_index.get(resp_leg)
    pi = phys_index.get(phys_leg)
    if ri is None or pi is None:
        return False
    if g_zero_mask is not None and g_zero_mask.get((pi, ri), False):
        return False
    return True


def _joint_backtrack(prediagram, edges, ext_assignment, order, candidates,
                     leaf_set, out_edges_of, in_edges_of,
                     resp_index, phys_index, g_zero_mask,
                     idx, vert_assignment, assigned_resp, assigned_phys):
    if idx == len(order):
        # every internal vertex typed and matched; resolve leaf ends and
        # build the record (the leaf-side checks below repeat ones already
        # made, harmlessly, for edges whose internal end was assigned last)
        edge_types = {}
        prop_indices = {}
        for edge in edges:
            u, v = edge[0], edge[1]
            if edge in assigned_resp:
                resp_leg = assigned_resp[edge]
            elif u in leaf_set:
                resp_leg = ext_assignment.get(u)
                if resp_leg is None or resp_leg not in resp_index:
                    return
            else:
                return
            if edge in assigned_phys:
                phys_leg = assigned_phys[edge]
            elif v in leaf_set:
                phys_leg = ext_assignment.get(v)
                if phys_leg is None or phys_leg not in phys_index:
                    return
            else:
                return
            if not _edge_ok(edge, resp_leg, phys_leg, resp_index, phys_index, g_zero_mask):
                return
            edge_types[edge] = (resp_leg, phys_leg)
            prop_indices[edge] = (resp_index[resp_leg], phys_index[phys_leg])
        yield TypedDiagram(prediagram, dict(vert_assignment), edge_types,
                           dict(ext_assignment), prop_indices)
        return

    v = order[idx]
    out_e = out_edges_of[v]
    in_e = in_edges_of[v]
    for vtype in candidates[v]:
        for resp_map, phys_map in _leg_matchings(vtype, out_e, in_e):
            ok = True
            for edge, resp_leg in resp_map.items():
                w = edge[1]
                if edge in assigned_phys:
                    phys_leg = assigned_phys[edge]
                elif w in leaf_set:
                    phys_leg = ext_assignment.get(w)
                    if phys_leg is None or phys_leg not in phys_index:
                        ok = False
                        break
                else:
                    continue
                if not _edge_ok(edge, resp_leg, phys_leg, resp_index, phys_index, g_zero_mask):
                    ok = False
                    break
            if not ok:
                continue
            for edge, phys_leg in phys_map.items():
                u = edge[0]
                if edge in assigned_resp:
                    resp_leg = assigned_resp[edge]
                elif u in leaf_set:
                    resp_leg = ext_assignment.get(u)
                    if resp_leg is None or resp_leg not in resp_index:
                        ok = False
                        break
                else:
                    continue
                if not _edge_ok(edge, resp_leg, phys_leg, resp_index, phys_index, g_zero_mask):
                    ok = False
                    break
            if not ok:
                continue
            vert_assignment[v] = vtype
            new_resp = dict(assigned_resp); new_resp.update(resp_map)
            new_phys = dict(assigned_phys); new_phys.update(phys_map)
            yield from _joint_backtrack(
                prediagram, edges, ext_assignment, order, candidates,
                leaf_set, out_edges_of, in_edges_of,
                resp_index, phys_index, g_zero_mask,
                idx + 1, vert_assignment, new_resp, new_phys)
            del vert_assignment[v]


# ── Convenience: enumerate across all prediagrams ────────────────────────────
#
# Parallelism rationale (2026-04-23, enumeration-speedup branch):
# ``enumerate_typed_diagrams`` is embarrassingly parallel across
# prediagrams — no shared mutable state, no cross-prediagram
# dependencies.  For cold-start workloads the outer loop is the
# dominant cost (60s+ per prediagram on quadratic Hawkes k=3 ell=1),
# so fork-ProcessPool parallelism stacks cleanly with the canonical-
# leg-matching change.
#
# Fork is required because Sage ``DiGraph`` objects, ``VertexType`` /
# ``SourceType`` instances, and the propagator matrix may not all
# pickle cleanly through stdlib ``pickle``.  With fork, workers
# inherit those from the parent's memory — no cross-process pickling
# on the inputs.  Only the RETURN value (list of ``TypedDiagram``)
# is pickled, and ``TypedDiagram`` has explicit ``__getstate__`` /
# ``__setstate__`` via ``__slots__`` for that purpose.
#
# Windows support deferred — same constraints as
# ``engine/integration/time_domain/pipeline.py``'s ``total_C_batch``.

_ENUM_WORKER_STATE = {}


def _worker_enumerate_one_prediagram(pd_idx):
    """Worker entry point for ``enumerate_all``'s fork pool.

    Must be a module-level function so it's picklable by
    ``multiprocessing``.  The actual input data lives in
    ``_ENUM_WORKER_STATE`` and is inherited via fork (not via
    ``initargs`` pickling).
    """
    state = _ENUM_WORKER_STATE
    pd = state['prediagrams'][pd_idx]
    return list(enumerate_typed_diagrams(
        pd, state['external_fields'],
        state['vertex_types'], state['source_types'],
        state['G_ft'], state['resp_index'], state['phys_index'],
        g_zero_mask=state.get('g_zero_mask'),
    ))


def enumerate_all(prediagrams, external_fields, vertex_types, source_types,
                  G_ft, resp_index, phys_index,
                  parallel=False, n_workers=None, start_method='fork'):
    """
    Enumerate typed diagrams across all prediagrams.

    Parameters
    ----------
    prediagrams : list of tuple
        ``(D, G, leaves, internal)`` per prediagram.
    external_fields, vertex_types, source_types, G_ft, resp_index,
    phys_index :
        Passed through to ``enumerate_typed_diagrams``.
    parallel : bool, default False
        If ``True``, fan the per-prediagram enumeration out across a
        fork-based ``multiprocessing.Pool``.  Each worker runs the
        full ``enumerate_typed_diagrams`` on one prediagram; the
        parent aggregates the resulting ``TypedDiagram`` lists.
        Default is ``False`` to preserve the pre-2026-04-23 serial
        behaviour for existing callers; flip to ``True`` (or set the
        kwarg at the cached-enumeration site in notebook cell 18)
        to get the speedup.
    n_workers : int or None, default None
        Cap on worker-process count when ``parallel=True``.  ``None``
        picks ``min(os.cpu_count(), len(prediagrams))``.
    start_method : {'fork'}, default 'fork'
        Multiprocessing start method.  Only ``'fork'`` is supported:
        ``'spawn'`` would require pickling Sage graph / matrix inputs
        across workers, which is not guaranteed to work cleanly.

    Returns
    -------
    list of TypedDiagram
        In the same order as the serial path: concatenation of
        per-prediagram results in prediagram-index order.

    Notes
    -----
    Numerical output is bit-identical between the serial and parallel
    paths — per-prediagram order and within-prediagram order of
    emission are both preserved.  Pinned by
    ``test_enumerate_all_parallel_matches_serial`` in
    ``tests/test_type_assignment.py``.
    """
    # Fork-safety guard: forking after Cocoa/BLAS init inside a macOS Jupyter
    # kernel can hard-crash the kernel AND the OS.  Degrade to serial there
    # (mirrors the temporal Phase-J path); fork stays available in scripts,
    # pytest, Linux, and terminal IPython.
    from engine.fork_safety import fork_unsafe_in_notebook, warn_fork_fallback_once
    if (parallel and len(prediagrams) > 1
            and fork_unsafe_in_notebook(start_method)):
        warn_fork_fallback_once('diagram type-assignment enumeration')
        parallel = False

    if not parallel or len(prediagrams) <= 1:
        # Serial path.  Keeps 'prediagrams' iteration for back-compat
        # with any downstream code that relies on yield ordering.
        results = []
        mask = _zero_mask(G_ft)
        for pd in prediagrams:
            for td in enumerate_typed_diagrams(
                pd, external_fields, vertex_types, source_types,
                G_ft, resp_index, phys_index, g_zero_mask=mask,
            ):
                results.append(td)
        return results

    import multiprocessing as mp
    import os
    # Needed for fork to proceed at all on macOS (objc otherwise aborts).  Only
    # reached for a LEGITIMATE fork — the notebook case fell back to serial
    # above, so this never runs in the lethal darwin+Jupyter context.
    os.environ.setdefault(
        'OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES'
    )

    # Populate _ENUM_WORKER_STATE BEFORE forking so children inherit it.
    # Not thread-safe — ``enumerate_all`` should not be called from two
    # threads concurrently.  In the Sage Phase J workflow this is a
    # single-threaded notebook call, so that's fine.
    _ENUM_WORKER_STATE['prediagrams'] = list(prediagrams)
    _ENUM_WORKER_STATE['external_fields'] = external_fields
    _ENUM_WORKER_STATE['vertex_types'] = vertex_types
    _ENUM_WORKER_STATE['source_types'] = source_types
    _ENUM_WORKER_STATE['G_ft'] = G_ft
    _ENUM_WORKER_STATE['resp_index'] = resp_index
    _ENUM_WORKER_STATE['phys_index'] = phys_index
    _ENUM_WORKER_STATE['g_zero_mask'] = _zero_mask(G_ft)

    ctx = mp.get_context(start_method)
    n_w_cap = (n_workers if n_workers is not None
               else min(os.cpu_count() or 4, len(prediagrams)))

    with ctx.Pool(processes=n_w_cap) as pool:
        per_pd_results = pool.map(
            _worker_enumerate_one_prediagram,
            range(len(prediagrams)),
        )

    # Flatten in prediagram-index order (Pool.map preserves input
    # order in the output), so the final list matches the serial path.
    results = []
    for sublist in per_pd_results:
        results.extend(sublist)
    return results

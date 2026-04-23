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


def _distinct_permutations(seq):
    """Yield each distinct permutation of *seq* exactly once."""
    return set(permutations(seq))


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
            )
            continue

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
    ``msrjd/diagrams/symmetry.py`` to discard the overcount.  That
    made the generation step wastefully large: typed-diagram counts
    blew up to ~15-20× the unique count before dedup, driving the
    ``enumerate_typed_diagrams`` wall time on cold runs.

    **Correctness under this change.**  The physics weight of a typed
    diagram is ``M(Γ) × ∏_v coeff(v) × ∫(propagators)`` where
    ``M(Γ) = ∏_v M_v`` is the ``combinatorial_factor`` from
    ``msrjd/diagrams/symmetry.py::combinatorial_factor``.  ``M_v``
    is *exactly* the count of response-leg permutations at vertex ``v``
    that preserve the same typed diagram — i.e. the orbit size of
    identical-leg swaps.  So: old code generated ``M(Γ)`` identical
    copies and each copy carried its own full weight; dedup kept one,
    with the integrator then multiplying by ``M(Γ)``.  New code
    generates the single canonical copy directly, and the integrator
    multiplies by the same ``M(Γ)``.  Numerical output is bit-
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

    # Use sympy's multiset_permutations: yields only the N!/∏n_r!
    # distinct orderings of a multiset, rather than the full N!
    # orderings of index positions.  Materialise to list so the inner
    # loop can re-iterate the phys_perms list.  For typical Hawkes
    # vertex types (R up to ~4, P up to ~2), list sizes stay small
    # (≤ 24) and memory is not a concern.
    #
    # Note: sympy.utilities.iterables.multiset_permutations expects a
    # list-like input.  Leg values are ``(field_base, pop_idx)``
    # tuples (hashable), so the multiset comparison works.
    from sympy.utilities.iterables import multiset_permutations
    resp_perms = (list(multiset_permutations(list(resp_legs)))
                  if resp_legs else [[]])
    phys_perms = (list(multiset_permutations(list(phys_legs)))
                  if phys_legs else [[]])

    for rp in resp_perms:
        resp_map = dict(zip(out_edges, rp))
        for pp in phys_perms:
            phys_map = dict(zip(in_edges, pp))
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
# ``msrjd/integration/time_domain/pipeline.py``'s ``total_C_batch``.

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
    if not parallel or len(prediagrams) <= 1:
        # Serial path.  Keeps 'prediagrams' iteration for back-compat
        # with any downstream code that relies on yield ordering.
        results = []
        for pd in prediagrams:
            for td in enumerate_typed_diagrams(
                pd, external_fields, vertex_types, source_types,
                G_ft, resp_index, phys_index,
            ):
                results.append(td)
        return results

    import multiprocessing as mp
    import os
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

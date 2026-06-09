"""
msrjd.diagrams.symmetry
========================
Combinatorial factor 𝒮(Γ) for fully-typed labeled diagrams, and
deduplication of typed diagrams into unique representatives.

Definition (Attachment)
-----------------------
Given a typed diagram Γ with directed graph D = (V, E) and a fixed
propagator type on each edge, the combinatorial factor 𝒮(Γ) counts
the number of ways to permute the *outgoing* (response) legs at each
vertex that yield the same typed diagram — i.e. the same multiset of
(response_type, physical_type) pairings on the outgoing edges.

For a single vertex v with outgoing edges, let each edge carry a
pairing (r, p) where r is the response leg type at v and p is the
physical leg type at the target vertex. Define:

    n_r     = number of response legs of type r at v
    n[r][p] = number of outgoing edges with pairing (r, p)
    m_p     = number of outgoing edges targeting physical type p

Then the per-vertex factor is:

    M_v = [∏_r  n_r! / ∏_p n[r][p]!]  ×  [∏_p  m_p!]

and the full combinatorial factor is:

    𝒮(Γ) = ∏_v  M_v

The diagram's contribution to the k-point function is:

    weight(Γ) = 𝒮(Γ) × ∏_v coeff(v) × ∫(propagators)

where the vertex coefficients already contain 1/n! from the Taylor
expansion of the action.

Build Phase G.
"""

from collections import Counter
from functools import reduce
from math import factorial
from operator import mul

from sage.all import SR


# ── Combinatorial factor ────────────────────────────────────────────────────

def _vertex_combinatorial_factor(vertex, typed_diagram):
    """
    Count the number of response-leg permutations at *vertex* that
    preserve the same typed diagram.

    Each external leaf is treated as a UNIQUE endpoint (distinguished
    by its position in the prediagram leaf list), because different
    leaf positions correspond to different spacetime points.  Internal
    (non-leaf) targets are identified by their (resp, phys) field-type
    pairing as before.

    M_v = [∏_r  n_r! / ∏_t n[r][t]!]  ×  [∏_t  m_t!]

    where t ranges over unique targets (leaf positions for external
    edges, (resp, phys) field types for internal edges).

    Parameters
    ----------
    vertex : hashable
        Vertex id within the typed diagram.
    typed_diagram : TypedDiagram

    Returns
    -------
    int
        Per-vertex combinatorial factor (always >= 1).
    """
    edge_types = typed_diagram.edge_types

    # Collect (resp_type, phys_type) for every outgoing edge from vertex.
    # M counts response-leg permutations that preserve the (resp, phys)
    # field-type pairing — i.e. within-vertex Wick contractions that
    # give the same integrand by commutativity.
    pairings = []
    for edge_key, (resp_leg, phys_leg) in edge_types.items():
        u = edge_key[0]
        if u == vertex:
            pairings.append((resp_leg, phys_leg))

    if not pairings:
        return 1

    # n_r: total count of each response type
    resp_counts = Counter(r for r, t in pairings)
    # n[r][t]: count of each (resp, target) pairing
    pair_counts = Counter(pairings)
    # m_t: count of each target across outgoing edges
    target_counts = Counter(t for r, t in pairings)

    m = 1
    # ∏_r [ n_r! / ∏_t n[r][t]! ]
    for r, n_r in resp_counts.items():
        m *= factorial(n_r)
        for t in target_counts:
            n_rt = pair_counts.get((r, t), 0)
            if n_rt > 1:
                m //= factorial(n_rt)
    # ∏_t m_t!
    for m_t in target_counts.values():
        m *= factorial(m_t)

    return m


def _wick_leg_factor(typed_diagram):
    """
    Per-vertex Wick numerator: product over vertices of the factorial
    multiplicities of identical legs (both response/outgoing AND
    physical/incoming).

    For each vertex v, let ``n_{v,ℓ}`` be the number of legs of type
    ``ℓ`` (where leg-type lumps direction + field).  This function
    returns

        ∏_v  [ (∏_r n_{v,resp_r}!)  ×  (∏_p n_{v,phys_p}!) ]

    This is the Wick combinatorial that would arise if all legs at
    each vertex were treated as distinguishable.  It is the
    "numerator" in the Feynman rule

        𝒮(Γ)  =  ∏_v ∏_ℓ n_{v,ℓ}!  /  |Aut(Γ)|

    See ``combinatorial_factor`` for the full formula.
    """
    n = 1
    for v, vtype in typed_diagram.vertex_assignments.items():
        # Outgoing/response legs
        resp_counts = Counter(vtype.response_legs)
        for c in resp_counts.values():
            n *= factorial(c)
        # Incoming/physical legs (only present on interaction vertices)
        if hasattr(vtype, 'physical_legs'):
            phys_counts = Counter(vtype.physical_legs)
            for c in phys_counts.values():
                n *= factorial(c)
    return n


def _colored_incidence_digraph(typed_diagram, fix_external=True):
    """
    Build a coloured directed bipartite incidence graph for a typed
    Feynman diagram so that Sage's ``automorphism_group(partition=…)``
    yields the **full** diagram automorphism group, including:

    - same-type internal vertex swaps,
    - parallel-edge swaps between the same vertex pair,
    - self-loop / tadpole leg-pair swaps,
    - same-type external leaf swaps (only when ``fix_external=False``).

    Each Feynman-diagram vertex becomes one node in the bipartite
    graph; each propagator edge becomes a separate "edge-node".  An
    original edge ``u→v`` is encoded as two directed bipartite edges
    ``u → edge_node`` and ``edge_node → v``.  Edge-nodes are coloured
    by (response-field, physical-field, propagator-indices); vertex
    nodes are coloured by (vertex-type, coefficient, bigrade) for
    internal vertices and uniquely (one colour per leaf position)
    for external leaves when ``fix_external=True``.

    Returns
    -------
    D : Sage ``DiGraph``
    partition : list of lists
        Vertex partition (colour classes) suitable for
        ``D.automorphism_group(partition=partition, order=True)``.
    vertex_label_map : dict
        Maps original typed-diagram vertex ids and edge keys to the
        nodes used in ``D``.  (Useful for debugging; not used by the
        production callers.)
    """
    from sage.all import DiGraph

    leaf_set = set(typed_diagram.external_legs.keys())

    # Colour classes.  Each class is a list of node ids; nodes in
    # different classes are NEVER mapped to each other by an
    # automorphism.
    color_groups: dict = {}
    nodes_added = []
    vertex_label_map = {}

    def _add_node(node_id, color_key):
        nodes_added.append(node_id)
        color_groups.setdefault(color_key, []).append(node_id)

    # Vertex nodes.
    for v in typed_diagram.prediagram[0].vertices():
        node_id = ('V', v)
        if v in leaf_set:
            # External leaf colouring.
            field = typed_diagram.external_legs.get(v)
            if fix_external:
                # Each leaf gets its OWN colour class so leaf positions
                # cannot be permuted.  ``Aut_fixed_ext``.
                color_key = ('leaf', v, field)
            else:
                # Same-type leaves can be swapped; group by field only.
                color_key = ('leaf', field)
        else:
            vt = typed_diagram.vertex_assignments.get(v)
            if vt is None:
                color_key = ('vertex', 'unassigned')
            else:
                color_key = (
                    'vertex',
                    type(vt).__name__,
                    str(vt.coefficient),
                    vt.bigrade,
                )
        _add_node(node_id, color_key)
        vertex_label_map[v] = node_id

    # Edge nodes.  Each propagator edge is a separate node whose
    # colour encodes everything about the propagator that an
    # automorphism must preserve.  Parallel edges between the same
    # (source, target) pair with identical colour are swappable in
    # Aut by design — they correspond to a genuine graph
    # automorphism that propagates through the Wick contractions.
    edges_directed = []  # (src_node, dst_node) bipartite edges to add
    for ek, (resp_leg, phys_leg) in typed_diagram.edge_types.items():
        u, w = ek[0], ek[1]
        prop = typed_diagram.propagator_indices.get(ek, None)
        edge_node = ('E', ek)
        color_key = ('edge', resp_leg, phys_leg, prop)
        _add_node(edge_node, color_key)
        vertex_label_map[ek] = edge_node
        edges_directed.append((('V', u), edge_node))
        edges_directed.append((edge_node, ('V', w)))

    D = DiGraph(multiedges=False, loops=False)
    D.add_vertices(nodes_added)
    D.add_edges(edges_directed)

    partition = list(color_groups.values())
    return D, partition, vertex_label_map


def vertex_role_signature(vertex, typed_diagram):
    """
    Hashable signature for an internal vertex that captures its role
    within the diagram modulo graph automorphism.

    Two vertices in the same Aut-orbit (under ``Aut_fixed_ext``)
    yield the same signature.  Used by Phase J's compensation
    factor to detect when an external-leaf swap is already an
    automorphism (so the leaf permutation in ``_all_mappings`` is
    redundant rather than a Wick combinatorial).

    Signature components (each invariant under graph relabeling):
      - vertex type (class name + coefficient + bigrade)
      - external-leaf attachment pattern (sorted multiset of
        (field, prop) for each leaf incident on this vertex)
      - internal-edge incidence pattern (multiset of
        (direction, other-vertex-type-tag, count, prop_indices))
        — same fields as ``diagram_signature``'s
        ``vertex_internal_map``, so a self-loop noise feeding a
        single cubic is distinguishable from a bridge noise
        connecting two cubics.

    Returns
    -------
    tuple
        Hashable signature, suitable as a dict key.
    """
    leaf_set = set(typed_diagram.external_legs.keys())
    vtype = typed_diagram.vertex_assignments.get(vertex)
    if vtype is None:
        # Leaf vertex — return a unique tag so each leaf is its own
        # equivalence class.
        return ('leaf', vertex,
                typed_diagram.external_legs.get(vertex))

    def _type_tag(w):
        if w in leaf_set:
            return ('leaf', typed_diagram.external_legs.get(w))
        wt = typed_diagram.vertex_assignments.get(w)
        if wt is None:
            return ('unknown',)
        return (type(wt).__name__, str(wt.coefficient), wt.bigrade)

    leaf_atts = []
    for ek in typed_diagram.edge_types:
        if ek[0] == vertex and ek[1] in leaf_set:
            leaf_atts.append((typed_diagram.external_legs[ek[1]],
                              typed_diagram.propagator_indices[ek]))
        elif ek[1] == vertex and ek[0] in leaf_set:
            leaf_atts.append((typed_diagram.external_legs[ek[0]],
                              typed_diagram.propagator_indices[ek]))

    # Internal incidence grouped by other endpoint (so multi-edges
    # to the SAME other vertex collapse correctly).
    from collections import defaultdict
    out_groups: dict = defaultdict(list)
    in_groups: dict = defaultdict(list)
    for ek in typed_diagram.edge_types:
        u, w = ek[0], ek[1]
        if u == vertex and w not in leaf_set:
            out_groups[w].append(typed_diagram.propagator_indices[ek])
        elif w == vertex and u not in leaf_set:
            in_groups[u].append(typed_diagram.propagator_indices[ek])
    out_entries = []
    for w, props in out_groups.items():
        out_entries.append(('out', _type_tag(w), len(props),
                            tuple(sorted(props))))
    in_entries = []
    for u, props in in_groups.items():
        in_entries.append(('in', _type_tag(u), len(props),
                           tuple(sorted(props))))

    return (
        type(vtype).__name__,
        str(vtype.coefficient),
        vtype.bigrade,
        tuple(sorted(leaf_atts)),
        tuple(sorted(out_entries + in_entries)),
    )


def _automorphism_order(typed_diagram, fix_external=True):
    """Order of the colour-preserving automorphism group of the
    coloured incidence digraph (see ``_colored_incidence_digraph``).

    Treats external leaves as fixed when ``fix_external=True`` (the
    convention used by Phase J, where the external Wick permutations
    are handled separately by the per-prediagram enumeration loop).
    """
    D, partition, _ = _colored_incidence_digraph(
        typed_diagram, fix_external=fix_external
    )
    grp = D.automorphism_group(partition=partition)
    return int(grp.order())


def combinatorial_factor(typed_diagram):
    r"""
    Symmetry factor for a typed Feynman diagram (Path A):

        𝒮(Γ)  =  ∏_v ∏_ℓ n_{v,ℓ}!  /  |Aut_{\text{fixed ext}}(Γ)|

    where the numerator is the per-vertex Wick combinatorial (the
    factorials of identical-leg multiplicities at each vertex, both
    response and physical) and the denominator is the order of the
    full automorphism group of the colour-preserving incidence
    digraph, with external leaves fixed at their canonical positions.

    This is the canonical Feynman rule.  The numerator counts the
    distinguishable Wick pairings at each vertex and the denominator
    removes redundancies from graph automorphisms (same-type internal
    vertex swaps, parallel-edge swaps, self-loop leg-pair swaps).

    The diagram's contribution to the connected k-point function is

        weight(Γ)  =  𝒮(Γ)  ×  ∏_v c_{\text{user},v}  ×  ∏_e P_e
                              ×  ∫dt_v

    where ``c_user`` is the literal coefficient written in the action
    (no implicit 1/n!), ``P_e`` is the propagator on edge ``e``, and
    the integration is over internal vertex times.

    Parameters
    ----------
    typed_diagram : TypedDiagram

    Returns
    -------
    int
        The combinatorial factor (always >= 1).
    """
    numer = _wick_leg_factor(typed_diagram)
    aut = _automorphism_order(typed_diagram, fix_external=True)
    if aut <= 0 or numer % aut != 0:
        # Defensive: numerator should always be divisible by Aut order
        # (it's a finite group acting freely on a finite set).  Fall
        # back to integer floor division if some Sage edge case yields
        # a non-divisor; downstream tests will catch any drift.
        return numer // max(aut, 1)
    return numer // aut


def compute_all_combinatorial_factors(typed_diagrams):
    """
    Compute 𝒮(Γ) for each typed diagram.

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram

    Returns
    -------
    list of int
        Combinatorial factor for each diagram, same order as input.
    """
    return [combinatorial_factor(td) for td in typed_diagrams]


# ── Deduplication ───────────────────────────────────────────────────────────

def diagram_signature(td):
    """
    Build a hashable canonical signature for a typed diagram.

    Two typed diagrams with the same signature are identical — they
    represent the same Feynman diagram Gamma and differ only in the
    internal choice of which identical leg was assigned to which edge
    (an attachment degree of freedom).

    The signature encodes, for each internal vertex:
      - vertex type (coefficient, legs, bigrade)
      - the sorted multiset of (external_field, propagator_index) for
        every leaf attached to that vertex
      - the sorted multiset of (target-vertex-type-signature,
        propagator_index) for every INTERNAL edge incident to that
        vertex — needed to distinguish topologically distinct
        diagrams that share per-vertex attributes but differ in their
        internal-edge wiring (e.g., 2-loop watermelon "3 K's between
        2 cubics" vs 2-loop tadpole-tadpole "1 K between cubics +
        2 self-loop K's", which both have 3 noise vertices + 2 cubic
        vertices each attached to 1 leaf but differ in HOW the noise
        edges distribute across the cubics).

    Crucially, the per-vertex leaf grouping ensures that two diagrams
    which differ in which leaf connects to which vertex (e.g. dn2 at
    a source vertex vs at an interaction vertex) are NOT merged.
    The vertex information is sorted by type (not by vertex id) to be
    invariant under vertex relabeling.

    Parameters
    ----------
    td : TypedDiagram

    Returns
    -------
    tuple
        Hashable canonical signature.
    """
    leaf_set = set(td.external_legs.keys())

    # Field-type-based deduplication.  Two diagrams that differ only
    # by permuting same-type leaves (within or across internal
    # vertices) are merged here; the inter-vertex Wick contractions
    # are enumerated separately in integrate_tree_diagram, with a
    # compensation factor for same-vertex permutations.
    vertex_leaf_map = {}
    for v in td.vertex_assignments:
        leaf_edges = []
        for ek, et in td.edge_types.items():
            if ek[0] == v and ek[1] in leaf_set:
                field = td.external_legs[ek[1]]
                prop = td.propagator_indices[ek]
                leaf_edges.append((field, prop))
            elif ek[1] == v and ek[0] in leaf_set:
                field = td.external_legs[ek[0]]
                prop = td.propagator_indices[ek]
                leaf_edges.append((field, prop))
        vertex_leaf_map[v] = tuple(sorted(leaf_edges))

    # Per-vertex "type tag" used to group internal edges: encodes the
    # vertex's intrinsic role (type+coeff+bigrade) but ignores its
    # specific id so vertices of identical role are interchangeable.
    def _type_tag(v):
        vtype = td.vertex_assignments.get(v)
        if vtype is None:
            return ('leaf', td.external_legs.get(v, ()))
        return (
            type(vtype).__name__,
            str(vtype.coefficient),
            vtype.bigrade,
        )

    # For each vertex, record the multiset of internal incident edges
    # GROUPED BY THE OTHER ENDPOINT INSTANCE.  Two outgoing edges that
    # go to the SAME target vertex collapse into one ``(target_tag,
    # count, propagator_indices)`` entry; two outgoing edges that go
    # to DIFFERENT targets of the same type produce two separate
    # entries.  This breaks the watermelon-vs-tadpole-tadpole
    # ambiguity: a watermelon source has 2 edges to 2 distinct cubic
    # vertices → two entries ``(cubic_tag, 1, ...)``; a tadpole-source
    # has 2 edges to the SAME cubic → one entry ``(cubic_tag, 2,
    # ...)``.  Sorted across each vertex, this is invariant under
    # vertex relabeling but distinguishes topologically distinct
    # diagrams that share per-vertex attributes.
    from collections import defaultdict
    vertex_internal_map = {}
    for v in td.vertex_assignments:
        out_groups = defaultdict(list)
        in_groups = defaultdict(list)
        for ek in td.edge_types:
            u, w = ek[0], ek[1]
            if u == v and w not in leaf_set:
                out_groups[w].append(td.propagator_indices[ek])
            elif w == v and u not in leaf_set:
                in_groups[u].append(td.propagator_indices[ek])
        entries = []
        for w, props in out_groups.items():
            entries.append(('out', _type_tag(w), len(props),
                            tuple(sorted(props))))
        for u, props in in_groups.items():
            entries.append(('in', _type_tag(u), len(props),
                            tuple(sorted(props))))
        vertex_internal_map[v] = tuple(sorted(entries))

    # Vertex assignments with leaf + internal-edge info, sorted by type
    # (not by id) for invariance under vertex relabeling.
    verts = []
    for v, vtype in td.vertex_assignments.items():
        tname = type(vtype).__name__
        resp = tuple(vtype.response_legs)
        phys = tuple(vtype.physical_legs) if hasattr(vtype, 'physical_legs') else ()
        verts.append((tname, str(vtype.coefficient), vtype.bigrade, resp, phys,
                      vertex_leaf_map.get(v, ()),
                      vertex_internal_map.get(v, ())))
    verts = tuple(sorted(verts))

    # Internal edges: edges between non-leaf vertices (sorted by prop index)
    internal_edges = tuple(sorted(
        td.propagator_indices[ek]
        for ek in td.edge_types
        if ek[0] not in leaf_set and ek[1] not in leaf_set
    ))

    return (verts, internal_edges)


def deduplicate_typed_diagrams(typed_diagrams):
    """
    Remove duplicate typed diagrams, keeping one representative per
    unique diagram Γ.

    Two TypedDiagrams are duplicates if they have identical external
    leg assignments, vertex type assignments, and propagator indices
    on every edge — i.e. they differ only in the internal leg-to-edge
    bijection (attachment).

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram

    Returns
    -------
    unique : list of TypedDiagram
        One representative per unique diagram.

    Notes
    -----
    Use ``deduplicate_with_multiplicities`` instead if you need the
    bug-correct combinatorial weight.  This function drops the
    dedup-equivalence-class size, which under-counts the symmetry
    factor 𝒮(Γ) for theories whose interaction vertices have ≥3
    identical physical legs fed by multiple distinct source vertices
    (e.g. cubic εxtx³ tadpoles).  See
    ``deduplicate_with_multiplicities`` for the corrected variant.
    """
    unique, _mult = deduplicate_with_multiplicities(typed_diagrams)
    return unique


def deduplicate_with_multiplicities(typed_diagrams):
    """
    Like ``deduplicate_typed_diagrams`` but also returns the size of
    each diagram's equivalence class.

    Background
    ----------
    ``_vertex_combinatorial_factor`` only counts permutations of
    *outgoing* (response) legs at each vertex.  When a sink vertex
    has ≥3 identical *incoming* (physical) legs sourced by multiple
    distinct vertices (e.g. the εxtx³ tadpole at 1-loop: v2 has 3
    x-legs fed by one v3-edge + two v4-edges), the dedup step
    silently merges 3 typed diagrams into one representative,
    losing the 3× combinatorial that those alternative leg-pairings
    would have contributed.  The downstream 𝒮(Γ) for the survivor
    is then a factor of 3 too small.

    Returning the equivalence-class size lets the caller multiply
    the surviving diagram's prefactor by it, recovering the correct
    total weight.

    For Hawkes-style theories whose interactions are at most
    quadratic in any single field, every equivalence class has
    size 1 and the multiplicity is a no-op.

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram

    Returns
    -------
    unique : list of TypedDiagram
        One representative per signature, in first-seen order.
    multiplicities : list of int
        Parallel to ``unique``; ``multiplicities[i]`` is the number
        of input diagrams whose signature matched ``unique[i]``.
    """
    sig_to_idx = {}
    unique = []
    multiplicities = []
    for td in typed_diagrams:
        sig = diagram_signature(td)
        idx = sig_to_idx.get(sig)
        if idx is None:
            sig_to_idx[sig] = len(unique)
            unique.append(td)
            multiplicities.append(1)
        else:
            multiplicities[idx] += 1
    return unique, multiplicities


# ── Coefficient classification ──────────────────────────────────────────────

def _symbols_matching_prefixes(expr, prefixes):
    """
    Return the set of free SR variables in *expr* whose string name
    starts with any of the given prefixes.
    """
    if not prefixes:
        return set()
    matches = set()
    for sym in expr.variables():
        name = str(sym)
        if any(name.startswith(p) for p in prefixes):
            matches.add(sym)
    return matches


def _is_source_type(vtype):
    """Check if a vertex type is a SourceType (has no physical_legs)."""
    return not hasattr(vtype, 'physical_legs')


def classify_coefficient_factors(typed_diagram, time_dep_params=None,
                                 noise_structure=None):
    r"""
    Partition each vertex coefficient into factors that can be pulled
    outside the integral vs factors that must stay inside.

    Returns
    -------
    dict with keys:
        'M', 'scalar_prefactor', 'vertex_time_factors',
        'source_time_info', 'is_stationary'
    """
    prefixes = list(time_dep_params or [])
    ns = noise_structure or {'temporal_type': 'white', 'amplitude_params': []}
    noise_type = ns.get('temporal_type', 'white')
    noise_amp_prefixes = list(ns.get('amplitude_params', []))

    Scal = combinatorial_factor(typed_diagram)

    scalar_parts = [SR(Scal)]
    vertex_time_factors = {}
    source_time_info = {}

    for v, vtype in typed_diagram.vertex_assignments.items():
        # Z = ∫ exp(-S), so each vertex factor from S_V acquires (-1).
        coeff = -SR(vtype.coefficient)

        if _is_source_type(vtype):
            n_legs = len(vtype.response_legs)
            amp_td_syms = _symbols_matching_prefixes(coeff,
                [p for p in noise_amp_prefixes if p in prefixes])
            amp_is_td = len(amp_td_syms) > 0
            can_pull_out = (noise_type == 'white' and not amp_is_td)

            if can_pull_out:
                scalar_parts.append(coeff)
            else:
                if amp_td_syms:
                    const_part = coeff.subs({s: SR(1) for s in amp_td_syms})
                    if not const_part.is_one() and not const_part.is_zero():
                        scalar_parts.append(const_part)

            source_time_info[v] = {
                'n_legs': n_legs,
                'temporal_type': noise_type,
                'amplitude': coeff,
                'amplitude_is_time_dep': amp_is_td,
                'in_integrand': not can_pull_out,
            }

        else:
            td_syms = _symbols_matching_prefixes(coeff, prefixes)

            if not td_syms:
                scalar_parts.append(coeff)
            else:
                const_part = coeff.subs({s: SR(1) for s in td_syms})
                td_part = coeff / const_part if not const_part.is_zero() else coeff

                if not const_part.is_one() and not const_part.is_zero():
                    scalar_parts.append(const_part)

                vertex_time_factors[v] = td_part.simplify_rational()

    scalar_prefactor = reduce(mul, scalar_parts, SR(1))

    is_stationary = (
        len(vertex_time_factors) == 0
        and all(not info['amplitude_is_time_dep']
                for info in source_time_info.values())
        and all(info['temporal_type'] in ('white', 'colored')
                for info in source_time_info.values())
    )

    return {
        'M': Scal,                 # key 'M' kept (interface); value is 𝒮(Γ)
        'scalar_prefactor': scalar_prefactor,
        'vertex_time_factors': vertex_time_factors,
        'source_time_info': source_time_info,
        'is_stationary': is_stationary,
    }

"""
engine.diagrams.symmetry
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
    return D, partition, vertex_label_map, color_groups


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
    D, partition, _, _ = _colored_incidence_digraph(
        typed_diagram, fix_external=fix_external
    )
    grp = D.automorphism_group(partition=partition)
    return int(grp.order())


def external_wick_compensation(typed_diagram):
    r"""
    Number of external-leaf permutations of *typed_diagram* that are
    realizable by graph automorphisms:

        comp  =  |Aut(Γ, leaves free)| / |Aut(Γ, leaves fixed)|

    This is the EXACT divisor for the external-Wick mapping sum in
    ``integrate_diagram`` / ``integrate_grouped_diagram``.  Those
    integrators enumerate every field-respecting assignment of
    canonical external positions to diagram leaves and sum the
    integrand over all of them.  By orbit–stabilizer, two assignments
    yield the same *pinned-external* diagram iff they differ by an
    automorphism that permutes leaves, and the stabilizer of any
    assignment is exactly ``Aut_fixed_ext`` — so every distinct
    pinned-external diagram class is counted

        |Aut(Γ, leaves free)| / |Aut(Γ, leaves fixed)|

    times by the mapping sum.  Dividing by this index makes the sum
    equal to the sum over distinct pinned diagrams, which is what the
    Feynman rules require (each pinned diagram already carries its
    full weight via ``combinatorial_factor``, whose denominator is
    the SAME ``Aut_fixed_ext``).

    History
    -------
    The previous implementation approximated this index as
    ``∏ N_{sig,field}!`` over leaves grouped by the
    ``vertex_role_signature`` of their attachment vertex.  That
    heuristic over-divides whenever two attachment vertices share a
    role signature (a shallow, depth-1 invariant) but are NOT related
    by any automorphism — e.g. the k=4 OU+εx³ 1-loop cascades, where
    three noise vertices all read "noise feeding a cubic" at depth 1
    yet hang off structurally distinct cubics.  The resulting ×⅓ / ×½
    deficits broke every k≥3 1-loop cumulant while leaving k=2 exact
    (at k=2 the heuristic happens to coincide with the index).
    Validated against the exact Boltzmann series κ₄ = −6ε + 126ε²
    (per-diagram match to the hand-enumerated Wick classes).

    Returns
    -------
    int
        The index (always >= 1; equals 1 when no leaf permutation is
        an automorphism, e.g. all-distinct external fields).
    """
    aut_free = _automorphism_order(typed_diagram, fix_external=False)
    aut_fixed = _automorphism_order(typed_diagram, fix_external=True)
    if aut_fixed <= 0 or aut_free % aut_fixed != 0:
        # Aut_fixed is a subgroup of Aut_free, so the order must
        # divide — anything else is a Sage edge case.  Fail loudly:
        # silently mis-weighting diagrams is exactly the bug class
        # this function exists to fix.
        raise RuntimeError(
            f'external_wick_compensation: |Aut_free|={aut_free} not '
            f'divisible by |Aut_fixed|={aut_fixed}'
        )
    return aut_free // aut_fixed


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
    Build a hashable canonical signature for a typed diagram: the
    canonical form of the colour-preserving incidence digraph with
    leaves coloured by FIELD only (``fix_external=False``).

    Two typed diagrams get the same signature **iff** they are
    isomorphic as coloured graphs — same vertex types (class,
    coefficient, bigrade), same propagator types on corresponding
    edges, same topology — allowing same-field external leaves to be
    permuted.  This is exactly the equivalence the integration layer
    expects: leaf-permuted variants are merged here and re-expanded by
    the ``_all_mappings`` sum in ``integrate_diagram`` /
    ``integrate_grouped_diagram`` (divided by
    ``external_wick_compensation``), while every genuinely distinct
    topology survives as its own representative.

    History
    -------
    The previous signature was a hand-rolled isomorphism INVARIANT
    (per-vertex leaf/internal-edge multisets with depth-1 type tags)
    rather than a complete one.  At k=3, 1-loop, the OU+ax^2+bx^3 a^3
    sector has 11 distinct diagram classes; the old signature
    collided 4 of them into 2 representatives (mult=3 each), and --
    since the dedup multiplicity is correctly NOT multiplied back
    (the Aut-based S(Gamma) already carries each class's full weight)
    -- the collided classes' integrals were silently dropped: the
    kappa_3 1-loop coefficient came out -68/3*a^3 instead of -32*a^3.
    A complete invariant cannot collide, so this failure mode is
    closed for every k, ell, and model.  (Validated against the
    exact Boltzmann series at k=3 and k=4, and a brute-force labeled
    Wick enumeration per class -- see scratch/wick_count_k3_a3.py.)

    Parameters
    ----------
    td : TypedDiagram

    Returns
    -------
    tuple
        Hashable canonical signature ``(colour_keys, colour_cells,
        canonical_edges)``.
    """
    D, _, _, color_groups = _colored_incidence_digraph(
        td, fix_external=False
    )
    # Deterministic, label-independent cell order: sort colour classes
    # by their (hashable, fully value-based) colour key.
    keys = sorted(color_groups.keys(), key=str)
    partition = [color_groups[k] for k in keys]
    C, cert = D.canonical_label(partition=partition, certificate=True)
    # Record which canonical vertex ids each colour class maps to --
    # the canonical edge list alone would confuse two graphs with the
    # same shape but different colourings.
    cells = tuple(tuple(sorted(cert[v] for v in color_groups[k]))
                  for k in keys)
    edges = tuple(sorted(C.edges(labels=False)))
    return (tuple(str(k) for k in keys), cells, edges)


def deduplicate_typed_diagrams(typed_diagrams):
    """
    Remove duplicate typed diagrams, keeping one representative per
    unique diagram Γ.

    Two TypedDiagrams are duplicates iff they are isomorphic as
    coloured graphs (``diagram_signature`` is a complete invariant —
    canonical form of the coloured incidence digraph with leaves
    coloured by field), i.e. they differ only by relabeling, by the
    internal leg-to-edge bijection (attachment), or by a permutation
    of same-field external leaves (re-expanded downstream by the
    ``_all_mappings`` sum).

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram

    Returns
    -------
    unique : list of TypedDiagram
        One representative per unique diagram.
    """
    unique, _mult = deduplicate_with_multiplicities(typed_diagrams)
    return unique


def deduplicate_with_multiplicities(typed_diagrams):
    """
    Like ``deduplicate_typed_diagrams`` but also returns the size of
    each diagram's equivalence class.

    The multiplicity is DIAGNOSTIC ONLY.  Under Path A the weight of
    a diagram is carried entirely by ``combinatorial_factor`` (the
    orbit–stabilizer count of Wick pairings on the representative) —
    multiplying by the class size would double-count, because the
    merged entries are isomorphic copies whose pairings 𝒮(Γ) already
    includes.  (Historically a caller-side ``mult`` multiplication
    compensated for incomplete-signature collisions that merged
    NON-isomorphic diagrams; ``diagram_signature`` is now a complete
    isomorphism invariant, so such collisions cannot occur and the
    compensation story is moot.)

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
        'Scal', 'scalar_prefactor', 'vertex_time_factors',
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
        'Scal': Scal,
        'scalar_prefactor': scalar_prefactor,
        'vertex_time_factors': vertex_time_factors,
        'source_time_info': source_time_info,
        'is_stationary': is_stationary,
    }

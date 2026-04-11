"""
msrjd.diagrams.symmetry
========================
Combinatorial factor M(Γ) for fully-typed labeled diagrams, and
deduplication of typed diagrams into unique representatives.

Definition (Attachment)
-----------------------
Given a typed diagram Γ with directed graph D = (V, E) and a fixed
propagator type on each edge, the combinatorial factor M(Γ) counts
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

    M(Γ) = ∏_v  M_v

The diagram's contribution to the k-point function is:

    weight(Γ) = M(Γ) × ∏_v coeff(v) × ∫(propagators)

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

    For each outgoing edge (vertex → w), the pairing is
    (resp_type, phys_type) where resp_type is the response leg used
    at *vertex* and phys_type is the physical leg at w.

    M_v = [∏_r  n_r! / ∏_p n[r][p]!]  ×  [∏_p  m_p!]

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
    # Edge keys may be 2-tuples (u, v) or 3-tuples (u, v, label).
    pairings = []
    for edge_key, (resp_leg, phys_leg) in edge_types.items():
        u = edge_key[0]
        if u == vertex:
            pairings.append((resp_leg, phys_leg))

    if not pairings:
        return 1

    # n_r: total count of each response type
    resp_counts = Counter(r for r, p in pairings)
    # n[r][p]: count of each (resp, phys) pairing
    pair_counts = Counter(pairings)
    # m_p: count of each physical type across outgoing edges
    phys_counts = Counter(p for r, p in pairings)

    m = 1
    # ∏_r [ n_r! / ∏_p n[r][p]! ]
    for r, n_r in resp_counts.items():
        m *= factorial(n_r)
        for p in phys_counts:
            n_rp = pair_counts.get((r, p), 0)
            if n_rp > 1:
                m //= factorial(n_rp)
    # ∏_p m_p!
    for m_p in phys_counts.values():
        m *= factorial(m_p)

    return m


def combinatorial_factor(typed_diagram):
    r"""
    Compute M(Γ) — the number of response-leg permutations across all
    vertices that preserve the same typed diagram.

    M(Γ) = ∏_v M_v

    This factor **multiplies** the diagram's contribution:

        weight = M(Γ) × ∏(vertex coefficients) × ∫(propagators)

    Parameters
    ----------
    typed_diagram : TypedDiagram

    Returns
    -------
    int
        The combinatorial factor (always >= 1).
    """
    m = 1
    for v in typed_diagram.vertex_assignments:
        m *= _vertex_combinatorial_factor(v, typed_diagram)
    return m


def compute_all_combinatorial_factors(typed_diagrams):
    """
    Compute M(Γ) for each typed diagram.

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
    and for internal (non-leaf) edges:
      - propagator indices

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

    # For each internal vertex, collect the multiset of (field, prop_idx)
    # for its leaf edges.
    vertex_leaf_map = {}
    for v in td.vertex_assignments:
        leaf_edges = []
        for ek, et in td.edge_types.items():
            # Edge from internal vertex v to a leaf
            if ek[0] == v and ek[1] in leaf_set:
                field = td.external_legs[ek[1]]
                prop = td.propagator_indices[ek]
                leaf_edges.append((field, prop))
            # Edge from a leaf to internal vertex v (rare but possible)
            elif ek[1] == v and ek[0] in leaf_set:
                field = td.external_legs[ek[0]]
                prop = td.propagator_indices[ek]
                leaf_edges.append((field, prop))
        vertex_leaf_map[v] = tuple(sorted(leaf_edges))

    # Vertex assignments with leaf info, sorted by type (not by id)
    verts = []
    for v, vtype in td.vertex_assignments.items():
        tname = type(vtype).__name__
        resp = tuple(vtype.response_legs)
        phys = tuple(vtype.physical_legs) if hasattr(vtype, 'physical_legs') else ()
        verts.append((tname, str(vtype.coefficient), vtype.bigrade, resp, phys,
                      vertex_leaf_map.get(v, ())))
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
    """
    seen = set()
    unique = []
    for td in typed_diagrams:
        sig = diagram_signature(td)
        if sig not in seen:
            seen.add(sig)
            unique.append(td)
    return unique


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

    M = combinatorial_factor(typed_diagram)

    scalar_parts = [SR(M)]
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
        'M': M,
        'scalar_prefactor': scalar_prefactor,
        'vertex_time_factors': vertex_time_factors,
        'source_time_info': source_time_info,
        'is_stationary': is_stationary,
    }

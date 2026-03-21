"""
msrjd.diagrams.symmetry
========================
Combinatorial factor M(Γ) for fully-typed labeled diagrams, and
deduplication of typed diagrams into unique representatives.

Definition (Attachment)
-----------------------
Given a typed diagram skeleton Γ with directed graph D = (V, E),
vertex leg multisets L_v^out (response) and L_v^in (physical), and
a fixed propagator type on each edge, an *attachment* is a collection
of bijections {f_v^out, f_v^in}_{v in V} such that:

  1. f_v^out : L_v^out → {outgoing edges of v}  is a bijection.
  2. f_v^in  : L_v^in  → {incoming edges of v}  is a bijection.
  3. Each edge e = (u → v) is matched by exactly one response leg
     at u and one physical leg at v.
  4. The resulting (response_leg, physical_leg) pair on each edge
     corresponds to a nonzero propagator component.
  5. Edges to/from external vertices match the assigned external fields.

The combinatorial factor is:

    M(Γ) = number of valid attachments.

Since identical legs (same field type) map to the same propagator
row or column, swapping identical legs among their edges never changes
the propagator assignment.  Therefore:

    M(Γ) = ∏_v  ∏_{groups of k identical response legs at v}  k!
              ×  ∏_{groups of k identical physical legs at v}  k!

The diagram's contribution to the k-point function is:

    weight(Γ) = M(Γ) × ∏_v coeff(v) × ∫(propagators)

where the vertex coefficients already contain 1/n! from the Taylor
expansion of the action.

Reference: Helias & Dahmen, "Statistical Field Theory for Neural
Networks", Ch. 9 (Springer, 2020).

Build Phase G.
"""

from collections import Counter
from functools import reduce
from math import factorial
from operator import mul

from sage.all import SR


# ── Combinatorial factor ────────────────────────────────────────────────────

def _vertex_attachment_count(vertex_type):
    """
    Count the number of valid leg-to-edge bijections at a single vertex.

    For each group of k identical response legs, any permutation among
    their k outgoing edges yields the same propagator indices → k! ways.
    Same for physical legs.

    Parameters
    ----------
    vertex_type : VertexType or SourceType

    Returns
    -------
    int
        Number of distinct attachments at this vertex (always >= 1).
    """
    resp_legs = vertex_type.response_legs
    has_phys = hasattr(vertex_type, 'physical_legs')
    phys_legs = vertex_type.physical_legs if has_phys else []

    m = 1
    for count in Counter(resp_legs).values():
        m *= factorial(count)
    for count in Counter(phys_legs).values():
        m *= factorial(count)
    return m


def combinatorial_factor(typed_diagram):
    r"""
    Compute M(Γ) — the number of distinct valid attachments (leg-to-edge
    bijections) that realize the typed diagram Γ.

    M(\Gamma) = \prod_{v} \prod_{\text{groups of } k \text{ identical legs}} k!

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
    for v, vtype in typed_diagram.vertex_assignments.items():
        m *= _vertex_attachment_count(vtype)
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
    represent the same Feynman diagram Γ and differ only in the
    internal choice of which identical leg was assigned to which edge
    (an attachment degree of freedom).

    The signature encodes:
      - External leg assignments  (which field at each leaf)
      - Vertex type at each internal vertex  (coefficient, legs, bigrade)
      - Propagator indices on every edge

    Parameters
    ----------
    td : TypedDiagram

    Returns
    -------
    tuple
        Hashable canonical signature.
    """
    # External legs: sorted (leaf, field) pairs
    ext = tuple(sorted(td.external_legs.items()))

    # Vertex assignments: sorted (vertex, type_key) pairs
    verts = []
    for v, vtype in sorted(td.vertex_assignments.items()):
        tname = type(vtype).__name__
        resp = tuple(vtype.response_legs)
        phys = tuple(vtype.physical_legs) if hasattr(vtype, 'physical_legs') else ()
        verts.append((v, tname, str(vtype.coefficient), vtype.bigrade, resp, phys))
    verts = tuple(verts)

    # Edge propagator assignments: sorted (edge, prop_indices) pairs
    edges = tuple(sorted(
        ((u, v), td.propagator_indices[(u, v)])
        for (u, v) in td.edge_types
    ))

    return (ext, verts, edges)


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

    >>> _symbols_matching_prefixes(SR('nstar1 * phi1_1'), ['nstar'])
    {nstar1}
    """
    if not prefixes:
        return set()
    matches = set()
    for sym in expr.variables():
        name = str(sym)
        if any(name.startswith(p) for p in prefixes):
            matches.add(sym)
    return matches


def classify_coefficient_factors(typed_diagram, time_dep_params=None):
    """
    Partition each vertex coefficient into constant and time-dependent
    symbolic factors.

    In the **stationary** case (``time_dep_params`` is empty or None),
    every coefficient is constant and the full product can be pulled
    outside the propagator integral.

    In the **nonstationary** case, symbols whose names start with any
    prefix in ``time_dep_params`` are time-dependent — they inherit the
    time variable of the vertex they sit on and must stay **inside**
    the integral.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    time_dep_params : list of str or None
        Parameter name prefixes that are time-dependent
        (e.g. ``['nstar', 'phi1', 'phi2']``).
        If None or empty, all coefficients are treated as constant.

    Returns
    -------
    dict with keys:
        ``'M'`` : int
            Combinatorial factor M(Γ).
        ``'scalar_prefactor'`` : SR expression
            Product of all constant (time-independent) factors across
            all vertices.  In the stationary case this equals
            M(Γ) × ∏_v coeff(v).
        ``'vertex_time_factors'`` : dict
            ``{vertex_id: SR expression}`` for each vertex whose
            coefficient contains time-dependent symbols.  Empty dict
            in the stationary case.
        ``'is_stationary'`` : bool
            True when no vertex has time-dependent symbols.
    """
    prefixes = list(time_dep_params or [])
    M = combinatorial_factor(typed_diagram)

    scalar_parts = [SR(M)]
    time_factors = {}

    for v, vtype in typed_diagram.vertex_assignments.items():
        coeff = SR(vtype.coefficient)
        td_syms = _symbols_matching_prefixes(coeff, prefixes)

        if not td_syms:
            # Entirely constant — goes into scalar prefactor
            scalar_parts.append(coeff)
        else:
            # Factor out the constant part
            # Use substitution: set all time-dep symbols to 1 → constant part
            const_part = coeff.subs({s: SR(1) for s in td_syms})
            # The time-dependent part = coeff / const_part
            td_part = coeff / const_part if not const_part.is_zero() else coeff

            if not const_part.is_one() and not const_part.is_zero():
                scalar_parts.append(const_part)

            time_factors[v] = td_part.simplify_rational()

    scalar_prefactor = reduce(mul, scalar_parts, SR(1))

    return {
        'M': M,
        'scalar_prefactor': scalar_prefactor,
        'vertex_time_factors': time_factors,
        'is_stationary': len(time_factors) == 0,
    }

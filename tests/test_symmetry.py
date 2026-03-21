"""
tests/test_symmetry.py
======================
Tests for msrjd.diagrams.symmetry — combinatorial factor M(Γ)
and typed diagram deduplication.

Build Phase G.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_symmetry.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph

from msrjd.diagrams.symmetry import (
    combinatorial_factor, compute_all_combinatorial_factors,
    diagram_signature, deduplicate_typed_diagrams,
)
from msrjd.diagrams.type_assignment import TypedDiagram
from msrjd.core.vertices import VertexType, SourceType


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_td(edges, leaves, vert_assignments=None, edge_types=None,
             ext_legs=None, prop_indices=None):
    D = DiGraph(edges)
    G = D.to_undirected()
    leaf_set = set(leaves)
    internal = sorted(set(D.vertices()) - leaf_set)
    pd = (D, G, list(leaves), internal)
    return TypedDiagram(
        pd,
        vert_assignments or {},
        edge_types or {},
        ext_legs or {},
        prop_indices or {},
    )


# ── Tests: M(Γ) = 1 for all-distinct legs ─────────────────────────────────

def test_all_distinct_legs_m_equals_one():
    """
    Vertex with all distinct legs → only 1 valid attachment.
    M(Γ) = 1! × 1! × 1! = 1.
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)],
        leaves=[0, 1],
        vert_assignments={2: vt},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    assert combinatorial_factor(td) == 1


def test_distinct_vertex_types_m_equals_one():
    """Two internal vertices with different types → M = 1 × 1 = 1."""
    vt1 = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    vt2 = VertexType(SR(2), [('nt', 2)], [('dn', 2)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 3), (3, 1)],
        leaves=[0, 1],
        vert_assignments={2: vt1, 3: vt2},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 3): (0, 0), (3, 1): (0, 0)},
    )
    assert combinatorial_factor(td) == 1


# ── Tests: identical response legs ─────────────────────────────────────────

def test_two_identical_response_legs():
    """
    Vertex with 2 identical response legs (nt, 1).
    Either leg can attach to either outgoing edge → M = 2! = 2.
    """
    vt = VertexType(SR(1), [('nt', 1), ('nt', 1)], [('dn', 1)], (2, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1), (2, 3)],
        leaves=[0, 1, 3],
        vert_assignments={2: vt},
        ext_legs={0: ('dn', 1), 1: ('dn', 1), 3: ('dn', 2)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0), (2, 3): (0, 1)},
    )
    assert combinatorial_factor(td) == 2


def test_three_identical_response_legs():
    """
    Vertex with 3 identical response legs → M = 3! = 6.
    """
    vt = VertexType(SR(1), [('nt', 1), ('nt', 1), ('nt', 1)], [('dn', 1)], (3, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1), (2, 3), (2, 4)],
        leaves=[0, 1, 3, 4],
        vert_assignments={2: vt},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0), (2, 3): (0, 1), (2, 4): (0, 2)},
    )
    assert combinatorial_factor(td) == 6


# ── Tests: identical physical legs ─────────────────────────────────────────

def test_two_identical_physical_legs():
    """
    Vertex with 2 identical physical legs (dn, 1).
    Either leg can attach to either incoming edge → M = 2! = 2.
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 1)], (1, 2))
    td = _make_td(
        edges=[(0, 2), (3, 2), (2, 1)],
        leaves=[0, 1, 3],
        vert_assignments={2: vt},
        prop_indices={(0, 2): (0, 0), (3, 2): (0, 0), (2, 1): (0, 0)},
    )
    assert combinatorial_factor(td) == 2


# ── Tests: multiple vertex product ─────────────────────────────────────────

def test_multi_vertex_product():
    """
    M(Γ) is the product of attachment counts over all vertices.
    Vertex 2: 2 identical resp legs → 2.
    Vertex 3: 2 identical phys legs → 2.
    Total M = 2 × 2 = 4.
    """
    vt_out = VertexType(SR(1), [('nt', 1), ('nt', 1)], [('dn', 1)], (2, 1))
    vt_in  = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 1)], (1, 2))

    D = DiGraph(multiedges=True)
    D.add_edges([(0, 2, 0), (2, 3, 0), (2, 3, 1), (3, 1, 0)])
    G = D.to_undirected()
    pd = (D, G, [0, 1], [2, 3])

    td = TypedDiagram(
        pd, {2: vt_out, 3: vt_in}, {},
        {0: ('nt', 1), 1: ('dn', 1)},
        {(0, 2): (0, 0), (2, 3): (0, 0), (3, 1): (0, 0)},
    )
    assert combinatorial_factor(td) == 4


def test_bubble_different_propagators():
    """
    Bubble with distinct legs at each vertex → M = 1.
    """
    vt_out = VertexType(SR(1), [('nt', 1), ('nt', 2)], [('dn', 1)], (2, 1))
    vt_in  = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 2)], (1, 2))

    D = DiGraph(multiedges=True)
    D.add_edges([(0, 2, None), (2, 3, 0), (2, 3, 1), (3, 1, None)])
    G = D.to_undirected()
    pd = (D, G, [0, 1], [2, 3])

    td = TypedDiagram(
        pd, {2: vt_out, 3: vt_in}, {},
        {0: ('nt', 1), 1: ('dn', 1)},
        {(0, 2, None): (0, 0), (2, 3, 0): (0, 0), (2, 3, 1): (0, 1),
         (3, 1, None): (0, 0)},
    )
    assert combinatorial_factor(td) == 1


# ── Tests: source vertex ──────────────────────────────────────────────────

def test_source_two_identical_legs():
    """
    Source vertex with 2 identical response legs.
    M = 2! = 2.  Together with coefficient 1/2: M × coeff = 2 × 1/2 = 1.
    """
    st = SourceType(SR(1)/2, [('nt', 1), ('nt', 1)], (2, 0))
    td = _make_td(
        edges=[(2, 0), (2, 1)],
        leaves=[0, 1],
        vert_assignments={2: st},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (0, 1)},
    )
    M = combinatorial_factor(td)
    assert M == 2
    # Net weight: M × coeff = 2 × (1/2) = 1
    assert M * st.coefficient == 1


def test_source_distinct_legs():
    """
    Source vertex with 2 distinct response legs → M = 1.
    """
    st = SourceType(SR(1), [('nt', 1), ('nt', 2)], (2, 0))
    td = _make_td(
        edges=[(2, 0), (2, 1)],
        leaves=[0, 1],
        vert_assignments={2: st},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (1, 1)},
    )
    assert combinatorial_factor(td) == 1


# ── Tests: compute_all ────────────────────────────────────────────────────

def test_compute_all_combinatorial_factors():
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td1 = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt}, prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    td2 = _make_td(
        edges=[(0, 1)], leaves=[0, 1],
        prop_indices={(0, 1): (0, 0)},
    )
    factors = compute_all_combinatorial_factors([td1, td2])
    assert factors == [1, 1]


# ── Tests: deduplication ──────────────────────────────────────────────────

def test_deduplication_removes_duplicates():
    """
    Two TypedDiagrams with the same signature should collapse to 1.
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td1 = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    # Same diagram, just a different Python object
    td2 = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    unique = deduplicate_typed_diagrams([td1, td2])
    assert len(unique) == 1


def test_deduplication_keeps_distinct():
    """
    Two TypedDiagrams with different propagator indices → both kept.
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td1 = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    td2 = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 1)},  # different!
    )
    unique = deduplicate_typed_diagrams([td1, td2])
    assert len(unique) == 2


def test_signature_deterministic():
    """Identical diagrams produce identical signatures."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    assert diagram_signature(td) == diagram_signature(td)


# ── Tests: M(Γ) matches enumeration multiplicity ─────────────────────────

def test_m_matches_multiplicity():
    """
    Enumerate duplicates explicitly: 2 identical resp legs produce 2
    typed diagrams with the same signature.  M(Γ) must equal 2.
    """
    st = SourceType(SR(1)/2, [('nt', 1), ('nt', 1)], (2, 0))

    # Two attachments: leg0→edge(2,0), leg1→edge(2,1)
    #              and  leg1→edge(2,0), leg0→edge(2,1)
    # Both produce the same propagator indices because both legs are (nt, 1).
    td1 = _make_td(
        edges=[(2, 0), (2, 1)], leaves=[0, 1],
        vert_assignments={2: st},
        edge_types={(2, 0): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 2))},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (0, 1)},
    )
    td2 = _make_td(
        edges=[(2, 0), (2, 1)], leaves=[0, 1],
        vert_assignments={2: st},
        edge_types={(2, 0): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 2))},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (0, 1)},
    )

    # Same signature → same diagram
    assert diagram_signature(td1) == diagram_signature(td2)

    # Dedup keeps 1
    unique = deduplicate_typed_diagrams([td1, td2])
    assert len(unique) == 1

    # M(Γ) = 2 = multiplicity
    assert combinatorial_factor(unique[0]) == 2


def test_no_internal_vertices_m_one():
    """A diagram with no internal vertices has M = 1 (empty product)."""
    td = _make_td(
        edges=[(0, 1)], leaves=[0, 1],
        vert_assignments={},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 1): (0, 0)},
    )
    assert combinatorial_factor(td) == 1

"""
tests/test_symmetry.py
======================
Tests for msrjd.diagrams.symmetry — Build Phase G.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_symmetry.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph

from msrjd.diagrams.symmetry import (
    build_subdivision_graph, symmetry_factor, compute_all_symmetry_factors,
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


# ── Tests: tree-level diagrams ──────────────────────────────────────────────

def test_simple_tree_symmetry_one():
    """
    Simple tree: 0 -> 2 -> 1 with distinct external legs.
    No internal vertex permutations possible → S = 1.
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)],
        leaves=[0, 1],
        vert_assignments={2: vt},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    S = symmetry_factor(td)
    assert S == 1


def test_distinct_vertex_types_no_symmetry():
    """Two internal vertices with different types → S = 1."""
    vt1 = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    vt2 = VertexType(SR(2), [('nt', 2)], [('dn', 2)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 3), (3, 1)],
        leaves=[0, 1],
        vert_assignments={2: vt1, 3: vt2},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 3): (0, 0), (3, 1): (0, 0)},
    )
    S = symmetry_factor(td)
    assert S == 1


# ── Tests: loop diagrams with symmetry ──────────────────────────────────────

def test_self_energy_bubble():
    """
    Self-energy bubble: two parallel edges between vertices 2 and 3,
    same propagator type on both edges.

        0 -> 2 ==> 3 -> 1  (double edge 2->3)

    Exchanging the two edges is an automorphism → S = 2.
    """
    vt_out = VertexType(SR(1), [('nt', 1), ('nt', 1)], [('dn', 1)], (2, 1))
    vt_in  = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 1)], (1, 2))

    # Must construct multiedge DiGraph explicitly
    D = DiGraph(multiedges=True)
    D.add_edges([(0, 2, 0), (2, 3, 0), (2, 3, 1), (3, 1, 0)])
    G = D.to_undirected()
    pd = (D, G, [0, 1], [2, 3])

    td = TypedDiagram(
        pd,
        {2: vt_out, 3: vt_in},
        {},
        {0: ('nt', 1), 1: ('dn', 1)},
        # Both 2->3 edges carry the same propagator type
        {(0, 2): (0, 0), (2, 3): (0, 0), (3, 1): (0, 0)},
    )
    S = symmetry_factor(td)
    # vt_out has 2 identical resp legs → 2!, vt_in has 2 identical phys legs → 2!
    # Total S = 2 * 2 = 4
    assert S == 4


def test_bubble_different_propagators():
    """
    Same topology as bubble but the two parallel edges carry different
    propagator types.  Each vertex has all-distinct legs → S = 1.
    """
    vt_out = VertexType(SR(1), [('nt', 1), ('nt', 2)], [('dn', 1)], (2, 1))
    vt_in  = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 2)], (1, 2))

    D = DiGraph(multiedges=True)
    # Use edge labels 0 and 1 to distinguish parallel edges
    D.add_edges([(0, 2, None), (2, 3, 0), (2, 3, 1), (3, 1, None)])
    G = D.to_undirected()
    pd = (D, G, [0, 1], [2, 3])

    td = TypedDiagram(
        pd,
        {2: vt_out, 3: vt_in},
        {},
        {0: ('nt', 1), 1: ('dn', 1)},
        # Use 3-tuple keys for multiedge propagator indices
        {(0, 2, None): (0, 0), (2, 3, 0): (0, 0), (2, 3, 1): (0, 1),
         (3, 1, None): (0, 0)},
    )
    S = symmetry_factor(td)
    assert S == 1


# ── Tests: source vertex with identical legs ─────────────────────────────────

def test_source_two_identical_legs():
    """
    Tree-level 2-point: source vertex with 2 identical response legs.
    S = 2! = 2, which cancels the 1/2 coefficient from the Taylor expansion.
    """
    st = SourceType(SR(1)/2, [('nt', 1), ('nt', 1)], (2, 0))
    td = _make_td(
        edges=[(2, 0), (2, 1)],
        leaves=[0, 1],
        vert_assignments={2: st},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (0, 1)},
    )
    S = symmetry_factor(td)
    assert S == 2
    # Net weight: S * coeff = 2 * (1/2) = 1
    assert S * st.coefficient == 1


def test_source_distinct_legs():
    """
    Source vertex with 2 distinct response legs → S = 1.
    """
    st = SourceType(SR(1), [('nt', 1), ('nt', 2)], (2, 0))
    td = _make_td(
        edges=[(2, 0), (2, 1)],
        leaves=[0, 1],
        vert_assignments={2: st},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (1, 1)},
    )
    S = symmetry_factor(td)
    assert S == 1


# ── Tests: external legs fixed ──────────────────────────────────────────────

def test_external_legs_fixed():
    """
    Even with identical internal vertices, external legs should never
    be permuted — each leaf has a unique color (singleton partition).
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)],
        leaves=[0, 1],
        vert_assignments={2: vt},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    S_graph, partition = build_subdivision_graph(td)

    # External legs should be in singleton partitions
    leaf_set = {0, 1}
    for part in partition:
        for v in part:
            if v in leaf_set:
                assert len(part) == 1, \
                    f'External leg {v} is in a non-singleton partition: {part}'


# ── Tests: compute_all ──────────────────────────────────────────────────────

def test_compute_all_symmetry_factors():
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td1 = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt}, prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    td2 = _make_td(
        edges=[(0, 1)], leaves=[0, 1],
        prop_indices={(0, 1): (0, 0)},
    )
    factors = compute_all_symmetry_factors([td1, td2])
    assert len(factors) == 2
    assert all(f >= 1 for f in factors)

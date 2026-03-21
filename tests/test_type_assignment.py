"""
tests/test_type_assignment.py
=============================
Tests for msrjd.diagrams.type_assignment — Build Phase E.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_type_assignment.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph, matrix

from msrjd.core.vertices import VertexType, SourceType
from msrjd.diagrams.type_assignment import (
    TypedDiagram, build_field_index_map,
    enumerate_typed_diagrams, enumerate_all,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_pd(edges, leaves):
    D = DiGraph(edges)
    G = D.to_undirected()
    leaf_set = set(leaves)
    internal = sorted(set(D.vertices()) - leaf_set)
    return (D, G, list(leaves), internal)


def _simple_index_maps():
    """
    Simple 2-field system:
      response: nt1 (index 0), nt2 (index 1)
      physical: dn1 (index 0), dn2 (index 1)
    """
    ring_var_names = ['nt1', 'nt2', 'dn1', 'dn2']
    n_tilde = 2
    return build_field_index_map(ring_var_names, n_tilde)


def _full_propagator_2x2():
    """A 2x2 propagator with all nonzero entries."""
    omega = SR.var('omega')
    return matrix(SR, 2, 2, [1/(omega + SR(1)), 1/(omega + SR(2)),
                              1/(omega + SR(3)), 1/(omega + SR(4))])


def _diagonal_propagator_2x2():
    """A 2x2 propagator where off-diagonal entries are zero."""
    omega = SR.var('omega')
    return matrix(SR, 2, 2, [1/(omega + SR(1)), 0,
                              0, 1/(omega + SR(2))])


# ── Tests: build_field_index_map ────────────────────────────────────────────

def test_field_index_map():
    resp_idx, phys_idx = _simple_index_maps()
    assert resp_idx == {('nt', 1): 0, ('nt', 2): 1}
    assert phys_idx == {('dn', 1): 0, ('dn', 2): 1}


def test_field_index_map_4field():
    """4-field Hawkes-like system."""
    names = ['vt1', 'vt2', 'nt1', 'nt2', 'dv1', 'dv2', 'dn1', 'dn2']
    resp_idx, phys_idx = build_field_index_map(names, 4)
    assert resp_idx[('vt', 1)] == 0
    assert resp_idx[('nt', 2)] == 3
    assert phys_idx[('dv', 1)] == 0
    assert phys_idx[('dn', 2)] == 3


# ── Tests: simple prediagrams ──────────────────────────────────────────────

def test_single_edge_tree():
    """
    Simplest prediagram: one edge  0 -> 1
    Leaf 0 = response leg, Leaf 1 = physical leg.
    No internal vertices.
    """
    pd = _make_pd([(0, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _full_propagator_2x2()

    # 2-point function of nt1 (resp) and dn1 (phys)
    external_fields = [('nt', 1), ('dn', 1)]

    results = list(enumerate_typed_diagrams(
        pd, external_fields, [], [], G_ft, resp_idx, phys_idx
    ))

    # Should get at least one diagram: nt1 -> dn1
    # With 2 external fields and 2 leaves, there are 2 permutations,
    # but only the one where resp field is at the outgoing leaf works.
    assert len(results) >= 1
    for td in results:
        assert isinstance(td, TypedDiagram)


def test_source_vertex_diagram():
    """
    Prediagram with source vertex:
        2 (source) -> 0 (leaf)
        2 (source) -> 1 (leaf)
    """
    pd = _make_pd([(2, 0), (2, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _full_propagator_2x2()

    stypes = [SourceType(SR(1), [('nt', 1), ('nt', 2)], (2, 0))]
    external_fields = [('dn', 1), ('dn', 2)]  # both physical at leaves

    results = list(enumerate_typed_diagrams(
        pd, external_fields, [], stypes, G_ft, resp_idx, phys_idx
    ))

    # Should produce typed diagrams with the source vertex
    assert len(results) > 0
    for td in results:
        assert 2 in td.vertex_assignments
        assert isinstance(td.vertex_assignments[2], SourceType)


def test_propagator_zero_rejects():
    """
    Diagonal propagator: G[0,1] = G[1,0] = 0.
    An edge requiring an off-diagonal propagator should be rejected.
    """
    pd = _make_pd([(0, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _diagonal_propagator_2x2()

    # nt1 (resp idx 0) -> dn2 (phys idx 1) requires G[0,1] which is 0
    external_fields = [('nt', 1), ('dn', 2)]

    results = list(enumerate_typed_diagrams(
        pd, external_fields, [], [], G_ft, resp_idx, phys_idx
    ))

    # The assignment nt1->dn2 should fail due to zero propagator
    # Only nt1->dn1 or nt2->dn2 should work on diagonal propagator
    for td in results:
        for edge, (resp_leg, phys_leg) in td.edge_types.items():
            ri = resp_idx[resp_leg]
            pi = phys_idx[phys_leg]
            assert not bool(SR(G_ft[ri, pi]).is_zero())


def test_no_propagator_check_when_none():
    """When G_ft is None, propagator consistency is not checked."""
    pd = _make_pd([(0, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()

    external_fields = [('nt', 1), ('dn', 2)]

    results = list(enumerate_typed_diagrams(
        pd, external_fields, [], [], None, resp_idx, phys_idx
    ))
    # Should still produce results (no propagator filtering)
    assert len(results) > 0


def test_interaction_vertex():
    """
    Prediagram:  0 (leaf) -> 2 (internal) -> 1 (leaf)
    Vertex 2: in=1, out=1
    """
    pd = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _full_propagator_2x2()

    vtypes = [
        VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1)),
        VertexType(SR(1), [('nt', 2)], [('dn', 2)], (1, 1)),
    ]

    # External: resp field at leaf 0 (outgoing), phys field at leaf 1 (incoming)
    external_fields = [('nt', 1), ('dn', 1)]

    results = list(enumerate_typed_diagrams(
        pd, external_fields, vtypes, [], G_ft, resp_idx, phys_idx
    ))

    assert len(results) > 0
    for td in results:
        assert 2 in td.vertex_assignments
        assert isinstance(td.vertex_assignments[2], VertexType)


def test_enumerate_all_multiple_prediagrams():
    """enumerate_all iterates over multiple prediagrams."""
    pd1 = _make_pd([(0, 1)], leaves=[0, 1])
    pd2 = _make_pd([(0, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _full_propagator_2x2()

    external_fields = [('nt', 1), ('dn', 1)]

    results = enumerate_all([pd1, pd2], external_fields, [], [],
                            G_ft, resp_idx, phys_idx)

    # Both prediagrams are identical, so should get same count doubled
    results1 = list(enumerate_typed_diagrams(
        pd1, external_fields, [], [], G_ft, resp_idx, phys_idx))
    assert len(results) == 2 * len(results1)


def test_no_valid_assignments():
    """Prediagram where no assignment works returns empty."""
    pd = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _full_propagator_2x2()

    # No vertex types with (in=1, out=1)
    vtypes = [VertexType(SR(1), [('nt', 1), ('nt', 2)], [('dn', 1)], (2, 1))]

    external_fields = [('nt', 1), ('dn', 1)]

    results = list(enumerate_typed_diagrams(
        pd, external_fields, vtypes, [], G_ft, resp_idx, phys_idx
    ))

    assert len(results) == 0

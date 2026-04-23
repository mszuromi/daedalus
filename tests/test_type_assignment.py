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


# ── Tests: canonical (multiset) leg-matching ──────────────────────────────
# These pin the behavior of `_leg_matchings` after the 2026-04-23
# switch from full-factorial permutations to canonical-multiset
# permutations.  See the enumeration-speedup branch docstring in
# `msrjd/diagrams/type_assignment.py::_leg_matchings` for the
# correctness argument.

def test_leg_matchings_all_distinct_legs():
    """Distinct legs: multiset count == factorial count == R! × P!.
    Skipping duplicates is a no-op when there ARE no duplicates, so
    this case must yield the same count as before the change."""
    from msrjd.diagrams.type_assignment import _leg_matchings

    # 2 distinct response legs, 2 distinct physical legs
    vt = VertexType(
        SR(1),
        [('nt', 1), ('nt', 2)],      # response_legs: A, B
        [('dn', 1), ('dn', 2)],      # physical_legs: C, D
        (2, 2),
    )
    out_edges = [('u0', 'v0', 0), ('u0', 'v1', 0)]
    in_edges  = [('u2', 'u0', 0), ('u3', 'u0', 0)]

    results = list(_leg_matchings(vt, out_edges, in_edges))
    # 2! × 2! = 4 distinct orderings
    assert len(results) == 4
    # All yielded pairs are distinct (canonicality)
    seen = set()
    for resp_map, phys_map in results:
        sig = (tuple(sorted(resp_map.items())),
               tuple(sorted(phys_map.items())))
        assert sig not in seen, (
            f'Duplicate (resp_map, phys_map) yielded: {sig}'
        )
        seen.add(sig)


def test_leg_matchings_duplicate_response_legs():
    """With two identical response legs, old code would yield 2!=2
    orderings that are physically identical; canonical code yields 1."""
    from msrjd.diagrams.type_assignment import _leg_matchings

    # Two identical response legs, one physical leg
    vt = VertexType(
        SR(1),
        [('nt', 1), ('nt', 1)],      # response_legs: A, A
        [('dn', 1)],                  # physical_legs: C
        (2, 1),
    )
    out_edges = [('u0', 'v0', 0), ('u0', 'v1', 0)]
    in_edges  = [('u2', 'u0', 0)]

    results = list(_leg_matchings(vt, out_edges, in_edges))
    # 2!/(2!·1!) × 1! = 1 × 1 = 1 distinct ordering
    assert len(results) == 1, (
        f'Expected 1 canonical leg-matching for resp_legs=[A,A] + '
        f'phys_legs=[C], got {len(results)}.  Duplicate legs should '
        f'collapse.'
    )


def test_leg_matchings_mixed_multiset():
    """Mixed multiset R=[A,A,B], P=[C]:  3!/2! × 1 = 3 orderings."""
    from msrjd.diagrams.type_assignment import _leg_matchings

    vt = VertexType(
        SR(1),
        [('nt', 1), ('nt', 1), ('nt', 2)],   # [A, A, B]
        [('dn', 1)],
        (3, 1),
    )
    out_edges = [('u0', 'v0', 0), ('u0', 'v1', 0), ('u0', 'v2', 0)]
    in_edges  = [('u3', 'u0', 0)]

    results = list(_leg_matchings(vt, out_edges, in_edges))
    assert len(results) == 3, (
        f'Expected 3 canonical orderings of multiset [A,A,B], '
        f'got {len(results)}.  Pre-change code would have yielded '
        f'3!=6 (with duplicates); canonical yields 3!/2!·1!=3.'
    )
    # Collect distinct (resp_map, phys_map) pairs and confirm all
    # yielded are already distinct (no dedup needed).
    seen = set()
    for resp_map, phys_map in results:
        sig = (tuple(sorted(resp_map.items())),
               tuple(sorted(phys_map.items())))
        assert sig not in seen, (
            f'Canonical enumeration yielded duplicate leg-matching: {sig}'
        )
        seen.add(sig)
    assert len(seen) == 3


def test_leg_matchings_empty_legs():
    """Empty leg lists should yield exactly one (empty, empty) pair
    — the degenerate case where the vertex has no legs on one side."""
    from msrjd.diagrams.type_assignment import _leg_matchings

    # Source type: response legs only (no physical_legs attribute)
    st = SourceType(SR(1), [('nt', 1), ('nt', 1)], (2, 0))
    out_edges = [('u0', 'v0', 0), ('u0', 'v1', 0)]

    results = list(_leg_matchings(st, out_edges, []))
    assert len(results) == 1  # 2!/2! = 1 canonical ordering
    resp_map, phys_map = results[0]
    assert phys_map == {}
    assert set(resp_map.keys()) == set(out_edges)
    # Both edges get the same leg value ('nt', 1)
    for leg in resp_map.values():
        assert leg == ('nt', 1)


def test_enumerate_typed_duplicate_leg_vertex_no_redundant_generation():
    """End-to-end: when a SourceType has identical response legs,
    enumerate_typed_diagrams should NOT generate the factorial
    overcount of identical typed diagrams.  Verifies by comparing the
    raw generation count to the dedup'd count — they should match.

    Before this change: raw count ≈ k! × dedup count (overcounted by
    leg permutations).  After: raw count == dedup count for this
    fixture because the leg orbit is the only source of overgeneration
    once the external-field permutations are handled by
    ``_distinct_permutations``."""
    from msrjd.diagrams.type_assignment import enumerate_typed_diagrams
    from msrjd.diagrams.symmetry import deduplicate_typed_diagrams

    # k=2 star tree: 1 source vertex with 2 out-edges to 2 leaves.
    # Source vertex has 2 response legs of the SAME field type, so
    # swapping them gives an identical typed diagram.
    pd = _make_pd([(2, 0), (2, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _full_propagator_2x2()

    # Source with duplicate legs: [('nt', 1), ('nt', 1)]
    stypes = [SourceType(SR(1), [('nt', 1), ('nt', 1)], (2, 0))]
    # External fields identical -> _distinct_permutations yields 1
    # (not 2), so the only remaining over-generation source is leg
    # permutations.
    external_fields = [('dn', 1), ('dn', 1)]

    raw = list(enumerate_typed_diagrams(
        pd, external_fields, [], stypes, G_ft, resp_idx, phys_idx
    ))
    dedup = deduplicate_typed_diagrams(raw)

    assert len(raw) == len(dedup), (
        f'Raw enumeration yielded {len(raw)} diagrams; dedup kept '
        f'{len(dedup)}.  With canonical leg-matching they should '
        f'match exactly for this fixture (the only over-generation '
        f'source would have been leg permutations of the '
        f'[nt1, nt1] multiset, which canonical enumeration skips).'
    )


def test_enumerate_typed_distinct_legs_regression():
    """Regression: when all legs are distinct, the canonical change
    must produce the EXACT same set of typed diagrams as the old
    factorial-based code.  No diagrams lost, no new spurious ones."""
    from msrjd.diagrams.type_assignment import enumerate_typed_diagrams
    from msrjd.diagrams.symmetry import (
        deduplicate_typed_diagrams, diagram_signature,
    )

    # Same fixture as test_interaction_vertex: distinct legs on the
    # interaction vertex, distinct external fields.  Canonical count
    # == factorial count in this case (no multiset collisions), so
    # we can sanity-check against a known-good value.
    pd = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])
    resp_idx, phys_idx = _simple_index_maps()
    G_ft = _full_propagator_2x2()
    vtypes = [
        VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1)),
        VertexType(SR(1), [('nt', 2)], [('dn', 2)], (1, 1)),
    ]
    external_fields = [('nt', 1), ('dn', 1)]

    raw = list(enumerate_typed_diagrams(
        pd, external_fields, vtypes, [], G_ft, resp_idx, phys_idx
    ))
    dedup = deduplicate_typed_diagrams(raw)

    # For this fixture each vertex has exactly 1 resp leg and 1 phys
    # leg, so leg-matching is trivial (1 per vertex).  Neither the
    # old nor the new code should generate duplicates, and the
    # results should match a small hand-verifiable count.
    assert len(dedup) == 2, (
        f'Expected 2 unique typed diagrams (one per VertexType), '
        f'got {len(dedup)}.'
    )
    # Both ends of the dedup should already be distinct under the
    # signature hash.
    sigs = {diagram_signature(td) for td in dedup}
    assert len(sigs) == len(dedup)

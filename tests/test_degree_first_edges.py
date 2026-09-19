"""
tests/test_degree_first_edges.py
================================
The degree-first (E2) generator (endpoint vectors of the added edges, orbit
deduplicated under the tree's automorphisms, then realized as edge multisets)
must produce exactly the same set of topologies as the former
enumerate-every-multiset path, and the 2-path endpoint condition used to
filter trees must not change the prediagram counts of Table I.

Run:  sage -python -m pytest tests/test_degree_first_edges.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.enumeration.loop_diagram_enumeration as L

TABLE = {(2, 1): 9, (3, 1): 80, (4, 1): 755, (2, 2): 283, (3, 2): 4496, (1, 3): 434}


def _topology_certs(k, ell, generator):
    old = L.EDGE_GENERATOR
    L.EDGE_GENERATOR = generator
    try:
        certs = set()
        for tree, j, nl in L.generate_trees_with_constraints(k, ell):
            certs |= {L._iso_cert(G) for G, _, _ in L.process_tree_parallel((tree, j, nl, k, ell))}
        return certs
    finally:
        L.EDGE_GENERATOR = old


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (1, 3)])
def test_degree_first_matches_multiset_topologies(k, ell):
    assert _topology_certs(k, ell, 'degree_first') == _topology_certs(k, ell, 'multiset')


@pytest.mark.parametrize('k,ell', sorted(TABLE))
def test_prediagram_counts_unchanged(k, ell):
    _trees, _topos, pds, _ = L.enumerate_all(k, ell, n_threads=1, verbose=False)
    assert len(pds) == TABLE[(k, ell)]


def test_endpoint_condition_is_what_the_lemma_says():
    """A path of m degree-2 vertices needs floor(m/2) endpoints; j retired
    leaves need j; budget 2*ell."""
    from sage.all import graphs
    P = graphs.PathGraph(7)                       # 5 degree-2 vertices in one run
    d2 = [v for v in P.vertices() if P.degree(v) == 2]
    assert L.two_path_lengths(P, d2) == [5]
    assert L.endpoint_condition_holds(P, d2, j=0, ell=1)        # 2 endpoints >= floor(5/2)=2
    assert not L.endpoint_condition_holds(P, d2, j=1, ell=1)    # 2 < 2 + 1


def test_realizations_enumerate_each_multiset_once():
    vs = [0, 1, 2, 3]
    rem = {0: 2, 1: 1, 2: 2, 3: 1}                # sum 6 -> 3 edges
    found = [tuple(sorted(tuple(sorted(e)) for e in F)) for F in L._realizations(vs, rem, set())]
    assert len(found) == len(set(found))
    assert all(sum(1 for e in F for x in e if x == v) == rem[v] for F in found for v in vs)
    assert found                                    # e.g. {(0,1),(0,2),(2,3)} and {(0,2),(0,2),(1,3)}

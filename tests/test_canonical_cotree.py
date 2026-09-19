"""
tests/test_canonical_cotree.py
==============================
The 'canonical' (E2) generator builds a (tree, added-edges) pair only when the
added edges are a locally maximal co-tree of the topology under a degree-based
edge invariant (``_cotree_is_local_max``).  Every topology has a globally
maximal co-tree, which is locally maximal, so the set of topologies must be
exactly that of the plain degree-first generator, which visits every spanning
tree.  Measured effect (serial, this machine): candidates built before the
isomorphism dedup fall from 18945 to 7733 at (3,2), 64071 to 89189->36035
unique at (4,2) with the per-vector orbit dedup switched off, and the (E2)
stage runs 2-4x faster at ell >= 2.

Run:  sage -python -m pytest tests/test_canonical_cotree.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.enumeration.loop_diagram_enumeration as L

TABLE = {(2, 1): 9, (3, 1): 80, (4, 1): 755, (2, 2): 283, (3, 2): 4496, (1, 3): 434}


def _topology_certs(k, ell, generator, delta_dedup='auto'):
    old = (L.EDGE_GENERATOR, L.DELTA_ORBIT_DEDUP)
    L.EDGE_GENERATOR, L.DELTA_ORBIT_DEDUP = generator, delta_dedup
    try:
        certs, n_raw = set(), 0
        for tree, j, nl in L.generate_trees_with_constraints(k, ell):
            for G, _, _ in L.process_tree_parallel((tree, j, nl, k, ell)):
                certs.add(L._iso_cert(G))
                n_raw += 1
        return certs, n_raw
    finally:
        L.EDGE_GENERATOR, L.DELTA_ORBIT_DEDUP = old


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (1, 3), (5, 1)])
def test_canonical_matches_degree_first_topologies(k, ell):
    canon, _ = _topology_certs(k, ell, 'canonical')
    plain, n_plain = _topology_certs(k, ell, 'degree_first')
    assert canon == plain
    if ell > 0:                                   # same orbit dedup on both sides:
        _, n_canon = _topology_certs(k, ell, 'canonical', delta_dedup='1')
        assert n_canon < n_plain                  # the filter actually removes copies


@pytest.mark.parametrize('k,ell', [(3, 2), (1, 3)])
def test_orbit_dedup_switch_does_not_change_topologies(k, ell):
    a, _ = _topology_certs(k, ell, 'canonical', delta_dedup='1')
    b, _ = _topology_certs(k, ell, 'canonical', delta_dedup='0')
    assert a == b


def test_default_generator_is_canonical_and_reproduces_table_i():
    assert L.EDGE_GENERATOR == 'canonical'
    for (k, ell), n in TABLE.items():
        _t, _topos, pds, _ = L.enumerate_all(k, ell, n_threads=1, verbose=False)
        assert len(pds) == n, (k, ell)


def test_tree_paths_are_the_tree_paths():
    from sage.all import graphs
    T = graphs.RandomTree(11)
    verts = sorted(T.vertices())
    paths = L._tree_paths(T, verts)
    for (u, w), edges in paths.items():
        assert u < w
        sp = T.shortest_path(u, w)
        expect = {tuple(sorted(p)) for p in zip(sp, sp[1:])}
        assert set(edges) == expect and len(edges) == len(expect)
    assert len(paths) == len(verts) * (len(verts) - 1) // 2


def test_cotree_filter_rejects_exactly_the_improvable_exchange():
    """Path tree 0-1-2-3 with added edge (0,3): G is the 4-cycle, all degrees
    2, every exchange ties -> accepted.  Path tree 0-1-2-3 plus pendant 4 at
    1 and added edge (0,3): deg(1)=3, so the tree edges (0,1) and (1,2) on
    the 0..3 path have invariant (3,2) > (2,2) of the added edge -> rejected;
    the decomposition with co-tree {(0,1)} is the one kept."""
    from sage.all import Graph
    P = Graph([(0, 1), (1, 2), (2, 3)])
    verts = sorted(P.vertices())
    deg = {0: 2, 1: 2, 2: 2, 3: 2}
    assert L._cotree_is_local_max([(0, 3)], deg, L._tree_paths(P, verts))
    Q = Graph([(0, 1), (1, 2), (2, 3), (1, 4)])
    vq = sorted(Q.vertices())
    degq = {0: 2, 1: 3, 2: 2, 3: 2, 4: 1}
    assert not L._cotree_is_local_max([(0, 3)], degq, L._tree_paths(Q, vq))
    # the exchanged decomposition: tree 3-2-1-4 + 0-3, co-tree {(0,1)}
    R = Graph([(1, 2), (2, 3), (0, 3), (1, 4)])
    vr = sorted(R.vertices())
    assert L._cotree_is_local_max([(0, 1)], degq, L._tree_paths(R, vr))


@pytest.mark.parametrize('k,ell', [(3, 1), (2, 2)])
def test_levelwise_experiment_still_agrees(k, ell):
    """The (slower) level-wise generator kept in engine/enumeration/orderly.py
    must keep producing the same topologies, so its recorded comparison stays
    meaningful."""
    from engine.enumeration import orderly as O
    lw = {L._iso_cert(G) for G, _, _ in O.enumerate_topologies_levelwise(k, ell)}
    ref, _ = _topology_certs(k, ell, 'canonical')
    assert lw == ref

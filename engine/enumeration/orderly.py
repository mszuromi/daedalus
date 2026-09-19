"""
engine.enumeration.orderly
==========================
EXPERIMENT (Sept 2026), NOT used by the pipeline: level-wise topology
generation, kept as the record of a measured negative result.

Idea.  The tree-based generator reaches a topology once per spanning tree, so
at (2,3) it certifies ~11 copies of each.  Build topologies one added edge at
a time instead: level 0 = the admissible trees, level t = every level-(t-1)
graph plus one edge, deduplicated by canonical certificate and kept only while
still completable (remaining-budget forms of the leaf-surplus and endpoint
conditions), with the full topology tests at the last level.

Result.  Correct (same topology sets as the tree-based generators at (2,1),
(3,1), (4,1), (2,2), (1,3), (3,2)) but SLOWER: a level-t graph is reached from
every one of its non-bridge edges, not from its ell added edges, so the
candidate count explodes --

    (3,2):  221 351 level-2 candidates for 3 181 topologies (x70), 11.9 s
            versus 18 945 (x6) and 3.6 s for plain degree-first,
    (1,3):   69 633 candidates for 507 topologies, 3.8 s versus 1.0 s.

Canonical augmentation (McKay) would remove the storage of duplicates but not
the candidates, which are the cost.  The redundancy was instead removed on the
tree side by the co-tree filter ``_cotree_is_local_max`` in
``loop_diagram_enumeration`` (generator 'canonical', the default), which keeps
one locally maximal spanning-tree decomposition per topology before any graph
is built.

``enumerate_topologies_levelwise`` remains importable and is checked against
the tree-based generator in tests/test_canonical_cotree.py.
"""
from __future__ import annotations

from itertools import combinations

from sage.all import Graph

from engine.enumeration import loop_diagram_enumeration as L


def _leaves(G):
    return [v for v in G.vertices() if G.degree(v) == 1]


def _two_path_cover_cost(G):
    d2 = [v for v in G.vertices() if G.degree(v) == 2]
    return sum(m // 2 for m in L.two_path_lengths(G, d2))


def completable(H, k, r):
    """Necessary conditions for a partial graph ``H`` (``r`` edges still to
    add) to extend to a topology at order ``(k, ell)``."""
    surplus = len(_leaves(H)) - k
    if surplus < 0:
        return False
    if surplus > 2 * r:
        return False
    return _two_path_cover_cost(H) + surplus <= 2 * r


def is_topology(G, k):
    leaves, _internal, d2, d3 = L.classify_vertices_sage(G)
    if len(leaves) != k:
        return False
    if L.has_adjacent_degree2_sage(G, d2):
        return False
    if not L.check_deg3_has_non_deg2_neighbor(G, d3, d2):
        return False
    if not L.check_leaf_neighbors_not_all_deg2(G, leaves, d2):
        return False
    return True


def augmentations(H, k, r):
    """Graphs ``H + e`` over the unordered vertex pairs ``e`` that can still
    lead to a topology: an edge may touch a leaf only while surplus remains,
    and a partial graph must stay completable."""
    verts = sorted(H.vertices())
    deg = {v: H.degree(v) for v in verts}
    surplus = sum(1 for v in verts if deg[v] == 1) - k
    for u, w in combinations(verts, 2):
        touched_leaves = (deg[u] == 1) + (deg[w] == 1)
        if touched_leaves > surplus:
            continue
        G = Graph(H, multiedges=True, loops=False)
        G.add_edge(u, w)
        yield G


def enumerate_topologies_levelwise(k, ell, trees=None, verbose=False, stats=None):
    """Unique topologies at ``(k, ell)`` by level-wise augmentation.

    Returns ``[(G, leaves, internal)]`` in the leaves-first labelling, like
    ``_enumerate_topologies_raw``.  ``stats`` (a dict) receives per-level
    candidate / unique counts.
    """
    if trees is None:
        trees = L.generate_trees_with_constraints(k, ell)
    if ell == 0:
        level = {L._iso_cert(t): t for t, j, nl in trees if nl == k}
        return L._remove_isomorphic_undirected(
            [(G, _leaves(G), [v for v in G.vertices() if G.degree(v) > 1]) for G in level.values()])

    level = {}
    for tree, j, nl in trees:
        H = Graph(tree, multiedges=True, loops=False)
        if completable(H, k, ell):
            level[L._iso_cert(H)] = H
    if stats is not None:
        stats[0] = (len(trees), len(level))

    for t in range(1, ell + 1):
        r = ell - t
        nxt = {}
        n_cand = 0
        for H in level.values():
            for G in augmentations(H, k, r + 1):
                n_cand += 1
                if r > 0:
                    if not completable(G, k, r):
                        continue
                elif not is_topology(G, k):
                    continue
                nxt.setdefault(L._iso_cert(G), G)
        if stats is not None:
            stats[t] = (n_cand, len(nxt))
        if verbose:
            print(f'  level {t}: {n_cand} candidates -> {len(nxt)} unique')
        level = nxt

    return L._remove_isomorphic_undirected(
        [(G, _leaves(G), [v for v in G.vertices() if G.degree(v) > 1]) for G in level.values()])

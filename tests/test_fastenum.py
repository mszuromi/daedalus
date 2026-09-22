"""
tests/test_fastenum.py
======================
The compiled hot loops (``engine/enumeration/_fastenum.pyx``, loaded by
``engine.enumeration.fastenum`` through pyximport) must reproduce the pure
Python enumeration exactly:

* ``aux_cert`` is a complete isomorphism invariant of (di)multigraphs, in the
  ``(order, sorted_edges)`` format, stable under relabelling and idempotent
  under ``cert_to_graph`` -> cert;
* ``orientation_patterns`` returns exactly the causal orientations that the
  Python checker returns;
* the streaming pipeline with the compiled backend yields the same prediagram
  classes as the shipped cells (compared up to isomorphism, since the two
  backends produce different bytes for the same class) and the Table I counts.

Skipped, with the reason, when the extension cannot be built (no C compiler)
or is disabled with DAEDALUS_FASTENUM=0.

Run:  sage -python -m pytest tests/test_fastenum.py -q
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.enumeration import fastenum as F
import engine.enumeration.loop_diagram_enumeration as L
from engine.enumeration import prediagram_cache as pdc

pytestmark = pytest.mark.skipif(not F.available, reason=f'fastenum unavailable: {F.reason}')

TABLE = {(2, 1): 9, (3, 1): 80, (4, 1): 755, (2, 2): 283, (3, 2): 4496, (1, 3): 434}


def _index_data(G):
    vs = sorted(G.vertices())
    idx = {v: i for i, v in enumerate(vs)}
    return len(vs), [(idx[u], idx[v]) for u, v in G.edges(labels=False)]


def _topologies(k, ell):
    return [L.cert_to_graph(L.unpack_cert(b), directed=False)
            for b in sorted(L.stream_topology_certs(k, ell, 1, 50, False))]


def test_backend_is_selected():
    assert L.CERT_BACKEND == 'fast'


@pytest.mark.parametrize('directed', [False, True])
def test_aux_cert_is_invariant_and_idempotent(directed):
    random.seed(7)
    Gs = _topologies(3, 2)[:300]
    if directed:
        Gs = [D for G in Gs[:100] for D in L.enumerate_orientations(G, L.leaves_of(G))]
    for G in Gs:
        n, e = _index_data(G)
        c = F.aux_cert(n, e, directed)
        perm = list(range(n)); random.shuffle(perm)
        e2 = [(perm[u], perm[v]) for u, v in e]; random.shuffle(e2)
        assert F.aux_cert(n, e2, directed) == c
        assert L._iso_cert(L.cert_to_graph(c, directed=directed)) == c
        assert c[0] == n and len(c[1]) == len(e)


def test_aux_cert_separates_classes_like_sage():
    """Same number of classes as Sage's canonical_label on a mixed sample of
    topologies and their orientations."""
    Gs = _topologies(3, 2)
    Ds = [D for G in Gs[:200] for D in L.enumerate_orientations(G, L.leaves_of(G))]
    for objs, directed in ((Gs, False), (Ds, True)):
        fast = {F.aux_cert(*_index_data(G), directed) for G in objs}
        sage = set()
        for G in objs:
            cls = type(G)
            S = cls(multiedges=True, loops=False); S.add_vertices(G.vertices())
            for u, v in G.edges(labels=False):
                S.add_edge(u, v)
            C = S.canonical_label()
            sage.add((C.order(), tuple(sorted(C.edges(labels=False)))))
        assert len(fast) == len(sage)


def test_orientation_patterns_match_python_checker():
    for G in _topologies(3, 2)[:500]:
        n, e = _index_data(G)
        deg = [0] * n
        for u, v in e:
            deg[u] += 1; deg[v] += 1
        forced = L._forced_orientation_bits(e, {v for v in range(n) if deg[v] == 1}, deg)
        ref = L.enumerate_orientations_sage(G, L.leaves_of(G))
        if forced is None:
            assert ref == []
            continue
        base, free = forced
        pats = F.orientation_patterns(n, [u for u, v in e], [v for u, v in e], base, free,
                                      [1 if d == 1 else 0 for d in deg], deg)
        assert len(pats) == len(ref)
        assert {L._iso_cert(D) for D in ref} == \
            {F.aux_cert(n, [(v, u) if (p >> i) & 1 else (u, v) for i, (u, v) in enumerate(e)], True) for p in pats}


@pytest.mark.parametrize('k,ell', sorted(TABLE))
def test_streaming_with_fast_backend_matches_shipped_cells(k, ell):
    fast = L.stream_prediagram_certs(k, ell, n_procs=1)
    assert len(fast) == TABLE[(k, ell)]
    assert fast == pdc.recanonicalize(pdc.load_shipped_certs(k, ell))


def test_raw_and_graph_paths_agree():
    """process_tree_parallel(raw=True) must yield the same topology classes as
    the Sage-graph path for every tree at (3,2)."""
    for tree, j, nl in L.generate_trees_with_constraints(3, 2):
        a = {F.aux_cert(n, e, False) for n, e, _ in L.process_tree_parallel((tree, j, nl, 3, 2), raw=True)}
        b = {L._iso_cert(G) for G, _, _ in L.process_tree_parallel((tree, j, nl, 3, 2))}
        assert a == b


@pytest.mark.parametrize('k,ell', [(3, 2), (2, 3), (4, 1)])
def test_compiled_edge_stage_matches_python_tree_by_tree(k, ell):
    """``_fastenum.tree_candidates`` (endpoint vectors, realizations, co-tree
    filter, structural checks, certificate, all in C) must give the same
    certificate set as the Python degree-first path for EVERY tree."""
    for tree, j, nl in L.generate_trees_with_constraints(k, ell):
        n, te = L._graph_index_data(tree)
        fast = L._tree_topology_certs_fast(n, te, j, nl, k, ell)
        ref = {L.pack_cert(L._iso_cert(G)) for G, _, _ in L.process_tree_parallel((tree, j, nl, k, ell))}
        assert fast == ref

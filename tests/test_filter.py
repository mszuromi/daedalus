"""
tests/test_filter.py
====================
Tests for engine.diagrams.filter — prediagram filtering (Build Phase D).

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_filter.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph

from engine.diagrams.filter import classify_prediagram_vertices, filter_prediagrams
from engine.core.vertices import VertexType, SourceType


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_pd(edges, leaves):
    D = DiGraph(edges)
    G = D.to_undirected()
    leaf_set = set(leaves)
    internal = sorted(set(D.vertices()) - leaf_set)
    return (D, G, list(leaves), internal)


def _make_vertex_type(in_deg, out_deg):
    """Minimal VertexType stub with the given degrees."""
    resp = [('nt', i + 1) for i in range(out_deg)]
    phys = [('dn', i + 1) for i in range(in_deg)]
    return VertexType(SR(1), resp, phys, (out_deg, in_deg))


def _make_source_type(out_deg):
    """Minimal SourceType stub with the given out_degree."""
    resp = [('nt', i + 1) for i in range(out_deg)]
    return SourceType(SR(1), resp, (out_deg, 0))


# ── Tests: classify_prediagram_vertices ─────────────────────────────────────

def test_classify_source_and_interaction():
    # Vertex 2 is source (in=0), vertex 3 is interaction (in=1)
    D = DiGraph([(2, 3), (3, 0), (3, 1)])
    source, interaction = classify_prediagram_vertices(D, leaves=[0, 1])
    assert source == [2]
    assert interaction == [3]


def test_classify_no_sources():
    D = DiGraph([(0, 2), (2, 1)])
    source, interaction = classify_prediagram_vertices(D, leaves=[0, 1])
    assert source == []
    assert interaction == [2]


def test_classify_leaves_excluded():
    D = DiGraph([(0, 2), (2, 1)])
    source, interaction = classify_prediagram_vertices(D, leaves=[0, 1])
    all_classified = set(source + interaction)
    assert 0 not in all_classified
    assert 1 not in all_classified


# ── Tests: filter_prediagrams ──────────────────────────────────────────────

def test_filter_keeps_valid():
    """Prediagram with (in=1, out=2) matches available vertex type."""
    pd = _make_pd([(0, 3), (3, 1), (3, 2)], leaves=[0, 1, 2])
    # Vertex 3: in=1, out=2
    vtypes = [_make_vertex_type(1, 2)]
    stypes = []
    kept, discarded = filter_prediagrams([pd], vtypes, stypes)
    assert len(kept) == 1
    assert discarded == 0


def test_filter_removes_invalid():
    """Prediagram with degree-4 vertex when only degree-3 types available."""
    pd = _make_pd([(0, 4), (1, 4), (4, 2), (4, 3)], leaves=[0, 1, 2, 3])
    # Vertex 4: in=2, out=2 → degree 4
    vtypes = [_make_vertex_type(1, 2)]  # only (in=1, out=2)
    stypes = []
    kept, discarded = filter_prediagrams([pd], vtypes, stypes)
    assert len(kept) == 0
    assert discarded == 1


def test_filter_checks_sources():
    """Source vertex with out_degree=3 when only out_degree=2 sources exist."""
    pd = _make_pd([(2, 0), (2, 1), (2, 3)], leaves=[0, 1, 3])
    # Vertex 2: source (in=0), out=3
    vtypes = []
    stypes = [_make_source_type(2)]  # only out_degree=2
    kept, discarded = filter_prediagrams([pd], vtypes, stypes)
    assert len(kept) == 0
    assert discarded == 1


def test_filter_source_passes():
    """Source vertex matches available source type."""
    pd = _make_pd([(2, 0), (2, 1)], leaves=[0, 1])
    # Vertex 2: source, out=2
    vtypes = []
    stypes = [_make_source_type(2)]
    kept, discarded = filter_prediagrams([pd], vtypes, stypes)
    assert len(kept) == 1


def test_filter_all_pass():
    pd1 = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])  # vertex 2: in=1, out=1
    pd2 = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])
    vtypes = [_make_vertex_type(1, 1)]
    stypes = []
    kept, discarded = filter_prediagrams([pd1, pd2], vtypes, stypes)
    assert len(kept) == 2
    assert discarded == 0


def test_filter_all_fail():
    pd1 = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])  # in=1, out=1
    pd2 = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])
    vtypes = [_make_vertex_type(2, 2)]  # no (1,1) type
    stypes = []
    kept, discarded = filter_prediagrams([pd1, pd2], vtypes, stypes)
    assert len(kept) == 0
    assert discarded == 2


def test_filter_mixed():
    """Mix of valid and invalid prediagrams."""
    pd_ok = _make_pd([(0, 3), (3, 1), (3, 2)], leaves=[0, 1, 2])   # in=1, out=2
    pd_bad = _make_pd([(0, 4), (1, 4), (4, 2), (4, 3)], leaves=[0, 1, 2, 3])  # in=2, out=2
    vtypes = [_make_vertex_type(1, 2)]
    stypes = []
    kept, discarded = filter_prediagrams([pd_ok, pd_bad], vtypes, stypes)
    assert len(kept) == 1
    assert discarded == 1


def test_filter_multiple_vertex_types():
    """Multiple vertex types available — both degree signatures accepted."""
    pd1 = _make_pd([(0, 2), (2, 1)], leaves=[0, 1])  # in=1, out=1
    pd2 = _make_pd([(0, 3), (3, 1), (3, 2)], leaves=[0, 1, 2])  # in=1, out=2
    vtypes = [_make_vertex_type(1, 1), _make_vertex_type(1, 2)]
    stypes = []
    kept, discarded = filter_prediagrams([pd1, pd2], vtypes, stypes)
    assert len(kept) == 2
    assert discarded == 0

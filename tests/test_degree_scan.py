"""
tests/test_degree_scan.py
=========================
Tests for msrjd.enumeration.degree_scan — Build Phase C.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_degree_scan.py -v
"""

import os, sys, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import DiGraph, Graph

from msrjd.enumeration.degree_scan import (
    max_vertex_degree, scan_source_vertices,
    check_taylor_order, ensure_taylor_order,
)
from msrjd.core.field_theory import FieldTheory
from msrjd.core.serialize import save_theory


# ── Helpers: build mock prediagrams ──────────────────────────────────────────

def _make_prediagram(edges, leaves, n_vertices=None):
    """
    Build a mock prediagram tuple (D, G, leaves, internal).

    Parameters
    ----------
    edges : list of (u, v)
        Directed edges.
    leaves : list of int
        Leaf (external leg) vertex IDs.
    """
    D = DiGraph(edges)
    if n_vertices:
        for v in range(n_vertices):
            if v not in D.vertices():
                D.add_vertex(v)
    G = D.to_undirected()
    all_verts = set(D.vertices())
    leaf_set = set(leaves)
    internal = sorted(all_verts - leaf_set)
    return (D, G, list(leaves), internal)


def _tree_prediagram():
    """
    A simple tree-level (ell=0) prediagram for k=2:

        0 (leaf) -> 2 (internal) -> 1 (leaf)

    Internal vertex 2 has in_degree=1, out_degree=1 → total degree 2.
    """
    return _make_prediagram([(0, 2), (2, 1)], leaves=[0, 1])


def _degree3_prediagram():
    """
    Prediagram with a degree-3 internal vertex:

        0 (leaf) -> 3 (internal, degree 3) -> 1 (leaf)
                                            -> 2 (leaf)

    Vertex 3: in_degree=1, out_degree=2 → total degree 3.
    """
    return _make_prediagram([(0, 3), (3, 1), (3, 2)], leaves=[0, 1, 2])


def _degree4_prediagram():
    """
    Prediagram with a degree-4 internal vertex:

        0 -> 4, 1 -> 4, 4 -> 2, 4 -> 3

    Vertex 4: in=2, out=2 → total degree 4.
    """
    return _make_prediagram([(0, 4), (1, 4), (4, 2), (4, 3)], leaves=[0, 1, 2, 3])


def _source_prediagram():
    """
    Prediagram with a source vertex (in_degree=0, not a leaf):

        2 (source, out=2) -> 0 (leaf)
                          -> 1 (leaf)

    Vertex 2: in_degree=0, out_degree=2 → source with out_degree 2.
    """
    return _make_prediagram([(2, 0), (2, 1)], leaves=[0, 1])


# ── Tests: max_vertex_degree ────────────────────────────────────────────────

def test_max_degree_tree():
    pds = [_tree_prediagram()]
    assert max_vertex_degree(pds) == 2


def test_max_degree_3():
    pds = [_degree3_prediagram()]
    assert max_vertex_degree(pds) == 3


def test_max_degree_4():
    pds = [_degree4_prediagram()]
    assert max_vertex_degree(pds) == 4


def test_max_degree_across_multiple():
    """Max degree is taken across all prediagrams."""
    pds = [_tree_prediagram(), _degree3_prediagram(), _degree4_prediagram()]
    assert max_vertex_degree(pds) == 4


def test_max_degree_empty():
    assert max_vertex_degree([]) == 0


def test_leaves_excluded():
    """Leaf vertices should not count toward max degree."""
    # Leaf vertex 0 has degree 1 in the graph but should be ignored
    pd = _degree3_prediagram()
    D = pd[0]
    # Leaves 1, 2 have in_degree 1 each in the DiGraph
    # Only internal vertex 3 should be counted
    assert max_vertex_degree([pd]) == 3


# ── Tests: scan_source_vertices ─────────────────────────────────────────────

def test_source_scan_finds_source():
    pds = [_source_prediagram()]
    assert scan_source_vertices(pds) == {2}


def test_source_scan_no_sources():
    """A prediagram where all non-leaf vertices have in_degree > 0."""
    pds = [_degree3_prediagram()]
    # Vertex 3 has in_degree=1 → not a source
    # Leaves are excluded
    # But wait — leaves 1, 2 have in_degree=1, leaf 0 has in_degree=0
    # Leaf 0 is excluded because it's a leaf
    assert scan_source_vertices(pds) == set()


def test_source_scan_mixed():
    """Mix of source and non-source prediagrams."""
    pds = [_source_prediagram(), _degree3_prediagram()]
    assert scan_source_vertices(pds) == {2}


# ── Tests: check_taylor_order ──────────────────────────────────────────────

def test_check_sufficient():
    meta = {'taylor_order': 4}
    sufficient, current, required = check_taylor_order(meta, 3)
    assert sufficient is True
    assert current == 4
    assert required == 3


def test_check_insufficient():
    meta = {'taylor_order': 2}
    sufficient, current, required = check_taylor_order(meta, 4)
    assert sufficient is False
    assert current == 2
    assert required == 4


def test_check_exact():
    meta = {'taylor_order': 3}
    sufficient, current, required = check_taylor_order(meta, 3)
    assert sufficient is True


# ── Tests: ensure_taylor_order ──────────────────────────────────────────────

def _load_hawkes_model():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    sys.path.insert(0, models_dir)
    from hawkes_sage import HAWKES_MODEL
    return HAWKES_MODEL


def test_ensure_no_reexpansion():
    """When Taylor order is sufficient, no re-expansion happens."""
    model = _load_hawkes_model()
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()

    tmpdir = tempfile.mkdtemp()
    try:
        theory_path = os.path.join(tmpdir, 'test_theory')
        model_file = os.path.relpath(
            os.path.join(os.path.dirname(__file__), '..', 'models', 'hawkes_sage.py'),
            os.path.dirname(__file__) + '/..',
        )
        save_theory(theory_path, ft, model_file=model_file,
                    model_var_name='HAWKES_MODEL')

        # Prediagrams with max degree 3 — order 4 is sufficient
        pds = [_degree3_prediagram()]
        project_root = os.path.join(os.path.dirname(__file__), '..')
        meta, data = ensure_taylor_order(theory_path, pds,
                                         project_root=project_root)
        assert meta['taylor_order'] == 4
    finally:
        shutil.rmtree(tmpdir)


def test_ensure_reexpands():
    """When Taylor order is insufficient, re-expansion happens."""
    model = _load_hawkes_model()
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()

    tmpdir = tempfile.mkdtemp()
    try:
        theory_path = os.path.join(tmpdir, 'test_theory')
        model_file = os.path.relpath(
            os.path.join(os.path.dirname(__file__), '..', 'models', 'hawkes_sage.py'),
            os.path.dirname(__file__) + '/..',
        )
        save_theory(theory_path, ft, model_file=model_file,
                    model_var_name='HAWKES_MODEL')

        # Prediagrams need degree 4
        pds = [_degree4_prediagram()]
        project_root = os.path.join(os.path.dirname(__file__), '..')
        meta, data = ensure_taylor_order(theory_path, pds,
                                         project_root=project_root)
        assert meta['taylor_order'] == 4
    finally:
        shutil.rmtree(tmpdir)

"""
tests/test_tree_bound_sharpened.py
==================================
The tree-decomposition bound |V2^T| <= k + 3*ell - 2*j - 1 (sharpened from
k + 3*ell - j - 1) must not change the enumeration output, and its per-class
bounds must sum to the orientability cap 3k + 3*ell - 3 for every j.

Run:  sage -python -m pytest tests/test_tree_bound_sharpened.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import daedalus.engine.enumeration.loop_diagram_enumeration as L

# Table I of the paper (prediagrams, counted up to isomorphism)
TABLE = {(2, 1): 9, (3, 1): 80, (4, 1): 755, (2, 2): 283, (1, 3): 434}


def test_bounds_sum_to_the_cap_for_every_j():
    for k in range(1, 7):
        for ell in range(0, 5):
            for j in range(0, ell + ell // 2 + 1):
                assert (k + j) + (k + j - 2) + (k + 3 * ell - 2 * j - 1) == L.v_max_orientable_bound(k, ell)


@pytest.mark.parametrize('k,ell', sorted(TABLE))
def test_prediagram_counts_unchanged(k, ell):
    trees, topos, pds, _ = L.enumerate_all(k, ell, n_threads=1, verbose=False)
    assert len(pds) == TABLE[(k, ell)]


def test_trees_respect_sharpened_bound():
    for k, ell in ((3, 1), (2, 2), (3, 2)):
        for tree, j, n_leaves in L.generate_trees_with_constraints(k, ell):
            leaves, _int, d2, d3 = L.classify_vertices_sage(tree)
            assert len(leaves) == k + j == n_leaves
            assert len(d3) <= k + j - 2
            assert len(d2) <= k + 3 * ell - 2 * j - 1
            assert tree.order() <= 3 * k + 3 * ell - 3


# -- lift the bound (audit W5, TST-08): the v2_max bound must not drop any prediagram -------------

def _lifted_trees(k, ell, extra=2):
    """``generate_trees_with_constraints`` with the |V2^T| bound removed: no per-tree degree-2
    check, and the vertex range opened ``extra`` beyond the orientability cap (which the v2 bound
    implies).  The other filters (leaf count, |V3^T| <= k + j - 2, the 2-path endpoint condition)
    are unchanged."""
    from sage.all import graphs
    out = []
    cap = L.v_max_orientable_bound(k, ell) + extra
    for j in range(0, ell + ell // 2 + 1):
        num_leaves = k + j
        v3_max = k + j - 2
        min_n = num_leaves if num_leaves == 1 else num_leaves + 1
        for n in range(min_n, cap + 1):
            for tree in graphs.trees(n):
                leaves, _int, degree_2, degree_3plus = L.classify_vertices_sage(tree)
                if len(leaves) != num_leaves or len(degree_3plus) > v3_max:
                    continue
                if not L.endpoint_condition_holds(tree, degree_2, j, ell):
                    continue
                out.append((tree, j, num_leaves))
    return out


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1), (2, 2)])
def test_lifting_the_v2_bound_drops_no_prediagram(monkeypatch, k, ell):
    def certs():
        return {L._iso_cert(D) for D, _G, _lv, _in in L.enumerate_all(k, ell, verbose=False)[2]}
    bounded = certs()
    assert len(bounded) == TABLE[(k, ell)]
    bounded_trees = L.generate_trees_with_constraints(k, ell)
    lifted_trees = _lifted_trees(k, ell)
    assert len(lifted_trees) > len(bounded_trees), 'the lifted search must look at more trees'
    # the comparison is sensitive: one less degree-2 vertex allowed drops prediagrams (the bound
    # is attained)
    tight = [t for t in bounded_trees
             if len(L.classify_vertices_sage(t[0])[2]) < k + 3 * ell - 2 * t[1] - 1]
    monkeypatch.setattr(L, 'v_max_orientable_bound', lambda _k, _l: 10 ** 6)
    monkeypatch.setattr(L, 'generate_trees_with_constraints', lambda _k, _l, _m=50: tight)
    assert certs() < bounded
    monkeypatch.setattr(L, 'generate_trees_with_constraints', lambda _k, _l, _m=50: lifted_trees)
    assert certs() == bounded

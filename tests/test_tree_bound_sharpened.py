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

import engine.enumeration.loop_diagram_enumeration as L

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

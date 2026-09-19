"""
tests/test_orientation_integer.py
=================================
The (E3) orientation stage now runs the causal checks on plain integers and
builds a Sage DiGraph only for the orientations that pass
(``enumerate_orientations_integer``).  It must return exactly the prediagrams
of the former Sage-checked path (``enumerate_orientations_sage``), which is
kept and selectable with ``DAEDALUS_ORIENTATION_CHECKER=sage``.

Measured stage time, serial: (3,2) 1.1 s -> 0.6 s, (4,2) 20.0 s -> 9.9 s,
(2,3) 7.0 s -> 2.7 s; of what remains, the survivors' certificates are ~60%.

Run:  sage -python -m pytest tests/test_orientation_integer.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.enumeration.loop_diagram_enumeration as L


def _prediagram_certs(k, ell, checker):
    old = L.ORIENTATION_CHECKER
    L.ORIENTATION_CHECKER = checker
    try:
        certs = set()
        for blob in L.stream_topology_certs(k, ell, 1, 50, False):
            certs |= L._topology_cert_to_prediagram_certs(blob)
        return certs
    finally:
        L.ORIENTATION_CHECKER = old


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (1, 3), (2, 3)])
def test_integer_checker_matches_sage_checker(k, ell):
    assert _prediagram_certs(k, ell, 'integer') == _prediagram_certs(k, ell, 'sage')


def test_default_is_integer_and_counts_hold():
    assert L.ORIENTATION_CHECKER == 'integer'
    assert len(_prediagram_certs(2, 3, 'integer')) == 14928
    assert len(_prediagram_certs(1, 4, 'integer')) == 22332


def test_each_rule_rejects_what_it_should():
    """Hand cases on one topology: the two-leaf bubble (leaves 0,1; internal
    2,3 joined by a double edge).  Forcings fix the leaf edges; the two free
    copies between 2 and 3 give four patterns, of which exactly one is causal:
    anti-parallel copies are a directed 2-cycle (Kahn), parallel copies out of
    the vertex whose leaf edge also leaves it make the other vertex a sink with
    in-degree 3 (no internal sink) -- only 'both copies 2 -> 3' or
    'both 3 -> 2' survive, and they are isomorphic, so the topology yields
    one prediagram (the tree-level C_2 bubble has |Aut|=2)."""
    from sage.all import Graph
    G = Graph([(0, 2), (1, 3), (2, 3), (2, 3)], multiedges=True)
    Ds = L.enumerate_orientations_integer(G, [0, 1])
    assert len(Ds) == 2
    assert len({L._iso_cert(D) for D in Ds}) == 1
    D = Ds[0]
    assert D.is_directed_acyclic()
    src = [v for v in D.vertices() if D.in_degree(v) == 0]
    assert len(src) == 1 and D.out_degree(src[0]) == 3
    # a contradictory forcing (leaf adjacent to a degree-2 vertex that must
    # point away from it) returns no orientation, as before
    P = Graph([(0, 1), (1, 2), (2, 3), (1, 3)], multiedges=True)   # vertex 1 has degree 3, 2 has degree 2
    assert L.enumerate_orientations_integer(P, [0]) == L.enumerate_orientations_sage(P, [0])

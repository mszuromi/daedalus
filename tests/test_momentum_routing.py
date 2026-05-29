"""
tests/test_momentum_routing.py
==============================
Unit tests for the spatial momentum-routing pass
(``msrjd.integration.spatial.momentum_routing.route_momenta``), the
§4a pre-integration step of the spatial re-architecture.

Uses SYNTHETIC diagram graphs (no enumeration pipeline) so the tests
are fast and isolate the routing linear-algebra:

  * tree 2-point  → L=0, every edge carries ±q (k² = q² uniformly)
  * 1-loop bubble → L=1, two internal edges carry ℓ and -(q+ℓ)
  * 2-loop sunset → L=2

Run:  sage -python -m pytest tests/test_momentum_routing.py -q
"""
from __future__ import annotations

import os
import sys

import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import DiGraph
from msrjd.integration.spatial.momentum_routing import route_momenta


class _FakeTD:
    """Minimal stand-in exposing only ``.prediagram`` = (D, G, leaves,
    internal), which is all ``route_momenta`` reads."""
    def __init__(self, D, leaves, internal):
        self.prediagram = (D, None, leaves, internal)


def _tree_2pt():
    D = DiGraph(multiedges=True)
    D.add_vertices([0, 1, 2])
    D.add_edge(2, 0, 0)
    D.add_edge(2, 1, 1)
    return _FakeTD(D, leaves=[0, 1], internal=[2])


def _bubble():
    D = DiGraph(multiedges=True)
    D.add_vertices([0, 1, 2, 3])
    D.add_edge(2, 0, 0)
    D.add_edge(3, 1, 1)
    D.add_edge(2, 3, 2)
    D.add_edge(2, 3, 3)
    return _FakeTD(D, leaves=[0, 1], internal=[2, 3])


def _sunset():
    # 2 leaves, 2 internal, THREE edges between them → 2 loops.
    D = DiGraph(multiedges=True)
    D.add_vertices([0, 1, 2, 3])
    D.add_edge(2, 0, 0)
    D.add_edge(3, 1, 1)
    D.add_edge(2, 3, 2)
    D.add_edge(2, 3, 3)
    D.add_edge(2, 3, 4)
    return _FakeTD(D, leaves=[0, 1], internal=[2, 3])


def test_tree_no_loops_uniform_q():
    r = route_momenta(_tree_2pt())
    assert r.n_loops == 0
    q0 = r.q_syms[0]
    k2 = r.edge_k2()
    # every edge carries ±q  ⇒  k² = q² on every edge
    assert all(sp.expand(v - q0**2) == 0 for v in k2.values())


def test_bubble_one_loop():
    r = route_momenta(_bubble())
    assert r.n_loops == 1
    q0, l0 = r.q_syms[0], r.loop_syms[0]
    mom = r.edge_momenta
    loop_edges = [mom[(2, 3, 2)], mom[(2, 3, 3)]]
    # exactly one loop momentum present, and it cancels in the leg sum
    assert any(l0 in m.free_symbols for m in loop_edges)
    assert l0 not in sp.expand(sum(loop_edges)).free_symbols
    # external legs carry ±q only (no loop momentum)
    assert l0 not in mom[(2, 0, 0)].free_symbols
    assert l0 not in mom[(3, 1, 1)].free_symbols


def test_sunset_two_loops():
    r = route_momenta(_sunset())
    assert r.n_loops == 2


def test_external_momentum_conservation():
    """The two external-leg momenta must be opposite (Σq = 0)."""
    r = route_momenta(_tree_2pt())
    mom = r.edge_momenta
    leg0, leg1 = mom[(2, 0, 0)], mom[(2, 1, 1)]
    assert sp.expand(leg0 + leg1) == 0

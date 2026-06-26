"""
tests/test_momentum_routing.py
==============================
Unit tests for the spatial momentum-routing pass
(``engine.integration.spatial.momentum_routing.route_momenta``), the
§4a pre-integration step of the spatial re-architecture.

Most tests use SYNTHETIC diagram graphs (no enumeration pipeline) so they
are fast and isolate the routing linear-algebra:

  * tree 2-point  → L=0, every edge carries ±q (k² = q² uniformly)
  * 1-loop bubble → L=1, two internal edges carry ℓ and -(q+ℓ)
  * 2-loop sunset → L=2

plus one integration test that routes the REAL typed diagrams the
enumeration pipeline produces for Allen-Cahn (the objects Stage C will
feed to the per-edge momentum substitution), checking that tree edges
carry ±q only, every 1-loop diagram has exactly one loop momentum, and
the Hartree tadpole (a k=0 connecting line) is present.

Run:  sage -python -m pytest tests/test_momentum_routing.py -q
"""
from __future__ import annotations

import os
import sys

import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import DiGraph
from engine.integration.spatial.momentum_routing import route_momenta


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


def test_edge_coeffs_reconstructs_momenta():
    """edge_coeffs() must reconstruct each edge momentum EXACTLY:
    k_e = Σ_i a_{ei} ℓ_i + Σ_j b_{ej} q_j.  This is the backend-C C0 input
    (the signed routing coefficients edge_k2 discards by squaring)."""
    for td in (_tree_2pt(), _bubble(), _sunset()):
        r = route_momenta(td)
        coeffs = r.edge_coeffs()
        for e, m in r.edge_momenta.items():
            a, b = coeffs[e]
            recon = (sum(a[i] * r.loop_syms[i] for i in range(len(r.loop_syms)))
                     + sum(b[j] * r.q_syms[j] for j in range(len(r.q_syms))))
            assert sp.expand(m - recon) == 0
            assert all(isinstance(c, int) for c in a + b)   # ±1/0 integers


def test_edge_coeffs_bubble_structure():
    """The bubble's two loop edges carry ±ℓ (a=(+1,) and (−1,)); the external
    legs carry no loop momentum (a=(0,)).  Matches the [1,−1] the 1-loop
    sigma_*_kernel hardcoded — which C0 now reads from routing instead."""
    r = route_momenta(_bubble())
    coeffs = r.edge_coeffs()
    loop_edges = [coeffs[(2, 3, 2)], coeffs[(2, 3, 3)]]
    assert sorted(a[0] for a, b in loop_edges) == [-1, 1]
    assert coeffs[(2, 0, 0)][0] == (0,) and coeffs[(3, 1, 1)][0] == (0,)


def test_edge_coeffs_sunset_two_loops():
    """The sunset's three internal edges, two loop momenta: edge_coeffs gives a
    2-vector ``a`` per edge; reconstruction is exact and the rows span L=2."""
    r = route_momenta(_sunset())
    coeffs = r.edge_coeffs()
    assert all(len(a) == 2 for a, b in coeffs.values())     # two loop momenta
    internal = [coeffs[(2, 3, lbl)][0] for lbl in (2, 3, 4)]
    assert sp.Matrix(internal).rank() == 2


def test_external_momentum_conservation():
    """The two external-leg momenta must be opposite (Σq = 0)."""
    r = route_momenta(_tree_2pt())
    mom = r.edge_momenta
    leg0, leg1 = mom[(2, 0, 0)], mom[(2, 1, 1)]
    assert sp.expand(leg0 + leg1) == 0


def test_route_real_enumerated_allen_cahn_diagrams():
    """Route the REAL typed diagrams from the enumeration pipeline (not
    synthetic graphs) for the Allen-Cahn λφ³ theory.

    Invariants Stage C relies on:
      * every tree (ℓ=0) diagram has L=0 and all edges k² = q₀²;
      * every 1-loop (ℓ=1) diagram has exactly one loop momentum, present
        on ≥1 edge alone (k² = ℓ₀², the clean integration variable);
      * the Hartree tadpole is present — a 1-loop diagram with an internal
        edge forced to k = 0 by momentum conservation (the line joining the
        closed self-loop to the external backbone).
    """
    import importlib.util

    from engine.core.field_theory import FieldTheory
    from engine.core.vertices import (
        extract_vertex_types, extract_source_types,
    )
    from engine.diagrams.type_assignment import build_field_index_map
    from api._propagator import build_propagator
    from api._diagrams import enumerate_unique_diagrams

    p = os.path.join(os.path.dirname(__file__), '..', 'theories',
                     'allen_cahn_1d_subcritical_infinite.theory.py')
    spec = importlib.util.spec_from_file_location('m', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build()

    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    vt = extract_vertex_types(ft)
    st = extract_source_types(ft)
    rv = list(ft._ns._ring_var_names)
    nt = ft._n_tilde
    ri, pi = build_field_index_map(rv, nt)
    ext = [('dphi', 1), ('dphi', 1)]
    ub, _, _ = enumerate_unique_diagrams(
        ft, model, k=2, max_ell=1, external_fields=ext, G_ft=prop['G_ft'],
        resp_idx=ri, phys_idx=pi, vtypes=vt, stypes=st,
        use_cache=False, verbose=False)

    # ℓ=0: tree, no loops, every edge k² = q₀²
    trees = ub.get(0, [])
    assert len(trees) >= 1
    for td in trees:
        r = route_momenta(td)
        assert r.n_loops == 0
        q0 = r.q_syms[0]
        assert all(sp.expand(v - q0**2) == 0 for v in r.edge_k2().values())

    # ℓ=1: exactly one loop momentum; a loop-only edge and a k=0 tadpole line
    one_loops = ub.get(1, [])
    assert len(one_loops) >= 1
    saw_loop_only_edge = False
    saw_tadpole_zero_line = False
    for td in one_loops:
        r = route_momenta(td)
        assert r.n_loops == 1
        assert len(r.loop_syms) == 1
        l0 = r.loop_syms[0]
        k2vals = list(r.edge_k2().values())
        # the loop momentum must actually appear
        assert any(l0 in sp.expand(v).free_symbols for v in k2vals)
        if any(sp.expand(v - l0**2) == 0 for v in k2vals):
            saw_loop_only_edge = True
        if any(sp.expand(v) == 0 for v in k2vals):
            saw_tadpole_zero_line = True
    assert saw_loop_only_edge, \
        'expected an edge carrying the loop momentum alone (k² = ℓ₀²)'
    assert saw_tadpole_zero_line, \
        'expected the Hartree tadpole (a k = 0 connecting line)'

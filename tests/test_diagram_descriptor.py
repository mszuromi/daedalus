"""
tests/test_diagram_descriptor.py
================================
Generic spatial loop pipeline — **Phase 1**: ``diagram_descriptor.diagram_to_cstack``
maps an enumerated typed diagram onto the C-stack edge list.  Validated on the
reaction-diffusion (φ̃φ²) model, whose ell=1 inventory is known
(``docs/spatial_loop_diagram_inventory.md``): 2 bubbles + 1 tadpole.

Asserts (classifying by STRUCTURE, not diagram index):
  * tree (ell=0): no internal vertices, a single external C edge between the
    two leaves, 0 loops;
  * ell=1: exactly 3 diagrams, 1 loop each; exactly ONE is tadpole-like (a C
    self-loop, ``u==v``) and TWO are bubbles (2 loop edges between the two
    internal vertices, no self-loop);
  * the two bubbles reproduce ``bubble_edges`` structure — one Σ_R-type
    (loop edges = {one R + one C}) and one Σ_K-type (loop edges = {C, C});
  * every external leg edge has loop-coefficient a = 0 (drops out of ∫dᵈℓ).

Run:  sage -python -m pytest tests/test_diagram_descriptor.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
from engine.integration.spatial.pipeline_bridge import (
    build_pipeline_records, _legs_to_phys_idx,
)
from engine.diagrams.type_assignment import build_field_index_map


@pytest.fixture(scope='module')
def rd_records():
    """Reaction-diffusion (φ̃φ²) d=1: built once, ell=0 and ell=1 records."""
    from api.model import ModelBuilder
    from api._propagator import build_propagator
    from api.compute import FieldTheory

    model = (ModelBuilder('rd1', n_populations=0)
             .physical_field('phi', spatial_dim=1)
             .parameter('mu', default=1.0, domain='positive')
             .parameter('D', default=1.0, domain='positive')
             .parameter('g', default=0.3, domain='real')
             .parameter('T', default=1.0, domain='positive')
             .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2')
             .set_action_text('phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
             .boundary('infinite').initial('stationary').build())
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    ft.sanity_check(verbose=False)
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], phys_idx)
    by_ell = build_pipeline_records(ft, model, prop, ext, max_ell=1, verbose=False)
    return by_ell


def test_tree_is_single_external_C_edge(rd_records):
    tree = [diagram_to_cstack(td) for td, _ in rd_records[0]]
    assert len(tree) == 1
    d = tree[0]
    assert d.n_loops == 0
    assert d.internal_vertices == ()                       # no interaction vertices
    assert len(d.edges) == 1                               # one C line
    e = d.edges[0]
    assert e.kind == 'C' and e.external
    assert set((e.u, e.v)) == set(d.external_legs)         # connects the two leaves
    assert not e.couples_loop()                            # no loop momentum


def test_ell1_inventory_two_bubbles_one_tadpole(rd_records):
    descr = [diagram_to_cstack(td) for td, _ in rd_records[1]]
    assert len(descr) == 3
    assert all(d.n_loops == 1 for d in descr)

    tadpoles = [d for d in descr if d.is_tadpole_like()]
    bubbles = [d for d in descr if not d.is_tadpole_like()]
    assert len(tadpoles) == 1, "expected exactly one tadpole (C self-loop)"
    assert len(bubbles) == 2, "expected exactly two bubbles"

    # tadpole: the self-loop edge carries pure loop momentum (a != 0, b == 0)
    tad = tadpoles[0]
    selfloops = [e for e in tad.edges if e.u == e.v]
    assert len(selfloops) == 1
    sl = selfloops[0]
    assert sl.kind == 'C' and sl.couples_loop()
    assert all(bi == 0 for bi in sl.b), "tadpole self-loop must not carry q"

    # bubbles: 2 loop edges between the SAME two internal vertices, no self-loop
    for b in bubbles:
        assert len(b.internal_vertices) == 2
        le = b.loop_edges()
        assert len(le) == 2
        assert all(e.u != e.v for e in le)
        assert all(e.couples_loop() for e in le)           # both carry ℓ

    # one Σ_R-type (kinds {R, C}) and one Σ_K-type (kinds {C, C})
    kindsets = sorted(tuple(sorted(e.kind for e in b.loop_edges())) for b in bubbles)
    assert kindsets == [('C', 'C'), ('C', 'R')], f"got loop-edge kinds {kindsets}"


def test_external_legs_have_zero_loop_coefficient(rd_records):
    for ell in (0, 1):
        for td, _ in rd_records[ell]:
            d = diagram_to_cstack(td)
            for e in d.edges:
                if e.external:
                    assert all(ai == 0 for ai in e.a), (
                        f"external leg {(e.u, e.v)} carries loop momentum a={e.a}")

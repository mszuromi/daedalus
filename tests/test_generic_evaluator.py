"""
tests/test_generic_evaluator.py
================================
Generic spatial loop pipeline — **Phase 2**: the per-diagram evaluator
(``generic_evaluator``).

Phase 2a (here): the descriptor's bubble **loop edges**, fed through
``sigma_parametric``, reproduce the hand-coded ``bubble_edges`` oracle — i.e. the
Phase-1 mapping produces the right loop kinematics (the Σ_R bubble matches
``bubble_edges('R')``, the Σ_K bubble matches ``bubble_edges('C')``), over a grid
of ``(q, t)``.

Run:  sage -python -m pytest tests/test_generic_evaluator.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
from msrjd.integration.spatial.generic_evaluator import loop_self_energy
from msrjd.integration.spatial.temporal_integrate import (
    sigma_parametric, bubble_edges,
)
from msrjd.integration.spatial.pipeline_bridge import (
    build_pipeline_records, _legs_to_phys_idx,
)
from msrjd.diagrams.type_assignment import build_field_index_map


@pytest.fixture(scope='module')
def rd_ell1():
    from pipeline.theory import TheoryBuilder
    from pipeline._propagator import build_propagator
    from pipeline.compute import FieldTheory

    model = (TheoryBuilder('rd1', n_populations=0)
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
    return [diagram_to_cstack(td) for td, _ in by_ell[1]]


def _classify(descr_list):
    """Return (sigma_R_descr, sigma_K_descr) by loop-edge kind structure."""
    bubbles = [d for d in descr_list if not d.is_tadpole_like()]
    sR = sK = None
    for b in bubbles:
        kinds = sorted(e.kind for e in b.loop_edges())
        if kinds == ['C', 'R']:
            sR = b
        elif kinds == ['C', 'C']:
            sK = b
    assert sR is not None and sK is not None
    return sR, sK


def test_bubble_loop_kinematics_match_oracle(rd_ell1):
    """Σ_R / Σ_K from the descriptor's loop edges == sigma_parametric on the
    hand-coded bubble_edges, over a (q,t) grid (the Phase-1 mapping is right)."""
    sR, sK = _classify(rd_ell1)
    mu, D, T = 1.0, 1.0, 1.0
    for q in (0.3, 0.9, 1.7):
        for t in (0.05, 0.4, 1.2):
            got_R = loop_self_energy(sR, q, t, mu, D, T)
            ref_R = sigma_parametric(bubble_edges('R'), q, t, mu, D, T)
            assert abs(got_R - ref_R) <= 1e-9 * (abs(ref_R) + 1e-12), \
                f"Σ_R mismatch at q={q},t={t}: {got_R} vs {ref_R}"

            got_K = loop_self_energy(sK, q, t, mu, D, T)
            ref_K = sigma_parametric(bubble_edges('C'), q, t, mu, D, T)
            assert abs(got_K - ref_K) <= 1e-9 * (abs(ref_K) + 1e-12), \
                f"Σ_K mismatch at q={q},t={t}: {got_K} vs {ref_K}"

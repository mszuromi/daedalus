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

import numpy as np
from sage.all import SR

import importlib.util

from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
from engine.integration.spatial.generic_evaluator import (
    loop_self_energy, bubble_delta_C, tadpole_delta_C,
)
from engine.integration.spatial.temporal_integrate import (
    sigma_parametric, bubble_edges,
)
from engine.integration.spatial import loop_dyson
from engine.integration.spatial.pipeline_bridge import (
    build_pipeline_records, _legs_to_phys_idx,
)
from engine.diagrams.type_assignment import build_field_index_map


@pytest.fixture(scope='module')
def rd_ell1():
    from api.theory import TheoryBuilder
    from api._propagator import build_propagator
    from api.compute import FieldTheory

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
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('g'): 0.3,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    return [(diagram_to_cstack(td), float(SR(pre).subs(base)))
            for td, pre in by_ell[1]]


def _classify(pairs):
    """Return (sigma_R_descr, sigma_K_descr) by loop-edge kind structure."""
    bubbles = [d for d, _ in pairs if not d.is_tadpole_like()]
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


def test_generic_bubble_dC_matches_loop_dyson(rd_ell1):
    """Phase 2b: the GENERIC per-diagram δC (Symanzik σ + single-mode Dyson
    convolution, weighted by 2^{-n_C}·𝒮(Γ)) summed over the bubble diagrams
    reproduces the bespoke loop_dyson δC(q,τ) — with NO pinned C_R/C_K (the
    weights come from the enumeration 𝒮(Γ): 16/4=4=C_R, 8/4=2=C_K)."""
    mu, D, T = 1.0, 1.0, 1.0
    A, B = mu, D                                    # single tree mode mass (μ,D)
    bubbles = [(d, pre) for d, pre in rd_ell1 if not d.is_tadpole_like()]
    assert len(bubbles) == 2
    for q in (0.4, 0.9, 1.6):
        taus = np.array([0.0, 0.5, 1.0, 2.0])
        gen = np.zeros(len(taus))
        for d, pre in bubbles:
            gen = gen + bubble_delta_C(d, pre, q, taus, A, B, mu, D)
        ref = loop_dyson.bubble_delta_C_q_tau(q, taus, mu, D, T, g=0.3)
        # coarse-grid convolution + interpolation ⇒ ~1% agreement is the bar
        for i, tau in enumerate(taus):
            denom = abs(ref[i]) + 1e-12
            assert abs(gen[i] - ref[i]) <= 1.5e-2 * denom, \
                f"δC mismatch q={q},τ={tau}: generic={gen[i]:.6e} vs loop_dyson={ref[i]:.6e}"


@pytest.fixture(scope='module')
def allencahn_ell1():
    """Allen-Cahn (φ̃φ³) d=1, λ=0.1: ell=1 (descriptor, 𝒮(Γ)·prefactor) pairs."""
    from api._propagator import build_propagator
    from api.compute import FieldTheory
    repo = os.path.join(os.path.dirname(__file__), '..')
    spec = importlib.util.spec_from_file_location(
        'ac', os.path.join(repo, 'theories',
                           'allen_cahn_1d_subcritical_infinite.theory.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build()
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    ft.sanity_check(verbose=False)
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], phys_idx)
    by_ell = build_pipeline_records(ft, model, prop, ext, max_ell=1, verbose=False)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('T'): 1.0,
            SR.var('lam'): 0.1, SR.var('phistar1'): 0.0}
    return [(diagram_to_cstack(td), float(SR(pre).subs(base)))
            for td, pre in by_ell[1]]


def test_tadpole_matches_allen_cahn_oracle(allencahn_ell1):
    """Phase 3: the GENERIC tadpole δC (instantaneous self-energy → mass shift,
    same Dyson machinery in the δ-limit) on Allen-Cahn reproduces the validated
    oracle.  Decisive end-to-end check: ∫dq/2π δC(q,0) == δ⟨φ²⟩ = 0.4625−0.5 =
    −0.0375 (the compute_spatial_correlator_one_loop number at λ=0.1) — with the
    coupling/combinatorics read from the enumeration 𝒮(Γ), NOT g=3λ hardcoded."""
    mu, D, T = 1.0, 1.0, 1.0
    A, B = mu, D
    tads = [(d, pre) for d, pre in allencahn_ell1
            if d.is_tadpole_like() and abs(pre) > 1e-9]
    assert len(tads) == 1, f"expected 1 live tadpole, got {len(tads)}"
    tad, pre = tads[0]

    # (a) shape: δC(q,τ) is a mass shift Σ·∂C₀/∂A (Σ from MY computation)
    for q in (0.0, 0.6, 1.4):
        taus = np.array([0.0, 0.5, 1.0])
        gen = tadpole_delta_C(tad, pre, q, taus, A, B, mu, D)
        m = A + B * q * q
        h = 1e-5
        dC0dA = ((T / (m + h)) * np.exp(-(m + h) * taus)
                 - (T / (m - h)) * np.exp(-(m - h) * taus)) / (2 * h)
        Sigma = gen[0] / dC0dA[0]                      # read Σ from τ=0
        for i in range(len(taus)):                     # same Σ at all τ
            assert abs(gen[i] - Sigma * dC0dA[i]) <= 1e-6 * (abs(gen[i]) + 1e-12)

    # (b) end-to-end: ∫dq/2π δC(q,0) == −0.0375 (the oracle's δ⟨φ²⟩ at λ=0.1)
    qg = np.linspace(0.0, 60.0, 6000)
    dC0 = np.array([tadpole_delta_C(tad, pre, float(q), np.array([0.0]),
                                    A, B, mu, D)[0] for q in qg])
    dvar = 2.0 * np.trapz(dC0, qg) / (2.0 * np.pi)     # even in q → 2∫₀
    assert abs(dvar - (-0.0375)) <= 5e-4, f"δ⟨φ²⟩={dvar:.6f}, expected −0.0375"

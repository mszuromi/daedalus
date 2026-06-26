"""
tests/test_spatial_pipeline_bridge.py
=====================================
Stage A production — the "symbolic-in-q bridge"
(``engine.integration.spatial.pipeline_bridge``).

These tests pin the bridge that routes a Tier-1 spatial theory THROUGH the
shared diagram pipeline (``compute_poles_and_residues`` +
``enumerate_unique_diagrams`` + ``compute_correction_td`` with
``Laplacian → -q²``), CERTIFIES that the pipeline's ``C(q, τ)`` equals the
propagator-derived heat-kernel modes ``Σ_α N_α/(A_α+B_α q²)e^{-(A_α+B_α q²)|τ|}``,
then does the external ``q → x`` FT analytically via ``free_two_point``.

Checks:
  * for every real spatial theory file the bridge reproduces the bespoke
    ``compute_spatial_correlator_tree`` oracle to <= 1e-10, and the pipeline
    certification residual is at machine precision (the diagrams the SHARED
    pipeline produced ARE the modes the analytic q-FT transforms);
  * the equal-time value matches the analytic closed form
    ``C(x,0) = T/(2√(μD)) e^{-|x|√(μ/D)}`` for the linear theories;
  * a Tier-2 (coupled) theory raises a clean NotImplementedError.

Run:  sage -python -m pytest tests/test_spatial_pipeline_bridge.py -q
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR

from engine.core.field_theory import FieldTheory
from api._propagator import build_propagator
from engine.integration.spatial.spatial_correlator import (
    compute_spatial_correlator_tree,
)
from engine.integration.spatial.pipeline_bridge import (
    compute_spatial_correlator_via_pipeline,
)
from api.theory import TheoryBuilder
from api import compute_cumulants

_THEORY_DIR = os.path.join(os.path.dirname(__file__), '..', 'theories')
_EXT = [('dphi', 1), ('dphi', 1)]
_TAUS = np.array([0.0, 0.5, 1.0])
_XS = np.array([0.0, 1.0, 2.0])


def _load(name):
    p = os.path.join(_THEORY_DIR, f'{name}.theory.py')
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.build()


def _setup(name):
    model = _load(name)
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    return ft, model, prop


# (name, params, closed_form_mu_D_T_or_None)
_CASES = [
    ('linear_diffusion_test', {'mu': 1.0, 'D': 1.0, 'T': 1.0}, (1.0, 1.0, 1.0)),
    ('edwards_wilkinson_1d', {'mu': 0.5, 'D': 2.0, 'T': 1.0}, (0.5, 2.0, 1.0)),
    ('allen_cahn_1d_subcritical_infinite',
     {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0}, (1.0, 1.0, 1.0)),
    ('allen_cahn_1d_subcritical_pbc',
     {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0, 'L': 20.0}, None),
]


@pytest.mark.parametrize('name,params,closed', _CASES,
                         ids=[c[0] for c in _CASES])
def test_bridge_matches_bespoke_oracle(name, params, closed):
    """The pipeline-routed bridge reproduces the bespoke oracle to <=1e-10,
    and its pipeline certification (diagram C(q,τ) vs propagator modes) is
    at machine precision."""
    ft, model, prop = _setup(name)
    nps = {SR.var(k): v for k, v in params.items()}
    nps[SR.var('phistar1')] = 0.0   # subcritical saddle (ignored if absent)

    Cb, _ = compute_spatial_correlator_tree(
        ft, model, prop, nps, _EXT, _TAUS, _XS, verbose=False)
    Cp, info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, nps, _EXT, _TAUS, _XS, verbose=False)

    assert info['pipeline_certified'], (
        f'{name}: pipeline certification failed '
        f'(max rel {info["certify_max_rel"]:.2e})')
    assert info['certify_max_rel'] < 1e-10, (
        f'{name}: certification residual {info["certify_max_rel"]:.2e} '
        f'not at machine precision')
    np.testing.assert_allclose(Cp, Cb, rtol=1e-9, atol=1e-12,
                               err_msg=f'{name}: bridge != bespoke oracle')


@pytest.mark.parametrize('name,params,closed',
                         [c for c in _CASES if c[2] is not None],
                         ids=[c[0] for c in _CASES if c[2] is not None])
def test_bridge_equal_time_matches_closed_form(name, params, closed):
    """Equal-time bridge value == analytic C(x,0) = T/(2√(μD)) e^{-|x|√(μ/D)}."""
    mu, D, T = closed
    ft, model, prop = _setup(name)
    nps = {SR.var(k): v for k, v in params.items()}
    nps[SR.var('phistar1')] = 0.0
    Cp, _ = compute_spatial_correlator_via_pipeline(
        ft, model, prop, nps, _EXT, _TAUS, _XS, verbose=False)
    it0 = int(np.argmin(np.abs(_TAUS)))
    closed_vals = T / (2 * math.sqrt(mu * D)) * np.exp(-_XS * math.sqrt(mu / D))
    np.testing.assert_allclose(Cp.real[it0], closed_vals,
                               rtol=1e-9, atol=1e-12)


def _two_field_model():
    return (
        TheoryBuilder('two-field decoupled bridge', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .physical_field('psi', spatial_dim=1)
        .parameter('mu1', default=1.0, domain='positive')
        .parameter('D1', default=1.0, domain='positive')
        .parameter('mu2', default=2.0, domain='positive')
        .parameter('D2', default=0.5, domain='positive')
        .parameter('T1', default=1.0, domain='positive')
        .parameter('T2', default=1.5, domain='positive')
        .set_action_text(
            'phit*((Dt+mu1-D1*Laplacian)*phi) '
            '+ psit*((Dt+mu2-D2*Laplacian)*psi) '
            '- T1*phit^2 - T2*psit^2')
        .equation(lhs='(Dt+mu1-D1*Laplacian)*phi', rhs='0')
        .equation(lhs='(Dt+mu2-D2*Laplacian)*psi', rhs='0')
        .boundary('infinite').initial('stationary').build())


_FUND2 = {'mu1': 1.0, 'D1': 1.0, 'mu2': 2.0, 'D2': 0.5, 'T1': 1.0, 'T2': 1.5}


@pytest.mark.parametrize('leg,fi,mu,D,T', [
    ('dphi', 0, 1.0, 1.0, 1.0),
    ('dpsi', 1, 2.0, 0.5, 1.5),
])
def test_bridge_multifield_resolves_each_field(leg, fi, mu, D, T):
    """The bridge must resolve the correct field INDEX for a 2-field
    decoupled theory and certify/transform each field's OWN heat-kernel
    mode (its own μ, D, T) — exercising the multi-field phys-column path
    the single-field bridge tests never touch."""
    model = _two_field_model()
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    nps = {SR.var(k): v for k, v in _FUND2.items()}
    ext = [(leg, 1), (leg, 1)]
    Cb, _ = compute_spatial_correlator_tree(
        ft, model, prop, nps, ext, _TAUS, _XS, verbose=False)
    Cp, info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, nps, ext, _TAUS, _XS, verbose=False)
    assert info['field_index'] == fi
    assert info['pipeline_certified'] and info['certify_max_rel'] < 1e-10
    np.testing.assert_allclose(Cp, Cb, rtol=1e-9, atol=1e-12)
    it0 = int(np.argmin(np.abs(_TAUS)))
    closed = T / (2 * math.sqrt(mu * D)) * np.exp(-_XS * math.sqrt(mu / D))
    np.testing.assert_allclose(Cp.real[it0], closed, rtol=1e-9, atol=1e-12)


def test_bridge_coupled_scalar_diffusion_supported():
    """A coupled (off-diagonal) multi-field theory with EQUAL (scalar) diffusion
    has no diagonal Tier-1 block, but the bridge now routes it to the spectral-
    Lyapunov coupled driver (Dyson 3b) and returns a finite C(x,τ)."""
    model = (
        TheoryBuilder('coupled spatial bridge (scalar D)', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .physical_field('psi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .parameter('g', default=0.3, domain='real')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt+mu-D*Laplacian)*phi + g*psi) '
            '+ psit*((Dt+mu-D*Laplacian)*psi + g*phi) '
            '- T*phit^2 - T*psit^2')
        .equation(lhs='(Dt+mu-D*Laplacian)*phi', rhs='-g*psi')
        .equation(lhs='(Dt+mu-D*Laplacian)*psi', rhs='-g*phi')
        .boundary('infinite').initial('stationary').build())
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    nps = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('g'): 0.3,
           SR.var('T'): 1.0}
    C, info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, nps, _EXT, _TAUS, _XS, verbose=False)
    assert info.get('coupled') is True
    assert np.all(np.isfinite(C.real))
    it0 = int(np.argmin(np.abs(_TAUS)))
    assert C[it0, 0].real > 0                      # C_phiphi(x=0, τ=0) is a variance


def test_bridge_coupled_unequal_diffusion_raises_clean():
    """A coupled theory with UNEQUAL diffusion (𝒟̂≠0) still needs the Dyson–Duhamel
    series, so the bridge raises a clean NotImplementedError."""
    model = (
        TheoryBuilder('coupled spatial bridge (unequal D)', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .physical_field('psi', spatial_dim=1)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D1', default=1.0, domain='positive')
        .parameter('D2', default=0.5, domain='positive')
        .parameter('g', default=0.3, domain='real')
        .parameter('T', default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt+mu-D1*Laplacian)*phi + g*psi) '
            '+ psit*((Dt+mu-D2*Laplacian)*psi + g*phi) '
            '- T*phit^2 - T*psit^2')
        .equation(lhs='(Dt+mu-D1*Laplacian)*phi', rhs='-g*psi')
        .equation(lhs='(Dt+mu-D2*Laplacian)*psi', rhs='-g*phi')
        .boundary('infinite').initial('stationary').build())
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    nps = {SR.var('mu'): 1.0, SR.var('D1'): 1.0, SR.var('D2'): 0.5,
           SR.var('g'): 0.3, SR.var('T'): 1.0}
    with pytest.raises(NotImplementedError, match='scalar-diffusion'):
        compute_spatial_correlator_via_pipeline(
            ft, model, prop, nps, _EXT, _TAUS, _XS, verbose=False)


# ── the GENERIC full-diagram 1-loop path (every diagram, no shortcut) ──
def test_generic_rd_full_diagram_1loop():
    """The φ̃φ² reaction-diffusion 1-loop through the GENERIC path, now powered by
    the full-diagram integrator: it runs (no close-pair hang), sums ALL 3 live
    ell=1 diagrams (2 bubbles + 1 tadpole) by the same genuine integral, and
    δC(x=0,τ=0) is finite and positive.  (The bubble part is validated == loop_dyson
    to ~1e-4 in tests, and the φ̃φ³ tadpole == the −0.0375 oracle; this end-to-end
    test just checks the φ̃φ² path runs and is sane — the multi-vertex tadpole's
    magnitude is left to the simulator.)"""
    from engine.integration.spatial.pipeline_bridge import (
        compute_spatial_correlator_generic,
    )
    g_true = 0.35
    model = _load('reaction_diffusion_quadratic_1d')
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    nps = {'mu': 1.0, 'D': 1.0, 'g': g_true, 'T': 1.0, 'phistar1': 0.0}
    taus = np.array([0.0, 1.0])
    xs = np.linspace(0.0, 6.0, 9)
    C1, info = compute_spatial_correlator_generic(
        ft, model, prop, nps, _EXT, taus, xs, verbose=False, n_q=48)
    assert info['generic'] is True and info['full_integrator'] is True
    assert info['n_ell1_diagrams'] == 3 and info['n_live_diagrams'] == 3
    C0, _ = compute_spatial_correlator_via_pipeline(
        ft, model, prop, nps, _EXT, taus, xs, verbose=False, certify=False)
    dC_x0 = float((C1[0, 0] - C0[0, 0]).real)
    assert np.all(np.isfinite(np.real(C1)))
    assert dC_x0 > 0                            # 1-loop correction is positive here


def test_bubble_loop_form_factor_extraction():
    """Phase 4c-2: ``bubble_loop_form_factor`` reads each interaction vertex's
    φ̃-leg momentum from ``route_momenta`` and assembles the derivative-vertex
    form factor.  For the φ̃φ² bubbles with a ``Lap`` chain (∇²φ²), the two
    self-energies give F = q²(q−ℓ)² (Σ_R: one external + one internal vertex)
    and F = q⁴ (Σ_K: both vertices external) — matching the hand derivation;
    an empty chain gives 1 (the plain bubble)."""
    import sympy as sp
    from engine.core.vertices import extract_vertex_types, extract_source_types
    from engine.diagrams.symmetry import classify_coefficient_factors
    from api._diagrams import enumerate_unique_diagrams
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.pipeline_bridge import (
        bubble_loop_form_factor, _diagram_is_bubble)

    model = _load('reaction_diffusion_quadratic_1d')
    ft = FieldTheory(model, taylor_order=3); ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    vt = extract_vertex_types(ft); st = extract_source_types(ft)
    rv = list(ft._ns._ring_var_names)
    ri, pi = build_field_index_map(rv, ft._n_tilde)
    ub, _, _ = enumerate_unique_diagrams(
        ft, model, k=2, max_ell=1, external_fields=[('dphi', 1), ('dphi', 1)],
        G_ft=prop['G_ft'], resp_idx=ri, phys_idx=pi, vtypes=vt, stypes=st,
        use_cache=False, verbose=False)
    bubbles = [td for td in ub.get(1, []) if _diagram_is_bubble(td)]
    assert len(bubbles) == 2

    q0, l0 = sp.Symbol('q0'), sp.Symbol('l0')
    lap = (('Lap',),)
    Fs = {sp.expand(bubble_loop_form_factor(td, lap)) for td in bubbles}
    assert Fs == {sp.expand(q0 ** 2 * (q0 - l0) ** 2), sp.expand(q0 ** 4)}
    # plain (no derivative) → form factor 1 on every bubble
    assert all(bubble_loop_form_factor(td, ()) == 1 for td in bubbles)


def test_diagram_classification_bubble_vs_tadpole():
    """``_diagram_is_bubble`` (q·ℓ cross-term) + ``_prefactor_is_live`` (φ*²
    dead at φ*=0) correctly separate the φ̃φ² LIVE bubbles from the φ²-tadpole,
    and mark a φ⁴ theory's φ*²-bubbles DEAD at φ*=0 (→ the tadpole path)."""
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _live_bubbles,
    )
    from engine.diagrams.type_assignment import build_field_index_map
    # reaction-diffusion: 2 LIVE bubbles (16/8 T²g²) + 1 tadpole
    model = _load('reaction_diffusion_quadratic_1d')
    ft = FieldTheory(model, taylor_order=3); ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    _, pi = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = _legs_to_phys_idx(_EXT, pi)
    ell1 = build_pipeline_records(ft, model, prop, ext, max_ell=1)[1]
    nps = {'mu': 1.0, 'D': 1.0, 'g': 0.3, 'T': 1.0, 'phistar1': 0.0}
    assert len(_live_bubbles(ell1, nps)) == 2
    # Allen-Cahn φ⁴ at φ*=0: its φ*²-bubbles are DEAD → 0 live bubbles
    modela = _load('allen_cahn_1d_subcritical_infinite')
    fta = FieldTheory(modela, taylor_order=4); fta.expand()
    propa = build_propagator(fta, modela, use_cache=False, verbose=False)
    _, pia = build_field_index_map(list(fta._ns._ring_var_names), fta._n_tilde)
    exta = _legs_to_phys_idx(_EXT, pia)
    ell1a = build_pipeline_records(fta, modela, propa, exta, max_ell=1)[1]
    npsa = {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0, 'phistar1': 0.0}
    assert len(_live_bubbles(ell1a, npsa)) == 0

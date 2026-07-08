"""
tests/test_spatial_operator_ir.py
=================================
The spatial operator IR (``pipeline.spatial_operator_ir``) — the momentum-space
foundation for spatial field theories.

Pins:
  * the operators are LINEAR (distribute over sums, pull out field/coord-free
    constants), and that linearity is the intrinsic algebra;
  * saddle expansion ``Lap(φ̄+δφ) → Lap(φ̄)+Lap(δφ)`` RETAINS the mean, and the
    homogeneous-mean annihilation is a SEPARATE, contingent pass — an
    inhomogeneous saddle keeps ``Lap(φ̄)``;
  * physics vertices come out right: Cahn–Hilliard ``∇²φ³`` and KPZ ``(∂ₓφ)²``;
  * ``∇⁴`` is a single derived generator; form factors ``Lap→−k²`` etc.

Run:  sage -python -m pytest tests/test_spatial_operator_ir.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, I, var, function

from api.spatial_operator_ir import (
    Lap, Dt, Dx, apply_linearity, expand_about_saddle, kill_means,
    to_derived_generators, form_factor, prepare_action, fourier_lower,
    classify_generators,
)

phi, psi, phibar, dphi, dpsi, mu, D, lam, k0, k1, om, x = var(
    'phi psi phibar dphi dpsi mu D lam k0 k1 omega x')


def _zero(e):
    return bool(SR(e).expand().is_trivial_zero())


# ── linearity (the operator algebra) ──────────────────────────────
def test_linearity_distributes_and_pulls_constants():
    out = apply_linearity(Lap(mu * phi + D * psi), [phi, psi])
    assert _zero(out - (mu * Lap(phi) + D * Lap(psi)))


def test_linearity_keeps_derivative_of_a_product_atomic():
    # ∇²(φψ) is a genuine vertex, NOT φ·∇²ψ + … — left atomic.
    out = apply_linearity(Lap(phi * psi), [phi, psi])
    assert str(out) == 'Lap(phi*psi)'


def test_position_dependent_coefficient_stays_atomic():
    # x is a coordinate → Lap(x·δφ) must NOT pull x out (Leibniz deferred).
    out = apply_linearity(Lap(x * dphi), [dphi])
    assert str(out) == 'Lap(dphi*x)' or str(out) == 'Lap(x*dphi)'


# ── saddle expansion: linearity applied, mean RETAINED ────────────
def test_saddle_expand_retains_mean():
    se = expand_about_saddle(Lap(phi), {phi: (phibar, dphi)})
    assert _zero(se - (Lap(phibar) + Lap(dphi)))


def test_kill_means_is_a_separate_contingent_pass():
    se = expand_about_saddle(Lap(phi), {phi: (phibar, dphi)})
    # homogeneous saddle → drop Lap(φ̄)
    assert _zero(kill_means(se, [phibar]) - Lap(dphi))
    # INHOMOGENEOUS saddle → Lap(φ̄) is retained (cancels the rest of the MF PDE)
    assert 'Lap(phibar)' in str(se)
    assert 'Lap(phibar)' in str(kill_means(se, [phibar], ops=('Dt',)))


def test_dt_gets_the_same_treatment():
    se = expand_about_saddle(Dt(phi), {phi: (phibar, dphi)})
    assert _zero(se - (Dt(phibar) + Dt(dphi)))
    assert _zero(kill_means(se, [phibar]) - Dt(dphi))      # stationary mean


# ── derived generators (the u=δφ, v=∇²δφ trick) ───────────────────
def test_phi_times_lap_phi_becomes_a_two_leg_vertex():
    t = kill_means(expand_about_saddle(phi * Lap(phi),
                                       {phi: (phibar, dphi)}), [phibar])
    g, gmap = to_derived_generators(t, [dphi])
    # δφ·v + φ̄·v  with v = ∇²δφ
    (gen, (base, chain)), = gmap.items()
    assert str(base) == 'dphi' and chain == (('Lap',),)
    assert _zero(g - gen * (dphi + phibar))


def test_nabla4_is_a_single_generator():
    t = kill_means(expand_about_saddle(Lap(Lap(phi)),
                                       {phi: (phibar, dphi)}), [phibar])
    g, gmap = to_derived_generators(t, [dphi])
    base, chain = gmap[g]                       # g reduced to the ∇⁴ generator
    assert str(base) == 'dphi' and chain == (('Lap',), ('Lap',))
    assert _zero(form_factor(chain, [k0]) - k0 ** 4)


# ── physics vertices ──────────────────────────────────────────────
def test_cahn_hilliard_conserved_nonlinearity():
    t = kill_means(expand_about_saddle(lam * Lap(phi ** 3),
                                       {phi: (phibar, dphi)}), [phibar])
    # 3λφ̄²·∇²δφ (bilinear) + 3λφ̄·∇²(δφ²) + λ·∇²(δφ³)
    assert _zero(t - lam * (3 * phibar ** 2 * Lap(dphi)
                            + 3 * phibar * Lap(dphi ** 2) + Lap(dphi ** 3)))
    _, gmap = to_derived_generators(t, [dphi])
    bases = sorted(str(b) for b, _ in gmap.values())
    assert bases == ['dphi', 'dphi^2', 'dphi^3']


def test_kpz_gradient_nonlinearity():
    t = kill_means(expand_about_saddle(Dx(phi, 0) ** 2,
                                       {phi: (phibar, dphi)}), [phibar])
    g, gmap = to_derived_generators(t, [dphi])
    (gen, (base, chain)), = gmap.items()
    assert str(base) == 'dphi' and chain == (('Dx', 0),)
    assert _zero(g - gen ** 2)


# ── Fourier form factors ──────────────────────────────────────────
def test_form_factors():
    assert _zero(form_factor((('Lap',),), [k0]) + k0 ** 2)               # −k²
    assert _zero(form_factor((('Lap',), ('Lap',)), [k0]) - k0 ** 4)      # k⁴
    assert _zero(form_factor((('Dx', 0),), [k0, k1]) - I * k0)           # i k_0
    assert _zero(form_factor((('Dx', 1),), [k0, k1]) - I * k1)
    assert _zero(form_factor((('Dt',),), [k0], omega=om) + I * om)       # −iω


def test_form_factor_multi_d_laplacian():
    # ∇² in 2-D → −(k0²+k1²)
    assert _zero(form_factor((('Lap',),), [k0, k1]) + (k0 ** 2 + k1 ** 2))


# ── generator classification (bilinear → kernel vs vertex → form factor) ──
def test_classify_reaction_diffusion_all_bilinear():
    phit, g, T = var('phit g T')
    S = phit * (Dt(phi) + mu * phi - D * Lap(phi) + g * phi ** 2) - T * phit ** 2
    Sg, gm = prepare_action(S, fields=[phi, phit])
    bil, vtx = classify_generators(Sg, gm, [phi, phit])
    assert len(vtx) == 0 and len(bil) == 2     # Dt(phi), Lap(phi): both bilinear


def test_classify_cahn_hilliard_conserved_vertex():
    phit, T, lam = var('phit T lam')
    # φ̃(Dtφ − D∇²φ + λ∇²φ³) − Tφ̃²  — ∇²φ³ is a degree-≥3 derivative vertex,
    # the linear ∇²φ is bilinear.
    S = phit * (Dt(phi) - D * Lap(phi) + lam * Lap(phi ** 3)) - T * phit ** 2
    Sg, gm = prepare_action(S, fields=[phi, phit])
    bil, vtx = classify_generators(Sg, gm, [phi, phit])
    bil_bases = sorted(str(gm[g][0]) for g in bil)
    vtx_bases = sorted(str(gm[g][0]) for g in vtx)
    assert bil_bases == ['phi', 'phi']         # Dt(phi), Lap(phi)
    assert vtx_bases == ['phi^3']              # ∇²(φ³)


def test_classify_kpz_gradient_vertex():
    phit, lam, T = var('phit lam T')
    # φ̃(Dtφ − D∇²φ + λ(∂ₓφ)²) − Tφ̃²  — (∂ₓφ)² is a degree-3 vertex.
    S = phit * (Dt(phi) - D * Lap(phi) + lam * Dx(phi, 0) ** 2) - T * phit ** 2
    Sg, gm = prepare_action(S, fields=[phi, phit])
    bil, vtx = classify_generators(Sg, gm, [phi, phit])
    assert sorted(gm[g][1] for g in vtx) == [(('Dx', 0),)]   # the ∂ₓ generator
    assert len(bil) == 2                                      # Dt, Lap bilinear


def test_operator_ir_derivative_vertex_raises_clean_phase4_error():
    """A derivative-VERTEX model (Cahn-Hilliard ∇²φ³) authored with
    ``.operator_ir()`` reaches a CLEAN, precise Phase-4 NotImplementedError on
    expand (not a crash, not silent wrong numbers) — the bilinear ∇²φ lowers,
    the ∇²(δφ²)/∇²(δφ³) vertices are correctly flagged as needing the
    momentum-first form-factor integrator."""
    import pytest
    from api.model import ModelBuilder
    from engine.core.field_theory import FieldTheory

    m = (ModelBuilder('ch_v2', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('lam', default=0.1, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text(
             'phit*(Dt(phi) + mu*phi - D*Lap(phi) + lam*Lap(phi^3)) - T*phit^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(m, taylor_order=4)
    # ∇²(φ³) is a base field-degree-3 vertex (a ≥3-physical-leg / sunset
    # topology) — still correctly rejected (base-degree 1 (∂φ)² and 2 ∂(φ²)
    # bubble vertices are the validated cases; KPZ/Burgers now run e2e).
    with pytest.raises(NotImplementedError, match='base field-degree 3'):
        ft.expand()


def test_temporal_model_untouched_by_spatial_v2():
    """A spatial_dim=0 (temporal-only) model must use the well-optimized
    TEMPORAL pipeline, fully untouched by the spatial-v2 / operator-IR work.
    The code GATES the spatial short-circuit on ``model['spatial']`` (compute.py)
    and the operator-IR overrides on ``ns._operator_ir`` (model_compiler), so
    asserting those are absent/off PROVES neither path can fire — ``Dt`` stays
    the bare v1 multiplicative symbol and the action expands exactly as before."""
    from api.model import ModelBuilder
    from engine.core.field_theory import FieldTheory

    m = (ModelBuilder('ou_temporal', n_populations=0)
         .physical_field('phi')                       # NO spatial_dim → temporal
         .parameter('mu', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('phit*((Dt + mu)*phi) - T*phit^2')
         .build())
    assert not m.get('spatial')                        # no spatial block emitted
    assert not m.get('operator_ir')                    # IR off by default

    ft = FieldTheory(m, taylor_order=2)
    ns, _R, _nt = ft._build_namespace()
    assert getattr(ns, '_operator_ir', False) is False  # IR not engaged
    assert SR(ns.Dt).is_symbol()                        # Dt is the bare v1 symbol
    # the action evaluates with the bare multiplicative Dt — no Lap/Dg nodes,
    # i.e. the operator-IR binding/lowering never ran for this temporal model.
    S = SR(m['action'](ns))
    assert 'Dt' in str(S) and 'Lap(' not in str(S) and 'Dg' not in str(S)


# ── end-to-end transform on the Phase-2 target model ─────────────
def test_reaction_diffusion_action_to_kernel_and_vertex():
    """The Phase-2 target: the φ̃φ² reaction-diffusion action authored with the
    operator IR,

        S = phit·(Dt φ + μφ − D∇²φ + gφ²) − Tφ̃² ,

    runs through ``prepare_action`` (linearity → derived generators) and then
    ``fourier_lower`` reproduces EXACTLY the v1 ingredients: the bilinear kernel
    ``K(ω,k) = −iω + μ + Dk²`` (the φ̃φ propagator denominator) and the
    momentum-independent ``g`` bubble vertex, with the white-noise ``−Tφ̃²``
    untouched.  (String authoring in ModelBuilder lands with Phase 3; this is
    the semantic content.)
    """
    phit, g, T, om = var('phit g T omega')
    S = phit * (Dt(phi) + mu * phi - D * Lap(phi) + g * phi ** 2) - T * phit ** 2
    S_gen, genmap = prepare_action(S, fields=[phi, phit])

    # Dt(phi) and Lap(phi) became derived generators; g φ² did NOT.
    chains = sorted(tuple(c) for _, (_, c) in genmap.items())
    assert chains == [(('Dt',),), (('Lap',),)]

    low = fourier_lower(S_gen, genmap, [k0], omega=om).expand()
    K = low.coefficient(phit, 1).coefficient(phi, 1)        # φ̃φ bilinear
    assert _zero(K - (-I * om + mu + D * k0 ** 2))          # = K(ω,k)
    assert _zero(low.coefficient(phit, 1).coefficient(phi, 2) - g)   # bubble vertex
    assert _zero(low.coefficient(phit, 2) + T)              # −Tφ̃² noise


def test_operator_ir_authoring_through_modelbuilder():
    """End-to-end of the AUTHORING path: a model authored with
    ``.operator_ir()`` + the ``Lap(phi)``/``Dt(phi)`` string syntax builds, and
    its action lambda (which now runs the IR passes internally) yields the
    generator form whose Fourier lowering reproduces ``K(ω,k)=−iω+μ+Dk²`` and
    the ``g`` vertex.  Proves the gate threads through ModelBuilder →
    field_theory namespace → model_compiler action lambda — with the IR ops
    overriding the bare symbols ONLY in this opted-in model's action namespace.
    """
    from api.model import ModelBuilder
    from engine.core.field_theory import FieldTheory

    m = (ModelBuilder('rd_v2_operator_ir', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('g', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text(
             'phit*(Dt(phi) + mu*phi - D*Lap(phi) + g*phi^2) - T*phit^2')
         .operator_ir()
         .boundary('infinite').initial('stationary')
         .build())
    assert m['operator_ir'] is True

    ft = FieldTheory(m, taylor_order=3)
    ns, _R, _nt = ft._build_namespace()
    S = SR(m['action'](ns))                       # runs the IR passes inside
    # genmap records what the IR lowered; the returned action is the v1-form
    # (Phase 3b-i: derived generators lowered to bare Dt/Laplacian symbols).
    genmap = ns._operator_ir_genmap
    chains = sorted(tuple(c) for _, (_, c) in genmap.items())
    assert chains == [(('Dt',),), (('Lap',),)]    # Dt(phi), Lap(phi) captured
    assert 'Dg' not in str(S)                      # generators lowered away

    # Substitute the v1 operator symbols to their Fourier images to read the
    # kernel: −iω + μ + Dk² + 2gφ̄  →  K(ω,k) at the φ*=0 saddle.
    phit_v = ns._tilde_sr_vars[0]
    phi_v = ns._phys_sr_vars[0]
    k0, om = var('k0 omega')
    Sf = S.subs({ns.Dt: -I * om, ns.Laplacian: -k0 ** 2}).expand()
    at_saddle = {s: 0 for s in Sf.variables() if 'star' in str(s)}
    K = Sf.coefficient(phit_v, 1).coefficient(phi_v, 1).subs(at_saddle)
    assert _zero(K - (-I * om + mu + D * k0 ** 2))
    assert _zero(Sf.coefficient(phit_v, 1).coefficient(phi_v, 2) - SR.var('g'))


def test_operator_ir_reduces_to_v1_action_for_reaction_diffusion():
    """Phase 3b-i: for a model whose vertices carry NO derivatives, the
    operator-IR (v2) action lowers to EXACTLY the v1 bare-symbol action — so a
    ``.operator_ir()`` reaction-diffusion model flows through the entire
    validated v1 pipeline unchanged.  Compares the evaluated/processed action
    SR expression of the two authorings term-for-term."""
    from api.model import ModelBuilder
    from engine.core.field_theory import FieldTheory

    def _build(use_ir):
        tb = (ModelBuilder('rd_cmp', n_populations=0)
              .physical_field('phi', spatial_dim=1)
              .parameter('mu', default=1.0, domain='positive')
              .parameter('D', default=1.0, domain='positive')
              .parameter('g', default=0.3, domain='real')
              .parameter('T', default=1.0, domain='positive'))
        if use_ir:
            tb = tb.set_action_text(
                'phit*(Dt(phi) + mu*phi - D*Lap(phi) + g*phi^2) - T*phit^2'
            ).operator_ir()
        else:
            tb = tb.set_action_text(
                'phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
        return tb.boundary('infinite').initial('stationary').build()

    def _eval(m):
        ft = FieldTheory(m, taylor_order=3)
        ns, _R, _nt = ft._build_namespace()
        return SR(m['action'](ns)).expand()

    assert _zero(_eval(_build(False)) - _eval(_build(True)))


def test_operator_ir_end_to_end_matches_v1_through_compute_cumulants():
    """End-to-end: ``compute_cumulants`` (tree) on a ``.operator_ir()``
    reaction-diffusion model is bit-identical to the v1 bare-symbol model,
    confirming the v2 authoring flows through the whole pipeline (MF solve,
    propagator, spatial bridge) unchanged."""
    import numpy as np
    from api.compute import compute_cumulants
    from api.model import ModelBuilder

    def _build(use_ir):
        tb = (ModelBuilder('rd_e2e', n_populations=0)
              .physical_field('phi', spatial_dim=1)
              .parameter('mu', default=1.0, domain='positive')
              .parameter('D', default=1.0, domain='positive')
              .parameter('g', default=0.35, domain='real')
              .parameter('T', default=1.0, domain='positive')
              .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2'))
        if use_ir:
            tb = tb.set_action_text(
                'phit*(Dt(phi) + mu*phi - D*Lap(phi) + g*phi^2) - T*phit^2'
            ).operator_ir()
        else:
            tb = tb.set_action_text(
                'phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
        return tb.boundary('infinite').initial('stationary').build()

    kw = dict(k=2, max_ell=0, fundamental={'mu': 1.0, 'D': 1.0, 'g': 0.35,
              'T': 1.0}, external_fields=[('phi', 1), ('phi', 1)],
              spatial_grid=np.linspace(0, 6, 9), tau_max=2.0, tau_step=1.0,
              verbose=False, use_cache=False, mf_dae_n_starts=4)
    c2 = np.real(compute_cumulants(_build(True), **kw)['C_tau'])
    c1 = np.real(compute_cumulants(_build(False), **kw)['C_tau'])
    assert np.max(np.abs(c2 - c1)) <= 1e-12 * (np.max(np.abs(c1)) + 1e-30)


def test_operator_ir_derivative_vertex_one_loop():
    """A QUADRATIC **derivative** vertex — the conserved ``−g∇²(φ²)`` reaction-
    diffusion (Model-B-type) — authored with ``.operator_ir()``.  The full-diagram
    integrator now HANDLES derivative vertices: the ∇ deposits a momentum-space
    form factor ``F(ℓ,q)`` on the loop, averaged over the loop-momentum Gaussian by
    Gauss–Hermite (exact for the polynomial ``F``; validated vs the ``loop_dyson``
    oracle to ~1%).  So ``compute_cumulants(max_ell=1)`` COMPUTES a finite
    ``C(x,τ)`` and the conserved bubble shifts the variance — v1 could not do this.

    Higher-loop (``max_ell≥2``) is also generic — the form factor is extracted +
    integrated per diagram for any topology (the L=2 momentum integral is validated
    to 1e-14 in ``test_full_integrator.test_diagram_form_factor_ell2_momentum``);
    it is correct but expensive end-to-end, so it is not exercised here."""
    import numpy as np
    import pytest
    from api.compute import compute_cumulants
    from api.model import ModelBuilder

    m = (ModelBuilder('rd_deriv_e2e', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=2.0, domain='positive')
         .parameter('g', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='g*Laplacian*phi^2')
         .set_action_text(
             'phit*(Dt(phi) + mu*phi - D*Lap(phi) - g*Lap(phi^2)) - T*phit^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    kw = dict(k=2, fundamental={'mu': 1.0, 'D': 2.0, 'g': 0.3, 'T': 1.0},
              external_fields=[('phi', 1), ('phi', 1)],
              spatial_grid=np.linspace(0, 4, 5), tau_max=0.0, tau_step=1.0,
              verbose=False, use_cache=False, mf_dae_n_starts=4)
    # tree (max_ell=0) works
    out0 = compute_cumulants(m, max_ell=0, **kw)
    mid0 = out0['C_tau_x'].shape[0] // 2
    tree0 = float(np.real(out0['C_tau_x'])[mid0][0])
    assert np.isfinite(tree0)
    # 1-loop derivative vertex now COMPUTES (the new capability)
    out1 = compute_cumulants(m, max_ell=1, **kw)
    c1 = np.real(out1['C_tau_x'])
    assert np.all(np.isfinite(c1))
    mid1 = c1.shape[0] // 2
    loop0 = float(c1[mid1][0])
    assert abs(loop0 - tree0) > 1e-6        # the conserved bubble shifts ⟨φ²⟩
    assert out1['spatial_info'].get('n_live_diagrams', 0) >= 1
    # max_ell>=2 is generic too (no NotImplementedError) — validated at the
    # integrator level (test_full_integrator); not run here (expensive e2e).


def test_operator_ir_multivertex_table_and_formfactor():
    """A model mixing Model B ∇²(φ²) (composite) and KPZ (∂ₓφ)² (perleg) — both
    φ̃φ² — lowers (NO single-mode gate) to a TWO-entry per-vertex form-factor
    table with coupling weights summing to 1; diagram_form_factor sums the two
    types PER NODE, reducing to the single-mode form factor when one weight→0."""
    import sympy as _sp
    from sage.all import SR
    from api.model import ModelBuilder
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, diagram_form_factor, _legs_to_phys_idx)
    from engine.diagrams.type_assignment import build_field_index_map

    m = (ModelBuilder('mb+kpz', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('g', default=0.15, domain='real')
         .parameter('kpz', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .set_action_text('phit*(Dt(phi) + mu*phi - D*Lap(phi) - g*Lap(phi^2) '
                          '- (kpz/2)*Dx(phi,0)^2) - T*phit^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()                                  # used to raise the single-mode gate

    # (a) two distinct vertex types, both φ̃φ², weights c_t/Σc summing to 1
    tbl = ft._ns._operator_ir_vertex_terms
    assert len(tbl) == 2
    assert {t['mode'] for t in tbl} == {'composite', 'perleg'}
    assert all(t['n_phys'] == 2 for t in tbl)
    assert (SR(sum(SR(t['weight']) for t in tbl)) - 1).simplify_full().is_zero()

    # (b) per-node form factor reduces to single-mode when the other weight is 0
    prop = build_propagator(ft, m, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    be = build_pipeline_records(ft, m, prop, ext, max_ell=1, verbose=False)
    td = be[1][0][0]                             # an ell=1 diagram (bubble)
    comp = [{'weight': 1.0, 'n_phys': 2, 'chain': (('Lap',),), 'mode': 'composite'}]
    perl = [{'weight': 1.0, 'n_phys': 2, 'chain': (('Dx', 0),), 'mode': 'perleg'}]
    comp0 = [comp[0], {'weight': 0.0, 'n_phys': 2, 'chain': (('Dx', 0),),
                       'mode': 'perleg'}]
    perl0 = [{'weight': 0.0, 'n_phys': 2, 'chain': (('Lap',),),
              'mode': 'composite'}, perl[0]]
    assert _sp.expand(diagram_form_factor(td, comp0)
                      - diagram_form_factor(td, comp)) == 0
    assert _sp.expand(diagram_form_factor(td, perl0)
                      - diagram_form_factor(td, perl)) == 0

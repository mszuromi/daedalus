"""
tests/test_propagator_spatial.py
================================
Phase 2 + 3 acceptance tests for the spatial heat-kernel propagator
(docs/spatial_implementation_plan.md Phases 2-3).

Covers:
  * build_propagator on a spatial model produces the spatial block
    (G_tx callables, G_tx_sym, k_var, mass/diffusion extraction)
  * G_tx matches the closed-form heat kernel to machine precision
    (infinite domain)
  * PBC: G_tx equals the image-source sum; is periodic; → infinite
    as L grows
  * precompute() builds + caches the propagator for both spatial
    models, and G_tx callables survive a cache round-trip
  * non-regression: a time-only model's propagator has no spatial
    block

Run::

    sage -python -m pytest tests/test_propagator_spatial.py -v
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_REPO = os.path.join(os.path.dirname(__file__), '..')


def _load_model(model_file: str):
    path = os.path.join(_REPO, 'models', model_file)
    spec = importlib.util.spec_from_file_location('m', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _build_prop(model):
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    return build_propagator(ft, model, use_cache=False, verbose=False)


_NUM = {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0, 'phistar1': 0.0}


# ── Phase 2: infinite-domain heat kernel ──────────────────────────
def test_spatial_block_present():
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.model.py'))
    assert prop.get('spatial_dim') == 1
    assert prop.get('bc_mode') == 'infinite'
    assert prop.get('G_tx') is not None
    assert (0, 0) in prop['G_tx']


def test_mass_diffusion_extraction():
    from sage.all import SR
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.model.py'))
    A, B = prop['G_tx_sym'][(0, 0)]
    # B = D (diffusion).
    assert str(B) == 'D'
    # A = mu + 3*lam*phistar1^2 ; at phistar1 = 0 the mass is mu.
    phistar1 = SR.var('phistar1')
    mu = SR.var('mu')
    assert (A.subs({phistar1: 0}) - mu).is_zero()


def test_g_tx_matches_heat_kernel_infinite():
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.model.py'))
    G = prop['G_tx'][(0, 0)]
    maxrel = 0.0
    for t in [0.05, 0.5, 2.0, 5.0]:
        for x in [0.0, 1.0, 3.0]:
            g = G(t, x, **_NUM)
            analytic = (1.0 / math.sqrt(4 * math.pi * t)
                        * math.exp(-x * x / (4 * t) - t))
            maxrel = max(maxrel, abs(g - analytic) / max(abs(analytic), 1e-300))
    assert maxrel < 1e-12, f'G_tx vs heat kernel max rel {maxrel:.3e}'


def test_g_tx_causal():
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.model.py'))
    G = prop['G_tx'][(0, 0)]
    assert G(-1.0, 0.0, **_NUM) == 0j
    assert G(0.0, 0.0, **_NUM) == 0j


# ── Phase 3: periodic boundary ────────────────────────────────────
def test_pbc_propagator_is_image_sum():
    from engine.integration.spatial.heat_kernel import image_sum
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_pbc.model.py'))
    assert prop['bc_mode'] == 'periodic'
    G = prop['G_tx'][(0, 0)]
    num = dict(_NUM, L=6.0)
    maxrel = 0.0
    for t in [0.2, 1.0, 3.0]:
        for x in [0.0, 1.5, 3.0]:
            ref = image_sum(t, x, 1.0, 1.0, 6.0)
            g = G(t, x, **num)
            maxrel = max(maxrel, abs(g - ref) / max(abs(ref), 1e-300))
    assert maxrel < 1e-12


def test_pbc_periodicity():
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_pbc.model.py'))
    G = prop['G_tx'][(0, 0)]
    num = dict(_NUM, L=6.0)
    for t in [0.5, 2.0]:
        for x in [0.0, 1.0, 2.5]:
            a = G(t, x, **num)
            b = G(t, x + 6.0, **num)
            assert abs(a - b) / max(abs(a), 1e-300) < 1e-12


def test_pbc_approaches_infinite_as_L_grows():
    from engine.integration.spatial.heat_kernel import gaussian_heat_kernel
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_pbc.model.py'))
    G = prop['G_tx'][(0, 0)]
    ginf = gaussian_heat_kernel(1.0, 0.0, 1.0, 1.0)
    prev = None
    for L in [4.0, 8.0, 16.0]:
        g = G(1.0, 0.0, **dict(_NUM, L=L))
        diff = abs(g - ginf)
        if prev is not None:
            assert diff < prev  # monotone convergence
        prev = diff
    assert abs(G(1.0, 0.0, **dict(_NUM, L=16.0)) - ginf) < 1e-6


# ── precompute + cache round-trip ─────────────────────────────────
@pytest.mark.parametrize('model_file', [
    'allen_cahn_1d_subcritical_infinite.model.py',
    'allen_cahn_1d_subcritical_pbc.model.py',
])
def test_precompute_builds_and_caches_spatial(model_file):
    from api._precompute import precompute
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    model = _load_model(model_file)
    out = precompute(model, force=True, verbose=False)
    assert out['propagator_built'] is True
    assert out['mf_check'] == 'PASS'
    # Cache load rebuilds the G_tx callables.
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=True, verbose=False)
    assert prop.get('G_tx') is not None
    assert (0, 0) in prop['G_tx']


# ── Non-regression: time-only model has no spatial block ─────────
def test_time_only_propagator_has_no_spatial_block():
    # Reuse a known time-only fixture model.
    prop = _build_prop(_load_model('single_population_quad_exp_test.model.py'))
    assert prop.get('G_tx') is None
    assert prop.get('G_tx_sym') is None


# ── Drift-generalized heat kernel (Burgers/KPZ + genuine advection) ──
def test_drift_heat_kernel_vs_analytic():
    """gaussian_heat_kernel with drift V=iv equals the analytic
    advection-diffusion Green's function, and V=0 is bit-identical to
    the pure heat kernel."""
    from engine.integration.spatial.heat_kernel import gaussian_heat_kernel
    A, B, v = 0.5, 1.0, 2.0
    maxerr = 0.0
    for t in (0.3, 1.0, 2.5):
        for x in (-1.0, 0.0, 2.0, 4.0):
            got = gaussian_heat_kernel(t, x, A, B, V=1j * v)          # V = i v
            ana = (math.exp(-A * t) * (4 * math.pi * B * t) ** (-0.5)
                   * math.exp(-(x - v * t) ** 2 / (4 * B * t)))
            maxerr = max(maxerr, abs(got - ana))
    assert maxerr < 1e-12, f'drift kernel vs analytic max err {maxerr:.2e}'
    # V=0 must reproduce the plain heat kernel exactly.
    assert abs(gaussian_heat_kernel(1.3, 0.7, A, B)
               - gaussian_heat_kernel(1.3, 0.7, A, B, V=0.0)) == 0.0


def test_extract_mass_diffusion_drift():
    """extract_mass_diffusion reads the linear-in-k DRIFT V = i·v from a
    genuine-advection inverse propagator, and V=0 for a Laplacian-only
    (even) kernel."""
    from sage.all import SR, var, I
    from engine.integration.spatial.heat_kernel import extract_mass_diffusion
    om = var('omega'); k = var('k'); lap = var('Laplacian'); gx = SR.var('GradX')
    A, B, v = 0.5, 1.0, 2.0
    Ae, Be, Ve = extract_mass_diffusion(I * om + A - B * lap + v * gx,
                                        om, k, lap, gx)
    assert (SR(Ae) - A).is_zero() and (SR(Be) - B).is_zero()
    assert (SR(Ve) - I * v).is_zero()
    # Laplacian-only (even) kernel ⇒ V = 0 (backward compatible).
    _, _, V0 = extract_mass_diffusion(I * om + A - B * lap, om, k, lap, gx)
    assert SR(V0).is_zero()


# ── Burgers / KPZ gradient-vertex compilation (operator IR v2) ──────
def test_burgers_compiles_with_saddle_drift():
    """Burgers ∂_x(φ²) lowers: the bilinear-Dx cross-term becomes a
    propagator DRIFT V ∝ φ* (→0 at the saddle), the propagator is the
    pure heat kernel (mass μ, diffusion D), and the vertex is composite."""
    from sage.all import SR
    prop = _build_prop(_load_model('burgers_1d.model.py'))
    A, B = prop['G_tx_sym'][(0, 0)]
    assert str(B) == 'D'
    assert (SR(A) - SR.var('mu')).is_zero()        # mass = mu
    # Drift is ∝ phistar1 (vanishes at the homogeneous saddle φ*=0).
    V = prop['ac_drift'][0]
    assert not SR(V).is_zero()
    assert SR(V).subs({SR.var('phistar1'): 0}).is_zero()


def test_burgers_kpz_vertex_modes():
    """The operator-IR lowering stashes the per-vertex form-factor mode:
    Burgers ∂_x(φ²) → 'composite', KPZ (∂_x h)² → 'perleg'."""
    from engine.core.field_theory import FieldTheory
    from api._propagator import build_propagator
    for model_file, want_mode in (('burgers_1d.model.py', 'composite'),
                                    ('kpz_1d.model.py', 'perleg')):
        model = _load_model(model_file)
        ft = FieldTheory(model, taylor_order=4)
        ft.expand()
        build_propagator(ft, model, use_cache=False, verbose=False)
        assert getattr(ft._ns, '_operator_ir_vertex_mode', None) == want_mode


def test_kpz_has_no_propagator_drift():
    """KPZ (∂_x h)² has NO bilinear Dx (∂_x of the homogeneous mean is 0),
    so the propagator drift is identically zero — pure heat kernel."""
    from sage.all import SR
    prop = _build_prop(_load_model('kpz_1d.model.py'))
    A, B = prop['G_tx_sym'][(0, 0)]
    assert str(B) == 'D' and (SR(A) - SR.var('mu')).is_zero()
    assert SR(prop['ac_drift'][0]).is_zero()

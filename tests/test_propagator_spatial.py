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
    theories, and G_tx callables survive a cache round-trip
  * non-regression: a time-only theory's propagator has no spatial
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


def _load_model(theory_file: str):
    path = os.path.join(_REPO, 'theories', theory_file)
    spec = importlib.util.spec_from_file_location('m', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _build_prop(model):
    from msrjd.core.field_theory import FieldTheory
    from pipeline._propagator import build_propagator
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    return build_propagator(ft, model, use_cache=False, verbose=False)


_NUM = {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0, 'phistar1': 0.0}


# ── Phase 2: infinite-domain heat kernel ──────────────────────────
def test_spatial_block_present():
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.theory.py'))
    assert prop.get('spatial_dim') == 1
    assert prop.get('bc_mode') == 'infinite'
    assert prop.get('G_tx') is not None
    assert (0, 0) in prop['G_tx']


def test_mass_diffusion_extraction():
    from sage.all import SR
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.theory.py'))
    A, B = prop['G_tx_sym'][(0, 0)]
    # B = D (diffusion).
    assert str(B) == 'D'
    # A = mu + 3*lam*phistar1^2 ; at phistar1 = 0 the mass is mu.
    phistar1 = SR.var('phistar1')
    mu = SR.var('mu')
    assert (A.subs({phistar1: 0}) - mu).is_zero()


def test_g_tx_matches_heat_kernel_infinite():
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.theory.py'))
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
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_infinite.theory.py'))
    G = prop['G_tx'][(0, 0)]
    assert G(-1.0, 0.0, **_NUM) == 0j
    assert G(0.0, 0.0, **_NUM) == 0j


# ── Phase 3: periodic boundary ────────────────────────────────────
def test_pbc_propagator_is_image_sum():
    from msrjd.integration.spatial.heat_kernel import image_sum
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_pbc.theory.py'))
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
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_pbc.theory.py'))
    G = prop['G_tx'][(0, 0)]
    num = dict(_NUM, L=6.0)
    for t in [0.5, 2.0]:
        for x in [0.0, 1.0, 2.5]:
            a = G(t, x, **num)
            b = G(t, x + 6.0, **num)
            assert abs(a - b) / max(abs(a), 1e-300) < 1e-12


def test_pbc_approaches_infinite_as_L_grows():
    from msrjd.integration.spatial.heat_kernel import gaussian_heat_kernel
    prop = _build_prop(_load_model('allen_cahn_1d_subcritical_pbc.theory.py'))
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
@pytest.mark.parametrize('theory_file', [
    'allen_cahn_1d_subcritical_infinite.theory.py',
    'allen_cahn_1d_subcritical_pbc.theory.py',
])
def test_precompute_builds_and_caches_spatial(theory_file):
    from pipeline._precompute import precompute
    from msrjd.core.field_theory import FieldTheory
    from pipeline._propagator import build_propagator
    model = _load_model(theory_file)
    out = precompute(model, force=True, verbose=False)
    assert out['propagator_built'] is True
    assert out['mf_check'] == 'PASS'
    # Cache load rebuilds the G_tx callables.
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=True, verbose=False)
    assert prop.get('G_tx') is not None
    assert (0, 0) in prop['G_tx']


# ── Non-regression: time-only theory has no spatial block ─────────
def test_time_only_propagator_has_no_spatial_block():
    # Reuse a known time-only fixture theory.
    prop = _build_prop(_load_model('single_population_quad_exp_test.theory.py'))
    assert prop.get('G_tx') is None
    assert prop.get('G_tx_sym') is None

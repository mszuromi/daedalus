"""
Unit tests for ``api/_expand_cache.py``.

Coverage:
  * Save → load round-trip at the same taylor_order produces identical
    ``_by_tp`` bigrades.
  * Save at high order → load at lower order downgrade-filters
    correctly (keeps only entries with total degree ``<= target``).
  * Save with one structure → load attempt after structural change
    rejects the stale cache (returns False) without corrupting ``ft``.
  * ``find_best_cached_order`` picks the smallest cached order
    ``>= target``.
  * Compute-time speedup: cold ``compute_cumulants`` vs warm rerun on
    the same model.

Run with:
    sage -python -m pytest tests/test_expand_cache.py -v
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest


# A model file we can hit cheaply (~9 s cold, ~0.2 s warm at order 4).
MODEL_FILENAME = 'single_population_linear_delta_spikes_test.model.py'


@pytest.fixture
def model_and_fund():
    """Load the test model + its standard fundamental.

    Returns (model_dict, fundamental_dict).
    """
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, '..', 'models', MODEL_FILENAME)
    spec = importlib.util.spec_from_file_location('test_th', model_path)
    th = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(th)
    model = th.build()
    fundamental = {
        'Em':  [0.8, 0.78],
        'tau': [10, 9],
        'w':   [[0.0, 0.25], [0.2, 0.0]],
    }
    return model, fundamental


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Redirect the cache root to a temporary directory so tests don't
    collide with the user's real cache.

    Yields the cache root path.
    """
    monkeypatch.chdir(tmp_path)
    yield str(tmp_path)


def _expand_fresh(model, taylor_order: int):
    """Build + expand a fresh ``FieldTheory`` at ``taylor_order``."""
    from engine.core.field_theory import FieldTheory
    ft = FieldTheory(model, taylor_order=taylor_order)
    ft.expand()
    return ft


# ──────────────────────────────────────────────────────────────────────


def test_round_trip_same_order(model_and_fund, isolated_cache):
    """Save and reload at the same order — ``_by_tp`` must match
    bigrade-by-bigrade."""
    from engine.core.field_theory import FieldTheory
    from api import _expand_cache as ec

    model, _ = model_and_fund
    ft1 = _expand_fresh(model, taylor_order=4)

    ec.save_expand(model, ft1, cache_dir_root=isolated_cache)

    ft2 = FieldTheory(model, taylor_order=4)
    ec.prepare_for_load(ft2)
    ok = ec.load_expand(model, ft2, target_order=4, cached_order=4,
                        cache_dir_root=isolated_cache)
    assert ok

    assert set(ft1._by_tp.keys()) == set(ft2._by_tp.keys())
    for key, poly1 in ft1._by_tp.items():
        poly2 = ft2._by_tp[key]
        assert poly1 == poly2, f'bigrade {key} differs after round-trip'

    assert ft2.sanity_check() is True


def test_downgrade_filter(model_and_fund, isolated_cache):
    """Save at order 4, load with target_order 2 — bigrades with
    total degree > 2 must be dropped."""
    from engine.core.field_theory import FieldTheory
    from api import _expand_cache as ec

    model, _ = model_and_fund
    ft1 = _expand_fresh(model, taylor_order=4)
    ec.save_expand(model, ft1, cache_dir_root=isolated_cache)

    ft2 = FieldTheory(model, taylor_order=2)
    ec.prepare_for_load(ft2)
    ok = ec.load_expand(model, ft2, target_order=2, cached_order=4,
                        cache_dir_root=isolated_cache)
    assert ok

    # Every retained bigrade must have a+b <= 2.
    for (a, b) in ft2._by_tp:
        assert a + b <= 2, f'bigrade ({a}, {b}) violates target_order=2'

    # Sanity check still passes — MF sectors are pre-zeroed and
    # survive the filter.
    assert ft2.sanity_check() is True


def test_find_best_cached_order_picks_smallest_above_target(
        model_and_fund, isolated_cache):
    """When multiple cached orders satisfy ``>= target``, pick the
    smallest one (least filter work)."""
    from engine.core.field_theory import FieldTheory
    from api import _expand_cache as ec

    model, _ = model_and_fund

    # Cache orders 2, 4, 6.
    for o in (2, 4, 6):
        ft = _expand_fresh(model, taylor_order=o)
        ec.save_expand(model, ft, cache_dir_root=isolated_cache)

    cached = ec.list_cached_orders(model, cache_dir_root=isolated_cache)
    assert cached == [2, 4, 6]

    # Target=2 → smallest cached >= 2 is 2.
    assert ec.find_best_cached_order(model, 2, isolated_cache) == 2
    # Target=3 → smallest >= 3 is 4.
    assert ec.find_best_cached_order(model, 3, isolated_cache) == 4
    # Target=5 → smallest >= 5 is 6.
    assert ec.find_best_cached_order(model, 5, isolated_cache) == 6
    # Target=7 → no cached order >= 7.
    assert ec.find_best_cached_order(model, 7, isolated_cache) is None


def test_compute_cumulants_warm_cache_speedup(
        model_and_fund, isolated_cache):
    """Calling ``compute_cumulants`` twice with the same arguments —
    the second call hits the cache and runs much faster (>= 10x),
    and produces identical numerical output."""
    import time
    import numpy as np
    from api import compute_cumulants

    model, fundamental = model_and_fund

    t0 = time.perf_counter()
    th1 = compute_cumulants(
        model=model, k=2, max_ell=0,
        fundamental=fundamental,
        external_fields=[('n', 1), ('n', 2)],
        tau_max=5.0, tau_step=1.0,
        parallel=False, verbose=False,
    )
    t_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    th2 = compute_cumulants(
        model=model, k=2, max_ell=0,
        fundamental=fundamental,
        external_fields=[('n', 1), ('n', 2)],
        tau_max=5.0, tau_step=1.0,
        parallel=False, verbose=False,
    )
    t_warm = time.perf_counter() - t0

    # Outputs must match exactly.
    assert np.allclose(th1['C_tau'], th2['C_tau']), \
        'C_tau differs between cold and warm runs'

    # Warm run must be substantially faster than cold.  Conservative
    # threshold (10x) — typical speedup on this small model is ~40x.
    assert t_warm < t_cold / 10, \
        (f'expected warm run < cold / 10; got cold={t_cold:.2f}s, '
         f'warm={t_warm:.2f}s')


def test_precompute_populates_cache(model_and_fund, isolated_cache):
    """``precompute(model)`` writes expand_taylor2.sobj + propagator.sobj
    and reports PASS for a healthy model."""
    from api import precompute
    from api._expand_cache import cache_dir

    model, _ = model_and_fund
    result = precompute(model, verbose=False)

    assert result['mf_check'] == 'PASS'
    assert result['sanity_ok'] is True
    assert result['taylor_order'] == 2
    assert result['propagator_built'] is True

    cd = cache_dir(model)
    files = set(os.listdir(cd))
    assert 'expand_taylor2.sobj' in files
    assert 'propagator.sobj' in files


def test_default_taylor_order_matches_diagrammatic_minimum(
        model_and_fund, isolated_cache):
    """When ``taylor_order=None``, ``compute_cumulants`` must auto-pick
    ``max(k + 2*max_ell, 2)`` — the mathematical minimum, not the
    historical 4-floor that forced over-expansion for ``k=2, max_ell=0``.

    Verifies via the cache: after precompute() writes expand_taylor2.sobj,
    a default-taylor compute_cumulants(k=2, max_ell=0) call must HIT
    that cache (and not produce expand_taylor4.sobj as a side effect).
    """
    from api import compute_cumulants, precompute
    from api._expand_cache import cache_dir

    model, fundamental = model_and_fund

    # 1) Precompute populates expand_taylor2.sobj.
    precompute(model, verbose=False)
    cd = cache_dir(model)
    before = set(os.listdir(cd))
    assert 'expand_taylor2.sobj' in before
    assert 'expand_taylor4.sobj' not in before

    # 2) Default-taylor compute_cumulants must hit the order-2 cache,
    # NOT trigger an order-4 expand.
    compute_cumulants(
        model=model, k=2, max_ell=0,           # no taylor_order kwarg
        fundamental=fundamental,
        external_fields=[('n', 1), ('n', 2)],
        tau_max=5.0, tau_step=1.0,
        parallel=False, verbose=False,
    )

    after = set(os.listdir(cd))
    new_files = after - before
    # No new expand_taylor*.sobj should have appeared — the default
    # picked order 2, which was already cached.
    new_expand_files = {f for f in new_files
                        if f.startswith('expand_taylor')}
    assert new_expand_files == set(), (
        f'default-taylor compute_cumulants produced new expand caches: '
        f'{sorted(new_expand_files)} — the auto-picker floor is too high')


# ══════════════════════════════════════════════════════════════════════
# Operator-IR form-factor signature (spatial derivative-vertex models)
# ──────────────────────────────────────────────────────────────────────
# The expand-cache slug is only ``model['name']`` + taylor order, which
# does NOT capture the per-vertex form-factor / mode table.  That table
# (``ns._operator_ir_vertex_terms``) is an *action-eval side effect*, not
# part of ``_by_tp``, so a cache load that skips ``expand()`` used to drop
# it silently — a Model-B⊕KPZ 1-loop value then loaded as the bare φ̃φ²
# number (vertex_mode None instead of 'composite+perleg').  These tests
# pin the two halves of the fix: (a) the table is rebuilt on load, and
# (b) a stale pre-signature bundle is rejected.

# The combined Allen-Cahn ⊕ Model B ⊕ KPZ model: φ³ plain vertex + Model
# B ∇²(φ²) composite vertex + KPZ (∂ₓφ)² per-leg vertex, all on one φ̃φ²
# node — the canonical mixed composite+perleg form-factor table.
COMBINED_MODEL_FILENAME = 'combined_allencahn_modelb_kpz_1d.model.py'


@pytest.fixture
def combined_model():
    """Load the combined operator-IR model model (absolute path, so it
    survives the ``isolated_cache`` chdir)."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, '..', 'models', COMBINED_MODEL_FILENAME)
    spec = importlib.util.spec_from_file_location('combined_th', model_path)
    th = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(th)
    return th.build()


# Standard parameter point + a small-but-representative external slice.
_COMBINED_FUND = {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'g': 0.15, 'kpz': 0.3,
                  'T': 1.0}


def _combined_kwargs():
    import numpy as np
    return dict(
        k=2, max_ell=1,
        fundamental=_COMBINED_FUND,
        external_fields=[('dphi', 1), ('dphi', 1)],
        spatial_grid=np.linspace(0.0, 6.0, 9),   # coarse → fast; bug is grid-free
        tau_max=0.0, tau_step=1.0,
        verbose=False,
    )


def test_operator_ir_table_rebuilt_on_prepare_for_load(combined_model):
    """``prepare_for_load`` (the cache-load namespace setup) must rebuild
    ``ns._operator_ir_vertex_terms`` identically to a fresh ``expand()``.

    A bare ``_build_namespace`` sets ``_operator_ir`` but NOT the table —
    that is what the cache-load path used to inherit.
    """
    from engine.core.field_theory import FieldTheory
    from api import _expand_cache as ec

    # Reference: a fresh expand populates the table.
    ft_fresh = FieldTheory(combined_model, taylor_order=4)
    ft_fresh.expand()
    fresh = getattr(ft_fresh._ns, '_operator_ir_vertex_terms', None)
    assert fresh, 'fresh expand produced no form-factor table'
    assert {t['mode'] for t in fresh} == {'composite', 'perleg'}

    # Cache-load setup: prepare_for_load alone must reproduce it.
    ft_load = FieldTheory(combined_model, taylor_order=4)
    ec.prepare_for_load(ft_load)            # NO expand()
    rebuilt = getattr(ft_load._ns, '_operator_ir_vertex_terms', None)
    assert rebuilt, 'prepare_for_load left the form-factor table empty'

    def _key(t):
        return (t['mode'], t['n_phys'], t['chain'], str(t['weight']))
    assert sorted(map(_key, rebuilt)) == sorted(map(_key, fresh))

    # And the signatures of the two namespaces agree.
    assert (ec.vertex_form_factor_signature(ft_load._ns)
            == ec.vertex_form_factor_signature(ft_fresh._ns)
            is not None)


def test_operator_ir_stale_unsigned_cache_rejected(combined_model,
                                                   isolated_cache):
    """A pre-signature bundle (``cache_version`` 1, no ``vertex_signature``)
    for a derivative-vertex model is the exact shape of the on-disk stale
    file that caused the bug.  ``load_expand`` must reject it (return
    ``False``) rather than load it with the form factors dropped."""
    from engine.core.field_theory import FieldTheory
    from api import _expand_cache as ec
    from sage.all import save as sage_save, load as sage_load

    # 1) Save a healthy (signed) bundle.
    ft = _expand_fresh(combined_model, taylor_order=4)
    path = ec.save_expand(combined_model, ft, cache_dir_root=isolated_cache)

    bundle = sage_load(path)
    assert bundle.get('vertex_signature') is not None  # signed by the fix

    # 2) Downgrade it to the old, unsigned format in place.
    bundle.pop('vertex_signature', None)
    bundle['cache_version'] = 1
    sage_save(bundle, path.removesuffix('.sobj'))

    # 3) load_expand must now reject the stale bundle.
    ft2 = FieldTheory(combined_model, taylor_order=4)
    ec.prepare_for_load(ft2)
    ok = ec.load_expand(combined_model, ft2, target_order=4, cached_order=4,
                        cache_dir_root=isolated_cache, verbose=False)
    assert ok is False, 'stale unsigned operator-IR cache was NOT rejected'


def test_operator_ir_signed_cache_round_trips(combined_model, isolated_cache):
    """A bundle saved by the fix carries a matching signature, so a
    subsequent ``load_expand`` accepts it (the cache still works — the
    signature only rejects *mismatches*)."""
    from engine.core.field_theory import FieldTheory
    from api import _expand_cache as ec

    ft = _expand_fresh(combined_model, taylor_order=4)
    ec.save_expand(combined_model, ft, cache_dir_root=isolated_cache)

    ft2 = FieldTheory(combined_model, taylor_order=4)
    ec.prepare_for_load(ft2)
    ok = ec.load_expand(combined_model, ft2, target_order=4, cached_order=4,
                        cache_dir_root=isolated_cache, verbose=False)
    assert ok is True, 'signed cache from the same model should load'
    # by_tp round-trips bigrade-for-bigrade.
    assert set(ft._by_tp.keys()) == set(ft2._by_tp.keys())
    for key, poly in ft._by_tp.items():
        assert poly == ft2._by_tp[key], f'bigrade {key} differs'


def test_combined_operator_ir_cached_matches_uncached(combined_model,
                                                      isolated_cache):
    """Headline regression: ``compute_cumulants`` for the combined
    Model-B⊕KPZ operator-IR model at ``max_ell=1`` agrees whether the
    expand cache is used or bypassed — and the cached run keeps the
    'composite+perleg' form factors (vertex_mode) rather than collapsing
    to the bare φ̃φ² value."""
    import numpy as np
    from api import compute_cumulants

    def _c00_mode(res):
        si = res.get('spatial_info') or {}
        return np.asarray(res['C_tau']), si.get('vertex_mode')

    kw = _combined_kwargs()

    # Reference: cache fully bypassed.
    C_nc, m_nc = _c00_mode(
        compute_cumulants(model=combined_model, use_cache=False, **kw))

    # Cold cache: isolated tmp is empty → fresh expand, then save (signed).
    C_cold, m_cold = _c00_mode(
        compute_cumulants(model=combined_model, use_cache=True, **kw))

    # Warm cache: the signed bundle must be HIT and its form-factor table
    # rebuilt on load.
    C_warm, m_warm = _c00_mode(
        compute_cumulants(model=combined_model, use_cache=True, **kw))

    assert m_nc == m_cold == m_warm == 'composite+perleg', (
        f'vertex_mode dropped: no-cache={m_nc!r}, cold={m_cold!r}, '
        f'warm={m_warm!r} — form-factor table was lost on cache load')
    assert np.allclose(C_cold, C_nc), 'cold-cache C(x) != uncached'
    assert np.allclose(C_warm, C_nc), 'warm-cache C(x) != uncached'
    # The 1-loop value must be the corrected one, not the plain +0.495.
    assert C_nc.flat[0] < 0.0, (
        f'C(0,0)={C_nc.flat[0]} — expected the composite+perleg 1-loop '
        f'value (~-0.888), not the dropped-form-factor +0.495')

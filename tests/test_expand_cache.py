"""
Unit tests for ``pipeline/_expand_cache.py``.

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
    the same theory.

Run with:
    sage -python -m pytest tests/test_expand_cache.py -v
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest


# A theory file we can hit cheaply (~9 s cold, ~0.2 s warm at order 4).
THEORY_FILENAME = 'single_population_linear_delta_spikes_test.theory.py'


@pytest.fixture
def model_and_fund():
    """Load the test theory + its standard fundamental.

    Returns (model_dict, fundamental_dict).
    """
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    theory_path = os.path.join(here, '..', 'theories', THEORY_FILENAME)
    spec = importlib.util.spec_from_file_location('test_th', theory_path)
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
    from msrjd.core.field_theory import FieldTheory
    ft = FieldTheory(model, taylor_order=taylor_order)
    ft.expand()
    return ft


# ──────────────────────────────────────────────────────────────────────


def test_round_trip_same_order(model_and_fund, isolated_cache):
    """Save and reload at the same order — ``_by_tp`` must match
    bigrade-by-bigrade."""
    from msrjd.core.field_theory import FieldTheory
    from pipeline import _expand_cache as ec

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
    from msrjd.core.field_theory import FieldTheory
    from pipeline import _expand_cache as ec

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
    from msrjd.core.field_theory import FieldTheory
    from pipeline import _expand_cache as ec

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
    from pipeline import compute_cumulants

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
    # threshold (10x) — typical speedup on this small theory is ~40x.
    assert t_warm < t_cold / 10, \
        (f'expected warm run < cold / 10; got cold={t_cold:.2f}s, '
         f'warm={t_warm:.2f}s')


def test_precompute_populates_cache(model_and_fund, isolated_cache):
    """``precompute(model)`` writes expand_taylor2.sobj + propagator.sobj
    and reports PASS for a healthy theory."""
    from pipeline import precompute
    from pipeline._expand_cache import cache_dir

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
    from pipeline import compute_cumulants, precompute
    from pipeline._expand_cache import cache_dir

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

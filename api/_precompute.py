"""
``precompute(model)`` — one-time structural pass for a theory.

What it does
------------
For a given ``model`` (the dict returned by ``TheoryBuilder.build()``)
this function:

  1. Expands the action at ``taylor_order=2``.  This is the smallest
     order that captures everything the structural validation +
     propagator construction need: the (0,0)/(1,0)/(0,1) MF sectors
     (verified to vanish at the saddle) plus the (1,1) bilinear
     propagator kernel.  Higher orders only bring in interaction
     vertices that diagrammatic computations consume downstream.
  2. Calls ``sanity_check()`` — confirms the action's MF sector
     vanishes at the saddle.
  3. Solves the mean-field equations (the same DAE / iteration
     solver used by ``compute_cumulants``) and reports the saddle.
  4. Builds the symbolic propagator and caches it under
     ``saved_theories/<theory>/propagator.sobj``.
  5. Caches the expand result under
     ``saved_theories/<theory>/expand_taylor2.sobj``.

The point: cheap (~seconds for any theory) and produces durable
artefacts that subsequent ``compute_cumulants`` calls hit
immediately — no matter what taylor_order the caller requests, the
propagator + MF check are free.  For higher-order interaction
vertices the user pays the taylor cost ONCE per order; subsequent
runs at that order (or any lower order via downgrade-filter) load
from cache.

Returns
-------
A status dict with keys:

  ``'mf_check'``      — ``'PASS'`` or a string describing the failure.
  ``'sanity_ok'``     — bool, the FieldTheory's internal sanity check.
  ``'mf_values'``     — dict of saddle values keyed by ``<var>star``.
  ``'taylor_order'``  — int, always ``2`` (this is the pre-compute order).
  ``'cache_dir'``     — str, the per-theory cache directory.
  ``'propagator_built'`` — bool.
  ``'wall_seconds'``  — float, total wall time.
  ``'log'``           — list of strings, status messages collected
                         (e.g. cache hits/misses).
"""
from __future__ import annotations

import os
import time
import traceback

__all__ = ['precompute']


def precompute(model: dict, *, force: bool = False,
               verbose: bool = True) -> dict:
    """Run the structural pre-compute pass for a theory.

    Parameters
    ----------
    model : dict
        Theory dict from ``TheoryBuilder.build()``.
    force : bool, default False
        If True, ignore any existing expand / propagator caches and
        rebuild from scratch.
    verbose : bool, default True
        If True, print the same per-step messages ``compute_cumulants``
        emits.  The structured ``log`` list in the return dict is
        always populated regardless.
    """
    log: list[str] = []

    def _log(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg)

    out: dict = {
        'mf_check':         'NOT_RUN',
        'sanity_ok':        False,
        'mf_values':        {},
        'taylor_order':     2,
        'cache_dir':        '',
        'propagator_built': False,
        'wall_seconds':     0.0,
        'log':              log,
    }

    t0 = time.perf_counter()

    try:
        # Imports inside the function so module import is cheap.
        from engine.core.field_theory import FieldTheory
        from api._propagator import build_propagator
        from api import _expand_cache as _ec
    except ImportError as e:
        _log(f'[precompute] FATAL: cannot import api modules ({e!r})')
        out['mf_check'] = f'IMPORT_FAILED: {e}'
        out['wall_seconds'] = time.perf_counter() - t0
        return out

    out['cache_dir'] = _ec.cache_dir(model)
    _log(f'[precompute] theory={model.get("name", "<unnamed>")!r}, '
         f'cache_dir={out["cache_dir"]}')

    # ── Stage 1: expand at order 2 (or load from cache) ────────────
    target_order = 2
    ft = FieldTheory(model, taylor_order=target_order)

    cache_hit = False
    if not force:
        cached_order = _ec.find_best_cached_order(model, target_order)
        if cached_order is not None:
            _ec.prepare_for_load(ft)
            cache_hit = _ec.load_expand(
                model, ft,
                target_order=target_order,
                cached_order=cached_order,
                verbose=verbose,
            )
            if cache_hit:
                _log(f'[precompute] expand: cache hit at order={cached_order} '
                     f'(filtered to {target_order} if needed)')

    if not cache_hit:
        _log(f'[precompute] expand: running fresh at order={target_order} '
             f'(force={force})')
        try:
            ft.expand()
        except Exception as e:
            _log(f'[precompute] FATAL: ft.expand() raised:\n'
                 f'{traceback.format_exc()}')
            out['mf_check'] = f'EXPAND_RAISED: {type(e).__name__}: {e}'
            out['wall_seconds'] = time.perf_counter() - t0
            return out
        try:
            _ec.save_expand(model, ft, verbose=verbose)
        except Exception as e:
            _log(f'[precompute] save_expand failed ({e!r}); continuing.')

    # ── Stage 2: sanity check (MF cancellation) ────────────────────
    try:
        out['sanity_ok'] = bool(ft.sanity_check(verbose=verbose))
        out['mf_check'] = 'PASS' if out['sanity_ok'] else 'FAIL'
        _log(f'[precompute] sanity_check: {out["mf_check"]}')
    except Exception as e:
        _log(f'[precompute] sanity_check raised: {e!r}')
        out['mf_check'] = f'SANITY_RAISED: {type(e).__name__}: {e}'

    if not out['sanity_ok']:
        out['wall_seconds'] = time.perf_counter() - t0
        return out

    # ── Stage 3: solve MF (DAE or legacy iteration) ────────────────
    try:
        fundamental = _build_fundamental_defaults(model)
        mf = _solve_mf_at_saddle(model, fundamental, ft)
        out['mf_values'] = mf
        _log(f'[precompute] mean-field saddle:')
        for k, v in mf.items():
            _log(f'    {k!r:10} = {v}')
    except Exception as e:
        _log(f'[precompute] MF solve failed: {e!r}\n'
             f'(non-fatal: propagator can still be built)')

    # ── Stage 4: build the propagator (and cache it) ───────────────
    try:
        build_propagator(ft, model, use_cache=True, verbose=verbose,
                         force=force)
        out['propagator_built'] = True
        _log(f'[precompute] propagator cached at '
             f'{out["cache_dir"]}/propagator.sobj')
    except Exception as e:
        _log(f'[precompute] build_propagator failed: '
             f'{type(e).__name__}: {e}')
        out['propagator_built'] = False

    out['wall_seconds'] = time.perf_counter() - t0
    _log(f'[precompute] done in {out["wall_seconds"]:.2f}s')
    return out


# ── Internal helpers ──────────────────────────────────────────────────


def _build_fundamental_defaults(model: dict) -> dict:
    """Build a ``fundamental`` dict from the model's per-parameter
    ``default=`` declarations.  Skips mean-field-marked parameters
    (they're saddles, computed not configured).
    """
    out: dict = {}
    for p in model.get('parameters', []):
        if p.get('mean_field'):
            continue
        if 'default' in p and p['default'] is not None:
            out[p['name']] = p['default']
    return out


def _solve_mf_at_saddle(model: dict, fundamental: dict, ft) -> dict:
    """Run the DAE solver if the model declares equations; otherwise
    fall back to the legacy iteration solver.  Returns the saddle
    values dict (``{<var>star: [vals]}``)."""
    if model.get('equations'):
        from api._mean_field_dae import solve_mean_field_dae
        result = solve_mean_field_dae(model, fundamental, verbose=False)
        return result.get('mf_values', {})

    # Legacy iteration solver path.
    from api._mean_field import solve_mean_field
    # The legacy solver takes the FieldTheory instance and the model
    # together with the fundamental dict.
    result = solve_mean_field(ft, model, fundamental, verbose=False)
    return result.get('num_saddles', {}) or {}

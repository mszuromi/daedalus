"""
Phase J refactor — fixture-running utilities.

Single entry point ``evaluate(config) -> dict`` that loads a theory,
runs ``compute_cumulants``, and evaluates ``total_C`` at the
configured probe τ-points.  Used by both ``_freeze.py`` (which writes
.npz files) and ``test_phase_j_refactor_regression.py`` (which
re-runs the same configs and compares to the frozen values).
"""
from __future__ import annotations

import importlib.util
import os
import time
from typing import Any

import numpy as np

from tests.phase_j_refactor_fixtures._configs import FixtureConfig


def _load_theory(theory_file: str):
    """Load a theories/<file> via importlib and return the model dict."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    full_path = os.path.join(repo_root, 'theories', theory_file)
    spec = importlib.util.spec_from_file_location(
        f'_phase_j_fixture_theory_{os.path.basename(theory_file)}',
        full_path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def evaluate(config: FixtureConfig) -> dict[str, Any]:
    """Run ``compute_cumulants`` for ``config`` and evaluate ``total_C``
    at every probe τ-point.

    Returns
    -------
    dict with keys:
        ``'tau_probes'`` — (n_probes, k) float64 array, copied from config
        ``'C_values'``   — (n_probes,)   complex128 array of total_C(*τ)
        ``'wall_time'``  — float seconds, end-to-end compute_cumulants
        ``'name'``       — str, fixture name (for downstream tagging)
        ``'rtol'``, ``'atol'`` — float tolerances from the config

    No exceptions are caught here — propagate them so the freeze script
    and the regression test can report them with full traceback.
    """
    # Local import so pytest collection doesn't pay the pipeline import
    # cost when a config can be resolved purely from _configs.py.
    from pipeline import compute_cumulants

    model = _load_theory(config.theory_file)

    # Pick the smallest τ-grid that compute_cumulants accepts so the
    # pipeline's internal sweep does minimal wasted work; we evaluate
    # at our own probe points afterwards.  Default to 1.0 if every
    # probe is at τ=0 (k=1 stationary case).
    nonzero_probe_taus = [
        abs(t) for probe in config.tau_probes for t in probe
        if abs(t) > 0
    ]
    grid_extent = max(nonzero_probe_taus) if nonzero_probe_taus else 1.0

    kwargs = dict(
        model=model,
        k=config.k,
        max_ell=config.max_ell,
        fundamental=config.fundamental,
        external_fields=config.external_fields,
        tau_max=grid_extent,
        tau_step=grid_extent,
        parallel=False,
        verbose=False,
        use_cache=True,
        use_grouped_phase_j=config.use_grouped_phase_j,
        origin_leaf_idx=config.origin_leaf_idx,
    )
    if config.taylor_order is not None:
        kwargs['taylor_order'] = config.taylor_order

    t0 = time.perf_counter()
    th = compute_cumulants(**kwargs)
    wall_time = time.perf_counter() - t0

    # compute_cumulants returns ``'total_C'`` (callable summing all ells)
    # and ``'total_C_by_ell'`` (dict).  Use total_C directly.
    total_C = th.get('total_C')
    if total_C is None:
        per_ell = th.get('total_C_by_ell', {})

        def total_C(*tau):  # type: ignore[misc]
            return sum(complex(fn(*tau)) for fn in per_ell.values())

    C_values = np.array(
        [complex(total_C(*probe)) for probe in config.tau_probes],
        dtype=np.complex128,
    )

    return {
        'name': config.name,
        'tau_probes': np.array(config.tau_probes, dtype=np.float64),
        'C_values': C_values,
        'wall_time': float(wall_time),
        'rtol': float(config.rtol),
        'atol': float(config.atol),
    }


def fixture_path(name: str) -> str:
    """Absolute path to the .npz file for a given fixture name."""
    return os.path.join(
        os.path.dirname(__file__), f'{name}.npz',
    )


def save_frozen(result: dict[str, Any]) -> str:
    """Write the frozen .npz for one fixture run.  Returns the path."""
    path = fixture_path(result['name'])
    np.savez(
        path,
        tau_probes=result['tau_probes'],
        C_values=result['C_values'],
        wall_time=np.array([result['wall_time']]),
        rtol=np.array([result['rtol']]),
        atol=np.array([result['atol']]),
    )
    return path


def load_frozen(name: str) -> dict[str, Any]:
    """Load the frozen .npz for one fixture name."""
    path = fixture_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Frozen fixture {name!r} not found at {path}.  Run '
            f'``tests/phase_j_refactor_fixtures/_freeze.py`` to '
            f'generate it.'
        )
    data = np.load(path)
    return {
        'name': name,
        'tau_probes': data['tau_probes'],
        'C_values': data['C_values'],
        'wall_time': float(data['wall_time'][0]),
        'rtol': float(data['rtol'][0]),
        'atol': float(data['atol'][0]),
    }

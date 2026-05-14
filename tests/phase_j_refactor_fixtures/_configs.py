"""
Phase J refactor — regression fixture configurations.

Each entry is a self-contained spec for one ``compute_cumulants`` call
plus a list of probe (k-tuple-of-floats) ``tau_points`` at which the
returned ``total_C(*tau)`` value will be frozen.

Adding a new fixture: append an entry to ``FIXTURES``, then re-run
``_freeze.py`` (which will append/overwrite the corresponding .npz).

Removing or renaming an entry is BREAKING — the regression test will
flag missing or extraneous .npz files at collection time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class FixtureConfig:
    name: str                       # short slug, used as .npz filename
    theory_file: str                # path under repo's theories/ dir
    k: int
    max_ell: int
    fundamental: dict
    external_fields: list
    tau_probes: list                # list of k-tuples (floats)
    # Optional knobs that affect the result:
    origin_leaf_idx: Optional[int] = 0
    taylor_order: Optional[int] = None        # None → pipeline default
    use_grouped_phase_j: bool = False
    # Tolerance for the regression test (set per-fixture in case some
    # configs are inherently noisier than others — e.g. theories with
    # very small C(τ) where machine ε dominates).
    rtol: float = 1e-10
    atol: float = 1e-12
    # Free-form notes for humans reading the .npz metadata.
    notes: str = ''


# --- Shared parameter dictionaries (one per theory) ------------------

_FUNDAMENTAL_SPIKE_RESET = {
    'Em':   [3.5, 3.5],
    'tau':  [10.0, 9.0],
    'a':    [2.5, 2.5],
    'taug': [[2.0, 3.0], [1.0, 3.0]],
    'w':    [[0.55, 0.65], [0.7, 0.8]],
}

_FUNDAMENTAL_QUAD = {
    'Em':   [0.8, 0.78],
    'tau':  [10.0, 9.0],
    'a':    [0.44, 0.44],
    'taug': [[2.0, 3.0], [1.0, 3.0]],
    'w':    [[0.25, 0.25], [0.2, 0.3]],
}


# --- Fixtures --------------------------------------------------------

FIXTURES: list[FixtureConfig] = [
    # ─── single_population_spike_reset_test ─────────────────────────
    FixtureConfig(
        name='spike_reset_k1_ell1',
        theory_file='single_population_spike_reset_test.theory.py',
        k=1,
        max_ell=1,
        fundamental=_FUNDAMENTAL_SPIKE_RESET,
        external_fields=[('n', 1)],
        # k=1: total_C is rate-independent of probe time for a stationary
        # process, so a single probe suffices.
        tau_probes=[(0.0,)],
        notes='Linear phi + spike reset, tadpole 1-loop rate shift.',
    ),
    FixtureConfig(
        name='spike_reset_k2_ell0',
        theory_file='single_population_spike_reset_test.theory.py',
        k=2,
        max_ell=0,
        fundamental=_FUNDAMENTAL_SPIKE_RESET,
        external_fields=[('n', 1), ('n', 2)],
        # k=2 cross-cumulant: probe at a spread of τ values.
        tau_probes=[
            (0.0, -10.0), (0.0, -5.0), (0.0, 0.0),
            (0.0, 5.0), (0.0, 10.0),
        ],
        notes='Tree-level cross-cumulant, linear phi + reset.',
    ),
    # NOTE: spike_reset_k2_ell1 (1-loop fixture exercising the m>=1
    # polytope integrator) is intentionally OMITTED here.  Even a
    # single-probe run takes ~30+ minutes on the current code path —
    # too slow for a tight refactor iteration loop.  When Stage 3b
    # (causal-poset integrator) lands and the m>=1 integration is fast
    # enough, add a fixture for it here (and re-freeze).

    # ─── single_population_quad_exp_test ────────────────────────────
    FixtureConfig(
        name='quad_exp_k2_ell0',
        theory_file='single_population_quad_exp_test.theory.py',
        k=2,
        max_ell=0,
        fundamental=_FUNDAMENTAL_QUAD,
        external_fields=[('n', 1), ('n', 2)],
        tau_probes=[
            (0.0, -10.0), (0.0, -5.0), (0.0, 0.0),
            (0.0, 5.0), (0.0, 10.0),
        ],
        notes='Quadratic phi, no reset.  Legacy propagator path with '
              'Newton-refined poles after the e8eec73 fix.',
    ),
]


def fixture_by_name(name: str) -> FixtureConfig:
    for fx in FIXTURES:
        if fx.name == name:
            return fx
    raise KeyError(f'No fixture named {name!r}.  Known: '
                   f'{[f.name for f in FIXTURES]}')

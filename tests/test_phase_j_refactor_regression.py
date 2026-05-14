"""
tests/test_phase_j_refactor_regression.py
=========================================
Regression tests for the Phase J integration refactor (phase-j-refactor
branch).

Each test re-runs one configuration from
``tests/phase_j_refactor_fixtures/_configs.FIXTURES``, evaluates
``total_C(*τ)`` at the probe τ-points, and asserts agreement with the
frozen ``.npz`` reference values (generated once on the pre-refactor
code state via ``_freeze.py``).

The point of these tests is NOT to validate physics — that's already
done by the sim_compare notebooks.  These tests catch silent magnitude
drift when an internal change (subset pre-pruning, EdgeModeSum
refactor, polygon integrators, …) accidentally changes the output of
the existing integration pipeline.

Run with::

    sage -python -m pytest tests/test_phase_j_refactor_regression.py -v

Tolerances are configured per-fixture in ``_configs.py``.  Default
``rtol=1e-10`` is tight enough that any real algorithmic change shows
up; raise per-fixture if a refactor justifies the looser bound (e.g.
switching to a different but mathematically-equivalent quadrature
strategy that has its own ε floor).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.phase_j_refactor_fixtures._configs import FIXTURES
from tests.phase_j_refactor_fixtures._runner import (
    evaluate,
    fixture_path,
    load_frozen,
)


def _fixture_ids():
    return [fx.name for fx in FIXTURES]


@pytest.fixture(scope='session', autouse=True)
def _check_all_fixtures_frozen():
    """Fail-fast at collection time if any fixture is missing its .npz.

    The regression suite is meaningless without all the frozen
    references, so we'd rather error loudly than silently skip.
    """
    missing = []
    for fx in FIXTURES:
        if not os.path.exists(fixture_path(fx.name)):
            missing.append(fx.name)
    if missing:
        pytest.fail(
            f'Frozen fixtures missing: {missing}.  Run '
            f'``sage -python tests/phase_j_refactor_fixtures/_freeze.py`` '
            f'on the pre-refactor branch, commit the .npz files, then '
            f're-run this test.',
            pytrace=False,
        )


@pytest.mark.parametrize('fx', FIXTURES, ids=_fixture_ids())
def test_total_C_matches_frozen_reference(fx):
    """For each fixture, total_C at the probe τ-points must match the
    frozen reference values to within the per-fixture tolerance.
    """
    frozen = load_frozen(fx.name)
    current = evaluate(fx)

    # τ-probe arrays should match by construction (both come from the
    # same FixtureConfig).  Guard against accidental config drift.
    np.testing.assert_allclose(
        current['tau_probes'], frozen['tau_probes'],
        err_msg=f'τ-probe drift in fixture {fx.name!r}: config was '
                f'edited but fixture not re-frozen?'
    )

    np.testing.assert_allclose(
        current['C_values'], frozen['C_values'],
        rtol=fx.rtol, atol=fx.atol,
        err_msg=(
            f'total_C regression in fixture {fx.name!r}:\n'
            f'  τ_probes = {frozen["tau_probes"].tolist()}\n'
            f'  frozen   = {frozen["C_values"].tolist()}\n'
            f'  current  = {current["C_values"].tolist()}\n'
            f'  rtol={fx.rtol:.0e}, atol={fx.atol:.0e}\n'
            f'\n'
            f'If the change is intentional (e.g. an integrator '
            f'rewrite with its own ε floor), update the tolerance '
            f'on this fixture in _configs.py.  If unintentional, '
            f'the refactor has changed the numerical output of an '
            f'integration path.'
        ),
    )

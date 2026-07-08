"""Regression tests for KEY=tau0enum (two LOW-risk correctness fixes).

FIX A — Itô τ=0 left-limit consistency between callable and grid.
    The Itô equal-time LEFT-limit nudge (evaluate at ``anchor − _ITO_EPS``
    instead of exactly the discontinuity at τ=0) was applied to the ``C_tau``
    grid but NOT to the user-facing ``total_C(t0, t1)`` callable.  As a result
    ``total_C(t0, t0)`` sampled the measure-zero τ=0 step discontinuity and
    disagreed with ``C_tau`` at τ=0 by ~``_ITO_EPS``·slope.  The per-ell (and
    hence master) callables are now wrapped so an equal-time evaluation gets
    the same left-limit nudge as the grid.

FIX B — remove the un-proven tree-vertex clamp in tree enumeration.
    ``generate_trees_with_constraints`` carried an extra, un-proven clamp
    ``max_n = min(max_n, num_leaves + min(10, v2_max + v3_max))`` on top of the
    PROVEN vertex bound.  It silently truncated the search at high vertex counts
    and could drop valid trees (completeness loss at loop order ≥ 3 / higher k).
    It is inert at the suite's tested surface (k ≤ 3, ell ≤ 2), where the proven
    bound already binds; removing it restores completeness without changing any
    currently-tested output.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import daedalus as dd                       # noqa: E402
from api import compute_cumulants           # noqa: E402
from api.compute import _ITO_EPS            # noqa: E402
from engine.enumeration.loop_diagram_enumeration import (  # noqa: E402
    generate_trees_with_constraints,
)


# ---------------------------------------------------------------------------
# FIX A
# ---------------------------------------------------------------------------

def test_total_C_callable_matches_C_tau_at_tau0():
    """total_C(t0, t0) must equal C_tau at τ=0 (same Itô left-limit nudge)."""
    model, _mod = dd.load_model('ou_quartic')
    tau = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    res = compute_cumulants(
        model, k=2, max_ell=1,
        external_fields=[('dx', 1), ('dx', 1)],
        parameters={'mu': 1.0, 'D': 1.0, 'eps': 0.02},
        tau_grid=tau, verbose=False, use_cache=False, parallel=False)

    C_tau = np.asarray(res['C_tau'])
    total_C = res['total_C']
    i0 = int(np.argmin(np.abs(tau)))

    # The equal-time callable evaluation must agree with the grid at τ=0.
    assert np.isclose(complex(total_C(0.0, 0.0)), complex(C_tau[i0]),
                      rtol=0, atol=1e-12)

    # Sanity: off-zero the callable still agrees with the grid (no regression).
    i1 = int(np.argmin(np.abs(tau - 0.5)))
    assert np.isclose(complex(total_C(0.0, 0.5)), complex(C_tau[i1]),
                      rtol=0, atol=1e-12)

    # The fix is load-bearing: the RAW (un-nudged) per-ell callables, summed,
    # would have differed from the grid by ~_ITO_EPS at τ=0.  Confirm the bug
    # existed and that the wrapped callable closes it.
    raw_at_zero = sum(
        complex(d['total_C'](0.0, 0.0))
        for d in res['phase_j_by_ell'].values() if d is not None)
    assert abs(raw_at_zero - complex(C_tau[i0])) > 1e-8     # bug was real
    assert abs(complex(total_C(0.0, 0.0)) - raw_at_zero) > 1e-8  # nudge applied


# ---------------------------------------------------------------------------
# FIX B
# ---------------------------------------------------------------------------

def _old_clamp_max_n(k, ell, num_leaves, j):
    """The vertex cap the DELETED un-proven clamp would have imposed."""
    v3_max = k + j - 2
    v2_max = 2 * v3_max + 3 * ell - k - 3 * j + 3
    proven_or_cap = min(num_leaves + v2_max + v3_max, 50)
    return min(proven_or_cap, num_leaves + min(10, v2_max + v3_max))


def _proven_max_n(k, ell, num_leaves, j):
    """The PROVEN bound (∧ the global max_vertices_search=50 cap) that remains."""
    v3_max = k + j - 2
    v2_max = 2 * v3_max + 3 * ell - k - 3 * j + 3
    return min(num_leaves + v2_max + v3_max, 50)


def test_clamp_removed_restores_truncated_trees():
    """At k=4, ell=2, j=0 the old clamp capped the search at n=14 while the
    proven bound allows n=15.  Trees with n=15 vertices (previously dropped)
    must now appear — proving the un-proven clamp is gone."""
    k, ell, j = 4, 2, 0
    num_leaves = k + j
    assert _old_clamp_max_n(k, ell, num_leaves, j) == 14   # old behavior
    assert _proven_max_n(k, ell, num_leaves, j) == 15      # proven allows more

    trees = generate_trees_with_constraints(k, ell)
    present = sorted({t.order() for (t, jj, nl) in trees if nl == num_leaves})
    # The n=15 tree(s) excluded by the old clamp are now enumerated.
    assert 15 in present
    assert max(present) == _proven_max_n(k, ell, num_leaves, j)


def test_clamp_removal_is_inert_at_tested_orders():
    """At the suite's tested surface (k ≤ 3, ell ≤ 2) the proven bound already
    binds, so removing the un-proven clamp changes NOTHING: every generated
    tree's vertex count stays within what the old clamp would have allowed."""
    for k, ell in [(2, 1), (2, 2), (3, 1), (3, 2)]:
        trees = generate_trees_with_constraints(k, ell)
        for (t, j, num_leaves) in trees:
            assert t.order() <= _old_clamp_max_n(k, ell, num_leaves, j)


def test_ito_eps_constant_present():
    """Guard: the left-limit regularizer magnitude is the documented value."""
    assert _ITO_EPS == 1e-6

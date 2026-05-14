"""
tests/test_causal_poset.py
==========================
Stage 3b-prep tests: partial-order extraction from retardation
constraints + linear-extension enumeration.  Each test isolates one
piece so failures are easy to localise before the closed-form
integrator (Stage 3b-nested) is built on top.

Run with::

    sage -python -m pytest tests/test_causal_poset.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.time_domain.final_integral import (
    _CausalPoset,
    _extract_causal_poset,
    _enumerate_linear_extensions,
    _causal_poset_consistent_scalar_lower,
    _causal_poset_consistent_scalar_upper,
)


# ───────────────────────────────────────────────────────────────────
# _extract_causal_poset
# ───────────────────────────────────────────────────────────────────

def test_extract_simple_chain_3_vars():
    """Constraints  s_1 > s_0,  s_2 > s_1  → edges (0,1), (1,2)."""
    constraints = [
        ([-1.0, 1.0, 0.0], [], 0.0),   # s_1 > s_0
        ([0.0, -1.0, 1.0], [], 0.0),   # s_2 > s_1
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=3)
    assert poset is not None
    assert poset.m == 3
    assert poset.edges == ((0, 1), (1, 2))
    assert poset.scalar_lowers == ()
    assert poset.scalar_uppers == ()


def test_extract_with_scalar_lower_bound():
    """Constraint  s_0 > 5.0  encoded as +1·s_0 + (−5.0) > 0."""
    constraints = [
        ([1.0, 0.0], [], -5.0),
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is not None
    assert poset.edges == ()
    assert poset.scalar_lowers == ((0, 5.0),)
    assert poset.scalar_uppers == ()


def test_extract_with_scalar_upper_bound():
    """Constraint  s_1 < 7.0  encoded as −1·s_1 + 7.0 > 0."""
    constraints = [
        ([0.0, -1.0], [], 7.0),
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is not None
    assert poset.scalar_uppers == ((1, 7.0),)


def test_extract_with_external_time_in_lower():
    """Constraint  s_0 > t_ext  with a_ext = [+1], c0 = 0.

    For t_ext = 2.5, lower is 2.5.
    """
    constraints = [
        ([1.0, 0.0], [-1.0], 0.0),  # s_0 - t_ext > 0
    ]
    poset = _extract_causal_poset(
        constraints, free_ext_vals=[2.5], m=2,
    )
    assert poset is not None
    assert len(poset.scalar_lowers) == 1
    var, val = poset.scalar_lowers[0]
    assert var == 0
    assert abs(val - 2.5) < 1e-12


def test_extract_rejects_mixed_constraint():
    """Constraint with three nonzero coefficients (not a clean ±1
    pair) → mixed → returns None."""
    constraints = [
        ([1.0, 1.0, -1.0], [], 0.0),  # not a pure ordering
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=3)
    assert poset is None


def test_extract_rejects_shifted_inter_axis():
    """Inter-axis constraint with c_eff ≠ 0  (e.g.  s_v > s_u + c
    for c ≠ 0) → not supported in the chain form → None."""
    constraints = [
        ([-1.0, 1.0], [], 3.0),  # s_1 > s_0 + 3 — not a pure poset edge
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is None


def test_extract_dedup_repeated_edges():
    """Two redundant copies of the same edge are deduplicated."""
    constraints = [
        ([-1.0, 1.0], [], 0.0),
        ([-1.0, 1.0], [], 0.0),
    ]
    poset = _extract_causal_poset(constraints, free_ext_vals=[], m=2)
    assert poset is not None
    assert poset.edges == ((0, 1),)


# ───────────────────────────────────────────────────────────────────
# _enumerate_linear_extensions
# ───────────────────────────────────────────────────────────────────

def test_linear_extensions_empty_poset():
    """No edges, m=3 → all 3! = 6 permutations are extensions."""
    poset = _CausalPoset(m=3, edges=(), scalar_lowers=(),
                         scalar_uppers=())
    exts = list(_enumerate_linear_extensions(poset))
    assert len(exts) == 6
    # Should include every permutation of (0, 1, 2)
    import itertools
    expected = set(itertools.permutations(range(3)))
    assert set(exts) == expected


def test_linear_extensions_chain_3_vars():
    """Chain  0 → 1 → 2  has exactly ONE linear extension."""
    poset = _CausalPoset(m=3, edges=((0, 1), (1, 2)),
                         scalar_lowers=(), scalar_uppers=())
    exts = list(_enumerate_linear_extensions(poset))
    assert exts == [(0, 1, 2)]


def test_linear_extensions_y_poset():
    """Y-shaped DAG: 0 → 2, 1 → 2 (two sources, one sink).
    Linear extensions: (0, 1, 2) and (1, 0, 2)."""
    poset = _CausalPoset(m=3, edges=((0, 2), (1, 2)),
                         scalar_lowers=(), scalar_uppers=())
    exts = list(_enumerate_linear_extensions(poset))
    assert sorted(exts) == [(0, 1, 2), (1, 0, 2)]


def test_linear_extensions_diamond():
    """Diamond:  0 → 1, 0 → 2, 1 → 3, 2 → 3.
    Linear extensions: (0, 1, 2, 3) and (0, 2, 1, 3)."""
    poset = _CausalPoset(
        m=4, edges=((0, 1), (0, 2), (1, 3), (2, 3)),
        scalar_lowers=(), scalar_uppers=(),
    )
    exts = list(_enumerate_linear_extensions(poset))
    assert sorted(exts) == [(0, 1, 2, 3), (0, 2, 1, 3)]


def test_linear_extensions_count_matches_known_formula():
    """For ``m`` totally unrelated vars, the number of extensions
    equals m! — sanity check for m=4, 5."""
    import math
    for m in (4, 5):
        poset = _CausalPoset(m=m, edges=(), scalar_lowers=(),
                              scalar_uppers=())
        exts = list(_enumerate_linear_extensions(poset))
        assert len(exts) == math.factorial(m), \
            f'm={m}: got {len(exts)} extensions, expected {math.factorial(m)}'


# ───────────────────────────────────────────────────────────────────
# _causal_poset_consistent_scalar_lower / upper
# ───────────────────────────────────────────────────────────────────

def test_consistent_scalar_lower_all_equal():
    """All variables share L = 1.0 → returns (1.0, True)."""
    poset = _CausalPoset(
        m=3, edges=(), scalar_lowers=((0, 1.0), (1, 1.0), (2, 1.0)),
        scalar_uppers=(),
    )
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is True
    assert abs(L - 1.0) < 1e-12


def test_consistent_scalar_lower_missing_var():
    """Only some variables have a scalar lower → ``False`` (the
    chain-simplex closed form requires uniform bounds)."""
    poset = _CausalPoset(
        m=3, edges=(), scalar_lowers=((0, 1.0), (1, 1.0)),
        scalar_uppers=(),
    )
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is False


def test_consistent_scalar_lower_differing_values():
    """Two variables have different lower bounds → ``False``."""
    poset = _CausalPoset(
        m=2, edges=(), scalar_lowers=((0, 1.0), (1, 2.0)),
        scalar_uppers=(),
    )
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is False


def test_consistent_scalar_lower_no_lowers_at_all():
    """No scalar lowers → returns ``(None, True)``  (caller will
    supply a bbox cap)."""
    poset = _CausalPoset(m=3, edges=(), scalar_lowers=(),
                         scalar_uppers=())
    L, ok = _causal_poset_consistent_scalar_lower(poset)
    assert ok is True
    assert L is None


def test_consistent_scalar_upper_per_var_min():
    """Per-variable scalar uppers are returned as a dict.  When a
    variable has multiple uppers the tightest (min) wins."""
    poset = _CausalPoset(
        m=2, edges=(),
        scalar_lowers=(),
        scalar_uppers=((0, 5.0), (0, 3.0), (1, 10.0)),
    )
    uppers = _causal_poset_consistent_scalar_upper(poset)
    assert uppers == {0: 3.0, 1: 10.0}

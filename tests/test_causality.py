"""
tests/test_causality.py
=======================
Tests for msrjd.diagrams.causality — Build Phase F.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_causality.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, I, DiGraph

from msrjd.diagrams.causality import check_pole_structure, check_causality, filter_causal
from msrjd.diagrams.type_assignment import TypedDiagram


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_td(edges, leaves, is_dag=True):
    """Build a minimal TypedDiagram for testing."""
    D = DiGraph(edges)
    G = D.to_undirected()
    leaf_set = set(leaves)
    internal = sorted(set(D.vertices()) - leaf_set)
    pd = (D, G, list(leaves), internal)
    return TypedDiagram(pd, {}, {}, {}, {})


# ── Tests: check_pole_structure ─────────────────────────────────────────────

def test_no_poles():
    passed, detail, conds = check_pole_structure([])
    assert passed is True
    assert 'No poles' in detail


def test_upper_half_plane_poles():
    """Poles with strictly positive imaginary parts pass."""
    tau = SR.var('tau', domain='positive')
    poles = [I / tau, 2 * I / tau]
    passed, detail, conds = check_pole_structure(poles)
    assert passed is True


def test_pure_imaginary_positive():
    poles = [3 * I, I / 2]
    passed, detail, conds = check_pole_structure(poles)
    assert passed is True
    assert len(conds) == 0


def test_lower_half_plane_fails():
    """A pole with negative imaginary part fails."""
    poles = [-I]
    passed, detail, conds = check_pole_structure(poles)
    assert passed is False
    assert 'FAILED' in detail


def test_real_pole_fails():
    """A real pole (Im = 0) fails."""
    poles = [SR(3)]
    passed, detail, conds = check_pole_structure(poles)
    assert passed is False


def test_mixed_poles_fail():
    """One good pole and one bad pole → fails."""
    poles = [I, -I]
    passed, detail, conds = check_pole_structure(poles)
    assert passed is False


def test_symbolic_pole_conditional():
    """A pole at i*a where a has no domain → conditional."""
    a = SR.var('a')
    poles = [I * a]
    passed, detail, conds = check_pole_structure(poles)
    # Should be conditional since sign of a is unknown
    assert 'CONDITIONAL' in detail or passed is True
    # If conditional, there should be conditions
    if 'CONDITIONAL' in detail:
        assert len(conds) > 0


# ── Tests: check_causality ──────────────────────────────────────────────────

def test_dag_passes_structural():
    td = _make_td([(0, 2), (2, 1)], leaves=[0, 1])
    passed, detail, _ = check_causality(td)
    assert passed is True


def test_cycle_fails():
    """A directed cycle is acausal."""
    D = DiGraph([(0, 1), (1, 2), (2, 0)])
    G = D.to_undirected()
    pd = (D, G, [], [0, 1, 2])
    td = TypedDiagram(pd, {}, {}, {}, {})
    passed, detail, _ = check_causality(td)
    assert passed is False
    assert 'cycle' in detail.lower()


def test_causality_with_poles():
    td = _make_td([(0, 1)], leaves=[0, 1])
    poles = [2 * I, 3 * I]
    passed, detail, _ = check_causality(td, pole_vals=poles)
    assert passed is True


def test_causality_bad_poles():
    td = _make_td([(0, 1)], leaves=[0, 1])
    poles = [-I]
    passed, detail, _ = check_causality(td, pole_vals=poles)
    assert passed is False


# ── Tests: filter_causal ────────────────────────────────────────────────────

def test_filter_causal_keeps_good():
    td1 = _make_td([(0, 2), (2, 1)], leaves=[0, 1])
    td2 = _make_td([(0, 1)], leaves=[0, 1])
    kept, n_disc, details = filter_causal([td1, td2])
    assert len(kept) == 2
    assert n_disc == 0


def test_filter_causal_removes_bad():
    td_good = _make_td([(0, 1)], leaves=[0, 1])
    kept, n_disc, details = filter_causal([td_good], pole_vals=[-I])
    assert len(kept) == 0
    assert n_disc == 1

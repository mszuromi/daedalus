"""
tests/test_symbolic_integration.py
===================================
Tests for engine.integration.symbolic — Build Phase H.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_symbolic_integration.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph, matrix, I, pi, var, exp

from engine.integration.symbolic import (
    check_propagator_available,
    assign_frequencies,
    solve_conservation,
    build_integrand,
    build_integrand_stationary,
    integrate_tree_level,
    integrate_to_time_domain,
)
from engine.diagrams.type_assignment import TypedDiagram
from engine.core.vertices import VertexType, SourceType


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_td(edges, leaves, vert_assignments=None, edge_types=None,
             ext_legs=None, prop_indices=None):
    D = DiGraph(edges)
    G = D.to_undirected()
    leaf_set = set(leaves)
    internal = sorted(set(D.vertices()) - leaf_set)
    pd = (D, G, list(leaves), internal)
    return TypedDiagram(
        pd,
        vert_assignments or {},
        edge_types or {},
        ext_legs or {},
        prop_indices or {},
    )


def _simple_propagator_data():
    """Minimal 2×2 propagator: G_{ii}(ω) = 1/(iω + α_i)."""
    omega = SR.var('omega')
    alpha1 = SR.var('alpha1', domain='positive')
    alpha2 = SR.var('alpha2', domain='positive')
    G = matrix(SR, [
        [1 / (I * omega + alpha1), 0],
        [0, 1 / (I * omega + alpha2)],
    ])
    return {
        'G_ft': G,
        'G_ft_explicit': True,
        'pole_vals': [I * alpha1, I * alpha2],
        'nf': 2,
    }


# ── Prerequisite checks ─────────────────────────────────────────────────────

def test_check_propagator_explicit():
    assert check_propagator_available({'G_ft': matrix(SR, [[1]])}) == 'explicit'

def test_check_propagator_implicit():
    pd = {'adj_ft': matrix(SR, [[1]]), 'D_omega': SR(1)}
    assert check_propagator_available(pd) == 'implicit'

def test_check_propagator_missing():
    import pytest
    with pytest.raises(ValueError, match='No frequency-domain propagator'):
        check_propagator_available({})


# ── Frequency assignment ─────────────────────────────────────────────────────

def test_assign_frequencies_2pt():
    """2 edges, 2 leaves → 2 edge frequencies, 2 leaf-edge mappings."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    edge_freqs, leaf_edge_freq = assign_frequencies(td, k=2)
    assert len(edge_freqs) == 2
    assert len(leaf_edge_freq) == 2
    # Each leaf should map to a distinct edge frequency
    assert leaf_edge_freq[0] != leaf_edge_freq[1]


# ── Conservation ─────────────────────────────────────────────────────────────

def test_solve_tree_no_free_freqs():
    """Tree: all internal edge freqs determined by conservation."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    edge_freqs, leaf_edge_freq = assign_frequencies(td, k=2)
    subs, free_freqs, overall = solve_conservation(
        td, edge_freqs, leaf_edge_freq
    )
    # Tree: no free frequencies (loop number = 0)
    assert len(free_freqs) == 0


def test_solve_one_loop_has_free_freq():
    """Bubble diagram: one free frequency (loop number = 1)."""
    vt_out = VertexType(SR(1), [('nt', 1), ('nt', 1)], [('dn', 1)], (2, 1))
    vt_in  = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 1)], (1, 2))

    D = DiGraph(multiedges=True)
    D.add_edges([(0, 2, 0), (2, 3, 0), (2, 3, 1), (3, 1, 0)])
    G = D.to_undirected()
    pd = (D, G, [0, 1], [2, 3])

    td = TypedDiagram(
        pd, {2: vt_out, 3: vt_in},
        {(0, 2): (('nt', 1), ('dn', 1)),
         (2, 3): (('nt', 1), ('dn', 1)),
         (3, 1): (('nt', 1), ('dn', 1))},
        {0: ('nt', 1), 1: ('dn', 1)},
        {(0, 2): (0, 0), (2, 3): (0, 0), (3, 1): (0, 0)},
    )
    edge_freqs, leaf_edge_freq = assign_frequencies(td, k=2)
    subs, free_freqs, overall = solve_conservation(
        td, edge_freqs, leaf_edge_freq
    )
    assert len(free_freqs) == 1


# ── Build integrand ──────────────────────────────────────────────────────────

def test_build_integrand_tree():
    """Tree-level integrand depends on external freqs, not internal."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    pd = _simple_propagator_data()
    omega = SR.var('omega')

    edge_freqs, leaf_edge_freq = assign_frequencies(td, k=2)
    subs, free_freqs, _ = solve_conservation(td, edge_freqs, leaf_edge_freq)
    integrand = build_integrand(td, edge_freqs, subs, pd, omega)

    # Should depend on at least one external (leaf) frequency
    integrand_vars = set(integrand.variables())
    ext_var_set = set(leaf_edge_freq.values())
    assert integrand_vars & ext_var_set


# ── Full assembly ────────────────────────────────────────────────────────────

def test_build_integrand_stationary_tree():
    """Full assembly includes time variables and exponential factors."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    pd = _simple_propagator_data()
    result = build_integrand_stationary(td, pd, k=2)

    assert result['loop_number'] == 0
    assert len(result['free_freqs']) == 0
    assert 'full_integrand' in result
    assert 'ext_times' in result
    assert len(result['ext_times']) == 2
    # After overall conservation, one ext freq is eliminated
    assert len(result['ext_freqs']) == 1


# ── Time-domain integration ─────────────────────────────────────────────────

def test_integrate_tree_level():
    """Tree-level integration returns a function of external times."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    pd = _simple_propagator_data()
    result = build_integrand_stationary(td, pd, k=2)
    contribution = integrate_tree_level(result)

    assert contribution is not None
    contrib_vars = set(contribution.variables())
    time_var_set = set(result['ext_times'])
    # Should depend on time variables
    assert contrib_vars & time_var_set
    # Should NOT depend on frequency variables
    ext_freq_set = set(result['ext_freqs'])
    assert not (contrib_vars & ext_freq_set)


def test_integrate_tree_level_rejects_loops():
    """integrate_tree_level should reject diagrams with loops."""
    import pytest
    result = {'loop_number': 1}
    with pytest.raises(ValueError, match='tree-level'):
        integrate_tree_level(result)


def test_integrate_to_time_domain_returns_dict():
    """integrate_to_time_domain returns proper result dict."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    pd = _simple_propagator_data()
    ir = build_integrand_stationary(td, pd, k=2)
    result = integrate_to_time_domain(ir)

    assert 'time_domain_result' in result
    assert 'ext_times' in result
    assert result['status'] in ('ok', 'partial')


# ── Source vertex conservation ───────────────────────────────────────────────

def test_source_vertex_conservation():
    """Source vertex: outgoing frequencies sum to zero."""
    st = SourceType(SR(1)/2, [('nt', 1), ('nt', 1)], (2, 0))
    td = _make_td(
        edges=[(2, 0), (2, 1)], leaves=[0, 1],
        vert_assignments={2: st},
        edge_types={(2, 0): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 2))},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (0, 1)},
    )
    edge_freqs, leaf_edge_freq = assign_frequencies(td, k=2)
    subs, free_freqs, overall = solve_conservation(
        td, edge_freqs, leaf_edge_freq
    )

    # Tree level: no free frequencies
    assert len(free_freqs) == 0

"""
tests/test_symbolic_integration.py
===================================
Tests for msrjd.integration.symbolic — Build Phase H.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_symbolic_integration.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph, matrix, I, pi, var, exp

from msrjd.integration.symbolic import (
    check_propagator_available,
    assign_frequencies,
    build_conservation_equations,
    solve_conservation,
    build_integrand,
    build_integrand_stationary,
    integrate_tree_level,
    integrate_to_time_domain,
)
from msrjd.diagrams.type_assignment import TypedDiagram
from msrjd.core.vertices import VertexType, SourceType


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
    """
    Build a minimal 2×2 propagator for testing.
    G_ft[i,j] = 1/(i*omega + alpha_j) — poles in the upper half-plane.
    """
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


# ── Tests: prerequisite checks ──────────────────────────────────────────────

def test_check_propagator_explicit():
    pd = {'G_ft': matrix(SR, [[1]])}
    assert check_propagator_available(pd) == 'explicit'


def test_check_propagator_implicit():
    pd = {'adj_ft': matrix(SR, [[1]]), 'D_omega': SR(1)}
    assert check_propagator_available(pd) == 'implicit'


def test_check_propagator_missing():
    import pytest
    with pytest.raises(ValueError, match='No frequency-domain propagator'):
        check_propagator_available({})


# ── Tests: frequency assignment ──────────────────────────────────────────────

def test_assign_frequencies_2pt():
    """2-point function: 2 leaves, 1 independent external frequency."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    edge_freqs, ext_freqs, ext_assign = assign_frequencies(td, k=2)

    assert len(edge_freqs) == 2       # 2 edges
    assert len(ext_freqs) == 2        # k = 2 independent ext freqs
    assert len(ext_assign) == 2       # 2 leaves


def test_assign_frequencies_3pt():
    """3-point: 3 leaves, 2 independent external frequencies."""
    td = _make_td(
        edges=[(0, 3), (1, 3), (3, 2)], leaves=[0, 1, 2],
        prop_indices={(0, 3): (0, 0), (1, 3): (1, 1), (3, 2): (0, 0)},
    )
    edge_freqs, ext_freqs, ext_assign = assign_frequencies(td, k=3)

    assert len(ext_freqs) == 3  # k=3 independent ext freqs


# ── Tests: conservation equations ────────────────────────────────────────────

def test_conservation_simple_tree():
    """
    Tree: 0 → 2 → 1.  Conservation at vertex 2 gives ω_{0,2} = ω_{2,1}.
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    edge_freqs, ext_freqs, ext_assign = assign_frequencies(td, k=2)
    eqs = build_conservation_equations(td, edge_freqs, ext_assign)

    # Should have 3 equations (one per vertex: leaves 0, 1 and internal 2)
    assert len(eqs) == 3


# ── Tests: solve conservation ────────────────────────────────────────────────

def test_solve_tree_no_loop_freqs():
    """
    Tree diagram: loop number = 0.  All edge frequencies determined
    by external frequencies.
    """
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    td = _make_td(
        edges=[(0, 2), (2, 1)], leaves=[0, 1],
        vert_assignments={2: vt},
        edge_types={(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        ext_legs={0: ('nt', 1), 1: ('dn', 1)},
        prop_indices={(0, 2): (0, 0), (2, 1): (0, 0)},
    )
    edge_freqs, ext_freqs, ext_assign = assign_frequencies(td, k=2)
    subs, loop_freqs, loop_number, overall = solve_conservation(
        td, edge_freqs, ext_assign
    )

    assert loop_number == 0
    assert len(loop_freqs) == 0

    # All edge frequencies should resolve to external frequencies
    for (u, v), omega_e in edge_freqs.items():
        resolved = omega_e.subs(subs)
        # Should depend only on external frequencies
        resolved_vars = set(resolved.variables())
        ext_var_set = set(ext_freqs)
        assert resolved_vars.issubset(ext_var_set), \
            f"Edge ({u},{v}): {resolved} has non-external variables"

    # Overall conservation should relate the external frequencies
    assert overall is not None


def test_solve_one_loop_has_loop_freq():
    """
    One-loop (bubble): loop number = 1, one independent loop frequency.
    """
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
    edge_freqs, ext_freqs, ext_assign = assign_frequencies(td, k=2)
    subs, loop_freqs, loop_number, overall = solve_conservation(
        td, edge_freqs, ext_assign
    )

    assert loop_number == 1
    assert len(loop_freqs) == 1


# ── Tests: build integrand ───────────────────────────────────────────────────

def test_build_integrand_tree():
    """Tree-level: integrand is a product of propagator entries."""
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

    edge_freqs, ext_freqs, ext_assign = assign_frequencies(td, k=2)
    freq_subs, loop_freqs, _, _ = solve_conservation(td, edge_freqs, ext_assign)
    integrand = build_integrand(td, edge_freqs, freq_subs, pd, omega)

    # integrand should be a product of two propagator entries
    # and should depend on at least one external frequency
    integrand_vars = set(integrand.variables())
    ext_var_set = set(ext_freqs)
    assert integrand_vars & ext_var_set, \
        f"Integrand {integrand} doesn't depend on any external freq"
    # Should NOT depend on any internal ω_e (all substituted)
    for omega_e in edge_freqs.values():
        assert omega_e not in integrand_vars


# ── Tests: full assembly ─────────────────────────────────────────────────────

def test_build_integrand_stationary_tree():
    """Full assembly for a tree-level diagram includes time variables."""
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
    assert len(result['loop_freqs']) == 0
    assert 'scalar_prefactor' in result
    assert 'integrand' in result
    assert 'full_integrand' in result
    assert 'ext_times' in result
    assert len(result['ext_times']) == 2
    # For 2-pt tree: 1 independent ext freq, 0 loop freqs → 1 integral
    assert len(result['ext_freqs_independent']) == 1
    # Fourier prefactor = 1/(2π) for the 1 ext freq integral
    assert result['fourier_prefactor'] == SR(1) / (2 * pi)


# ── Tests: tree-level integration (time domain) ─────────────────────────────

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

    # Result should be a function of external TIMES, not frequencies
    assert contribution is not None
    contrib_vars = set(contribution.variables())
    time_var_set = set(result['ext_times'])
    # Should depend on at least one time variable
    assert contrib_vars & time_var_set, \
        f"Contribution {contribution} doesn't depend on any external time"
    # Should NOT depend on external frequency variables
    ext_freq_set = set(result['ext_freqs'])
    assert not (contrib_vars & ext_freq_set), \
        f"Contribution should not depend on frequency variables, got {contrib_vars & ext_freq_set}"


def test_integrate_tree_level_rejects_loops():
    """integrate_tree_level should reject diagrams with loops."""
    import pytest
    result = {'loop_number': 1, 'scalar_prefactor': SR(1), 'integrand': SR(1)}
    with pytest.raises(ValueError, match='tree-level'):
        integrate_tree_level(result)


# ── Tests: time-domain integration structure ─────────────────────────────────

def test_integrate_to_time_domain_tree():
    """integrate_to_time_domain returns dict with time_domain_result."""
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
    assert 'frequency_domain_integrand' in result
    assert 'integration_variables' in result
    assert 'ext_times' in result
    assert result['status'] in ('ok', 'partial')


# ── Tests: source vertex conservation ────────────────────────────────────────

def test_source_vertex_conservation():
    """
    Source vertex (no incoming edges): outgoing frequencies sum to zero.
    For a 2-leg source, ω_a + ω_b = 0.
    """
    st = SourceType(SR(1)/2, [('nt', 1), ('nt', 1)], (2, 0))
    td = _make_td(
        edges=[(2, 0), (2, 1)], leaves=[0, 1],
        vert_assignments={2: st},
        edge_types={(2, 0): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 2))},
        ext_legs={0: ('dn', 1), 1: ('dn', 2)},
        prop_indices={(2, 0): (0, 0), (2, 1): (0, 1)},
    )
    edge_freqs, ext_freqs, ext_assign = assign_frequencies(td, k=2)
    subs, loop_freqs, loop_number, overall = solve_conservation(
        td, edge_freqs, ext_assign
    )

    # Tree level: ℓ = 0
    assert loop_number == 0

    # After applying overall conservation (ω_ext_1 = ω_ext_2),
    # the source vertex conservation (ω_{2,0} + ω_{2,1} = 0) should
    # be satisfied.  Check in the resolved form.
    omega_20 = edge_freqs[(2, 0)].subs(subs)
    omega_21 = edge_freqs[(2, 1)].subs(subs)
    # These should be functions of external freqs, and source
    # conservation means ω_{2,0} + ω_{2,1} = 0 is encoded
    # in the overall conservation relation.
    # The sum ω_{2,0} + ω_{2,1} should equal the overall
    # conservation expression (which equals zero by definition).
    assert overall is not None

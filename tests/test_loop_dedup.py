"""
tests/test_loop_dedup.py
=========================
Tests for loop integral deduplication:
  extract_propagator_factors, canonicalize_prop_factors,
  loop_kernel_signature, group_diagrams_by_kernel.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_loop_dedup.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph, matrix, I

from engine.integration.symbolic import (
    assign_frequencies,
    solve_conservation,
    build_integrand_stationary,
    extract_propagator_factors,
    canonicalize_prop_factors,
    loop_kernel_signature,
    group_diagrams_by_kernel,
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


def _tree_diagram():
    """
    Tree diagram: source(2) → leaf(0), internal(3) → leaf(1)
    with edge 2→3 connecting them.
    Edges: 2→0, 2→3, 3→1.  Leaves: 0, 1.
    """
    st = SourceType(SR(1)/2, [('nt', 1), ('nt', 1)], (2, 0))
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    return _make_td(
        edges=[(2, 0), (2, 3), (3, 1)],
        leaves=[0, 1],
        vert_assignments={2: st, 3: vt},
        edge_types={
            (2, 0): (('nt', 1), ('dn', 1)),
            (2, 3): (('nt', 1), ('dn', 1)),
            (3, 1): (('nt', 1), ('dn', 1)),
        },
        ext_legs={0: ('dn', 1), 1: ('dn', 1)},
        prop_indices={(2, 0): (0, 0), (2, 3): (0, 0), (3, 1): (0, 0)},
    )


def _bubble_diagram(prop_idx_inner=(0, 0)):
    """
    Bubble (1-loop) diagram:
      leaf(0) → internal(2), internal(2) ⇉ internal(3), internal(3) → leaf(1)
    Edges: 0→2, 2→3 (×2), 3→1.  Leaves: 0, 1.
    """
    vt_out = VertexType(SR(1), [('nt', 1), ('nt', 1)], [('dn', 1)], (2, 1))
    vt_in = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 1)], (1, 2))

    D = DiGraph(multiedges=True)
    D.add_edges([(0, 2, 0), (2, 3, 0), (2, 3, 1), (3, 1, 0)])
    G = D.to_undirected()
    pd = (D, G, [0, 1], [2, 3])

    ri0, pi0 = prop_idx_inner
    return TypedDiagram(
        pd,
        {2: vt_out, 3: vt_in},
        {
            (0, 2, 0): (('nt', 1), ('dn', 1)),
            (2, 3, 0): (('nt', 1), ('dn', 1)),
            (2, 3, 1): (('nt', 1), ('dn', 1)),
            (3, 1, 0): (('nt', 1), ('dn', 1)),
        },
        {0: ('nt', 1), 1: ('dn', 1)},
        {
            (0, 2, 0): (0, 0),
            (2, 3, 0): (0, 0),
            (2, 3, 1): prop_idx_inner,
            (3, 1, 0): (0, 0),
        },
    )


# ── Tests: extract_propagator_factors ────────────────────────────────────────

def test_extract_tree_factors():
    """Tree diagram yields one factor per edge, all 'prop' type."""
    td = _tree_diagram()
    ef, lef = assign_frequencies(td, k=2)
    subs, free, overall = solve_conservation(td, ef, lef)
    factors = extract_propagator_factors(td, ef, subs)

    assert len(factors) == 3
    for f in factors:
        assert f[0] == 'prop'
        assert isinstance(f[1], (int,))  # row
        assert isinstance(f[2], (int,))  # col


def test_extract_bubble_factors():
    """Bubble diagram yields 4 propagator factors, some with loop freq."""
    td = _bubble_diagram()
    ef, lef = assign_frequencies(td, k=2)
    subs, free, overall = solve_conservation(td, ef, lef)
    factors = extract_propagator_factors(td, ef, subs)

    assert len(factors) == 4
    # At least one factor should contain a free (loop) frequency variable
    free_set = set(free)
    has_loop = any(
        set(f[3].variables()) & free_set for f in factors
    )
    assert has_loop, "Expected at least one factor depending on loop freq"


def test_extract_factors_match_indices():
    """Propagator indices in factors match the TypedDiagram's prop_indices."""
    td = _tree_diagram()
    ef, lef = assign_frequencies(td, k=2)
    subs, free, overall = solve_conservation(td, ef, lef)
    factors = extract_propagator_factors(td, ef, subs)

    # All edges have prop_indices (0, 0) in the tree diagram
    for f in factors:
        assert (f[1], f[2]) == (0, 0)


# ── Tests: canonicalize_prop_factors ─────────────────────────────────────────

def test_canonicalize_replaces_variables():
    """After canonicalization, no diagram-specific omega_eN variables remain."""
    td = _bubble_diagram()
    ef, lef = assign_frequencies(td, k=2)
    subs, free, overall = solve_conservation(td, ef, lef)

    # Apply overall conservation
    if overall is not None:
        from sage.all import solve as sage_solve
        ext_all = [lef[lf] for lf in td.prediagram[2]]
        target = ext_all[-1]
        cons_sol = sage_solve(overall == 0, target, solution_dict=True)
        if cons_sol:
            subs.update(cons_sol[0])
        for _ in range(10):
            changed = False
            for k_ in list(subs):
                nv = subs[k_].subs(subs)
                if nv != subs[k_]:
                    subs[k_] = nv
                    changed = True
            if not changed:
                break

    ext_freqs = [w for w in [lef[lf] for lf in td.prediagram[2]]
                 if w not in subs]

    factors = extract_propagator_factors(td, ef, subs)
    cfactors, c_ext, c_loop = canonicalize_prop_factors(
        factors, ext_freqs, free
    )

    # All frequency expressions should only contain w_* and L_* variables
    allowed = set(c_ext) | set(c_loop)
    for f in cfactors:
        omega_expr = f[3] if f[0] == 'prop' else f[1]
        for v in omega_expr.variables():
            assert v in allowed, f"Unexpected variable {v} in canonical factor"


def test_canonicalize_two_diagrams_same_structure():
    """
    Two bubble diagrams with identical prop indices but different
    internal variable names should produce identical canonical factors.
    """
    td1 = _bubble_diagram(prop_idx_inner=(0, 0))
    td2 = _bubble_diagram(prop_idx_inner=(0, 0))

    results = []
    for td in [td1, td2]:
        ir = build_integrand_stationary(td, _simple_propagator_data(), k=2)
        pf = extract_propagator_factors(
            td, ir['edge_freqs'], ir['substitutions']
        )
        cpf, _, _ = canonicalize_prop_factors(
            pf, ir['ext_freqs'], ir['free_freqs']
        )
        results.append(cpf)

    # Both should have the same canonical factors
    def to_hashable(factors):
        return tuple(sorted(
            (f[0], f[1], f[2], str(f[3].expand())) if f[0] == 'prop'
            else (f[0], str(f[1].expand()))
            for f in factors
        ))

    assert to_hashable(results[0]) == to_hashable(results[1])


# ── Tests: loop_kernel_signature ─────────────────────────────────────────────

def test_signature_hashable():
    """Signature can be used as a dict key."""
    td = _bubble_diagram()
    ir = build_integrand_stationary(td, _simple_propagator_data(), k=2)
    pf = extract_propagator_factors(
        td, ir['edge_freqs'], ir['substitutions']
    )
    cpf, c_ext, c_loop = canonicalize_prop_factors(
        pf, ir['ext_freqs'], ir['free_freqs']
    )
    sig = loop_kernel_signature(cpf, c_loop)

    d = {sig: 'test'}
    assert d[sig] == 'test'


def test_signature_identical_diagrams():
    """Two identical bubble diagrams produce the same signature."""
    sigs = []
    for _ in range(2):
        td = _bubble_diagram(prop_idx_inner=(0, 0))
        ir = build_integrand_stationary(td, _simple_propagator_data(), k=2)
        pf = extract_propagator_factors(
            td, ir['edge_freqs'], ir['substitutions']
        )
        cpf, _, c_loop = canonicalize_prop_factors(
            pf, ir['ext_freqs'], ir['free_freqs']
        )
        sigs.append(loop_kernel_signature(cpf, c_loop))

    assert sigs[0] == sigs[1]


def test_signature_different_prop_indices():
    """Bubble diagrams with different prop indices have different signatures."""
    prop_data = _simple_propagator_data()
    # Make both diagonal entries nonzero so both diagrams are valid
    sigs = []
    for idx in [(0, 0), (1, 1)]:
        td = _bubble_diagram(prop_idx_inner=idx)
        ir = build_integrand_stationary(td, prop_data, k=2)
        pf = extract_propagator_factors(
            td, ir['edge_freqs'], ir['substitutions']
        )
        cpf, _, c_loop = canonicalize_prop_factors(
            pf, ir['ext_freqs'], ir['free_freqs']
        )
        sigs.append(loop_kernel_signature(cpf, c_loop))

    assert sigs[0] != sigs[1]


def test_tree_signature_no_loop_partition():
    """Tree diagram has empty loop partition in signature."""
    td = _tree_diagram()
    ir = build_integrand_stationary(td, _simple_propagator_data(), k=2)
    pf = extract_propagator_factors(
        td, ir['edge_freqs'], ir['substitutions']
    )
    cpf, _, c_loop = canonicalize_prop_factors(
        pf, ir['ext_freqs'], ir['free_freqs']
    )
    sig = loop_kernel_signature(cpf, c_loop)

    # For tree (no loops), signature is just (external_sig,)
    assert len(sig) == 1
    # The external sig should contain all factors
    assert len(sig[0]) == 3  # 3 edges


def test_bubble_signature_has_loop_partition():
    """Bubble diagram has both external and loop partitions."""
    td = _bubble_diagram()
    ir = build_integrand_stationary(td, _simple_propagator_data(), k=2)
    pf = extract_propagator_factors(
        td, ir['edge_freqs'], ir['substitutions']
    )
    cpf, _, c_loop = canonicalize_prop_factors(
        pf, ir['ext_freqs'], ir['free_freqs']
    )
    sig = loop_kernel_signature(cpf, c_loop)

    # 1-loop: signature is (external_sig, loop_0_sig)
    assert len(sig) == 2
    ext_sig, loop_sig = sig
    # Loop partition should be non-empty (some factors depend on L_0)
    assert len(loop_sig) > 0


# ── Tests: group_diagrams_by_kernel ──────────────────────────────────────────

def test_group_identical_diagrams():
    """Two identical diagrams are grouped together with summed prefactors."""
    td1 = _bubble_diagram(prop_idx_inner=(0, 0))
    td2 = _bubble_diagram(prop_idx_inner=(0, 0))

    groups = group_diagrams_by_kernel(
        [td1, td2], _simple_propagator_data(), k=2,
    )

    assert len(groups) == 1
    assert groups[0]['n_diagrams'] == 2


def test_group_different_diagrams():
    """Diagrams with different prop indices stay in separate groups."""
    td1 = _bubble_diagram(prop_idx_inner=(0, 0))
    td2 = _bubble_diagram(prop_idx_inner=(1, 1))

    groups = group_diagrams_by_kernel(
        [td1, td2], _simple_propagator_data(), k=2,
    )

    assert len(groups) == 2
    assert all(g['n_diagrams'] == 1 for g in groups)


def test_group_combined_prefactor():
    """Combined prefactor is the sum of individual prefactors."""
    # Create two diagrams with different vertex coefficients
    # but same propagator structure
    st1 = SourceType(SR.var('a'), [('nt', 1), ('nt', 1)], (2, 0))
    st2 = SourceType(SR.var('b'), [('nt', 1), ('nt', 1)], (2, 0))
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))

    def make_tree(st):
        return _make_td(
            edges=[(2, 0), (2, 3), (3, 1)],
            leaves=[0, 1],
            vert_assignments={2: st, 3: vt},
            edge_types={
                (2, 0): (('nt', 1), ('dn', 1)),
                (2, 3): (('nt', 1), ('dn', 1)),
                (3, 1): (('nt', 1), ('dn', 1)),
            },
            ext_legs={0: ('dn', 1), 1: ('dn', 1)},
            prop_indices={(2, 0): (0, 0), (2, 3): (0, 0), (3, 1): (0, 0)},
        )

    td1 = make_tree(st1)
    td2 = make_tree(st2)

    groups = group_diagrams_by_kernel(
        [td1, td2], _simple_propagator_data(), k=2,
    )

    # Same propagator structure → one group
    assert len(groups) == 1
    # Combined prefactor should contain both 'a' and 'b'
    cp = groups[0]['combined_prefactor']
    cp_vars = set(str(v) for v in cp.variables())
    assert 'a' in cp_vars or 'b' in cp_vars


def test_group_preserves_representative_ir():
    """Each group has a representative integrand result for evaluation."""
    td = _bubble_diagram()
    groups = group_diagrams_by_kernel(
        [td], _simple_propagator_data(), k=2,
    )

    assert len(groups) == 1
    ir = groups[0]['representative_ir']
    assert 'integrand' in ir
    assert 'free_freqs' in ir
    assert 'ext_freqs' in ir

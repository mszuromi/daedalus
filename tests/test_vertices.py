"""
tests/test_vertices.py
======================
Tests for engine.core.vertices — vertex decomposition (Build Phase B).

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_vertices.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, PolynomialRing

from engine.core.vertices import (
    VertexType, SourceType, _parse_field_name,
    decompose_sector, extract_vertex_types, extract_source_types,
    available_degrees,
)
from engine.core.field_theory import FieldTheory


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_hawkes():
    """Load the Hawkes model and expand at taylor_order=4."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'simulations')
    sys.path.insert(0, models_dir)
    from hawkes_sage import HAWKES_MODEL
    ft = FieldTheory(HAWKES_MODEL, taylor_order=4)
    ft.expand()
    return ft


def _make_ring_and_poly():
    """Build a small polynomial ring and a test polynomial for unit tests."""
    # 2-population: response fields nt1, nt2; physical fields dn1, dn2
    names = ['nt1', 'nt2', 'dn1', 'dn2']
    R = PolynomialRing(SR, names)
    nt1, nt2, dn1, dn2 = R.gens()
    return R, names, nt1, nt2, dn1, dn2


# ── Tests: _parse_field_name ─────────────────────────────────────────────────

def test_parse_simple():
    assert _parse_field_name('nt1') == ('nt', 1)
    assert _parse_field_name('dn2') == ('dn', 2)
    assert _parse_field_name('vt12') == ('vt', 12)


def test_parse_no_digits():
    assert _parse_field_name('omega') == ('omega', 0)


# ── Tests: decompose_sector ─────────────────────────────────────────────────

def test_single_monomial_vertex():
    """A single (1,2) monomial decomposes into one VertexType."""
    R, names, nt1, nt2, dn1, dn2 = _make_ring_and_poly()
    alpha = SR.var('alpha')
    poly = alpha * nt1 * dn1 * dn2  # bigrade (1,2)
    result = decompose_sector(poly, 2, names)

    assert len(result) == 1
    vt = result[0]
    assert isinstance(vt, VertexType)
    assert vt.bigrade == (1, 2)
    assert vt.out_degree == 1
    assert vt.in_degree == 2
    assert vt.total_degree == 3
    assert vt.coefficient == alpha
    assert sorted(vt.response_legs) == [('nt', 1)]
    assert sorted(vt.physical_legs) == [('dn', 1), ('dn', 2)]


def test_single_monomial_source():
    """A (2,0) monomial decomposes into one SourceType."""
    R, names, nt1, nt2, dn1, dn2 = _make_ring_and_poly()
    beta = SR.var('beta')
    poly = beta * nt1 * nt2  # bigrade (2,0)
    result = decompose_sector(poly, 2, names)

    assert len(result) == 1
    st = result[0]
    assert isinstance(st, SourceType)
    assert st.bigrade == (2, 0)
    assert st.out_degree == 2
    assert st.coefficient == beta


def test_exponent_multiplicity():
    """A monomial with nt1^2 produces two response legs of the same type."""
    R, names, nt1, nt2, dn1, dn2 = _make_ring_and_poly()
    poly = SR(3) * nt1**2 * dn1  # bigrade (2,1)
    result = decompose_sector(poly, 2, names)

    assert len(result) == 1
    vt = result[0]
    assert isinstance(vt, VertexType)
    assert vt.bigrade == (2, 1)
    assert len(vt.response_legs) == 2
    assert vt.response_legs == [('nt', 1), ('nt', 1)]
    assert vt.in_degree == 1
    assert vt.out_degree == 2


def test_multiple_monomials():
    """A sector with two monomials produces two objects."""
    R, names, nt1, nt2, dn1, dn2 = _make_ring_and_poly()
    a, b = SR.var('a'), SR.var('b')
    poly = a * nt1 * dn1 * dn2 + b * nt2 * dn1 * dn2
    result = decompose_sector(poly, 2, names)

    assert len(result) == 2
    assert all(isinstance(r, VertexType) for r in result)
    bigrades = {r.bigrade for r in result}
    assert bigrades == {(1, 2)}


def test_coefficient_extraction():
    """The SR coefficient is exactly what the polynomial ring stores."""
    R, names, nt1, nt2, dn1, dn2 = _make_ring_and_poly()
    tau, w = SR.var('tau'), SR.var('w')
    poly = tau * w * nt1 * dn1  # coefficient is tau*w
    result = decompose_sector(poly, 2, names)

    assert len(result) == 1
    assert bool(result[0].coefficient == tau * w)


# ── Tests: extract from Hawkes model ────────────────────────────────────────

def test_extract_vertex_types_hawkes():
    """Extract vertex types from expanded Hawkes model, check basic properties."""
    ft = _load_hawkes()
    vtypes = extract_vertex_types(ft)

    assert len(vtypes) > 0
    for vt in vtypes:
        assert isinstance(vt, VertexType)
        assert vt.total_degree >= 3
        assert vt.in_degree > 0  # interaction vertices have physical legs


def test_extract_source_types_hawkes():
    """Extract source types from expanded Hawkes noise kernel."""
    ft = _load_hawkes()
    stypes = extract_source_types(ft)

    assert len(stypes) > 0
    for st in stypes:
        assert isinstance(st, SourceType)
        assert st.out_degree >= 2
        assert st.bigrade[1] == 0  # no physical legs


def test_vertex_bigrades_match_sectors():
    """Every bigrade from vertex types should appear in ft.vertices()."""
    ft = _load_hawkes()
    vtypes = extract_vertex_types(ft)
    sector_keys = set(ft.vertices().keys())
    # Remove pure noise-kernel sectors (n_p == 0)
    sector_keys = {k for k in sector_keys if k[1] > 0}

    vertex_bigrades = {vt.bigrade for vt in vtypes}
    assert vertex_bigrades.issubset(sector_keys)


def test_source_bigrades_match_noise():
    """Every bigrade from source types should appear in ft.noise_kernel()."""
    ft = _load_hawkes()
    stypes = extract_source_types(ft)
    noise_keys = set(ft.noise_kernel().keys())

    source_bigrades = {st.bigrade for st in stypes}
    assert source_bigrades.issubset(noise_keys)


def test_available_degrees():
    """Check that available_degrees returns the correct structure."""
    ft = _load_hawkes()
    vtypes = extract_vertex_types(ft)
    stypes = extract_source_types(ft)

    int_degs, src_degs = available_degrees(vtypes, stypes)

    assert isinstance(int_degs, set)
    assert isinstance(src_degs, set)
    # All interaction degrees have in_degree > 0
    for (in_d, out_d) in int_degs:
        assert in_d > 0
    # Source degrees are all >= 2
    for d in src_degs:
        assert d >= 2


def test_total_degree_property():
    """total_degree == in_degree + out_degree for VertexType."""
    ft = _load_hawkes()
    vtypes = extract_vertex_types(ft)
    for vt in vtypes:
        assert vt.total_degree == vt.in_degree + vt.out_degree


def test_empty_theory():
    """A free theory (only (1,1) sector) returns empty vertex/source lists."""
    # Build a trivial model with only a free action (no interaction terms)
    trivial_model = {
        'name': 'free',
        'index_sets': {'pop': [0]},
        'response_fields': [{'name': 'nt', 'indexed': True, 'latex': r'\tilde{n}'}],
        'physical_fields': [{'name': 'dn', 'indexed': True, 'latex': r'\delta n'}],
        'parameters': [],
        'kernels': [],
        'operators': [],
        'functions': [],
        # Free action only: S = nt1 * dn1
        'action': lambda ns: ns.nt[0] * ns.dn[0],
    }
    ft = FieldTheory(trivial_model, taylor_order=4)
    ft.expand()

    vtypes = extract_vertex_types(ft)
    stypes = extract_source_types(ft)

    assert vtypes == []
    assert stypes == []


def test_monomial_count_consistency():
    """Total monomials from vertex types should match the number of monomials
    in the interaction sectors of the polynomial."""
    ft = _load_hawkes()
    vtypes = extract_vertex_types(ft)

    # Count monomials directly from the polynomial sectors
    expected_count = 0
    for (n_t, n_p), poly in ft.vertices().items():
        if n_p == 0:
            continue
        expected_count += len(poly.dict())

    assert len(vtypes) == expected_count

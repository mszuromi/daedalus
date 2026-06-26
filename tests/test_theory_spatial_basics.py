"""
tests/test_theory_spatial_basics.py
===================================
Phase 1 acceptance tests for the spatial field-theory extension
(``docs/spatial_implementation_plan.md`` Phase 1).

Covers the theory-side wiring only — no propagator (Phase 2) or
(t, x) integration (Phase 5):

  * ``physical_field(spatial_dim=...)`` + per-field FieldSpec storage
  * ``.spatial_dim(d)`` convenience bulk-set (D1)
  * ``model['spatial']`` / ``model['boundary']`` / ``model['initial']``
    emission, with sensible defaults
  * the ``Laplacian`` operator symbol is registered and survives
    ``FieldTheory.expand()`` as an inert multiplicative symbol in the
    bilinear inverse-propagator sector
  * the saddle resolves to φ*=0 (Laplacian killed on the uniform
    saddle in all THREE places: action lambda, mf_bg lambda, DAE
    numerical+stability namespaces)
  * boundary/initial/spatial_dim round-trip through the serializer
  * validation: mixed-dim, bad mode, .boundary on a time-only theory
  * non-regression: a time-only theory carries none of the new keys

Run::

    sage -python -m pytest tests/test_theory_spatial_basics.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.theory import TheoryBuilder


# ── Builders ─────────────────────────────────────────────────────
def _allen_cahn_1d():
    """The v1 spatial test theory (1D subcritical Allen-Cahn)."""
    return (
        TheoryBuilder('Allen-Cahn 1D test', n_populations=0)
        .physical_field('phi', spatial_dim=1, description='order parameter')
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('D',   default=1.0, domain='positive')
        .parameter('lam', default=0.1, domain='positive')
        .parameter('T',   default=1.0, domain='positive')
        .set_action_text(
            'phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
        .stability_analysis(True)
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


def _time_only_phi4():
    """The same φ⁴ theory minus the spatial structure — the
    non-regression control."""
    return (
        TheoryBuilder('Time-only phi4 control', n_populations=0)
        .physical_field('phi')
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('lam', default=0.1, domain='positive')
        .parameter('T',   default=1.0, domain='positive')
        .set_action_text('phit*((Dt + mu)*phi + lam*phi^3) - T*phit^2')
        .equation(lhs='(Dt + mu)*phi', rhs='-lam*phi^3')
        .stability_analysis(True)
        .build()
    )


# ── 1.1 / build() model-dict structure ───────────────────────────
def test_spatial_block_emitted():
    m = _allen_cahn_1d()
    assert 'spatial' in m
    assert m['spatial']['dim'] == 1
    # Internal fluctuation name is 'dphi' (auto from natural 'phi').
    assert m['spatial']['fields_with_spatial'] == ['dphi']


def test_laplacian_operator_registered():
    m = _allen_cahn_1d()
    op_names = [o['name'] for o in m['operators']]
    assert 'Dt' in op_names
    assert 'Laplacian' in op_names


def test_per_field_spatial_dim_on_model():
    m = _allen_cahn_1d()
    phi = next(f for f in m['physical_fields'] if f['name'] == 'dphi')
    assert phi.get('spatial_dim') == 1


def test_boundary_initial_defaults():
    m = _allen_cahn_1d()
    assert m['boundary']['mode'] == 'infinite'
    assert m['initial']['mode'] == 'stationary'


def test_spatial_dim_convenience_bulk_set():
    """``.spatial_dim(d)`` sets every declared field AND the default
    for fields declared afterward; explicit per-field wins."""
    b = (TheoryBuilder('bulk', n_populations=0)
         .physical_field('a')              # declared before → retro-set
         .spatial_dim(2)
         .physical_field('b')              # declared after → inherits 2
         .physical_field('c', spatial_dim=0))   # explicit override
    dims = {f.natural_name: f.spatial_dim for f in b.physical_fields}
    assert dims['a'] == 2
    assert dims['b'] == 2
    assert dims['c'] == 0


def test_periodic_boundary_named_length():
    m = (TheoryBuilder('pbc', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D',  default=1.0, domain='positive')
         .parameter('T',  default=1.0, domain='positive')
         .parameter('L',  default=20.0, domain='positive')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .boundary('periodic', length='L')
         .initial('stationary')
         .build())
    assert m['boundary']['mode'] == 'periodic'
    assert m['boundary']['length'] == 'L'


def test_periodic_boundary_inline_length_creates_hidden_param():
    m = (TheoryBuilder('pbc-inline', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D',  default=1.0, domain='positive')
         .parameter('T',  default=1.0, domain='positive')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .boundary('periodic', length=12.0)
         .initial('stationary')
         .build())
    # Inline number → hidden parameter, boundary length rewritten to it.
    hidden = m['boundary']['length']
    assert isinstance(hidden, str)
    pnames = [p['name'] for p in m['parameters']]
    assert hidden in pnames


# ── Validation ────────────────────────────────────────────────────
def test_mixed_dim_rejected():
    with pytest.raises(ValueError, match='mixed spatial dimensions'):
        (TheoryBuilder('mixed', n_populations=0)
         .physical_field('a', spatial_dim=1)
         .physical_field('b', spatial_dim=2)
         .parameter('mu', default=1.0)
         .set_action_text('at*a + bt*b')
         .build())


def test_boundary_on_time_only_rejected():
    with pytest.raises(ValueError, match='no physical field is spatial'):
        (TheoryBuilder('notspatial', n_populations=0)
         .physical_field('phi')
         .parameter('mu', default=1.0)
         .set_action_text('phit*(Dt + mu)*phi')
         .boundary('periodic', length=10.0)
         .build())


def test_bad_boundary_mode_rejected():
    with pytest.raises(ValueError, match="'infinite' and 'periodic'"):
        (TheoryBuilder('badbc', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .boundary('dirichlet'))


def test_bad_initial_mode_rejected():
    with pytest.raises(ValueError, match="only 'stationary'"):
        (TheoryBuilder('badic', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .initial('transient'))


def test_periodic_without_length_rejected():
    with pytest.raises(ValueError, match="'length' is required"):
        (TheoryBuilder('nolen', n_populations=0)
         .physical_field('phi', spatial_dim=1)
         .boundary('periodic'))


# ── 1.2 / 1.3 — expand() + Laplacian inert + saddle resolves ──────
def test_expand_runs_and_laplacian_in_bilinear_sector():
    from engine.core.field_theory import FieldTheory
    m = _allen_cahn_1d()
    ft = FieldTheory(m, taylor_order=2)
    ft.expand()   # raises if the MF saddle does not resolve to φ*=0
    bilinear = ft._by_tp.get((1, 1))
    assert bilinear is not None
    s = str(bilinear)
    # Inverse-propagator structure: Dt + mu - D*Laplacian (+ saddle
    # term that vanishes at φ*=0).
    assert 'Laplacian' in s
    assert 'Dt' in s
    assert 'mu' in s


# ── Non-regression: time-only theory is untouched ─────────────────
def test_time_only_theory_has_no_spatial_keys():
    m = _time_only_phi4()
    assert 'spatial' not in m
    assert 'boundary' not in m
    assert 'initial' not in m
    op_names = [o['name'] for o in m['operators']]
    assert 'Laplacian' not in op_names


def test_time_only_phi4_still_expands():
    from engine.core.field_theory import FieldTheory
    m = _time_only_phi4()
    ft = FieldTheory(m, taylor_order=2)
    ft.expand()
    bilinear = ft._by_tp.get((1, 1))
    assert bilinear is not None and 'Laplacian' not in str(bilinear)


# ── 1.5 — serializer round-trip ───────────────────────────────────
def test_serializer_roundtrip_spatial_boundary_initial():
    from api.theory_serialize import (
        render_theory_file, load_spec_from_file,
    )
    spec = {
        'name': 'RT spatial',
        'populations': [],
        'physical_fields': [{
            'name': 'phi', 'natural_name': 'phi', 'indexed': True,
            'spatial_dim': 1, 'description': 'order parameter'}],
        'response_fields': [], 'parameters': [], 'kernels': [],
        'functions': [], 'cgf_terms': [], 'mf_equations': [],
        'equations': [],
        'action_text': 'phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2',
        'boundary': {'mode': 'periodic', 'length': 'L'},
        'initial': {'mode': 'stationary'},
        'stability_analysis': True,
    }
    src = render_theory_file(spec)
    with tempfile.NamedTemporaryFile(
            'w', suffix='.theory.py', delete=False) as f:
        f.write(src)
        path = f.name
    try:
        spec2 = load_spec_from_file(path)
    finally:
        os.unlink(path)

    assert spec2['boundary'] == {'mode': 'periodic', 'length': 'L'}
    assert spec2['initial'] == {'mode': 'stationary'}
    phi = next(f for f in spec2['physical_fields'] if f['name'] == 'phi')
    assert phi.get('spatial_dim') == 1


def test_serializer_roundtrip_operator_ir_and_dyson():
    """operator_ir (LOAD-BEARING for KPZ/Burgers/Model B derivative
    vertices) and the Dyson policy + reference_diffusion must round-trip
    render -> load -> re-render -> build.  Regression for the bug where
    .operator_ir() was silently dropped, changing the physics of a
    re-saved derivative theory."""
    from api.theory_serialize import (
        render_theory_file, load_spec_from_file, save_theory_to_file,
    )
    import importlib.util
    spec = {
        'name': 'RT opir dyson',
        'populations': [],
        'physical_fields': [
            {'name': 'a', 'natural_name': 'a', 'spatial_dim': 1},
            {'name': 'b', 'natural_name': 'b', 'spatial_dim': 1}],
        'response_fields': [], 'kernels': [], 'functions': [],
        'cgf_terms': [], 'mf_equations': [],
        'parameters': [
            {'name': 'mu', 'default': 1.0, 'domain': 'positive'},
            {'name': 'D', 'default': 1.0, 'domain': 'positive'},
            {'name': 'g', 'default': 0.3, 'domain': 'real'},
            {'name': 'T', 'default': 1.0, 'domain': 'positive'}],
        'equations': [
            {'lhs': '(Dt+mu-D*Laplacian)*a', 'rhs': '0', 'population': None},
            {'lhs': '(Dt+mu-D*Laplacian)*b', 'rhs': '0', 'population': None}],
        'action_text': ('at*(Dt(a)+mu*a-D*Lap(a)+g*a^2) - T*at^2'
                        ' + bt*(Dt(b)+mu*b-D*Lap(b)) - T*bt^2'),
        'operator_ir': True,
        'boundary': {'mode': 'infinite'}, 'initial': {'mode': 'stationary'},
        'dyson': {'mode': 'fixed', 'order': 3},
        'reference_diffusion': 1.0,
    }
    src = render_theory_file(spec)
    assert '.operator_ir()' in src
    assert '.dyson_order(3)' in src
    assert '.reference_diffusion(1.0)' in src

    d = tempfile.mkdtemp()
    path = os.path.join(d, 'rt_opir.theory.py')
    save_theory_to_file(spec, path)
    # parse back: every property survives
    spec2 = load_spec_from_file(path)
    assert spec2.get('operator_ir') is True
    assert spec2.get('dyson') == {'mode': 'fixed', 'order': 3}
    assert spec2.get('reference_diffusion') == 1.0
    # idempotent re-render
    assert '.operator_ir()' in render_theory_file(spec2)
    # the generated file builds and the model carries operator_ir
    sm = importlib.util.spec_from_file_location('rt_opir', path)
    mod = importlib.util.module_from_spec(sm)
    sm.loader.exec_module(mod)
    model = mod.build()
    assert model.get('operator_ir') is True


def test_serializer_spatial_dim_convenience_roundtrip():
    """A hand-written ``.spatial_dim(d)`` call (convenience form)
    parses back to per-field spatial_dim."""
    from api.theory_serialize import load_spec_from_file
    src = '''"""Convenience spatial_dim round-trip."""
from api.theory import TheoryBuilder


def build():
    return (
        TheoryBuilder('conv', n_populations=0)
        .spatial_dim(1)
        .physical_field('phi')
        .parameter('mu', default=1.0, domain='positive')
        .set_action_text('phit*((Dt + mu)*phi)')
        .build()
    )


DEFAULT_FUNDAMENTAL = {}
METADATA = {}
'''
    with tempfile.NamedTemporaryFile(
            'w', suffix='.theory.py', delete=False) as f:
        f.write(src)
        path = f.name
    try:
        spec = load_spec_from_file(path)
    finally:
        os.unlink(path)
    phi = next(f for f in spec['physical_fields'] if f['name'] == 'phi')
    assert phi.get('spatial_dim') == 1


# ── Reserved-name validation (scoped hard error, incl. x) ──────────
def _spatial_phi_builder(field_name='phi', extra_param=None):
    """Minimal spatial builder; field/param names overridable to probe
    the reserved-name guard."""
    b = (TheoryBuilder('rn', n_populations=0)
         .physical_field(field_name, spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive'))
    if extra_param:
        b = b.parameter(extra_param, default=1.0, domain='positive')
    ft = field_name
    return (b.set_action_text(f'{ft}t*((Dt + mu - D*Laplacian)*{ft}) - {ft}t^2')
            .boundary('infinite'))


def test_reserved_spatial_field_x_rejected():
    with pytest.raises(ValueError, match=r"'x' is reserved \(spatial"):
        _spatial_phi_builder(field_name='x').build()


def test_reserved_spatial_param_k_rejected():
    with pytest.raises(ValueError, match=r"'k' is reserved \(spatial"):
        _spatial_phi_builder(extra_param='k').build()


def test_reserved_global_param_t_rejected():
    # 't' is reserved in EVERY theory (the Fourier time variable).
    with pytest.raises(ValueError, match=r"'t' is reserved \(all"):
        (TheoryBuilder('rn2', n_populations=0)
         .physical_field('phi')
         .parameter('t', default=1.0, domain='positive')
         .set_action_text('phit*((Dt + t)*phi) - phit^2')
         .build())


def test_x_allowed_in_time_only_theory():
    # The conventional time-only field name 'x' must still build.
    model = (TheoryBuilder('rn3', n_populations=0)
             .physical_field('x')
             .parameter('mu', default=1.0, domain='positive')
             .set_action_text('xt*((Dt + mu)*x) - xt^2')
             .build())
    assert any((f.get('natural_name') or f['name']) == 'x'
               for f in model['physical_fields'])
    assert 'spatial' not in model

"""
tests/test_theory_builder_split.py
==================================
Pins the TheoryBuilder split (docs/theory_builder_split_plan.md, step 1):

  * the forward builders ``TemporalTheoryBuilder`` / ``SpatialTheoryBuilder``
    produce model dicts IDENTICAL (on the structured keys) to the back-compat
    ``TheoryBuilder`` shim — i.e. the split changed NO model schema / routing;
  * clean per-domain API: ``SpatialTheoryBuilder`` lacks the temporal methods,
    ``TemporalTheoryBuilder`` lacks the spatial methods, the base has neither;
  * domain guards: a ``TemporalTheoryBuilder`` rejects a ``spatial_dim>0`` field,
    a ``SpatialTheoryBuilder`` requires at least one spatial field.

Run:  sage -python -m pytest tests/test_theory_builder_split.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.theory import (                       # noqa: E402
    TheoryBuilder, SpatialTheoryBuilder, TemporalTheoryBuilder,
    _BaseTheoryBuilder,
)

# Structured (comparable) model keys — excludes the lambda hooks (action,
# phi_concrete) which are not == across builds.
_CMP = ['name', 'spatial', 'boundary', 'initial', 'operators', 'parameters',
        'physical_fields', 'response_fields', 'equations', 'populations',
        'index_sets', 'operator_ir', 'stability_analysis']


def _spatial(BB, **kw):
    return (BB('lin_diff', **kw)
            .physical_field('phi', spatial_dim=1)
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .parameter('T', default=1.0, domain='positive')
            .set_action_text('phit*((Dt+mu-D*Laplacian)*phi) - T*phit^2')
            .equation(lhs='(Dt+mu-D*Laplacian)*phi', rhs='0')
            .boundary('infinite').initial('stationary').build())


def _temporal(BB, **kw):
    return (BB('ou', **kw)
            .physical_field('x')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('T', default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu)*x) - T*xt^2')
            .equation(lhs='(Dt+mu)*x', rhs='0').build())


def test_spatial_forward_builder_equals_shim():
    """SpatialTheoryBuilder(...).build() == TheoryBuilder(...).build() on every
    structured key (the split is purely an authoring-layer change)."""
    shim = _spatial(TheoryBuilder, n_populations=0)
    fwd = _spatial(SpatialTheoryBuilder)
    for k in _CMP:
        assert repr(shim.get(k)) == repr(fwd.get(k)), f'spatial model mismatch in {k!r}'
    assert fwd['spatial']['dim'] == 1


def test_temporal_forward_builder_equals_shim():
    shim = _temporal(TheoryBuilder)
    fwd = _temporal(TemporalTheoryBuilder)
    for k in _CMP:
        assert repr(shim.get(k)) == repr(fwd.get(k)), f'temporal model mismatch in {k!r}'
    assert 'spatial' not in shim and 'spatial' not in fwd


def test_api_separation():
    for m in ('boundary', 'initial', 'spatial_dim'):
        assert not hasattr(TemporalTheoryBuilder, m), \
            f'TemporalTheoryBuilder should not expose spatial method {m!r}'
    for m in ('markovianize', 'declare_cgf_term', 'correlated_noise', 'kernel',
              'define_kernel', 'use_synaptic_kernel', 'add_gtas_noise', 'population'):
        assert not hasattr(SpatialTheoryBuilder, m), \
            f'SpatialTheoryBuilder should not expose temporal method {m!r}'
    # the back-compat shim keeps BOTH method sets
    assert hasattr(TheoryBuilder, 'boundary') and hasattr(TheoryBuilder, 'markovianize')
    # the shared base owns neither domain's public methods
    assert not hasattr(_BaseTheoryBuilder, 'boundary')
    assert not hasattr(_BaseTheoryBuilder, 'markovianize')
    # ...but the base DOES build (autopop no longer needs the relocated population())
    assert hasattr(_BaseTheoryBuilder, 'build') and hasattr(_BaseTheoryBuilder, '_inject_autopop')


def test_temporal_rejects_spatial_field():
    with pytest.raises(ValueError, match='spatial_dim'):
        (TemporalTheoryBuilder('t').physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('phit*((Dt+mu)*phi) - T*phit^2')
         .equation(lhs='(Dt+mu)*phi', rhs='0').build())


def test_spatial_requires_spatial_field():
    with pytest.raises(ValueError, match='no physical field is spatial'):
        (SpatialTheoryBuilder('s').physical_field('phi')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('phit*((Dt+mu)*phi) - T*phit^2')
         .equation(lhs='(Dt+mu)*phi', rhs='0').build())


def test_serializer_phase1_round_trip():
    """Serializer Phase 1: render_theory_file emits the domain-specific forward
    builder (Spatial for a spatial spec, Temporal otherwise), and
    load_spec_from_file accepts the new constructor names."""
    import os
    import shutil
    import tempfile
    from pipeline.theory_serialize import load_spec_from_file, render_theory_file

    root = os.path.join(os.path.dirname(__file__), '..')

    def emit(rel):
        return render_theory_file(load_spec_from_file(os.path.join(root, rel)))

    sp = emit('theories/kpz_1d.theory.py')               # spatial fixture
    assert 'from pipeline.theory import SpatialTheoryBuilder' in sp
    assert 'SpatialTheoryBuilder(' in sp

    tp = emit('theories/linear_hawkes.theory.py')        # temporal fixture
    assert 'from pipeline.theory import TemporalTheoryBuilder' in tp
    assert 'TemporalTheoryBuilder(' in tp

    # the loader round-trips the rendered (new-name) source
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'rt.theory.py')
        with open(p, 'w') as fh:
            fh.write(sp)
        spec2 = load_spec_from_file(p)
        assert spec2['name']
        assert any(f.get('spatial_dim') for f in spec2['physical_fields'])
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── Dyson policy authoring (D-4, docs/dyson_duhamel_integration_plan.md) ──

def _spatial_dyson(**dyson_calls):
    """Spatial fixture with optional .dyson_order(N)/.reference_diffusion(D0)
    chained before .build()."""
    b = (SpatialTheoryBuilder('lin_diff')
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('phit*((Dt+mu-D*Laplacian)*phi) - T*phit^2')
         .equation(lhs='(Dt+mu-D*Laplacian)*phi', rhs='0'))
    if 'order' in dyson_calls:
        b = b.dyson_order(dyson_calls['order'])
    if 'D0' in dyson_calls:
        b = b.reference_diffusion(dyson_calls['D0'])
    return b.build()


def test_dyson_policy_lands_in_model():
    """.dyson_order(2) + .reference_diffusion(0.5) land in model['spatial'];
    never calling them ⇒ {'mode': 'off'} and no reference_diffusion key."""
    m = _spatial_dyson(order=2, D0=0.5)
    assert m['spatial']['dyson'] == {'mode': 'fixed', 'order': 2}
    assert m['spatial']['reference_diffusion'] == 0.5

    m0 = _spatial_dyson()
    assert m0['spatial']['dyson'] == {'mode': 'off'}
    assert 'reference_diffusion' not in m0['spatial']


def test_dyson_policy_validation():
    b = SpatialTheoryBuilder('s')
    # v2+ policies raise NotImplementedError naming the mode
    for v2_mode in ('auto', 'adaptive', 'resum'):
        with pytest.raises(NotImplementedError, match=v2_mode):
            b.dyson(mode=v2_mode)
    # unknown mode is a plain ValueError
    with pytest.raises(ValueError, match='unrecognized'):
        b.dyson(mode='bogus')
    # mode='fixed' requires an int order >= 0
    with pytest.raises(ValueError, match='order'):
        b.dyson_order(-1)
    with pytest.raises(ValueError, match='order'):
        b.dyson(mode='fixed')
    # reference_diffusion must be > 0
    with pytest.raises(ValueError, match='D0'):
        b.reference_diffusion(0.0)
    with pytest.raises(ValueError, match='D0'):
        b.reference_diffusion(-1.0)


def test_dyson_methods_absent_on_temporal_builder():
    """The Dyson surface is spatial-only — TemporalTheoryBuilder doesn't
    expose it (so calling it is an AttributeError)."""
    for m in ('dyson', 'dyson_order', 'reference_diffusion'):
        assert not hasattr(TemporalTheoryBuilder, m), \
            f'TemporalTheoryBuilder should not expose spatial method {m!r}'
        assert not hasattr(_BaseTheoryBuilder, m)
    assert hasattr(SpatialTheoryBuilder, 'dyson_order')
    assert hasattr(TheoryBuilder, 'dyson_order')


def test_dyson_on_non_spatial_shim_raises():
    """The back-compat TheoryBuilder exposes .dyson_order, but build()
    rejects the policy when no physical field is spatial (mirrors the
    .boundary()-on-non-spatial validation)."""
    with pytest.raises(ValueError, match=r'\.dyson\(\)'):
        (TheoryBuilder('ou')
         .physical_field('x')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('xt*((Dt+mu)*x) - T*xt^2')
         .equation(lhs='(Dt+mu)*x', rhs='0')
         .dyson_order(2).build())


def test_serializer_dyson_round_trip():
    """Serializer D-4: spec['dyson'] / spec['reference_diffusion'] render as
    .dyson_order(N) / .reference_diffusion(X); load_spec_from_file recovers
    both; and the rendered file BUILDS with the policy in model['spatial']."""
    import shutil
    import tempfile
    from pipeline.theory_serialize import load_spec_from_file, render_theory_file

    spec = {
        'name': 'lin_diff_dyson',
        'physical_fields': [{'name': 'phi', 'spatial_dim': 1}],
        'parameters': [
            {'name': 'mu', 'default': 1.0, 'domain': 'positive'},
            {'name': 'D',  'default': 1.0, 'domain': 'positive'},
            {'name': 'T',  'default': 1.0, 'domain': 'positive'},
        ],
        'action_text': 'phit*((Dt+mu-D*Laplacian)*phi) - T*phit^2',
        'equations': [{'lhs': '(Dt+mu-D*Laplacian)*phi', 'rhs': '0',
                       'population': None}],
        'boundary': {'mode': 'infinite'},
        'initial': {'mode': 'stationary'},
        'dyson': {'mode': 'fixed', 'order': 2},
        'reference_diffusion': 0.5,
    }
    src = render_theory_file(spec)
    assert '.dyson_order(2)' in src
    assert '.reference_diffusion(0.5)' in src

    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'dy.theory.py')
        with open(p, 'w') as fh:
            fh.write(src)
        spec2 = load_spec_from_file(p)
        assert spec2['dyson'] == {'mode': 'fixed', 'order': 2}
        assert spec2['reference_diffusion'] == 0.5
        # the rendered file BUILDS (exec + build()) with the policy wired
        ns: dict = {}
        exec(compile(src, p, 'exec'), ns)
        model = ns['build']()
        assert model['spatial']['dyson']['order'] == 2
        assert model['spatial']['reference_diffusion'] == 0.5
    finally:
        shutil.rmtree(d, ignore_errors=True)

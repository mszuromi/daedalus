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

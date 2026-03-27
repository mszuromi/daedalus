"""Tests for msrjd.core.cache — PipelineCache round-trip serialization."""

import os
import tempfile

import pytest
from sage.all import SR, DiGraph

from msrjd.core.cache import PipelineCache
from msrjd.core.vertices import VertexType, SourceType
from msrjd.diagrams.type_assignment import TypedDiagram


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_typed_diagram():
    """Build a minimal TypedDiagram for testing."""
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1), ('dn', 1)], (1, 2))
    D = DiGraph(multiedges=True)
    D.add_edges([(0, 2, 0), (2, 3, 0), (2, 3, 1), (3, 1, 0)])
    G = D.to_undirected()
    pd = (D, G, [0, 1], [2, 3])
    return TypedDiagram(
        pd, {2: vt, 3: vt},
        {(0, 2, 0): (('nt', 1), ('dn', 1)),
         (2, 3, 0): (('nt', 1), ('dn', 1)),
         (2, 3, 1): (('nt', 1), ('dn', 1)),
         (3, 1, 0): (('nt', 1), ('dn', 1))},
        {0: ('nt', 1), 1: ('dn', 1)},
        {(0, 2, 0): (0, 0), (2, 3, 0): (0, 0),
         (2, 3, 1): (0, 0), (3, 1, 0): (0, 0)},
    )


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, 'test_cache')


# ── Tests ────────────────────────────────────────────────────────────────

def test_save_load_roundtrip(cache_dir):
    cache = PipelineCache(cache_dir)
    td = _make_typed_diagram()
    cache.save('unique_typed', [td], k=2, loop_order=1)
    loaded = cache.load('unique_typed', k=2, loop_order=1)
    assert len(loaded) == 1
    assert len(loaded[0].prediagram[0].vertices()) == 4


def test_exists(cache_dir):
    cache = PipelineCache(cache_dir)
    assert not cache.exists('prediagrams', k=2, loop_order=0)
    cache.save('prediagrams', {'data': 42}, k=2, loop_order=0)
    assert cache.exists('prediagrams', k=2, loop_order=0)
    assert not cache.exists('prediagrams', k=2, loop_order=1)


def test_load_missing_raises(cache_dir):
    cache = PipelineCache(cache_dir)
    with pytest.raises(FileNotFoundError):
        cache.load('nonexistent', k=1, loop_order=0)


def test_get_or_compute_cache_hit(cache_dir):
    cache = PipelineCache(cache_dir)
    cache.save('result', 'cached_value', k=2, loop_order=0)
    call_count = [0]
    def fn():
        call_count[0] += 1
        return 'computed'
    val = cache.get_or_compute('result', fn, k=2, loop_order=0)
    assert val == 'cached_value'
    assert call_count[0] == 0


def test_get_or_compute_cache_miss(cache_dir):
    cache = PipelineCache(cache_dir)
    val = cache.get_or_compute('result', lambda: 'computed', k=2, loop_order=0)
    assert val == 'computed'
    # Should now be cached.
    assert cache.exists('result', k=2, loop_order=0)


def test_clear_single_entry(cache_dir):
    cache = PipelineCache(cache_dir)
    cache.save('a', 1, k=2, loop_order=0)
    cache.save('b', 2, k=2, loop_order=0)
    cache.clear('a', k=2, loop_order=0)
    assert not cache.exists('a', k=2, loop_order=0)
    assert cache.exists('b', k=2, loop_order=0)


def test_clear_all(cache_dir):
    cache = PipelineCache(cache_dir)
    cache.save('a', 1, k=2, loop_order=0)
    cache.save('b', 2, k=2, loop_order=1)
    cache.clear()
    assert not os.path.isdir(cache_dir)


def test_manifest(cache_dir):
    cache = PipelineCache(cache_dir)
    cache.save('prediagrams', 'pd', k=2, loop_order=0)
    cache.save('typed', 'td', k=2, loop_order=1)
    entries = cache.list_cached()
    assert len(entries) == 2
    keys = {e['key'] for e in entries}
    assert 'prediagrams_k2_l0' in keys
    assert 'typed_k2_l1' in keys


def test_vertex_type_roundtrip(cache_dir):
    cache = PipelineCache(cache_dir)
    alpha = SR.var('alpha')
    vt = VertexType(alpha**2, [('nt', 1), ('nt', 2)], [('dn', 1)], (2, 1))
    cache.save('vtypes', [vt])
    loaded = cache.load('vtypes')
    assert len(loaded) == 1
    assert loaded[0].bigrade == (2, 1)
    assert loaded[0].coefficient == alpha**2
    assert loaded[0].response_legs == [('nt', 1), ('nt', 2)]


def test_source_type_roundtrip(cache_dir):
    cache = PipelineCache(cache_dir)
    st = SourceType(SR(1)/2, [('nt', 1), ('nt', 1)], (2, 0))
    cache.save('stypes', [st])
    loaded = cache.load('stypes')
    assert loaded[0].coefficient == SR(1)/2
    assert loaded[0].bigrade == (2, 0)


def test_no_k_no_loop_order(cache_dir):
    """Stage with no k or loop_order (e.g. model-wide data)."""
    cache = PipelineCache(cache_dir)
    cache.save('propagator_data', {'nf': 2})
    assert cache.exists('propagator_data')
    assert cache.load('propagator_data') == {'nf': 2}


def test_typed_diagram_attributes_preserved(cache_dir):
    """All TypedDiagram attributes survive serialization."""
    cache = PipelineCache(cache_dir)
    td = _make_typed_diagram()
    cache.save('td', td, k=2, loop_order=0)
    td2 = cache.load('td', k=2, loop_order=0)

    assert set(td2.vertex_assignments.keys()) == set(td.vertex_assignments.keys())
    assert td2.edge_types == td.edge_types
    assert td2.external_legs == td.external_legs
    assert td2.propagator_indices == td.propagator_indices
    assert td2.prediagram[2] == td.prediagram[2]  # leaves
    assert td2.prediagram[3] == td.prediagram[3]  # internal

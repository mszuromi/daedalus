"""The streaming cert pipeline must reproduce the eager one exactly.

These are equivalence tests, not smoke tests: the streamed cert SET is compared
against the shipped cache element by element, so a refactor that silently drops
or invents a prediagram fails here rather than in a downstream cumulant.
"""
import os
import pytest

pytest.importorskip("sage.all")
from sage.all import load                                     # noqa: E402
import engine.enumeration.loop_diagram_enumeration as L       # noqa: E402

CACHE = 'saved_prediagrams'
CELLS = [(2, 1), (3, 1), (2, 2), (4, 1)]      # fast cells; (3,2)+ marked slow


def _cached_certs(k, ell):
    path = f'{CACHE}/prediagrams_v1_k{k}_l{ell}.sobj'
    if not os.path.exists(path):
        pytest.skip(f'{path} not built')
    return {L.pack_cert(L._iso_cert(r[0])) for r in load(path)}


@pytest.mark.parametrize('k,ell', CELLS)
def test_streamed_certs_match_cache_exactly(k, ell):
    assert L.stream_prediagram_certs(k, ell, n_procs=1) == _cached_certs(k, ell)


@pytest.mark.parametrize('k,ell', CELLS)
def test_pack_unpack_roundtrip(k, ell):
    for blob in L.stream_prediagram_certs(k, ell, n_procs=1):
        assert L.pack_cert(L.unpack_cert(blob)) == blob


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1)])
def test_parallel_matches_serial(k, ell):
    assert (L.stream_prediagram_certs(k, ell, n_procs=1)
            == L.stream_prediagram_certs(k, ell, n_procs=2))


def test_packed_cert_is_far_smaller_than_the_tuple():
    """The premise of the refactor: packing is what makes the set fit in RAM."""
    import sys
    certs = list(L.stream_prediagram_certs(3, 1, n_procs=1))
    packed = sum(len(c) for c in certs) / len(certs)
    tup = sum(sys.getsizeof(L.unpack_cert(c)) +
              sys.getsizeof(L.unpack_cert(c)[1]) for c in certs) / len(certs)
    assert packed < 64
    assert tup > 4 * packed


@pytest.mark.slow
@pytest.mark.parametrize('k,ell', [(3, 2), (4, 2)])
def test_streamed_certs_match_cache_large_cells(k, ell):
    assert L.stream_prediagram_certs(k, ell, n_procs=4) == _cached_certs(k, ell)

"""Certificate round-trip: the gate for the streaming enumeration refactor.

``_iso_cert`` -> ``cert_to_graph`` -> ``_iso_cert`` must be the identity, and
``leaves_of`` must recover exactly the k external vertices.  If either fails,
certs cannot be used as the pipeline's currency and the streaming design is
invalid.  Asserted against every prediagram in the shipped cache, so this is a
statement about real enumeration output rather than synthetic graphs.
"""
import os
import pytest

sage = pytest.importorskip("sage.all")
from sage.all import load                                    # noqa: E402
from engine.enumeration.loop_diagram_enumeration import (    # noqa: E402
    _iso_cert, cert_to_graph, leaves_of,
)

CACHE = 'saved_prediagrams'
# (k, ell) cells small enough to check exhaustively on every run.
CELLS = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]


def _load(k, ell):
    path = f'{CACHE}/prediagrams_v1_k{k}_l{ell}.sobj'
    if not os.path.exists(path):
        pytest.skip(f'{path} not built')
    return load(path)


@pytest.mark.parametrize('k,ell', CELLS)
def test_cert_roundtrip_is_identity(k, ell):
    for rec in _load(k, ell):
        D = rec[0]
        c = _iso_cert(D)
        assert _iso_cert(cert_to_graph(c, directed=True)) == c


@pytest.mark.parametrize('k,ell', CELLS)
def test_leaves_recovered_from_cert_alone(k, ell):
    for rec in _load(k, ell):
        D = rec[0]
        R = cert_to_graph(_iso_cert(D), directed=True)
        assert len(leaves_of(R)) == k


@pytest.mark.parametrize('k,ell', CELLS)
def test_cert_preserves_edge_and_vertex_counts(k, ell):
    """Multiplicity survives: repeated pairs in the cert are repeated edges."""
    for rec in _load(k, ell):
        D = rec[0]
        R = cert_to_graph(_iso_cert(D), directed=True)
        assert R.order() == D.order()
        assert R.size() == D.size()


@pytest.mark.parametrize('k,ell', CELLS)
def test_cached_set_is_deduplicated_up_to_plain_isomorphism(k, ell):
    """Prediagrams are distinct up to PLAIN isomorphism -- leaves interchangeable.

    A prediagram carries no measurement points; externals are pinned only at
    typing (``type_assignment`` permutes ``external_fields`` over the leaves).
    This pins down the semantics the refactor must preserve exactly.
    """
    pds = _load(k, ell)
    certs = {_iso_cert(rec[0]) for rec in pds}
    assert len(certs) == len(pds)

"""``record_from_cert`` (fast) against ``_record_from_cert_reference``.

The fast rebuild must return EXACTLY what the reference returns --- same
vertex lists in the same order, same labelled edges in the same iteration
order, same ``leaves``/``internal`` lists, same multigraph flags --- because
``type_assignment`` walks ``leaves`` positionally and every downstream stage
inherits the record's order.  Order matters at the Sage-backend level (the
iteration order of ``vertices()``/``edges()`` depends on slot assignment), so
equality is checked with ``sort=False`` and on lists, never on sets.
"""
import random

import pytest

pytest.importorskip("sage.all")
from engine.enumeration import prediagram_cache as pdc            # noqa: E402

#: Cells compared in full; all are small, and (5,1) has graphs with more than
#: 10 vertices, where Sage's backend orders ``vertices()`` non-trivially.
SMALL_CELLS = [(1, 3), (2, 2), (2, 3), (3, 2), (4, 1), (5, 1)]


def _snapshot(rec):
    D, G, leaves, internal = rec
    return (D.vertices(), D.edges(labels=True, sort=False),
            G.vertices(), G.edges(labels=True, sort=False),
            list(leaves), list(internal),
            D.allows_multiple_edges(), D.allows_loops(),
            G.allows_multiple_edges(), G.allows_loops(),
            D.is_directed(), G.is_directed())


def _assert_same(blob):
    ref = _snapshot(pdc._record_from_cert_reference(blob))
    new = _snapshot(pdc.record_from_cert(blob))
    assert new == ref, blob.hex()


@pytest.mark.parametrize('k,ell', SMALL_CELLS)
def test_fast_matches_reference_bit_for_bit(k, ell):
    certs = sorted(pdc.load_shipped_certs(k, ell))
    for blob in certs:
        _assert_same(blob)


def test_records_from_certs_uses_the_fast_path(monkeypatch):
    certs = sorted(pdc.load_shipped_certs(2, 2))
    seen = []
    real = pdc.record_from_cert
    monkeypatch.setattr(pdc, 'record_from_cert',
                        lambda b: (seen.append(b), real(b))[1])
    pdc.records_from_certs(certs)
    assert seen == certs


def test_fast_matches_reference_on_random_certs_of_mixed_cells():
    pool = []
    for k, ell in [(2, 3), (3, 2), (4, 2), (5, 1), (6, 1), (1, 4)]:
        pool.extend(pdc.load_shipped_certs(k, ell))
    pool.sort()
    for blob in random.Random(20261012).sample(pool, 500):
        _assert_same(blob)


@pytest.mark.parametrize('blob', [
    b'',                      # no order byte
    bytes([3, 0, 1, 2]),      # half an edge
    bytes([2, 1, 1]),         # a loop (the graphs forbid them)
    bytes([2, 0, 5]),         # endpoint beyond the order
    bytes([3]),               # no edges
    bytes([3, 0, 1]),         # an isolated vertex
])
def test_malformed_certs_behave_as_the_reference(blob):
    def run(fn):
        try:
            return ('ok', _snapshot(fn(blob)))
        except Exception as exc:                   # noqa: BLE001
            return (type(exc).__name__, str(exc))

    assert run(pdc.record_from_cert) == run(pdc._record_from_cert_reference)


@pytest.mark.parametrize('k,ell,name',
                         [(2, 2, 'rfc-k2'), (3, 1, 'rfc-k3'), (4, 1, 'rfc-k4')])
def test_typed_pipeline_identical_with_reference_swapped_in(k, ell, name,
                                                            monkeypatch):
    """Typed -> causal -> unique counts, multiplicities, factors and the sorted
    diagram signatures (a complete isomorphism invariant) are the same whether
    the shipped cell is rebuilt by the fast or the reference path."""
    from tests.test_prediagram_cache import _ctx, _stages

    ctx = _ctx(k, ell, name)
    certs = pdc.load_shipped_certs(k, ell)

    fast = _stages(ctx, pdc.records_from_certs(certs))
    with monkeypatch.context() as mp:
        mp.setattr(pdc, 'record_from_cert', pdc._record_from_cert_reference)
        ref = _stages(ctx, pdc.records_from_certs(certs))

    for key in ('n_typed', 'n_causal', 'n_unique', 'mult', 'sigs', 'factors'):
        assert fast[key] == ref[key], key
    assert fast['n_unique'] > 0

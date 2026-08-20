"""
engine.enumeration.prediagram_cache
===================================
Two on-disk formats for the model-independent prediagram set at ``(k, ell)``.

v1 (``saved_prediagrams/prediagrams_v1_k{k}_l{ell}.sobj``)
    A pickled list of ``(D, G, leaves, internal)`` records --- live Sage
    graphs.  Measured at ~13.8 kB of resident memory per prediagram, which is
    what caps the reachable ``(k, ell)`` cells: (6,2) would need ~170 GB.

v2 (``saved_prediagrams/streaming_v2/prediagrams_v2_k{k}_l{ell}.pkl``)
    A pickled ``set`` of PACKED CERTIFICATES (:func:`pack_cert` of
    :func:`_iso_cert`), tens of bytes each --- a ~350x shrink live, and 4-5x
    on disk against v1's gzipped ``.sobj`` (measured: (4,1) 78,952 -> 17,660
    bytes; (2,2) 29,509 -> 5,778).  This is the format
    :func:`stream_prediagram_certs` already emits, so the expensive cells
    computed by the streaming path drop straight in.  Records are rebuilt on
    load by :func:`record_from_cert`.

Why the record rebuild is not a verbatim replay of v1
-----------------------------------------------------
A certificate is an isomorphism-class representative: it stores nauty's
canonical labelling, not the labelling the enumerator happened to produce.
Reconstruction therefore returns a graph ISOMORPHIC to the v1 record but not
identical to it, and neither the vertex numbering nor the edge labels can be
recovered.  Two consequences, both handled here:

1. **Vertex convention.**  v1 records come out of ``relabel_leaves_first``,
   which puts the sorted leaves on ``0..k-1`` and the sorted internal vertices
   on ``k..|V|-1``.  ``type_assignment`` assigns external fields to
   ``leaves[i]`` POSITIONALLY, so the convention is re-applied here: every
   rebuilt record satisfies ``leaves == list(range(k))`` exactly as v1's does.
   (The canonical labelling on its own does not: nauty is free to number the
   degree-1 vertices anywhere.)

2. **Edge labels.**  ``orient_edges`` tags each edge copy with a distinct
   integer, and the whole downstream stack keys ``edge_types`` /
   ``propagator_indices`` on ``(u, v, label)``.  Parallel edges sharing a
   label would COLLIDE in those dicts and silently merge two propagators, so
   the rebuild re-tags every edge with a distinct integer.  The specific
   integers are not reproducible from a cert --- and are not reproducible from
   v1 either, since a v1 record's labels reflect the enumerator's live
   ``G.edges()`` iteration order at generation time, which a save/load round
   trip does not preserve.  Only distinctness is load-bearing.

What the surviving difference costs
----------------------------------
The difference that cannot be designed away is WHICH labelled representative
of each isomorphism class reaches the integrator.  Measured, not assumed
(``tests/test_prediagram_cache.py``):

* The diagram SET is preserved EXACTLY.  Prediagram, typed, causal and unique
  counts all match; so do the ``diagram_signature`` multiset (a complete
  isomorphism invariant), the dedup multiplicities, and every
  ``combinatorial_factor`` / ``external_wick_compensation``.  Nothing is
  dropped, duplicated or re-weighted --- which is the failure mode the
  positional leaf assignment in ``type_assignment`` invites.

* The FLOAT total moves by at most 1-2 ULP (<= 4.5e-16 relative), and only
  because the leaves arrive in a different order.  This is rounding, not
  physics: at ``k=2, ell=1`` simply SWAPPING the two leaves of the v1 records
  --- same graphs, same everything else --- reproduces the v2 total bit for
  bit (``-0x1.e96d428f8073ap-6`` against v1's ``...73bp-6``).  The
  ``deduplicate_with_multiplicities`` + ``external_wick_compensation``
  machinery does make the total leaf-order-independent in exact arithmetic;
  what it cannot make order-independent is IEEE rounding, because the
  integrand is assembled in leaf order.

So switching a cell from v1 to v2 is equivalent to relabelling its leaves, and
costs the same last bit that relabelling costs.  Exact bit-identity with a v1
file is not attainable from a certificate --- the labelling a certificate
forgets is precisely the labelling that fixes the rounding.

Not addressed here: ``load_prediagrams`` still materialises every record, so
the v2 win is disk and the ability to STORE a cell like (6,2) at all; making
the typed-assignment stage itself streaming is separate work.
"""

import os
import pickle

from sage.all import DiGraph, Graph

from engine.enumeration.loop_diagram_enumeration import (
    _iso_cert,
    cert_to_graph,
    enumerate_all as _enumerate_eager,
    pack_cert,
    relabel_leaves_first,
    stream_prediagram_certs,
    unpack_cert,
)

#: Subdirectory of the prediagram cache root holding the v2 cert files.  Fixed
#: by the files ``stream_prediagram_certs`` runs have already written.
V2_SUBDIR = 'streaming_v2'

#: Filename stems.  Both match what :class:`~engine.core.cache.PipelineCache`
#: would produce for these stages at ``(k, ell)``, so the shipped v1 ``.sobj``
#: files and the already-streamed v2 ``.pkl`` files are found unchanged.
V1_STAGE = 'prediagrams_v1'
V2_STAGE = 'prediagrams_v2'

#: Format preference, overridable for A/B testing:
#:   'auto' (default) -- v2 if present, else v1, else compute
#:   'v1'             -- v1 only (never reads or writes v2)
#:   'v2'             -- v2 only (never reads v1)
_FORMAT_ENV = 'DAEDALUS_PREDIAGRAM_FORMAT'
_PROCS_ENV = 'DAEDALUS_PREDIAGRAM_PROCS'


def cache_format():
    """Resolve the on-disk format preference; see :data:`_FORMAT_ENV`."""
    fmt = os.environ.get(_FORMAT_ENV, 'auto').strip().lower()
    if fmt not in ('auto', 'v1', 'v2'):
        raise ValueError(
            f'{_FORMAT_ENV}={fmt!r}: expected one of auto, v1, v2')
    return fmt


def _enum_procs():
    return max(1, int(os.environ.get(_PROCS_ENV, '1')))


# ── Paths ───────────────────────────────────────────────────────────────────

def v2_path(root, k, ell):
    """Path of the v2 cert file for ``(k, ell)`` under cache *root*."""
    return os.path.join(os.path.expanduser(root), V2_SUBDIR,
                        f'{V2_STAGE}_k{int(k)}_l{int(ell)}.pkl')


def v1_path(root, k, ell):
    """Path of the v1 record file for ``(k, ell)`` under cache *root*."""
    return os.path.join(os.path.expanduser(root),
                        f'{V1_STAGE}_k{int(k)}_l{int(ell)}.sobj')


def v2_exists(root, k, ell):
    return os.path.isfile(v2_path(root, k, ell))


def v1_exists(root, k, ell):
    return os.path.isfile(v1_path(root, k, ell))


# ── Cert <-> record ─────────────────────────────────────────────────────────

def record_from_cert(blob):
    """Rebuild one ``(D, G, leaves, internal)`` record from a packed cert.

    ``leaves`` are the degree-1 vertices and ``internal`` the rest, both under
    the ``relabel_leaves_first`` convention (leaves ``0..k-1``, internal
    ``k..|V|-1``) that v1 records carry and that ``type_assignment`` relies on
    when it walks ``leaves`` positionally.

    Edge labels are re-issued as ``0..|E|-1`` so that parallel edges stay
    distinguishable as ``(u, v, label)`` dict keys.
    """
    D_canon = cert_to_graph(unpack_cert(blob), directed=True)
    D_canon, _ = relabel_leaves_first(D_canon)

    # Undirected topology, same vertex labels -- prediagram slot 1.
    G = Graph(multiedges=True, loops=False)
    G.add_vertices(D_canon.vertices())
    for u, v in D_canon.edges(labels=False):
        G.add_edge(u, v)

    # Re-tag with distinct labels.  ``sorted`` only fixes a deterministic
    # order; any injection from edge copies to labels would do.
    D = DiGraph(multiedges=True, loops=False)
    D.add_vertices(D_canon.vertices())
    for i, (u, v) in enumerate(sorted(D_canon.edges(labels=False))):
        D.add_edge(u, v, i)

    leaves = [v for v in D.vertices() if D.degree(v) == 1]
    internal = [v for v in D.vertices() if D.degree(v) != 1]
    return (D, G, leaves, internal)


def records_from_certs(certs):
    """Rebuild the record list from an iterable of packed certs.

    Sorted by packed cert so the record ORDER --- and hence which member of
    each downstream dedup class becomes the representative --- is a function
    of the cert set alone, not of set-iteration order.
    """
    return [record_from_cert(b) for b in sorted(certs)]


def certs_from_records(records):
    """Packed certs of a v1-shaped record list.  Inverse of the above modulo
    the labelling that a cert deliberately forgets."""
    return {pack_cert(_iso_cert(rec[0])) for rec in records}


# ── Load / save ─────────────────────────────────────────────────────────────

def load_v2_certs(root, k, ell):
    """Read the packed-cert set for ``(k, ell)``."""
    with open(v2_path(root, k, ell), 'rb') as f:
        certs = pickle.load(f)
    if not isinstance(certs, (set, frozenset, list, tuple)):
        raise ValueError(
            f'{v2_path(root, k, ell)}: expected a set of packed certs, '
            f'got {type(certs).__name__}')
    return certs


def save_v2_certs(root, k, ell, certs):
    """Write the packed-cert set for ``(k, ell)``, creating the subdir."""
    path = v2_path(root, k, ell)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(set(certs), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_prediagrams(root, k, ell, *, use_cache=True, verbose=False):
    """Prediagram records for ``(k, ell)``, cheapest available source first.

    Lookup order is v2, then v1, then compute.  A compute miss is served by
    :func:`stream_prediagram_certs` (the packed-cert path) and written back as
    v2 --- v1 files are never created, only read.

    Because v1 wins over computing, a cell that already has a v1 file keeps
    being served from v1 and keeps its exact previous numbers: no shipped cell
    silently acquires the leaf relabelling described in the module docstring.
    Only new cells, and cells whose v2 file was written by a streaming run,
    come from v2.  ``DAEDALUS_PREDIAGRAM_FORMAT=v2`` forces the other choice
    (it is what the A/B tests use); ``DAEDALUS_PREDIAGRAM_PROCS`` sets the
    worker count for a compute miss (default 1).

    ``use_cache=False`` means "do not involve the cache at all", and the cert
    round trip is a cache-format concern: that path therefore runs the EAGER
    enumerator and returns its records verbatim, exactly as before this module
    existed.  It matters -- ``build_pipeline_records`` (the spatial entry
    point) always passes ``use_cache=False``, and downstream code indexes the
    resulting diagram list positionally, so handing it canonically-relabelled
    records in a different order picks out a different diagram.

    Returns
    -------
    records : list of (D, G, leaves, internal)
    source : {'v2', 'v1', 'computed', 'eager'}
    """
    from sage.all import load as sage_load

    if not use_cache:
        return list(_enumerate_eager(k=k, ell=ell, verbose=verbose)[2]), 'eager'

    fmt = cache_format()

    if fmt in ('auto', 'v2') and v2_exists(root, k, ell):
        return records_from_certs(load_v2_certs(root, k, ell)), 'v2'

    if fmt in ('auto', 'v1') and v1_exists(root, k, ell):
        return list(sage_load(v1_path(root, k, ell))), 'v1'

    certs = stream_prediagram_certs(k, ell, n_procs=_enum_procs(),
                                    verbose=verbose)
    if fmt in ('auto', 'v2'):
        save_v2_certs(root, k, ell, certs)
    return records_from_certs(certs), 'computed'

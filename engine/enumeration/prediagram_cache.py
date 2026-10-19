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

    Every v2 file is STAMPED (:data:`V2_FORMAT`, :data:`CONVENTION`,
    :data:`EDGE_PACK`, certificate backend, Sage version, cell, class count):
    a file whose convention stamp differs from this code's is refused with a
    message naming both; a legacy bare-set file is accepted.

shipped (``engine/enumeration/shipped_prediagrams/prediagrams_v2_k{k}_l{ell}.pkl``)
    The stamped v2 files for the cells in :data:`SHIPPED_CELLS`, EXACTLY these
    21 cells (class counts in parentheses; 5.7 MB beyond the original
    3.4 MB):

    ====  =======================================================
    k     ell
    ====  =======================================================
    1     1 (1), 2 (15), 3 (434), 4 (22,332)
    2     0 (1), 1 (9), 2 (283), 3 (14,928), 4 (1,152,032)
    3     0 (3), 1 (80), 2 (4,496), 3 (358,983)
    4     0 (13), 1 (755), 2 (65,956)
    5     0 (69), 1 (7,412), 2 (922,728)
    6     0 (448), 1 (75,253)
    ====  =======================================================

    The four largest ((2,4), (3,3), (5,2), (6,1)) are ``.pkl.xz`` (the same
    stamped pickle, xz-compressed, read with ``lzma``); the rest are plain
    ``.pkl``.  ``MANIFEST.json`` in that directory lists every file with its
    cell, class count, size and SHA-256 (:func:`build_manifest`).  They are
    tracked in git and found relative to this module, so a fresh clone never
    recomputes them and does not depend on the working directory.  Nothing
    under ``saved_prediagrams/`` is tracked: that directory is the per-machine
    cache, and the large cells that live there on the development machine
    ((6,2), (4,3)) are NOT shipped (see :func:`fetch_cache`).  Rebuild the
    shipped files with ``sage -python -m engine.enumeration.prediagram_cache
    --rebuild-shipped --procs N`` and the manifest with ``--rebuild-manifest``;
    ``tests/test_shipped_prediagrams.py`` checks the manifest and compares
    the small cells with a fresh enumeration.

    One portability caveat.  A certificate is ``canonical_label()`` of the
    graph, and Sage picks the backend by graph, not by machine: bliss when it
    is installed and the graph has no parallel edges, its own algorithm
    otherwise.  On a machine without bliss the simple-graph prediagrams
    therefore canonicalise to DIFFERENT bytes for the SAME isomorphism class.
    Every consumer here only rebuilds a representative from a certificate, so
    loading is unaffected; but two certificate sets from different backends
    must be compared up to isomorphism (:func:`recanonicalize`), never as
    bytes, and the record order (sorted by certificate) can differ between
    backends, which moves totals by the same last bit that the v1/v2 switch
    does (below).

streamed (``load_prediagrams(..., use_cache=False)``, the default there)
    No file at all: the certs are streamed in memory and rebuilt exactly as a
    v2 load would rebuild them, so the records, their labels and their order
    are those of a v2 file written by the same certificate backend.  The
    spatial path (``build_pipeline_records``, which never uses the cache)
    gets these records.  The TEMPORAL compute path does not, yet: with
    ``use_cache=False`` it still gets the eager enumerator's records verbatim
    (``stream=False``, see :data:`TEMPORAL_CACHE_OFF_STREAMS` for why).
    ``DAEDALUS_PREDIAGRAM_EAGER`` overrides both defaults (:data:`_EAGER_ENV`).

    The shipped files hold Sage-backend certs written on a Sage install with
    the optional bliss package.  Where the compiled 'fast' backend is active
    (the default whenever the extension builds), streamed records and a fresh
    clone's cache-on records are therefore different representatives of the
    same classes, in a different order.  With the Sage backend
    (``DAEDALUS_FASTENUM=0`` or ``DAEDALUS_CERT_BACKEND=sage``) on a Sage
    install that has bliss, they are identical; without bliss they differ
    again (the portability caveat above).

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
   rebuilt record has ``sorted(leaves) == list(range(k))`` as v1's does.  The
   LIST order is the backend's vertex iteration order, which for graphs with
   more than 10 vertices is not ascending (a fraction of a percent of the
   records of the large cells, e.g. 176 of 65,956 at (4,2)), so
   ``leaves == list(range(k))`` does not hold for every record.  The rebuilt
   records have always had this order and :func:`record_from_cert` reproduces
   it exactly; it is kept on purpose.
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

That bound holds where every integration region is evaluated analytically.
A region that falls back to ``scipy.nquad`` is integrated only to the
fallback's tolerance, and another representative can send other regions
there, so such a total moves by the fallback's error instead.  Measured with
the ``scipy.nquad`` fallback at its default tolerance as it stood at commit
16b5564 (any change to the fallback changes these numbers), on
``single_population_spike_reset_test``, k=2, ell=1: fast-backend streamed
records vs the eager records, or vs the (Sage-backend) shipped records, move
the one-loop term by up to 1.0e-10 absolute and ``C_tau`` by up to 3.6e-7
relative; with a tight fallback tolerance the representatives agree to
1.7e-15 absolute.  The same mechanism separates cache-on and cache-off runs
whenever their records come from different sources (the eager enumerator
and the shipped files already took different routes for
``single_population_quad_exp_test``: 8.0e-11 relative in ``C_tau``).  This
is why the temporal path keeps the eager records for ``use_cache=False``
(:data:`TEMPORAL_CACHE_OFF_STREAMS`).

Not addressed here: ``load_prediagrams`` still materialises every record, so
the v2 win is disk and the ability to STORE a cell like (6,2) at all; making
the typed-assignment stage itself streaming is separate work.
"""

import logging
import lzma
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

#: Directory of the v2 cert files tracked in git, next to this module.
SHIPPED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'shipped_prediagrams')

#: The cells shipped with the package (see the module docstring).
SHIPPED_CELLS = ((1, 1), (1, 2), (1, 3), (1, 4),
                 (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                 (3, 0), (3, 1), (3, 2), (3, 3),
                 (4, 0), (4, 1), (4, 2),
                 (5, 0), (5, 1), (5, 2),
                 (6, 0), (6, 1))


# ── Version stamp ───────────────────────────────────────────────────────────
#: A v2 file is ``{'stamp': {...}, 'certs': [packed certs, sorted]}``.  The
#: stamp records what the loader must know to use the certs safely:
#:
#: * ``format``      layout of the file itself (:data:`V2_FORMAT`);
#: * ``convention``  the prediagram convention (:data:`CONVENTION`, the
#:                   equivalent of the ``prediagrams_v1`` stage name): which
#:                   objects a cert stands for (dedup up to plain isomorphism,
#:                   leaves = degree-1 vertices, ...).  A MISMATCH IS REFUSED;
#: * ``edge_pack``   byte layout of ``pack_cert`` (:data:`EDGE_PACK`), also
#:                   refused on mismatch;
#: * ``cert_backend`` canonical-labelling backend that produced the bytes
#:                   ('sage', 'fast'); informational, because loading never
#:                   canonicalises (compare sets through ``recanonicalize``);
#: * ``sage_version`` informational;
#: * ``k``, ``ell``, ``count`` the cell and its class count (checked on load).
#:
#: A file that is a bare ``set`` of certs (written before the stamp existed) is
#: a LEGACY file: accepted as the current convention, noted once at debug level.
V2_FORMAT = 1
CONVENTION = 'prediagrams_v1'
EDGE_PACK = 1

_log = logging.getLogger(__name__)
_LEGACY_NOTED = False


def _sage_version():
    try:
        import sage.version
        return str(sage.version.version)
    except Exception:                                   # pragma: no cover
        return 'unknown'


def make_stamp(k, ell, certs, cert_backend=None):
    """The stamp for a cert set at ``(k, ell)``; *cert_backend* defaults to
    the backend active in this process."""
    if cert_backend is None:
        import engine.enumeration.loop_diagram_enumeration as _L
        cert_backend = _L.CERT_BACKEND
    return {'format': V2_FORMAT, 'convention': CONVENTION,
            'edge_pack': EDGE_PACK, 'cert_backend': str(cert_backend),
            'sage_version': _sage_version(), 'k': int(k), 'ell': int(ell),
            'count': len(certs)}


def check_stamp(stamp, where, k=None, ell=None):
    """Refuse a stamp this code cannot safely use; return it unchanged."""
    if not isinstance(stamp, dict):
        raise ValueError(f'{where}: malformed stamp {stamp!r}')
    if stamp.get('convention') != CONVENTION:
        raise ValueError(
            f"{where}: prediagram convention stamp {stamp.get('convention')!r} "
            f'does not match this code ({CONVENTION!r}); the cells were built '
            f'under another convention.  Delete the file (or the cache '
            f'directory) so it is recomputed, or use the code version that '
            f'wrote it.')
    if stamp.get('edge_pack') != EDGE_PACK or stamp.get('format') != V2_FORMAT:
        raise ValueError(
            f"{where}: file format {stamp.get('format')!r}/edge pack "
            f"{stamp.get('edge_pack')!r} does not match this code "
            f'({V2_FORMAT!r}/{EDGE_PACK!r}); regenerate the file.')
    if k is not None and (stamp.get('k'), stamp.get('ell')) != (int(k), int(ell)):
        raise ValueError(
            f"{where}: stamped for cell {(stamp.get('k'), stamp.get('ell'))}, "
            f'requested {(int(k), int(ell))}')
    return stamp


def read_cert_file(path, k=None, ell=None):
    """``(certs, stamp)`` of a v2 file.  A legacy bare-set file returns
    ``stamp=None`` (accepted); a stamped file is checked by
    :func:`check_stamp` and its class count is verified."""
    global _LEGACY_NOTED
    with open(path, 'rb') as f:
        xz = f.read(6) == b'\xfd7zXZ\x00'          # by content, not extension
    opener = lzma.open if xz else open
    with opener(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        stamp = check_stamp(obj.get('stamp'), path, k, ell)
        certs = set(obj['certs'])
        if len(certs) != stamp.get('count'):
            raise ValueError(f"{path}: stamp says {stamp.get('count')} "
                             f'classes, file holds {len(certs)}')
        return certs, stamp
    if isinstance(obj, (set, frozenset, list, tuple)):
        if not _LEGACY_NOTED:
            _LEGACY_NOTED = True
            _log.debug('unstamped (legacy) v2 cert file %s accepted as '
                       'convention %s', path, CONVENTION)
        return set(obj), None
    raise ValueError(f'{path}: expected a stamped cert file or a set of '
                     f'packed certs, got {type(obj).__name__}')


def write_cert_file(path, k, ell, certs, cert_backend=None):
    """Atomically write a stamped v2 file; returns the stamp."""
    certs = set(certs)
    stamp = make_stamp(k, ell, certs, cert_backend)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    if str(path).endswith('.xz'):
        with lzma.open(tmp, 'wb', preset=9 | lzma.PRESET_EXTREME) as f:
            pickle.dump({'stamp': stamp, 'certs': sorted(certs)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tmp, 'wb') as f:
            pickle.dump({'stamp': stamp, 'certs': sorted(certs)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return stamp


def shipped_path(k, ell):
    """Path of the shipped v2 cert file for ``(k, ell)``: the plain ``.pkl``
    or, when only that exists, the xz-compressed ``.pkl.xz`` variant (a
    stamped pickle read with ``lzma``); the ``.pkl`` name when neither does."""
    base = os.path.join(SHIPPED_DIR, f'{V2_STAGE}_k{int(k)}_l{int(ell)}.pkl')
    if not os.path.isfile(base) and os.path.isfile(base + '.xz'):
        return base + '.xz'
    return base


def shipped_exists(k, ell):
    return os.path.isfile(shipped_path(k, ell))


def load_shipped_certs(k, ell):
    """The shipped packed-cert set for ``(k, ell)`` (stamp checked)."""
    return read_cert_file(shipped_path(k, ell), k, ell)[0]


def shipped_stamp(k, ell):
    """The stamp of the shipped file for ``(k, ell)`` (None if legacy)."""
    return read_cert_file(shipped_path(k, ell), k, ell)[1]


def recanonicalize(certs):
    """Re-express packed certs in THIS machine's canonical form.

    Identity when the certs were produced by the same canonical-labelling
    backend; otherwise the map from one backend's representatives to the
    other's.  Use it before comparing a shipped set with a locally computed
    one (see the module docstring)."""
    return {pack_cert(_iso_cert(cert_to_graph(unpack_cert(b), directed=True)))
            for b in certs}


def write_shipped_cell(k, ell, n_procs=1, verbose=False):
    """(Re)build the shipped file for one cell from a fresh enumeration."""
    certs = set(stream_prediagram_certs(k, ell, n_procs=n_procs, verbose=verbose))
    path = shipped_path(k, ell)
    write_cert_file(path, k, ell, certs)
    return path, len(certs)


# ── Explicit fetch tier ─────────────────────────────────────────────────────
#: Environment variable holding the base URL (``https://...`` or
#: ``file:///...``) the cells in ``fetch_manifest.json`` are downloaded from.
#: There is no built-in default URL, and nothing is ever fetched implicitly.
FETCH_URL_ENV = 'DAEDALUS_CACHE_URL'
FETCH_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'fetch_manifest.json')


def load_fetch_manifest(path=None):
    import json
    with open(path or FETCH_MANIFEST) as f:
        return json.load(f)


def fetch_cache(cells='core', dest=None, base_url=None, *, manifest=None,
                force=False, verbose=True):
    """Download prediagram cells that are NOT tracked in git.

    ``cells`` is ``'core'`` (the cells ``fetch_manifest.json`` marks core),
    ``'all'`` (every cell it lists) or a list of ``(k, ell)``.  Only MISSING
    cells are downloaded (not in the shipped directory, not already under
    *dest*) unless ``force``.  *dest* is a cache root (default
    ``saved_prediagrams``); files land in ``<dest>/streaming_v2/`` where
    :func:`load_prediagrams` finds them as local v2 files.

    *base_url* defaults to the environment variable :data:`FETCH_URL_ENV`; a
    file is fetched from ``<base_url>/<name>`` (or from the entry's own
    ``url`` when set).  Every download is verified against the manifest's
    size and SHA-256 and then parsed (stamp, convention, class count) BEFORE
    it is moved into place atomically; any mismatch removes the partial file
    and raises ``ValueError``.  Returns ``{(k, ell): 'fetched' | 'present' |
    'shipped'}`` and prints what it did.  Never called implicitly.
    """
    import hashlib
    import urllib.request

    man = manifest if manifest is not None else load_fetch_manifest()
    entries = {(e['k'], e['ell']): e for e in man['files']}
    if cells == 'core':
        want = [c for c, e in entries.items() if e.get('core')]
    elif cells == 'all':
        want = list(entries)
    else:
        want = [(int(k), int(l)) for k, l in cells]
        unknown = [c for c in want if c not in entries]
        if unknown:
            raise ValueError(f'fetch_cache: no manifest entry for cells {unknown}; '
                             f'known: {sorted(entries)}')
    base_url = base_url or os.environ.get(FETCH_URL_ENV)
    dest = dest or 'saved_prediagrams'
    outdir = os.path.join(os.path.expanduser(dest), V2_SUBDIR)
    report = {}
    for cell in sorted(want):
        e = entries[cell]
        k, ell = cell
        if not force and shipped_exists(k, ell):
            report[cell] = 'shipped'
            continue
        if not force and v2_exists(dest, k, ell):
            report[cell] = 'present'
            continue
        if not e.get('sha256'):
            raise ValueError(f'fetch_cache: manifest entry {e["name"]} has no sha256')
        url = e.get('url') or (base_url.rstrip('/') + '/' + e['name'] if base_url else None)
        if not url:
            raise ValueError(f'fetch_cache: no base URL; pass base_url= or set '
                             f'{FETCH_URL_ENV} (file {e["name"]} is not in git)')
        os.makedirs(outdir, exist_ok=True)
        final = os.path.join(outdir, e['name'])
        tmp = final + '.part'
        h = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
                for chunk in iter(lambda: r.read(1 << 20), b''):
                    h.update(chunk)
                    size += len(chunk)
                    f.write(chunk)
            if e.get('size') is not None and size != e['size']:
                raise ValueError(f'fetch_cache: {e["name"]}: got {size} bytes, '
                                 f'manifest says {e["size"]}')
            if h.hexdigest() != e['sha256']:
                raise ValueError(f'fetch_cache: {e["name"]}: SHA-256 mismatch '
                                 f'(got {h.hexdigest()[:16]}..., manifest '
                                 f'{e["sha256"][:16]}...); file refused')
            certs, _ = read_cert_file(tmp, k, ell)       # stamp + count checked
            if e.get('count') is not None and len(certs) != e['count']:
                raise ValueError(f'fetch_cache: {e["name"]}: {len(certs)} classes, '
                                 f'manifest says {e["count"]}')
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        os.replace(tmp, final)
        report[cell] = 'fetched'
        if verbose:
            print(f'fetch_cache: ({k},{ell}) {len(certs)} classes, {size} bytes '
                  f'-> {final}')
    if verbose:
        for cell, what in sorted(report.items()):
            if what != 'fetched':
                print(f'fetch_cache: ({cell[0]},{cell[1]}) {what}, nothing to do')
    return report


# ── Shipped manifest ────────────────────────────────────────────────────────
#: Tracked next to the shipped files: for every file its name, cell, class
#: count, size and SHA-256.  ``tests/test_shipped_prediagrams.py`` checks it.
MANIFEST_NAME = 'MANIFEST.json'


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(directory=None):
    """The manifest entries of every ``prediagrams_v2_k*_l*.pkl[.xz]`` in
    *directory* (default :data:`SHIPPED_DIR`), sorted by cell.  Reads each
    file's stamp for the class count."""
    import re
    directory = directory or SHIPPED_DIR
    rx = re.compile(rf'^{V2_STAGE}_k(\d+)_l(\d+)\.pkl(\.xz)?$')
    entries = []
    for name in os.listdir(directory):
        m = rx.match(name)
        if not m:
            continue
        k, ell = int(m.group(1)), int(m.group(2))
        path = os.path.join(directory, name)
        certs, stamp = read_cert_file(path, k, ell)
        entries.append({'name': name, 'k': k, 'ell': ell,
                        'count': len(certs), 'size': os.path.getsize(path),
                        'sha256': _sha256(path),
                        'convention': (stamp or {}).get('convention'),
                        'cert_backend': (stamp or {}).get('cert_backend')})
    entries.sort(key=lambda e: (e['k'], e['ell']))
    return entries


def write_manifest(directory=None):
    """(Re)write ``MANIFEST.json`` in *directory*; returns the entries."""
    import json
    directory = directory or SHIPPED_DIR
    entries = build_manifest(directory)
    with open(os.path.join(directory, MANIFEST_NAME), 'w') as f:
        json.dump({'convention': CONVENTION, 'files': entries}, f, indent=1)
        f.write('\n')
    return entries


def load_manifest(directory=None):
    import json
    with open(os.path.join(directory or SHIPPED_DIR, MANIFEST_NAME)) as f:
        return json.load(f)


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

#: Source of ``load_prediagrams(use_cache=False)`` for every caller:
#:   unset or empty   -- each caller's ``stream`` argument decides (the
#:                       spatial path streams; the temporal path passes
#:                       :data:`TEMPORAL_CACHE_OFF_STREAMS`);
#:   1/true/yes/on    -- the EAGER enumerator's records verbatim (source
#:                       ``'eager'``): rollback and A/B;
#:   0/false/no/off   -- the streamed records (source ``'streamed'``), the
#:                       temporal path included.
#: Any other value raises ``ValueError``.
_EAGER_ENV = 'DAEDALUS_PREDIAGRAM_EAGER'

#: Whether the TEMPORAL compute path (``api.compute.compute_cumulants``)
#: streams its ``use_cache=False`` prediagrams.  False: it keeps the eager
#: enumerator's records, as before streaming existed, so its numbers do not
#: move.  Streamed records are other labelled representatives of the same
#: classes, and Phase J's ``scipy.nquad`` fallback is not
#: representative-independent: on models whose regions reach it, another
#: representative sends other regions there and the total moves by the
#: fallback's error.  Where every region is analytic the switch moves
#: totals by rounding only (``ou_quartic`` k=2, ell<=3: 4.6e-16).  Measured
#: 2026-10-06 with the fallback of commit d68e383:
#: ``single_population_spike_reset_test`` k=2, ell=1 agreed to rtol 1e-13 and
#: ``single_population_quad_exp_test`` (k=2, ell<=1) differed by 1.21e-11
#: relative.  Since 0.2.0 (box-free Phase J by default) all three agree to
#: rtol 1e-13 (measured 2026-10-11), but fewer regions reach the fallback,
#: not none: ``quad_exp`` still sends 8 m=3 regions there at tau=0.5 and 1
#: (a variable without an inherited lower bound, declined by the exact DBM
#: as ill-conditioned), ``quadratic_hawkes_alpha`` k=2, ell=1 one.  Flip
#: this once no tracked model reaches the fallback, or the fallback agrees
#: across representatives to rtol 1e-13.  The gate is the
#: slow parametrized test
#: ``tests/test_prediagram_cache.py::test_temporal_streamed_totals_match_eager_on_a_fallback_model``
#: (about 7 min): run it (``pytest -m slow ... -k fallback_model``) after any
#: change to Phase J routing or its fallback.
TEMPORAL_CACHE_OFF_STREAMS = False

_TRUE = ('1', 'true', 'yes', 'on')
_FALSE = ('0', 'false', 'no', 'off')


def cache_format():
    """Resolve the on-disk format preference; see :data:`_FORMAT_ENV`."""
    from engine import _env
    return _env.choice(_FORMAT_ENV, ('auto', 'v1', 'v2'), 'auto')


def _enum_procs():
    from engine import _env
    return _env.integer(_PROCS_ENV, 1, minimum=1)


def _cache_off_source(stream):
    """``'streamed'`` or ``'eager'``: the source of a ``use_cache=False``
    load whose caller asked for ``stream``, after :data:`_EAGER_ENV`."""
    raw = os.environ.get(_EAGER_ENV, '')
    val = raw.strip().lower()
    if val in _TRUE:
        return 'eager'
    if val in _FALSE:
        return 'streamed'
    if val:
        raise ValueError(
            f'{_EAGER_ENV}={raw!r}: expected one of '
            f'{", ".join(_TRUE + _FALSE)}, or unset')
    return 'streamed' if stream else 'eager'


# ── In-process memo of streamed cert sets ───────────────────────────────────
#: ``use_cache=False`` cert sets already streamed in this process, keyed by
#: the cell and every enumeration setting that can change the cert BYTES.
#: A streamed call pays a fixed ~5-90 ms at small cells (one external tree
#: generator process per tree order), which callers that enumerate the same
#: cell many times per process would otherwise pay each time.  Certs are
#: immutable bytes; the records are rebuilt on every call, so no caller ever
#: shares a mutable graph.  Memory only, never disk; bounded below.
_STREAMED_MEMO = {}
_STREAMED_MEMO_MAX_CELLS = 32
_STREAMED_MEMO_MAX_CERTS = 200_000          # cells larger than this are not kept


def _streamed_memo_key(k, ell):
    import engine.enumeration.loop_diagram_enumeration as _L
    return (int(k), int(ell), _L.CERT_BACKEND, _L.ORIENTATION_CHECKER,
            _L.EDGE_GENERATOR)


def clear_streamed_memo():
    """Forget every memoised ``use_cache=False`` cert set."""
    _STREAMED_MEMO.clear()


def _streamed_certs(k, ell, verbose=False):
    """The packed-cert set of ``(k, ell)`` from :func:`stream_prediagram_certs`,
    memoised in process (see :data:`_STREAMED_MEMO`)."""
    key = _streamed_memo_key(k, ell)
    certs = _STREAMED_MEMO.get(key)
    if certs is None:
        certs = frozenset(stream_prediagram_certs(
            k, ell, n_procs=_enum_procs(), verbose=verbose))
        if len(certs) <= _STREAMED_MEMO_MAX_CERTS:
            while len(_STREAMED_MEMO) >= _STREAMED_MEMO_MAX_CELLS:
                _STREAMED_MEMO.pop(next(iter(_STREAMED_MEMO)))
            _STREAMED_MEMO[key] = certs
    return certs


# ── Paths ───────────────────────────────────────────────────────────────────

def v2_path(root, k, ell):
    """Path of the v2 cert file for ``(k, ell)`` under cache *root*: the plain
    ``.pkl`` or, when only that exists, a fetched ``.pkl.xz``."""
    base = os.path.join(os.path.expanduser(root), V2_SUBDIR,
                        f'{V2_STAGE}_k{int(k)}_l{int(ell)}.pkl')
    if not os.path.isfile(base) and os.path.isfile(base + '.xz'):
        return base + '.xz'
    return base


def v1_path(root, k, ell):
    """Path of the v1 record file for ``(k, ell)`` under cache *root*."""
    return os.path.join(os.path.expanduser(root),
                        f'{V1_STAGE}_k{int(k)}_l{int(ell)}.sobj')


def v2_exists(root, k, ell):
    return os.path.isfile(v2_path(root, k, ell))


def v1_exists(root, k, ell):
    return os.path.isfile(v1_path(root, k, ell))


# ── Cert <-> record ─────────────────────────────────────────────────────────

def _record_from_cert_reference(blob):
    """Reference implementation of :func:`record_from_cert` (the test oracle).

    Rebuilds one ``(D, G, leaves, internal)`` record from a packed cert through
    three intermediate Sage graphs.  Kept verbatim: ``record_from_cert`` must
    return exactly what this returns.

    ``leaves`` are the degree-1 vertices and ``internal`` the rest, both under
    the ``relabel_leaves_first`` convention (leaves ``0..k-1``, internal
    ``k..|V|-1``) that v1 records carry and that ``type_assignment`` relies on
    when it walks ``leaves`` positionally.  ``sorted(leaves)`` is always
    ``list(range(k))``; the list itself follows the backend's vertex iteration
    order, which is not ascending for some graphs with more than 10 vertices.

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


_VERTEX_ORDER_MEMO = {}


def _canon_vertex_order(order):
    """``vertices()`` of the graph ``cert_to_graph`` starts from, as Sage's
    sparse backend iterates it.

    Not ``range(order)`` once ``order`` exceeds the backend's initial capacity
    (10): the label 10 then gets an explicit slot and comes out FIRST.  The
    reference's final vertex list inherits that order (it relabels this graph
    and adds the result's ``vertices()`` to the new ones), and its ``leaves`` /
    ``internal`` lists follow it, so it is asked of Sage itself, once per
    ``order``, instead of being assumed.
    """
    got = _VERTEX_ORDER_MEMO.get(order)
    if got is None:
        H = Graph(multiedges=True, loops=False)
        H.add_vertices(range(order))
        got = _VERTEX_ORDER_MEMO[order] = tuple(H.vertices())
    return got


def record_from_cert(blob):
    """Rebuild one ``(D, G, leaves, internal)`` record from a packed cert.

    ``leaves`` are the degree-1 vertices and ``internal`` the rest, both under
    the ``relabel_leaves_first`` convention (leaves ``0..k-1``, internal
    ``k..|V|-1``) that v1 records carry and that ``type_assignment`` relies on
    when it walks ``leaves`` positionally.  ``sorted(leaves)`` is always
    ``list(range(k))``; the list itself follows the backend's vertex iteration
    order, which is not ascending for some graphs with more than 10 vertices.

    Edge labels are re-issued as ``0..|E|-1`` so that parallel edges stay
    distinguishable as ``(u, v, label)`` dict keys.

    Bit-for-bit the record of :func:`_record_from_cert_reference` (same vertex
    lists and orders, same edge multiset, labels and iteration order, same
    ``leaves``/``internal``).  The degrees, the leaves-first relabelling and
    the sorted edge list are computed in plain Python from the packed bytes,
    so each of ``D`` and ``G`` is built once instead of the reference's three
    Sage graphs plus a per-vertex ``D.degree`` call.  Two details are
    deliberate copies of the reference rather than tidier choices:

    * Vertices go in as ``[perm[v] for v in canonical_order]``, the order the
      reference's relabelled graph hands over, because that order fixes the
      backend's slot assignment and hence the iteration order of
      ``vertices()`` and ``edges()`` (see :func:`_canon_vertex_order`).
    * Edges go in through ``add_edges`` in sorted order; the ``data=``
      constructors are faster still but iterate the edges in a different
      order.
    """
    n = len(blob)
    order = blob[0] if n else 0
    us = blob[1:n:2]
    vs = blob[2:n:2]
    if (n - 1) % 2 or any(u >= order or v >= order or u == v
                          for u, v in zip(us, vs)):
        # Malformed cert: let the reference produce its own behaviour.
        return _record_from_cert_reference(blob)
    deg = [0] * order
    for u in us:
        deg[u] += 1
    for v in vs:
        deg[v] += 1
    n_leaves = deg.count(1)
    perm = [0] * order
    next_leaf, next_internal = 0, n_leaves
    for x in range(order):
        if deg[x] == 1:
            perm[x] = next_leaf
            next_leaf += 1
        else:
            perm[x] = next_internal
            next_internal += 1
    edges = sorted([(perm[u], perm[v]) for u, v in zip(us, vs)])
    verts = [perm[v] for v in _canon_vertex_order(order)]

    G = Graph(multiedges=True, loops=False)
    G.add_vertices(verts)
    G.add_edges(edges)

    D = DiGraph(multiedges=True, loops=False)
    D.add_vertices(verts)
    D.add_edges([(u, v, i) for i, (u, v) in enumerate(edges)])

    vertices = D.vertices()
    return (D, G,
            [v for v in vertices if v < n_leaves],
            [v for v in vertices if v >= n_leaves])


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
    """Read the packed-cert set for ``(k, ell)`` (stamp checked)."""
    return read_cert_file(v2_path(root, k, ell), k, ell)[0]


def save_v2_certs(root, k, ell, certs, cert_backend=None):
    """Write the stamped packed-cert set for ``(k, ell)``, creating the
    subdir.  *cert_backend* names the backend that produced *certs* (default:
    the one active in this process)."""
    path = v2_path(root, k, ell)
    write_cert_file(path, k, ell, certs, cert_backend)
    return path


def _copy_file_atomic(src, dest):
    """Copy ``src`` to ``dest`` through a temporary file (no half-written
    cache file if the run is killed)."""
    import shutil
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.tmp'
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def load_prediagrams(root, k, ell, *, use_cache=True, verbose=False,
                     stream=True):
    """Prediagram records for ``(k, ell)``, cheapest available source first.

    Lookup order is local v2, then local v1, then the shipped v2 file, then
    compute.  A shipped hit is copied into the local root as v2 (so the next
    run is a local v2 hit, exactly as after a compute); a compute miss is
    served by :func:`stream_prediagram_certs` (the packed-cert path) and
    written back as local v2.  v1 files are never created, only read, and the
    shipped files are never written to.

    Because v1 wins over computing, a cell that already has a v1 file keeps
    being served from v1 and keeps its exact previous numbers: no shipped cell
    silently acquires the leaf relabelling described in the module docstring.
    Only new cells, and cells whose v2 file was written by a streaming run,
    come from v2.  ``DAEDALUS_PREDIAGRAM_FORMAT=v2`` forces the other choice
    (it is what the A/B tests use); ``DAEDALUS_PREDIAGRAM_PROCS`` sets the
    worker count for a compute miss (default 1).

    ``use_cache=False`` means "do not involve the cache at all": nothing is
    read from or written to disk (no local file, no shipped file).  With
    ``stream=True`` (the default) the records are computed by
    :func:`stream_prediagram_certs` (the compiled streaming enumerator when
    it is available, with ``DAEDALUS_PREDIAGRAM_PROCS`` workers; the fork
    guard in ``_map_unordered`` keeps a Jupyter kernel on macOS serial) and
    rebuilt in memory by :func:`records_from_certs`, source ``'streamed'``.
    The cert set is memoised in process (:func:`_streamed_certs`; certs
    only, the records are rebuilt on every call).  The records are exactly
    those a v2 file written by the same certificate backend would give: same
    representatives, same labels, same order (sorted by packed cert).  With
    ``stream=False`` they are the eager enumerator's records verbatim, in
    generation order, source ``'eager'``: what this path returned before
    streaming existed, and what the temporal compute path still asks for
    (:data:`TEMPORAL_CACHE_OFF_STREAMS`).  ``DAEDALUS_PREDIAGRAM_EAGER``
    overrides ``stream`` either way (:data:`_EAGER_ENV`).  ``stream`` is
    ignored when ``use_cache`` is true.

    Streamed records are NOT in general the records a cache-on run or the
    eager enumerator uses.  The eager enumerator, the shipped files
    (Sage-backend certs) and local files written under another backend hold
    other representatives of the same classes, in another order.  Where
    every integration region is analytic the totals then differ by rounding;
    where regions fall back to ``scipy.nquad`` they differ by the fallback's
    error (module docstring: 3.6e-7 relative in ``C_tau`` for
    ``single_population_spike_reset_test``, k=2, ell=1, streamed vs eager,
    with the fallback of commit 16b5564).
    The order also depends on the source and the certificate backend, so no
    consumer may rely on list position (pick diagrams by structure, e.g.
    ``diagram_signature``).

    Returns
    -------
    records : list of (D, G, leaves, internal)
    source : {'v2', 'v1', 'shipped', 'computed', 'streamed', 'eager'}
    """
    from sage.all import load as sage_load

    # the manifests' count (or an extrapolation): refuse before any file is
    # read or any cert computed; the counted checks below catch the rest
    _load_guard(k, ell, None)
    if not use_cache:
        if _cache_off_source(stream) == 'eager':
            return (list(_enumerate_eager(k=k, ell=ell, verbose=verbose)[2]),
                    'eager')
        certs = _streamed_certs(k, ell, verbose)
        _load_guard(k, ell, len(certs))
        return records_from_certs(certs), 'streamed'

    fmt = cache_format()

    # A local cell is served only when its class count matches the
    # package's manifests (where they know the cell): a stale, truncated or
    # foreign local file falls through to the shipped data / a compute,
    # which rewrites the local v2 file.
    if fmt in ('auto', 'v2') and v2_exists(root, k, ell):
        certs = load_v2_certs(root, k, ell)
        if _local_cell_ok(root, k, ell, len(certs), 'v2'):
            _load_guard(k, ell, len(certs))
            return records_from_certs(certs), 'v2'

    if fmt in ('auto', 'v1') and v1_exists(root, k, ell):
        recs = list(sage_load(v1_path(root, k, ell)))
        if _local_cell_ok(root, k, ell, len(recs), 'v1'):
            return recs, 'v1'

    if fmt in ('auto', 'v2') and shipped_exists(k, ell):
        src_path = shipped_path(k, ell)
        certs, stamp = read_cert_file(src_path, k, ell)
        _load_guard(k, ell, len(certs))
        if str(src_path).endswith('.xz'):
            # the compressed file as it is (re-pickling it uncompressed put
            # 38 MB of (2,4) into every working directory; audit RED-11)
            dest = v2_path(root, k, ell)
            _copy_file_atomic(src_path,
                              dest if dest.endswith('.xz') else dest + '.xz')
        else:
            save_v2_certs(root, k, ell, certs,
                          (stamp or {}).get('cert_backend', 'sage'))
        return records_from_certs(certs), 'shipped'

    certs = stream_prediagram_certs(k, ell, n_procs=_enum_procs(),
                                    verbose=verbose)
    if fmt in ('auto', 'v2'):
        save_v2_certs(root, k, ell, certs)
    _load_guard(k, ell, len(certs))
    return records_from_certs(certs), 'computed'


def _load_guard(k, ell, n):
    """Refuse a cell whose records would not fit the run's memory budget
    (``PHASE_J_MEM_BUDGET_GB``; default ``default_mem_budget``), BEFORE they
    are built: :data:`_LOAD_BYTES_PER_CLASS` per class (two Sage graphs; the
    records of (2,4) or (5,2) need 10-13 GB, of (4,3) / (6,2) 81 / 140 GB).
    ``n``: the class count when known (the cert set is in hand), else the
    manifests' count or an extrapolation (none: no check).  Raises
    :class:`engine.integration.time_domain.final_integral.PhaseJResourceError`
    (audit MEM-01)."""
    if n is None:
        n, how = estimate_class_count(k, ell)
        if n is None:
            return
        how = 'known' if how == 'known' else 'estimated'
    else:
        how = 'counted'
    need = int(n) * _LOAD_BYTES_PER_CLASS
    from engine.integration.time_domain import final_integral as _fi
    budget, rule = _fi.mem_budget('PHASE_J_MEM_BUDGET_GB')
    # an extrapolated count is good to a factor of two: refuse on it only
    # beyond that (the count is checked again once the certs are in hand)
    if budget is None or need <= budget * (2 if how == 'estimated' else 1):
        return
    ram = _fi._ram_bytes()
    # the same machine sentence as the Phase J pre-pass refusal
    machine = (f'This machine has {ram / 1e9:.1f} GB of RAM; the estimate is '
               f'{need / 1e9:.2f} GB ({need / ram:.0%} of it).' if ram else
               f'The estimate is {need / 1e9:.2f} GB (this machine\'s RAM is '
               f'unknown).')
    raise _fi.PhaseJResourceError(
        f'the prediagrams of (k={k}, ell={ell}) do not fit the memory budget: '
        f'{int(n):,} classes ({how}) need ~{_fmt_bytes(need)} as records '
        f'(~{_LOAD_BYTES_PER_CLASS:,} B per class), the budget is '
        f'{_fi._fmt_budget(budget, rule)}.  {machine}  Nothing was built.  '
        f'Options: {_fi._BUDGET_HOW}.',
        estimate={'k': k, 'ell': ell, 'classes': int(n), 'classes_how': how,
                  'bytes': need, 'budget_bytes': budget, 'budget_rule': rule,
                  'where': 'prediagram records'})


_KNOWN_COUNTS = []


def _local_cell_ok(root, k, ell, n, fmt):
    """Whether a local ``fmt`` file of cell ``(k, ell)`` holding ``n``
    classes may be served: always for a cell no manifest knows, otherwise
    only when ``n`` is the manifest's count (else a warning, and the caller
    falls through to the shipped data or a compute)."""
    if not _KNOWN_COUNTS:
        try:
            _KNOWN_COUNTS.append(known_class_counts())
        except (OSError, ValueError, KeyError):
            _KNOWN_COUNTS.append({})
    want = _KNOWN_COUNTS[0].get((int(k), int(ell)))
    if want is None or int(want) == int(n):
        return True
    import warnings
    warnings.warn(
        f'prediagram cache: the local {fmt} file of cell (k={k}, ell={ell}) '
        f'in {root} holds {n} classes, the package manifest {want}; not '
        f'served (rebuilt from the shipped data or recomputed).',
        RuntimeWarning, stacklevel=3)
    return False


# ── Building cells: cost estimate, then build ───────────────────────────────
#: Cost model of a cold build of one cell, measured on 2026-10-12 with the
#: compiled enumerator, one process (``DAEDALUS_PREDIAGRAM_PROCS`` unset):
#: 25-40 microseconds per class from (2,3) to (6,1), 69 at (2,4) (80 s for its
#: 1.15M classes) and 82 at (1,4), plus a fixed ~30 ms for the tree-generator
#: processes.  The rate is set between the two groups, so every measured
#: cell is within a factor 2; :func:`build_cache` rescales it by timing the
#: small calibration cell on the machine it runs on.  The cert set costs
#: ~90 bytes per class in a Python ``set`` at the small cells and 183 at
#: (2,4) (table growth); the records :func:`load_prediagrams` later rebuilds
#: from it cost ~11 kB per class (two Sage graphs), which is what limits the
#: cells one can USE.
_BUILD_OVERHEAD_S = 0.03
_BUILD_US_PER_CLASS = 50.0
_BUILD_BYTES_PER_CLASS = 200
_LOAD_BYTES_PER_CLASS = 11_200
_CALIBRATION_CELL = (3, 2)
_CALIBRATION_S = 0.145          # that cell's cold build on the reference machine


def known_class_counts():
    """``{(k, ell): class count}`` of every cell the package has a manifest
    entry for: the shipped files and the fetchable ones."""
    counts = {}
    for e in load_manifest()['files']:
        counts[(e['k'], e['ell'])] = e['count']
    try:
        for e in load_fetch_manifest()['files']:
            if e.get('count') is not None:
                counts[(e['k'], e['ell'])] = e['count']
    except OSError:
        pass
    return counts


def estimate_class_count(k, ell, counts=None):
    """``(n, how)``: the class count of ``(k, ell)``, from the manifests when
    known (``how='known'``), otherwise extrapolated from the cells of the same
    ``k`` below it (``'extrapolated'``; ``None`` when there are fewer than two
    to go on).

    The ratio of consecutive cells grows about linearly in ``ell`` (for k = 2
    it is 9, 31, 53, 77), so the last ratio is advanced by the last increment
    for each step.  With two cells only the ratio is repeated.  Checked by
    hiding a known cell: (2,3) 15.2k vs 14,928; (3,3) 385k vs 358,983; (4,3)
    7.70M vs 7.26M.  Treat it as good to a factor of two or so.
    """
    counts = known_class_counts() if counts is None else counts
    if (k, ell) in counts:
        return counts[(k, ell)], 'known'
    below = sorted(e for (kk, e) in counts if kk == k and e < ell)
    # the unbroken run of cells ending at the largest known ell below the target
    run = []
    for e in reversed(below):
        if run and e != run[-1] - 1:
            break
        run.append(e)
    run.reverse()
    if len(run) < 2:
        return None, None
    n = float(counts[(k, run[-1])])
    ratio = n / counts[(k, run[-2])]
    step = ratio - counts[(k, run[-2])] / counts[(k, run[-3])] if len(run) >= 3 else 0.0
    for _ in range(ell - run[-1]):
        ratio = max(ratio + step, 1.0)
        n *= ratio
    return int(round(n)), 'extrapolated'


def _speed_factor():
    """How much slower than the reference machine a cold build is here: the
    small calibration cell is enumerated once, in memory, nothing written."""
    import time
    from engine.enumeration.loop_diagram_enumeration import stream_prediagram_certs
    t0 = time.perf_counter()
    stream_prediagram_certs(*_CALIBRATION_CELL, n_procs=1, verbose=False)
    return (time.perf_counter() - t0) / _CALIBRATION_S


def _physical_memory():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        return None


def _fmt_seconds(s):
    if s < 120:
        return f'{s:.1f} s'
    if s < 7200:
        return f'{s / 60:.0f} min'
    return f'{s / 3600:.1f} h'


def _fmt_bytes(b):
    for unit, size in (('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if b >= size:
            return f'{b / size:.1f} {unit}'
    return f'{b:.0f} B'


def build_cache(cells, dest=None, dry_run=False, *, force=False,
                calibrate=True, verbose=True):
    """Say what building prediagram cells costs, then (unless *dry_run*) build
    the missing ones.

    ``cells`` is a list of ``(k, ell)`` (or one pair).  For each cell the
    report gives its status (``'shipped'``: in the package; ``'local'``: a v2
    or v1 file already under *dest*, default ``saved_prediagrams``;
    ``'missing'``), the class count (from the manifests when known,
    otherwise extrapolated), and for a missing cell the estimated time and
    memory of the build and the memory of the records that using the cell
    will materialise (:data:`_LOAD_BYTES_PER_CLASS` per class).  The time is
    for one process; the rate comes from :data:`_BUILD_US_PER_CLASS` scaled by
    a one-off timing of the calibration cell on this machine
    (``calibrate=False`` skips that and assumes the reference machine).

    The estimate is printed BEFORE any work.  With ``dry_run=True`` that is
    all that happens: nothing is written, not even *dest*.  Otherwise each
    missing cell is built by :func:`stream_prediagram_certs` (the path
    :func:`load_prediagrams` takes on a miss, ``DAEDALUS_PREDIAGRAM_PROCS``
    workers) and written to *dest* as a stamped v2 file.  A build whose
    estimated memory exceeds the machine's is refused unless ``force=True``;
    shipped and local cells are left alone unless ``force=True``.

    Returns ``{(k, ell): {...}}`` with ``status``, ``n_classes``,
    ``n_classes_how``, ``est_seconds``, ``est_build_bytes``,
    ``est_load_bytes`` and, for a built cell, ``action='built'`` with the
    measured ``seconds`` and ``n_built``.
    """
    import time
    from engine.enumeration.loop_diagram_enumeration import stream_prediagram_certs

    if (isinstance(cells, (tuple, list)) and len(cells) == 2
            and all(isinstance(c, int) for c in cells)):
        cells = [tuple(cells)]
    want = sorted({(int(k), int(l)) for k, l in cells})
    dest = dest or 'saved_prediagrams'
    counts = known_class_counts()

    report = {}
    for k, ell in want:
        if shipped_exists(k, ell):
            status = 'shipped'
        elif v2_exists(dest, k, ell) or v1_exists(dest, k, ell):
            status = 'local'
        else:
            status = 'missing'
        n, how = estimate_class_count(k, ell, counts)
        report[(k, ell)] = dict(status=status, n_classes=n, n_classes_how=how,
                                est_seconds=None, est_build_bytes=None,
                                est_load_bytes=None, action=None)
    todo = [c for c, r in report.items() if r['status'] == 'missing' or force]

    speed = 1.0
    if calibrate and any(report[c]['n_classes'] for c in todo):
        speed = _speed_factor()
    for c in todo:
        r = report[c]
        if r['n_classes']:
            r['est_seconds'] = (_BUILD_OVERHEAD_S * speed
                                + r['n_classes'] * _BUILD_US_PER_CLASS * 1e-6 * speed)
            r['est_build_bytes'] = r['n_classes'] * _BUILD_BYTES_PER_CLASS
            r['est_load_bytes'] = r['n_classes'] * _LOAD_BYTES_PER_CLASS

    ram = _physical_memory()
    if verbose:
        print(f'build_cache: dest={dest!r}, {len(want)} cell(s), '
              f'{"dry run" if dry_run else "build"}, '
              f'machine factor {speed:.2f} x reference')
        for (k, ell), r in report.items():
            n = r['n_classes']
            ncl = ('unknown class count' if n is None else
                   f'{"~" if r["n_classes_how"] != "known" else ""}{n:,} classes')
            line = f'  ({k},{ell}) {r["status"]:8s} {ncl}'
            if r['est_seconds'] is not None:
                line += (f' | build ~{_fmt_seconds(r["est_seconds"])}, '
                         f'~{_fmt_bytes(r["est_build_bytes"])}'
                         f' | using it later ~{_fmt_bytes(r["est_load_bytes"])}')
            elif r['status'] != 'missing':
                line += ' | nothing to build'
            print(line)
        if ram:
            print(f'  machine memory {_fmt_bytes(ram)}')

    if not dry_run:
        if ram:
            too_big = [c for c in todo if (report[c]['est_build_bytes'] or 0) > 0.8 * ram]
            if too_big and not force:
                raise ValueError(
                    f'build_cache: the estimated memory of {too_big} exceeds 80 % '
                    f'of this machine ({_fmt_bytes(ram)}); pass force=True to try anyway')
        for c in todo:
            k, ell = c
            t0 = time.perf_counter()
            certs = stream_prediagram_certs(k, ell, n_procs=_enum_procs(),
                                            verbose=False)
            path = save_v2_certs(dest, k, ell, certs)
            report[c].update(action='built', seconds=time.perf_counter() - t0,
                             n_built=len(certs), path=path)
            if verbose:
                print(f'build_cache: ({k},{ell}) {len(certs):,} classes in '
                      f'{_fmt_seconds(report[c]["seconds"])} -> {path}')
    return report


if __name__ == '__main__':                      # sage -python -m engine.enumeration.prediagram_cache
    import argparse
    ap = argparse.ArgumentParser(description='Rebuild the shipped prediagram cert files.')
    ap.add_argument('--rebuild-shipped', action='store_true')
    ap.add_argument('--rebuild-manifest', action='store_true',
                    help='rewrite shipped_prediagrams/MANIFEST.json')
    ap.add_argument('--procs', type=int, default=1)
    ns = ap.parse_args()
    if ns.rebuild_shipped:
        for _k, _l in SHIPPED_CELLS:
            _p, _n = write_shipped_cell(_k, _l, n_procs=ns.procs)
            print(f'({_k},{_l}) {_n:7d} certs -> {_p}')
    elif ns.rebuild_manifest:
        for _e in write_manifest():
            print(f"({_e['k']},{_e['ell']}) {_e['count']:8d} classes {_e['size']:9d} B  {_e['name']}")
    else:
        ap.print_help()

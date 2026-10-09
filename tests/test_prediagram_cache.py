"""The v2 packed-cert prediagram cache must not change the physics.

Two things are asserted here, and the difference between them is the point:

* the DIAGRAM SET is preserved exactly --- counts, the complete isomorphism
  invariant ``diagram_signature``, dedup multiplicities and both symmetry
  factors.  This is the failure mode the v2 format invites: reconstructing a
  prediagram from a certificate returns nauty's canonical labelling, and
  ``type_assignment`` assigns external fields to ``leaves[i]`` POSITIONALLY,
  so a wrong leaf convention would silently re-weight or drop diagrams.

* the FLOAT total moves by at most a couple of ULP, and only through leaf
  ORDER.  ``test_leaf_swap_reproduces_the_v2_total_bit_for_bit`` pins the
  mechanism: swapping the two leaves of the v1 records --- same graphs, same
  cache format --- reproduces the v2 total exactly, so the residual is IEEE
  rounding in a leaf-ordered integrand assembly and not a v2 artefact.

See ``engine/enumeration/prediagram_cache.py`` for the full argument.
"""
import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

pytest.importorskip('sage.all')

from sage.all import SR                                          # noqa: E402
import engine.enumeration.loop_diagram_enumeration as L          # noqa: E402
from engine.enumeration import prediagram_cache as pdc           # noqa: E402

CACHE = 'saved_prediagrams'


@pytest.fixture(autouse=True, scope='module')
def _exact_record_order():
    """This module pins ``load_prediagrams``' exact record order and labels,
    so it always runs against the real loader and typed enumeration: a
    whole-run reordering with ``DAEDALUS_TEST_DIAGRAM_ORDER``
    (``tests/conftest.py``) is undone here for the module's duration."""
    from tests._diagram_order import unwrapped_order_patches
    undo = unwrapped_order_patches()
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in undo]
    for mod, name, value in undo:
        setattr(mod, name, value)
    try:
        yield
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)


def _v1_root(k, ell, tmp_dir):
    """A cache root holding a v1 file for ``(k, ell)``: the local one when it
    exists, else one built into ``tmp_dir`` from the eager enumerator (the
    v1 format IS the eager record list).  Lets the v1/v2 comparisons run on a
    fresh clone instead of skipping."""
    if pdc.v1_exists(CACHE, k, ell):
        return CACHE
    from sage.all import save as sage_save
    root = os.path.join(str(tmp_dir), 'v1root')
    os.makedirs(root, exist_ok=True)
    records = list(pdc._enumerate_eager(k=k, ell=ell, verbose=False)[2])
    sage_save(records, pdc.v1_path(root, k, ell))
    return root


# ── Format mechanics ────────────────────────────────────────────────────────

@pytest.fixture
def fmt_auto(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'auto')


def test_v2_round_trips_through_the_cache(tmp_path, fmt_auto):
    """save -> load -> rebuild -> re-certify is the identity on the cert set."""
    certs = L.stream_prediagram_certs(2, 1, n_procs=1)
    path = pdc.save_v2_certs(str(tmp_path), 2, 1, certs)
    assert os.path.basename(path) == 'prediagrams_v2_k2_l1.pkl'
    assert os.path.basename(os.path.dirname(path)) == pdc.V2_SUBDIR

    assert pdc.load_v2_certs(str(tmp_path), 2, 1) == certs
    records, source = pdc.load_prediagrams(str(tmp_path), 2, 1)
    assert source == 'v2'
    assert pdc.certs_from_records(records) == certs


def test_rebuilt_records_carry_the_v1_conventions(fmt_auto):
    """leaves == 0..k-1, internal == k..|V|-1, and edge labels distinct.

    ``type_assignment`` walks ``leaves`` positionally and keys ``edge_types``
    on ``(u, v, label)``, so both conventions are load-bearing: a mislabelled
    leaf mis-assigns an external field, and two parallel edges sharing a label
    collapse into one propagator.
    """
    for k, ell in [(2, 1), (3, 1), (2, 2)]:
        for blob in sorted(L.stream_prediagram_certs(k, ell, n_procs=1))[:40]:
            D, G, leaves, internal = pdc.record_from_cert(blob)
            assert leaves == list(range(k))
            assert internal == list(range(k, D.order()))
            assert all(D.degree(v) == 1 for v in leaves)
            assert len({(u, v, lbl) for u, v, lbl in D.edges()}) == D.size()
            assert G.size() == D.size() and G.order() == D.order()


def test_rebuild_preserves_the_isomorphism_class(fmt_auto):
    """The cert of a rebuilt record is the cert it was rebuilt from."""
    for k, ell in [(2, 1), (2, 2), (3, 1), (4, 1)]:
        certs = L.stream_prediagram_certs(k, ell, n_procs=1)
        assert pdc.certs_from_records(pdc.records_from_certs(certs)) == certs


@pytest.mark.parametrize('k,ell', [(2, 0), (2, 1), (2, 2), (3, 1), (4, 1)])
def test_v1_files_still_load_and_describe_the_same_set(k, ell, monkeypatch, tmp_path):
    """v1 files (local, or built here from the eager enumerator) keep working,
    and agree with the streamed certs up to isomorphism."""
    root = _v1_root(k, ell, tmp_path)
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v1')
    v1, source = pdc.load_prediagrams(root, k, ell)
    assert source == 'v1'
    assert all(len(rec) == 4 for rec in v1)
    assert pdc.certs_from_records(v1) == L.stream_prediagram_certs(
        k, ell, n_procs=1)


def test_lookup_order_is_v2_then_v1_then_shipped_then_compute(tmp_path, fmt_auto):
    from sage.all import save as sage_save

    root = str(tmp_path)
    # (a) nothing on disk, shipped cell -> shipped, and the hit is copied in as v2
    records, source = pdc.load_prediagrams(root, 2, 1)
    assert source == 'shipped'
    assert pdc.v2_exists(root, 2, 1)
    assert not pdc.v1_exists(root, 2, 1)

    # (a') nothing on disk, cell not shipped -> compute, written back as v2
    assert not pdc.shipped_exists(1, 0)
    assert pdc.load_prediagrams(root, 1, 0)[1] == 'computed'
    assert pdc.v2_exists(root, 1, 0)

    # (b) v2 present -> v2 wins even with a v1 file alongside
    sage_save(list(records), pdc.v1_path(root, 2, 1).removesuffix('.sobj'))
    assert pdc.load_prediagrams(root, 2, 1)[1] == 'v2'

    # (c) v1 only -> v1 beats the shipped file (keeps a machine's old numbers)
    os.remove(pdc.v2_path(root, 2, 1))
    assert pdc.load_prediagrams(root, 2, 1)[1] == 'v1'

    # (d) neither -> shipped again
    os.remove(pdc.v1_path(root, 2, 1))
    assert pdc.load_prediagrams(root, 2, 1)[1] == 'shipped'


def _record_key(rec):
    """Everything a record carries, label-exact: the labelled edges of D, the
    undirected multigraph G, and the leaf / internal vertex lists."""
    D, G, leaves, internal = rec
    return (sorted(D.vertices()), sorted(D.edges()),
            sorted(G.edges(labels=False)), list(leaves), list(internal))


@pytest.fixture
def no_eager_knob(monkeypatch):
    monkeypatch.delenv(pdc._EAGER_ENV, raising=False)


def test_use_cache_false_streams_in_memory_and_touches_no_disk(
        tmp_path, fmt_auto, no_eager_knob, monkeypatch):
    """Opting out of the cache opts out of every file: the records are
    streamed and rebuilt in memory (source 'streamed'), exactly as a v2 file
    of the same cert backend would rebuild them, and nothing is read or
    written -- not the local cache, not the shipped files."""
    def _no_disk(*a, **kw):
        raise AssertionError('use_cache=False touched a prediagram file')
    for name in ('save_v2_certs', 'load_v2_certs', 'load_shipped_certs'):
        monkeypatch.setattr(pdc, name, _no_disk)

    root = str(tmp_path)
    records, source = pdc.load_prediagrams(root, 2, 1, use_cache=False)
    assert source == 'streamed'
    assert len(records) == 9
    assert not pdc.v2_exists(root, 2, 1) and not pdc.v1_exists(root, 2, 1)
    assert os.listdir(root) == []
    want = pdc.records_from_certs(L.stream_prediagram_certs(2, 1, n_procs=1))
    assert [_record_key(r) for r in records] == [_record_key(r) for r in want]


def test_use_cache_false_eager_knob_is_verbatim(tmp_path, fmt_auto,
                                               monkeypatch):
    """``stream=False``, or ``DAEDALUS_PREDIAGRAM_EAGER=1`` (the rollback
    knob) whatever ``stream`` says, gives the eager enumerator's records
    verbatim: same order, same labels, no disk."""
    from engine.enumeration.loop_diagram_enumeration import enumerate_all
    eager = [_record_key(r) for r in enumerate_all(k=2, ell=1,
                                                   verbose=False)[2]]
    root = str(tmp_path)
    monkeypatch.delenv(pdc._EAGER_ENV, raising=False)
    for stream, env in ((False, None), (True, '1'), (False, 'yes')):
        if env is not None:
            monkeypatch.setenv(pdc._EAGER_ENV, env)
        records, source = pdc.load_prediagrams(root, 2, 1, use_cache=False,
                                               stream=stream)
        assert source == 'eager', (stream, env)
        assert [_record_key(r) for r in records] == eager
    assert not pdc.v2_exists(root, 2, 1) and os.listdir(root) == []
    # the knob leaves the cache-on lookup alone ...
    assert pdc.load_prediagrams(root, 2, 1)[1] == 'shipped'
    assert pdc.load_prediagrams(str(tmp_path / 'b'), 2, 1,
                                stream=False)[1] == 'shipped'


@pytest.mark.parametrize('env,stream,want', [
    (None, True, 'streamed'), (None, False, 'eager'),
    ('1', True, 'eager'), ('on', False, 'eager'),
    ('0', False, 'streamed'), (' No ', False, 'streamed'),
    ('', False, 'eager'), ('off', True, 'streamed'),
])
def test_cache_off_source_resolution(env, stream, want, monkeypatch):
    """``DAEDALUS_PREDIAGRAM_EAGER``: unset or empty defers to the caller's
    ``stream``; a true value forces eager and a false one forces streaming
    for every caller."""
    if env is None:
        monkeypatch.delenv(pdc._EAGER_ENV, raising=False)
    else:
        monkeypatch.setenv(pdc._EAGER_ENV, env)
    assert pdc._cache_off_source(stream) == want


def test_cache_off_source_rejects_an_unknown_value(tmp_path, monkeypatch,
                                                    fmt_auto):
    monkeypatch.setenv(pdc._EAGER_ENV, 'sometimes')
    with pytest.raises(ValueError, match='DAEDALUS_PREDIAGRAM_EAGER'):
        pdc.load_prediagrams(str(tmp_path), 2, 1, use_cache=False)
    # a cache-on load never reads the knob
    assert pdc.load_prediagrams(str(tmp_path), 2, 1)[1] == 'shipped'


def test_use_cache_false_never_forks_inside_a_notebook_kernel(
        tmp_path, monkeypatch, no_eager_knob):
    """With ``DAEDALUS_PREDIAGRAM_PROCS`` > 1 the streamed path asks for
    workers; inside a macOS Jupyter kernel the existing fork guard must turn
    that into a warning and a serial run (forking there has crashed the
    machine).  The kernel is simulated; any process pool is a failure."""
    import concurrent.futures

    import engine.fork_safety as fs

    def _no_pool(*a, **kw):
        raise AssertionError('a process pool was started in a notebook kernel')
    monkeypatch.setattr(fs, 'fork_unsafe_in_notebook', lambda *a, **kw: True)
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', _no_pool)
    monkeypatch.setenv(pdc._PROCS_ENV, '4')
    monkeypatch.setattr(pdc, '_STREAMED_MEMO', {})      # force a real stream
    with pytest.warns(UserWarning, match='fork'):
        records, source = pdc.load_prediagrams(str(tmp_path), 2, 1,
                                               use_cache=False)
    assert source == 'streamed' and len(records) == 9
    want = pdc.records_from_certs(L.stream_prediagram_certs(2, 1, n_procs=1))
    assert [_record_key(r) for r in records] == [_record_key(r) for r in want]


def test_use_cache_false_memoises_the_certs_not_the_records(
        tmp_path, monkeypatch, no_eager_knob):
    """A cell is streamed once per process and enumeration setting; every
    call still gets freshly rebuilt records (no shared mutable graphs), and
    another enumeration setting is another memo entry.  The setting varied
    is the orientation checker, which every installation can run; the
    certificate backend is varied too where the compiled extension exists
    (without it only the 'sage' backend runs)."""
    from engine.enumeration import fastenum
    calls = []
    real = pdc.stream_prediagram_certs

    def counting(k, ell, **kw):
        calls.append((k, ell, L.CERT_BACKEND, L.ORIENTATION_CHECKER))
        return real(k, ell, **kw)
    monkeypatch.setattr(pdc, 'stream_prediagram_certs', counting)
    monkeypatch.setattr(pdc, '_STREAMED_MEMO', {})
    root = str(tmp_path)
    a, _ = pdc.load_prediagrams(root, 2, 1, use_cache=False)
    b, _ = pdc.load_prediagrams(root, 2, 1, use_cache=False)
    assert len(calls) == 1
    assert [_record_key(r) for r in a] == [_record_key(r) for r in b]
    assert all(ra[0] is not rb[0] for ra, rb in zip(a, b))
    monkeypatch.setattr(L, 'ORIENTATION_CHECKER',
                        'sage' if L.ORIENTATION_CHECKER == 'integer'
                        else 'integer')
    c, _ = pdc.load_prediagrams(root, 2, 1, use_cache=False)
    assert len(calls) == 2
    assert [_record_key(r) for r in c] == [_record_key(r) for r in a]
    pdc.load_prediagrams(root, 2, 1, use_cache=False)
    assert len(calls) == 2
    if fastenum.available:
        monkeypatch.setattr(L, 'CERT_BACKEND',
                            'sage' if L.CERT_BACKEND == 'fast' else 'fast')
        pdc.load_prediagrams(root, 2, 1, use_cache=False)
        assert len(calls) == 3
    assert len(pdc._STREAMED_MEMO) == len(calls)
    pdc.clear_streamed_memo()
    assert pdc._STREAMED_MEMO == {}


def _sage_certs_match_the_shipped_writer(k, ell):
    """True when THIS installation's Sage canonical labelling re-derives the
    shipped certs byte for byte.  The shipped files were written by the Sage
    backend on an install with the optional bliss package; without bliss
    (or with any other labelling algorithm) Sage canonicalises simple graphs
    to other bytes for the same class."""
    shipped = pdc.load_shipped_certs(k, ell)
    return pdc.recanonicalize(shipped) == set(shipped)


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1)])
def test_sage_backend_streams_exactly_the_shipped_records(
        k, ell, tmp_path, monkeypatch, no_eager_knob):
    """The shipped files hold Sage-backend certs written with bliss.  Under
    that backend (``DAEDALUS_FASTENUM=0``, or ``DAEDALUS_CERT_BACKEND=sage``)
    on a Sage install with bliss, the streamed records are therefore exactly
    the records a fresh clone's cache-on run loads; under the compiled
    'fast' backend, or without bliss, they are other representatives of the
    same classes (``test_shipped_prediagrams`` checks the classes).  Skipped
    when this install's Sage labelling does not re-derive the shipped
    bytes."""
    monkeypatch.setattr(L, 'CERT_BACKEND', 'sage')
    monkeypatch.setattr(pdc, '_STREAMED_MEMO', {})
    if not _sage_certs_match_the_shipped_writer(k, ell):
        pytest.skip('this Sage canonical labelling is not the one that wrote '
                    'the shipped files (no bliss?)')
    streamed, src = pdc.load_prediagrams(str(tmp_path), k, ell,
                                         use_cache=False)
    shipped = pdc.records_from_certs(pdc.load_shipped_certs(k, ell))
    assert src == 'streamed'
    assert [_record_key(r) for r in streamed] == \
           [_record_key(r) for r in shipped]


@pytest.mark.parametrize('k,ell', [
    (2, 0), (2, 1), (2, 2), (3, 1), (4, 0), (4, 1), (1, 2),
    pytest.param(2, 3, marks=pytest.mark.slow),
    pytest.param(3, 2, marks=pytest.mark.slow),
    pytest.param(1, 3, marks=pytest.mark.slow),
])
def test_streamed_records_equal_the_canonicalised_eager_records(
        k, ell, tmp_path, no_eager_knob):
    """The streamed records ARE the eager records sent through the cert round
    trip -- same set, same representatives, same labels, same order -- so the
    switch changes only which labelled member of each class is used."""
    records, source = pdc.load_prediagrams(str(tmp_path), k, ell,
                                           use_cache=False)
    assert source == 'streamed'
    eager = list(pdc._enumerate_eager(k=k, ell=ell, verbose=False)[2])
    canon = pdc.records_from_certs(pdc.certs_from_records(eager))
    assert len(records) == len(eager) == len(canon)
    assert [_record_key(r) for r in records] == [_record_key(r) for r in canon]


@pytest.mark.parametrize('k,ell', [(2, 1), (2, 2), (3, 1)])
def test_streamed_records_equal_a_computed_v2_load(k, ell, tmp_path,
                                                   monkeypatch, no_eager_knob):
    """Cache off (streamed) and a cold cache-on v2 compute hand the pipeline
    the SAME records: same certs, same rebuild, same order.  (The shipped
    files are hidden so the cell is computed; they hold certs written by a
    different canonical-labelling backend, see the module docstring.)"""
    streamed, s1 = pdc.load_prediagrams(str(tmp_path / 'a'), k, ell,
                                        use_cache=False)
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v2')
    monkeypatch.setattr(pdc, 'shipped_exists', lambda kk, ll: False)
    computed, s2 = pdc.load_prediagrams(str(tmp_path / 'b'), k, ell)
    assert (s1, s2) == ('streamed', 'computed')
    assert [_record_key(r) for r in streamed] == \
           [_record_key(r) for r in computed]


def _ou_quartic_curves(monkeypatch, max_ell, taus, eager):
    """compute_cumulants for OU quartic k=2 with every cache bypassed, the
    prediagrams served eagerly or streamed; also returns the sources seen."""
    import importlib.util

    import numpy as np

    from api.compute import compute_cumulants

    spec = importlib.util.spec_from_file_location(
        'ou_quartic_model', 'models/ou_quartic.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build()
    sources = set()
    orig_load = pdc.load_prediagrams

    def spy(root, k, ell, **kw):
        recs, src = orig_load(root, k, ell, **kw)
        sources.add(src)
        return recs, src

    with monkeypatch.context() as mp:
        mp.setattr(pdc, 'load_prediagrams', spy)
        # the temporal path defaults to eager (TEMPORAL_CACHE_OFF_STREAMS);
        # set the knob both ways so the test does not depend on that default
        mp.setenv(pdc._EAGER_ENV, '1' if eager else '0')
        res = compute_cumulants(model, k=2, max_ell=max_ell,
                                external_fields=[('dx', 1)] * 2,
                                tau_grid=np.asarray(taus), use_cache=False,
                                parallel=False, verbose=False)
    return {ell: np.asarray(c) for ell, c in res['C_tau_by_ell'].items()}, \
        sources


@pytest.mark.parametrize('max_ell', [2, pytest.param(3, marks=pytest.mark.slow)])
def test_streamed_totals_match_eager_through_compute_cumulants(max_ell,
                                                               monkeypatch):
    """Where every region is analytic, streaming moves totals only by
    rounding: streamed vs eager records through the real wiring
    (``compute_cumulants(use_cache=False)``).  Measured max relative gap at
    max_ell=3: 4.6e-16."""
    import numpy as np

    taus = [0.0, 0.5, 1.0, 3.0]
    s, src_s = _ou_quartic_curves(monkeypatch, max_ell, taus, eager=False)
    e, src_e = _ou_quartic_curves(monkeypatch, max_ell, taus, eager=True)
    assert (src_s, src_e) == ({'streamed'}, {'eager'})
    assert sorted(s) == sorted(e) == list(range(max_ell + 1))
    for ell in s:
        assert np.all(np.isfinite(s[ell])) and np.any(s[ell] != 0)
        np.testing.assert_allclose(s[ell], e[ell], rtol=1e-13, atol=1e-15)


def _sources_seen(monkeypatch, run):
    """The ``load_prediagrams`` sources that ``run()`` hit."""
    seen = set()
    orig = pdc.load_prediagrams

    def spy(root, k, ell, **kw):
        recs, src = orig(root, k, ell, **kw)
        seen.add(src)
        return recs, src
    with monkeypatch.context() as mp:
        mp.setattr(pdc, 'load_prediagrams', spy)
        run()
    return seen


def _temporal_run():
    import numpy as np

    from api.compute import compute_cumulants
    import daedalus as dd
    model, _ = dd.load_model('ou_quartic')
    compute_cumulants(model, k=2, max_ell=1, external_fields=[('dx', 1)] * 2,
                      tau_grid=np.asarray([0.0, 1.0]), use_cache=False,
                      parallel=False, verbose=False)


def _spatial_run():
    import numpy as np

    from api.compute import compute_cumulants
    import daedalus as dd
    model, _ = dd.load_model('reaction_diffusion_2d')
    compute_cumulants(model, k=2, max_ell=1,
                      external_fields=[('dphi', 1)] * 2,
                      parameters={'mu': 1.0, 'D': 1.0, 'g': 0.2, 'T': 1.0},
                      tau_grid=np.asarray([0.0]),
                      chi_grid=np.asarray([0.4, 1.0]), use_cache=False,
                      verbose=False, spatial_parallel=False)


def test_cache_off_sources_of_the_temporal_and_spatial_paths(monkeypatch):
    """Cache off, the temporal compute path keeps the eager records
    (``TEMPORAL_CACHE_OFF_STREAMS`` is False: its nquad fallback is not
    representative-independent) and the spatial path streams.
    ``DAEDALUS_PREDIAGRAM_EAGER`` overrides both, either way."""
    assert pdc.TEMPORAL_CACHE_OFF_STREAMS is False
    monkeypatch.delenv(pdc._EAGER_ENV, raising=False)
    assert _sources_seen(monkeypatch, _temporal_run) == {'eager'}
    assert _sources_seen(monkeypatch, _spatial_run) == {'streamed'}
    monkeypatch.setattr(pdc, 'TEMPORAL_CACHE_OFF_STREAMS', True)
    assert _sources_seen(monkeypatch, _temporal_run) == {'streamed'}
    monkeypatch.setattr(pdc, 'TEMPORAL_CACHE_OFF_STREAMS', False)
    monkeypatch.setenv(pdc._EAGER_ENV, '0')
    assert _sources_seen(monkeypatch, _temporal_run) == {'streamed'}
    monkeypatch.setenv(pdc._EAGER_ENV, '1')
    assert _sources_seen(monkeypatch, _spatial_run) == {'eager'}


class _FallbackRepresentativeGap(Exception):
    """Raised only by the gap check of the strict xfails below, so any other
    failure of that test (a crash, a gap above the documented bound) is a
    real failure."""


def _representative_gap(measured):
    return pytest.mark.xfail(
        strict=True, raises=_FallbackRepresentativeGap, reason=(
            'Phase J nquad fallback is representative-dependent (streamed vs '
            'eager records, fallback of commit d68e383, measured 2026-10-06; '
            'a change to the fallback changes these '
            f'numbers): {measured}  Flip prediagram_cache.'
            'TEMPORAL_CACHE_OFF_STREAMS only when every case of this test '
            'XPASSes.'))


#: The models whose regions reach the ``scipy.nquad`` fallback.  Measured
#: 2026-10-06 with the fallback of commit d68e383 (a change to the fallback
#: changes these numbers): the two spike-reset cases agree to rtol 1e-13
#: (they pass); ``quad_exp`` still differs, by 1.21e-11 relative.  Every
#: case must pass, not just some, before ``TEMPORAL_CACHE_OFF_STREAMS``
#: flips.
_FALLBACK_GATE_CASES = [
    pytest.param(
        'single_population_spike_reset_test', 'P_SPIKE', (0.0, 10.0), False,
        id='spike_reset-perdiag'),
    pytest.param(
        'single_population_spike_reset_test', 'P_SPIKE', (0.0, 10.0), True,
        id='spike_reset-grouped'),
    pytest.param(
        'single_population_quad_exp_test', 'P_SP', (0.0, 1.0, 5.0), False,
        id='quad_exp-perdiag', marks=_representative_gap(
            'single_population_quad_exp_test k=2 ell<=1 (P_SP), '
            'tau=(0, 1, 5): C_tau 1.21e-11 relative (about 400 s).')),
]


@pytest.mark.slow
@pytest.mark.parametrize('model_name, params, taus, grouped',
                         _FALLBACK_GATE_CASES)
def test_temporal_streamed_totals_match_eager_on_a_fallback_model(
        model_name, params, taus, grouped, monkeypatch):
    """The gate of ``TEMPORAL_CACHE_OFF_STREAMS``: streamed and eager records
    give the same totals to rtol 1e-13 on the models whose regions reach the
    ``scipy.nquad`` fallback.  Today they differ by the fallback's error
    (strict xfails); a gap above the documented 1e-6 is a real failure.
    Flip the flag only when every case passes unexpectedly: the cases do
    not move together (a tighter fallback can close one gap and not
    another)."""
    import numpy as np

    import daedalus as dd
    from api.compute import compute_cumulants
    from tests.tools import phase_j_zoo_baseline as zoo

    model, _ = dd.load_model(model_name)
    curves = {}
    for env in ('1', '0'):
        monkeypatch.setenv(pdc._EAGER_ENV, env)
        res = compute_cumulants(
            model, k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)],
            parameters=getattr(zoo, params), tau_grid=np.asarray(taus),
            use_cache=False, parallel=False, verbose=False,
            use_grouped_phase_j=grouped)
        curves[env] = np.asarray(res['C_tau'])
    e, s = curves['1'], curves['0']
    assert np.all(np.isfinite(e)) and np.all(e != 0)
    rel = float(np.max(np.abs(s - e) / np.abs(e)))
    assert rel <= 1e-6, f'C_tau gap {rel:.3e} above the documented bound'
    if rel > 1e-13:
        raise _FallbackRepresentativeGap(f'C_tau gap {rel:.3e}')


def test_format_override_is_validated(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v3')
    with pytest.raises(ValueError):
        pdc.cache_format()


def test_shipped_v2_file_is_picked_up():
    """The (1,4) cell ships with the package and must be found by name."""
    path = pdc.shipped_path(1, 4)
    assert os.path.isfile(path), path
    certs, stamp = pdc.read_cert_file(path, 1, 4)
    assert isinstance(certs, set) and len(certs) == 22332
    assert stamp['convention'] == pdc.CONVENTION
    # Spot-check the rebuild rather than materialising all 22k Sage graphs.
    # Compared up to isomorphism: the bytes depend on the canonical-labelling
    # backend of the machine that wrote the file (module docstring).
    for blob in sorted(certs)[:50]:
        D, _, leaves, _ = pdc.record_from_cert(blob)
        assert leaves == [0]
        c = L._iso_cert(D)
        assert L._iso_cert(L.cert_to_graph(c, directed=True)) == c
        assert D.is_isomorphic(L.cert_to_graph(L.unpack_cert(blob), directed=True))


# ── Numerical identity ──────────────────────────────────────────────────────

def _ctx(k, max_ell, name):
    """OU quartic as a text-driven model, plus everything typing needs."""
    from engine.core.field_theory import FieldTheory
    from engine.core.vertices import extract_source_types, extract_vertex_types
    from engine.diagrams.type_assignment import build_field_index_map
    from api._propagator import build_propagator, compute_poles_and_residues
    from api.model import TemporalModelBuilder

    eps = 0.05
    model = (TemporalModelBuilder(name).physical_field('x')
             .parameter('mu', default=1.0, domain='positive')
             .parameter('T', default=1.0, domain='positive')
             .parameter('eps', default=eps)
             .set_action_text('xt*((Dt+mu)*x + eps*x^3) - T*xt^2')
             .equation(lhs='(Dt+mu)*x + eps*x^3', rhs='0').build())
    ft = FieldTheory(model, taylor_order=max(k + 2 * max_ell, 4))
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    num_params = {SR.var('mu'): 1.0, SR.var('T'): 1.0,
                  SR.var('xstar1'): 0.0, SR.var('eps'): eps}
    compute_poles_and_residues(prop, num_params, verbose=False)
    resp_idx, phys_idx = build_field_index_map(
        list(ft._ns._ring_var_names), ft._n_tilde)
    return dict(prop=prop, num_params=num_params, k=k, eps=eps,
                resp_idx=resp_idx, phys_idx=phys_idx,
                vtypes=extract_vertex_types(ft),
                stypes=extract_source_types(ft),
                ext=[('dx', 1)] * k)


def _stages(ctx, records):
    """prediagrams -> typed -> causal -> unique, plus the label-free invariants."""
    from engine.diagrams.causality import filter_causal
    from engine.diagrams.symmetry import (combinatorial_factor,
                                          deduplicate_with_multiplicities,
                                          diagram_signature,
                                          external_wick_compensation)
    from engine.diagrams.type_assignment import enumerate_all as typed_all

    typed = typed_all(records, ctx['ext'], ctx['vtypes'], ctx['stypes'],
                      G_ft=ctx['prop']['G_ft'], resp_index=ctx['resp_idx'],
                      phys_index=ctx['phys_idx'], parallel=False)
    causal, _, _ = filter_causal(typed)
    unique, mult = deduplicate_with_multiplicities(causal)
    return dict(
        n_typed=len(typed), n_causal=len(causal), n_unique=len(unique),
        mult=sorted(mult),
        sigs=sorted(str(diagram_signature(td)) for td in unique),
        factors=sorted((int(combinatorial_factor(td)),
                        int(external_wick_compensation(td)))
                       for td in unique),
        unique=unique)


def _total(ctx, unique, taus):
    from engine.diagrams.symmetry import classify_coefficient_factors
    from engine.integration.time_domain.final_integral import integrate_diagram

    k = ctx['k']
    pdic = {key: ctx['prop'][key] for key in (
        'K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta',
        't_var', 'omega', 'nf', 'pole_vals', 'C_mats')}
    tvars = [SR.var('t%d' % i) for i in range(1, k + 1)]
    out = [0.0] * len(taus)
    for td in unique:
        info = classify_coefficient_factors(
            td, [], {'temporal_type': 'white', 'amplitude_params': []})
        pref = SR(info['scalar_prefactor'])
        if abs(float(pref.subs(ctx['num_params']))) < 1e-14:
            continue
        res = integrate_diagram(td, pdic, pref, tvars,
                                num_params=ctx['num_params'],
                                external_fields=ctx['ext'])
        for i, t in enumerate(taus):
            args = [0.0, t] + [0.0] * (k - 2)
            out[i] += complex(res['contribution'](*args)).real
    return out


def _both_sources(ctx, k, ell, monkeypatch, tmp_root):
    """Records for ``(k, ell)`` from the shipped v1 file and from v2.

    The v2 side is rooted in ``tmp_root``, NOT in the shipped cache: writing
    a v2 file next to a v1 file would flip that cell to v2 for every later run
    on this machine, which is exactly the silent behaviour change these tests
    exist to characterise.
    """
    root1 = _v1_root(k, ell, tmp_root)
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v1')
    v1, s1 = pdc.load_prediagrams(root1, k, ell)
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v2')
    v2, s2 = pdc.load_prediagrams(os.path.join(str(tmp_root), 'v2root'), k, ell)
    assert s1 == 'v1' and s2 in ('v2', 'shipped', 'computed')
    assert len(v1) == len(v2)
    return v1, v2


#: Known (prediagram, unique-diagram) counts for OU quartic, so an equality
#: test cannot pass by comparing two empty pipelines.
EXPECTED = {(2, 0): (1, 1), (2, 1): (9, 4), (2, 2): (283, 66),
            (4, 0): (13, 3), (4, 1): (755, 91)}


@pytest.mark.parametrize('ell', [0, 1])
def test_v1_and_v2_give_the_same_diagram_set_k2(ell, tmp_path, monkeypatch):
    """Exact equality of everything that is not a floating-point rounding."""
    ctx = _ctx(2, 1, 'pdcache-k2')
    v1, v2 = _both_sources(ctx, 2, ell, monkeypatch, tmp_path)
    a, b = _stages(ctx, v1), _stages(ctx, v2)
    n_pd, n_uniq = EXPECTED[(2, ell)]
    assert len(v1) == len(v2) == n_pd
    assert a['n_unique'] == b['n_unique'] == n_uniq
    assert (a['n_typed'], a['n_causal'], a['n_unique']) == \
           (b['n_typed'], b['n_causal'], b['n_unique'])
    assert a['sigs'] == b['sigs']
    assert a['mult'] == b['mult']
    assert a['factors'] == b['factors']


@pytest.mark.parametrize('ell', [0, 1])
def test_v1_and_v2_totals_agree_to_rounding_k2(ell, tmp_path, monkeypatch):
    """Measured max gap is 1 ULP; 1e-13 relative leaves ~450x headroom."""
    taus = [0.0, 1.0, 3.0]
    ctx = _ctx(2, 1, 'pdcache-k2')
    v1, v2 = _both_sources(ctx, 2, ell, monkeypatch, tmp_path)
    ta = _total(ctx, _stages(ctx, v1)['unique'], taus)
    tb = _total(ctx, _stages(ctx, v2)['unique'], taus)
    for x, y in zip(ta, tb):
        assert x == pytest.approx(y, rel=1e-13, abs=1e-15), (ell, ta, tb)
    # ell=0 is the bare propagator: exp(-mu*tau) with mu = 1.
    if ell == 0:
        assert ta[1] == pytest.approx(0.36787944117144233, rel=1e-12)


def test_leaf_swap_reproduces_the_v2_total_bit_for_bit(tmp_path, monkeypatch):
    """The residual v1/v2 gap IS leaf order, and nothing else.

    Take the v1 records at (k=2, ell=1), swap leaf 0 with leaf 1 and change
    nothing else, and the total becomes the v2 total exactly --- same bits.
    That identifies the gap as IEEE rounding in a leaf-ordered integrand
    assembly, not as a diagram the v2 format got wrong.
    """
    from sage.all import DiGraph, Graph

    taus = [3.0]                       # where the last bit actually differs
    ctx = _ctx(2, 1, 'pdcache-leafswap')
    v1, v2 = _both_sources(ctx, 2, 1, monkeypatch, tmp_path)

    def swap_leaves(rec):
        D, G, leaves, internal = rec
        p = {0: 1, 1: 0}
        p.update({v: v for v in D.vertices() if v > 1})
        D2 = DiGraph(multiedges=True, loops=False)
        D2.add_vertices([p[v] for v in D.vertices()])
        for u, v, lbl in D.edges():
            D2.add_edge(p[u], p[v], lbl)
        G2 = Graph(multiedges=True, loops=False)
        G2.add_vertices([p[v] for v in G.vertices()])
        for u, v in G.edges(labels=False):
            G2.add_edge(p[u], p[v])
        return (D2, G2, sorted(p[v] for v in leaves),
                sorted(p[v] for v in internal))

    t_v1 = _total(ctx, _stages(ctx, v1)['unique'], taus)
    t_v2 = _total(ctx, _stages(ctx, v2)['unique'], taus)
    t_swap = _total(ctx, _stages(ctx, [swap_leaves(r) for r in v1])['unique'],
                    taus)
    # The claim under test: leaf order accounts for the gap COMPLETELY.
    assert [x.hex() for x in t_swap] == [x.hex() for x in t_v2], (
        f'leaf swap gave {t_swap}, v2 gave {t_v2}')
    if t_v1 == t_v2:
        pytest.skip('v1 and v2 now agree bit for bit here; the demonstration '
                    'is vacuous but nothing is wrong -- re-derive the '
                    'documented 1-ULP example if this persists')


def test_compute_cumulants_agrees_through_the_real_wiring(tmp_path,
                                                          monkeypatch):
    """The A/B above bypasses ``api._diagrams``; this one goes through it.

    ``compute_cumulants`` -> ``enumerate_unique_diagrams`` ->
    ``load_prediagrams`` is the path a user actually takes, and its
    typed-diagram cache has to be cold on both runs or the second run just
    reloads the first one's answer -- hence the per-run model cache dir.
    The v2 run is rooted in tmp for the same reason ``_both_sources`` is:
    a test must not leave a v2 file that flips a shipped cell.
    """
    import importlib.util

    import numpy as np

    import api._diagrams as diagrams_mod
    from api.compute import compute_cumulants

    spec = importlib.util.spec_from_file_location(
        'ou_quartic_model', 'models/ou_quartic.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build()

    taus = np.array([0.0, 1.0, 3.0])

    def curves(fmt, slot, pd_root):
        monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', fmt)
        monkeypatch.setattr(diagrams_mod, 'PREDIAGRAM_CACHE_ROOT', pd_root)
        monkeypatch.setattr(
            diagrams_mod, '_model_cache_dir',
            lambda m, t, c, _s=slot: f'{tmp_path}/{_s}/{m["name"]}')
        res = compute_cumulants(model, k=2, max_ell=1,
                                external_fields=[('dx', 1)] * 2,
                                tau_grid=taus, use_cache=True,
                                parallel=False, verbose=False)
        return {ell: [complex(fn(0.0, t)).real for t in taus]
                for ell, fn in res['total_C_by_ell'].items()}

    # Guard: this test's v2 run must write only under tmp_path.  On a fresh
    # clone earlier tests can legitimately have populated the real cache at
    # (2,0) (a compute miss, or the copy of the shipped cell), so compare
    # before/after rather than asserting absence.
    had_v2 = pdc.v2_exists(CACHE, 2, 0)
    a = curves('v1', 'a', CACHE)
    b = curves('v2', 'b', f'{tmp_path}/pd')
    assert pdc.v2_exists(CACHE, 2, 0) == had_v2, 'the test wrote into the real cache'
    assert pdc.v2_exists(f'{tmp_path}/pd', 2, 0)
    assert sorted(a) == sorted(b) == [0, 1]
    for ell in a:
        for x, y in zip(a[ell], b[ell]):
            assert x == pytest.approx(y, rel=1e-13, abs=1e-15), (ell, a, b)
    # Tree cumulant of the OU process: C(tau) = (D/mu) * exp(-mu*tau), D=mu=1.
    assert a[0][1] == pytest.approx(0.36787944117144233, rel=1e-9)
    assert a[1][0] != 0.0, 'the 1-loop correction must not be identically zero'


@pytest.mark.slow
def test_v1_and_v2_agree_at_k4(tmp_path, monkeypatch):
    """k=4: exact diagram set, and both paths hit the Boltzmann series.

    kappa_4 = -6*eps + 126*eps^2 exactly, so the tree and 1-loop totals are
    independent anchors on top of the v1-vs-v2 comparison.
    """
    eps = 0.05
    ctx = _ctx(4, 1, 'pdcache-k4')
    exact = {0: -6 * eps, 1: 126 * eps ** 2}
    for ell in (0, 1):
        v1, v2 = _both_sources(ctx, 4, ell, monkeypatch, tmp_path)
        a, b = _stages(ctx, v1), _stages(ctx, v2)
        n_pd, n_uniq = EXPECTED[(4, ell)]
        assert len(v1) == len(v2) == n_pd
        assert a['n_unique'] == b['n_unique'] == n_uniq
        assert (a['n_typed'], a['n_causal'], a['n_unique']) == \
               (b['n_typed'], b['n_causal'], b['n_unique'])
        assert a['sigs'] == b['sigs']
        assert a['factors'] == b['factors']
        ta = _total(ctx, a['unique'], [0.0])[0]
        tb = _total(ctx, b['unique'], [0.0])[0]
        assert ta == pytest.approx(tb, rel=1e-13)
        assert ta == pytest.approx(exact[ell], abs=1e-12)
        assert tb == pytest.approx(exact[ell], abs=1e-12)


@pytest.mark.slow
def test_v1_and_v2_agree_at_k2_two_loop(tmp_path, monkeypatch):
    ctx = _ctx(2, 2, 'pdcache-k2-l2')
    v1, v2 = _both_sources(ctx, 2, 2, monkeypatch, tmp_path)
    a, b = _stages(ctx, v1), _stages(ctx, v2)
    n_pd, n_uniq = EXPECTED[(2, 2)]
    assert len(v1) == len(v2) == n_pd
    assert a['n_unique'] == b['n_unique'] == n_uniq
    assert (a['n_typed'], a['n_causal'], a['n_unique']) == \
           (b['n_typed'], b['n_causal'], b['n_unique'])
    assert a['sigs'] == b['sigs']
    assert a['factors'] == b['factors']
    taus = [0.0, 1.0, 3.0]
    ta = _total(ctx, a['unique'], taus)
    tb = _total(ctx, b['unique'], taus)
    for x, y in zip(ta, tb):
        assert x == pytest.approx(y, rel=1e-13, abs=1e-15), (ta, tb)

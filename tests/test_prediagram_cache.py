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
def test_v1_files_still_load_and_describe_the_same_set(k, ell, monkeypatch):
    """The 16 shipped v1 files keep working, and agree with v2 up to iso."""
    if not pdc.v1_exists(CACHE, k, ell):
        pytest.skip(f'v1 cache for ({k},{ell}) not built')
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v1')
    v1, source = pdc.load_prediagrams(CACHE, k, ell)
    assert source == 'v1'
    assert all(len(rec) == 4 for rec in v1)
    assert pdc.certs_from_records(v1) == L.stream_prediagram_certs(
        k, ell, n_procs=1)


def test_lookup_order_is_v2_then_v1_then_compute(tmp_path, fmt_auto):
    from sage.all import save as sage_save

    root = str(tmp_path)
    # (a) nothing on disk -> compute, and the miss is written back as v2
    records, source = pdc.load_prediagrams(root, 2, 1)
    assert source == 'computed'
    assert pdc.v2_exists(root, 2, 1)
    assert not pdc.v1_exists(root, 2, 1)

    # (b) v2 present -> v2 wins even with a v1 file alongside
    sage_save(list(records), pdc.v1_path(root, 2, 1).removesuffix('.sobj'))
    assert pdc.load_prediagrams(root, 2, 1)[1] == 'v2'

    # (c) v1 only -> v1
    os.remove(pdc.v2_path(root, 2, 1))
    assert pdc.load_prediagrams(root, 2, 1)[1] == 'v1'


def test_use_cache_false_uses_the_eager_path_and_touches_no_disk(tmp_path,
                                                                 fmt_auto):
    """Opting out of the cache must opt out of the cert round trip too.

    ``build_pipeline_records`` (spatial) calls with ``use_cache=False`` and
    then indexes the diagram list positionally, so re-ordering or
    canonically relabelling these records changes WHICH diagram it picks --
    it broke ``test_full_integrator`` when this path streamed instead.
    """
    root = str(tmp_path)
    records, source = pdc.load_prediagrams(root, 2, 1, use_cache=False)
    assert source == 'eager'
    assert not pdc.v2_exists(root, 2, 1)
    assert len(records) == 9
    # Verbatim eager output: same objects, same order, same labels.
    from engine.enumeration.loop_diagram_enumeration import enumerate_all
    eager = enumerate_all(k=2, ell=1, verbose=False)[2]
    assert [sorted(r[0].edges()) for r in records] == \
           [sorted(r[0].edges()) for r in eager]
    assert [list(r[2]) for r in records] == [list(r[2]) for r in eager]


def test_format_override_is_validated(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v3')
    with pytest.raises(ValueError):
        pdc.cache_format()


def test_precomputed_streaming_v2_files_are_picked_up():
    """The expensive cells already streamed to disk must be found by name."""
    path = pdc.v2_path(CACHE, 1, 4)
    if not os.path.isfile(path):
        pytest.skip('streaming_v2/prediagrams_v2_k1_l4.pkl not built')
    with open(path, 'rb') as f:
        certs = pickle.load(f)
    assert isinstance(certs, set) and len(certs) == 22332
    # Spot-check the rebuild rather than materialising all 22k Sage graphs.
    for blob in sorted(certs)[:50]:
        D, _, leaves, _ = pdc.record_from_cert(blob)
        assert leaves == [0]
        assert L.pack_cert(L._iso_cert(D)) == blob


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
    if not pdc.v1_exists(CACHE, k, ell):
        pytest.skip(f'v1 cache for ({k},{ell}) not built')
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v1')
    v1, s1 = pdc.load_prediagrams(CACHE, k, ell)
    monkeypatch.setenv('DAEDALUS_PREDIAGRAM_FORMAT', 'v2')
    v2, s2 = pdc.load_prediagrams(str(tmp_root), k, ell)
    assert s1 == 'v1' and s2 in ('v2', 'computed')
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

    a = curves('v1', 'a', CACHE)
    b = curves('v2', 'b', f'{tmp_path}/pd')
    assert not pdc.v2_exists(CACHE, 2, 0), 'the test wrote into the real cache'
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

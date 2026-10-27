"""M5 (P6): the per-diagram setup levers are pure speed-ups.

* L1  a numerically-zero prefactor returns before any setup;
* L2  model-level data built once per ``compute_correction_td`` call;
* L3  SR objects built on demand;
* L5  shared automorphism work (``daedalus.engine.diagrams.symmetry``).

Every lever is always on: their switches (``USE_SETUP_ZERO_EXIT``,
``USE_SETUP_PROP_TD``, ``USE_SETUP_LAZY_SR``, ``symmetry.USE_AUT_MEMO``, the
``DAEDALUS_SETUP_*`` variables and the umbrella
``DAEDALUS_PHASE_J_LEGACY_SETUP``) were retired in audit W7, after every
result with a lever on had been checked bit-identical (``np.array_equal``)
to the lever off on these models.  What is checked now: the levers fire,
the zero exit gives the structure and values of the full path (still
reachable through ``integrate_diagram(_setup_full=...)``), a shared
``PropagatorTD`` gives the values of a per-diagram build, the lazy SR
objects equal the eager ones (``G_t_entry``), and the memo the uncached
order.

Run with::

    PYTHONHASHSEED=0 python -m pytest tests/test_phase_j_setup_fastpath.py -v
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import daedalus as dd
import daedalus.api.compute as AC
import daedalus.engine.integration.time_domain.final_integral as FI
import daedalus.engine.integration.time_domain.pipeline as PL

_TAUS = np.linspace(0.0, 3.0, 7)
_P2 = [(0.0, 0.7), (0.0, 1.9), (0.0, -1.1)]
_P4 = [(0.0, 0.3, 0.6, 0.9), (0.0, 0.5, 0.4, 0.7)]
import daedalus.engine.diagrams.symmetry as SYM
_LH_PARAMS = {'E': [0.78, 0.81], 'w': [[0.30, 0.25], [0.30, 0.35]],
              'tau': 10.0, 'a': 1.0, 'tau_g': 2.5}


# ── helpers ────────────────────────────────────────────────────────────────
def _model(key):
    return dd.load_model(key)[0]


def _run(key, k, ell, ext_field='dx', model=None, **kw):
    """``compute_cumulants`` -> dict of arrays (per-ell totals, per-diagram
    values at fixed points) and the Phase J counters."""
    FI._reset_runtime_counters()
    ext = [(ext_field, 1)] * k
    with contextlib.redirect_stdout(io.StringIO()):
        res = AC.compute_cumulants(
            _model(key) if model is None else model, k=k, max_ell=ell,
            tau_grid=_TAUS, use_cache=True,
            parallel=False, verbose=False, external_fields=ext, **kw)
    pts = _P2 if k == 2 else _P4
    out = {}
    if k == 2:
        out['C_tau'] = np.asarray(res['C_tau'])
        for e, a in res['C_tau_by_ell'].items():
            out[f'C_tau_ell{e}'] = np.asarray(a)
    else:
        for e, f in res['total_C_by_ell'].items():
            out[f'total_ell{e}'] = np.array([f(*p) for p in pts], complex)
    for e, v in res['phase_j_by_ell'].items():
        groups = v['groups']
        arr = np.zeros((len(groups), len(pts)), complex)
        for i, g in enumerate(groups):
            for j, p in enumerate(pts):
                arr[i, j] = g['contribution'](*p)
        out[f'perdiag_ell{e}'] = arr
    return out, dict(FI._RUNTIME_COUNTERS)


def _same(a, b):
    return (set(a) == set(b)
            and all(np.array_equal(a[k], b[k]) for k in a))


@pytest.fixture(autouse=True)
def _fresh():
    """An empty automorphism memo and zeroed counters."""
    SYM._aut_memo_clear()
    FI._reset_runtime_counters()
    yield
    SYM._aut_memo_clear()
    FI._reset_runtime_counters()


@pytest.fixture(scope='module')
def captured_ou():
    """The keyword arguments of every ``integrate_diagram`` call of
    ``ou_quartic`` k=2, ell <= 2 (the real inputs: typed diagram, propagator
    data, prefactor, num_params)."""
    calls = []
    orig = PL.integrate_diagram

    def spy(**kw):
        # the per-call object (L2) is the pipeline's, not an input of the diagram
        calls.append({k: v for k, v in kw.items() if k != 'prop_td'})
        return orig(**kw)

    PL.integrate_diagram = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            AC.compute_cumulants(
                _model('ou_quartic'), k=2, max_ell=2, tau_grid=_TAUS,
                use_cache=True, parallel=False, verbose=False,
                external_fields=[('dx', 1)] * 2)
    finally:
        PL.integrate_diagram = orig
    assert len(calls) > 60
    return calls


def _full_path(c):
    """``integrate_diagram`` without the zero exit (the full setup: what the
    lazy keys of a zero-exit result re-run)."""
    return FI.integrate_diagram(_setup_full=FI._next_diagram_serial(), **c)


# ── L1: zero-prefactor early exit ──────────────────────────────────────────
@pytest.mark.parametrize('key,k,ell', [('ou_quartic', 2, 2),
                                       ('ou_quartic', 4, 1),
                                       ('ou_quartic_colored', 2, 1)])
def test_l1_fires_and_two_runs_agree(key, k, ell):
    """The zero exit fires, and the values (every per-diagram slot kept)
    are main's: frozen at dca8804, where the retired DAEDALUS_SETUP_ZERO_EXIT
    off path gave them bit for bit (tests/fixtures/w7_default_values.npz)."""
    from tests.test_phase_j_structural_zeros import _frozen_close
    _run(key, k, ell)                          # warm the disk cache
    on, c_on = _run(key, k, ell)
    again, _c = _run(key, k, ell)
    assert _same(on, again)
    _frozen_close(f'l1|{key}|{k}|{ell}|', on)
    assert c_on['setup_zero_exit'] > 0                 # it fired


def _two_field_ou():
    """A plain two-field coupled OU with quartic terms, inline."""
    from daedalus.api.model import TemporalModelBuilder
    return (TemporalModelBuilder('two-field OU quartic (test)', n_populations=0)
            .physical_field('x', indexed=True)
            .physical_field('y', indexed=True)
            .parameter('mu1', default=1.0, domain='real')
            .parameter('mu2', default=1.0, domain='real')
            .parameter('eps1', default=0.05, domain='positive')
            .parameter('eps2', default=0.05, domain='positive')
            .parameter('J1', default=0.5, domain='real')
            .parameter('J2', default=0.5, domain='real')
            .parameter('D1', default=1.0, domain='positive')
            .parameter('D2', default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu1)*x + eps1*x^3 - J1*y) - D1*xt^2 '
                             '+ yt*((Dt+mu2)*y + eps2*y^3 - J2*x) - D2*yt^2')
            .mf_residual(lhs='(Dt+mu1)*x', rhs='-eps1*x^3 + J1*y')
            .mf_residual(lhs='(Dt+mu2)*y', rhs='-eps2*y^3 + J2*x')
            .stability_analysis(True)
            .build())


def test_l1_counts_on_the_two_field_ou(monkeypatch):
    """The two-field OU is the motivating case (most diagrams have a zero
    prefactor); an inline plain model, so it runs in every checkout."""
    m = _two_field_ou()
    _run('two_field_ou', 2, 1, model=m)
    _on, c = _run('two_field_ou', 2, 1, model=m)
    assert c['setup_zero_exit'] > 10


def test_l1_never_exits_for_a_noise_or_conv_vertex(captured_ou):
    """A diagram with a noise-source (``cumulant_specs``) or a
    ``ConvVertexType`` vertex keeps the full path: its kernel is substituted
    into the prefactor, so the plain prefactor decides nothing.  (No model in
    this checkout reaches those branches of ``integrate_diagram``, so the
    decision is tested on a real zero-prefactor diagram with such a vertex
    added.)"""
    import types
    from daedalus.engine.core.vertices import ConvVertexType, NoiseSourceType
    c = _zero_diagram(captured_ou)
    td, pd = c['typed_diagram'], c['propagator_data']
    leaves = list(td.prediagram[2])
    zero, ef = FI.SR(0), c['external_fields']
    assert FI._zero_exit_applies(td, pd, zero, ef, leaves)

    def with_vertex(vtype):
        assigned = dict(td.vertex_assignments)
        assigned[max(assigned, default=0) + 1] = vtype
        return types.SimpleNamespace(vertex_assignments=assigned)

    noise = NoiseSourceType(FI.SR(1), ['xt', 'xt'], (0, 2),
                            [{'legs': (0, 0)}])
    assert not FI._zero_exit_applies(with_vertex(noise), pd, zero, ef, leaves)
    # a NoiseSourceType without cumulant specs is a plain source
    plain = NoiseSourceType(FI.SR(1), ['xt', 'xt'], (0, 2), [])
    assert FI._zero_exit_applies(with_vertex(plain), pd, zero, ef, leaves)
    conv = ConvVertexType.__new__(ConvVertexType)
    assert not FI._zero_exit_applies(with_vertex(conv), pd, zero, ef, leaves)


def _is_zero(c):
    pf = FI.SR(c['combined_prefactor']).subs(c['num_params'])
    return complex(FI.CDF(pf)) == 0


def _nonzero_diagrams(calls):
    """The captured calls whose prefactor is not zero (they have a value)."""
    out = [c for c in calls if not _is_zero(c)]
    assert len(out) >= 4
    return out


def _zero_diagram(calls):
    """A captured call whose prefactor is numerically 0."""
    for c in calls:
        if _is_zero(c):
            return c
    raise AssertionError('no zero-prefactor diagram captured')


def test_l1_result_has_the_structure_of_the_full_path(captured_ou):
    c = _zero_diagram(captured_ou)
    full = _full_path(c)
    FI._reset_runtime_counters()
    fast = FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_zero_exit'] == 1
    assert list(fast.keys()) == list(full.keys())
    for key in full:
        a, b = full[key], fast[key]
        if key == 'contribution':
            continue
        if key == 'integration_vars':
            assert [str(x) for x in a] == [str(x) for x in b]
        elif key in ('stripped_integrand',):
            assert bool(a == b)
        elif key == 'constraints':
            assert [str(x) for x in a] == [str(x) for x in b]
        elif key == 'edge_info':
            assert len(a) == len(b)
            for ea, eb in zip(a, b):
                assert list(ea) == list(eb)
                assert (ea['u'], ea['v'], ea['lbl'], ea['ri'], ea['pi']) == \
                       (eb['u'], eb['v'], eb['lbl'], eb['ri'], eb['pi'])
                assert bool(ea['dt_sym'] == eb['dt_sym'])
                assert ea['delta_coeff'] == eb['delta_coeff']
                assert bool(ea['smooth_factor'] == eb['smooth_factor'])
        else:
            assert type(a) is type(b), key
            assert a == b, key
    # same value, same type, same argument checking
    for pt in _P2:
        va, vb = full['contribution'](*pt), fast['contribution'](*pt)
        assert type(va) is type(vb) is complex
        assert np.array_equal(va, vb)
    with pytest.raises(ValueError):
        full['contribution'](0.0)
    with pytest.raises(ValueError):
        fast['contribution'](0.0)


def test_l1_lazy_keys_stay_present_and_resolve(captured_ou):
    c = _zero_diagram(captured_ou)
    fast = FI.integrate_diagram(**c)
    assert {'edge_info', 'constraints', 'stripped_integrand'} <= set(fast)
    assert len(dict(fast)) == len(fast)               # dict(...) resolves
    assert all(v is not None for k, v in fast.items()
               if k in ('edge_info', 'constraints'))
    assert isinstance(fast.copy(), dict)


def test_l1_declines_without_external_fields_or_poles(captured_ou):
    c = _zero_diagram(captured_ou)
    td, pd = c['typed_diagram'], c['propagator_data']
    leaves = list(td.prediagram[2])
    zero = FI.SR(0)
    assert FI._zero_exit_applies(td, pd, zero, c['external_fields'], leaves)
    assert not FI._zero_exit_applies(td, pd, zero, None, leaves)
    assert not FI._zero_exit_applies(td, dict(pd, pole_vals=[]), zero,
                                     c['external_fields'], leaves)
    # a nonzero or non-numeric prefactor never exits
    assert not FI._zero_exit_applies(td, pd, FI.SR(1e-300),
                                     c['external_fields'], leaves)
    assert not FI._zero_exit_applies(td, pd, FI.SR.var('stray'),
                                     c['external_fields'], leaves)


def test_l1_threads_see_the_same_results(monkeypatch, captured_ou):
    calls = captured_ou[:40]
    serial = [FI.integrate_diagram(**c)['contribution'](*_P2[0])
              for c in calls]

    def work(_):
        return [FI.integrate_diagram(**c)['contribution'](*_P2[0])
                for c in calls]

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(work, range(4)))
    for r in results:
        assert all(np.array_equal(a, b) for a, b in zip(serial, r))


# ── L2: model-level data once per compute_correction_td call ───────────────
def _l2_inputs(captured):
    c = captured[0]
    return c['propagator_data'], c['num_params']


def test_l2_fires_on_models():
    for key, k, ell in [('ou_quartic', 2, 2), ('ou_quartic', 4, 1),
                        ('ou_quartic_colored', 2, 1)]:
        _run(key, k, ell)                                   # warm the cache
        on, c_on = _run(key, k, ell)
        assert c_on['setup_prop_td_used'] > 0
        assert c_on['setup_prop_td_stale'] == 0
        # one G(t) build per compute_correction_td call (one per ell), not
        # one per diagram
        assert 0 < c_on['setup_prop_td_g_t_builds'] <= ell + 1


def test_l2_prop_td_none_builds_locally(monkeypatch, captured_ou):
    """No ``prop_td`` (the default): the per-diagram path of before.  A
    shared ``PropagatorTD`` gives the same contribution values."""
    calls = _nonzero_diagrams(captured_ou)[:30]
    FI._reset_runtime_counters()
    plain = [FI.integrate_diagram(**c)['contribution'](*_P2[1]) for c in calls]
    assert FI._RUNTIME_COUNTERS['setup_prop_td_used'] == 0
    c0 = calls[0]
    ptd = FI.PropagatorTD(c0['propagator_data'], c0['num_params'])
    shared = [FI.integrate_diagram(prop_td=ptd, **c)['contribution'](*_P2[1])
              for c in calls]
    assert FI._RUNTIME_COUNTERS['setup_prop_td_used'] == len(calls)
    assert FI._RUNTIME_COUNTERS['setup_prop_td_g_t_builds'] == 1
    assert all(np.array_equal(a, b) for a, b in zip(plain, shared))


def test_l2_edge_mode_sums_equal_the_per_diagram_build(captured_ou):
    pdata, nps = _l2_inputs(captured_ou)
    ptd = FI.PropagatorTD(pdata, nps)
    n_poles = len(pdata['pole_vals'])
    assert n_poles >= 1
    edge_info = [dict(ri=0, pi=0, delta_coeff=0.0),
                 dict(ri=0, pi=0, delta_coeff=1.5 - 0.5j)]
    assert (FI._build_edge_mode_sums(edge_info, pdata)
            == FI._build_edge_mode_sums(edge_info, pdata, prop_td=ptd))
    # an entry whose conversion fails: ``None`` both ways, and cached
    bad = [dict(ri=0, pi=99, delta_coeff=0.0)]
    assert FI._build_edge_mode_sums(bad, pdata) is None
    assert FI._build_edge_mode_sums(bad, pdata, prop_td=ptd) is None
    assert FI._build_edge_mode_sums(bad, pdata, prop_td=ptd) is None
    # missing pole data: ``None`` before any table is touched
    assert FI._build_edge_mode_sums(edge_info, dict(pdata, pole_vals=None),
                                    prop_td=ptd) is None


def test_l2_stale_propagator_data_is_not_served(monkeypatch, captured_ou):
    """A ``PropagatorTD`` is recognised by the identity of the data it was
    built from.  Handed a changed propagator (new pole list) or other
    ``num_params`` it is ignored, the diagram is built locally and the
    result is that of an independent call (the pole engine's poles: the
    state-space route reads K_ft, not the pole list)."""
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    c = _nonzero_diagrams(captured_ou)[-1]
    pdata, nps = c['propagator_data'], c['num_params']
    ptd = FI.PropagatorTD(pdata, nps)
    ref = FI.integrate_diagram(**c)['contribution'](*_P2[0])
    FI._reset_runtime_counters()
    # the propagator is re-solved: new lists in a new dict (what the spatial
    # bridge does for every q), poles slightly moved
    pdata2 = dict(pdata)
    pdata2['pole_vals'] = [p * 1.5 for p in pdata['pole_vals']]
    c2 = dict(c, propagator_data=pdata2)
    got = FI.integrate_diagram(prop_td=ptd, **c2)['contribution'](*_P2[0])
    want = FI.integrate_diagram(**c2)['contribution'](*_P2[0])
    assert FI._RUNTIME_COUNTERS['setup_prop_td_stale'] == 1
    assert FI._RUNTIME_COUNTERS['setup_prop_td_used'] == 0
    assert np.array_equal(got, want) and not np.array_equal(got, ref)
    # same dict, pole list replaced in place: also not served
    pdata3 = dict(pdata)
    ptd3 = FI.PropagatorTD(pdata3, nps)
    pdata3['pole_vals'] = [p * 1.5 for p in pdata['pole_vals']]
    got3 = FI.integrate_diagram(
        prop_td=ptd3, **dict(c, propagator_data=pdata3))['contribution'](*_P2[0])
    assert np.array_equal(got3, want)
    # other num_params object: not served
    ptd4 = FI.PropagatorTD(pdata, dict(nps))
    FI.integrate_diagram(prop_td=ptd4, **c)
    assert FI._RUNTIME_COUNTERS['setup_prop_td_stale'] == 3


def test_l2_two_successive_calls_are_independent_calls(monkeypatch):
    """The spatial bridge calls ``compute_correction_td`` once per q with the
    propagator re-solved in between.  Each call must give what an
    independent call gives (a per-call object, no state kept across): the
    same q twice gives the same values.  (Until audit W7 also checked
    against the lever off, bit for bit.)"""
    from sage.all import SR
    import tests.test_spatial_pipeline_bridge as T
    from daedalus.engine.integration.spatial import pipeline_bridge as PB
    from daedalus.api._propagator import build_propagator
    from daedalus.engine.core.field_theory import FieldTheory
    from daedalus.engine.diagrams.type_assignment import build_field_index_map
    params = {'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0}
    model = T._load('allen_cahn_1d_subcritical_infinite')
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    nps = {SR.var(k): v for k, v in params.items()}
    nps[SR.var('phistar1')] = 0.0
    _, pi = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = PB._legs_to_phys_idx(T._EXT, pi)
    records = PB.build_pipeline_records(ft, model, prop, ext, max_ell=1)
    taus = np.array([0.0, 0.5, 1.0, 2.0])

    def C(q, ell):
        return PB.pipeline_C_q_tau(prop, records[ell], ext, nps, q, taus)

    qs = [0.7, 1.5, 0.0, 0.7, 3.0]
    FI._reset_runtime_counters()
    on = [[C(q, ell) for ell in (0, 1)] for q in qs]
    assert FI._RUNTIME_COUNTERS['setup_prop_td_used'] > 0
    assert FI._RUNTIME_COUNTERS['setup_prop_td_stale'] == 0
    # the same q twice in one process: equal results
    assert all(np.array_equal(x, y) for x, y in zip(on[0], on[3]))
    # different q: different values (the propagator really changed)
    assert not np.array_equal(on[0][0], on[1][0])


def test_l2_threads_share_one_object(captured_ou):
    calls = _nonzero_diagrams(captured_ou)[:30]
    serial = [FI.integrate_diagram(**c)['contribution'](*_P2[2]) for c in calls]
    c0 = calls[0]
    ptd = FI.PropagatorTD(c0['propagator_data'], c0['num_params'])

    def work(i):
        # a different starting point per thread: the first build races
        order = calls[i * 7:] + calls[:i * 7]
        return {id(c): FI.integrate_diagram(prop_td=ptd, **c)['contribution'](*_P2[2])
                for c in order}

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(work, range(4)))
    for r in results:
        assert all(np.array_equal(r[id(c)], v) for c, v in zip(calls, serial))
    assert FI._RUNTIME_COUNTERS['setup_prop_td_g_t_builds'] == 1


# ── L3: lazy SR ────────────────────────────────────────────────────────────
def _pair_tree(n_legs=3):
    """A star tree (one source, ``n_legs`` edges to the leaves) on the 2x2
    instantaneous fixture of ``test_time_domain``: entry (0, 0) has a delta
    part AND a smooth part, so the delta-subset enumeration, the shot-noise
    branch (with smooth edges left over) and the continuous branch all run."""
    from sage.all import DiGraph
    from daedalus.engine.core.vertices import SourceType
    from daedalus.engine.diagrams.type_assignment import TypedDiagram
    from tests.test_time_domain import _propagator_data_instantaneous_pair
    pd = _propagator_data_instantaneous_pair()
    st = SourceType(FI.SR(1), [('nt', 1)] * n_legs, (n_legs, 0))
    D = DiGraph()
    D.add_edges([(0, i) for i in range(1, n_legs + 1)])
    td = TypedDiagram(
        prediagram=(D, D.to_undirected(), list(range(1, n_legs + 1)), [0]),
        vertex_assignments={0: st},
        edge_types={(0, i, None): (('nt', 1), ('dn', 1))
                    for i in range(1, n_legs + 1)},
        external_legs={i: ('dn', 1) for i in range(1, n_legs + 1)},
        propagator_indices={(0, i, None): (0, 0)
                            for i in range(1, n_legs + 1)})
    ts = [FI.SR.var(f't_{i}') for i in range(1, n_legs + 1)]
    return dict(typed_diagram=td, propagator_data=pd,
                combined_prefactor=FI.SR(-2), ext_time_vars=ts,
                num_params=None, origin_leaf_idx=0,
                external_fields=[('dn', 1)] * n_legs)


def _delta_values(res, pts):
    """Everything numeric a result exposes: the callable at ``pts`` and each
    delta contribution (equality, coefficient at a few points, retardation)."""
    out = [complex(res['contribution'](*p)) for p in pts]
    for dc in res['delta_contributions']:
        out.extend(dc['equality_a'])
        out.append(dc['equality_c'])
        out.extend(complex(dc['coeff_fc'](*p[1:])) for p in pts)
        out.extend(x for ad in dc['retardation_data'] for x in (*ad[0], ad[1]))
    return np.array(out, complex)


def test_l3_shotnoise_branch_builds_the_smooth_factors_on_demand():
    """The shot-noise branch multiplies the leftover smooth edges'
    ``smooth_factor`` in; they are built there, on demand."""
    kw = _pair_tree(3)
    pts = [(0.0, 0.5, 1.5), (0.0, 1.0, 1.0), (0.0, 0.7, 2.3)]
    FI._reset_runtime_counters()
    on = FI.integrate_diagram(**kw)
    on_vals = _delta_values(on, pts)
    # the eager path's values (the retired DAEDALUS_SETUP_LAZY_SR=0), frozen
    # at main dca8804
    from tests.test_phase_j_structural_zeros import _frozen_close
    _frozen_close('l3_shotnoise|', {'vals': on_vals})
    assert FI._RUNTIME_COUNTERS['setup_lazy_sr_deferred'] == 3
    assert FI._RUNTIME_COUNTERS['setup_lazy_sr_built'] > 0      # it was needed
    assert on['n_shotnoise_skipped'] > 0 and len(on['delta_contributions']) > 0


def test_l3_sr_integrand_branch_builds_the_smooth_factors():
    """No mode-sum cache (the builder answers ``None``): every subset takes
    the SR integrand + scipy path, which multiplies the ``smooth_factor``s."""
    kw = dict(_pair_tree(2), edge_mode_sums_builder=lambda ei, pd: None)
    pts = [(0.0, 0.5), (0.0, 1.5), (0.0, 0.2)]
    FI._reset_runtime_counters()
    on = FI.integrate_diagram(**kw)
    on_vals = _delta_values(on, pts)
    assert FI._RUNTIME_COUNTERS['setup_lazy_sr_built'] > 0
    from tests.test_phase_j_structural_zeros import _frozen_close
    _frozen_close('l3_sr|', {'vals': on_vals}, rtol=1e-12)   # scipy path
    assert on['status'] == 'ok'


def test_l3_analytic_path_never_builds_the_smooth_factor(captured_ou):
    FI._reset_runtime_counters()
    for c in _nonzero_diagrams(captured_ou)[:20]:
        FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_lazy_sr_deferred'] > 20
    assert FI._RUNTIME_COUNTERS['setup_lazy_sr_built'] == 0
    assert FI._RUNTIME_COUNTERS['setup_lazy_display_built'] == 0


def test_l3_keys_stay_present_and_resolve_to_the_eager_values(captured_ou):
    """Each lazy ``smooth_factor`` is the eager one (``G_t_entry`` on the
    same G(t) matrix, as the retired eager path built it)."""
    c = _nonzero_diagrams(captured_ou)[0]
    lazy = FI.integrate_diagram(**c)
    assert 'stripped_integrand' in lazy and 'edge_info' in lazy
    G = FI.build_G_t_matrix(c['propagator_data'], FI.SR.var('_t_td_'),
                            num_params=c['num_params'])
    sf_all = FI.SR(1)
    for eb in lazy['edge_info']:
        assert list(eb) == ['u', 'v', 'lbl', 'ri', 'pi', 'dt_sym',
                            'delta_coeff', 'smooth_factor']
        eager = FI.G_t_entry(G, eb['pi'], eb['ri'], eb['dt_sym'],
                             include_heaviside=False)
        assert bool(eager == eb['smooth_factor'])
        sf_all = sf_all * eb['smooth_factor']
    # the display product: the prefactor times every smooth factor
    assert bool((lazy['stripped_integrand'] / sf_all).simplify_full()
                .is_constant())
    # copies and plain-dict views resolve everything
    assert all(v is not None for v in dict(lazy['edge_info'][0]).values())
    import copy
    import pickle
    assert isinstance(copy.deepcopy(lazy['edge_info'][0]), dict)
    assert pickle.loads(pickle.dumps(dict(lazy['edge_info'][0]))) is not None


def test_l3_failed_results_carry_the_eager_display_value(monkeypatch):
    """A failure return (an unexpected free symbol in the SR integrand) keeps
    a real ``stripped_integrand``, not a placeholder."""
    kw = dict(_pair_tree(2), edge_mode_sums_builder=lambda ei, pd: None,
              combined_prefactor=FI.SR.var('stray_symbol'))
    res = FI.integrate_diagram(**kw)
    assert res['status'] == 'failed'
    assert res['stripped_integrand'] is not None
    assert 'stray_symbol' in str(res['stripped_integrand'])
    assert type(res['stripped_integrand']).__name__ == 'Expression'


def test_l3_fires_on_models():
    for key, k, ell in [('ou_quartic', 2, 2), ('ou_quartic', 4, 1),
                        ('ou_quartic_colored', 2, 1)]:
        _run(key, k, ell)
        _on, c_on = _run(key, k, ell)
        assert c_on['setup_lazy_sr_deferred'] > 0


def test_l3_lazy_dict_basics():
    calls = []
    d = FI._LazyDict([('a', 1), ('b', None), ('c', 3)],
                     {'b': lambda: calls.append(1) or 2})
    assert list(d) == ['a', 'b', 'c'] and len(d) == 3 and 'b' in d
    assert not calls
    assert d['b'] == 2 and d['b'] == 2 and calls == [1]
    d2 = FI._LazyDict([('a', 1), ('b', None)], {'b': lambda: 5})
    assert d2.get('b') == 5
    d3 = FI._LazyDict([('a', 1), ('b', None)], {'b': lambda: 6})
    assert dict(d3) == {'a': 1, 'b': 6}
    d4 = FI._LazyDict([('a', 1), ('b', None)], {'b': lambda: 7})
    assert list(d4.items()) == [('a', 1), ('b', 7)]
    d5 = FI._LazyDict([('a', 1), ('b', None)], {'b': lambda: 8})
    d5['b'] = 9                                    # an explicit set wins
    assert d5['b'] == 9
    d6 = FI._LazyDict([('a', 1), ('b', None)], {'b': lambda: 10})
    assert {**d6} == {'a': 1, 'b': 10} and d6.copy() == {'a': 1, 'b': 10}


def test_l3_threads_resolve_the_same_values(captured_ou):
    c = _nonzero_diagrams(captured_ou)[0]
    res = FI.integrate_diagram(**c)
    ref = [str(e['smooth_factor']) for e in
           FI.integrate_diagram(**c)['edge_info']]

    def work(_):
        return [str(e['smooth_factor']) for e in res['edge_info']] + \
               [str(res['stripped_integrand'])]

    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(work, range(8)))
    for o in outs:
        assert o[:-1] == ref and o == outs[0]


# ── L5: shared automorphism work ───────────────────────────────────────────
@pytest.fixture(scope='module')
def captured_lh():
    """Typed diagrams of ``linear_hawkes`` k=2 ell=1 with DISTINCT external
    fields (n1, n2): one Wick mapping per diagram."""
    calls = []
    orig = PL.integrate_diagram

    def spy(**kw):
        calls.append({k: v for k, v in kw.items() if k != 'prop_td'})
        return orig(**kw)

    PL.integrate_diagram = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            AC.compute_cumulants(
                _model('linear_hawkes'), k=2, max_ell=1,
                tau_grid=np.array([0.0, 2.5, 10.0]), use_cache=True,
                parallel=False, verbose=False, parameters=_LH_PARAMS,
                external_fields=[('n', 1), ('n', 2)])
    finally:
        PL.integrate_diagram = orig
    assert calls
    return calls


def test_l5_memo_value_equals_the_uncached_order(captured_ou, captured_lh):
    for c in captured_ou[:25] + captured_lh:
        td = c['typed_diagram']
        for fe in (True, False):
            want = SYM._automorphism_order_uncached(td, fe)
            assert SYM._automorphism_order(td, fe) == want       # miss
            assert SYM._automorphism_order(td, fe) == want       # hit
            assert type(SYM._automorphism_order(td, fe)) is int


def test_l5_counters_hits_and_misses(captured_ou):
    td = captured_ou[0]['typed_diagram']
    FI._reset_runtime_counters()
    SYM._automorphism_order(td, True)
    SYM._automorphism_order(td, True)
    SYM._automorphism_order(td, False)
    SYM._automorphism_order(td, True)
    assert FI._RUNTIME_COUNTERS['setup_aut_memo_misses'] == 2
    assert FI._RUNTIME_COUNTERS['setup_aut_memo_hits'] == 2
    # combinatorial_factor + external_wick_compensation share the fixed order
    SYM._aut_memo_clear()
    FI._reset_runtime_counters()
    SYM.combinatorial_factor(td)
    SYM.external_wick_compensation(td)
    assert FI._RUNTIME_COUNTERS['setup_aut_memo_misses'] == 2
    assert FI._RUNTIME_COUNTERS['setup_aut_memo_hits'] == 1


def test_l5_the_table_is_bounded_and_holds_its_diagrams(monkeypatch,
                                                        captured_ou):
    monkeypatch.setattr(SYM, '_AUT_MEMO_MAX', 5)
    tds = [c['typed_diagram'] for c in captured_ou[:12]]
    FI._reset_runtime_counters()
    for td in tds:
        SYM._automorphism_order(td, True)
    assert len(SYM._aut_memo) <= 5
    assert FI._RUNTIME_COUNTERS['setup_aut_memo_evictions'] >= 1
    # every entry keeps its diagram alive, so its id cannot be reused
    assert all(v[0] is not None for v in SYM._aut_memo.values())
    for (i, _fe), (td, _fp, _o) in SYM._aut_memo.items():
        assert i == id(td)


def test_l5_a_changed_diagram_is_recomputed(captured_ou):
    import copy as _copy
    td = captured_ou[0]['typed_diagram']
    base = SYM._automorphism_order(td, False)
    td2 = _copy.copy(td)                              # distinct object, same data
    assert SYM._automorphism_order(td2, False) == base
    # swap a container of the memoised diagram: the fingerprint no longer
    # matches, the memo entry is not served
    td3 = _copy.copy(td)
    SYM._automorphism_order(td3, True)
    td3.external_legs = dict(td3.external_legs)
    FI._reset_runtime_counters()
    SYM._automorphism_order(td3, True)
    assert FI._RUNTIME_COUNTERS['setup_aut_memo_misses'] == 1
    assert FI._RUNTIME_COUNTERS['setup_aut_memo_hits'] == 0


def test_l5_single_mapping_compensation_is_exactly_one(captured_lh):
    """The shortcut's premise: with one Wick mapping the compensation index
    |Aut(leaves free)| / |Aut(leaves fixed)| is 1 -- checked against the
    real computation on every diagram of a model with distinct externals."""
    n = 0
    for c in captured_lh:
        td = c['typed_diagram']
        free = SYM._automorphism_order_uncached(td, False)
        fixed = SYM._automorphism_order_uncached(td, True)
        assert free == fixed
        n += 1
    assert n >= 2


def test_l5_the_shortcut_fires_only_for_a_single_mapping(captured_ou,
                                                         captured_lh):
    FI._reset_runtime_counters()
    for c in captured_lh:
        FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_aut_wick_skipped'] == len(captured_lh)
    # identical external fields: two mappings, the compensation is computed
    FI._reset_runtime_counters()
    for c in _nonzero_diagrams(captured_ou)[:8]:
        FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_aut_wick_skipped'] == 0


def test_l5_fires_on_models():
    cases = [('ou_quartic', 2, 2, {}), ('ou_quartic', 4, 1, {}),
             ('linear_hawkes', 2, 1, {'parameters': _LH_PARAMS})]
    for key, k, ell, extra in cases:
        field = 'n' if key == 'linear_hawkes' else 'dx'
        ext = ([('n', 1), ('n', 2)] if key == 'linear_hawkes'
               else [(field, 1)] * k)

        def run():
            FI._reset_runtime_counters()
            with contextlib.redirect_stdout(io.StringIO()):
                res = AC.compute_cumulants(
                    _model(key), k=k, max_ell=ell,
                    tau_grid=np.array([0.0, 2.5, 10.0]) if key == 'linear_hawkes' else _TAUS,
                    use_cache=True, parallel=False, verbose=False,
                    external_fields=ext, **extra)
            pts = _P4 if k == 4 else ([(0.0, 2.5), (0.0, 7.0)] if key == 'linear_hawkes' else _P2)
            arrs = {}
            for e, v in res['phase_j_by_ell'].items():
                if not v:
                    continue
                arrs[f'pd{e}'] = np.array(
                    [[g['contribution'](*p) for p in pts] for g in v['groups']], complex)
            if k == 2:
                arrs['C'] = np.asarray(res['C_tau'])
            return arrs, dict(FI._RUNTIME_COUNTERS), res

        SYM._aut_memo_clear()
        run()
        SYM._aut_memo_clear()
        on, c_on, _ = run()
        again, _c, _ = run()
        assert _same(on, again), key
        assert c_on['setup_aut_memo_misses'] > 0
        if key == 'linear_hawkes':
            assert c_on['setup_aut_wick_skipped'] > 0


def test_l5_threads_agree_with_serial(captured_ou):
    tds = [c['typed_diagram'] for c in captured_ou[:30]]
    serial = [(SYM._automorphism_order_uncached(t, True),
               SYM._automorphism_order_uncached(t, False)) for t in tds]

    def work(i):
        order = tds[i * 5:] + tds[:i * 5]
        return {id(t): (SYM._automorphism_order(t, True),
                        SYM._automorphism_order(t, False)) for t in order}

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(work, range(4)))
    for r in results:
        assert all(r[id(t)] == v for t, v in zip(tds, serial))


# ── findings of the independent verification ───────────────────────────────
def test_l1_declines_for_incomplete_residues_and_symbolic_zero(captured_ou):
    c = _zero_diagram(captured_ou)
    td, pd, ef = c['typed_diagram'], c['propagator_data'], c['external_fields']
    leaves = list(td.prediagram[2])
    zero = FI.SR(0)
    assert FI._zero_exit_applies(td, pd, zero, ef, leaves)
    assert not FI._zero_exit_applies(td, dict(pd, C_mats=None), zero, ef, leaves)
    assert not FI._zero_exit_applies(td, dict(pd, C_mats=[]), zero, ef, leaves)
    # a symbolic expression that evaluates to 0.0 may still be a nonzero
    # number once multiplied by a delta coefficient in the full path
    sym0 = FI.SR(7).sqrt() * FI.SR(5).sqrt() - FI.SR(35).sqrt()
    assert complex(FI.CDF(sym0)) == 0 and not sym0.is_numeric()
    assert not FI._zero_exit_applies(td, pd, sym0, ef, leaves)


def test_compute_correction_td_with_no_diagrams_needs_no_propagator():
    res = PL.compute_correction_td(typed_diagrams=[], prefactors=[],
                                   propagator_data=None, k=2)
    assert res['total_C'](0.0, 1.0) == 0

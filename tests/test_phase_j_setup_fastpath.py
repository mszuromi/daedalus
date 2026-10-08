"""M5 (P6): the per-diagram setup levers are pure speed-ups.

Every result with a lever on must be bit-identical (``np.array_equal``, never
a tolerance) to the same run with the lever off, in ONE process:

* L1 ``USE_SETUP_ZERO_EXIT``  a numerically-zero prefactor returns before any setup;
* L2 ``USE_SETUP_PROP_TD``    model-level data built once per ``compute_correction_td`` call;
* L3 ``USE_SETUP_LAZY_SR``    SR objects built on demand;
* L5 ``USE_AUT_MEMO``         shared automorphism work (``engine.diagrams.symmetry``).

``DAEDALUS_PHASE_J_LEGACY_SETUP=1`` forces all of them off.

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
import api.compute as AC
import engine.integration.time_domain.final_integral as FI
import engine.integration.time_domain.pipeline as PL

_TAUS = np.linspace(0.0, 3.0, 7)
_P2 = [(0.0, 0.7), (0.0, 1.9), (0.0, -1.1)]
_P4 = [(0.0, 0.3, 0.6, 0.9), (0.0, 0.5, 0.4, 0.7)]
_ALL_FLAGS = ('USE_SETUP_ZERO_EXIT',)       # grows with every lever


# ── helpers ────────────────────────────────────────────────────────────────
def _model(key):
    return dd.load_model(key)[0]


def _run(key, k, ell, ext_field='dx', **kw):
    """``compute_cumulants`` -> dict of arrays (per-ell totals, per-diagram
    values at fixed points) and the Phase J counters."""
    FI._reset_runtime_counters()
    ext = [(ext_field, 1)] * k
    with contextlib.redirect_stdout(io.StringIO()):
        res = AC.compute_cumulants(
            _model(key), k=k, max_ell=ell, tau_grid=_TAUS, use_cache=True,
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
def _flags(monkeypatch):
    """Every lever on by default; the umbrella unset; counters zeroed."""
    monkeypatch.delenv('DAEDALUS_PHASE_J_LEGACY_SETUP', raising=False)
    monkeypatch.setattr(FI, 'USE_SETUP_ZERO_EXIT', True)
    FI._reset_runtime_counters()
    yield
    FI._reset_runtime_counters()


@pytest.fixture(scope='module')
def captured_ou():
    """The keyword arguments of every ``integrate_diagram`` call of
    ``ou_quartic`` k=2, ell <= 2 (the real inputs: typed diagram, propagator
    data, prefactor, num_params)."""
    calls = []
    orig = PL.integrate_diagram

    def spy(**kw):
        calls.append(kw)
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


def _lever(monkeypatch, name, on):
    monkeypatch.setattr(FI, name, on)


# ── L1: zero-prefactor early exit ──────────────────────────────────────────
@pytest.mark.parametrize('key,k,ell', [('ou_quartic', 2, 2),
                                       ('ou_quartic', 4, 1),
                                       ('ou_quartic_colored', 2, 1)])
def test_l1_on_vs_off_array_equal(monkeypatch, key, k, ell):
    _lever(monkeypatch, 'USE_SETUP_ZERO_EXIT', False)
    _run(key, k, ell)                          # warm the disk cache
    off, c_off = _run(key, k, ell)
    _lever(monkeypatch, 'USE_SETUP_ZERO_EXIT', True)
    on, c_on = _run(key, k, ell)
    assert _same(off, on)
    assert c_off['setup_zero_exit'] == 0
    assert c_on['setup_zero_exit'] > 0                 # it fired
    # the diagrams stay in the list, at their positions (a zero entry)
    for name in off:
        assert off[name].shape == on[name].shape


def test_l1_counts_on_the_two_field_ou(monkeypatch):
    """The two-field OU is the motivating case (most diagrams have a zero
    prefactor).  Its model file is git-ignored, so skip without it."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models',
                        'ou_quartic_two_dim.model.py')
    if not os.path.exists(path):
        pytest.skip('models/ou_quartic_two_dim.model.py is not in this checkout')
    _lever(monkeypatch, 'USE_SETUP_ZERO_EXIT', False)
    _run('ou_quartic_two_dim', 2, 1)
    off, _ = _run('ou_quartic_two_dim', 2, 1)
    _lever(monkeypatch, 'USE_SETUP_ZERO_EXIT', True)
    on, c = _run('ou_quartic_two_dim', 2, 1)
    assert _same(off, on)
    assert c['setup_zero_exit'] > 10


def test_l1_never_exits_for_a_noise_or_conv_vertex(captured_ou):
    """A diagram with a noise-source (``cumulant_specs``) or a
    ``ConvVertexType`` vertex keeps the full path: its kernel is substituted
    into the prefactor, so the plain prefactor decides nothing.  (No model in
    this checkout reaches those branches of ``integrate_diagram``, so the
    decision is tested on a real zero-prefactor diagram with such a vertex
    added.)"""
    import types
    from engine.core.vertices import ConvVertexType, NoiseSourceType
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


def _zero_diagram(calls):
    """A captured call whose prefactor is numerically 0."""
    for c in calls:
        pf = FI.SR(c['combined_prefactor']).subs(c['num_params'])
        if complex(FI.CDF(pf)) == 0:
            return c
    raise AssertionError('no zero-prefactor diagram captured')


def test_l1_result_has_the_structure_of_the_full_path(monkeypatch, captured_ou):
    c = _zero_diagram(captured_ou)
    _lever(monkeypatch, 'USE_SETUP_ZERO_EXIT', False)
    full = FI.integrate_diagram(**c)
    _lever(monkeypatch, 'USE_SETUP_ZERO_EXIT', True)
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


def test_l1_lazy_keys_stay_present_and_resolve(monkeypatch, captured_ou):
    c = _zero_diagram(captured_ou)
    _lever(monkeypatch, 'USE_SETUP_ZERO_EXIT', True)
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


def test_l1_flag_and_umbrella(monkeypatch, captured_ou):
    c = _zero_diagram(captured_ou)
    FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_zero_exit'] == 1
    monkeypatch.setattr(FI, 'USE_SETUP_ZERO_EXIT', False)
    FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_zero_exit'] == 1       # flag off
    monkeypatch.setattr(FI, 'USE_SETUP_ZERO_EXIT', True)
    monkeypatch.setenv('DAEDALUS_PHASE_J_LEGACY_SETUP', '1')
    FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_zero_exit'] == 1       # umbrella
    monkeypatch.setenv('DAEDALUS_PHASE_J_LEGACY_SETUP', '0')
    FI.integrate_diagram(**c)
    assert FI._RUNTIME_COUNTERS['setup_zero_exit'] == 2
    monkeypatch.setattr(FI, 'USE_SETUP_ZERO_EXIT', 'yes')
    with pytest.raises(ValueError):
        FI.integrate_diagram(**c)


def test_setup_flags_from_the_environment():
    f = FI._initial_setup_flag
    assert f('X', {}) is True
    for v in ('1', 'true', 'YES', 'on', ''):
        assert f('X', {'X': v}) is True
    for v in ('0', 'false', 'No', 'off'):
        assert f('X', {'X': v}) is False
    with pytest.raises(ValueError):
        f('X', {'X': 'maybe'})


def test_the_legacy_umbrella_is_not_the_setup_umbrella(monkeypatch, captured_ou):
    """``DAEDALUS_PHASE_J_LEGACY_SETUP`` only restores the old setup path; it
    is a different switch from ``DAEDALUS_PHASE_J_LEGACY`` (older numbers)."""
    monkeypatch.setenv('DAEDALUS_PHASE_J_LEGACY', '0')
    assert FI._setup_lever_on('USE_SETUP_ZERO_EXIT') is True
    monkeypatch.setenv('DAEDALUS_PHASE_J_LEGACY_SETUP', '1')
    assert FI._setup_lever_on('USE_SETUP_ZERO_EXIT') is False


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

"""M0.1 (docs/integration_speedup_plan.md §4.1): ``final_integral._SUBSET_HOOK``,
the new observational runtime counters / bail reasons, and the per-subset
differential harness ``tests/tools/phase_j_subset_diff.py``.

Nothing here may move a number:

(i)   with the hook unset, results equal the values captured on b2bf559
      BEFORE M0.1 (PYTHONHASHSEED=0) to rtol 1e-13 (plan §4.4).  Not ``==``:
      MEASURED, the last ulp depends on the cache state -- the capture was
      made with a developer's local v1 prediagram cells, this module runs on
      a fresh cache (below), and a fresh run differs from a cache-loaded run
      of the same diagrams by up to 3.8e-16 rel on spike reset -- and spike
      reset's k=2 ell=1 output is also not bit-reproducible ACROSS PROCESSES
      at PYTHONHASHSEED=0 (<=2.6e-14 rel).  Strict old-vs-new bit-identity
      was proven in-process on the ``ou`` and ``spike`` configurations
      below, per-diagram and grouped (the pre-M0.1 sources exec'd into the
      live modules vs the M0.1 sources, all ``array_equal``).  The
      one-population spike-reset (``spike1``) values were captured on the
      f822c33 sources (engine identical to b2bf559, PYTHONHASHSEED=0), fresh
      cwd, same run order.  With the hook INSTALLED, the recorded run
      matches the same pre-M0.1 values to rtol 1e-13 too.
(ii)  with the hook installed, calls are recorded on the per-diagram AND the
      grouped path and the results are ``np.array_equal`` to the hook-unset
      run in the same process and the same cache state (both runs load the
      diagrams from the cache a warm-up run wrote); the hook is read at call
      time.
(iii) Σ value / compensation over the recorded subset calls reproduces the
      pipeline total to 1e-10.
(iv)  ``_reset_runtime_counters`` resets the new entries, incl. the nested
      ``nquad_fallback_by_reason`` dict; ``nquad_calls`` counts only m>=1
      entries (real nquad fallbacks), m=0 entries are ``polytope_m0_direct``.

Hermetic: the whole module runs in a private temporary cwd, so every
(cwd-relative) cache starts empty -- what a fresh clone sees -- and never the
developer's local caches, whose prediagram representatives change the
Phase J routes and the last ulp (MEASURED).

Models (public only): ``ou_quartic`` (smooth propagators: m1 / poset paths
only) and the spike-reset model ``single_population_spike_reset_test``
(δ / instantaneous propagator parts: m=0 δ-collapsed subsets, zero-normal
constraint rows, m1 / polygon / poset paths and poset nquad fallbacks),
both as the repo model (two populations, ``spike``) and as a one-population
instance of the same action (``spike1``, built below).

Cost: the default suite runs ``ou_quartic`` and ``spike1`` (both cheap:
``spike1``'s fresh-cache setup is ~3 s, the two-population model's ~15 s).
The repo spike-reset model -- the one with guard-bailed polygons
(``polygon_triangle_guard``) -- is ``@pytest.mark.slow``.

M1 (Θ(0) = 0 for constant constraint rows, plan §2.4) moves the one-loop of
the δ models, so the pre-M0.1 captures (i) are now reproduced with
``final_integral.THETA0_CONST_ROW_MODE = 'legacy_clip'`` (``_PRE_M1_MODE``);
``ou_quartic`` has no constant rows and is compared in the default mode.
Everything else runs in the default ('ito') mode, where ``spike1`` no longer
reaches the scipy.nquad fallback at all; the hook's nquad-path records and
counters are exercised on ``spike1`` under 'legacy_clip' (the hook does not
depend on the mode) and on ``spike`` (slow) in both modes.  The payload's
``row_kinds`` is the M1 row provenance.  The harness's references are
computed per (subset, free values, tie context) -- every tie orientation of
a subset separately (``test_harness_references_every_tie_orientation``,
model-free).
"""
import contextlib
import inspect
import math
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import daedalus as dd                                            # noqa: E402
from api import compute_cumulants                                # noqa: E402
import engine.integration.time_domain.final_integral as FI       # noqa: E402
import engine.integration.time_domain.grouped_integral as GI     # noqa: E402
from tests.tools import phase_j_subset_diff as PSD               # noqa: E402

_SPIKE_PARAMS = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
                 'w': [[0.55, 0.65], [0.7, 0.8]]}
_SPIKE1_PARAMS = {'Em': [3.5], 'tau': [10.0], 'a': [2.5], 'w': [[0.55]]}
_OU_PARAMS = {'mu': 1.0, 'D': 1.0, 'eps': 0.02}


def _spike_reset_one_population():
    """``models/single_population_spike_reset_test.model.py`` with ONE
    population (the repo model has two): the same fields, kernel, action
    and mean-field equations.  Same δ / zero-normal-row structure at a
    fifth of the fresh-cache setup cost."""
    from api.model import TemporalModelBuilder
    return (
        TemporalModelBuilder('Single population Spike Reset Test')
        .population('E', size=1, description='Excitatory')
        .physical_field('n', population='E', description='spike train')
        .physical_field('v', population='E', description='voltage')
        .parameter('Em', default=[1.0], indexed_by=['E'], domain='positive')
        .parameter('tau', default=[10], indexed_by=['E'], domain='positive')
        .parameter('a', default=[0.5], indexed_by=['E'], domain='positive')
        .parameter('w', default=[[0.25]], indexed_by=['E', 'E'],
                   domain='positive')
        .define_function('phi', args=['v'], expression='a[i] * v',
                         population='E', description='transfer (exc)')
        .define_kernel('g', time_expr='dirac_delta(t)', latex_name='g')
        .set_action_text('''
            sum( nt[i]*n[i]
            - (exp(nt[i])-1)*phi[i](v[i])
            + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i] + v[i]*n[i]
            - sum(w[i,j]*Conv(g,n[j]) for j in E))
            for i in E)
        ''')
        .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*nstar[j] for j in E))'
                                  ' / (1 + nstar[i])')
        .set_mf_equation('nstar', 'phi[i](vstar[i])')
        .build()
    )


_CONFIGS = {
    'ou': dict(model='ou_quartic', k=2, max_ell=1,
               external_fields=[('dx', 1), ('dx', 1)],
               parameters=_OU_PARAMS, tau_grid=[0.0, 0.5]),
    # One-population spike reset, ⟨n n⟩: m=0 δ-collapsed subsets, zero-normal
    # rows, m1/polygon/poset paths and poset nquad fallbacks, in ~3 s.
    'spike1': dict(model=_spike_reset_one_population, k=2, max_ell=1,
                   external_fields=[('n', 1), ('n', 1)],
                   parameters=_SPIKE1_PARAMS, tau_grid=[0.0, 3.0]),
    # The repo model, ⟨n1 n2⟩: the same paths plus guard-bailed polygons.
    'spike': dict(model='single_population_spike_reset_test', k=2,
                  max_ell=1, external_fields=[('n', 1), ('n', 2)],
                  parameters=_SPIKE_PARAMS, tau_grid=[0.0, 3.0]),
}
# Configurations whose DEFAULT-mode runs reach the scipy.nquad fallback
# (m>=1).  Before M1 ``spike1`` did too (poset_extract_none /
# poset_no_extension on constant rows and order cycles); with the Θ(0) rule
# it answers every subset analytically.
_NQUAD_CONFIGS = {'spike'}
# The mode that reproduces the pre-M1 captures in ``_BEFORE`` (``None``: the
# default mode; the model has no constant constraint rows).
_PRE_M1_MODE = {'ou': None, 'spike1': 'legacy_clip', 'spike': 'legacy_clip'}

# C_tau_by_ell captured on b2bf559 before the M0.1 edit (exact reprs).
_BEFORE = {
    ('ou', False): {
        0: [0.9999990000005 - 6.12322174927796e-17j,
            0.6065306597126334 - 1.8569645775045228e-17j],
        1: [-0.059999999999969994 + 7.347880794876769e-18j,
            -0.054587759374137006 + 6.127983105764925e-18j]},
    ('ou', True): {
        0: [0.9999990000005 - 6.12322174927796e-17j,
            0.6065306597126334 - 1.8569645775045228e-17j],
        1: [-0.059999999999969994 + 7.347880794876769e-18j,
            -0.054587759374137006 + 6.127983105764925e-18j]},
    # captured on the f822c33 sources (pre-M0.1 engine), see the docstring
    ('spike1', False): {
        0: [-0.4759643876300507 + 2.914439591429286e-17j,
            -0.08039546532128111 - 3.8318398056310674e-18j],
        1: [0.019073172166059516 - 2.335789995410257e-18j,
            0.012632042663118966 - 8.297314695362206e-19j]},
    ('spike1', True): {
        0: [-0.4759643876300507 + 2.914439591429286e-17j,
            -0.08039546532128111 - 3.8318398056310674e-18j],
        1: [0.019073172166059516 - 2.335789995410256e-18j,
            0.012632042663118966 - 8.297314695362202e-19j]},
    ('spike', False): {
        0: [0.5309889247725774 - 5.479611401951156e-16j,
            0.0025024048332344248 - 5.476278882510659e-17j],
        1: [-0.042525758149856235 + 1.285900923694143e-16j,
            -0.004618608811512337 + 4.9085013452590543e-17j]},
    ('spike', True): {
        0: [0.5309889247725773 - 5.479611401951154e-16j,
            0.0025024048332344187 - 5.4762788825106595e-17j],
        1: [-0.04252575815925349 + 1.285900923694143e-16j,
            -0.00461860881995695 + 4.908501345259056e-17j]},
}

_SLOW = pytest.mark.slow


def _key_param(name, grouped, slow=False):
    return pytest.param(
        (name, grouped), id=f'{name}-{"grouped" if grouped else "perdiag"}',
        marks=[_SLOW] if slow else [])


def _name_param(name, slow=False):
    return pytest.param(name, id=name, marks=[_SLOW] if slow else [])


_KEYS = [_key_param(n, g, slow=(n == 'spike'))
         for n in ('ou', 'spike1', 'spike') for g in (False, True)]
_DELTA_MODELS = ['spike1', _name_param('spike', slow=True)]


# ── hermetic cwd + lazily computed runs ──────────────────────────────

@pytest.fixture(scope='module', autouse=True)
def _fresh_cache_cwd(tmp_path_factory):
    """Run the whole module in an empty cwd: every cache root is
    cwd-relative, so the caches start empty (a fresh clone)."""
    prev = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('phase_j_hook_cache'))
    try:
        yield
    finally:
        os.chdir(prev)


_MODELS = {}
_RUNS = {}
_HOOKED = {}
_PRE = {}
_WARM = set()


@contextlib.contextmanager
def _theta0_mode(mode):
    """Run with ``THETA0_CONST_ROW_MODE = mode`` (``None``: unchanged).
    'legacy_clip' reaches the scipy quadrature fallback on the δ models, and
    the pre-M1 captures were made with the pre-M2b (default-tolerance)
    fallback, so ``NQUAD_HARDENED`` is False with it (M2b)."""
    if mode is None:
        yield
        return
    saved = FI.THETA0_CONST_ROW_MODE
    saved_nh = FI.NQUAD_HARDENED
    FI.THETA0_CONST_ROW_MODE = mode
    if mode == 'legacy_clip':
        FI.NQUAD_HARDENED = False
    try:
        yield
    finally:
        FI.THETA0_CONST_ROW_MODE = saved
        FI.NQUAD_HARDENED = saved_nh


def _model(spec):
    """A repo model by name, or a builder callable (each built once per
    module); a model dict is returned as is."""
    if isinstance(spec, dict):
        return spec
    if spec not in _MODELS:
        _MODELS[spec] = (dd.load_model(spec)[0] if isinstance(spec, str)
                         else spec())
    return _MODELS[spec]


def _compute(cfg, grouped, tau_grid=None):
    assert FI._SUBSET_HOOK is None
    return compute_cumulants(
        _model(cfg['model']), k=cfg['k'], max_ell=cfg['max_ell'],
        external_fields=cfg['external_fields'], parameters=cfg['parameters'],
        tau_grid=np.array(tau_grid if tau_grid is not None
                          else cfg['tau_grid']),
        use_cache=True, parallel=False, verbose=False,
        use_grouped_phase_j=grouped)


def _warm(name):
    """One throw-away run (last τ only) so that every later run of this
    configuration loads its diagrams, expansion and propagator from the
    cache: a fresh run and a cache-loaded run can differ in the last ulp."""
    if name not in _WARM:
        cfg = _CONFIGS[name]
        _compute(cfg, False, tau_grid=cfg['tau_grid'][-1:])
        _WARM.add(name)


def _plain_arrays(res):
    return {int(ell): np.asarray(v, dtype=complex).copy()
            for ell, v in res['C_tau_by_ell'].items()}


def _hooked_run(cfg, grouped, **run_kw):
    """The same run through the harness (hook installed)."""
    run = PSD.run(_model(cfg['model']), k=cfg['k'], max_ell=cfg['max_ell'],
                  external_fields=cfg['external_fields'],
                  parameters=cfg['parameters'], tau_grid=cfg['tau_grid'],
                  use_grouped_phase_j=grouped, keep_live=False, **run_kw)
    assert FI._SUBSET_HOOK is None            # restored on exit
    from api.compute import _ITO_EPS
    by_ell = {}
    for (ell, pt), v in run['pipeline'].items():
        by_ell.setdefault(ell, {})[pt] = v
    arrays = {ell: np.array([d[PSD._grid_point(cfg['k'], t, _ITO_EPS)]
                             for t in cfg['tau_grid']], dtype=complex)
              for ell, d in by_ell.items()}
    return run, arrays


def _runs(key):
    """(model, grouped) -> {'plain': arrays, 'run': harness run,
    'hooked': arrays}: warm-up, then plain, then hooked, same process."""
    if key not in _RUNS:
        name, grouped = key
        _warm(name)
        plain = _plain_arrays(_compute(_CONFIGS[name], grouped))
        run, hooked = _hooked_run(_CONFIGS[name], grouped)
        _RUNS[key] = {'plain': plain, 'run': run, 'hooked': hooked}
    return _RUNS[key]


def _hooked(key):
    """(model, grouped) -> (harness run, arrays).  Reuses the hooked run of
    ``_runs`` when that exists; otherwise ONE harness run with no warm-up
    (the first run of a configuration fills its caches)."""
    if key in _RUNS:
        return _RUNS[key]['run'], _RUNS[key]['hooked']
    if key not in _HOOKED:
        name, grouped = key
        _HOOKED[key] = _hooked_run(_CONFIGS[name], grouped)
    return _HOOKED[key]


def _pre_m1_runs(key):
    """Like ``_runs`` but in the mode that reproduces the pre-M1 captures
    (``_PRE_M1_MODE``): warm-up, then plain, then hooked, same process."""
    name, grouped = key
    if _PRE_M1_MODE[name] is None:
        return _runs(key)
    if key not in _PRE:
        _warm(name)
        with _theta0_mode(_PRE_M1_MODE[name]):
            plain = _plain_arrays(_compute(_CONFIGS[name], grouped))
            run, hooked = _hooked_run(_CONFIGS[name], grouped)
        _PRE[key] = {'plain': plain, 'run': run, 'hooked': hooked}
    return _PRE[key]


def _assert_before(arrays, key):
    before = _BEFORE[key]
    assert sorted(arrays) == sorted(before)
    for ell, vals in before.items():
        # cache-state / cross-process ULP jitter (see module docstring)
        np.testing.assert_allclose(arrays[ell], np.array(vals, dtype=complex),
                                   rtol=1e-13, atol=0)


# ── (i) pre-M0.1 values ───────────────────────────────────────────────

@pytest.mark.parametrize('key', _KEYS)
def test_hook_unset_matches_pre_m01_values(key):
    """(δ models under 'legacy_clip' since M1, see ``_PRE_M1_MODE``.)"""
    _assert_before(_pre_m1_runs(key)['plain'], key)


@pytest.mark.parametrize('key', _KEYS)
def test_hooked_run_matches_pre_m01_values(key):
    """The recorded run itself (hook installed) reproduces the pre-M0.1
    values -- on spike reset through the δ / zero-normal-row paths."""
    _assert_before(_pre_m1_runs(key)['hooked'], key)


@pytest.mark.parametrize('key', _KEYS)
def test_legacy_mode_hook_is_transparent(key):
    r = _pre_m1_runs(key)
    assert sorted(r['plain']) == sorted(r['hooked'])
    for ell in r['plain']:
        assert np.array_equal(r['plain'][ell], r['hooked'][ell]), ell


# ── (ii) hook installed: transparent, records both dispatches ─────────

@pytest.mark.parametrize('key', _KEYS)
def test_hook_is_transparent(key):
    r = _runs(key)
    assert sorted(r['plain']) == sorted(r['hooked'])
    for ell in r['plain']:
        assert np.array_equal(r['plain'][ell], r['hooked'][ell]), ell


def _known_bail_reasons():
    """Every fine-grained reason string passed to ``_bail`` in FI / GI."""
    src = inspect.getsource(FI) + inspect.getsource(GI)
    return set(re.findall(r"_bail\('([a-z0-9_]+)'\)", src))


def _check_records(recs):
    known = _known_bail_reasons() | {'not_eligible', 'unknown'}
    for r in recs:
        if r['path'] == 'nquad':
            assert r['bail_category'] in FI._NQUAD_FALLBACK_REASONS
            assert r['bail_reason'] in known, r['bail_reason']
        else:
            assert r['bail_category'] is None and r['bail_reason'] is None
        assert r['m'] == (len(r['constraints'][0][0])
                          if r['constraints'] else r['m'])
        # M1 row provenance: one kind per constraint row
        assert isinstance(r['row_kinds'], tuple)
        assert len(r['row_kinds']) == len(r['constraints'])
        assert set(r['row_kinds']) <= set(FI.ROW_KINDS)
        assert r['perm_index'] is not None and r['compensation'] is not None


# The bail REASONS seen at M0 (e.g. 'poset_no_extension', 'poset_extract_
# none', and on spike 'polygon_triangle_guard') are transient: M1 (Θ(0)
# helper) and M2 (structural zeros) are designed to remove them.  Only their
# validity is asserted; the answering PATHS are structural.
@pytest.mark.parametrize('name', _DELTA_MODELS)
def test_hook_records_per_diagram(name):
    recs = _hooked((name, False))[0]['records']
    assert recs and {r['source'] for r in recs} == {'per_diagram'}
    paths = {(r['path'], r['branch']) for r in recs}
    assert {('m1', 'plan'), ('polygon', 'plan'), ('poset', 'plan'),
            ('m0', None)} <= paths
    assert (('nquad', 'plan') in paths) == (name in _NQUAD_CONFIGS)
    _check_records(recs)
    # the nquad-path payload, on the pre-M1 route
    lrecs = _pre_m1_runs((name, False))['run']['records']
    assert ('nquad', 'plan') in {(r['path'], r['branch']) for r in lrecs}
    _check_records(lrecs)


@pytest.mark.parametrize('name', ['ou'] + _DELTA_MODELS)
def test_hook_records_grouped(name):
    grecs = _hooked((name, True))[0]['records']
    assert 'grouped' in {r['source'] for r in grecs}
    _check_records(grecs)
    gpaths = {(r['path'], r['branch']) for r in grecs
              if r['source'] == 'grouped'}
    assert ('m1', 'grouped') in gpaths
    assert {('poset', 'noplan'), ('polygon', 'noplan')} & gpaths
    if name in _NQUAD_CONFIGS:
        assert {('polygon', 'noplan'), ('nquad', 'noplan')} <= gpaths
    if _PRE_M1_MODE[name] is not None:
        lrecs = _pre_m1_runs((name, True))['run']['records']
        _check_records(lrecs)
        lpaths = {(r['path'], r['branch']) for r in lrecs
                  if r['source'] == 'grouped'}
        assert {('polygon', 'noplan'), ('nquad', 'noplan')} <= lpaths


def test_hook_is_read_at_call_time():
    """Setting the hook AFTER the closures were built still records."""
    cfg = _CONFIGS['ou']
    res = _compute(cfg, False, tau_grid=[0.5])
    fn = res['phase_j_by_ell'][1]['total_C']
    seen = []
    with PSD.installed_hook(seen.append):
        v = complex(fn(0.0, 0.5))
    assert FI._SUBSET_HOOK is None
    assert seen and all(p['ctx']['ext_time_values'] == (0.0, 0.5)
                        for p in seen)
    assert v == complex(res['C_tau_by_ell'][1][0])
    n_before = len(seen)
    fn(0.0, 0.5)                                   # hook gone: no record
    assert len(seen) == n_before


# ── (iii) reconstruction ──────────────────────────────────────────────

@pytest.mark.parametrize('key', _KEYS)
def test_reconstruction_reproduces_pipeline(key):
    rows, orphans = PSD.reconstruction_check(_hooked(key)[0])
    assert rows and not orphans
    for row in rows:
        assert row['n_records'] > 0 or row['pipeline'] == 0
        assert row['abs_diff'] <= 1e-10 * max(abs(row['pipeline']), 1e-300) \
            + 1e-14, row


_ATTEMPT_KEYS = ('interval_attempted', 'polygon_attempted', 'poset_attempted',
                 'nquad_calls', 'polytope_m0_direct')


@pytest.mark.parametrize('name', ['ou'] + _DELTA_MODELS)
def test_points_mode_records_only_the_requested_points(name):
    """``run(points=...)`` at k=2 without a τ grid evaluates a one-point
    auxiliary grid (not the 201-point default) and drops its records and
    counters: only the requested points are recorded and reconstructed."""
    cfg = _CONFIGS[name]
    _warm(name)
    pts = [(0.0, 0.5), (0.5, 0.0), (0.0, -1e-6)]
    run = PSD.run(_model(cfg['model']), k=2, max_ell=1,
                  external_fields=cfg['external_fields'],
                  parameters=cfg['parameters'], points=pts, keep_live=False)
    assert run['config']['aux_tau_grid'] == [0.0]
    assert {r['ext_time_values'] for r in run['records']} == set(pts)
    rows, orphans = PSD.reconstruction_check(run)
    assert not orphans and len(rows) == 2 * len(pts)
    assert all(row['ok'] for row in rows)
    c, aux = run['counters'], run['counters_aux_grid']
    assert c['nquad_calls'] == sum(
        1 for r in run['records'] if r['path'] == 'nquad')
    assert c['polytope_m0_direct'] == sum(
        1 for r in run['records'] if r['evaluator'] == 'polytope_m0')
    # the auxiliary grid did evaluate (its counters were split off)
    assert sum(aux[k] for k in _ATTEMPT_KEYS) > 0
    assert sum(c[k] for k in _ATTEMPT_KEYS) > 0
    if name in _NQUAD_CONFIGS:
        assert aux['nquad_calls'] > 0 and c['nquad_calls'] > 0
    srcs = run['config']['diagram_sources']
    assert srcs and all(d['source'] in ('typed_cache', 'v2', 'v1', 'shipped',
                                        'computed', 'eager') for d in srcs)


# ── (iv) counters ─────────────────────────────────────────────────────

@pytest.mark.parametrize('pre_m1', [True, False], ids=['legacy', 'ito'])
@pytest.mark.parametrize('name', _DELTA_MODELS)
def test_new_counters_present_and_consistent(name, pre_m1):
    run = (_pre_m1_runs((name, False)) if pre_m1
           else {'run': _hooked((name, False))[0]})['run']
    c = run['counters']
    by = c['nquad_fallback_by_reason']
    assert set(by) == set(FI._NQUAD_FALLBACK_REASONS)
    n_mge1 = (c['scipy_nquad_called_m1'] + c['scipy_nquad_called_m2']
              + c['scipy_nquad_called_mge3'])
    assert sum(by.values()) == n_mge1
    assert (n_mge1 > 0) == (pre_m1 or name in _NQUAD_CONFIGS)
    # nquad_calls counts entries into the scipy.nquad fallback (m>=1) only;
    # the m=0 δ-collapsed subsets that reach _integrate_polytope are direct
    # evaluations, counted separately.
    assert c['nquad_calls'] == n_mge1
    assert c['zero_normal_rows_seen'] > 0
    recs = run['records']
    assert sum(1 for r in recs if r['path'] == 'nquad') == n_mge1
    n_m0 = sum(1 for r in recs if r['evaluator'] == 'polytope_m0')
    assert c['polytope_m0_direct'] == n_m0 > 0
    assert by['polygon_guard'] == sum(
        1 for r in recs if r['bail_reason'] == 'polygon_triangle_guard')
    # M1 Θ(0) counters: only the Itô rule calls the helper
    n_theta0 = c['theta0_const_empty'] + c['theta0_const_drop']
    assert (n_theta0 > 0) == (not pre_m1)
    assert c['theta0_tie'] <= c['theta0_const_empty']


def test_reset_runtime_counters_resets_new_entries():
    nested = FI._RUNTIME_COUNTERS['nquad_fallback_by_reason']
    FI._RUNTIME_COUNTERS['nquad_calls'] = 7
    FI._RUNTIME_COUNTERS['polytope_m0_direct'] = 4
    FI._RUNTIME_COUNTERS['zero_normal_rows_seen'] = 3
    FI._RUNTIME_COUNTERS['polygon_attempted'] = 5
    nested['poset_chain_none'] = 2
    nested['bogus_reason'] = 9
    FI._reset_runtime_counters()
    assert FI._RUNTIME_COUNTERS['nquad_fallback_by_reason'] is nested
    assert nested == {r: 0 for r in FI._NQUAD_FALLBACK_REASONS}
    for k, v in FI._RUNTIME_COUNTERS.items():
        if k != 'nquad_fallback_by_reason':
            assert v == 0, k


@pytest.mark.parametrize('name', ['ou'] + _DELTA_MODELS)
def test_stub_report_labels_the_counters(name):
    """With --stub-nquad the real counters stay 0; the report says why.
    (``ou_quartic`` never reaches the fallback: 0 calls are intercepted.)"""
    cfg = _CONFIGS[name]
    _warm(name)
    run = PSD.run(_model(cfg['model']), k=2, max_ell=1,
                  external_fields=cfg['external_fields'],
                  parameters=cfg['parameters'], tau_grid=cfg['tau_grid'][-1:],
                  stub_nquad=True, keep_live=False)
    c = run['counters']
    n_stub = sum(n for m, n in run['stub_calls'].items() if m >= 1)
    assert c['nquad_calls'] == 0
    assert (n_stub > 0) == (name in _NQUAD_CONFIGS)
    assert sum(c['nquad_fallback_by_reason'].values()) == n_stub
    rep = PSD.format_report(run)
    assert f'nquad STUBBED: {n_stub} m>=1 calls intercepted' in rep


# ── bail-reason bookkeeping is complete ───────────────────────────────

@pytest.mark.parametrize('fn', [
    FI._integrate_1d_polytope_modesum, FI._integrate_2d_polygon_modesum,
    FI._integrate_nd_polytope_poset_modesum,
    GI._evaluate_grouped_m0_modesum, GI._integrate_grouped_m1_modesum,
])
def test_every_analytic_bail_records_a_reason(fn):
    src = inspect.getsource(fn)
    code = '\n'.join(line.split('#')[0] for line in src.splitlines())
    assert not re.search(r'\breturn None\b', code), fn.__name__


def test_bail_category_mapping():
    assert FI._bail_category('polygon_triangle_guard') == 'polygon_guard'
    assert FI._bail_category('polygon_gamma_overflow') == 'polygon_other'
    assert FI._bail_category('poset_no_extension') == 'poset_no_extension'
    assert FI._bail_category('poset_lower_inconsistent') == 'other'
    assert FI._bail_category(None) == 'other'
    assert FI._bail('x') is None and FI._pop_bail_reason() == 'x'
    assert FI._pop_bail_reason() is None


# ── harness references on a closed-form subset ───────────────────────

def test_harness_references_closed_form_m2():
    """∫_{s0<s1<0} e^{-(s1-s0)} e^{s1} ds0 ds1 = 1 (λ=-1 on both edges);
    a zero-normal row with c=0 empties it (Itô), with c>0 it only drops."""
    cons = [((-1.0, 1.0), (0.0,), 0.0),      # s1 - s0 > 0
            ((0.0, -1.0), (1.0,), 0.0)]      # t - s1 > 0 (t = free val 0)
    lam = -1.0 + 0.0j

    def f(s0, s1, t):
        return np.exp(lam * (s1 - s0)) * np.exp(lam * (t - s1))

    terms = [(1.0 + 0.0j, (lam, lam))]
    val, info = PSD.tight_quad_reference(f, cons, (0.0,), 2, box=200.0)
    assert abs(val - 1.0) < 1e-10 and info['worst_status'] == 0
    for box in (150.0, 400.0):
        assert abs(PSD.mp_fan_reference(terms, cons, (0.0,), box=box)
                   - 1.0) < 1e-12
    empty = cons + [((0.0, 0.0), (0.0,), 0.0)]
    assert PSD.tight_quad_reference(f, empty, (0.0,), 2, box=200.0)[0] == 0
    assert PSD.mp_fan_reference(terms + [], empty, (0.0,), box=150.0) == 0
    kept = cons + [((0.0, 0.0), (0.0,), 0.3)]
    assert abs(PSD.tight_quad_reference(f, kept, (0.0,), 2,
                                        box=200.0)[0] - 1.0) < 1e-10
    with pytest.raises(NotImplementedError):
        PSD.dbm_reference()
    with pytest.raises(NotImplementedError):
        PSD.certify_box_reference()


# ── the harness's own constant-row rule ──────────────────────────────

def _tie_cases():
    """(a_ext, c0, free_vals, tie_ctx) constant rows: signs, exact ties of
    two legs in every Wick framing of k = 3 (origin 0/1/2), raw-time
    near ties that round to 0 after the translation, rows that are not a
    two-leg difference, and no context."""
    out = [((1.0,), 0.0, (0.3,), None), ((1.0,), 0.0, (-0.3,), None),
           ((0.0,), 0.0, (0.0,), None), ((0.0,), 0.5, (0.0,), None),
           ((1.0,), -0.3, (0.3,), ((0.0, 0.3), (1,), 0))]    # c0 != 0 tie
    for times in ((0.0, 0.7, 0.7), (0.7, 0.7, 0.0), (0.4, 0.4, 0.4),
                  (0.0, 0.7, 0.1 + 0.2 + 0.4), (1e6, 1e6 + 0.7, 1e6 + 0.7)):
        for origin in range(3):
            free = tuple(j for j in range(3) if j != origin)
            fv = tuple(times[j] - times[origin] for j in free)
            ctx = (times, free, origin)
            for a_ext in ((1.0, -1.0), (-1.0, 1.0), (1.0, 0.0), (0.0, 1.0),
                          (-1.0, 0.0), (0.0, -1.0), (1.0, 1.0), (0.0, 0.0),
                          (2.0, -2.0)):
                out.append((a_ext, 0.0, fv, ctx))
    return out


def test_harness_const_row_rule_agrees_with_production():
    """The harness decides constant rows with its OWN implementation of
    the 0.2.0 rule; on every case it agrees with production, and its
    divergence log stays empty."""
    n_tie = 0
    for a_ext, c0, fv, ctx in _tie_cases():
        tc = None if ctx is None else FI._TieContext(*ctx)
        a_int = (0.0,) * 2
        prod, basis = FI._const_row_decision(a_int, a_ext, c0, fv, 'edge', tc)
        mine = PSD.reference_const_row_verdict(a_ext, c0, fv, 'edge', ctx)
        assert mine == prod, (a_ext, c0, fv, ctx, mine, prod)
        n_tie += basis == 'tie_order'
    assert n_tie >= 20                  # the tie order is really exercised
    with pytest.raises(ValueError):
        PSD.reference_const_row_verdict((0.0,), 0.0, (0.0,), 'noise_box')


def test_harness_row_kind_tag_flags_constant_rows():
    """With M1 provenance every smooth-edge row is 'edge', so the constant
    rows are tagged from the rows themselves ('--ref-scope zero_normal')."""
    plain = {'row_kinds': ('edge', 'edge'),
             'constraints': [((1.0, -1.0), (0.0,), 0.0),
                             ((0.0, 1.0), (1.0,), 0.0)]}
    const = {'row_kinds': ('edge', 'edge'),
             'constraints': [((1.0, -1.0), (0.0,), 0.0),
                             ((0.0, 0.0), (1.0,), 0.0)]}
    legacy = {'row_kinds': None, 'constraints': const['constraints']}
    assert PSD.row_kind_tag(plain) == 'edge'
    assert not PSD.has_zero_normal_row(plain)
    assert PSD.row_kind_tag(const) == 'edge+zero_normal'
    assert PSD.has_zero_normal_row(const)
    assert PSD.row_kind_tag(legacy) == 'edge+zero_normal'


def test_harness_reports_a_wrong_production_tie_order(monkeypatch):
    """A production tie order flipped at exact ties is caught: the harness
    keeps its own verdict and logs the divergence."""
    real = FI._tie_order_sign
    monkeypatch.setattr(FI, '_tie_order_sign',
                        lambda *a, **k: -real(*a, **k))
    start = len(PSD.RULE_DIVERGENCES)
    ctx = ((0.0, 0.7, 0.7), (1, 2), 0)
    # the row t_1 − t_2 at t_1 == t_2 (leg 2 is later-listed: earlier) holds
    rows = [((0.0, 0.0), (1.0, -1.0), 0.0)]
    verdict, _ = PSD._resolve_rows(rows, (0.7, 0.7), ctx, ('edge',))
    assert verdict == 'OK'
    new = PSD.RULE_DIVERGENCES[start:]
    assert len(new) == 1
    assert (new[0]['harness'], new[0]['production']) == ('DROP', 'EMPTY')
    del PSD.RULE_DIVERGENCES[start:]


# ── references at an exact tie: one per (subset, tie context) ──────────

def _tie_payload(tie_ctx, perm_index, value, integrand):
    """A hook payload for ONE m=1 subset (t − s > 0, plus the constant row
    Δ = (free leg) − (origin leg)) at the free value 0.0, i.e. an exact tie
    of legs 0 and 1 of a k = 2 call, from Wick permutation ``perm_index``.
    Model-free: the closure is the closed form e^{s − t}."""
    return {
        'free_ext_vals': (0.0,), 'diagram_serial': 7, 'subset_index': 0,
        'tie_ctx': tie_ctx,
        'ctx': {'compensation': 1, 'call_serial': 3, 'eval_serial': 0,
                'ext_time_values': (0.0, 0.0), 'perm_index': perm_index,
                'n_perms': 2},
        'prefactor': 1.0 + 0j, 'source': 'per_diagram', 'loop_number': 1,
        'subset_id': 0, 'delta_edges': (), 'smooth_edges': (0, 1), 'm': 1,
        'constraints': [((-1.0,), (1.0,), 0.0), ((0.0,), (1.0,), 0.0)],
        'row_kinds': ('edge', 'edge'), 'modes_summary': None,
        'path': 'm1', 'evaluator': '_integrate_1d_polytope_modesum',
        'branch': 'plan', 'attempted': 'm1', 'bail_reason': None,
        'bail_category': None, 'value': value, 'integrand': integrand,
        'modes': None, 'plan': None, 'pole_tuples': None, 'diagram': None,
    }


def test_harness_references_every_tie_orientation():
    """(Review round 3.)  At an exact tie two Wick permutations evaluate
    the same subset at the same free values, but with different legs
    behind the free value, so the constant row t_1 − t_0 empties the
    region in one (leg 1 is later-listed: earlier) and holds in the other.
    The references are keyed by (subset, tie context): each orientation
    gets its own reference, and the disagreement table and the attribution
    compare each record with its own.  (Keyed by the subset alone, the
    second orientation was compared with the first one's reference: a
    spurious attribution of +1 here.)"""
    def f(s, t):
        return np.exp(s - t)
    identity = ((0.0, 0.0), (1,), 0)        # free value = t_1 − t_0
    swapped = ((0.0, 0.0), (0,), 1)         # free value = t_0 − t_1
    tc = [FI._TieContext(*c) for c in (identity, swapped)]
    rows = [((-1.0,), (1.0,), 0.0), ((0.0,), (1.0,), 0.0)]
    assert [FI._any_const_row_empty(rows, (0.0,), None, c) for c in tc] == [
        True, False]
    one = 1.0 - math.exp(-200.0)            # ∫_{-200}^{0} e^{s} ds
    rec = PSD.SubsetRecorder()
    rec(_tie_payload(identity, 0, 0j, f))
    rec(_tie_payload(swapped, 1, one + 0j, f))
    assert rec.records[0]['key'] == rec.records[1]['key']
    assert rec.records[0]['ref_key'] != rec.records[1]['ref_key']
    run_ = {'config': {'stub_nquad': False}, 'records': rec.records,
            'live': rec.live, 'refs': {}, 'ref_meta': {}}
    start = len(PSD.RULE_DIVERGENCES)
    PSD.compute_references(run_, ('a',), ref_max_m=1)
    assert len(PSD.RULE_DIVERGENCES) == start
    assert run_['ref_meta']['n_candidates'] == 2
    refs = {k: v['a'] for k, v in run_['refs'].items()}
    assert refs[rec.records[0]['ref_key']] == 0
    assert abs(refs[rec.records[1]['ref_key']] - one) < 1e-10
    table, worst = PSD.disagreement_table(run_)
    assert sum(row['n'] for row in table.values()) == 2
    assert sum(row['n_disagree'] for row in table.values()) == 0, worst
    (s, n_with, n_without), = PSD.attribution(run_, 'a').values()
    assert (n_with, n_without) == (2, 0) and abs(s) < 1e-10
    # a time without a float value (a symbolic one) still keys the record
    from sage.all import SR
    assert PSD.tie_ctx_key(((SR.var('t'), 0.5), (1,), 0)) == (
        ('t', 0.5), (1,), 0)

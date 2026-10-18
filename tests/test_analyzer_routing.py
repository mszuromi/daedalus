"""D2: the model analyzer wired into ``compute_cumulants``, the backend registry, provenance, the G6 run
diagnostics, and the D1 follow-ups.

A  wiring on and off give bit-identical numbers (per-diagram, spatial); the full zoo A/B is a separate
   in-process harness (reports/16_report.md).
B  every deliberately bad model stops at the cheapest gate with its specific code and a remedy, BEFORE the
   diagram enumeration (the enumerator is replaced by a tripwire); warning-level defects are recorded, and stop
   only under ``strict``.
C  the registry reproduces the dispatch of 0.2.0 for every request shape (a transcription of the old if-chain is
   the oracle; the table of the tracked models is printed), and a toy backend registered here is picked without
   touching the built-in ones.
D  saved results carry the route and the assumptions ledger; files without them still load.
G6 the total-relative accuracy estimate and the M8 verdicts (synthetic counters, and real runs).
D1 follow-ups: tests for the finding codes that had none, G3 for spatial models, the expand-cache validity
   check shared by ``load_expand`` and the analyzer, and the ``n_poles`` convention (wired == standalone)."""
import importlib.util
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR, sin                                                  # noqa: E402

import api.compute as C                                                       # noqa: E402
from api import compute_cumulants, load_npz, load_provenance, routing         # noqa: E402
from api import analyze as A                                                  # noqa: E402
from api.analyze import (AnalyzerError, FINDING_CODES, Policy, Question,      # noqa: E402
                         analyze)
from api.model import SpatialModelBuilder, TemporalModelBuilder              # noqa: E402
from api.save import save_csv, save_npz                                       # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / 'models'


def _load(name):
    spec = importlib.util.spec_from_file_location(f'd2_{name}', MODELS / f'{name}.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(), mod


def _run(model, ext=(('x', 1), ('x', 1)), k=2, max_ell=1, **kw):
    kw.setdefault('tau_grid', np.array([0.0, 1.0]))
    return compute_cumulants(model, k=k, max_ell=max_ell, external_fields=list(ext), parallel=False,
                             verbose=False, **kw)


@pytest.fixture
def no_enumeration(monkeypatch):
    """Replace the diagram enumerator by a tripwire: B asserts that no bad model reaches it."""
    calls = []

    def tripwire(*a, **kw):
        calls.append(1)
        raise AssertionError('the diagram enumeration was reached')
    import api._diagrams as _diagrams
    monkeypatch.setattr(C, 'enumerate_unique_diagrams', tripwire)          # temporal (module-level import)
    monkeypatch.setattr(_diagrams, 'enumerate_unique_diagrams', tripwire)  # spatial bridge (local import)
    return calls


# ── gallery models (small, built here) ──────────────────────────────────────────────────────────

def _ou(name, action, eqs=(('(Dt+mu)*x', '0'),), params=(('mu', 1.0), ('D', 1.0)), stab=True, fields=('x',)):
    b = TemporalModelBuilder(name)
    for f in fields:
        b.physical_field(f)
    for p, d in params:
        b.parameter(p, default=d, domain='positive')
    b.set_action_text(action)
    for lhs, rhs in eqs:
        b.equation(lhs=lhs, rhs=rhs)
    if stab:
        b.stability_analysis(True)
    return b.build()


def _kernel(name, image, mf='0'):
    return (TemporalModelBuilder(name)
            .population('A', size=1)
            .physical_field('x', population='A')
            .parameter('mu', default=[1.0], indexed_by=['A'], domain='positive')
            .parameter('J', default=[0.3], indexed_by=['A'], domain='positive')
            .parameter('D', default=[1.0], indexed_by=['A'], domain='positive')
            .parameter('tg', default=2.0, domain='positive').parameter('d', default=1.5, domain='positive')
            .define_kernel('g', freq_image=image, latex_name='g')
            .set_action_text('sum(xt[i]*((Dt + mu[i])*x[i] - J[i]*g*x[i]) - D[i]*xt[i]^2 for i in A)')
            .set_mf_equation('xstar', mf)
            .build())


def _time_dependent():
    model = _load('ou_quartic')[0]
    act = model['action']
    model = dict(model)
    model['action'] = lambda ns: act(ns) + SR.var('A') * sin(SR.var('t')) * ns.xt[0]
    return model


def _marginal():
    return (TemporalModelBuilder('D2 gallery marginal mode')
            .population('A', size=1)
            .physical_field('x', population='A')
            .parameter('g', default=[1.0], indexed_by=['A'], domain='positive')
            .parameter('D', default=[0.1], indexed_by=['A'], domain='positive')
            .set_action_text('sum(xt[i]*(Dt*x[i] + g[i]*x[i]^3) - D[i]*xt[i]^2 for i in A)')
            .set_mf_equation('xstar', '0')
            .build())


def _spatial(name, action, mu=1.0, Dd=1.0, stab=True):
    b = (SpatialModelBuilder(name)
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=mu, domain='real').parameter('D', default=Dd, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text(action)
         .equation(lhs='(Dt + mu)*phi', rhs='0'))
    if stab:
        b.stability_analysis(True)
    return b.build()


_SP_GRID = dict(spatial_grid=np.array([0.0, 1.0]), tau_grid=np.array([0.0, 0.5]))
_SP_EXT = (('phi', 1), ('phi', 1))

# case: (builder, kwargs for compute_cumulants, code, stage)
STOPS = {
    'explicit time':        (_time_dependent, {}, 'G2_TIME_DEPENDENT', 'declarations'),
    'missing external':     (lambda: _load('ou_quartic')[0], {'ext': (('y', 1), ('y', 1))},
                             'G0_EXTERNAL_FIELD', 'declarations'),
    'domain violation':     (lambda: _load('ou_quartic')[0], {'parameters': {'mu': -1.0, 'eps': 0.02, 'D': 1.0}},
                             'G0_DOMAIN_VIOLATION', 'declarations'),
    'unpaired field':       (lambda: _ou('D2 gallery unpaired', 'xt*((Dt+mu)*x) - D*xt^2 - D*yt^2',
                                         eqs=(('(Dt+mu)*x', '0'), ('mu*y', '0')), fields=('x', 'y'), stab=False),
                             {}, 'G1_UNPAIRED_FIELD', 'expanded'),
    # an advance exp(+i omega d) is acausal (a delay exp(-i omega d) is routed: api.delay)
    'advance kernel':       (lambda: _kernel('D2 gallery advance', 'exp(I*omega*d)/(1 + I*omega*tg)'), {},
                             'G3_ACAUSAL_POLE', 'delay'),
    'sqrt kernel':          (lambda: _kernel('D2 gallery sqrt', '1/sqrt(1 + I*omega*tg)'), {},
                             'G3_NONRATIONAL_KERNEL', 'propagator'),
    'unstable (stab off)':  (lambda: _ou('D2 gallery unstable', 'xt*((Dt - mu)*x) - D*xt^2',
                                         eqs=(('(Dt - mu)*x', '0'),), stab=False), {},
                             'G2_UNSTABLE_SADDLE', 'mean_field'),
    'unstable (stab on)':   (lambda: _ou('D2 gallery unstable stab', 'xt*((Dt - mu)*x) - D*xt^2',
                                         eqs=(('(Dt - mu)*x', '0'),)), {},
                             'G2_UNSTABLE_SADDLE', 'mean_field'),
    'no saddle':            (lambda: _ou('D2 gallery no saddle', 'xt*((Dt+mu)*x + x^2 + 1) - D*xt^2',
                                         eqs=(('(Dt+mu)*x', '-(x^2 + 1)'),)), {},
                             'G2_NO_SADDLE', 'expanded'),     # the expansion's saddle check fails first
    'negative variance':    (lambda: _ou('D2 gallery negative variance', 'xt*((Dt+mu)*x) + D*xt^2'), {},
                             'G4_NEGATIVE_VARIANCE', 'mean_field'),
    'marginal mode':        (_marginal, {}, 'G2_MARGINAL_MODE', 'poles'),
    'acausal kernel':       (lambda: _acausal(), {}, 'G3_ACAUSAL_POLE', 'poles'),
    'spatial anti-diffusion': (lambda: _spatial('D2 gallery anti-diffusion',
                                                'phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2', Dd=-1.0),
                               dict(ext=_SP_EXT, **_SP_GRID), 'G3_SPATIAL_UNSTABLE', 'mean_field'),
    'undriven (strict)':    (lambda: _ou('D2 gallery undriven', 'xt*((Dt+mu)*x) - D*xt^2 + yt*((Dt+mu)*y)',
                                         eqs=(('(Dt+mu)*x', '0'), ('(Dt+mu)*y', '0')), fields=('x', 'y')),
                             {'policy': {'mode': 'strict', 'allow_taylor_truncation': True}},
                             'G1_UNDRIVEN_FIELD', 'mean_field'),
    'two wells (strict)':   (lambda: _load('ou_quartic_double_well')[0],
                             {'parameters': {'mu': -1.0, 'eps': 0.1, 'D': 1.0}, 'policy': {'mode': 'strict'}},
                             'G2_MULTISTABLE', 'mean_field'),
}


def _acausal():
    """An anti-causal kernel image g = 1/(1 - i omega tg) in the action (a pole in the lower half plane) while the
    declared mean-field equation is stable."""
    return (TemporalModelBuilder('D2 gallery acausal')
            .physical_field('x')
            .parameter('mu', default=1.0, domain='positive').parameter('J', default=0.3, domain='positive')
            .parameter('D', default=1.0, domain='positive').parameter('tg', default=2.0, domain='positive')
            .define_kernel('g', freq_image='1/(1 - I*omega*tg)', latex_name='g')
            .set_action_text('xt*((Dt + mu)*x - J*g*x) - D*xt^2')
            .equation(lhs='(Dt + mu)*x', rhs='0')
            .stability_analysis(True)
            .build())


@pytest.mark.parametrize('case', list(STOPS))
def test_bad_model_stops_before_enumeration(case, no_enumeration):
    build, kw, code, stage = STOPS[case]
    model = build()
    kw = dict(kw)
    ext = kw.pop('ext', (('x', 1), ('x', 1)))
    with pytest.raises(AnalyzerError) as ei:
        _run(model, ext=ext, **kw)
    err = ei.value
    assert code in err.codes, str(err)
    assert err.stage == stage, (err.stage, str(err))
    hit = [f for f in err.findings if f.code == code][0]
    assert hit.remedy and hit.level == 'error'
    assert f'[{code}]' in str(err) and 'remedy:' in str(err)
    assert isinstance(err, ValueError)
    assert not no_enumeration
    assert err.report is not None and code in err.report.codes


def test_unstable_with_stability_analysis_keeps_the_pipeline_message(no_enumeration):
    """The pipeline's own mean-field error stays in the message (callers matching it keep working)."""
    model = STOPS['unstable (stab on)'][0]()
    with pytest.raises(AnalyzerError, match='NONE were classified linearly stable'):
        _run(model)


def test_time_dependent_stops_before_the_expansion(monkeypatch):
    calls = []
    orig = C.FieldTheory.expand

    def spy(self, *a, **kw):
        calls.append(1)
        return orig(self, *a, **kw)
    monkeypatch.setattr(C.FieldTheory, 'expand', spy)
    with pytest.raises(AnalyzerError):
        _run(_time_dependent(), use_cache=False)
    assert not calls


def test_warning_level_defects_are_recorded_not_raised():
    res = _run(STOPS['undriven (strict)'][0](), max_ell=0)
    rep = res['analysis']
    assert rep.has('G1_UNDRIVEN_FIELD') and rep.ok
    assert [f for f in rep.findings if f.code == 'G1_UNDRIVEN_FIELD'][0].level == 'warning'
    model = _load('ou_quartic_double_well')[0]
    res = _run(model, max_ell=0, parameters={'mu': -1.0, 'eps': 0.1, 'D': 1.0})
    assert res['analysis'].has('G2_MULTISTABLE')


def test_downgrade_accepts_a_chosen_error():
    model = STOPS['negative variance'][0]()
    with pytest.raises(AnalyzerError) as ei:
        _run(model, max_ell=0)
    assert ei.value.codes == ('G4_NEGATIVE_VARIANCE',)
    res = _run(model, max_ell=0, policy={'downgrade': ['G4_NEGATIVE_VARIANCE']})
    f = [f for f in res['analysis'].findings if f.code == 'G4_NEGATIVE_VARIANCE'][0]
    assert f.level == 'warning' and 'downgrade' in f.remedy
    assert 'downgrade:G4_NEGATIVE_VARIANCE' in {e['name'] for e in res['config']['assumptions']}
    # codes the run cannot go past are never downgraded
    r = analyze(model, policy={'downgrade': ['G2_NO_SADDLE', 'G3_STIFF']})
    assert r.has('G0_POLICY_INVALID')


def test_out_of_range_fixed_point_index_is_a_warning():
    """The pipeline clamps it with a warning (0.2.0 behaviour); the analyzer records it, it does not stop."""
    with pytest.warns(UserWarning, match='fixed_point_index=3'):
        res = _run(_load('ou_quartic')[0], max_ell=0, fixed_point_index=3)
    f = [f for f in res['analysis'].findings if f.code == 'G2_FIXED_POINT_INDEX'][0]
    assert f.level == 'warning'


def test_array_valued_parameters_are_numeric():
    """numpy arrays as parameter values (accepted by the pipeline) pass the G0 domain check."""
    model = _load('ou_quartic')[0]                       # mu, eps, D declared positive
    par = {'mu': np.array(1.0), 'eps': np.array([0.02]), 'D': np.float64(1.0)}
    r = analyze(model, question={'parameters': par})
    assert not r.has('G0_DOMAIN_VIOLATION'), r.summary()
    r = analyze(model, question={'parameters': dict(par, mu=np.array([-1.0]))})
    assert r.has('G0_DOMAIN_VIOLATION')


def test_routing_errors_pickle_and_unknown_backend_fails_early(monkeypatch):
    import pickle
    req = routing.RouteRequest(model_name='m', spatial=True, k=2, max_ell=1)
    with pytest.raises(routing.NoBackendValueError) as ei:
        routing.select(req)
    back = pickle.loads(pickle.dumps(ei.value))
    assert type(back) is routing.NoBackendValueError and str(back) == str(ei.value)
    assert isinstance(back, ValueError)

    def boom(*a, **kw):
        raise AssertionError('the expansion ran')
    monkeypatch.setattr(C.FieldTheory, 'expand', boom)
    with pytest.raises(ValueError, match='not registered'):
        _run(_load('ou_quartic')[0], max_ell=0, backend='nope', use_cache=False)


def test_g6_under_worker_processes_reports_no_estimate():
    rep = _g6_report({'dd_err_abs_sum': 1e-3, 'dd_poset': 2}, [0.5, 0.5], complete=False)
    assert rep.diagnostics['accuracy_estimate'] is None and not rep.has('G6_ACCURACY_LOW')
    assert rep.has('G6_COUNTERS_INCOMPLETE')


def test_strict_policy_and_consents():
    model = _load('linear_hawkes')[0]
    ext = (('n', 1), ('n', 2))
    with pytest.raises(AnalyzerError) as ei:
        _run(model, ext=ext, max_ell=0, policy={'mode': 'strict'})
    assert ei.value.stage == 'declarations' and 'G1_TAYLOR_CONSENT_MISSING' in ei.value.codes
    res = _run(model, ext=ext, max_ell=0, policy=Policy(mode='strict', allow_taylor_truncation=True,
                                                         interpretation='ito'))
    assert res['analysis'].ok and not res['analysis'].warnings


def test_internal_error_in_a_gate_never_stops_a_run(monkeypatch):
    def boom(ctx):
        raise RuntimeError('synthetic gate bug')
    monkeypatch.setattr(A, '_g4_noise', boom)
    res = _run(_load('ou_quartic')[0], max_ell=0, policy={'mode': 'strict'})
    f = [f for f in res['analysis'].findings if f.code == 'INTERNAL_ERROR'][0]
    assert f.level == 'warning' and 'synthetic gate bug' in f.message


# ── A: wiring on vs off (in-suite subset) ─────────────────────────────────────────────────────

def _arrays(res):
    out = {'C_tau': np.asarray(res['C_tau'])}
    for ell, v in (res.get('C_tau_by_ell') or {}).items():
        out[f'ell{ell}'] = np.asarray(v)
    if 'C_tau_x' in res:
        out['C_tau_x'] = np.asarray(res['C_tau_x'])
    return out


@pytest.mark.parametrize('name,ext,kw', [
    ('ou_quartic', (('x', 1), ('x', 1)), {'max_ell': 1}),
    ('single_population_spike_reset_test', (('n', 1), ('n', 2)), {'max_ell': 1}),
    ('linear_hawkes', (('n', 1), ('n', 2)), {'max_ell': 1}),
    ('edwards_wilkinson_1d', (('phi', 1), ('phi', 1)), dict(max_ell=0, **_SP_GRID)),
])
def test_wiring_on_off_bit_identical(name, ext, kw):
    model, mod = _load(name)
    par = getattr(mod, 'DEFAULT_FUNDAMENTAL', None) or None
    # discarded warm-up: a cold on-disk cache and a warm one pick different (isomorphic) diagram
    # representatives and differ at ~1e-16 relative, which is not
    # what this test compares; on a fresh clone the first run below would otherwise be the cold one
    _run(model, ext=ext, parameters=par, preflight=False, **kw)
    on = _run(model, ext=ext, parameters=par, preflight=True, **kw)
    off = _run(model, ext=ext, parameters=par, preflight=False, **kw)
    a, b = _arrays(on), _arrays(off)
    assert set(a) == set(b)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    assert on['route'] == off['route']
    assert on['analysis'] is not None and off['analysis'] is None


def test_preflight_env_switch(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PREFLIGHT', '0')
    assert _run(_load('ou_quartic')[0], max_ell=0)['analysis'] is None
    monkeypatch.setenv('DAEDALUS_PREFLIGHT', '1')
    assert _run(_load('ou_quartic')[0], max_ell=0)['analysis'] is not None


# ── C: routing ────────────────────────────────────────────────────────────────────────────────

def _legacy_dispatch(req):
    """The dispatch of compute_cumulants at 0.2.0 (e6f3d4c), transcribed: the backend name, or the exception
    (type, first line) it raised.  Grouped Phase J (``phase_j_grouped``) is retired: every temporal request
    goes to the per-diagram backend."""
    spatial, k = req.spatial, req.k
    if spatial and (req.has_spatial_grid or req.has_spatial_points):
        if k != 2 and not req.has_spatial_points:
            return NotImplementedError, f'spatial k={k}: pass spatial_points'
        if k == 2 and not req.has_spatial_grid:
            return ValueError, 'spatial k=2 returns C(x,τ) on a grid'
        if k != 2:
            return 'spatial_integrator'
        if req.max_ell > 2:
            return NotImplementedError, 'spatial v1 implements tree'
        if req.initial_mode != 'stationary':
            return NotImplementedError, "spatial v1 supports the 'stationary' initial condition"
        return 'spatial_integrator'
    if spatial:
        return ValueError, f'model {req.model_name!r} is spatial'
    return 'phase_j_per_diagram'


def _requests():
    for spatial in (False, True):
        for k in (1, 2, 3):
            for ell in (0, 2, 3):
                for grid in (False, True):
                    for pts in (False, True):
                        for ic in ('stationary', 'switch_on'):
                            yield routing.RouteRequest(
                                model_name='m', spatial=spatial, k=k, max_ell=ell, has_spatial_grid=grid,
                                has_spatial_points=pts, initial_mode=ic)


def test_registry_reproduces_the_legacy_dispatch():
    n = 0
    for req in _requests():
        want = _legacy_dispatch(req)
        if isinstance(want, str):
            assert routing.select(req).name == want, req
        else:
            exc_type, text = want
            with pytest.raises(exc_type) as ei:
                routing.select(req)
            assert isinstance(ei.value, routing.NoBackendError)
            assert str(ei.value).startswith(text), (req, str(ei.value))
            assert 'Rejected backends:' in str(ei.value)
        n += 1
    assert n == 144


def _tracked_names():
    from tests.test_analyze import _tracked_model_names
    return sorted(_tracked_model_names())


def test_routing_table_of_the_tracked_models(capsys):
    rows = []
    for name in _tracked_names():
        model, mod = _load(name)
        spatial = bool(model.get('spatial'))
        req = routing.RouteRequest(model_name=model['name'], spatial=spatial, k=2, max_ell=1,
                                   has_spatial_grid=spatial,
                                   initial_mode=(model.get('initial') or {}).get('mode', 'stationary'))
        got = routing.select(req).name
        assert got == _legacy_dispatch(req)
        rows.append(f'{name:40s} ->  {got}')
    with capsys.disabled():
        print('\nrouting table (k=2, max_ell=1):\n  ' + '\n  '.join(rows))


def test_toy_backend_is_picked_without_editing_old_ones():
    before = [(b.name, b.priority, b.applies, b.run) for b in routing.backends()]
    seen = []

    def run(state):
        seen.append(state.route.name)
        return {'toy': True, 'C_tau': None, 'config': {'k': state.k, 'max_ell': state.max_ell}}

    toy = routing.Backend(name='toy_backend', priority=1,
                          applies=lambda req: ((True, ()) if req.model_name == 'D2 toy model' else
                                               (False, (routing.Reason('model_name', 'not the toy model'),))),
                          run=run, description='test-only backend')
    routing.register(toy)
    try:
        model = dict(_load('ou_quartic')[0], name='D2 toy model')
        res = _run(model, max_ell=0)
        assert res['toy'] and seen == ['toy_backend']
        assert res['route']['backend'] == 'toy_backend' and res['config']['route'] == 'toy_backend'
        # every other model still routes as before; the toy is listed among the rejected
        res = _run(_load('ou_quartic')[0], max_ell=0)
        assert res['route']['backend'] == 'phase_j_per_diagram'
        assert res['route']['rejected'][0]['backend'] == 'toy_backend'
        with pytest.raises(ValueError, match='already registered'):
            routing.register(toy)
    finally:
        routing.unregister('toy_backend')
    assert [(b.name, b.priority, b.applies, b.run) for b in routing.backends()] == before


def test_honest_failure_names_the_failing_trait(no_enumeration):
    model = _kernel('D2 gallery sqrt (downgraded)', '1/sqrt(1 + I*omega*tg)')
    with pytest.raises(routing.NoBackendError) as ei:
        _run(model, policy={'downgrade': ['G3_NONRATIONAL_KERNEL']})
    msg = str(ei.value)
    assert msg.startswith('propagator is not rational in omega') and 'first failing trait: propagator_class' in msg
    # the retired grouped Phase J and Phase H backends are not registered (audit W7)
    names = [b.name for b in routing.backends()]
    assert names == ['spatial_integrator', 'phase_j_per_diagram']
    for name in names:
        assert f'  - {name}:' in msg
    assert 'phase_j_grouped' not in msg and 'phase_h_symbolic' not in msg
    assert not no_enumeration
    for retired in ('phase_j_grouped', 'phase_h_symbolic'):
        with pytest.raises(ValueError, match='not registered'):
            _run(_load('ou_quartic')[0], max_ell=0, backend=retired)
    with pytest.raises(ValueError, match='not registered'):
        _run(_load('ou_quartic')[0], max_ell=0, backend='nope')


# ── D: provenance ─────────────────────────────────────────────────────────────────────────────

def test_saved_results_carry_route_and_ledger(tmp_path):
    model = _load('ou_quartic')[0]
    npz, csv = tmp_path / 'r.npz', tmp_path / 'r.csv'
    res = _run(model, max_ell=1, output_npz=str(npz), output_csv=str(csv))
    prov = load_provenance(load_npz(str(npz)))
    assert prov['route'] == 'phase_j_per_diagram'
    names = {e['name'] for e in prov['assumptions']}
    assert {'loop_truncation', 'stationary_state', 'phase_j_accuracy'} <= names
    assert {'code': 'G6_ACCURACY', 'level': 'info'} in prov['analyzer_findings']
    assert prov['assumptions'] == res['config']['assumptions']
    text = csv.read_text()
    assert '# route: phase_j_per_diagram' in text and '# assumptions: [' in text


def test_files_without_route_or_ledger_still_load(tmp_path):
    old = tmp_path / 'old.npz'
    np.savez(old, k=np.array([2]), max_ell=np.array([0]), tau_grid=np.array([0.0]),
             C_total=np.array([1.0 + 0j]), daedalus_version=np.asarray(['0.2.0']),
             phase_j_convention=np.asarray(['theta0_ito_const_rows']))
    data = load_npz(str(old))
    prov = load_provenance(data)
    assert prov['route'] is None and prov['assumptions'] is None and prov['analyzer_findings'] is None
    assert prov['daedalus_version'] == '0.2.0'
    # a result computed with the analyzer off: route kept, no ledger
    res = _run(_load('ou_quartic')[0], max_ell=0, preflight=False)
    path = save_npz(res, str(tmp_path / 'off.npz'))
    prov = load_provenance(load_npz(path))
    assert prov['route'] == 'phase_j_per_diagram' and prov['assumptions'] is None
    save_csv(res, str(tmp_path / 'off.csv'))


# ── G6 ────────────────────────────────────────────────────────────────────────────────────────

def _g6_report(counters, C_tau, complete=True, accuracy=None):
    pf = A.Preflight(_load('ou_quartic')[0], Question(k=2, max_ell=1, accuracy=accuracy))
    pf.after_run({'C_tau': np.asarray(C_tau)}, counters, complete=complete, evaluated=True)
    return pf.report()


def test_g6_names_diagrams_phase_j_could_not_evaluate():
    """G6_DIAGRAMS_SKIPPED (warning): diagrams left out of the cumulant
    (``result['phase_j_incomplete']``, see test_phase_j_never_silent.py for
    the routes); a complete run has no such finding."""
    pf = A.Preflight(_load('ou_quartic')[0], Question(k=2, max_ell=1))
    pf.after_run({'C_tau': np.array([0.5, 0.4]), 'phase_j_incomplete': {
        1: [{'diagram': 3, 'n_diagrams': 1, 'reason': 'evaluator status=failed (x)'},
            {'diagram': 5, 'n_diagrams': 2, 'reason': 'evaluator status=failed (x)'}]}},
        {'dd_poset': 1}, complete=True, evaluated=True)
    rep = pf.report()
    f = [f for f in rep.findings if f.code == 'G6_DIAGRAMS_SKIPPED'][0]
    assert f.level == 'warning' and f.gate == 'G6' and f.remedy
    assert '3 diagram(s)' in f.message and rep.diagnostics['n_diagrams_skipped'] == 3
    assert not _g6_report({'dd_poset': 1}, [0.5, 0.4]).has('G6_DIAGRAMS_SKIPPED')


def test_g6_bound_and_verdicts_on_synthetic_counters():
    rep = _g6_report({'divergent_subsets': 2, 'dd_ill_conditioned': 3, 'dd_err_ratio_max': 7.0,
                      'dd_err_abs_sum': 4e-7, 'dbm_err_abs_sum': 1e-7, 'dd_poset': 5, 'nquad_calls': 1},
                     [0.5, -0.5j])
    d = rep.diagnostics
    assert d['accuracy_estimate'] == pytest.approx(5e-7, rel=1e-12, abs=0)
    assert d['accuracy_threshold'] == A._G6_DEFAULT_RTOL == 1e-10
    lv = {f.code: f.level for f in rep.findings}
    assert lv['G6_ACCURACY_LOW'] == 'warning' and lv['G6_DIVERGENT_SUBSETS'] == 'warning'
    assert lv['G6_ILL_CONDITIONED'] == lv['G6_NQUAD_FALLBACK'] == 'info'
    assert {e.name: e for e in rep.ledger}['phase_j_accuracy'].satisfied is False
    # a declared accuracy target replaces the default threshold
    rep = _g6_report({'dd_err_abs_sum': 4e-7, 'dd_poset': 5}, [0.5, 0.5], accuracy=1e-3)
    assert not rep.has('G6_ACCURACY_LOW') and rep.diagnostics['accuracy_threshold'] == 1e-3
    assert {e.name: e for e in rep.ledger}['phase_j_accuracy'].satisfied is True
    # nothing estimated (every subset elsewhere): no check, not "satisfied"
    rep = _g6_report({'nquad_calls': 4}, [0.5, 0.5])
    assert rep.diagnostics['accuracy_estimate'] is None and not rep.has('G6_ACCURACY_LOW')
    assert {e.name: e for e in rep.ledger}['phase_j_accuracy'].satisfied is None
    # a non-finite cumulant
    rep = _g6_report({'dd_err_abs_sum': 1e-20, 'dd_poset': 1}, [np.nan, 0.5])
    assert rep.diagnostics['accuracy_estimate'] == float('inf') and rep.has('G6_ACCURACY_LOW')
    # an exactly zero cumulant with a nonzero error estimate: infinite estimate
    rep = _g6_report({'dd_err_abs_sum': 1e-20, 'dd_poset': 1}, [0.0, 0.0])
    assert rep.diagnostics['accuracy_estimate'] == float('inf') and rep.has('G6_ACCURACY_LOW')


def test_g6_real_runs_and_counter_restoration():
    import engine.integration.time_domain.final_integral as FI
    FI._reset_runtime_counters()
    FI._RUNTIME_COUNTERS['dd_err_ratio_max'] = 123.0       # a caller's earlier maximum
    res = _run(_load('ou_quartic')[0], max_ell=1)
    d = res['analysis'].diagnostics
    assert d['counters_complete'] and d['evaluated']
    assert 0 < d['accuracy_estimate'] < 1e-12
    assert FI._RUNTIME_COUNTERS['dd_err_ratio_max'] == 123.0   # max(before, run) restored
    res = _run(_load('edwards_wilkinson_1d')[0], ext=_SP_EXT, max_ell=0, **_SP_GRID)
    assert res['analysis'].has('G6_NOT_EVALUATED')


def test_g6_stiff_without_cancellation_is_quiet():
    """A stiff colored OU (fast/slow decay-rate ratio 1e4): the stiffness is flagged before the run (G3_STIFF);
    nothing cancels, so the bound stays at rounding level and G6 raises no false alarm."""
    model = _load('ou_quartic_colored')[0]
    res = _run(model, max_ell=1, parameters={'mu': 1.0, 'eps': 0.02, 'D': 1.0, 'tauc': 1e-4},
               policy={'allow_markov_embedding': True})
    rep = res['analysis']
    assert rep.has('G3_STIFF') and rep.traits.fast_slow_ratio == pytest.approx(1e4, rel=1e-12)
    d = rep.diagnostics
    assert d['n_estimated_subsets'] > 0 and d['accuracy_estimate'] < 1e-12
    assert not rep.has('G6_ACCURACY_LOW')


def test_g6_flags_the_digits_lost_to_near_degenerate_poles(monkeypatch):
    """Two poles 1e-4 apart (colored OU with tauc = 1/(mu (1 + 1e-4))): the residues ~ 1/gap cancel in the
    pole-residue sums.  Since item 21 compute_cumulants averages deformed propagators (api._pole_clusters):
    the one-loop value at tau = 0 is the exact 3 eps C0(0) dC0/dmu = -0.0900074997 and G6 stays quiet.  With
    the deformation switched off the old rounding noise of order 1 comes back, and G3_POLE_CLUSTER_UNRESOLVED
    and G6_ACCURACY_LOW say so (R of order 10); the default gap 1 (tauc = 0.5) stays at rounding level."""
    # the pole engine's deformation and lost digits (S3: the default
    # state-space route uses no poles; tests/test_statespace_default.py)
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    model = _load('ou_quartic_colored')[0]
    kw = dict(max_ell=1, policy={'allow_markov_embedding': True}, tau_grid=np.array([0.0, 0.5, 2.0]))
    near = {'mu': 1.0, 'eps': 0.02, 'D': 1.0, 'tauc': 1 / (1 + 1e-4)}
    res = _run(model, parameters=near, **kw)
    assert abs(complex(res['C_tau_by_ell'][1][0]).real - (-0.0900074997)) < 1e-11
    rep = res['analysis']
    assert res['pole_clusters']['status'] == 'resolved' and rep.has('G3_CLOSE_POLES')
    assert not rep.has('G6_ACCURACY_LOW') and rep.diagnostics['accuracy_estimate'] < 1e-10
    # an unresolved cluster: the exact residue tier unavailable (before
    # audit W7: the deformation switched off, DAEDALUS_POLE_CLUSTERS=0)
    from api import _propagator as PR
    with monkeypatch.context() as mp_:
        mp_.setattr(PR, '_compute_residues_via_polynomial_fracfield',
                    lambda *a, **k: (None, None, None))
        with pytest.warns(UserWarning, match='digits lost'):
            bad = _run(model, parameters=near, **kw)
    assert abs(complex(bad['C_tau_by_ell'][1][0]).real - (-0.0900074997)) > 1e-3        # the lost digits
    rep = bad['analysis']
    assert rep.has('G3_POLE_CLUSTER_UNRESOLVED') and rep.has('G6_ACCURACY_LOW')
    assert rep.diagnostics['accuracy_estimate'] > 1e-2
    good = _run(model, parameters={'mu': 1.0, 'eps': 0.02, 'D': 1.0, 'tauc': 0.5}, **kw)
    assert good['pole_clusters'] is None
    good = good['analysis']
    assert not good.has('G6_ACCURACY_LOW') and good.diagnostics['accuracy_estimate'] < 1e-12


# ── D1 follow-ups ─────────────────────────────────────────────────────────────────────────────

def _colored(name, kernel):
    b = (TemporalModelBuilder(name).physical_field('x')
         .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
         .parameter('tc', default=0.5, domain='positive')
         .set_action_text('xt*((Dt+mu)*x) - D*xt^2')
         .equation(lhs='(Dt+mu)*x', rhs='0').stability_analysis(True))
    b.correlated_noise('eta', response_legs={2: ['xt', 'xt']}, cumulants={2: kernel})
    return b.build()


CODE_GALLERY = {
    'G0_DUPLICATE_NAME': lambda: (dict(_load('ou_quartic')[0], parameters=list(
        _load('ou_quartic')[0]['parameters']) + [dict(_load('ou_quartic')[0]['parameters'][0])]), None),
    'G0_DOMAIN_UNKNOWN': lambda: (dict(_load('ou_quartic')[0], parameters=[
        dict(p, domain='nonnegative') if p['name'] == 'D' else p
        for p in _load('ou_quartic')[0]['parameters']]), None),
    'G0_PARAMETERS_MISSING': lambda: (dict(_load('ou_quartic')[0], parameters=[
        {k: v for k, v in p.items() if k != 'default'} if p['name'] == 'D' else p
        for p in _load('ou_quartic')[0]['parameters']]), None),
    'G0_UNKNOWN_PARAMETER': lambda: (_load('ou_quartic')[0],
                                     {'parameters': {'mu': 1.0, 'eps': 0.02, 'D': 1.0, 'muu': 2.0}}),
    'G0_MODEL_INVALID': lambda: ([1, 2], None),
    'G1_ACTION_EVAL': lambda: (dict(_load('ou_quartic')[0], action=lambda ns: ns.no_such_symbol), None),
    'G1_UNPAIRED_FIELD': lambda: (STOPS['unpaired field'][0](), None),
    'G2_NO_SADDLE': lambda: (STOPS['no saddle'][0](), None),
    'G2_ASSERTION_VIOLATED': lambda: (_load('ou_quartic_double_well')[0],
                                      {'parameters': {'mu': -1.0, 'eps': 0.1, 'D': 1.0}}),
    'G2_STABILITY_UNAVAILABLE': lambda: (_ou('D2 algebraic', 'xt*((Dt+mu)*x) - D*xt^2', eqs=(('mu*x', '0'),),
                                             stab=False), None),
    'G3_REPEATED_POLE': lambda: (_load('ou_quartic_colored')[0],
                                 {'parameters': {'mu': 1.0, 'eps': 0.02, 'D': 1.0, 'tauc': 1.0}}),
    'G3_IMPROPER': lambda: (_ou('D2 improper', 'xt*(x + Dt*y) - D*xt^2 + yt*(y) - D*yt^2',
                                eqs=(('x', '0'), ('y', '0')), fields=('x', 'y'), stab=False), None),
    'G3_ACAUSAL_POLE': lambda: (_acausal(), None),
    'G4_NO_NOISE': lambda: (_ou('D2 no noise', 'xt*((Dt+mu)*x)'), None),
    'G4_NEGATIVE_VARIANCE': lambda: (STOPS['negative variance'][0](), None),
    'G4_COLORED_RATIONAL': lambda: (_colored('D2 colored exp', lambda ns, i, j, tau: SR.var('D') * SR(
        'exp(-abs(tau)/tc)'.replace('tau', str(tau)))), None),
    'G4_NONRATIONAL_NOISE': lambda: (_colored('D2 colored gauss', lambda ns, i, j, tau: SR.var('D') * SR(
        'exp(-(tau/tc)^2)'.replace('tau', str(tau)))), None),
    'G3_SPATIAL_UNSTABLE': lambda: (STOPS['spatial anti-diffusion'][0](), None),
    # mu = 0 makes every homogeneous value a saddle (a zero eigenvalue); without stability_analysis one of them is
    # selected and its marginal q = 0 mode is a massless field
    'G3_MASSLESS': lambda: (_spatial('D2 massless', 'phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2', mu=0.0,
                                     stab=False), None),
}
_ASSERTIONS = {'G2_ASSERTION_VIOLATED': {'assertions': {'positive_at_saddle': ['x'], 'saddle_confirmed': True}}}


@pytest.mark.parametrize('code', sorted(CODE_GALLERY))
def test_finding_code_gallery(code):
    model, q = CODE_GALLERY[code]()
    r = analyze(model, question=q, policy=_ASSERTIONS.get(code))
    assert code in r.codes, r.summary()
    f = [f for f in r.findings if f.code == code][0]
    assert f.level == FINDING_CODES[code][1] and f.gate == FINDING_CODES[code][0]
    if f.level != 'info':
        assert f.remedy, f
    assert 'INTERNAL_ERROR' not in r.codes, r.summary()


def test_rational_spectrum_classification():
    from api.analyze import _exp_poly_form
    tau, c, w = SR.var('tau'), SR.var('c'), SR.var('w')
    rational = [SR('exp(-abs(tau)/c)'), SR('cos(w*tau)*exp(-abs(tau)/c)'), SR('tau*exp(-abs(tau)/c)'),
                SR('abs(tau)*exp(-abs(tau)/c) + 3*exp(-2*abs(tau)/c)'), SR('heaviside(tau)*exp(-tau/c)'),
                SR('sin(w*tau)^2*exp(-abs(tau)/c)'), SR('cosh(tau/c)*exp(-2*abs(tau)/c)')]
    other = [SR('exp(-(tau/c)^2)'), SR('1/(1 + tau^2)'), SR('sin(tau)/tau'), SR('exp(-abs(tau - 1))'),
             SR('cos(w*tau)'), SR('tau^2'), SR(3) * SR('cos(w*tau)') + SR('exp(-abs(tau)/c)')]
    assert all(_exp_poly_form(k, tau) for k in rational), [k for k in rational if not _exp_poly_form(k, tau)]
    assert not any(_exp_poly_form(k, tau) for k in other), [k for k in other if _exp_poly_form(k, tau)]


def test_g3_residues_numerical_tier(monkeypatch):
    """The exact propagator tier cannot compute the residues (its root
    labelling undecided, its 50-digit route failing): the numerical tier
    runs, with a PoleResidueWarning and G3_RESIDUES_NUMERICAL_TIER."""
    import warnings
    from api import _propagator as PR
    from api.noise import Gaussian
    from api.sde import SDE
    monkeypatch.setattr(PR, '_entry_pole_membership', lambda *a, **k: (None, 'undecided'))

    def boom(*a, **k):
        raise ArithmeticError('forced failure')
    monkeypatch.setattr(PR, '_exact_route', boom)
    monkeypatch.setattr(PR, '_cluster_path', lambda *a, **k: None)
    m = (SDE('ou_driven_g3').parameter('a', default=1.0).parameter('b', default=0.7).parameter('D', default=0.1)
         .field('x').field('z').drift('x', '-a*x + z').drift('z', '-b*z')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).noise('zeta', Gaussian(var='2*D'), couples={'z': 1})
         .build())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = compute_cumulants(m, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)], parallel=False,
                              verbose=False, use_cache=False, tau_grid=np.array([0.5]))
    assert any(issubclass(w.category, PR.PoleResidueWarning) for w in caught)
    f = [f for f in r['analysis'].findings if f.code == 'G3_RESIDUES_NUMERICAL_TIER'][0]
    assert f.level == 'warning' and f.remedy and 'forced failure' in f.message
    assert r['analysis'].traits.residue_route == 'failed'
    # exact: the OU part plus x filtering the OU z
    a, b, D, t = 1.0, 0.7, 0.1, 0.5
    cxx = D / a * np.exp(-a * t) + D / b * (a * np.exp(-b * t) - b * np.exp(-a * t)) / (a * (a ** 2 - b ** 2))
    assert abs(complex(r['C_tau'][0]).real - cxx) <= 1e-8 * abs(cxx)


def test_g6_reports_the_state_space_route_and_its_fallbacks(monkeypatch):
    """S3: on the default (state-space) route G6 records the engine census (G6_PHASE_J_ENGINE, the ledger) and its
    own accuracy estimate; a call-time fallback on the pole route (here forced: every segment refused as stiff) is
    G6_SS_FALLBACK, recorded and never raised (also under the strict policy)."""
    import engine.integration.time_domain.statespace as SS
    monkeypatch.delenv('DAEDALUS_PHASE_J_ENGINE', raising=False)
    model = _load('ou_quartic')[0]
    kw = dict(ext=[('dx', 1), ('dx', 1)], parameters={'mu': 1.0, 'eps': 0.02, 'D': 1.0})
    rep = _run(model, **kw)['analysis']
    assert rep.has('G6_PHASE_J_ENGINE') and not rep.has('G6_SS_FALLBACK')
    assert 0 < rep.diagnostics['accuracy_estimate'] < 1e-12
    monkeypatch.setattr(SS, 'TRI_DENSE_MAX', -1)
    monkeypatch.setattr(SS, 'TAYLOR_S_HARD', 0)
    for policy, level in ((None, 'warning'), ({'mode': 'strict'}, 'error')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = _run(model, policy=policy, **kw)            # never raised
        f = {x.code: x for x in res['analysis'].findings}
        # strict promotes it in the report, as every G6 warning
        assert f['G6_SS_FALLBACK'].level == level and 'stiff_segment' in f['G6_SS_FALLBACK'].message
        assert res['phase_j_engine']['fallback_evals'] > 0
    # a run predicted covered whose subsets decline the route: G6_SS_PARTIAL
    monkeypatch.setattr(SS, 'TRI_DENSE_MAX', 600)
    monkeypatch.setattr(SS, 'TAYLOR_S_HARD', 2000)
    monkeypatch.setattr(SS, 'MAX_DOWNSETS', 0)
    rep = _run(model, **kw)['analysis']
    assert rep.has('G6_SS_PARTIAL') and not rep.has('G6_SS_FALLBACK')


def test_finding_codes_all_have_a_test():
    """Every warning/error code is exercised somewhere in the analyzer tests (this file, test_analyze.py or
    test_analyzer_regime.py)."""
    here = Path(__file__).read_text()
    d1 = ((Path(__file__).parent / 'test_analyze.py').read_text()
          + (Path(__file__).parent / 'test_analyzer_regime.py').read_text())
    missing = [c for c, (_, lvl) in FINDING_CODES.items()
               if lvl != 'info' and c not in here and c not in d1 and c not in (
                   'G0_NAME_MISSING', 'G0_NAMESPACE', 'G1_EXPAND_FAILED', 'G1_SINGULAR_PROPAGATOR', 'G2_MF_FAILED',
                   'G3_UNRESOLVED_SYMBOLS', 'G3_POLES_UNAVAILABLE', 'INTERNAL_ERROR')]
    assert not missing, missing


def test_spatial_g3_on_the_tracked_spatial_models():
    """G3 for spatial models: decay rates of every probed mode positive, diffusion positive, no new warning."""
    for name in ('allen_cahn_1d_subcritical_infinite', 'coupled_rd_2species_1d', 'reaction_diffusion_2d',
                 'edwards_wilkinson_1d'):
        model, mod = _load(name)
        r = analyze(model, question={'parameters': getattr(mod, 'DEFAULT_FUNDAMENTAL', None) or None})
        t = r.traits
        assert t.spatial_mass_min > 0 and t.spatial_rate_min > 0 and t.spatial_diffusion_min > 0, (name, t)
        assert not r.warnings and r.ok, r.summary()


def test_spatial_leg_labels_are_not_populations():
    """Spatial callers label the legs ('dphi', 1), ('dphi', 2): the second entry is a slot, so G0 checks the field
    name only (a wrong name is still caught)."""
    model = _load('edwards_wilkinson_1d')[0]
    r = analyze(model, question={'k': 2, 'external_fields': [('dphi', 1), ('dphi', 2)]})
    assert not r.has('G0_EXTERNAL_FIELD')
    r = analyze(model, question={'k': 2, 'external_fields': [('dpsi', 1), ('dphi', 2)]})
    assert r.has('G0_EXTERNAL_FIELD')
    res = _run(model, ext=(('dphi', 1), ('dphi', 2)), max_ell=0, **_SP_GRID)
    assert res['route']['backend'] == 'spatial_integrator'


def test_massless_spatial_is_a_warning_not_a_marginal_error():
    r = analyze(CODE_GALLERY['G3_MASSLESS']()[0])
    assert r.has('G3_MASSLESS') and not r.has('G2_MARGINAL_MODE') and r.ok


def test_expand_cache_validity_is_one_shared_check(tmp_path, monkeypatch):
    """``load_expand`` and the analyzer's read-only reader call the same ``bundle_stale_reason``: forcing it
    either way flips both."""
    from api import _expand_cache as EC
    from engine.core.field_theory import FieldTheory
    monkeypatch.chdir(tmp_path)
    model = _load('ou_quartic')[0]
    _run(model, max_ell=0)                       # writes saved_models/<model>/expand_taylor2.sobj here
    assert EC.find_best_cached_order(model, 2) == 2

    def readers():
        ft = FieldTheory(model, taylor_order=2)
        EC.prepare_for_load(ft)
        loaded = EC.load_expand(model, ft, target_order=2, cached_order=2)
        ctx = A._Ctx(model, Question(), Policy())
        ctx.ft = FieldTheory(model, taylor_order=2)
        read = A._read_expand_cache(ctx, 2) is not None
        return loaded, read

    assert readers() == (True, True)
    monkeypatch.setattr(EC, 'bundle_stale_reason', lambda m, f, b: 'forced stale')
    assert readers() == (False, False)
    monkeypatch.setattr(EC, 'bundle_stale_reason', lambda m, f, b: None)
    monkeypatch.setattr(EC, '_get_ring_var_names', lambda ft: ['not', 'the', 'ring'])
    assert readers() == (True, True)             # neither keeps a private copy of the checks


@pytest.mark.parametrize('name,q', [
    ('ou_quartic_colored', None),
    ('linear_hawkes', None),
    ('single_population_spike_reset_test', None),
])
def test_n_poles_convention_wired_equals_standalone(name, q):
    """``n_poles`` = deg lcm(entry denominators of G), the same number from the analyzer's own exact inverse
    and from the pipeline's tier-1 record (ModelTraits docstring)."""
    model, mod = _load(name)
    par = getattr(mod, 'DEFAULT_FUNDAMENTAL', None) or None
    alone = analyze(model, question={'parameters': par}).traits
    ext = [(f['name'], 1) for f in model['physical_fields']][:1] * 2
    wired = _run(model, ext=ext, max_ell=0, parameters=par)['analysis'].traits
    assert alone.pole_source == 'exact' and wired.pole_source == 'pipeline'
    assert wired.n_poles == alone.n_poles and wired.repeated_poles == alone.repeated_poles
    assert wired.delta_component == alone.delta_component and wired.causal == alone.causal
    np.testing.assert_allclose(np.array(wired.poles), np.array(alone.poles), rtol=1e-12, atol=1e-14)


# ── dd run config path ────────────────────────────────────────────────────────────────────────

def test_config_policy_reaches_the_analyzer(tmp_path):
    from api.config_run import ConfigError, run_config
    cfg = tmp_path / 'c.yaml'
    cfg.write_text('model: ou_quartic\nquestion: {k: 2, max_ell: 0, external_fields: [[x, 1], [x, 1]], '
                   'tau_grid: [0.0, 1.0]}\npolicy: {mode: strict, downgrade: [G4_NEGATIVE_VARIANCE]}\n'
                   'outputs: {npz: out/c.npz}\n')
    res, resolved = run_config(str(cfg))
    assert resolved['policy']['mode'] == 'strict'
    assert res['analysis'].policy.mode == 'strict'
    assert res['analysis'].policy.downgrade == ('G4_NEGATIVE_VARIANCE',)
    info = json.loads((tmp_path / 'out' / 'run_info.json').read_text())
    assert info['route'] == 'phase_j_per_diagram' and info['assumptions']
    cfg.write_text('model: ou_quartic\npolicy: {downgrade: [G3_STIFF]}\n')
    with pytest.raises(ConfigError, match='not an error code'):
        run_config(str(cfg))
    cfg.write_text('model: ou_quartic\npolicy: {downgrade: [G2_NO_SADDLE]}\n')
    with pytest.raises(ConfigError, match='cannot be downgraded'):
        run_config(str(cfg))
    cfg.write_text('model: ou_quartic\npolicy: {preflight: false}\n'
                   'question: {k: 2, max_ell: 0, external_fields: [[x, 1], [x, 1]], tau_grid: [0.0]}\n')
    res, _ = run_config(str(cfg))
    assert res['analysis'] is None


def test_g6_names_an_ill_conditioned_realization():
    """A realization refused as 'ill_conditioned' (api._realization: exactly
    Hurwitz, unusable once rounded) is a counted, named fallback: G6 warning
    G6_SS_ILL_CONDITIONED and a ledger assertion (end to end: the slow test in
    tests/test_delay_kernels.py, a Pade-24 kernel on a 4-field cycle)."""
    from api.analyze import _Ctx, _g6_engine, Question, Policy
    ctx = _Ctx({}, Question(), Policy())
    census = {'engine': 'statespace', 'covers_run': False, 'reason': 'realization_ill_conditioned',
              'ss_subsets': 0, 'ss_declined': 2,
              'ss_declined_by_reason': {'realization_ill_conditioned': 2}, 'fallback_evals': 0}
    _g6_engine(ctx, {}, {'phase_j_engine': census}, None)
    hit = [f for f in ctx.findings if f.code == 'G6_SS_ILL_CONDITIONED']
    assert hit and hit[0].level == 'warning' and '2 subset(s)' in hit[0].message
    led = [e for e in ctx.ledger if e.name == 'realization_conditioning']
    assert led and led[0].satisfied is False


def test_spatial_accuracy_findings():
    """G6 names a spatial result the integrator cannot certify (result
    status 'approximate'): G6_SPATIAL_DERIVATIVE_LOOP for a derivative-vertex
    loop, G6_SPATIAL_EQUAL_TIME for a coupled loop at tau = 0 (product grid)."""
    import warnings
    import numpy as np
    from api import compute_cumulants
    kpz = (SpatialModelBuilder('kpz g6', n_populations=0).physical_field('p', spatial_dim=1)
           .parameter('mu', default=1.0, domain='positive').parameter('D', default=0.7, domain='positive')
           .parameter('c', default=0.3, domain='real').parameter('T', default=1.0, domain='positive')
           .equation(lhs='(Dt + mu - D*Laplacian)*p', rhs='0')
           .set_action_text('pt*(Dt(p)+mu*p-D*Lap(p)-(c/2)*Dx(p,0)^2)-T*pt^2')
           .operator_ir().boundary('infinite').initial('stationary').build())
    crd, _mod = _load('coupled_rd_2species_1d')
    kw = dict(k=2, max_ell=1, tau_grid=np.array([0.0]), spatial_grid=np.array([0.5]),
              use_cache=False, parallel=False, spatial_parallel=False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r1 = compute_cumulants(kpz, external_fields=[('p', 1)] * 2, **kw)
        r2 = compute_cumulants(crd, external_fields=[('a', 1)] * 2, **kw)
    assert 'G6_SPATIAL_DERIVATIVE_LOOP' in {f.code for f in r1['analysis'].findings}
    assert 'G6_SPATIAL_EQUAL_TIME' in {f.code for f in r2['analysis'].findings}
    assert r1['status'] == r2['status'] == 'approximate'

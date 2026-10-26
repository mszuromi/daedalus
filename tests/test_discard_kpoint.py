"""Evaluate-and-discard at k >= 3 (``discard=True`` / ``'auto'`` with fixed
evaluation points).

A k >= 3 result is read at points, not on a τ grid: ``compute_cumulants``
takes them as ``points`` (the external-time tuples), ``tau_grid=[0.0]`` is
the equal-time point, and ``dd.run`` passes the points of its slices (and
of its permuted-leg runs, full grid and moment blocks).  With the points
known up front each diagram is evaluated there right after it is built and
released, so Phase J holds one diagram at a time.  The values are the
same as reading a ``discard=False`` result at those points, bit for bit;
anywhere else the callables raise ``DiscardedDiagramsError``.

``'auto'`` turns it on from the pre-pass (the summed estimate over the
budget, the largest diagram within it) or mid-run, when the run's measured
growth crosses ``DISCARD_AUTO_GROWTH_FRACTION`` of the budget.  The growth
here is simulated deterministically: the footprint reading counts the live
per-diagram callables."""
import weakref

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                    # noqa: E402
from api import compute_cumulants                        # noqa: E402
from api.noise import Gaussian                           # noqa: E402
from api.sde import SDE                                  # noqa: E402
from engine.integration.time_domain import final_integral as FI   # noqa: E402
from engine.integration.time_domain import pipeline as PL         # noqa: E402

POINTS = [(0.0, 0.0, 0.0), (0.0, 0.3, 1.1), (0.0, 1.1, 0.3),
          (0.0, -0.4, 0.0)]


def _cubic():
    return (SDE('discard kpoint cubic').parameter('mu', default=1.2)
            .parameter('g', default=0.5).parameter('D', default=0.4)
            .field('x').drift('x', '-mu*x - g*x^2')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def _two_field():
    return (SDE('discard kpoint two field').parameter('mu', default=1.2)
            .parameter('nu', default=0.8).parameter('g', default=0.5)
            .parameter('c', default=0.3).parameter('D', default=0.4)
            .field('x').field('y')
            .drift('x', '-mu*x - g*x^2 + c*y').drift('y', '-nu*y + c*x*y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .noise('eta', Gaussian(var='D'), couples={'y': 1}).build())


def _cc(model, k, ell, **kw):
    kw.setdefault('external_fields', [('x', 1)] * k)
    return compute_cumulants(model, k=k, max_ell=ell, verbose=False,
                             use_cache=False, **kw)


def _at(res, pts):
    return ({e: [complex(f(*p)) for p in pts]
             for e, f in res['total_C_by_ell'].items()},
            [complex(res['total_C'](*p)) for p in pts])


@pytest.fixture(autouse=True)
def _budget(monkeypatch):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '4')


@pytest.mark.parametrize('k,ell', [(3, 0), (3, 1), (4, 0), (4, 1)])
def test_discard_values_identical(k, ell):
    pts = [p + (0.0,) * (k - 3) for p in POINTS] + [
        tuple(0.2 * i for i in range(k))]
    model = _cubic()
    a = _cc(model, k, ell, parallel=False, tau_grid=np.array([0.0]))
    b = _cc(model, k, ell, parallel=False, points=pts, discard=True)
    va, ta = _at(a, pts)
    vb, tb = _at(b, pts)
    assert va == vb and ta == tb
    assert list(b['C_points']) == ta
    assert {e: list(v) for e, v in b['C_points_by_ell'].items()} == va
    assert np.array_equal(b['points'], np.array(pts))
    pm = b['phase_j_memory']
    assert pm['discard'] is True
    assert pm['discard_refunded'] + pm['discard_kept_for_delta'] == sum(
        len(r['groups']) for r in b['phase_j_by_ell'].values() if r)


def test_off_point_queries_raise():
    b = _cc(_cubic(), 3, 1, parallel=False, points=POINTS, discard=True)
    with pytest.raises(dd.DiscardedDiagramsError, match='discard=False'):
        b['total_C'](0.0, 0.2, 0.2)
    with pytest.raises(dd.DiscardedDiagramsError):
        b['total_C_by_ell'][1](0.0, 0.2, 0.2)
    g = b['phase_j_by_ell'][1]['groups'][0]['contribution']
    assert getattr(g, 'discarded', False)
    with pytest.raises(dd.DiscardedDiagramsError, match='released'):
        g(*POINTS[0])
    # on the points: the same sums, whatever the argument types
    assert complex(b['total_C'](*POINTS[1])) == b['C_points'][1]
    assert complex(b['total_C'](*[np.asarray(t) for t in POINTS[1]])) == \
        b['C_points'][1]


def test_equal_time_point_and_validation():
    model = _cubic()
    a = _cc(model, 3, 1, parallel=False, tau_grid=np.array([0.0]))
    c = _cc(model, 3, 1, parallel=False, tau_grid=np.array([0.0]),
            discard=True)
    assert c['points'].tolist() == [[0.0, 0.0, 0.0]]
    assert c['C_points'][0] == complex(a['total_C'](0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='k >= 3 without fixed'):
        _cc(model, 3, 0, parallel=False, discard=True)       # default grid
    with pytest.raises(ValueError, match=r'shape \(n, k\)'):
        _cc(model, 3, 0, parallel=False, points=[(0.0, 0.0)], discard=True)
    with pytest.raises(ValueError, match='k >= 3'):
        _cc(model, 2, 0, parallel=False, points=[(0.0, 0.0)])


def test_parallel_runs_identical():
    model = _cubic()
    ref = _cc(model, 3, 1, parallel=False, tau_grid=np.array([0.0]))
    want = _at(ref, POINTS)
    for nw in (1, 4):
        b = _cc(model, 3, 1, parallel=True, n_workers=nw, points=POINTS,
                discard=True)
        assert _at(b, POINTS) == want


def _cfg(disc, **kw):
    base = dict(k=3, max_ell=1, external_fields=[('x', 1), ('y', 1),
                                                 ('x', 1)],
                tau_grid=np.array([-0.5, 0.0, 0.7]), use_cache=False,
                verbose=False, discard=disc)
    base.update(kw)
    return dd.Config(**base)


@pytest.mark.parametrize('extra', [{}, {'kpoint_base_lags': [0.3, 0.0]},
                                   {'output': 'central_moment'}])
def test_run_slices_identical(extra):
    """dd.run (tree): the slices (with the permuted-leg runs of a two-field
    cumulant), the base-lag slices and the moment blocks, bit for bit."""
    model = _two_field()
    a = dd.run(model, _cfg(False, max_ell=0, **extra))
    b = dd.run(model, _cfg(True, max_ell=0, **extra))
    assert b['phase_j_memory']['discard'] is True
    for j in a['C_tau_slices']:
        assert np.array_equal(a['C_tau_slices'][j], b['C_tau_slices'][j])
        for e in a['C_tau_slices_by_ell'][j]:
            assert np.array_equal(a['C_tau_slices_by_ell'][j][e],
                                  b['C_tau_slices_by_ell'][j][e])
    if 'moment' in a:
        assert np.array_equal(np.asarray(a['moment']),
                              np.asarray(b['moment']))


def test_run_one_loop_identical():
    model = _cubic()
    kw = dict(external_fields=[('x', 1)] * 3,
              kpoint_base_lags=[0.3, 0.0])
    a = dd.run(model, _cfg(False, **kw))
    b = dd.run(model, _cfg(True, **kw))
    assert b['phase_j_memory']['discard'] is True
    for j in a['C_tau_slices']:
        assert np.array_equal(a['C_tau_slices'][j], b['C_tau_slices'][j])


def test_run_full_grid_identical():
    model = _cubic()
    kw = dict(external_fields=[('x', 1)] * 3, max_ell=0,
              kpoint_full_grid=True)
    a = dd.run(model, _cfg(False, **kw))
    b = dd.run(model, _cfg(True, **kw))
    assert np.array_equal(a['C_tau_grid'], b['C_tau_grid'])
    assert np.array_equal(a['C_tau'], b['C_tau'])


def _simulated_growth(monkeypatch, per_diagram):
    """The footprint reading = live per-diagram callables x ``per_diagram``
    (released ones drop out), so the guard sees each kept diagram."""
    live = weakref.WeakSet()
    orig = PL.integrate_diagram

    def counted(*a, **k):
        out = orig(*a, **k)
        if callable(out.get('contribution')):
            live.add(out['contribution'])
        return out
    monkeypatch.setattr(PL, 'integrate_diagram', counted)
    monkeypatch.setattr(FI, 'process_footprint_bytes',
                        lambda: 10e9 + len(live) * per_diagram)


def test_guard_stopped_run_completes_with_discard(monkeypatch):
    model = _cubic()
    ref = _cc(model, 3, 1, parallel=False, tau_grid=np.array([0.0]))
    want = _at(ref, POINTS)
    n_diag = sum(len(r['groups']) for r in ref['phase_j_by_ell'].values()
                 if r)
    assert n_diag >= 8
    # budget: a quarter of the diagrams kept at once
    budget = 1e9
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1')
    _simulated_growth(monkeypatch, 4.0 * budget / n_diag)
    with pytest.raises(FI.PhaseJResourceError) as ei:
        _cc(model, 3, 1, parallel=False, points=POINTS)
    assert 'discard=True' in str(ei.value)
    # without points the refusal says what is needed
    with pytest.raises(FI.PhaseJResourceError, match='points='):
        _cc(model, 3, 1, parallel=False, tau_grid=np.array([0.3]))
    b = _cc(model, 3, 1, parallel=False, points=POINTS, discard=True)
    assert _at(b, POINTS) == want
    # 'auto': the pre-pass estimate is under the budget, the measured
    # growth switches it on mid-run; the same values
    c = _cc(model, 3, 1, parallel=False, points=POINTS, discard='auto')
    pm = c['phase_j_memory']
    assert pm['discard'] is True and pm.get('discard_switched_at')
    assert _at(c, POINTS) == want
    assert list(c['C_points']) == want[1]
    with pytest.raises(dd.DiscardedDiagramsError):
        c['total_C'](0.0, 0.2, 0.2)


def test_auto_stays_off_without_pressure():
    model = _cubic()
    c = _cc(model, 3, 1, parallel=False, points=POINTS, discard='auto')
    assert c['phase_j_memory']['discard'] is False
    assert 'C_points' not in c
    # the callables answer any point
    complex(c['total_C'](0.0, 0.2, 0.2))
    # k = 2: unchanged (no points, the grid; off-grid points still work)
    r2 = _cc(model, 2, 1, parallel=False, tau_grid=np.array([0.0, 0.5]),
             discard='auto')
    assert r2['phase_j_memory']['discard'] is False and 'points' not in r2
    complex(r2['total_C'](0.0, 0.7))


def test_delay_auto_order_unchanged_by_discard():
    """The delay order controller estimates its error at fixed sample points
    (``api.delay._observable``); a discard run carries them along, so it
    picks the same order and gives the same values."""
    m = (SDE('discard kpoint delay').parameter('a', default=1.0)
         .parameter('c', default=-2.0)
         .parameter('d', default=0.5, domain='positive')
         .parameter('b', default=0.3).parameter('D', default=0.1)
         .field('x').drift('x', '-a*x + c*x(t - d) - b*x^2')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    # both runs search: the order cache would start the second one at the first's order
    kw = dict(parallel=False, delay={'tol': 5e-5, 'order_cache': False})
    a = _cc(m, 3, 0, tau_grid=np.array([0.0]), **kw)
    b = _cc(m, 3, 0, tau_grid=np.array([0.0]), discard=True, **kw)
    assert b['delay']['n'] == a['delay']['n']
    assert b['delay'].get('est_by_order') == a['delay'].get('est_by_order')
    assert b['phase_j_memory']['discard'] is True
    i = [tuple(p) for p in b['points'].tolist()].index((0.0, 0.0, 0.0))
    assert b['C_points'][i] == complex(a['total_C'](0.0, 0.0, 0.0))


def test_discard_per_diagram_identical_at_scale():
    """Thousands of one-loop diagrams (a two-field k = 3 run): every
    diagram's value and every sum is the same with and without discard
    (both runs cold: ``use_cache=False``; the on-disk caches are a separate
    matter)."""
    model = _two_field()
    kw = dict(k=3, max_ell=1, external_fields=[('x', 1), ('y', 1), ('x', 1)],
              parallel=False, verbose=False, use_cache=False)
    pt = (0.0, 0.0, 0.0)
    a = compute_cumulants(model, tau_grid=np.array([0.0]), **kw)
    b = compute_cumulants(model, points=[pt], discard=True, **kw)
    n = 0
    for ell, ra in a['phase_j_by_ell'].items():
        if not ra:
            continue
        va = [complex(g['contribution'](*pt)) for g in ra['groups']
              if g.get('contribution') is not None]
        vb = b['phase_j_by_ell'][ell]['eval_per_diagram_batch'](
            [pt], parallel=False)[0][0]
        assert va == [complex(v) for v in vb]
        assert complex(a['total_C_by_ell'][ell](*pt)) == \
            b['C_points_by_ell'][ell][0]
        n += len(va)
    assert n > 4000
    assert complex(a['total_C'](*pt)) == b['C_points'][0]


@pytest.mark.parametrize('disc', [True, 'auto'])
def test_slice_failure_still_reported(monkeypatch, disc):
    """A k >= 3 slice whose points cannot be computed is reported as with
    discard=False: the result is incomplete, warns, and has no moment."""
    from api.compute import PhaseJIncompleteWarning

    def boom(*a, **kw):
        raise RuntimeError('slice evaluation failed')
    monkeypatch.setattr(dd, '_kpoint_slice_times', boom)
    monkeypatch.setattr(dd, '_kpoint_slice_point', boom)
    for out in ('cumulant', 'moment'):
        with pytest.warns(PhaseJIncompleteWarning, match='slices'):
            r = dd.run(_cubic(), dd.Config(
                k=3, max_ell=0, tau_grid=[0.0, 1.0], discard=disc,
                external_fields=[('x', 1)] * 3, output=out,
                use_cache=False, verbose=False))
        assert r['status'] == 'incomplete' and r['C_tau'] is None
        assert 'slice evaluation failed' in r['_kpoint_slice_error']
        if out == 'moment':
            assert r['moment'] is None

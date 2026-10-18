"""Audit wave W3a: silent wrong numbers on the default ``dd.run`` route.

* ``output='moment'`` builds every block from ITS OWN legs (field and
  population, the lag between them) -- it used leg 0 for every leg and folded
  the lag; reference: Isserlis + Lyapunov for a 2-field linear SDE.
* the loop-level raw moment includes the loop corrections of the means
  (the 1-point tadpoles); reference: the exact stationary density of a
  globally monostable cubic drift.
* METADATA ``fixed_point_index_default`` / ``seed_box_default`` are read.
* a stale population index in a METADATA leg recommendation is replaced.
* entry checks shared by ``compute_cumulants`` and ``dd.Config``.
* a failed k >= 3 slice evaluation marks the result incomplete and warns.

Inline public models only.  Run: sage -python -m pytest tests/test_audit_w3a_core.py -q
"""
import os
import sys
import types
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import daedalus as dd                                               # noqa: E402
from api.compute import (compute_cumulants, PhaseJIncompleteError,  # noqa: E402
                         PhaseJIncompleteWarning)
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE, SDEError                                   # noqa: E402


# ── models ───────────────────────────────────────────────────────────────────

_A = np.array([[-0.7, 0.3], [0.4, -1.3]])
_B = np.array([1.5, -1.0])
_Q = np.diag([0.6, 0.9])


def _linear2():
    return (SDE('w3a linear 2-field').field('x').field('y')
            .drift('x', '1.5 - 0.7*x + 0.3*y').drift('y', '-1 - 1.3*y + 0.4*x')
            .noise('xi', Gaussian(var='0.6'), couples={'x': 1})
            .noise('eta', Gaussian(var='0.9'), couples={'y': 1}).build())


def _cubic():
    # dx = (1 - x - 0.1 x^3) dt + sqrt(0.3) dW: one stable root, monostable
    # on the whole line, so its stationary density is exact
    return (SDE('w3a cubic').field('x').drift('x', '1 - x - 0.1*x**3')
            .noise('xi', Gaussian(var='0.3'), couples={'x': 1}).build())


def _ou():
    return (SDE('w3a ou').parameter('mu', default=1.0)
            .parameter('D', default=1.0, domain='positive').field('x')
            .drift('x', '-mu*x').noise('xi', Gaussian(var='2*D'),
                                       couples={'x': 1}).build())


def _double_well():
    # dx = (x - 0.1 x^3) dt + noise (mu = -1): stable roots -sqrt(10),
    # +sqrt(10), the unstable root 0 rejected by the stability analysis
    from api.model import TemporalModelBuilder
    return (TemporalModelBuilder('w3a double well')
            .population('pop', size=1)
            .physical_field('x', population='pop', domain='real')
            .parameter('mu', default=-1.0, domain='real')
            .parameter('eps', default=0.1, domain='positive')
            .parameter('D', default=0.1, domain='positive')
            .set_action_text('sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) '
                             '- D*xt[i]^2 for i in pop)')
            .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='pop')
            .stability_analysis(True).build())


def _two_units():
    return (SDE('w3a two units').population('E', size=2)
            .parameter('a', default=[1.0, 2.5], indexed=True)
            .field('x', population='E')
            .noise('xi', Gaussian(var='1'), couples={'x': 1})
            .equation('x', lhs='(Dt + 1)*x',
                      rhs='a + 0.2*sum(x[j] for j in E)').build())


def _run(m, **cfg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return dd.run(m, dd.Config(**cfg))


# ── ACC-01: every block from its own legs ────────────────────────────────────

def _isserlis(fields, tau, central):
    from scipy.linalg import expm, solve_continuous_lyapunov
    mu = -np.linalg.solve(_A, _B)
    S = solve_continuous_lyapunov(_A, -_Q)
    ix = {'x': 0, 'y': 1}

    def cov(i, ti, j, tj):                         # <dX_i(ti) dX_j(tj)>
        if tj >= ti:
            return (expm(_A * (tj - ti)) @ S)[j, i]
        return (expm(_A * (ti - tj)) @ S)[i, j]

    out = []
    for t in tau:
        times = [0.0] * len(fields)
        times[1] = t

        def rec(rem):
            if not rem:
                return 1.0
            a, tot = rem[0], 0.0
            if not central:
                tot += mu[ix[fields[a]]] * rec(rem[1:])
            for j in rem[1:]:
                tot += cov(ix[fields[a]], times[a], ix[fields[j]], times[j]) \
                    * rec([q for q in rem[1:] if q != j])
            return tot
        out.append(rec(list(range(len(fields)))))
    return np.array(out)


@pytest.mark.parametrize('fields', [['x', 'y'], ['y', 'x', 'x'],
                                    ['x', 'y', 'x', 'y']])
@pytest.mark.parametrize('output', ['moment', 'central_moment'])
def test_mixed_field_moment_matches_isserlis(fields, output):
    tau = np.array([-1.2, -0.4, 0.0, 0.5, 1.5])
    r = _run(_linear2(), k=len(fields), max_ell=0, tau_grid=tau,
             external_fields=[(f, 1) for f in fields], output=output)
    got = np.real(np.asarray(r['moment']))
    want = _isserlis(fields, tau, output == 'central_moment')
    # k = 2: tau = 0 is the exact tie (no nudge); k >= 3 slices still place
    # coincident legs 1e-6 apart (``_kpoint_slice_times``)
    tol = 1e-12
    assert np.allclose(got, want, rtol=tol, atol=tol), (got, want)


def test_population_legs_use_their_own_saddle():
    """moment - cumulant = mu_a mu_b with the saddle of each leg's unit."""
    A = -np.eye(2) + 0.2 * np.ones((2, 2))
    mu = -np.linalg.solve(A, [1.0, 2.5])                  # 13/6, 11/3
    tau = np.array([-1.0, 0.0, 1.0])
    for ext in ([('x', 1), ('x', 2)], [('x', 2), ('x', 2)]):
        r = _run(_two_units(), k=2, max_ell=0, tau_grid=tau,
                 external_fields=ext, output='moment')
        diff = np.real(np.asarray(r['moment'])) - np.real(np.asarray(r['C_tau']))
        want = mu[ext[0][1] - 1] * mu[ext[1][1] - 1]
        assert np.allclose(diff, want, rtol=1e-12), (ext, diff, want)


# ── ACC-02: loop corrections of the means ────────────────────────────────────

def _cubic_exact_moments():
    import mpmath as mp
    mp.mp.dps = 30
    w = lambda x: mp.exp(2 * (x - x**2 / 2 - 0.1 * x**4 / 4) / 0.3)  # noqa: E731
    Z = mp.quad(w, [-mp.inf, 0, 1, mp.inf])
    return [float(mp.quad(lambda x: x**n * w(x), [-mp.inf, 0, 1, mp.inf]) / Z)
            for n in (1, 2, 3)]


def test_loop_moment_includes_the_mean_tadpoles():
    """<x^k> at max_ell = 1 and 2 against the exact stationary density: the
    error falls by more than 10x per loop order (it stayed at the tree's
    5 % for <x^2> when the mean's tadpole was left out)."""
    exact = _cubic_exact_moments()
    m = _cubic()
    for k in (1, 2, 3):
        err = []
        for L in (0, 1, 2):
            r = _run(m, k=k, max_ell=L, external_fields=[('x', 1)] * k,
                     tau_grid=[0.0], output='moment')
            got = float(np.real(np.asarray(r['moment'])).ravel()[0])
            err.append(abs(got - exact[k - 1]) / exact[k - 1])
        assert err[1] < 0.1 * err[0] and err[2] < 0.1 * err[1], (k, err)
    # the cluster identity at L = 1 for k = 2, from the run's own pieces
    r = _run(m, k=2, max_ell=1, external_fields=[('x', 1)] * 2,
             tau_grid=[0.0], output='moment')
    mu0, mu1 = r['moment_means'][('x', 1)]
    k2 = sum(float(np.real(np.asarray(v)).ravel()[0])
             for v in r['C_tau_by_ell'].values())
    M = float(np.real(np.asarray(r['moment'])).ravel()[0])
    assert abs(M - (k2 + mu0 * mu0 + 2 * mu0 * mu1)) < 1e-12


# ── ACC-12: METADATA fixed-point defaults ────────────────────────────────────

def test_metadata_fixed_point_index_default_is_used():
    m = _double_well()
    mod = types.SimpleNamespace(METADATA={'fixed_point_index_default': 1})
    tau = dict(tau_grid=[0.0, 1.0], external_fields=[('dx', 1)] * 2)
    r0 = _run(m, k=2, max_ell=0, **tau)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r1 = dd.run(m, dd.Config(k=2, max_ell=0, **tau), mod)
        r2 = dd.run(m, dd.Config(k=2, max_ell=0, fixed_point_index=0, **tau),
                    mod)
    x0 = r0['mf_values']['xstar'][0]
    x1 = r1['mf_values']['xstar'][0]
    assert r1['mf_index_used'] == 1 and abs(x1 + x0) < 1e-9 and abs(x0) > 3
    assert r2['mf_index_used'] == 0                # the Config value wins


@pytest.mark.parametrize('bad', [-1, 1.7])
def test_metadata_fixed_point_index_default_is_checked(bad):
    mod = types.SimpleNamespace(METADATA={'fixed_point_index_default': bad})
    with pytest.raises((ValueError, TypeError), match='fixed_point_index'):
        dd.run(_double_well(), dd.Config(k=2, max_ell=0, tau_grid=[0.0],
                                         external_fields=[('dx', 1)] * 2), mod)


def test_metadata_seed_box_must_be_a_dict():
    mod = types.SimpleNamespace(METADATA={'seed_box_default': "{'x': (1, 5"})
    with pytest.raises(ValueError, match='seed_box_default'):
        dd.run(_double_well(), dd.Config(k=2, max_ell=0, tau_grid=[0.0],
                                         external_fields=[('dx', 1)] * 2), mod)


# ── LOUD-07: a stale population index in a recommendation ────────────────────

def test_stale_population_leg_recommendation_is_replaced():
    m1, m2 = _ou(), _two_units()
    assert not dd._ext_is_valid(m1, [('x', 1), ('x', 2)], 2)
    assert not dd._ext_is_valid(m1, [('x', 1), ('x', 1.0)], 2)
    assert dd._ext_is_valid(m1, [('x', 1), ('x', 1)], 2)
    assert dd._ext_is_valid(m2, [('x', 1), ('x', 2)], 2)
    assert not dd._ext_is_valid(m2, [('x', 1), ('x', 3)], 2)
    mod = types.SimpleNamespace(
        METADATA={'recommended_external_fields': [('x', 1), ('x', 2)]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dd.run(m1, dd.Config(k=2, max_ell=0, tau_grid=[0.0]), mod)
    assert [tuple(e) for e in r['_resolved']['external_fields']] == \
        [('x', 1), ('x', 1)]


# ── ACC-19 / ACC-20 / CON-03: entry checks ───────────────────────────────────

_KW = dict(k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)])


@pytest.mark.parametrize('bad, exc, match', [
    (dict(parameters={'muu': 5.0}), ValueError, 'unknown parameter'),
    (dict(fundamental={'muu': 5.0}), ValueError, 'unknown parameter'),
    (dict(tau_grid=(-5, 5, 51)), ValueError, 'ambiguous'),
    (dict(tau_grid=[-5, 5, 51]), ValueError, 'ambiguous'),
    (dict(accuracy=0), ValueError, 'accuracy'),
    (dict(accuracy=-1e-8), ValueError, 'accuracy'),
    (dict(accuracy=float('nan')), ValueError, 'accuracy'),
    (dict(n_workers=0), ValueError, 'n_workers'),
    (dict(discard=1), ValueError, 'discard'),
    (dict(discard='yes'), ValueError, 'discard'),
    (dict(fixed_point_index=-1), ValueError, 'fixed_point_index'),
])
def test_compute_cumulants_entry_checks(bad, exc, match):
    kw = dict(_KW, tau_grid=[0.0])
    kw.update(bad)
    with pytest.raises(exc, match=match):
        compute_cumulants(_ou(), verbose=False, parallel=False, **kw)


@pytest.mark.parametrize('k, max_ell', [(2.0, 0), (2, 0.5), (True, 0)])
def test_compute_cumulants_integer_k_and_max_ell(k, max_ell):
    with pytest.raises(TypeError, match='integer'):
        compute_cumulants(_ou(), k=k, max_ell=max_ell,
                          external_fields=[('x', 1), ('x', 1)],
                          tau_grid=[0.0], verbose=False, parallel=False)


def test_compute_cumulants_accepts_numpy_bool_discard_and_unsorted_grid():
    m = _ou()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, tau_grid=[1.0, 0.0, 0.5], discard=np.False_,
                              verbose=False, parallel=False, **_KW)
        s = compute_cumulants(m, tau_grid=[0.0, 0.5, 1.0], verbose=False,
                              parallel=False, **_KW)
    # an unsorted grid is taken as given: values aligned with its points
    C, Cs = np.real(r['C_tau']), np.real(s['C_tau'])
    assert np.allclose(C, Cs[[2, 0, 1]], rtol=1e-12, atol=0)
    assert np.allclose(Cs[1:], np.exp(-np.array([0.5, 1.0])), rtol=1e-12)


@pytest.mark.parametrize('bad', [dict(k=0), dict(max_ell=-1),
                                 dict(tau_step=0), dict(tau_max=-1.0),
                                 dict(n_workers=0), dict(k=2.5),
                                 dict(fixed_point_index=-1)])
def test_config_entry_checks(bad):
    with pytest.raises((ValueError, TypeError)):
        dd.Config(**bad)


def test_config_accepts_sage_and_numpy_integers():
    from sage.all import Integer
    c = dd.Config(k=Integer(2), max_ell=np.int64(1), n_workers=Integer(1))
    assert c.k == 2 and c.max_ell == 1


def test_config_checked_again_at_run():
    cfg = dd.Config(k=2, max_ell=0, tau_grid=[0.0])
    cfg.max_ell = -1
    with pytest.raises(ValueError, match='max_ell'):
        dd.run(_ou(), cfg)


@pytest.mark.parametrize('name', ['xt', 'dx', 'xstar'])
def test_sde_refuses_derived_names(name):
    s = (SDE('w3a derived').parameter(name, default=1.0).field('x')
         .drift('x', '-x').noise('xi', Gaussian(var='1'), couples={'x': 1}))
    with pytest.raises(SDEError, match='derives'):
        s.build()


def test_sde_refuses_a_parameter_declared_twice():
    with pytest.raises(SDEError, match='twice'):
        SDE('w3a twice').parameter('a', default=1.0).parameter('a', default=2.0)


# ── ACC-73: k >= 3 slice failures and resampling are reported ────────────────

def test_k3_slice_failure_marks_incomplete_and_warns(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError('slice evaluation failed')
    monkeypatch.setattr(dd, '_kpoint_slice_times', boom)
    cfg = dict(k=3, max_ell=0, tau_grid=[0.0, 1.0],
               external_fields=[('x', 1)] * 3)
    with pytest.warns(PhaseJIncompleteWarning, match='slices'):
        r = dd.run(_ou(), dd.Config(**cfg))
    assert r['status'] == 'incomplete' and r['C_tau'] is None
    assert 'slice evaluation failed' in r['_kpoint_slice_error']
    with pytest.raises(PhaseJIncompleteError):
        dd.run(_ou(), dd.Config(policy={'mode': 'strict'}, **cfg))


def test_k3_moment_after_a_slice_failure_is_none(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError('slice evaluation failed')
    monkeypatch.setattr(dd, '_kpoint_slice_times', boom)
    with pytest.warns(PhaseJIncompleteWarning, match='slices'):
        r = dd.run(_ou(), dd.Config(k=3, max_ell=0, tau_grid=[0.0, 1.0],
                                    external_fields=[('x', 1)] * 3,
                                    output='moment'))
    assert r['moment'] is None and r['status'] == 'incomplete'


def test_k3_full_grid_failure_keeps_the_slices(monkeypatch):
    real = dd._kpoint_pinned_time

    def boom(*a, **kw):                 # used by the full grid only
        raise RuntimeError('grid evaluation failed')
    monkeypatch.setattr(dd, '_kpoint_pinned_time', boom)
    monkeypatch.setattr(dd, '_kpoint_slice_times',
                        lambda base, j, t, eps, _f=dd._kpoint_slice_times:
                        _with(real, _f, base, j, t, eps))
    with pytest.warns(PhaseJIncompleteWarning, match='full k-point grid'):
        r = dd.run(_ou(), dd.Config(k=3, max_ell=0, tau_grid=[0.0, 1.0],
                                    external_fields=[('x', 1)] * 3,
                                    kpoint_full_grid=True))
    assert r['C_tau'] is not None and 'C_tau_grid' not in r
    assert r['status'] == 'incomplete' and '_kpoint_grid_error' in r


def _with(real, f, *args):
    # run the slice-times helper with the real pinned-time function
    saved = dd._kpoint_pinned_time
    dd._kpoint_pinned_time = real
    try:
        return f(*args)
    finally:
        dd._kpoint_pinned_time = saved


def test_spatial_moment_accepts_list_legs():
    m, mod = dd.load_model('linear_diffusion_test')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dd.run(m, dd.Config(k=2, max_ell=0, chi_grid=[0.0, 1.0],
                                tau_max=0.0, tau_step=1.0, output='moment',
                                external_fields=[['phi', 1], ['phi', 1]]), mod)
    assert np.allclose(np.real(r['moment']), np.real(r['C_tau_x'])
                       + r['moment_means'][('phi', 1)][0] ** 2, rtol=1e-12, atol=1e-12)


def test_k3_resampled_grid_warns():
    with pytest.warns(UserWarning, match='replaced by 41'):
        r = dd.run(_ou(), dd.Config(k=3, max_ell=0, tau_grid=(0.0, 2.0, 50),
                                    external_fields=[('x', 1)] * 3))
    assert np.asarray(r['tau_grid']).size == 41
    assert '_kpoint_tau_resampled' in r

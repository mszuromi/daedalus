"""
tests/test_covariance_matrix.py
===============================
The k = 2 covariance matrix over many fields in one call, and its parameter
derivatives (``api/covariance.py``; ``dd.covariance_matrix``,
``dd.Config(covariance=...)``).

* every computed entry equals the per-entry ``compute_cumulants`` result on
  the same grid bit for bit, per loop order too (OU quartic two-field, a
  three-field linear-plus-cubic model, the linear Hawkes population model),
  with the parameter-point stages run once (``stage_memo``);
* ``Sigma_ij(tau) = Sigma_ji(-tau)``; the tau = 0 point of a lower entry is
  its own left limit, not the reflection;
* ``dSigma/dtheta`` against the exact Lyapunov / Frechet derivative of a
  linear model, and against central differences of the full pipeline at one
  loop, also for a parameter that moves the saddle; a stencil that crosses a
  bifurcation is refused;
* the shared pass costs less CPU than the N(N+1)/2 separate calls.

Inline public models or tracked ones only.
"""
import contextlib
import io
import os
import sys

import numpy as np
import pytest

pytest.importorskip('sage.all')

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)

import daedalus as dd                                                # noqa: E402
from api import compute_cumulants                                    # noqa: E402
from api import covariance as CV                                     # noqa: E402
from api.compute import stage_memo                                   # noqa: E402
from api.noise import Gaussian, Poisson                              # noqa: E402
from api.sde import SDE                                              # noqa: E402
from scipy.linalg import expm_frechet, solve_continuous_lyapunov     # noqa: E402
from tests._cputime import CpuClock                                  # noqa: E402

GRID = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
QUARTIC_P = {'J1': 0.7, 'J2': 0.2}


def _entry(model, a, b, ell, grid, **kw):
    return compute_cumulants(model, 2, ell, external_fields=[a, b],
                             tau_grid=grid, verbose=False, parallel=False, **kw)


def _assert_bitwise(model, cm, ell, grid, lower=False, **kw):
    """Every computed entry (the upper triangle; with ``lower`` every one)
    is the per-entry run, bit for bit, in total and per loop order."""
    legs = cm['fields']
    for i, a in enumerate(legs):
        for j, b in enumerate(legs):
            if j < i and not lower:
                continue
            r = _entry(model, a, b, ell, grid, **kw)
            assert np.array_equal(r['C_tau'], cm['Sigma'][:, i, j]), (a, b)
            assert sorted(r['C_tau_by_ell']) == sorted(cm['Sigma_by_ell'])
            for ell_, v in r['C_tau_by_ell'].items():
                assert np.array_equal(v, cm['Sigma_by_ell'][ell_][:, i, j]), \
                    (a, b, ell_)


def _linear():
    return (SDE('covariance linear two-field OU')
            .parameter('a', default=1.0).parameter('b', default=1.7)
            .parameter('c', default=0.4).parameter('g', default=-0.3)
            .parameter('D', default=0.6).parameter('F', default=0.9)
            .field('x').field('y')
            .drift('x', '-a*x + c*y').drift('y', 'g*x - b*y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .noise('eta', Gaussian(var='2*F'), couples={'y': 1})).build()


def _three_field():
    return (SDE('covariance three-field cubic')
            .parameter('a', default=1.0).parameter('b', default=1.4)
            .parameter('c', default=0.8).parameter('u', default=0.1)
            .parameter('k1', default=0.3).parameter('k2', default=0.2)
            .parameter('D', default=0.5)
            .field('x').field('y').field('z')
            .drift('x', '-a*x - u*x^3 + k1*y')
            .drift('y', 'k1*x - b*y + k2*z').drift('z', 'k2*y - c*z')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .noise('eta', Gaussian(var='2*D'), couples={'y': 1})
            .noise('zeta', Gaussian(var='2*D'), couples={'z': 1})).build()


@pytest.fixture(scope='module')
def quartic():
    return dd.load_model('ou_quartic_coupled')[0]


# ── bit identity ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('pairs', ['upper', 'all'])
def test_quartic_two_field_matrix_is_the_per_entry_runs(quartic, pairs):
    cm = CV.covariance_matrix(quartic, [('x', 1), ('y', 1)], 1,
                              parameters=QUARTIC_P, tau_grid=GRID,
                              pairs=pairs, parallel=False)
    assert cm['status'] == 'ok'
    # the parameter-point stages ran once for the whole matrix
    assert cm['shared_stages'] == {'hits': len(cm['runs']) - 1, 'misses': 1}
    _assert_bitwise(quartic, cm, 1, GRID, lower=pairs == 'all',
                    parameters=QUARTIC_P)
    total = sum(cm['Sigma_by_ell'].values())
    np.testing.assert_allclose(total, cm['Sigma'], rtol=1e-15, atol=1e-17)


def test_three_field_linear_plus_cubic_is_the_per_entry_runs():
    m = _three_field()
    cm = CV.covariance_matrix(m, ['x', 'y', 'z'], 1, tau_grid=GRID,
                              parallel=False, use_cache=False)
    assert cm['fields'] == [('x', 1), ('y', 1), ('z', 1)]
    assert cm['shared_stages']['misses'] == 1
    _assert_bitwise(m, cm, 1, GRID, use_cache=False)
    s0 = cm['Sigma0'].real
    np.testing.assert_allclose(s0, s0.T, rtol=1e-13)
    assert np.all(np.linalg.eigvalsh(s0) > 0)


def test_population_model_is_the_per_entry_runs():
    hk = dd.load_model('linear_hawkes')[0]
    legs = [('n', 1), ('n', 2), ('v', 1)]
    grid = np.array([-2.0, 0.0, 1.0, 2.0])
    cm = CV.covariance_matrix(hk, legs, 1, tau_grid=grid, parallel=False)
    assert cm['status'] == 'ok' and cm['shared_stages']['misses'] == 1
    _assert_bitwise(hk, cm, 1, grid)
    # tau = 1 has no mirror on this grid: those lower entries come from the
    # (j, i) runs on the unmirrored points, the others by reflection
    pts = [r for r in cm['runs'] if r['tau_grid'] == 'points']
    assert len(pts) == 3 and all(r['n_tau'] == 2 for r in pts)
    assert CV.default_fields(hk) == [('n', 1), ('n', 2), ('v', 1), ('v', 2)]


def test_stage_memo_separates_parameter_points_and_is_off_outside(quartic):
    p2 = dict(QUARTIC_P, J1=0.6)
    with stage_memo() as memo:
        a1 = _entry(quartic, ('x', 1), ('y', 1), 1, GRID, parameters=QUARTIC_P)
        b2 = _entry(quartic, ('x', 1), ('x', 1), 1, GRID, parameters=p2)
        a2 = _entry(quartic, ('x', 1), ('y', 1), 1, GRID, parameters=p2)
    assert (memo.hits, memo.misses) == (1, 2)
    assert np.array_equal(
        a1['C_tau'], _entry(quartic, ('x', 1), ('y', 1), 1, GRID,
                            parameters=QUARTIC_P)['C_tau'])
    assert np.array_equal(
        a2['C_tau'], _entry(quartic, ('x', 1), ('y', 1), 1, GRID,
                            parameters=p2)['C_tau'])
    assert b2['propagator'] is a2['propagator']
    assert a1['propagator'] is not a2['propagator']
    # outside the block nothing is shared
    r1 = _entry(quartic, ('x', 1), ('y', 1), 1, GRID, parameters=QUARTIC_P)
    r2 = _entry(quartic, ('x', 1), ('y', 1), 1, GRID, parameters=QUARTIC_P)
    assert r1['propagator'] is not r2['propagator']


# ── symmetry ─────────────────────────────────────────────────────────────────

def test_symmetry_sigma_ij_tau_is_sigma_ji_minus_tau(quartic):
    grid = np.linspace(-3.0, 3.0, 7)
    full = CV.covariance_matrix(quartic, ['x', 'y'], 1, parameters=QUARTIC_P,
                                tau_grid=grid, pairs='all', parallel=False)
    S = full['Sigma']
    np.testing.assert_allclose(S, np.transpose(S[::-1], (0, 2, 1)),
                               rtol=1e-12, atol=1e-15)
    for ell, v in full['Sigma_by_ell'].items():
        np.testing.assert_allclose(v, np.transpose(v[::-1], (0, 2, 1)),
                                   rtol=1e-11, atol=1e-15)
    up = CV.covariance_matrix(quartic, ['x', 'y'], 1, parameters=QUARTIC_P,
                              tau_grid=grid, parallel=False)
    assert up['reflected'] == 6
    # reflected entries are the (0, 1) run at -tau exactly
    assert np.array_equal(up['Sigma'][grid != 0, 1, 0],
                          up['Sigma'][::-1][grid != 0, 0, 1])
    np.testing.assert_allclose(up['Sigma'], S, rtol=1e-13, atol=1e-16)


def _spiking():
    """Spikes n of rate a*v; v jumps by w/tau at each spike, so <n(0) v(tau)>
    jumps at tau = 0."""
    return (SDE('covariance spike-driven rate')
            .parameter('Em', default=0.5).parameter('tau', default=2.0)
            .parameter('a', default=1.0).parameter('w', default=0.4)
            .define_function('phi', args=['v'], expression='a*v')
            .noise('spikes', Poisson(rate='phi(v)'), expose='n')
            .field('v').equation('v', lhs='(tau*Dt + 1)*v',
                                 rhs='Em + w*n')).build()


def test_tau0_of_a_lower_entry_is_its_own_left_limit():
    """C_nv(0-) != C_nv(0+): the (v, n) entry at tau = 0 is C_vn(0-) =
    C_nv(0+), computed by its own run, not the reflection of C_nv(0)."""
    m = _spiking()
    grid = np.array([-1.0, 0.0, 1.0])
    cm = CV.covariance_matrix(m, None, 1, tau_grid=grid, parallel=False,
                              use_cache=False)
    assert cm['fields'] == [('n', 1), ('v', 1)]
    near = np.array([-1e-3, 0.0, 1e-3])
    nv = _entry(m, ('n', 1), ('v', 1), 1, near, use_cache=False)['C_tau'].real
    vn = _entry(m, ('v', 1), ('n', 1), 1, np.array([0.0]),
                use_cache=False)['C_tau'].real
    assert abs(nv[2] - nv[0]) > 0.1                   # the jump
    np.testing.assert_allclose(nv[1], nv[0], rtol=1e-3)   # C_nv(0) = 0-
    np.testing.assert_allclose(vn[0], nv[2], rtol=1e-3)   # C_vn(0) = C_nv(0+)
    np.testing.assert_allclose(cm['Sigma0'].real,
                               [[cm['Sigma0'][0, 0].real, nv[1]],
                                [vn[0], cm['Sigma0'][1, 1].real]],
                               rtol=1e-13)
    pts = [x for x in cm['runs'] if x['tau_grid'] == 'points']
    assert len(pts) == 1 and pts[0]['n_tau'] == 1 and cm['reflected'] == 2


# ── derivatives ──────────────────────────────────────────────────────────────

def _lyapunov_reference(p, grid, lab, h=1e-6):
    def A(q):
        return np.array([[-q['a'], q['c']], [q['g'], -q['b']]])

    def Q(q):
        return np.diag([2 * q['D'], 2 * q['F']])
    pp, pm = dict(p), dict(p)
    pp[lab] += h
    pm[lab] -= h
    a, dA = A(p), (A(pp) - A(pm)) / (2 * h)       # A, Q are linear in p
    dQ = (Q(pp) - Q(pm)) / (2 * h)
    S = solve_continuous_lyapunov(a, -Q(p))
    dS = solve_continuous_lyapunov(a, -(dA @ S + S @ dA.T + dQ))
    C, dC = [], []
    for t in grid:          # C(tau) = <x(0) x(tau)^T> = S e^{A^T tau}, tau >= 0
        if t >= 0:
            E, dE = expm_frechet(a.T * t, dA.T * t)
            C.append(S @ E)
            dC.append(dS @ E + S @ dE)
        else:
            E, dE = expm_frechet(a * -t, dA * -t)
            C.append(E @ S)
            dC.append(dE @ S + E @ dS)
    return np.array(C), np.array(dC)


def test_derivatives_of_a_linear_model_are_the_lyapunov_derivative():
    p0 = dict(a=1.0, b=1.7, c=0.4, g=-0.3, D=0.6, F=0.9)
    cm = CV.covariance_matrix(_linear(), ['x', 'y'], 0, tau_grid=GRID,
                              parallel=False, use_cache=False,
                              derivatives=['a', 'c', 'D'])
    C, _ = _lyapunov_reference(p0, GRID, 'a')
    np.testing.assert_allclose(cm['Sigma'].real, C, rtol=1e-13, atol=1e-15)
    for lab in ('a', 'c', 'D'):
        d = cm['derivatives'][lab]
        _, dC = _lyapunov_reference(p0, GRID, lab)
        err = np.abs(d['dSigma'] - dC)
        assert err.max() < 1e-9, (lab, err.max())
        assert np.all(err <= d['error'] + 1e-12), lab   # the estimate bounds it
        assert d['status'] == 'ok'
        assert d['step'] == pytest.approx(1e-3 * p0[lab])
        # the saddle of a linear model does not move
        assert d['dmf'] == {'xstar': [0.0], 'ystar': [0.0]} or all(
            abs(v[0]) < 1e-12 for v in d['dmf'].values())


@pytest.mark.parametrize('lab, params', [
    ('eps1', QUARTIC_P),
    # mu1 < 0 with a strong coupling: a nonzero saddle that mu1 moves
    ('mu1', {'mu1': -0.4, 'mu2': 1.0, 'J1': 0.2, 'J2': 0.3}),
])
def test_derivatives_match_central_differences_of_the_full_pipeline(
        quartic, lab, params):
    cm = CV.covariance_matrix(quartic, ['x', 'y'], 1, parameters=params,
                              tau_grid=GRID, parallel=False,
                              derivatives=[lab])
    d = cm['derivatives'][lab]
    base = CV._resolved_parameters(quartic, params)
    h = 1e-4 * max(1.0, abs(base[lab]))
    fd = np.zeros_like(cm['Sigma'])
    fd_ell = {e: np.zeros_like(v) for e, v in cm['Sigma_by_ell'].items()}
    for sgn in (1, -1):
        p = dict(base, **{lab: base[lab] + sgn * h})
        for i, a in enumerate(cm['fields']):
            for j, b in enumerate(cm['fields']):
                r = _entry(quartic, a, b, 1, GRID, parameters=p)
                fd[:, i, j] += sgn * r['C_tau'] / (2 * h)
                for e, v in r['C_tau_by_ell'].items():
                    fd_ell[e][:, i, j] += sgn * v / (2 * h)
    scale = np.max(np.abs(d['dSigma']))
    assert scale > 1e-3
    assert np.max(np.abs(d['dSigma'] - fd)) < 1e-6 * scale
    for e in fd_ell:
        assert np.max(np.abs(d['dSigma_by_ell'][e] - fd_ell[e])) < 1e-6 * max(
            scale, 1e-3)
    if lab == 'mu1':
        x0 = np.asarray(cm['mf_values']['xstar'])
        assert abs(x0[0]) > 0.1                       # off the origin
        for key, v in d['dmf'].items():
            np.testing.assert_allclose(v, d['dmf_fd'][key], rtol=1e-6,
                                       atol=1e-9)
        assert abs(d['dmf']['xstar'][0]) > 0.1        # and it moves


def test_a_stencil_across_a_bifurcation_is_refused():
    """x* = -sqrt(m) for m > 0 and 0 for m <= 0: a stencil reaching m < 0
    leaves the branch, and is refused instead of differenced."""
    m = (SDE('covariance pitchfork').parameter('m', default=0.01)
         .parameter('D', default=1e-4).field('x').drift('x', 'm*x - x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()
    with pytest.raises(CV.CovarianceDerivativeError, match='branch'):
        CV.covariance_matrix(m, ['x'], 0, tau_grid=np.array([0.0]),
                             parallel=False, use_cache=False,
                             derivatives={'parameters': ['m'],
                                          'abs_step': {'m': 0.008}})
    # a step that stays on the branch: dx*/dm = -1 / (2 sqrt(m)) = -5
    ok = CV.covariance_matrix(m, ['x'], 0, tau_grid=np.array([0.0]),
                              parallel=False, use_cache=False,
                              derivatives=['m'])
    np.testing.assert_allclose(ok['derivatives']['m']['dmf']['xstar'], [-5.0],
                               rtol=1e-6)


def test_derivative_parameter_specs():
    hk = dd.load_model('linear_hawkes')[0]
    vals = CV._resolved_parameters(hk, None)
    t = CV._parameter_targets(['w[0,1]', ('E', 1), 'tau'], vals, hk)
    assert [x[0] for x in t] == ['w[0,1]', 'E[1]', 'tau']
    assert [x[0] for x in CV._parameter_targets('E', vals, hk)] == \
        ['E[0]', 'E[1]']
    with pytest.raises(ValueError, match='outside the shape'):
        CV._parameter_targets(['w[2,0]'], vals, hk)
    with pytest.raises(ValueError, match='not a parameter'):
        CV._parameter_targets(['nope'], vals, hk)
    with pytest.raises(ValueError, match='unknown option'):
        CV._derivative_options({'parameters': ['tau'], 'order': 6})


# ── cost ─────────────────────────────────────────────────────────────────────

def test_shared_pass_costs_less_than_the_separate_calls(quartic):
    grid = np.linspace(-3.0, 3.0, 13)
    kw = dict(parameters=QUARTIC_P, tau_grid=grid, parallel=False)
    CV.covariance_matrix(quartic, ['x', 'y'], 1, **kw)          # warm
    best = None
    for _ in range(2):
        clock = CpuClock()
        CV.covariance_matrix(quartic, ['x', 'y'], 1, **kw)
        shared = clock.elapsed()
        clock.reset()
        for a, b in ([('x', 1), ('x', 1)], [('x', 1), ('y', 1)],
                     [('y', 1), ('y', 1)]):                   # N(N+1)/2 calls
            compute_cumulants(quartic, 2, 1, external_fields=[a, b],
                              verbose=False, **kw)
        ratio = shared / clock.elapsed()
        best = ratio if best is None else min(best, ratio)
    # ~0.55 measured (MF and poles are ~60 % of one call here)
    assert best < 0.85, best


# ── entry points and refusals ────────────────────────────────────────────────

def test_config_covariance_runs_the_matrix(quartic):
    cfg = dd.Config(max_ell=1, parameters=QUARTIC_P, tau_grid=GRID,
                    covariance={'fields': ['x', 'y'], 'derivatives': ['J1']})
    res = dd.run(quartic, cfg)
    direct = dd.covariance_matrix(quartic, ['x', 'y'], 1,
                                  parameters=QUARTIC_P, tau_grid=GRID,
                                  derivatives=['J1'])
    assert np.array_equal(res['Sigma'], direct['Sigma'])
    assert np.array_equal(res['derivatives']['J1']['dSigma'],
                          direct['derivatives']['J1']['dSigma'])
    assert res['_resolved']['fields'] == [('x', 1), ('y', 1)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dd.config_options(False)
    assert 'covariance' in buf.getvalue()


@pytest.mark.parametrize('kw, match', [
    (dict(covariance={'pairs': 'lower'}), 'pairs'),
    (dict(covariance={'fieldz': ['x']}), 'unknown key'),
    (dict(covariance=True, output='moment'), 'output'),
    (dict(covariance=True, k=3), 'k=3'),
    (dict(covariance=True, spectra=True), 'spectra'),
])
def test_config_covariance_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        dd.Config(**kw)


def test_refusals(quartic):
    for key, val in (('external_fields', [('x', 1)] * 2), ('spectra', True),
                     ('t_min', 0.0), ('k', 2)):
        with pytest.raises(ValueError, match=f'{key}='):
            CV.covariance_matrix(quartic, ['x'], 0, tau_grid=GRID,
                                 **{key: val})
    with pytest.raises(ValueError, match='pairs'):
        CV.covariance_matrix(quartic, ['x'], 0, tau_grid=GRID, pairs='lower')
    with pytest.raises(ValueError, match='twice'):
        CV.covariance_matrix(quartic, ['x', ('x', 1)], 0, tau_grid=GRID)
    with pytest.raises(ValueError, match='repeated'):
        CV.covariance_matrix(quartic, ['x'], 0, tau_grid=np.array([0., 0.]))
    kpz = dd.load_model('kpz_1d')[0]
    with pytest.raises(NotImplementedError, match='temporal'):
        CV.covariance_matrix(kpz, None, 0, tau_grid=GRID)

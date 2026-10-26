"""
tests/test_resum.py
===================
The screened (optimized-perturbation) loop series, ``api/resum.py``
(``dd.Config(resum='screened')``).

* the counterterm parameters shift the propagator and nothing else: a run of
  the counterterm model equals the plain model at the shifted mass, order by
  order;
* order 1 at the Hartree mass (``κ = 1``) is the Hartree curve ``C(τ)`` and
  its spectrum (the counterterm cancels the tadpole);
* OU quartic against the exact Boltzmann / Fokker-Planck answer, at the
  prototype's numbers: PMS at three loops, equal-time error 8.6e-4 at
  ``g = 1`` and 4.5e-3 at ``g = 1000``; the lag correlator within 1 % of the
  exact one at ``g = 1`` (prototype table: 8.3e-3);
* an even order has no stationary point and takes the odd order's mass;
* the two-field coupled model (2-D Boltzmann quadrature) through the
  covariance layout, and a jump model (``dd.master_equation``) through the
  noise counterterm;
* a linear model: the trial is the tree, the series is the loop series;
* the default (``resum=None``) is the loop path, unchanged.

Inline public models or tracked ones only.
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import daedalus as dd  # noqa: E402
from api.compute import compute_cumulants  # noqa: E402
from api.resum import (screened_model, resolve_resum_options,  # noqa: E402
                       fd_weights, ResumWarning, _LogCheb)

C_LOOP = [1.0, -3.0, 24.0, -297.0]      # <x^2> = (D/mu) sum_l c_l g^l


def _exact_variance(mu, eps, D):
    from scipy.integrate import quad
    U = lambda x: (mu * x * x / 2 + eps * x ** 4 / 4) / D
    xm = 12 * math.sqrt(D / max(abs(mu), math.sqrt(eps * D)))
    num = quad(lambda x: x * x * math.exp(-U(x)), -xm, xm, limit=400)[0]
    return num / quad(lambda x: math.exp(-U(x)), -xm, xm, limit=400)[0]


def _screened_series(eps, m, N):
    """The equal-time screened variance of OU quartic (μ = D = 1) at trial
    mass ``m``, from the exact loop coefficients (the prototype's formula)."""
    tot = 0.0
    for ell in range(N + 1):
        p = 2 * ell + 1
        for k in range(N - ell + 1):
            fall = 1.0
            for j in range(k):
                fall *= -(p + j)
            tot += (C_LOOP[ell] * eps ** ell * fall * m ** (-p - k)
                    * (1.0 - m) ** k / math.factorial(k))
    return tot


def _pms_mass(eps, N, m_guess):
    """The stationary point of the screened series nearest ``m_guess``."""
    from scipy.optimize import brentq
    ms = np.geomspace(m_guess / 20, m_guess * 20, 4001)
    V = np.array([_screened_series(eps, m, N) for m in ms])
    dV = np.gradient(V, ms)
    idx = np.where(np.sign(dV[:-1]) != np.sign(dV[1:]))[0]
    best = idx[np.argmin(np.abs(np.log(ms[idx] / m_guess)))]
    f = lambda m: (_screened_series(eps, m * (1 + 1e-6), N)
                   - _screened_series(eps, m * (1 - 1e-6), N))
    return brentq(f, ms[best], ms[best + 1])


def _hartree_mass(eps):
    s2 = (-1 + math.sqrt(1 + 12 * eps)) / (6 * eps)
    return 1 / s2


def _fp_C(eps, tau, n=3001):
    """C(τ) of OU quartic (μ = D = 1) from a detailed-balance finite-volume
    Fokker-Planck operator (the prototype's reference)."""
    from scipy.linalg import eigh_tridiagonal
    from scipy.optimize import brentq
    L = brentq(lambda y: (y * y / 2 + eps * y ** 4 / 4) - 60, 1e-12, 1e6)
    x = np.linspace(-L, L, n)
    h = x[1] - x[0]
    U = x * x / 2 + eps * x ** 4 / 4
    dU = np.diff(U)
    kp = np.exp(-dU / 2) / h ** 2
    km = np.exp(dU / 2) / h ** 2
    diag = np.zeros(n)
    diag[:-1] -= kp
    diag[1:] -= km
    lam, V = eigh_tridiagonal(-diag, -np.full(n - 1, 1 / h ** 2))
    pi = np.exp(-(U - U.min()))
    pi /= pi.sum()
    w = ((x * np.sqrt(pi)) @ V) ** 2
    keep = lam > 1e-9 * lam.max()
    lam, w = lam[keep], w[keep]
    return (w[None, :] * np.exp(-np.outer(tau, lam))).sum(axis=1)


@pytest.fixture(scope='module')
def ou_quartic():
    return dd.load_model('ou_quartic')


def _run(model_mod, eps, N, resum, tau, **cfg):
    m, mod = model_mod
    return dd.run(m, dd.Config(k=2, max_ell=N, resum=resum, tau_grid=tau,
                               parameters={'mu': 1.0, 'eps': eps, 'D': 1.0},
                               **cfg), mod)


def test_counterterm_is_a_pure_propagator_shift(ou_quartic):
    """At trial position s the counterterm model is OU quartic with mass
    μ + s (m_H - μ), loop order by loop order (vertices unchanged)."""
    m = ou_quartic[0]
    p = {'mu': 1.0, 'eps': 0.3, 'D': 1.0}
    h = dd.hartree(m, p)
    mct, values = screened_model(m, h)
    assert mct['name'].endswith('[screened]')
    assert mct['spec_signature'] != m.get('spec_signature')
    assert [e[1] for e in mct['_screened_counterterm']] == ['h']
    s = 0.6
    tau = np.array([0.0, 0.5, 1.0])
    r1 = compute_cumulants(mct, 2, 2, dict(p, **values(s)),
                           [('dx', 1), ('dx', 1)], tau_grid=tau,
                           verbose=False)
    mass = 1.0 + s * (-h.A[0, 0] - 1.0)
    r2 = compute_cumulants(m, 2, 2, dict(p, mu=mass),
                           [('dx', 1), ('dx', 1)], tau_grid=tau,
                           verbose=False)
    for e in (0, 1, 2):
        assert np.allclose(np.real(r1['C_tau_by_ell'][e]),
                           np.real(r2['C_tau_by_ell'][e]), rtol=1e-12,
                           atol=0)


def test_order_one_at_hartree_mass_is_hartree(ou_quartic, tmp_path):
    tau = np.array([0.0, 0.2, 0.7, 1.5])
    om = np.array([0.0, 2.0])
    r = _run(ou_quartic, 3.0, 1, {'method': 'screened', 'mass': 'hartree'},
             tau, spectra={'omega': om})
    h = dd.hartree(ou_quartic[0], {'mu': 1.0, 'eps': 3.0, 'D': 1.0})
    info = r['resummation']
    assert info['kappa'] == 1.0 and info['kappa_source'] == 'hartree'
    assert np.allclose(np.real(r['C_tau_by_order'][1]), h.C(tau),
                       rtol=0, atol=1e-8 * h.cov[0, 0])
    assert np.allclose(np.real(r['spectra']['S']), np.real(h.spectrum(om)),
                       rtol=1e-7)
    assert abs(r['spectra']['tau_c'] - h.spectra()['tau_c']) < 1e-7
    # the plain loop series rides along, from the same call a plain run makes
    plain = _run(ou_quartic, 3.0, 1, None, tau)
    for e in (0, 1):
        assert np.array_equal(r['C_tau_loops_by_ell'][e],
                              plain['C_tau_by_ell'][e])
    assert np.array_equal(r['C_tau_loops'], plain['C_tau'])
    assert 'C_tau_by_ell' not in r and np.array_equal(
        r['C_tau'], r['C_tau_by_order'][1])
    # A_trial at κ = 1 is the Hartree drift
    assert np.allclose(info['A_trial'], h.A, rtol=1e-14)
    assert r['status'] == 'ok' and info['trial_runs_not_ok'] == []
    # a saved file says it is resummed and keeps the plain series
    from api.save import save_npz
    f = np.load(save_npz(r, str(tmp_path / 'screened.npz')))
    assert str(f['resum_method']) == 'screened'
    assert np.array_equal(f['C_total'], np.real(r['C_tau'])) or \
        np.array_equal(f['C_total'], r['C_tau'])
    assert 'C_screened_order_1' in f and 'C_loops_total' in f


def test_pms_three_loops_strong_coupling(ou_quartic):
    """g = 1000: the loop series is off by 1e13; PMS at three loops is within
    0.45 % (prototype table: +4.5e-3), at the analytic PMS mass."""
    g = 1000.0
    r = _run(ou_quartic, g, 3, 'screened', np.array([0.0]))
    info = r['resummation']
    mH = _hartree_mass(g)
    m3 = _pms_mass(g, 3, mH)
    assert info['kappa_source'] == 'pms order 3'
    assert abs(info['kappa'] - (m3 - 1) / (mH - 1)) < 1e-8
    ex = _exact_variance(1.0, g, 1.0)
    rel = float(np.real(r['C_tau'][0])) / ex - 1
    assert abs(rel - 4.47e-3) < 2e-5
    assert abs(float(np.real(r['C_tau'][0]))
               - _screened_series(g, m3, 3)) < 1e-7 * ex
    # the spread of the last two orders is a conservative estimate here
    assert info['spread_last_two'] > abs(rel)
    assert info['pms_stencil_check'] < 1e-6
    # the PMS curve has one stationary point at order 1 and 3, none at 2
    sp = info['stationary_points']
    assert len(sp[1]) == 1 and sp[2] == [] and len(sp[3]) == 1


def test_lag_correlator_g1(ou_quartic):
    """g = 1, three loops at the PMS mass: equal-time error 8.6e-4 and C(τ)
    within 1 % of the Fokker-Planck answer on [0, 6/m_H] (prototype table:
    8.3e-3; the plain three-loop series is off by 589x)."""
    g = 1.0
    mH = _hartree_mass(g)
    kappa = (_pms_mass(g, 3, mH) - 1) / (mH - 1)
    tau = np.linspace(0.0, 6.0 / mH, 13)
    r = _run(ou_quartic, g, 3, {'method': 'screened', 'mass': kappa}, tau)
    assert r['resummation']['kappa_source'] == 'given'
    ex = _fp_C(g, tau)
    c3 = np.real(r['C_tau'])
    assert abs(c3[0] / _exact_variance(1.0, g, 1.0) - 1 - 8.64e-4) < 1e-5
    err = np.max(np.abs(c3 - ex)) / ex[0]
    assert 5e-3 < err < 1e-2
    loops = np.real(r['C_tau_loops'])
    assert np.max(np.abs(loops - ex)) / ex[0] > 100
    # the screened orders converge toward the exact curve
    errs = [np.max(np.abs(np.real(r['C_tau_by_order'][M]) - ex)) / ex[0]
            for M in range(4)]
    assert errs[0] > errs[1] > errs[2] > errs[3]


def test_even_order_takes_the_odd_order_mass(ou_quartic):
    g = 1.0
    with pytest.warns(ResumWarning, match='no stationary point'):
        r = _run(ou_quartic, g, 2, 'screened', np.array([0.0]))
    info = r['resummation']
    mH = _hartree_mass(g)
    assert info['kappa_source'] == 'pms order 1 (order 2 has no stationary ' \
                                   'point)'
    assert abs(info['kappa'] - (_pms_mass(g, 1, mH) - 1) / (mH - 1)) < 1e-8
    assert info['stationary_points'][2] == []


COUPLED = {'mu1': 1.0, 'mu2': 1.0, 'eps1': 1.0, 'eps2': 0.5, 'J1': 0.5,
           'J2': 0.5, 'D1': 1.0, 'D2': 1.0}


@pytest.mark.slow
def test_two_field_covariance_against_boltzmann():
    """The coupled gradient model at strong coupling (2-D Boltzmann
    quadrature): Hartree is 8-13 % low, the loop series diverges (x15 at
    one loop); screened order 1 (PMS, the covariance layout) is within 10 %
    and order 2 at that mass within 2.5 % on the cross covariance."""
    from tests.test_hartree import _coupled_exact
    m, mod = dd.load_model('ou_quartic_coupled')
    ex = _coupled_exact(COUPLED)
    exm = np.array([[ex['xx'], ex['xy']], [ex['xy'], ex['yy']]])
    tau = np.array([0.0])
    r = dd.run(m, dd.Config(max_ell=1, resum='screened', parameters=COUPLED,
                            covariance=True, tau_grid=tau), mod)
    info = r['resummation']
    assert info['kappa_source'] == 'pms order 1'
    assert r['fields'] == [('x', 1), ('y', 1)]
    S = np.real(r['Sigma'][0])
    h = dd.hartree(m, COUPLED)
    err = lambda M: np.max(np.abs(M - exm) / np.abs(exm))
    assert err(S) < 0.1 < err(h.cov) < 0.15
    assert err(np.real(r['Sigma_loops'][0])) > 10
    assert np.allclose(S, S.T, rtol=1e-10)
    assert set(r['Sigma_by_order']) == {0, 1}
    r2 = dd.run(m, dd.Config(max_ell=2, parameters=COUPLED, tau_grid=tau,
                             external_fields=[('x', 1), ('y', 1)],
                             resum={'method': 'screened',
                                    'mass': info['kappa']}), mod)
    assert abs(float(np.real(r2['C_tau'][0])) / ex['xy'] - 1) < 0.03
    assert r2['resummation']['spread_last_two'] > 0.03


def test_jump_model_noise_counterterm():
    """Immigration, death and pairwise annihilation: the multiplicative
    Poisson noise gives a noise counterterm; screened order 2 is as close
    to ``dd.master_equation`` as the plain two-loop series is."""
    P = {'lam': 60.0, 'mu': 1.0, 'c': 0.02}
    me = dd.master_equation(
        transitions=[('lam', +1), ('mu*x', -1), ('c*x^2', -2)], fields=['x'],
        params=P, h=1.0, box=(0, 250))
    ex_v = float(me.cumulants(kmax=2).cumulants[2][0, 0])
    m = me.to_sde().build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ResumWarning)
        r = dd.run(m, dd.Config(k=2, max_ell=2, resum='screened',
                                parameters=dict(P, h=1.0),
                                tau_grid=np.array([0.0])))
    kinds = sorted({e[1] for e in r['resummation']['counterterm']})
    assert kinds == ['h', 'q']
    v = float(np.real(r['C_tau'][0]))
    v_l = float(np.real(r['C_tau_loops'][0]))
    assert abs(v / ex_v - 1) < 1e-4
    assert abs(v / ex_v - 1) < 3 * abs(v_l / ex_v - 1) + 1e-6


def test_linear_model_is_the_loop_series():
    from tests.test_hartree import _linear_two_field
    m = _linear_two_field()
    tau = np.array([0.0, 0.5])
    r = dd.run(m, dd.Config(k=2, max_ell=1, resum='screened', tau_grid=tau,
                            external_fields=[('x', 1), ('y', 1)]))
    assert r['resummation']['kappa_source'] == 'trivial (A_H = J0)'
    assert r['resummation']['counterterm'] is None
    assert np.array_equal(r['C_tau'], r['C_tau_loops'])


def test_default_is_the_loop_path(ou_quartic):
    tau = np.array([0.0, 0.5])
    a = _run(ou_quartic, 0.1, 1, None, tau)
    b = dd.run(ou_quartic[0], dd.Config(k=2, max_ell=1, tau_grid=tau,
                                        parameters={'mu': 1.0, 'eps': 0.1,
                                                    'D': 1.0}), ou_quartic[1])
    assert 'resummation' not in a
    for e in (0, 1):
        assert np.array_equal(a['C_tau_by_ell'][e], b['C_tau_by_ell'][e])


def test_options_and_refusals(ou_quartic):
    o = resolve_resum_options('screened')
    assert o['mass'] == 'pms' and o['n_scan'] == 17 and o['step'] == 5e-3
    for bad in ({'method': 'screened', 'mass': -1.0},
                {'method': 'screened', 'n_scan': 10},
                {'method': 'screened', 'kappa_range': (1.5, 3.0)},
                {'method': 'screened', 'step': 0.5},
                {'method': 'screened', 'colour': 1}):
        with pytest.raises(ValueError):
            resolve_resum_options(bad)
    with pytest.raises(ValueError, match='k = 2'):
        dd.Config(resum='screened', k=3)
    with pytest.raises(ValueError, match='max_ell >= 1'):
        dd.Config(resum='screened', max_ell=0)
    with pytest.raises(ValueError, match='derivatives'):
        dd.Config(resum='screened', max_ell=1,
                  covariance={'derivatives': ['mu']})
    with pytest.raises(ValueError, match='t_min'):
        dd.Config(resum='screened', max_ell=1, t_min=0.0)
    # max_ell inherited from the model file (0) is refused at run time
    with pytest.raises(ValueError, match='max_ell >= 1'):
        dd.run(ou_quartic[0], dd.Config(resum='screened'), ou_quartic[1])
    with pytest.raises(dd.HartreeUnsupported, match='kernels'):
        dd.run(dd.load_model('single_population_quad_exp_test')[0],
               dd.Config(k=2, max_ell=1, resum='screened'))


def test_numerics_helpers():
    # Fornberg weights reproduce derivatives of a polynomial exactly
    x = 1.3 + 0.01 * np.arange(-3, 4)
    W = fd_weights(1.3, x, 4)
    f = lambda t: 2 - t + 3 * t ** 2 - t ** 3 + 0.5 * t ** 4
    vals = f(x)
    exact = [f(1.3), -1 + 6 * 1.3 - 3 * 1.3 ** 2 + 2 * 1.3 ** 3,
             6 - 6 * 1.3 + 6 * 1.3 ** 2, -6 + 12 * 1.3, 12.0]
    for d in range(5):
        assert abs(W[d] @ vals - exact[d]) < 1e-6 * max(1, abs(exact[d]))
    # the log-Chebyshev interpolant: derivatives in s of s^-7 (1 + 50 s)^-3
    n = 17
    t = -np.cos(np.pi * np.arange(n) / (n - 1))
    lo, hi = math.log(0.25), math.log(6.0)
    s = np.exp(0.5 * (hi + lo) + 0.5 * (hi - lo) * t)
    F = lambda u: (1 + 50 * u) ** -3.0
    ip = _LogCheb(s, F(s), lo, hi)
    k = 1.7
    exact = [F(k), -150 * (1 + 50 * k) ** -4, 30000 * (1 + 50 * k) ** -5,
             -7.5e6 * (1 + 50 * k) ** -6]
    for d in range(4):
        assert abs(ip.deriv(np.array([k]), d)[0] / exact[d] - 1) < 1e-6

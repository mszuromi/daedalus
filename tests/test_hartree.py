"""
tests/test_hartree.py
=====================
The self-consistent Gaussian (Hartree) closure, ``api/hartree.py``
(``dd.hartree``, ``dd.Config(resum='hartree')``).

* OU quartic: the closure equals the closed form
  ``Σ = (-μ + sqrt(μ² + 12 ε D)) / (6 ε)`` from weak to strong coupling, its
  lag correlator is the single exponential ``Σ e^{-m_H |τ|}``, its weak
  coupling series is ``1 - 3g + 18g²`` (the exact Boltzmann series has
  ``+24g²``: the sunset is missing), and at strong coupling it stays within
  the known bound of the exact Boltzmann variance
  (ratio ``Γ(1/4) / (2 sqrt(3) Γ(3/4)) = 0.854`` as ``g -> inf``);
* multi-field: the O(N) model reproduces its ``(N + 2) / N`` closed form; the
  coupled two-field gradient model is within the Hartree error of its 2-D
  Boltzmann quadrature;
* a ``Dt`` pencil ``τ Dt x``: the noise is transformed with the pencil;
* colored noise (Markovian embedding): the linear limit is the exact
  stationary covariance of the embedded system;
* a jump model: the closure is the Gaussian moment closure of the master
  equation, checked against its closed equations and ``dd.master_equation``;
* double well: several stable solutions give the bistability advisory;
* spectra and the covariance layout agree with ``dd.spectra`` /
  ``dd.covariance_matrix`` on a linear model (where Hartree is exact);
* refusals: kernels, spatial models, derivative vertices, ``Config`` misuse.

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
from api.hartree import (hartree, HartreeUnsupported,  # noqa: E402
                         HartreeBistabilityWarning)
from api.model import TemporalModelBuilder  # noqa: E402


def _closed(mu, eps, D):
    return (-mu + math.sqrt(mu * mu + 12 * eps * D)) / (6 * eps)


def _boltzmann_var(mu, eps, D):
    from scipy.integrate import quad
    U = lambda x: (mu * x * x / 2 + eps * x ** 4 / 4) / D
    xm = 12 * math.sqrt(D / max(mu, math.sqrt(eps * D)))
    num = quad(lambda x: x * x * math.exp(-U(x)), -xm, xm, limit=400)[0]
    return num / quad(lambda x: math.exp(-U(x)), -xm, xm, limit=400)[0]


@pytest.fixture(scope='module')
def ou_quartic():
    return dd.load_model('ou_quartic')[0]


@pytest.mark.parametrize('eps', [1e-3, 0.1, 1.0, 100.0])
def test_ou_quartic_closed_form(ou_quartic, eps):
    mu, D = 1.3, 0.7
    r = hartree(ou_quartic, {'mu': mu, 'eps': eps, 'D': D})
    s2 = _closed(mu, eps, D)
    assert r.converged and r.advisories == []
    assert len(r.solutions) == 1
    assert abs(r.cov[0, 0] / s2 - 1) < 1e-12
    assert abs(r.mean[0]) < 1e-12
    mH = mu + 3 * eps * s2
    assert abs(-r.A[0, 0] / mH - 1) < 1e-12
    assert abs(r.Q[0, 0] - 2 * D) < 1e-12
    assert abs(r.J0[0, 0] + mu) < 1e-14 and abs(r.Q0[0, 0] - 2 * D) < 1e-14
    tau = np.array([-0.7, 0.0, 0.3, 2.0])
    assert np.allclose(r.C(tau), s2 * np.exp(-mH * np.abs(tau)),
                       rtol=1e-12, atol=0)
    assert np.allclose(r.C(tau, ('dx', 1), 'x'), r.C(tau), rtol=0, atol=0)
    # S(ω) = Q / (ω² + m²), and ∫ C dτ = S(0)
    w = np.array([0.0, 1.5])
    assert np.allclose(r.spectrum(w).real, 2 * D / (w ** 2 + mH ** 2),
                       rtol=1e-12)
    sp = r.spectra(omega=w, T=[0.5, 1e3])
    assert abs(sp['S0'] - 2 * D / mH ** 2) < 1e-12 * D
    assert abs(sp['tau_c'] - 1 / mH) < 1e-12 / mH
    assert abs(sp['integral'][0] - s2 * (1 - math.exp(-0.5 * mH)) / mH) \
        < 1e-12 * s2
    assert abs(sp['integral'][1] - sp['integral_inf']) < 1e-12 * s2


def test_weak_coupling_series(ou_quartic):
    """Σ = 1 - 3g + 18g² + O(g³) (μ = D = 1): the one-loop tadpole is
    exact, the two-loop coefficient is the double tadpole alone."""
    for g in (1e-3, 5e-4):
        s2 = hartree(ou_quartic, {'mu': 1.0, 'eps': g, 'D': 1.0}).cov[0, 0]
        c2 = (s2 - (1 - 3 * g)) / g ** 2
        assert abs(c2 - 18) < 300 * g      # next coefficient is -135
    ex = _boltzmann_var(1.0, 1e-3, 1.0)
    # exact two-loop coefficient 24, Hartree's 18: the difference is 6g²
    assert abs((ex - _s2(ou_quartic, 1e-3)) / 1e-6 - 6) < 0.5


def _s2(model, g):
    return hartree(model, {'mu': 1.0, 'eps': g, 'D': 1.0}).cov[0, 0]


def test_strong_coupling_bound(ou_quartic):
    lim = math.gamma(0.25) / (2 * math.sqrt(3) * math.gamma(0.75))
    ratios = [_s2(ou_quartic, g) / _boltzmann_var(1.0, g, 1.0)
              for g in (1.0, 10.0, 1e4)]
    assert all(lim - 1e-3 < q < 1.0 for q in ratios)
    assert ratios[0] > ratios[1] > ratios[2]
    assert abs(ratios[2] - lim) < 2e-3


def test_on_model_large_n():
    N = 3
    m = (TemporalModelBuilder('O(N) quartic').population('pop', size=N)
         .physical_field('x', population='pop')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('eps', default=2.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .set_action_text(
             'sum(xt[i]*((Dt+mu)*x[i] + (eps/%d)*x[i]*sum(x[j]^2 for j in '
             'pop)) - D*xt[i]^2 for i in pop)' % N)
         .equation(lhs='(Dt+mu)*x[i]',
                   rhs='-(eps/%d)*x[i]*sum(x[j]^2 for j in pop)' % N,
                   population='pop')
         .build())
    r = hartree(m, {})
    gg = 2.0 * (N + 2) / N
    s2 = (-1 + math.sqrt(1 + 4 * gg)) / (2 * gg)
    assert r.converged and len(r.fields) == N
    assert r.legs == [('dx', 1), ('dx', 2), ('dx', 3)]
    assert np.allclose(np.diag(r.cov), s2, rtol=1e-12)
    assert np.max(np.abs(r.cov - np.diag(np.diag(r.cov)))) < 1e-12


def _coupled_exact(P, L=8.0):
    """⟨x²⟩, ⟨y²⟩, ⟨xy⟩ of exp(-U/D) for the symmetric coupled model
    (J1 = J2: a gradient system)."""
    from scipy.integrate import dblquad
    U = lambda x, y: (P['mu1'] * x * x / 2 + P['eps1'] * x ** 4 / 4
                      + P['mu2'] * y * y / 2 + P['eps2'] * y ** 4 / 4
                      - P['J1'] * x * y) / P['D1']
    Z = dblquad(lambda y, x: math.exp(-U(x, y)), -L, L, -L, L)[0]
    return {nm: dblquad(lambda y, x: fn(x, y) * math.exp(-U(x, y)),
                        -L, L, -L, L)[0] / Z
            for nm, fn in (('xx', lambda x, y: x * x),
                           ('yy', lambda x, y: y * y),
                           ('xy', lambda x, y: x * y))}


COUPLED = {'mu1': 1.0, 'mu2': 1.0, 'eps1': 1.0, 'eps2': 0.5, 'J1': 0.5,
           'J2': 0.5, 'D1': 1.0, 'D2': 1.0}


def test_coupled_two_field():
    m = dd.load_model('ou_quartic_coupled')[0]
    r = hartree(m, COUPLED)
    assert r.converged and r.fields == ['dx1', 'dy1']
    L = r.A @ r.cov + r.cov @ r.A.T + r.Q
    assert np.max(np.abs(L)) < 1e-10
    assert np.max(np.abs(r.mean)) < 1e-10
    # C(τ) at τ=0 is Σ, and C_ij(τ) = C_ji(-τ)
    assert abs(r.C(0.0, 0, 1)[0] - r.cov[0, 1]) < 1e-14
    assert abs(r.C(0.4, 0, 1)[0] - r.C(-0.4, 1, 0)[0]) < 1e-14
    # the stronger nonlinearity confines x more; within the Hartree error
    # (a few percent at this coupling) of the 2-D Boltzmann integral
    ex = _coupled_exact(COUPLED)
    assert r.cov[0, 0] < r.cov[1, 1]
    for nm, (i, j) in (('xx', (0, 0)), ('yy', (1, 1)), ('xy', (0, 1))):
        rel = r.cov[i, j] / ex[nm] - 1
        assert -0.15 < rel < 0.0, (nm, rel)


def test_dt_pencil_noise_transformed():
    """``τ ẋ = -x + η``, ``⟨ηη⟩ = 2D δ``: Var x = D / τ.  The noise of the
    action lives on the ``M Dt x`` equation and is ``M^{-1} Q M^{-T}`` for
    ``x``."""
    m = (TemporalModelBuilder('pencil OU').physical_field('x')
         .parameter('tau', default=2.5, domain='positive')
         .parameter('gp', default=0.0, domain='real')
         .parameter('D', default=1.2, domain='positive')
         .set_action_text('xt*((tau*Dt+1)*x + gp*x^3) - D*xt^2')
         .equation(lhs='(tau*Dt+1)*x', rhs='-gp*x^3')
         .build())
    r = hartree(m, {})
    assert abs(r.cov[0, 0] / (1.2 / 2.5) - 1) < 1e-12
    assert abs(r.A[0, 0] + 1 / 2.5) < 1e-14
    assert abs(r.Q[0, 0] - 2 * 1.2 / 2.5 ** 2) < 1e-14
    # and it equals the tree level of the loop pipeline
    res = dd.run(m, dd.Config(k=2, max_ell=0, tau_grid=np.array([0.0, 0.7])))
    assert np.allclose(np.real(res['C_tau']), r.C([0.0, 0.7]), rtol=1e-12)


def test_colored_noise_linear_limit():
    """x' = -μx - εx³ + ξ, ξ an OU process of variance 2D/τc and time
    τc: at ε -> 0, Var x = σ_ξ² τc / (μ (1 + μ τc))."""
    m = dd.load_model('ou_quartic_colored')[0]
    mu, D, tc = 0.8, 1.0, 0.5
    r = hartree(m, {'mu': mu, 'eps': 1e-12, 'D': D, 'tauc': tc})
    assert r.fields == ['dx1', 'dxi1']
    sxi = 2 * D / tc
    assert abs(r.cov[1, 1] / sxi - 1) < 1e-9
    assert abs(r.cov[0, 0] / (sxi * tc / (mu * (1 + mu * tc))) - 1) < 1e-9
    # the nonlinearity lowers the variance of x, not that of the noise
    r2 = hartree(m, {'mu': mu, 'eps': 1.0, 'D': D, 'tauc': tc})
    assert r2.cov[0, 0] < r.cov[0, 0]
    assert abs(r2.cov[1, 1] / sxi - 1) < 1e-9


def test_double_well_bistability_advisory():
    m = dd.load_model('ou_quartic_double_well')[0]
    p = {'mu': -1.0, 'eps': 1.0, 'D': 0.1}
    with pytest.warns(HartreeBistabilityWarning, match='multistable'):
        sym = hartree(m, p, mean0=[0.0])
    assert abs(sym.mean[0]) < 1e-12
    assert abs(sym.cov[0, 0] / _closed(-1.0, 1.0, 0.1) - 1) < 1e-12
    # the symmetric Gaussian and the two broken-symmetry ones
    assert len(sym.solutions) == 3
    assert [a['code'] for a in sym.advisories] == ['HARTREE_MULTIPLE_SOLUTIONS']
    means = sorted(s['mean'][0] for s in sym.solutions)
    assert abs(means[1]) < 1e-12 and abs(means[0] + means[2]) < 1e-9
    with pytest.warns(HartreeBistabilityWarning):
        br = hartree(m, p, mean0=[1.0])
    assert 0.85 < br.mean[0] < 0.95
    # broken-symmetry closure: ⟨f⟩ = 0 and Σ = D / |A|
    mb, s2 = br.mean[0], br.cov[0, 0]
    assert abs(mb * (-1.0 + mb ** 2 + 3 * s2)) < 1e-10
    assert abs(s2 * (-br.A[0, 0]) - 0.1) < 1e-10
    # at D = 1 only the symmetric solution exists: no bistability, but the
    # closure started in the selected well (x* = -1) settles at the barrier
    # top, nearer the unstable root 0, and says so
    with warnings.catch_warnings():
        warnings.simplefilter('error', HartreeBistabilityWarning)
        with pytest.warns(dd.HartreeAdvisoryWarning, match='settled'):
            one = hartree(m, {'mu': -1.0, 'eps': 1.0, 'D': 1.0})
    assert len(one.solutions) == 1
    assert [a['code'] for a in one.advisories] == ['HARTREE_LEFT_SADDLE']
    assert abs(one.mean[0]) < 1e-10


def test_jump_model_gaussian_moment_closure():
    """Immigration, death and pairwise annihilation (rates λ, μx, c x²; the
    SDE front end's Poisson sources).  Hartree keeps the first two jump
    moments, so it is the Gaussian closure of the exact moment equations
    ``λ - μm - 2c(m² + Σ) = 0`` and ``2AΣ + Q = 0`` with ``A = -μ - 4cm``,
    ``Q = λ + μm + 4c(m² + Σ)``; against ``dd.master_equation`` it moves the
    mean from the saddle's 1 % error to 0.01 %, and its variance is within
    the closure's O(1/Ω) error."""
    from scipy.optimize import fsolve
    P = {'lam': 60.0, 'mu': 1.0, 'c': 0.02}
    me = dd.master_equation(
        transitions=[('lam', +1), ('mu*x', -1), ('c*x^2', -2)], fields=['x'],
        params=P, h=1.0, box=(0, 250))
    cm = me.cumulants(kmax=2)
    ex_m, ex_v = float(cm.mean[0]), float(cm.cumulants[2][0, 0])
    m = me.to_sde().build()
    h = hartree(m, dict(P, h=1.0))
    lam, mu, c = P['lam'], P['mu'], P['c']

    def eqs(v):
        mm, S = v
        A = -mu - 4 * c * mm
        Q = lam + mu * mm + 4 * c * (mm * mm + S)
        return [lam - mu * mm - 2 * c * (mm * mm + S), 2 * A * S + Q]
    mm, S = fsolve(eqs, [ex_m, ex_v], xtol=1e-14)
    assert abs(h.mean[0] - mm) < 1e-10 * mm
    assert abs(h.cov[0, 0] - S) < 1e-10 * S
    assert abs(h.mean[0] - ex_m) < 0.02 * abs(h.saddle[0] - ex_m)
    assert abs(h.cov[0, 0] / ex_v - 1) < 0.015
    # the lag correlator decays at the averaged rate
    tc = np.asarray(me.correlation(np.array([0.5])).C).ravel()[0]
    assert abs(h.C([0.5])[0] / tc - 1) < 0.03


def _linear_two_field():
    return (TemporalModelBuilder('linear two-field').physical_field('x')
            .physical_field('y')
            .parameter('a', default=1.0, domain='positive')
            .parameter('b', default=0.7, domain='real')
            .parameter('c', default=1.6, domain='positive')
            .set_action_text('xt*((Dt+a)*x - b*y) - xt^2 + yt*((Dt+c)*y) '
                             '- 0.5*yt^2')
            .equation(lhs='(Dt+a)*x', rhs='b*y')
            .equation(lhs='(Dt+c)*y', rhs='0')
            .build())


def test_linear_model_spectra_and_covariance_match_the_pipeline():
    """Hartree is exact for a linear model: its C(τ), spectra (cross pair,
    sign conventions included) and covariance layout equal the tree level
    of ``compute_cumulants`` / ``dd.spectra`` / ``dd.covariance_matrix``."""
    m = _linear_two_field()
    tau = np.array([-1.1, 0.0, 0.4, 2.0])
    om = np.array([0.0, 0.8, 3.0])
    T = [0.3, 5.0]
    ext = [('x', 1), ('y', 1)]
    r_h = dd.run(m, dd.Config(resum='hartree', external_fields=ext,
                              tau_grid=tau,
                              spectra={'omega': om, 'T': T}))
    r_t = dd.run(m, dd.Config(k=2, max_ell=0, external_fields=ext,
                              tau_grid=tau, spectra={'omega': om, 'T': T}))
    assert np.allclose(r_h['C_tau'], np.real(r_t['C_tau']), rtol=1e-11,
                       atol=1e-13)
    sh, st = r_h['spectra'], r_t['spectra']
    for key in ('S', 'integral', 'integral_neg'):
        assert np.allclose(sh[key], st[key], rtol=1e-10, atol=1e-13), key
    for key in ('S0', 'integral_inf', 'integral_inf_neg', 'C0', 'tau_c'):
        assert abs(sh[key] - st[key]) < 1e-10 * (1 + abs(st[key])), key
    c_h = dd.run(m, dd.Config(resum='hartree', covariance=True,
                              tau_grid=tau))
    c_t = dd.covariance_matrix(m, None, 0, tau_grid=tau, verbose=False)
    assert c_h['fields'] == [('x', 1), ('y', 1)] == c_t['fields']
    assert c_h['ring_fields'] == ['dx1', 'dy1']
    assert np.allclose(c_h['Sigma'], np.real(c_t['Sigma']), rtol=1e-11,
                       atol=1e-13)
    # k = 1: the closure keeps the mean at the saddle of a linear model
    r1 = dd.run(m, dd.Config(resum='hartree', k=1, external_fields=[ext[0]],
                             tau_grid=np.array([0.0])))
    assert abs(r1['C_tau'][0]) < 1e-14


def test_run_resum_hartree_matches_dd_hartree(ou_quartic):
    p = {'mu': 1.0, 'eps': 3.0, 'D': 1.0}
    tau = np.array([0.0, 0.25, 1.0])
    r = dd.run(ou_quartic, dd.Config(resum='hartree', parameters=p,
                                     tau_grid=tau))
    h = dd.hartree(ou_quartic, p)
    assert np.array_equal(r['C_tau'], h.C(tau))
    assert r['resummation']['method'] == 'hartree'
    assert r['_resolved']['resum'] == 'hartree'


def test_refuses_kernels():
    m = dd.load_model('single_population_quad_exp_test')[0]
    with pytest.raises(HartreeUnsupported, match='kernels'):
        hartree(m, {})


def test_refuses_spatial():
    m = dd.load_model('edwards_wilkinson_1d')[0]
    with pytest.raises(HartreeUnsupported, match='spatial'):
        hartree(m, {})
    with pytest.raises(HartreeUnsupported, match='spatial'):
        dd.run(m, dd.Config(resum='hartree'))


def test_refuses_derivative_vertices():
    m = (TemporalModelBuilder('derivative vertex').physical_field('x')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('av', default=0.2, domain='real')
         .parameter('D', default=1.0, domain='positive')
         .set_action_text('xt*((Dt+mu)*x + av*x*Dt*x) - D*xt^2')
         .equation(lhs='(Dt+mu)*x', rhs='0')
         .build())
    with pytest.raises(HartreeUnsupported, match='time derivatives'):
        hartree(m, {})


def test_config_refusals():
    with pytest.raises(ValueError, match='max_ell'):
        dd.Config(resum='hartree', max_ell=1)
    with pytest.raises(ValueError, match='k must be 1 or 2'):
        dd.Config(resum='hartree', k=3)
    with pytest.raises(ValueError, match='output'):
        dd.Config(resum='hartree', output='moment')
    with pytest.raises(ValueError, match='method'):
        dd.Config(resum='variational')
    with pytest.raises(ValueError, match='screened series'):
        dd.Config(resum={'method': 'hartree', 'mass': 'pms'})
    with pytest.raises(ValueError, match='derivatives'):
        dd.Config(resum='hartree', covariance={'derivatives': ['mu']})


def test_multiplicative_noise_exact_moment():
    """dx = -mu x dt + sqrt(2D(1 + a x²)) dW (Itô): the second-moment
    equation closes, Var x = D / (mu - a D), and the closure (noise vertex
    ``xt² x²`` averaged in the Gaussian) must reproduce it."""
    m = (TemporalModelBuilder('multiplicative OU').physical_field('x')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('a', default=0.3, domain='positive')
         .parameter('D', default=1.2, domain='positive')
         .set_action_text('xt*((Dt+mu)*x) - D*(1+a*x^2)*xt^2')
         .equation(lhs='(Dt+mu)*x', rhs='0')
         .build())
    r = hartree(m, {})
    assert r.converged
    assert abs(r.cov[0, 0] / (1.2 / (1.0 - 0.3 * 1.2)) - 1) < 1e-12
    assert abs(r.Q[0, 0] - 2 * 1.2 * (1 + 0.3 * r.cov[0, 0])) < 1e-12

"""
tests/test_hartree_prototype.py
===============================
The self-consistent Gaussian (Hartree) prototype, ``api/hartree.py``.

* OU quartic: the closure equals the closed form
  ``Σ = (-μ + sqrt(μ² + 12 ε D)) / (6 ε)`` from weak to strong coupling, its
  lag correlator is the single exponential ``Σ e^{-m_H |τ|}``, its weak
  coupling series is ``1 - 3g + 18g²`` (the exact Boltzmann series has
  ``+24g²``: the sunset is missing), and at strong coupling it stays within
  the known bound of the exact Boltzmann variance
  (ratio ``Γ(1/4) / (2 sqrt(3) Γ(3/4)) = 0.854`` as ``g -> inf``);
* multi-field: the O(N) model reproduces its ``(N + 2) / N`` closed form,
  and the coupled two-field model returns a solution of the Lyapunov
  equation with zero mean drift;
* colored noise (Markovian embedding): the linear limit is the exact
  stationary covariance of the embedded system;
* double well: a symmetric start gives the symmetric Gaussian, a start in
  one well the broken-symmetry one.

Inline public models or tracked ones only.
"""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import daedalus as dd  # noqa: E402
from api.hartree import hartree  # noqa: E402


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
    assert r.converged
    assert abs(r.cov[0, 0] / s2 - 1) < 1e-12
    assert abs(r.mean[0]) < 1e-12
    mH = mu + 3 * eps * s2
    assert abs(-r.A[0, 0] / mH - 1) < 1e-12
    assert abs(r.Q[0, 0] - 2 * D) < 1e-12
    tau = np.array([-0.7, 0.0, 0.3, 2.0])
    assert np.allclose(r.C(tau), s2 * np.exp(-mH * np.abs(tau)),
                       rtol=1e-12, atol=0)
    # S(ω) = Q / (ω² + m²), and ∫ C dτ = S(0)
    w = np.array([0.0, 1.5])
    assert np.allclose(r.spectrum(w).real, 2 * D / (w ** 2 + mH ** 2),
                       rtol=1e-12)


def test_weak_coupling_series(ou_quartic):
    """Σ = 1 - 3g + 18g² + O(g³) (μ = D = 1): the one-loop tadpole is
    exact, the two-loop coefficient is the double tadpole alone."""
    for g in (1e-3, 5e-4):
        s2 = hartree(ou_quartic, {'mu': 1.0, 'eps': g, 'D': 1.0}).cov[0, 0]
        c2 = (s2 - (1 - 3 * g)) / g ** 2
        assert abs(c2 - 18) < 300 * g      # next coefficient is -135
    ex = _boltzmann_var(1.0, 1e-3, 1.0)
    # exact two-loop coefficient 24, Hartree's 18: the difference is 6g²
    assert abs((ex - s2_at(ou_quartic, 1e-3)) / 1e-6 - 6) < 0.5


def s2_at(model, g):
    return hartree(model, {'mu': 1.0, 'eps': g, 'D': 1.0}).cov[0, 0]


def test_strong_coupling_bound(ou_quartic):
    lim = math.gamma(0.25) / (2 * math.sqrt(3) * math.gamma(0.75))
    ratios = [s2_at(ou_quartic, g) / _boltzmann_var(1.0, g, 1.0)
              for g in (1.0, 10.0, 1e4)]
    assert all(lim - 1e-3 < q < 1.0 for q in ratios)
    assert ratios[0] > ratios[1] > ratios[2]
    assert abs(ratios[2] - lim) < 2e-3


def test_on_model_large_n():
    from api.model import TemporalModelBuilder
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
    assert np.allclose(np.diag(r.cov), s2, rtol=1e-12)
    assert np.max(np.abs(r.cov - np.diag(np.diag(r.cov)))) < 1e-12


def test_coupled_two_field_solves_closure():
    m = dd.load_model('ou_quartic_coupled')[0]
    r = hartree(m, {'eps1': 0.5, 'eps2': 0.2})
    assert r.converged and r.fields == ['dx1', 'dy1']
    L = r.A @ r.cov + r.cov @ r.A.T + r.Q
    assert np.max(np.abs(L)) < 1e-10
    assert np.max(np.abs(r.mean)) < 1e-10
    # asymmetric couplings give an asymmetric covariance diagonal
    assert r.cov[0, 0] < r.cov[1, 1]
    # C(τ) at τ=0 is Σ, and C_ij(τ) = C_ji(-τ)
    assert abs(r.C(0.0, 0, 1)[0] - r.cov[0, 1]) < 1e-14
    assert abs(r.C(0.4, 0, 1)[0] - r.C(-0.4, 1, 0)[0]) < 1e-14


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


def test_double_well_branches():
    m = dd.load_model('ou_quartic_double_well')[0]
    p = {'mu': -1.0, 'eps': 1.0, 'D': 0.1}
    sym = hartree(m, p, mean0=[0.0])
    assert abs(sym.mean[0]) < 1e-12
    assert abs(sym.cov[0, 0] / _closed(-1.0, 1.0, 0.1) - 1) < 1e-12
    br = hartree(m, p, mean0=[1.0])
    assert 0.85 < br.mean[0] < 0.95
    # broken-symmetry closure: ⟨f⟩ = 0 and Σ = D / |A|
    mb, s2 = br.mean[0], br.cov[0, 0]
    assert abs(mb * (-1.0 + mb ** 2 + 3 * s2)) < 1e-10
    assert abs(s2 * (-br.A[0, 0]) - 0.1) < 1e-10


def test_refuses_kernels():
    m = dd.load_model('single_population_quad_exp_test')[0]
    with pytest.raises(NotImplementedError):
        hartree(m, {})


def test_multiplicative_noise_exact_moment():
    """dx = -mu x dt + sqrt(2D(1 + a x²)) dW (Itô): the second-moment
    equation closes, Var x = D / (mu - a D), and the closure (noise vertex
    ``xt² x²`` averaged in the Gaussian) must reproduce it."""
    from api.model import TemporalModelBuilder
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

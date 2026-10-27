"""Validation tables of the resummations (``dd.hartree``,
``dd.Config(resum='screened')``) for the white-noise OU quartic model,

    dx/dt = -mu x - eps x^3 + sqrt(2D) eta,       g = eps D / mu^2,

against the exact Boltzmann variance and the Fokker-Planck lag correlator.
The numbers are those of ``docs/resummation.md``.

* equal time: relative error of the loop series (3 loops), Hartree and the
  screened series at orders 1-3 (PMS mass), with the PMS kappa and the
  spread of the last two orders;
* lag: max_tau |C - C_exact| / C_exact(0) on [0, 6/m_H] and the relative
  error of tau_int = int_0^inf C / C(0), for Hartree, the 3-loop series and
  screened order 3.

Run:  sage -python docs/resummation_validation.py      (a few minutes)
"""
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))


def exact_variance(mu, eps, D):
    U = lambda x: (mu * x * x / 2 + eps * x ** 4 / 4) / D
    xm = 12 * math.sqrt(D / max(abs(mu), math.sqrt(eps * D)))
    num = quad(lambda x: x * x * math.exp(-U(x)), -xm, xm, limit=400)[0]
    den = quad(lambda x: math.exp(-U(x)), -xm, xm, limit=400)[0]
    return num / den


def fp_spectrum(mu, eps, D, n=3001):
    """Rates and weights of C(tau) = sum_n w_n exp(-lam_n tau) from a
    detailed-balance finite-volume Fokker-Planck operator."""
    from scipy.optimize import brentq
    L = brentq(lambda y: (mu * y * y / 2 + eps * y ** 4 / 4) / D - 60,
               1e-12, 1e6)
    x = np.linspace(-L, L, n)
    h = x[1] - x[0]
    U = (mu * x * x / 2 + eps * x ** 4 / 4) / D
    dU = np.diff(U)
    kp = D / h ** 2 * np.exp(-dU / 2)
    km = D / h ** 2 * np.exp(dU / 2)
    diag = np.zeros(n)
    diag[:-1] -= kp
    diag[1:] -= km
    lam, V = eigh_tridiagonal(-diag, -np.full(n - 1, D / h ** 2))
    pi = np.exp(-(U - U.min()))
    pi /= pi.sum()
    w = ((x * np.sqrt(pi)) @ V) ** 2
    keep = lam > 1e-9 * lam.max()
    return lam[keep], w[keep]


def cheb_grid(n, T):
    return 0.5 * T * (1 - np.cos(np.pi * np.arange(n) / (n - 1)))


def cheb_integral(y, T):
    from numpy.polynomial import chebyshev as Ch
    n = len(y)
    u = -np.cos(np.pi * np.arange(n) / (n - 1))
    ci = Ch.chebint(Ch.chebfit(u, y, n - 1))
    return 0.5 * T * (Ch.chebval(1, ci) - Ch.chebval(-1, ci))


def main():
    import daedalus as dd
    from daedalus.api.resum import ResumWarning
    warnings.simplefilter('ignore', ResumWarning)
    model, mod = dd.load_model('ou_quartic')
    mu = D = 1.0

    print('== equal-time <x^2>, relative error vs exact (mu = D = 1) ==')
    print('%-7s %-10s %-10s %-10s %-10s %-10s %-10s %-8s %-9s %s' % (
        'g', 'exact', 'L3', 'hartree', 'pms1', 'pms2', 'pms3', 'kappa3',
        'spread', 'wall'))
    for g in (0.01, 0.1, 1.0, 10.0, 1000.0):
        p = {'mu': mu, 'eps': g, 'D': D}
        ex = exact_variance(mu, g, D)
        h = dd.hartree(model, p)
        t0 = time.time()
        r = dd.run(model, dd.Config(k=2, max_ell=3, resum='screened',
                                    parameters=p, tau_grid=np.array([0.0])),
                   mod)
        wall = time.time() - t0
        info = r['resummation']
        # orders 1 and 2 at their own PMS masses (order 2: order 1's)
        lower = {}
        for N in (1, 2):
            rn = dd.run(model, dd.Config(k=2, max_ell=N, resum='screened',
                                         parameters=p,
                                         tau_grid=np.array([0.0])), mod)
            lower[N] = float(np.real(rn['C_tau'][0]))
        rel = lambda v: (v - ex) / ex
        print('%-7g %-10.5g %-+10.2e %-+10.2e %-+10.2e %-+10.2e %-+10.2e '
              '%-8.4f %-9.2e %.0fs' % (
                  g, ex, rel(float(np.real(r['C_tau_loops'][0]))),
                  rel(h.cov[0, 0]), rel(lower[1]), rel(lower[2]),
                  rel(float(np.real(r['C_tau'][0]))), info['kappa'],
                  info['spread_last_two'], wall))

    print('\n== lag correlator: max |C - C_exact| / C_exact(0) on [0, 6/m_H] '
          '(tau_int relative error) ==')
    for g in (0.1, 1.0, 3.0, 10.0):
        p = {'mu': mu, 'eps': g, 'D': D}
        h = dd.hartree(model, p)
        mH = -h.A[0, 0]
        T = 30.0 / mH
        tau = cheb_grid(65, T)
        lam, w = fp_spectrum(mu, g, D)
        Cex = (w[None, :] * np.exp(-np.outer(tau, lam))).sum(axis=1)
        tint_ex = float(np.sum(w / lam) / np.sum(w))
        r = dd.run(model, dd.Config(k=2, max_ell=3, resum='screened',
                                    parameters=p, tau_grid=tau), mod)
        win = tau <= 6.0 / mH
        row = []
        for name, c in (('hartree', h.C(tau)),
                        ('L3', np.real(r['C_tau_loops'])),
                        ('pms3', np.real(r['C_tau']))):
            err = float(np.max(np.abs(c - Cex)[win]) / Cex[0])
            tint = cheb_integral(c, T) / c[0]
            row.append('%s %.2e (%+.1f%%)' % (name, err,
                                              100 * (tint / tint_ex - 1)))
        print('g=%-5g kappa=%.4f  ' % (g, r['resummation']['kappa'])
              + '   '.join(row))


if __name__ == '__main__':
    main()

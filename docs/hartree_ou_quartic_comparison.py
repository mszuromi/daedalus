"""Self-consistent Gaussian (Hartree) resummation of the white-noise OU quartic
model, compared with the exact answer, the loop series and simulation.

    dx/dt = -mu x - eps x^3 + sqrt(2D) eta,       g = eps D / mu^2

Columns of the comparison (k = 2, the autocorrelation of x):

* exact: the Boltzmann density exp(-(mu x^2/2 + eps x^4/4)/D) for the
  equal-time variance (quadrature), and a detailed-balance discretisation of
  the Fokker-Planck operator for the lag correlator C(tau) and the
  correlation time tau_int = int_0^inf C / C(0).
* loops L=1..3: the plain loop series from compute_cumulants (max_ell=3).
* hartree: api.hartree (self-consistent Gaussian; one exponential mode).
* screened N=2,3: the loop series reorganised around the Hartree mass m_H:
  mu = m + lambda (mu - m), eps -> lambda eps, truncated at lambda^N
  (optimized / variational perturbation theory with the Hartree choice of m;
  N = 1 reproduces Hartree exactly).  The mass derivatives of the loop
  orders come from compute_cumulants runs at a 7-point stencil of masses.
* pms N=2,3 (equal time): the same series with m fixed by minimal
  sensitivity, dV_N/dm = 0.
* sim: Euler-Maruyama (numba, threads), at a few g only (--sim).

Further sections: the O(N) model (Hartree is exact as N -> inf) and the
symmetric double well (where a single Gaussian cannot work).

Run:  sage -python docs/hartree_ou_quartic_comparison.py [--sim] [--out DIR]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

C_LOOP = [1.0, -3.0, 24.0, -297.0]      # <x^2> = (D/mu) sum_l c_l g^l


def kappa4_coefficients(L=3):
    """kappa_4 = (D/mu)^2 sum_l d_l g^l from the 0-D Boltzmann integral
    (the series the k = 4 loop orders reproduce: d_1 = -6, d_2 = 126)."""
    import sympy as sp
    g = sp.Symbol('g')
    def mom(n):          # <x^n> under exp(-x^2/2 - g x^4/4), as a series
        num = sum((-g / 4) ** j / sp.factorial(j) * sp.factorial2(n + 4 * j - 1)
                  for j in range(L + 1))
        den = sum((-g / 4) ** j / sp.factorial(j) * sp.factorial2(4 * j - 1)
                  for j in range(L + 1))
        return sp.series(num / den, g, 0, L + 1).removeO()
    k4 = sp.expand(sp.series(mom(4) - 3 * mom(2) ** 2, g, 0, L + 1).removeO())
    c2 = sp.expand(mom(2))
    assert [float(c2.coeff(g, l)) for l in range(L + 1)] == C_LOOP[:L + 1]
    return [float(k4.coeff(g, l)) for l in range(L + 1)]


# ── exact references ──────────────────────────────────────────────────────

def exact_variance(mu, eps, D):
    U = lambda x: (mu * x * x / 2 + eps * x ** 4 / 4) / D
    xm = 12 * math.sqrt(D / max(abs(mu), math.sqrt(eps * D)))
    num = quad(lambda x: x * x * math.exp(-U(x)), -xm, xm, limit=400)[0]
    den = quad(lambda x: math.exp(-U(x)), -xm, xm, limit=400)[0]
    return num / den


def fp_spectrum(mu, eps, D, n=3001):
    """Eigen-decomposition of the Fokker-Planck operator (detailed-balance
    finite volume).  Returns rates lam_n >= 0 and weights w_n with
    C(tau) = sum_n w_n exp(-lam_n tau) (the lam=0 mode dropped)."""
    # domain: where the potential has risen 60 D above its minimum
    Umin = -mu * mu / (4 * eps) if mu < 0 else 0.0
    from scipy.optimize import brentq
    L = brentq(lambda y: (mu * y * y / 2 + eps * y ** 4 / 4 - Umin) / D - 60,
               1e-12, 1e6)
    x = np.linspace(-L, L, n)
    h = x[1] - x[0]
    U = (mu * x * x / 2 + eps * x ** 4 / 4) / D
    dU = np.diff(U)
    kp = D / h ** 2 * np.exp(-dU / 2)       # i -> i+1
    km = D / h ** 2 * np.exp(dU / 2)        # i+1 -> i
    diag = np.zeros(n)
    diag[:-1] -= kp
    diag[1:] -= km
    off = np.full(n - 1, D / h ** 2)        # sqrt(kp * km)
    lam, V = eigh_tridiagonal(-diag, -off)
    pi = np.exp(-(U - U.min()))
    pi /= pi.sum()
    amp = (x * np.sqrt(pi)) @ V
    w = amp ** 2
    keep = lam > 1e-9 * lam.max()
    return lam[keep], w[keep]


def fp_C(lam, w, tau):
    tau = np.atleast_1d(tau)
    return (w[None, :] * np.exp(-np.outer(tau, lam))).sum(axis=1)


# ── Hartree ───────────────────────────────────────────────────────────────

def hartree_closed(mu, eps, D):
    s2 = (-mu + math.sqrt(mu * mu + 12 * eps * D)) / (6 * eps)
    return s2, D / s2                       # variance, mass m_H


# ── equal-time series (exact coefficients) ───────────────────────────────

def loop_variance(mu, eps, D, L):
    g = eps * D / mu ** 2
    return D / mu * sum(C_LOOP[l] * g ** l for l in range(L + 1))


def screened_variance(mu, eps, D, m, N, coeffs=None, power=1):
    """sum_{l+n<=N} (mu-m)^n/n! d^n/dm^n [c_l D^{l+P} eps^l m^{-(2l+P)}]
    (P = 1: the variance; P = 2 with the kappa_4 coefficients: kappa_4)."""
    C = C_LOOP if coeffs is None else coeffs
    tot = 0.0
    for l in range(N + 1):
        p = 2 * l + power
        for k in range(N - l + 1):
            # d^k/dm^k m^{-p} = (-1)^k p(p+1)...(p+k-1) m^{-p-k}
            fall = 1.0
            for j in range(k):
                fall *= -(p + j)
            tot += (C[l] * D ** (l + power) * eps ** l * fall
                    * m ** (-p - k) * (mu - m) ** k / math.factorial(k))
    return tot


def pms_variance(mu, eps, D, N, m_guess):
    """Minimal-sensitivity mass: the stationary point of V_N(m) nearest to
    the Hartree mass (falls back to the Hartree mass when none exists)."""
    ms = np.geomspace(m_guess / 20, m_guess * 20, 4001)
    V = np.array([screened_variance(mu, eps, D, m, N) for m in ms])
    dV = np.gradient(V, ms)
    idx = np.where(np.sign(dV[:-1]) != np.sign(dV[1:]))[0]
    if len(idx) == 0:
        return float('nan'), float('nan')
    best = idx[np.argmin(np.abs(np.log(ms[idx] / m_guess)))]
    from scipy.optimize import brentq
    f = lambda m: (screened_variance(mu, eps, D, m * (1 + 1e-6), N)
                   - screened_variance(mu, eps, D, m * (1 - 1e-6), N))
    try:
        m = brentq(f, ms[best], ms[best + 1])
    except ValueError:
        m = ms[best]
    return screened_variance(mu, eps, D, m, N), m


# ── lag correlator from compute_cumulants ────────────────────────────────

_MODEL = None


def daedalus_by_ell(mu, eps, D, tau, max_ell=3):
    global _MODEL
    import daedalus as dd
    if _MODEL is None:
        _MODEL = dd.load_model('ou_quartic')
    m, mod = _MODEL
    r = dd.run(m, dd.Config(k=2, max_ell=max_ell, tau_grid=np.asarray(tau),
                            fundamental={'mu': mu, 'eps': eps, 'D': D},
                            ), mod)
    return {e: np.real(np.asarray(v)) for e, v in r['C_tau_by_ell'].items()}


_FD = {  # central stencils on j = -3..3 for derivatives 1..3 (O(h^4)+)
    1: np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0,
    2: np.array([2, -27, 270, -490, 270, -27, 2]) / 180.0,
    3: np.array([1, -8, 13, 0, -13, 8, -1]) / 8.0,
}


def screened_lag(mu, eps, D, m, tau, N_max=3, rel_h=0.005):
    """Screened series C_N(tau) for N = 0..N_max, plus the plain loop
    partial sums at the physical mass (same tau grid)."""
    h = rel_h * m
    stencil = [daedalus_by_ell(m + j * h, eps, D, tau, N_max)
               for j in range(-3, 4)]
    deriv = {}
    for l in range(N_max + 1):
        vals = np.array([s[l] for s in stencil])
        deriv[(l, 0)] = vals[3]
        for k in range(1, N_max - l + 1):
            deriv[(l, k)] = _FD[k] @ vals / h ** k
    out = {}
    for N in range(N_max + 1):
        tot = np.zeros(len(tau))
        for l in range(N + 1):
            for k in range(N - l + 1):
                tot += deriv[(l, k)] * (mu - m) ** k / math.factorial(k)
        out[N] = tot
    plain = daedalus_by_ell(mu, eps, D, tau, N_max)
    loops = {L: sum(plain[l] for l in range(L + 1))
             for L in range(N_max + 1)}
    return out, loops


def cheb_grid(n, T):
    return 0.5 * T * (1 - np.cos(np.pi * np.arange(n) / (n - 1)))


def cheb_integral(y, T):
    """int_0^T of the Chebyshev interpolant through cheb_grid nodes."""
    from numpy.polynomial import chebyshev as Ch
    n = len(y)
    u = -np.cos(np.pi * np.arange(n) / (n - 1))       # nodes on [-1, 1]
    c = Ch.chebfit(u, y, n - 1)
    ci = Ch.chebint(c)
    return 0.5 * T * (Ch.chebval(1, ci) - Ch.chebval(-1, ci))


# ── simulation ────────────────────────────────────────────────────────────

def simulate(mu, eps, D, dt, n_steps, sub, n_chains=8, seed=1):
    import numba

    @numba.njit(parallel=True, cache=False)
    def run(mu, eps, D, dt, n_steps, sub, n_chains, seed):
        n_out = n_steps // sub
        out = np.empty((n_chains, n_out))
        amp = math.sqrt(2 * D * dt)
        for c in numba.prange(n_chains):
            np.random.seed(seed + 7919 * c)
            x = 0.0
            for _ in range(20000):
                x += -(mu * x + eps * x ** 3) * dt + amp * np.random.randn()
            k = 0
            for i in range(n_out * sub):
                x += -(mu * x + eps * x ** 3) * dt + amp * np.random.randn()
                if (i + 1) % sub == 0:
                    out[c, k] = x
                    k += 1
        return out
    return run(mu, eps, D, dt, n_steps, sub, n_chains, seed)


def sim_stats(X, dt_out, lags):
    var = float(np.mean(X * X) - np.mean(X) ** 2)
    per_chain = (X * X).mean(axis=1)
    err = float(per_chain.std(ddof=1) / math.sqrt(X.shape[0]))
    C = []
    for L in lags:
        k = int(round(L / dt_out))
        C.append(float(np.mean(X[:, :X.shape[1] - k] * X[:, k:])))
    return var, err, np.array(C)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = {}
    mu, D = 1.0, 1.0

    # 1. equal-time variance across g
    gs = [1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1e3]
    rows = []
    print('\n== equal-time variance <x^2>  (mu = D = 1, eps = g); '
          'relative error vs exact ==')
    hdr = ('g', 'exact', 'L1', 'L2', 'L3', 'hartree', 'scr2', 'scr3',
           'pms2', 'pms3')
    print(('%-7s' + '%-11s' + '%-10s' * 8) % hdr)
    for g in gs:
        eps = g
        ex = exact_variance(mu, eps, D)
        s2, mH = hartree_closed(mu, eps, D)
        vals = {
            'L1': loop_variance(mu, eps, D, 1),
            'L2': loop_variance(mu, eps, D, 2),
            'L3': loop_variance(mu, eps, D, 3),
            'hartree': s2,
            'scr2': screened_variance(mu, eps, D, mH, 2),
            'scr3': screened_variance(mu, eps, D, mH, 3),
            'pms2': pms_variance(mu, eps, D, 2, mH)[0],
            'pms3': pms_variance(mu, eps, D, 3, mH)[0],
        }
        assert abs(screened_variance(mu, eps, D, mH, 1) - s2) < 1e-12 * s2
        rel = {k: (v - ex) / ex for k, v in vals.items()}
        rows.append({'g': g, 'exact': ex, **vals})
        print(('%-7g' + '%-11.5g' + '%-+10.2e' * 8) % (
            g, ex, *[rel[k] for k in hdr[2:]]))
    out['equal_time'] = rows

    # 2. Hartree from the library adapter equals the closed form
    import daedalus as dd
    from api.hartree import hartree
    m_q, _ = dd.load_model('ou_quartic')
    chk = []
    for g in (0.01, 1.0, 100.0):
        r = hartree(m_q, {'mu': mu, 'eps': g, 'D': D})
        chk.append(abs(r.cov[0, 0] / hartree_closed(mu, g, D)[0] - 1))
    print('\napi.hartree vs closed form: max rel diff %.1e' % max(chk))
    out['adapter_check'] = max(chk)

    # 3. lag correlator and correlation time
    print('\n== lag correlator: max_tau |C - C_exact| / C_exact(0) on '
          '[0, 6/m_H]; tau_int relative error ==')
    lag_rows = []
    for g in (0.1, 0.3, 1.0, 3.0, 10.0):
        eps = g
        s2, mH = hartree_closed(mu, eps, D)
        lam, w = fp_spectrum(mu, eps, D)
        T = 30.0 / mH
        tau = cheb_grid(65, T)
        Cex = fp_C(lam, w, tau)
        tint_ex = float(np.sum(w / lam) / np.sum(w))
        scr, loops = screened_lag(mu, eps, D, mH, tau)
        Ch = s2 * np.exp(-mH * tau)
        win = tau <= 6.0 / mH
        m_pms = pms_variance(mu, eps, D, 3, mH)[1]
        pms, _ = screened_lag(mu, eps, D, m_pms, tau)
        # the stencil derivatives reproduce the exact equal-time series
        assert abs(pms[3][0] - screened_variance(mu, eps, D, m_pms, 3)
                   ) < 1e-7 * s2
        assert abs(scr[3][0] - screened_variance(mu, eps, D, mH, 3)
                   ) < 1e-7 * s2
        cands = {'hartree': Ch, 'L1': loops[1], 'L2': loops[2],
                 'L3': loops[3], 'scr2': scr[2], 'scr3': scr[3],
                 'pms3': pms[3]}
        row = {'g': g, 'tau_int_exact': tint_ex, 'm_pms3': m_pms,
               'slowest_rate_exact': float(lam.min()), 'm_H': mH,
               'C0_exact_grid': float(Cex[0])}
        assert np.max(np.abs(scr[1] - Ch)) < 1e-6 * s2   # N=1 is Hartree
        for k, c in cands.items():
            row['maxerr_' + k] = float(np.max(np.abs(c - Cex)[win]) / Cex[0])
            row['tint_' + k] = float(cheb_integral(c, T) / c[0])
        lag_rows.append(row)
        print('g=%-5g m_H=%.4f exact slowest rate=%.4f tau_int=%.4f' % (
            g, mH, lam.min(), tint_ex))
        for k in cands:
            print('   %-8s maxerr %.2e   tau_int rel err %+.2e' % (
                k, row['maxerr_' + k], row['tint_' + k] / tint_ex - 1))
        if a.out and g in (1.0,):
            out['curve_g1'] = {'tau': tau.tolist(), 'exact': Cex.tolist(),
                               **{k: v.tolist() for k, v in cands.items()}}
    out['lag'] = lag_rows

    # 3b. kappa_4: the non-Gaussian part Hartree misses
    d4 = kappa4_coefficients(3)
    print('\n== kappa_4 (d_l = %s); excess kurtosis kappa_4/<x^2>^2 ==' % d4)
    print('%-7s%-10s%-10s%-10s%-10s%-10s%-10s' % (
        'g', 'exact', 'hartree', 'L1', 'L3', 'pms2', 'pms3'))
    k4_rows = []
    for g in (0.01, 0.1, 1.0, 10.0, 100.0):
        eps = g
        U = lambda x: (mu * x * x / 2 + eps * x ** 4 / 4) / D
        xm = 12 * math.sqrt(D / max(mu, math.sqrt(eps * D)))
        Z = quad(lambda x: math.exp(-U(x)), -xm, xm, limit=400)[0]
        m2 = quad(lambda x: x ** 2 * math.exp(-U(x)), -xm, xm, limit=400)[0] / Z
        m4 = quad(lambda x: x ** 4 * math.exp(-U(x)), -xm, xm, limit=400)[0] / Z
        k4 = m4 - 3 * m2 * m2
        s2, mH = hartree_closed(mu, eps, D)
        m_p = pms_variance(mu, eps, D, 3, mH)[1]
        gg = eps * D / mu ** 2
        L1 = (D / mu) ** 2 * d4[1] * gg
        L3 = (D / mu) ** 2 * sum(d4[l] * gg ** l for l in range(4))
        p2 = screened_variance(mu, eps, D, m_p, 2, d4, 2)
        p3 = screened_variance(mu, eps, D, m_p, 3, d4, 2)
        v3 = screened_variance(mu, eps, D, m_p, 3)
        k4_rows.append({'g': g, 'exact': k4 / m2 ** 2, 'L1': L1, 'L3': L3,
                        'pms2': p2 / v3 ** 2, 'pms3': p3 / v3 ** 2})
        print('%-7g%-+10.4f%-+10.4f%-+10.3g%-+10.3g%-+10.4f%-+10.4f' % (
            g, k4 / m2 ** 2, 0.0, L1 / m2 ** 2, L3 / m2 ** 2,
            p2 / v3 ** 2, p3 / v3 ** 2))
    out['kappa4'] = k4_rows

    # 4. O(N): Hartree is exact as N -> infinity
    print('\n== O(N) model, dx_i = -mu x_i - (eps/N) x_i |x|^2 + noise: '
          '<x_1^2> relative error of Hartree ==')
    on_rows = []
    for g in (1.0, 10.0):
        line = []
        for N in (1, 2, 4, 8, 16, 64, 256):
            U = lambda r: (mu * r * r / 2 + g * r ** 4 / (4 * N)) / D
            rm = 12 * math.sqrt(N * D)
            rr = np.linspace(1e-9, rm, 20001)
            lw = lambda r, p: p * math.log(r) - U(r)
            shift = max(lw(r, N + 1) for r in rr)
            num = quad(lambda r: math.exp(lw(r, N + 1) - shift), 0, rm,
                       limit=400, points=[rr[np.argmax([lw(r, N + 1) for r in rr])]])[0]
            den = quad(lambda r: math.exp(lw(r, N - 1) - shift), 0, rm,
                       limit=400, points=[rr[np.argmax([lw(r, N - 1) for r in rr])]])[0]
            ex = num / den / N
            gg = g * (N + 2) / N
            s2 = (-mu + math.sqrt(mu * mu + 4 * gg * D)) / (2 * gg)
            line.append((N, (s2 - ex) / ex))
            on_rows.append({'g': g, 'N': N, 'exact': ex, 'hartree': s2})
        print('g=%-4g ' % g + '  '.join('N=%d:%+.1e' % t for t in line))
    out['O_N'] = on_rows

    # 5. symmetric double well: where one Gaussian cannot work
    print('\n== double well mu=-1, eps=1 ==')
    dw = []
    for Dd in (0.1, 0.3, 1.0):
        lam, w = fp_spectrum(-1.0, 1.0, Dd)
        ex = float(np.sum(w))
        s2s, mS = hartree_closed(-1.0, 1.0, Dd)
        r_b = hartree(dd.load_model('ou_quartic_double_well')[0],
                      {'mu': -1.0, 'eps': 1.0, 'D': Dd}, mean0=[1.0])
        x2_b = float(r_b.mean[0] ** 2 + r_b.cov[0, 0])
        dw.append({'D': Dd, 'exact_x2': ex, 'exact_slowest_rate':
                   float(lam.min()), 'sym_hartree_x2': s2s,
                   'sym_hartree_rate': mS, 'broken_mean': float(r_b.mean[0]),
                   'broken_x2': x2_b, 'broken_rate': float(-r_b.A[0, 0])})
        print('D=%-4g exact <x^2>=%.4f slowest rate=%.3e | symmetric '
              'Hartree <x^2>=%.4f rate=%.3f | broken Hartree mean=%.3f '
              '<x^2>=%.4f rate=%.3f' % (Dd, ex, lam.min(), s2s, mS,
                                        r_b.mean[0], x2_b, -r_b.A[0, 0]))
    out['double_well'] = dw

    # 6. simulation spot checks
    if a.sim:
        print('\n== Euler-Maruyama (16 chains) ==')
        sims = []
        for g in (0.1, 1.0, 10.0):
            s2, mH = hartree_closed(mu, g, D)
            stiff = mu + 3 * g * 9 * s2
            dt = 2e-3 / stiff
            sub = max(1, int(round(0.02 / mH / dt)))
            t0 = time.time()
            X = simulate(mu, g, D, dt, int(4e7), sub, n_chains=16)
            lags = [0.0, 0.5 / mH, 1.0 / mH, 2.0 / mH]
            var, err, C = sim_stats(X, dt * sub, lags)
            ex = exact_variance(mu, g, D)
            lam, w = fp_spectrum(mu, g, D)
            Cex = fp_C(lam, w, np.array(lags))
            sims.append({'g': g, 'var': var, 'err': err, 'exact': ex,
                         'C': C.tolist(), 'C_exact': Cex.tolist(),
                         'lags': lags, 'dt': dt, 'wall': time.time() - t0})
            print('g=%-5g sim var=%.4f +- %.4f exact %.4f (dt=%.1e) | '
                  'C(lags)/C_exact = %s' % (
                      g, var, err, ex, dt,
                      np.array2string(C / Cex, precision=3)))
        out['sim'] = sims

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, 'hartree_ou_quartic.json'), 'w') as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == '__main__':
    main()

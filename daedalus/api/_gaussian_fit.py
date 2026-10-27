r"""
Rational fit of a time-shifted Gaussian density (the Gaussian jitter of an
exogenous input)
==========================================================================

A Gaussian density ``g(t) = exp(-t^2/(2 sigma^2))/sqrt(2 pi sigma^2)`` is not
causal, and its image ``exp(-sigma^2 omega^2/2)`` is not rational, so neither
the pole stage nor the state-space realization can take it.  On an EXOGENOUS
stationary input (a source no field feeds back into) every stationary
statistic of the network is unchanged when the whole input is shifted in time
by ``T``: the k-th cumulant of the input carries ``delta(omega_1 + ... +
omega_k)``, and ``prod_i exp(-i omega_i T) = 1`` on that shell.  So ``g(t)``
may be replaced by ``g(t - T)``, which is causal up to the mass
``Phi(-T/sigma)`` it has at ``t < 0``, and the delay ``T`` is dropped (it costs
nothing: it is never realized).

``gaussian_fit(tol)`` returns a strictly proper rational ``R(z)``,
``z = i omega sigma`` (the fit is universal in units of ``sigma``), whose
impulse response ``r`` is close to ``g(t - T)`` (``sigma = 1``):

1. ``T``: the smallest multiple of ``1/4`` (at least 3) with
   ``Phi(-T) <= tol / 10``.
2. The truncated density ``g(t - T) theta(t)`` is expanded in the Laguerre
   functions ``l_k(t) = sqrt(2 lam) exp(-lam t) L_k(2 lam t)`` (``lam = 2``,
   orthonormal on ``t >= 0``, image ``sqrt(2 lam)/(z + lam) ((z - lam)/(z +
   lam))^k``) to at most ``M = 480`` terms (the last ones below ``1e-12`` of
   the largest, checked; the negligible tail is dropped).  Its state-space
   realization (the Laguerre network) is input-normal (controllability
   Gramian = identity), well conditioned at any order.
3. Balanced truncation (square-root method on the Hammarling factors of
   ``api._realization.hankel_singular_values``) gives the poles: the order
   ``r`` starts at the fewest states whose H-infinity bound ``2 sum(discarded
   Hankel singular values)`` meets ``tol`` and grows until the measured
   ``L1`` error of the final fit (below) is at most ``tol``.
4. The poles, as first / second order sections (real poles, conjugate
   pairs), are rounded to double; the numerators are the least-squares fit
   of the image ``exp(-omega^2/2 - i omega T)`` with those poles (Parseval:
   the L2 error over the whole line), constrained to unit area.  More
   accurate than the residues of the truncated system, whose eigenvectors
   lose ~1e-10 at 22 states.
5. The kernel string (``image``) is ``N(z)/D(z)`` multiplied out EXACTLY from
   the rounded sections (dyadic rationals), scaled to unit area exactly:
   the mean field is exact.  The pipeline's realization factors ``D``
   exactly into the same sections and realises them as a cascade (no
   polynomial roots).

Errors (measured on the rounded sections; ``sigma = 1``):

``l1``        ``int |r(t) - g(t - T)| dt`` over the whole line (the mass
              ``Phi(-T)`` at ``t < 0`` included).  It bounds the image error
              ``sup_omega |R(i omega) - exp(-omega^2/2 - i omega T)| <= l1``.
``sup``       that image error, sampled on a frequency grid.
``spectral``  ``sup_omega ||R(i omega)|^2 - exp(-omega^2)|``, sampled: second
              order statistics see ``g`` only through ``|g~|^2``.

Per cumulant order ``k`` (``bound(k)``): a contribution with ``m`` jittered
input legs (a tree k-point cumulant of a linear network has ``m = k``) moves
by at most ``(1 + l1)^m - 1`` relative to the same contribution with every
kernel image replaced by its modulus (for ``k = 2`` and a linear network,
``spectral`` relative to the variance driven by the unjittered input); the
mean (``k = 1``) is exact.  The bounds are relative to that unjittered scale:
when sigma is small against the network's time constants the jitter's own
effect on a statistic is small too, and a loose fit reproduces that effect only
roughly (measured: ~13 % of it at ``tol = 1e-2``, 5e-3 at ``1e-3``, 1.5e-5 at
``1e-6``); take ``tol <= 1e-4`` when the jitter correction itself is wanted.

Cost (``tol``: states, shift ``T/sigma``, ``l1``): 1e-2: 6, 3.25, 9e-3;
1e-4: 12, 4.5, 3e-5; 1e-6: 16, 5.25, 7e-7; 1e-8: 22, 6.0, 8e-9; 1e-9: 27,
6.5, 6e-10.  About 0.8 s per ``tol`` (cached per process).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ['GaussianFit', 'gaussian_fit', 'TOL_RANGE']

#: Honoured fit tolerances (the L1 error of the fit): near 1e-9 the residues
#: of the sections pass 1e4 and the least-squares numerators stop improving
#: (the L1 error levels off at ~3e-10 from 30 states on).
TOL_RANGE = (1e-9, 1e-2)

_LAM = 2.0                   # Laguerre scale (units 1/sigma)
_M = 480                     # Laguerre terms
_NQ = 4000                   # quadrature nodes for the Laguerre coefficients
_R_MAX = 40                  # largest fit order


@dataclass(frozen=True)
class GaussianFit:
    """The fit for ``tol`` in units ``sigma = 1`` (see the module docstring)."""
    tol: float
    shift: float                       # T / sigma
    order: int                         # states (degree of the denominator)
    poles: tuple                       # complex, one per state
    residues: tuple
    sections: tuple                    # ((n1, n0, d1, d0) | (n0, d0)), real
    l1: float
    sup: float
    spectral: float
    truncated_mass: float
    hsv: tuple = field(repr=False, default=())

    def bound(self, k: int) -> float:
        """Relative error bound of a k-th order statistic (module docstring)."""
        k = int(k)
        if k <= 1:
            return 0.0
        if k == 2:
            return float(self.spectral)
        return float((1.0 + self.l1) ** k - 1.0)

    def image(self, z: str) -> str:
        """The rational image as text in the variable ``z`` (an expression):
        ``N(z)/D(z)``, both multiplied out EXACTLY from the sections (dyadic
        rationals), ``D`` monic and ``N(0) = D(0)`` (unit area, exactly).
        The symbolic stages handle one expanded fraction with exact
        coefficients much faster than a sum of sections, a product of
        factors or float coefficients, and the realization's exact
        factorization recovers the sections' factors (small Schwarz blocks:
        a non-dyadic ``sigma`` would otherwise give one block of a degree-16
        polynomial whose coefficient heights grow like ``53 k`` bits)."""
        z = f'({z})'
        N, D = self.fraction()

        def poly(cs):
            terms = []
            for k, c in enumerate(cs):
                if c == 0:
                    continue
                q = str(c.numerator) if c.denominator == 1 else f'{c.numerator}/{c.denominator}'
                zk = '' if k == 0 else (f'*{z}' if k == 1 else f'*{z}^{k}')
                terms.append(zk[1:] if c == 1 and k else f'({q}){zk}')
            return ' + '.join(terms)
        return f'({poly(N)})/({poly(D)})'

    def fraction(self):
        """``(N, D)``: ascending ``Fraction`` coefficients of the image."""
        return _fraction(self.sections)

    def transfer(self, z):
        """``R(z)`` (numpy, z complex), from the sections."""
        return _sections_transfer(self.sections, np.asarray(z, dtype=complex))

    def impulse(self, t):
        """``r(t)``, ``t >= 0`` (numpy)."""
        t = np.asarray(t, dtype=float)
        p, rho = np.array(self.poles), np.array(self.residues)
        return (np.exp(np.multiply.outer(t, p)) @ rho).real

    def record(self) -> dict:
        return {'tol': self.tol, 'shift_in_sigma': self.shift, 'order': self.order,
                'l1': self.l1, 'sup': self.sup, 'spectral': self.spectral,
                'truncated_mass': self.truncated_mass,
                'bound_by_order': {k: self.bound(k) for k in range(1, 7)}}


def _phi(t, T):
    return np.exp(-0.5 * (t - T) ** 2) / math.sqrt(2.0 * math.pi)


def _laguerre_coefficients(T):
    from scipy.special import roots_legendre
    L = T + 14.0
    x, w = roots_legendre(_NQ)
    t = 0.5 * (x + 1.0) * L
    w = 0.5 * w * L
    g = w * _phi(t, T)
    xx = 2.0 * _LAM * t
    lkm = np.exp(-0.5 * xx)
    lk = (1.0 - xx) * lkm
    c = np.empty(_M)
    c[0], c[1] = g @ lkm, g @ lk
    for k in range(1, _M - 1):
        lkm, lk = lk, ((2 * k + 1 - xx) * lk - k * lkm) / (k + 1)
        c[k + 1] = g @ lk
    c *= math.sqrt(2.0 * _LAM)
    if not np.max(np.abs(c[-40:])) <= 1e-12 * np.max(np.abs(c)):
        raise ArithmeticError('gaussian_fit: Laguerre expansion not converged')
    # drop the negligible tail (the Gramians cost M^3)
    keep = int(np.nonzero(np.abs(c) > 1e-15 * np.max(np.abs(c)))[0][-1]) + 17
    return c[:min(keep, _M)]


def _laguerre_network(c):
    M = c.size
    a = -_LAM * np.eye(M) - 2.0 * _LAM * np.tril(np.ones((M, M)), -1)
    b = math.sqrt(2.0 * _LAM) * np.ones(M)
    return a, b, c.copy()


def _pole_sections(a):
    """The poles of ``a`` as real sections: ``(d0,)`` for ``z + d0`` and
    ``(d1, d0)`` for ``z^2 + d1 z + d0`` (one per conjugate pair)."""
    lam = np.linalg.eigvals(a)
    lam = lam[np.argsort(np.abs(lam.imag) + 1e-9 * lam.real)]
    dens, used = [], np.zeros(lam.size, dtype=bool)
    for i, p in enumerate(lam):
        if used[i]:
            continue
        used[i] = True
        if abs(p.imag) <= 1e-12 * abs(p):
            dens.append((float(-p.real),))
            continue
        j = min((q for q in range(lam.size) if not used[q]),
                key=lambda q: abs(lam[q] - np.conj(p)))
        used[j] = True
        p = complex(p.real, abs(0.5 * (p.imag - lam[j].imag)))
        dens.append((float(-2.0 * p.real), float(abs(p) ** 2)))
    return dens


def _numerators(dens, T):
    """Least-squares numerators of the sections over the image error
    ``int |R(i w) - exp(-w^2/2 - i w T)|^2 dw`` (Parseval: the L2 error in
    time, over the whole line), subject to ``R(0) = 1`` (exact unit area)."""
    from scipy.special import roots_legendre
    x, w = roots_legendre(24)
    edges = np.linspace(0.0, 16.0, 161)
    h = np.diff(edges)
    om = (edges[:-1, None] + 0.5 * h[:, None] * (x[None, :] + 1.0)).ravel()
    ww = np.sqrt((0.5 * h[:, None] * w[None, :]).ravel())
    z = 1j * om
    cols, dc = [], []
    for d in dens:
        if len(d) == 1:
            cols.append(1.0 / (z + d[0]))
            dc.append(1.0 / d[0])
        else:
            q = z * z + d[0] * z + d[1]
            cols += [z / q, 1.0 / q]
            dc += [0.0, 1.0 / d[1]]
    B = np.array(cols).T * ww[:, None]
    g = np.exp(-0.5 * om * om - 1j * om * T) * ww
    A = np.vstack([B.real, B.imag])
    y = np.concatenate([g.real, g.imag])
    # R(0) = dc . x = 1: x = x0 + N u
    dc = np.array(dc)
    x0 = dc / (dc @ dc)
    N = np.linalg.svd(dc[None, :])[2][1:].T
    u = np.linalg.lstsq(A @ N, y - A @ x0, rcond=None)[0]
    xs = x0 + N @ u
    secs, k = [], 0
    for d in dens:
        if len(d) == 1:
            secs.append((float(xs[k]), d[0]))
            k += 1
        else:
            secs.append((float(xs[k]), float(xs[k + 1]), d[0], d[1]))
            k += 2
    # unit area after rounding to double (to ~1e-16; the kernel string is exact)
    area = sum(s_[-3] / s_[-1] if len(s_) == 4 else s_[0] / s_[1] for s_ in secs)
    secs = [((s_[0] / area, s_[1]) if len(s_) == 2 else
             (s_[0] / area, s_[1] / area, s_[2], s_[3])) for s_ in secs]
    return secs


def _poles_residues(secs):
    poles, res = [], []
    for sec in secs:
        if len(sec) == 2:
            poles.append(complex(-sec[1]))
            res.append(complex(sec[0]))
            continue
        n1, n0, d1, d0 = sec
        disc = np.sqrt(complex(d1 * d1 - 4.0 * d0))
        p, q = (-d1 + disc) / 2.0, (-d1 - disc) / 2.0
        for x, y in ((p, q), (q, p)):
            poles.append(complex(x))
            res.append(complex((n1 * x + n0) / (x - y)))
    return tuple(poles), tuple(res)


def _sections_transfer(secs, z):
    R = np.zeros_like(z)
    for sec in secs:
        if len(sec) == 2:
            R = R + sec[0] / (z + sec[1])
        else:
            R = R + (sec[0] * z + sec[1]) / (z * z + sec[2] * z + sec[3])
    return R


def _fraction(secs):
    """``N/D`` of the sections multiplied out exactly (``Fraction``,
    ascending coefficients), ``D`` monic, ``N`` scaled to ``N(0) = D(0)``
    (unit area exactly: the sections' area after rounding is 1 only to
    ~1e-15)."""
    from fractions import Fraction as Fr

    def mul(a, b):
        out = [Fr(0)] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] += x * y
        return out
    dens = [[Fr(q[1]), Fr(1)] if len(q) == 2 else [Fr(q[3]), Fr(q[2]), Fr(1)] for q in secs]
    nums = [[Fr(q[0])] if len(q) == 2 else [Fr(q[1]), Fr(q[0])] for q in secs]
    D = [Fr(1)]
    for d in dens:
        D = mul(D, d)
    N = [Fr(0)] * (len(D) - 1)
    for k, n in enumerate(nums):
        t = n
        for j, d in enumerate(dens):
            if j != k:
                t = mul(t, d)
        N = [x + y for x, y in zip(N, t + [Fr(0)] * (len(N) - len(t)))]
    scale = D[0] / N[0]
    return tuple(x * scale for x in N), tuple(D)


def _errors(fit_poles, fit_res, secs, T):
    """``l1``, ``sup``, ``spectral`` of the sections (``sigma = 1``)."""
    from scipy.special import roots_legendre, ndtr
    p, rho = np.array(fit_poles), np.array(fit_res)
    L = T + 30.0
    x, w = roots_legendre(16)
    edges = np.linspace(0.0, L, 1601)
    h = np.diff(edges)
    t = (edges[:-1, None] + 0.5 * h[:, None] * (x[None, :] + 1.0)).ravel()
    ww = (0.5 * h[:, None] * w[None, :]).ravel()
    r = (np.exp(np.multiply.outer(t, p)) @ rho).real
    l1 = float(ww @ np.abs(r - _phi(t, T)))
    # tails: t > L (|r| bounded by its modes' envelopes, negligible at T + 30;
    # the Gaussian's mass beyond L), t < 0
    l1 += float(np.sum(np.abs(rho) * np.exp(p.real * L) / np.abs(p.real)))
    l1 += float(ndtr(-(L - T))) + float(ndtr(-T))
    om = np.concatenate([np.linspace(0.0, 12.0, 6001), np.geomspace(12.0, 1e4, 400)])
    R = _sections_transfer(secs, 1j * om)
    G = np.exp(-0.5 * om * om - 1j * om * T)
    sup = float(np.max(np.abs(R - G)))
    spectral = float(np.max(np.abs(np.abs(R) ** 2 - np.exp(-om * om))))
    return l1, sup, spectral


@functools.lru_cache(maxsize=32)
def gaussian_fit(tol: float = 1e-6) -> GaussianFit:
    """The rational fit of the shifted unit Gaussian with L1 error ``<= tol``
    (module docstring); cached per ``tol``."""
    from scipy.special import ndtr
    from api._realization import hankel_singular_values
    tol = float(tol)
    if not TOL_RANGE[0] <= tol <= TOL_RANGE[1]:
        raise ValueError(f'gaussian_fit: tol={tol:g} outside {TOL_RANGE}')
    T = 3.0
    while ndtr(-T) > tol / 10.0:
        T += 0.25
    a, b, c = _laguerre_network(_laguerre_coefficients(T))
    sv, Rf, Lf, U, Vh, ae, be, ce = hankel_singular_values(a, b, c)
    tail = 2.0 * np.concatenate([np.cumsum(sv[::-1])[::-1], [0.0]])
    r = max(int(np.argmax(tail <= tol * sv[0])), 2)
    while True:
        if r > _R_MAX:
            raise ArithmeticError(f'gaussian_fit: tol={tol:g} not reached with {_R_MAX} states')
        srt = np.sqrt(sv[:r])
        Tm = (Rf @ Vh[:r].conj().T) / srt
        W = (Lf @ U[:, :r]) / srt
        ar = (W.conj().T @ ae @ Tm).real
        if np.all(np.linalg.eigvals(ar).real < 0):
            secs = _numerators(_pole_sections(ar), T)
            poles, res = _poles_residues(secs)
            if all(p.real < 0 for p in poles):
                l1, sup, spectral = _errors(poles, res, secs, T)
                if l1 <= tol:
                    break
        r += 1
    return GaussianFit(tol=tol, shift=T, order=len(poles), poles=poles, residues=res,
                       sections=tuple(secs), l1=l1, sup=sup, spectral=spectral,
                       truncated_mass=float(ndtr(-T)), hsv=tuple(float(x) for x in sv[:r + 4]))

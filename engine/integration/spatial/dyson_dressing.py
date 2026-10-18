"""
engine.integration.spatial.dyson_dressing
=========================================
Dyson–Duhamel dressing for UNEQUAL diffusion (paper Appendix B §B.24–B.30;
``docs/dyson_duhamel_integration_plan.md`` steps D-2/D-3).

With ``𝒟 = D₀·I + 𝒟̂`` (``𝒟̂ ≠ 0``) the retarded propagator expands in powers of
the residual-diffusion insertion ``𝒟̂|k|²``::

    G_R(t,k) = Σ_{n≥0} G_n(t,k),
    G_n(t,k) = (−|k|²)^n · e^{−D₀|k|²t} · 𝓗_n(t),                      (B26)
    𝓗_n(t)  = Σ_{α_0..α_n} P_{α_0}𝒟̂P_{α_1}⋯𝒟̂P_{α_n} ·
              e^{−m_{α_0}t} · Φ_n(t; m_{α_1}−m_{α_0}, …, m_{α_n}−m_{α_0}),   (B27)

derived from the n-fold Duhamel convolution
``G_n(t) = (−|k|²)^n ∫_{t≥s_1≥…≥s_n≥0} e^{−M(t−s_1)}𝒟̂e^{−M(s_1−s_2)}𝒟̂⋯e^{−Ms_n}``
whose nested exponential integral is the divided difference
``(−1)^n f[m_{α_0},…,m_{α_n}]`` of ``f(z)=e^{−zt}`` — shift-invariance gives the
``e^{−m_{α_0}t}·Φ_n`` form with :func:`spectral_propagator.phi_n` (Opitz/expm,
confluent-safe).

Tree-level dressed 2-point (D-3, q-space)::

    C^{(N)}(q,τ≥0) = ∫_0^∞ ds  G_R^{(N)}(τ+s, q) · N · G_R^{(N)}(s, q)ᵀ,

evaluated by Gauss–Legendre on the σ-concentrated substitution (the same
``σ = cap·v²`` trick as the chamber integrator).  Truncation order ``N`` comes
from the model policy ``model['spatial']['dyson']`` (builder D-4); convergence
requires the residual ratio ``‖𝒟̂‖/D₀ < 1`` (geometric in ``N`` at large ``q``;
exact at ``q=0`` where the insertion vanishes).

Exact oracle for validation: ``C(q,τ)=expm(−A(q)τ)·Σ(q)``, ``A(q)=M+𝒟q²``,
``A Σ + Σ Aᵀ = N`` (``simulations/coupled_rd_1d_sim.coupled_box_correlator`` — itself
pinned against the unequal-D Langevin simulation).
"""
from __future__ import annotations

import itertools

import numpy as np

from engine.integration.spatial.spectral_propagator import (
    spectral_projectors, phi_n_batch,
)


def hcal_n(ts, M, Dhat, n):
    """``𝓗_n(t)`` (B27) for an array of times — returns ``(n_t, nf, nf)``.

    ``n = 0`` → ``e^{−Mt}`` (the bare matrix decay).  Cost ``nf^{n+1}`` strings;
    each string is an outer-product chain ``P_{α_0}𝒟̂P_{α_1}⋯𝒟̂P_{α_n}`` times
    the scalar ``e^{−m_{α_0}t}Φ_n(t; m_{α_i}−m_{α_0})`` evaluated by the
    confluent-safe Opitz ``Φ_n``."""
    ts = np.asarray(ts, dtype=float)
    M = np.asarray(M, dtype=float)
    Dhat = np.asarray(Dhat, dtype=float)
    eig, proj = spectral_projectors(M)
    nf = len(eig)
    out = np.zeros((ts.size, nf, nf), dtype=complex)
    for string in itertools.product(range(nf), repeat=n + 1):
        mat = proj[string[0]]
        for a_i in string[1:]:
            mat = mat @ Dhat @ proj[a_i]
        if not np.any(np.abs(mat) > 1e-300):
            continue
        m0 = eig[string[0]]
        nus = [eig[a_i] - m0 for a_i in string[1:]]
        scal = np.exp(-m0 * ts) * phi_n_batch(ts, nus)
        out += scal[:, None, None] * mat[None, :, :]
    return out


def dressed_GR(ts, qsq, M, Dhat, D0, order):
    """Truncated dressed retarded propagator ``G_R^{(N)}(t, k)`` (B24/B26) for an
    array of times at one ``|k|² = qsq`` — returns ``(n_t, nf, nf)`` complex.

    ``Σ_{n=0}^{order} (−qsq)^n e^{−D₀·qsq·t} 𝓗_n(t)``; ``order=0`` is the
    scalar-``D₀`` reference ``G₀`` (exact when ``𝒟̂=0``)."""
    ts = np.asarray(ts, dtype=float)
    acc = np.zeros((ts.size, np.asarray(M).shape[0], np.asarray(M).shape[0]),
                   dtype=complex)
    for n in range(order + 1):
        acc += ((-qsq) ** n) * hcal_n(ts, M, Dhat, n)
    return np.exp(-D0 * qsq * ts)[:, None, None] * acc


def dressed_tree_C(q, tau, M, Dhat, D0, N, order, n_s=48, s_cap_scale=32.0):
    """Dressed tree-level 2-point **matrix** ``C^{(N)}(q, τ)`` (D-3).

    ``∫_0^∞ ds G_R^{(N)}(|τ|+s, q) N G_R^{(N)}(s, q)ᵀ`` for ``τ ≥ 0``
    (``τ < 0`` → transpose), by Gauss–Legendre on ``s = cap·v²`` (nodes
    concentrated near ``s=0``, cap set by the slowest eigenvalue).  Exact in the
    ``order→∞`` limit; at ``order=0`` with ``𝒟̂=0`` it reproduces the
    spectral-Lyapunov ``coupled_two_point``."""
    M = np.asarray(M, dtype=float)
    eig, _ = spectral_projectors(M)
    mu_min = float(np.min(eig.real))
    if not mu_min > 0.0:
        raise ValueError(f'dressed_tree_C: min Re eig(M) = {mu_min} <= 0.')
    cap = s_cap_scale / (mu_min + float(D0) * float(q) ** 2)
    xv, wv = np.polynomial.legendre.leggauss(n_s)
    vv = 0.5 * (xv + 1.0)
    s_nodes = cap * vv * vv
    s_w = wv * cap * vv                                  # ∫ds = Σ w·cap·v
    at = abs(float(tau))
    qsq = float(q) ** 2
    G_late = dressed_GR(at + s_nodes, qsq, M, Dhat, D0, order)   # (n_s, nf, nf)
    G_early = dressed_GR(s_nodes, qsq, M, Dhat, D0, order)
    N = np.asarray(N, dtype=float)
    C = np.einsum('s,sij,jk,slk->il', s_w, G_late, N, G_early)
    return C if tau >= 0 else C.T


def dressed_tree_C_q_grid(qs, taus, M, Dhat, D0, N, order, n_s=48):
    """Vectorized convenience: ``C^{(N)}_{ij}`` over a q-grid × τ-grid —
    returns ``(n_tau, n_q, nf, nf)`` complex (the tree driver's FT input)."""
    qs = np.asarray(qs, dtype=float)
    taus = np.asarray(taus, dtype=float)
    nf = np.asarray(M).shape[0]
    out = np.empty((taus.size, qs.size, nf, nf), dtype=complex)
    for iq, q in enumerate(qs):
        for it, tau in enumerate(taus):
            out[it, iq] = dressed_tree_C(q, tau, M, Dhat, D0, N, order, n_s=n_s)
    return out


# ── exact q→x transform of the dressed tree (no q-grid) ───────────────
# ``G_R^{(N)}(t,q) = e^{−D₀q²t} Σ_n (−q²)^n 𝓗_n(t)`` with q-INDEPENDENT 𝓗_n,
# so every term of ``C^{(N)}(q,τ) = ∫_0^∞ G(|τ|+s) N G(s)ᵀ ds`` is a matrix
# function of s times ``|q|^{2p} e^{−a q²}``, ``a = D₀(|τ|+2s)``, ``p = n+m``.
# Its inverse FT is the Laguerre–Gaussian closed form
#     ∫dᵈq/(2π)ᵈ e^{iq·x} |q|^{2p} e^{−a q²} = K_a(x) · p! · a^{−p} · L_p^{(d/2−1)}(|x|²/4a),
# ``K_a = (4πa)^{−d/2} e^{−|x|²/4a}`` (``(−Δ)^p`` of the heat kernel), so
#     C^{(N)}(x,τ) = ∫_0^∞ ds Σ_{n,m≤N} (−1)^{n+m} F_{n+m}(x; a(s)) [𝓗_n(|τ|+s) N 𝓗_m(s)ᵀ],
# one smooth s-integral by adaptive quadrature with a checked error estimate.
# A periodic box sums images of F (or, for a wide Gaussian, the dual mode
# sum) to convergence; nothing is truncated at a fixed q_cut / mode count.
def _laguerre_gauss(p, r2, a, d):
    """``F_p`` on the infinite domain for squared distances ``r2`` (array)."""
    from math import factorial
    from scipy.special import eval_genlaguerre
    u = r2 / (4.0 * a)
    return ((4.0 * np.pi * a) ** (-0.5 * d) * np.exp(-u) * factorial(p)
            * a ** (-p) * eval_genlaguerre(p, 0.5 * d - 1.0, u))


def _periodic_1d(p, x, a, L, rtol=1e-16):
    """1-d periodic ``F_p``: image sum of :func:`_laguerre_gauss` (narrow
    Gaussian) or the dual mode sum ``(1/L) Σ_k q_k^{2p} e^{−a q_k²} cos(q_k x)``
    (wide one), each summed until the terms are below ``rtol`` of the total
    past the peak of the polynomial factor."""
    x = np.asarray(x, dtype=float)
    if a <= L * L / (4.0 * np.pi):
        tot = _laguerre_gauss(p, x * x, a, 1)
        n = 1
        n_min = int(np.ceil((np.max(np.abs(x)) + np.sqrt(8.0 * p * a)) / L)) + 1
        quiet = 0
        while True:
            add = (_laguerre_gauss(p, (x + n * L) ** 2, a, 1)
                   + _laguerre_gauss(p, (x - n * L) ** 2, a, 1))
            tot = tot + add
            quiet = quiet + 1 if np.max(np.abs(add)) <= rtol * max(
                np.max(np.abs(tot)), 1e-300) else 0
            if n >= n_min and quiet >= 2:
                return tot
            n += 1
            if n > 100000:
                raise RuntimeError('periodic image sum did not converge.')
    tot = np.full(x.shape, (1.0 / L) * (1.0 if p == 0 else 0.0))
    k = 1
    k_min = int(np.ceil(np.sqrt(p / a) * L / (2.0 * np.pi))) + 1
    while True:
        q = 2.0 * np.pi * k / L
        bound = (2.0 / L) * q ** (2 * p) * np.exp(-a * q * q)
        tot = tot + bound * np.cos(q * x)
        # stop on the term's BOUND (cos -> 1): an x on a zero of cos(q x)
        # must not end the sum early
        if k >= k_min and bound <= rtol * max(np.max(np.abs(tot)), 1e-300):
            return tot
        k += 1
        if k > 100000:
            raise RuntimeError('periodic mode sum did not converge.')


def _F_p(p, xs, a, d, L=None):
    """``F_p(x; a)`` at distances ``xs`` (x along axis 0 on a periodic box)."""
    xs = np.asarray(xs, dtype=float)
    if L is None:
        return _laguerre_gauss(p, xs * xs, a, d)
    if d == 1:
        return _periodic_1d(p, xs, a, L)
    # |q|^{2p} = Σ_{p_0+…+p_{d−1}=p} p!/∏p_c! ∏_c q_c^{2p_c}: a product of
    # 1-d periodic factors (x on axis 0, the other components at 0)
    from math import factorial
    out = np.zeros(xs.shape)
    for parts in itertools.product(range(p + 1), repeat=d):
        if sum(parts) != p:
            continue
        coef = factorial(p)
        for pc in parts:
            coef //= factorial(pc)
        term = coef * _periodic_1d(parts[0], xs, a, L)
        for pc in parts[1:]:
            term = term * _periodic_1d(pc, np.zeros(1), a, L)[0]
        out = out + term
    return out


def dressed_tree_C_x(xs, tau, M, Dhat, D0, N, order, legs, spatial_dim=1,
                     L=None, epsrel=1e-11):
    """EXACT real-space dressed tree ``C^{(N)}_{ij}(x, τ)`` (package convention
    ``⟨φ_i(0,0) φ_j(x,τ)⟩``) at the distances ``xs`` (along axis 0 for a
    periodic box of side ``L``): the closed-form inverse FT of
    :func:`dressed_tree_C` (see the block comment above) — no q-grid, no
    ``q_cut``, no mode cap.  ``legs = (i, j)``.  The remaining s-integral is
    adaptive (``scipy.integrate.quad_vec``); its error estimate is checked and
    a failure raises instead of returning a number.  ``d ≥ 2`` refuses the
    UV-divergent equal point ``x = τ = 0``."""
    from scipy.integrate import quad_vec
    xs = np.asarray(xs, dtype=float)
    d = int(spatial_dim)
    M = np.asarray(M, dtype=float)
    Dhat = np.asarray(Dhat, dtype=float)
    Nm = np.asarray(N, dtype=float)
    D0 = float(D0)
    i, j = legs
    at = abs(float(tau))
    ii, jj = (j, i) if tau >= 0 else (i, j)
    if d >= 2 and at == 0.0 and np.any(xs == 0.0):
        from engine.integration.spatial.heat_kernel import (
            SpatialPropagatorError)
        raise SpatialPropagatorError(
            f'equal-point (x=0, tau=0) dressed tree correlator is '
            f'UV-divergent at d={d}.')
    eig, proj = spectral_projectors(M)
    mu_min = float(np.min(eig.real))
    if not mu_min > 0.0:
        raise ValueError(f'dressed_tree_C_x: min Re eig(M) = {mu_min} <= 0.')
    # the n-th order strings P_{α0}𝒟̂P_{α1}⋯ once (shared by every s)
    strings = []
    for n in range(order + 1):
        sn = []
        for st in itertools.product(range(len(eig)), repeat=n + 1):
            mat = proj[st[0]]
            for a_i in st[1:]:
                mat = mat @ Dhat @ proj[a_i]
            if np.any(np.abs(mat) > 1e-300):
                sn.append((mat, eig[st[0]], [eig[a_i] - eig[st[0]]
                                             for a_i in st[1:]]))
        strings.append(sn)

    def _H(n, t):
        out = np.zeros(M.shape, dtype=complex)
        for mat, m0, nus in strings[n]:
            out += mat * (np.exp(-m0 * t) * phi_n_batch(np.array([t]), nus)[0])
        return out

    S = 1.0 / mu_min

    def _f(v):
        s = S * v * v
        a = D0 * (at + 2.0 * s)
        Hl = [_H(n, at + s) for n in range(order + 1)]
        He = [_H(m, s) for m in range(order + 1)]
        acc = np.zeros(xs.shape, dtype=complex)
        Fc = {}
        for n in range(order + 1):
            HlN = Hl[n] @ Nm
            for m in range(order + 1):
                w = (HlN @ He[m].T)[ii, jj]
                if w == 0:
                    continue
                p = n + m
                if p not in Fc:
                    Fc[p] = _F_p(p, xs, a, d, L)
                acc = acc + ((-1) ** p) * w * Fc[p]
        return acc * (2.0 * S * v)

    val, err = quad_vec(_f, 0.0, np.inf, epsabs=0.0, epsrel=epsrel,
                        norm='max', limit=4000)
    scale = float(np.max(np.abs(val))) or 1.0
    if not np.all(np.isfinite(val)) or err > 1e3 * epsrel * scale:
        raise RuntimeError(
            f'dressed_tree_C_x: the s-integral did not converge (error '
            f'estimate {err:.2e} against {scale:.2e}); no value returned.')
    return val

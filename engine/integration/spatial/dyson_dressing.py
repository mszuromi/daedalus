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

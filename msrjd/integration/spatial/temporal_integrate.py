"""
msrjd.integration.spatial.temporal_integrate
=============================================
Backend C — **C2 (the causal time-simplex)**.  After the C0/C1 momentum
reduction (``spatial_reduce``), a self-energy diagram is internal vertices joined
by retarded ``G_R`` and correlation ``C`` lines; C2 integrates over the edge
**time parameters** with the causal structure.  Design: ``docs/backend_C_design.md``
(C2); math: ``docs/backend_C_math.md`` §3.

Scope (this milestone): **2-vertex self-energies** — the 1-loop bubble and the
2-loop sunset — i.e. all internal edges span the SAME inter-vertex time ``t``
(one ordering chamber).  In the heat-kernel ``(k,t)`` representation each edge is
``e^{−(μ+Dk_e²) w_e}``:

  * a **retarded** ``G_R`` edge has a FIXED duration ``w_e = t``  (``θ(t)``);
  * a **correlation** ``C`` edge carries a Schwinger parameter ``w_e = s_e``,
    INTEGRATED over ``s_e ∈ [|t|, ∞)``  (since ``C(k,Δt)=T∫_{|Δt|}^∞ ds e^{−m_k s}``),
    each contributing a factor ``T``.

So the self-energy is

  Σ(q,t) = T^{n_C} ∫_{[|t|,∞)^{n_C}} ∏ ds_C · e^{−μ Σ_e w_e}
                 · I_mom( {(a_e,b_e)}, {w_e}, q )

with ``I_mom`` the C1 momentum integral (``spatial_reduce.momentum_integral``),
``w_e = t`` on retarded edges and ``w_e = s_e`` on correlation edges.  The
``∫dℓ`` is already done analytically in ``I_mom`` (no momentum poles → close-pair
cannot arise — math §4b), so this is a smooth, finite Schwinger/time quadrature.

Validated against backend B (``loop_parametric.sigma_R_kernel/sigma_K_kernel``,
itself pinned vs direct ``∫dℓ``) for the 1-loop bubble, and against a direct
``∫dℓ₁dℓ₂`` for the 2-loop sunset at ``t=0``.

**Beyond this milestone:** multi-vertex self-energies need the Phase-J ordering
chambers (retarded ``θ``-orderings carve >1 chamber); this 2-vertex assembler is
the single-chamber case.  The cutoff (``gaussian_edge|hard_spherical|lattice_bz``,
design §7) is the next first-class input to thread into ``I_mom``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from msrjd.integration.spatial.spatial_reduce import momentum_integral

_RET = ('R', 'retarded')
_COR = ('C', 'correlation')


def sigma_parametric(edges, q, t, mu, D, T, spatial_dim=1, s_cap=None,
                     quad_opts=None):
    """Self-energy ``Σ(q,t)`` of a 2-vertex diagram via the causal parametric
    route (C2).

    edges : list of ``(a, b, kind)`` per internal (loop) edge, where ``a`` is the
            length-``L`` loop-momentum coefficient tuple, ``b`` the length-``n_ext``
            external coefficient tuple (from ``RoutingResult.edge_coeffs``), and
            ``kind`` ∈ {'R','retarded','C','correlation'}.
    q     : external momentum (scalar or length-``n_ext`` sequence).
    t     : inter-vertex time (retarded edges use ``|t|``; correlation Schwinger
            params run ``[|t|, ∞)``).  ``t=0`` ⇒ equal-time (use a tiny floor).

    Returns ``Σ(q,t)`` (float).  ``T^{n_C}`` and ``e^{−μΣw}`` are included; the
    diagram's combinatorial factor ``M(Γ)`` is applied by the caller.
    """
    a_all = [e[0] for e in edges]
    b_all = [e[1] for e in edges]
    kinds = [e[2] for e in edges]
    c_idx = [i for i, k in enumerate(kinds) if k in _COR]
    r_idx = [i for i, k in enumerate(kinds) if k in _RET]
    if len(c_idx) + len(r_idx) != len(edges):
        raise ValueError(f"edge kinds must be in {_RET + _COR}; got {kinds}")
    nC = len(c_idx)
    tt = abs(float(t))
    lo = max(tt, 1e-9)                       # avoid the U→0 corner (s_e→0)
    hi = (lo + (s_cap if s_cap is not None else 60.0 / max(mu, 1e-6)))
    opts = quad_opts or {'limit': 200}

    def _integrand(svals):
        w = [0.0] * len(edges)
        for i in r_idx:
            w[i] = tt
        for j, i in enumerate(c_idx):
            w[i] = svals[j]
        mom = momentum_integral(a_all, b_all, w, q, D, spatial_dim=spatial_dim)
        return math.exp(-mu * sum(w)) * mom

    if nC == 0:
        val = _integrand([])
    elif nC <= 2:
        # low-dim: adaptive quad (fast, pinned vs backend B for the bubble).
        if nC == 1:
            val, _ = integrate.quad(lambda s: _integrand([s]), lo, hi, **opts)
        else:
            val, _ = integrate.dblquad(
                lambda s2, s1: _integrand([s1, s2]), lo, hi,
                lambda _s1: lo, lambda _s1: hi)
    else:
        # nC ≥ 3 (sunset …): adaptive nquad is intractable (and the t→0 corner is
        # singular).  Use a Gauss–Laguerre TENSOR rule, which integrates the
        # e^{−μ s_C} weight exactly: with s_C = lo + x/μ,
        #   ∫_lo^∞ ds e^{−μs} g(s) = (e^{−μ lo}/μ) Σ_k w_k g(lo + x_k/μ).
        # The retarded edges contribute the constant e^{−μ·#R·t}.
        import itertools
        deg = 40
        xk, wk = np.polynomial.laguerre.laggauss(deg)
        sk = lo + xk / mu
        const = (math.exp(-mu * tt * len(r_idx))
                 * (math.exp(-mu * lo) / mu) ** nC)
        acc = 0.0
        w = [tt if i in r_idx else 0.0 for i in range(len(edges))]
        for combo in itertools.product(range(deg), repeat=nC):
            wprod = 1.0
            for d_, ci in enumerate(c_idx):
                w[ci] = sk[combo[d_]]
                wprod *= wk[combo[d_]]
            acc += wprod * momentum_integral(a_all, b_all, w, q, D,
                                             spatial_dim=spatial_dim)
        val = const * acc
    return (T ** nC) * val


# ── convenience edge specs for the validated 2-vertex topologies ──
def bubble_edges(kind_R='R'):
    """The φ̃φ² 1-loop bubble loop edges (a over the single loop ℓ, b over the
    single external q): ``k=ℓ`` and ``k=q−ℓ``.  ``kind_R='R'`` gives the retarded
    self-energy Σ_R (one G_R + one C); ``kind_R='C'`` gives the Keldysh Σ_K
    (both correlation)."""
    return [((1.0,), (0.0,), kind_R), ((-1.0,), (1.0,), 'C')]


def sunset_edges():
    """The 2-loop sunset loop edges: ``k=ℓ₁``, ``k=ℓ₂``, ``k=q−ℓ₁−ℓ₂`` (all
    correlation, the equal-time Keldysh sunset)."""
    return [((1.0, 0.0), (0.0,), 'C'),
            ((0.0, 1.0), (0.0,), 'C'),
            ((-1.0, -1.0), (1.0,), 'C')]


# ── C3-lite capstone: the full equal-time bubble δC(q,0) via the C stack ──
def bubble_delta_equal_time_via_C(q, mu, D, T, g=1.0, C_R=4.0, C_K=2.0,
                                  n_a=160, a_max=None):
    """End-to-end **C0→C1→C2→C3-lite**: the φ̃φ² 1-loop equal-time bubble
    ``δC(q,0)`` assembled entirely from the new stack — tabulate ``Σ_R(q,a)`` and
    ``Σ_K(q,a)`` via :func:`sigma_parametric` (which uses the C1 ``momentum_integral``
    over the C0 Symanzik form), then collapse with the MSR Dyson equation:

        δC(q,0) = g²·[ C_R·(T/m²)∫₀^∞ Σ_R(a)e^{−ma}da  +  C_K·(1/m)∫₀^∞ Σ_K(a)e^{−ma}da ].

    Reproduces ``loop_dyson.bubble_delta_S`` (the backend-B / golden reference) —
    i.e. the new stack yields the validated physical correlator.  ``C_R=4, C_K=2``
    are the pinned ``M(Γ)`` weights (W9 will derive them analytically).  This is a
    finite-cutoff (Regime-1/2) evaluation: with the continuum (no-cutoff) σ the
    small-``a`` UV tail is the integrable ``a^{−1/2}`` (handled by the quadrature).
    """
    m = mu + D * q * q
    if a_max is None:
        a_max = 40.0 / m
    ag = np.linspace(a_max / (10 * n_a), a_max, n_a)   # start near 0 (small-a tail)
    sR = np.array([sigma_parametric(bubble_edges('R'), q, a, mu, D, T)
                   for a in ag])
    sK = np.array([sigma_parametric(bubble_edges('C'), q, a, mu, D, T)
                   for a in ag])
    e = np.exp(-m * ag)
    t1 = (T / (m * m)) * np.trapz(sR * e, ag)
    t2 = (1.0 / m) * np.trapz(sK * e, ag)
    return g * g * (C_R * t1 + C_K * t2)

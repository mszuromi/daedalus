"""
msrjd.integration.spatial.full_integrator
==========================================
Backend C — **the full-diagram integrator**.  ONE genuine integral evaluates
*every* enumerated diagram (tree, bubble, tadpole, sunset, … any ``k``, ``ell``,
``d``) — no Dyson convolution, no mass-shift shortcut, no 1PI bookkeeping.

For a typed diagram mapped to the C-stack
(:func:`diagram_descriptor.diagram_to_cstack`) with internal interaction vertices
at times ``{t_v}`` (integrated), external leaves at FIXED times ``{τ_j}`` /
momenta ``{q_j}``, and edges (retarded ``R`` or correlation ``C``, each with a
momentum routing ``k_e = a_e·ℓ + b_e·q``), the diagram's contribution to the
connected ``k``-point cumulant is

  Γ(q,{τ}) = 2^{−n_C}·𝒮(Γ) · ∫ ∏_v dt_v ∏_{C edges} dσ_e  𝟙(θ's) ·
                              e^{−μ Σ_e w_e} · MomFactor(w, q),

  w_e = t_head − t_tail   (R, with θ: w_e ≥ 0),
      = |t_a − t_b| + σ_e  (C, the Schwinger parameter ≥ |Δt|, σ_e ∈ [0,∞)),
  MomFactor = (4πD)^{−Ld/2} U(w)^{−d/2} e^{−D qᵀ Q_eff(w) q}   (``spatial_reduce``),

the **Symanzik momentum reduction** done analytically (general in ``L`` and ``d``;
the loop integral is smooth ⇒ no close-pair pathology).  The residual integral is
over the internal vertex times (the causal θ's carve the ordering chambers) and
the correlation Schwinger parameters — done here by quadrature.

**Normalization is derived, not fitted.**  The enumeration prefactor uses the
``2T`` noise-vertex value; a kinematic ``C`` edge here is the unit-amplitude
Schwinger factor ``∫dσ e^{−mσ}=1/m``; the ``2^{−n_C}`` converts between them.
Check (no loops): the tree (one ``C`` edge between the two leaves, ``n_C=1``,
prefactor ``2T``) gives ``2^{−1}·2T·(1/m)e^{−mτ} = (T/m)e^{−mτ} = C₀`` exactly.

Scope of THIS module: simple (non-derivative, non-convolution) interaction
vertices.  Derivative/form-factor edges multiply the loop integrand by ``F(ℓ)``
and are layered on separately.
"""
from __future__ import annotations

import math

import numpy as np

from msrjd.integration.spatial.causal_chambers import causal_chambers


# ── Notation (code ↔ paper App. B) ───────────────────────────────────
#   Lam      Λ        loop / first-Symanzik matrix  Σ_e w_e a_e a_eᵀ  (U=det Lam)
#   Bcal     𝓑(w)     external quadratic form  = D·Q_eff = Q − Nᵀ Lam⁻¹ N
#   N, Q     N_rb,Q_ab   Symanzik cross / external blocks ;  Q_eff = 𝓑/D
#   a, b     B_er,C_eb   edge routing coefficients (plain B,C in paper)
#   D        D_0      scalar reference diffusion ;  w,q ↔ w_e,q_b
#   Scal     𝒮(Γ)     symmetry factor — prose 𝒮(Γ); local var Scal; dict key 'M' kept
# ─────────────────────────────────────────────────────────────────────


def _momentum_factor_batch(a, b, w_batch, q_vec, D, spatial_dim, u_floor=1e-300,
                           return_gaussian=False):
    """``MomFactor(w,q)`` for a BATCH of Schwinger-weight vectors ``w_batch``
    (shape ``P×E``) — vectorized Symanzik reduction.

    ``a`` (``E×L``), ``b`` (``E×n_ext``): the per-edge routing coefficients.
    Returns ``(P,)``.  ``L=0`` (tree) → ``exp(−D qᵀQq)``, ``Q=Σ_e w_e b_e b_eᵀ``;
    ``L≥1`` → ``(4πD)^{−Ld/2} U^{−d/2} exp(−D qᵀ Q_eff q)`` with ``U=det Lam``,
    ``Q_eff=Q−Nᵀ Lam⁻¹ N`` (batched ``det``/``solve``).

    ``return_gaussian`` also returns ``(Lam, N, ok)`` (the loop-momentum Gaussian's
    precision ``Lam=Σw_e a_e a_eᵀ`` and cross-term ``N=Σw_e a_e b_eᵀ``, with a
    non-degenerate mask) so a derivative-vertex **form factor** can be averaged
    over ``ℓ~N(−Lam⁻¹Nq, (2D Lam)⁻¹)`` — see :func:`_formfactor_average`.  ``Lam,N`` are
    ``None`` for ``L=0``."""
    E, L = a.shape
    qv = np.atleast_1d(np.asarray(q_vec, dtype=float))
    Q = np.einsum('pe,ej,ek->pjk', w_batch, b, b)            # (P, n_ext, n_ext)
    if L == 0:
        quad = np.einsum('j,pjk,k->p', qv, Q, qv)
        mf = np.exp(-D * quad)
        return (mf, None, None, None) if return_gaussian else mf
    Lam = np.einsum('pe,el,em->plm', w_batch, a, a)            # (P, L, L)
    N = np.einsum('pe,el,ej->plj', w_batch, a, b)            # (P, L, n_ext)
    U = np.linalg.det(Lam)                                     # (P,)
    ok = U > u_floor
    out = np.zeros(w_batch.shape[0])
    if np.any(ok):
        Lamok, Nok, Qok, Uok = Lam[ok], N[ok], Q[ok], U[ok]
        LamiN = np.linalg.solve(Lamok, Nok)                      # (P', L, n_ext)
        Qeff = Qok - np.einsum('plj,plk->pjk', Nok, LamiN)
        quad = np.einsum('j,pjk,k->p', qv, Qeff, qv)
        pref = (4.0 * math.pi * D) ** (-0.5 * spatial_dim * L) \
            * np.power(Uok, -0.5 * spatial_dim)
        out[ok] = pref * np.exp(-D * quad)
    return (out, Lam, N, ok) if return_gaussian else out


def _symanzik_kernel_batch(a, b, w_batch, D, spatial_dim, u_floor=1e-300,
                           return_gaussian=False):
    """Per-Schwinger-sample heat-kernel ingredients for the ANALYTIC spatial IFT
    (Case A — plain vertices).  Returns ``(pref, Bcal, ok)``: the q-Gaussian
    ``pref·exp(−q⃗ᵀ𝓑q⃗)`` UN-collapsed from q, with
    ``pref = (4πD)^{−Ld/2} U^{−d/2}`` and ``𝓑 = D·Q_eff``.

    ``n_ext = 1`` (k=2): ``Bcal`` is scalar ``(P,)`` and the IFT is the heat
    kernel

        ∫dᵈq/(2π)ᵈ e^{iq·x} pref·e^{−Bcal q²} = pref·(4πBcal)^{−d/2} e^{−|x|²/4Bcal}.

    ``n_ext ≥ 2`` (k ≥ 3): ``Bcal`` is the MATRIX ``(P, n, n)`` and the IFT is
    the multivariate Gaussian (see :func:`_heat_kernel_x_general`)

        ∫∏ⱼdᵈqⱼ/(2π)^{nd} e^{iΣqⱼ·Xⱼ} pref·e^{−q⃗ᵀ𝓑q⃗}
            = pref·(4π)^{−nd/2} det(𝓑)^{−d/2} e^{−¼ Σ_c X⃗_cᵀ𝓑⁻¹X⃗_c}

    (one factor per spatial component ``c``; the same ``𝓑`` for every
    component because the propagators are isotropic).  Either way NO q-grid
    and NO numerical FT.  ``L=0`` (tree): ``pref=1``, ``𝓑=D·Q``.  Mirrors
    :func:`_momentum_factor_batch` but does not contract with q (the
    x-dependence stays analytic).  ``return_gaussian`` also returns
    ``(Lam, N, Q)`` (``None`` for L=0) for the derivative-vertex Phase-2
    form-factor average (:func:`_formfactor_average_x`)."""
    E, L = a.shape
    P = w_batch.shape[0]
    Q = np.einsum('pe,ej,ek->pjk', w_batch, b, b)            # (P, n_ext, n_ext)
    n = Q.shape[1]
    if L == 0:
        B0 = D * Q[:, 0, 0] if n == 1 else D * Q
        ret = (np.ones(P), B0, np.ones(P, dtype=bool))
        return ret + (None, None, Q) if return_gaussian else ret
    Lam = np.einsum('pe,el,em->plm', w_batch, a, a)            # (P, L, L)
    N = np.einsum('pe,el,ej->plj', w_batch, a, b)            # (P, L, n_ext)
    U = np.linalg.det(Lam)                                     # (P,)
    ok = U > u_floor
    pref = np.zeros(P)
    Bcal = np.zeros(P) if n == 1 else np.zeros((P, n, n))
    if np.any(ok):
        Lamok, Nok, Qok, Uok = Lam[ok], N[ok], Q[ok], U[ok]
        LamiN = np.linalg.solve(Lamok, Nok)
        Qeff = Qok - np.einsum('plj,plk->pjk', Nok, LamiN)     # (P', n, n)
        pref[ok] = (4.0 * math.pi * D) ** (-0.5 * spatial_dim * L) \
            * np.power(Uok, -0.5 * spatial_dim)
        Bcal[ok] = D * Qeff[:, 0, 0] if n == 1 else D * Qeff
    return (pref, Bcal, ok, Lam, N, Q) if return_gaussian else (pref, Bcal, ok)


def _heat_kernel_x_general(Bcal_g, xs_arr, spatial_dim):
    """Analytic spatial IFT of the per-sample q-Gaussian ``e^{−q⃗ᵀ𝓑q⃗}`` at the
    evaluation points ``xs_arr`` — the (multivariate) heat kernel.  Returns
    ``hk`` of shape ``(P', n_x)``.

    ``n_ext = 1``: ``Bcal_g`` is ``(P',)`` and ``xs_arr`` is ``(n_x,)`` [d=1
    scalar offsets] or ``(n_x, d)`` [d≥2 vectors]; ``hk = (4πB)^{−d/2}
    e^{−|x|²/4B}``.

    ``n_ext ≥ 2``: ``Bcal_g`` is ``(P', n, n)`` and ``xs_arr`` is ``(n_x, n)``
    [d=1] or ``(n_x, n, d)`` [d≥2], where column ``j`` is the FT conjugate
    ``X_j`` of the j-th external momentum in the descriptor's ``b`` routing
    (``q_syms[j]`` = the momentum of ``external_legs[j]``; momentum
    conservation eliminated the LAST leaf, so ``X_j = x_{leg_j} −
    x_{leg_{k−1}}``).  ``hk = (4π)^{−nd/2} det(𝓑)^{−d/2}
    exp(−¼ Σ_c X⃗_cᵀ𝓑⁻¹X⃗_c)`` (product over the ``d`` spatial components,
    which share 𝓑 by isotropy).

    Callers must pre-filter degenerate samples (``det 𝓑 > 0``)."""
    d = float(spatial_dim)
    xs_arr = np.asarray(xs_arr, dtype=float)
    if Bcal_g.ndim == 1:                                     # n_ext = 1 (k=2)
        if xs_arr.ndim == 1:                                 # scalar offsets
            x2 = xs_arr ** 2                                  # (n_x,)
        else:                                                # (n_x, d) vectors
            x2 = np.sum(xs_arr ** 2, axis=-1)
        return ((4.0 * math.pi * Bcal_g)[:, None] ** (-0.5 * d)
                * np.exp(-x2[None, :] / (4.0 * Bcal_g[:, None])))
    n = Bcal_g.shape[1]
    if xs_arr.ndim == 1:
        raise ValueError(
            f'n_ext={n} multivariate IFT needs evaluation points of shape '
            f'(n_x, {n}) [d=1] or (n_x, {n}, d); got 1-d xs.')
    detB = np.linalg.det(Bcal_g)                             # (P',)
    Binv = np.linalg.inv(Bcal_g)                             # (P', n, n)
    if xs_arr.ndim == 2:                                     # d=1: (n_x, n)
        quad = np.einsum('xj,pjk,xk->px', xs_arr, Binv, xs_arr)
    else:                                                    # d≥2: (n_x, n, d)
        quad = np.einsum('xjc,pjk,xkc->px', xs_arr, Binv, xs_arr)
    return ((4.0 * math.pi) ** (-0.5 * n * d)
            * detB[:, None] ** (-0.5 * d)
            * np.exp(-0.25 * quad))


def _formfactor_average(formfactor, Lam, N, q_vec, D, ok, gh_order=6, spatial_dim=1):
    """``⟨F(ℓ,q)⟩`` of a derivative-vertex form factor over the loop-momentum
    Gaussian ``ℓ ~ N(ℓ̄, Σ)``, ``ℓ̄ = −Lam⁻¹N q``, ``Σ = (2D Lam)⁻¹``, by
    Gauss–Hermite — **EXACT** for a polynomial ``F`` (a momentum-space derivative
    vertex deposits exactly a polynomial: ``Lap→−|k|²``, ``∂_x→ik``).  The base
    ``MomFactor`` already carries the Gaussian normalization + the ``1/(2π)^{Ld}``
    measure, so the full derivative-vertex loop integral is ``MomFactor·⟨F⟩``
    (validated to 1e-12 vs brute ``∫dℓ``).

    Generic in the loop number ``L``, the number of externals ``n_ext``, AND the
    spatial dimension ``d=spatial_dim``.  The loop covariance factorizes as
    ``Σ ⊗ I_d`` (isotropic propagators ⇒ the ``d`` spatial components are
    independent, same ``L×L`` precision ``Lam``, means from the matching component
    of ``q``).  Placing ``q`` on **axis 0** (legit for rotation-invariant Lap /
    full-gradient vertices), the parallel component (α=0) gets ``ℓ̄=−Lam⁻¹N|q|`` and
    the transverse components (α≥1) are zero-mean — an ``L·d``-dimensional GH grid.

    ``d=1``: ``formfactor(ell, q)`` with ``ell`` ``(P',G,L)`` and ``q`` ``(n_ext,)``.
    ``d≥2``: ``ell`` is ``(P',G,L,d)`` and ``q`` is ``(n_ext,d)`` (``q[:,0]=|q|``,
    rest 0).  Returns ``(P,)``; ``1.0`` where the loop is degenerate."""
    P, L = Lam.shape[0], Lam.shape[1]
    out = np.ones(P, dtype=complex)                          # COMPLEX: ∂_x→ik
    if not np.any(ok):
        return out
    qv = np.atleast_1d(np.asarray(q_vec, dtype=float))       # (n_ext,) magnitudes
    Lamok, Nok = Lam[ok], N[ok]                                  # (P',L,L), (P',L,n_ext)
    lbar0 = -np.einsum('plj,j->pl', np.linalg.solve(Lamok, Nok), qv)  # parallel mean (P',L)
    Sig = np.linalg.inv(Lamok) / (2.0 * D)                     # (P',L,L)
    Ch = np.linalg.cholesky(Sig)                             # (P',L,L)
    xg, wg = np.polynomial.hermite_e.hermegauss(gh_order)    # weight e^{−x²/2}
    d = int(spatial_dim)

    if d == 1:                                               # ── validated scalar path
        Z = np.stack([g.ravel() for g in
                      np.meshgrid(*([xg] * L), indexing='ij')], axis=-1)   # (G,L)
        Wg = np.ones(xg.size ** L)
        for wgrid in np.meshgrid(*([wg] * L), indexing='ij'):
            Wg = Wg * wgrid.ravel()
        Wg = Wg / (2.0 * math.pi) ** (0.5 * L)
        ell = lbar0[:, None, :] + np.einsum('plm,gm->pgl', Ch, Z)          # (P',G,L)
        Fv = np.asarray(formfactor(ell, qv), dtype=complex)               # (P',G)
        out[ok] = np.einsum('pg,g->p', Fv, Wg)
        return out

    # ── d ≥ 2: L·d-dim GH (q on axis 0; α=0 shifted, α≥1 zero-mean) ──────
    Ld = L * d
    Z = np.stack([g.ravel() for g in
                  np.meshgrid(*([xg] * Ld), indexing='ij')], axis=-1)      # (G, L·d)
    Wg = np.ones(xg.size ** Ld)
    for wgrid in np.meshgrid(*([wg] * Ld), indexing='ij'):
        Wg = Wg * wgrid.ravel()
    Wg = Wg / (2.0 * math.pi) ** (0.5 * Ld)                                # (G,)
    Zr = Z.reshape(Z.shape[0], d, L)                          # (G, α, loop) — α-major
    nP, G = Lamok.shape[0], Z.shape[0]
    ell = np.zeros((nP, G, L, d))
    for al in range(d):
        comp = np.einsum('plm,gm->pgl', Ch, Zr[:, al, :])    # (P',G,L)  zero-mean draw
        if al == 0:
            comp = comp + lbar0[:, None, :]                  # parallel: shift by ℓ̄
        ell[:, :, :, al] = comp
    q_comp = np.zeros((qv.size, d)); q_comp[:, 0] = qv        # q on axis 0
    Fv = np.asarray(formfactor(ell, q_comp), dtype=complex)  # (P',G)
    out[ok] = np.einsum('pg,g->p', Fv, Wg)
    return out


def _formfactor_average_x(formfactor, Lam, N, Q, D, ok, xs, spatial_dim=1,
                          gh_order=6, q_deg=8):
    """Phase 2 — analytic q→x IFT of a derivative-vertex form factor (d=1, k=2).

    Returns ``FF`` of shape ``(P, n_x)``: the loop-averaged form factor's
    contribution to the spatial IFT, EXCLUDING the heat-kernel prefactor ``pref·
    K(Bcal,x)`` (applied by the caller).  Method (the polynomial-fit route):

      1. ``P(q) = ⟨F(ℓ,q)⟩_ℓ`` is a polynomial in ``q`` (the ℓ-average of a
         polynomial form factor) of degree ≤ ``q_deg`` (= total degree of F).
         Recover it by interpolating the EXISTING ℓ-Gauss–Hermite average
         (:func:`_formfactor_average`) at ``q_deg+1`` real q-nodes.
      2. The q→x transform is then analytic: ``∫dq/2π e^{iqx} q^n e^{−Bcal q²} =
         K(Bcal,x)·E[(u+ix/2Bcal)^n]`` with ``u~N(0,1/2Bcal)`` (closed-form heat-kernel
         moments).  So ``FF(x) = Σ_n p_n E[(u+ix/2Bcal)^n]`` and the full diagram
         contribution is ``pref·K(Bcal,x)·FF(x)``.

    This replaces the n_q-point numerical FT with ``q_deg+1`` ℓ-GH evaluations —
    exact (no ringing / q_cut), and ~``n_q/(q_deg+1)`` fewer form-factor evals."""
    from math import comb
    P, L = Lam.shape[0], Lam.shape[1]
    xv = np.asarray(xs, dtype=float)
    out = np.zeros((P, xv.size), dtype=complex)
    if spatial_dim != 1:
        raise NotImplementedError(
            'Phase 2 analytic IFT (derivative vertices) is d=1 only so far; '
            'd≥2 transverse handling is Phase 3.')
    if not np.any(ok):
        return out
    Lamok, Nok, Qok = Lam[ok], N[ok], Q[ok]
    Qeff = (Qok - np.einsum('plj,plk->pjk', Nok,
                            np.linalg.solve(Lamok, Nok)))[:, 0, 0]   # (P',)
    Bcal = D * Qeff
    good = Bcal > 1e-300
    if not np.any(good):
        return out
    Lamg, Ng = Lamok[good], Nok[good]
    Bcal_g = Bcal[good]                                              # (Pg,)
    Pg = Lamg.shape[0]

    # PRINCIPLED route — the joint-(ℓ,q)-Gaussian moment (Case C): one pass per
    # diagram, NO q-node loop / NO GH grid.  ℓ̄=−Lam⁻¹N·q gives a=ℓ̄/q; Σ=(2D·Lam)⁻¹.
    # (Falls back to the polynomial fit below if the moment callable is absent.)
    moment_x = getattr(formfactor, 'moment_x', None)
    if moment_x is not None:
        a = -np.linalg.solve(Lamg, Ng)[:, :, 0]                # (Pg, L): ℓ̄ = a·q
        Sg = np.linalg.inv(Lamg) / (2.0 * D)                   # (Pg, L, L): Σ
        out_good = np.zeros((Lamok.shape[0], xv.size), dtype=complex)
        out_good[good] = moment_x(a, Sg, Bcal_g, xv)             # (Pg, n_x)
        out[ok] = out_good
        return out

    # 1. interpolate P(q)=⟨F⟩_ℓ from (q_deg+1) real nodes — scaled (t=q/qsc) for
    #    a well-conditioned Vandermonde.  EXACT (P is a polynomial of degree q_deg).
    n_nodes = int(q_deg) + 1
    qsc = 1.0 / float(np.sqrt(np.median(Bcal_g)))                # ~ Gaussian q-width
    tnodes = 0.35 + 2.3 * (0.5 - 0.5 * np.cos(
        np.pi * (np.arange(n_nodes) + 0.5) / n_nodes))       # ~Chebyshev in (0,~3)
    qnodes = qsc * tnodes
    Fbar = np.empty((Pg, n_nodes), dtype=complex)
    okg = np.ones(Pg, dtype=bool)
    for j in range(n_nodes):
        Fbar[:, j] = _formfactor_average(formfactor, Lamg, Ng, [float(qnodes[j])],
                                         D, okg, gh_order, spatial_dim)
    Vt = np.vander(tnodes, n_nodes, increasing=True)         # in t (well-cond.)
    ptil = np.linalg.solve(Vt, Fbar.T).T                     # (Pg, n_nodes): coeffs in t
    pcoef = ptil / (qsc ** np.arange(n_nodes))[None, :]      # back to q: p_n = p̃_n/qsc^n

    # 2. heat-kernel q-moments E_n(x) = E[(u+ix/2Bcal)^n], u~N(0, 1/2Bcal).
    c = 1j * xv[None, :] / (2.0 * Bcal_g[:, None])               # (Pg, n_x): ix/2Bcal
    sig2 = 1.0 / (2.0 * Bcal_g)                                  # (Pg,): Var(u)=1/2Bcal
    FF = np.zeros((Pg, xv.size), dtype=complex)
    for n in range(n_nodes):
        En = np.zeros((Pg, xv.size), dtype=complex)
        for k in range(0, n + 1, 2):                         # E[u^k]=σ^k(k-1)!! (even k)
            df = 1.0
            kk = k - 1
            while kk > 0:
                df *= kk
                kk -= 2
            En += comb(n, k) * c ** (n - k) * (sig2[:, None] ** (k // 2)) * df
        FF += pcoef[:, n][:, None] * En
    out_good = np.zeros((Lamok.shape[0], xv.size), dtype=complex)
    out_good[good] = FF
    out[ok] = out_good
    return out


def external_times_2pt(descr, tau):
    """For a 2-point correlator: leaf 0 at time 0, leaf 1 at time ``τ``."""
    legs = descr.external_legs
    if len(legs) != 2:
        raise ValueError(f"external_times_2pt expects k=2 (2 leaves); got {legs}.")
    return {legs[0]: 0.0, legs[1]: float(tau)}


def _gl_on(lower, upper, n):
    """Gauss–Legendre nodes/weights on ``[lower, upper]`` (broadcastable arrays),
    √-concentrated toward ``upper`` (where the retarded integrand peaks): for each
    point, ``n`` nodes ``t = upper − (upper−lower)·v²`` (``v∈[0,1]``).  Returns
    ``(t_nodes, t_w)`` each of shape ``lower.shape + (n,)``."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    v = 0.5 * (xg + 1.0)
    span = (upper - lower)[..., None]
    t = upper[..., None] - span * (v * v)
    w = span * (wg * v)                                      # Jacobian 2·span·v · ½
    return t, w


def _diagram_bessel_xs(a, b, edges, internal, idx, internal_R, external_times,
                       xs, mu, D, spatial_dim, formfactor, N, seed):
    """``method='bessel'`` — the radial-Bessel-K × angular-MC backend for `δC(x)`.

    Reparametrize each causal-region point `(u_v=−t_v, σ_e) = λ·ŝ`, `ŝ` on the
    `(n−1)`-simplex (`n=n_V+n_C`).  The Symanzik polynomials are homogeneous
    (`U→λ^L Û`, `F→λ^{L+1}F̂`, `W→λŴ`), so the radial `λ`-integral is EXACTLY a
    modified Bessel function `∫₀^∞ λ^P e^{−aλ−c/λ}dλ = 2(c/a)^{(P+1)/2}K_{P+1}(2√(ac))`
    (`a=μŴ`, `c=|x|²Û/4DF̂`, `P=n−1−(L+1)d/2`).  Only the smooth angular simplex is
    sampled — Dirichlet(1,…,1) with **causal poset rejection** for the internal R
    edges; the measure is `1/((n−1)!·N)`.  The radial reduction does the `det Lam→0`
    (degenerate-loop) direction analytically, regularizing what breaks pure MC.

    Plain (`formfactor=None` / no `moment_bessel`): a single Bessel-K.  Derivative
    (`ff.moment_bessel`): the form-factor moment is `M_F(λ)=Σ_m λ^{−m}EF_m`, so the
    radial sum is `Σ_m EF_m·K(P−m)`.  Returns `(n_x,)` real.  `x=0` (equal point) is
    UV-sensitive (divergent term-by-term); only the convergent part is kept."""
    from math import factorial as _fact
    from scipy.special import kv as _kv, gamma as _gamma
    n_V = len(internal)
    n_C = sum(1 for e in edges if e.kind == 'C')
    L = a.shape[1]
    n = n_V + n_C
    xs = np.asarray(xs, dtype=float)
    total = np.zeros(xs.size)
    if n == 0 or L == 0:
        return total                                        # tree / no loop scale
    rng = np.random.default_rng(seed)
    Es = rng.standard_exponential((int(N), n))
    s = Es / Es.sum(1, keepdims=True)                       # Dirichlet(1..1) on the simplex
    tvals = {leaf: np.full(int(N), tt) for leaf, tt in external_times.items()}
    for k, v in enumerate(internal):
        tvals[v] = -s[:, k]                                 # t_v = −ŝ_v  (in the past)
    sig = [s[:, n_V + c] for c in range(n_C)]
    w = np.empty((int(N), len(edges)))
    valid = np.ones(int(N), dtype=bool)
    ci = 0
    for ei, e in enumerate(edges):
        tu, tv = tvals[e.u], tvals[e.v]
        if e.kind == 'R':
            dd = tv - tu
            w[:, ei] = dd
            if (e.u in idx) and (e.v in idx):
                valid &= (dd >= 0.0)                        # causal poset (R-edge ≥ 0)
            else:
                w[:, ei] = np.maximum(dd, 1e-15)
        else:
            w[:, ei] = np.abs(tu - tv) + sig[ci]
            ci += 1
    wv = w[valid]
    if wv.shape[0] == 0:
        return total
    _pref, Bcal_k, ok, Lam, Nn, Q = _symanzik_kernel_batch(
        a, b, wv, D, spatial_dim, return_gaussian=True)
    if Lam is None or not np.any(ok & (Bcal_k > 1e-300)):
        return total
    good = ok & (Bcal_k > 1e-300)
    Lamg, Ng, Qg, wg = Lam[good], Nn[good], Q[good], wv[good]
    Uhat = np.linalg.det(Lamg)
    Qeff = (Qg - np.einsum('plj,plk->pjk', Ng,
                           np.linalg.solve(Lamg, Ng)))[:, 0, 0]
    Fhat = Uhat * Qeff
    What = wg.sum(1)
    okF = (Uhat > 1e-300) & (np.real(Fhat) > 1e-300) & (What > 0)
    if not np.any(okF):
        return total
    Lamg, Ng = Lamg[okF], Ng[okF]
    Uhat, Fhat, What = Uhat[okF], Fhat[okF], What[okF]
    Pg = Lamg.shape[0]
    c0 = ((4.0 * math.pi * D) ** (-0.5 * L * spatial_dim)
          * (4.0 * math.pi * D * Fhat) ** (-0.5 * spatial_dim))
    aa = mu * What
    Pp = n - 1 - (L + 1) * spatial_dim / 2.0
    norm = _fact(n - 1) * int(N)
    mom_b = getattr(formfactor, 'moment_bessel', None) if formfactor is not None else None
    if formfactor is not None and mom_b is None:
        raise NotImplementedError(
            "the Bessel backend's λ-graded moment is built for d=1 derivative "
            "vertices only (ff.moment_bessel is None — e.g. a d≥2 derivative "
            "vertex); d≥2 derivative analytic IFT is Phase 3.  Use method='grid' "
            "(numerical FT) for this case.")
    if mom_b is not None:                                   # derivative vertex: Σ_m EF_m λ^{−m}
        ahat = -np.linalg.solve(Lamg, Ng)[:, :, 0]
        Shat = np.linalg.inv(Lamg) / (2.0 * D)
        Bhat = D * Fhat / Uhat
        powers, g = mom_b(ahat, Shat, Bhat, xs)            # (n_m,), (n_m, Pg, n_x)
    else:                                                   # plain: single Bessel-K
        powers = np.array([0.0])
        g = np.ones((1, Pg, xs.size), dtype=complex)
    for ix in range(xs.size):
        x = xs[ix]
        acc = np.zeros(Pg, dtype=complex)
        if x == 0.0:                                        # c=0 → Γ; drop UV-divergent terms
            for im in range(powers.size):
                P1 = Pp - powers[im] + 1.0
                if P1 > 0:
                    acc += g[im, :, ix] * _gamma(P1) / aa ** P1
        else:
            cc = x * x * Uhat / (4.0 * D * Fhat)
            z = 2.0 * np.sqrt(aa * cc)
            for im in range(powers.size):
                P1 = Pp - powers[im] + 1.0
                acc += g[im, :, ix] * 2.0 * (cc / aa) ** (P1 / 2.0) * _kv(P1, z)
        total[ix] = float(np.real(np.sum(c0 * acc)))
    return total / norm


def diagram_kinematic(descr, q_vec, external_times, mu, D, spatial_dim=1,
                      W=None, n_t=22, n_s=24, formfactor=None, gh_order=6,
                      xs=None, method='grid', mc_n=1000000, mc_seed=0):
    """The kinematic (unit-amplitude, no couplings) full-diagram integral
    ``∫ ∏dt_v ∏dσ_e 𝟙(θ) e^{−μΣw} MomFactor`` by **causal-chamber quadrature**.

    The retarded ``θ``'s are turned into the integration LIMITS (not a mask),
    so the integrand is SMOOTH within each chamber — every ``|Δt|`` sign is
    fixed by the ordering, so there are no cusps and the quadrature converges
    fast.  Internal vertex times are integrated chamber-by-chamber (the orderings
    from :func:`causal_chambers`, nested latest→earliest with each level bounded
    by the next-later time and the external retarded legs), Gauss–Legendre per
    level (√-concentrated at the upper bound); correlation Schwinger params
    ``σ_e∈[0,∞)`` by Gauss–**Laguerre** (which integrates ``e^{−μσ}`` exactly).
    The Symanzik reduction is batched over the whole grid.

    ``q_vec`` : external momenta (length ``n_ext=k−1``; ``[q]`` for ``k=2``).
    Returns a float.  Grid size ``≈ n_t^{n_V}·n_s^{n_C}`` per chamber."""
    edges = list(descr.edges)
    internal = list(descr.internal_vertices)
    n_V = len(internal)
    idx = {v: i for i, v in enumerate(internal)}
    n_C = sum(1 for e in edges if e.kind == 'C')
    a = np.array([e.a for e in edges], dtype=float).reshape(len(edges), -1)
    b = np.array([e.b for e in edges], dtype=float).reshape(len(edges), -1)

    if W is None:
        W = 22.0 / mu
    ext_t = list(external_times.values())
    me, mn = max(ext_t), min(ext_t)
    lo, hi = mn - W, me + 3.0 / mu

    # retarded structure on the internal vertices: internal→internal R edges give
    # the ordering poset; R edges to/from a leaf give a fixed-time scalar bound.
    internal_R = []
    s_up = [hi] * n_V
    s_lo = [lo] * n_V
    for e in edges:
        if e.kind != 'R':
            continue
        ui, vi = e.u in idx, e.v in idx
        if ui and vi:
            internal_R.append((idx[e.u], idx[e.v]))          # t_{e.v} > t_{e.u}
        elif ui:                                             # e.u internal, leaf head
            s_up[idx[e.u]] = min(s_up[idx[e.u]], external_times[e.v])
        elif vi:                                             # leaf tail, e.v internal
            s_lo[idx[e.v]] = max(s_lo[idx[e.v]], external_times[e.u])

    # Correlation Schwinger param σ_e∈[0,∞): substitute σ = s_cap·v² (v∈[0,1]) so
    # the nodes CONCENTRATE near σ=0 — that resolves the integrable U^{−d/2}∼σ^{−d/2}
    # singularity of a self-loop (U=σ) that plain Gauss–Laguerre under-resolves.
    # The e^{−μσ} weight is folded into ``s_w``; ``mu_resid`` excludes σ.
    s_cap = 32.0 / mu
    xv, wv = np.polynomial.legendre.leggauss(n_s)
    vv = 0.5 * (xv + 1.0)
    s_nodes = s_cap * vv * vv
    s_w = wv * s_cap * vv * np.exp(-mu * s_nodes)            # w·s_cap·v·e^{−μσ}
    # the correlation Schwinger grid (n_s^{n_C} points), shared across chambers
    if n_C > 0:
        sg = np.meshgrid(*([s_nodes] * n_C), indexing='ij')
        swg = np.meshgrid(*([s_w] * n_C), indexing='ij')
        s_flat = [g.ravel() for g in sg]
        s_wflat = np.ones(s_flat[0].size)
        for g in swg:
            s_wflat = s_wflat * g.ravel()
    else:
        s_flat, s_wflat = [], np.array([1.0])
    Ps = s_wflat.size

    # ANALYTIC spatial IFT (xs given): accumulate the heat kernel over the output
    # grid instead of evaluating MomFactor at a single q — Σ_chambers ∫dw runs
    # ONCE, the x-dependence is analytic.  Phase 1 = plain vertices (no form
    # factor); the derivative-vertex joint-(ℓ,q) case is Phase 2.
    xs_arr = None if xs is None else np.asarray(xs, dtype=float)
    total = np.zeros(len(xs_arr)) if xs_arr is not None else 0.0
    if method == 'bessel' and xs_arr is not None:           # ── radial-Bessel × angular-MC
        if b.shape[1] > 1:
            raise NotImplementedError(
                "method='bessel' (radial Bessel-K × angular MC) is k=2 only "
                f"(single |x|); got n_ext={b.shape[1]}.  Use method='grid' "
                "(multivariate analytic IFT) for k>=3.")
        return _diagram_bessel_xs(
            a, b, edges, internal, idx, internal_R, external_times, xs_arr,
            mu, D, spatial_dim, formfactor, int(mc_n), mc_seed)
    chambers = causal_chambers(n_V, internal_R) if n_V else [()]
    _use_mc = method == 'mc' and (n_V + n_C) > 0
    _rng = np.random.default_rng(mc_seed) if _use_mc else None
    for order in chambers:
        if _use_mc:
            # ── Monte-Carlo: importance-sample the (n_V+n_C)-D chamber/Schwinger
            #    integral (P=mc_n random points, NOT n_t^{n_V}·n_s^{n_C}) — bounded
            #    memory, O(1/√N).  Internal times via nested Exp(μ) gaps (matching
            #    the retarded poset bounds), correlation σ's ~ Exp(μ); each σ-edge's
            #    e^{−μσ} is consumed by its proposal.  (Validated for PLAIN vertices;
            #    derivative-vertex form factors are biased — det Lam→0 singularity.)
            P = int(mc_n)
            placed = {}
            later = None
            Sgap = np.zeros(P)
            for vi in reversed(order):
                upper = (np.full(P, s_up[vi]) if later is None
                         else np.minimum(s_up[vi], later))
                g = -np.log(_rng.random(P)) / mu              # gap ~ Exp(μ)
                placed[vi] = upper - g
                later = placed[vi]
                Sgap += g
            tvals = {leaf: np.full(P, tt) for leaf, tt in external_times.items()}
            for v in internal:
                tvals[v] = placed[idx[v]]
            sig = [(-np.log(_rng.random(P)) / mu) for _ in range(n_C)]
        else:
            # ── deterministic causal-chamber PRODUCT GRID (the validated path) ──
            # nested internal-time grid (latest→earliest; each level bounded above
            # by the next-later time and its external scalar bound), × the σ-grid.
            placed = {}                                       # vertex idx → (Pt,) array
            wt = np.array([1.0])
            later = None
            for vi in reversed(order):
                Pt = wt.size
                upper = np.full(Pt, s_up[vi]) if later is None \
                    else np.minimum(s_up[vi], later)
                lower = np.full(Pt, s_lo[vi])
                tnode, wnode = _gl_on(lower, upper, n_t)       # (Pt, n_t)
                for k in placed:
                    placed[k] = np.repeat(placed[k], n_t)
                placed[vi] = tnode.ravel()
                wt = (wt[:, None] * wnode).ravel()
                later = placed[vi]
            Pt = wt.size
            P = Pt * Ps
            tvals = {leaf: np.full(P, tt) for leaf, tt in external_times.items()}
            for i, v in enumerate(internal):
                tvals[v] = np.repeat(placed[idx[v]], Ps)
            sig = [np.tile(sf, Pt) for sf in s_flat]
            wfull = np.repeat(wt, Ps) * np.tile(s_wflat, Pt)

        # edge weights + Symanzik + e^{−μΣw} (SHARED by grid and MC)
        w_batch = np.empty((P, len(edges)))
        mu_resid = np.zeros(P)
        ci = 0
        for ei, e in enumerate(edges):
            tu, tv = tvals[e.u], tvals[e.v]
            if e.kind == 'R':
                w_batch[:, ei] = np.maximum(tv - tu, 1e-12)
                mu_resid += np.maximum(tv - tu, 0.0)
            else:
                dt = np.abs(tu - tv)
                w_batch[:, ei] = dt + sig[ci]
                mu_resid += dt
                ci += 1
        # per-sample amplitude weight: grid = wfull·e^{−μ·mu_resid}; MC = the
        # importance weight e^{−μ(mu_resid−Σgap)}/(μ^{n_V+n_C}·N) (each σ's e^{−μσ}
        # cancels against its Exp(μ) proposal; ÷N is the MC mean).
        amp = (np.exp(-mu * (mu_resid - Sgap)) / (mu ** (n_V + n_C) * P)
               if _use_mc else wfull * np.exp(-mu * mu_resid))

        if xs_arr is not None:                            # ── analytic heat-kernel IFT
            if formfactor is None:                        # Phase 1: plain → pure heat kernel
                pref, Bcal_k, okk = _symanzik_kernel_batch(a, b, w_batch, D, spatial_dim)
                # degenerate-sample filter: B>0 (scalar) / det B>0 (matrix)
                _Bpos = (Bcal_k > 1e-300) if Bcal_k.ndim == 1 \
                    else (np.linalg.det(Bcal_k) > 1e-300)
                good = okk & _Bpos
                if np.any(good):
                    wamp = (amp * pref)[good]
                    hk = _heat_kernel_x_general(Bcal_k[good], xs_arr, spatial_dim)
                    total = total + np.einsum('p,px->x', wamp, hk)
            else:                                         # Phase 2: derivative → heat kernel × form-factor moments
                pref, Bcal_k, okk, Lamb, Nb, Qb = _symanzik_kernel_batch(
                    a, b, w_batch, D, spatial_dim, return_gaussian=True)
                if Bcal_k.ndim != 1:
                    raise NotImplementedError(
                        'derivative-vertex analytic IFT (form-factor moments) '
                        f'is k=2 only so far; got n_ext={Bcal_k.shape[1]}.  '
                        'Use the q-path (numerical FT) for k>=3 derivative '
                        'vertices.')
                good = (okk & (Bcal_k > 1e-300)) if Lamb is not None \
                    else np.zeros(len(pref), dtype=bool)
                if np.any(good):
                    Bcal_g = Bcal_k[good]
                    wamp = (amp * pref)[good]
                    hk = ((4.0 * math.pi * Bcal_g)[:, None] ** (-0.5 * spatial_dim)
                          * np.exp(-(xs_arr[None, :] ** 2) / (4.0 * Bcal_g[:, None])))
                    qdeg = getattr(formfactor, 'q_poly_deg', None) or 8
                    eff_gh = getattr(formfactor, 'gh_order_needed', None) or gh_order
                    FF = _formfactor_average_x(
                        formfactor, Lamb[good], Nb[good], Qb[good], D,
                        np.ones(int(np.sum(good)), dtype=bool), xs_arr,
                        spatial_dim=spatial_dim, gh_order=eff_gh, q_deg=qdeg)
                    total = total + np.einsum('p,px,px->x', wamp, hk, FF)
            continue                                      # next chamber (skip the q-eval)
        if formfactor is None:
            momfac = _momentum_factor_batch(a, b, w_batch, q_vec, D, spatial_dim)
        else:
            # derivative-vertex form factor F(ℓ,q): the loop integral becomes
            # MomFactor·⟨F⟩, ⟨F⟩ a Gauss–Hermite average (exact for the polynomial
            # form factor) over the loop-momentum Gaussian — d-dim (the d spatial
            # components are independent, q on axis 0; transverse moments).
            momfac, Lamb, Nb, okb = _momentum_factor_batch(
                a, b, w_batch, q_vec, D, spatial_dim, return_gaussian=True)
            if Lamb is not None:                               # L>=1 (loop diagram)
                # The polynomial form factor needs only its minimal exact GH order
                # per variable (the d≥2 grid is gh_order^{L·d} — a big saving).
                eff_gh = getattr(formfactor, 'gh_order_needed', None) or gh_order
                momfac = momfac * _formfactor_average(
                    formfactor, Lamb, Nb, q_vec, D, okb, eff_gh,
                    spatial_dim=spatial_dim)
        total += np.sum(amp * momfac)
    # `formfactor=None` → real (unchanged float return); a derivative/∇ form
    # factor (∂_x→ik) can be complex per diagram (e.g. odd # of ∂'s), so return
    # complex — the physical C(q,τ) is real and the imaginary parts cancel in the
    # diagram sum / are dropped at the real-space output.
    if xs_arr is not None:                                # analytic IFT → (n_x,) real
        return np.real(total)
    return complex(total) if formfactor is not None else float(np.real(total))


def spectral_rows(descr):
    """The expanded ROW list for the coupled-field (Dyson 3c) kinematic.

    Each ``R`` edge is one retarded segment (one row).  Each ``C`` edge is
    **two glued retarded segments** sharing one Schwinger ``σ`` (the noise-
    source time integrated out): with endpoint times ``t_u, t_v`` and
    ``s = min(t_u,t_v) − σ`` the half durations are

        w_u = t_u − s = relu(t_u − t_v) + σ        (mass m_u),
        w_v = t_v − s = relu(t_v − t_u) + σ        (mass m_v),

    each carrying the SAME routed momentum (rows duplicated), so the edge's
    heat-kernel weight is ``D(|Δt|+2σ)`` and the σ integral produces the
    Lyapunov denominator ``1/(m_u+m_v+2D|k|²)``.  Uniform-mass check:
    ``∫dσ e^{−m(|Δt|+2σ)} = e^{−m|Δt|}/(2m)`` — exactly ``2^{−1}`` per ``C``
    edge of the single-field one-segment convention, so a coupled diagram is
    ``pv·Σ_assign W·I_spec`` with **no** ``2^{−n_C}`` (the enumeration ``pv``
    keeps its ``2T``-convention noise coefficients).

    Returns ``[(edge_index, edge, half)]`` with ``half ∈ {'R','Cu','Cv'}``.
    """
    rows = []
    for ei, e in enumerate(descr.edges):
        if e.kind == 'R':
            rows.append((ei, e, 'R'))
        else:
            rows.append((ei, e, 'Cu'))
            rows.append((ei, e, 'Cv'))
    return rows


def diagram_kinematic_spectral(descr, q_vec, external_times, mass_table, D,
                               spatial_dim=1, W=None, n_t=22, n_s=24, xs=None,
                               mu_scale=None, power_table=None, insert_row=None):
    """Coupled-field kinematic integral with PER-SEGMENT masses (Dyson 3c).

    The spectral-assignment companion of :func:`diagram_kinematic`: same
    causal-chamber quadrature, same Symanzik momentum reduction, but the mass
    factor is ``∏_rows e^{−m_r w_r}`` over the :func:`spectral_rows` expansion
    (R edges one row; C edges two glued retarded segments sharing a σ), with a
    **batch of mass assignments** evaluated against ONE shared quadrature/
    Symanzik pass — the ``N_modes^{labels}`` spectral sum costs almost nothing
    beyond the single-mass integral.

    Parameters
    ----------
    mass_table : (n_rows, n_assign) complex array
        Column ``j`` = the per-row masses ``m_α`` of spectral assignment ``j``
        (rows ordered as :func:`spectral_rows`).  Complex eigenvalues are
        allowed (conjugate pairs; the driver's weighted sum is real).
    xs : None | array
        ``None`` → return ``(n_assign,)`` complex values at external ``q_vec``;
        array → analytic heat-kernel IFT, return ``(n_assign, n_x)`` complex.
    power_table : None | (n_rows, n_assign) int array  (Dyson loop dressing)
        Per-row polynomial powers κ: amplitude ``∏_r w_r^{κ_r} e^{−m_r w_r}``
        — the CONFLUENT dressed-segment form ``w·e^{−mw}`` (equal-eigenvalue
        Duhamel string) that partial fractions cannot produce.
    insert_row : None | int  (Dyson loop dressing, order n=1)
        Multiply the momentum integral by ``(−|k_r|²)`` for this row's routed
        momentum — evaluated EXACTLY via the derivative identity
        ``(−k_r²)·e^{−D w_r k_r²} = (1/D)·∂/∂w_r`` applied to the closed-form
        Symanzik factors:  ``∂U/∂w_r = U·g_r`` (rank-1, ``g_r=a_rᵀΛ⁻¹a_r``),
        ``∂𝓑/∂w_r = D·(b_r − a_rᵀΛ⁻¹N)²``, so

          xs path: factor = (1/D)[−(d/2)·g_r − ∂𝓑_r·(d/(2𝓑) − x²/(4𝓑²))·D⁻¹·D]
          q  path: factor = (1/D)[−(d/2)·g_r − q²·∂𝓑_r]

        (the heat kernel's 𝓑-derivative gives the x-dependent term).  This is
        the B26 ``(−|k|²)^n`` insertion at ``n=1`` — the leading O(𝒟̂) loop
        dressing.  Higher ``n`` is not implemented here (tree-level dressing
        supports all orders via ``dyson_dressing``).

    Plain vertices only (no form factor), deterministic grid only — the
    coupled v1 surface.  The single-field/diagonal path is untouched
    (:func:`diagram_kinematic`); this function is additive.
    """
    edges = list(descr.edges)
    internal = list(descr.internal_vertices)
    n_V = len(internal)
    idx = {v: i for i, v in enumerate(internal)}
    n_C = sum(1 for e in edges if e.kind == 'C')
    rows = spectral_rows(descr)
    n_rows = len(rows)
    a = np.array([rows[r][1].a for r in range(n_rows)],
                 dtype=float).reshape(n_rows, -1)
    b = np.array([rows[r][1].b for r in range(n_rows)],
                 dtype=float).reshape(n_rows, -1)
    mass_table = np.asarray(mass_table, dtype=complex)
    if mass_table.ndim == 1:
        mass_table = mass_table[:, None]
    if mass_table.shape[0] != n_rows:
        raise ValueError(
            f'diagram_kinematic_spectral: mass_table has {mass_table.shape[0]} '
            f'rows, expected {n_rows} (= n_R + 2·n_C segments).')
    n_assign = mass_table.shape[1]
    if power_table is not None:
        power_table = np.asarray(power_table, dtype=float)
        if power_table.ndim == 1:
            power_table = power_table[:, None]
        if power_table.shape != mass_table.shape:
            raise ValueError(
                f'power_table shape {power_table.shape} != mass_table '
                f'{mass_table.shape}.')
    if insert_row is not None and not (0 <= int(insert_row) < n_rows):
        raise ValueError(f'insert_row {insert_row} out of range (n_rows={n_rows}).')
    re_min = float(np.min(mass_table.real))
    if not re_min > 0.0:
        raise ValueError(
            f'diagram_kinematic_spectral: all segment masses need Re m > 0 '
            f'(stability); got min Re m = {re_min}.')
    if mu_scale is None:
        mu_scale = re_min                              # slowest decay sets scales
    mu_scale = float(min(mu_scale, re_min))            # never narrower than needed

    if W is None:
        W = 22.0 / mu_scale
    ext_t = list(external_times.values())
    me, mn = max(ext_t), min(ext_t)
    lo, hi = mn - W, me + 3.0 / mu_scale

    # retarded poset / leaf bounds — identical to diagram_kinematic
    internal_R = []
    s_up = [hi] * n_V
    s_lo = [lo] * n_V
    for e in edges:
        if e.kind != 'R':
            continue
        ui, vi = e.u in idx, e.v in idx
        if ui and vi:
            internal_R.append((idx[e.u], idx[e.v]))
        elif ui:
            s_up[idx[e.u]] = min(s_up[idx[e.u]], external_times[e.v])
        elif vi:
            s_lo[idx[e.v]] = max(s_lo[idx[e.v]], external_times[e.u])

    # σ grid: GEOMETRIC weights only (the per-segment e^{−mσ} decay lives in
    # the complex amplitude, not folded into the weights as in the uniform path)
    s_cap = 32.0 / mu_scale
    xv, wv = np.polynomial.legendre.leggauss(n_s)
    vv = 0.5 * (xv + 1.0)
    s_nodes = s_cap * vv * vv
    s_w = wv * s_cap * vv                              # dσ = 2·s_cap·v·(dv/2)
    if n_C > 0:
        sg = np.meshgrid(*([s_nodes] * n_C), indexing='ij')
        swg = np.meshgrid(*([s_w] * n_C), indexing='ij')
        s_flat = [g.ravel() for g in sg]
        s_wflat = np.ones(s_flat[0].size)
        for g in swg:
            s_wflat = s_wflat * g.ravel()
    else:
        s_flat, s_wflat = [], np.array([1.0])
    Ps = s_wflat.size

    xs_arr = None if xs is None else np.asarray(xs, dtype=float)
    L = a.shape[1]
    total = (np.zeros((n_assign, len(xs_arr)), dtype=complex)
             if xs_arr is not None else np.zeros(n_assign, dtype=complex))
    chambers = causal_chambers(n_V, internal_R) if n_V else [()]
    for order in chambers:
        # deterministic causal-chamber product grid (as diagram_kinematic)
        placed = {}
        wt = np.array([1.0])
        later = None
        for vi in reversed(order):
            Pt = wt.size
            upper = np.full(Pt, s_up[vi]) if later is None \
                else np.minimum(s_up[vi], later)
            lower = np.full(Pt, s_lo[vi])
            tnode, wnode = _gl_on(lower, upper, n_t)
            for k in placed:
                placed[k] = np.repeat(placed[k], n_t)
            placed[vi] = tnode.ravel()
            wt = (wt[:, None] * wnode).ravel()
            later = placed[vi]
        Pt = wt.size
        P = Pt * Ps
        tvals = {leaf: np.full(P, tt) for leaf, tt in external_times.items()}
        for i, v in enumerate(internal):
            tvals[v] = np.repeat(placed[idx[v]], Ps)
        sig = [np.tile(sf, Pt) for sf in s_flat]
        wfull = np.repeat(wt, Ps) * np.tile(s_wflat, Pt)

        # per-ROW durations (R segment; Cu/Cv glued halves share the C edge's σ)
        w_rows = np.empty((P, n_rows))
        ci_of = {}
        ci = 0
        for ei, e in enumerate(edges):
            if e.kind == 'C':
                ci_of[ei] = ci
                ci += 1
        for r, (ei, e, half) in enumerate(rows):
            tu, tv = tvals[e.u], tvals[e.v]
            if half == 'R':
                w_rows[:, r] = np.maximum(tv - tu, 1e-12)
            elif half == 'Cu':
                w_rows[:, r] = np.maximum(tu - tv, 0.0) + sig[ci_of[ei]]
            else:                                      # 'Cv'
                w_rows[:, r] = np.maximum(tv - tu, 0.0) + sig[ci_of[ei]]

        # batched mass amplitude: (P, n_assign), shared Symanzik per chamber;
        # power_table adds the confluent ∏ w_r^{κ_r} (via exp(κ·log w))
        expo = -(w_rows @ mass_table)
        if power_table is not None:
            expo = expo + np.log(np.maximum(w_rows, 1e-300)) @ power_table
        amp = wfull[:, None] * np.exp(expo)

        # n=1 momentum insertion (−|k_r|²): per-sample Gaussian pieces for row r
        def _ins_pieces(Lam_b, N_b, okm):
            """(g_r, dB_r) per sample: g=a_rᵀΛ⁻¹a_r, dB=D(b_r−a_rᵀΛ⁻¹N)²."""
            r = int(insert_row)
            br = float(b[r, 0])
            if L == 0 or Lam_b is None:
                return np.zeros(P), np.full(P, D * br * br)
            g = np.zeros(P)
            u = np.zeros(P)
            if np.any(okm):
                LamiA = np.linalg.solve(Lam_b[okm],
                                        np.broadcast_to(a[r], (int(np.sum(okm)), L))[..., None])[..., 0]
                g[okm] = LamiA @ a[r]
                u[okm] = np.einsum('pl,pl->p', LamiA, N_b[okm][:, :, 0])
            return g, D * (br - u) ** 2

        if xs_arr is not None:                         # analytic heat-kernel IFT
            if L == 0:                                 # tree: Bcal = D·Σ w_r b_r²
                Bcal_k = D * (w_rows @ (b[:, 0] ** 2))
                pref = np.ones(P)
                okk = Bcal_k > 1e-300
                Lamb = Nb = None
            elif insert_row is not None:
                pref, Bcal_k, okk, Lamb, Nb, _Qb = _symanzik_kernel_batch(
                    a, b, w_rows, D, spatial_dim, return_gaussian=True)
            else:
                pref, Bcal_k, okk = _symanzik_kernel_batch(a, b, w_rows, D,
                                                           spatial_dim)
                Lamb = Nb = None
            good = okk & (Bcal_k > 1e-300)
            if np.any(good):
                Bcal_g = Bcal_k[good]
                hk = ((4.0 * math.pi * Bcal_g)[:, None] ** (-0.5 * spatial_dim)
                      * np.exp(-(xs_arr[None, :] ** 2) / (4.0 * Bcal_g[:, None])))
                wamp = amp[good, :] * pref[good][:, None]
                if insert_row is not None:
                    g_r, dB_r = _ins_pieces(Lamb, Nb, okk)
                    d_ = float(spatial_dim)
                    fac = (-(0.5 * d_) * g_r[good, None] / D
                           - (dB_r[good, None] / D)
                           * (0.5 * d_ / Bcal_g[:, None]
                              - (xs_arr[None, :] ** 2) / (4.0 * Bcal_g[:, None] ** 2)))
                    total = total + np.einsum('pj,px,px->jx', wamp, hk, fac)
                else:
                    total = total + np.einsum('pj,px->jx', wamp, hk)
        else:
            if L == 0:                                 # tree: e^{−D q² Σ w_r b_r²}
                qv = float(q_vec[0])
                momfac = np.exp(-D * (qv * qv) * (w_rows @ (b[:, 0] ** 2)))
                Lam_b = N_b = None
                okm = np.ones(P, dtype=bool)
            elif insert_row is not None:
                momfac, Lam_b, N_b, okm = _momentum_factor_batch(
                    a, b, w_rows, q_vec, D, spatial_dim, return_gaussian=True)
            else:
                momfac = _momentum_factor_batch(a, b, w_rows, q_vec, D,
                                                spatial_dim)
            if insert_row is not None:
                g_r, dB_r = _ins_pieces(Lam_b, N_b, okm)
                d_ = float(spatial_dim)
                qv = float(q_vec[0])
                fac = (-(0.5 * d_) * g_r - (qv * qv) * dB_r) / D
                momfac = momfac * fac
            total = total + amp.T @ momfac
    return total


def diagram_value(descr, prefactor_val, q_vec, external_times, mu, D,
                  spatial_dim=1, **kw):
    """One diagram's contribution to the cumulant: ``2^{−n_C}·prefactor·kinematic``.

    ``prefactor_val`` is the enumeration ``𝒮(Γ)·prefactor`` evaluated at the
    params (couplings + noise amplitudes, e.g. ``2T`` for the tree, ``8T²g²`` /
    ``16T²g²`` for the bubbles); the ``2^{−n_C}`` converts the ``2T`` noise-vertex
    convention to the kinematic unit-amplitude ``C`` edges."""
    n_C = sum(1 for e in descr.edges if e.kind == 'C')
    kin = diagram_kinematic(descr, q_vec, external_times, mu, D,
                            spatial_dim=spatial_dim, **kw)
    return (2.0 ** (-n_C)) * float(prefactor_val) * kin


def _is_retarded_type(descr):
    """True iff the diagram's two external legs have DIFFERENT propagator kinds
    (one ``C``, one ``R``) — a **retarded** self-energy insertion, which dresses
    both the retarded and advanced sides of the line and so appears as the pair
    ``Σ_R(τ)+Σ_A(τ) = Γ(τ)+Γ(−τ)``.  A ``{R,R}`` (Keldysh) insertion is its own
    conjugate ⇒ a single ``Γ(τ)``."""
    kinds = sorted(e.kind for e in descr.edges if e.external)
    return kinds == ['C', 'R']


def diagram_correlator(descr, prefactor_val, q, tau, mu, D, spatial_dim=1, **kw):
    """One diagram's contribution to ``C(q,τ)``, with the retarded+advanced sum
    applied: ``Γ(τ)+Γ(−τ)`` for a retarded-type insertion (``{C,R}`` external
    legs), else ``Γ(τ)``."""
    et = external_times_2pt(descr, tau)
    val = diagram_value(descr, prefactor_val, [q], et, mu, D,
                        spatial_dim=spatial_dim, **kw)
    if _is_retarded_type(descr) and tau != 0.0:
        et_m = external_times_2pt(descr, -tau)
        val += diagram_value(descr, prefactor_val, [q], et_m, mu, D,
                             spatial_dim=spatial_dim, **kw)
    elif _is_retarded_type(descr):
        val *= 2.0                                           # τ=0: Γ(0)+Γ(0)
    return val


def correlator_2pt(descrs_prefactors, q, tau, mu, D, spatial_dim=1, **kw):
    """The connected 2-point cumulant ``C(q,τ) = Σ_Γ Γ(q,τ)`` — the sum over ALL
    enumerated diagrams (tree + every loop), each via the SAME full integral, with
    the retarded+advanced sum applied per diagram (:func:`diagram_correlator`).

    ``descrs_prefactors`` : iterable of ``(CStackDiagram, 𝒮(Γ)·prefactor value)``.
    Returns the momentum-space ``C(q,τ)`` (FT to position is done by the caller)."""
    total = 0.0
    for descr, pre in descrs_prefactors:
        if abs(float(pre)) < 1e-14:
            continue
        total += diagram_correlator(descr, pre, q, tau, mu, D,
                                    spatial_dim=spatial_dim, **kw)
    return total


# ─────────────────────── ANALYTIC heat-kernel IFT (Case A) ───────────────────────
# Real-space δC(x,τ) directly, via the per-Schwinger-sample heat kernel — no
# q-grid, no numerical q→x FT (no ringing, no n_q, no q_cut).  Plain (non-
# derivative) vertices only (Phase 1); derivative vertices are Phase 2.

def diagram_value_x(descr, prefactor_val, xs, external_times, mu, D,
                    spatial_dim=1, **kw):
    """Analytic-IFT analogue of :func:`diagram_value` — one diagram's REAL-SPACE
    δC(x) as a vector over ``xs`` (``2^{−n_C}·prefactor·kinematic_x``)."""
    n_C = sum(1 for e in descr.edges if e.kind == 'C')
    kin = diagram_kinematic(descr, [0.0], external_times, mu, D,
                            spatial_dim=spatial_dim, xs=xs, **kw)
    return (2.0 ** (-n_C)) * float(prefactor_val) * kin       # (n_x,) real


def diagram_correlator_x(descr, prefactor_val, xs, tau, mu, D, spatial_dim=1, **kw):
    """Analytic-IFT analogue of :func:`diagram_correlator` — δC(x,τ) (vector) with
    the retarded+advanced sum applied per diagram."""
    et = external_times_2pt(descr, tau)
    val = diagram_value_x(descr, prefactor_val, xs, et, mu, D,
                          spatial_dim=spatial_dim, **kw)
    if _is_retarded_type(descr) and tau != 0.0:
        et_m = external_times_2pt(descr, -tau)
        val = val + diagram_value_x(descr, prefactor_val, xs, et_m, mu, D,
                                    spatial_dim=spatial_dim, **kw)
    elif _is_retarded_type(descr):
        val = val * 2.0                                       # τ=0: Γ(0)+Γ(0)
    return val


def correlator_2pt_x(descrs_prefactors, xs, tau, mu, D, spatial_dim=1, **kw):
    """Analytic-IFT analogue of :func:`correlator_2pt` — the real-space δC(x,τ)
    summed over all diagrams via the heat-kernel IFT (no q-grid / numerical FT)."""
    xs_arr = np.asarray(xs, dtype=float)
    total = np.zeros(len(xs_arr))
    for descr, pre in descrs_prefactors:
        if abs(float(pre)) < 1e-14:
            continue
        total = total + diagram_correlator_x(descr, pre, xs_arr, tau, mu, D,
                                             spatial_dim=spatial_dim, **kw)
    return total


# ── General-k external-event evaluation (June 2026) ─────────────────────────

def field_respecting_mappings(point_fields, leaf_fields):
    """All bijections canonical-point-slot → leaf-position that respect the
    field type, as tuples ``m`` with ``m[j] = slot assigned to leaf j``.

    Mirrors the temporal ``_all_mappings`` enumeration in
    ``final_integral.integrate_diagram`` (there keyed slot→leaf; here we
    return the leaf-indexed inverse, which is what the kinematic assembly
    consumes).  Each mapping is one external Wick contraction; summing the
    kinematic over all of them counts every pinned-external diagram exactly
    ``external_wick_compensation(td)`` times (orbit–stabilizer), so the
    caller divides by that index — the SAME architecture validated to
    machine precision against the Boltzmann series at k ≤ 5 in the temporal
    pipeline (tests/test_all_k_boltzmann.py)."""
    import itertools as _it
    k = len(leaf_fields)
    if len(point_fields) != k:
        return [tuple(range(k))]
    slots_by_field = {}
    for s, f in enumerate(point_fields):
        slots_by_field.setdefault(f, []).append(s)
    leaves_by_field = {}
    for j, f in enumerate(leaf_fields):
        leaves_by_field.setdefault(f, []).append(j)
    if {f: len(v) for f, v in slots_by_field.items()} != \
            {f: len(v) for f, v in leaves_by_field.items()}:
        return [tuple(range(k))]
    mappings = [{}]
    for f in sorted(slots_by_field, key=str):
        slots = slots_by_field[f]
        lfs = leaves_by_field[f]
        new = []
        for m in mappings:
            for perm in _it.permutations(slots):
                nm = dict(m)
                for j, s in zip(lfs, perm):
                    nm[j] = s
                new.append(nm)
        mappings = new
    return [tuple(m[j] for j in range(k)) for m in mappings]


def diagram_correlator_pts(descr, prefactor_val, x_pts, t_pts, mu, D,
                           spatial_dim=1, mappings=None, comp=1, **kw):
    """One diagram's contribution to the k-point cumulant at general external
    EVENTS — the k-generic replacement for the 2-point ``Γ(τ)+Γ(−τ)``
    completion in :func:`diagram_correlator_x`.

    ``x_pts`` : ``(n_pts, k)`` (d=1) or ``(n_pts, k, d)`` absolute positions,
        one column per canonical external slot (the user's external_fields
        order).
    ``t_pts`` : ``(k,)`` absolute times per canonical slot (all ``n_pts``
        share the time configuration; vectorization is over positions).
    ``mappings`` : list of slot-assignment tuples from
        :func:`field_respecting_mappings` (``m[j]`` = slot on leaf j).
        Default = identity only (single pinned assignment — oracle use).
    ``comp`` : ``external_wick_compensation`` of the typed diagram.

    For each mapping the kinematic runs with leaf times ``t_pts[m[j]]`` and
    IFT conjugates ``X_j = x(m[j]) − x(m[k−1])`` (momentum conservation
    eliminated the last leaf's momentum in the ``b`` routing).  k=2 with both
    mappings and the right comp reproduces :func:`diagram_correlator_x`
    exactly: retarded-type insertions have comp=1 → Γ(τ)+Γ(−τ); symmetric
    insertions have comp=2 and a τ-even kinematic → Γ(τ).

    Returns ``(n_pts,)`` real."""
    legs = list(descr.external_legs)
    k = len(legs)
    x_arr = np.asarray(x_pts, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[None, :]
    t_arr = np.asarray(t_pts, dtype=float)
    if x_arr.shape[1] != k or t_arr.shape != (k,):
        raise ValueError(
            f'diagram_correlator_pts: expected x_pts (n_pts, {k}[, d]) and '
            f't_pts ({k},); got {x_arr.shape} and {t_arr.shape}.')
    if mappings is None:
        mappings = [tuple(range(k))]
    n_C = sum(1 for e in descr.edges if e.kind == 'C')
    total = np.zeros(x_arr.shape[0])
    # Group mappings that produce the SAME (leaf-times, X-conjugates)
    # configuration — e.g. permuting two slots that sit at identical
    # events — and evaluate each unique configuration once, weighted by
    # its multiplicity.  Pure optimization: the sum is unchanged.
    config_count = {}
    config_data = {}
    for m in mappings:
        et = {legs[j]: float(t_arr[m[j]]) for j in range(k)}
        # FT conjugates of q_syms[0..k−2]: X_j = x_{slot on leg j} − x_{slot on leg k−1}
        X = np.stack([x_arr[:, m[j]] - x_arr[:, m[k - 1]]
                      for j in range(k - 1)], axis=1)       # (n_pts, k−1[, d])
        key = (tuple(sorted(et.items())), X.tobytes())
        config_count[key] = config_count.get(key, 0) + 1
        config_data[key] = (et, X)
    for key, (et, X) in config_data.items():
        if k == 2 and X.ndim == 2:
            xs_eval = X[:, 0]                               # scalar path (k=2, d=1)
        else:
            xs_eval = X
        kin = diagram_kinematic(descr, [0.0] * (k - 1), et, mu, D,
                                spatial_dim=spatial_dim, xs=xs_eval, **kw)
        total = total + config_count[key] * np.asarray(kin, dtype=float)
    return (2.0 ** (-n_C)) * float(prefactor_val) * total / float(comp)

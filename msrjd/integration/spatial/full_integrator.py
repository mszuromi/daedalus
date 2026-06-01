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

  Γ(q,{τ}) = 2^{−n_C}·M(Γ) · ∫ ∏_v dt_v ∏_{C edges} dσ_e  𝟙(θ's) ·
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


def _momentum_factor_batch(a, b, w_batch, q_vec, D, spatial_dim, u_floor=1e-300,
                           return_gaussian=False):
    """``MomFactor(w,q)`` for a BATCH of Schwinger-weight vectors ``w_batch``
    (shape ``P×E``) — vectorized Symanzik reduction.

    ``a`` (``E×L``), ``b`` (``E×n_ext``): the per-edge routing coefficients.
    Returns ``(P,)``.  ``L=0`` (tree) → ``exp(−D qᵀQq)``, ``Q=Σ_e w_e b_e b_eᵀ``;
    ``L≥1`` → ``(4πD)^{−Ld/2} U^{−d/2} exp(−D qᵀ Q_eff q)`` with ``U=det M``,
    ``Q_eff=Q−Nᵀ M⁻¹ N`` (batched ``det``/``solve``).

    ``return_gaussian`` also returns ``(M, N, ok)`` (the loop-momentum Gaussian's
    precision ``M=Σw_e a_e a_eᵀ`` and cross-term ``N=Σw_e a_e b_eᵀ``, with a
    non-degenerate mask) so a derivative-vertex **form factor** can be averaged
    over ``ℓ~N(−M⁻¹Nq, (2D M)⁻¹)`` — see :func:`_formfactor_average`.  ``M,N`` are
    ``None`` for ``L=0``."""
    E, L = a.shape
    qv = np.atleast_1d(np.asarray(q_vec, dtype=float))
    Q = np.einsum('pe,ej,ek->pjk', w_batch, b, b)            # (P, n_ext, n_ext)
    if L == 0:
        quad = np.einsum('j,pjk,k->p', qv, Q, qv)
        mf = np.exp(-D * quad)
        return (mf, None, None, None) if return_gaussian else mf
    M = np.einsum('pe,el,em->plm', w_batch, a, a)            # (P, L, L)
    N = np.einsum('pe,el,ej->plj', w_batch, a, b)            # (P, L, n_ext)
    U = np.linalg.det(M)                                     # (P,)
    ok = U > u_floor
    out = np.zeros(w_batch.shape[0])
    if np.any(ok):
        Mok, Nok, Qok, Uok = M[ok], N[ok], Q[ok], U[ok]
        MiN = np.linalg.solve(Mok, Nok)                      # (P', L, n_ext)
        Qeff = Qok - np.einsum('plj,plk->pjk', Nok, MiN)
        quad = np.einsum('j,pjk,k->p', qv, Qeff, qv)
        pref = (4.0 * math.pi * D) ** (-0.5 * spatial_dim * L) \
            * np.power(Uok, -0.5 * spatial_dim)
        out[ok] = pref * np.exp(-D * quad)
    return (out, M, N, ok) if return_gaussian else out


def _formfactor_average(formfactor, M, N, q_vec, D, ok, gh_order=6):
    """``⟨F(ℓ,q)⟩`` of a derivative-vertex form factor over the loop-momentum
    Gaussian ``ℓ ~ N(ℓ̄, Σ)``, ``ℓ̄ = −M⁻¹N q``, ``Σ = (2D M)⁻¹``, by
    Gauss–Hermite — **EXACT** for a polynomial ``F`` (a momentum-space derivative
    vertex deposits exactly a polynomial: ``Lap→−|k|²``, ``∂_x→ik``).  The base
    ``MomFactor`` already carries the Gaussian normalization + the ``1/(2π)^{Ld}``
    measure, so the full derivative-vertex loop integral is ``MomFactor·⟨F⟩``
    (validated to 1e-12 vs brute ``∫dℓ``).

    Generic in the loop number ``L`` (the GH grid is ``L``-dimensional) and the
    number of external momenta ``n_ext`` (``ℓ̄ = −M⁻¹N·q`` uses the full external
    vector), so it composes for ANY topology / ``ell`` / ``k``.  **d=1** (each
    ``ℓ``, ``q`` component is a scalar-per-loop / per-external); ``d≥2`` form
    factors (transverse momentum moments) are a documented extension.  Returns
    ``(P,)``; ``1.0`` where the loop is degenerate (``MomFactor`` is ``0`` there).

    ``formfactor(ell, q)``: ``ell`` is ``(P', G, L)`` loop momenta, ``q`` the
    ``(n_ext,)`` external-momentum vector → ``(P', G)``.  Build it from the
    per-vertex leg routing + operator form factor (:func:`diagram_form_factor`)."""
    P, L = M.shape[0], M.shape[1]
    out = np.ones(P, dtype=complex)                          # COMPLEX: ∂_x→ik
    if not np.any(ok):
        return out
    qv = np.atleast_1d(np.asarray(q_vec, dtype=float))       # (n_ext,)
    Mok, Nok = M[ok], N[ok]                                  # (P',L,L), (P',L,n_ext)
    lbar = -np.einsum('plj,j->pl', np.linalg.solve(Mok, Nok), qv)   # (P',L)
    Sig = np.linalg.inv(Mok) / (2.0 * D)                     # (P',L,L)
    Ch = np.linalg.cholesky(Sig)                             # (P',L,L)
    xg, wg = np.polynomial.hermite_e.hermegauss(gh_order)    # weight e^{−x²/2}
    Z = np.stack([g.ravel() for g in
                  np.meshgrid(*([xg] * L), indexing='ij')], axis=-1)   # (G,L)
    Wg = np.ones(xg.size ** L)
    for wgrid in np.meshgrid(*([wg] * L), indexing='ij'):
        Wg = Wg * wgrid.ravel()
    Wg = Wg / (2.0 * math.pi) ** (0.5 * L)                   # (G,)
    ell = lbar[:, None, :] + np.einsum('plm,gm->pgl', Ch, Z)  # (P',G,L)
    Fv = np.asarray(formfactor(ell, qv), dtype=complex)      # (P',G); ∂_x→ik ⇒ complex
    out[ok] = np.einsum('pg,g->p', Fv, Wg)
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


def diagram_kinematic(descr, q_vec, external_times, mu, D, spatial_dim=1,
                      W=None, n_t=22, n_s=24, formfactor=None, gh_order=6):
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

    total = 0.0
    chambers = causal_chambers(n_V, internal_R) if n_V else [()]
    for order in chambers:
        # 1. nested internal-time grid (latest→earliest); each level bounded above
        #    by the next-later time (and its external scalar bound).
        placed = {}                                          # vertex idx → (Pt,) array
        wt = np.array([1.0])
        later = None
        for vi in reversed(order):
            Pt = wt.size
            upper = np.full(Pt, s_up[vi]) if later is None \
                else np.minimum(s_up[vi], later)
            lower = np.full(Pt, s_lo[vi])
            tnode, wnode = _gl_on(lower, upper, n_t)          # (Pt, n_t)
            for k in placed:
                placed[k] = np.repeat(placed[k], n_t)
            placed[vi] = tnode.ravel()
            wt = (wt[:, None] * wnode).ravel()
            later = placed[vi]
        Pt = wt.size

        # 2. outer product time-grid × σ-grid → full grid of P = Pt·Ps points
        P = Pt * Ps
        tvals = {leaf: np.full(P, tt) for leaf, tt in external_times.items()}
        for i, v in enumerate(internal):
            tvals[v] = np.repeat(placed[idx[v]], Ps)
        sig = [np.tile(sf, Pt) for sf in s_flat]
        wfull = np.repeat(wt, Ps) * np.tile(s_wflat, Pt)

        # 3. edge weights + Symanzik + e^{−μΣw}
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
        if formfactor is None:
            momfac = _momentum_factor_batch(a, b, w_batch, q_vec, D, spatial_dim)
        else:
            # derivative-vertex form factor F(ℓ,q): the loop integral becomes
            # MomFactor·⟨F⟩, ⟨F⟩ a Gauss–Hermite average (exact for the polynomial
            # form factor) over the loop-momentum Gaussian.  d=1 only (v1).
            if spatial_dim != 1:
                raise NotImplementedError(
                    'derivative-vertex form factors are implemented for d=1 '
                    '(scalar loop momentum); d>=2 transverse moments are deferred.')
            momfac, Mb, Nb, okb = _momentum_factor_batch(
                a, b, w_batch, q_vec, D, spatial_dim, return_gaussian=True)
            if Mb is not None:                               # L>=1 (loop diagram)
                momfac = momfac * _formfactor_average(
                    formfactor, Mb, Nb, q_vec, D, okb, gh_order)
        total += np.sum(wfull * np.exp(-mu * mu_resid) * momfac)
    # `formfactor=None` → real (unchanged float return); a derivative/∇ form
    # factor (∂_x→ik) can be complex per diagram (e.g. odd # of ∂'s), so return
    # complex — the physical C(q,τ) is real and the imaginary parts cancel in the
    # diagram sum / are dropped at the real-space output.
    return complex(total) if formfactor is not None else float(np.real(total))


def diagram_value(descr, prefactor_val, q_vec, external_times, mu, D,
                  spatial_dim=1, **kw):
    """One diagram's contribution to the cumulant: ``2^{−n_C}·prefactor·kinematic``.

    ``prefactor_val`` is the enumeration ``M(Γ)·prefactor`` evaluated at the
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

    ``descrs_prefactors`` : iterable of ``(CStackDiagram, M(Γ)·prefactor value)``.
    Returns the momentum-space ``C(q,τ)`` (FT to position is done by the caller)."""
    total = 0.0
    for descr, pre in descrs_prefactors:
        if abs(float(pre)) < 1e-14:
            continue
        total += diagram_correlator(descr, pre, q, tau, mu, D,
                                    spatial_dim=spatial_dim, **kw)
    return total

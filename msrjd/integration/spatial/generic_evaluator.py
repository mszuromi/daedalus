"""
⚠ ORACLE-ONLY — not on the production path. Superseded by ``full_integrator.py``
(see ``docs/spatial_pipeline.md``); reached only by its own test(s).  Kept as an
independent numerical cross-check — ``compute_cumulants`` does NOT use this module.

msrjd.integration.spatial.generic_evaluator
============================================
Backend C — **Phase 2 of the generic spatial loop pipeline**
(``docs/spatial_generic_pipeline_plan.md``): evaluate one
:class:`~msrjd.integration.spatial.diagram_descriptor.CStackDiagram`'s
contribution to the correlator, by the momentum-first route

    enumerate → map (Phase 1) → Symanzik ∫dᵈℓ (C0/C1) → causal-chamber ∫dt (C2).

This is the **one** evaluator every diagram goes through.  There is no
bubble/tadpole branch: the loop momentum couples (or not) to ``q`` purely through
the Symanzik polynomials, which are built mechanically from the edge list.

Two entry points (built/validated incrementally):
  * :func:`loop_self_energy` — Phase 2a: the amputated 2-vertex self-energy of a
    diagram's loop edges, via the validated
    :func:`~msrjd.integration.spatial.temporal_integrate.sigma_parametric`.  Used
    to confirm the descriptor → C-stack edge mapping reproduces the hand-coded
    ``bubble_edges`` oracle.
  * :func:`evaluate_diagram` — Phase 2b: the FULL diagram value ``C_Γ(q,τ)``
    (external legs included), the strict-fixed-order momentum-first replacement
    for ``compute_correction_td``.  *(under construction)*

Normalization: **kinematic only** — couplings, noise amplitudes and the
combinatorial ``𝒮(Γ)`` are the enumeration's ``scalar_prefactor`` and are applied
by the caller.  A bare ``C`` edge contributes its Schwinger ``e^{−m|Δt|}/m`` with
unit weight (no extra ``T``).
"""
from __future__ import annotations

import math

import numpy as np

from msrjd.integration.spatial.temporal_integrate import sigma_parametric


def loop_self_energy(descr, q, t, mu, D, T=1.0, spatial_dim=1, **quad):
    """Phase 2a — the amputated self-energy ``Σ(q,t)`` of ``descr``'s **loop**
    edges (the internal, non-external lines) via :func:`sigma_parametric`.

    Valid for a 2-vertex loop (a bubble): all loop edges span the same
    inter-vertex time ``t`` (one ordering chamber), which is exactly
    ``sigma_parametric``'s domain.  Returns ``Σ(q,t)`` (kinematic; the caller
    multiplies the enumeration ``𝒮(Γ)·prefactor``).  Diagrams whose loop has a
    self-loop (a tadpole) or >2 internal vertices are handled by the full
    :func:`evaluate_diagram` (Phase 2b), not here.
    """
    le = descr.loop_edges()
    selfloops = [e for e in le if e.u == e.v]
    if selfloops:
        raise ValueError(
            "loop_self_energy is the 2-vertex (bubble) helper; this diagram has a "
            "self-loop (tadpole) — use evaluate_diagram (Phase 2b).")
    edges = [(tuple(float(x) for x in e.a),
              tuple(float(x) for x in e.b), e.kind) for e in le]
    return sigma_parametric(edges, q, t, mu, D, T, spatial_dim=spatial_dim, **quad)


def sigma_grid_direct(descr, q, u_grid, mu, D, spatial_dim=1, n_l=2600,
                      L_cut=None, formfactor=None):
    """**Vectorized** kinematic self-energy ``σ_Γ(q, u)`` of a 2-vertex bubble's
    loop edges, on the whole ``u_grid`` at once, by a direct ``∫dᵈℓ`` (the
    descriptor-driven, generic analog of the bespoke ``loop_dyson._sigma_grids``).

    All loop edges of a 2-vertex bubble span the SAME inter-vertex time ``u``, so

        σ_Γ(q,u) = ∫dᵈℓ/(2π)ᵈ  F(q,ℓ) [∏_{C edges} 1/m_{k_e}] · e^{−(Σ_e m_{k_e})·u},
        k_e = a_e·ℓ + b_e·q,   m_{k_e} = μ + D|k_e|²   (kinematic; T=1).

    ``formfactor`` (Phase 6): an optional callable ``F(ℓ)`` (the product of the
    derivative-vertices' per-leg momentum factors, e.g. ``q²ℓ²`` for ∇²φ²) that
    multiplies the loop integrand; ``None`` ⇒ the plain bubble.  For d=1 ``ℓ`` is
    the line grid; for d≥2 it is the first (q-aligned) axis component.

    Returns ``σ`` of shape ``u_grid``.  Matches :func:`loop_self_energy`
    (``sigma_parametric``) pointwise but ~100× faster — this is the production
    σ.  d=1 uses a line grid; d≥2 a Cartesian grid truncated at ``L_cut``
    (Regime-1 cutoff)."""
    u_grid = np.asarray(u_grid, dtype=float)
    le = descr.loop_edges()
    if any(e.u == e.v for e in le):
        raise ValueError("sigma_grid_direct is for bubbles (no self-loop).")
    d = int(spatial_dim)
    if L_cut is None:
        L_cut = max(60.0, abs(q) + 40.0) if d == 1 else max(20.0, abs(q) + 15.0)
    if d == 1:
        lg = np.linspace(-L_cut, L_cut, n_l)
        dvol = lg[1] - lg[0]
        grids = (lg,)
    else:
        n_l = min(n_l, 110 if d == 2 else 60)
        axes = np.linspace(-L_cut, L_cut, n_l)
        grids = np.meshgrid(*([axes] * d), indexing='ij')
        dvol = (axes[1] - axes[0]) ** d
        grids = tuple(g.ravel() for g in grids)
    pref = dvol / (2.0 * math.pi) ** d

    msum = np.zeros_like(grids[0])
    cpref = np.ones_like(grids[0])
    for e in le:
        a0 = float(e.a[0])                          # L=1: single loop momentum
        b0 = float(e.b[0]) if e.b else 0.0          # single external q
        # k_e = a0·ℓ + b0·q  (ℓ along axis 0 carries the external q in d≥2)
        kx = a0 * grids[0] + b0 * q
        k2 = kx * kx + sum(a0 * a0 * g * g for g in grids[1:])
        m_e = mu + D * k2
        msum = msum + m_e
        if e.kind == 'C':
            cpref = cpref / m_e
    if formfactor is not None:                      # derivative-vertex F(ℓ)
        cpref = cpref * np.asarray(formfactor(grids[0]), dtype=float)
    E = np.exp(-np.outer(msum, u_grid))             # (n_grid, n_u)
    sig = (cpref[:, None] * E).sum(axis=0) * pref
    return sig


# ── the generic single-mode Dyson convolution (external-leg dressing) ──
# This is the strict-fixed-order first-order correction for a self-energy
# inserted into the tree correlator of a SINGLE mode ``(A, B, N)``
# (``m = A + B q²``, ``G_R⁰(t)=θ(t)e^{−mt}``, ``C⁰(t)=(N/m)e^{−m|t|}``):
#
#   retarded:  δC_R(τ) = (G_R⁰⊛Σ_R⊛C⁰)(τ) + (C⁰⊛Σ_A⊛G_A⁰)(τ)
#   Keldysh :  δC_K(τ) = (G_R⁰⊛Σ_K⊛G_A⁰)(τ)
#
# Model-independent: it depends ONLY on the tree mode (A,B,N) and the self-energy
# grid ``σ(a)`` — no theory-specific constants.  (This is the same convolution the
# now-retired bespoke ``loop_dyson`` did for the φ̃φ² bubble, generalized to any
# self-energy and parametrized by the mode; the small-``a`` power-law sliver is
# kept so equal-time / derivative-vertex self-energies integrate accurately.)


def _sigma_grid_axis(m, taus, n_floor=2000, t_max_cap=60.0):
    """The adaptive ``a``-grid resolving the convolution kernels (decay ``1/m``)
    and the τ-reach — the time axis on which ``σ(a)`` is tabulated."""
    tau_max = max((abs(float(t)) for t in taus), default=0.0)
    t_max = min(t_max_cap, max(2.0 * tau_max + 12.0 / m, 12.0 / m))
    n_t = int(min(max(n_floor, t_max * m * 50.0), 8000.0))
    return np.linspace(t_max / n_t, t_max, n_t)


def _dyson_retarded(ag, sR, m, N, taus):
    """``δC_R(τ) = Term1(τ)+Term1(−τ)``, ``Term1=(N/m)∫₀^∞ σ_R(a)K(τ−a)da`` with
    the closed inner kernel ``K`` (``G_R⁰⊛C⁰``) and the small-``a`` power-law
    sliver (an equal-time / derivative-vertex σ_R may diverge integrably as
    ``a→0⁺``)."""
    a1 = ag[0]

    def _slope(s):
        if len(s) > 1 and s[0] > 0.0 and s[1] > 0.0:
            return math.log(s[1] / s[0]) / math.log(ag[1] / ag[0])
        return 0.0
    p = max(_slope(sR), -0.95)
    sliv = sR[0] * a1 / (1.0 + p)
    aeff = a1 * (1.0 + p) / (2.0 + p)

    def _K(c):
        c = np.asarray(c, dtype=float)
        return np.where(c >= 0.0, np.exp(-m * np.abs(c)) * (np.abs(c) + 0.5 / m),
                        np.exp(-m * np.abs(c)) / (2.0 * m))

    out = np.empty(len(taus))
    for i, tau in enumerate(taus):
        t1 = (N / m) * (sliv * _K(tau - aeff) + np.trapz(sR * _K(tau - ag), ag))
        t1m = (N / m) * (sliv * _K(-tau - aeff) + np.trapz(sR * _K(-tau - ag), ag))
        out[i] = t1 + t1m
    return out


def _dyson_keldysh(ag, sK, m, taus):
    """``δC_K(τ) = (1/2m)∫ σ_K(|τ−d|) e^{−m|d|} dd`` (``G_R⁰⊛Σ_K⊛G_A⁰``); σ_K is
    even and finite at 0, so no sliver is needed."""
    ag0 = np.concatenate(([0.0], ag))
    sK0 = np.concatenate(([sK[0]], sK))
    dg = np.concatenate((-ag[::-1], ag0))
    out = np.empty(len(taus))
    for i, tau in enumerate(taus):
        sh = np.interp(np.abs(tau - dg), ag0, sK0, left=sK0[0], right=0.0)
        out[i] = (1.0 / (2.0 * m)) * np.trapz(sh * np.exp(-m * np.abs(dg)), dg)
    return out


def _kinematic_to_physical(descr):
    """The universal kinematic↔enumeration normalization for a diagram: the
    enumeration ``𝒮(Γ)·prefactor`` uses the noise-source action coefficient, while
    the kinematic convolution uses a unit (T=1) correlation amplitude per C edge.
    They differ by ``2^{−n_C}`` (``n_C`` = number of correlation edges = number of
    contracted 2-point noise sources).  VERIFIED universal across the φ̃φ² Σ_R/Σ_K
    bubbles (n_C=2 → 1/4 = C_R/M_R = C_K/M_K) and the tadpole; re-checked at the
    2-loop sunset (n_C=3)."""
    n_C = sum(1 for e in descr.edges if e.kind == 'C')
    return 2.0 ** (-n_C)


def bubble_delta_C(descr, prefactor_val, q, taus, A, B, mu, D,
                   spatial_dim=1, formfactor=None):
    """The full 1-loop **bubble** contribution ``δC_Γ(q,τ)`` of one diagram,
    the generic momentum-first way: ``σ_Γ`` from the Symanzik route
    (:func:`loop_self_energy`), dressed by the generic single-mode Dyson
    convolution, weighted by ``2^{−n_C}·(𝒮(Γ)·prefactor)``.

    ``descr`` must be a non-tadpole (no self-loop) 2-vertex bubble; classified
    Σ_R (loop kinds ``{R,C}``) → retarded dressing, Σ_K (``{C,C}``) → Keldysh.
    ``prefactor_val`` is the enumeration ``𝒮(Γ)·prefactor`` evaluated at the
    params (carries the couplings AND all noise amplitudes — one ``T`` per C
    edge).  The kinematics are therefore computed at **unit noise**: ``σ_Γ`` with
    ``T=1`` and the external-leg dressing (``C₀``) with unit amplitude, so the
    total noise power ``T^{n_C}`` comes solely from ``prefactor_val`` (using the
    real ``T`` in the dressing too would double-count it → wrong ``T``-scaling).
    Returns an array over ``taus``."""
    taus = np.asarray(taus, dtype=float)
    m = A + B * q * q
    loop_kinds = tuple(sorted(e.kind for e in descr.loop_edges()))
    ag = _sigma_grid_axis(m, taus)
    sg = sigma_grid_direct(descr, q, ag, mu, D, spatial_dim=spatial_dim,
                           formfactor=formfactor)
    if loop_kinds == ('C', 'R'):
        kin = _dyson_retarded(ag, sg, m, 1.0, taus)      # unit-noise C₀ dressing
    elif loop_kinds == ('C', 'C'):
        kin = _dyson_keldysh(ag, sg, m, taus)            # G_R⁰⊛Σ_K⊛G_A⁰: no C₀
    else:
        raise NotImplementedError(
            f"bubble_delta_C: unsupported loop-edge kinds {loop_kinds}.")
    return _kinematic_to_physical(descr) * float(prefactor_val) * kin


def _phi2_zero(sl, q, mu, D, spatial_dim, L_cut, n_l=600):
    """The self-loop's loop integral ``⟨φ²⟩₀ = ∫dᵈℓ/(2π)ᵈ 1/(μ+D|k_ℓ|²)``
    (kinematic, T=1), ``k_ℓ = a·ℓ`` (a tadpole self-loop carries no ``q``).
    d=1 is UV-finite → the exact ``sigma_parametric`` value; d≥2 is UV-divergent
    → a **finite-cutoff** grid (Regime 1; ``|ℓ_i|<L_cut``)."""
    if int(spatial_dim) == 1:
        return sigma_parametric([(tuple(float(x) for x in sl.a),
                                  tuple(float(x) for x in sl.b), 'C')],
                                q, 0.0, mu, D, 1.0)
    d = int(spatial_dim)
    n_l = min(n_l, 220 if d == 2 else 70)
    axes = np.linspace(-L_cut, L_cut, n_l)
    grids = np.meshgrid(*([axes] * d), indexing='ij')
    a0 = float(sl.a[0])
    l2 = sum((a0 * g) ** 2 for g in grids)
    dvol = (axes[1] - axes[0]) ** d
    return float((1.0 / (mu + D * l2)).sum() * dvol / (2.0 * math.pi) ** d)


def tadpole_delta_C(descr, prefactor_val, q, taus, A, B, mu, D,
                    spatial_dim=1, L_cut=None):
    """The full 1-loop **tadpole** contribution ``δC_Γ(q,τ)`` of one diagram —
    an INSTANTANEOUS (equal-time self-loop) self-energy, i.e. a mass shift.

    This is the ``σ(a)=Σ·δ(a)`` limit of the SAME Dyson convolution: feeding it
    through :func:`_dyson_retarded` gives ``Σ·(−∂C₀/∂A)`` analytically, so

        δC_Γ(q,τ) = Σ_Γ · (−∂C₀/∂A),   −∂C₀/∂A = (1/m)e^{−m|τ|}(|τ|+1/m),
        Σ_Γ = 2^{−n_C}·(𝒮(Γ)·prefactor)·⟨φ²⟩₀^kin,

    with ``⟨φ²⟩₀^kin`` the self-loop's loop integral (``sigma_parametric`` on the
    self-loop edge at equal time, T=1), ``m = A + B q²``, and a factor
    ``∏ 1/m_{k_c}`` from any loop-momentum-free **connector** edges (e.g. the
    φ̃φ² rd ``k=0`` line — integrating its intermediate vertex time).  As in
    :func:`bubble_delta_C`, the kinematics use **unit noise** (the ``T^{n_C}``
    lives entirely in ``prefactor_val``).  Signs take care of themselves through
    the signed enumeration ``prefactor`` and the convolution's intrinsic
    ``−∂C₀/∂A`` (verified vs the Allen-Cahn oracle).

    Handles single-vertex tadpoles (φ̃φ³ Allen-Cahn — no connector) and
    multi-vertex tadpoles whose connectors carry no loop momentum (φ̃φ² rd —
    one ``k=0`` connector).  A connector carrying ``q`` (q-dependent Σ) raises."""
    taus = np.asarray(taus, dtype=float)
    m = A + B * q * q
    internal = descr.loop_edges()
    selfloops = [e for e in internal if e.u == e.v]
    if not selfloops:
        raise ValueError("tadpole_delta_C: no self-loop edge (not a tadpole).")
    non_self = [e for e in internal if e.u != e.v]
    sl = selfloops[0]
    if L_cut is None:
        L_cut = max(20.0, abs(q) + 15.0)
    phi2 = _phi2_zero(sl, q, mu, D, spatial_dim, L_cut)
    # Connector edges (non-self-loop internal lines, e.g. the φ̃φ² rd ``k=0``
    # line) carry NO loop momentum (a=0 — the loop is confined to the self-loop)
    # and integrating their intermediate vertex time contributes ``1/m_{k_c}``,
    # ``k_c = b_c·q``.  A pure tadpole's connectors have ``b_c=0`` (q-independent
    # mass shift); a q-dependent connector would make Σ q-dependent (deferred).
    conn = 1.0
    for e in non_self:
        if any(ai != 0 for ai in e.a):
            raise NotImplementedError(
                f"tadpole connector {(e.u, e.v, e.kind)} carries loop momentum "
                f"(a={e.a}); only loop-momentum-free connectors are supported.")
        kc = sum(float(bi) for bi in e.b) * q if e.b else 0.0
        conn *= 1.0 / (mu + D * kc * kc)
    Sigma = _kinematic_to_physical(descr) * float(prefactor_val) * phi2 * conn
    minus_dC0_dA = (1.0 / m) * np.exp(-m * np.abs(taus)) * (np.abs(taus) + 1.0 / m)
    return Sigma * minus_dC0_dA


def diagram_delta_C(descr, prefactor_val, q, taus, A, B, mu, D,
                    spatial_dim=1, L_cut=None, formfactor=None):
    """The full 1-loop contribution ``δC_Γ(q,τ)`` of ONE enumerated diagram — the
    single entry point, no bubble/tadpole branch in the *caller*.  Dispatches on
    the diagram's own structure (a property, not a physics choice): a self-loop
    edge ⇒ instantaneous self-energy (:func:`tadpole_delta_C`); otherwise a
    2-vertex loop ⇒ smooth self-energy (:func:`bubble_delta_C`).  ``formfactor``
    (a derivative-vertex ``F(ℓ)``) applies to the bubble loop only."""
    if descr.is_tadpole_like():
        return tadpole_delta_C(descr, prefactor_val, q, taus, A, B, mu, D,
                               spatial_dim=spatial_dim, L_cut=L_cut)
    return bubble_delta_C(descr, prefactor_val, q, taus, A, B, mu, D,
                          spatial_dim=spatial_dim, formfactor=formfactor)


def delta_C_one_loop(descrs_with_prefactors, q, taus, A, B, mu, D,
                     spatial_dim=1, prefactor_tol=1e-14, L_cut=None,
                     formfactors=None):
    """The **complete** 1-loop correction ``δC(q,τ) = Σ_Γ δC_Γ`` — the sum over
    ALL enumerated, LIVE (nonzero prefactor at the saddle) ell=1 diagrams, each
    evaluated by :func:`diagram_delta_C`.  This is where the dropped-tadpole
    problem disappears: bubbles AND tadpoles are summed uniformly, no diagram is
    special-cased away.

    ``descrs_with_prefactors`` : iterable of ``(CStackDiagram, prefactor_val)``
    where ``prefactor_val`` is the enumeration ``𝒮(Γ)·prefactor`` evaluated at
    the params/saddle.  Dead diagrams (``|prefactor| < tol``, e.g. φ*²-vertices
    at φ*=0) are skipped.  ``formfactors`` : optional list parallel to
    ``descrs_with_prefactors`` of per-diagram derivative-vertex form factors
    ``F(ℓ)`` (``None`` per entry ⇒ plain).  Returns an array over ``taus``."""
    taus = np.asarray(taus, dtype=float)
    total = np.zeros(len(taus))
    for i, (descr, pre) in enumerate(descrs_with_prefactors):
        if abs(float(pre)) < prefactor_tol:
            continue
        ff = formfactors[i] if formfactors is not None else None
        total = total + diagram_delta_C(descr, pre, q, taus, A, B, mu, D,
                                        spatial_dim=spatial_dim, L_cut=L_cut,
                                        formfactor=ff)
    return total

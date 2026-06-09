"""
⚠ ORACLE-ONLY — not on the production path. Superseded by ``full_integrator.py``
(see ``docs/spatial_pipeline.md``); reached only by its own test(s).  Kept as an
independent numerical cross-check — ``compute_cumulants`` does NOT use this module.

msrjd.integration.spatial.loop_dyson
====================================
1-loop Dyson assembly for the spatial bubble (Stage C.5) — turns the
momentum-first self-energies (``loop_parametric``) into the dressed
correlator correction.

For the ``φ̃φ²`` reaction-diffusion theory the 1-loop self-energy is a
**bubble** (momentum-DEPENDENT), with a retarded part ``Σ_R = G_R·C`` and a
Keldysh part ``Σ_K = C·C``.  The dressed correlation is the standard MSR Dyson

    δC(q,ω) = G_R⁰ Σ_R C⁰ + G_R⁰ Σ_K G_A⁰ + C⁰ Σ_A G_A⁰ ,

with ``G_R⁰=1/(m-iω)``, ``G_A⁰=1/(m+iω)``, ``C⁰=2T/(m²+ω²)``, ``m=μ+Dq²``,
``Σ_A=Σ_R*``.  The **equal-time** structure-factor correction ``δC(q,τ=0)``
has the closed convolution form (derived by inverse-FT at ``τ=0``)

    δC(q,0) = (T/m²) ∫₀^∞ Σ_R(q,u) e^{-mu} du
            + (1/m)  ∫₀^∞ Σ_K(q,u) e^{-mu} du ,

(the Keldysh double-time integral ``∫∫e^{-m(a₁+a₂)}Σ_K(|a₁-a₂|)`` collapses to a
1-D integral under ``(a₁,a₂)→(s=a₁+a₂, u=a₁-a₂)`` since ``Σ_K`` is even),
validated frequency-route == time-route (``tests/test_loop_dyson.py``).

The self-energy ``∫dℓ`` is **pole-free** (a momentum integral of a product of
exponentials/Lorentzians) whether done directly or by the parametric Symanzik
route — the ``m≥3`` close-pair bug lived ONLY in Phase J's time-polytope, which
this assembly bypasses.  So the bubble integrator is fast and robust at any q.

Normalization here is per ``Σ_R = ∫dℓ G_R·C``, ``Σ_K = ∫dℓ C·C`` with NO
coupling / combinatorial factor — the caller multiplies by ``𝒮(Γ)·(coupling)``
(``g²`` for the bubble), pinned from the pipeline.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate


def _mk(k, mu, D):
    return mu + D * k * k


# ── self-energy time kernels (direct ∫dℓ; pole-free, fast) ─────────
# ``formfactor`` (Phase 4): an optional callable ``F(ℓ)`` multiplying the loop
# integrand — the product of the two vertices' per-leg momentum form factors for
# a DERIVATIVE-vertex theory (e.g. ``F(ℓ)=−ℓ²`` for a ∇² on the loop leg, or
# ``F(ℓ)=−ℓ·(q−ℓ)`` for a KPZ gradient pair).  ``None`` ⇒ the plain bubble
# (``F=1``), exactly reproducing the validated Stage-C.5 result.  The momentum-
# first ∫dℓ stays pole-free with the polynomial factor, so this is robust.
def sigma_R_time(q, t, mu, D, T, formfactor=None):
    """Retarded bubble ``∫dℓ/2π F(ℓ) G_R(ℓ,t) C(q-ℓ,t)``  (t>0)."""
    if t <= 0:
        return 0.0
    ff = formfactor if formfactor is not None else (lambda l: 1.0)
    f = lambda l: (ff(l) * math.exp(-_mk(l, mu, D) * t)
                   * (T / _mk(q - l, mu, D)) * math.exp(-_mk(q - l, mu, D) * t))
    v, _ = integrate.quad(f, -np.inf, np.inf, limit=120)
    return v / (2 * math.pi)


def sigma_K_time(q, t, mu, D, T, formfactor=None):
    """Keldysh bubble ``∫dℓ/2π F(ℓ) C(ℓ,t) C(q-ℓ,t)``  (even in t)."""
    at = abs(t)
    ff = formfactor if formfactor is not None else (lambda l: 1.0)
    f = lambda l: (ff(l) * (T / _mk(l, mu, D)) * math.exp(-_mk(l, mu, D) * at)
                   * (T / _mk(q - l, mu, D)) * math.exp(-_mk(q - l, mu, D) * at))
    v, _ = integrate.quad(f, -np.inf, np.inf, limit=120)
    return v / (2 * math.pi)


# Principled per-diagram normalizations, pinned from the framework's own
# uniform-momentum diagram values (Σ_R diagram d[1][0] 𝒮(Γ)=16, Σ_K diagram
# d[1][1] 𝒮(Γ)=8): with the Dyson terms T1 (Σ_R) and T2 (Σ_K) normalized as
# below, the physical bubble correction is  c_R·T1 + c_K·T2  with
#
# CONFIRMED at 1-loop vs simulation: a φ²-only (lam=0) sim at the perturbative
# sweet spot (g=0.20, moderate run BEFORE the metastable −gφ³ potential drifts)
# gives fit coefficient B = sim-bubble/principled-bubble = 0.99 — bang on the
# 1-loop prediction B=1.  Longer runs / larger g inflate B (1.7–2.6) because the
# φ²-only theory is metastable (higher-order drift), NOT because the factors are
# wrong.  So c_R=4, c_K=2 are the correct 1-loop normalization.
C_R, C_K = 4.0, 2.0


def _dyson_terms(q, mu, D, T, formfactor=None):
    """Return the two Dyson terms ``(T1, T2)`` of ``δC(q,0)`` (normalization 1):
    ``T1 = (T/m²)∫Σ_R e^{-mu}``  (retarded+advanced Σ_R),
    ``T2 = (1/m) ∫Σ_K e^{-mu}``  (Keldysh).

    ``formfactor`` (Phase 4): an optional loop-momentum form factor ``F(ℓ)``
    (the product of the two vertices' per-leg momentum factors) threaded into
    both self-energies; ``None`` ⇒ the plain bubble."""
    m = _mk(q, mu, D)
    t1f = lambda u: sigma_R_time(q, u, mu, D, T, formfactor) * math.exp(-m * u)
    t1, _ = integrate.quad(t1f, 0, np.inf, limit=200)
    t1 *= T / (m * m)
    t2f = lambda u: sigma_K_time(q, u, mu, D, T, formfactor) * math.exp(-m * u)
    t2, _ = integrate.quad(t2f, 0, np.inf, limit=200)
    t2 /= m
    return t1, t2


def bubble_delta_S(q, mu, D, T, g=1.0, formfactor=None):
    """PHYSICAL bubble contribution to the equal-time structure factor
    ``δC(q, τ=0)`` for the ``φ̃φ²`` theory: ``g²·(C_R·T1 + C_K·T2)`` with the
    principled weights ``C_R=4, C_K=2`` (from the framework's 𝒮(Γ)).  Even in q.
    Excludes the q-independent ``φ²``-tadpole (the mass shift, d[1][2]).

    ``formfactor=F(ℓ)`` (Phase 4) injects a derivative-vertex form factor into
    the loop; ``None`` is the plain bubble (validated B=0.99 vs sim)."""
    t1, t2 = _dyson_terms(q, mu, D, T, formfactor)
    return g * g * (C_R * t1 + C_K * t2)


def bubble_delta_phi2(mu, D, T, g=1.0, q_cut=40.0):
    """``δ⟨φ²⟩ = ∫dq/2π δC(q,0)`` from the bubble (PHYSICAL, g²-scaled).

    ``q_cut`` bounds the (fast-decaying ``~1/q⁴``) momentum integral.
    """
    f = lambda q: bubble_delta_S(q, mu, D, T, g)
    v, _ = integrate.quad(f, 0.0, q_cut, limit=200)
    return 2.0 * v / (2 * math.pi)        # even in q → 2·∫₀


# ── full τ-dependent bubble correction (time route) ───────────────
def _sigma_grids(q, mu, D, T, t_max, n_t, n_l=2600, formfactor=None,
                 formfactor_K=None):
    """Tabulate ``σ_R(t), σ_K(t)`` on ``t∈(0,t_max]`` (n_t points) by a single
    VECTORIZED ``∫dℓ`` over the whole t-grid at once — one trapezoid on an
    ℓ-grid wide enough to cover the ``C(q−ℓ)`` peak at ``ℓ=q``.  ~100× faster
    than per-t ``scipy.quad`` and matches it to <1e-4 (validated).

    ``formfactor`` (Phase 4): a numpy-vectorized ``F_R(ℓ)`` on the retarded
    bubble; ``formfactor_K`` the (possibly different) Keldysh form factor — a
    derivative-vertex theory has F_R≠F_K (e.g. ∇²φ²: F_R=q²ℓ², F_K=q⁴).  If
    ``formfactor_K`` is None it falls back to ``formfactor``; both None ⇒ the
    plain bubble.
    """
    tg = np.linspace(t_max / n_t, t_max, n_t)
    L = max(60.0, abs(q) + 40.0)
    lg = np.linspace(-L, L, n_l)
    ml = mu + D * lg * lg
    mql = mu + D * (q - lg) ** 2
    ffR = (formfactor(lg) if formfactor is not None else 1.0) * np.ones_like(lg)
    _fk = formfactor_K if formfactor_K is not None else formfactor
    ffK = (_fk(lg) if _fk is not None else 1.0) * np.ones_like(lg)
    E_l = np.exp(-np.outer(ml, tg))                  # (n_l, n_t)
    Cq = (T / mql)[:, None] * np.exp(-np.outer(mql, tg))
    sR = np.trapz(ffR[:, None] * E_l * Cq, lg, axis=0) / (2 * math.pi)   # ∫dℓ F_R G_R·C
    sK = np.trapz(ffK[:, None] * (T / ml)[:, None] * E_l * Cq, lg, axis=0) / (2 * math.pi)
    return tg, sR, sK


def _sigma_grids_dD(q, mu, D, T, t_max, n_t, spatial_dim, n_l=110, L_cut=None):
    """``σ_R(t), σ_K(t)`` for the φ̃φ² bubble in ``spatial_dim`` ∈ {2,3} via a
    DIRECT, vectorized ``∫dᵈℓ`` over a Cartesian ℓ-grid (external q along the
    first axis), truncated at ``|ℓ_i|<L_cut`` (Regime 1 — a physical cutoff; the
    d≥2 loop is UV-sensitive, so ``L_cut`` matters and should match the simulator
    when comparing).  This is the fast production analog of the 1-D ``_sigma_grids``
    (d=1) — same object the C-stack ``sigma_parametric`` computes analytically
    (validated equal); used here vectorized over the whole t-grid for speed.

    No form factor (plain φ̃φ² vertex); derivative vertices in d>1 are deferred.
    """
    tg = np.linspace(t_max / n_t, t_max, n_t)
    if L_cut is None:
        L_cut = max(20.0, abs(q) + 15.0)
    axes = np.linspace(-L_cut, L_cut, n_l)
    grids = np.meshgrid(*([axes] * spatial_dim), indexing='ij')
    l2 = sum(gi ** 2 for gi in grids)                       # |ℓ|²
    ql2 = (q - grids[0]) ** 2 + sum(gi ** 2 for gi in grids[1:])  # |q−ℓ|² (q∥axis0)
    ml = (mu + D * l2).ravel()
    mql = (mu + D * ql2).ravel()
    dl = (axes[1] - axes[0]) ** spatial_dim
    pref = dl / (2.0 * math.pi) ** spatial_dim
    msum = ml + mql
    # vectorized over t (chunk to bound memory if the grid is large)
    Cq = T / mql
    TmlCq = (T / ml) * Cq
    E = np.exp(-msum[:, None] * tg[None, :])                # (n_grid, n_t)
    sR = (Cq[:, None] * E).sum(axis=0) * pref
    sK = (TmlCq[:, None] * E).sum(axis=0) * pref
    return tg, sR, sK


def bubble_delta_C_q_tau(q, taus, mu, D, T, g=1.0, t_max=60.0, n_t=4000,
                         formfactor=None, formfactor_K=None,
                         spatial_dim=1, L_cut=None):
    """PHYSICAL bubble correction ``δC(q, τ)`` for ALL ``τ`` in ``taus``, via the
    **time route** (the frequency route converges as 1/ω because Σ_R has a t=0
    step — Gibbs — so it is not used).  Each Dyson term collapses to a fast,
    accurate 1-D integral over the tabulated self-energy:

        Term1(τ) = (T/m) ∫₀^∞ σ_R(a)·K(τ−a) da,   (G_R⁰ ⊛ Σ_R ⊛ C⁰)(τ)
            K(c) = e^{-mc}(c + 1/2m)  (c≥0),  e^{mc}/2m  (c<0),
        Σ_R+Σ_A contribution  =  Term1(τ) + Term1(−τ),
        Term2(τ) = (1/2m) ∫ σ_K(|τ−d|) e^{-m|d|} dd,   (G_R⁰ ⊛ Σ_K ⊛ G_A⁰)(τ)

    ⇒ δC(q,τ) = g²[ C_R·(Term1(τ)+Term1(−τ)) + C_K·Term2(τ) ].  At τ=0 this
    equals the closed form ``bubble_delta_S(q)`` (validated to <1e-3); for τ≠0 it
    is the full time-displaced correlator.  ``σ_R, σ_K`` are tabulated once on
    ``(0, t_max]`` (n_t points) and the 1-D integrals done by trapezoid.

    Returns an array parallel to ``taus`` (real, even in τ).
    """
    m = _mk(q, mu, D)
    # Adaptive a-grid: the convolution kernels decay on the timescale 1/m, so the
    # grid must resolve [0, ~12/m] (and the τ-shift) with enough points — a fixed
    # t_max/n_t under-resolves large-m (large-q) modes (kernel decays within a few
    # cells → the steep σ_R·kernel product is over-counted).  Use ≥50 points per
    # 1/m and cap t_max at the kernel/τ reach; keep the caller's n_t as a floor.
    # CAP the point count: very-large-q modes are heavily damped (amplitude
    # T/m→0, δC→0) so they need not be resolved — without a cap n_t∝m would blow
    # up over the bridge's q-sweep.  The a→0⁺ σ_R singularity (the only
    # non-negligible large-q piece, at τ=0) is captured grid-independently by the
    # power-law sliver below, so capping is safe.
    tau_max = max((abs(float(t)) for t in taus), default=0.0)
    t_max = min(t_max, max(2.0 * tau_max + 12.0 / m, 12.0 / m))
    n_t = int(min(max(n_t, t_max * m * 50.0), 8000.0))
    if int(spatial_dim) == 1:
        ag, sR, sK = _sigma_grids(q, mu, D, T, t_max, n_t, formfactor=formfactor,
                                  formfactor_K=formfactor_K)  # a∈[a1,t_max]
    else:
        # d≥2: direct vectorized ∫dᵈℓ self-energy (no form factor; Regime-1 cutoff
        # L_cut).  The d-dependence lives entirely here; the Dyson convolution
        # below (Term1/Term2 + C_R/C_K) is d-independent.
        n_t = min(n_t, 1800)                                  # d≥2 grid is heavier
        ag, sR, sK = _sigma_grids_dD(q, mu, D, T, t_max, n_t,
                                     int(spatial_dim), L_cut=L_cut)
    a1 = ag[0]                                                # = t_max/n_t

    # [0,a1] sliver of ∫σ(a)·kernel da, done by the local power law.  A
    # derivative-vertex F_R (e.g. ∇²φ²: F_R=q²ℓ²) makes σ_R(a)~A·a^p with
    # p∈(-1,0) (here p=-½) — an INTEGRABLE singularity at a→0⁺.  The old
    # a=0 prepend (σ at 1e-7, then trapezoid) grossly OVER-counts that first
    # interval (≈100× for p=-½).  Integrate the power law exactly instead:
    #     ∫₀^{a1} A·a^p da = σ(a1)·a1/(1+p),   σ-weighted mean a_eff=a1(1+p)/(2+p).
    # For the plain bubble (σ_R,σ_K finite ⇒ p≈0) this is the σ(a1)·a1 rectangle
    # (a_eff=a1/2) — the validated B=0.99 behavior is preserved.  σ_K (F_K=q⁴,
    # ℓ-flat) is finite at 0, so Term2 needs no sliver.
    def _slope(s):
        if len(s) > 1 and s[0] > 0.0 and s[1] > 0.0:
            return math.log(s[1] / s[0]) / math.log(ag[1] / ag[0])
        return 0.0
    pR = max(_slope(sR), -0.95)        # clamp; p≤−1 would be a true UV divergence
    slivR = sR[0] * a1 / (1.0 + pR)
    aeffR = a1 * (1.0 + pR) / (2.0 + pR)

    def _K(c):                                                # closed inner
        return np.where(c >= 0.0,
                        np.exp(-m * np.abs(c)) * (np.abs(c) + 0.5 / m),
                        np.exp(-m * np.abs(c)) / (2.0 * m))

    # σ_K interp grid incl. a=0 (σ_K(0⁺)≈σ_K(a1) since σ_K is finite); symmetric
    # d-grid over d∈[−t_max,t_max] including 0 for the |τ−d| integral.
    ag0 = np.concatenate(([0.0], ag))
    sK0 = np.concatenate(([sK[0]], sK))
    dg = np.concatenate((-ag[::-1], ag0))

    out = np.empty(len(taus), dtype=float)
    for i, tau in enumerate(taus):
        # Term1 = (T/m)∫₀^∞ σ_R(a) K(τ−a) da = sliver[0,a1] + trapz[a1,t_max]
        term1 = (T / m) * (slivR * _K(tau - aeffR)
                           + np.trapz(sR * _K(tau - ag), ag))
        term1m = (T / m) * (slivR * _K(-tau - aeffR)
                            + np.trapz(sR * _K(-tau - ag), ag))
        # Term2 = (1/2m)∫ σ_K(|τ−d|) e^{−m|d|} dd  (σ_K finite ⇒ no sliver)
        argk = np.abs(tau - dg)
        sKshift = np.interp(argk, ag0, sK0, left=sK0[0], right=0.0)
        term2 = (1.0 / (2.0 * m)) * np.trapz(sKshift * np.exp(-m * np.abs(dg)),
                                             dg)
        out[i] = g * g * (C_R * (term1 + term1m) + C_K * term2)
    return out

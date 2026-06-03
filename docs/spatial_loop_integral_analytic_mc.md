# Spatial loop integral: analytic reductions past the Wick IFT + Monte Carlo

*Branch `spatial-extension`, June 2026.* After the analytic momentum (q→x) IFT
and the Wick form-factor moment, every diagram reduces to a **Schwinger-parametric
(Symanzik) integral** over internal vertex times and correlation Schwinger
parameters. This note records (1) how far that integral can be pushed
analytically and (2) what Monte-Carlo buys, both grounded in numerical
experiments (prototypes in `scratch/mc*.py`).

## The integral

For one diagram (`d` spatial dims, `L` loops, `k=2`):

    δC(x,τ) = Σ_chambers ∫ ∏_v dt_v ∏_{e∈C} dσ_e · 2^{−n_C} M(Γ) ·
              e^{−μ W} · (4πD)^{−Ld/2} (4πD F)^{−d/2} · e^{−x² U /(4D F)} · M_F

over an **`(n_V + n_C)`-dimensional** domain (`n_V` internal-vertex times, `n_C`
correlation σ's). For a KPZ 2-loop diagram `n_V=4, n_C=3` ⇒ **7-D**.

- `W = Σ_e w_e` (edge weights; R-edge `w=Δt`, C-edge `w=|Δt|+σ`); `e^{−μW}` is
  the IR/mass damping.
- `U = det M`, `M = Σ_e w_e a_e a_eᵀ` — **1st Symanzik**, multilinear in `w`, deg `L`.
- `F = U·Q_eff` — **2nd Symanzik**, deg `L+1`; `B = D·Q_eff = D F/U`.
- `M_F` — the Wick form-factor moment (`=1` for a plain vertex).

**Key simplification (verified in code):** the momentum prefactor `U^{−d/2}` and
the heat-kernel normalization `(4πB)^{−d/2}=U^{d/2}(4πDF)^{−d/2}` **cancel their
`U` powers**, leaving the clean `(4πDF)^{−d/2} e^{−x²U/4DF}` form above. Work in
`F`, not `U^{−1/2}`.

## Part 1 — How far analytically past the Wick IFT?

1. **`U^{−d/2}` cancellation** → F-only integrand. Exact, general.
2. **Radial (overall-scale) integral = a Bessel-K, exactly.** Scaling all
   times+σ's by `λ` (clean at τ=0, external times at the origin): `U→λ^L U`,
   `F→λ^{L+1}F`, `W→λW`, so the radial integral is
   `∫₀^∞ λ^p e^{−μλ − c/λ} dλ = 2 (c/μ)^{(p+1)/2} K_{p+1}(2√(μc))`, `c = x²Û/(4DF̂)`.
   **Removes exactly ONE dimension, any L, in closed form.** (Special case
   `∫₀^∞ u^{−1/2}e^{−au−c/u}du = √(π/a) e^{−2√(ac)}`.)
3. **Per-σ closed forms.** A *single* σ-integral (`U,F` linear in `σ_e`) is
   closed-form (√/erfc/Bessel). But `U,F` couple all edges, so after the first σ
   the remaining integrals are non-elementary → **at most ~1 extra dimension**.
4. **Full multi-loop = polylogarithms / elliptic.** No elementary closed form
   (standard for ≥2-loop parametric integrals).

**Bottom line:** ≈ **1–2 of the `(n_V+n_C)` dimensions** reduce to closed form —
the radial Bessel-K being the clean, general, exact one. The remaining ≈5-D
(KPZ 2-loop) is an irreducibly numerical, **smooth, bounded** integral over the
angular simplex.

## Part 2 — Monte Carlo

Importance-sampled MC (`scratch/mc.py`, `mc2.py`, `mcconf.py`): internal times via
**nested `Exp(μ)` gaps** (matching the retarded poset bounds), σ's `~ Exp(μ)`;
per-sample weight `e^{−μ(mu_resid − Σgaps)}/μ^{n_V+n_C}` (the `e^{−μσ}` cancels
against the σ proposal). Bounded memory (process `N` samples in chunks), `O(1/√N)`.

| Case | result |
|---|---|
| **q-space integral**, 1-loop | MC vs grid **0.2%**, 1e6 pts in **0.18 s** ✓ |
| **PLAIN δC(x)**, 1-loop | **0.03–0.35%** at all x, stable over 5 seeds (vs fine nt26 grid) ✓ |
| **PLAIN δC(x)**, 2-loop | MC(5e6)=`[0.09449,…,0.04826]`; grid converges nt6=0.0913→nt8=0.0951→**nt10=0.09457** ⇒ MC matches the converged grid to **<0.1%**, **memory-safe** where the nt16 grid OOMs (180M pts/chamber, 72 GB). 135 s. ✓ |
| **DERIVATIVE-vertex δC(x)** (KPZ/Model B) | **BIASED** (46% @ x=0; 2-loop signal swamped, MC 10× low & sign-wrong) ✗ |

**Why the derivative case fails:** the Wick moment `M_F = E[F(a·Q+ξ,Q)]` has
`a=−M⁻¹N`, `Σ=(2DM)⁻¹` that **blow up as `U=det M → 0`** (degenerate loop), so the
integrand `~ U^{−(1/2 + deg F/2)}` (e.g. `U^{−5/2}` for KPZ) and the **variance
integral `∫U^{−(1+deg F)}` diverges** → infinite-variance MC. (The *plain* heat
kernel `~F^{−1/2}` is a mild, integrable singularity → MC is fine; this was
confirmed — the plain xs-path MC converges, only the form-factor moment biases.)

## Synthesis / recommendation

- **Plain theories (Allen-Cahn, reaction-diffusion, all φⁿ):** MC is **ready and
  validated** — a memory-safe, ~0.1% accurate, fast ℓ=2 path. Worth wiring as an
  integrator option (`SPATIAL_INTEGRATOR=mc`, `SPATIAL_MC_N`). Resolves the ℓ=2
  dimensionality wall for these models.
- **Derivative theories (KPZ, Model B, Burgers):** plain MC is **not enough** —
  needs variance reduction targeting the loop-degeneracy (`det M → 0`) singularity:
  the **analytic radial (Bessel-K) reduction is the natural cure** (the singularity
  lives in the scale direction; doing it analytically regularizes the angular MC).
  Alternatives: importance-sample toward non-degenerate loops (large `det M`),
  control variates, or stratification in `det M`.
- **Recommended estimator:** **analytic radial (Bessel-K) × importance-sampled
  angular MC** — removes one dimension *and* tames the singularity in one move.

## Caveats / scope
- Numbers are KPZ/plytic d=1, τ=0, μ=D=T=1, c=0.3. The MC weight derivation
  assumes the equal-time (τ=0) projective scaling; τ≠0 needs the external-time
  shift carried through.
- The radial Bessel-K reduction is derived but **not yet implemented**; it is the
  concrete next step for a feasible *derivative-vertex* ℓ=2.

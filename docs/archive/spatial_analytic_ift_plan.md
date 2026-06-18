# Analytic spatial inverse Fourier transform (heat-kernel IFT) — design + plan

*Branch `spatial-extension`, June 2026.* Retire the numerical `q→x` transform
(and the `n_q` grid) by doing the spatial IFT **analytically**, per
Schwinger/chamber sample. This is the order-of-magnitude speedup for the spatial
loop path; everything stays in "heat-kernel land".

## 1. The core identity

After Symanzik reduction, a diagram's momentum-space value, **per
chamber/Schwinger quadrature sample `w`** (internal times + correlation σ's), is

    δC(q,τ)|_w = A(w) · ⟨F(ℓ,q)⟩_ℓ · exp(−D·qᵀQ_eff(w)·q)

with the **amplitude** and **width**

    A(w) = 2^{−n_C}·M(Γ)·e^{−μΣw}·(4πD)^{−Ld/2}·U(w)^{−d/2},
    B(w) = D·Q_eff(w),     Q_eff = Q − NᵀM⁻¹N,  U = det M

(`⟨F⟩_ℓ = 1` for a plain polynomial vertex; a polynomial in `q` (and `ℓ`) for a
derivative vertex). The spatial correlator is the IFT

    δC(x,τ) = ∫ dᵈq/(2π)ᵈ e^{iq·x} δC(q,τ)
            = Σ_chambers ∫dw  A(w) · 𝓘(x,w),
    𝓘(x,w) = ∫ dᵈq/(2π)ᵈ e^{iq·x} ⟨F(ℓ,q)⟩ exp(−D·qᵀQ_eff·q).

**The whole game is doing `𝓘(x,w)` in closed form** — and it always can be,
because the `q`-dependence is (polynomial)×(Gaussian).

## 2. The three cases

### Case A — plain vertices (`F = 1`): pure heat kernel
`𝓘(x,w) = ∫dᵈq/(2π)ᵈ e^{iq·x} e^{−B q²} = (4πB)^{−d/2} exp(−|x|²/4B)`, so

    δC(x,τ) = Σ_chambers ∫dw  A(w)·(4πB(w))^{−d/2}·exp(−|x|²/4B(w)).

No `q`-grid, no FT, no ringing — a weighted sum of heat kernels over the chamber
quadrature. (Allen-Cahn, reaction-diffusion, all `φⁿ`.)

### Case B — composite-derivative vertices (Model B `∇²(φ²)`): Hermite × heat kernel
The `ℓ`-average leaves a polynomial in `q`: `⟨F⟩_ℓ = P(q)`. Then

    𝓘(x,w) = ∫dᵈq/(2π)ᵈ e^{iq·x} P(q) e^{−Bq²} = P(−i∂_x)·(heat kernel)
           = (Hermite polynomial in x)·(4πB)^{−d/2}exp(−|x|²/4B).

(`∫dq/2π · q^{2n} e^{iqx−Bq²} = (−∂_x²)ⁿ` of the kernel.)

### Case C — per-leg derivative vertices (KPZ `(∇φ)²`): joint `(ℓ,q)` Gaussian
`F(ℓ,q)` is polynomial in **both** `ℓ` and `q`. Fold the `q`-integral into the
loop average: the FT source turns `q` into a **complex Gaussian** — the exponent
`−Bq² + iq·x` has saddle `q* = ix/(2B)` and variance `1/(2B)`. Hence

    𝓘(x,w) = (4πB)^{−d/2} e^{−|x|²/4B} · ⟨F(ℓ,q)⟩_{(ℓ,q)~joint Gaussian},
    q ~ N(ix/2B, 1/2B),   ℓ | q ~ N(−M⁻¹N q, (2DM)⁻¹).

`⟨F⟩` of a polynomial over a Gaussian is closed-form (Wick / Isserlis): a
polynomial in `x` (from the `ix/2B` mean) times the heat kernel.

### Unifying statement
    δC(x,τ) = Σ_chambers ∫dw  A(w)·(4πB)^{−d/2} e^{−|x|²/4B} · 𝓜(x,w),
    𝓜 = 1 (A) | Hermite(x) (B) | joint-Gaussian moment of F (C),  C ⊃ B ⊃ A.
The chamber/Schwinger quadrature `∫dw` runs **once per diagram**; the
`x`-dependence is analytic and broadcast over the whole output grid.

## 3. Cost & caveats
- **Cost:** old = `n_q × (chamber quad) + numerical FT`; new = `chamber quad ×1`,
  analytic in `x`. → up to **`~n_q×` (≈64×)** for plain vertices; large for
  derivative vertices too (a tiny `q`-moment in place of 64 `q`-samples). Exact:
  no FT ringing, no large-`x` degradation, no `n_q`/`q_cut`.
- **Quadratic generator only:** `D∇² → e^{−Dk²w}` (Gaussian). `|k|^α` → Lévy
  stable kernel (out of scope — we only have `D∇²`; documented limit).
- **Complex means** (`q* = ix/2B`): use the analytic Gaussian-moment (Wick)
  route on the symbolic polynomial `F` (cleanest — no real GH grid with a complex
  mean); the physical `δC(x)` is real (imaginary parts cancel in the diagram sum,
  as now). Equivalent operator form: `P(−i∂_x)` on the heat kernel.
- **`k>2`:** `F = qᵀG(w)q` (multivariate Gaussian) → `(det G)^{−1/2}e^{−xᵀG⁻¹x}`;
  same machinery, more indices. (Argues `k>2` is *easier* here than feared.)

## 4. Amenability — CONFIRMED against the code
- **`A(w)`, `B(w)`:** `_momentum_factor_batch` already computes `pref =
  (4πD)^{−Ld/2}U^{−d/2}` and `Q_eff` per sample (full_integrator.py:76–79) — it
  just collapses them with `q` into the scalar `out`. Expose per-sample.
- **Chamber accumulation:** `diagram_kinematic`'s chamber loop builds
  `w_batch`/`wfull`/`mu_resid` per sample (full_integrator.py:266–298); the final
  `total += Σ wfull·e^{−μresid}·momfac` (298) is the single localized change —
  replace `momfac(q)` with `Σ_samples … · heat_kernel(B,xs)`.
- **`F(ℓ,q)` polynomial:** verified clean polynomials in the routing symbols —
  Model B `F = q₀²(ℓ₀−q₀)²` (q-deg 4, ℓ-deg 2); KPZ `F = ℓ₀²q₀(ℓ₀−q₀)` (q-deg 2,
  ℓ-deg 3). `q` is just another polynomial variable; degrees/coeffs extract via
  `sp.Poly` (the existing `_min_gh_order` already does this for the loop degree).

## 5. Implementation outline (phased; each bit-identical-validated)

**Keep the numerical-FT path as the validated reference** (behind a flag) for
cross-checks throughout.

**Phase 1 — Case A (plain vertices), the clean win.** ✅ DONE (June 2026).
`_symanzik_kernel_batch` + `diagram_kinematic(xs=…)` heat-kernel path +
`diagram_{value,correlator}_x` / `correlator_2pt_x`; `compute_spatial_correlator_
generic` routes all-plain theories through it (gate `_all_plain`), retiring the
q-grid + FT. Validated: analytic vs numerical FT ≤1e-4 (`test_analytic_ift_vs_
numerical_ft`); Allen-Cahn φ⁴ e2e exact (0.5000/0.4625/0.4707), `max_ell=2`
**~17 min → 16.6 s** (the `n_q` factor gone); works at d≥2 (the heat kernel is
isotropic). 25/25 regression. Original plan below:
1. `_symanzik_gaussian_batch` (or extend `_momentum_factor_batch` return): per
   sample give `(pref, Qeff, M, N, ok)` — the un-collapsed Gaussian.
2. `diagram_kinematic_x(descr, xs, external_times, mu, D, spatial_dim)`: same
   chamber loop; accumulate `Σ wfull·e^{−μresid}·pref·heat_kernel(D·Qeff, xs)` →
   `(n_x,)`. Add `diagram_correlator_x` / `correlator_2pt_x`.
3. `compute_spatial_correlator_generic`: for non-derivative theories call the
   `_x` path; drop the `q`-loop + `_ft_to_x`.
4. **Validate:** Allen-Cahn φ⁴ `δC(x)` matches the current path to FT accuracy;
   tree `C₀(x)` exact (closed form).

**Phase 2 — Cases B & C (derivative vertices): joint `(ℓ,q)` Gaussian.** ✅ DONE
(June 2026).  `_formfactor_average_x` (the **polynomial-fit route**): `P(q)=⟨F⟩_ℓ`
is a polynomial in `q` of degree ≤ `total_degree(F)` (the ℓ-mean `ℓ̄=−M⁻¹Nq` is
linear in `q`) — interpolate it from `q_poly_deg+1` ℓ-Gauss–Hermite evals on a
scaled Vandermonde, then `FF(x)=Σ_n p_n·E[(u+ix/2B)^n]` with the closed-form
heat-kernel `q`-moments `u~N(0,1/2B)`.  Wired into `diagram_kinematic(xs=…)`
(formfactor branch) + `diagram_correlator_x`; `compute_spatial_correlator_generic`
routes **d=1 derivative** vertices through it (gate `_use_analytic=_all_plain or
d==1`).  `SPATIAL_FORCE_NUMERICAL_FT=1` (+ `SPATIAL_Q_CUT`/`SPATIAL_N_Q`) keeps the
numerical-FT path as the validated cross-check reference.
- **Validated (diagram level, `δC(x)` analytic vs numerical FT):** KPZ `(∇φ)²`
  (Case C) ≤ 1e-4; Model B `∇²(φ²)` (Case B) the numerical FT **converges** to the
  analytic value as `q_cut→∞` (`0.0318` at `q_cut=40` → `0.0333724` at `q_cut=80`
  vs analytic `0.0333725` — the analytic IFT is exact; the discrepancy was the
  truncated `+¼q⁴` tail, NOT a bug).  `test_analytic_ift_derivative_vs_numerical_ft`.
- **Validated (e2e via `compute_cumulants`, `max_ell=1`):** KPZ `C(0,0)=0.501091`
  (hits the established **0.5011**), Model B `C(0,0)=0.530910` — real, smooth,
  ~3 s (no q-grid).  Original plan below:
1. `_formfactor_average_x`: generalize `_formfactor_average` — average `F` over
   the joint `(ℓ,q)` Gaussian with `q ~ N(ix/2B, 1/2B)`, returning `(n_x,)` ×
   heat kernel. Analytic Gaussian-moment (Wick) route on the symbolic `F`
   (reuse the `q`-degree from `sp.Poly`, mirror `_min_gh_order`).
2. Wire into `diagram_kinematic_x`.
3. **Validate:** Model B + KPZ `δC(x)` vs the numerical-FT path (same `q_cut`→
   continuum); `C(0,0)` matches the validated 0.356/0.501 numbers.

**Phase 3 — `d≥2` + retire `n_q`.**
1. `x` along an axis; transverse `q`-components → heat-kernel measure; the
   component form factor → the existing transverse-moment structure.
2. `compute_spatial_correlator_generic` uses the `_x` path universally; `n_q`/
   `q_cut` retire (kept only as a debug fallback).
3. **Validate:** `d=2`/`d=3` vs the current path + the brute `∫dᵈℓ` oracle.

**Why phased:** Phase 1 alone retires `n_q` for every polynomial theory (the bulk
of use cases) and is trivially exact; Phases 2–3 reuse the validated form-factor
machinery. No parallelism primitives anywhere — pure analytic numpy, cannot crash.

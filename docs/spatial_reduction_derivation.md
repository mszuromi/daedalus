# Spatial loop integral — the full reduction chain

*Branch `spatial-extension`.* How one MSR–JD Feynman diagram for a spatial
stochastic field theory is reduced from a `(time × momentum)` integral to a small
angular quadrature, stage by stage:

> **construction → temporal poles → Schwinger → momentum (Symanzik) → Wick → Bessel**

Conventions (white noise; colored noise is Markovianized to this form upstream):
- physical field `φ`, response field `φ̃`; mass/relaxation `m_k = μ + D|k|²`.
- free propagators in `(k,t)`:
  `G_R(k,t) = θ(t) e^{−m_k t}` (response/retarded),
  `C(k,t) = (T/m_k) e^{−m_k|t|}` (correlation).
- one external momentum `q` (k=2 two-point); `L` loops; `d` spatial dims.

---

## 0. Construction — diagram ⟶ (time × momentum) integral

The connected correlator is a sum over MSR–JD diagrams `Γ`. Each has internal
vertices (integrated over `t_v, x_v`), external legs (fixed), and edges = free
propagators (`R` or `C`). Vertices come from the action's nonlinear terms: a
coupling × a **form factor** (a derivative vertex deposits a momentum polynomial:
`∇² → −|k|²`, `∂_x → ik`).

Fourier over space and route momenta — each edge carries
`k_e = a_e·ℓ + b_e·q` (`a_e ∈ {0,±1}^L` loop incidence, `b_e` external routing);
the `∫d^dx_v` enforce momentum conservation and collapse to `L` loop integrals:

    Γ = 2^{−n_C} M(Γ) ∫ ∏_v dt_v  ∫ ∏_{i=1}^{L} d^dℓ_i/(2π)^d
            ∏_e [ propagator_e(k_e, Δt_e) ] · F(ℓ, q)

`M(Γ)` = combinatorial/symmetry factor, `n_C` = # correlation edges, `F` = product
of vertex form factors (`F≡1` for plain `φⁿ` vertices).

## 1. Temporal poles — causal chambers + the `1/m_k` residues

The time structure has two parts:

- **Retarded `θ`'s ⟶ causal chambers.** Each `G_R` carries `θ(Δt)`, so the internal
  vertex times are **partially ordered** (the retarded edges define a poset). Split
  the `∫∏dt_v` into the **causal chambers** = the total orders consistent with the
  poset. *Inside a chamber every `|Δt|` sign is fixed*, so the integrand is smooth
  (no cusps) and each edge has a definite time extent `w_e⁽⁰⁾ = Δt` (R) or `|Δt|` (C).

- **`1/m_k` is the pole.** In the frequency domain the propagators are rational in
  `ω` with poles at `ω = ±i m_k`; the temporal-only pipeline closes the contour and
  sums **residues**. Spatially `m_k = μ+D|k|²` stays *symbolic in `k`*, so we do NOT
  take `ω`-residues. The residue content survives as the **static factor `1/m_k`
  inside each correlation edge** `C(k,t)=(T/m_k)e^{−m_k|t|}`. That `1/m_k` is the
  object the next step removes.

## 2. Schwinger — linearize the poles

Use `1/m_k = ∫_0^∞ dσ \, e^{−σ m_k}` on each correlation edge:

    C(k,t) = T ∫_0^∞ dσ_e \, e^{−m_k (|Δt| + σ_e)} .

Now **every** edge contributes a pure exponential `e^{−m_{k_e} w_e}` with a single
positive time weight

    w_e = Δt      (R-edge),        w_e = |Δt| + σ_e   (C-edge, σ_e ∈ [0,∞)).

Because `m_k = μ + D|k|²` splits additively,

    ∏_e e^{−m_{k_e} w_e} = e^{−μ W} · e^{−D Σ_e w_e |k_e|²},   W = Σ_e w_e .

The mass piece `e^{−μW}` is the overall IR/exponential damping; the diffusion piece
is **Gaussian in the momenta**. So:

    Γ = 2^{−n_C} M(Γ) T^{n_C} Σ_chambers ∫∏_v dt_v ∏_{e∈C} dσ_e \, e^{−μ W}
            ∫ ∏_i d^dℓ_i/(2π)^d \, e^{−D Σ_e w_e |k_e|²} F(ℓ,q).

## 3. Momentum integral — Symanzik reduction (exact Gaussian)

With `k_e = a_e·ℓ + b_e·q`,

    Σ_e w_e |k_e|² = ℓᵀ M ℓ + 2 ℓᵀ N q + qᵀ Q q,
      M = Σ_e w_e a_e a_eᵀ   (L×L),   N = Σ_e w_e a_e b_eᵀ,   Q = Σ_e w_e b_e b_eᵀ.

Complete the square (`ℓ = ℓ̄ + ξ`, `ℓ̄ = −M⁻¹N q`) and do the Gaussian `∫d^dℓ`:

    ∫ ∏_i d^dℓ_i/(2π)^d e^{−D(…)} F(ℓ,q)
        = (4πD)^{−Ld/2} U^{−d/2} e^{−D qᵀ Q_eff q} · ⟨F(ℓ,q)⟩_ℓ ,
      U = det M  (1st Symanzik, deg L),
      Q_eff = Q − Nᵀ M⁻¹ N,   F_sym ≡ U·Q_eff  (2nd Symanzik, deg L+1),
      ξ ~ N(0, Σ_ℓ),   Σ_ℓ = (2D M)⁻¹ ,   ℓ̄ = −M⁻¹N q .

So *per chamber sample* `(t,σ)` the momentum-space contribution is
`e^{−μW} (4πD)^{−Ld/2} U^{−d/2} e^{−B q²} ⟨F⟩`, with `B = D Q_eff` (scalar at k=2).
`⟨F⟩ = 1` for plain vertices.

## 4. Wick — loop-momentum average + the `q→x` inverse FT

Two Gaussian averages, both closed form:

**(a) Loop average.** `⟨F(ℓ,q)⟩_ℓ` of the polynomial `F` over `ℓ ~ N(ℓ̄(q), Σ_ℓ)`
is Isserlis/Wick — a polynomial in `q`.

**(b) `q → x` inverse transform.** The real-space correlator is
`δC(x) = ∫ d^dq/(2π)^d e^{iq·x} δC(q)`. The bare Gaussian gives the **heat kernel**

    ∫ d^dq/(2π)^d e^{iq·x} e^{−B q²} = (4πB)^{−d/2} e^{−|x|²/4B} ≡ K(B,x),

and the momentum prefactor's `U^{−d/2}` **cancels** the kernel's normalization
`(4πB)^{−d/2}=U^{d/2}(4πD F_sym)^{−d/2}` ⟹ clean `(4πD F_sym)^{−d/2} e^{−|x|²U/4D F_sym}`.

The form factor's `q`-dependence is folded *into* the transform: the source
`e^{iq·x − Bq²}` makes `q` a **complex Gaussian** `q ~ N(ix/2B, 1/2B)`. The full
`(ℓ,q)` is then jointly Gaussian, and

    M_F(a, Σ_ℓ, B, x) ≡ E_{(ℓ,q) joint Gaussian}[ F(ℓ,q) ]   (a = −M⁻¹N)

is again Isserlis/Wick — a **polynomial in `x`** with coefficients rational in the
Symanzik data. Collecting, per chamber sample:

    δC(x)|_w = 2^{−n_C} M(Γ) T^{n_C} · e^{−μ W}
               · (4πD)^{−Ld/2} (4πD F_sym)^{−d/2} · e^{−|x|²U/4D F_sym} · M_F .

(`M_F = 1` for plain vertices; this is the analytic IFT that retired the q-grid.)

## 5. Bessel — radial reduction of the remaining time/σ integral

What's left is the `(n_V + n_C)`-D **Schwinger-parametric integral** over the
internal times and the `σ`'s. Reparametrize by an overall scale: with
`u_v = −t_v ≥ 0` (time-in-the-past; external times at the origin for τ=0),

    (u_v, σ_e) = λ · ŝ,   ŝ on the (n−1)-simplex (Σ ŝ = 1),   n = n_V + n_C .

The Symanzik polynomials are homogeneous, so `U → λ^L Û`, `F_sym → λ^{L+1} F̂`,
`W → λ Ŵ`, and the heat-kernel exponent `|x|²U/4D F_sym → c/λ`. The **radial
`λ`-integral is a modified Bessel function**:

    ∫_0^∞ dλ \, λ^p e^{−a λ − c/λ} = 2 (c/a)^{(p+1)/2} K_{p+1}(2√(a c)),
      a = μ Ŵ ,   c = |x|² Û / (4D F̂) ,
      p = (n−1)  −  (L+1)d/2  +  p_moment .

- `(n−1)` = simplex Jacobian; `−(L+1)d/2` from `F̂^{−d/2}` (the **only** `d`
  dependence — it shifts the Bessel **order** `p+1`, with `a,c` `d`-independent);
  `p_moment` = the `λ`-power carried by `M_F`.
- **Plain vertices** (`M_F=1`): a single `p` ⟹ **one Bessel-K** (verified exact,
  R²=1.0).
- **Derivative vertices**: `M_F = Σ_k g_k(ŝ,x) λ^{p_k}` ⟹ a **sum of Bessel-K's**,
  `Σ_k g_k · 2(c/a)^{(p_k+1)/2} K_{p_k+1}(2√(ac))`.

The radial direction is exactly where the `det M → 0` (degenerate-loop) singularity
lives, so doing it analytically **regularizes** what remains: the **angular
integral over the (ordered) simplex**, which is smooth, bounded, `(n−1)`-D — done by
low-order quadrature or Monte-Carlo.

## Assembly

    δC(x,τ) = Σ_Γ Σ_chambers ∫_{simplex} dŝ [angular measure]
                 · [radial Bessel-K (plain) or Σ_k Bessel-K (derivative)] ,

plus the retarded+advanced `τ → −τ` sum per diagram, summed over all diagrams (loop
orders `1…max_ell`) with weights `2^{−n_C} M(Γ)·prefactor`.

## What is exact vs numerical
| stage | status |
|---|---|
| construction, momentum (Symanzik), Wick (loop avg + q→x IFT) | **exact, closed form** |
| temporal `θ` ⟶ chambers; Schwinger `1/m_k → ∫dσ` | exact reparametrization (adds the `σ` integrals) |
| Bessel radial `λ`-integral | **exact** (single K plain / sum of K's derivative) |
| **angular simplex integral** | the one remaining **numerical** piece — `(n−1)`-D, smooth (currently: grid; MC; Bessel×angular-quadrature = the planned path) |

Backends today: `method='grid'` (causal-chamber product quadrature, default) and
`method='mc'` (importance-sampled Monte-Carlo). The **Bessel radial × angular
quadrature** backend (this derivation) is the next build — it does stage 5's radial
λ analytically and only quadratures the smooth angular simplex.

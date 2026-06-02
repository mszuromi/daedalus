# Spatial field theory at d = 2 and d = 3 — what changes, and what to build

> **STATUS: DONE + VALIDATED (June 2026).** Every vertex type — polynomial AND
> derivative/form-factor — runs at `d ∈ {1,2,3}`. The `L·d`-dim transverse-moment
> Gauss–Hermite average (`_formfactor_average(…, spatial_dim)`) + the
> per-component form factor (`diagram_form_factor(…, d)`: `Lap→−|p|²`,
> `Dx_i→i p_i`) match a brute `∫dᵈℓ` to **1.0e-14 (d=2)** / brute-grid (d=3) for
> Model B `∇²(φ²)` and KPZ `(∇h)²=Σ_i(∂_i h)²`; both run e2e through
> `compute_cumulants` at d=2 (`test_full_integrator.test_formfactor_d_ge_2_vs_brute`,
> 4 cases).  **2-D simulator** (`spatial_field_2d_sim.simulate_2d` gained `g_lap`
> Model B + `lam_kpz` KPZ forcings): the d=2 KPZ excess velocity
> `⟨φ⟩=(κ/2μ)⟨(∇φ)²⟩` (both axes) matches the lattice tree to **0.1 %**
> (`test_kpz_burgers_sim.test_kpz_d2_excess_velocity`).  The only `d≥2` caveat is
> physical: higher superficial degree of divergence ⇒ the *bare* loop is
> cutoff-sensitive (needs renormalisation; Model B's conserved ∇² is also stiff
> in the sim → smaller dt).

*Branch `spatial-extension`, June 2026.* Goal: **every** vertex type (polynomial
AND derivative/form-factor) at `d ∈ {1,2,3}`. This doc is the math analysis
(grounded in the code) that precedes the implementation.

## The pieces, and which are already d-general

The spatial loop pipeline is `enumerate → route momenta → Symanzik ∫dᵈℓ →
causal-chamber time integral → ret+adv → q→x FT`. Walking each piece:

### 1. Propagator (heat kernel) — **already d-general**
`m_k = μ + D|k|²` with `|k|² = Σ_{α=1}^{d} k_α²`. Real space
`G(x,t) = θ(t)(4πDt)^{−d/2} exp(−|x|²/4Dt − μt)`. `gaussian_heat_kernel`
already takes `spatial_dim` and uses the `(4πBt)^{−d/2}` prefactor; the q→x
transform is `radial_inverse_ft` for `d≥2`. The drift `V` (KPZ/Burgers saddle
cross-term) is a *vector* in `d≥2`, but it is `0` at the homogeneous saddle
`φ*=0`, so it never enters — no change needed.

### 2. Symanzik momentum integral — **already d-general**
A Gaussian loop integral in `d` dimensions **factorizes over the `d` spatial
components** because the propagators depend only on `|ℓ|²` (isotropy). With the
Schwinger weights `w_e` and the per-edge routing `k_e = Σ_i a_{ei}ℓ_i +
Σ_j b_{ej}q_j` (the coefficients `a,b` are integers, **d-independent**),

    M = Σ_e w_e a_e a_eᵀ   (L×L),   N = Σ_e w_e a_e b_eᵀ,   Q = Σ_e w_e b_e b_eᵀ,

and the momentum factor is

    MomFactor = (4πD)^{−Ld/2} · U^{−d/2} · exp(−D·qᵀ Q_eff q),
    U = det M,  Q_eff = Q − Nᵀ M⁻¹ N.

The **only** `d`-dependence is the powers `−Ld/2`, `−d/2` (the Gaussian
normalization in `d` dims), and `qᵀQ_eff q` uses `|q|` (place `q` along axis 0).
`_momentum_factor_batch(…, spatial_dim)` already does exactly this — validated
`d=2` Keldysh sunset vs brute `∫d²ℓ` to 2.5e-4.

### 3. Vertices / `expand` — **already d-general (operator IR)**
`spatial_operator_ir.form_factor(chain, k)` already maps `Lap→−|k|²=−Σ_α k_α²`,
`Dx_i→i k_i` on a **vector** `k`. The lowering (`to_derived_generators`,
`classify_generators`, the per-vertex table `ns._operator_ir_vertex_terms`) is
purely algebraic and dimension-free. **KPZ in `d≥2`** is the *full gradient*
`(∇h)² = Σ_{α} (∂_α h)²` — authored as `Σ_i Dx(phi,i)²`, i.e. `d` per-axis
perleg vertices that the **multi-vertex table** (just built) sums automatically;
its form factor is the rotational-invariant dot product `𝔣 = Σ_α (ip₁_α)(ip₂_α)
= −p₁·p₂`. Model B `∇²(φ²)` is isotropic already (`𝔣 = −|p|²`).

### 4. The form-factor LOOP AVERAGE — **the one piece that is d=1-specific**
A derivative vertex deposits a **polynomial** form factor `F(ℓ,q)` on the loop;
the loop integral is `MomFactor · ⟨F⟩`, `⟨F⟩` the average over the loop Gaussian

    ℓ ~ N(ℓ̄, Σ),   ℓ̄ = −M⁻¹N q,   Σ = (2D M)⁻¹.

In `d=1` (`_formfactor_average`) `ℓ` is `L` scalars and `⟨F⟩` is an `L`-dim
Gauss–Hermite (exact for the polynomial `F`). In `d≥2` `ℓ` is `L` **vectors**
(`L·d` scalar components) and `F` depends on the components (via `Dx_i` / the
dot product), so we need the **`d`-dim Gaussian average** — the *transverse
momentum moments*. This is the deferred piece (`diagram_kinematic` raises for
`formfactor + spatial_dim≠1`).

## The d≥2 form-factor average — the math

The full loop covariance is **`Σ ⊗ I_d`**: the `d` spatial components are
independent, each an `L`-dim Gaussian with the **same** precision `M` (from the
`d`-independent `a,b`), and means set by the corresponding component of `q`.
Place the external momentum along axis 0, `q = (|q|, 0, …, 0)` (legit by
rotational invariance for the isotropic Lap / `(∇h)²` vertices). Then

    ℓ̄^{(0)} = −M⁻¹N |q|   (parallel: shifted mean),
    ℓ̄^{(α)} = 0           (transverse α≥1: zero mean),     all with Σ = (2DM)⁻¹.

So `⟨F⟩` is an **`L·d`-dimensional** Gauss–Hermite: sample each component
`ℓ_i^{(α)} = ℓ̄_i^{(α)} + (Ch·Z^{(α)})_i` with independent standard-normal blocks
`Z^{(α)}` (`Ch = chol(Σ)`), and `⟨F⟩ = Σ_grid W_g F(ℓ_g, q)`. Exact for a
polynomial `F` at GH order `≥ ⌈(deg F + 1)/2⌉` (deg `F ≤ 2·#vertices`; 1-loop ⇒
deg ≤ 4 ⇒ order 3 suffices). The base `MomFactor` already carries the Gaussian
normalization and the `1/(2π)^{Ld}` measure, so the full loop integral stays
`MomFactor · ⟨F⟩` — the **same identity as `d=1`**, just an `L·d` grid.

The form factor itself in `d≥2`: each routed momentum `p_e` (a `d`-independent
linear combo of `ℓ_i, q_j`) has components `p_e^{(α)} = Σ_i a_{ei}ℓ_i^{(α)} +
Σ_j b_{ej}q_j^{(α)}` (the *same* combo per component), and

    Lap → −|p_e|² = −Σ_α (p_e^{(α)})²,   Dx_i → i·p_e^{(i)}.

## Implementation plan

1. **`_formfactor_average`** → `d`-dim: `q_vec` is `(n_ext, d)` (or a magnitude
   placed on axis 0), `ℓ̄` is `(L,d)`, the GH grid is `L·d`-dim, `ell` is
   `(P',G,L,d)`. *Gate:* a brute `∫dᵈℓ` of a `d=2`/`d=3` derivative bubble.
2. **`diagram_form_factor`** → add `d`: build each routed momentum's `d`
   component symbols (`ℓ_{i,α}, q_{j,α}`), apply `Lap→−Σ_α p_α²`, `Dx_i→i p_i`.
3. **`_formfactor_callable`** → lambdify over the `(L,d)` loop components +
   `(n_ext,d)` externals (`q` on axis 0).
4. **`diagram_kinematic`** → drop the `spatial_dim≠1` gate; thread `d`.
5. **Wire** `d` through `compute_spatial_correlator_generic`; **validate** a
   `d=2` Model B / KPZ bubble vs brute `∫d²ℓ`, then vs a 2-D simulator.

Anisotropic single-axis composites (e.g. `d≥2` Burgers `∂_x(φ²)` along one axis)
break rotational invariance → a non-radial correlator; deferred (the isotropic
Lap / full-gradient KPZ cases are the physical d≥2 targets and stay radial).

# The Spatial Loop Pipeline (current-state reference)

*Branch `spatial-extension`, June 2026. This is the authoritative description of how
Daedalus computes spatial-field-theory correlators today. For the analytic `q→x` IFT +
Monte-Carlo + Bessel-K reduction of the loop integral see
`spatial_loop_integral_analytic_mc.md`; for the full reduction chain see
`spatial_reduction_derivation.md`; for the forward-looking momentum-native rearchitecture
see `spatial_v2_architecture.md`; for the historical planning docs that led here (now
superseded) see `docs/archive/spatial/`.*

## What it computes

For a single-field MSR-JD spatial stochastic theory

```
(∂_t + μ − D∇²) φ  =  −V'(φ) + η ,        ⟨η(x,t)η(x',t')⟩ = 2T δ(x−x') δ(t−t'),
```

it returns the two-point correlator `C(x,τ) = ⟨φ(0,0) φ(x,τ)⟩` to loop order `ℓ`,
as a perturbative ladder `C = C₀ + δC⁽¹⁾ + δC⁽²⁾ + …`. Heat-kernel building blocks:
`G_R(k,t) = θ(t) e^{−m_k t}`, `C₀(k,t) = (T/m_k) e^{−m_k|τ|}`, `m_k = μ + Dk²`.

## The principle: one genuine integral for every diagram

Every enumerated Feynman diagram — tree, tadpole, bubble, sunset, … any `k`, `ℓ`, `d` —
is evaluated by the **same** integral, with **no shortcuts** (no Dyson resummation, no
mass-shift, no bubble-vs-tadpole branch, no model-specific formula):

```
Γ(q,{τ}) = 2^(−n_C) · M(Γ) · ∫∏dt_v ∏dσ_e · e^(−μ Σ w_e) · MomFactor(w, q)
```

- **Momentum** is done analytically by Symanzik reduction and batched over the
  quadrature grid: `MomFactor = (4πD)^(−Ld/2) U^(−d/2) e^(−D qᵀQ_eff q)`. Whether the
  loop momentum couples to the external `q` (→ bubble, q-dependent) or not (→ tadpole,
  q-independent) falls out of the Symanzik polynomials automatically — it is never a
  branch in the code.
- **Time** uses **causal chambers**: the retarded θ's become integration *limits*, not a
  mask, so the integrand is smooth inside each chamber (every |Δt| sign fixed by the
  ordering) → fast convergence, no close-pair pathology. Internal vertex times are nested
  latest→earliest; each correlation line adds a Schwinger σ = z² integral concentrated at
  0 (resolves the `U^(−d/2) ∼ σ^(−d/2)` self-loop singularity).
- **Retarded + advanced**: a `{C,R}`-external diagram (retarded self-energy) contributes
  `Γ(τ) + Γ(−τ)`; a `{R,R}` (Keldysh) one contributes `Γ(τ)`. Detected from the external
  edge kinds.
- **Normalization is derived, not fitted**: the enumeration prefactor `M(Γ)·coupling`
  already carries couplings, noise amplitudes, and the combinatorial factor, so the
  kinematic integral runs with noise/coupling = 1; the universal `2^(−n_C)` (n_C = number
  of correlation edges = noise sources) converts the all-`G_R` + noise-source
  representation back to physical normalization. The tree (1 C edge, prefactor `2T`)
  reproduces `C₀` to machine precision — that pins it.

A correlation line `C` is represented as two `G_R` edges meeting at a 2-point noise source;
`diagram_to_cstack` contracts them. External legs are just `R` edges with loop-coefficient
`a = 0` (they carry only `±q`, drop out of the `∫dᵈℓ`, and contribute a plain time factor) —
uniform with every other edge. Because the integrator sums **every** enumerated diagram,
there is no place left to silently drop one.

## Data flow

```
compute_cumulants(model, k=2, max_ell=ℓ, spatial_grid=xs, …)        [pipeline/compute.py]
  │  [1/7] expand action → vertices/sources     [2/7] propagator     [3/7] mean field
  │       (shared with the temporal path — same code)
  ▼  [4/7] momentum stays symbolic (Laplacian) → skip ω pole-finder + Phase J
  │  [5/7] read per-mode (A,B,N), certify tree modes vs the shared-pipeline C(q,τ)
  │  [6/7] enumerate prediagrams → typed diagrams (enumerate_unique_diagrams) →
  │        classify M(Γ)·prefactor → map each to a C-stack descriptor → live set
  │  [7/7] for every live diagram at every 1 ≤ ell ≤ max_ell:
  │            Symanzik ∫dᵈℓ  →  causal-chamber ∫dt  →  ret+adv  →  Γ(q,τ)
  │        Σ_Γ 2^(−n_C) M(Γ) Γ(q,τ)  →  ANALYTIC q→x heat-kernel IFT  →  C(x,τ)
  ▼  returns {C_tau_x, spatial_grid, spatial_info, …}
```

Run with `verbose=True` to see the staged `[1/7]…[7/7]` trace (parallels the temporal
pipeline's staging). `compute_cumulants` is **the same function** the temporal theories
call; spatial-ness is detected at runtime (`model['spatial'] and spatial_grid is not None`)
and the back half diverges into the integrator instead of the ω-domain Phase J.

## Module map

| File | Role |
|---|---|
| `msrjd/integration/spatial/full_integrator.py` | THE integrator: `diagram_correlator`/`_value`/`_kinematic` (momentum-space, ret+adv, Symanzik chamber quadrature); the analytic-IFT analogues `diagram_correlator_x`/`_value_x` + `_symanzik_kernel_batch` (per-sample heat-kernel) + `_formfactor_average_x` (derivative-vertex moment); `diagram_kinematic(method='grid'|'mc', mc_n, mc_seed)` (switchable backend); `_momentum_factor_batch` (Symanzik, vectorized). |
| `msrjd/integration/spatial/causal_chambers.py` | enumerate the retarded-poset chambers; the smooth nested-time quadrature primitive. |
| `msrjd/integration/spatial/diagram_descriptor.py` | `diagram_to_cstack(td)` — typed diagram → C-stack (contract noise sources, classify edges). |
| `msrjd/integration/spatial/pipeline_bridge.py` | `compute_spatial_correlator_generic` (the loop orchestrator: analytic-IFT vs numerical-FT gate, the memory guard, the `SPATIAL_INTEGRATOR` backend switch), `_formfactor_callable` + `_build_wick_moment` (the joint-`(ℓ,q)` Wick form-factor moment → `ff.moment_x`), `_via_pipeline` (tree + (A,B,N) certification), `build_pipeline_records` (same `enumerate_unique_diagrams` as temporal). |
| `pipeline/compute.py` | `compute_cumulants` spatial branch + the `q→x` transforms. |
| `msrjd/integration/spatial/spatial_reduce.py` | Symanzik U/F polynomials. |

**Oracle-only (NOT on the production path)**, kept as independent cross-checks:
`loop_dyson.py`, `generic_evaluator.py`, `loop_parametric.py`, `temporal_integrate.py`.
These predate `full_integrator` and are reached only by their own tests — see their module
headers.

## Supported scope

| Axis | Supported | Deferred |
|---|---|---|
| correlator order `k` | `k = 2` (two-point) | `k > 2` (needs the multi-point external FT) |
| loop order `ℓ` | `0, 1, 2` (gated; higher works by construction but is costly) | automatic `ℓ ≥ 3` cost control |
| dimension `d` | `d ∈ {1,2,3}` for **all** vertex types — polynomial AND derivative/form-factor (the `d≥2` transverse-momentum average is the `L·d`-dim GH, validated vs brute `∫dᵈℓ` to 1e-14 at d=2) | `d ≥ 2` loops are more UV-divergent (the bare value is cutoff-sensitive — needs renormalisation, like the `d≥2` tadpole) |
| vertices | simple polynomial `φⁿ` (any degree); **composite-derivative ∇/∂ vertices** `∇²(φⁿ)`/`∂ₓ(φ²)` (Model B, Burgers), **per-leg-derivative** `(∂ₓφ)²` (KPZ), AND **any MIX of them** in one theory (per-node coupling-weighted form-factor table — Allen-Cahn⊕Model B⊕KPZ computes), generic in `ℓ` and `k` | field-degree≥3 composite (`∇²φ³`), genuine constant drift `v·∂ₓφ` in loops (integrator-gated), convolution/non-local vertices. NB a *same-signature* cross of two derivative vertices (e.g. Model B `∇²(φ²)` × KPZ `(∂φ)²`) gives a higher-degree loop form factor that can be **UV-divergent** — computed honestly, but the bare value is cutoff-dependent (needs renormalisation). |
| initial condition | stationary | transient ICs |

### Derivative (∇) vertices — momentum-space form factors

A derivative interaction vertex (e.g. the conserved Model-B `g∇²(φ²)`) deposits a
**momentum-space form factor** `F(ℓ,q)` on the loop: `Lap→−|k|²`, `∂_x→ik`, with `k`
the vertex's leg momentum (from `route_momenta`). The loop integral then factorizes
as `MomFactor·⟨F⟩`, and because `F` is a *polynomial*, the Gaussian average `⟨F⟩`
over `ℓ ~ N(−M⁻¹Nq, (2DM)⁻¹)` is computed **exactly by Gauss–Hermite** quadrature —
general in the form factor, no per-theory hardcoding (`full_integrator._formfactor_average`).
This is a **local, per-vertex feature, not bubble-specific**: a diagram's form
factor is the product over its interaction vertices,
`F = ∏_v 𝔣_chain(p_v)` (`p_v` = vertex `v`'s response-leg momentum from
`route_momenta`, `diagram_form_factor`), so vertices "wire together" for **any
loop order `ℓ` and any `k`**. The `L`-loop Gaussian average is an `L`-dimensional
Gauss–Hermite grid. Authored with `.operator_ir()`; the per-diagram `F` is
extracted by `pipeline_bridge._formfactor_callable` (mapping loop symbol `ℓᵢ→`
loop column `i`, external `qⱼ→q[j]`) and applied automatically. The conservation
law falls out: the `∇²(φ²)` tadpole gets `F=0`, the bubble `F∝q²` ⇒ `Σ(q→0)→0`.

**Validated generically:** the `L=2` form-factor momentum integral matches a brute
`∫dℓ₀dℓ₁` to **1e-14** (`test_diagram_form_factor_ell2_momentum`), confirming the
per-vertex product composes and the loop-basis↔column mapping is right at 2 loops.

**Scope:** `d=1`. Two derivative-vertex *modes*, selected automatically by the
operator-IR lowering from each vertex's base field-degree:
- **`composite`** (the ∂/∇ acts on a `φⁿ` composite — Model B `∇²(φ²)`, Burgers
  `∂ₓ(φ²)`): form factor on the response-leg momentum.
- **`perleg`** (the ∂ acts on *each physical leg* — KPZ `(∂ₓφ)²`): form factor
  `∏_legs i·p_leg`, the `ℓ·(q−ℓ)` dot-product KPZ signature.

`∂ₓ→ik` is carried complex (imaginary part drops at the real-space output). A
*first-derivative bilinear* (from a gradient nonlinearity's saddle cross-term, or
a genuine drift `v·∂ₓφ`) lowers to a propagator **drift** `V` via the drift-
generalized heat kernel (`extract_mass_diffusion → (A,B,V)`); for a homogeneous
saddle `φ*=0` (KPZ/Burgers) `V→0` and the propagator is the pure heat kernel.
`ℓ=1` runs fast end-to-end; `ℓ≥2` is *correct but expensive* (a runtime warning,
not a hard gate).  **MULTIPLE distinct derivative vertices in one theory** are
supported: the operator-IR lowering stashes a per-vertex-type TABLE
(`ns._operator_ir_vertex_terms` = each type's coupling weight `c_t/Σc`, leg
count, chain, mode), and `diagram_form_factor` sums the matching types PER NODE
(`𝔉(v)=Σ_t w_t 𝔣_t`), so a mixed diagram reconstructs every cross term while the
prefactor's merged coupling cancels the weight normalisation — exactly the
single-type behaviour when one vertex.  **Remaining (genuine, non-bespoke)
limits:** field-degree≥3 composite vertices (`∇²φ³` — a ≥3-leg/sunset topology,
gated in `theory_compiler`); a genuine constant drift
`v·∂ₓφ` with `V≠0` at the saddle (validated at the heat-kernel oracle level but
not yet wired into the Symanzik loop reduction — bridge raises cleanly).
**`d=2`/`d=3` derivative vertices are now done** (the `L·d`-dim transverse-moment
GH average, `_formfactor_average(…, spatial_dim)` + the per-component
`diagram_form_factor(…, d)`; `Lap→−|p|²`, `Dx_i→i p_i`, KPZ `(∇h)²=Σ_i(∂_i h)²`)
— see `docs/spatial_d_ge_2.md`. The remaining `d≥2` caveat is physical: the loop
form factor raises the superficial degree of divergence, so the *bare* loop is
cutoff-sensitive (needs renormalisation).

## Real-space output: the analytic `q→x` IFT

Closing the pipeline back to real space is **analytic**, not a numerical FT. After
Symanzik reduction each chamber sample's `q`-dependence is `(polynomial)×Gaussian`, so the
inverse transform is closed form:
- **plain vertices** (Phase 1): `∫dᵈq/(2π)ᵈ e^{iq·x} e^{−Bq²} = (4πB)^{−d/2} e^{−|x|²/4B}` —
  a heat kernel, summed over the chamber quadrature. No `q`-grid, no ringing, no `n_q`/`q_cut`.
- **derivative vertices, d=1** (Phase 2): the form factor's `q`-dependence folds *into* the
  transform — the FT source makes `q ~ N(ix/2B, 1/2B)` a complex Gaussian, so `δC(x)` is the
  **joint-`(ℓ,q)` Wick moment** `M_F(a,Σ,B,x)`, built symbolically once per diagram
  (`pipeline_bridge._build_wick_moment` → `ff.moment_x`, the Isserlis sum) and EXACT. This
  retired the old per-diagram polynomial-fit (which re-ran the loop average `q_deg+1`× and
  caused a ~10× ℓ=2 blow-up).

Gate (`compute_spatial_correlator_generic`): `_use_analytic = _all_plain or d==1`. `d≥2`
derivative vertices fall back to the numerical FT (kept as the validated cross-check — env
`SPATIAL_FORCE_NUMERICAL_FT=1`, `SPATIAL_Q_CUT`/`SPATIAL_N_Q`). Full derivation:
`docs/spatial_reduction_derivation.md`; analytic-vs-MC study: `docs/spatial_loop_integral_analytic_mc.md`.

## Integrator backends & the ℓ≥2 cost wall

After the analytic IFT a diagram is a Schwinger-parametric integral over the `n_V` internal
times + `n_C` correlation σ's — an `(n_V+n_C)`-D chamber/Schwinger quadrature. **ℓ=2 hits the
curse of dimensionality**: a KPZ 2-loop diagram has `n_V=4, n_C=3` ⇒ a **7-D** grid,
`≈1.8e8` points/chamber at the accuracy grid `(n_t=16, n_s=14)` — one `(P, n_x)` array is
tens of GB → OOM. Controls (all in `compute_cumulants`/the bridge via env or `method=`):

| Backend / knob | Effect |
|---|---|
| **memory guard** (always on) | estimates the worst chamber's `(P, n_x)` allocation up front; raises `SpatialPropagatorError` with the `n_V/n_C/P` numbers if it exceeds `SPATIAL_MEM_BUDGET_GB` (default 6) — ℓ≥2 can **never silently OOM-crash**. |
| `SPATIAL_INTEGRATOR=grid` (default) | deterministic causal-chamber product quadrature — the validated path; **bit-identical** regardless of the other knobs. |
| `SPATIAL_INTEGRATOR=mc` (+ `SPATIAL_MC_N`, default 1e6) | importance-sampled Monte-Carlo (`diagram_kinematic(method='mc')`): nested `Exp(μ)` time-gaps + `Exp(μ)` σ's; bounded memory, `O(1/√N)`. **Validated <0.1% for PLAIN φⁿ vertices** (the feasible ℓ=2 path for Allen-Cahn/RD where the grid OOMs); **BIASED for derivative vertices** (the `det M→0` loop-degeneracy singularity → infinite variance). |
| `SPATIAL_GRID_NT` / `SPATIAL_GRID_NS` | coarsen the loop grid (accuracy tradeoff — validate vs the simulator). |

**Practical guidance:** ℓ=1 is fast + exact for every theory; for ℓ=2, plain theories use
`SPATIAL_INTEGRATOR=mc`, derivative theories await the Bessel-K backend.

**In progress — the Bessel-K × angular-MC backend (`method='bessel'`):** does the radial
(overall-scale) `λ`-integral **analytically** via a modified Bessel function,
`∫₀^∞ λ^p e^{−aλ−c/λ}dλ = 2(c/a)^{(p+1)/2} K_{p+1}(2√(ac))` — exactly the direction the
`det M→0` singularity lives in, so it **regularizes** what breaks pure MC — and quadratures
only the smooth angular simplex. Unlocks feasible, accurate **derivative-vertex ℓ≥2**.
Foundational check (`scratch/besselk_rayfit.py`): plain integrand = a single Bessel-K
(R²=1.0, `p=−(L+1)d/2`, robust across `d=1,2,3`); derivative = a sum of Bessel-K's. Plan:
`spatial_loop_integral_analytic_mc.md` §3.

## Validation

| Test | Result |
|---|---|
| tree `Γ == C₀(q,τ)` | machine precision (≤1e-9), `tests/test_full_integrator.py` |
| `d=1` Keldysh sunset vs brute `∫dℓ₁dℓ₂` | ~1e-6–1e-4 |
| `d=2` Keldysh sunset vs brute `∫d²ℓ₁d²ℓ₂` | ~2.5e-4 |
| Allen-Cahn φ⁴ `d=1` ladder vs SPDE sim | tree 0.5 → 1-loop 0.4625 → 2-loop 0.4707, sim 0.4690 (|Δ| 0.031→0.0065→0.0017) |
| **φ⁶ generalization** (Allen-Cahn + `−γφ⁵`) | new `φ̃φ⁵` (deg-6) vertex handled with zero special-casing: γ correctly absent at tree/1-loop (degree-6 vertex needs `taylor_order = k+2·max_ell = 6` ⇒ only at `ℓ=2`), enters at 2-loop as the double-tadpole. At λ=0.05, γ=0.005 the isolated γ contribution is −0.0047, moving 2-loop from 0.4833 (φ⁴-only) to 0.4786 — 3× closer to sim 0.4797. |
| **derivative-vertex form factor** (GH vs brute) | `⟨F⟩·MomFactor` reproduces brute `∫dℓ F(ℓ,q)·Gaussian` to **1e-12** for `F∈{ℓ², ℓ²q², ℓ²(ℓ-q)², ℓ(ℓ-q)}` (Gauss–Hermite is exact for the polynomial form factor). |
| **Model-B conserved `g∇²(φ²)`** (full integrator vs oracle) | the 1-loop form-factor bubble matches the independent, sim-validated `loop_dyson` oracle to **~1%** per q (`tests/test_full_integrator.py::test_formfactor_bubble_vs_oracle`). Runs end-to-end through `compute_cumulants(max_ell=1)`. *(Note: the equal-time variance shift is conservation-suppressed — small and a weak end-to-end target; the per-q oracle agreement is the rigorous validation.)* |
| **analytic `q→x` IFT** (Wick moment vs polynomial-fit vs numerical FT) | KPZ + Model B `δC(x)`: the joint-`(ℓ,q)` Wick moment matches the polynomial-fit to **1e-12–1e-17** (both exact); both match the numerical FT (Model B `0.03337` vs numFT `0.03184` confirms the numerical FT under-resolves the `q⁴` tail). KPZ e2e `max_ell=1`: `C(0,0)=0.50109`. |
| **Monte-Carlo backend** (`SPATIAL_INTEGRATOR=mc`) | `test_mc_integrator_matches_grid_plain`; the grid path stays bit-identical (regression 16/16). PLAIN δC(x): 1-loop 0.03–0.35% (stable/5 seeds vs fine grid), 2-loop matches the converged grid to **<0.1%** where the `n_t=16` grid OOMs. e2e plain φ⁴ `max_ell=2` via MC: no OOM. |

## Notebooks

- `notebooks/spatial/pipeline_allen_cahn_1d_full_loop_sim_compare.ipynb` — φ⁴, config cell
  `MAX_ELL ∈ {0,1,2}`, cumulative per-loop progression, sim overlay, `VERBOSE` staged trace.
- `notebooks/spatial/pipeline_allen_cahn_quintic_1d_full_loop_sim_compare.ipynb` — φ⁶
  generalization test (default `MAX_ELL=2` to exercise γ).
- `notebooks/spatial/pipeline_linear_diffusion_1d_sim_compare.ipynb`,
  `pipeline_linear_field_2d_sim_compare.ipynb`,
  `pipeline_reaction_diffusion_2d_loop_sim_compare.ipynb` — earlier spatial validations.

Theory files: `theories/allen_cahn_1d_subcritical_infinite.theory.py`,
`theories/allen_cahn_quintic_1d_subcritical_infinite.theory.py`. Simulators:
`models/spatial_field_1d_sim.py`, `models/spatial_field_phi6_1d_sim.py`,
`models/spatial_field_2d_sim.py`.

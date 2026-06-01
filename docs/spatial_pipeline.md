# The Spatial Loop Pipeline (current-state reference)

*Branch `spatial-extension`, May 2026. This is the authoritative description of how
Daedalus computes spatial-field-theory correlators today. For the forward-looking
momentum-native rearchitecture see `spatial_v2_architecture.md`; for the historical
planning docs that led here (now superseded) see `docs/archive/spatial/`.*

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
  │        Σ_Γ 2^(−n_C) M(Γ) Γ(q,τ)  →  C(q,τ)  →  q→x (radial / erf)  →  C(x,τ)
  ▼  returns {C_tau_x, spatial_grid, spatial_info, …}
```

Run with `verbose=True` to see the staged `[1/7]…[7/7]` trace (parallels the temporal
pipeline's staging). `compute_cumulants` is **the same function** the temporal theories
call; spatial-ness is detected at runtime (`model['spatial'] and spatial_grid is not None`)
and the back half diverges into the integrator instead of the ω-domain Phase J.

## Module map

| File | Role |
|---|---|
| `msrjd/integration/spatial/full_integrator.py` | THE integrator: `diagram_correlator` (ret+adv), `diagram_value` (2^−n_C · M · kinematic), `diagram_kinematic` (causal-chamber quadrature), `_momentum_factor_batch` (Symanzik, vectorized). |
| `msrjd/integration/spatial/causal_chambers.py` | enumerate the retarded-poset chambers; the smooth nested-time quadrature primitive. |
| `msrjd/integration/spatial/diagram_descriptor.py` | `diagram_to_cstack(td)` — typed diagram → C-stack (contract noise sources, classify edges). |
| `msrjd/integration/spatial/pipeline_bridge.py` | `compute_spatial_correlator_generic` (the loop orchestrator, sums all `1≤ell≤max_ell`), `_via_pipeline` (tree + (A,B,N) certification), `build_pipeline_records` (same `enumerate_unique_diagrams` as temporal). |
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
| dimension `d` | general (`d = 1` validated end-to-end; `d = 2` via brute-force oracle) | `d ≥ 2` tadpole UV-cutoff polish |
| vertices | simple polynomial `φⁿ` (any degree); **composite-derivative/∇ vertices `∇²(φⁿ)`** (Model B), generic in `ℓ` and `k` via momentum form factors — see below | KPZ-type `(∂φ)²` (derivative on *physical* legs), field-degree≠2 (`∇²φ³`) / multiple distinct deriv-vertex types (compiler-gated), convolution/non-local vertices |
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

**Scope:** `d=1`; **composite-derivative** vertices (the ∇ acts on the field
composite — Model B `∇²(φⁿ)`). `ℓ=1` runs fast end-to-end; `ℓ≥2` is *correct but
expensive* (the GH grid multiplies the heavy `ℓ≥2` chamber quadrature — a runtime
warning is emitted, not a hard gate). **Remaining (genuine, non-bespoke) limits:**
per-PHYSICAL-leg derivatives (KPZ `(∂φ)²` — needs a per-leg operator map on the
vertex type, the next architectural piece); `d≥2` (transverse-momentum moments,
gated in `full_integrator`); field-degree≠2 and multiple distinct deriv-vertex
types (gated in `theory_compiler`).

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

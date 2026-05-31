# Spatial v2 — momentum-space-native diagrammatic pipeline (architecture)

Status: **design + Phase 1 (operator IR) landed.** This supersedes the v1
"symbolic-`Laplacian` bridge" (`docs/spatial_stageC5_general_integrator_design.md`),
which is kept running in parallel until v2 reproduces its regressions.

## 0. Why v2

v1 carried the spatial operator as a bare *multiplicative* symbol
`SR.var('Laplacian')` and substituted `Laplacian = −q²` at evaluation. That is
correct only on the plane-wave eigenbasis and only for a single momentum, so it
breaks for (i) derivative operators inside nonlinear vertices, (ii) non-constant
coefficients, (iii) the homogeneous-saddle annihilation (hand-patched), and
(iv) loops (one global symbol cannot hold the distinct internal momenta `q, ℓ,
q−ℓ`). v2 makes the engine **momentum-space-native**: derivatives are algebraic
operators that deposit per-leg momentum **form factors**, propagators and the
integrator carry explicit spatial frequencies, and translation-invariant
theories are the case `vertex = δ(Σk)·∏𝔣ₗ`, `loop = ∫dᵈℓ`.

## 1. Decisions (and how to reverse them)

Each decision is recorded with its rationale AND the seam where it would be
undone, so none of them is a one-way door.

- **D1 — momentum-first `(k,t)` representation** (do the spatial loop integral
  `∫dᵈℓ` first, the temporal integral second). *Rationale:* the alternative
  (time-first: build temporal poles at fixed momenta, integrate, then `∫dℓ`) was
  the v1 path and it hangs — the poles `m_{k_e}=μ+Dk_e²` sweep through
  near-degeneracy as `ℓ` varies, hitting the open `m≥3` close-pair slow path
  generically. Momentum-first removes all momentum-dependent poles before any
  time integral, so close-pair cannot arise. *Reversal seam:* the
  `spatial_reduce` / `temporal_integrate` split (§4). If momentum-first proves
  wrong, that boundary is where a different order is introduced; nothing
  upstream (IR, propagator, vertices, enumerator) or downstream (output FT)
  depends on the order.

- **D2 — temporal backend = A (reuse the Phase-J ordering polytope, integrate
  each chamber numerically).** Long-term we expect to move to **C** (full
  parametric / Schwinger) for speed. *Rationale + the three options:* see §5.
  *Reversal seam:* `temporal_integrate` is a **pluggable strategy** (A | B | C);
  swapping it touches nothing else.

- **D3 — v2 alongside v1**, retire v1 only once v2 passes the same regressions.
  *Reversal seam:* `compute.py` spatial short-circuit dispatch; both paths
  coexist behind it.

- **D4 — first end-to-end target: `reaction_diffusion_quadratic_1d`** (the φ̃φ²
  bubble), as a *regression* reproducing the validated `B=0.99` / `δC(q,0)`
  before any new-capability theory (Model B `∇²φ³`, KPZ `(∂ₓφ)²`).

- **D5 — basis scope for v2.0: d=1, infinite line + periodic ring only**;
  operator-eigenmode basis (inhomogeneous / bounded domains) deferred. Every
  signature is written for a vector `k` so d>1 is a backend, not a rewrite.

## 2. Data flow

```
theory (action with Lap,Dt,Dx + fields + BC + noise)
  1. PARSE → operator IR                              [NEW parser; DONE: spatial_operator_ir]
  2. SADDLE: solve MF PDE → φ̄                         [REUSE MF solver; inhomogeneous φ̄ deferred]
  3. EXPAND about φ̄: linearity → kill_means →         [DONE: IR passes; REUSE FieldTheory.expand]
     derived generators → multivariate Taylor
  4. PROPAGATOR K(ω,k); G=K⁻¹  (bilinear ∇²δφ → −k²)  [NEW k-explicit kernel; REUSE ω pole machinery]
  5. VERTICES: per-leg form factors 𝔣ₗ(k)             [NEW form-factor extraction]
  6. ENUMERATE diagrams (topology)                     [REUSE enumerate_unique_diagrams, M(Γ)]
  7. spatial_reduce → temporal_integrate               [REUSE/EXTEND route_momenta, Gaussian ∫dℓ;
     (∫dᵈℓ Gaussian) + (∫dt backend A)                  Phase-J polytope as the A backend]
  8. OUTPUT δC(q,ω/τ) → inverse-FT q→x → C(x,τ), S(q) [REUSE tree closed form; NEW loop q-FT]
```

## 3. The operator IR (Phase 1 — landed)

`pipeline/spatial_operator_ir.py` (+ `tests/test_spatial_operator_ir.py`, 12
tests). Operators are inert, argument-**binding** Sage function nodes
(`Lap(phi)`, `Dt(phi)`, `Dx(phi,i)`); all semantics live in explicit passes:

- `apply_linearity` — the intrinsic algebra: distribute over sums, pull out
  field- AND coordinate-free constants. `Lap(δφ·δψ)` (derivative of a product)
  and `Lap(f(x)·δφ)` (position-dependent coefficient) stay **atomic**.
- `expand_about_saddle` — `Lap(φ̄+δφ) → Lap(φ̄)+Lap(δφ)`; **linearity applied,
  mean RETAINED.**
- `kill_means` — a **separate, contingent** pass: annihilate `Lap(φ̄)` only for a
  homogeneous/stationary saddle. An inhomogeneous MF solution keeps `Lap(φ̄)` to
  cancel the rest of the stationarity condition. *Linearity and annihilation are
  two passes, by design.*
- `to_derived_generators` — lower atomic `Op(δφ)` to fresh ring generators (the
  `u=δφ, v=∇²δφ` trick); `∇⁴` collapses to one generator.
- `form_factor` — `Lap→−|k|²`, `Dt→−iω`, `Dx_i→i k_i`, composed.

Validated on Cahn–Hilliard `λ∇²φ³ → 3λφ̄²∇²δφ + 3λφ̄∇²(δφ²) + λ∇²(δφ³)` and
KPZ `(∂ₓφ)²`.

## 4. The integrator interface (the reversal seam for D1 & D2)

```
spatial_reduce(diagram, routed_momenta) ─► temporal_integrand(t₁…t_v)   # the ∫dᵈℓ Gaussian
temporal_integrate(integrand, ordering_polytope) ─► amplitude            # backend: A | B | C
```

- `spatial_reduce` does the analytic `∫dᵈℓ` (Gaussian in the loop momenta;
  polynomial form factors → Gaussian moments), returning a **smooth** function
  of the internal vertex times. This is the generalization of `loop_parametric`.
- `temporal_integrate` consumes that smooth integrand over the time-ordering
  polytope. Backend is a strategy object; A is the default.

Keeping these two as a clean boundary is what makes D1 (the order) and D2 (the
backend) reversible without disturbing the propagator, vertices, enumerator, or
output FT.

## 5. The temporal backend — A now, C later (D2 detail)

After `spatial_reduce`, a diagram is internal vertices at times `t₁…t_v` joined
by retarded `G_R(Δt)=θ(Δt)e^{−mΔt}` and correlation `C(Δt)` lines; the question
is how to integrate over the times.

- **A — reuse the Phase-J ordering polytope, numerical per chamber.** The `θ`'s
  carve time into ordering chambers (a polytope); Phase-J already enumerates
  them for any topology. v1 integrated each chamber with an analytic
  pole-residue formula whose `1/(mᵢ−mⱼ)` factors blow up at near-degeneracy
  (the close-pair bug). Because `∫dℓ` already removed the discrete poles, the
  per-chamber integrand is now **smooth**, so we integrate it by ordinary
  numerical quadrature. *General for free* (inherits Phase-J's topology
  coverage); cost is `(#vertices−1)`-dim quadrature per chamber.
- **B — explicit Dyson convolutions** (current `loop_dyson`). A self-energy is a
  sandwich `G_R⁰ ⊛ Σ ⊛ C⁰`; compute the loop `Σ(q,t)` once, convolve with the
  legs (1-D integrals, often closed form). *Fastest where applicable* but
  bespoke per topology. **Kept as a fast-path + independent oracle** (A must
  agree with B on the bubble to ~1e-6).
- **C — full Schwinger/parametric.** Write each propagator as
  `1/(−iω+m)=∫₀^∞ ds e^{−s(−iω+m)}`, one parameter per edge; do *all* loop
  integrals (`∫dℓ` and `∫dω`) analytically; the diagram collapses to a single
  smooth integral over the parameters `{s_e}` (Symanzik/Lee–Pomeransky form),
  evaluated numerically by sector decomposition. **Most general, scalable to
  high loop order, expected fastest.** Cost: the real-time/Keldysh subtlety —
  our propagators are *retarded* (causal `θ`'s, the `|t|` in `C`, real-time
  contour `i`'s), so the standard *Euclidean* Symanzik machinery must be adapted
  to causal propagators. This is the long-term target; reading list in §9.

## 6. Basis + dimension (D5)

Spatial structure = a basis `{ψ_a(x)}` diagonalizing the quadratic part; `a` is
the "generalized momentum." One interface, three backends:

| BC | label `a` | sum/integral | vertex | v2.0 |
|---|---|---|---|---|
| infinite line | continuous `k` | `∫dᵈk/(2π)ᵈ` | `δ(Σk)·∏𝔣ₗ` | ✅ |
| periodic ring | discrete `kₙ=2πn/L` | `(1/L)Σₙ` | `δ_{Σn}·∏𝔣ₗ` | ✅ (heat-kernel image sums already do this) |
| finite/inhomog. | operator eigenmodes | `Σ` modes | `∫ψψψ` overlaps | deferred |

`d=1` wired first; all signatures take a vector `k`.

## 7. Deferred layers (separable, documented)

- **Non-constant coefficients** `f(x)∇²φ` → Leibniz / `f̂(p)` momentum injection
  (a static auxiliary field) or the eigenbasis backend. The IR already leaves
  `Lap(f(x)·δφ)` atomic as the hook.
- **Inhomogeneous saddle** `φ̄(x)` → `kill_means` skipped for `Lap`; `Lap(φ̄)`
  becomes a position-dependent coefficient → same machinery as above.
- `max_ell > 1`, `k > 2`, multi-field loops.

## 8. Module map

| module | role | status |
|---|---|---|
| `pipeline/spatial_operator_ir.py` | operators, linearity, saddle, generators, form factors | ✅ Phase 1 |
| `pipeline/theory.py` (parse) | author `Lap(phi)` syntax → IR | Phase 2 |
| `…/spatial/propagator_k.py` (new) | `K(ω,k)`, fold bilinear-`v`, `G=K⁻¹` | Phase 3 |
| vertex extraction (form factors) | per-leg `𝔣ₗ(k)` from generators | Phase 3 |
| `…/spatial/spatial_reduce.py` (new) | analytic `∫dᵈℓ` (generalize `loop_parametric`) | Phase 4 |
| `…/spatial/temporal_integrate.py` (new) | backend A (reuse Phase-J polytope) | Phase 4 |
| `…/spatial/loop_dyson.py` | backend B (fast-path + oracle) | ✅ exists |
| output q→x FT | tree closed form + loop numeric | Phase 5 |

## 9. Further reading — the long-term **C** route

Parametric / Schwinger representation and its numerical evaluation:

- C. Bogner & S. Weinzierl, *Feynman graph polynomials*, Int. J. Mod. Phys. A
  25 (2010) 2585 — [arXiv:1002.3458](https://arxiv.org/abs/1002.3458). The
  canonical reference for the **first and second Symanzik polynomials** (`U`,
  `F`) read directly off a graph — the heart of the parametric representation.
- S. Weinzierl, *Feynman Integrals* (Springer, UNITEXT for Physics, 2022) —
  [arXiv:2201.03593](https://arxiv.org/abs/2201.03593). Textbook covering the
  **Feynman, Schwinger, Lee–Pomeransky, Baikov, Mellin–Barnes** representations
  side by side; the Lee–Pomeransky single-polynomial form is especially clean
  for numerics.
- S. Borowka et al., *pySecDec: a toolbox for the numerical evaluation of
  multi-scale integrals*, Comput. Phys. Commun. (2018); recent:
  [arXiv:2202.13647](https://arxiv.org/abs/2202.13647),
  [arXiv:2311.00492](https://arxiv.org/abs/2311.00492). **Sector decomposition**
  — the standard way the final parameter integral is made finite + evaluated by
  Monte-Carlo; the practical reason C can be fast/robust.

The real-time / MSR subtlety (why our case is not textbook-Euclidean):

- U. C. Täuber, *Critical Dynamics: A Field Theory Approach to Equilibrium and
  Non-Equilibrium Scaling Behavior* (Cambridge, 2014), ISBN 9780521842235 — the
  MSR–Janssen–De Dominicis response functional and how dynamic loop integrals
  (Model A/B/C, reaction–diffusion) are actually organized. Closest to *our*
  setting.
- U. C. Täuber, *Field Theory Approaches to Nonequilibrium Dynamics*,
  [arXiv:cond-mat/0511743](https://arxiv.org/abs/cond-mat/0511743) — open-access
  lecture-note version of the above; readable now.
- A. Kamenev, *Field Theory of Non-Equilibrium Systems* (Cambridge) — the
  Keldysh / closed-time-path formalism, for the retarded/Keldysh propagator
  structure that the parametric representation must be adapted to.

**Open piece for C in our context:** combining the Euclidean parametric
machinery (Bogner–Weinzierl, Weinzierl) with causal/retarded propagators
(Täuber, Kamenev) — adapting the Symanzik polynomials to the real-time contour —
is not a solved, packaged result; it is the research content of the eventual C
backend.

## 10. Build order (Phases)

1. ✅ Operator IR + linearity/saddle/generators/form-factors (landed).
2. ✅ Action → IR → (fields + derived generators) transform: `prepare_action`
   (composes the passes for both authoring conventions) + `fourier_lower` (the
   generator → `form_factor·base` bridge). Proven on `reaction_diffusion`:
   reproduces `K(ω,k)=−iω+μ+Dk²` + the `g` vertex.
3. ✅ Gated `Lap(phi)`/`Dt(phi)`/`Dx(phi,i)` authoring (`.operator_ir()`),
   threaded TheoryBuilder → field_theory `ns._operator_ir` → theory_compiler.
   The action lambda runs the IR passes; **Phase 3b-i** lowers the derived
   generators back to the v1 bare-symbol form, so a derivative-free-vertex
   theory (reaction-diffusion) becomes IDENTICAL to v1 and flows through the
   whole pipeline — `compute_cumulants` tree+bubble **bit-identical** to v1.
   Default OFF; all existing theories untouched (33 regression tests green).
   *(Remaining within Phase 3: IR-enable the `.equation()` MF text too — for now
   the saddle equation is still authored in v1 syntax alongside an IR action.)*
4. **The divergence from v1 — derivative VERTICES.**
   - ✅ **4a** `classify_generators`: split derived generators into bilinear
     (→ kernel) vs derivative-vertex (→ form factors), by per-term field-degree.
     Wired into `_lower_operator_ir_action`: bilinear lower to v1, vertex raise a
     precise Phase-4 `NotImplementedError` (validated end-to-end: a
     `.operator_ir()` Cahn-Hilliard hits the clean boundary).
   - ✅ **4b** integrator primitive: `loop_dyson` self-energy kernels take an
     optional `formfactor=F(ℓ)` (the product of the vertices' per-leg momentum
     form factors); `F=1` reproduces the bubble, `F=−ℓ²`/`−ℓ(q−ℓ)` are applied
     and validated vs direct ∫dℓ.
   - ✅ **4c-1** integrator side COMPLETE: the form factor is threaded through
     the whole bubble path — `sigma_R_time/sigma_K_time` (primitive), `_dyson_terms`
     / `bubble_delta_S` (equal-time), the vectorized `_sigma_grids` /
     `bubble_delta_C_q_tau` (time-displaced), and `compute_spatial_correlator_bubble`
     (which takes `formfactor(q) → (ℓ↦F_q(ℓ))`).  `F=None` is bit-identical to the
     validated plain bubble; a non-trivial `F` is applied and stays finite.  So
     **given `F(ℓ)`, the integrator computes the derivative-vertex bubble.**
   - ✅ **4c-2** extraction: `bubble_loop_form_factor(td, op_chain)` reads each
     interaction vertex's response-leg momentum from `route_momenta` (the vertex's
     unique outgoing edge in the all-`G_R` representation) and assembles
     `F = ∏ f_chain(p_v)` (`Lap→−p²`, `Dx→ip`).  **Validated vs hand derivation:**
     the φ̃φ² bubbles with a `Lap` chain give `F_R=q²ℓ²` (Σ_R: external q + loop ℓ)
     and `F_K=q⁴` (Σ_K: both external) — NOT the naive uniform `q⁴`.
   - ✅ **4d** sim-validation: added a conserved derivative-vertex forcing
     `g∂ₓ²(φ²)` to the 1D simulator and validated the target theory
     `∂_tφ=−μφ+D∂ₓ²φ+g∂ₓ²(φ²)+η`.  The form-factor bubble reproduces the sim at
     **B=0.944, R²=0.946** (where the plain bubble landed, 0.99), and the
     ℓ-resolved route-extracted `F` **beats** the wrong guesses (uniform `q⁴`:
     R²=0.927; `q²(q−ℓ)²`: R²=0.848). The integrator + extraction are validated
     end-to-end on a theory v1 fundamentally could not compute.
   - **Remaining within 4 (operational wiring):** connect the extraction to
     `compute_cumulants` so a `.operator_ir()` derivative-vertex theory runs the
     whole path automatically (unfold vertex generators → enumerate → extract
     `F` → form-factor bubble), instead of the current clean `NotImplementedError`.
     The pieces (extraction, integrator, sim-validated F) are all in place.
5. Output `q→x` FT; Model B / KPZ as the new capability v1 could not do.

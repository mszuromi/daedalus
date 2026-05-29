# Spatial Phase 5 — re-architecture plan (shared pipeline, momentum space)

**Status:** proposal for review — no code yet.
**Supersedes:** the bespoke `compute_spatial_correlator_tree` path and the
`docs/spatial_implementation_plan.md` §5/§5b "(t,x) heat-kernel
integrator" framing.
**Date:** 2026-05-29.

## 1. What this fixes

The overnight build took a shortcut: spatial two-point functions are
computed by a bespoke evaluator
(`msrjd/integration/spatial/spatial_correlator.py:compute_spatial_correlator_tree`),
reached via a short-circuit in `pipeline/compute.py:375` that `return`s
**before** the diagram→Phase-J path (`compute_poles_and_residues` at
`compute.py:464`, `compute_correction_td` at `:608`) is ever entered.

That bespoke path hard-codes the free tree-level 2-point structure
(noise vertex × two retarded heat-kernel propagators, convolved
analytically). It does not generalize: no loops, no k≠2 cumulants, no
multi-field coupling. It is the reason "Phase 5b" looked like a
separate effort instead of falling out of the same machinery.

**The correct architecture** (and the user's intent): a spatial theory
goes through the *same* high-level pipeline as a time-only theory —
`enumerate_unique_diagrams` → typed diagrams → `compute_correction_td`
→ `integrate_diagram`. The diagram→integral *construction* is shared;
the spatial case differs only in (a) the propagator carrying momentum
dependence and (b) an added momentum-integration layer. How those
integrals are *evaluated* efficiently (closed form vs quadrature) is a
separate, later concern.

## 2. The pivotal finding: the pipeline is momentum-space-shaped

`integrate_diagram` (`final_integral.py:2411`) builds, for each typed
diagram, the integral

```
∫_polytope  Π_edges [ Σ_α C_α · exp(λ_α · Δt_e) ]  · prefactor   d^m s
```

over internal vertex times `s_v`, where `Δt_e` is a linear form in the
vertex/external times and `λ_α = i·p_α` are the retarded ω-poles
(`final_integral.py:172`). Every analytic evaluator
(`_integrate_1d/2d/nd_modesum`, defs at `:1976 / :2175 / :1726`) assumes
this **sum-of-exponentials-in-time** structure; the per-pole-tuple
exponent `Σ_v α_v s_v + γ` is assembled in `_build_modesum_plan`
(`:648-661`).

Two ways to add space:

| | **(t, x) real space** (what I built) | **(t, k) momentum space** (pipeline-native) |
|---|---|---|
| edge factor | heat kernel `θ(t)(4πDt)^{-d/2} e^{-x²/4Dt - μt}` | `G(Δt, k) = Σ_α C_α(k) e^{λ_α(k)·Δt}` |
| fits `Σ C e^{λΔt}`? | **No** — `1/√t` and `x²/t` couple t,x | **Yes** — exactly that form at fixed `k` |
| time evaluators | need rewriting (erf m≥2 — the deferred "5b") | **reused verbatim**, per momentum point |
| new work | per-`m` erf integrators + position integrals | momentum routing + an outer momentum integral |
| internal-vertex integration | over positions `x_v` | over loop momenta `k_ℓ` |

The decisive point: **the existing evaluators consume `(t,k)` edges
with zero change** — `λ_α`, `C_α` just become functions of the edge
momentum. This is precisely what the design comment at
`final_integral.py:111-114` already anticipated:

> "…in a future spatial theory `λ_α` and `C_α` become callables of
> momentum `k`… the EdgeModeSum interface stays the same."

The `(t,x)` route is what forced the bespoke bypass — heat kernels
can't ride the evaluators, so loops would have required brand-new
m≥2 erf integrators *and* a separate position-integration layer.

## 3. Decision to confirm: pivot (t,x) → (t,k)

This reverses the v1 design choice (`docs/spatial_implementation_outline.md`
chose `(t,x)` to keep tree-level closed-form and avoid loop-momentum
integration). The re-architecture mandate — "construct the integrals
the same way as time-domain" — pushes to `(t,k)`, because that is the
representation the shared pipeline is built for.

**What the pivot costs:** loop diagrams now require numerical
loop-momentum integrals `∫dk_ℓ/2π` (the very thing `(t,x)` tried to
avoid). But that cost is unavoidable for loops in *either*
representation — the `(t,x)` route trades the momentum integral for an
internal-position integral of equal difficulty, and additionally needs
the erf-evaluator rewrite. For the canonical case (Allen-Cahn 1-loop
tadpole) the loop momentum integral is a single finite 1D integral
(`⟨φ²⟩₀ = ∫dk/2π · T/(μ+Dk²) = T/(2√(μD))`), trivial numerically or
closed-form.

**What the pivot keeps:** the heat-kernel propagator (`G_tx`, Phase
2/3) survives as the **tree-level closed-form oracle** — the exact
`C(x,τ)` it produces (validated to machine precision + simulation) is
the regression target the new pipeline path must reproduce before the
bespoke path is retired. The erf primitives may still serve as
closed-form evaluators for specific momentum integrals.

**Recommendation:** pivot to `(t,k)`. It is the only representation
that satisfies the shared-pipeline mandate, it generalizes to all loop
orders / cumulants / multi-field, and it matches the integrator's
original design intent.

## 4. The shared construction, and where momentum threads in

Grounded in the integrator map (file:line):

### 4a. Per-edge momentum (NEW — the core structural addition)
The time-domain integrator assigns a **time** to each vertex and
integrates internal times. The momentum-space analog assigns a
**momentum to each edge** with **conservation at each vertex**
(spatial translation invariance). Independent momenta = `L` loop
momenta (one per loop) + the external momentum `q`.

- Build a momentum-routing pass over the diagram graph `D`
  (`final_integral.py:2499` has `D`; edges at `:2836`): solve momentum
  conservation, express each edge momentum `k_e` as a linear form in
  `(loop momenta, external momentum)`. This parallels the existing
  `Δt_e` linear-form construction (`dt_sym` at `:2847`,
  `subset_constraint_data` at `:3407-3422`) — same idea, on momenta.
- Tree diagrams: `L=0`, every `k_e` fixed by `q` alone.

### 4b. k-dependent edge modes
`_build_edge_mode_sums` (`:150-200`) currently builds `modes` from
scalar `pole_vals`/`C_mats` (`λ_α = i·p_α`, `:172`). Make the modes a
function of the edge momentum: `λ_α(k_e)`, `C_α(k_e)`.

- `compute_poles_and_residues` (`_propagator.py:834`) already accepts
  `num_params` and substitutes into `K_ft` before root-finding
  (`_propagator.py:120`). After `Laplacian → -k²`, feeding a numerical
  `k` value through `num_params` yields the per-k poles/residues — the
  Agent-B "Option A" (no fracfield surgery). So "evaluate the edge
  modes at momentum k" = "call the existing residue machinery with k
  in num_params."
- The `EdgeModeSum` interface (`:116-147`) is unchanged per momentum
  point; only the values differ.

### 4c. Momentum-integration layer (NEW — wraps the evaluators)
For each diagram, the existing time-polytope evaluator returns the
time integral *as a function of the momenta*. Wrap it:

```
C(q, τ) = ∫ Π_ℓ (dk_ℓ/2π)  [ time-polytope-integral(edge modes @ momenta) ]
C(x, τ) = ∫ (dq/2π) e^{iqx} C(q, τ)          # external inverse FT
```

- Tree level: no `k_ℓ`; just the external-`q` inverse FT.
- Evaluation of the momentum integrals: **numerical (scipy) by
  default**, with closed-form where the structure allows (Gaussian /
  rational → the heat-kernel and erf closed forms become *evaluators*
  for these integrals, not a separate architecture).

### 4d. Noise + vertices (mostly unchanged)
- White noise enters via the propagator (`G_ft` carries the ⟨φφ⟩
  block); momentum-dependent the same way the poles are. No integrand
  change (matches the time-only treatment, Q5 of the map).
- Interaction vertices: local in space ⇒ momentum conservation at the
  vertex (already in 4a). The prefactor machinery
  (`compute.py:525`, applied `final_integral.py:2869`) is unchanged.

### 4e. The gate
`pipeline/compute.py:375` short-circuits spatial to the bespoke path.
Re-route: when `model.get('spatial')`, thread the spatial dim +
momentum metadata into `propagator_data` and proceed through
`compute_poles_and_residues` / `compute_correction_td`, with the
momentum layer (4c) wrapping the per-diagram contribution. Add the
external inverse-FT to produce `C_tau_x`.

## 5. Sequencing (each stage gated on a checkpoint)

**Stage A — tree-level through the shared pipeline.**
Route the free linear theory (Edwards-Wilkinson / linear-diffusion)
through `compute_correction_td` with the momentum layer (external `q`
only, no loops). 
*Checkpoint:* reproduce the bespoke path's `C(x,τ)` to ≤1e-10
(equal-time and two-time) — the bespoke evaluator is the oracle.

**Stage B — retire the bespoke path.**
Delete the `compute.py:375` short-circuit and
`compute_spatial_correlator_tree`; spatial tree-level now comes from
the pipeline. Keep `G_tx` closed forms as a test oracle in
`tests/`. All spatial tests still pass.

**Stage C — 1-loop.**
With momentum routing + loop integration live, the Allen-Cahn λφ³
tadpole falls out as just another diagram.
*Checkpoint:* the 1-loop `C(0,0)` matches the strict-1-loop mass-shift
prediction (`≈0.4625` at λ=0.1: `Σ=3λ⟨φ²⟩₀=0.15`, `δC=Σ·∂C₀/∂μ`) and
agrees with the simulator (the −0.071 Hartree shift the sim showed at
λ=0.3). Build the Allen-Cahn sim-comparison notebook.

**Stage D — generalize.** k≠2 cumulants and multi-field coupled
spatial follow from the same construction (no longer special-cased).

## 6. What's kept / retired

| Keep | Retire / supersede |
|---|---|
| Propagator builder (`K_ft` with `Laplacian→-k²`), Phase 2 | `compute_spatial_correlator_tree` (after Stage B) |
| `G_tx` heat-kernel closed form → **test oracle** | `compute.py:375` spatial short-circuit |
| erf / heat-kernel closed forms → momentum-integral *evaluators* | the "(t,x) erf m≥2 integrator" framing of §5b |
| Theory side (Laplacian, BC/IC, reserved names, UI) — all unchanged | |
| The whole diagram enumeration/typing pipeline — already shared | |

## 7. Risks & open questions

1. **Momentum routing implementation.** The integrator has no momentum
   bookkeeping today; 4a is genuinely new code (a conservation solve
   over the diagram graph). Medium effort, well-defined.
2. **Loop-momentum integral convergence/UV.** d=1 Allen-Cahn tadpole
   is UV-finite; d≥2 is divergent and needs regularization — out of
   scope, but the architecture should not assume finiteness silently
   (log a warning if a momentum integral doesn't converge).
3. **Cost.** A numerical `∫dk_ℓ` per loop × `∫dq` external × the
   time-polytope evaluator per momentum point. Tree level is cheap
   (one `q` integral, or the closed-form heat kernel). 1-loop adds one
   `k` integral. Higher loops scale; acceptable for v1 (1-loop).
4. **PBC.** Periodic domain ⇒ the momentum integral becomes a discrete
   sum over `k_n = 2πn/L` (the image-sum's Fourier dual). Clean in
   momentum space — arguably cleaner than the real-space image sum.
5. **Does the external inverse-FT reproduce the heat kernel exactly?**
   Stage A's checkpoint answers this; if the closed-form `G_tx` and the
   numerical `q`-FT disagree, the routing/normalization is wrong.

## 8. The decision in front of us

**Confirm the (t,x) → (t,k) pivot** (§3). Everything else follows. If
we instead keep `(t,x)`, the shared-pipeline mandate cannot be met
without writing the heat-kernel-aware m≥2 erf evaluators *and* a
position-integration layer — strictly more work, less general, and
still a partial fit. The momentum-space route is the one the
integrator was designed for.

If confirmed, Stage A is the first concrete build, with the existing
machine-precision `C(x,τ)` as its regression oracle.

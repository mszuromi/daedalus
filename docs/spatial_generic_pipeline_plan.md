# The generic spatial loop pipeline (the plan)

> **STATUS UPDATE (superseded by the full-diagram integrator).** The 6-phase plan
> below built the generic per-diagram evaluator out of *shortcuts* (a Dyson
> convolution for the bubble, a mass-shift for the tadpole), which left a genuine
> multi-vertex-tadpole gap.  Per direction to "build the whole thing, no
> shortcuts," that was replaced by **`full_integrator.py`** — ONE genuine integral
> for *every* diagram (tree, bubble, tadpole, … any k, ell, d):
> `enumerate → route → Symanzik ∫dᵈℓ → causal-chamber time integral → ret+adv sum`.
> It is wired into `compute_cumulants` and **validated**: tree `== C₀` (1e-14),
> rd φ̃φ² bubble `== loop_dyson` (~1e-4), Allen-Cahn φ̃φ³ tadpole `δ⟨φ²⟩ = −0.0375`
> (the oracle), and a φ⁴ sim-vs-theory notebook (tree 0.5 → 1-loop 0.4625 → sim
> 0.469).  Scope today: **simple (non-derivative) vertices, single field**; d=1
> validated, d general by construction.  Deferred: derivative/∇ form-factor
> vertices, the d≥2 tadpole UV cutoff, ell≥2 (sunset) validation, production speed.
> `loop_dyson` / `generic_evaluator` survive only as **independent test oracles**.
>
> The architecture below (the data flow, the C-stack components, the normalization)
> still describes the pipeline; only the per-diagram *evaluation* changed from
> shortcuts to the genuine integral.


*Branch `spatial-extension`, May 2026.  This is the authoritative plan for the
spatial loop calculation.  It replaces the **bespoke** bubble/tadpole paths with
a single generic per-diagram evaluator.  Companion: `spatial_loop_diagram_inventory.md`
(how the enumerator represents a spatial diagram) and `backend_C_math.md` /
`backend_C_design.md` (the Schwinger/Symanzik math).*

## The principle

**One path.  Evaluate every enumerated diagram the same way and sum them.**
Tree, bubble, tadpole, sunset — all just diagrams.  The bubble-vs-tadpole
distinction is *never a branch in the code*; it is purely a property of the
Symanzik output (does the loop momentum couple to the external `q`?), computed
automatically.

We follow **strict fixed order** (sum diagrams; no Dyson resummation) — the same
convention the validated time-only `compute_correction_td` uses.  In fact:

> **This is the momentum-first replacement for `compute_correction_td`** —
> identical job (evaluate a full correlator diagram's contribution to the
> cumulant), but the loop momentum is done by Schwinger/Symanzik instead of the
> broken uniform `Laplacian = −q²` substitution, and the time integral by causal
> chambers + smooth quadrature instead of the close-pair-prone exp-product
> time-polytope.

### Why "full diagram", not "self-energy + Dyson"

The enumerator emits **full** correlator diagrams (external legs + noise sources
included), not amputated self-energies.  Evaluating the full diagram → `δC`
directly means:
* no Dyson assembly, no `Σ_R`/`Σ_K` split, no 1PI-vs-1PR taxonomy;
* external legs are just **R edges with loop-coefficient `a = 0`** (they carry
  only `±q`, so they drop out of the loop `∫dᵈℓ` and contribute a plain
  `e^{−m_q·Δt}` time factor) — uniform with every other edge;
* the dropped-tadpole problem **evaporates**: there is no place left to drop a
  diagram — you sum all of them.

## The data flow

```
compute_cumulants (spatial)                          [compute.py — ONE call, all ell]
  │
  ▼   for each enumerated diagram td  (tree | bubble | tadpole | sunset | …):
  │
  │   M(Γ)·coupling  ←  classify_coefficient_factors          [EXISTING]
  │
  ├─ diagram_to_cstack(td)                                    [NEW glue — Phase 1]
  │     internal interaction vertices → integrated times {t_v}
  │     external legs                 → fixed times {0, τ} + q-injection
  │     edges  → [(a, b, kind, (u→v))]
  │        kind:  contract noise-source(2,0) → ONE C edge
  │               every other G_R edge stays an R edge (external legs: a=0)
  │        (a,b): route_momenta(td).edge_coeffs()             [EXISTING, W2]
  │
  ├─ Symanzik loop-momentum integral   (C0/C1, spatial_reduce)[EXISTING, validated]
  │     ∫dᵈℓ ∏ e^{−D k_e² w_e}  →  (4πD)^{−Ld/2} U^{−d/2} e^{−D F/U}
  │        w_e = (t_v−t_u) on R edges,  s_e on C edges
  │        ← q·ℓ cross-term in F ⇒ q-dependent (bubble); none ⇒ q-indep (tadpole)
  │
  ├─ causal-chamber time integral  (C2, causal_chambers)      [core BUILT; +Schwinger wrap]
  │     Σ_chambers ∫d{t_v} ∫d{s_e}  e^{−μ Σ w_e} · [momentum factor]
  │        chambers from the retarded poset; external times = poset bounds
  │
  ▼   diagram value  C_Γ(q, τ)
  Σ_Γ M(Γ)·C_Γ(q,τ)  →  C(q,τ)  →  q→x (radial/erf)  →  C(x,τ)   [EXISTING transforms]
```

Everything labeled EXISTING is built and validated in isolation
(`momentum_routing`, `spatial_reduce` C0/C1, `temporal_integrate` + the new
`causal_chambers` C2, and the `q→x` transforms).  The work is the NEW glue +
wiring + deletion.

### Normalization (the one subtlety to pin in Phase 2)

The enumeration prefactor (e.g. `8·T²·g²`) already carries the couplings, noise
amplitudes, AND the combinatorial `M(Γ)`.  So the C-stack must supply only the
**kinematic** part — run it with noise/coupling set to 1 (a bare C edge
contributes its Schwinger `e^{−m|Δt|}/m`, no extra `T`) and multiply by the
enumeration `M(Γ)·prefactor`.  Mixing both would double-count `T`.  This is
exactly the normalization that validation against `loop_dyson` will pin.

## Phases (each gated by a validation)

| Phase | Deliverable | Validation gate |
|------|-------------|-----------------|
| **1** | `diagram_to_cstack(td)` — pure descriptor mapping | reproduces `bubble_edges('R')` from diagram 0, `bubble_edges('C')` from diagram 1; flags diagram 2 as a tadpole self-loop |
| **2** | generic per-diagram evaluator; full bubble at fixed `(q,τ)` (Symanzik + chamber/Schwinger + external-leg poset threading) | `Σ` over bubble diagrams reproduces `loop_dyson`'s `δC(q,τ)` (B≈0.99-vs-sim, machine-precision-vs-∫dℓ) **with NO pinned `24/C_R/C_K`** |
| **3** | tadpole through the *same* evaluator (no new code) | matches analytic `⟨φ²⟩₀·∂C₀/∂A`; q-independence falls out of Symanzik `F` automatically |
| **4** | sum all diagrams + wire into `compute.py` (replace bespoke dispatch) | end-to-end reaction-diffusion `C(x,τ)` at 1-loop = Phase-2 bubble + Phase-3 tadpole; **decisive test vs simulation** (complete 1-loop vs bubble-alone) |
| **5** | multi-vertex / ≥3 chambers | 2-loop sunset via the generic path vs the brute-force `∫dℓ₁dℓ₂` oracle (W11) |
| **6** | delete the bespoke code | `compute_spatial_correlator_bubble`/`_one_loop`, `_BUBBLE_MGAMMA=24`, `loop_dyson`'s `C_R/C_K`, `_diagram_is_bubble` dispatch — all gone; `loop_dyson` demoted to a test oracle then removed; docs + full regression |

## The honest hard parts (where the risk is)

1. **External-leg → poset threading** (Phase 2): map the two external times
   `{0, τ}` onto the causal poset's fixed bounds (`_CausalPoset.scalar_lowers /
   scalar_uppers`).  The temporal `_extract_causal_poset` already does the
   analogous thing — adapt, don't reinvent.
2. **Correlation-Schwinger × chamber composition** (Phase 2/5): each C edge adds
   an `s_e ∈ [|Δt|,∞)` integral nested inside the chamber quadrature.
   `sigma_parametric` does this for 2-vertex; generalizing to the chamber core is
   the new bit — validated against the sunset oracle.
3. **d≥2 UV cutoff consistency** (Phase 3/4): the tadpole `∫dᵈℓ` is UV-sensitive;
   the Symanzik small-`w` corner is the cutoff (Regime 1).  Thread the cutoff
   consistently.

## What dies (Phase 6)

`compute_spatial_correlator_bubble`, `compute_spatial_correlator_one_loop`, the
pinned `_BUBBLE_MGAMMA = 24`, `loop_dyson`'s `C_R=4/C_K=2`, and `_diagram_is_bubble`
as a dispatch fork — **all replaced** by one evaluator where the distinctions are
automatic.  `loop_dyson` survives only as an independent test oracle until the
generic path is fully validated, then is removed.

## Status

- [x] C2-full chamber core (`causal_chambers.py`) — the multi-vertex time
      integral primitive (tests green).
- [x] Reconnaissance: the enumerator's diagram representation + the
      enumerated-diagram → C-stack mapping rule (`spatial_loop_diagram_inventory.md`).
- [ ] Phase 1 … Phase 6 (this plan).

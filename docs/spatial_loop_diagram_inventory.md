# Spatial 1-loop diagram inventory & the C-stack mapping rule

*Reconnaissance for the model-independence keystone (May 2026, branch
`spatial-extension`).  Probe: `/tmp/probe_bubble_edges.py`.*

## Why this note

The production spatial bubble path (`pipeline_bridge.compute_spatial_correlator_bubble`)
extracts the coupling by dividing the summed bubble prefactor by a **pinned**
constant `_BUBBLE_MGAMMA = 24.0`, and the Dyson collapse in `loop_dyson` uses
**pinned** `C_R = 4`, `C_K = 2`.  These are φ̃φ²-bubble topology constants.  To be
genuinely model-independent we want to evaluate each *enumerated* diagram's
self-energy directly (route → Symanzik → momentum → causal time, with `M(Γ)` from
enumeration) so those constants *emerge* rather than being pinned.  This note
records exactly how the shared enumerator represents a spatial loop diagram, so
that mapping can be built on solid ground.

## The representation: all-`G_R` + explicit noise sources

Reaction–diffusion `∂ₜφ = D∇²φ − μφ − gφ² + η`, `⟨ηη⟩ = 2Tδ`, single field,
`taylor_order = 4` (= `k + 2·max_ell` for `k=2, max_ell=1`).  The propagator is a
**1×1** matrix:

```
ring_gen_names = ['phit1', 'dphi1']      # one response (φ̃), one physical (φ)
nf (n_tilde)   = 1
G_ft = [ -1 / (D·Laplacian − 2g·φ* − μ − iω) ]      # the retarded G_R only
```

Every propagator edge is the **same `G_R`** (`edge_types[(u,v,lbl)] = ((phit,1),(dphi,1))`,
`propagator_indices = (0,0)`).  There is **no separate "C" propagator type.**  A
**correlation line `C(k)` is represented as two `G_R` edges meeting at a
noise-source vertex** — a `SourceType(bigrade=(2,0), resp=[phit,phit], coeff=−T)`,
i.e. the `⟨φ̃φ̃⟩ = 2T` insertion.  The two `G_R` edges into a noise source carry
equal-and-opposite momenta (the source injects no external momentum), so they
combine into one `C` line at that momentum.  This is just the MSR identity
`C = G_R · 2D · G_A` made graph-explicit.

## The φ̃φ² `ell=1` inventory — THREE diagrams

`enumerate_unique_diagrams(k=2, max_ell=1)` returns **three** typed diagrams,
each with `M(Γ)·prefactor` a pure coupling monomial in `T, g`:

| # | `M(Γ)·prefactor` | `_diagram_is_bubble` | structure |
|---|------------------|----------------------|-----------|
| 0 | `16·T²·g²`       | **True** (bubble)    | Σ_R-type: two interaction vertices joined by **one `G_R` (retarded) + one `C`** line; loop momentum mixes with `q` (`q−ℓ`, `ℓ`) |
| 1 | ` 8·T²·g²`       | **True** (bubble)    | Σ_K-type: two interaction vertices joined by **two `C`** lines (`ℓ`, `ℓ+q`) |
| 2 | ` 8·T²·g²`       | **False** (tadpole)  | one interaction vertex closes its two φ-legs into a `⟨φ²⟩` self-loop `C(ℓ)`; connected to the rest by a `k=0` `G_R` line → **q-independent**, 1-particle-reducible |

`_diagram_is_bubble` correctly separates these: it tests for an edge `k²` with a
nonzero mixed second partial (a `q·ℓ` cross term).  Diagrams 0,1 have `(q−ℓ)²`
edges (cross term); diagram 2's loop is a pure self-loop `ℓ²` with the connector
at `k=0` (no cross term).

### Where the pinned "24" comes from

`_live_bubbles` selects diagrams 0 and 1 (both `is_bubble` and prefactor live at
the saddle).  Their prefactor sum is `16·T²·g² + 8·T²·g² = 24·T²·g²`.  The noise
spectral weight read from the propagator is `N0 = T`, so
`g² = pref_sum / (24·N0²) = 24T²g² / (24T²) = g²` — **exact**.  The "24" is just
`16 + 8`, the φ̃φ² bubble combinatorial sum.  (`loop_dyson`'s `C_R=4, C_K=2` are
the same topology data in the Dyson-collapse normalization.)  Confirmed
machine-precision for reaction-diffusion *and* the conserved `∇²(φ²)` theory (W9).

## The mapping rule: enumerated diagram → C-stack `(a, b, kind)` edges

To feed `temporal_integrate.sigma_parametric` / the C-stack a diagram generically:

1. **Find the noise-source vertices** (`SourceType` with `bigrade=(2,0)`).
2. **Contract each noise source** into ONE **correlation `C` edge**: its two
   incident `G_R` edges carry `±k`; emit a `C` edge at momentum `k` and a factor
   of the noise amplitude (here `2T`; `sigma_parametric` already multiplies `T^{n_C}`).
3. **Keep the remaining `G_R` edges** (interaction↔interaction, or
   interaction↔external) as **retarded `R` edges**.
4. **Interaction-vertex times are the integration variables.**  For ≤2 internal
   vertices there is a single ordering chamber (`sigma_parametric`'s domain); for
   ≥3 the retarded θ's carve several chambers — that is what
   `causal_chambers.integrate_over_chambers` is for.
5. The per-edge routing `(a, b)` comes from `route_momenta(td).edge_coeffs()`
   (already validated, W2).  `M(Γ)·coupling` comes from
   `classify_coefficient_factors` (the `scalar_prefactor`).

Validation target for the mapping: reproduce `bubble_edges('R')` from diagram 0's
loop and `bubble_edges('C')` from diagram 1's loop, then confirm the summed
C-stack `Σ_R, Σ_K` + Dyson collapse reproduces `loop_dyson`'s bubble (the
B≈0.99-vs-sim, machine-precision-vs-∫dℓ reference) **without** the pinned
`24 / C_R / C_K`.

## The dropped tadpole (diagram 2) — open correctness question

`compute_spatial_correlator_bubble` computes diagrams 0,1 (via `loop_dyson`) and
**explicitly drops diagram 2** (`NOTE (scope): returns ONLY the momentum-dependent
bubble`).  But diagram 2 is a legitimate connected `ell=1` contribution to the
`k=2` cumulant (the enumerator emits it with prefactor `8T²g²`).  At one loop the
connected correlator is the linear superposition of all connected one-loop
diagrams (1PI *and* 1PR), so the complete answer should be

```
δC_total(x,τ) = δC_bubble(x,τ)  +  δC_tadpole(x,τ),
δC_tadpole    = Σ_tad · ∂C₀/∂A,   Σ_tad = g_tad · ⟨φ²⟩₀,
```

with `g_tad` **read from the enumerated tadpole diagram's prefactor** (the
validated mechanism already in `compute_spatial_correlator_one_loop`) — NOT a
hardcoded Hartree formula.  `⟨φ²⟩₀ = free_two_point(A,B,N,0,0)`.

**Resolution (decided): the generic pipeline.**  Rather than special-case the
tadpole as a second bespoke path, the tadpole and the bubble are unified into a
single per-diagram evaluator — see **`spatial_generic_pipeline_plan.md`**.  In
that pipeline there is no place to drop a diagram: every enumerated diagram flows
through `route → Symanzik → causal chambers` and is summed, and the
bubble-vs-tadpole distinction is just whether the Symanzik `F` couples `ℓ` to `q`
(computed automatically).  An attempt to *measure* the dropped tadpole's
magnitude via the temporal Phase-J path confirmed it **cannot** be evaluated that
way for φ̃φ² (2 vertices + 2 noise sources ⇒ the m≥3 close-pair hang at the
uniform `Laplacian=−q²`) — which is exactly why the momentum-first C-stack is
necessary, and why the tadpole coupling (like the bubble's, W9) must come from
the analytic enumeration, not the time-polytope.

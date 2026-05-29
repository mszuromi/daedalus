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
2/3) and the erf primitives are **not discarded — they become the
analytic momentum-integral evaluators** inside the pipeline's momentum
layer (§4c). Doing the k-integral by residues (the clean, no-ringing
path) *yields exactly the `G_tx` closed form*. They also serve as the
tree-level regression oracle (the machine-precision `C(x,τ)` the new
pipeline path must reproduce before the bespoke path is retired).

> **Ringing concern (raised + resolved 2026-05-29).** The original
> (t,x) choice was motivated by numerical ω-inverse-FT ringing. A
> numerical *k*-FFT would ring the same way on the equal-time slice
> (verified: ~1e-1 error, wrong-sign tails). But that is avoided
> entirely by doing the k-integral **analytically by residues** — the
> same technique that makes the framework's ω-treatment clean. The
> two-time slice additionally carries Gaussian `exp(-Dk²τ)` damping
> from the time integral, so even a numerical fallback is clean there.
> So (t,k) does **not** reintroduce ringing. See §4c.

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
- **Implementation note:** the routing must respect the MSR *directed*
  edge structure — response (`G_R`) and correlation (`C`) edges carry
  momentum the same way, but edge orientation sets the sign in each
  conservation equation. Read orientation off `D.edges()` /
  `propagator_indices` (the same source the time form `Δt_e = t_v − t_u`
  uses at `:2847`).
- **Dimension:** v1 is d=1, so each momentum is a scalar and the layer
  integrals are 1D. For d≥2 they become d-dimensional (with an angular
  reduction for isotropic kernels); the routing algebra is identical —
  only the integral dimension grows.

### 4b. k-dependent edge modes
`_build_edge_mode_sums` (`:150-200`) currently builds `modes` from
scalar `pole_vals`/`C_mats` (`λ_α = i·p_α`, `:172`). Make the modes a
function of the edge momentum: `λ_α(k_e)`, `C_α(k_e)`.

- **Diagonal heat-kernel case (all of v1): closed-form modes, `k` stays
  symbolic.** After `Laplacian → -k²` the inverse-propagator entry is
  `A + B·k² + iω`, with `(A, B)` the mass / diffusion *already* extracted
  in Phase 2 (`ac_mass`, `ac_diffusion`). The single retarded mode is
  written directly — `C=1, λ(k) = -(A + B·k²)` — with **no root-finding**,
  and `k` is carried *symbolically* into the analytic momentum integral
  (§4c′, residue / Gaussian path). This is the path that avoids ringing.
- **General case only (multi-field / off-diagonal, OR the numerical
  fallback): evaluate per momentum point.** `compute_poles_and_residues`
  (`_propagator.py:834`) accepts `num_params` and substitutes into
  `K_ft` before root-finding (`_propagator.py:120`); a numerical `k`
  through `num_params` gives the poles/residues at that `k` (Agent-B
  "Option A", no fracfield surgery). Used only when no closed-form
  `λ(k)` exists or the momentum integral is being done numerically —
  NOT the default.
- The `EdgeModeSum` interface (`:116-147`) is unchanged either way; only
  whether `λ` is a symbolic function of `k` or a per-point numeric value.

### 4c. Momentum-integration layer (NEW — wraps the evaluators)
For each diagram, the existing time-polytope evaluator returns the
time integral *as a function of the momenta*. Wrap it:

```
C(q, τ) = ∫ Π_ℓ (dk_ℓ/2π)  [ time-polytope-integral(edge modes @ momenta) ]
C(x, τ) = ∫ (dq/2π) e^{iqx} C(q, τ)          # external inverse FT
```

- Tree level: no `k_ℓ`; just the external-`q` inverse FT.
- **Evaluation is ANALYTIC-first (residues / closed form), NOT
  numerical FFT.** This is the same principle that makes the framework's
  ω-integral clean — it does ω by residues (→ t-domain exponentials),
  never by numerical FFT. The k-integral gets the identical treatment:
  the k-plane poles of `1/(μ+Dk²)` (at `k=±i√(μ/D)`) integrate by
  residues to the `e^{-|x|/ξ}` exponential — which *is* the
  `G_tx` heat-kernel closed form (Phase 2/3). **So `G_tx` and the erf
  primitives are not an oracle to be discarded — they are the
  momentum-integral evaluators**, slotted into the pipeline's
  momentum layer.
- **Why this matters (empirically verified 2026-05-29):** a naive
  *numerical* k-FFT of the equal-time (τ=0) slice rings badly — slow
  `1/k²` decay, Gibbs oscillation, wrong-sign tails (max err ~1e-1 at
  modest cutoff; still ~1e-2 at K=30). This is the same ringing that
  motivated the original (ω→t) → (t) choice. The two-time (τ≠0) slice
  does NOT ring — the time integral leaves a Gaussian `exp(-Dk²τ)`
  damping that kills the high-k tail (numerical FFT hits ~1e-11 at a
  tiny cutoff). So: use the closed form (residues) everywhere — exact,
  no ringing; numerical k-integration is a **guarded fallback only**,
  and never on the equal-time slice (where the closed form always
  applies). See the spike result archived in this commit's message.

### 4c′. Evaluation hierarchy + feasibility ladder

"Analytic momentum integration" is **two** tools, not one — and for
the RD / heat-kernel theory class the second is the workhorse:

1. **Residues** — for *rational* integrands: equal-time loops
   (`∫dℓ/(μ+Dℓ²)`) and the external `q→x` FT of a rational `C(q)`.
2. **Gaussian / erf closed forms** — for *Gaussian-damped* integrands.
   At fixed internal times each heat-kernel propagator is a **Gaussian
   in k** (`e^{-Dk²t}`), so a product integrated over loop momenta is a
   Gaussian integral — **closed-form at any loop order** (determinant
   formula). This is the `G_tx` / erf machinery, reused as an
   evaluator. (Unlike generic QFT's Lorentzian propagators, the
   *spatial* integral is the easy part here.)

**Committed evaluation order (per momentum integral):**

1. rational integrand → **residues** (exact);
2. Gaussian-damped integrand → **erf / Gaussian closed form** (exact);
3. else, numerical fallback → **Gauss-Hermite** (exact on
   polynomial×Gaussian; exponentially convergent — verified, see below);
4. the **external equal-time `q→x` FT is ALWAYS by residue** (it is the
   only oscillatory integral, and at τ=0 it is rational) — never a
   numerical FFT, so the ω-style ringing can never recur.

Naive numerical FFT of an oscillatory, slowly-decaying integrand is
**not** in the hierarchy — that is the ringing case, and it is
structurally avoided (step 4).

**Feasibility ladder (when analytic stops, numerical starts):**

| Loop order | Analytic status | Notes |
|---|---|---|
| tree, **1-loop** | **always closed-form** | single residue or single erf-family integral — **v1's entire scope** |
| **2-loop** | **usually** closed-form | nested erf/Dawson/incomplete-gamma; occasionally a non-elementary 2D special function → that sub-integral goes numerical |
| **3-loop+** | **generally numerical** | closed forms rarely elementary; and the pole-tuple enumeration cost (≈ poles^loops, the same `2^N` blow-up as the time-domain chain simplex) makes numerical cheaper even when a closed form exists |

Two independent axes of "feasible": (a) a closed form **exists**
(math) — yes / mostly / no by row; (b) the closed form is **cheaper
than numerics** (cost) — the residue/pole-tuple combinatorics scale
like `poles^loops`, so past some order numerical wins regardless. The
decision is **per-integral and structural**: stay analytic until the
chain produces a function the next integration can't close (or the
combinatorics blow up), then drop *that* integral to numerical —
exactly the hierarchy the time-domain side already runs (analytic
chain-simplex + `scipy.quad` fallback for the residual m=1).

**Why the numerical fallback is benign here** (verified 2026-05-29):
loop integrals are **non-oscillatory** (loop momenta integrate against
decaying propagators, no `e^{ikx}` phase) and **Gaussian-damped**, so
Gauss-Hermite converges exponentially and monotonically — 1-loop-type
`∫dℓ e^{-Dℓ²t}/(μ+Dℓ²)` reaches ~1e-6 by n=16 (t=2); a coupled 2-loop-
type 2D integral reaches ~4e-6 by n=32². No ringing, because nothing
oscillates.

**Optimization lever (order of integration).** Momentum and time
integrals commute, so the order is a free choice per diagram:
*momentum-first* is a pure Gaussian (closed-form, any loop count) but
leaves an erf-type time integral; *time-first* (the pipeline default)
keeps the time-polytope evaluators on the pure-exponential path but
leaves a rational×Gaussian momentum integral. For a purely-Gaussian
loop, doing it first maximizes analytic reach — a useful knob at
2-loop+.

### 4d. Noise + vertices (mostly unchanged)
- White noise enters via the propagator (`G_ft` carries the ⟨φφ⟩
  block); momentum-dependent the same way the poles are. No integrand
  change (matches the time-only treatment, Q5 of the map).
- Interaction vertices: local in space ⇒ momentum conservation at the
  vertex (already in 4a). The prefactor machinery
  (`compute.py:525`, applied `final_integral.py:2869`) is unchanged.
- A loop that closes two *physical* legs (the Allen-Cahn tadpole) is a
  **correlation-block** edge `⟨φφ⟩`, NOT a response `G_R`. Its
  equal-point value is the finite `⟨φ²⟩₀ = ∫dk/2π·C(k) = T/(2√(μD))` —
  not the singular `G_R(0,0)`. The typing already assigns the correct
  block via `propagator_indices`, so this is automatic; recorded
  because it is *why* the d=1 loop is UV-finite.

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

**Stage C — 1-loop (tadpole).**
With momentum routing + loop integration live, the Allen-Cahn λφ³
tadpole falls out as just another diagram. The combinatorial factor
(the `3` in `Σ=3λ⟨φ²⟩`) is the topological `M(Γ)` —
**dimension-independent**, so it is the *same* factor the framework
already validates for the time-only OU+εx³ tadpole; momentum adds no
new combinatorial risk.
*Checkpoint:* the 1-loop `C(0,0)` matches the strict-1-loop mass-shift
prediction (`≈0.4625` at λ=0.1: `Σ=3λ⟨φ²⟩₀=0.15`, `δC=Σ·∂C₀/∂μ`) and
agrees with the simulator (the −0.071 Hartree shift the sim showed at
λ=0.3). Build the Allen-Cahn sim-comparison notebook.
*Caveat:* the tadpole has **trivial** momentum routing — the self-loop
carries an unconstrained `ℓ`, the line carries `q`, and no edge is a
non-trivial combination of the two. So it validates the loop integral
+ self-energy + `M(Γ)`, but NOT the §4a routing solve.

**Stage C.5 — bubble (the routing stress test).**
Add a theory with a *quadratic* nonlinearity (a `φ̃φ²` vertex — e.g. a
simple quadratic-Langevin / reaction RD model). Its 1-loop 2-point
self-energy is a **bubble**: one propagator carries `ℓ`, the other
`q−ℓ` — the first diagram whose routing is non-trivial. This is the
actual exercise of §4a (the conservation solve must produce the `q−ℓ`
edge momentum).
*Checkpoint:* the bubble self-energy `Σ(q)` matches its closed form
(the convolution `∫dℓ/2π · C(ℓ)·G_R(q−ℓ)`, doable by residues), and the
corrected `C(x,τ)` matches a direct simulation of that theory.

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
   over the diagram graph). Medium effort, well-defined. **NB:** the v1
   canonical diagram (Allen-Cahn tadpole) has *trivial* routing and so
   does NOT exercise this code — Stage C.5 (a bubble from a `φ̃φ²`
   theory) is the actual routing test, and must land alongside any
   "routing works" claim. The combinatorial factor `M(Γ)` is
   topological (dimension-independent), so it carries over unchanged
   from the validated time-only loops — not a new risk.
2. **Loop-momentum integral convergence/UV.** d=1 Allen-Cahn tadpole
   is UV-finite; d≥2 is divergent and needs regularization — out of
   scope, but the architecture should not assume finiteness silently
   (log a warning if a momentum integral doesn't converge).
3. **Cost / analytic-feasibility ceiling.** See §4c′ for the full
   evaluation hierarchy and feasibility ladder. Summary: 1-loop is
   always closed-form (v1); 2-loop usually closed-form (occasional
   numerical sub-integral); 3-loop+ is numerical-dominated, both
   because elementary closed forms run out *and* because the
   pole-tuple residue cost scales like `poles^loops` (the same `2^N`
   blow-up the time-domain chain simplex already has). The numerical
   fallback is well-conditioned, NOT ring-prone — loop integrands are
   non-oscillatory + Gaussian-damped (Gauss-Hermite, exponentially
   convergent), and the only oscillatory integral (external equal-time
   FT) is always done by residue. So the cost/feasibility question is a
   v2+ (2-loop+) concern and does **not** gate v1.
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

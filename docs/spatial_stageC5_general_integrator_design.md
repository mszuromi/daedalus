# Stage C.5+ — the general per-edge momentum loop integrator (design)

Goal: make **any** enumerated spatial loop diagram computable — momentum-
dependent self-energies (the `φ̃φ²` bubble), higher cumulants, and N-loop —
not just the momentum-independent tadpole mass-shift the Stage-C code special-
cases. This retires the `g`-q-independence guard and the `max_ell>1` block.

## The seam (from the integrator map, 2026-05-29)

The time-domain integrator (`final_integral.integrate_diagram`) already does all
the hard temporal work, and it is **purely temporal in `k`**:

- Each propagator edge is an `EdgeModeSum`: `Σ_α C_α · exp(λ_α · Δt)` plus a
  `delta_coeff` (the instantaneous part). Built by `_build_edge_mode_sums`
  (`final_integral.py:150`) at the `integrate_diagram` call `:2866`.
- **Poles `λ_α` are currently GLOBAL** (one `propagator_data['pole_vals']` for
  every edge); only the residues `C_α = C_mats[k][pi,ri]` differ per edge.
- The δ-subset enumeration, the `Δt` linear-form constraints, and the
  time-polytope evaluator (`_build_fast_subset_evaluator_from_modes`,
  `:4296`) consume **only** the per-edge `(C_α, λ_α)` and the temporal
  constraint data. They never see momentum.

⇒ **Momentum enters a diagram ONLY through each edge's `(C_α, λ_α)`.** For the
heat-kernel field, edge `e` at routed momentum `k_e` has pole
`λ_e = −(A + B·k_e²)` (its own mass `m_e = A + B k_e²`), and the
correlation/response residue structure of that 1×1 block at mass `m_e`. So:

> Build the per-edge `EdgeModeSum`s from the **routed** `k_e²`
> (`momentum_routing.route_momenta`, already validated on real diagrams), feed
> them to the existing time-polytope evaluator, and the diagram's time integral
> `Phase_J(q, {ℓ_i}, τ)` comes out. Then do the loop integral `∫∏dℓ_i` around it.

## Architecture (chosen)

**Time-first + numerical `∫dℓ` (reuses Phase J).**

1. `route_momenta(td)` → per-edge `k_e(q, ℓ_i)`.
2. For each loop-momentum quadrature node `{ℓ_i}` (Gauss–Hermite; the loop
   integrand is Gaussian-damped → exponentially convergent, §4c′ verified):
   - per edge, `m_e = A + B·k_e²` at that node; build its `EdgeModeSum` from the
     1×1 heat-kernel block at mass `m_e` (poles/residues from a per-momentum
     `compute_poles_and_residues`, or directly from the closed heat-kernel form);
   - call `integrate_diagram` with an **`edge_mode_sums_override`** (new optional
     kwarg; defaults to the current `_build_edge_mode_sums`, so the time-only
     path is byte-for-byte unchanged) → `Phase_J(q, {ℓ_i}, τ)`;
   - accumulate with the GH weight.
3. External `q→x` FT as today (analytic for the dressed modes, else numeric).

Why time-first/numerical and not momentum-first parametric (Symanzik): the
Gaussian momentum integral is closed-form at any loop order (determinant /
Symanzik `U,F`), and that is the eventual exact path — but it is a *separate*
integrator. Time-first **reuses the validated Phase-J polytope machinery** and
adds only (a) a per-edge mode override and (b) a GH loop wrapper. The `∫dℓ`
being numerical is benign here (non-oscillatory, Gaussian-damped).

The one shared-path change is additive: `integrate_diagram(...,
edge_mode_sums_override=None)`. Everything else lives in a new spatial module.

## Validation target

`theories/reaction_diffusion_quadratic_1d.theory.py` — `∂_tφ=(D∂²−μ)φ−gφ²+ξ`,
the minimal momentum-**dependent** 1-loop self-energy. Confirmed (2026-05-29):
saddle `φ*=0`; `max_ell=1` enumerates 3 diagrams `∝ g²`, two of which are
bubbles carrying `ℓ` and `q−ℓ` (resp. `ℓ+q`), one a `φ²`-tadpole (`k=0` line).
The Stage-C tadpole code correctly rejects it (`g` is q-dependent).

## Sequencing (each gated on a checkpoint)

- **C.5a — bubble physics spike (no integrator surgery).** Compute the bubble
  self-energy `Σ_R(q,τ) = M(Γ)·g²·∫dℓ/2π G_R(ℓ,τ)·C(q−ℓ,τ)` directly; validate
  the `∫dℓ` (Gauss–Hermite vs adaptive quad) and the resulting `δC(q,τ)` against
  the analytic/closed reference. *Checkpoint:* `∫dℓ` converges; `Σ_R(q)` matches
  the convolution oracle. ← **next**
- **C.5b — `edge_mode_sums` hook + spatial loop integrator.**
  - **Hooks landed (`d8b5489` + this commit):** `integrate_diagram(...,
    edge_mode_sums_builder=…)` and `compute_correction_td(...,
    edge_mode_sums_builder_fn=…)` — both additive, default off, time-only path
    byte-unchanged (test_time_domain 21 + OU smoke green). Each `edge_info`
    entry carries its `(u,v,lbl)` key so it maps cleanly onto
    `route_momenta`'s per-edge `k_e²`.
  - **Spike (`stageC5b_loop_integrator_spike.py`):** structurally correct
    (C.5a already validated the `∫dℓ`), but a **performance WIP** — re-running
    the full `integrate_diagram` per `ℓ`-node is too slow because the 1-loop
    polytope (the `2^|E|` δ-subset sum + polygon/poset evaluation) is itself
    seconds-scale and is **`ℓ`-independent**.
  - **Remaining (the real C.5b work):** build the time-polytope ONCE and
    re-evaluate only the per-edge `(C_α, λ_α)` per `ℓ`-node — cache the
    per-subset evaluator (`_build_fast_subset_evaluator_from_modes`) and feed it
    node-dependent edge modes — so the cost per `ℓ`-node is one cheap evaluator
    call, not a full `integrate_diagram`. (Alternative: momentum-first Gaussian
    `∫dℓ` at fixed internal times.) Then `∫dℓ` by fixed Gauss–Legendre /
    Gauss–Hermite. *Checkpoint:* reproduce the Stage-C tadpole
    `δC(0.8,0.5)=−0.0447` through the override; then the bubble vs C.5a.
- **C.5c — wire into `compute.py` + retire guards.** `max_ell=1` routes any
  1-loop self-energy through the general integrator (drop the q-independence
  special case); `max_ell>1` lifts to nested `∫dℓ₁dℓ₂…`. Validate vs a direct
  simulation of the RD theory.
- **C.6 — 2-loop (sunset).** Same machinery, nested GH. Pole-tuple cost grows
  `~poles^loops`; per §4c′ ladder, drop individual sub-integrals to numerics as
  needed.

## C.5b performance findings (profiled 2026-05-29)

Per-call timings (Allen-Cahn tadpole, 1 non-zero ell=1 diagram):
- `integrate_diagram` itself: **1 ms** (the time-polytope is NOT the bottleneck —
  my earlier "rebuild the polytope per node" worry was wrong).
- `compute_poles_and_residues`: 46 ms → the override overhead (~5 per node) is
  the only real per-node cost; **cache propagator data per `k²`** removes it.
- BUT one `ℓ`-node (`ℓ=−9.95`, loop mass ≈100) took **269 s**.

Root cause: the surviving tadpole diagram has a **double edge** (both loop lines
carry `±ℓ` → identical mass → exactly degenerate poles). At MODERATE momentum
this is fast (0.11 s/node); at LARGE `|ℓ|` (large pole magnitude) the
integrator's degenerate / high-precision pole-handling path (the open
`m≥3` chain-simplex precision issue, `docs/m_ge3_precision_bug_audit.md`)
churns. So the blocker is not the polytope rebuild — it is the **existing
degenerate-pole slow path, triggered at large loop momentum**.

Tail behaviour also matters: the **tadpole** loop decouples → a `1/ℓ²`
power-law tail (slow numerical `∫dℓ` convergence, and it forces the
large-`|ℓ|` nodes that hit the slow path). It is genuinely better handled by
the Stage-C **closed form**. The **bubble** (two distinct loop edges
`ℓ`, `q−ℓ`) decays as `1/ℓ⁴` → a moderate `∫dℓ` range suffices, and only the
single point `ℓ=q/2` is exactly degenerate. **⇒ validate the general
integrator on the BUBBLE, not the tadpole.**

Practical path for C.5b cont.:
1. cache `compute_poles_and_residues` per `k²` (kills the 46 ms × 5 overhead);
2. integrate `ℓ` over the PHYSICAL range only (drop negligible large-mass
   nodes — the integrand is `≲e^{-Dℓ²τ}/ℓ²`), avoiding the large-pole slow path;
3. target the **bubble** δC vs an independent (ω,k) oracle (the C.5a `∫dℓ`
   already validates the loop-momentum integral itself);
4. for exactly-degenerate points (double edges; `ℓ=q/2`) either perturb the
   node or add an analytic degenerate-limit branch — and/or pick up the open
   `m≥3` precision fix, which this work now has a second motivation for.

## C.5c — the decisive finding: time-first is blocked, go momentum-first

Built the cached bubble loop integrator (`stageC5c_bubble_loop_integrator_spike.py`).
Two results:

1. **PLUMBING validated (oracle-free).** Forcing the override to put EVERY edge
   at the same momentum `q²` reproduces the standard global-propagator path
   **exactly** (`|Δ| = 0` at all τ). The per-edge override mechanism + the
   `k²`-cache are correct bit-for-bit. The seam works.

2. **Time-first hits the close-pair slow path GENERICALLY.** Profiling the
   bubble `∫dℓ` at `q=0.8`: `ℓ=0` (edge masses `{1, 1.64, 1}` — exactly
   degenerate, small) is fast (0.04 s); `ℓ=1` (masses `{2, 1.04, 1.64, 1}` —
   the pair `1.04, 1` **close but not equal**) hangs. This is exactly the open
   `m≥3` chain-simplex close-pair precision bug
   (`docs/m_ge3_precision_bug_audit.md`). The catch: spatial loops hit it
   **generically**, because the loop momentum makes edge masses sweep past one
   another across the `∫dℓ` — close pairs are the rule, not a special point.

**Conclusion.** The time-first path (reuse Phase J + numerical `∫dℓ`) is the
RIGHT mechanism (plumbing proven) but is gated by an unsolved precision bug it
trips on continuously. The robust general integrator is **momentum-first**:

> Do the loop integral `∫∏dℓ_i` ANALYTICALLY first. With the heat-kernel
> Schwinger rep each edge is `e^{-(A+B k_e²) w_e}`, so the momentum integral is
> a pure Gaussian → closed form at any loop order (the Symanzik `U,F`
> polynomials / a determinant). It never forms the time-polytope with close
> momentum-dependent poles, so the `m≥3` bug never fires. What remains is an
> erf-family integral over the Schwinger/vertex times — the SAME closed-form
> family the tree-level `G_tx`/erf machinery already evaluates.

So the path forward is the parametric (momentum-first) evaluator, NOT more work
on the time-first quadrature. The time-first seam stays useful for the
exactly-degenerate / decoupled cases (the tadpole) and as a cross-check.

### Next concrete build (momentum-first 1-loop)
1. For a 1-loop diagram, parametrize: response edge → Schwinger `w_e = Δt_e`;
   correlation edge → extra `∫_{|Δt_e|}^∞ ds`. Each edge becomes
   `e^{-(A+B k_e²) w_e}`.
2. Gaussian `∫dℓ` (linear `k_e(q,ℓ)`): closed form `U(w)^{-1/2}
   exp(−q²·F(w)/U(w))` (1-D; `U,F` from the routing's incidence).
3. Remaining `∫` over vertex times + correlation-edge `s` → erf family (reuse
   `heat_kernel.erf_time_integral`).
4. Validate on the bubble vs the C.5a `Σ(q,τ)` and an `(ω,k)` reference.

## C.5 pivot — validated state (momentum-first)

The momentum-first loop integral is now validated comprehensively, and the
close-pair blocker is gone (it cannot arise — no momentum-dependent poles):

- **Core** (`msrjd/integration/spatial/loop_parametric.py`, tests
  `tests/test_loop_parametric.py`, 21 passed in 0.67 s):
  `gaussian_momentum_integral(a,b,w,q,D)` = `exp(-Dq²(W-V²/U))/(4πDU)^{d/2}`,
  `U=Σa²w, V=Σab w, W=Σb²w`.
- **Self-energy kernels** (time): `Σ_R = ∫dℓ G_R(ℓ,t)C(q-ℓ,t)` (1-D Schwinger),
  `Σ_K = ∫dℓ C(ℓ,t)C(q-ℓ,t)` (2-D Schwinger) — both match direct `∫dℓ` to ≤1e-9.
- **Frequency** `Σ_R(q,ω)`: validated two independent ways — the `(ω,k)` route
  (ν-integral by residue → `Σ_R(q,ω)=∫dℓ/2π · T/[m_{q-ℓ}(iω+m_ℓ+m_{q-ℓ})]`)
  and the FT of the time kernel agree to 1e-12 on the real part (imag parts
  agree up to the `e^{±iωt}` convention). The static `Σ_R(q,0)` and equal-time
  `Σ(q,0⁺)=⟨φ²⟩₀` come out right.

### Remaining build (on this validated foundation)
1. **`Σ_K(q,ω)`** by the same `(ω,k)` residue route (convolution of two
   Lorentzians) — the second oracle piece.
2. **Dyson δC(q,ω)** = `G_R⁰Σ_R C⁰ + G_R⁰Σ_K G_A⁰ + C⁰Σ_A G_A⁰`; inverse-FT by
   residue → `δC(q,τ)`; `δ⟨φ²⟩ = ∫dq/2π δC(q,0)`. Pole-free throughout.
3. **Combinatorial factor** `M(Γ)` from the pipeline's
   `classify_coefficient_factors` on the bubble diagrams (same mechanism the
   tadpole used; the bubble prefactors are `∝ g²`).
4. **Framework wiring**: a per-diagram parametric evaluator that reads the
   diagram's edge structure (`td.prediagram`, `edge_types`, routing) and time
   structure, evaluates momentum-first, multiplies by the pipeline prefactor —
   bypassing the close-pair-prone polytope evaluator. Route `max_ell=1`
   momentum-dependent self-energies through it (drop the q-independence guard).
5. **Validate** the full bubble δ⟨φ²⟩ vs the `(ω,k)` oracle and a stable sim
   (use `φ²+φ³` so the potential is bounded).

## C.5 — bubble diagram structure mapped (the all-G_R + noise representation)

Dumped the φ̃φ² ``max_ell=1`` diagrams (``reaction_diffusion_quadratic_1d``).
The theory has ONE interaction vertex ``(1,2) g·φ̃φ²`` and ONE source
``(2,0) −T·φ̃²``.  Every propagator edge is ``G_R`` (resp ``phit`` → phys
``dphi``); a "correlation line" ``C`` is a noise source joining two ``G_R``s.
A vertex's role is read from its leg counts: 2 ``phit`` (out) ⇒ noise source;
1 ``phit`` + 2 ``dphi`` ⇒ interaction.  The three diagrams are exactly the
Dyson terms:

| diag | M(Γ) | structure | loop | external legs |
|---|---|---|---|---|
| d[1][0] | **16** | ``G_R⁰ Σ_R C⁰`` | ``G_R(q−ℓ)·C(ℓ)`` (1 noise src) | G_R + C⁰ |
| d[1][1] | **8** | ``G_R⁰ Σ_K G_A⁰`` | ``C(ℓ)·C(ℓ+q)`` (2 noise src) | G_R + G_A |
| d[1][2] | **8** | φ²-tadpole | ``⟨φ²⟩`` self-loop + ``k=0`` line | mass shift (q-indep) |

(``M(Γ)`` = scalar_prefactor / ∏coeffs, with ∏coeffs = ``g²T²``.)

This is precisely what ``loop_dyson`` computes (``Σ_R``, ``Σ_K``, Dyson), and the
``Σ_R:Σ_K`` multiplicity ``16:8 = 2:1`` matches ``loop_dyson``'s
``2·δC₁ : δC₂`` weighting — independent confirmation the assembly is right.
d[1][2] (the φ²-tadpole) is the q-independent ``A·(−T/m²)`` piece of the sim
fit; ``loop_dyson.bubble_delta_S`` is the q-DEPENDENT d[1][0]+d[1][1].

**Factor pinning** (remaining): the overall ``c·g²`` (sim ≈ 6.8) follows from
the bare all-``G_R`` diagram integral × the pipeline prefactor.  Either evaluate
one bare diagram (the per-diagram momentum-first evaluator: ∫dℓ Gaussian over the
ℓ-edges, then the vertex-time integral with retarded Heavisides — all ``G_R`` so
NO Schwinger needed) and read off ``c``, or finish the T/coupling bookkeeping.
Then wire ``compute_cumulants(max_ell=1)`` to the loop_dyson + tadpole sum.

## Risks

- **edge_info ↔ routing key match.** The override must map each `EdgeModeSum`
  to the right routed `k_e²`. `route_momenta` keys on `(u,v,lbl)`; the
  integrator's `edge_info` is built from `list(D.edges())`. C.5b must pin this
  correspondence (same edge order / explicit key) and assert it.
- **Correlation-edge mode structure.** The `⟨φφ⟩` block at mass `m_e` has the
  retarded×advanced pole pair; build it from the framework's own
  `compute_poles_and_residues` at `Laplacian=−k_e²` (don't hand-roll) so the
  residues match the time-only convention exactly.
- **Shared-path safety.** The override defaults off; the 12 time-only
  regression tests must stay green at every step.

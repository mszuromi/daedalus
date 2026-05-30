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
- **C.5b — `edge_mode_sums_override` + spatial loop integrator.** Add the kwarg;
  build per-edge modes from routing; GH `∫dℓ`. Validate the override path
  reproduces C.5a on the bubble AND the Stage-C tadpole (as a 1-node trivial
  loop). *Checkpoint:* time-only suite byte-identical; bubble matches C.5a.
- **C.5c — wire into `compute.py` + retire guards.** `max_ell=1` routes any
  1-loop self-energy through the general integrator (drop the q-independence
  special case); `max_ell>1` lifts to nested `∫dℓ₁dℓ₂…`. Validate vs a direct
  simulation of the RD theory.
- **C.6 — 2-loop (sunset).** Same machinery, nested GH. Pole-tuple cost grows
  `~poles^loops`; per §4c′ ladder, drop individual sub-integrals to numerics as
  needed.

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

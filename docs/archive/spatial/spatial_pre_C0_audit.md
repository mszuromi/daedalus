# Spatial subsystem audit — pre-C0 readiness for backend C

Read-only audit of the spatial subsystem (4 parallel agents, May 2026) to decide
what to **build on**, **edit**, or **drop** before implementing backend C
(`docs/backend_C_design.md`, `docs/backend_C_math.md`), plus a ranked set of
**easy pre-C0 wins**. This doc is the durable record + execution log.

## Headline findings

- **The one structural gap for C0:** nothing extracts the per-edge loop/external
  coefficient matrices `(aₑᵢ, bₑⱼ)` from `route_momenta.edge_momenta` — consumers
  use `edge_k2()` (squares → discards cross-terms) or hardcode the bubble's
  `[1,−1],[0,1]`. That `(a,b)` extraction is the literal missing C0 input. → **W2**.
- **Conflict adjudicated (close-pair sidecar):** the bubble path's *coupling*
  extraction **still calls** `pipeline_C_q_tau → compute_correction_td` (the m≳4
  close-pair slow path) and back-solves `g²=V_bub·m⁴/(2N0²)` — this is the q0=1.3
  hang worked around this session with adaptive `q0_samples`. (One agent thought
  this was already retired; it conflated the *form-factor* extraction, which does
  use `route_momenta`, with the *coupling* extraction, which does not.) Retiring
  it via an analytic `M(Γ)` read is a **real** win → **W9**.
- **Most of the spine is reusable.** `route_momenta` is already k- and L-general;
  `gaussian_momentum_integral` is the literal C0/C1 Symanzik seed; the
  `(μ,D,T)`/form-factor extractors are d-independent. The bespoke parts are the
  1-loop Dyson bubble assembly (keep as backend-B oracle) and the d=1 q→x / loop
  line-integrals.

## KEEP / EDIT / DROP map

**KEEP — build on directly**
- `momentum_routing.route_momenta` + `RoutingResult` — k- and L-loop-general; the C0 front-end.
- `loop_parametric.gaussian_momentum_integral` — the C0/C1 Symanzik seed (`spatial_dim` exponent already right; scalar→matrix is the only change).
- `spatial_operator_ir.form_factor`, `classify_generators`, `prepare_action`, `fourier_lower` — d-general per-leg momentum factors.
- `heat_kernel.extract_mass_diffusion`, `spatial_correlator.extract_noise_coefficients`, `pipeline_bridge.diagonal_modes_from_propagator` — the `(μ,D,T)` per-mode extractors (affine-in-`k²`, d-independent).
- `heat_kernel.gaussian_heat_kernel` — d-general position-space kernel; the q→x validation target.
- Oracle harness: `test_loop_parametric`, `test_momentum_routing` (covers **sunset/L=2 routing**), `test_spatial_pipeline_bridge`, `test_spatial_correlator`; spikes `stageC5_bubble_sim_validation` (B=0.99), `stageC5_momentumfirst_spike` (1e-12), `stageC5_derivative_vertex_validation` (B=0.944), `allen_cahn_1loop_validation`.
- Theory files: `reaction_diffusion_quadratic_1d` (III.0/III.1), `allen_cahn_1d_subcritical_{infinite,pbc}` (tadpole), `edwards_wilkinson_1d` + `linear_diffusion_test` (tree oracles).

**EDIT — reusable with specific changes**
- `loop_dyson.bubble_delta_C_q_tau` / `_sigma_grids` — keep as backend-B fast-path + III.0 oracle; bespoke-1-loop & d=1, so *mine* the causal kernel `_K`, the `a^{−1/2}` sliver, the adaptive grid/cap as the **C3-lite design pattern**, don't lift the code.
- `pipeline_bridge.compute_spatial_correlator_bubble` — **the seam**: lines ~734–747 (the `bubble_delta_C_q_tau` call + inlined cosine FT) is backend B; lift behind `temporal_integrate(strategy)`. Everything above (records/classification/form factors) and below (q→x FT) is strategy-independent. → **W7**.
- `compute.py` spatial dispatch — replace `try one_loop / except NotImplementedError → bubble` with explicit strategy selection (the `except` also swallows multi-mode errors and mis-routes them).
- `pipeline_bridge.bubble_loop_form_factor` — delegate hardcoded 1-D `Lap→−p²` to `spatial_operator_ir.form_factor`.
- `heat_kernel.build_spatial_propagator`, `spatial_correlator.free_two_point` — relax `d != 1` gate / isolate the q→x transform behind an interface (Hankel slots in later). → **W6**.
- `loop_dyson.C_R, C_K = 4.0, 2.0` + the bubble g-extraction — read `M(Γ)` analytically. → **W9**.

**DROP (or demote)**
- `loop_parametric.m_k` (dup of `loop_dyson._mk`); `sigma_R_kernel`/`sigma_K_kernel` (superseded by `_sigma_grids`; keep only as a parametric-route unit oracle after verifying no caller). → **W3**.
- spikes `stageC5b_loop_integrator_spike.py`, `stageC5c_bubble_loop_integrator_spike.py` (superseded time-first WIP). → **W3**.
- `ou_quartic_two_dim*` as a d=2 oracle — it is **two coupled 0-d OU fields, not a spatial d=2 theory**; do not mistake it for III.2 validation.

## Ranked pre-C0 wins

| # | Win | Tier | Why it helps C | Status |
|---|---|---|---|---|
| W1 | Pin-test bubble+tadpole golden numbers | XS | III.0 oracle + refactor safety net | — |
| W2 | `RoutingResult.edge_coeffs()` | XS | the missing C0 input | — |
| W3 | Delete dead dups + superseded spikes | XS | reduce surface before generalizing | — |
| W4 | Extract `symanzik_UF(...)` standalone | S | the matrix-promotion seam (C1 seed) | — |
| W5 | Cutoff first-class `cutoff` dict | S | C1's closed-vs-numerical branch keys on it | — |
| W6 | Thread `spatial_dim` through signatures | S | the d>1 seam | — |
| W7 | `temporal_integrate(strategy='B')` seam | S | the plug-point for C | — |
| W8 | Simulator `S(q)` + explicit `k_max` | S | matched-cutoff oracle (III.0/III.2) | — |
| W9 | Retire close-pair g-extraction via analytic `M(Γ)` | M | removes the only loop-path hang + scaffolding | — |
| W10 | Conserved `∇²φ²` theory as `.theory.py` | S | a first-class C2 test case | — |
| W11 | Brute-force `∫dℓ₁dℓ₂` sunset oracle | M | the III.1 (2-loop) validation oracle | — |

## Traps to carry into C

- **Cutoff mismatch is the central correctness risk.** The simulator is a *lattice*
  theory (`m_k=μ+(2D/dx²)(1−cos k·dx)`); the loop code is *continuum* (`μ+Dk²`)
  integrated to ±∞. They agree only in the low-q band → `lattice_bz` C1 must use
  the sim's dispersion; comparisons must be band-limited.
- **Close-pair re-entry via multi-pole noise.** The moment a correlation edge
  becomes a *sum over modes* (colored/Markovian/multi-field), summing it by naive
  residue differences reintroduces `1/(λᵢ−λⱼ)`. Carry it parametrically (math §4b).
- **Single-mode lock.** `len(modes)!=1 → NotImplementedError` at three guard sites
  (`pipeline_bridge` 469–471, 610–612; tadpole path) — multi-mode must thread
  through all three.
- **Coverage thinnest where C0 starts.** The matrix Symanzik (L≥2) has *no* test;
  *no test imports the simulator*. W1 + W8 are the safety net, not optional.
- **`gaussian_momentum_integral` raises on `U≤0`.** Correct for the continuum; the
  Gaussian-edge cutoff (`w_e += σ²/D`) keeps `U>0`, but hard/lattice can probe
  `w→0` — the raise must become regime-aware once the cutoff is plumbed.

## Execution log

Branch `spatial-extension`. Overnight autonomous run (May 2026):

**DONE + validated (162 tests green across the full spatial+temporal suite):**
- **W1** golden-number pins (`test_loop_dyson`): bubble_delta_S(q)×5, bubble_delta_phi2,
  bubble_delta_C_q_tau — the III.0 oracle + refactor net.
- **W2** `RoutingResult.edge_coeffs()` (`momentum_routing`) — the C0 input; tests reconstruct
  tree/bubble/sunset momenta + rank checks.
- **W3** dropped the dead `loop_parametric.m_k` dup (kept `sigma_*_kernel` as the C1 oracle).
- **W4** extracted `loop_parametric.symanzik_UF(...)` (the C1 seed); `gaussian_momentum_integral`
  is now its thin wrapper; bubble-reduction test.
- **W8** simulator `meta['k_max']` + `structure_factor(snaps,meta)` (the matched-cutoff oracle);
  `test_spatial_sim` (3) closes the 'no test imports the sim' gap.
- **W10** `theories/reaction_diffusion_conserved_1d.theory.py` (the ∇²φ² derivative-vertex
  theory as a first-class file; builds + sanity passes).
- **W11** the 2-loop brute-force `∫dℓ₁dℓ₂` sunset oracle (embedded in `test_spatial_reduce`).
- **C0** `spatial_reduce.symanzik_polynomials` → (U=det M, Q_eff) from edge_coeffs; matches the
  hand bubble & sunset polynomials (math §6).
- **C1** `spatial_reduce.momentum_integral` (L-loop); = `gaussian_momentum_integral` at L=1,
  = brute-force ∫dℓ₁dℓ₂ at L=2 (1e-6).
- **C2** `temporal_integrate.sigma_parametric` (2-vertex causal time-simplex; Gauss–Laguerre for
  nC≥3); 1-loop bubble Σ_R/Σ_K = backend B (III.0 oracle), 2-loop sunset = direct ∫dℓ₁dℓ₂.

**Result:** C0→C1→C2 form a working finite-scale loop evaluator at L=1 AND L=2, d-general,
validated against backend B (the III.0 oracle) and direct momentum integrals.

**W9 — DONE + validated.** Retired the close-pair g-extraction sidecar:
`compute_spatial_correlator_bubble` no longer calls `pipeline_C_q_tau →
compute_correction_td` (the m≳4 hang). Coupling read analytically from the diagram
prefactor — the φ̃φ² bubble has `Σ M(Γ)·prefactor = 24·N0²·g²`, so `g² = pref/(24·N0²)`.
The `24` verified to machine precision vs the old numerical `V_bub` for BOTH
reaction-diffusion (g=0.35; g=0.2,T=2) AND the conserved ∇²(φ²) theory (g=0.3,D=2),
invariant under g,T,D (a pinned topology constant like `c_R=4/c_K=2`). No
compute_correction_td on the spatial loop path now ⇒ no hang at any mass/D; faster.
Tests green: bridge (analytic g=g_true, spread 0, δC unchanged), operator_ir 22.

**d>1 building blocks — DONE + validated.** The audit's d>1 study assumed the
d-dim loop needs numerical angular quadrature; it does NOT — C1's analytic Symanzik
reduction does `∫dᵈℓ` closed-form (d only in the `U^{−d/2}` exponent). Validated:
C1 momentum_integral at d=2 vs brute-force `∫d²ℓ` + dimension factorization (d=1–4);
C2 `sigma_parametric` bubble Σ_R at d=2 vs direct `∫d²ℓ`; and the EXTERNAL q→x
output transform `spatial_correlator.radial_inverse_ft` (d=1 cosine, d=2 J₀ Hankel,
d=3 sinc) vs the closed-form free correlators (`K₀`/Yukawa). So both d>1 building
blocks — the self-energy and the output transform — are proven; the d>1 loop is a
parameter flip, not a re-derivation.

Also landed: the **2-D simulator** `models/spatial_field_2d_sim.py` (the d=2 oracle;
S(k)=T/ω_k vs the exact lattice + variance vs the lattice sum, validated); and the
**d=2 bubble assembles end-to-end** through the stack (`bubble_delta_equal_time_via_C`
+ `bubble_delta_phi2_via_C` gained `spatial_dim`; d=2 δC(q,0) finite/positive/exact
g²-scaling/genuinely d-dependent — correct by composition). Remaining d>1: (i) the
full-resolution d=2 δ⟨φ²⟩-vs-2D-sim study (perf-gated on a vectorized `sigma_parametric`
Σ_K; sim is also tadpole/bubble-conflated); (ii) wiring d=2 through `compute_cumulants`
(the bridge/dispatch gate d≠1 today).

**d=2 TREE through `compute_cumulants` — LANDED + reviewable.** Relaxed the
`heat_kernel` d≠1 gate (d∈{1,2,3}); the bridge + `total_C` use `radial_inverse_ft`
for the q→x at d≥2; sped up Σ_K (Gauss–Laguerre, ~2×). A d=2 linear theory runs
through the PUBLIC `compute_cumulants(max_ell=0)` and `C(r,0)` matches the exact
`K₀` to ~1% (r≤3); demo notebook `notebooks/pipeline_linear_field_2d_sim_compare.ipynb`
(vs `K₀` + the 2-D sim). The 1-loop **bubble** at d=2 (`max_ell=1`) is the next
increment — route `compute_spatial_correlator_bubble` through the C-stack
(`sigma_parametric` d=2 + `radial_inverse_ft`) instead of the d=1 `loop_dyson`.

**Deferred (next session, with review):**
- **W5** cutoff-dict (`gaussian_edge|hard_spherical|lattice_bz`) + **W6** `spatial_dim` threading
  through the OLD bubble path. (The NEW C0/C1/C2 code is already d-general; the cutoff threads
  naturally into C2's `momentum_integral` — these wins are old-path setup.)
- **W7** wire C2 into `compute_spatial_correlator_bubble` as a selectable backend (the C2 module
  IS the seam; routing the bridge to it = C3-lite/C4 = the Dyson assembly on top of C2's Σ —
  the next milestone beyond C2).
- Then: C3-lite (finite-cutoff δC end-to-end vs sim at matched cutoff), multi-vertex ordering
  chambers, the d=2/3 path.

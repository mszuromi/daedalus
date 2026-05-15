# Phase J refactor — Stage 3 / Stage 4a notes

This document captures the analytic-integration refactor of Phase J
(vertex-time integration) on the `phase-j-refactor` branch, leading
up to the pre-numba checkpoint tagged `phase-j-stage-3b-analytic`.

## Problem

The pre-refactor Phase J path integrated each typed-diagram subset
via `scipy.nquad` on a Sage-compiled `fast_callable`.  For
higher-loop / higher-`k` configurations this was the dominant cost:

* `k=2, max_ell=1` singlepop quad: 13.5 min parallel.
* `k=1, max_ell=2` singlepop quad: did not finish in 30+ min;
  diagnostic-mode serial run hit **5.7 hours**.

A cProfile of the 2-loop case showed 95% of wall time in
`scipy.nquad → _qagse`, with 5.7 B Heaviside-filter calls
wrapping 1.5 B actual evaluator calls (73% sampling waste).

## Approach

Replace `scipy.nquad` with **closed-form analytic integrators** at
every value of `m` (= number of integration variables surviving
δ-elimination):

| m | Path | Closed-form |
|---|---|---|
| 0 | trivial | evaluate integrand at free externals |
| 1 | interval modesum | `∫_L^U Σ_α C_α exp(λ_α·Δt) ds`, pole-tuple expansion + per-term 1D integral, ±∞ bounds handled exactly |
| 2 | polygon modesum | fan-triangulation of feasible polygon + closed-form `∫∫ exp(α·s_0 + β·s_1) dA` on each triangle |
| ≥ 3 | poset / chain simplex | causal-poset extraction → linear-extension enumeration → 2^N closed form for nested ∫ exp(α·s) on each chain |
| ≥ 3 | poset, polynomial extension | when cumulative pole sums `β = α_inner + α_outer` vanish, carry a polynomial prefactor `(s − L)^k` through the recursion |
| ≥ 3 | poset, intermediate-upper extension | when scalar upper bounds sit on non-maximal poset elements, decompose the chain into 2^q case-tuples of independent chain simplexes |

Each integrator is **mathematically exact** for rational propagators
(no quadrature error).  scipy.nquad remains as a fallback for
non-local cumulant kernels (`NoiseSourceType` vertices).

## Stages (in commit order on `phase-j-refactor`)

### Stage 0 — validation harness  (`189522f`)
Regression fixtures: 3 frozen `.npz` reference values
(`spike_reset_k1_ell1`, `spike_reset_k2_ell0`, `quad_exp_k2_ell0`)
re-evaluated and compared bit-identical on every subsequent commit.

### Stage 1a — pre-prune δ-subset enumeration  (`23ba93a`)
Drops subsets where any δ-edge has zero coefficient before
expensive symbolic work.

### Stage 1b — Heaviside filter opt-in  (`27b9d63`)
`DEBUG_HEAVISIDE_GUARD` flag.  Default off in production.

### Stage 2 — `EdgeModeSum` canonical representation  (`077825b`)
Pre-compute pole/residue mode-sum cache per edge at the top of
`integrate_diagram`; subset evaluators consume the cache instead
of re-extracting per call.  Spatial-extension-ready.

### Stage 3a — m=2 polygon integrator  (`60755e1`)
* `_integrate_2d_polygon_modesum(smooth_edge_modes, prefactor,
  subset_constraint_data, free_ext_vals)`
* Sutherland-Hodgman half-plane clipping → fan triangulation.
* Per-pole-tuple analytic ∫∫ over each triangle.
* `USE_POLYGON_M2_INTEGRATOR` flag (default True).
* 20 unit tests.

### Stage 3b — causal-poset m≥3 integrator
* **Prep** (`212e2cf`): `_extract_causal_poset` extracts the DAG
  on integration variables from retardation constraints;
  `_enumerate_linear_extensions` yields all topological sorts.
* **Nested** (`cd60991`): `_exp_over_chain_simplex(alphas, L, U)`
  closed-form 2^N-term sum for the nested integral
  ∫…∫ exp(α₁s₁ + … + αₙsₙ) on `L ≤ s₁ ≤ … ≤ sₙ ≤ U`.
* **Wire** (`5e89bc0`): `_integrate_nd_polytope_poset_modesum`
  routes per-tuple, per-extension into `_exp_over_chain_simplex`.
* **Correctness fixes** (`0d09cd9`): empty-simplex `upper ≤ lower`
  → 0 return; maximality check bail (later replaced — see below).
* **Overflow guard** (`aa05954`): per-term `EXP_REAL_LIMIT = 600`
  for both polygon and poset paths.

### Stage 4a-grouped — analytic merged-residue path  (`b32c50a`, `0e0bb15`)
Add `pole_tuples=` override parameter to the m=2 polygon and m≥3
poset analytic integrators.  Lets the grouped Phase J path
(`pipeline/_grouped_phase_j.py`) inject merged residues
`B_α = Σ_td cp_td · Π_e C^(td)_{α_e, e}` in place of the per-edge
Cartesian product.  Brings the prototype grouped path from
scipy.nquad (~1e-8 rel agreement with per-diagram) to machine
precision.

### Stage 4a-perdiag — analytic m=1 in per-diagram path  (`ed1eb39`)
`_integrate_1d_polytope_modesum` — closed-form 1D interval
integral with exact ±∞ bound handling.  Replaces scipy.quad on
the pole-residue closure for m=1 subsets.  10 unit tests.

### Stage 3b-extended — polynomial-prefactor chain simplex  (`09d937c`)
`_exp_over_chain_simplex_polynomial` extends the standard chain
closed form to handle degenerate β (cumulative-pole-sum
vanishing) by carrying a polynomial prefactor `(s − L)^k` through
the recursion.  Each degenerate level adds one to the polynomial
degree; non-degenerate levels use the closed-form
`∫(s−L)^k · exp(β·s) ds` via integration-by-parts.  Caller
(`_integrate_nd_polytope_poset_modesum`) falls back to this path
when the standard chain simplex returns None on degenerate β.
21 unit tests.

### Diagnostic counters  (`5c9909d`)
`_RUNTIME_COUNTERS` module-level dict + `_reset_runtime_counters()`
helper.  Increments at every analytic-path decision point so a
notebook can distinguish *intent* (`_evaluator_label`, set at
subset setup) from *runtime* (whether the analytic path actually
completed or silently fell back to scipy).

### Stage 3b — tight bounds + intermediate-upper handling  (`2a4d113`)
The Cell C runtime counters revealed two large remaining scipy
fallback sources in the user's k=2 max_ell=1 spike-reset run
(116 / 144 m≥3 subsets):

* **Tight physical lower** — fall back to `min(free_ext_vals) −
  POSET_PHYSICAL_MARGIN` (margin = 50.0) instead of
  `-POLYGON_BBOX_CAP = -200` when no scalar lower is extracted.
  Prevents `exp(β · L)` overflow on chains with cumulative pole
  sums `|Re β| > 3` (16 subsets recovered).

* **Intermediate-upper chain simplex** —
  `_chain_with_intermediate_uppers(alphas, L, upper_per_position,
  U_chain_top)` decomposes the chain integral when scalar uppers
  sit on non-maximal poset elements.  For `q` cuts the integral
  is a sum over cut-position tuples `(c_0, …, c_{q-1})` (each
  `c_i ≥ p_i`, the original constraint position; tuples
  monotonic in chain position) of products of independent chain
  simplexes.  Replaces the previous "maximality bail" with full
  analytic coverage (100 subsets recovered).  9 unit tests.

Result: post-fix counter shows **0 m≥3 scipy fallbacks** on the
same config; k=2 max_ell=1 spike-reset full pipeline drops from
unknown (would have been > 1 hour) to ~37 min on 12 cores.

### Notebook diagnostics
* **Section 3.6** added to both singlepop notebooks
  (`58c42e7`, `7fc2210`) with three opt-in diagnostic cells:
  parallelism sanity check, cProfile snapshot, runtime path
  counters (`3640284`, `203797b`).  Cells are wrapped in
  `if False:` by default — flip to `True` to enable individually.
* **Per-order cumulative plotting** (`d4d6eeb`, `7fc2210`):
  the theory side shows separate bars / lines for tree, tree+1-loop,
  tree+1-loop+2-loop, … with accurate labels.  Per-order residual
  breakdown in the residual cell.

### Other fixes
* `subset_bits → branch_bits` typo in `integrate_diagram` error
  messages (`56f4871`) — masked the real error when an unexpected
  free symbol slipped through `num_params`.
* `.gitignore` for `*.prof` (`4eaadcf`).
* τ-grid count correction in Cells B and C (`849ed38`).

## Performance summary

| Config | Before refactor | After Stage 3b (this checkpoint) | Speedup |
|---|---|---|---|
| k=2 max_ell=1 singlepop quad | 13.5 min parallel | 4.8 min parallel | 2.8× |
| k=1 max_ell=2 singlepop quad | did not finish (5.7 h serial single-τ) | 18 min parallel | > 20× |
| k=2 max_ell=1 singlepop spike-reset | (previously DNF) | 37 min parallel | — |

All 100 Phase J tests pass on this checkpoint:

```
sage -python -m pytest tests/test_polygon_m2_integrator.py \
    tests/test_causal_poset.py tests/test_chain_simplex_polynomial.py \
    tests/test_chain_intermediate_uppers.py \
    tests/test_1d_polytope_modesum.py \
    tests/test_grouped_vs_perdiag.py \
    tests/test_phase_j_refactor_regression.py -q
```

## Knobs

Module-level flags in `msrjd.integration.time_domain.final_integral`:

* `USE_POLYGON_M2_INTEGRATOR = True` — m=2 analytic polygon.
* `USE_POSET_INTEGRATOR = True` — m≥3 analytic poset/chain.
* `USE_1D_INTEGRATOR = True` — m=1 analytic interval.
* `POSET_PHYSICAL_MARGIN = 50.0` — lower-bound fallback margin.
* `POLYGON_BBOX_CAP = 200.0` — upper-bound fallback (retarded
  poles underflow safely here).
* `QUAD_OPTS = {'limit': 200}` — scipy.nquad opts for residual
  fallback cases.

Module-level flag in `msrjd.integration.time_domain.grouped_integral`:

* `USE_GROUPED_ANALYTIC_MODESUM = True` — grouped path's analytic
  merged-residue route.

## Revert / checkpoint

This state is tagged `phase-j-stage-3b-analytic`:

```bash
# return to this exact state
git checkout phase-j-stage-3b-analytic

# or branch off from it
git checkout -b my-experiment phase-j-stage-3b-analytic
```

The next planned work is `numba`-compiling the chain simplex hot
loop (`_exp_over_chain_simplex` and
`_exp_over_chain_simplex_polynomial`).  Expected 30-100× per chain
evaluation, ~5-10× wall time.  The Python implementations stay in
place as the reference / fallback; the numba version will be a
new function selected via a flag, so revert is just a flag flip.

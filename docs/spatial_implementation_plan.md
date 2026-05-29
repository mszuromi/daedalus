# Spatial field theories — unified code-change plan (v1)

**Companion to** `docs/spatial_implementation_outline.md` (the *what*
and *why*).  This document is the *how*: a single ordered list of
code changes synthesized from three parallel audit passes (Agent A:
theory-side, Agent B: propagator-side, Agent C: Phase J integrator)
plus a fourth, integrative pass that resolves the conflicts and
sequencing those audits left open.

*Last updated: 2026-05-28.*

## How to read this document

Each phase below has the same five sub-sections:

| Sub-section | Use |
|---|---|
| **Inputs**          | What must already exist before this phase can start |
| **Output**          | The user-visible / framework-visible artefact this phase produces |
| **Touch points**    | Concrete `file.py:line` ranges and what to do at each |
| **LOC + risk**      | Effort sizing and what could go wrong |
| **Checkpoint test** | What must pass for the phase to be called done |

Phases are ordered so that each one's *Inputs* are produced by an
earlier phase's *Output*.  Phases marked **[Sequential]** strictly
block the next; phases marked **[Parallelizable]** can be worked
concurrently with the prior phase by a second hand.

## Critical-path findings up front

These are findings from the audit pass that change the v1 scope or
order before any code is touched:

1. **Heat-kernel factors break the fast Phase J integrators *as
   currently written* (Agent C) — but the wall is rescueable.**
   The existing closed-form `_exp_over_*` evaluators in
   `msrjd/integration/time_domain/final_integral.py` assume the
   integrand exponent is *linear in s* (the integration variable).
   Heat-kernel factors `Δt^{-1/2} · exp(-Δx²/(4 D Δt))` are not
   linear-in-Δt; the `_exp_over_chain_simplex`,
   `_exp_over_triangle`, and `_integrate_1d_polytope_modesum`
   closed forms cannot evaluate them *without modification*.

   **But heat-kernel-times-exponential is still a closed-form
   integrable family** — just one rung up the special-function
   ladder.  Concretely, after spatial collapse, integrals of the
   form

   ```
   ∫_L^U  T(s)^(-1/2)  exp[ -X² / (4D T(s))  -  μ T(s) ]  ds
   ```

   (with `T(s)` linear in `s`) have a closed form in terms of
   `erfc(·)` via the standard substitution `v = √(α u) ± √(β/u)`;
   the infinite-range version is the K_{1/2} Bessel identity.  The
   existing fast evaluators bottom out in
   `exp(λ·U) − exp(λ·L)`; the heat-kernel-aware versions bottom
   out in `erfc(α + β/√U) − erfc(α + β/√L)`.  Same structural
   recursion, different leaf functions.

   **Consequence for v1 scope**: see §"Heat-kernel rescue paths"
   below.  Three concrete rescue approaches exist; the choice
   between "scipy.quad fallback at m=1" (the baseline plan, ships
   fastest) and a rescue tier is a Phase 0 decision.  The
   recommended tier (Rescue A — Path 1 m=1 erfc) costs only one
   extra week, kills scipy.quad in v1 entirely, and lifts the
   precision floor from ~1e-8 to ~1e-15.

2. **Spatial-extension dependency chain is strictly serial through
   Phases 2-5 (Agents A, B, C agree).**  Each downstream phase
   consumes a data structure that the previous phase produces:

   ```
   Phase 1 (Theory)
     │   produces:  model['spatial'] block, ft._all_field_sr_vars
     │              with Laplacian symbol registered
     ▼
   Phase 2 (Propagator builder)
     │   produces:  propagator dict gains G_tx, k_var, spatial_dim,
     │              bc_mode, bc_params
     ▼
   Phase 3 (PBC image sums)
     │   produces:  G_tx wrapped in Σ_n G_inf(t, x + nL) when PBC
     ▼
   Phase 4 (IC machinery)        ← Parallelizable with Phase 5
     │
     ▼
   Phase 5 (Phase J spatial)
     │   produces:  vertex_position tracking, spatial evaluator
     │              wired before time evaluator
     ▼
   Phase 6 (Output API + tests)
   ```

   The only phase that can be parallelized is Phase 4 (Initial
   Conditions) — its code touches `pipeline/compute.py` and the
   theory-side serializer but does not depend on or feed the
   propagator/integrator chain.

3. **Three open design questions need user signoff before Phases 1-2
   can start** (one from each agent).  Listed in §"Open questions"
   at the end of this document.  Resolving them is Phase 0.  A
   fourth Phase 0 decision is which heat-kernel rescue tier (if
   any) to bake into v1 — see §"Heat-kernel rescue paths" below.

## Heat-kernel rescue paths

Three concrete approaches to extending the fast evaluators to
handle heat-kernel-times-exponential integrands.  Listed in order
of increasing structural change.  These are alternatives /
supplements to the "scipy.quad fallback at m=1" approach the
baseline Phase 5 plan uses.

### Path 1 — re-derive each evaluator's leaf integrals with erfc

The direct route: rewrite the closed-form leaves at each `m` so
the recursion bottoms out in `erfc(·)` instead of `exp(λ·U) −
exp(λ·L)`.

- **m=1** `_integrate_1d_polytope_modesum` → one change of
  variables (`v = √(α u) ± √(β/u)`), two `erfc` calls per pole.
  ~200 LOC, ~1 week.  **Sufficient on its own for Allen-Cahn
  1-loop in d=1**, because the spatial Gaussian convolution always
  collapses the loop's spatial coordinate first, leaving only an
  m=1 temporal residual after the spatial integral.
- **m=2** `_integrate_2d_polygon_modesum` → fan triangulation
  already separates the polygon; on each triangle, change variables
  so the time-direction integrates to `erfc` and the orthogonal
  direction collapses (Gaussian in the heat-kernel case too).
  ~600 LOC, ~1.5 weeks.  **Open question** (to resolve before
  sizing tightly): whether the orthogonal direction always
  collapses cleanly — Agent C's audit suggested it does but I'd
  want a one-page derivation pass to confirm.
- **m≥3** `_exp_over_chain_simplex` → 2^N recursion still works,
  but each split produces an `erfc` ladder instead of `exp` ladder.
  `erfc` for complex argument near zero is precision-sensitive
  (analog of the close-paired-pole concern the original
  ``USE_CHAIN_SIMPLEX_PRECISION_FIX`` audit chased) — likely needs
  an `mpmath` safety net.  ~800 LOC, ~1.5 weeks.

**Total Path 1 if all three**: ~3-4 weeks, ~1600 LOC.
**Benefit**: closed-form to arbitrary loop order in d=1; no
scipy.quad anywhere in v1; no `max_ell` wall.

**Risk**: medium-high.  `erfc` with complex `μ` (oscillatory) is
precision-sensitive — the analog of the close-paired-pole concern
that the original audit chased (now confirmed dead on current
spike-reset by the 2-loop stress test at
`docs/m_ge3_precision_bug_audit.md` §Resolution, but the regime
remains real in principle).

### Path 2 — Fourier-mode decomposition (PBC-only natural fit)

For PBC of period L, the heat kernel decomposes *exactly* into a
finite sum of pure exponentials in `Δt`:

```
G_PBC(Δt, Δx)  =  (1/L) Σ_n  exp(i k_n Δx)  exp(−(D k_n² + μ) Δt)
                                  with  k_n = 2π n / L
```

Each Fourier mode is in the input format the existing
`_exp_over_*` evaluators already accept.  The rescue for PBC =
"decompose into N modes, run each through the existing evaluator,
sum."

**Effort**: ~1.5 weeks, ~400 LOC.
**Risk**: low — no new closed-form math, just bookkeeping.
**Cost at runtime**: factor of `N_modes` overhead (typically 30-
100 for tight convergence at moderate L); embarrassingly parallel.
**Limitation**: PBC-only.  Infinite-domain case still needs Path
1 or Path 3.

### Path 3 — sum-of-exponentials approximation

Approximate the heat kernel by a finite sum

```
e^{-X²/(4D Δt)} / sqrt(4π D Δt)  ≈  Σ_k  w_k(X) · exp(−λ_k(X) Δt)
```

via Gaussian quadrature on the heat-kernel's Laplace-representation
auxiliary parameter.  `(w_k, λ_k)` depend on `X` (the external
spatial separation) but not on the integration variables `s`, so
they're computed once per `(τ, x_external)` probe.  Each integrand
factors into the existing pure-exponential form — *exactly* the
shape the existing evaluators eat.

Conceptually the same idea as the colored-noise time
Markovianization from Phase 1 (memory note
`project_grouped_phase_j_precision.md`), but applied to the
spatial heat kernel.

**Effort**: ~2 weeks, ~500 LOC.
**Risk**: medium — convergence rate of the sum depends on `X` and
`Δt` range; needs adaptive node count.  UV behaviour as `X → 0`
(heat kernel sharply peaked) needs careful handling.
**Benefit**: works for both infinite and PBC; reuses the existing
fast evaluators verbatim.
**Cost at runtime**: factor of `N_nodes` overhead (typically 10-30).

### Recommended rescue tiers

Three natural rescue tiers, depending on v1 ambition:

| Rescue tier | Phases added | What v1 gets | Extra effort |
|---|---|---|---|
| **None** (baseline) | None | Tree-level closed-form; 1-loop via `scipy.quad` fallback at m=1; hard wall at `max_ell > 1` | +0 wk |
| **Rescue A** — Path 1 m=1 erfc | +1 wk inside Phase 5 | Tree + 1-loop closed-form everywhere; **no `scipy.quad` in v1**; precision floor 1e-15 (was 1e-8); still hard wall at `max_ell > 1` because m≥2 cases need Path 1 m=2 + m≥3 too | +1 wk |
| **Rescue B** — full Path 1 | Phase 5b (new) | Tree + 1-loop + 2-loop (and beyond) closed-form; **no `max_ell` wall in d=1** | +3 wk |
| **Rescue C** (optional add-on to A or B) — Path 2 PBC mode decomp | +0.5 wk inside Phase 3 | PBC case ALSO routes through the unified mode-decomp path (cross-validates the heat-kernel implementation against the same closed form via a different route — independent sanity check) | +0.5 wk |

Tiers are *additive* — Rescue A is a strict superset of None; Rescue
B is a strict superset of A.  Choosing a higher tier doesn't close
any door the lower tier opened.

**My recommendation: Rescue A.**  Buys the right thing (kills
`scipy.quad` from v1, lifts precision floor) for the lowest extra
cost (+1 wk), and leaves the door open to do Rescue B as a
follow-up if d=1 2-loop becomes a near-term priority.  Path 3 is
not recommended for v1 — Path 1 m=1 is mathematically cleaner and
similar effort; Path 3 would become attractive if d ≥ 2 enters
scope (where the closed-form `erfc` route gets messier).

## Effort summary

Pricing depends on the rescue tier chosen in Phase 0:

| Phase | Description | None | Rescue A | Rescue B |
|---|---|---|---|---|
| 0 | Design signoff | 1 wk | 1 wk | 1 wk |
| 1 | Theory namespace + `boundary` / `initial` APIs | 1.5 wk | 1.5 wk | 1.5 wk |
| 2 | Propagator builder: `Laplacian` → `z_lap`, FT-in-k, heat-kernel module | 2 wk | 2 wk | 2 wk |
| 3 | PBC image-source sum | 1 wk | 1 wk | 1 wk |
| 4 | IC machinery (parallelizable with 5) | 0.5 wk | 0.5 wk | 0.5 wk |
| 5 | Phase J spatial extension | 2.5 wk *(scipy.quad fallback)* | 2.5 wk *(Path 1 m=1 erfc instead)* | 2.5 wk *(Path 1 m=1 erfc)* |
| 5b | Path 1 m=2 + m≥3 erfc extension | — | — | +3 wk |
| 6 | `compute_cumulants` spatial_grid + test theory + validation suite | 1 wk | 1 wk | 1 wk |
| **Cumulative** | | **9 wk** | **9 wk** | **12 wk** |

Notes:
- Rescue A is *the same wall-time as None* in Phase 5 — the erfc
  derivation takes ~1 wk but replaces the scipy.quad fallback
  development that the baseline plan would have spent ~1 wk on.
  So the v1 milestone date is unchanged; you just get better
  precision and no scipy dependency in the loop path.
- Rescue B's +3 wk is for the m=2 polygon (+1.5 wk) and m≥3 chain
  (+1.5 wk) erfc derivations.  Phase 5b runs after Phase 5
  (sequential).
- Rescue C (+0.5 wk inside Phase 3) is omitted from this table —
  it's an optional cross-validation add-on, not a decision axis.

**Default recommendation: Rescue A → total 9 wk** (same as the
baseline plan; same milestone date; better precision and no
scipy.quad dependency).

## Phase 0 — design signoff (1 week, no code)

**Inputs**: `docs/spatial_implementation_outline.md`, this document.
**Output**: written decisions on the three open questions below; one
revision to the outline if any decision changes its scope.

Resolve (see §"Open questions" for full context):

1. `spatial_dim` per-field vs per-theory
2. Boundary-length parameter convention
3. v1's behaviour when the user sets `max_ell > 1` on a spatial
   theory (hard error, soft warning, or just slow) — coupled to
   question 4 below
4. **Rescue tier**: None / Rescue A / Rescue B (see §"Heat-kernel
   rescue paths" above for the trade-offs).  This decision affects
   Phase 5's deliverable and whether Phase 5b is needed at all.
   Optional add-on: Rescue C (Path 2 PBC mode-decomp
   cross-validation).

No code; just a markdown decisions file under `docs/`.

**Checkpoint**: PR adding `docs/spatial_design_decisions_v1.md` is
merged; outline's Stage 0 list is annotated with the chosen options.

---

## Phase 1 — Theory namespace + spatial declarations  [Sequential]

**Inputs**: Phase 0 decisions.
**Output**: a theory file that uses `Laplacian(phi)` in its action
text and declares `.boundary(...)` / `.initial(...)` can be `build()`-
ed end-to-end; the resulting `model` dict carries the spatial block
and `ft.expand()` runs to completion.

### 1.1 — `FieldSpec` and `physical_field()`

| File | Lines | Change |
|---|---|---|
| `pipeline/theory.py` | 72-78 (`FieldSpec`) | Add `spatial_dim: int = 0` field |
| `pipeline/theory.py` | 231-309 (`physical_field()`) | Add `spatial_dim: int = 0` kwarg, validate `≥ 0`, store on `FieldSpec` |
| `pipeline/theory.py` | 1475-1505 (`build()`) | When any `FieldSpec.spatial_dim > 0`, emit `model['spatial'] = {'dim': D, 'fields_with_spatial': [...]}` |

### 1.2 — `Laplacian` SR symbol in the namespace

| File | Lines | Change |
|---|---|---|
| `msrjd/core/field_theory.py` | 1221-1230 (`_build_namespace`) | After the existing `z_delta`, `z_delta_p` registrations, add `ns.Laplacian = SR.var('Laplacian', latex_name=r'\nabla^2')` *only when* any field has `spatial_dim > 0` |
| `msrjd/core/field_theory.py` | 893-898 (`expand()`) | No code change — `Laplacian` passes through Taylor expansion as an inert SR symbol exactly like `Dt` does |
| `pipeline/theory_compiler.py` | 482-484 (`_ns_var_namespace()`) | Add `Laplacian` to the exposed names so the action-text evaluator sees it |
| `pipeline/theory_compiler.py` | 488 (`_build_namespace_for_eval`) | Bind `Laplacian` as a callable that returns the SR symbol times its argument |

### 1.3 — Saddle-killer rule for spatially-uniform saddles

| File | Lines | Change |
|---|---|---|
| `pipeline/theory_compiler.py` | 746-767 (`make_action_lambda` saddle-killer for `Dt`) | Add parallel rule: `Laplacian(<spatially-uniform saddle>) → 0` |
| `pipeline/theory_compiler.py` | 1198-1214 (`make_mf_bg_conditions_lambda`) | Same parallel rule — must be added in two places because both lambdas separately walk the expression tree |

### 1.4 — `.boundary()` and `.initial()` builder methods

| File | Lines | Change |
|---|---|---|
| `pipeline/theory.py` | 638-668 (`.markovianize()`) and 670-696 (`.stability_analysis()`) | Use these as templates — they're the closest existing examples of a builder method that stores a structured dict on `self._model_extras` and emits it under a top-level key in `build()` |
| `pipeline/theory.py` | new methods, ~720 area | Add `.boundary(mode, **params)` (`mode ∈ {'infinite', 'periodic'}`) and `.initial(mode, **params)` (`mode ∈ {'stationary'}` in v1). Both validate and store on builder state. |
| `pipeline/theory.py` | 1475-1505 (`build()`) | Emit `model['boundary']`, `model['initial']` from the stored values |

### 1.5 — Round-trip serializer support

| File | Lines | Change |
|---|---|---|
| `pipeline/theory_serialize.py` | 271-298 (template: `.stability_analysis(True)` emitter) | Add parallel emitter blocks for `.boundary(...)` and `.initial(...)` |
| `pipeline/theory_serialize.py` | 546-576 (chain walker) | Recognize `boundary` / `initial` keys when reading a model dict back into a TheoryBuilder chain |

### LOC + risk

- **~190-230 LOC**, mostly small additions.
- Risk: **Low/Medium**.  The two parallel saddle-killer rules
  (1.3) are the only place where a missed edit produces silent
  wrong answers — both `make_action_lambda` and
  `make_mf_bg_conditions_lambda` must get the rule or the MF solver
  will treat `Laplacian` as a non-zero symbol and refuse to converge
  on the uniform saddle.

### Checkpoint test

```python
ft = build_theory('theories/allen_cahn_1d_subcritical_infinite.theory.py')
ft.expand()
# Bilinear sector contains: phit · (Dt + mu - D·Laplacian) · phi
# MF saddle reduces to:    mu · phi_star + lam · phi_star^3 == 0
# (Laplacian killed at uniform saddle)
assert 'spatial' in ft.model
assert ft.model['spatial']['dim'] == 1
assert ft.model['boundary']['mode'] == 'infinite'
```

A unit test in `tests/test_theory_spatial_basics.py` covering the
five items above is the deliverable.

---

## Phase 2 — Propagator builder (heat-kernel-aware)  [Sequential]

**Inputs**: Phase 1 output (theory carries `model['spatial']` and
expands with `Laplacian` as inert symbol in the bilinear sector).
**Output**: `build_propagator(model, ...)` returns a dict whose new
keys `G_tx`, `k_var`, `spatial_dim`, `bc_mode`, `bc_params` are
populated for spatial theories; old time-only theories see no
change.

### 2.1 — Recognize `Laplacian` in `_to_kernel`

| File | Lines | Change |
|---|---|---|
| `pipeline/_propagator.py` | 507-521 (`_to_kernel`) | Recognize `Laplacian` as a pass-through symbol (mirrors how `Dt` is handled at this stage); do NOT yet substitute `Laplacian → -k²` — that happens at the FT step |
| `pipeline/_propagator.py` | new helper near 520 | `_lap_symbol_in_kernel(expr) -> bool` for downstream decisions |

### 2.2 — FT substitution `z_lap → -k²`

| File | Lines | Change |
|---|---|---|
| `pipeline/_propagator.py` | 609-623 (FT step) | Extend the existing `z_delta → 1`, `z_delta_p → I·ω` substitution dict to include `Laplacian → -k²` where `k = SR.var('k')` (or `k_x, k_y` for higher dim — but in v1, single `k`) |

### 2.3 — Per-k pole recomputation (Option A)

Agent B's audit identified two implementations:

**Option A (recommended)**: Treat `k` as one more entry in the
`num_params` dict at pole-finding time.  `_compute_residues_via_polynomial_fracfield`
already accepts a `num_params` dict at lines 81-313; if `k` flows
through as just another numerical key, no surgery is needed in that
function.  `compute_poles_and_residues` (line 781) similarly just
needs `k` threaded through.

**Option B (deferred)**: Symbolic pole-tracking in `k`.  Out of v1
scope — would require substantial rework of the fracfield path.

| File | Lines | Change |
|---|---|---|
| `pipeline/_propagator.py` | 81-313, 781 | No code change for v1 — `k` flows through `num_params` |
| `pipeline/_propagator.py` | 524-778 (`build_propagator`) | Add an `_inverse_ft_to_spatial(...)` stage after the existing inverse-FT-in-ω, conditional on `model.get('spatial')`; produces `G_tx` |

### 2.4 — Heat-kernel module (new)

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/spatial/__init__.py` | new | (empty) |
| `msrjd/integration/spatial/heat_kernel.py` | new, ~250 LOC | Two-tier module: **Tier 1** — closed-form heat kernel × exponential for Allen-Cahn-like propagators (diagonal in field index, scalar `D · k²` term), detected by structural pattern-matching on `K_ft`; **Tier 2** — numerical inverse FT on a `k`-grid for the fallback general case.  Both return a callable `G_tx(t, x, **num_params) -> complex`. |

### 2.5 — Propagator dict shape

| File | Lines | Change |
|---|---|---|
| `pipeline/_propagator.py` | 749-763 (prop dict assembly) | When `model['spatial']` exists, add: `G_tx` (callable), `k_var` (the `SR.var('k')`), `spatial_dim` (int), `bc_mode` (str, just stored here for downstream), `bc_params` (dict) |
| `msrjd/integration/time_domain/propagator_td.py` | 135-350 (`build_G_t_matrix`) | No change in Phase 2 — `build_G_tx_matrix` is added in Phase 5, follows the same `smooth[i,j](t) = Σ_k C[k][i,j] · exp(I·pole[k]·t)` assembly pattern but with the heat-kernel factor multiplied in per smooth edge |

### 2.6 — Cache key extension

| File | Lines | Change |
|---|---|---|
| `pipeline/_diagrams.py` | 42-51 (`_model_cache_dir`) | No change.  The diagram set is independent of spatial structure (topology unchanged); reuse the existing `<model-tag>_taylor<N>/` directory |
| `pipeline/_propagator.py` | cache key construction | Bump the propagator-cache stage name (e.g. add `_spatial_v1` suffix) when `model['spatial']` exists, so old time-only caches don't collide |

### LOC + risk

- **~700-900 LOC**, of which **~250 are the new `heat_kernel.py`**.
- Risk: **Medium**.  The single biggest unknown is whether
  Sage's symbolic inverse FT step actually returns clean
  closed-form for non-Allen-Cahn-like (multi-Laplacian,
  multi-field) cases.  Mitigation: Tier 1 / Tier 2 split detects
  the easy case and routes the hard case to numerical fallback —
  no closed-form-or-bust assumption.

### Open question (Agent B)

- Where exactly does the spatial-Fourier image of `Laplacian` get
  resolved — at `_to_kernel` (early), at the FT step (middle), or
  at residue extraction (late)?  Phase 2.2 chooses the **FT step**
  on the grounds that this is where existing `z_delta_p → I·ω`
  substitutions happen, but Agent B flagged it as worth double-
  checking that no earlier stage depends on a numerical `Laplacian`.

### Checkpoint test

```python
prop = build_propagator(allen_cahn_model, ...)
G_tx = prop['G_tx']
# Sample several (t, x) points; compare to closed form:
#   1 / sqrt(4 pi D t) * exp(-x^2/(4 D t) - mu t)
# Expect agreement to 1e-12 for t ∈ [0.01, 10], x ∈ [-5, 5]
assert prop['spatial_dim'] == 1
assert prop['bc_mode'] == 'infinite'
```

Unit tests in `tests/test_propagator_spatial.py`.

---

## Phase 3 — PBC image-source sum  [Sequential]

**Inputs**: Phase 2 output (`G_tx` exists for infinite domain).
**Output**: when `model['boundary']['mode'] == 'periodic'`, `G_tx`
is replaced by `G_tx_PBC(t, x) = Σ_n G_inf(t, x + n·L)`, truncated at
exponentially decaying tail.

### 3.1 — Image-sum wrapper

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/spatial/heat_kernel.py` | new | `image_sum(G_inf, L, n_max_auto=True) -> G_PBC` helper.  Truncation: auto-pick `n_max` such that `|G_inf(t_max, n_max · L)| < eps · |G_inf(t_max, 0)|`, `eps = 1e-12` |
| `pipeline/_propagator.py` | `build_propagator` near `G_tx` assembly | When `bc_mode == 'periodic'`, wrap `G_tx = image_sum(G_inf_tx, L=bc_params['length'])` |

### 3.2 — User-facing `.boundary('periodic', length=...)`

Already added in Phase 1.4 — Phase 3 just reads it.

### 3.3 — *(Optional, Rescue C)* Path 2 PBC mode-decomposition cross-validation

If the user picks Rescue C in Phase 0, add a *second*
PBC-evaluation route: instead of (or in addition to) the
image-source sum, decompose `G_PBC` as a finite Fourier sum

```
G_PBC(Δt, Δx)  =  (1/L) Σ_n  exp(i k_n Δx)  exp(−(D k_n² + μ) Δt)
                                  with  k_n = 2π n / L
```

and feed each pure-exponential mode through the *existing*
`_exp_over_*` evaluators.  Sum the results; cross-check against the
image-source-sum path to ~1e-10 relative.

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/spatial/heat_kernel.py` | new | `mode_decomposition_pbc(G_inf_FT, L, n_modes_auto=True) -> list of (k_n, decay_rate, weight)` |
| `msrjd/integration/time_domain/final_integral.py` | unchanged | Mode-decomposed path uses the EXISTING evaluators — no analytic changes needed |
| `tests/test_pbc_mode_decomp_vs_image_sum.py` | new | Cross-validation test |

**Effort**: +0.5 wk.  **Benefit**: independent sanity check that
the heat-kernel implementation gives the same answer via two
different routes.  Catches sign / normalization bugs that any
single-path test would miss.

### LOC + risk

- **~150-200 LOC**, almost all in the new helper.
- Risk: **Low**.  Image-sum truncation is straightforward; the
  failure mode (truncation too aggressive) is detectable by the
  PBC→∞-limit checkpoint test.

### Checkpoint test

```python
# Free-theory equal-time variance under PBC matches the lattice-sum:
prop_pbc = build_propagator(allen_cahn_pbc_model, L=20.0, ...)
v_pbc = prop_pbc['G_tx'](t=0.0, x=0.0, mu=1.0, D=1.0)
v_lattice = sum(1.0 / (1.0 * (2*pi*n/20.0)**2 + 1.0) for n in range(-100, 101))
# Multiply by T/L convention factor
assert abs(v_pbc - T / L * v_lattice) / v_lattice < 1e-6

# PBC → infinite limit as L grows
for L in [10, 20, 50, 200]:
    v_pbc_L = build_propagator(model, L=L)['G_tx'](0, 0, mu=1, D=1)
v_pbc_L → T / (2 * sqrt(mu * D))  as L → ∞
```

---

## Phase 4 — IC machinery  [Parallelizable with Phase 5]

**Inputs**: Phase 1 output (`.initial(...)` builder method exists).
**Output**: `compute_cumulants` validates IC + observable
consistency; the only v1 mode is `stationary`, which requires no
additional action term.

### 4.1 — Validation in `compute_cumulants`

| File | Lines | Change |
|---|---|---|
| `pipeline/compute.py` | 237-244 (validation) | When `model.get('initial', {}).get('mode') == 'stationary'`, validate that requested external fields are two-time-or-higher (not single-time mean observables); raise a clear error if they aren't |
| `pipeline/compute.py` | new helper | `_validate_ic_observable_compat(model, external_fields)` |

### 4.2 — Round-trip

Already added in Phase 1.5.

### LOC + risk

- **~80-120 LOC**, mostly the validation helper + tests.
- Risk: **Low**.  Pure plumbing; no math.

### Checkpoint test

```python
# Two-time correlator: OK
compute_cumulants(model=allen_cahn_model, k=2,
                  external_fields=[('phi', 1), ('phi', 2)])  # passes

# Single-time observable under stationary IC: error
with pytest.raises(ValueError, match='stationary IC requires'):
    compute_cumulants(model=allen_cahn_model, k=1,
                      external_fields=[('phi', 1)])
```

---

## Phase 5 — Phase J spatial extension (variant per rescue tier)  [Sequential]

**Inputs**: Phase 2 output (propagator has `G_tx`), Phase 3 output
(PBC variant of `G_tx`).
**Output**: `compute_correction_td` returns a callable that, when
evaluated at `(τ, x_external)` for a spatial theory, returns the
correct tree-level and 1-loop value for Allen-Cahn-like theories.

This is the largest, riskiest phase.  Agent C's audit identified
the load-bearing issue (heat-kernel factors break the fast m≥1
evaluators) — sub-section 5.4 below describes the two variants:

- **Variant 5.4-None** (baseline): residual m=1 temporal integral
  routes through `scipy.quad` (1 wk).  Ships fastest; precision
  floor 1e-8 (matches scipy.quad default `epsabs`).
- **Variant 5.4-A** (Rescue A, *recommended*): residual m=1
  temporal integral routes through a new closed-form
  `_integrate_1d_polytope_with_erfc` evaluator (1 wk —
  *same wall-time* as the scipy.quad work in the None variant).
  No scipy.quad in v1 loop path; precision floor 1e-15.

Both variants share sub-sections 5.1-5.3, 5.5, 5.6.  The
divergence is only in 5.4.

### 5.1 — `EdgeModeSum` carries heat-kernel factor

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/time_domain/final_integral.py` | 116-147 (`EdgeModeSum`) | Extend dataclass: add `heat_kernel: Optional[HeatKernelFactor]` field, default `None` (preserves time-only behaviour); add `dx_*` parallel fields mirroring the `Δt` linear form structure |
| `msrjd/integration/time_domain/final_integral.py` | 150-200 (`_build_edge_mode_sums`) | When `propagator_data['spatial_dim'] > 0`, populate `heat_kernel` field per edge using `propagator_data['G_tx']` |
| `msrjd/integration/spatial/spatial_integral.py` | new, ~150 LOC | `HeatKernelFactor` dataclass: `(D_eff, mu_eff)` parameters; `evaluate(dt, dx)` method; methods to compose factors when convolving along a chain |

### 5.2 — `vertex_position` parallel to `vertex_time`

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/time_domain/final_integral.py` | 2624-2638 (`vertex_time` assignment in `integrate_diagram`) | Add parallel `vertex_position` assignment.  External vertices' positions come from `external_positions` argument; internal vertices get an `x_v` SR symbol per spatial dim |
| `msrjd/integration/time_domain/final_integral.py` | 2411-3830 (`integrate_diagram`) | Thread `vertex_position`, `free_pos_vals`, `propagator_data['spatial_dim']` through the integration loop |
| `msrjd/integration/time_domain/final_integral.py` | 3656-3707 (`_make_subset_contrib`) | Call spatial evaluator BEFORE the time evaluator (spatial integral collapses first, leaving residual `dt` integrals) |

### 5.3 — Spatial evaluator (the analytic collapse)

For Allen-Cahn-like free theory in d=1, the spatial integral is a
Gaussian convolution that collapses in closed form:

$$\int dx_v \; G(t, x - x_v) \cdot G(t', x_v - x') = G(t + t', x - x')$$

(heat kernel semigroup property).  This is the only spatial-integral
case v1 handles closed-form.

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/spatial/spatial_integral.py` | new | `_spatial_gaussian_convolution_chain(heat_kernels, vertex_positions, external_positions)` — multiplies the chain of heat kernels along a connected component, returns the residual time-domain integrand |
| `msrjd/integration/spatial/spatial_integral.py` | new | `_spatial_fallback_quadrature(...)` — for cases the closed-form path doesn't recognize, route to `scipy.integrate.nquad` over the spatial coordinates.  Off in v1 (raises `NotImplementedError`); stub in place for v2 |

### 5.4 — Residual m=1 temporal integral (variant per rescue tier)

After the spatial collapse, the residual temporal integral is of
the form

```
∫_L^U  T(s)^(-1/2)  exp[ -X² / (4D T(s))  -  μ T(s) ]  ds
```

with `T(s)` linear in `s`.  The exponent is NOT linear in `s`, so
the standard `_integrate_1d_polytope_modesum` closed form does not
apply.  Two variants:

#### 5.4-None (baseline) — `scipy.quad` fallback

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/time_domain/final_integral.py` | 1976 area (`_integrate_1d_polytope_modesum`) | Detect heat-kernel-prefactor flag on the `EdgeModeSum`; if set, route to a new `_integrate_1d_polytope_with_heat_kernel` helper that wraps the integrand in a `scipy.integrate.quad` call |
| `msrjd/integration/time_domain/final_integral.py` | new helper ~2050 | `_integrate_1d_polytope_with_heat_kernel(...)` — same polytope bounds extraction as the analytic path, but with `scipy.quad` instead of pole-residue closure |

**Effort**: ~1 wk.  **Precision floor**: 1e-8 (scipy.quad default
`epsabs`).  **v1 risk**: scipy.quad may underflow at large `Δt`
when the heat-kernel `1/sqrt(Δt)` multiplies `exp(-μ Δt)` in the
wrong order — route through `numpy.float128` if it diverges.

#### 5.4-A (*recommended*) — Path 1 m=1 erfc closed form

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/time_domain/final_integral.py` | 1976 area (`_integrate_1d_polytope_modesum`) | Detect heat-kernel-prefactor flag; if set, route to a new `_integrate_1d_polytope_with_erfc` helper |
| `msrjd/integration/time_domain/final_integral.py` | new helper ~2050, ~200 LOC | `_integrate_1d_polytope_with_erfc(α, β, γ, L, U)` — closed form via the `v = √(α u) ± √(β/u)` substitution; bottoms out in two `erfc(·)` calls per pole.  Handles `L = −∞` and `U = +∞` via the limiting K_{1/2} identity |
| `msrjd/integration/time_domain/final_integral.py` | new tests, ~150 LOC | Unit tests vs scipy.quad on synthetic integrands across 6+ regimes (small/large `μ`, small/large `X`, finite/semi-infinite intervals, complex `μ`) |

**Effort**: ~1 wk (same as 5.4-None — replaces, not adds).
**Precision floor**: 1e-15 (closed form).  **Risk**: `erfc` for
complex argument near zero needs an `mpmath` safety net analogous
to `USE_CHAIN_SIMPLEX_PRECISION_FIX` — add the gating flag at
implementation time even if it defaults to off, so future
debugging is easy.

**Why Rescue A is recommended**: same wall-time as the scipy.quad
path, but kills the scipy dependency in the v1 loop integrand,
lifts precision by 7 orders of magnitude, and produces an artefact
(`_integrate_1d_polytope_with_erfc`) that's a building block for
Rescue B's m≥2 work.

### 5.5 — `max_ell > 1` behaviour for spatial theories (rescue-tier-dependent)

The `max_ell > 1` wall exists *only if* the chosen rescue tier
doesn't extend Path 1 to m≥2.  With Rescue B, no wall.

| File | Lines | Change |
|---|---|---|
| `pipeline/compute.py` | 237-244 (validation) | When `model.get('spatial')` exists AND `max_ell > 1` AND rescue tier ∈ {None, A}: either raise (decision A in Q3) or warn + slow path (decision B in Q3) — Phase 0 decides.  When rescue tier == B: no wall, no warning |

### 5.6 — `build_G_tx_matrix`

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/time_domain/propagator_td.py` | 135-350 area, new function | `build_G_tx_matrix` parallel to `build_G_t_matrix`; same hand-coded closed-form structure (`smooth[i,j](t,x) = Σ_k C[k][i,j] · exp(I·pole[k]·t) · heat_kernel(t, x; D_eff)`) |

### LOC + risk

- **~700-800 LOC** for Variant 5.4-None; **~750-850 LOC** for
  Variant 5.4-A (the erfc helper is slightly larger than the
  scipy.quad wrapper, mostly due to its unit-test suite).
- Risk: **Medium/High** in both variants.  Three things can go wrong:
  1. The spatial Gaussian-convolution collapse (5.3) silently
     produces the wrong residual `D_eff, mu_eff` parameters when
     the chain has more than one heat kernel — needs careful unit
     tests on each chain length.
  2. **Variant 5.4-None**: scipy.quad may underflow at large `Δt`
     if the heat kernel factor `1/sqrt(Δt)` is multiplied with
     exponentially decaying `exp(-μ Δt)` in the wrong order —
     route through `numpy.float128` if `float64` quadrature
     diverges.
     **Variant 5.4-A**: the erfc closed form is precision-sensitive
     for complex argument near zero — needs an `mpmath` safety
     net flag analogous to `USE_CHAIN_SIMPLEX_PRECISION_FIX` (the
     same close-pole concern from `docs/m_ge3_precision_bug_audit.md`).
  3. The interaction between Phase J's existing `δ`-edge handling
     and the new spatial coordinates (a `δ(t-t')` edge collapses
     two vertex *times* but not their *positions* — the position
     integral still has to be done) — Agent C flagged this as the
     least-well-understood interaction and requested a worked
     example before coding.

### Open question (Agent C)

- Should `G_tx` be a Sage SR expression or a numpy callable?  v1
  Phase 2 produces a callable (faster runtime, no per-evaluation
  Sage substitution overhead).  Phase 5 has to compose heat
  kernels symbolically for the chain collapse, which argues for
  SR.  **Resolution**: produce both — `G_tx` (callable) for direct
  evaluation, `G_tx_sym` (SR expression) for symbolic composition.
  Add this to Phase 2's deliverable retroactively.

### Checkpoint test

```python
# Free tree-level, infinite domain
result = compute_cumulants(model=allen_cahn_model, k=2, max_ell=0,
                           external_fields=[('phi', 1), ('phi', 2)],
                           spatial_grid=np.linspace(-5, 5, 21))
# Closed-form expected:  C(x, 0) = (T / (2 sqrt(mu D))) * exp(-|x| / sqrt(D/mu))
# Match to 1e-6 relative

# 1-loop self-energy correction
result_loop = compute_cumulants(model=allen_cahn_model, k=2, max_ell=1, ...)
# Closed-form expected:  -3 lam T^2 / (8 mu sqrt(mu D))
# Variant 5.4-None: match to 1e-4 relative (scipy.quad floor)
# Variant 5.4-A:    match to 1e-10 relative (erfc closed form)
```

---

## Phase 5b — Path 1 m=2 + m≥3 erfc extension *(Rescue B only)*  [Sequential, follows Phase 5]

**Inputs**: Phase 5 output with Variant 5.4-A in place (erfc m=1
closed form working).
**Output**: m=2 polygon and m≥3 chain-simplex integrators extended
to recognize heat-kernel-prefactored integrands; `max_ell > 1`
wall on spatial theories is lifted.

This phase is only entered if Phase 0 chose Rescue B.  It builds on
the m=1 erfc machinery from Variant 5.4-A.

### 5b.1 — m=2 polygon erfc extension

Existing `_integrate_2d_polygon_modesum` (`final_integral.py:2175`)
does fan triangulation and integrates each triangle's pure
`exp(α s₀ + β s₁)` integrand in closed form.  The heat-kernel
extension: on each triangle, change variables so the
time-direction integrates to erfc and the orthogonal direction
collapses (claim: Gaussian collapse — to be confirmed by Phase 0
derivation pass).

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/time_domain/final_integral.py` | 2175 area (`_integrate_2d_polygon_modesum`) | Detect heat-kernel-prefactor flag; if set, route to new `_integrate_2d_polygon_with_erfc` |
| `msrjd/integration/time_domain/final_integral.py` | new helper, ~400 LOC | `_integrate_2d_polygon_with_erfc(...)` — per-triangle change of variables; `erfc` closed form along the time direction × Gaussian collapse on the orthogonal direction.  Reuses `_integrate_1d_polytope_with_erfc` (from 5.4-A) on the inner time integral |
| `tests/test_polygon_m2_erfc.py` | new, ~200 LOC | Unit tests vs `_integrate_2d_polygon_modesum` on synthetic time-only integrands (heat_kernel=None) to verify backward compat; then on heat-kernel integrands vs mpmath reference |

**Effort**: ~1.5 wk.  **Open question** (Phase 0 derivation pass):
whether the orthogonal-direction collapse is *always* Gaussian for
the heat-kernel case or only in specific configurations.

### 5b.2 — m≥3 chain-simplex erfc extension

Existing `_exp_over_chain_simplex` (`final_integral.py:877`) does
a 2^N recursion bottoming out in `(exp(α U) − exp(α L)) / α` per
split.  The heat-kernel extension: each split produces an `erfc`
ladder.

| File | Lines | Change |
|---|---|---|
| `msrjd/integration/time_domain/final_integral.py` | 877 area (`_exp_over_chain_simplex`) | Add a `heat_kernel: HeatKernelFactor | None = None` parameter; when non-None, route to `_erfc_over_chain_simplex` |
| `msrjd/integration/time_domain/final_integral.py` | new helper, ~500 LOC | `_erfc_over_chain_simplex(...)` — 2^N recursion identical to the existing one but with `erfc` leaves.  Reuses the close-pole detection logic from `USE_CHAIN_SIMPLEX_PRECISION_FIX` (since `erfc` cancellation has the same character as `exp` cancellation near zero argument) |
| `msrjd/integration/time_domain/final_integral.py` | mpmath gate | Add `USE_ERFC_CHAIN_SIMPLEX_PRECISION_FIX` flag, default False; on True, route close-erfc cases to `mpmath.erfc` |
| `tests/test_chain_simplex_erfc.py` | new, ~250 LOC | Synthetic close-pole stress test; verify mpmath gate fires when needed |

**Effort**: ~1.5 wk.

### LOC + risk for Phase 5b

- **~1500 LOC** total (split ~600/900 between m=2 and m≥3).
- Risk: **High**.  Two compounding concerns:
  1. The m=2 orthogonal-collapse claim needs verification before
     coding; if it doesn't hold for the heat-kernel case, the
     m=2 path becomes considerably uglier (still doable, but
     more bookkeeping).
  2. `erfc` precision near zero is the analog of the
     close-paired-pole concern from the chain-simplex audit.
     Whereas `exp` cancellation is mostly latent on currently-
     stress-tested theories (per the
     `docs/m_ge3_precision_bug_audit.md` §Resolution finding),
     `erfc` cancellation may manifest more readily because
     the argument structure `α + β/√U` produces small differences
     more easily when `U` is large.  Plan for the mpmath gate
     to actually fire on at least one test theory.

### Checkpoint test for Phase 5b

```python
# 2-loop self-energy correction, infinite domain, d=1
result = compute_cumulants(model=allen_cahn_model, k=2, max_ell=2,
                           external_fields=[('phi', 1), ('phi', 2)],
                           spatial_grid=np.linspace(-5, 5, 21))
# Closed-form expected:  the 2-loop "sunset" diagram value (TBD,
# but tractable — Allen-Cahn 2-loop is in standard RD textbooks)
# Match to 1e-10 relative
```

---

## Phase 6 — Output API + tests + validation  [Sequential]

**Inputs**: Phase 5 output.
**Output**: `compute_cumulants(spatial_grid=...)` returns a result
dict with `C_tau_x` populated; the test theory file ships; the
simulator and validation notebook are checked in.

### 6.1 — `compute_cumulants` API extension

| File | Lines | Change |
|---|---|---|
| `pipeline/compute.py` | 108-129 (signature) | Add `spatial_grid: Optional[np.ndarray] = None` kwarg |
| `pipeline/compute.py` | 237-244 (validation) | If `model.get('spatial')` AND `spatial_grid is None`, raise; if `spatial_grid is not None` AND no spatial fields, raise |
| `pipeline/compute.py` | 467-473 (`tau_points` construction) | When `spatial_grid` given, build Cartesian product `(tau, x)` external grid |
| `pipeline/compute.py` | 312-432 (`total_C_batch`) | Embarrassingly parallel — enlarge the probe tuple to `(*tau, *x)`; existing parallel machinery handles the rest |
| `pipeline/compute.py` | 619-646 (result dict) | Add `C_tau_x` of shape `(len(tau_grid), len(spatial_grid))` |

### 6.2 — Test theory files

| File | Lines | Change |
|---|---|---|
| `theories/allen_cahn_1d_subcritical_infinite.theory.py` | new, ~40 LOC | Exactly as outlined in `spatial_implementation_outline.md` §"Test theory" |
| `theories/allen_cahn_1d_subcritical_pbc.theory.py` | new, ~40 LOC | Identical except `.boundary('periodic', length='L')` and `L` parameter |

### 6.3 — Simulator

| File | Lines | Change |
|---|---|---|
| `models/allen_cahn_1d_sim_numba.py` | new, ~300 LOC | 1D Langevin simulator on a uniform grid; spectral propagation step (FFT-based Crank-Nicolson for the linear part, explicit Euler for `-λ φ³`).  Standard textbook implementation |

### 6.4 — Validation notebook + tests

| File | Lines | Change |
|---|---|---|
| `notebooks/pipeline_allen_cahn_1d_sim_compare.ipynb` | new | Reproduces every entry in the outline's "Validation suite" table |
| `tests/test_spatial_rd_basics.py` | new | Closed-form benchmark assertions (free tree-level, free PBC, PBC→∞ sweep, 1-loop tadpole) |

### LOC + risk

- **~600-800 LOC** total.
- Risk: **Low** for the framework code; the simulator can take a
  bit to debug but the math is well-known.

### Checkpoint test

All entries in the outline's "Validation suite" table pass.

---

## Open questions (Phase 0)

These need user signoff before Phase 1 code lands.

### Q1 (Agent A) — `spatial_dim`: per-field or per-theory?

**Per-field** (each `physical_field` declares its own `spatial_dim`):

- Pro: matches the rest of the field API; allows mixed-dim theories
  in v2 (e.g. an order parameter `φ(x, t)` coupled to a spatially-
  averaged variable `m(t)`)
- Con: adds validation logic — all fields with `spatial_dim > 0`
  must share the same value in v1 (no mixed dims)

**Per-theory** (a top-level `.spatial_dim(1)` builder method):

- Pro: simpler v1; one declaration; no mixed-dim validation needed
- Con: forces all fields in the theory to be spatial; no spatially-
  averaged auxiliary fields

**Recommended**: **per-field**, default 0.  The validation cost
is one `len(set(spatial_dims)) <= 1` check at `build()`.  Gains
mixed-dim support for free in v2.

### Q2 (Agent A) — Boundary-length parameter

The PBC variant needs a length `L`.  Two options:

**Inline number** (`.boundary('periodic', length=20.0)`): L is fixed
at theory-build time.

**Named parameter** (`.boundary('periodic', length='L')` referring
to a `.parameter('L', default=20.0)`): L is sweepable like any
other parameter.

**Recommended**: **named parameter** (the outline's example
already uses this form).  Gains the PBC→∞-limit checkpoint test
as a single sweep across one parameter, which is the most natural
sanity-check we can run.

### Q3 (Agent C + outline) — `max_ell > 1` on spatial theory

Coupled to Q4 (rescue tier).  If rescue tier is **Rescue B**, Q3
is moot — no wall.  If rescue tier is **None** or **Rescue A**,
Q3 picks one of:

- **A: Hard error**: `max_ell > 1` on a spatial model raises
  `NotImplementedError("v1 supports max_ell ≤ 1 for spatial
  theories; see docs/spatial_implementation_plan.md §rescue paths
  for the Rescue B option that lifts this")`
- **B: Soft warning**: emit a `UserWarning` and proceed via the
  scipy.quad fallback at every subset (slow, but works)
- **C: Silent slow path**: no warning, just route everything
  through scipy.quad

**Recommended**: **B** for v1 (regardless of None vs Rescue A).
The user may want to spend the wall time to get a 2-loop answer
for a single sanity check even if it's slow.  Hard error is
friendlier in v0.5 (intermediate milestone after Stage 4) when
the user hasn't yet been told about the limitation.

### Q4 (new) — Rescue tier

See §"Heat-kernel rescue paths" earlier in this document for the
full options and trade-offs.  Pick one of:

- **None** (baseline) — scipy.quad fallback at m=1; hard wall at
  `max_ell > 1`; precision floor 1e-8.  Total v1: 9 wk.
- **Rescue A** (recommended) — Path 1 m=1 erfc closed form
  replaces scipy.quad at m=1; same hard wall at `max_ell > 1`
  but no scipy in the loop; precision floor 1e-15.  Total v1: 9 wk
  (*same as None* — replaces, not adds).
- **Rescue B** — full Path 1 (m=1 + m=2 + m≥3 erfc); no
  `max_ell` wall in d=1; precision floor 1e-12 to 1e-15.  Total
  v1: 12 wk (+3 wk for Phase 5b).
- **Rescue C add-on** to A or B — Path 2 PBC mode-decomp
  cross-validation.  +0.5 wk inside Phase 3.

**Recommended**: **Rescue A** (no extra time vs None, kills scipy
in the v1 loop, sets up Rescue B as a clean follow-on).  Add
Rescue C if you want belt-and-suspenders cross-validation on PBC.

---

## What is explicitly out of v1 scope

Stated here so the implementation doesn't drift:

1. **Higher-derivative kinetic operators**: Cahn-Hilliard `∇⁴`,
   fractional `(-Δ)^α`, anisotropic Laplacians.  These break the
   heat-kernel closed form (and the rescue paths that depend on
   it).
2. **Multi-Laplacian in one action**: e.g. `Laplacian(φ) +
   Laplacian(ψ)` with cross-coupling.  Phase 2's Tier 1
   structural pattern-match handles the diagonal case only.
3. **Non-Gaussian spatial noise**: shot noise, Poisson noise on a
   lattice.  Path A from the existing memory notes handles non-
   Gaussian *temporal* noise but doesn't extend to spatial without
   additional work.
4. **Critical phenomena, ε-expansion, RG**: the entire (t, x)
   choice is wrong for these.  v2+ work; would likely add an (ω, k)
   alternative integrator path.
5. **Dirichlet, Neumann, Robin BCs**: only periodic + infinite in
   v1.  Lefèvre-Biroli §2.6's reservoir-coupled boundaries are the
   v2 framing.
6. **Transient ICs**: only stationary in v1.  Lefèvre-Biroli §2.5's
   `S_I` action term is the v1.5 plan.
7. **d ≥ 2**: dimension is parameterized from day one (per the
   outline's pitfall list) but v1 ships with d=1 only.  Path 1's
   m=1 erfc generalization to d ≥ 2 is straightforward (the K_{1/2}
   identity becomes K_{d/2}); Path 1 m=2 / m≥3 in d ≥ 2 gets messy
   and Path 3 (sum-of-exponentials) likely becomes the preferred
   route for higher dim.
8. **No closed-form ell ≥ 2 in d=1** — *only if Rescue tier is
   None or A*.  Rescue B lifts this; see §"Heat-kernel rescue
   paths".

---

## Suggested execution order with checkpoints

### Rescue tier None or A (9 wk total)

```
Week 1         Phase 0: design signoff (incl. rescue tier choice)
                                       → docs/spatial_design_decisions_v1.md
Week 2-3       Phase 1: theory namespace + .boundary/.initial APIs
               ▸ Checkpoint: Allen-Cahn theory file build()s, ft.expand() runs
Week 4-5       Phase 2: propagator builder + heat_kernel module
               ▸ Checkpoint: G_tx matches closed form at sample points
Week 5-6       Phase 3: PBC image-sum  (overlaps Phase 2's last week)
               ▸ Optional +0.5 wk: Rescue C (Path 2 mode-decomp cross-check)
               ▸ Checkpoint: equal-time PBC variance matches lattice sum
Week 6         Phase 4: IC machinery  (concurrent with start of Phase 5)
               ▸ Checkpoint: stationary IC validates; transient request errors
Week 6-8.5     Phase 5: Phase J spatial extension
               ▸ Variant 5.4-None: scipy.quad fallback (Rescue tier None)
               ▸ Variant 5.4-A:    Path 1 m=1 erfc        (Rescue tier A)
                 ─ both variants take ~1 wk in 5.4; same overall Phase 5
                   duration
               ▸ Mid-checkpoint (week 7): tree-level free theory matches
               ▸ End-checkpoint (week 8.5): 1-loop tadpole matches
                 ─ to 1e-4 rel (None) or 1e-10 rel (A)
Week 9         Phase 6: compute_cumulants API + test theory + validation suite
               ▸ Final checkpoint: all entries in outline's validation table pass
```

### Rescue tier B (12 wk total)

Same as above through week 9, then:

```
Week 9-12      Phase 5b: Path 1 m=2 polygon erfc + Path 1 m≥3 chain erfc
               ▸ Week 10: m=2 polygon derivation + implementation
               ▸ Week 11: m≥3 chain erfc + mpmath safety net
               ▸ Week 12: 2-loop self-energy checkpoint test
                 ─ closed-form 2-loop Allen-Cahn matches expected
                   value to 1e-10 rel; max_ell wall lifted in d=1
```

### v0.5 demo milestones

Two natural intermediate demo points:

1. **End of Phase 4 (~week 6)** — tree-level spatial correlators
   end-to-end on any user-written theory, validated against
   free-theory closed forms.  Same for both Rescue tier choices.
2. **End of Phase 5 (~week 8.5)** — 1-loop spatial corrections
   working.  Quality difference is the precision floor
   (1e-4 None vs 1e-10 A) and the scipy dependency presence.
3. **End of Phase 5b (~week 12)** — *Rescue B only* — 2-loop
   closed-form working; `max_ell` wall lifted in d=1.

---

*See `docs/spatial_implementation_outline.md` for the design
rationale that justifies the (t, x) integration approach over
(ω, k), and for the test theory definition.*

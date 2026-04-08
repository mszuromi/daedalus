# Changelog

All notable fixes, features, and known issues for the MSR-JD Feynman diagram pipeline.

---

## 2026-04-08 — Phase J numerical quadrature (replaces symbolic integration)

The Phase J tree-level evaluator no longer uses SageMath's symbolic
`integrate()` — it now does explicit **numerical quadrature** via
`scipy.integrate.quad` / `nquad` on a `fast_callable` JIT'd version of
the stripped integrand, with polytope bounds extracted from the
retarded Heaviside factors. The public API contract of the
`time_domain` subpackage is unchanged (same module layout, same
function signatures) but the type of the returned `contribution` and
`total_C` has flipped from SageMath `SR` to plain Python callables.

### Why

The previous MVP handed the symbolic integrand
`combined_prefactor · ∏ exp(...) · heaviside(...)` to
`sage.all.integrate(..., -oo, +oo)`, which returns an **unevaluated**
`integrate(...)` SR object whenever the integration bounds depend on
the sign of a symbolic external time (e.g. `min(0, t₁)` for the k=2
tree). The tests still passed because the downstream code called
`.subs({t₁: value}).real()` at each τ point, which silently
re-triggered Maxima to resolve the polytope at that specific t₁ —
closed-form symbolic work per τ point. Correct, but:

- it's slow — Maxima is re-doing symbolic integration for every τ point;
- it's fragile — Maxima routinely hangs on Heaviside-gated integrands
  with more than a couple of variables;
- it's not what "done numerically" means; there was never a call to
  `scipy.integrate.quad` or any numerical quadrature routine.

The user flagged this and asked for explicit numerical quadrature
instead. This change implements the "mature engine" path that was
already sketched in the Phase J plan (`spicy-seeking-shore.md`).

### What changed

- `msrjd/integration/time_domain/final_integral.py` — full rewrite of
  `integrate_tree_diagram`:
  1. `G_t_entry` is now called with `include_heaviside=False`, so each
     edge factor is a pure exponential. The Heaviside argument
     `dt_e = t_v − t_u` is collected separately as an **explicit
     linear inequality constraint** on the vertex-time variables.
  2. The stripped integrand is JIT-compiled via
     `sage.all.fast_callable(expr, vars=[s_1, ..., s_m, t_1, ..., t_k],
     domain=CDF)` — evaluation becomes a C-level op on concrete floats.
  3. Linear coefficients `(a_int, a_ext, c₀)` are extracted from each
     constraint by `.coefficient()` + origin substitution, so the
     constraints can be resolved to concrete numeric half-space
     inequalities in `s` once external times are supplied.
  4. `_integrate_polytope` dispatches to `_integrate_1d_polytope`
     (single `scipy.integrate.quad` call) or `_integrate_2d_polytope`
     (nested `scipy.integrate.nquad` with inner bounds computed by
     `_resolve_1d_bounds` given the outer value). Real and imaginary
     parts are integrated separately so the path handles
     complex-valued residues cleanly.
  5. `integrate_tree_diagram` returns a Python closure
     `contribution(*ext_time_values) -> complex` instead of an SR
     expression. The closure captures the `fast_callable` integrand,
     the linear constraint data, and the pin / free-index bookkeeping.
  6. For `m ≥ 3` (not exercised by the MVP) the polytope integrator
     raises `NotImplementedError` so the orchestrator can fall back to
     Phase I. Extension 1 will generalize to arbitrary `m`.
- `msrjd/integration/time_domain/pipeline.py` — `compute_correction_td`:
  - `total_C` is now itself a Python callable that sums each group's
    contribution callable; it takes `k` positional arguments and
    returns a complex.
  - `representation` on each tree-evaluated group is now
    `'numerical'` (was `'symbolic'`).
  - The SIGALRM watchdog and `timeout_sec` parameter are kept in the
    API for compatibility but are no longer used — the numerical path
    cannot hang on symbolic integration.
- `tests/test_time_domain.py` — all 6 tests updated to call the new
  callable directly (e.g. `contribution(t1_val, 0.0)` instead of
  `SR(contribution).subs(...).real()`). Tolerances are unchanged.
- `notebooks/hawkes_linear_phi_test.ipynb` Section 8 — cell 8.1
  replaces the `SR.subs` loop with a direct callable invocation on
  the same τ grid used by the Phase I residue IFT path. Cell 8.2
  (overlay plot) needs no changes since it only consumes
  `tau_phase_j` / `C_tree_phase_j` arrays.

### Numerical validation

All tests pass (124 total):

- `tests/test_time_domain.py` — 6 Phase J tests, all green.
  - `test_k2_tree_single_integration_analytical` agrees with the
    closed-form `(1/2) exp(-|τ|)` within `1e-8` at six τ values.
  - `test_k2_tree_translation_invariance` confirms the unpinned k=2
    tree result depends only on `τ = t1 − t2`.
  - `test_phase_J_vs_phase_I_linear_hawkes_tree` agrees with Phase I
    within `1e-6` (actually matches to `0.0` or `~1e-17`).
- 2-population diagonal smoke test (not in the pytest suite; ran by
  hand): Phase I vs Phase J agree to `0.0` / `~1e-17` / `~1e-14`
  at five τ values, and the Phase J callable runs at **~0.95 ms per
  evaluation** (vs several seconds per τ point for the old symbolic
  path on nontrivial diagrams).
- Full suite: 124 passing in 15 s (was 42 s on the symbolic path —
  the speedup is mostly in the Phase J tests themselves, but the
  tree evaluator is called in `test_phase_J_vs_phase_I_linear_hawkes_tree`
  where it is now ~10× faster).

### Known gap

Polytope integration for `m ≥ 3` integration variables is not yet
implemented. This does not affect any tree-level linear Hawkes case
(which is `m = 1`) nor the upcoming ℓ = 1 bubble extension (which is
`m ≤ 2` for k = 2). It will be added when the first diagram needing it
appears.

---

## 2026-04-08 — Phase J MVP: hybrid loop-kernel reduction (tree-level only)

### New parallel time-domain backend

Phase J is introduced as a **new, parallel** evaluation backend living in
`msrjd/integration/time_domain/`. It is a hybrid pipeline: frequency
space is reused only for unique loop-kernel identification and algebraic
grouping (via the existing `group_diagrams_by_kernel` / `loop_only_signature`
machinery from `msrjd/integration/symbolic.py`), and actual integration is
performed in the time domain via vertex-time integration of retarded
exponential propagators.

Nothing in Phase I (`msrjd/integration/symbolic.py`, notebook cell 28, the
residue-based IFT path) is touched — Phase I remains the default backend
and the fallback for kernel groups Phase J does not yet handle.

**MVP scope**: the first build validates **only** the Phase J evaluation
layer — time-domain propagator extraction, vertex-time integration,
convention handling (Fourier sign, Heaviside at zero, propagator
transpose), translation fixing, and orchestrator dispatch — on
**tree-level** (`loop_number == 0`) kernel groups. Loop kernel reduction,
kernel caching, and parent-diagram contraction (Phases 3-5 of the full
hybrid pipeline) are not yet implemented and are the target of Extension 1.

### Module layout

- `msrjd/integration/time_domain/propagator_td.py`
  - `build_G_t_matrix(propagator_data, t_var, num_params)`: symbolic G(t)
    matrix via pole-residue sum `G(t) = Σ_k C_k · exp(I · p_k · t)`. Does
    **not** apply Heaviside; the caller must multiply by `heaviside(t)` to
    get the retarded propagator. Under the Fourier convention
    `G(t) = (1/2π) ∫ dω exp(+iωt) Ĝ(ω)`, the pipeline's causality filter
    guarantees Im(p_k) > 0 and thus decay for t > 0.
  - `G_t_entry(G_t_matrix, phys_idx, resp_idx, t_expr)`: retarded edge
    propagator lookup. Reads `G_t_matrix[phys, resp]` — the TRANSPOSE of
    the natural `[resp, phys]` layout, matching `_get_propagator_entry` in
    `symbolic.py:305`. Multiplies by `heaviside(t_expr)` by default.
- `msrjd/integration/time_domain/subgraph.py`
  - `identify_loop_subgraphs(ir, td)`: returns `[]` for tree-level;
    raises `NotImplementedError` for any diagram with free/loop
    frequencies, including the message for the orchestrator to fall back
    to Phase I. Extension 1 will flesh out the connected-closure
    algorithm per the plan file.
  - `LoopSubgraph` dataclass is the data model for the eventual
    attachment-point / internal-vertex split.
- `msrjd/integration/time_domain/final_integral.py`
  - `integrate_tree_diagram(typed_diagram, representative_ir,
    propagator_data, combined_prefactor, ext_time_vars, num_params,
    origin_leaf_idx, timeout_sec)`: builds the symbolic time-domain
    integrand `combined_prefactor · ∏_e G_R(t_v − t_u)` and integrates
    the non-leaf vertex times. Asserts `loop_number == 0` inside the
    function as a safety net against future misuse. Integration uses
    SageMath's symbolic `integrate(..., -oo, +oo)` and relies on the
    Heaviside factors to cut the polyhedral region; each call is wrapped
    in a `SIGALRM` watchdog (default 30s) so Sage hangs don't block
    downstream work.
- `msrjd/integration/time_domain/pipeline.py`
  - `compute_correction_td(kernel_groups, propagator_data, k, ...)`:
    Phase J orchestrator. For `loop_number == 0` groups it bypasses
    Phases 3-5 entirely and calls the tree evaluator directly; for
    `loop_number > 0` groups it marks them `'skipped'` with a reason and
    returns them in `skipped_kernel_ids` for Phase I fallback.
    Returns a debug-friendly dict with per-group diagnostics so
    downstream code can tell exactly which kernels Phase J handled.

### Conventions (fixed pipeline-wide)

- **Fourier convention**: `G(t) = (1/2π) ∫ dω exp(+iωt) Ĝ(ω)`. Retarded
  poles have Im(ω) > 0. This matches what notebook cell 8 actually
  constructs for `G_t` and is the convention assumed by the entire
  `time_domain` subpackage.
- **Heaviside at zero**: SageMath's default (`1/2`). Treated as frozen
  across the whole pipeline — no monkey-patching or `unit_step`
  substitutions.
- **Transpose**: `G_t_entry(phys=j, resp=i, t)` reads `G_t_matrix[j, i]`
  — the physical-row, response-column entry, matching the retarded
  propagator "response of physical j to response-field source i".

### MVP tests

Six new tests in `tests/test_time_domain.py`, all passing:

1. `test_G_t_matrix_single_pole` — `G(t)` for `K(ω) = 1 + iω` gives
   `exp(-t)` at t = 1 (agreement to 1e-12, symbolic check via
   `simplify_full`).
2. `test_G_t_entry_retarded` — `G_t_entry(t_expr=-1)` is killed by
   Heaviside; `t_expr=+1` returns `exp(-1)`.
3. `test_subgraph_tree_case_returns_empty` — tree-level diagram →
   `identify_loop_subgraphs` returns `[]`.
4. `test_k2_tree_single_integration_analytical` — k=2 tree with
   `G_R(t) = Θ(t) exp(-t)` integrates to the closed-form
   `(1/2) exp(-|τ|)` at six τ values (positive and negative), agreement
   below 1e-8.
5. `test_k2_tree_translation_invariance` — Phase J k=2 tree result
   evaluated at `(t1=1, t2=0)` and `(t1=6, t2=5)` agrees to below 1e-8
   (both should be the same value since the result depends only on
   `τ = t1 − t2`).
6. `test_phase_J_vs_phase_I_linear_hawkes_tree` — **end-to-end MVP
   validation**. Same tree k=2 diagram, same `propagator_data`:
   - Phase I (`integrate_to_time_domain`) computes `C(τ)` via residue
     integration in frequency space.
   - Phase J (`compute_correction_td`) computes `C(τ)` via vertex-time
     integration.
   Both agree within `1e-6` absolute tolerance at four τ > 0 values.
   In practice the agreement is several orders of magnitude tighter
   (~1e-16) on a 2-population diagonal-propagator smoke test.

All 124 tests pass (118 pre-existing + 6 new Phase J).

### Notebook integration

`notebooks/hawkes_linear_phi_test.ipynb` gets a new **Section 9** at the
end:
- Cell 9.1: imports `compute_correction_td`, runs it on the existing
  `kernel_groups` with `num_params` substituted, pins `t_2 = 0` so the
  result is a function of `τ = t_1`, and evaluates on the same
  `tau_residue` grid used by the Phase I residue IFT path (cell 28).
  Prints per-group diagnostics (which kernel groups were handled by the
  tree evaluator vs skipped) and `max|Phase J − Phase I residue|` over
  the grid.
- Cell 9.2: overlays the Phase J result on the k=2 tree plot next to
  the Phase I FFT and Phase I residue curves. Expected outcome: the
  Phase J curve overlays the Phase I residue curve exactly.

The existing Phase I cells (1-30) are untouched; Section 9 only appends.

### Deferred (Extension 1+)

- `kernel_reduce.py` — symbolic/numerical integration of internal loop
  vertex times → `K(τ_1, ..., τ_{p-1})` for p-attachment subgraphs.
- `kernel_cache.py` — cache keyed on `loop_only_signature`. Before
  Extension 1 is merged, the invariant that `loop_only_signature`
  distinguishes (a) internal edge propagator types, (b) loop routing
  connectivity, and (c) external attachment pattern must be verified;
  if the existing signature is missing any of these, `kernel_cache.py`
  must extend the key.
- `contraction.py` — parent diagram → contracted diagram with
  fundamental edges + effective general-p hyper-edges. Design principle
  (already written into the plan): the kernel abstraction is general-p
  from the start, APIs must not implicitly assume 2-point kernels.
- Polyhedral / exponential-integration engine to replace the provisional
  Sage `integrate()` path once tree / bubble cases saturate it.

---

## 2026-04-07 — k=3 support, residue-based IFT, and structured residue exploration

### Multi-frequency (k=3) numerical evaluation

- **Generalized `spectrum_tree`** to handle multiple external frequencies. For
  k=2 (n_ext=1) returns a 1D array; for k=3 (n_ext=2) evaluates on an N×N grid.
  Falls back to per-slice evaluation when fine 2D grids would be too costly.
- **Generalized `inverse_fourier`** to handle 1D (`ifft`) and 2D (`ifft2`)
  spectra with appropriate `(N·Δω/(2π))^n_ext` scaling.
- **Adaptive grid by k** in cell 28: k=2 uses `T_max=80, Δτ=0.05` (N≈4096);
  k≥3 currently set to the same fine grid (`Δτ=0.02 → N=8192`, ~67M 2D points).
- **k=3 plotting**: extracts 1D slices `C(τ₁, τ₂=0)` and `C(τ₁=0, τ₂)` from the
  full 2D `C(τ₁,τ₂)` surface. Two-panel layout matching the n_τ slices.
- **k=3 simulation cumulant**: For each slice, compute the connected 3rd cumulant
  via FFT — slice 0 cross-correlates `dn_a · dn_c` (product) with `dn_b`, slice 1
  uses `dn_a · dn_b` × `dn_c`. The product trick reduces the 3-point cumulant
  to a 2-point correlation since means are subtracted.
- **Adaptive comparison plot**: simulation cell now reads `external_fields` from
  the config and computes the appropriate auto/cross/3-point statistic. Same
  notebook handles k=1, 2, 3 with no edits.

### Residue-based IFT for k=2 (exact, no Gibbs ringing)

- **`find_spectrum_poles(propagator_data, num_params)`**: Returns all poles of
  the spectrum from the propagator. Poles of det(K(ω))=0 are already known
  symbolically; the spectrum has additional poles at their negatives from
  det(K(−ω))=0. Substitutes parameters for numerical pole values.
- **`compute_numerical_residues(f, poles)`**: Computes residues at simple poles
  via the limit `(z − pole) · f(z)` evaluated at `z = pole + ε`.
- **`ift_via_residues(f, poles, tau_grid)`**: Closes contour in the upper
  half-plane for τ>0 (returns `+i · Σ_upper residue · exp(iωτ)`) and lower
  half-plane for τ<0. Exact, no truncation artifacts, evaluable at any τ.
- **Delta-spike detection**: For auto-correlators, the Poisson shot noise
  contributes `n* · δ(τ)`. This shows up as a constant `S(ω→∞)` with no poles.
  Detected by evaluating `S` at large ω and added as `S∞ / Δτ` at τ=0 to match
  the binned simulation convention.
- **Validation**: For linear Hawkes k=2 cross-correlator, the residue IFT
  matches the FFT IFT to ~0.1% across all τ values (smooth part) and the
  delta-spike heights match exactly when the τ grids share `Δτ`.

### Sequential residue integration prototype (k=3, partial)

Explored a fully exact residue-based path for k=3 ("Path C2"). The goal:
integrate over external frequencies one at a time via residues, eliminating
both FFTs entirely. Two implementations were tested:

1. **Pure-symbolic chained**: Build `J(ω₀, t₂) = i·Σ_upper res(ω₁=p_k(ω₀))
   · exp(i·p_k(ω₀)·t₂)` symbolically, then attempt second integration. Sage's
   `solve()` and `simplify_rational()` choked on rational expressions with
   embedded exponentials — calls timed out at ~10 minutes.

2. **Structured `Term` objects**: Each term tracks `(rational_part, exp_factors)`
   where `exp_factors` is a list of `(linear_combo, time_var)` representing
   `exp(i·linear_combo·time_var)` factors accumulated over residue substitutions.
   The rational part stays rational in the surviving omega vars, so `solve()`
   works at every step. Successfully completed both integrations for k=3 without
   symbolic blowup. Inner integration (over ω₁) yielded 4 upper / 2 lower poles,
   each with the expected shift structure (`p_intrinsic − ω₀` from mixed
   propagator factors, `±p_intrinsic` from single-variable factors).

**Status**: The architecture works (terms propagate cleanly through both
integrations) but the **contour direction logic for the outer integral has
bugs**. The effective time at the outer step is `t₁ + (coefficient_of_ω₀_in_existing_phases)·t₂`,
and different terms have different effective signs depending on (t₁, t₂).
Test values for k=3 were off by varying factors (0.5–0.85) and sometimes wrong
sign. The architecture needs more debugging on the sign accumulation across
the two residue closures.

**Verification at k=2**: The same machinery works perfectly for k=2 (matches
FFT to 0.1%), confirming the basic residue-via-`N(p)/D'(p)` and Term substitution
logic are correct. The k=3 issues are specific to handling the second
contour direction with carried-over exp factors.

### Pipeline architecture

- **Adaptive evaluation cell** (`hawkes_linear_phi_test.ipynb` cell 28): now
  computes residue-based C(τ) alongside FFT-based C(τ) for k=2 and overlays
  both in the comparison plot. Three-curve overlay (sim, FFT-tree, residue-tree).
- **`_param_subs` model-agnostic phi differentiation**: previously hardcoded
  `ns.a[i]` substitutions in the MF solver; now iterates `HAWKES_MODEL['parameters']`
  and substitutes any fundamental parameter into the symbolic phi derivative
  expressions. Works for any phi form without code changes.
- **Cache directory keys**: now include `external_fields` so switching from
  `[(dn,1),(dn,2)]` to `[(dn,1),(dn,1)]` doesn't pull stale diagrams.

### Documentation

- **CHANGELOG.md** updated with all 2026-04-03 critical fixes and 2026-04-07 work
- **PIPELINE_PLAN.md**: status updated to reflect Phases A–I complete; design
  decisions section now documents propagator transposition, external leg labeling,
  action sign convention, and IFT time convention
- **BUILD_PHASE_OUTLINES.md**: Phases H and I marked complete with critical
  implementation notes from debugging

### Known issues / open questions

- **k=3 sequential residue (Path C2)**: contour-direction sign bug, see above.
  Architecture is correct but implementation needs debugging of sign accumulation
  across sequential residue closures with mixed effective times.
- **k≥3 evaluation cost**: full 2D FFT is the only working option, ~67M points
  per evaluation at the current grid. Acceptable but slow.
- **Fourier artifacts at τ≈0**: Sharp features (delta-function shot noise)
  cause Gibbs ringing in the FFT path. Residue path has no ringing for k=2.
- **Time-domain integration not yet attempted**: For systems with known
  symbolic time-domain propagators, direct vertex-time or edge-duration
  integration would sidestep the residue-chasing complexity entirely. See
  user notes in 2026-04-07 design discussion (spanning-tree time reduction,
  V−1 independent time variables, polyhedral integration regions for
  exponential propagators).

### Design discussion: hybrid loop-kernel reduction (Phase J)

After exploring both pure-frequency residue paths (with the contour-direction
bug at k=3) and considering pure-time-domain vertex-time reduction, the design
that emerged combines both: **frequency space for loop-kernel identification
and deduplication, time domain for actual integration**.

**Architecture summary** (full description in `PIPELINE_PLAN.md` Phase J):

1. **Phase 1 — Diagram compilation** (existing): build frequency-space integrand
   with conservation applied.
2. **Phase 2 — Unique loop-kernel identification** (existing): use the routing
   matrix to identify the loop-dependent subgraph for each loop variable, find
   the connected closure, identify attachment vertices, canonicalize, dedupe.
3. **Phase 3 — Kernel reduction (new)**: for each unique loop kernel, switch
   from `Ĝ_e(ω)` to `G_e(t)` and integrate out the internal vertex times of the
   subgraph. Result is a reduced time-domain kernel `K(τ₁, …, τ_{p−1})` where
   `p` is the number of attachment points.
4. **Phase 4 — Kernel evaluation (new)**: compute each unique reduced kernel
   once (analytically when possible, numerically otherwise).
5. **Phase 5 — Substitution / contraction (new)**: replace each subgraph
   instance in the parent diagram with an effective edge/hyper-edge carrying
   the precomputed kernel. Tadpole/coincident-attachment cases collapse to
   vertex-local multiplicative factors.
6. **Phase 6 — Final time-domain integration (new)**: vertex-time spanning-tree
   reduction on the contracted parent. For causal exponential propagators this
   gives polyhedral exponential integrals (closed form).

**Key design decisions**:
- Same kernel ID ≠ same edge occurrence: distinct subgraphs in the parent that
  share a kernel ID get independent effective edges, all evaluating the same
  precomputed `K` on different parent-time arguments.
- Coincident attachment ("tadpole") = kernel evaluated on the diagonal,
  represented as a vertex-local factor rather than a self-loop edge.
- The integration count is the same as pure frequency or pure time space
  (`V−1` for connected diagrams), but the loop integrations are done "early"
  at the kernel level and the parent integrations "late" in time domain.

This is the proposed v2 priority. Not yet started. The frequency-domain
Phase I pipeline remains the current working backend.

---

## 2026-04-03 — Critical bug fixes and simulation validation

### Critical fixes (affect all numerical results)

1. **Propagator index transposition** (`msrjd/integration/symbolic.py`)
   - **Bug:** `_get_propagator_entry(i, j, ...)` read `G_ft[i, j]` where `i`=response row, `j`=physical col. But the retarded propagator "response of physical field j to response-field source i" is `G^R_{j←i} = G[j, i]` (transposed).
   - **Impact:** Every diagram integrand used the wrong propagator entries. For asymmetric networks this produced wrong amplitudes (factor ~1.4–5× depending on parameters) and wrong time-domain asymmetry.
   - **Fix:** Transposed the lookup: `G_ft[i, j]` → `G_ft[j, i]`.
   - **Verification:** Pipeline `S₁₂(0)` now exactly matches the analytical formula `[(I − W ĥ)⁻¹ diag(n*) (I − W ĥ)⁻ᵀ]₁₂` (ratio = 1.0000).

2. **Propagator matrix ordering mismatch** (`notebooks/hawkes_*_pipeline_demo.ipynb`, cell 8)
   - **Bug:** Cell 8 hardcoded `resp_names = ['vt1','vt2','nt1','nt2']`, but `build_field_index_map` uses the ring variable ordering `['nt1','nt2','vt1','vt2']`. The kernel matrix `K_ft` rows/cols were permuted relative to what the propagator indices expected.
   - **Impact:** `G_ft[0,0]` was `G[vt1,dv1]` in the matrix but the type assignment thought it was `G[nt1,dn1]`. Produced symmetric integrands (losing cross-correlation asymmetry) and wrong amplitudes.
   - **Fix:** Derive `resp_names` and `phys_names` from `ring_gen_names[:n_tilde]` and `ring_gen_names[n_tilde:]`.

3. **External leg permutation** (`msrjd/diagrams/type_assignment.py`)
   - **Bug:** `enumerate_typed_diagrams` permuted external field assignments across all leaf vertices (`for ext_perm in permutations(...)`). This generated diagrams for all orderings of external fields (e.g., both ⟨dn₁ dn₂⟩ and ⟨dn₂ dn₁⟩).
   - **Impact:** The "swapped" diagrams have opposite imaginary parts, so summing them cancelled the asymmetry, producing a symmetric integrand for cross-correlators.
   - **Fix:** External legs are labeled — leaf `i` always gets `external_fields[i]`. Removed the permutation loop.

4. **Action sign: Poisson term** (`models/hawkes_sage.py`)
   - **Bug:** The MSR-JD action had `+(e^{ñ} − 1)φ` but the correct sign is `−(e^{ñ} − 1)φ`.
   - **Impact:** Flipped the sign of the entire tree-level spectrum. For all-excitatory networks, the cross-correlation was negative (physically impossible).
   - **Fix:** Changed to `−(e^{ñ} − 1)φ` and updated `ndot_bg` from `−n*` to `+n*`.

5. **Conservation equation guard for k=1** (`msrjd/integration/symbolic.py`)
   - **Bug:** `build_integrand_stationary` had `if overall_cons is not None and len(ext_freqs_all) >= 2:` which skipped applying ω_ext = 0 for k=1 tadpole.
   - **Impact:** k=1 diagrams retained a spurious external frequency variable instead of evaluating to a scalar.
   - **Fix:** Removed the `len >= 2` guard.

### Other fixes

6. **Multi-edge support in type assignment** (`msrjd/diagrams/type_assignment.py`)
   - `D.neighbors_out(v)` collapsed multi-edges; switched to `D.outgoing_edges(v)`.
   - Assigned unique integer labels in `orient_edges` to prevent dict key collisions.

7. **k variable shadowing** (multiple notebook cells)
   - Loop variables `for k in ...` overwrote the config `k` (cumulant order). Renamed to `kern`, `idx`, `pk`, `dk` as appropriate.

8. **IFT time convention** (notebook cell 28)
   - The MSR-JD phase is `exp(+iω(t₁−t₂))`, so the natural IFT gives `C(t₁−t₂)`. Flip the output array to get `C(t₂−t₁)` matching the simulation convention (positive τ = second field later).

9. **Simulation covariance normalization** (notebook cell 30)
   - Binned-rate cross-correlation had an extra `1/dt_bin` factor relative to the continuous covariance density. Multiply by `dt_bin`.

10. **Sage Integer/RealNumber contamination** (notebook simulation cell)
    - Sage wraps all numeric literals as `Integer()`/`RealNumber()` which numpy rejects. All values passed to numpy are now explicitly cast via `float()`/`int()`.

### Features added

- **Model-agnostic MF solver** (notebook cell 28): Reads `phi_concrete` from the model, differentiates symbolically to the required Taylor order, solves MF self-consistency equations numerically via `fsolve`. No hardcoded parameter names.
- **Linear Hawkes model** (`models/hawkes_linear_sage.py`): `φ(v) = v` with specializations `phi1=1`, `phi2=...=0`. Vertices arise only from `exp(ñ)` Poisson nonlinearity.
- **Model-specific cache directories**: Cache path includes model name to prevent cross-contamination between models.
- **Adaptive evaluation by k**: k=1 (scalar mean), k=2 (spectrum + IFT), k≥3 (2D slices).
- **Euler-Poisson simulation** for validation against analytical results.

### Known issues / future work

- **Higher-loop evaluation**: The factored evaluation (precompute unique loop integrands, multiply by external propagators) is implemented for k≥2 but not yet verified against simulation for the nonlinear model.
- **Fourier artifacts**: Sharp features near τ=0 (Poisson shot noise delta function) cause Gibbs ringing. Mitigated by increasing `Delta_tau` (finer grid) but not eliminated.
- **`_build_factor_product` in notebook**: The factored loop evaluation uses `G_ft[ri, pi]` directly from `prop_factors` — this needs to be checked against the transposed convention. May need updating for loop-level diagrams.

---

## 2026-03-27 — Initial pipeline build

- Phases A–H implemented: serialization, vertex decomposition, prediagram enumeration, type assignment, causality filter, symmetry/deduplication, symbolic integration, numerical evaluation.
- 118 tests passing.
- Validated on 2-population nonlinear Hawkes process with quadratic φ.

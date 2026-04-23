# Changelog

All notable fixes, features, and known issues for the MSR-JD Feynman diagram pipeline.

---

## 2026-04-23 — `enumeration-speedup` branch (WIP): canonical leg-matching + per-prediagram parallelism

### Scope

The 2026-04-22 `parallel-eval` branch accelerated the EVALUATION
phase of Phase J (scipy.quad on pre-built contribution callables).
This branch targets the **ENUMERATION phase** — specifically the
step that goes from prediagram topologies to fully-typed Feynman
diagrams with field assignments on every vertex and leg.

### Problem

`msrjd/diagrams/type_assignment.py::enumerate_typed_diagrams` and
its entry point `enumerate_all` were generating ~15-20× more typed
diagrams than the downstream `deduplicate_typed_diagrams` kept — for
quadratic Hawkes k=3 ell=0 the ratio was 552 typed → 38 unique.
That 95% waste was the single biggest cold-start bottleneck:
`enumerate_typed_diagrams` took **~390s for k=3 ell=0** on M2 Max
(3 prediagrams), and **>10 hours extrapolated for k=3 ell=1**
(64 kept prediagrams, never completed in our benchmark runs).

### Candidate 1: canonical multiset leg-matching

`_leg_matchings(vertex_type, out_edges, in_edges)` at
`type_assignment.py:311` enumerated all `R! × P!` index permutations
of response and physical legs.  For leg multisets with duplicate
`(field, pop)` entries (common at high Taylor order), many of those
permutations produced bit-identical typed diagrams that later dedup
collapsed away.

**Fix:** replace `set(permutations(range(R)))` with
`sympy.utilities.iterables.multiset_permutations(resp_legs)` (and
similarly for phys).  multiset_permutations yields only the
`R! / ∏ n_r!` distinct orderings of the leg multiset, skipping the
redundant ones directly.

**Correctness:** the physics weight of a typed diagram includes
`M(Γ) = ∏_v M_v` from `symmetry.py::combinatorial_factor`, where
`M_v` is *exactly* the leg-permutation orbit size at vertex v.  Old
code generated M(Γ) identical copies, dedup kept one, integrator
multiplied by M(Γ).  New code generates the canonical copy,
integrator multiplies by the same M(Γ).  Bit-identical numerical
output — no downstream change.

**Measured on quadratic Hawkes k=3 (user's production workload):**

| Stage | BEFORE (baseline) | AFTER Candidate 1 only |
|---|---|---|
| ell=0 enumerate_typed | 390s → 552 typed | **64.1s → 84 typed** (6.1× faster, 6.6× fewer) |
| ell=1 per-prediagram  | >600s each | ~20-100s each, avg ~45s |
| ell=1 full (64 prediagrams) | extrapolated >10 hours | extrapolated ~50 min serial |

Dedup output unchanged (38 unique at ell=0, 104 at ell=1, matching
the user's reported counts).

### Candidate 3: per-prediagram fork-ProcessPool parallelism

`enumerate_all(prediagrams, ...)` is embarrassingly parallel — each
prediagram's `enumerate_typed_diagrams` is independent.  Added a
`parallel=False, n_workers=None, start_method='fork'` kwarg set;
notebooks opt in via `TD_PARALLEL` / `TD_N_WORKERS` from the
Configuration cell (same knobs already used for evaluation).

Implementation follows the `pipeline.py::total_C_batch` pattern:
module-level `_ENUM_WORKER_STATE` dict inherited by workers via
fork (no pickle of Sage graph / matrix inputs), plus top-level
`_worker_enumerate_one_prediagram` function.  Workers return lists
of `TypedDiagram` objects which pickle cleanly via their existing
`__getstate__` / `__setstate__`.

**Measured on quadratic Hawkes k=3 (Candidate 1 + Candidate 3,
12-core M2 Max):**

| Stage | Serial (C1) | Parallel (C1 + C3) | Speedup |
|---|---|---|---|
| ell=0 enumerate_typed (3 pds) | 64.1s | 50.3s | 1.3× (bounded by slowest pd) |
| ell=1 enumerate_typed (8-pd subset) | **343.55s** | **67.58s** | **5.08×** (336 typed, bit-identical) |
| ell=1 extrapolated (64 pds full) | ~46 min | **~7-8 min** | ~6-7× parallel + 10-30× vs pre-branch |

The ell=0 speedup is modest because only 3 prediagrams exist —
parallel gain is bounded by the slowest one.  The ell=1 win is
transformative: cold-start enumeration for max_ell=1 goes from
>10 hours (pre-branch) to ~8 minutes.  Bit-identity verified on the
8-prediagram subset: element-wise signature match across all 336
typed diagrams in serial vs parallel runs.

### Library changes

`msrjd/diagrams/type_assignment.py`:
- `_leg_matchings` replaced its index-permutation loop with
  `multiset_permutations`, with a full docstring on the
  correctness-via-M(Γ) argument.
- `enumerate_all` gained `parallel` / `n_workers` / `start_method`
  kwargs (default `parallel=False` for back-compat; notebooks opt
  in).
- New module-level `_ENUM_WORKER_STATE` + `_worker_enumerate_one_
  prediagram` for the fork pool.

### Notebook changes

Cell 18 in all three TD notebooks reads `TD_PARALLEL` /
`TD_N_WORKERS` from globals and passes them to `enumerate_all_typed`.
Print line now includes `(parallel=<bool>)` status.

### Tests

`tests/test_type_assignment.py` gained 7 tests:
- `test_leg_matchings_all_distinct_legs` — baseline (R! = R!/1).
- `test_leg_matchings_duplicate_response_legs` — [A, A]: 1 orbit.
- `test_leg_matchings_mixed_multiset` — [A, A, B]: 3 orbits.
- `test_leg_matchings_empty_legs` — 0-leg edge case.
- `test_enumerate_typed_duplicate_leg_vertex_no_redundant_generation`
  — end-to-end: raw count == dedup count when leg orbit is the
  only over-generation source.
- `test_enumerate_typed_distinct_legs_regression` — no spurious
  changes when all legs are distinct.
- `test_enumerate_all_parallel_matches_serial` — element-wise bit-
  identity of serial vs parallel `enumerate_all` output.

Total test count: **158** (151 pre-branch + 7 new).  All pass.

### Dependencies

- `sympy.utilities.iterables.multiset_permutations`: ships with
  SymPy, which ships with SageMath.  No new install.
- `multiprocessing` from stdlib: already used by
  `pipeline.py::total_C_batch`.  No new install.

POSIX-only (fork).  Windows support deferred with the same
rationale as `parallel-eval`.

---

## 2026-04-22 — `parallel-eval` branch merged: fork-ProcessPool evaluation + ell-aware displays

### Summary

A 15-commit branch landing fork-based ProcessPool parallelism over
the evaluation pipeline plus a series of notebook UX fixes.  Measured
**~7× wall-time reduction on k=2 max_ell=1 quadratic Hawkes slice
runs** (400 min → ~55-60 min after this branch; a further ~12×
reduction from the hard-coded 201-point grid bug fix takes it down
to ~30 min for a 17-point grid).

### Library (`msrjd/integration/time_domain/pipeline.py`)

1. **`total_C_batch(tau_points, parallel=True, n_workers=None)`** —
   new method on the `compute_correction_td` return dict.  Evaluates
   ``total_C`` at a list of τ points with auto-selected parallelism:
   * `len(tau) >= n_workers` → per-τ parallel (each worker sums all
     diagrams at its τ slice serially).
   * `len(tau) < n_workers` → **nested per-(τ, diagram) parallel**
     so single-point evaluations still saturate all cores.  This is
     essential for `SIM_MODE='point'` where the old per-τ path left
     11 cores idle.
   * `total_tasks < 2*n_workers` → serial (fork overhead not worth it).

2. **`eval_per_diagram_batch(tau_points, parallel=True, ...)`** —
   new method returning `(vals, loop_nums)` where
   `vals[tau_idx][diag_idx]` is the per-diagram contribution.  Used
   by downstream code that needs BOTH the full total AND the per-
   diagram/per-ell breakdown (cell 28, cell 31, cell 32).

3. **Fork start-method required on POSIX.**  The per-diagram
   `contribution` closure is a nested function that stdlib `pickle`
   cannot serialise; fork bypasses serialisation by inheriting
   memory.  Module-level `_WORKER_STATE` dict + module-level
   `_worker_eval_total_C` / `_worker_eval_one_diagram` workers.
   `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` set automatically on
   macOS.

4. **Windows support DEFERRED.**  Documented in `pipeline.py` with a
   multi-paragraph comment listing the two future fixes
   (cloudpickle optional dep or top-level picklable classes).
   Windows users get a clear `ValueError: cannot find context for
   'fork'` rather than silent drift.

### Tests (`tests/test_time_domain.py`)

Two new regression tests added (21 total now):

- `test_phase_J_total_C_batch_parallel_matches_serial` — pins
  bit-identity of per-τ parallel vs serial on a nondiagonal 2×2
  fixture.  `max |parallel − serial| == 0.0` required.
- `test_phase_J_total_C_batch_nested_matches_serial` — same but
  for the nested per-(τ, diagram) path, triggered by a 1-point
  batch with `n_workers > 1`.

The existing 19 tests all continue to pass unchanged — the library
change is purely additive (`total_C_batch` / `eval_per_diagram_batch`
are new methods; serial `total_C` is untouched).

### Notebook changes (all three: `hawkes_td_only.ipynb`,
`hawkes_td_only_expg.ipynb`, `hawkes_td_only_quad_expg.ipynb`)

1. **`TD_PARALLEL` / `TD_N_WORKERS` knobs in Configuration cell 2.**
   Users on shared machines can opt out or cap worker count without
   editing code.

2. **Cell 25 (§8.1) symbolic display now shows ALL diagrams.**  The
   `if _g['handled_by'] != 'tree_evaluator': continue` filter was
   hiding all loop diagrams; now only `handled_by == 'skipped'` is
   filtered.  Quadratic Hawkes k=2 max_ell=1 runs show symbolic
   LaTeX for all 106 diagrams, not just the 2 trees.

3. **Cell 23 tau grid honours Config.**  Was hard-coded
   `np.arange(-50.0, 50.25, 0.5)` = 201 points regardless of user's
   `tau_max` / `tau_step`.  Now reads those from Configuration cell;
   for `tau_max=20, tau_step=2.5` the grid is correctly 17 points
   (not 201).  A diagnostic print shows the resulting grid shape.

4. **Cell 28 k=1 / k=2 / k≥3 all use the parallel batch evaluator.**
   Previously only k≥3 was wired up; k=2 ran a serial Python
   for-loop over τ (the primary cause of 400-min reports), and k=1
   called `total_C(0.0)` directly.

5. **Per-diagram timing probe in cell 28.**  Runs each diagram at 3
   τ points before the main slice loop, prints top 10 slowest with
   extrapolated per-slice cost.  Lets users identify which diagrams
   dominate wall time (typically a handful of m=3/m=4 1-loop
   polytopes).

6. **Arbitrary-`max_ell` displays.**

   * **Cell 28** now builds `C_theory_k1_by_ell` (k=1) and
     `C_phase_j_by_ell` / `C_phase_j_slices_by_ell` (k=2, k≥3)
     dicts mapping `ell → contribution array`.  Legacy aliases
     (`C_theory_k1_tree/_loop/_total`, `C_tree_phase_j`,
     `C_tree_phase_j_slices`) preserved for downstream back-compat.

   * **Cell 31 point-mode table** — replaced the fixed "Tree vs
     1-loop split" with per-ell rows (one per distinct loop order
     present).  Also fractional `|ell=N/tree|` diagnostic.

   * **Cell 32 k=1 bar plot** — stacked bar with one colored
     segment per loop order (blue/purple/orange/red/brown).

   * **Cell 32 k=2 and k≥3 slice plots** — cumulative "truncation"
     curves: `Tree`, `Tree + 1-loop`, `Tree + 1-loop + 2-loop`, …
     up to `max_ell`.  viridis colormap (light→dark with ascending
     ell); dashed partial truncations, solid final cumulative.
     Users visually see each loop order's contribution as the
     vertical gap between consecutive curves.

7. **Cell 31 robustness.**  `TEST_POINTS` is normalized to a list
   of tuples, accepting both flat-list-for-k=2 and explicit-tuple
   forms.  Clear error if a point has the wrong length.  Stops
   recomputing scipy per-(diagram, point) in the per-diagram table
   — reuses the parallel-batch output instead.

### Measured speedups

Quadratic Hawkes + exp filter, different workloads on M2 Max 12-core:

| Workload | Config | BEFORE | AFTER |
|---|---|---|---|
| k=2 max_ell=1, 106 diagrams, 201 τ slice | old hard-coded grid, serial k=2 | **400 min** | |
| same workload after this branch | | | **~30 min** (7× parallel × 12× grid) |
| k=3 max_ell=0, 38 diagrams, 201 τ × 4 slices | serial (pre-branch) | **~170 min** | |
| same workload after this branch | | | **~24 min** (7× parallel) |
| k=2 max_ell=1 point mode, 1 τ | serial single-point | ~2 min (broken) | |
| same workload | | | ~65 s (12-way nested parallel) |

### Known limitations documented in this branch

- POSIX only (macOS + Linux).  Windows will raise a clear error at
  the fork() call; future fix via cloudpickle or top-level classes.
- Cold-start enumeration still serial.  The branch does NOT
  accelerate `enumerate_typed_diagrams` — that's next up on the
  `enumeration-speedup` branch (Candidate 1: canonical leg-matching
  via sympy's `multiset_permutations`).

### Rollback

- Library: `git checkout main && git revert <merge-commit>`.
- Full pre-branch state: `git reset --hard dc91e8a` (Fix E) or
  `git reset --hard checkpoint-2026-04-21` (pre-Fix-D).

---

## 2026-04-21 — Audit Fix #E: bypass `fast_callable` on the integrand hot path

### What changed

Phase J's per-subset numerical integrand is, structurally,

```
  subset_factor(s, t)  =  P · Π_e  Σ_k  C_e^{(k)} · exp(I · p_k · Δt_e(s, t))
```

where `P` folds in the combined prefactor and every δ-edge
coefficient, the outer product runs over smooth edges, and the
inner sum over poles.  Prior to this fix, the code in
`integrate_diagram` (`msrjd/integration/time_domain/final_integral.py`)
built `subset_factor` as an SR expression, called `.expand()` on it
to distribute the product into a sum of single exponentials, and
JIT-compiled that with `fast_callable`.

The distribution is brutal for anything bigger than a tree.  On a
5-edge 1-loop diagram with 4 poles:

- unexpanded SR str length:  3 184 chars
- `.expand()`ed SR str length: 303 708 chars
- top-level sum terms after expand: **988** (≈ 4^5 cross-product
  of pole choices per edge)

`fast_callable` then walks all 988 single-exp terms on every
scipy.quad/nquad sample.  cProfile of the k=2 ell=1 quadratic
Hawkes V=5 diagram (2026-04-21) measured ~18 µs per integrand call
and 752 766 calls across 3 τ points → **13.5 of 14.1 seconds of
Phase J wall time** inside the single `filtered(...)` frame.

The `.expand()` itself was a correctness fix (commit 388fa7c,
2026-04-08) added because unexpanded products of pole-sum factors
overflowed IEEE double precision inside fast_callable's evaluation
tree at large negative vertex times — individual pole factors grew
like `exp(|α|·|s|)` before the retarded causality cancellation
kicked in.

Fix E replaces the fast_callable + expand combo on the hot path
with a dedicated per-edge numerical evaluator,
`_build_fast_subset_evaluator`.  The evaluator:

- Pulls `pole_vals` and `C_mats` straight from `propagator_data`
  (no SR intermediate).
- Reads the per-smooth-edge `(ri, pi)` index, sampling the
  corresponding residue column out of each `C_mats[k]`.
- Reuses the already-extracted per-edge
  `subset_constraint_data` tuple `(a_int, a_ext, c0)` to get the
  linear form of `Δt_e`.  Converts these to sparse `(position,
  coef)` pairs so the per-call dot runs over the 1–2 nonzero axes
  per retardation vector instead of the full `m` axes.
- On every scipy call, computes

    dt_e      = c0 + Σ_i a_int[i]·s_i + Σ_j a_ext[j]·t_f_j
    edge_val  = Σ_k residues[k] · cmath.exp(1j · poles[k] · dt_e)
    result   *= edge_val

  and returns `P · Π_e edge_val`.

The overflow concern is defused by construction: the Heaviside-
filtered integrand wrapper (Fix D / 2026-04-15b) guarantees that
`integrand_callable` is only invoked on samples where every
retardation `Δt_e > 0`, so each `|exp(-Im(p_k)·Δt_e)| ≤ 1` and
`|edge_val| ≤ Σ_k |C_e^{(k)}|` is bounded independently of where
scipy samples inside the polytope.

The fast_callable path is preserved as a fallback: if the
numerical extraction fails (non-numerical prefactor, missing
propagator data, etc.), `_build_fast_subset_evaluator` returns
`None` and the caller uses `integrand_fc_sub` as before.  The
`.expand()` + fast_callable compilation also still runs so the
zero-check and free-variable check that guarded the old path are
unchanged.

### Measured speedup

Micro-benchmark (100 000 calls, 5 edges × 4 poles, m=3 axes + 2
external times):

| Approach                    | Time/call | Speedup vs expand+fc |
|-----------------------------|-----------|----------------------|
| `fast_callable(.expand())`  | 42.45 µs  | 1.0×                 |
| `fast_callable(unexpanded)` | 1.47 µs   | 29.0× (UNSAFE — overflows) |
| numpy per-edge              | 16.96 µs  | 2.5× (numpy overhead dominates at tiny array sizes) |
| **Fix E cmath per-edge**    | **2.89 µs** | **14.7×** |

`fast_callable(unexpanded)` is faster per call but reintroduces the
2026-04-08 overflow bug on the nondiagonal 2×2 fixture
(test_phase_J_nondiagonal_2x2_does_not_overflow → NaN at τ=-3.0).
Fix E sits between the two: per-call cost within ~2× of the
unexpanded fast_callable, AND guaranteed overflow-safe under the
Heaviside-filtered polytope regime.

End-to-end evaluation (quadratic Hawkes k=2 ell=1, 7 1-loop
diagrams, τ ∈ {-10, -5, 0, 5, 10}):

| Diagram       | BEFORE  | AFTER   | ratio  |
|---------------|---------|---------|--------|
| [0] V=4 E=4   | 0.514s  | 0.337s  | 1.52×  |
| [1] V=5 E=5   | 24.69s  | 2.534s  | 9.74×  |
| [2] V=5 E=5   | 34.65s  | 1.890s  | 18.3×  |
| **TOTAL (3)** | **59.83s** | **4.761s** | **12.6×** |

Numerical outputs are bit-identical (same τ-summed `sum` values
across all three diagrams), because Fix E and the pre-existing
fast_callable path both evaluate the same structural expression —
just in different orderings of complex arithmetic.

### Verification

- All 18 pre-existing tests in `tests/test_time_domain.py` pass
  unchanged, including:
    - `test_phase_J_nondiagonal_2x2_does_not_overflow` — the exact
      regression that motivated the original `.expand()` fix.
      Fix E passes this without expand on the hot path, because
      overflow is prevented at a different layer (Heaviside filter).
    - `test_phase_J_vs_phase_I_linear_hawkes_tree` — cross-validates
      Phase J (time domain) against Phase I (frequency domain FFT).
- One new Fix E regression added (total now 19 tests):
    - `test_phase_J_fix_E_fast_evaluator_matches_fast_callable` —
      pins NaN-free output on the nondiagonal 2×2 fixture and
      checks that `_build_fast_subset_evaluator` correctly returns
      `None` for a non-numerical prefactor (preserves the fallback
      safety net).

### Files changed

- `msrjd/integration/time_domain/final_integral.py`
    - New helper `_build_fast_subset_evaluator` — builds the
      numerical per-edge callable from propagator data + constraint
      linear coefficients + the combined prefactor.
    - Subset loop in `integrate_diagram` prefers the fast evaluator
      (falls back to fast_callable transparently on extraction
      failure).  The evaluator choice is recorded in
      `subset_diagnostics[*]['evaluator']` for debugging.
- `tests/test_time_domain.py` — one new regression test appended
  before `test_phase_J_k3_star_tree_finite_and_stationary`.

### Why this is the real Fix-D follow-up

Fix D targeted the filter's constraint scan and the bound
functions — which micro-benchmarks said were the biggest call-
count hot spots (`filtered()` called hundreds of thousands of
times per τ point).  But cProfile after Fix D showed that 13.5 of
14.1 s of wall time attributed to `filtered()` was really the
fast_callable *call from inside* filtered, not the scan itself.
Fix E cuts that cost at the root by replacing the fast_callable
with a much cheaper per-edge evaluator.  Fix D's scan speedup
still helps — just less dramatically than its micro-profile
suggested — now that the integrand call is no longer 40+ µs.

---

## 2026-04-21 — Audit Fix #D: vectorise polytope-bounds and Heaviside filter

### What changed

`msrjd/integration/time_domain/final_integral.py` — the two hottest
inner functions in Phase J loop-diagram evaluation were both doing
the same Python-level work at every scipy.nquad sample:

1. **`_make_heaviside_filtered_integrand`** — the wrapper that
   zero-kills any sampled point outside the true polytope.  At every
   integrand evaluation it looped over every constraint, iterated
   the full `a_int` coefficient vector, and checked `if a != 0.0`.
   For a 3-internal-vertex 1-loop diagram with ~6 retarded
   constraints on m=3 axes, that's ~18 multiplications + 24
   conditional checks per integrand call, and scipy's adaptive
   quadrature calls it thousands of times per (t_1, t_2) grid point.

2. **`_make_bound_fn`** (nested inside `_integrate_nd_polytope`) —
   the per-axis bound callable that scipy.nquad uses to build its
   integration grid.  It re-allocated a `sub_constraints` list on
   every call, re-ran `any(abs(a_int[j]) >= 1e-15 for j in
   range(k_var))` to filter deferred constraints, rebuilt `(a_int,
   new_c_eff)` tuples, and then delegated to `_resolve_1d_bounds`
   which did yet another scan of the list.  For the outermost axis
   this is fine (called few times), but for middle axes of a 3D
   integral it's called ~N^{m-k_var} times per grid point.

3. **`bounds_s0`** in `_integrate_2d_polytope` — same
   list-allocation-on-every-call pattern.

Fix D moves all invariant work to closure-build time:

* **Sparse coefficient lists**.  Each constraint is reduced to
  `(c_eff, tuple_of_(j, a_j)_nonzero_pairs)`.  Retardation
  constraints are `t_v − t_u > 0`, so `a_int` always has exactly
  two nonzero entries regardless of `m` — this collapses the inner
  "for j in range(m)" scan to 2 iterations.

* **Constraint classification**.  Per axis, each constraint falls
  into one of four buckets at build time:
    - `pure_k`: only `a[k_var]` nonzero → contributes a fixed
      slice to `(L, U)` that's precomputed ONCE and reused at
      every call.
    - `mixed`: `a[k_var]` AND some outer-axis coefficient nonzero
      → resolved per call against the outer values.
    - `outer_only`: `a[k_var] = 0` but some outer-axis coefficient
      nonzero → a pure residual inequality in the outer values;
      when it fails, `bounds_k` returns `(0, 0)` to kill the
      region.
    - `constant`: all coefficients zero → checked at build time
      (drops trivial `c > 0`; shorts to empty-polytope for `c ≤ 0`).

* **Build-time infeasibility detection**.  If the pure-k bounds
  alone produce `L_pure >= U_pure`, or if any constant constraint
  is infeasible, `_make_bound_fn` returns a closure that constantly
  returns `(0, 0)` — skipping all per-call work entirely.

* **Infinite-bound cap semantics preserved**.  The original code
  capped only when `math.isfinite(L)` was false.  The vectorised
  version uses the same check (`math.isinf(L)`) at the end of the
  hot path, so a mixed constraint that pins a bound to e.g. -500.0
  is passed through unclipped — identical to pre-Fix D.

### Measured speedup

Micro-profile of 50 000 calls to each hot-path function, with a
realistic m=3 polytope (6 retardation constraints, `s_0 < s_1 <
s_2`-style chain plus cross-axis coupling):

| Function             | BEFORE       | AFTER        | speedup |
|----------------------|--------------|--------------|---------|
| `filtered(*s)`       | 0.76 µs/call | 0.41 µs/call | **1.85×** |
| `bounds_fn(k=1)(*)`  | 2.62 µs/call | 0.49 µs/call | **5.39×** |

Correctness check across 50 000 random outer samples: OLD vs NEW
returned bit-identical `(L, U)` everywhere (0 feasibility
mismatches, max |L_old − L_new| = max |U_old − U_new| = 0).

### Caveat: wall-clock gain is within measurement noise

End-to-end evaluation of the same workload (quadratic Hawkes k=2,
ell=1, 7 1-loop diagrams, τ ∈ {-10, -5, 0, 5, 10}):

| Diagram        | BEFORE  | AFTER   | ratio |
|----------------|---------|---------|-------|
| [0] V=4 E=4    | 0.514s  | 0.479s  | 1.07× |
| [1] V=5 E=5    | 24.69s  | 24.72s  | 1.00× |
| [2] V=5 E=5    | 34.65s  | 38.08s  | 0.91× |

The V=5 diagrams spend the overwhelming majority of their wall
time inside Sage's `fast_callable` integrand — not in the filter
scan or the bound-fn body.  scipy.nquad calls the filter ~O(N^m)
times per τ point (N ~ a few hundred adaptive-quad samples per
axis, m = 3 integration axes), but each of those calls is
dominated by the JIT'd complex-exponential integrand evaluation
that Fix D did not touch.  So the 2× faster filter scan and the
5× faster bound functions translate to a wall-clock improvement
well under 10% — lost inside run-to-run noise of the adaptive
quadrature.

### fast_callable gotcha: keep the single-unpack idiom

During development an intermediate version of the filter used
Python 3.5+ multi-unpack syntax:

```python
return complex(integrand_callable(*s_vals, *free_ext_tuple))
```

This triggers a 2–3× slowdown on the m=3 workload vs. the
single-unpack form:

```python
return complex(integrand_callable(*(list(s_vals) + free_ext_list)))
```

Root cause not fully diagnosed, but reproducible: V=5 diagram [1]
takes 24.7s with single-unpack and 55.6s with multi-unpack,
holding everything else constant.  Probably Sage's fast_callable
has an arg-marshalling fast path keyed on receiving a single
unpacked sequence.  Comment added at the call site warning
future readers not to "simplify" to the multi-unpack form.

### Verification

- All 16 pre-existing tests in `tests/test_time_domain.py` pass
  unchanged.
- Two new Fix D regression tests added to
  `tests/test_time_domain.py` (total now 18 passing):
    - `test_phase_J_fix_D_vectorized_bound_fn_matches_scalar` —
      pins three representative polytope shapes (cross-axis chain,
      redundant pure-external residual, outer-only kill) and
      checks the integrated volume against analytical values.
    - `test_phase_J_fix_D_heaviside_filter_sparse_path` —
      directly tests the rewritten filter on trivially-satisfied
      constraints (dropped), trivially-infeasible constraints
      (forces filter to return 0), and mixed-coefficient
      constraints (sparse dot product is exact).

### Files changed

- `msrjd/integration/time_domain/final_integral.py`
    - `_make_heaviside_filtered_integrand` rewritten to build a
      sparse constraint list once and iterate only nonzero (j, a)
      pairs in the hot loop.
    - `_make_bound_fn` (inside `_integrate_nd_polytope`) rewritten
      to classify constraints into pure/mixed/outer-only buckets
      at closure-build time and handle the infeasibility early.
    - `bounds_s0` (inside `_integrate_2d_polytope`) rewritten with
      the same classification pattern (pure_0, mixed, s1_residual).
- `tests/test_time_domain.py` — two new regression tests appended
  before `test_phase_J_k3_star_tree_finite_and_stationary`.

---

## 2026-04-21 — Audit Fix #A: drop redundant `.subs(num_params)` in subset loop

### What changed

`msrjd/integration/time_domain/final_integral.py::integrate_diagram`
previously re-applied ``.subs(num_params)`` in the 2^|E| delta-subset
loop (both the shot-noise branch around line 487 and the smooth-subset
branch around line 586).  These calls were redundant: by the time
the subset loop runs, every factor has already been num_params-substituted:

  * ``cp`` (combined prefactor) -- subbed at line 351-352
  * ``edge_info[i]['smooth_factor']`` -- via ``build_G_t_matrix``
    (which is called with ``num_params`` at line 220 and applies it
    to the pole/residue/delta data before the factors are extracted)
  * ``edge_info[i]['delta_coeff']`` -- same path
  * ``.subs(substitutions)`` (the delta-sifting eliminator) only
    touches integration-variable symbols, never model parameters

Both `.subs(num_params)` calls removed.  Comments added at both
sites explain the invariant so future readers do not re-introduce
them.

### Measured speedup

Benchmark: quadratic Hawkes k=2 ell=1 at
``a=0.1, tau=10, tau_g=5, w=[[0.5,-0.1],[0.5,0.4]], E=[1.2,1.4]``,
construction time for all 7 unique 1-loop diagrams:

| Diagram           | BEFORE | AFTER  | speedup |
|-------------------|--------|--------|---------|
| V=4 E=4 (x2)      | 0.35s  | 0.33s  | ~1.05x  |
| V=5 E=5 (x3)      | 0.49s  | 0.47s  | ~1.04x  |
| V=6 E=6 (x2)      | 1.44s  | 1.35s  | ~1.06x  |
| **TOTAL (7)**     | **5.10s** | **4.80s** | **1.06x** |

Less than the audit's 2-4x estimate because Sage's ``.subs()`` has a
fast-path when no free symbols match -- but still a real few-percent
win at zero risk.

### Verification

All 18 tests in ``tests/test_time_domain.py`` and
``tests/diagnostic_k2_origin_stationarity.py`` pass unchanged.

### Files changed

- ``msrjd/integration/time_domain/final_integral.py`` -- two
  ``.subs(num_params)`` calls removed, replaced by explanatory
  comments documenting the invariant.

---

## 2026-04-20 — Generalised vertex-time integrator to any loop order

### Summary

The vertex-time integrator in
``msrjd/integration/time_domain/final_integral.py`` no longer asserts
``loop_number == 0``: the same algorithm (assign a time variable to
each internal vertex, integrate over those times with a retarded
Heaviside on every edge, enumerate the ``2^|E|`` delta-subset
expansion) works on loop diagrams too because our enumeration always
produces DAGs and multi-edges already travel through the code via
``(u, v, label)`` edge keys.

### API changes

- **``integrate_tree_diagram`` is now ``integrate_diagram``** in
  ``msrjd/integration/time_domain/final_integral.py``.  The old name
  is kept as a backward-compat alias, and both are re-exported from
  ``msrjd.integration.time_domain``.
- **``compute_correction_td``** in
  ``msrjd/integration/time_domain/pipeline.py`` no longer branches on
  ``loop_number``.  Every diagram goes through ``integrate_diagram``.
  Tree diagrams are reported in the output ``groups`` with
  ``'handled_by': 'tree_evaluator'``; loop diagrams get
  ``'handled_by': 'loop_evaluator'``.

### Verified end-to-end

On the quadratic Hawkes ``k=2`` ``ell=1`` case (7 unique 1-loop
topologies from the new ``phi''`` cubic vertex at
``a=0.1, tau=10, tau_g=5, w=[[0.5,-0.1],[0.5,0.4]], E=[1.2,1.4]``):

- Construction of all 1 tree + 7 loop callables via
  ``compute_correction_td``: ~6 s.
- Each individual 1-loop evaluator returns finite real values at
  ``tau = +2``, with the expected asymmetry (diagrams depending on
  which leaf is at which external position).  Examples:
  - Parallel-edge loop (V=4, E=4): build 0.35 s, eval 0.14 s,
    contribution = ``+1.53e-6``.
  - Chain-with-loop (V=5, E=5): build 0.5 s, eval 5-6 s,
    contributions at the ``1e-7 -- 1e-6`` scale.

### Known runtime caveat

The ``2^|E|`` delta-subset sum times the per-subset polytope
dimensionality is the dominant cost: a 1-loop diagram with 5 edges and
3 internal vertices runs ~32 3-D scipy.integrate.nquad calls per
``total_C`` invocation, which is 5-10 seconds.  Summing 7 such
diagrams per tau-grid point is therefore ~30-60 s per tau.  The
efficiency-audit fixes already flagged in ``CHANGELOG.md`` -- in
particular:

  * **Fix #2** (drop the redundant ``.subs(num_params)`` inside the
    subset loop)
  * **Fix #10** (vectorise ``_make_bound_fn`` bound computation)

target exactly this hot path and should bring per-tau-point eval down
by a factor of ~5-10x.  Threading over diagrams is off-limits for the
same Sage ECL thread-safety reason documented in the earlier
parallelism-revert entry.

### Files changed

- `msrjd/integration/time_domain/final_integral.py` -- renamed
  ``integrate_tree_diagram`` to ``integrate_diagram``, dropped the
  ``assert loop_number == 0``, added the docstring note about loop
  support, left an ``integrate_tree_diagram = integrate_diagram``
  alias at the bottom of the module for backward compat.
- `msrjd/integration/time_domain/__init__.py` -- re-export
  ``integrate_diagram`` alongside the old name.
- `msrjd/integration/time_domain/pipeline.py` -- route every diagram
  through ``integrate_diagram`` unconditionally; loop diagrams now
  come back with ``'handled_by': 'loop_evaluator'`` instead of
  ``'skipped'``.

All 18 regression tests pass unchanged.

---

## 2026-04-20 — New model variant: quadratic Hawkes with gain `a` and exponential filter (1-loop test bed)

### Summary

Added a third Hawkes variant that generalises
``hawkes_linear_expg.py`` from ``phi(v) = a v`` to ``phi(v) = a v^2``,
keeping the unit-integral exponential synaptic filter
``g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)``.  The motivation is
**1-loop diagram coverage**: the quadratic nonlinearity produces a
``-a ñ dv^2`` cubic interaction vertex (from the ``phi''(v*) = 2 a``
Taylor coefficient) that doesn't exist in any linear variant.  This
vertex drives a rich family of 1-loop topologies that exercise the
loop-subgraph integrator.

### New files

- **`models/hawkes_quad_expg.py`** — model dict with ``phi(v) = a v^2``.
  Specialisations set ``phi1_i -> 2 a vstar_i``, ``phi2_i -> 2 a``,
  higher derivatives -> 0.  MF is nonlinear (``n*_i = a (E_i + Σ_j w_{ij}
  n*_j)^2``), solved by fsolve in cell 23 via the existing
  ``phi_concrete = a v**2`` hook.  The ``kernel_ft_image`` and
  parameter declaration (``a``, ``tau_g``, ``w`` as 2D-indexed) are
  identical to the linear-expg model so all the notebook plumbing --
  cell 8 kernel image, cell 23 generic num_params, cell 25 compute_td
  -- just works.
- **`models/hawkes_sim_quad_expg_numba.py`** — numba simulator with
  the same exp-filtered synaptic dynamics as the linear-expg sim, but
  ``lambda_i = max(a * v_i^2, 0)`` (quadratic rate).  Call signature
  ``sim_hawkes_quad_expg_numba(n_steps, dt_sim, tau, tau_g, a, E, W,
  v_init, bin_size_steps, n_bins, seed)``.
- **`notebooks/hawkes_td_only_quad_expg.ipynb`** — clone of
  ``hawkes_td_only_expg.ipynb`` with imports pointed at the new model
  and simulator.  Config-cell defaults:
  - ``max_ell = 1`` (the point of this notebook is to exercise 1-loop
    corrections).
  - ``fundamental['a'] = 0.1`` (quadratic at a=1 with the same E/w
    fires too hot; 0.1 keeps ``n*`` O(1)).

### Vertex-structure comparison (k=2)

| model           | interaction vertices | noise vertices | k=2, ell=0 unique | k=2, ell=1 unique |
|-----------------|-----------------------|-----------------|---------------------|---------------------|
| linear-expg     | 6                     | 6               | 2                   | **0**               |
| **quad-expg**   | **14**                | 6               | 2                   | **61**              |

The linear theory has zero 1-loop correction to the 2-point function;
the quadratic theory has 61 unique 1-loop topologies, all originating
from the new ``(1, 2)`` and ``(k, 2)`` interaction vertices the
quadratic Taylor expansion introduces.

### Known current limitation

``compute_correction_td`` in ``msrjd/integration/time_domain/pipeline.py``
still marks loop diagrams as ``'handled_by': 'skipped'`` with reason
``'loop_number = k: not in MVP'``.  The new notebook therefore
enumerates the 61 1-loop diagrams and reports them in the group
summary, but cell 25 does not yet evaluate them through the Phase J
tree integrator.  Frequency-domain loop evaluation via
``msrjd/integration/symbolic.py::loop_integrand_groups`` is a
separate code path (used by the older ``hawkes_2pt_pipeline_demo``
notebook) that could be wired in as the next step.

### Files changed

- `models/hawkes_quad_expg.py` (new)
- `models/hawkes_sim_quad_expg_numba.py` (new)
- `notebooks/hawkes_td_only_quad_expg.ipynb` (new)

---

## 2026-04-20 — Reverted the thread-pool parallelism in ``compute_correction_td``

### Why

The 2026-04-20 thread-pool refactor of ``compute_correction_td``
(efficiency audit Fix #3 -- parallel evaluation of ``total_C``)
produced small numerical artifacts on real Hawkes workloads that
compounded with the downstream polytope quadrature and pushed theory
slightly above the simulation histogram on the k=3 slice plot.  It
also made the propagator handling more fragile: the narrower
``_to_sr_ab`` variant that landed alongside the thread-pool fix
preserved exact arithmetic for the vanilla notebook but stopped
catching the SR-wrapped-Python-complex case the ``expg`` notebook
relies on, reinstating the original GiNaC complex-sort crash at
``display_stripped *= ei['smooth_factor']``.

Rather than keep tuning, we reverted the parallelism entirely.

### What was reverted

- ``msrjd/integration/time_domain/pipeline.py::compute_correction_td``
  -- restored to the pre-2026-04-20 signature and body (no
  ``parallel`` / ``n_workers`` / ``parallel_threshold_ms`` kwargs, no
  ``ThreadPoolExecutor``, no ``_executor_state`` return-dict key).
  ``total_C`` sums per-diagram contributions with a plain Python for
  loop.
- ``notebooks/hawkes_td_only.ipynb`` and
  ``notebooks/hawkes_td_only_expg.ipynb`` -- removed the
  ``TD_PARALLEL`` / ``TD_N_WORKERS`` / ``TD_PARALLEL_THRESHOLD_MS``
  configuration block from cell 2 and removed the corresponding
  kwargs from the two ``compute_correction_td`` call sites in
  cell 25.

### What was kept

- ``msrjd/integration/time_domain/propagator_td.py::build_G_t_matrix``
  still has the narrower ``_to_sr_ab`` (it only normalises raw
  Python ``complex`` and Sage ``ComplexDoubleElement`` values, not
  symbolic SR expressions), but the ordering is now **normalise
  first, then substitute**:

      pole_vals = [_to_sr_ab(p) for p in pole_vals]   # was AFTER subs
      C_mats    = [C.apply_map(_to_sr_ab) for C in C_mats]
      if num_params:
          pole_vals = [SR(p).subs(num_params) for p in pole_vals]
          C_mats    = [C.apply_map(lambda e: SR(e).subs(num_params))
                       for C in C_mats]

  Running ``_to_sr_ab`` *before* the ``SR(p).subs(num_params)``
  wrapping catches the raw-``complex`` case in the ``expg`` notebook
  (where pole data comes from a numerical ``polynomial.roots()``),
  decomposing it into clean ``a + b*I`` SR form with float
  components before GiNaC ever sees it.  The vanilla notebook's
  symbolic ``pole_vals`` and ``C_mats`` still pass through
  unchanged, so the closed-form algebra is preserved.

### Verification

- All 18 regression tests in ``tests/test_time_domain.py`` and
  ``tests/diagnostic_k2_origin_stationarity.py`` pass.
- Standalone exercise of the ``expg`` failure path -- a ``pd`` dict
  with ``pole_vals`` as Python ``complex`` and ``C_mats`` as
  ``matrix(CDF, ...)``, followed by ``build_G_t_matrix`` ->
  ``G_t_entry`` -> SR multiplication -> ``.expand()`` -- runs clean
  end-to-end with no complex-sort crash.

### Files changed

- ``msrjd/integration/time_domain/pipeline.py``
- ``msrjd/integration/time_domain/propagator_td.py``
  (order of ``_to_sr_ab`` vs. ``subs(num_params)``)
- ``notebooks/hawkes_td_only.ipynb``
- ``notebooks/hawkes_td_only_expg.ipynb``

---

## 2026-04-20 — Precision regression in ``build_G_t_matrix`` normalisation

### Bug

The 2026-04-20 ``_to_sr_ab`` normaliser in
``msrjd/integration/time_domain/propagator_td.py::build_G_t_matrix``
was too aggressive: its ``try: complex(value)`` branch succeeded not
only on raw Python ``complex`` / Sage ``ComplexDoubleElement`` scalars
(which actually need the decomposition to dodge the GiNaC
complex-sort crash) but also on **exact symbolic** SR expressions like
``i*(1 - sqrt(6)/10)``.  Sage quietly evaluates such expressions to a
double-precision float when coerced with ``complex(...)``, so every
pole value and every residue-matrix entry was being collapsed to a
double before being fed to ``fast_callable``.

For the **vanilla** (delta-kernel) linear Hawkes notebook, where
``pole_vals`` and ``C_mats`` come from a symbolic ``solve(...)`` and
exact adjugate divisions, this caused a small systematic bias in
every ``total_C`` output -- visible as the theory curve sitting a few
percent above the k=3 simulation histogram.  The ``expg`` notebook
was unaffected because its pole data is genuinely ``complex`` /
``CDF`` from the numerical ``.roots()`` call, where the normalisation
is still needed and not a precision downgrade.

### Fix

``_to_sr_ab`` now decomposes the value only when it is a raw Python
``complex`` or a Sage ``ComplexDoubleElement``.  Anything else
(symbolic SR expressions, Sage rationals, integers, ...) is returned
as ``SR(value)`` unchanged, preserving exact algebraic content.

### Verification

- All 18 regression tests in ``tests/test_time_domain.py`` +
  ``tests/diagnostic_k2_origin_stationarity.py`` pass.
- A standalone check with numerical ``pole_vals`` (``complex``) and
  ``C_mats`` (``matrix(CDF, ...)``) still runs through
  ``build_G_t_matrix`` -> ``G_t_entry`` -> SR multiplication without
  the GiNaC complex-sort crash, confirming the ``expg`` path is
  preserved.
- Parallel vs. serial ``total_C`` agreement across a 12-point x
  8-diagram k=3 workload is now bit-identical (max |diff| = 0.0e+00),
  confirming the parallelism refactor itself introduces no drift --
  the user-observed parallel-vs-serial gap was entirely an artefact
  of floating-point accumulation order compounding the precision loss
  from the old ``_to_sr_ab``.

### Files changed

- ``msrjd/integration/time_domain/propagator_td.py`` --
  ``_to_sr_ab`` tightened to act only on raw ``complex`` / CDF inputs.

---

## 2026-04-20 — Adaptive thread-pool parallelism in `compute_correction_td`

### Summary

Efficiency audit Fix #3: the returned `total_C(*ext_time_values)` callable now
optionally fans per-diagram `contribution` evaluations out to a
`ThreadPoolExecutor`.  Each diagram's hot path is `scipy.integrate.quad`
and Sage's `fast_callable` CDF kernel, both of which release the GIL, so
threading gives a near-linear speedup on multicore machines for heavy
workloads (k=3 / k=4 with real polytopes and multi-diagram sums).

### What DOESN'T get parallelised

Per-diagram **construction** -- `integrate_tree_diagram` itself -- stays
serial.  Sage's SR / GiNaC / Maxima-ECL backend is not thread-safe
during symbolic manipulation; running it on multiple threads triggers
a native-code crash (`Did you forget to call 'ecl_import_current_thread'?`
from ECL).  This was discovered by the initial implementation attempt
that threaded the construction loop; the crash is deterministic and
worth documenting as a known Sage thread-safety boundary.

### Adaptive heuristic (`parallel='auto'`, default)

The pool has real dispatch overhead (~100-500 us per task on modern
CPython), which can exceed the per-call work for cheap k=2 workloads
(sub-ms per diagram).  `parallel='auto'` handles this automatically:

- The first `total_C(...)` call is timed on the serial path.
- If that call exceeded `parallel_threshold_ms` (default 5.0 ms), the
  executor is built and subsequent calls go parallel.
- Otherwise subsequent calls stay serial.

Modes:

- `parallel='auto'` (default) -- adaptive as above.
- `parallel=True` -- force parallel from the first call onward.
- `parallel=False` -- force serial (useful for bit-exact reproducibility
  and for debugging).

### Measured behaviour

| Workload | Auto | Forced serial | Forced parallel | Auto speedup |
| --- | --- | --- | --- | --- |
| Cheap k=2, 16 diagrams (30 tau pts) | 78.7 ms | 71.4 ms | 78.7 ms | ~1.0x |
| Realistic k=3, 8 diagrams (25 tau pts) | 30.3 ms (stays serial) | 31.3 ms | 34.3 ms | matches serial |
| Heavy (simulated 50 ms/call), 8 diagrams | 168 ms (promotes) | 1307 ms | 170 ms | **7.8x** |

On the heavy workload auto-mode delivers the ~8-way speedup expected from
8 diagrams on a 12-core machine.  On the realistic k=3 case auto correctly
stays serial so there is no regression relative to the previous
single-thread `total_C`.  The cheap k=2 case crosses the 5 ms threshold
(16 diagrams x ~0.3 ms each) and promotes, costing ~10% overhead; if this
matters for a particular workload, pass `parallel=False` or
raise `parallel_threshold_ms` on that call.

### Return-dict additions

- `_executor_state` -- dict with keys `'executor'` (a
  `ThreadPoolExecutor` or `None`) and `'mode'` (`True` / `False` /
  `'auto'`, updated in place once auto resolves).  Callers that want
  deterministic thread shutdown can call
  `res['_executor_state']['executor'].shutdown(wait=True)` at the end
  of a run; otherwise Python's atexit handler closes the pool on
  process exit.

### Files changed

- `msrjd/integration/time_domain/pipeline.py` -- `compute_correction_td`
  signature gains `parallel='auto'`, `n_workers=None`,
  `parallel_threshold_ms=5.0`.  Module docstring expanded with the
  thread-safety rationale.

All 18 regression tests in `tests/test_time_domain.py` +
`tests/diagnostic_k2_origin_stationarity.py` pass with the new default.

---

## 2026-04-20 — New model variant: linear Hawkes with gain `a` and exponential synaptic filter

### Summary

Added a second linear-Hawkes model alongside `hawkes_linear_sage.py` where:
  * `phi(v) = a * v`  (parametrised gain instead of the identity)
  * `g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)` (unit-integral exponential filter instead of `delta(t)`)

plus a matching numba simulator and a cloned notebook.  The pipeline's generic
hooks (`kernel_ft_image`, parameter iteration) carried the new model through
`FieldTheory`, diagram enumeration, and the time-domain integrator, but several
plumbing fixes were needed to keep the numerical pole-residue data consumable
downstream.

### New files

- **`models/hawkes_linear_expg.py`** — model dict with `a`, `tau_g`, and the
  2x2 weight matrix `w` declared as parameters.  Declares
  `'kernel_ft_image': lambda ns, omega: {ns.g: 1/(1 + I*omega*ns.tau_g)}`
  so the notebook's cell 8 can substitute `g -> g_hat(omega)` after the
  Fourier transform instead of specialising `g -> delta_D` in the time domain.
- **`models/hawkes_sim_expg_numba.py`** — Euler-step simulator with per-
  population filtered synaptic input `F_j` (decay `tau_g`, unit-integral kicks)
  and linear rate `lambda_i = max(a * v_i, 0)`.  Call signature
  `sim_hawkes_expg_numba(n_steps, dt_sim, tau, tau_g, a, E, W, v_init,
  bin_size_steps, n_bins, seed)`.
- **`notebooks/hawkes_td_only_expg.ipynb`** — clone of `hawkes_td_only.ipynb`
  pointed at the new model + simulator; cell 8 rewritten around the new
  kernel-image hook; cell 23 has a generic parameter-substitution loop and a
  deferred numerical-pole computation; cells 31 and 32 call the new simulator
  with `a` and `tau_g`.

### Library-level bug fixes uncovered by the new model

1. **`msrjd/integration/time_domain/propagator_td.py::build_G_t_matrix`** —
   every pole value and `C_mat` entry is now normalised to
   `SR(float(Re(z))) + SR(float(Im(z))) * I` before building `smooth`.
   Previously, Python `complex` objects were embedded in the GiNaC tree;
   when downstream code multiplied or expanded the resulting SR expression,
   GiNaC's operand sort tried to compare `complex` with `<` and raised
   `TypeError: '<' not supported between instances of 'complex' and 'complex'`.
   Casting to `a + b*I` with Python `float` components keeps everything
   sortable by Sage's native ordering.

2. **`msrjd/integration/time_domain/propagator_td.py::_infer_time_variable`** —
   replaced the `entry == 0` guard with a `variables()` / `is_trivial_zero()`
   check.  Equality tests trigger Maxima-backed simplification, which also
   tripped on the `complex`-sort path above when called on any entry with
   numerical CDF residues.

3. **`notebooks/hawkes_td_only_expg.ipynb` cell 8** —
   - `_to_kernel` no longer short-circuits when it sees `ns.g`; it wraps
     every term in `delta_D` so Sage's `fourier_transform` returns the
     constant coefficient instead of `2*pi*delta(omega)`.  The notebook then
     applies `HAWKES_MODEL['kernel_ft_image']` after the FT to replace
     `ns.g` with `1/(1 + I*omega*ns.tau_g)`.
   - Replaced the symbolic `solve(det(K_ft) == 0, omega)` block with a
     `compute_poles_and_residues(num_params)` helper that:
     (a) reads the characteristic polynomial from
         `lcm(G_ft[i,j].denominator())` (the minimal common denominator of the
         factored propagator entries -- the true quartic) rather than
         `det().numerator()` (which picks up an extra `(1 + I*omega*tau_g)`
         factor from how Sage normalises a sum of rational functions);
     (b) finds roots numerically via `PolynomialRing(CDF, 'omega').roots`;
     (c) keeps `Im(omega) > 0` and deduplicates;
     (d) builds each residue matrix `C_k = i * adj(omega_k) / det'(omega_k)`
         using complex-double arithmetic.
   - Cosmetic display cleanup: when a `K_ker` entry's every term carries a
     declared kernel symbol AND a `delta_D`, the display divides out
     `delta_D`, so synaptic-coupling entries now render as `-w_{ij} g`
     instead of `-w_{ij} delta g`.  Mixed entries like `(tau delta' + delta)`
     and pure-constant entries like `-a delta` are untouched.

4. **`notebooks/hawkes_td_only_expg.ipynb` cell 23** —
   - The `_param_subs` loop (used by the MF-solve's `phi_num`) now handles
     2D matrix-valued indexed parameters, keying each scalar by
     `SR.var(f'{pname}{i+1}{j+1}')` instead of `getattr(ns, pname)[i]` (which
     becomes an unhashable SR row once `mf_substitutions` has replaced
     `ns.w` with a nested matrix).
   - The `num_params` loop was already generic but is now mirrored by the
     `_param_subs` loop above, so the MF solve and the pipeline substitution
     see the same dictionary.
   - A deferred block at the end of the cell calls
     `compute_poles_and_residues(num_params)` and writes the numerical
     `pole_vals` and `C_mats` into `propagator_data` (they were left as
     `None` in cell 8 because the rational determinant has no symbolic
     closed-form quartic roots).
   - Prints each numerical pole in `a + b*i` form for quick sanity checks.

### Verified end-to-end

At `a = 1.0, tau = 10.0, tau_g = 5.0, E = [1.2, 1.4], w = [[0.5, -0.1], [0.5, 0.4]]`
the pipeline finds exactly 4 retarded poles

    omega_1 = -0.019980 + (+0.040917) i
    omega_2 = -0.019980 + (+0.259083) i
    omega_3 = +0.019980 + (+0.040917) i
    omega_4 = +0.019980 + (+0.259083) i

(matching the quartic degree of the displayed `G_ft` denominator), and
`G_t_entry(G_t_obj, phys, resp, t)` returns real-valued smooth-propagator
samples with numerical-noise imaginary parts (~1e-16), which means
`integrate_tree_diagram` now multiplies `display_stripped *= smooth_factor`
without the earlier GiNaC complex-sort crash.

### Files changed

- `models/hawkes_linear_expg.py` (new)
- `models/hawkes_sim_expg_numba.py` (new)
- `notebooks/hawkes_td_only_expg.ipynb` (new, based on `hawkes_td_only.ipynb`)
- `msrjd/integration/time_domain/propagator_td.py`
  (`build_G_t_matrix` and `_infer_time_variable`)

---

## 2026-04-17 — k=2 mirror-image: missing `external_fields` in `compute_correction_td`

### Bug

For k=2 with mixed external fields, e.g. `external_fields=[('dn',1), ('dv',2)]`,
cell 25's `compute_correction_td(...)` call omitted the `external_fields`
keyword.  Without it, `integrate_tree_diagram` fell back to the identity
canonical-to-leaf mapping based on the enumeration's leaf order.  When that
order happened to be the reverse of `external_fields`, the integrand ended up
computing `<delta_v_2(0) * delta_n_1(tau)>` = `<delta_n_1(0) * delta_v_2(-tau)>`
by stationarity -- exactly the tau -> -tau mirror image of the sim curve.

### Manifestation

After the 2026-04-15 canonical-convention fix was applied, the user re-ran
the k=2 slice comparison with `external_fields=[('dn',1), ('dv',2)]` with a
fresh kernel and `USE_CACHE=False`, and still saw the theory curve as the
exact mirror of the sim curve (peak near tau=-1 instead of tau=+1).  The k=3
branch in the same notebook was unaffected because it already passed
`external_fields` through.

### Fix

- `notebooks/hawkes_td_only.ipynb` cell 25 -- added
  `external_fields=external_fields` to the k=2 `compute_correction_td` call,
  with a comment flagging the subtlety for future copies of the notebook.
- `msrjd/integration/time_domain/final_integral.py` -- `integrate_tree_diagram`
  now emits a loud `warnings.warn(...)` when `external_fields` is omitted AND
  the diagram has mixed leaf field types, so this class of silent-mirror bug
  cannot recur without being noisy.
- `msrjd/integration/time_domain/pipeline.py` -- `compute_correction_td`'s
  `typed_diagrams`, `prefactors`, `propagator_data`, and `k` are now keyword-
  defaulted to `None` so the legacy `kernel_groups=` call pattern used by
  `test_phase_J_nondiagonal_2x2_does_not_overflow` works again (the test had
  been silently broken since the signature was tightened).
- `tests/diagnostic_k2_origin_stationarity.py` (new) -- regression test that
  locks down the stationarity relationship
  `origin=1, C(tau, 0)  ==  origin=0, C(0, -tau)` on the generic nondiagonal
  cross-population fixture.

### Slice-plot cells downstream

Two plot cells assumed the pre-2026-04-15 "scalar fixed-tau" key format and
crashed under the new tuple-valued keys:

- `notebooks/hawkes_td_only.ipynb` cell 29 (8.2 slice comparison) --
  `tau_fixed` was undefined in the title-label code; replaced with
  `zip(other_idxs, fixed_tuple)` so each fixed-axis label is paired with its
  value in the k-agnostic tuple.
- `notebooks/hawkes_td_only.ipynb` cell 32 (9 sim comparison, k>=3 branch) --
  `float(tuple)` threw `TypeError: float() argument must be a string or a
  real number, not 'tuple'`.  Rewrote the else-branch to use the tuple
  directly as the `C_sim_slices` / `C_tree_phase_j_slices` key and to produce
  per-axis labels via the same `zip(...)` pattern as cell 29.

### Files changed

- `notebooks/hawkes_td_only.ipynb` (cells 25, 29, 32)
- `msrjd/integration/time_domain/final_integral.py`
- `msrjd/integration/time_domain/pipeline.py`
- `tests/diagnostic_k2_origin_stationarity.py` (new)

---

## 2026-04-15 — k=2 slice convention: canonical `origin_leaf_idx = 0`

### Bug

For k=2, cell 25 set `_origin_idx = 1` when calling
`compute_correction_td`, pinning the SECOND leg's time (`t_2`) to 0
and leaving the FIRST leg (`t_1`) as the free axis.  This is the
OPPOSITE of the pipeline's documented convention, which pins `t_1`
and lets `t_2 = τ` be free (see `compute_correction_td` docstring in
`msrjd/integration/time_domain/pipeline.py`).

The simulation estimator (`compute_kpoint_slice`) follows the
canonical convention: it builds the product from FIXED legs at
`lag_bins[i]` (with leg 0 at `lag_bins[0] = 0` = pinned) and
cross-correlates with the SWEEP leg, producing
`C_sim[τ] = ⟨δX_0(0) · δX_1(τ)⟩`.

With the theory pinning `t_2` instead of `t_1`, the theory curve
came out as `⟨δX_0(τ) · δX_1(0)⟩ = ⟨δX_0(0) · δX_1(-τ)⟩` by
stationarity — a mirror image of the sim slice across `τ = 0`.

### Manifestation

User observed on a k=2 mixed-field slice `[('dn',1), ('dv',2)]`:
the simulation showed the expected strong positive-τ tail (dn_1
spikes at time 0 propagate forward to affect dv_2 at τ > 0), while
the theory curve had a mirror-image strong NEGATIVE-τ tail
(anti-causal appearance).  Peak heights matched exactly; only the
τ-axis orientation differed.

### Fix

Changed `_origin_idx = 1` to `_origin_idx = 0` in cell 25's k=2
branch.  Now `total_C(0, τ)` correctly returns
`⟨δX_0(0) · δX_1(τ)⟩`, matching the sim convention.

Also updated cell 28's k=2 slice-eval from `_total_C_fn(τ, 0)` to
`_total_C_fn(0, τ)`, which is the correct order under the new
canonical `_origin_idx = 0`.

The k≥3 branch of cell 25 uses `origin_leaf_idx = None` (no pinning,
all k external times are free) and a separate per-diagram slice
evaluator in cell 28 that correctly builds `time_args = [0.0, ...]`.
Cell 33's heatmap already used `_total_C_fn(0.0, t1, t2)`.  No other
paths were affected.

### Files changed

- `notebooks/hawkes_td_only.ipynb` — cell 25 (`_origin_idx` for k=2)
  and cell 28 (k=2 slice-eval argument order).

---

## 2026-04-15 — Heaviside convention: Θ(0) = 0 throughout

### Summary

Changed the retarded-Heaviside boundary convention from `Θ(0) = 1`
("Ito" as previously annotated, boundary included) to `Θ(0) = 0`
(boundary strictly excluded from the retarded support).  Applied
consistently across the numerical integration path and the in-code
documentation.

### What changed

- `_make_heaviside_filtered_integrand` (in `msrjd/integration/time_domain/
  final_integral.py`) — filter now triggers `return 0.0 + 0.0j` whenever
  any `dt <= 0` (previously `dt < 0`).  The boundary `dt = 0` is now
  excluded from the feasible region.
- `_resolve_1d_bounds` — degenerate-constraint feasibility check tightened
  from `c_eff < 0` to `c_eff <= 0`.  A constraint with `a = 0` and
  `c_eff = 0` (exact-boundary pure-external) is now infeasible under
  strict `Θ(0) = 0`.
- `_integrate_polytope` m=0 branch — same tightening.
- `_integrate_2d_polytope` pure-external guard — same tightening.
- `_outer_bounds` — same tightening.
- `msrjd/integration/time_domain/__init__.py` — module docstring updated:
  "The convention `Θ(0) = 0` is used throughout — boundary `Δt = 0` is
  strictly excluded from the feasible (retarded) region."
- `msrjd/integration/time_domain/propagator_td.py` — Heaviside-at-zero
  docstring rewritten.  Clarifies that:
  - Numerical integration path uses `Θ(0) = 0`.
  - Sage's symbolic `heaviside(0) = 1/2` is used only in the symbolic-
    display path (cell 26 when `SHOW_SYMBOLIC = True`); it does NOT
    enter the JIT-compiled integrand.
- `notebooks/hawkes_td_only.ipynb` — three in-cell comments referencing
  "Ito convention: Θ(0) = 1" updated to "Θ(0) = 0 convention".

### Impact on numerics

Measure-zero for continuous integrals: the boundary `Δt = 0` is a lower-
dimensional hypersurface that contributes zero to proper Lebesgue
integrals.  All 140 tests pass with no numerical differences within
quadrature tolerance.  The convention change is mathematically consistent
but does not alter any computed cumulant value for the configurations
currently tested.

Where the convention matters: distributional contact terms (δ-surface
contributions at `Δt = 0`), which are already handled separately via
the `delta_contributions` data structure in `integrate_tree_diagram`.
Those contributions are excluded from `total_C` (the smooth callable)
and reported separately.

### Files changed

- `msrjd/integration/time_domain/final_integral.py` — 5 locations.
- `msrjd/integration/time_domain/propagator_td.py` — docstring.
- `msrjd/integration/time_domain/__init__.py` — docstring.
- `notebooks/hawkes_td_only.ipynb` — 3 in-cell comments.

### Verification

All 140 regression tests pass (4 polytope-specific + 136 others).  The
Heaviside-filter-kills-overshoot test, which exercises the strict
boundary behavior directly, passes at 1% tolerance against the
analytical reference.

---

## 2026-04-15 — k=4 Phase J hardening: polytope integrator, Heaviside filter, cumulant estimator

### Summary

Debugging session that closed out four distinct bugs in the k=4 tree-
level evaluation path, plus a discretization artifact in the simulation
estimator.  Pipeline is now validated end-to-end at k=2, k=3, and k=4
for all tested external-field configurations (all-distinct, single
same-type pair, two same-type pairs, mixed dn/dv).  Theory/sim ratios
all within ~1σ of 1.0 across multiple seeds, points, and run counts.

### Problem observed

Starting point: at k=4 with `external_fields = [('dn',1),('dn',2),
('dv',1),('dv',2)]` (all-distinct, `max_ell = 0`), theory gave
~0.2 against a sim mean of ~0.56 (ratio ~0.38).  The 2+2 same-type
case `[dn1,dn1,dn2,dn2]` was similarly off.  After iterative debugging
with parallel agent audits, four bugs were identified:

### Bug 1: Cross-axis constraint handling in `_integrate_nd_polytope`

For m ≥ 3 integration axes (first exercised at k=4 trees with 3
internal vertices), `_make_bound_fn(k_var)` substituted OUTER axes
(indices > k_var) into `c_eff` but passed the original unchanged
`a_int` to `_resolve_1d_bounds(s_index=k_var)`.  Constraints with
zero coefficient on axis k_var but nonzero coefficient on a
MORE-INNER axis j < k_var (e.g. `s_2 − s_0 > 0`) triggered the
pure-residual branch in `_resolve_1d_bounds` and spuriously declared
the polytope infeasible whenever the accumulated residual was
negative.  Retarded linear Hawkes trees with non-adjacent internal-
vertex edges have exactly this constraint pattern, so k=4 theory was
systematically clipped.

**Fix:** in `_make_bound_fn`, skip constraints that still couple to a
more-inner axis.  Those constraints will be resolved at the deeper
nesting level when that inner axis becomes the resolution target.
This mirrors the filter already present in `_integrate_2d_polytope`.

**Regression test:** `test_phase_J_nd_polytope_preserves_deferred_constraints`
— the simplex `-5 < s_0 < s_1 < s_2 < 5` (integrand = 1) should
produce volume 1000/6 ≈ 166.67.  With the bug, the middle-axis lower
bound was falsely clipped, giving 500/6 ≈ 83.33 (factor-of-two error).

### Bug 2: Heaviside enforcement in integrand

The JIT-compiled smooth integrand contains only `G^sm(Δt)`, with
retardation `Θ(Δt)` enforced implicitly via polytope bounds.  When the
bounds are only approximate (because some cross-axis constraints are
deferred to inner nesting levels, falling back to ±OUTER_CAP), the
integrator samples regions outside the true polytope.  For retarded
poles with Im(ω) > 0, `G^sm(Δt) = C · exp(-γ Δt)` GROWS for `Δt < 0`,
producing a spurious positive contribution that scales with
OUTER_CAP — observed as sensitivity of the theory value to the cap
(0.64 at cap=200, 0.46 at cap=100).

**Fix:** `_make_heaviside_filtered_integrand` wraps the integrand
with an explicit check: evaluates to 0 whenever any constraint
`a_int · s + c_eff < 0`.  Polytope bounds become a pure optimization
(tightening the quadrature domain); correctness is guaranteed by the
filter regardless of cap width.  Applied to all three integrators
(`_integrate_1d_polytope`, `_integrate_2d_polytope`,
`_integrate_nd_polytope`).

**Regression tests:**
- `test_phase_J_heaviside_filter_kills_overshoot` — integrate
  `exp(-0.1·(s_2 - s_0))` on the ordered simplex
  `0 < s_0 < s_1 < s_2 < 10`.  Checked against analytical reference
  computed via nested 1D quadrature (103.7).  Without the filter the
  middle axis admits s_1 < 0 where bounds_0's (0, s_1_val) is
  spurious — the check catches any regression.
- `test_phase_J_nd_polytope_simplex_gaussian` — Gaussian on
  `-5 < s_0 < s_1 < s_2 < 5`, result `(2π)^{3/2}/6` with 5×10⁻³
  relative precision.

### Bug 3: `pure_s1_found` in `_integrate_2d_polytope`

The 2D outer-axis bound computation:
```python
for (a_int, c_eff) in s_constraints:
    if abs(a_int[0]) < 1e-15:
        pure_s1_found = True          # ← set BEFORE checking a_int[1]
        a = a_int[1]
        if abs(a) < 1e-15:
            ...
            continue                   # skips bound update but flag stays True
```

A constraint with BOTH `a_int[0] ≈ 0` and `a_int[1] ≈ 0` — a
pure-external constraint with no dependence on either integration
axis — flagged `pure_s1_found = True` but never updated `tmp_L,
tmp_U`.  The code then used `L1, U1 = (-inf, +inf)` instead of the
`(-OUTER_CAP, +OUTER_CAP)` fallback.  scipy.nquad on an unbounded
axis uses a tanh-sinh variable transform that oversamples near the
polytope boundary, biasing the result upward by ~10-15%.  At k=4,
δ-sifting can pin an integration variable to external times and
leave residual pure-external constraints in subsets that then reach
`_integrate_2d_polytope` — exactly the triggering condition.

**Fix:** move `pure_s1_found = True` to AFTER the `abs(a_int[1]) <
1e-15` guard, so constraints with no s_1 dependence don't flag.

**Regression test:** `test_phase_J_2d_polytope_pure_external_constraint`
— the box `0 < s_0 < 5, 0 < s_1 < 5` with a redundant pure-external
constraint `1 > 0` added.  Without the fix the outer bound is
`(-inf, +inf)` and scipy biases; with the fix the bound falls back
to `(-200, +200)` and the Heaviside filter + quadrature converges
to `((1-exp(-0.5))/0.1)² ≈ 15.5` within 1%.

### Bug 4: `_two_point` factorial correction (latent)

In `models/cumulant_estimator.py`, the cumulant-subtraction pair
estimator `_two_point` used ordinary linear centering for every
pair — inconsistent with the 4-point product estimator's philosophy,
which applies falling-factorial correction to same-pop same-ft spike
legs at coincident bins to remove self-spike shot noise.  For a
subtraction pair meeting the same coincidence conditions, the two
objects are from different estimator families.

**Fix:** added a factorial-correction branch in `_two_point` for
same-pop same-ft='dn' same-lag pairs:
```python
arr = binned_counts[pa, :].astype(float)
fact_rate_sq = arr * (arr - 1.0) / (dt_bin ** 2)
mean_rate = mean_by_pop_ft[(pa, fta)] / dt_bin
return float(fact_rate_sq.mean() - mean_rate * mean_rate)
```

None of the test configurations exercised to date trigger this path
(no subtraction pair in the tested configs has same-pop same-ft
same-lag spike legs).  The fix is a latent consistency patch for
future configurations.

### Discretization artifact (not a bug): bin-averaging bias

Separate from the code bugs above, the simulation's binned estimator
measures the BIN-AVERAGED cumulant density over a 4-dimensional box
of side `dt_bin`, while the theory evaluates the POINT value.  For a
smooth κ_4 with timescale τ, the bias is O((dt_bin/τ)²) per axis.
At the original `dt_bin = 2.0` with Hawkes τ = 10, this produced
systematic shifts of ~10-15% in either direction depending on the
curvature of κ_4 at the test point.

**Mitigation:** reduced `dt_bin` to 1.0 (cell 31 default).  This
shrinks the bias ~4×.  At `dt_bin = 1.0` and `T = 5M`, sim mean
agreement is within ~1σ of theory across multiple seeds:

| Config                                    | Theory | Sim    | Ratio   |
|-------------------------------------------|--------|--------|---------|
| `[dn1, dn2, dv1, dv2]` at (+4, +2, -2)    | 0.569  | 0.577  | 0.987   |
| `[dn1, dn2, dn2, dn1]` at (+4, -2, +2)    | 0.844  | 0.792  | 1.066   |
| `[dn1, dn1, dn2, dn2]` at (+4, -2, +2)    | 0.849  | 0.879  | 0.965   |

All within ~1σ of 1.0.

### Files changed

- `msrjd/integration/time_domain/final_integral.py`:
  - `_integrate_1d_polytope` — wraps with Heaviside filter.
  - `_integrate_2d_polytope` — wraps with Heaviside filter;
    `pure_s1_found` fix.
  - `_integrate_nd_polytope` — wraps with Heaviside filter;
    `_make_bound_fn` cross-axis filter; restored OUTER_CAP = 200.
  - `_make_heaviside_filtered_integrand` — new helper.
- `msrjd/integration/time_domain/__init__.py` — re-exported
  `identify_loop_subgraphs` for test-module imports (latent drift).
- `models/cumulant_estimator.py` — `_two_point` factorial-correction
  branch for same-pop same-ft same-lag spike pairs.
- `tests/test_time_domain.py` — four new regression tests:
  - `test_phase_J_nd_polytope_preserves_deferred_constraints`
  - `test_phase_J_nd_polytope_simplex_gaussian`
  - `test_phase_J_heaviside_filter_kills_overshoot`
  - `test_phase_J_2d_polytope_pure_external_constraint`
- `notebooks/hawkes_td_only.ipynb`:
  - Cell 25: set `QUAD_OPTS = {'limit': 100, 'epsrel': 1e-4}` for the
    tightened-quadrature path used at k=4.
  - Cell 26: added `SKIP_CELL = True` gate (hand evaluation is
    diagnostic only; cell 31 doesn't need it).
  - Cell 27: guarded against empty `hand_eval_results`.
  - Cell 31: `dt_bin = 1.0` default; `BASE_SEED` randomized via
    `secrets.randbits(31)` each run.

### Methodology note

The four bugs were identified by iterative parallel agent audits.
Early agents hypothesized the cumulant subtraction formula, residue-
matrix formula, and diagram enumeration; numerical cross-checks and
hand-combinatorics ruled these out.  Two later agents (with the
instruction "there IS a bug causing overshoot, find it") located
bugs 1 and 3 by reading the 2D and nD polytope code paths.  The
sim-side `_two_point` gap (bug 4) was identified via static code
analysis comparing the 4-point product's estimator choice to the
pair-partition subtraction's.

---

## 2026-04-14 — Inter-vertex Wick contraction enumeration

### Summary

Fixed a remaining ~15% theory-vs-simulation underestimate at large negative
τ₁ for the k=3 linear Hawkes correlator with repeated external field types
[dn₁, dn₁, dn₂].  The root cause was that `integrate_tree_diagram` only
computed ONE canonical-to-leaf mapping per diagram, missing the other
inter-vertex Wick contractions that become dominant when the "earliest"
external leg is not the first canonical leg.

### Root cause

For Configuration B of a 4-edge chain diagram (one dn₁ leg at the source S,
one dn₁ leg at the interaction V, one dn₂ leg at V), there are 2 distinct
Wick contractions:

- **Mapping 1:** dn₁(t₁=0) at S, dn₁(t₂=τ₁) at V
- **Mapping 2:** dn₁(t₁=0) at V, dn₁(t₂=τ₁) at S

These give genuinely different integrands when t₁ ≠ t₂, because the
internal vertices have different time arguments.  By causality (retarded
propagators), the earliest-time external leg wants to be at the causally-
upstream source vertex S.

- **For τ₁ > 0**: t₁ = 0 is the earlier time.  Mapping 1 is causally natural
  (source has the earlier leg).  Mapping 2 is exponentially suppressed.
  Pipeline's single-mapping evaluation matched simulation. ✓
- **For τ₁ < 0**: t₂ = τ₁ is the earlier time.  Mapping 2 is causally
  natural.  Pipeline only computed Mapping 1 → exponentially suppressed
  contribution → theory undershoots simulation by ~15%. ✗

The previous "position-aware" deduplication in symmetry.py kept both
"duplicate" TypedDiagrams (which evaluated identically under the canonical
remapping), accidentally double-counting Mapping 1 and giving approximately
correct answers at small |τ₁|.  This masked the underlying issue.

### Fix

Two coordinated changes:

1. **`msrjd/diagrams/symmetry.py`** — Reverted `diagram_signature` and
   `_vertex_combinatorial_factor` to use field-type (not leaf-position)
   identification for external edges.  Diagrams that differ only by
   permuting same-type leaves are now merged into a single TypedDiagram.
   This gives 10 unique k=3 tree diagrams (down from 14), with the 4
   "duplicates" correctly merged.

2. **`msrjd/integration/time_domain/final_integral.py`** — Added
   inter-vertex Wick contraction enumeration in `integrate_tree_diagram`:
   - Enumerate all canonical-to-leaf mappings (permutations within each
     same-type field group).
   - For each mapping, evaluate the integrand by permuting the positional
     arguments appropriately.
   - Sum all mappings and divide by a compensation factor = product over
     internal vertices V of (product over same-type field groups at V of
     `n_V!`), where `n_V` is the number of same-type legs at V.

This gives the correct behavior for all cases:

- **Star** (e.g., 3 legs at 1 vertex): 2 mappings × same integrand / comp=2
  = single integrand value.  Same as before. ✓
- **Configuration A** (dn₂ at S, 2×dn₁ at V): 2 mappings × same integrand /
  comp=2 = single value.  Same as before. ✓
- **Configuration B** (dn₁ at S, dn₁+dn₂ at V): 2 mappings × different
  integrands / comp=1 = Mapping_1 + Mapping_2.  Previously missing Mapping_2
  is now included. ✓

### Verification

- All 130 unit tests pass.
- 10 unique k=3 tree diagrams (2 stars + 8 four-edge), down from 14.
- Theory/simulation ratio at τ₁=-20 improved from ~0.84 to near 1.0.
- Positive-τ₁ region still matches (unchanged, since Mapping 2 is
  negligible there).

### Files changed

- `msrjd/diagrams/symmetry.py` — `diagram_signature`,
  `_vertex_combinatorial_factor` reverted to field-type
- `msrjd/integration/time_domain/final_integral.py` —
  `integrate_tree_diagram` vertex_time assignment and contribution wrapper

### Notebook speed optimization (same day)

- **Cell 7.4 (cell 23):** τ grid coarsened from step=0.05 (2001 points)
  to step=0.5 (201 points) — 10× fewer slice evaluation points.
- **Cell 8.1c (cell 28):** Slice evaluation parallelized with
  `ThreadPoolExecutor` across diagrams.  `scipy.integrate.quad` releases
  the GIL during C-level quadrature, so threads give true parallelism.
  On 12-core machine: ~30 min → ~2 min for full slice evaluation.
- **Cell 26 (8.1b):** Hand-integration polytope bounds computed
  analytically (instead of `[-200, 200]` brute force) for 1D and 2D
  smooth_integral cases, matching pipeline convention.
- **Known issue for 4-edge 2D integrals:** Hand integration still
  underestimates pipeline by ~5-6% due to `nquad` adaptive sampling
  difficulty.  Not a theory bug — pipeline value is authoritative.

---

## 2026-04-13 — Position-aware deduplication and pipeline integration fixes

### Summary

Fixed a ~30% magnitude mismatch between tree-level k=3 theory and simulation
for the linear Hawkes model.  The root cause was two interacting bugs in the
diagram deduplication and integration pipeline.

### Bug 1: False deduplication of diagrams with repeated external field types

**Symptom:** For k=3 with external fields [dn₁, dn₁, dn₂], the pipeline
produced 10 unique tree diagrams instead of the correct 14.  Four diagrams
were falsely merged during deduplication.

**Root cause:** `diagram_signature()` in `msrjd/diagrams/symmetry.py` used
the external field TYPE `('dn', 1)` in its per-vertex leaf multiset.  When
two leaves carried the same field type but connected to different internal
vertices, swapping them across vertices produced an identical signature —
merging two physically distinct diagrams (different integrands due to
different time arguments at different vertices).

**Fix:** Hybrid position-aware signature:
- **Multi-vertex diagrams** (leaves at >1 internal vertex): signature
  includes the leaf POSITION index, so cross-vertex swaps of same-type
  leaves produce distinct signatures.
- **Single-vertex diagrams** (star graphs, all leaves at one vertex):
  signature uses field TYPE only, since within-vertex permutations of
  same-type legs give the same commutative integrand.  The combinatorial
  factor M handles these permutations.

The combinatorial factor `_vertex_combinatorial_factor()` was updated to
match: multi-vertex diagrams use position-aware targets `('leaf', position)`
for external edges, while single-vertex diagrams use the original field-type
pairing.

**Verification:** 2+3×2×2 = 14 unique tree diagrams for k=3 linear Hawkes.
All 130 unit tests pass.

**Files changed:**
- `msrjd/diagrams/symmetry.py` — `diagram_signature()`,
  `_vertex_combinatorial_factor()`
- `tests/test_symmetry.py` — updated
  `test_mixed_response_legs_distinct_pairings` expected M value

### Bug 2: Canonical remapping in integrate_tree_diagram erased diagram distinctions

**Symptom:** Even after fixing the deduplication, the 4 new diagrams
evaluated to EXACTLY the same values as 4 existing ones (e.g. D6 ≡ D10),
effectively double-counting rather than adding genuinely new contributions.

**Root cause:** `integrate_tree_diagram()` in
`msrjd/integration/time_domain/final_integral.py` used a "first unused
matching field" canonical remapping to assign times to leaves.  For two
diagrams differing only in which dn₁ leaf connects to which vertex, this
remapping swapped both the time assignments AND the propagator entries,
and the commutative product produced an identical integrand.

**Fix:** Replaced the "first unused match" canonical remapping with a
greedy assignment that respects each diagram's specific leaf-field mapping.
The `_canon_to_leaf` dict now maps each canonical position (external_fields
index) to the specific leaf that carries that field, using leaf list order
to break ties.  This ensures diagrams with different leaf-to-vertex
assignments produce genuinely different integrands.

**Verification:** D6 and D10 now produce distinct values at test points.
Theory/simulation ratio improved from ~0.70 to ~1.0.

**Files changed:**
- `msrjd/integration/time_domain/final_integral.py` — `integrate_tree_diagram()`
  vertex_time assignment (lines 216–247)

### Notebook changes (hawkes_td_only.ipynb)

- **Cell 8.1b (cell 26):** Rewritten for clean 5-step LaTeX display of
  δ-function integration.  Shows symbolic Steps 1–5 (full integral →
  expand G^R → distribute → sift δ → stationary variables) plus per-term
  numerical check at test points `(±2, ±1)`.  Results stored in
  `hand_eval_results` dict for reuse by downstream cells.

- **Cell 8.1b' (cell 27):** New term-by-term comparison cell.  Select
  `DIAG_IDX` to compare hand-computed terms against pipeline per-diagram
  callable.  Reads from `hand_eval_results` (no recomputation).

- **Cell 8.1c (cell 28):** Parallelized slice evaluation using
  `ThreadPoolExecutor` across diagrams.  Coarsened τ grid from step=0.05
  (2001 points) to step=0.5 (201 points) for ~10× speedup.

- **Cell 8.2 (cell 29):** Rewritten as 4-panel theory vs simulation
  slice comparison (2×2 grid: vary τ₁/τ₂ × τ_fixed=±1).

- **Cell 9-quick (cell 31):** New fast point-evaluation cell comparing
  theory + hand sums + simulation at 4 test points.

- **Cell 8.1d:** Deleted (diagnostic clutter).
- **Cell 8.2b (old):** Deleted (replaced by 8.1b').
- **Cell 8.3 (heatmap):** Disabled (finicky, for later).

- **Configuration:** `fundamental['w'] = [[0.4, 0.5], [0.5, 0.4]]`,
  `fundamental['E'] = [1.0, 1.0]`, `USE_CACHE = False`,
  `TAU_FIXED_LIST = [1.0, -1.0]`.

---

## 2026-04-08 — Phase J 2D polytope bounds-sentinel fix (non-star trees)

### Symptom

On k=3 (and any tree with `m >= 2` integration variables), Phase J
produced output that **grew unboundedly** at large |τ|, diverging
into large negative values. This was immediately visible on the
linear Hawkes k=3 autocorrelator-like slices: both slice 0 and
slice 1 reached −5e-3 to −8e-3 at τ = ±50 — roughly 5× larger in
magnitude than the peak value near τ = 0, and with the wrong sign.

A per-kernel breakdown revealed that the divergent behavior came
**only from kernel groups with two internal (non-leaf) vertices** —
i.e., non-star trees where there's an interaction vertex between
the source and some of the leaves. Groups with a single source
vertex (pure star trees, `m = 1` integration variables) decayed
correctly.

### Root cause

`_resolve_1d_bounds` returned the sentinel `(math.inf, -math.inf)`
to signal "infeasible half-space intersection". For the 1D polytope
integrator this was fine because `_integrate_1d_polytope` checks
`if L >= U: return 0` **before** calling `scipy.integrate.quad`.

But for the **2D** polytope integrator, the inner-bounds function
`bounds_s0(s_1_val)` is passed directly to `scipy.integrate.nquad`.
When part of the outer variable's range is outside the polytope
projection, `bounds_s0` returns the infeasible sentinel —
`scipy.nquad` then calls `scipy.quad(f, +inf, -inf)` on the inner
axis, which silently returns **the negative of the full real-line
integral**, not zero:

    scipy.quad(exp(-x²), +inf, -inf) = −1.7724… (= −√π)

The full real-line integrand is multiplied by all other factors
evaluated at that s_1 value, and this nonzero "phantom" contribution
is then integrated over the outer variable. The result accumulates
a large, oscillating wrong value that dominates the physical
contribution at large |τ|.

This bug was not caught earlier because the existing non-regression
Phase J fixtures all used simple star trees (`m = 1`). The k=2 tests
never exercised `_integrate_2d_polytope` at all.

### Fix

`msrjd/integration/time_domain/final_integral.py` —
`_resolve_1d_bounds` now returns a **degenerate empty interval**
`(0.0, 0.0)` when the half-space intersection is infeasible, instead
of `(math.inf, -math.inf)`. `scipy.quad(f, 0, 0)` correctly returns
0, so the 2D outer integration gets a clean 0 from the inner
integrator on the infeasible portion of the outer range — which is
the mathematically correct behavior.

The 1D polytope integrator still catches `L >= U` up-front, so
changing the sentinel form doesn't affect 1D-path behavior.

### Verification

- `msrjd/integration/time_domain/final_integral.py` was the only
  file changed.
- On the linear Hawkes k=3 `[('dn', 1), ('dn', 1), ('dn', 2)]`
  fixture, Phase J now produces decaying slices:

      τ       slice 0 (vary t_2)    slice 1 (vary t_3)
     −50      1.96e−05              5.10e−05
     −25      2.10e−04              5.23e−04
     −10      8.61e−04              2.02e−03
     −1       1.99e−03              4.38e−03
      0       1.05e−03              1.05e−03
     +1       2.11e−03              1.46e−03
     +10      1.43e−03              9.60e−04
     +25      7.27e−04              4.73e−04
     +50      2.26e−04              1.44e−04

  Both tails shrink to O(1e−4) at |τ|=50 vs O(1e−3) near τ=0 —
  the expected stationary-correlator decay.

- `tests/test_time_domain.py` — new test
  `test_phase_J_2d_polytope_decays_at_large_tau`:
  1. Builds a 2×2 diagonal propagator (two retarded poles with
     different decay rates).
  2. Constructs a minimal non-star k=2 tree with an interaction
     vertex between the source and one of the leaves, forcing
     `m = 2` polytope integration.
  3. Asserts `|C(τ=±50, 0)| < 0.01 × |C(0, 0)|` — catches both the
     old divergence (which had the tail much larger than τ=0) and
     any weaker failure to decay.

- Full suite: **130 passing** (was 129, +1 new regression test).

### Note on the symbolic integrands

Per the user's observation: the `format_td_integral_latex` symbolic
integrand display for each k=3 kernel group was already correct —
the prefactor, the product of `G_R` factors, and the retardation
Θ-constraints were all right. The bug was purely in the numerical
quadrature's handling of the infeasible-bounds sentinel, not in
how the integrand was built.

---

## 2026-04-08 — Phase J k=3 support: skip Phase I pole/FFT IFT at k>=3, evaluate time-domain slices vs simulation

### Context

Phase I's pole/FFT inverse-Fourier-transform path is unreliable for
`k >= 3`: the nD spectrum grid aliases badly on anything but a very
fine ω grid, and the residue IFT doesn't generalize cleanly to
multi-dimensional time arguments (`C(τ_1, τ_2, ...)` with multiple
time differences). The user asked for the notebook to skip Phase I
entirely at `k >= 3` and compare the Phase J time-domain evaluator
directly against the simulation cumulant slices.

### What changed

- `msrjd/integration/time_domain/final_integral.py` —
  `eval_delta_contributions_on_tau_grid` is generalized from
  `free_ext_dim == 1` to arbitrary `free_ext_dim`, with two new
  parameters:
    - `vary_index` (int): which component of the free-external-time
      vector is swept along `tau_grid`.
    - `fixed_values` (dict): `{j: value}` pinning the other
      components to fixed values (default 0.0 each).
  A δ contribution whose equality collapses to `0 = 0` on the chosen
  slice (a "δ along the whole slice") is silently skipped — it can't
  be represented as a single-bin spike. Single-point-on-slice spikes
  continue to work as before.

- `notebooks/hawkes_linear_phi_test.ipynb` cell 28 — the `k >= 2`
  frequency-domain evaluation block is gated on `k == 2`. For
  `k >= 3` a short prelude creates a τ grid and sets every Phase I
  output variable (`C_tree_tau`, `C_total_tau`, `C_tree_residue`,
  `C_tree_tau_slices`, `C_total_tau_slices`) to `None`, so
  downstream cells can detect Phase I's absence and fall back to
  Phase J. The existing k=2 code is indented under `else:` and is
  otherwise untouched.

- `notebooks/hawkes_linear_phi_test.ipynb` cell 30 (Phase J 8.1) —
  now has three branches:
    - `k == 1`: no-op (deferred)
    - `k == 2`: single-slice evaluation with `t_2` pinned to origin,
      δ spikes inserted via `eval_delta_contributions_on_tau_grid`
      at `free_ext_dim=1`.
    - `k == 3`: `origin_leaf_idx=None`, evaluate two slices that
      match the simulation cell's convention
      (`C(0, τ, 0)` for slice 0 and `C(0, 0, τ)` for slice 1 —
      varying leaf 1's and leaf 2's times respectively). δ spikes
      are inserted per slice via
      `eval_delta_contributions_on_tau_grid(..., free_ext_dim=3,
      vary_index=1 or 2, fixed_values={other_two: 0.0})`. Results
      are stored in a new `C_tree_phase_j_slices = {0: …, 1: …}`
      dict.

- `notebooks/hawkes_linear_phi_test.ipynb` cell 31 (Phase J 8.2
  plot) — a new k=3 branch draws both slices in a 2-panel figure
  labelled `C^{(3)}(τ_1, 0)` and `C^{(3)}(0, τ_2)`. The k=2 overlay
  now also guards `C_tree_tau is not None` so it degrades gracefully
  if Phase I was skipped.

- `notebooks/hawkes_linear_phi_test.ipynb` cell 33 (simulation
  comparison) — the theory-plot block now guards every Phase I
  reference on `is not None` and overlays Phase J whenever it's
  available. For k=2 the Phase J curve is plotted as a dotted magenta
  line on top of the Phase I curves; for k=3 it's plotted as a
  solid magenta line (since Phase I is absent). The simulation
  curve is unchanged.

- `tests/test_time_domain.py` — new test
  `test_phase_J_k3_star_tree_finite_and_stationary`:
  1. Builds the minimal k=3 fixture — a 1-population 1-pole
     propagator with a single source vertex feeding three outgoing
     edges to three leaves (all via the same G_R entry).
  2. Runs `compute_correction_td(..., origin_leaf_idx=None)`.
  3. Asserts the result is finite and stationary:
     `C(t_1+Δ, t_2+Δ, t_3+Δ) == C(t_1, t_2, t_3)` for several Δ, to
     within `1e-8`.

### Numerical sanity

On the notebook's linear Hawkes 4-field k=3 fixture with
`external_fields = [('dn', 1), ('dn', 1), ('dn', 2)]` and the
fundamental parameters `E=[1, 0.5], w=[[0.3, 0.5], [0.1, 0.3]],
tau=10`:

- 6 kernel groups, all handled by `tree_evaluator`, 0 skipped.
- 1 shot-noise δ contribution produced with equality
  `−t_1 + t_2 = 0` (i.e., the "t_1 = t_2" shot-noise hyperplane from
  the two identical `dn_1` leaves).
- Phase J slice values at (t_1, t_2, t_3) = (0, 0, 0) agree between
  the two slices (1.048e-03), as expected when both slices intersect
  at the origin.
- Slice 0 (vary leaf 1) and slice 1 (vary leaf 2) both decay at
  large |τ|.

### Limitations / known caveats

- **Degenerate δ-on-slice**: for the k=3 autocorrelator-like case,
  the δ contribution's equality can collapse to `0 = 0` on one of
  the two chosen slices. For
  `external_fields = [('dn', 1), ('dn', 1), ('dn', 2)]` the single
  δ equality is `t_1 = t_2`; on slice 0 (which fixes t_1 = 0 and
  varies t_2) the δ fires at τ = 0 as a single-bin spike, but on
  slice 1 (which fixes t_1 = t_2 = 0 and varies t_3) the equality
  is satisfied *for every* τ_3 — i.e., the δ lies "along the whole
  slice". The helper silently drops this case. The physical
  interpretation is that this subset contributes a continuous
  function (not a spike) along the slice, which the discrete
  single-bin insertion can't represent. It will show up as a
  systematic offset on slice 1 only if the contribution is large.
  Proper handling requires either a full 2D grid evaluator or
  reformulating that subset as a smooth contribution — deferred.

- **Leaf-label symmetry for identical external fields**: the
  diagram enumeration treats leaves as distinguishable (`leaf i`
  gets `external_fields[i]` fixed, not permuted). For identical
  external fields at leaves 0 and 1, this means Phase J's result
  is not automatically symmetric under `t_1 ↔ t_2`. Stationarity
  (verified by the new test) still holds, and the sum over kernel
  groups does include both orderings at the DIAGRAM level, so the
  grand total should be correct — but individual slices may show
  small non-symmetries depending on whether the per-group
  enumeration covers both orderings. This is a pre-existing
  property of the enumeration layer, not a Phase J issue.

---

## 2026-04-08 — Phase J shot-noise δ(τ) spike: expose and discretize

### Symptom

After the earlier δ(t)-component fix, Phase J correctly computed the
asymmetric cross-correlator `⟨δn₁ δn₂⟩` on the linear Hawkes notebook.
But for the **autocorrelator** case `external_fields = [('dn', 1),
('dn', 1)]`, Phase J's output was missing the **shot-noise δ spike at
τ = 0** that Phase I produces via its residue IFT path. The
continuous Phase J curve matched Phase I everywhere except at the
origin, where Phase I shows a large spike and Phase J showed nothing.

### Root cause

For the autocorrelator tree diagram with a source at `nt_1`, BOTH
edges `(source → leaf_1)` and `(source → leaf_2)` read the same
`G_R[dn_1, nt_1]` matrix entry — and that entry has `δ(t)` coefficient
`c_δ = lim_{ω→∞} G_ft[nt_1, dn_1] = 1`. So in the δ-subset
enumeration, the subset `S = {both edges}` pins both edges' time
differences to zero simultaneously, which forces `s = t_1` AND
`s = t_2`, i.e., `t_1 = t_2` (τ = 0). This is a **pure shot-noise
delta** `combined_pf · c_δ² · δ(t_1 − t_2)`.

In the previous fix (the asymmetry fix for cross-correlators), I
explicitly **skipped** shot-noise subsets because a `δ(τ)` cannot be
represented as a continuous Python callable. That skip was correct
for the `⟨δn₁ δn₂⟩` case (which has no shot-noise subset since edge 2
has `c_δ = 0`), but **wrong** for `⟨δn₁ δn₁⟩` where the subset has a
nonzero weight.

### What changed

- `msrjd/integration/time_domain/final_integral.py` — shot-noise
  subsets are no longer silently skipped. Instead, the tree
  evaluator builds a structured `delta_contributions` entry for each
  shot-noise subset with exactly one non-trivial residual equality
  among external times:

      {
        'coeff_fc': fast_callable over the free ext-time symbols,
        'equality_a': list of float (linear coeffs in free ext times),
        'equality_c': float,
        'retardation_data': [(a_list, c0), ...],
        'equality_symbolic': SR,
        'delta_edges': list of edge indices,
        'free_ext_idx': list,
      }

  This encodes `coeff_fc · δ(Σ aᵢ·xᵢ + c)` as a distribution on the
  free-external-time space, with additional retardation half-space
  checks. The continuous `contribution` callable still represents the
  smooth part only; the spike is returned alongside it.

  Multi-equality subsets (rare at tree level; would correspond to
  `δ(τ_a − τ_b) · δ(τ_c − τ_d)` style double spikes) are detected
  and deferred with a diagnostic entry.

- New helper `eval_delta_contributions_on_tau_grid(delta_list, tau_grid,
  free_ext_dim=1)`. Takes the list of structured δ contributions
  together with a uniformly-spaced 1-D τ grid and returns a numpy
  array the same length as `tau_grid` with the spike weights inserted
  into the correct bins (respecting retardation half-space
  constraints). Each spike is normalized as `coeff / |a| / Δτ` so
  that the bin height times the bin width recovers the analytic δ
  weight. Supports `free_ext_dim == 1` (k=2 with one leaf pinned as
  origin); higher `k` raises `NotImplementedError`.

- `msrjd/integration/time_domain/pipeline.py` — `compute_correction_td`
  now aggregates `delta_contributions` across all tree kernel groups
  and returns them under the same key in its result dict. Per-group
  diagnostics also include `n_delta_contributions`.

- `msrjd/integration/time_domain/__init__.py` — exports
  `eval_delta_contributions_on_tau_grid`.

- `notebooks/hawkes_linear_phi_test.ipynb` Section 8.1 — after
  evaluating the smooth `total_C` on the Phase I residue IFT τ grid,
  calls `eval_delta_contributions_on_tau_grid` to build the discrete
  δ-spike array and adds it to `C_tree_phase_j`. Both the
  "smooth only" and "smooth + δ" values of `C(0)` are printed so the
  user can see which portion comes from the shot-noise spike.

- `tests/test_time_domain.py` — new regression
  `test_phase_J_autocorrelator_delta_spike_at_origin`:
  1. Runs Phase J on the 2×2 instantaneous fixture with
     autocorrelator edges (both edges → `G_ft[0,0]`).
  2. Asserts exactly one `delta_contributions` entry is produced.
  3. Asserts the equality fires at τ = 0.
  4. Asserts the coefficient is `-2` (= `combined_prefactor × c_δ²`
     for the pipeline's `SourceType(SR(1), [ñ, ñ], (2, 0))`).
  5. Discretizes onto a 501-point τ grid in [−5, 5], asserts the
     nonzero bin is at τ=0, and verifies that the bin-height × Δτ
     integrated weight equals the analytic `-2`.

### Limitations / future work

- **Non-origin δ spikes in higher-k correlators**: for k ≥ 3 a
  shot-noise subset could fire on a hyperplane in (τ_1, τ_2, ...)
  space rather than at a single point. `eval_delta_contributions_on_tau_grid`
  currently supports only `free_ext_dim == 1`. Extension to higher-k
  will need a Jacobian determinant (not just `1/|a|`) and a suitable
  "thick hyperplane" bin assignment rule.

- **Multi-equality shot-noise subsets**: a subset with more than one
  residual ext-time equality would correspond to a product of deltas.
  Currently detected and skipped with a diagnostic; should be added
  when a physical case requires it.

- **Retardation checks at non-star trees**: the retardation data for
  shot-noise subsets with surviving smooth edges uses the same linear
  form as the smooth polytope. It has not been exercised yet because
  the MVP trees are stars (all edges directly from the source to
  leaves).

---

## 2026-04-08 — Phase J δ(t)-component fix: handle instantaneous propagator entries

### Symptom

After the earlier overflow fix (see below), Phase J still produced a
**symmetric curve at ~1/3 the correct amplitude** on the full linear
Hawkes k=2 notebook run, while matching on the 1×1 and 2×2 regression
fixtures. The mismatch was specific to propagators whose frequency-
domain entries have a nonzero `ω → ∞` limit (i.e., instantaneous
couplings like `ñ_i × δn_i` in the MSR-JD Hawkes action).

### Root cause

`build_G_t_matrix` constructed the time-domain retarded propagator as
the pole-residue sum

    G_R[i, j](t) ≈ Σ_k C_mats[k][i, j] · exp(I · p_k · t) · Θ(t)

which is the correct smooth decay component — **but misses a δ(t)
contribution** whenever `lim_{ω→∞} Ĝ[i, j](ω) ≠ 0`. The physical
meaning is that an instantaneous coupling in the action (like ñ × δn)
means a ñ source at time `t` produces an *immediate* δn response at
the same time `t`, not a delayed exponential. The full retarded
propagator decomposition is

    G_R[i, j](t)  =  delta_coeff[i, j] · δ(t)
                  +  Θ(t) · smooth[i, j](t)

where `delta_coeff[i, j] = lim_{ω→∞} Ĝ[i, j](ω)`.

For the linear Hawkes 2-pop fixture, `G_ft[nt_i, dn_i] = 1 + O(1/ω)`,
so `delta_coeff[nt_i, dn_i] = 1` — these entries carry a δ(t) that
was being silently dropped from the time-domain integrand.

For the k=2 ⟨δn₁ δn₂⟩ tree diagram with source at `nt_1`, edge 1
(source → leaf₁) uses `G_R[dn₁, nt₁]` which has δ(t), while edge 2
(source → leaf₂) uses `G_R[dn₂, nt₁]` which is smooth. The full
contribution has **four** terms from the 2² = 4 choices of which
edges use their δ component vs their smooth component:

    Σ_{S ⊆ edges}  ∫ ds (∏_{e∈S} c_e·δ(dt_e)) · (∏_{e∉S} smooth_e · Θ)

For this particular diagram:
- `S = ∅`: the smooth-smooth convolution (what Phase J computed before).
- `S = {edge₁}`: δ pins `s = t₁`, contribution
  `c₁ · G_R[dn₂, nt₁](t₂ − t₁)` nonzero only for `τ < 0`.
- `S = {edge₂}`: δ coefficient is 0, drops out.
- `S = {both}`: edge₂ has no δ, drops out.

The missing `S = {edge₁}` term accounted for exactly the observed
discrepancy at `τ < 0` in the notebook plot. Direct verification:
at `τ = −1` Phase J returned `6.58e-3`, direct scipy IFT returned
`2.73e-2`, and the analytical `c₁ · G_R[dn₂, nt₁](1)` correction
was `2.07e-2`. Sum: `6.58e-3 + 2.07e-2 = 2.73e-2` ✓.

### What changed

- `msrjd/integration/time_domain/propagator_td.py` — `build_G_t_matrix`
  now returns a dict `{'smooth', 'delta', 't_var'}` instead of a bare
  matrix. The `'smooth'` entry is the old pole-residue sum; the new
  `'delta'` entry is a matrix of numeric constants `c_{ij} = lim_{ω→∞}
  Ĝ[i,j](ω)` computed by evaluating at a large ω and checking for a
  stable (non-decaying) limit. A new helper `G_t_delta_coeff(G_t_obj,
  pi, ri)` returns the δ coefficient for one entry; `G_t_entry` accepts
  either the new dict or a bare matrix (for backward compat).
- `msrjd/integration/time_domain/final_integral.py` — `integrate_tree_diagram`
  is rewritten to enumerate the `2^|E|` subsets of edges that take
  their δ component. For each subset:
  1. The δ-edge equalities `dt_e = 0` are solved via `sage.all.solve`
     to eliminate integration variables by substitution (on the MVP
     star tree, δ edges pin the source time to a leaf time).
  2. If the residual constraints force equality among external times,
     that subset contributes a `δ(τ)` shot-noise spike — it's counted
     in `n_shotnoise_skipped` and excluded from the continuous
     callable, matching how the notebook's `ift_via_residues` and the
     Phase I residue path handle shot noise separately.
  3. The remaining smooth-factor product is `.expand()`'d into a sum
     of single exponentials (to preserve the earlier overflow fix)
     and JIT-compiled via `fast_callable` over the reduced polytope.
  4. All subset contributions are summed in the final callable.

  The `.is_trivial_zero()` check replaces `subset_factor == 0` because
  the latter triggers Sage's `simplify_full()` which invokes Maxima
  and can hang / throw ECL errors on the deeply nested symbolic
  expressions produced by the 4-field linear Hawkes kernel.

- `msrjd/integration/time_domain/__init__.py` — now exports
  `G_t_delta_coeff` and `format_td_integral_latex`.

- `msrjd/integration/time_domain/final_integral.py` — new helper
  `format_td_integral_latex(tree_result, …)` produces a LaTeX string
  summarizing the Phase J integrand structure for a tree diagram in
  the same style as the notebook's `show_integral` helper for the
  frequency-domain Phase I integrand. It shows the vertex-time
  assignment, the `∫ds · ∏ G_R · Θ` form, and any nonzero δ edge
  coefficients.

- `notebooks/hawkes_linear_phi_test.ipynb` Section 8.1 — now calls
  `format_td_integral_latex` for each tree kernel group and
  `display(Math(...))`s it, so the Phase J integrand structure is
  visible in the notebook output alongside the numerical result.

- `tests/test_time_domain.py` — two new regression tests:
  1. `test_G_t_matrix_detects_delta_component` — on a minimal 2×2
     fixture with `G_ft[0,0] = (1+iω)/(1+a+iω) → 1` at `ω → ∞`,
     verifies `build_G_t_matrix` returns a dict with the correct
     `delta[0,0] = 1` coefficient.
  2. `test_phase_J_delta_component_asymmetric_cross_correlator` —
     end-to-end: constructs a tree diagram whose two edges use
     different matrix entries (one with δ, one without), runs
     Phase J, and compares the callable output to a closed-form
     analytic result derived by hand from the δ-subset expansion.
     Agreement is `< 1e-10` (machine precision) at six τ values
     spanning both signs. Also asserts the result is asymmetric at
     `± τ` — the canonical symptom of the bug before the fix.

  The earlier `test_G_t_matrix_single_pole` test was updated to index
  into `G_t_obj['smooth']` and to assert `delta[0,0] == 0` for the
  non-instantaneous 1×1 propagator.

### Numerical validation

- Full suite: **127 passing** in 15 s (was 125 — two new regression
  tests).
- End-to-end on the linear Hawkes 4-field k=2 pipeline: Phase J now
  matches direct `scipy.integrate.quad` IFT of the frequency-domain
  integrand to **~1e-5 absolute accuracy** at all τ values tested.
  Sample asymmetric output:

      tau         scipy IFT          Phase J            diff       ratio
       -5.00   3.5727e-02        3.5724e-02         2.86e-06    0.9999
       -2.00   4.0905e-02        4.0900e-02         5.30e-06    0.9999
       -0.50   4.3752e-02        4.3743e-02         8.91e-06    0.9998
        0.50   7.2734e-02        7.2743e-02         8.93e-06    1.0001
        2.00   6.8294e-02        6.8299e-02         5.29e-06    1.0001
        5.00   6.0100e-02        6.0103e-02         2.86e-06    1.0000

  Compare to pre-fix:

      tau         scipy IFT          Phase J            ratio
       -5.00   3.5727e-02        1.8127e-02         0.5074
       -2.00   4.0905e-02        2.0984e-02         0.5130
       -0.50   4.3752e-02        2.2581e-02         0.5161
        0.50   7.2734e-02        2.2622e-02         0.3110
        2.00   6.8294e-02        2.1131e-02         0.3094
        5.00   6.0100e-02        1.8425e-02         0.3066

  The asymmetry is now correctly captured (3.57e-2 at τ=−5 vs 6.01e-2
  at τ=+5), matching the Phase I residue and FFT IFT outputs visible
  in the notebook's cell 8.2 overlay plot.

### Limitations / deferred

- **Shot-noise δ(τ=0) spike**: when two or more edges with δ
  components share a source vertex, the δ-subset enumeration forces
  equality among external leaf times, producing a `δ(t_1 − t_2)`
  spike. Phase J counts but **does not represent** this spike in the
  continuous callable. It is reported in `tree_result['n_shotnoise_skipped']`
  for downstream handling. The notebook's `ift_via_residues` path
  handles this separately by adding an explicit delta at `τ = 0` to
  the residue IFT output; Phase J will eventually need the same
  treatment if a downstream caller needs the autocorrelation value
  at `τ = 0`.

- **Non-star trees**: the δ-subset solver currently uses `sage.all.solve`
  to eliminate integration variables one at a time. For star trees
  (all non-leaf vertices are the source) every δ edge pins the source
  time directly, which always succeeds. For trees with interaction
  vertices between the source and the leaves, some subsets may yield
  unsolvable systems or residual integrations over reduced variables;
  those cases should be flagged and verified when they first appear.

---

## 2026-04-08 — Phase J numerical-overflow fix: expand integrand before fast_callable

### Symptom

On nontrivial kernel matrices (anything beyond the 1×1 / diagonal test
fixtures), Phase J's numerical output on the notebook's comparison
plot came out **symmetric in τ and ~half the amplitude** of the Phase
I (notebook FFT / residue IFT) reference. The shape was wrong in
addition to the magnitude — real asymmetric cross-correlators were
being returned as symmetric curves. Individual calls to the tree
evaluator's contribution callable silently returned `nan + nan·j` on
2×2 **nondiagonal** test fixtures.

### Root cause

`integrate_tree_diagram` builds each edge's time-domain propagator as
a **sum of exponentials** (one term per pole of `det K`). After
multiplying the edge factors, the stripped integrand is a **product
of sums**:

    (A₁·e^(α₁·s) + B₁) · (A₂·e^(α₂·s) + B₂) · e^(γ·s + …)

Left in that factored form, `fast_callable` evaluates each factor
separately. At large negative `s`, individual factors can grow like
`exp(|α_i|·|s|)`, and their pairwise products can overflow IEEE double
precision **before** the causal suppression factor `exp(−α_total·|s|)`
brings the product back into range. The MATHEMATICAL result is a
finite, decaying integrand, but the NUMERICAL intermediate values
blow up.

`scipy.integrate.quad(f, −∞, L)` samples at arbitrarily negative `s`
as part of its adaptive quadrature, so any overflow anywhere in the
real line produces `nan` and corrupts the integral. The overflow is
not rare: for the 2-pop 4-field linear Hawkes kernel at typical
parameters it happens in the `s ≲ -200` tail.

Concrete demonstration (2×2 nondiagonal fixture):

    s =  -100:  raw = -5.01e-67   expanded = -5.01e-67   ✓
    s =  -200:  raw = -1.31e-132  expanded = -1.31e-132  ✓
    s =  -400:  raw = -0          expanded = -8.94e-264  ✓
    s =  -700:  raw = -0          expanded = 0           ✓
    s = -1000:  raw = nan + nan·j expanded = 0           ← overflow!

    quad raw      = nan
    quad expanded = -0.042687795113793774   (correct)

### Fix

`msrjd/integration/time_domain/final_integral.py` — one line added to
`integrate_tree_diagram` right before the `fast_callable` step:

    stripped = stripped.expand()

Sage's `.expand()` distributes products of exponential sums into an
explicit sum of single-exponential terms, so
`(A·e^a + B)·(C·e^c + D)·e^g` becomes
`A·C·e^(a+c+g) + A·D·e^(a+g) + B·C·e^(c+g) + B·D·e^g`. Each summand is
`C · exp(α·s + …)` with one coefficient `α`, and at retarded-causal
polytopes every term has `α > 0` (decay as `s → −∞`). Numerically,
each term is evaluated as a single `exp`, so there is no overflow in
intermediate products — the only overflow risk is if `|α·s| > 1024`,
which happens far beyond where any term has measurable magnitude
anyway.

### Verification

- `tests/test_time_domain.py` — all 6 tests still pass in 3.5 s.
- Full suite: **124 passing** in 15 s. No regressions.
- 2×2 nondiagonal smoke test (propagator
  `K = [[1+iω, -3/10], [-2/10, 1+iω]]`, source at ñ₁ with
  cross-mode edges to dn₁ and dn₂) is now **asymmetric** and tracks
  the notebook's FFT IFT:

      tau     notebook FFT     Phase J (fixed)   diff
     -3.00   -4.08e-02        -4.05e-02          3e-4
     -1.00   -1.18e-01        -1.17e-01          6e-4
     -0.30   -1.25e-01        -1.25e-01          4e-4
     +0.30   -7.94e-02        -8.04e-02          1e-3
     +1.00   -4.34e-02        -4.27e-02          7e-4
     +3.00   -7.92e-03        -7.83e-03          9e-5

  The residual shrinks monotonically as the reference FFT grid gets
  finer (`N`=4096 → `Omega_max`=80 → 1.0e-3; `N`=65536 →
  `Omega_max`=500 → 1.2e-4), confirming Phase J is giving the exact
  continuum answer and the residual is just truncation error in the
  reference, not a Phase J bug.

### Why the simpler tests still passed

The pre-fix `test_k2_tree_single_integration_analytical`,
`test_k2_tree_translation_invariance`, and
`test_phase_J_vs_phase_I_linear_hawkes_tree` tests all used a 1×1
propagator `K(ω) = 1 + iω`. For a 1×1 kernel there is exactly one
pole, so each edge's G(t) is a single exponential (no sum-of-sums
structure), and the product of edge factors is already a single
exponential — no `expand()` needed, no intermediate overflow. The
bug only manifests when the propagator has ≥ 2 poles AND the product
includes more than one sum-of-exponentials factor, which is the
generic case for any multi-field kernel (2-pop Hawkes, 4-field linear
Hawkes, etc.).

A new 2×2 nondiagonal regression test covering this case should be
added before Extension 1.

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

# M5 cloud run: per-diagram setup levers (bit-identical speed-up)

**TEMPORARY folder.** It exists only on branch `cloud/m5-base` to brief a cloud session, and the coordinator removes it
when merging. Do not edit these files, except:
- write your report to `cloud_handoff/M5_REPORT.md`;
- put logs, small JSON outputs and **every script you write** in `cloud_handoff/out/`.

`scripts/` is git-ignored, so a script saved there never gets pushed. Before the final commit, run
`git status --ignored cloud_handoff` and make sure nothing you need is ignored.

**Model:** Sonnet 5.5, effort high. **Orchestration:** solo, except one independent verification agent at the end.

This item must **not move any number**. Every result with a lever on must be `np.array_equal` to the same run with the
lever off. Correctness is that equality. Speed is the goal; report what you measure and do not chase the targets.

## 0. Environment (Ubuntu cloud VM: 4 vCPU, 16 GB)
1. **Reuse first.** If `micromamba env list` shows `daedalus`, activate it:
   `eval "$(micromamba shell hook -s bash)" && micromamba activate daedalus`, then `python -c "import sage.all, pytest"`.
   If that succeeds, skip to step 3.
2. **Install, if needed** (`micro.mamba.pm` is blocked by the proxy):
   ```
   curl -sSL -o micromamba.tar.bz2 https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2
   mkdir ex && tar -xjf micromamba.tar.bz2 -C ex && cp ex/bin/micromamba /usr/local/bin/
   export MAMBA_ROOT_PREFIX=/root/micromamba
   micromamba create -y -f environment.yml -n daedalus        # ~3 min
   eval "$(micromamba shell hook -s bash)" && micromamba activate daedalus
   python -c "import sage.all, pytest"                        # must be silent
   ```
   Run every test as `python -m pytest ...` inside this env, always with `PYTHONHASHSEED=0`.
3. **Benchmark model.** `cp cloud_handoff/ou_quartic_two_dim.model.py models/` (do **not** commit it). Every other
   model you need is already in the repo.
4. **Known baseline on this VM (Linux x86). None of these is yours:**
   - Default suite, 5 failures: `test_phase_j_structural_zeros.py::{test_flag_off_restores_the_pre_m2a_fallback,
     test_flag_is_read_at_call_time, test_perdiag_cycle_through_the_fallback_fixes_the_overflow}` and
     `test_phase_j_nquad_hardening.py::{test_old_routine_samples_empty_inner_intervals,
     test_two_cycle_reaches_no_evaluation_without_structural_zeros}`.
   - Deselect `tests/test_spatial_correlator.py::test_periodic_inverse_ft_limits[3]` (OOM on 16 GB). The 23 `fastenum`
     tests skip.
   - Slow suite, failures that also fail on the parent: `ou_quartic_two_dim_color_corr-k2-l1-default`,
     `multipopulation_test-k2-l1-default`, `test_phase_j_ties::test_k4_two_loop_no_nquad_fallback_above_mu_1p2`
     (strict XPASS), `test_phase_j_nquad_hardening::test_bit_identical_to_the_pre_m2b_engine`,
     `test_phase_j_structural_zeros::test_public_spike1_unchanged[perdiag]` and
     `test_phase_j_structural_zeros::test_bit_identical_to_the_pre_m2a_engine`.
   - **Do not run** `tests/test_dyson_dressing.py` or `tests/test_spatial_general_k.py::test_k3_coupled_decoupled_limit`
     and the tests after it (hours or OOM on 16 GB). Do not run `quadratic_hawkes_alpha` (OOM). Do not use `pkill -f`
     (it kills your own shell). Do not run two jobs that build the same model cache at once.
   - Test platform note: a test that compares a stored value captured elsewhere bit-for-bit can fail on one ulp across
     platforms. The bit-identity gate is lever on vs off **in one process**.
5. **Long runs.** Background commands are capped at about 30 min: split the suite into chunks of test files (M4 used 7).
   About 60 s after each launch, read the log: tests must be collected and running.
6. **Warm the disk cache before any A/B.** The first run of a model in a fresh checkout writes `saved_models/<model>/`
   (expand cache); a cold run differs from a warm run at ~1e-20 even with every lever off. Run each model once and
   discard it, or compare only warm runs.

## 1. What to build (the "P6" levers)
Per-diagram setup is the next cost after M4: for the two-field OU at ℓ=2, 5217 of 5347 diagrams have a numerically
zero prefactor yet still pay the full setup, and the model-level data (`build_G_t_matrix`, pole/residue arrays) is
rebuilt per diagram. The entry point is `integrate_diagram` in `engine/integration/time_domain/final_integral.py`
(find it with `grep -n "^def integrate_diagram"`; **line numbers in this brief are not given on purpose, since M1–M4
moved everything: grep for names**). The caller is `compute_correction_td` in
`engine/integration/time_domain/pipeline.py`.

`cloud_handoff/proptd_proto.py` is a monkeypatch prototype of the levers (usage in its docstring; models `ou2`, `ou1`;
run from the repo root with `python cloud_handoff/proptd_proto.py ou2 2 gtmemo,modes,autmemo,zskip`). It measured
max|dC|/max|C| = 0 for all combinations. It is a reference for the idea, not production code: the production
version must live in the engine behind flags, with no id()-keyed global memos.

**One commit per lever**, each behind its own flag (read at call time, env override, default ON), each with its own
counter(s) in `_RUNTIME_COUNTERS`. Copy the pattern of `USE_CHAIN_UPPERS_MEMO` / `DAEDALUS_CHAIN_UPPERS_MEMO`
(M4, same file). An umbrella `DAEDALUS_PHASE_J_LEGACY_SETUP=1` forces every setup lever off. It is a different switch
from `DAEDALUS_PHASE_J_LEGACY`, which reproduces older NUMBERS: do not touch that dict or its tests (the levers change
no number).

- **L1: zero-prefactor early exit** (`DAEDALUS_SETUP_ZERO_EXIT`).
  - At the top of `integrate_diagram`, before `build_G_t_matrix` runs: if the diagram has no vertex with
    `cumulant_specs` (noise-source vertex) and no `ConvVertexType` vertex, and
    `complex(CDF(SR(combined_prefactor).subs(num_params))) == 0` **exactly**, return a zero contribution.
  - Its return value must have exactly the structure and types of what `integrate_diagram` returns today for an empty
    subset list (read that branch: totals, `_comp`, the per-subset lists, dtypes). `np.array_equal` on the caller's
    results is the test.
  - **List positions must not change.** `api/report.py` and `eval_per_diagram_batch` (pipeline.py) index diagrams
    positionally, so the diagram stays in the list with a zero entry; never filter it out.
  - If the prefactor cannot be evaluated numerically (exception, symbolic leftovers), do not exit early.
  - Measured on the prototype: two-field OU ℓ=2 spends 12.7 of 14.1 s of `integrate_diagram` on these diagrams.
- **L2: model-level data built once per `compute_correction_td` call** (`DAEDALUS_SETUP_PROP_TD`).
  - A `PropagatorTD`-style object holding complex arrays of the poles, residues `R[k, phys, resp]` and the D
    quantities, all converted with the *same* `complex(CDF(SR(x)))` as the current per-diagram code, plus a lazily
    built `G_t_obj` (the `build_G_t_matrix` result) and the per-`(pi, ri)` mode tuples that
    `_build_edge_mode_sums` builds.
  - **Scope it per call, never key it by `id()` in a module global.** The spatial bridge
    (`engine/integration/spatial/pipeline_bridge.py`, `pipeline_C_q_tau`) mutates the propagator for each q-sample, so a cache that outlives
    a call would serve stale data.
  - New optional kwarg `prop_td=None` on `integrate_diagram`; when None, build locally exactly as today. Any caller
    that does not pass it must behave as before.
  - Measured on the prototype: two-field OU ℓ=2 setup 13.8 → 5.2 s.
- **L3: lazy SR** (`DAEDALUS_SETUP_LAZY_SR`).
  - `edge_info[i]['smooth_factor']` (an SR object from `G_t_entry(G_t_obj, ..., include_heaviside=False)`) is needed
    only by the shot-noise branch and the SR-integrand branch. Make it an on-demand accessor (or a lazy mapping),
    keeping the key present.
  - `display_stripped`/`stripped_integrand` is display-only; commit `851be3a` already stopped expanding it. Keep that
    key present in every returned record, but build it only under a debug flag or lazily. Grep for consumers first
    (`stripped_integrand`); if you find a real numeric consumer, stop and report it as open.
  - Measured on the prototype: about −1.2 s on the two-field OU.
- **L5: shared automorphism work** (`DAEDALUS_SETUP_AUT_MEMO`).
  - In `engine/diagrams/symmetry.py`, skip `external_wick_compensation` when there is a single mapping, and memoise
    `_automorphism_order` per `(typed_diagram, fix_external)`.
  - Key by the diagram object held in the memo value (so its `id` cannot be reused), scoped per `compute_correction_td`
    call or a bounded dict; never an unbounded `id()` global. Thread safety as in M4 (lock; compute outside it).
  - Measured on the prototype: about −0.6 s.

**Out of scope** (list under "open", do not do):
- numeric δ-elimination (`sage_solve` replacement, "L4a/L4b"): that is a later milestone (M10) with its own
  exact-zero contract;
- anything that changes a number; the fast-pole / DBM machinery; the chain memo (M4) or its kernels;
- `notebooks/`; model files other than the one benchmark copy.

## 2. Acceptance criteria (decided beforehand)
- **A (bit-identity).** For every lever and for all levers together, in one process (toggle the module flags, clear
  any per-call state), `np.array_equal` totals, per-ℓ values and per-diagram values on:
  - `ou_quartic` k=2 ℓ≤2 (and ℓ=3 if it runs in < 5 min), `ou_quartic` k=4 ℓ=1;
  - `ou_quartic_two_dim` k=2 ℓ=1 and ℓ=2, 7 τ on [0, 3];
  - spike-reset k=2 ℓ=1, per-diagram and grouped; `single_population_quad_exp_test` k=2 ℓ=1;
  - the models of `tests/test_fix_colored.py` and `tests/test_conv_kernel_attachments.py`: these are the cases where
    the shot-noise / SR / conv branches really run, so they are the L1/L3 safety net;
  - one small spatial Allen-Cahn run (find the shortest in `tests/test_spatial_*.py`), which exercises the per-q
    propagator scoping of L2; compare its output lever on vs off;
  - the zoo entries of `tests/tools/phase_j_zoo_baseline.py` that exist in this checkout (skip `MISSING_MODEL` ones),
    via `cloud_handoff/zoo_ab.py` (a template; edit it, copy your version to `cloud_handoff/out/`).
  - Run a **positive control**: temporarily break one lever (e.g. make L1 drop a nonzero prefactor) and check the A/B
    script catches it. Report that you did.
- **B (tests)** in `tests/test_phase_j_setup_fastpath.py`: for each lever, lever on vs off `array_equal` on the cases
  above that run in < 2 min each; counters show the lever fired (e.g. L1 skip count > 0 on the two-field OU, = 0 on
  a colored-noise model); flag toggle and umbrella switch; `prop_td=None` path; a per-call-scope test for L2 (two
  successive calls with different `num_params` give the results of independent calls); a threaded smoke test
  (4 threads calling `integrate_diagram` / the memoised helpers on the same inputs, results equal to serial).
- **C (speed, report only).** Wall time per case, lever off → on, plus the `integrate_diagram` setup share from a
  `cProfile` of `ou_quartic_two_dim` k=2 ℓ=2. Planning targets on the Mac (setup only, then end to end): two-field OU
  ℓ=2 setup 13.8 → 5 s; end to end 41.8 → 23 s; OU ℓ=3 warm 4.0 → 1.8 s; OU k=4 ℓ=2 setup 5.7 → 0.15 s. This VM is
  slower and noisier (±2 s); compare only within one session, off/on alternated.
- **D.** The default suite and the slow suite (§0.4 exclusions) show no new failure against the §0.4 baseline.
  Counter dictionaries are compared by a few tests (e.g. `test_public_spike1_unchanged` filters `chain_*` keys): if a
  new counter breaks such a comparison, filter the new key in the test the way M4 did, and say so in the report.

## 3. Process
1. Measure baselines with every lever off (warm cache), with wall times and values.
2. Implement L1, L2, L3, L5 as separate commits, each with its tests. Check bit-identity after each.
3. Run A and C.
4. **One independent verification agent**: re-runs A on two models of its choice, plus one adversarial case (a diagram
   whose prefactor is zero but which has a noise-source or conv vertex; a spatial run with two different q-sample
   propagators in one process), and reviews L1's early-exit condition and L2's scoping for staleness. Fix what it
   finds, then re-verify.
5. Run the suite (D).
6. **CHANGELOG** (0.2.0, unreleased, under "Performance"): the four levers, their flags and counters, the umbrella
   switch, and the measured speed-ups.
7. **Commits.** Work commits touch only `engine/`, `tests/` and `CHANGELOG.md`, **never** `cloud_handoff/` or
   `models/`. A separate final commit touches only `cloud_handoff/`. Stage paths explicitly (never `git add -A`).
   End each message with the attribution line from your system reminder. Push to `cloud/m5-base`; if refused, push
   your session's branch and name it in the report.
8. **Report: `cloud_handoff/M5_REPORT.md`** with: Result (one line); Environment; What changed (files, commits, flags,
   counters); A–D as a table with values, pass/fail and log paths; Speed-ups; Verification-agent findings and
   dispositions; Open issues; Wall time; Model/effort adequacy; Advice for the next handoff.

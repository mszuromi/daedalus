# RFC cloud run: speed up `record_from_cert` (bit-exact) so a first use from the shipped cache is fast

**TEMPORARY folder** on branch `cloud/rfc-base`; the coordinator removes it when merging. Write your report to `cloud_handoff/RFC_REPORT.md`; put logs and EVERY script you write in `cloud_handoff/out/`
(run `git status --ignored cloud_handoff` before the final commit). **Model:** Sonnet 5.5, effort high. **Orchestration:** solo, no subagents. **Budget:** hard cost ceiling; targeted tests while you work, ONE default-suite run at the end, no slow suite.
Commits: one work commit touching only `engine/enumeration/prediagram_cache.py` (the function and its private helpers), `tests/` and `CHANGELOG.md`; a separate final commit touching only `cloud_handoff/`; stage paths explicitly; end each commit message with the attribution line from your system reminder; push to the branch. No private model; do not touch notebooks or other files.

## What this is
The repository ships 21 prediagram cells as packed certificates (`engine/enumeration/shipped_prediagrams/`, stamped `.pkl`/`.pkl.xz`; read `prediagram_cache.py`: `load_shipped_certs`, `unpack_cert`, `cert_to_graph`, `relabel_leaves_first`, `record_from_cert`, `records_from_certs`). On a fresh clone the first use of a cell rebuilds every record with `record_from_cert`,
which a local profile showed to be 99 % of the first-use time (about 1.0 s for (2,3) with 14,928 classes, 4.8 s for (4,2), plus about 2.3 s of package import that you cannot change): each record builds THREE Sage graphs (the canonical `DiGraph`, an undirected `Graph`, and a relabelled `DiGraph` with distinct edge labels) and calls `D.degree(v)` per vertex. For the larger cells the cost scales with the class count
(for (2,4), 1,152,032 classes: report the projected time and memory; do not run it end to end unless it takes under 10 minutes).
**Goal:** a faster `record_from_cert` that returns EXACTLY the same record as today: same `(D, G, leaves, internal)`, same vertex lists and order, same edge multiset with the same labels in the same order, same `leaves`/`internal` lists, so every downstream consumer (`type_assignment` etc.) is unaffected.
Keep the current implementation in the module as `_record_from_cert_reference` (used by the tests as the oracle) and make `record_from_cert` the fast one. Do not change the cert format, the stamp, the loaders, the manifest or any cache file.
Ideas, in the order I would try them (measure each; stop when it is fast enough; report what did not help): avoid the intermediate `D_canon`/`relabel_leaves_first` Sage graphs by computing the relabelling and the sorted edge list in pure Python from the unpacked edge list, then construct the final `D` and `G` once with `add_edges` / the `data=` constructors; compute degrees from the edge list instead of `D.degree(v)`;
check whether the graph constructors can skip validation (`immutable`, `sparse` backends, `data_structure`, `loops`/`multiedges` flags) WITHOUT changing the backend type or any observable attribute (compare `type(D._backend)`/behaviour if downstream code could depend on it; if unsure, keep the default).

## Acceptance (decided beforehand)
- **A (bit-exact):** for every shipped cell with at most 100,000 classes, and for a fixed-seed sample of 20,000 certificates from each larger shipped cell (read them with `load_shipped_certs`), the new and the reference record are identical: `D.vertices()` lists, `D.edges(labels=True, sort=False)` lists, `G.vertices()` and `G.edges(labels=True, sort=False)` lists, `leaves`, `internal` (equal as lists, same order), and graph-level flags (`D.allows_multiple_edges()`, `D.allows_loops()`, same for `G`). Also: for the cells (2,2), (3,1), (4,1) the typed-diagram pipeline (the same call `api._diagrams.enumerate_unique_diagrams` makes for k = 2 ell = 2, k = 3 ell = 1, k = 4 ell = 1 with a tracked model such as `ou_quartic`, `use_cache=True` from an empty working directory so the shipped cells are used) produces identical diagram counts and an identical sorted list of canonical keys before and after (compute both, with the reference swapped in by monkeypatch).
- **B (speed, report only):** wall time of `records_from_certs` for (2,3) and (4,2) from an empty directory, old vs new, 3 repeats; peak memory not worse than the reference (measure with `/usr/bin/time -v` or `resource`). Target: at least 2x; report what you reach.
- **C (tests)** in `tests/test_record_from_cert.py`: the bit-exact comparison on the small shipped cells (cells up to (2,2), (3,1), (4,1) and (5,0) or similar, runtime under 30 s), a malformed-cert error case unchanged from the reference, and a test that `record_from_cert` and `_record_from_cert_reference` stay identical on 500 random certificates of mixed cells.
- **D:** the default suite once; the known Linux baseline failures only (below), plus your new tests passing.
Report: Result; what you changed and what you tried; the A-D table with numbers; the (2,4) projection; open issues.

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
3. **Models.** Use only models in the repo (tracked): e.g. `ou_quartic`, `ou_sextic`, `ou_quartic_double_well`, `ou_quartic_colored`, `linear_hawkes`. Local-only models are NOT in the clone: build any extra reference inside your tests.
4. **Known Linux baseline, none of these is yours:**
   - Default suite, 5 failures: `test_phase_j_structural_zeros.py::{test_flag_off_restores_the_pre_m2a_fallback,
     test_flag_is_read_at_call_time, test_perdiag_cycle_through_the_fallback_fixes_the_overflow}` and
     `test_phase_j_nquad_hardening.py::{test_old_routine_samples_empty_inner_intervals,
     test_two_cycle_reaches_no_evaluation_without_structural_zeros}`.
   - Deselect `tests/test_spatial_correlator.py::test_periodic_inverse_ft_limits[3]` (OOM on 16 GB). The 23 `fastenum`
     tests skip.
   - **Do not run the slow suite** (the coordinator runs it on the Mac afterwards).
   - Run the default suite in chunks (background commands are capped at ~30 min; 7-8 chunks). About 60 s after each
     launch, read the log: tests must be collected and running.
   - Do not use `pkill -f` (kills your own shell). Do not run two jobs that build the same model cache at once.
5. **Warm the disk cache** (`saved_models/<model>/`) with one discarded run of a model before any A/B. A cold run and a
   warm run differ at ~1e-16 even with no change. Never compare across two checkouts for delta models: toggle the flag
   **in one process in one checkout** (the delta models show ~1e-14 cross-process jitter that is unrelated to this work).


   - One more failure is expected in this clone and is not yours: `tests/test_chain_uppers_memo.py::test_two_tau_end_to_end_array_equal` (MISSING_MODEL: a local-only model file).

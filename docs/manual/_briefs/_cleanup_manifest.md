# Cleanup manifest (June 2026)

Read-only survey of repo clutter. Nothing was moved or deleted — this is a recommendation list.
Repo root: `/Users/matthewszuromi/Documents/Education/BU PhD/Ocker Lab/Automated Feynman Calculations`

Legend — action: `delete` | `legacy` (move to legacy/) | `archive` (move to docs/archive/) | `keep` | `investigate`.

## TL;DR
- **scratch/ is NOT gitignored** and 28 files are tracked. Easiest clean fix: add `scratch/` to `.gitignore` and untrack it, OR delete the disposable probe scripts/logs and keep only the research PDFs + a couple of reusable generators. The freshly-untracked probes (`d_*.py`, `*.log`, `*.bak`) listed in `git status` are throwaway.
- **`scratch/borinsky_2008.pdf|2302.pdf|2310.pdf`** are the tropical-Monte-Carlo papers cited by `docs/spatial_efficiency_research.md` — research assets, KEEP (move out of scratch if you want them permanent).
- **2 genuinely-dead module stubs**: `msrjd/core/propagator.py` and `msrjd/integration/numerical.py` (both "Status: Not started/Not yet extracted", 0 imports).
- **pipeline_outputs/** is already gitignored (local-only clutter; safe to delete on disk).
- The oracle-only spatial modules (`loop_dyson`, `loop_parametric`, `generic_evaluator`, etc.) are deliberately-kept cross-checks — KEEP.
- The hardcoded `pipeline/theories/*.py` ARE still imported by `pipeline_demo.ipynb` + tests — KEEP.

---

## scratch/ — throwaway probes & logs (untracked, disposable)
| Path | Kind | Action | Reason |
|------|------|--------|--------|
| `scratch/d_audit.py`, `d_audit2.py`, `d_coupled_exact.py`, `d_coupled_validate.py`, `d_drift.py`, `d_dyson.py`, `d_dyson_fast.py`, `d_dyson_min.py`, `d_dyson_oracle.py`, `d_periodic.py` | scratch | delete | One-off Dyson/coupled-D probe scripts, untracked, superseded by committed integrator |
| `scratch/d_coupled_exact.py.bak2` | scratch | delete | `.bak` editor backup of a probe; already matched by `*.bak` ignore intent |
| `scratch/bound_check.py`, `orient_spotcheck.py`, `theta_counterexamples.py`, `_grouped_probe.py` | scratch | delete | Untracked spot-check probes from the enumeration-bound / orientation work |
| `scratch/*.log` (dyson_fast_full, dyson_min, dyson_oracle, exec_sweep, grouped_regression, k3_verify, out_k2_l2, out_k2_l2_n13, out_k2_l3, out_k3_l1_n12, out_k3_l2, out_k4_l1, out_mirror_k3_l2, out_theta_cex) | scratch | delete | Captured stdout from probe runs; several are 0-9 bytes; regenerable |
| `scratch/wave1_args.json`, `wave1a_args.json`, `wave1a_compact.json`, `wave2_compact.json` | scratch | delete | Untracked migration-wave arg dumps; `wave2_final.json` already tracked is the keeper |

## scratch/ — research assets & reusable tooling (keep, but move out of scratch)
| Path | Kind | Action | Reason |
|------|------|--------|--------|
| `scratch/borinsky_2008.pdf`, `borinsky_2302.pdf`, `borinsky_2310.pdf` | keep | keep | Tropical-MC papers (arXiv 2008.12310 / 2302.08955 / 2310...) cited in `docs/spatial_efficiency_research.md`; research provenance — consider moving to `docs/papers/` or `Literature/` |
| `scratch/borinsky_2008.txt`, `borinsky_2302.txt`, `borinsky_2310.txt` | scratch | investigate | OCR/text extracts of the PDFs; keep if used for grep, else delete (regenerable from PDF) |
| `scratch/gen_examples.py` | keep | keep | Referenced by `notebooks/examples/README.md`; generator for the 9 example notebooks |
| `scratch/gen_runner.py`, `gen_simcompare.py`, `gen_simcompare_spatial.py`, `gen_templates.py`, `gen_ex_*.py`, `repo_mirror.py` | scratch | investigate | Notebook/template generators + repo-mirror helper (tracked); reusable but belong in `scripts/` not scratch — keep if part of the notebook-build workflow |
| `scratch/mc.py`, `mc2.py`, `mcconf.py` | scratch | investigate | Monte-Carlo prototypes referenced by `docs/spatial_loop_integral_analytic_mc.md`; keep as research provenance or fold into that doc |
| `scratch/bessel_backend.py`, `bessel_backend_deriv.py`, `besselk_rayfit.py` | scratch | investigate | Bessel-backend prototypes (tracked); functionality landed in `heat_kernel.py` — likely superseded, verify before delete |
| `scratch/audit_builder_smoke.py`, `serializer_roundtrip_audit.py`, `comp_audit_k4.py`, `wick_count_k3_a3.py`, `audit_descriptors.json` | scratch | investigate | Tracked audit harnesses from serializer/combinatorial work; keep if rerun, else delete |
| `scratch/build_migrate_args.py`, `check_nb.py`, `exec_nb.py`, `migrate_args.json`, `migrate_workflow.js`, `wave2_final.json` | scratch | investigate | Tracked notebook-migration tooling; disposable once migration is done |

## Top-level stray files/dirs
| Path | Kind | Action | Reason |
|------|------|--------|--------|
| `pipeline_outputs/` (whole dir, incl. demo/, *.npz, *_report.pdf, .DS_Store) | generated-artifact | delete | Already gitignored (regenerable run outputs); safe to delete on disk to declutter |
| `scratch/` (as a directory) | stray | investigate | Not gitignored — recommend adding `scratch/` to `.gitignore` and untracking the 28 tracked files, so future probes never get committed |
| `.DS_Store` (9 on disk, repo-wide) | stray | delete | macOS cruft; already gitignored, not tracked — safe local delete |
| `.pytest_cache/` | generated-artifact | keep | Gitignored, not tracked; harmless |

## docs/manual/ — LaTeX build artifacts
| Path | Kind | Action | Reason |
|------|------|--------|--------|
| `docs/manual/daedalus_manual.aux`, `.log`, `.out`, `.toc` | generated-artifact | investigate | LaTeX intermediates (regenerable); NOT currently tracked or gitignored — add `*.aux *.log *.out *.toc` (scoped to docs/manual) to `.gitignore` to keep them out |

## DEAD/ORPHAN python modules
| Path | Kind | Action | Reason |
|------|------|--------|--------|
| `msrjd/core/propagator.py` | dead-module | delete | Docstring-only stub, "Status: Not yet extracted"; 0 imports anywhere (logic lives in `_propagator.py`) |
| `msrjd/integration/numerical.py` | dead-module | delete | Docstring-only stub, "Status: Not started"; 0 imports (the `numerical` grep hits are the English word) |
| `msrjd/integration/time_domain/subgraph.py` | dead-module | investigate | `identify_loop_subgraphs` returns `[]` / raises NotImplementedError; still re-exported by `time_domain/__init__.py` and touched by `test_time_domain.py` — dead body but a live import surface, audit before removing |
| `msrjd/integration/spatial/generic_evaluator.py` | dead-module | keep | Banner-annotated ORACLE-ONLY cross-check; reached only by `test_generic_evaluator.py`; deliberately kept per project memory |
| `msrjd/integration/spatial/loop_dyson.py`, `loop_parametric.py`, `spatial_reduce.py`, `temporal_integrate.py` | dead-module | keep | Oracle-only / cross-check cluster (banner-annotated, mutually tested); `compute_cumulants` does not use them but they are intentional numerical oracles |
| `pipeline/theories/linear_hawkes_2pop_*.py`, `quad_hawkes_2pop_*.py` (5 files) | keep | keep | Still imported by `pipeline_demo.ipynb`, 3+ temporal sim-compare notebooks, and `pipeline/tests/test_theory_equivalence.py` — NOT superseded by `theories/*.theory.py` |
| `pipeline/examples/run_linear_gtas.py` | orphan | investigate | Example entry-point script; verify it still runs against current API, else legacy |

## docs/ — superseded / overlapping design docs
| Path | Kind | Action | Reason |
|------|------|--------|--------|
| `docs/spatial_kpz_burgers_plan.md`, `spatial_analytic_ift_plan.md`, `spatial_loop_integral_analytic_mc.md`, `spatial_reduction_derivation.md`, `spatial_d_ge_2.md` | superseded-doc | archive | "Plan"/"design" docs whose status is now DONE; `docs/spatial_pipeline.md` is the authoritative reference — move to `docs/archive/spatial/` alongside the others |
| `docs/spatial_v2_architecture.md` | superseded-doc | investigate | Forward roadmap (Phase 1 landed, 4c/4d open) — keep while v2 in progress; archive once v2 lands |
| `docs/spatial_efficiency_research.md` | keep | keep | Active efficiency roadmap (cites the borinsky PDFs); research-only but current |
| `docs/spatial_pipeline.md` | keep | keep | Authoritative current spatial reference |
| `docs/m_ge3_chain_simplex_fix_proposal.md` | superseded-doc | investigate | "fix_proposal" companion to `m_ge3_precision_bug_audit.md`; if the proposal is implemented or stale, archive |
| `docs/phase_j_refactor_notes.md`, `docs/theory_builder_split_plan.md`, `docs/dyson_duhamel_integration_plan.md`, `docs/backend_C_design.md`, `docs/backend_C_math.md`, `docs/conductance_vertex_kernels_design.md` | superseded-doc | investigate | Plan/design/notes docs for completed work; archive the ones whose feature has shipped (verify each against current state) |
| `docs/PIPELINE_PLAN.md`, `docs/BUILD_PHASE_OUTLINES.md` | superseded-doc | investigate | Early whole-pipeline build plans (May); likely historical — archive if no longer the working plan |
| `docs/pipeline_cleanup_audit.md` | keep | keep | The June code-level dead-code audit; current and useful |
| `docs/kpz_burgers_sim_validation.py` | stray | keep | A `.py` sitting in docs/, but referenced by `test_kpz_burgers_sim.py` and 2 notebooks — keep (consider relocating to a validation/ dir) |
| `docs/archive/spatial/` (+ spikes/) | keep | keep | Already-archived provenance with a README explaining each; leave as-is |

## Notebooks & existing legacy
| Path | Kind | Action | Reason |
|------|------|--------|--------|
| `notebooks/legacy/` | keep | keep | Already segregated legacy notebooks; note only |
| `legacy/` (31 tracked files: Enumeration Code, Field Theory Framework, *_sympy.py/.ipynb, Theory Builder) | keep | keep | The pre-pipeline sympy-era codebase, already isolated as legacy; note only |
| `notebooks/saved_results/`, `notebooks/saved_theories/` | generated-artifact | keep | Already gitignored cache dirs; regenerable, harmless |

## Already-clean (no action)
- `.gitignore` already covers `__pycache__`, `*.sobj`, `*.bak`, `*.prof`, `saved_theories/`, `pipeline_outputs/`, `Literature/`, `notebooks/*.pdf`, `.DS_Store`, `.claude/`.
- No stray `*.sobj` / `*.prof` / `*.bak` outside scratch; no tracked `.DS_Store`/`.pytest_cache`.

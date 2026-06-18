# docs/ Freshness Manifest

**Generated:** 2026-06-18 · branch `spatial-extension`
**Cross-checked against:** `docs/spatial_pipeline.md` (authoritative spatial state),
`docs/temporal_lessons_from_spatial.md` (temporal reverse-transfer audit), `docs/CHANGELOG.md`.

**Recent code facts used as the ground truth:**
- `Config.fundamental` → renamed `Config.parameters` (`notebooks/daedalus.py:259-319`); `fundamental`
  kept as a **deprecated alias** (engine dict internally still named `fundamental`). The `_briefs`
  already document this correctly.
- New white-noise theory `theories/ou_quartic.theory.py` exists (alongside `ou_quartic_colored`,
  `ou_quartic_double_well`, etc.).
- `Config.output ∈ {'cumulant','moment','central_moment'}` (`daedalus.py:261-266`,
  `_assemble_moment_*`).
- Fork-safety guards landed (temporal `a141fdd`, spatial serial-only `fd3af81` / smart thread-gate
  `d8246c3`).

Status legend: **current** (accurate, stays top-level) · **superseded** (rolled into a newer doc —
named) · **plan-archive** (historical plan/outline → `docs/archive`) · **stale** (out of date vs
current code).

---

## Authoritative / current references (keep top-level)

| Doc | Status | Notes |
|---|---|---|
| `spatial_pipeline.md` | **current** | The authoritative spatial-state reference. Matches code (full_integrator, analytic IFT, MC/Bessel backends, k-general, coupled Dyson). |
| `temporal_lessons_from_spatial.md` | **current** | Authoritative reverse-transfer audit; fork-guard `a141fdd` documented. |
| `CHANGELOG.md` | **current** | Chronological fix log; actively maintained (latest 2026-06-01). |
| `related_work.md` | **current** | June 2026 novelty/literature map; matches model class + backends. |
| `future_directions.md` | **current** | v2 roadmap (May 2026), forward-looking; "current state v1" table still accurate. |
| `correlated_noise_capabilities.md` | **current** | `declare_cgf_term`/`correlated_noise` still in code; capability matrix accurate. |
| `spatiotemporal_convolutions.md` | **current** | Form-factor survey, June 2026; consistent with operator-IR vertices. |

### Spatial supporting docs cross-referenced as *current companions* by `spatial_pipeline.md`

| Doc | Status | Notes |
|---|---|---|
| `spatial_reduction_derivation.md` | **current** | The full reduction chain; cited by spatial_pipeline.md as the derivation companion. |
| `spatial_loop_integral_analytic_mc.md` | **current** | Analytic-vs-MC + Bessel-K §3; cited as the backend-math companion. |
| `spatial_v2_architecture.md` | **current** | Forward momentum-native rearch; cited as the v2 roadmap (Phases 1-3 + 4a/4b landed). Supersedes archived `spatial_stageC5_general_integrator_design.md` (already archived). |
| `spatial_d_ge_2.md` | **current** | Header marked DONE+VALIDATED June 2026; matches the d∈{1,2,3} backend matrix in spatial_pipeline.md. |
| `spatial_efficiency_research.md` | **current** | Tropical-MC efficiency roadmap (research only, not implemented) — matches MEMORY roadmap entry. |
| `spatial_kpz_burgers_plan.md` | **current** | Header marked DONE end-to-end; serves as the KPZ/Burgers validation record. Plan-ish title but body is a status/validation doc. |
| `backend_C_design.md` | **current** | Forward design for the arbitrary-(L,d) Schwinger backend; explicitly the long-term target named by spatial_v2_architecture §5. Not yet built — forward design, not stale. |
| `backend_C_math.md` | **current** | Math foundation companion to backend_C_design. Forward research, not stale. |

### Resolved-bug audits (keep as durable records; self-mark RESOLVED)

| Doc | Status | Notes |
|---|---|---|
| `m_ge3_precision_bug_audit.md` | **current** | Self-marked RESOLVED 2026-05-28 with regression fixture; durable evidence record (documents flag-gated dead paths). Keep. |
| `ou_quartic_large_mu_phase_j_audit.md` | **current** | Diagnosed audit + workaround; matches MEMORY `ou_quartic_large_mu` note. Keep. |
| `scalar_mode_field_coercion_fix.md` | **current** | Fix record (May 2026), matches MEMORY scalar-mode note. Keep. |
| `pipeline_cleanup_audit.md` | **current** | June 2026 cleanup audit; "Applied" commit 5565192 + deferred suggestions. Living TODO record. Keep. |

---

## Plan-archive (completed plans/outlines → move to `docs/archive`)

| Doc | Status | Move to | Why |
|---|---|---|---|
| `BUILD_PHASE_OUTLINES.md` | **plan-archive** | `docs/archive/` | "Phases A–J complete" (Apr 2026). Detailed pre-implementation outlines; the build is done and the chronology lives in CHANGELOG. Historical plan. |
| `PIPELINE_PLAN.md` | **plan-archive** | `docs/archive/` | Apr 2026 architecture+build plan, "Phases A–J complete." Superseded as a *current* description by CHANGELOG + the manual; valuable as provenance. |
| `phase_j_refactor_notes.md` | **plan-archive** | `docs/archive/` | Stage 3/4a refactor notes leading to a tagged checkpoint; work landed, captured in CHANGELOG + grouped-phasej brief. |
| `m_ge3_chain_simplex_fix_proposal.md` | **plan-archive** | `docs/archive/` | "Design proposal, no code changes yet" for a bug its own companion audit now marks **RESOLVED** (different fix). Historical proposal — never executed as written. |
| `spatial_analytic_ift_plan.md` | **plan-archive** | `docs/archive/` | Design+plan for the analytic IFT, which is **DONE** (spatial_pipeline.md "Real-space output" §, commits 0ca5a78/0f1ee48). Plan fully executed → archive like the other completed spatial plans. |
| `theory_builder_split_plan.md` | **plan-archive** | `docs/archive/` | "Step 1 + serializer Phase-1 DONE"; a build plan whose executed steps are in CHANGELOG. Remaining steps are minor; an outline doc, not current-state. |

### Borderline (lean keep, flagged for the doc owner)

| Doc | Status | Note |
|---|---|---|
| `dyson_duhamel_integration_plan.md` | **investigate** | Titled "plan" but the body is a status table — most steps DONE, one item (loop-level dressing) **GATED** with the implementation route recorded inline. Because the gated route is live design info not captured elsewhere, lean **keep** until loop dressing lands, then archive. Cross-ref: MEMORY `project_coupled_dyson`. |
| `conductance_vertex_kernels_design.md` | **investigate** | "Design — implementation in progress", branch `convolution-operator` (branch still exists). Not on `spatial-extension`. Status unverified against that branch's code; keep but flag as stale-risk if convolution-operator was abandoned/merged. |

---

## Manual (`docs/manual/`) — in-progress, keep

| Item | Status | Note |
|---|---|---|
| `daedalus_manual.tex` | **current** | Master LaTeX file; in active authoring. |
| `sections/00-how-to-read.tex`, `01-overview.tex` | **current** | Filled chapters. |
| `sections/02..21, A..D *.tex` | **keep (stub)** | 24 `(pending: …)` stubs — placeholders, not stale content. In-progress scaffolding. |
| `_briefs/*.md` (18 files) | **current** | Source material for the manual; verified to already reflect the `parameters` rename, `output=` modes, and current backends. The authoring corpus — keep. |
| `daedalus_manual.{aux,log,out,toc}` | **keep (build artifact)** | LaTeX build outputs; regenerated on compile. Not docs. Could be gitignored but harmless. |

---

## Static assets / reference material (keep)

| Item | Status | Note |
|---|---|---|
| `algorithm_flow/*` (.drawio/.pdf/.png) | **current** | Algorithm flowchart source+exports. Reference asset. |
| `figures/test_runs/*.png` | **keep** | Validation-run screenshots; referenced from notebooks/docs. Asset. |
| `papers/*.pdf` (FoliasI2026, helias_ch9) | **keep** | External reference papers. |
| `walkthrough/Notebook_Walkthrough_v1.docx` | **keep** | User walkthrough doc (Word). |
| `*.pdf`/`*.docx` at docs root (Contributing Diagrams, Feynman graph counting, General systems, Notes for Feynman Paper) | **keep** | Source/reference material (math notes, paper drafts). Not code-derived; freshness N/A. |
| `kpz_burgers_sim_validation.py` | **keep** | Validation script (not a doc per se); referenced by the KPZ/Burgers work. Belongs with docs as a runnable check. |

---

## `docs/archive/spatial/` — already archived (keep as-is)

All 9 `.md` + 12 `spikes/*.py` are already under `docs/archive/spatial/` with a README that
correctly labels them **superseded** and points to `spatial_pipeline.md`. No action — they are
already in their archive home. Listed here for completeness:
`README.md`, `spatial_design_decisions_v1.md`, `spatial_generic_pipeline_plan.md`,
`spatial_implementation_outline.md`, `spatial_implementation_plan.md`,
`spatial_loop_diagram_inventory.md`, `spatial_phase5_rearchitecture_plan.md`,
`spatial_pre_C0_audit.md`, `spatial_stageC5_general_integrator_design.md`, `spikes/*.py`.

---

## No `stale` content found

No top-level doc was found to contradict current code (the `Config.parameters` rename, the new
`ou_quartic` white theory, `output=` modes, and fork guards are all either reflected or not yet
referenced — none are *wrongly* described). The only freshness risk flagged is
`conductance_vertex_kernels_design.md`, whose `convolution-operator` branch status is unverified.

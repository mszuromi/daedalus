# Holistic Completeness Critic — Daedalus Manual

_What does the codebase contain that the manual does NOT adequately cover?_
Pass run 2026-06-18 against the real source tree (`msrjd/`, `pipeline/`,
`models/`, `theories/`, `notebooks/`). Method: read `C-file-index.tex` +
the chapter list in `daedalus_manual.tex`, enumerate every source file, then
grep each chapter for the capability/identifier (accounting for LaTeX
`\_` underscore-escaping, which silently defeats naïve greps).

## Headline verdict

The manual is **substantially complete at the file and capability level.**
The Appendix-C file index maps *every* non-test source file to a chapter, and
the 21 chapters are written to a deliberate, uniform depth budget
(~900–1500 source lines each). Most "is X covered?" hypotheses were
**disproven on close reading**: `describe_model`, `list_theories`, the routing
predicates, `summary`, the `output='moment'` assembly, the k≥3 synthesis, all
three spatial backends (`grid`/`mc`/`bessel`), the `Conv` operator, and all
nine `SPATIAL_*` env knobs each have a real home. The naïve grep undercounts
because LaTeX escapes underscores; the escaped grep finds them.

The surviving gaps are therefore **few and specific** — and they cluster in
two places: (1) the *getting-started / install* on-ramp, and (2) the
*result-output* tail (save + report), which the file index promises to a
chapter that does not actually deliver it.

---

## Findings (verified, high-confidence)

### 1. [MAJOR] No installation / getting-started procedure
**Where:** missing chapter; partially orphaned between `00-how-to-read.tex`
(§111–115) and `B-external-tools.tex` (§18).

The manual's stated audience "may have little experience with the specific
software tooling … SageMath, nauty, sympy, numba." Yet there is **no procedure
for standing the engine up**:
- `00-how-to-read.tex:111-115` correctly *orients* — "you need SageMath, launch
  with `sage -python`, use the Sage Jupyter kernel, see Appendix B" — but gives
  no steps.
- `B-external-tools.tex:18` explicitly punts: *"You do not install any of these
  by hand … They all [ship with Sage]."* So it assumes Sage is already present.
- Quickstart (`02-quickstart.tex`) opens at `import daedalus` / `load_theory`,
  i.e. *after* a working environment.
- **Nothing** documents: how to obtain SageMath; how to clone the repo; the
  `MSRJD_diagrams` conda environment (named only incidentally at
  `10-type-assignment.tex:1212` and `B-external-tools.tex:20`); a
  `requirements.txt`/`environment.yml`; or a first-run smoke test.
- A documented footgun is omitted entirely: the base miniforge numpy is broken
  (libgfortran rpath) — a new user who installs into the wrong env hits it
  immediately. (Tracked in the project's own memory, not the manual.)

**Why it matters:** this is the single largest *new-user* gap. Everything
downstream assumes a green environment; the manual never tells you how to get
one or how to confirm you have one.

**Suggested fix:** add a short "Chapter 0 / Getting Started" (or a §0 in the
Quickstart, or a dedicated Appendix): obtain Sage → clone → create/activate the
`MSRJD_diagrams` env → the numpy-rpath caveat → a 3-line smoke test
(`import daedalus as dd; dd.list_theories()` then a sub-second
`ou_quartic` k=2 run) that prints an expected number.

### 2. [MAJOR] Result output: `report.py` is orphaned; `save.py` schema undocumented
**Where:** `C-file-index.tex` points both `pipeline/report.py` and
`pipeline/save.py` at Ch.19 (`ch:cumulants-moments`); Ch.19 covers **neither**.

- `pipeline/report.py` → `generate_report` is a **public exported symbol**
  (`pipeline/__init__.py:52` re-exports it: `from pipeline import
  generate_report`). It is the multi-page-PDF result visualizer (prediagrams →
  typed-diagram assignments → per-diagram numerical contributions). It appears
  **only** in the file index — `grep -ri report` across the 21 chapters finds no
  walkthrough; Ch.19's lone "report" hit is unrelated. A user who wants the
  built-in PDF output has no documented entry point.
- `pipeline/save.py` → `save_npz`/`save_csv` (also public exports,
  `__init__.py:53`) get one passing mention in the orchestration chapter
  (`17-compute-orchestration.tex:1083`: "if `output_npz`/`output_csv` was given
  … that module owns the schema"). But the **schema itself** — the
  k/ℓ/model-adaptive array names (`C_tree`, `C_<n>_loop`, `C_total`, plus the
  mean-field arrays) — is documented **nowhere**. The file index promises Ch.19;
  Ch.19 does not deliver.

**Why it matters:** "how do I export / save / report my results" is a first-week
question with a public API and no documented home. The cross-references
(index → Ch.19) are also *wrong*, which erodes trust in the index.

**Suggested fix:** either (a) add a short "Saving and reporting results"
section to Ch.19 (it is the natural home and the index already points there) or
to Ch.17, covering `save_npz`/`save_csv` (with the actual array schema) and
`generate_report`; and (b) correct the Ch. column in `C-file-index.tex` for
`report.py`/`save.py` to wherever that section lands.

### 3. [MINOR] Depth-vs-importance: temporal Phase-J core under-budgeted
**Where:** Ch.12 (`12-phasej-temporal.tex`, 1141 lines) covers
`final_integral.py` (**5269 lines — by far the largest module in the repo**).

The chapters are uniform (~1.1k lines) regardless of subsystem size, so the
single biggest, most numerically load-bearing module gets the same budget as the
1028-line reference `symbolic.py`. The closed-form mode-sum machinery
(`_integrate_1d_polytope_modesum`, `_integrate_2d_polygon_modesum`,
`_integrate_nd_polytope_poset_modesum`, `EdgeModeSum`, the pole-tuple
enumerator) is named in the index but the chapter cannot do 5k lines justice at
1.1k. By contrast the authoring surface is similar: `theory.py` (2126) +
`theory_compiler.py` (1944) ≈ 4070 lines of "what can I type" share Ch.3
(936 lines) — thin for the most user-facing reference in the manual.

**Why it matters:** not a *coverage* gap (the index maps them) but a *depth*
gap — when the answer is "wrong and I suspect the loop integral," Ch.12 is where
the index sends you, and it is proportionally the thinnest deep-dive.

**Suggested fix:** flag both as candidates for expansion (or for an explicit
"this chapter is a map of a very large module; the authoritative reference is
the source + docstrings" disclaimer). Lower priority than #1/#2.

### 4. [NIT] No consolidated knobs/env-var reference
**Where:** cross-cutting; the nine `SPATIAL_*` env vars + the `compute_cumulants`
keyword knobs are documented **scattered** across Ch.14/15/16/17.

All nine knobs ARE covered (verified with underscore-escaped grep:
`SPATIAL_INTEGRATOR`, `SPATIAL_GRID_NT/NS`, `SPATIAL_MC_N`, `SPATIAL_N_Q`,
`SPATIAL_Q_CUT`, `SPATIAL_FORCE_NUMERICAL_FT`, `SPATIAL_MEM_BUDGET_GB`,
`SPATIAL_DYSON_ORDER` — Ch.16 names all of them). But there is no single table a
user can scan when a run is slow or OOMs. Same for `compute_cumulants`'s
performance/control kwargs (`spatial_parallel`, `spatial_n_q`, `use_cache`,
`use_grouped_phase_j`, `mf_dae_*`).

**Suggested fix:** a one-page "Tuning & environment knobs" reference table
(name | default | effect | which chapter) — pure discoverability, no new prose.

---

## Things checked and found ADEQUATELY covered (so they are NOT findings)
- Convenience/introspection verbs (`describe_model` §374, `list_theories` §293,
  routing predicates §354, `summary` §897) — all in Ch.18.
- `output='moment'` / cumulant→moment set-partition assembly — Ch.18 §715 +
  whole of Ch.19.
- k≥3 slice/grid synthesis — Ch.18 §660 + Ch.19 §538.
- The three spatial backends `grid`/`mc`/`bessel` incl. the MC-bias-for-
  derivative-vertices gotcha — Ch.14 §"The three backends" + Ch.15.
- All nine `SPATIAL_*` env knobs — Ch.14/15/16 (Ch.16 is the most complete).
- `Conv`/`ConvVertexType` operator reduction — Ch.3, Ch.6.
- `.theory.py` author contract (`build`/`DEFAULT_FUNDAMENTAL`/`METADATA`) —
  Ch.4, Ch.3.
- Oracle-only spatial modules + the two docstring stubs
  (`msrjd/core/propagator.py`, `integration/numerical.py`) + the `subgraph.py`
  near-stub — all correctly flagged "—" / "Stub only" in the index.
- `Literature/` (PDFs), `scripts/profile_pipeline_k3.py` — peripheral, not
  subsystems; reasonable to omit/footnote.

## Note on the index's own cross-references
The `report.py`/`save.py` → Ch.19 pointers in `C-file-index.tex` are the only
*incorrect* chapter cross-references found (the target chapter doesn't cover
them). Worth fixing alongside finding #2.

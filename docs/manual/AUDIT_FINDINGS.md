# Daedalus Manual — Audit Findings & Code Inconsistencies

_Working log of things found while documenting the codebase, for us to go over together. Three kinds of entry: **(A)** likely code inconsistencies / dead code worth a decision, **(B)** manual-vs-code accuracy fixes (already applied to the chapters), **(C)** accepted limitations that are documented, not bugs._

**Status (2026-06-18, FINAL):** Audit complete — all 20 per-chapter accuracy audits + 3 holistic audits ran. **All 7 major accuracy findings are fixed** (Part B); the manual compiles to **406 pp, 0 errors, 0 undefined references**. The ~60 remaining minor/nit findings are catalogued per chapter in `docs/manual/_briefs/_audit_<slug>.md` (per-chapter verdicts in Part E) and were left for review rather than each individually patched. Code-side cleanup items are in Part A; holistic findings in Part F.

---

## Part A — Likely code inconsistencies / dead code (for review, NOT fixed)

These came out of the read-only subsystem deep-reads. None is necessarily a bug; each is a "should this be cleaned up / is this intentional?" item.

### Engine API (`notebooks/daedalus.py`)
1. **`Config.components` looks dead/unwired.** Declared at `daedalus.py:307` and advertised in the module docstring (`:24-25`) as honored by plotting, but no plotter in the file ever reads it. Either wire it or drop it from the docstring + field.
2. **`Config.is_multifield` dispatch is incomplete.** The run/plot docstring (`:23-25`) says `plot_cumulant` dispatches on `(spatial?, multi-field?, k)`, but `plot_cumulant` only branches on spatial/k — the multi-field axis is never consulted in this file.
3. **`METADATA['spatial_grid']` is ignored by `run()`** for the spatial k=2 path (`daedalus.py:574-577` defaults to `np.linspace(-6,6,49)`). A theory's recommended grid is effectively dropped unless the notebook passes `spatial_grid=` explicitly. Confirm intentional.
4. **Unused import:** `from dataclasses import dataclass, field` (`:31`) — `field` is never used. Harmless.
5. **Silent fallback in the k≥3 slice/grid synthesis:** wrapped in a bare `try/except Exception: pass` (`:632`, `:671-672`) that discards any `total_C` evaluation failure, silently leaving the result in scalar form. Consider surfacing a warning.
6. **`plot_kpoint` (spatial k≥3, `:1054`) has no `sim=` parameter** unlike every other plotter, so a simulator overlay for spatial k≥3 is unreachable through `plot_cumulant`. Confirm spatial k≥3 sim-compare is intentionally unsupported.

### Theory specification (`pipeline/theory.py`, `theory_compiler.py`)
7. **`theory.py` module docstring (`:1-62`) is stale.** It documents an older API surface (set_action lambda-first, `transfer_function`, `use_action_template` as a roadmap item, a YAML loader) that no longer matches the implemented text-compilation + template flow. Misleading — flag for rewrite.
8. **Dead parameter:** `make_specializations_lambda` still accepts `mf_eqs`, which its own comment (`theory_compiler.py:1091`) says is unused/retained for API compat.
9. **No-op branch:** `_parse_response_legs` (`theory_compiler.py:1707`) has `legs = parts if len(parts) > 1 else parts` — both arms identical. Copy-paste leftover (perhaps meant `parts[:1]`?).
10. **Fragile duplication:** the "operator·saddle → 0" kill rule is mirrored in two places (`make_action_lambda` and `make_mf_bg_conditions_lambda`) with an explicit warning that a divergence silently breaks MF convergence. Factor into one helper.
11. **Asymmetry:** `set_kernel_ft_image` exists but there is no `set_kernel_td_image` lambda-setter, although `build()`/the compiler emit a `kernel_td_image` hook.

### Theory files / serialization (`pipeline/theory_serialize.py`)
12. **Dead/confusing code:** `_emit_response_field` (`:103`) builds its argument string with a `_kw_chain(('', None))` placeholder that always yields the empty string, then concatenates a second hand-built string.

### Dead / orphan modules (from the cleanup manifest)
13. **`msrjd/core/propagator.py`** — docstring-only stub ("Status: Not yet extracted"), **0 imports** (real logic is `pipeline/_propagator.py`). Delete candidate.
14. **`msrjd/integration/numerical.py`** — docstring-only stub ("Status: Not started"), **0 imports**. Delete candidate.
15. **`msrjd/integration/time_domain/subgraph.py`** — `identify_loop_subgraphs` returns `[]` / `NotImplementedError`, but is still re-exported by `time_domain/__init__.py` and touched by `test_time_domain.py`. Dead body, live import surface — audit before removing.
16. **`pipeline/examples/run_linear_gtas.py`** — example entry-point; verify it still runs against the current API, else move to legacy.

---

## Part B — Manual-vs-code accuracy fixes (APPLIED to the chapters)

**All 7 major accuracy findings are fixed.** Every fix was re-verified against the code before editing.

- **[major, FIXED] Ch.3 Fourier sign.** Chapter said `fourier_transform` computes `∫g(t)e^{+iωt}dt`; code uses **e^{−iωt}** (`msrjd/core/field_theory.py:35-39`), consistent with `1/(1+iωτ_g)`. Corrected.
- **[major, FIXED] Ch.6 wrong consumer.** Said the enumerator `degree_scan.py` "reads `available_degrees`"; it is actually imported/called only in `msrjd/diagrams/filter.py:60-61`. Reworded to name the prediagram **filter**.
- **[major, FIXED] Ch.6 Conv over-claim.** Said `models/hawkes_quad_expg.py` "exercises every feature" incl. the `Conv` operator; that model uses plain multiplication (no `Conv` atom). Reworded: `Conv` is exercised by the conductance theory files.
- **[major, FIXED] Ch.8 `_solve_mf_at_saddle`.** Said it "re-runs the solver to verify the linear term vanishes"; it only returns the saddle-values dict (`_precompute.py:200`). The linear-term check is `sanity_check` at Stage 2 (`_precompute.py:141`). Corrected.
- **[major, FIXED] Ch.9 fabricated attribution (×2).** Topology counts 9/67/289 were attributed to "the code comment at `loop_diagram_enumeration.py:138`"; that comment records the slack-verified **orders** `{(2,1),(3,1),(2,2),(3,2),(4,1),(2,3)}`, not counts (9/67/289 appear nowhere in the file). Reworded both spots.
- **[major, FIXED] Ch.9 broken `\ref{gotcha:…}`.** The `gotcha` box was a counterless `tcolorbox`, so its `\label`/`\ref` resolved to the wrong (enclosing) counter. Added `[auto counter, number within=chapter]` to the `gotcha` box in the preamble → all 7 gotcha refs now resolve (e.g. "Gotcha 9.3"), 0 undefined refs.
- **[major, FIXED] Ch.10 non-existent tests (×2).** Cited `test_leg_matchings_canonical` and `test_enumerate_typed_signatures_match_pre_change` — neither exists (copied from a **stale source docstring**, `type_assignment.py:380-381`). Replaced with the real `test_leg_matchings_*` family + `test_enumerate_typed_distinct_legs_regression`. *(The source docstring is itself stale — a code-side cleanup, logged in Part A.)*
- **[major, FIXED] Ch.16 1500-line-off citation.** Cited `pipeline_bridge.py:509` for the "coincide at k=2" comment; line 509 is unrelated (a Dyson env-var read). Real comment is at `:2007`. Corrected.
- **[minor, FIXED] Ch.3 lambda count.** "six companion lambdas" → **seven** (incl. `mf_substitutions()`). Corrected.
- **[holistic, ADDED] Getting-started on-ramp.** The completeness critic flagged no install/first-run guidance. Added a "Getting the engine running" section to Ch.0 (obtain Sage → make repo importable → `MSRJD_diagrams` env + `sage -python` → broken-base-numpy caveat → a 1-line smoke test).
- **[holistic, ADDED] Ch.12 contour clause.** The readability critic flagged the contour-closing derivation skipped *why* it closes upward. Added a Jordan's-lemma sentence.
- **[fixed during compile] Appendix A glossary** bracket bug (a `\item[$f[\dots]$]` prematurely closing the optional arg) — brace-wrapped.
- **[accurate] Ch.5, Ch.12, Ch.15, Ch.18** — audit verdict accurate, no corrections.

---

## Part C — Accepted limitations (documented, not bugs)

These are known and tracked elsewhere; listed so we don't re-litigate them as "findings".

- **m≥3 chain-simplex precision** on close-paired poles (spike-reset k≥2, ℓ≥1). `docs/m_ge3_precision_bug_audit.md`.
- **μ = 1/τc exact degeneracy** (double pole) not handled by the v1 Markovian embedding; keep μτc ≠ 1.
- **Colored 2-loop (max_ell=2) is pathologically slow** — 66 embedded diagrams × per-τ `nquad`. NOT a regression (predates recent work). Use the white `ou_quartic` theory for fast 2-loop. (This is the issue that prompted the white-noise example.)
- **OU-quartic Phase-J blow-up at large |μ|, ℓ≥1.** `docs/ou_quartic_large_mu_phase_j_audit.md`.
- **Spatial d≥2 derivative-vertex transverse handling** is still numerical (analytic IFT Phase 3 remains).
- **Spatial coupled fields with unequal D:** only the tree correlator (via Dyson dressing) is wired; loop corrections are gated.
- **Spatial k≥3 loop records** need reduced chamber quadrature; `compute_cumulants` public API is still k=2-gated for the loop path.

---

## Part E — Per-chapter audit results (verdicts + finding counts)

Full per-chapter findings (including all minors/nits) are in `docs/manual/_briefs/_audit_<slug>.md`. Summary:

| Chapter | Verdict | Findings | Majors (all fixed) |
|---|---|---|---|
| 02 quickstart | minor | 3 | — |
| 03 theory-spec | minor | (fixed) | Fourier sign |
| 04 theory-files | minor | 3 | — |
| 05 colored-markovian | accurate | 0 | — |
| 06 fieldtheory-core | major | 6 | wrong consumer; Conv over-claim |
| 07 propagator | minor | 5 | — |
| 08 mean-field | minor | 3 | `_solve_mf_at_saddle` |
| 09 enumeration | minor | 6 | fabricated attribution; gotcha refs |
| 10 type-assignment | minor | 4 | non-existent tests |
| 11 caching | minor | 3 | — |
| 12 phasej-temporal | accurate | 3 (nits) | — |
| 13 grouped-phasej | minor | 4 | — |
| 14 spatial-core | minor | 3 | — |
| 15 spatial-heatkernel | accurate | 3 | — |
| 16 spatial-coupled | minor | 4 | :509→:2007 citation |
| 17 compute-orchestration | minor | 3 | — |
| 18 engine-api | accurate | 1 | — |
| 19 cumulants-moments | minor | 2 | — |
| 20 simulators | minor | 3 | — |
| 21 ui | minor | 7 | — |

The ~60 minor/nit items (slightly-off line citations, mild overstatements, notation nits) are **not yet individually patched** — they are catalogued in the `_audit_*.md` files for a follow-up sweep if you want one.

---

## Part F — Holistic audit findings

- **Completeness:** the manual is substantially complete at the file/capability level (Appendix C maps every non-test source file to a chapter). The one **major** gap — no install/getting-started on-ramp — is now **fixed** (Part B). Minor: Appendix C promises a `save`/`report` output-tail chapter that the body only lightly delivers — noted for review.
- **Consistency:** **clean.** `𝒮(Γ)` (not `M(Γ)`) used as the symmetry factor everywhere; propagator/pole/residue naming uniform; Phase-J gate-flag names match across Ch.12/13. Nits only: `Aut_{fixed-ext}` hyphenation drift, and the symmetry-factor numerator abbreviated two equivalent ways. (Cosmetic; not changed.)
- **Readability (novice lens):** strong overall — nauty, SymPy, numba, einsum, fork-vs-spawn all introduced from scratch as promised. One **major** gap (Ch.12 contour-closing skipped *why* it closes upward) is now **fixed** (Part B). Other minor "term used a few lines before its definition" spots are listed in `_briefs/_holistic_readability.md`.

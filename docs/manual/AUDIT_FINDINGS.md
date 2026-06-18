# Daedalus Manual — Audit Findings & Code Inconsistencies

_Working log of things found while documenting the codebase, for us to go over together. Three kinds of entry: **(A)** likely code inconsistencies / dead code worth a decision, **(B)** manual-vs-code accuracy fixes (already applied to the chapters), **(C)** accepted limitations that are documented, not bugs._

**Status (2026-06-18):** Preliminary. The deep-read briefs are complete and seeded Part A below. Of the 19 per-chapter accuracy audits, **2 landed before the session limit** (theory-spec, colored-markovian); the remaining 17 + the holistic final audit will run when the agent quota resets (2:40pm ET) and their findings get appended. Nothing here is auto-fixed except where Part B says "applied".

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

From the 2 per-chapter audits that completed:

- **[major, FIXED] Ch.3 Fourier sign.** Chapter said `fourier_transform` computes `∫g(t)e^{+iωt}dt`; the code uses the **e^{−iωt}** convention (`msrjd/core/field_theory.py:35-39`), consistent with the exponential-synapse image `1/(1+iωτ_g)`. Corrected in `03-theory-spec.tex`.
- **[minor, FIXED] Ch.3 lambda count.** "six companion lambdas" → **seven** (`use_action_template` also pulls `mf_substitutions()`); `HawkesAction` method list now includes `mf_substitutions()`. Corrected in `03-theory-spec.tex`.
- **[accurate] Ch.5 colored→Markovian.** Audit verdict: accurate, no corrections.
- **[fixed during compile] Appendix A glossary** had a `\item[...]` whose bracketed math `$f[x_0,\dots,x_n]$` prematurely closed the optional argument (23 cascading LaTeX errors). Brace-wrapped; manual now compiles to 0 errors.

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

## Part D — Pending (to append after the 2:40pm quota reset)

- Per-chapter accuracy audits NOT yet run: 02-quickstart, 04-theory-files, 06-fieldtheory-core, 07-propagator, 08-mean-field, 09-enumeration, 10-type-assignment, 11-caching, 12-phasej-temporal, 13-grouped-phasej, 14-spatial-core, 15-spatial-heatkernel, 16-spatial-coupled, 17-compute-orchestration, 18-engine-api, 19-cumulants-moments, 20-simulators, 21-ui.
- The big **holistic final audit** (manual read end-to-end vs the code + a completeness critic).
- Their findings will be appended to Parts A/B above, and any **critical/major** manual errors fixed in the chapters before the final commit.

# Audit: `sections/11-caching.tex`

**Verdict: accurate (minor-issues).** LaTeX is valid (`latex_ok = true`).

Adversarially checked every named symbol, every `file:line` citation, and every
behavioural claim against the four described source files
(`pipeline/_diagrams.py`, `pipeline/_expand_cache.py`, `pipeline/_precompute.py`,
`msrjd/core/cache.py`) plus the cross-referenced `pipeline/compute.py`,
`msrjd/diagrams/symmetry.py`, and `msrjd/integration/spatial/pipeline_bridge.py`.

The chapter is unusually faithful. All identifiers exist
(`PipelineCache`, `_stage_key`, `_to_int`, `_sobj_path`, `get_or_compute`,
`_load_manifest`, `_update_manifest`, `clear`, `_slug`, `list_cached_orders`,
`find_best_cached_order`, `downgrade_by_tp_dict`, `_poly_to_dict_form`,
`_dict_form_to_poly`, `_by_tp_to_dict_form`, `_by_tp_from_dict_form`,
`vertex_form_factor_signature`, `_canon_chain`, `_reconstruct_operator_ir_table`,
`prepare_for_load`, `save_expand`, `load_expand`, `precompute`,
`_build_fundamental_defaults`, `_solve_mf_at_saddle`,
`enumerate_unique_diagrams`, `_ext_fields_tag`, `_model_cache_dir`,
`combinatorial_factor`, `_wick_leg_factor`, `diagram_signature`,
`deduplicate_with_multiplicities`, `_build_cumulant_action`). Every line citation
I spot-checked (≈40 of them across all files, including all seven `load_expand`
sub-citations `:379–384`, `:388–394`, `:396–399`, `:411–418`, `:421–423`,
`:425–435`, `:464–472`) landed on exactly the cited construct. All three docstring
*quotations* (`_expand_cache.py:6–9`, `:14–18`, `:263–264`) are verbatim. The
load-bearing physics — `S(Gamma) = prod n_leg! / |Aut_fixed_ext|` carried by
`combinatorial_factor` (symmetry.py:438–439) with the dedup multiplicity being
diagnostic-only/double-counting-if-multiplied, the strict-superset / exact-downgrade
theorem, the `v1->v2->v3` version-bump history with dates, and the `-68/3 a^3` vs
`-32 a^3` cautionary tale ("4 of 11 a^3-sector 1-loop classes at k=3 for
OU+ax^2+bx^3") — all reproduce the source faithfully.

LaTeX: chapter is `\input` at `daedalus_manual.tex:194`; the build log shows zero
errors (no `!`, no "Undefined control sequence", no "Runaway argument"); a fresh
PDF was produced today. All macros/environments used (`\code \file \term \Daedalus
\msrjd \dd \ee \Dt \Lap \Gret \Sym \Aut`, tcolorboxes `note/gotcha/defn`) are
defined in the preamble. Every bare `_ # & \\` in the file is inside a
`lstlisting`/`console` block or a `tabular` (where `&`/`\\` are legitimate and all
`_` are escaped as `\_`).

## Findings

- **nit / Redundant triple-citation of the `sage_save(...removesuffix)` line.**
  Chapter (lines 212–215): *"That exact line appears at `cache.py:108`,
  `_expand_cache.py:345`, and inside `save_expand`."* —
  Code reality: `_expand_cache.py:345` **is** the line inside `save_expand`, so the
  list names the same occurrence twice and implies three distinct production sites.
  There are only two in production code (`cache.py:108`, `_expand_cache.py:345`); the
  only other occurrence is in `tests/test_expand_cache.py:360`.
  Location: `11-caching.tex:212–215` vs `pipeline/_expand_cache.py:345`.

- **nit / `sanity_check` shown without its argument.** Chapter writes
  `ft.sanity\_check()` (line 402 "which is what `sanity_check` verifies"; line 798
  Stage-2 description). — Code calls `ft.sanity_check(verbose=verbose)`
  (`_precompute.py:143`). The `verbose` kwarg is omitted in the prose; immaterial to
  behaviour, but the call signature as written is incomplete.
  Location: `11-caching.tex:402, :798` vs `pipeline/_precompute.py:143`.

- **nit / Slight paraphrase of the dedup-collision wording.** Chapter (line 972):
  *"four of the eleven one-loop classes at k=3 had been merged into the wrong
  representatives."* — Source (`symmetry.py:493–496`) is more precise: the old
  signature *"collided 4 of them into 2 representatives (mult=3 each)"* and the
  collided classes' integrals were *dropped*, not "merged into the wrong
  representatives." Same root cause and same numbers; wording is looser but not
  incorrect.
  Location: `11-caching.tex:964–972` vs `msrjd/diagrams/symmetry.py:489–498`.

## Spot-checks that PASSED (high-risk claims that turned out correct)

- `cache_version: 2` int in the bundle and Table~\ref{tab:bundle} `int (=2)` —
  matches `_expand_cache.py:343` (= 2). (Note this is *distinct* from the `v3`
  enumeration version; the chapter correctly keeps them separate.)
- The chapter consistently uses `unique_typed_mult_**v3**_...` in all five
  filename references and does **not** inherit the stale `v1` that appears in the
  `_expand_cache.py:41` module docstring's layout sketch.
- `'S_raw_dict'` "written by `save_expand` but never read by `load_expand`" — true:
  `load_expand` reads only `bundle['by_tp']` and `bundle['mf_sector_raw']` and
  rebuilds `_S_raw` by summing the (filtered) by_tp polys (`:425–432`).
- `taylor_order = max(k + 2*max_ell, 2)` is inside the `if taylor_order is None`
  auto-pick branch (`compute.py:279`); chapter's "By default" framing is correct.
- ASCII data-flow step markers `[1] FieldTheory.expand` and
  `[5] enumerate_unique_diagrams` match `compute.py`'s `[1/7]` (`:312`) and `[5/7]`
  (`:637`) phase labels.
- `parallel` default is `False` (`_diagrams.py:68`); `precompute` status
  `taylor_order` is hardcoded `2`; `_build_cumulant_action` returns `SR(0)` and is
  keyed on `model['correlated_noises']`; `pipeline_bridge.py:239–242` calls
  `enumerate_unique_diagrams(..., use_cache=False)`.
- Graceful-degradation contrast (expand cache → `False`; enumeration cache rebuilds
  the single ell slot at `_diagrams.py:170–173`) is accurate.

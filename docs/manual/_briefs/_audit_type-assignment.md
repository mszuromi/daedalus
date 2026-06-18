# Adversarial audit — `sections/10-type-assignment.tex`

Verdict: **minor-issues** (one major accuracy defect, three nits). LaTeX: **valid**.

Audited every named function/class/file/identifier and behavioural claim against:
`msrjd/diagrams/{type_assignment,causality,symmetry,filter}.py`, `msrjd/fork_safety.py`,
`msrjd/core/vertices.py`, `pipeline/_diagrams.py`, `tests/test_type_assignment.py`,
and the manual preamble `docs/manual/daedalus_manual.tex`.

The overwhelming majority of the chapter is **accurate** — line citations land on the right
functions, the lstlisting snippets match the source verbatim or near-verbatim, the convention
notes ((pi,ri) order, string zero-mask, fork guard, Path-A symmetry factor, complete-invariant
dedup, orbit–stabilizer compensation) are all faithful, and the historical-bug anecdotes
(−68/3 a³ vs −32 a³ collision; GTaS m̃→δn off-diagonal [ri,pi] bug; vertex_role_signature
over-division) reproduce the source docstrings correctly.

## Findings

- **MAJOR — two named regression tests do not exist.**
  Manual claim: the canonical-leg-matching equivalence "is pinned by the regression tests
  `test_leg_matchings_canonical` and `test_enumerate_typed_signatures_match_pre_change` in
  `tests/test_type_assignment.py`" (chapter ~lines 609–612), and gotcha #3 repeats
  "pinned by `test_leg_matchings_canonical`" (chapter line ~1276).
  Code reality: neither test is defined anywhere in the repo. `grep -rn` finds those two
  identifiers ONLY inside the `type_assignment.py` docstring (lines 380–381), never as a
  `def test_...`. The actual tests in `tests/test_type_assignment.py` are
  `test_leg_matchings_all_distinct_legs` (236), `test_leg_matchings_duplicate_response_legs`
  (266), `test_leg_matchings_mixed_multiset` (290), `test_leg_matchings_empty_legs` (322),
  `test_enumerate_typed_duplicate_leg_vertex_no_redundant_generation` (341), and
  `test_enumerate_typed_distinct_legs_regression` (479). The chapter copied the stale names
  verbatim from the source docstring (which is itself out of date — the tests were renamed),
  but presents them to the reader as greppable, existing tests. A reader who follows the cite
  finds nothing.
  Location: `docs/manual/sections/10-type-assignment.tex` ~609–612 and ~1276;
  source of the stale names: `msrjd/diagrams/type_assignment.py:380-381`.

- **NIT — "single edge, end to end" example is a reconstruction, not the cited test.**
  Manual claim: "The micro-example from `tests/test_type_assignment.py` makes the data shapes
  concrete," then shows `pd = (DiGraph([(1, 2)]), None, [1, 2], [])` with edge keys like
  `(1, 2, None)` (chapter lines 485–502).
  Code reality: the real test `test_single_edge_tree` (test_type_assignment.py:78) uses edge
  `(0, 1)` / `leaves=[0, 1]` built via a `_make_pd(...)` helper, not a literal
  `DiGraph([(1,2)])` 4-tuple with nodes 1→2. The illustrated data SHAPES
  (edge_types / external_legs / propagator_indices / empty vertex_assignments) are correct, and
  the chapter does frame it as illustrative — but the wording "from `tests/...`" slightly
  overstates fidelity; the node labels and construction idiom differ.
  Location: chapter lines 485–502 vs `tests/test_type_assignment.py:78-100`.

- **NIT — `_leg_matchings` docstring cited as `:356-387`; actual proof text is ≈352–382.**
  Manual claim (canonical-leg-matching subsection, chapter ~585): "the code's docstring
  (`type_assignment.py:356–387`) carefully proves [it]."
  Code reality: the multinomial formula is at line 355 and the "Correctness under this change"
  argument plus test names end at ~382 (line 384 begins "Yields"). The cite is off by a few
  lines on both ends but points at the correct docstring.
  Location: chapter ~line 585.

- **NIT — dedup "diagnostic only" rationale cited as `symmetry.py:556-568`; text is 558–567.**
  Manual claim (dedup gotcha, chapter ~947 and gotcha #4 line ~1279): the multiplicities being
  diagnostic-only is "explicit" at `symmetry.py:556-568`.
  Code reality: the "DIAGNOSTIC ONLY" paragraph in `deduplicate_with_multiplicities` runs
  lines 558–567. Within a couple of lines of the cite; harmless region drift.
  Location: chapter ~947 and ~1279.

## Spot-checks that PASSED (high-value confirmations)

- Edge convention comment `type_assignment.py:11-13`; (pi,ri) warnings at `:293-296` and
  `:490-502` (incl. the GTaS m̃→δn phys-idx-0 resp-idx-4 example) — verbatim accurate.
- `g_zero_mask` string test `str(G_ft[i,j]) == '0'` at line 165; comment `:144-157` — accurate.
- All `enumerate_typed_diagrams` step citations (`:158-165, :167-177, :179-184, :186-201,
  :204-213, :223-242, :244-266`) land correctly.
- `TypedDiagram.__slots__` = 5 attrs (48-49); `__getstate__/__setstate__` (60-65) — accurate.
- `build_field_index_map` (76-110), response-first ordering (`i < n_tilde`) — accurate.
- `_leg_matchings` body, `multiset_permutations` import at `:406`, `has_phys` source test — accurate.
- `_backtrack` base case (431-472) and recursive early-prune step (474-538) — accurate.
- `enumerate_all` def (582), `parallel=False` default (584), guard fires (630-634),
  lazy `import multiprocessing` (648), `_ENUM_WORKER_STATE` thread-safety note (657-660) — accurate.
- filter.py: `classify_prediagram_vertices` (12-40), `filter_prediagrams` (43-81), in-deg-0 =
  source — accurate. `available_degrees` (777), int_degs (793), src_degs (794);
  VertexType.in_degree = len(physical_legs), out_degree = len(response_legs) — accurate.
  `_parse_field_name` (327), pop_idx 1-based, 0 = no digits — accurate.
- causality.py: `check_pole_structure` (19-94), `check_causality` (97-123), `filter_causal`
  (126-152) returning `(kept, n_discarded, details)`; `>0` pass / `<=0` fail / `==0` hard-fail;
  try/except (TypeError, ValueError); inconclusive → True+conditions — all accurate.
- symmetry.py: ALL def line numbers match (`_vertex_combinatorial_factor` 51,
  `_wick_leg_factor` 115, `_colored_incidence_digraph` 149, `vertex_role_signature` 247,
  `_automorphism_order` 328, `external_wick_compensation` 343, `combinatorial_factor` 403,
  `compute_all_combinatorial_factors` 449, `diagram_signature` 467, `deduplicate_typed_diagrams`
  527, `deduplicate_with_multiplicities` 553, `classify_coefficient_factors` 618).
- Incidence-digraph colours: internal `('vertex', type(vt).__name__, str(vt.coefficient),
  vt.bigrade)`; leaf `('leaf', v, field)` / `('leaf', field)`; edge `('edge', resp_leg,
  phys_leg, prop)` — verbatim accurate.
- `combinatorial_factor` numer/aut floor-division fallback (440-446); `external_wick_compensation`
  raises on non-divisor (391-399) — accurate, and the deliberate asymmetry is real.
- `diagram_signature` body (cells/edges/canonical_label) verbatim; history note (484-498)
  with 11→collide-4-into-2 and −68/3 a³ / −32 a³ — accurate.
- `classify_coefficient_factors` returns keys {'Scal','scalar_prefactor','vertex_time_factors',
  'source_time_info','is_stationary'}; coeff = −SR(vtype.coefficient); td_part/const_part split;
  reduce(mul,...) — accurate.
- fork_safety.py: `fork_unsafe_in_notebook` (27-44) verbatim, ZMQInteractiveShell fingerprint;
  `warn_fork_fallback_once` (47-61), `_WARNED` dedup; OBJC flag "controlled abort → hard crash"
  matches docstring 9-10 — accurate.
- Orchestrator: `enumerate_unique_diagrams` in `pipeline/_diagrams.py` (def 54); four lines
  176-186; `filter_causal` @185, `deduplicate_with_multiplicities` @186;
  `enumerate_all_typed` = alias of `enumerate_all` (import line 31); cache tag
  `unique_typed_mult_v3` (line 150) — accurate.
- `test_enumerate_all_parallel_matches_serial` EXISTS (line 383) and does element-wise
  signature compare of serial vs fork-parallel; runs under pytest on the fork path — accurate.

## LaTeX validity — PASS (latex_ok = true)

- All environments balanced: begin/end counts equal for center(1), defn(3), description(3),
  enumerate(1), equation(4), example(3), figure(1), gotcha(6), itemize(6), lstlisting(17),
  note(4), quote(1), tikzpicture(2).
- Custom environments defined in preamble: `note`/`gotcha`/`defn` via `\newtcolorbox`
  (daedalus_manual.tex:107/109/111), `example` via `\newtheorem` (line 130).
- All chapter macros defined: `\code`,`\file`,`\term`,`\msrjd`,`\dd`,`\ee`,`\Dt`,`\avg`,
  `\Gret`,`\Sym`,`\Aut` (preamble lines 98–125).
- Colours used in chapter tikz (`notebg`, `codebg`) defined (lines 56, 51); tikz libs
  arrows.meta/positioning/calc loaded (line 40); packages listings/tcolorbox/tikz loaded.
- Chapter is `\input` at daedalus_manual.tex:193, so preamble is in scope.
- No bare unescaped specials in text: 306 inline `$` (even/balanced); every flagged `_`/`$`/`&`
  is inside math (`$...$`, `\[...\]`, `equation`, tikz nodes) or written `\_`. Zero unescaped
  `_` inside any `\code{...}`/`\file{...}` text argument.

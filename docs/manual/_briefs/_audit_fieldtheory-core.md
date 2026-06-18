# Audit: `sections/06-fieldtheory-core.tex`

Adversarial accuracy + LaTeX audit against the real source
(`msrjd/core/{field_theory,vertices,convolution,serialize}.py`,
`models/hawkes_quad_expg.py`, `theories/*.theory.py`).

**Verdict: minor-issues.** The chapter is overwhelmingly accurate: every named
symbol exists, the line-number citations are correct (some are decorator-vs-def
off-by-one, all immaterial), the listings match the source verbatim, and the
LaTeX compiles cleanly (0 fatal errors; PDF regenerated after the chapter's last
edit; only cosmetic Overfull \hbox warnings). The findings below are the few
real deviations.

`latex_ok = true` — all custom macros (`\code`, `\file`, `\term`, `\msrjd`,
`\ee`, `\dd`, `\Gret`) and tcolorbox environments (`note`, `gotcha`, `defn`,
`example`) are defined in the preamble (`daedalus_manual.tex` lines 98–130);
all 13 lstlisting / 13 gotcha / 3 note / 2 defn / 1 example / 5 itemize /
3 enumerate / 2 description / 2 table / 2 tabular environments balance;
`grep -c "^! " daedalus_manual.log` = 0.

---

## Findings

- **major** — *Wrong consumer file for `available_degrees`.* Chapter
  §"What the enumerator actually asks for" (lines 942–946) says: "The diagram
  enumerator (`msrjd/enumeration/degree_scan.py`) reads exactly this." Code
  reality: `available_degrees` (vertices.py:777) is imported and called only in
  **`msrjd/diagrams/filter.py:60–61`** (`filter_prediagrams`). `degree_scan.py`
  does NOT import or call `available_degrees`; it computes degrees from
  prediagram graph objects via its own `max_vertex_degree` /
  `scan_source_vertices`. Pointing at the wrong file for a named-consumer claim.
  (The broader directional claim — output feeds the enumerator of
  Ch. enumeration — is defensible.)
  Location: chapter L942–946 vs `msrjd/diagrams/filter.py:60–61`,
  `msrjd/enumeration/degree_scan.py:1–67`.

- **major** — *Hawkes "running real example" overstated re: the `Conv`
  operator.* Chapter (L39–41) designates `models/hawkes_quad_expg.py` as the
  example that "exercises every feature the subsystem has," with "a synaptic
  convolution kernel," then dedicates §The convolution operator to `Conv(g,n)`
  with the `v·Conv(g,n)` conductance example. Code reality: the
  `hawkes_quad_expg.py` action (line 246) uses plain multiplication
  `ns.g * (ns.nstar[j] + ns.dn[j])` — it contains **no `Conv` atom**, so step 3
  (`reduce_conv_in_action`) is a no-op for this model. The literal `Conv(...)`
  operator and the `v·Conv(g,n)` form are exercised by the
  `theories/*conductance*.theory.py` files (e.g.
  `single_population_conductance_test.theory.py:33`,
  `single_population_linear_conductance_test.theory.py:34`), not by the named
  model. (`g` IS declared as a kernel with a `kernel_ft_image`, so calling it a
  "convolution kernel" is physically fine; the "exercises every feature" framing
  is what misleads — a reader cross-referencing the model file for `Conv` will
  not find it.)
  Location: chapter L38–43 / §sec:conv vs `models/hawkes_quad_expg.py:113,236-250`,
  `theories/single_population_conductance_test.theory.py:33`.

- **minor** — *"again inside the verifier (field\_theory.py:949)" mis-points.*
  Chapter §sec:mf gotcha (L780) says specializations run "again inside the
  verifier (field\_theory.py:949)." Line 949 is in `expand` (it *prepares*
  `spec_subs` immediately before calling `_verify_and_zero_mf_sector`); the
  actual second `.subs(spec_subs)` application is **inside** the verifier at
  `field_theory.py:621`. The substance (specializations applied a second time)
  is correct; the cited line is the expand-side preparation, not "inside the
  verifier." Location: chapter L780 vs `field_theory.py:621` (apply) /
  `field_theory.py:949` (prepare).

- **nit** — *"The four reduction rules" header undercounts.* The subsection is
  titled "The four reduction rules" (L553) but the `\begin{description}` that
  follows enumerates **five** items, Rule 0 through Rule 4 (matching the
  convolution.py docstring's "0.–4." list). Internal inconsistency: either the
  title should say five, or it should say "Rules 0–4." Location: chapter
  L553–579 vs `convolution.py:115–141`.

- **nit** — *`_mf_numerical_residual` attribution of the `default` preference.*
  Chapter §sec:mf note (L763–766) says `_mf_numerical_residual`
  "builds a representative parameter point (preferring each parameter's declared
  `default`…)." Strictly, `_mf_numerical_residual` (field_theory.py:662) does
  `model.get('fundamental_defaults') or _default_fundamental_point(...)`; the
  per-parameter `default` preference lives in `_default_fundamental_point`
  (730), which the chapter does cite separately one sentence later. Harmless
  conflation. Location: chapter L763–772 vs `field_theory.py:672–673, 730–762`.

- **nit** — *`E` vs `Em` cosmetic.* Chapter (L772) writes the no-real-root
  example as "$\phi=av^2$ with $E=a=w=1$"; the `_default_fundamental_point`
  docstring writes `Em=a=w=1` (the parameter is named `Em` in the model).
  Cosmetic; the algebra ($2v^2-v+1=0$) is identical and correct. Location:
  chapter L772 vs `field_theory.py:738`.

---

## Spot-checks that PASSED (representative, not exhaustive)

Symbols / line citations verified exact or off-by-one-immaterial:
`_Namespace` (185), imports 26–29, `_poly_taylor` (193), `_iter_multi_indices`
(202) + autoreload comment (210–215), `_multi_index_suffix` (225), `_sr_to_ring`
(241) + add-name gotcha (266–267) + single-term wrap (268–271) + loop (272–281),
`_build_cumulant_action` (293) + return `SR(0)` (332) + δ-peel loop (460–463) +
`is_trivial_zero` (490) + order≥3 drop `warnings.warn` (550–565), `_collect_bigrade`
(570) verbatim, `_verify_and_zero_mf_sector` (585) + two-tier + AssertionError text
("Either the MF solver is wrong…", 644–645) + num_tol 1e-9, `_mf_numerical_residual`
(662) prefers DAE when `equations` declared (684) with `xstar=-eps*xstar^3` example,
`_default_fundamental_point` (730) + `2v²-v+1=0` docstring, `expand` (814) +
docstring "sequential taylor()" (818) + one-shot taylor at 894 (listing verbatim) +
classify-before-MF comment (913–922), `_build_namespace` (1041) ordering (index→fields→
ring tilde-first→params→kernels/operators→functions→mf_subs), kernel naming
(`z_<name>_<i+1>`, matrix `z_<name>_<i+1>_<j+1>`, scalar by sage_name),
`ns.delta_D='z_delta'`/`delta_Dp` (1245–46), `z`-sorts-after-tau/phi comment (1243),
`'expression'` path diff-at-0 + skip-numeric (1305–1314), `taylor_coeffs`→`_poly_taylor`
(1346–1361), `sanity_check` (966) always-prints-FAIL (975/990), `free_action` (1010)
=(1,1), `noise_kernel` (1015) =(≥2,0), `vertices` (1021) =total≥3, `sectors` (1027),
`summary` (999), `_sector_label` (1380/1381).

vertices.py: `_NamespaceBoundKernel` (25, module-level) pre-eval at 56,
`VertexType` (80) slots + in/out/total_degree (114–126), `ConvVertexType` (134)
kernel_attachments keys 'symbol'/'leg'/'leg_index'/'kernel_td_fn' (162–182),
`SourceType` (216) + explicit `SourceType.__slots__` in `__getstate__` (240) +
comment (235–238), `NoiseSourceType` (256) + `cumulant_specs` (298),
`_parse_field_name` (327) trailing-digit split, `decompose_sector` (349) listing
(386–393 verbatim), `extract_vertex_types` (500) unresolved-kernel-leaves-coeff
(589–595), `extract_source_types` (629) NoiseSourceType upgrade (766),
`available_degrees` (777) listing verbatim, no `multiprocessing` import.

convolution.py: `Conv=_sr_function('Conv',nargs=2)` (70), `reduce_conv_in_action`
(110) `normalized=True`, rules 0–4 (234–289), `.substitute_function` (291),
add-name test (254), normalized DC ref (182–185), v·Conv(g,n) expansion matches
docstring (152) and chapter (L544).

serialize.py: `save_theory` (52), `load_theory` (156) returns (meta,data),
`reload_model` (192) re-imports from model_file, metadata.json fields,
symbolic_data.sobj `sym` = R/S_raw/by_tp/n_tilde + propagator mats (134–148),
stale "Use SageMath's load()" comment (232) vs `exec(compile(...))` (237).

Cross-file: OU action `xt[i]*((Dt+mu)*x[i]+eps*x[i]^3)-D*xt[i]^2`
= `theories/ou_quartic.theory.py:28` verbatim; Hawkes action @
`hawkes_quad_expg.py:236`; φ specialization phi1→2a·vstar, phi2→2a @
`hawkes_quad_expg.py:186`; `pipeline/compute.py` builds FieldTheory (314) +
`ft.expand()` (329). Chapter 06 `\input` at `daedalus_manual.tex:189`.

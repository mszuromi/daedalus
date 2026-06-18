# Audit: `sections/16-spatial-coupled.tex` (Coupled Fields, Dyson Dressing, the Bridge)

**Scope.** Full read of the 1281-line chapter. Every named identifier, file:line
citation, code snippet, default argument, and behavioural claim verified by targeted
grep/Read against the five source files. LaTeX validity verified by an actual
`pdflatex` compile of the chapter against the real manual preamble.

**Verdict: minor-issues.** The chapter is exceptionally accurate. Function names,
signatures, defaults, control flow, the inline code listings, and the ~60 file:line
citations almost all match the source verbatim. One file:line citation is wrong
(points at unrelated code ~1500 lines away). The remaining items are nits.

**LaTeX: OK (latex_ok = true).** Standalone compile (full preamble + this chapter +
`\end{document}`) succeeded with `pdflatex -halt-on-error`, exit 0, 19 pages, zero
LaTeX errors. All environments balanced (note 5/5, gotcha 12/12, defn 13/13,
example 1/1, lstlisting 13/13, equation 17/17, align 1/1, figure/tikzpicture 1/1).
All macros/environments used are defined in the preamble (`\code`,`\file`,`\term`,
`\msrjd`,`\ee`,`\ii`,`\dd`,`\Lap`,`\Gret`,`\avg`,`\Sym`,`\Aut`; tcolorboxes
`note`/`gotcha`/`defn`; `\newtheorem{example}`; tikz libs arrows.meta/positioning/calc).
No bare `_ # & % $` outside math/listings. The 23 "undefined reference" warnings in
the isolated build are expected cross-chapter `\ref`s and resolve in the full manual.

---

## Findings

- **major / Wrong file:line citation for the "two views coincide at k=2" comment.**
  In the General-k section the chapter says: "As the comment at
  `\file{pipeline\_bridge.py:509}` notes, the two views coincide at $k=2$."
  Code reality: line 509 is `_env = os.environ.get('SPATIAL_DYSON_ORDER')` inside
  `compute_coupled_tree_correlator` — the Dyson env-var policy, nothing about k=2.
  The actual comment ("the two coincide at k=2: mixed-field leaves give singleton
  mappings with comp=1 ...") lives in the `compute_coupled_kpoint` docstring at
  **lines 2006-2008** (a near-identical remark is also at line 1902). A reader
  following the citation lands ~1500 lines away on unrelated code.
  *Location:* chapter §"General-k cumulants" (the sentence ending
  "...the mapping sum is the general machinery"), citing pipeline_bridge.py:509.

- **nit / `loop_dyson.py` banner line range slightly off.** Chapter: the oracle
  banner is at "(\file{loop\_dyson.py:1-5})". Code reality: the banner is lines
  **1-4**; line 5 is blank and line 6 starts the module-title docstring block. The
  quoted text itself is reproduced correctly.
  *Location:* chapter §"The oracle module", first sentence.

- **nit / `moment_x` perf factorization attributed to the docstring, but it is a
  code comment.** Chapter: "The key performance factorization, recorded in the
  docstring of `moment\_x`: the polynomial-in-$X$ form factor has coefficients
  $g_k$ that are $x$-independent ...". Code reality: this is a `#` comment block at
  **lines 899-904**, immediately above the `def moment_x` (line 913); the function's
  own docstring (line 914) is only a one-line shape annotation. The technical content
  of the claim is correct (g_k are x-independent; ~n_x speedup; cse folds 1/Bcal^k).
  *Location:* chapter §"The analytic spatial inverse Fourier transform", paragraph
  after eq:MF.

- **nit / Dyson-divergence gotcha line range understated.** Chapter cites
  "(\file{pipeline\_bridge.py:519-523})" for the rho>=1 raise + rho>0.6 warn. The
  raise spans 521-525 and the warn 526-528, so the cited range covers the raise's
  start but not the warn. Both behaviours are described correctly (raise on
  rho>=1, warn on rho>0.6). Off-by-a-region only.
  *Location:* chapter §"The bridge architecture", "The Dyson divergence guard" gotcha.

---

## Spot-checked and CONFIRMED accurate (representative, not exhaustive)

Identifiers / signatures / defaults:
- `split_reference_diffusion` (spectral_propagator.py:55), `spectral_projectors` (:70),
  `_COND_CAP=1e10` (:52), `SpectralReference` (:91), `build_reference` (:116),
  `lyapunov_covariance` (:156), `coupled_two_point` (:169), `_opitz_bidiagonal` (:211),
  `phi_n` (:221), `phi_n_batch` (:259) — all line numbers and the two inline listings
  (split_reference_diffusion, spectral_projectors, lyapunov_covariance,
  coupled_two_point, phi_n) match verbatim. n=1 worked check at :240 ✓.
- `hcal_n` (dyson_dressing.py:47), `dressed_GR` (:73),
  `dressed_tree_C(q,tau,M,Dhat,D0,N,order,n_s=48,s_cap_scale=32.0)` (:87) — signature
  and the cap/leggauss/s_nodes/einsum snippet match.
- pipeline_bridge.py def lines all exact: `_legs_to_phys_idx` 89, `pipeline_C_q_tau`
  259, `certify_modes` 286, `compute_spatial_correlator_via_pipeline` 304,
  `compute_coupled_tree_correlator` 449, `_diagram_is_bubble` 664,
  `diagram_form_factor` 685, `_isserlis` 802, `_build_wick_moment` 815,
  `_zero_ff_with_moments` 972, `_prefactor_is_live` 1058,
  `compute_coupled_loop_correlator` 1090, `compute_spatial_correlator_generic` 1434,
  `compute_spatial_kpoint` 1788, `_build_wick_moment_multi` 1893,
  `compute_coupled_kpoint` 2000. File length 2157 ✓.
- spatial_operator_ir.py: `_LAP/_DT/_DX` 68-70 (import at 59), `GRADX_SYM` 78,
  `apply_linearity` 147, `expand_about_saddle` 220, `kill_means` 245 (ops default
  ('Lap','Dt','Dx')), `to_derived_generators` 275 (prefix='Dg'), `prepare_action` 346,
  `classify_generators` 388, `form_factor` 420. form_factor body listing matches.
- lambdify citations 911/937/997/1043/1979 all exact; cse=True present.

Behavioural claims:
- Coupled tree requires V=0 -> NotImplementedError (:497-500); scalar-diffusion ->
  coupled_two_point, unequal -> dressed_tree_C; rho>=1 raise (:521), rho>0.6 warn (:526).
- Dispatch predicates: `prop['G_tx_sym'] is None` -> coupled (:322);
  cross-correlator `_field_index(...)!=fi` -> coupled (:364). Hard-gate certify:
  raises SpatialPropagatorError if certify_max_rel>certify_tol, default 1e-8 (:307,400).
- coupled_loop `2^{-n_C}` does NOT apply (docstring :1108-1110); diagram_to_cstack,
  spectral_rows, fpairs, itertools.product(range(nf),repeat=n_rows); weights
  `proj[a][pi,ri]` = [P]_{p,r} (:1213); cross-correlator i!=j fix (:1368-1400).
- Memory guard P=n_t^{n_V}·n_s^{n_C}, KPZ 2-loop n_V=4 n_C=3 ~1.8e8 at (16,14),
  default budget 6 GB, 4 knobs, bypassed for mc/bessel (:1633-1665).
- Analytic/numerical gate `_use_analytic=(_all_plain or d==1) and not _force_num`
  (:1677). Smart-gated ThreadPoolExecutor, _heavy = loop-order>=2 (:1712-1714).
- Drift V!=0 refused on generic loop path (:1540-1554) and coupled tree (:497).
- `_modes_C_q_tau` reference formula (:198) matches the chapter's certify-reference
  Σ_α κ_α/(μ_α+D_α q²)e^{-(μ_α+D_α q²)|τ|}.
- loop_dyson.py: oracle banner text (verbatim), Σ_R/Σ_K via scipy.integrate.quad
  (Gauss-Kronrod), C_R,C_K=4.0,2.0 at :89, B=0.99 validation. Σ_R formula matches.
- field_respecting_mappings + external_wick_compensation named correctly; coupled
  k-point scoped to scalar-D / plain vertices / infinite boundary (:2022-2024).

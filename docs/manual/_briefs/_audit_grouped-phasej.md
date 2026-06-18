# Audit: `sections/13-grouped-phasej.tex`

**Verdict: accurate (minor-issues).** Every named identifier exists, every behavioural
claim checks out against the code, and the chapter compiles cleanly (the manual's
`daedalus_manual.log` from the latest build has zero `!` errors and no
"Undefined control sequence" / "Missing \$" around `13-grouped-phasej.tex`,
which renders to pages 204+). The findings below are all citation-region
imprecisions and one lightly-cleaned "verbatim" quote — no fabricated symbols,
no backwards gates, no LaTeX breakage.

## Findings

- **minor** — `RuntimeError` line citation points at the History docstring, not the raise.
  - manual claim: in the `\begin{note}` near ch. line 824, "[`external_wick_compensation`] raises a `RuntimeError` rather than return a wrong divisor (\file{symmetry.py:368--381})".
  - code reality: lines 368–381 of `msrjd/diagrams/symmetry.py` are the **History** section of the docstring (about the old `∏N!` heuristic). The divisibility check is at line **391** and the actual `raise RuntimeError(...)` is at **396–399**. The behavioural claim itself (raises if `aut_free % aut_fixed != 0`, with the subgroup argument) is correct; only the line region is wrong.
  - location: `docs/manual/sections/13-grouped-phasej.tex` ch. line ~826 → `msrjd/diagrams/symmetry.py:391,396`.

- **nit** — second `symmetry.py:368--381` citation is region-correct but points at prose, not the computation.
  - manual claim: ch. line ~847, "It was replaced by the exact Sage automorphism ratio (\file{symmetry.py:368--381})."
  - code reality: 368–381 is the History docstring that *describes* the replacement (so it is defensible), but the ratio is actually computed at `aut_free // aut_fixed`, line **400** (and `_automorphism_order` calls at 389–390). Borderline acceptable since the cited region is exactly the narrative the sentence paraphrases.
  - location: `docs/manual/sections/13-grouped-phasej.tex` ch. line ~847 → `msrjd/diagrams/symmetry.py:389-400`.

- **nit** — the `[style=console]` block quoting the module docstring (lines 9–13) is lightly cleaned, not strictly verbatim.
  - manual claim: ch. lines 72–77 present a console block attributed to `grouped_integral.py:9--13` reading "Typed diagrams sharing the same prediagram have identical graph topology, leaves, internal vertices, and integration variables. They differ only in per-edge propagator field-type indices (ri, pi) and per-vertex coefficients."
  - code reality: the source (lines 9–13) reads "...carries its parent `prediagram` tag. Typed diagrams sharing the same prediagram have **identical graph topology, leaves, internal vertices, and integration variables**." The quote drops the leading sentence and the bold, and the quoted span actually begins mid-line-10. Faithful in meaning; flagged only because it is rendered as a quotation.
  - location: `docs/manual/sections/13-grouped-phasej.tex` ch. lines 72–77 → `msrjd/integration/time_domain/grouped_integral.py:9-13`.

- **nit** — `EdgeModeSum` convention citation region is slightly off.
  - manual claim: ch. lines 183–186 cite "\file{final\_integral.py:128--133}, with the index ordering $G[p_i, r_i] = \langle \varphi_{p_i}\,\tilde n_{r_i}\rangle$ --- physical row, response column."
  - code reality: `class EdgeModeSum` is at line **117**; the index-ordering convention `G[pi, ri] = ⟨φ_pi ñ_ri⟩` is in the `ri, pi` field doc at line **124**. Lines 128–133 are the `modes` field doc (which does carry `λ_α = i·p_α`, `C_α = C_mats[α][pi, ri]`, and the smooth-propagator formula the chapter also attributes there). So the cited span supports most of the sentence but not the `G[pi,ri]` ordering, which sits a few lines earlier. Right file, off by ~4 lines.
  - location: `docs/manual/sections/13-grouped-phasej.tex` ch. lines 183–186 → `msrjd/integration/time_domain/final_integral.py:117,124,128-133`.

## Spot-checks that PASSED (high-value, in case of doubt)

- `compute.py:129` `use_grouped_phase_j: bool = False` — exact. Dispatch block `compute.py:757–774` — exact match incl. kwargs.
- `build_G_t_matrix` at `propagator_td.py:135`; decomposition `G_R = delta_coeffs·δ(t) + Θ(t)·Σ_k C_k e^{i p_k t}` at 140–141; Fourier-convention `Im(p)>0` / Heaviside text at 157–161 — all exact.
- `_grouped_phase_j.py`: `_SUBSET_SKIP_STATUSES` frozenset (49), "production bug we're fixing" comment (42–48), `compute_correction_td_grouped` (55), ext_time_vars auto-build (72–76), `_ext_legs_key` (87), `groups` defaultdict (96–100), `_perdiag_fallback` (107), `has_skipped_subset` scan (147–153), three-tier dispatch incl. loud `skipped` (154/165/192/213–228), `total_C`/`total_C_batch` (231–241), return dict with `eval_per_diagram_batch=None` + `delta_contributions=[]` (243–252) — all verified. `total_C_batch` does ignore `parallel`/`n_workers` (serial list comp) — confirmed.
- `grouped_integral.py`: `USE_GROUPED_ANALYTIC_MODESUM=True` (130), `_GROUPED_EXP_REAL_LIMIT=600.0` (135), 709-ceiling comment (133–134), `_evaluate_grouped_m0_modesum` (138) incl. `c_eff<=0` at 155, `_integrate_grouped_m1_modesum` (179) incl. `c_eff<=0` at 217 and unused `bbox_cap` param (confirmed never referenced in body), bbox_cap/`quad_exp_k2_ell0` docstring (198–200, within cited 196–202), `integrate_grouped_diagram` (283), prediagram-identity check (349–350), shared scaffolding (360–379), Wick enumeration + compensation-agreement (381–452, `status='failed'` 437–448), vertex times w/ `SR(0)` pin (461), noise-source loop (471–520), per-td edge_info w/ `include_heaviside=False` (522–591), `grouped_analytic_enabled` gate (602–607), λ=`complex(_CDF(SR(p)))*1j` (612–613), residue tensor (619–628), subset loop `range(2**n_edges)` (654), summed integrand (759–770), fast_callable (815–818), constraint extraction (830–847), merged-residue pole tuples (925–948), dummy `EdgeModeSum` (949–961), contribution closure (1049–1074) and `/ _comp` (1074) — all verified.
- `final_integral.py`: `_enumerate_pole_tuples` (529) yields `∏_e C_α_e` Cartesian product — exact. `_integrate_2d_polygon_modesum` (2175) fan-triangulates from `polygon[0]` (2230–2233) — exact. `_exp_over_unit_triangle` (356) is the `J(p,q)` whose docstring gives the w-first derivation and 4th-order Taylor fallback "when any of |p|,|q|,|p−q| drops below `_J_TAYLOR_EPS`" (364–367) — exact; chapter uses `J(p,q)` only as math notation (matching the docstring), never claims the function is named `J`. `_integrate_nd_polytope_poset_modesum` (1726) enumerates linear extensions → chain simplices `L ≤ s_{σ(0)} ≤ … ≤ U` (1750–1753) — exact. `USE_POLYGON_M2_INTEGRATOR=True` (287), `USE_POSET_INTEGRATOR=True` (1661), `_loop_number_from_graph` (2401, `L=|E|−|V|+1`), `_lookup_prop_indices` (5230), `_integrate_polytope` (4372), `vertex_role_signature` ref-comment (2573), `POLYGON_BBOX_CAP` (288) — all verified.
- `USE_POSET_CAP_MATCH_SCIPY` discrepancy block at `final_integral.py:1690–1700`: contains exactly the per-diag `+2.66e-3` vs grouped `+6.38e-4` 4× discrepancy and "bookkeeping difference in pole-tuple construction" conclusion the chapter cites. `USE_POSET_MPMATH_ACCUMULATION=False` (1723) with the "did NOT address the spike-reset bug" note (1719–1722) — exact. The `1j`-fold `λ_k=i p_k` mode convention — consistent throughout.
- `symmetry.py`: `vertex_role_signature` (247) components (type+coefficient+bigrade, external-leaf attachment pattern, internal-edge incidence pattern) and "depth-1 / shallow" wording — exact. `external_wick_compensation` (343) with `comp = |Aut_free|/|Aut_fixed|` and Boltzmann `κ₄ = −6ε + 126ε²` — exact. `_automorphism_order` (328) "coloured incidence digraph" + `automorphism_group(...).order()` (339–340), two calls (389–390) — exact.
- LaTeX: all environments balanced in the chapter (lstlisting 17/17, note 7/7, gotcha 7/7, defn 4/4, equation 13/13, description 4/4, itemize 1/1). All macros resolve in the preamble: `\code`/`\file`/`\term` (98–100), `\dd`/`\ii`/`\ee` (117–119), `\Gret` (123), `\Aut` (125), `bm` (21), `amsmath/mathtools` (20) for `\eqref`/`\texorpdfstring`, environments `note`/`gotcha`/`defn` (107–111), `[style=console]` and default `\lstset{style=py}` (93). Every `_` inside `\code{}`/`\file{}` is escaped (grep for unescaped variants returned empty). No bare `# % & $` in prose. Manual builds — `daedalus_manual.pdf` present, log error-free.

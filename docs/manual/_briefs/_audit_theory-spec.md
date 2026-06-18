# Audit — `sections/03-theory-spec.tex` (Specifying a Model: the TheoryBuilder)

Audited against `pipeline/theory.py`, `pipeline/theory_compiler.py`,
`pipeline/theory_templates.py`, plus the referenced
`theories/ou_quartic.theory.py`, `theories/kpz_1d.theory.py`,
`pipeline/compute.py`, `msrjd/core/field_theory.py`,
`pipeline/colored_to_markovian.py`, `pipeline/spatial_operator_ir.py`.

**Verdict: minor-issues.** The chapter is overwhelmingly accurate — every named
class/method/file/identifier exists, the cited line numbers land on the right
region (defs, docstrings, or the relevant statement block), the model-dict key
inventory is exact, and the LaTeX compiles cleanly. Three substantive findings
below: one **major** (a wrong sign in a stated formula that contradicts the code
*and* the chapter's own kernel image), two **minor** (a lambda-count off by one,
appearing in two places).

## Findings

- **[major] Fourier-transform exponent sign is wrong.** Chapter (line 451–452):
  "`make_kernel_ft_image_lambda` ... calls `msrjd.core.field_theory.fourier_transform`
  — a wrapper around Sage's `integrate` computing **∫g(t)e^{iωt}dt**." The actual
  function uses the **e^{−iωt}** convention: its docstring states `F(s) = ∫ f(t) e^{-i s t} dt`
  and the integrand is `f.subs({heaviside(t):1}) * exp(-I * s * t)` / `f * exp(-I*s*t)`.
  The code's e^{−iωt} is also the convention consistent with the chapter's own
  stated exponential-synapse image ĝ(ω)=1/(1+iωτ_g) (lines 434–435; `ExpSynapticKernel`
  returns `1/(1 + I*omega*tau_g)`); the e^{+iωt} written in the chapter would give
  1/(1−iωτ_g). So the prose formula is internally inconsistent and wrong.
  Location: chapter line 452 vs `msrjd/core/field_theory.py:35-39,59,62`.

- **[minor] "six companion lambdas" undercount in `use_action_template`.** Chapter
  (line 525): "`use_action_template` ... wires a template in by pulling its **six**
  companion lambdas off it." The code pulls **seven**: `action()`, `phi_concrete()`,
  `specializations()`, `mf_bg_conditions()`, `mf_equations()`, `mf_substitutions()`,
  and `functions_list()`. The omitted one is `mf_substitutions()`.
  Location: chapter line 525 vs `pipeline/theory.py:673-679`.

- **[minor] `HawkesAction` method list omits `mf_substitutions()`.** Chapter
  (lines 506–512): "Its methods each return a lambda: `action()` ..., `phi_concrete()`,
  `specializations()`, `mf_bg_conditions()`, `mf_equations()`, and `functions_list()`."
  This lists six and omits `mf_substitutions()`, which `HawkesAction` does define
  (and which `use_action_template` consumes). Same root omission as the count above.
  Location: chapter lines 506–512 vs `pipeline/theory_templates.py:169` (`def mf_substitutions`).

## Nits (not counted in the structured findings)

- **[nit] "Plain (v1)" example uses call syntax.** Chapter (lines 553–554) describes
  the plain/multiplicative Laplacian style yet writes the example with call-looking
  `Dt(p)` / `Lap(p)`: `pt*(Dt(p) + mu*p - DD*Lap(p) + g*p^2) - T*pt^2`. This is
  faithfully copied from the `set_action_text` docstring (`pipeline/theory.py:1031`),
  which has the same arguably-confusing form, so the chapter is accurate *to the
  source*. Worth a future cleanup in the docstring itself, not a chapter error.

- **[nit] A few line citations point at the start of the relevant comment/code block
  rather than the exact statement** (all land within the right region, no reader would
  be misled): `:1697` autopop (gate condition `n_populations <= 1` is at 1721; 1697 is
  the comment-block start); `:606` mt1-duplicate gotcha (the `ValueError: variable name
  'mt1' appears more than once` string is at 614; 606 is the explaining comment); `:1530`
  kernel-FT ValueError (the `raise` is at 1533, inside the `try` at 1530); `:1850`
  reserved-name check (the raise is at 1877; 1850 is the `_reserved_always` set that
  opens the block). All acceptable.

- **[nit] `_safe_eval` code excerpt** (chapter lines 624–630) omits the wrapping
  `try:` that surrounds `return eval(parsed, eval_globals)` in the source (line 707–708).
  An abbreviated excerpt, faithful in substance.

## Confirmed accurate (spot list)

- Builder classes & lines: `TemporalTheoryBuilder` (2087), `SpatialTheoryBuilder`
  (2102, ctor `n_populations=0` at 2108), `TheoryBuilder` (2121); mixin bases
  `_TemporalMethods`/`_SpatialMethods`/`_BaseTheoryBuilder`; `_resolve_spatial_dim`
  overrides raise at 2092 (temporal) / 2111 (spatial) — verbatim match to the quoted
  snippet. Every builder method returns `self`.
- `physical_field` (806): new vs legacy style, `dn` derivation (858), auto-response
  `<natural>t` (885), auto-saddle `<natural>star` with `mean_field=True` and domain
  `'positive'` only for `n` (899–902). Three-symbol summary (dx/xt/xstar) correct.
  `response_field` (796). `FieldSpec.spatial_dim` per-field (79). bulk `spatial_dim(d)`
  (122).
- `ParameterSpec` dataclass fields match the quoted listing; matrix grid
  `SR.var(f'{_w}{i+1}{j+1}', domain=_d)` at 1634. `parameter` (914).
- `set_action_text` (1008), docstring through 1049; "non-Gaussian white noise =
  response-field monomial of degree ≥ 3" at 1040.
- `define_function` (969), `set_transfer_function` default `'phi'` (1220).
  `_IndexedFormalFunction` 1-based suffix `i+1` (73,110–114). Hawkes `functions_list`
  formal lambda `lambda i, v: function(f'phi_{i+1}')(v)` and the "phi(0)=0 ... uncancelled
  (1,0) tadpole" rationale (templates 319–345).
- `.equation()` (1087): validation lines 1143–1184 — non-empty strings; `Conv(...)`
  rejected via `\bConv\s*\(` case-insensitive on both sides; `Dt` in rhs rejected via
  `\bDt\b`; kind = 'differential' iff `\bDt\b` in lhs. multi-start Newton + sort +
  `fixed_point_index`. `stability_analysis(True)` (1187), off by default.
- `set_mf_equation` (1069); `_auto_populate_mf_eqs_from_equations` (1276): Dt→0,
  single state-var, `<var>star`, word-boundary RHS rewrite (Em vs E).
- `define_kernel` (370): raises if both `time_expr`/`freq_image` None; `make_kernel_ft_image_lambda`
  (1493) FT failure ValueError (1530–1537). `declare_cgf_term` (461). `markovianize(True)`
  default on (429), rewrite in `pipeline/colored_to_markovian.py::markovianize_spec` (198).
- Templates: `HawkesAction` (69), `ExpSynapticKernel` (353), `DeltaKernel` (374)
  — `kernel_ft_image()` returns None, `extra_specializations()` returns
  `lambda ns: {ns.g: ns.delta_D}` (385–393). `GTaSNoise` (400) supplies κ²/κ³/κ⁴ (482–486).
  `add_gtas_noise` (591), `mt1` duplicate rationale (606–614). `use_action_template` (665).
- Spatial form factors (Lap→−k², Dt→−iω, ∂_xi→ik_i); `operator_ir()` (1053);
  `_lower_operator_ir_action` (compiler 720); base field-degree ∈{1,2} else
  `NotImplementedError` (784); bilinear transverse `Dx` raises (837); lowering machinery
  in `pipeline/spatial_operator_ir.py` (apply_linearity/kill_means/to_derived_generators/
  classify_generators/Lap/Dt/Dx/GRADX_SYM all present). `boundary` (148, infinite/periodic,
  length required), `initial` (182, stationary-only), `dyson` (206) + `dyson_order` (270),
  `reference_diffusion` (276).
- Compilation core: `_safe_eval` (682) preparse+globals; "user name I shadows imaginary
  unit, intentional" (697–700). Wrapper classes `_FullPhysicalField` (206), `_FieldScalar`
  (268), `_MatrixView` (316), `_IndexedFormalFunction` (73). Unwrap helpers
  `_unwrap_field_arg` (49), `_unwrap_builtin` (342), `_builtin_namespace` (359).
  `make_action_lambda` (851) returns the `_action(ns)` closure. Kill-rule duplicated in
  `make_action_lambda` and `make_mf_bg_conditions_lambda`, "must stay in sync" comment at
  compiler 1378–1382 (verbatim).
- `build()` (1688): autopop gated on `n_populations <= 1` (1721); markovianize-before-compile;
  required-hook validation (`action` always; `phi_concrete` only if a single-arg function
  exists; `kernel_ft_image` optional); reserved-name set
  `{t,omega,Dt,delta_D,delta_Dp}` always + `{k,Laplacian,x,y,z}` spatial (1850–1851);
  `k` silent-wrong-mass comment names `heat_kernel.py` (1842); rename hint `x→phi` (1865).
  Model dict assembled at 1943.
- Model-dict keys: 17 always-present keys match exactly (1943–1973). Conditional keys
  `mf_bg_conditions_action`/`mf_bg_conditions`/`mf_equations`/`kernel_ft_image`/
  `kernel_td_image`/`specializations`/`correlated_noises` (1974–1996). Spatial-only
  `spatial`/`boundary`/`initial` (2002–2025). The "stored twice" mf_bg gotcha
  (closure-baked `_mf_bg` vs solver-friendly `_mf_bg_solver`, with legacy fallback)
  matches 1974–1986. `naming_convention` example shape correct.
- `compute_cumulants` (`pipeline/compute.py:108`) docstring required-hook list
  (response_fields, physical_fields, parameters, kernels, operators, functions,
  mf_substitutions, mf_bg_conditions, specializations, kernel_ft_image, phi_concrete,
  mf_equations, action; correlated_noises optional) — exact match.
- Both example theory files (`theories/ou_quartic.theory.py`, `theories/kpz_1d.theory.py`)
  match their chapter listings (modulo cosmetic line-wrapping in the listings).

## LaTeX validity: PASS (latex_ok = true)

- Environments balanced: note 1/1, gotcha 10/10, defn 7/7, lstlisting 10/10,
  description 6/6, itemize 3/3, enumerate 2/2, equation 3/3, center 2/2, tikzpicture
  1/1, tabular 1/1.
- No unescaped specials (`_ # & % $ ^`) inside `\code{}`/`\file{}`/`\term{}` — all
  identifiers escape `_`→`\_`, `^`→`\^{}`, `#`→`\#`, `{`/`}`→`\{`/`\}`. Bare-`_`/`&`
  hits in the scan are all inside `\[...\]` math (subscripts) or `tabular` cells —
  legal.
- Macros `\code \file \term \msrjd` and envs `note/gotcha/defn` defined in the main
  preamble (lines 98–111); math macros `\Dt \Lap \avg \dd \ii \ee` defined (117–122) and
  used only in math mode; `\toprule/\midrule/\bottomrule` (booktabs), tikz `Stealth`
  (arrows.meta), `right=of`/`below=of` (positioning), `\color` (xcolor), `style=console`
  lstlisting style (preamble 83) — all available. The `\\word` tokens flagged by the
  command scan (`\declarations \lambdas \integrate \hooks`) are tikz node line-breaks
  `\\` + ordinary words, not undefined macros. Chapter is `\input` at main-file line 181,
  so all preamble definitions are in scope.

# Audit: `sections/05-colored-markovian.tex`

Adversarial accuracy + LaTeX-validity audit of Chapter 5 against
`pipeline/colored_to_markovian.py` (and the test/theory/wiring files it cites).

**Verdict: accurate.** Every named identifier exists, every file:line citation
resolves to the claimed content, every behavioural claim matches the code, and
the LaTeX is valid (balanced environments, all macros preamble-defined, no
unescaped specials outside listings/math). No corrections required.

## What was verified (all PASS)

### Identifiers / symbols (all exist)
- `detect_lorentzian` (colored_to_markovian.py:79), signature
  `(kernel_text, coefficient_text, declared_params) -> Optional[dict]` — exact match (manual §6 listing).
- `markovianize_spec(builder) -> None` (line 198) — exact.
- `_pick_aux_field_name(source_nat, taken, additional_taken=())` (line 449) — exact.
- `_inject_aux_coupling(action_text, resp_field, aux_natural)` (line 472) — exact.
- `_sr_to_text(expr)` (line 511), body `return str(SR(expr))` — exact.
- Import line `from sage.all import (SR, sage_eval, exp, abs_symbolic, simplify, sqrt as sage_sqrt,)`
  at lines 70–73 — exact; manual's claim that `simplify`/`sage_sqrt` are imported-but-unused is correct
  (`.simplify_full()` is used instead; no free `simplify(` or `sage_sqrt(` call in the file).
- `msrjd.integration.time_domain.final_integral` — real module (`msrjd/integration/time_domain/final_integral.py`).
- `make_correlated_noises_block` (pipeline/theory_compiler.py:1833) — exists, produces `model['correlated_noises']`.
- `_CGFKernelCallable` (pipeline/theory_compiler.py:1727), ctor `(coeff_text, kernel_text, order)` — matches the
  positional call `_CGFKernelCallable('4*D/tauc^2', 'dirac_delta(tau)', 2)` and the §"Downstream" reference.
- `compute_cumulants` (pipeline/compute.py:108) — exists.
- Builder attrs read/written: `_cgf_terms`, `_markovianize_default` (default `True`, theory.py:765;
  setter `markovianize()` theory.py:429), `parameters` (list of `ParameterSpec.name`), `physical_fields`
  (`FieldSpec.natural_name`/`.name`), `_action_text`, `_markovianize_applied` — all real.
- `physical_field` docstring (theory.py:806+) confirms auto-response `<name>t` AND saddle `<name>star`
  (Piece 1, manual lines 231/610/794).

### File:line citations (spot-checked, all correct)
- Module docstring "pain" quote at 6–10 ✓; SDE system at 13–23 ✓ (lines 15–16); cross-cumulant 33–39 ✓;
  v2 roadmap 51–64 ✓; positivity docstring 91–93 ✓.
- `detect_lorentzian` steps: empty guard L110 ✓; tau-pollution comment 116–127 ✓; eval-ns 128–135 ✓;
  parse 137–141 ✓; top-level-exp 143–148 ✓; decompose 154–161 ✓; tauc 163–170 (L163 "must be POSITIVE") ✓;
  amplitude 174–180 ✓; aux drive L183 (`(SR(2)*c_expr/tauc_expr).simplify_full()`) ✓; return dict 185–192 ✓.
- `markovianize_spec` steps: snapshot/guard 223–225 ✓; gating 227–230 ✓; params L232 ✓; classify 240–276 ✓
  (order!=2 passthrough 250–252 ✓); early-out 278–279 ✓; gather legs 292–298 ✓; invert resp→nat 303–317 ✓;
  single-τc enforce 323–334 (NotImplementedError 328–333) ✓; assign names 339–352 ✓; inject fields 355–360 ✓;
  inject OU eqns 363–367 ✓; build white rows 370–405 (markovianize=False flag L404; Piece-3 397–405) ✓;
  swap L408 ✓; action 424–436 (aux-kinetic append 431–435) ✓; diagnostics 439–443 ✓.
- `_inject_aux_coupling` three-moves 486–508 ✓ (regex `rf'\b{re.escape(resp_field)}\s*\*\s*\('`,
  `start = m.end()-1`, depth-counter, insert `f' - {aux_natural}'`); opt-out remedy comments 418–423 & 481–485 ✓.
- `_sr_to_text` re-eval note 517–519 ✓.
- `build()` wiring theory.py:1735–1737 (`if self._cgf_terms: ... markovianize_spec(self)`), BEFORE
  `_compile_text_declarations()` at 1742 ✓.

### External-file citations (all correct)
- `theories/ou_quartic_colored.theory.py:18` CGF row, `:19` action text — verbatim match;
  `.equation(lhs='(Dt+mu)*x', rhs='-eps*x^3')` at :20 backs the "After: two equations" claim and the
  rendered user equation `(Dt+μ)x = -εx³` (manual line 798).
- `theories/ou_quartic_two_dim_color_corr.theory.py:25–27` (Cxx/Cyy/Cxy) — verbatim match.
- `tests/test_markovianize.py`: aux_drive `'4*D/tauc^2'` at :73 ✓ and tauc_text `'tauc'`+aux_drive at :71–73 ✓;
  `phys_names == ['dx','dxi']` at :116 ✓; `name.endswith('_markov_aux')` at :130 and
  `spec['response_legs'][2] == ['xit','xit']` at :132 ✓; tree-level params μ=0.1,τc=1,D=1 at :177 and
  analytic `2*D/(mu*(mu*tauc+1))` at :190 ✓; degeneracy inline note at :165–168 ✓; opt-out test
  `test_markovianize_opt_out_keeps_colored_row` and detection test `test_lorentzian_kernel_detection` exist;
  the three-verdict checks (match / Gaussian-reject / bare-exp-reject) present at :90/:100.

### Behavioural claims (all match)
- OU stationary variance derivation σ²τc/2 = c·e^{-|τ|/τc}; aux drive = 2c/τc; worked value 4D/τc² — correct.
- Idempotency rests on `markovianize=False` on injected rows (L404) — correct.
- Gating semantics (`.markovianize(False)` builder-off; per-row True asserts/raises ValueError;
  None/'auto' = match-or-silently-pass) — matches L244–274 and L227–230.
- Single-τc → `NotImplementedError`; non-match with explicit `markovianize=True` → `ValueError` — correct.
- `_pick_aux_field_name` fallback chain `xi` → `xi_<src>` → `xi_<src>_n` (n from 2) — correct (L458–469).
- Gotcha "positivity documented but not enforced": code only checks `.has(abs_tau)`/`.has(tau_sym)`,
  never numeric positivity — correct.
- `_lorentz_tau` pollution guard and the 3-row verdict table (e^{-|τ|/τc} match; bare e^{-τ/τc} reject as
  r still has τ; Gaussian reject) — correct.

### LaTeX validity (PASS, latex_ok = true)
- Environments balanced: align 2/2, center 1/1, defn 4/4, description 4/4, enumerate 4/4, equation 5/5,
  gotcha 4/4, itemize 3/3, lstlisting 13/13, note 5/5, quote 1/1, tabular 1/1.
- All custom macros preamble-defined in `daedalus_manual.tex`: `\code` (98), `\file` (99), `\term` (100),
  `\Daedalus` (101), `\msrjd` (102), `\dd/\ii/\ee/\Dt/\avg` (117–122); tcolorboxes `note/gotcha/defn`
  (107–112) with `[1][]` optional-arg signature → chapter's bracket-free `\begin{note}` etc. are valid.
  `booktabs` (26) and `amssymb` (20, for `\checkmark`) loaded; `\lstdefinestyle{console}` (83) and `{py}` (66)
  defined → `[style=console]` listings valid. Chapter `\input` at daedalus_manual.tex:183.
- Special-char scan (tracking verbatim + display-math + multi-line inline-math regions): NO unescaped
  `_`, `#`, or `^` in body text. Identifiers consistently use `\_` and `\^{}` (e.g. `tauc\^{}2`,
  `\_cgf\_terms`, `\_markov\_aux`); regex backslashes use `\textbackslash{}`.

## Minor / non-blocking observations (NOT errors)
- A stale build log `docs/manual/daedalus_manual.log` shows a fatal `'/tcb/Bigrade'` error — that is from a
  PROBE file (`\begin{defn}[Bigrade]`), NOT chapter 5 ("Bigrade" absent from the chapter). The real artifact
  `daedalus_manual.pdf` (1.75 MB) was produced; chapter 5 is not implicated.
- Manual line 132 prose ("resolved legs are `['xit','xit']`") describes the VALUE correctly; the test's
  `response_legs[2]` indexes the order-2 key of a dict (not a Python list index). No reader-facing inaccuracy.
- §"Downstream" presents the order-2 callable as a single `_CGFKernelCallable(...)`; `make_correlated_noises_block`
  in general sums multiple rows per (name,order) into one callable, but for the single auto-cumulant in the
  worked example the one-callable description is exactly right.

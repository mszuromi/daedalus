# Audit: `sections/08-mean-field.tex` vs. code

Scope: `pipeline/_mean_field.py`, `pipeline/_mean_field_dae.py`, plus cross-file
citations into `compute.py`, `_precompute.py`, `theory.py`, `_propagator.py`,
`theory_compiler.py`. Adversarial accuracy + LaTeX-validity pass.

**Verdict: minor-issues.** One major behavioural overstatement; two nits. LaTeX
compiles (all environments balanced, all macros/environments defined in the
`daedalus_manual.tex` preamble, no bare specials outside math/listings).

---

## Findings

- **major** — `_solve_mf_at_saddle`'s purpose is misdescribed.
  - *Manual claim* (ch. lines 154-157): "A second consumer, `_solve_mf_at_saddle`
    (`pipeline/_precompute.py:200`), **re-runs the solver to verify the action's
    linear term vanishes at the saddle**."
  - *Code reality*: `_solve_mf_at_saddle` (`_precompute.py:200`) does re-run the MF
    solver, but only to **obtain the saddle-value dict for the precompute summary**
    (`out['mf_values'] = mf`, `_precompute.py:157-158`); its own docstring says only
    "Returns the saddle values dict." It performs **no** linear-term/tadpole
    verification. The MF-cancellation check is a *separate, earlier* stage —
    Stage 2 `ft.sanity_check(...)` (`_precompute.py:141-145`), whose module docstring
    (`_precompute.py:15`) explicitly reads "Calls `sanity_check()` — confirms the
    action's MF sector [cancels]". `sanity_check` does not route through
    `_solve_mf_at_saddle`. So the symbol and the `:200` line cite are correct, but the
    described *behaviour* (verifying the linear term) is wrong — that is `sanity_check`'s
    job, not this function's.
  - *Location*: ch. lines 154-157 (the "What consumes its output" paragraph).

- **nit** — Legacy-file line count off by one.
  - *Manual claim* (ch. line 27): "`pipeline/_mean_field.py` (553 lines)".
  - *Code reality*: `wc -l` reports **552** lines.
  - *Location*: ch. line 27.

- **nit** — DAE-file line count off by one.
  - *Manual claim* (ch. line 28): "`pipeline/_mean_field_dae.py` (977 lines)".
  - *Code reality*: `wc -l` reports **976** lines.
  - *Location*: ch. line 28.

---

## Verified accurate (spot-checks that PASSED)

Line-precise citations — all confirmed exact:
- `solve_mean_field(ft, model, fundamental, verbose=True)` @ `_mean_field.py:14`; sage
  import `from sage.all import SR, diff` @ `:10`; `from scipy.optimize import fsolve` @ `:11`.
- Hetero-branch routing `if model.get('populations'):` @ `_mean_field.py:40`.
- param-expand listing region `:62-87`, phi-deriv `:90-102`, vstar `:105-118`,
  nstar target `:124-141`, `mf_residual`/`initial_guesses`/`fsolve` loop `:149-179`,
  finiteness `RuntimeError` `:193-197`, `num_params` assembly `:234-258`, phi0-rescue
  loop `:272-279` — every quoted listing matches the source (including
  `initial_guesses = [0.1, 0.5, 1.0, 0.01]·npop` and `if attempt[2] == 1` / `ier==1`).
- `_solve_mean_field_hetero` @ `_mean_field.py:291`; saddle-detection `:338-351`
  (`mean_field=True`), iter/compound classification `:367-375`
  (`model['iteration_saddles']` preferred), `kernel_to_one` over scalar/vector/matrix
  `:387-401`. Hetero return has `phi_deriv_vals: {}` (`:547`) — matches the note.
- DAE: `_sage_to_python` @ `:41`, `_MATH_NS` @ `:33-38` (np-backed tanh/exp/…),
  `_state_variables` @ `:75` (prefers `natural_name`), `_PhiCallableList` @ `:114`,
  `__call__` @ `:134-144` (bare-call needs exactly one pop position),
  `_make_phi_callables` @ `:147` with `if len(args_text) != 1: continue` @ `:174-175`,
  `_numpy_params` @ `:198`, `_build_residual` @ `:236` with the `R(x)` listing `:268-307`
  (Dt=0, Laplacian=0, `__builtins__:{}`, `sum`, 1e10 penalty), PEP-3104 globals comment
  @ `:273-279`.
- `solve_mean_field_dae` @ `:390`, `n_starts` default **64** (`:394`/`:413`),
  `default_rng(rng_seed)` @ `:475`, seed-box domain logic `:336-342`
  (positive → `[0,5s]`, else `[-3s,3s]`, default `'positive'`), multi-start filter
  listing `:481-499` (re-checked residual `>1e-7` reject @ `:492-494`), `_dedup_roots`
  @ `:368` (`diff < atol + rtol*scale`), sort listing `:506-508`.
- Stability toggle/index region: `stab_on` + `selectable_records` + raise `:569-586`,
  clamp `:594-603`, per-root `except Exception` wrap `:542-552`, `stable=None` tag for
  OFF path. `.stability_analysis(...)` builder @ `theory.py:1187`; default False.
- `linear_stability` @ `:631`; matrix build `:798-811` ("∂Dt first to drop the degree"),
  `scipy.linalg.eig(-B_mat, A_mat)` @ `:817`, EIG_TOL=1e-9, stable-iff-all-finite-Re<0,
  unstable list `:831-832`; lazy `from sage.all import SR, diff` @ `:676`, spatial-k
  note @ `:769-773`; `import scipy.linalg` @ `:682`.
- Compat wrapper `solve_mean_field_dae_compat` @ `:853`; saddle-bake listing `:943-950`
  (`getattr(ft._ns, saddle_name, None)`); returns the legacy keys + DAE extras (the
  chapter's list is a faithful subset; code additionally returns `mf_values`).
- Cross-file: routing `if model.get('equations'):` @ `compute.py:364` AND `_precompute.py:204`;
  `compute_poles_and_residues` @ `_propagator.py:851`; `_FieldScalar` class @
  `theory_compiler.py:268`; `_deriv_rename_subs[val] = target` @ `theory_compiler.py:1048`
  with `target = SR.var(f'{tname}{suffix}_{i+1}')` (→ `phi1_<i+1>` etc., matches the
  `phik_<i+1>` claim); `equation(...)` builder @ `theory.py:1087`; "INTENTIONALLY
  decoupled" @ `_mean_field_dae.py:16`.

Data-structure sections: legacy return dict (`nstar_vals/vstar_vals/phi_deriv_vals/
num_params/param_subs` + hetero `saddle_values/saddle_info`) and DAE return dict
(`mf_values/mf_all_roots/mf_stable_roots/mf_unstable_roots/mf_index_used/state_var_order/
n_seeds_converged/stability_analysis`) both match the actual `return {...}` blocks.

## LaTeX validity

`latex_ok = true`.
- Environments balanced: equation 5/5, align 3/3, defn 4/4, gotcha 11/11, note 6/6,
  lstlisting 24/24, itemize 4/4, description 2/2.
- All custom macros/envs used are defined in `daedalus_manual.tex`:
  `\code` (98), `\file` (99), `\term` (100), `\msrjd` (102), `\dd` (117), `\ee` (119);
  `note`/`gotcha`/`defn` are `newtcolorbox` (107/109/111); listing styles `py` (default
  via `\lstset`) and `console` (83) both defined. Chapter is `\input` @ line 191.
- No bare `_ # & $ ^` outside math or listings. Every underscore in prose `\code{...}`/
  `\file{...}` arguments is escaped (`\_`); literal carets use `\^{}`. The multi-line
  `\code{TypeError: ufunc 'bitwise\_xor' not supported}` (lines 586-587) is well-formed
  (argument simply spans a line break). `%` occurrences are LaTeX comments (lines 1-3).
- Cross-refs resolve: `\label{sec:mf-tools}` (952) ↔ refs; `eq:mf-volt/-rate/-closure/
  -taylor/-genfunc` labels ↔ `\eqref`s; `\ref{ch:quickstart}` (→ 02-quickstart.tex:4)
  and `\ref{ch:propagator}` (→ 07-propagator.tex:4) both exist.

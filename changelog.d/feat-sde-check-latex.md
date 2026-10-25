#### `SDE.check()`, `SDE.latex()` and caret parse errors

The SDE front end can now show what it derives, and says where an expression is wrong. This is presentation and
diagnostics only: the action, the mean-field equations and the built models are unchanged (the byte-identical
snapshot of 23 models still passes).

- **`m.check(k=2, max_ell=1, params=None)`** returns a report (a dict, `SDEReport`) and prints it. It holds the
  declarations (fields with response fields, parameters, functions, kernels, operators); the derived action split by
  origin (one term per field and per noise source, with the source type); the mean-field equations; the
  interpretation corrections, each labelled with its source (Stratonovich -> Ito drift terms, Marcus noise terms with
  route and order); the vertex table (numbers of response and field legs, coefficient at the saddle, degree, and the
  loop orders a vertex can enter for `k` external legs); the propagator structure (class, poles, stability, saddle,
  linear operator, noise matrix); and warnings (a truncated Marcus series or noise CGF, terms that enter the mean
  field only, the analyzer's warnings). It integrates nothing; a small model takes about a second (the expansion of
  the action). A model the analyzer cannot expand (a parameter without a default) is reported, not raised.
- **`m.latex()`** returns `{'sde', 'noise', 'action', 'mean_field'}` as LaTeX strings (`api/sde_latex.py`): the SDE as
  written (`dx = (...) dt + ... dW`, `dN` for jump sources, the operator form when the left side is not `Dt x`), with
  `\partial_t` and `\nabla^2` for the operators, `\sum_{j \in E}` for population sums and `x(t - d)` for a delay; the
  derived action; and the stationary mean-field equations. A Stratonovich noise term is written `\circ dW`, a Marcus
  jump `\diamond dN`, and a compound-Poisson source shows its jump law. The output is deterministic across runs and sessions.
  Nothing is compiled.
- **Caret parse errors.** An expression that is not an expression is refused when it is declared (`equation`,
  `drift`, `noise` coefficients, `define_function`, `define_kernel`, `point_process`, `extra_action`, and the
  parameters of a noise source). The message shows the string with a `^` under the failing character, the field or
  noise it belongs to, and a hint: a missing `*`, an unmatched bracket, a Python keyword used as a name, an
  undeclared name (with the closest declared one), an operator written inside a function argument or a noise
  coefficient. The exception types are unchanged (`SDEError`, a `NoiseError` and a `ValueError`). Also new:
  `api.noise.expression_syntax_problem(text)`.
- Tests: `tests/test_sde_check_latex.py` (check on an OU quartic, a Stratonovich multiplicative model, a Marcus jump
  model, a population model and an SPDE through the bridge; deterministic LaTeX; caret errors for typical mistakes),
  and a README example in `tests/test_readme_snippets.py`.
- Known cosmetic difference: a model with a call-style `dt(u)` and an operator acting on an expression (`lap(u^2)`)
  is typeset as `\partial_t u = ... + \dot W` rather than `\mathrm{d}u = (...)\mathrm{d}t + ...`; both are correct.

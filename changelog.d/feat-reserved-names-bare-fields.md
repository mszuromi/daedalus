#### Reserved paper-syntax names; bare external-field names; plain field names documented

- **Reserved names.** The increments of the paper-style syntax cannot name an operator: `dt`, `dW`, `dN`, `dW_<name>`,
  `dN_<name>`, `d<X>_<name>` (and, as before, `d<field>` for every field, in either declaration order).
  `operator('dt', ...)` raises `SDEError` with the hint to use `Dt` (the default). `∂t` / `∂_t` in an equation (or a
  paper-style operator line) is the time derivative, and `operator('∂t', TimeDerivative())` declares the default
  `Dt`. Docs and examples that named the time derivative `dt` now use the default `Dt` (or `D_t` where a rename is
  shown); the tests that renamed it to `dt` use `D_t`.
- **Bare external-field names.** `external_fields=['x', 'x']` (or `['dx', 'dx']`) is `[('x', 1), ('x', 1)]` for a
  field with one unit, in `compute_cumulants`, `dd.Config` / `dd.run`, `covariance_matrix(fields=...)`, the analyzer
  and the YAML `question.external_fields` (`[x, x]`). Results are byte-identical to the pair form. A bare name for a
  field in a population of several units raises `ValueError` (`ConfigError` with the line in YAML) asking for the
  unit; before, `covariance_matrix` silently took unit 1.
- **Documented form.** The plain field name (`'x'`) is the documented form everywhere (README, docstrings, error
  remedies), with the fluctuation name (`'dx'`) an accepted alias. The docs now say that at `k = 1` the result is the
  loop shift `<x> - x*` (tree level 0), that the mean is `mf_values['xstar']` plus the shifts, and that
  `dd.Config(output='moment')` gives the loop-corrected mean (`res['moment']`); and that the values in
  `into={x: c}` are coupling coefficients, not populations.
- Messages: the analyzer's `G0_EXTERNAL_FIELD` remedies name the plain-name and bare-name forms.
- Tests: `tests/test_reserved_names_bare_fields.py`.

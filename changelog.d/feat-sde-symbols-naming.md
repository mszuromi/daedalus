#### SDE front end: declarations return Sage symbols; `into=` / `as_field=`

- **Symbols.** `field`, `parameter`, `operator`, `define_function`, `define_kernel`, `population`, `point_process` (and
  `noise(..., as_field=)`) return the declared name's Sage symbol (`api.sde_symbols.Symbol`, a Sage `Expression`), so
  expressions can be written as Sage objects: `x = m.field('x')`, `mu = m.parameter('mu', default=1.0)`,
  `m.drift(x, -mu*x - eps*x**3).noise('xi', Gaussian(var=2*D), into={x: 1})`. A call of an operator, function or field
  symbol is that call (`dt(x)`, `phi(v)`, `x(t - d)`; never Sage's substitution), `x[i]` / `w[i, j]` the indexed
  name, and `api.sde_symbols.Conv` writes `Conv(K, x)`. Every method that takes an expression or a field name accepts
  a string or a symbol / Sage expression; a Sage expression is read as its printed text (Sage's order of terms), so a
  symbol model equals the string model with that text byte for byte. Strings work unchanged.
- **Chaining is unchanged.** A returned symbol forwards every builder method (`SDE(...).field('x').drift(...)` builds
  as before); the two names Sage expressions also have behave as the builder's (`operator(name, op)`, `show()`;
  `operator()` with no arguments is still Sage's).
- **Renamed keywords** (old names keep working, with a `DeprecationWarning`): `noise(couples=)` -> `into=`,
  `noise(expose=)` -> `as_field=`; giving both spellings is refused. The YAML `sde:` noise entries take `into:`
  (`couples:` still accepted) and a per-source `interpretation:`. Messages and docs use the new names. The config
  error for a noise without couplings now says `'into' must be a mapping` (was `'couples' ...`), and the SDE error for a
  source with neither names `into=` / `as_field=`.
- `dd.Config` describes `k` as the cumulant order (was "correlator order"; the audit's terminology finding).
- Other names from the audit's naming list were reviewed and kept: `Poisson(size=)` / `point_process(size=)` (the jump
  size, consistently in both), `parameter(indexed=)` (shorthand next to `indexed_by=`), the location parameter `mu`
  of the jump laws (the standard name of each law), and `SDE.equation` vs `TemporalModelBuilder.equation` (different
  builders; renaming would break every model file).
- Identity: every catalogue model is byte-identical under the new keywords and under the old ones (golden snapshot).
- Tests: `tests/test_sde_symbols.py`.

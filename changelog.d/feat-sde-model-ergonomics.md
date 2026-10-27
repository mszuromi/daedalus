#### SDE front end: delayed fields, `connectivity`, `observable`, and the text form `to_text` / `from_text`

- **Delayed fields** `x(t - d)` now also work for an indexed field in a population sum
  (`sum(w[i,j]*v[j](t - d) for j in E)`, before: refused) and in a point-process rate (before: refused); each is
  the hard-delay kernel `Conv(delay_d, x)` declared on the fly, as for a scalar field. `sde_text()` writes it back
  as `x(t - d)` (before: `Conv(delay_d, x)`, which did not read back without the kernel). A delay in brackets,
  `x(t - (d))`, is the delay `d`.
- **`m.connectivity(name, W)`**: a weight matrix from a numpy array (nested lists, `tolist()` / `toarray()`
  objects), `w[i,j]` onto unit `i` from unit `j`; `between=(post, pre)` for two populations. It is
  `parameter(name, default=W.tolist(), indexed_by=[...])` (or `indexed='matrix'` for `SDE(n_populations=N)`), with
  the shape checked against the population sizes and finite entries required.
- **`m.observable(name, expr)`**: a function of the fields (and parameters, declared functions) taken by
  `compute_cumulants` as an external leg (`['rate']`, `('rate', u)` for a unit of a population observable; field
  legs may be mixed in). New `api/observables.py`: the composite-operator expansion
  `f(x* + dx) = f* + f_a dx_a + f_ab dx_a dx_b / 2 + ...` assembled from field cumulants, exact at every `k` and loop
  order for an observable linear in the fields; for a nonlinear one the mean to one loop
  (`f_a <dx_a> + f_ab C_ab(0)/2`; `result['observable']['mean']` = `f*` + the shift) and the `k = 2` tree. Higher
  orders need coincident-time cumulants of the composite operator, which the engine does not compute: refused with
  that reason (an `ObservableError`, a `DaedalusError` and a `ValueError`), as are spectra / switch-on / file outputs with observable legs, SPDE observables, and a spike train or
  exposed source entering nonlinearly. The built model gains an `observables` entry only when one is declared
  (every other model is unchanged).
- **`m.to_text()` / `SDE.from_text(text)`** (new `api/sde_text.py`): the model as the YAML `sde:` block of a
  `daedalus run` config, read back through the config's own parser and builder (`api/config_run.py`:
  `sde_block_from_text`, `sde_from_block`). The equations are kept as written (higher-order lhs with `velocity`,
  delays as `x(t - d)`); derived things (velocities, coloured-noise states, delay kernels, the product kernels of
  `Conv(a*b, x)`) are derived again.
  `to_text()` reads its text back and compares action, equations and built dict; a model declared in an order the
  block cannot record raises with the differing order.
- **YAML `sde:` block**: new keys `n_populations`, `space`, `operators`, `populations`, `functions`, `kernels`
  (a `Kernel.gaussian` jitter as `{gaussian: {sigma: s, tol: 1e-6}}`, fitted again on reading),
  `point_processes`, `extra_action`, `boundary`, `dyson`, `observables`; parameter keys `indexed`, `indexed_by`
  (before: `indexed` was an unknown key); field entries with attributes (`- v: {population: E}`); an `equations`
  entry that is an expression is a drift, and an equation takes `velocity`; noise entries take `as_field`,
  `population`, `domain` and a Gaussian's `kernel` (an image, or `{freq_image: ...}`); `noise` is optional with
  `point_processes`, and `into` with `as_field`. Existing configs read as before.
- Tests (`tests/test_sde_model_ergonomics.py`): delayed indexed fields and rates equal the explicit `Conv` forms, also in a
  switch-on (`t_min`) run; a Gaussian jitter kernel round-trips; the
  connectivity equals the parameter call, with its refusals; the one-loop mean of `exp(x/2) + x^2` vs the exact
  stationary density of a 1-d SDE (error ratio 3.9 when `D` halves, O(D^2); saddle only: ratio 2); the k = 2 tree
  `f_a C_ax`; a linear observable at one loop equals the sum of the field cumulants per loop order; a population
  observable (`phi(v)` of the linear Hawkes model) is `n*` at the saddle; every catalogue model and nine adversarial
  models round-trip through the text (digest identical); `to_text()` + a question section is a config whose run
  equals the Python model bit for bit. README snippet registered.

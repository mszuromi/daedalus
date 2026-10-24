#### Declared operators in the SDE front end, and SPDEs from `SDE(..., space=d)`

`SDE` models can now name the operators their equations use, with `m.operator(name, op)` and the operator objects
of the new module `api/operators.py`: `TimeDerivative()` and `Laplacian()`. Each operator object says which engine
paths support it, and how each path writes it. That leaves room for gradient / divergence, biharmonic, fractional
and kernel operators later.

- **Unchanged by default.** `Dt` stays declared as the time derivative. Every existing SDE model gives the same
  action text, mean-field equations and built model dict, byte for byte. `tests/test_declared_operators.py` checks
  23 SDE models (OU, Hawkes, point-process, delay, kernel, Stratonovich, Marcus and jump-source models) against a
  snapshot taken before this change.
- **Two equivalent spellings.** Product style (`(tau*dt + 1)*v`, `D*lap*u`) and call style (`dt(v)`, `lap(u)`).
  `dt(x)` gives exactly the text of `Dt*x`. Declaring the time derivative under another name renames it, and `Dt` is
  then refused. An operator name that collides with a field, a field's `xt` / `dx` / `xstar`, a parameter, a
  function, a kernel or a population is refused, whichever is declared first. So are built-in names.
  `noise(..., into={...})` is the same keyword as `couples=`.
- **SPDEs.** `SDE(name, space=d)` with a `Laplacian()` builds the spatial pipeline's model. It is the same model a
  hand-written `SpatialModelBuilder` gives: the same dict, an action text equal as an expression, and the same
  expanded field theory. The tests check this against seven tracked spatial models: both Allen-Cahn models, the 1-D
  and 2-D reaction-diffusion models, the coupled two-species model, linear diffusion and the conserved model. A
  one-loop k = 2 run of Allen-Cahn written as `dt(phi) = D*lap(phi) - mu*phi - lam*phi^3` equals the hand-written
  model bit for bit. Noise is space-time white, `Gaussian(var=..., space='white')`. `SDE.boundary(...)` and
  `SDE.dyson(...)` pass through to the spatial builder. A Laplacian of an expression (`g*lap(phi^2)`) writes the
  action in the spatial operator IR (`Lap(phi^2)`, `.operator_ir()`).
- **Refusals.** With a Laplacian, temporal-only features are refused with a message, never dropped: populations,
  kernels and delays, point processes and exposed sources, non-Gaussian or time-only noise, the Stratonovich and
  Marcus rules, and the SDE simulator. A Laplacian without `space=`, and spatial noise in a time-only model, are
  refused too. The spatial pipeline's own refusals (periodic loops, switch-on transients, reserved names such as a
  field `x`, UV-divergent subgraphs, the spatial k and loop-order gates, and the `approximate` status of a
  derivative-vertex loop) are the same as for the hand-written models. Conserved noise
  (`Gaussian(..., space='conserved')`) is refused; it is not supported yet. So are an operator acting on no field
  (`lap(mu)`), and an operator (or the engine's `Dt` / `Laplacian` / `Lap` spelling, undeclared) inside a
  `define_function` body or a noise expression.

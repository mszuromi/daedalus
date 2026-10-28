#### Symbolic output: `values=` that move the saddle now feed the mean-field solve

Fixes a silent wrong number. The affected calls are `dd.one_loop_mean`, `dd.one_loop_spectrum`, `dd.mean_relation` and
`dd.tree_covariance`, when called with `mean_field='solve'` and a `values=` entry naming a parameter the saddle
depends on. Their expressions carried that value, but the saddle was solved at the model defaults (or `parameters=`).

Effects on main:
- The `'integral'` and `'residues'` forms of `OneLoopMean.evaluate` came out silently off.
- `'rational'` refused (its pipeline reference saw the mismatch).
- `OneLoopSpectrum.evaluate` refused any such `values=`.

Now the solve runs at `parameters=` (or the defaults) with those `values=` written in. The result is the same as passing
them as `parameters=`.

Old and new values (new = `dd.run(parameters=...)`, and for the first model the hand value):
- `dx = (-x - g x^2 + h) dt + sqrt(2 D) dW` with `values={'g': 0.5}`; the hand value is `-g D / (1 + 2 g x*)^2`:
  - one-loop mean `-9.0000e-3 -> -9.6154e-3`;
  - with `h = 1.5` at evaluation, `-5.5510e-3 -> -6.2500e-3`;
  - the substituted saddle `x* = 0.66667 -> 0.61245`.
- `quadratic_hawkes_alpha` with every parameter in `values=` and `w21 = 0.35` (default 0.2):
  - one-loop mean `dn1 4.2637e-3 -> 5.6231e-3`;
  - `dv2 3.4225e-3 -> 4.5598e-3`.

Now evaluated where main refused (no old number). These match `dd.run(parameters=...)` to <= 2e-15:
- `single_population_quad_exp_test`, every parameter in `values=` with `Em = [1.07, 1.07]`:
  - the one-loop spectrum `(n1, v2)`;
  - the one-loop mean.
  - On main: `saddle_mismatch`, `nstar1 0.340` vs `0.665`.
- `single_population_spike_reset_test`, built with `values=` at `Em`, `tau`, `a` and `w` all scaled by 1.07, then
  `evaluate()` with no values:
  - the one-loop mean, every form;
  - the spectrum `(v1, v2)`.
  - On main: refused with "Em1 was fixed to 1.07 ... cannot be changed to 1.0".
  - An explicit `evaluate` value for a symbol fixed at the build is still refused.

Related changes:
- `TreeCovariance.default_values()` also solves its reference saddle at the `values=` it was built with. Example:
  `dd.tree_covariance(model, values={'g': 0.5})` gives `x* 0.66667 -> 0.61245`.
- `MeanRelation.saddle_at` measures a change against the build's `values=` as well as the defaults, and its re-solve
  keeps them. So `saddle_at({'g': 0.5})` on a relation built with `values={'g': 0.5}` is now `None` (no move), as
  `evaluate` already assumed.
- `MeanRelation.with_defaults` puts the symbols fixed at the build between the defaults and the values given.
- A parameter that has no default may now be given in `values=` alone with `mean_field='solve'`.
- The `too_large` refusal of the rational equal-time parts (more than `max_symbols` symbols) now suggests `values=`.

Without `values=` naming such a parameter, nothing changes.

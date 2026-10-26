#### SDE front end: coloured Gaussian noise (`kernel=`) and correlated vector noise (`GaussianVector`)

- **`Gaussian(var, kernel=h)`**: white noise of intensity `var` filtered by a causal kernel `h`, an `api.delay.Kernel`
  with a rational image (`Kernel.exponential('tau')`, `Kernel.alpha('tau')`, `Kernel.gamma(n, 'mean')`,
  `Kernel.image('(1 + I*omega*a)/(1 + I*omega*tau)^2')`, products). The SDE builder embeds the filter exactly: with
  `H(I omega) = N/D`, the state field `z` (the source's name, or `as_field=`; returned as a symbol) obeys
  `D(Dt) z = xi` (a higher-order `D` gets velocities, as for higher-order equations), `eta = N(Dt) z`, and `c*eta`
  enters the rhs of each field in `into=` (`c` may depend on the fields). Normalisation: `<eta eta>(s) = var (h * h~)(s)`,
  so `Kernel.exponential(tau)` gives the Ornstein-Uhlenbeck noise `var/(2 tau) exp(-|s|/tau)` (white noise of
  intensity `var` as `tau -> 0`). A proper image adds its direct part as white noise into the fields; a mean is
  filtered (`H(0) mean`). Refused: delayed kernels, non-rational or improper images, a constant image, and a
  field-dependent `var`.
- **`GaussianVector(cov=[[...]], mean=None)`**: correlated white noise of `n` components, coupled with
  `into={field: [c_1, ..., c_n]}`; action term `-(1/2) theta^T cov theta`, Stratonovich drift
  `1/2 sum_kl cov_kl sum_j b_jl d_j b_ik`. `cov` must be symmetric, positive semidefinite (checked when numeric) and
  free of the fields. Also accepted in the YAML `sde:` noise block (`type: GaussianVector`).
- The simulator runs both (coloured noise through its state fields; vector noise through `cov = L L^T` at the run's
  parameters); `latex()` shows `c eta dt` terms, vector increments `dW_{xi,k}` and the covariance matrix.
  `sde_text()` refuses both (no paper-style term).
- White Gaussian sources are unchanged (golden snapshot).
- Tests (`tests/test_sde_colored_noise.py`): OU-driven, alpha-kernel and proper-kernel (direct part) models vs exact
  Lyapunov covariances and C(tau) to 1e-11; the OU noise's own covariance; the filtered mean; a state-dependent
  coloured coupling vs the hand-written two-field model at one loop; cross-correlated vector noise vs Lyapunov (all
  four pairs) and the simulator; the vector Stratonovich correction; refusals; YAML.

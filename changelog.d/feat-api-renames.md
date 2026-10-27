#### 0.4.0: one name per concept (audit CON-06)

Six spelling changes. Every old spelling keeps working for one more release with exactly ONE `DeprecationWarning` and
builds the byte-identical model (tests compare the model signatures, equations and parameters); only the `dd.Kernel`
alias of `PairingKernel` is removed.

- **`dd.Kernel` is the kernel-shape builder** (`api.delay.Kernel`: `Kernel.exponential('tau')`, `Kernel.delay(d)`).
  The deprecated alias `Kernel` of `PairingKernel` is removed from `dd` and from `api.symbolic_out`; use
  `dd.PairingKernel` / `api.symbolic_out.PairingKernel` (unchanged).
- **`ModelBuilder.mf_residual(lhs=, rhs=, population=)`** declares one residual of the mean-field system. The builder's
  `.equation(lhs=, rhs=)` is a deprecated alias; `SDE.equation(field, lhs, rhs)` (a field's dynamics) is unchanged. The
  tracked models, notebooks, docs and the model-file writer use the new name; the model-file reader accepts both.
- **`point_process(jump_size=)`** is the jump of the spike train at a spike; `size=` is the deprecated name.
  `population(size=)` (number of units) and `Poisson(size=)` are unchanged.
- **`parameter(indexed_by=[...])`** is the one way to name the indexing. `indexed=True` / `'vector'` / `'matrix'` stays as
  the shorthand for the model's single population; with several named populations it is ambiguous, `build()` warns once
  per parameter (it will be refused in a later release).
- **Wording:** "correlator order" is "cumulant order" in the docs, notebooks and messages (function names such as
  `compute_spatial_correlator_*` keep theirs).
- **Noise vocabulary:** the location parameter is `loc=`: `NormalInverseGaussian(alpha, beta, delta, loc)`,
  `Meixner(alpha, beta, delta, loc)` (was `mu=`) and `VarianceGamma(mu, sigma, nu, loc)` (was `drift=`; `mu` stays: it is
  the drift of the Brownian motion on the gamma clock, not the location). Inline YAML `sde:` noise blocks accept the old
  keys with the same warning. `mu` / `sigma` remain in `GaussianJump` and `LogNormal` (and `VarianceGamma`'s `mu`, `sigma`).
- No numbers move.

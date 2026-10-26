#### Gaussian-jittered exogenous inputs (`Kernel.gaussian`)

- **`Kernel.gaussian(sigma, tol=1e-6)`** (`api/delay.py`, `api/_gaussian_fit.py`): the Gaussian density
  `exp(-t^2/(2 sigma^2))/sqrt(2 pi sigma^2)` as a kernel of an EXOGENOUS input, e.g. Poisson mother events whose
  shots are jittered, `Conv(kX*g, n)`. The density is acausal and its image is not rational; on a stationary input
  no field feeds back into, a time shift of the whole input changes no stationary statistic, so the density is
  shifted by `T` (`Phi(-T/sigma) <= tol/10`), fitted by a rational kernel with L1 error at most `tol`, and the shift is
  dropped (never realized). The fit is universal in units of sigma: a Laguerre expansion of the shifted density,
  balanced truncation for the poles, least-squares numerators with unit area; the kernel string is one fraction with
  exact (dyadic) coefficients, so the mean is exact. Orders: 6 / 12 / 16 / 22 / 27 states at `tol` 1e-2 / 1e-4 / 1e-6
  / 1e-8 / 1e-9 (`tol` in `[1e-9, 1e-2]`), about 0.8 s per `tol`, cached.
- **Error report.** `result['input_jitter'][<input>]['fit']` and the ledger entry `jitter:<input>` (kind
  `approximation`): `l1`, the sampled image error `sup`, the spectral error `sup |(|R|^2 - exp(-sigma^2 omega^2))|`
  and `bound_by_order` (order 1: 0; order 2: the spectral error, relative to the variance driven by the unjittered
  input; order `k >= 3`: `(1 + l1)^k - 1`). `SDE.check()` prints a jitter kernel as a one-line summary.
- **Refusals (by name).** At the build: a jitter on a field that is not a point process / exposed source, a source
  whose rate (or size, or noise parameters) depends on a field (feedback), a use of the input without the jitter or
  with a different one (the shift must be common to every use), an indexed jitter kernel or sigma, two jitters in
  one product. In `compute_cumulants`: the jittered input (or its response field) as an external leg, and switch-on
  runs (`t_min` / `times` / `initial_covariance`). `TemporalModelBuilder.define_kernel(kernel=Kernel.gaussian(...))`
  is refused (the builder cannot check exogeneity). A Gaussian written as `time_expr` is still refused
  (`G3_NONRATIONAL_KERNEL`), whose remedy now points to `Kernel.gaussian`.
- **Realization speed (`api/_realization.py`).** The exact Kalman reduction takes its Krylov ranks modulo a prime and
  certifies them exactly (the pivot columns span an A-invariant subspace containing B iff two exact solves are
  consistent; the echelon basis of that subspace is unique), falling back to the exact ranks otherwise. Results are
  bit-identical; a two-population rate network with alpha synapses and a jittered input: tree 94 s -> 11 s (the
  global observability reduction 80 s -> 0.23 s).
- Tests: `tests/test_gaussian_jitter.py` (the fit's L1 error by an independent quadrature; k = 2 tree against the
  frequency integral with the true `exp(-sigma^2 omega^2)` within the reported bound, at sigma = 0.5, 0.3 and the
  narrow 0.01; k = 3 tree against a time-domain quadrature with exponentially modified Gaussians, within the bound,
  and a loose fit whose error shows; the exact mean; a logistic rate model, tree + one loop, against a simulation
  with the true Gaussian bumps; every refusal), and a README example in `tests/test_readme_snippets.py`.

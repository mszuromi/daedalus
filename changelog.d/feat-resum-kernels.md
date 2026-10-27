#### Resummation: kernels and hard delays are embedded automatically

`dd.hartree`, `Config(resum='hartree')` and `Config(resum='screened')` refused every model with a kernel (`Conv`) or a
delay ("embed the kernel states as fields first"), so a filtered model had to be written stage by stage. A kernel acts
on the fields linearly, so the closure now embeds it before it runs:

- **Rational kernels** (exponential, alpha, sums of these, any rational frequency image) become linear state fields by
  the engine's own exact state-space realization of the kernel entry (the one the state-space Phase J route uses, with
  its diagonal balancing and balanced truncation), not a new embedder. The results are on the model's own fields; the
  closure on the fields and the states is in `result.full`, the states in `result.info['kernel_states']`. The states
  are linear, so they add no quadrature dimension. An alpha-synapse `Conv` equals the model written with its two filter
  stages by hand (Hartree mean, covariance, lag correlator and spectrum to 1e-12; the screened series at k = 2 and
  k = 1 to 1e-8 at one trial propagator).
- **Hard delays** go through their Pade approximant. `delay=` (the keys of `Config.delay`) selects the order like the
  loop path: `n_start`, doubling to `n_max`, the first order that agrees with the one before to `tol` (default 1e-8) in
  the mean, the covariance and the spectra, with the order cached per model and parameter point (`'search'` /
  `'replay'`). A target that is not reached is REFUSED (`HartreeError`: the estimate and the orders tried); an order
  given explicitly is accepted without an estimate. `result.info['delay']` records the order, the estimate and the
  source, and the advisory `HARTREE_DELAY_PADE` is always set (with the change of the lag correlator, which converges
  only algebraically). For a linear model with a delay the closure is exact: its spectrum equals the delay spectrum
  within the reported error (1e-8 relative at order 16). The screened series takes the same `delay=`; with
  `delay={'order': n, 'tree': 'pade'}` its order 1 at the Hartree mass equals the Hartree curve to 5e-15.

Still refused, by name: non-local noise cumulant kernels (declare the noise through its Markovian embedding), a kernel
without a frequency image, spatial models, and exogenous input fields (a point-process input into a kernel, which
includes the Gaussian-jitter inputs: they have no dynamics of their own). The closure's covariance test now accepts a
rounding-size negative eigenvalue (a stiff kernel state can be numerically uncontrollable). No number of a model
without kernels moves (the models of `tests/test_hartree.py` and `tests/test_resum.py` are unchanged).

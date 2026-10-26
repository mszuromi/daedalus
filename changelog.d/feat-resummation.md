#### Opt-in resummation at strong coupling: Hartree and the screened loop series

Two new resummations, both opt-in through `dd.Config(resum=...)`. Without `resum=` every code path and every number
is unchanged.

- `dd.hartree(model, parameters)` and `Config(resum='hartree')` (`api/hartree.py`) compute the self-consistent
  Gaussian of a temporal model. It solves $\langle f\rangle = 0$, $A\Sigma + \Sigma A^{\mathsf T} + \langle Q\rangle = 0$,
  $A = \langle\partial f\rangle$, read off the expanded action. This sums every cactus diagram and is the Gaussian moment
  closure (for a jump model, that of the master equation).
  - Outputs: mean, covariance, dressed drift, decay rates and $C(\tau) = \Sigma e^{A^{\mathsf T}\tau}$. `.spectra()`
    returns the `dd.spectra` keys and conventions. `Config(covariance=True)` gives the `covariance_matrix` layout.
  - It starts from every mean-field root. Several stable solutions give the advisory `HARTREE_MULTIPLE_SOLUTIONS` and a
    `HartreeBistabilityWarning`: one Gaussian describes one basin, and hopping is missing.
  - The advisory `HARTREE_LEFT_SADDLE` is given when the closure started at the selected saddle settles nearer another
    mean-field root.
  - It refuses kernels, delays, spatial models, derivative vertices and singular `Dt` pencils.
  - A non-polynomial drift is Taylor-truncated, with an advisory.
  - Compared with the research prototype, the noise of an action with a `Dt` pencil `M Dt x` is now transformed to
    $M^{-1} Q M^{-\mathsf T}$. For `(tau*Dt + 1)*x` the variance moves from `D tau` to the exact `D / tau`.
- `Config(resum='screened')` (`api/resum.py`, k = 2) re-expands the existing loop orders around a trial propagator
  (optimized perturbation theory).
  - The trial lies on the line `A(kappa) = J0 + kappa (A_H - J0)` from the tree through Hartree. It is injected as a
    bilinear counterterm on a copy of the model (`'<name> [screened]'`, its own caches).
  - `kappa` is fixed by minimal sensitivity, from equal-time runs on Chebyshev nodes in `log kappa`. An even order takes
    the odd order's mass.
  - The orders are built from a central stencil about `kappa` (7 points through N = 3).
  - Outputs: `C_tau` (order N = max_ell), `C_tau_by_order`, `C_tau_spread` (`C_N - C_{N-1}`, the error estimate), the
    plain series in `C_tau_loops_by_ell`, and `resummation` (kappa, its source, the PMS curve, the trial matrices,
    the Hartree closure). `spectra=` and `covariance=` are combined order by order.
  - OU quartic at three loops: equal-time error 8.6e-4 at g = 1 and 4.5e-3 at g = 1000 (plain series: -589 and
    -1.4e13). The lag correlator is within 1 % at g = 1.
- `save_npz` of a resummed result also writes `resum_method`, `resum_kappa`, `C_screened_order_N` and `C_loops_total`
  (`C_total` is the resummed value). The files of plain results are unchanged.
- The G5 loop-parameter advisory (`G5_LOOP_PARAMETER`) now suggests `Config(resum='screened')` among its remedies (text
  only).
- New docs: `docs/resummation.md` and `docs/resummation_validation.py` (the validation tables).

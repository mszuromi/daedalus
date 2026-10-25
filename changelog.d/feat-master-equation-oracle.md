#### Exact master-equation oracle for jump processes (`dd.master_equation`)

`dd.master_equation(transitions, ...)` (`api/master_equation.py`) solves a lattice jump (continuous-time Markov chain)
model exactly, with no simulation, as a validation tool for the jump-noise models of the SDE front end. A process is a
list of `(rate, jump)` transitions on a box of `Z^d`, with rates as strings in the field names, parameters and the
lattice step `h`, or as callables.

- **Stationary law and cumulants.** A sparse LU with iterative refinement gives `p`; `cumulants(kmax)` returns the full
  symmetric tensors for `k = 1..4`.
- **Lag correlations.** `correlation(taus)` gives `C_jk(tau) = <dx_j(0) dx_k(tau)>` (the package's `C_tau` convention).
  It marches `e^{Dt Q}` with `expm_multiply`, or uses shift-and-invert Arnoldi with one LU for the whole grid. The Krylov
  dimension grows until two projections agree to `krylov_tol`, and that last change is reported. `integrated_correlation()`
  gives `int_0^inf C` from one Poisson solve.
- **Diagnostics on every result.** The residual of `Q^T p = 0`, the probability flux suppressed at the box edge (and the
  mass of the edge states), the mass clipped by LU round-off, and the Krylov error. A box that cuts the law gives a
  `TruncationWarning` (`on_truncation='raise'` for an error). Several closed classes (a conservation law), negative rates
  and oversize boxes are refused; transient states are dropped.
- **Diffusion limit.** `kramers_moyal(refine=(1, 2, 4))` solves the central-difference Fokker-Planck generator of the same
  first two jump moments on the lattices `h/m` and extrapolates in `m^-2`. At `m = 1` it is the jump process.
- **Bridge to `compute_cumulants`.** `me.to_sde()` builds the SDE-front-end model with one state-dependent `Poisson`
  source per transition (or the Gaussian diffusion), and `me.compare_with_daedalus()` tabulates, cumulant by cumulant,
  the exact value against the loop-order partial sums. The loop expansion is asymptotic in the jump size `h`: it is
  expected to converge to the oracle as `h -> 0`, and not to match at fixed `h`, near a bifurcation, at boundaries or in
  the tails. See `docs/master_equation.md`.

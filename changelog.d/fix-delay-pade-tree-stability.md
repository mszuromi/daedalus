#### Delay route: delayed filters of spike trains converge at τ = 0, an honest order estimate, no spurious unstable poles

- A kernel `Kernel.delay(d) * Kernel.exponential(tau)` (a delay times a filter of relative degree one) filtering a
  spike train now takes the strictly proper `[n-1/n]` Padé approximant (`result['delay']['proper_kernels']`).
  Before, the all-pass `(n, n)` image `(-1)^n + (strictly proper)` sent its direct term through the filter, so the
  approximated kernel started at `(-1)^n / tau` instead of 0. A point-process covariance holds `Λ K(τ)` at short lags,
  so `C(0)` was off by `(-1)^n w Λ / tau` at every order, in the Padé tree and in the loops. One linear unit
  (μ = 0.6, w = −0.2, d = 1.3, tau = 0.5): `C(0)` was −0.180 / +0.220 at even / odd n against the exact 0.0201; it is
  now 0.0156, 0.0180, 0.0191 at n = 4, 8, 16. One unit with two delays (no exact tree), default order: `C(0)` was
  −0.244 against +0.0335; it is now 0.0299. A nonlinear (exp-rate) unit with such a kernel and one loop, refused at
  every order by the tree-vs-exact guard, now runs, and its loop converges with n.
- The doubling estimate compared orders of one parity (`n // 2`, `n`), in which an error that alternates with the
  parity of n cancels. The order reached is now also compared with the smallest order of the other parity above
  `n // 2` (`result['delay']['parity_probe']`), and `est_rel_error` is the larger of the two. On the two-delay unit
  the estimate went from 0.25 against an actual error of 1.8 to 0.38 against 0.25.
- The refusal "no Padé order is usable" no longer advises raising `n_max` when the Padé tree's distance to the
  exact tree does not shrink with the order.
- Stability of a rational (Padé) model: when a double-precision root of the propagator denominator lies on or below
  the real axis, all roots are recomputed from the exact coefficients in high precision. A stable 3-unit Hawkes
  network with two delays was refused as `G2_UNSTABLE_SADDLE` at Padé n = 16 by about 50 such spurious roots of a
  degree-158 polynomial; it now runs (`C₁₁(0)` = 0.0906 against the exact 0.0925). A truly unstable root stays, and an
  unstable delayed model is still refused by its characteristic function.
- `spectra`: a loop order without a matrix-exponential triple is integrated numerically from its Phase J callable.
  The integration's cost is now predicted from one timed evaluation (`result['spectra']['quad_cost']`), and above
  600 s a `SpectraWarning` says so before it starts. A two-delay model at Padé n = 16, whose realization the
  state-space route declines as ill-conditioned (every evaluation is then a mode sum, 0.5 s), spent more than
  40 minutes in `spectra=` after a 1.7 s run, without a word. The delay record lists each order's share of subsets
  on the state-space route (`result['delay']['statespace_share_by_order']`).

#### Delay route: the error next to a covariance jump is estimated, declined orders are avoided

- A kernel `Kernel.delay(d) * Kernel.exponential(tau)` filtering a spike train gives a covariance that jumps at
  `|τ| = d`. No Padé order has the jump: each smooths it over about `d/n`, so lags next to it carry about half the
  jump at every order, and two orders agree there. The order estimate could not see this. At k = 2 the approximated
  part of the lag correlator (the tree on a Padé tree, the loops on the exact tree) is now fitted near each jump to the
  approximant's own step and ramp responses, from one extra run of the order reached on 21 lags per jump.
  `result['delay']['jump']` holds the fitted jumps, the estimated error per lag (`error_by_tau`) and the lags within
  `2 d/n` of a jump (`near_jump`). `est_rel_error` includes this error, and the `DelayOrderWarning` names the lags.
  On one linear unit (μ = 0.6, w = −0.2, d = 1.3, tau = 0.5, Padé tree) at n = 16 the estimate was 0.167 against an
  actual error of 0.78 (relative to max |C|, at τ = 1.29 / 1.31); it is now 1.08. At n = 8 it is 1.15 against 0.90.
- The automatic order search replaces an order whose realization the state-space route declines as ill-conditioned by
  the first neighbouring order `n − 1, n + 1, n − 2, n + 2` the route takes (`result['delay']['declined_orders']`). A
  declined order runs its subsets on mode sums, and its spectra's quadrature takes minutes. One field with two hard
  delays and multiplicative noise, k = 2 with one loop and `spectra=`: order 16 was declined (spectra advisory "at least
  690 s"). The search now runs order 15 on the state-space route, with exact spectra, in 14 s. A pinned order runs as
  given.
- `dd.hartree` keeps the all-pass `(n, n)` image for a delayed filter of a spike train (`info['delay']['approximant']`,
  `proper_kernels = []`). The closure eliminates its input fields and reports only dynamical fields, which integrate
  the kernel. The all-pass image reproduces the delay of an exogenous input exactly (3e-15). The `[n-1/n]` image
  would leave an error of 1e-3 at n = 8.

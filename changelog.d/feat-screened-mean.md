#### Resummation: the screened series at k = 1 (the resummed mean)

`dd.Config(k=1, max_ell=N, resum='screened')` resums the mean of one field, which was refused (`method='screened'`
accepted `k = 2` only): a rate such as `<phi> = (<v>^2 + Var v) / a` needs the mean as well as the covariance. The
mean uses the same variational trial propagator, the same order-`N` re-expansion and the same PMS choice as `k = 2`.
The PMS criterion is the equal-time covariance for either `k` (`observable='trace_cov'` by default), so a `k = 1` run
and a `k = 2` run of one model at one parameter point share `kappa`, `A_trial` and `Q_trial`; with `k = 1` the scan
still runs the `k = 2` loop orders to find that propagator. The result has the keys of `k = 2` (`C_tau` is the shift
about the saddle, with `C_tau_by_order`, `C_tau_spread`, `C_tau_loops`, `C_tau_loops_by_ell` and the
`resummation` record), plus `mean` and `mean_by_order` (absolute). Against the exact stationary density of
`dx = (-x + 0.3 x^2 - 0.2 x^3) dt + noise` (exact mean 0.09675), the plain series gives 0.150, 0.0638, 0.1056 at
`N = 1, 2, 3` and the screened series 0.121, 0.0941, 0.0965; at `g = 1` (exact 2.608) the plain order 3 is 4.72 and
the screened 2.02. `spectra=` and `covariance=` are refused at `k = 1`; `k >= 3`, kernels, delays, spatial models and
derivative vertices are refused as before. `k = 2` results are unchanged. `api.resum.screened` takes `k=2`;
`check_config` takes `spectra=`.

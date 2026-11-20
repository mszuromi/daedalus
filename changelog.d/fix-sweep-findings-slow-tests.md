#### Delay route: bare-name external fields keep the exact tree; a swallowed propagator timeout takes effect

- `external_fields=['y', 'y']` on a model with delay kernels is now the same run as `[('y', 1), ('y', 1)]`, byte for
  byte. Previously the delay route passed the bare names, unresolved, to the analyzer and the exact delay tree. The
  exact tree raised internally, and the run fell back on the Padé tree: `status='target_missed'`, values 2.6e-6 off
  in relative terms. The `DelayOrderWarning` then blamed the Padé order. Whenever the exact tree is unavailable, the
  warning now also names the reason (`result['delay']['tree_note']`).
- The time budget of the exact propagator inverses now takes effect when its timeout is raised inside a Sage
  callback that ignores exceptions (printed as "Exception ignored in: 'sage.symbolic.expression.py_is_real'"). The
  timeout is raised again every second until it lands, and a result computed after a swallowed raise is discarded.
  On a two-unit network with four indexed Erlang-3 kernels the first call went from 2499 s to 140 s, with the same
  numbers.
- Delay order search at k >= 3: the lazy values are now evaluated once per order at the controller's sample points,
  where they used to be evaluated up to four times. Their cost now counts in the order's recorded wall, so the next
  order's predicted time is honest. On a delayed model at k = 3, one loop, order 4 took more than 300 s while
  recording a few seconds, and order 8 was predicted at 30 s. When the next automatic order is predicted above
  600 s and no `wall_max` is set, a `DelayOrderWarning` now says so before the run and names
  `delay={'order': n}` for the order reached (`result['delay']['cost_advisory']`). The order the search picks is
  unchanged: the estimate alone decides it.
- README: an unindexed point-process input `nX` in a population model is one independent copy per unit; a shared
  input needs a population of size 1.
- Slow tests: the delayed spike/voltage equality test accepts the `target_missed` status that a pinned delay order
  reports above the default `tol`. The two Padé-16 tests have tolerances wide enough for the precision of hosted
  runners (the problem is ill-conditioned, with ‖A‖₁ ~ 1e40). The slow workflow records the runner's CPU and BLAS.

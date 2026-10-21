#### Evaluate-and-discard at k >= 3

`discard=True` / `'auto'` now works for k >= 3 runs whose evaluation points are known up front. Before, every
k >= 3 diagram kept its lazy point callable, about 45 KB each, so a large one-loop run could hit the run-growth
guard even though no single diagram was large.

- **Points.** `compute_cumulants(points=...)` takes the `(n, k)` external-time tuples the result will be read at
  (duplicates are dropped). `tau_grid=[0.0]` with `discard` means the equal-time point. `dd.run` passes the points
  it will read: those of its slices, of the runs with permuted legs (cumulants with legs of different fields), of
  the full grid (`kpoint_full_grid`) and of the moment's cumulant blocks.
- **What is kept.** Each diagram is evaluated at those points right after it is built, one call per point (as a
  `discard=False` result is read). Then it is released and its charge is refunded. The values are bit-identical to
  reading a `discard=False` result at the same points, and the summation order is unchanged.
  `result['C_points']` / `result['C_points_by_ell']` hold the values at `result['points']`. The callables answer
  those points only and raise `DiscardedDiagramsError` anywhere else.
- **`'auto'` mid-run.** As before, `'auto'` turns on when the pre-pass estimate is over the budget and the largest
  diagram fits. It now also turns on mid-run: when the estimate is under the budget, the run starts with the
  callables kept. At the first diagram boundary where measured growth exceeds `DISCARD_AUTO_GROWTH_FRACTION` (0.5)
  of the budget, it switches. The diagrams built so far are evaluated at the points in diagram order and released,
  so the values are the same. This applies to k = 1, 2 as well. A loop order finished before the switch keeps its
  callables. `result['phase_j_memory']['discard_switched_at']` records where the switch happened.
- **Messages.** The growth guard's refusal now says when evaluate-and-discard applies to the run. For a k >= 3 run
  without points, it says what it needs (`points=`). The pre-pass refusal says the same when the largest diagram
  fits.

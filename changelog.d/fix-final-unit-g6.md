#### Analyzer: unit-free accuracy estimate on the state-space route and G5 pole rounding (final check)

Two analyzer gaps found by time-rescaled twins (every rate divided by T, T = 1e-12 .. 1e12). The cumulant values
were already equal to T = 1 within 1e-15 on both engines; only the analyzer's report changed with T.

- **G6 accuracy estimate, state-space engine**: the residual of the stationary resolvent solves is now measured
  componentwise against the size of the terms, `max_i |r + M y|_i / (|M| |y| + |r|)_i`. It used to be
  `|r + M y|_inf / |r|_inf`, which compares entries carrying different powers of the time unit, so it grew linearly
  with T. On a two-field damped rotation, R was 3.9e-15 at T <= 1, 6.3e-10 at T = 1e6 (a false
  `G6_ACCURACY_LOW` warning), 1.2e-6 at 1e9 and 8.1e-3 at 1e12. It is now 3.8e-15 to 8.2e-15 at every T. The
  pole engine was already invariant (4.0e-15 at every T).
- **G5 regime gate** (`_tree_sigmas`): underdamped poles were merged after rounding to 9 decimal places in
  absolute terms, so a model with rates ~1e-12 rounded every pole to 0 and the sort then divided by zero. The error
  was caught as `INTERNAL_ERROR` and the regime check was skipped silently, on both engines. Poles are now merged
  at 1e-9 of the pole scale.

Numbers that move: only the G6 accuracy estimate R of state-space runs. ZOO_RESULT

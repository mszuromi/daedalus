#### Delay kernels: the automatic Pade order is searched once per entry and reused as a start for the others

A delayed model's automatic Pade order search runs the pipeline at `n_start`, `2 n_start`, ... until two orders agree.
Every call used to repeat it for every entry of a covariance matrix, every lag grid and every repeated call (5-9 s per
entry against 0.2 s for white noise). The outcome is now cached in the process, keyed by the model text and index sets,
the full parameter point (delays and saddle included), the question's accuracy and Taylor order, the delay options and
the balanced-truncation tolerance in effect, and then by ENTRY (external fields, lag grid, evaluation points).

- **The same entry again** replays its search: one run at the order it settled on, with the truncation tolerance the
  search used there. The result, `est_rel_error` and `target_met` are those of the search, bit for bit.
- **Another entry** starts its own search at the smallest cached order and the order before it (two runs instead of the
  whole ladder when the cached order is above `n_start`), measures ITS OWN estimate from them, and goes on upward when its
  target is missed; it never reports another entry's estimate. The order it settles on is cached for it.
  `result['delay']['order_source']` is `'search'`, `'replay'` or `'verified'`.
- **A wall-capped search is pinned** per entry: `wall_max` stops a search where the machine's speed put it, and the
  replay does not re-decide.
- **Opt out:** `delay={'order_cache': False}`; `api.delay.clear_order_cache()` empties the cache. Not cached: fixed
  orders, hand-written dict models (no text signature), runs that left diagrams skipped; a replay that fails or skips falls
  back to a full search.
- **No numbers move** for a call that searches (the first call of an entry at a key). A `'verified'` entry starts its
  ladder at the cached order, so its lower order runs at the base truncation tolerance rather than the adaptive one of a
  full ladder: the same estimate to the accuracy of the target (test: 1e-6 relative on the cumulant).

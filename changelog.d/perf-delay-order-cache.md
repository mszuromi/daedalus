#### Delay kernels: the automatic Pade order is searched once per model and parameter point

A delayed model's automatic Pade order search runs the pipeline at `n_start`, `2 n_start`, ... until two orders agree.
Every entry of a covariance matrix, every lag grid and every repeated call used to repeat it (5-9 s per entry against
0.2 s for white noise). The order a search settled on is now cached in the process, keyed by the model text and index
sets, the full parameter point (delays and saddle included), the question's accuracy and Taylor order, the delay
options and the balanced-truncation tolerance in effect. A later call with the same key runs ONCE at that order, with
the truncation tolerance the search used there.

- **Same entry, same numbers.** The replayed run is the run the search made at that order, so the result is bit-identical
  to the uncached automatic run (`result['delay']['est_by_order']`, `est_rel_error` and `target_met` are the search's).
- **Another entry or lag grid** takes the cached order, not its own estimate (it can differ from what its own search would
  pick; `delay={'order_cache': False}` searches every call). `result['delay']['order_from_cache']` says which.
- **A wall-capped search is pinned.** `wall_max` stops a search where the machine's speed put it; the replay does not
  re-decide, so later calls at that key no longer depend on how fast the machine ran them.
- **Not cached:** fixed orders, hand-written dict models (no text signature), searches that left diagrams skipped;
  a replay that fails or skips falls back to a full search. `api.delay.clear_order_cache()` empties the cache.
- **No numbers move** for a call that searches (the first call at a key); `delay` gained the option `order_cache`.

#### One worker knob: `n_workers` reaches the streamed prediagram enumeration (W8p S3)

- `compute_cumulants(parallel=True, n_workers=n)`, and so `dd.Config.n_workers` and the YAML `policy.n_workers`, now
  also set the process count of a streamed prediagram enumeration: a cache miss on an unshipped cell
  (`prediagram_cache.load_prediagrams(..., n_procs=)`, `api._diagrams.enumeration_procs`). Before, only the
  environment variable `DAEDALUS_PREDIAGRAM_PROCS` reached it.
- With `n_workers` unset the enumeration stays in one process, as before; a streamed cell is usually small.
- `DAEDALUS_PREDIAGRAM_PROCS` still overrides when set (kept for scripts; prefer `n_workers`).
- macOS never forks, so there the enumeration is serial whatever the count.
- The certificate set, and so every result, does not depend on the worker count.

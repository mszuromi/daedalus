#### Spawned Sage-free Phase J workers and per-worker memory caps on every pool (W8p S2 + S4)

- **Spawned workers** (`engine.spawn_pool`, `engine._spawn_worker`) are fresh interpreters started with
  `subprocess`. Nothing of the parent is inherited, the user's `__main__` is not re-imported, and they never import
  Sage: about 0.4 s and 40 MB each.
  - They evaluate Phase J diagrams from the numeric state the parent already built (`statespace.diagram_payload`;
    shared model and entry ops travel once per worker). The values are the same bytes as the parent's.
  - Used automatically where fork is not available (macOS scripts, Windows), when a timing sample predicts at least
    `pipeline.SPAWN_MIN_SECONDS` = 3 s of work. Inside a Jupyter kernel they are opt-in (`start_method='spawn'` or
    `DAEDALUS_SPAWN_IN_KERNEL=1`) until the manual notebook checklist in `docs/parallel.md` has passed.
  - Default 4 workers (`n_workers` overrides). Workers run with `OPENBLAS_NUM_THREADS` (and OMP/MKL/VECLIB) = 1.
  - A diagram a worker refuses, or one held by a worker that died, is evaluated in the parent, with its pole-route
    fallback.
- **The Sage-free evaluation core:** `time_domain._light` now holds the runtime counters, the Θ(0) / tie rule for
  constant rows and the Wick-permutation sum (`diagram_sum`). `final_integral` re-exports every name (same
  objects). `integrate_diagram`'s `contribution` and a spawned worker run the same `diagram_sum` and
  `statespace._subset_value`.
- **Memory caps on every pool** (`engine.worker_budget`):
  - Each pool sizes itself from what the run's budget has left after the run's growth so far (W8's
    `run_growth_scope`): fewer workers, then serial.
  - A watcher polls each worker's own footprint (macOS `phys_footprint`; Linux private pages). A spawned worker over
    its share is killed and its diagrams are redone in the parent. A fork pool over its share (Phase J, type
    assignment) is stopped and the stage runs serially. The streamed enumeration stops with
    `PhaseJResourceError`.
  - Counters: `parallel_workers_capped_by_budget`, `parallel_workers_over_share`, `parallel_spawn_batches`,
    `parallel_spawn_diagrams_here`.
- `engine.fork_safety.fork_unsafe` is also True where the platform has no fork (Windows).
- New doc: `docs/parallel.md` (methods, knobs, memory, the notebook checklist).
- Numbers: no public number moves (parallel = serial bytes; the serial path is unchanged).

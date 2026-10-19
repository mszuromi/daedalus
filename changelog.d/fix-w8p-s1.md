#### Parallel Phase J batches: diagram-only partitioning, thread fan-out on every platform (W8p S1)

- `total_C_batch` / `eval_per_diagram_batch` (and so `compute_cumulants(parallel=True)`, the default) now
  partition the batch by diagram only. A worker evaluates whole diagrams on the whole tau grid, and the parent
  sums each point's diagram values in diagram order. The result is the same bytes as `parallel=False` for any
  worker count and method.
  - Before, on Linux, a fork worker evaluated whole tau points without the state-space grid prefetch. That
    differed from the serial batch by rounding (state-space engine).
  - On macOS (S0) the batch was serial.
- **Threads on every platform, notebooks included.** A diagram goes to a worker thread when its largest
  stationary-state block has at least `pipeline.THREAD_BLOCK_MIN` = 256 entries; that work releases the GIL. The
  main thread evaluates the small diagrams meanwhile; on threads they are dispatch-bound and slower.
  - Measured on quadratic_hawkes_alpha k=2 ell=1 (130 diagrams, warm): 2.7-3.0x on the large-block diagrams at
    4 threads.
  - Measured on dendritic_quad_soma_sigmoid (5,212 diagrams, blocks <= 81): no pool starts; threads there would
    run at 0.75-0.86x.
  - Up to `pipeline.DEFAULT_THREADS` = 8 threads; `n_workers` overrides.
- **Fork (Linux only)** takes batches with at least `max(4, 2 n_workers)` diagrams. Its workers now return their
  `_RUNTIME_COUNTERS` deltas, so a run's counters (analyzer G6) stay complete.
- `start_method`: `None` (new default, automatic), `'threads'` or `'fork'`. An explicit `'fork'` on macOS warns
  once and uses the safe method. `'spawn'` raises `ValueError` until spawn workers land.
- `final_integral._RUNTIME_COUNTERS` is a dict subclass. In a worker thread (`final_integral.counter_delta`)
  increments go to a thread-local overlay that the parent merges (`merge_counter_delta`: sums; `*_max` keys take
  the maximum). So no increment is lost to a race.
- Inside a switch-on scope (per thread) the batch stays serial. `DAEDALUS_PARALLEL=0` keeps every batch serial.
- Numbers: no public number moves. Serial is unchanged, and parallel is now bit-identical to it.

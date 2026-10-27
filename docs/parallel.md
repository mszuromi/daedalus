# Parallel evaluation: methods, knobs and safety

`compute_cumulants(parallel=True)` (the default) evaluates in parallel where that helps, using a method that is safe
on your platform. **The numbers do not depend on it:** every parallel path returns the same bytes as
`parallel=False`, for any worker count.

## What runs in parallel

| stage | threads | spawned workers | fork |
|---|---|---|---|
| Phase J τ batch (`total_C_batch`, `eval_per_diagram_batch`) | every platform, notebooks included | every platform, scripts (automatic); notebooks opt-in | Linux, only with `start_method='fork'` |
| type assignment (cache miss) | no | no | Linux only |
| streamed prediagram enumeration (cache miss, `n_workers` given) | no | no | Linux only |
| spatial loop integration (loop order >= 2) | every platform | no | never |

- **Partition by diagram only.** A Phase J worker evaluates whole diagrams on the whole τ grid. The parent sums
  each point's diagram values in diagram order, exactly as the serial batch does, so the result is bit-identical
  for any worker count and method.
- **Threads** take only the diagrams whose largest stationary-state block has at least 256 entries
  (`pipeline.THREAD_BLOCK_MIN`). Their Kronecker / BLAS work releases the GIL: 2.7–3.0x at 4 threads, measured. Small
  diagrams are dispatch-bound and run on the main thread meanwhile. Default: up to 8 threads.
- **Spawned workers** (`daedalus.engine.spawn_pool`) are fresh Python interpreters started with `subprocess`. They never
  import Sage: start-up is about 0.4 s and 40 MB each. They receive each diagram's numeric state, already built by
  the parent (`statespace.diagram_payload`).
  - A diagram that cannot leave the parent stays there: off the state-space route, with a subset hook, or under a
    switch-on. So does any diagram a worker refuses or loses.
  - In automatic mode they start only when a timing sample predicts at least 3 s of work
    (`pipeline.SPAWN_MIN_SECONDS`). Default: 4 workers.
- **Fork** of a Phase J batch is Linux only and only on request (`start_method='fork'`); the automatic method never
  forks, so a Linux run no longer triggers CPython's fork-in-a-multithreaded-process warning there. Type assignment
  and the streamed enumeration still fork on Linux (their workers need Sage). **macOS never forks**: a fork after Cocoa/BLAS/ECL initialisation has hard-crashed
  a Jupyter kernel and the OS, and a Sage process is multithreaded after its first symbolic step. Windows has no
  fork.

## Knobs

- `parallel` (`compute_cumulants`, `dd.Config`, YAML `policy.parallel`): on/off.
- `n_workers`: the pool size of every stage above, including the streamed enumeration.
  `DAEDALUS_PREDIAGRAM_PROCS` still overrides the enumeration's count.
- `total_C_batch(..., start_method=)`: `None` (automatic: threads or spawned workers), `'threads'`, `'spawn'` or
  `'fork'` (Linux; elsewhere it falls back with one warning).
- `DAEDALUS_PARALLEL=0`: every pool off, threads included. Read at call time.
- `DAEDALUS_SPAWN_IN_KERNEL=1`: allow automatic spawned workers inside a Jupyter kernel. An explicit
  `start_method='spawn'` is the other opt-in.

## Memory

Every pool sizes itself from what the run's memory budget (`PHASE_J_MEM_BUDGET_GB`) has left after the run's growth
so far (`daedalus.engine.worker_budget`). Workers that would not fit mean fewer workers, then serial.

A watcher polls each worker process's own footprint: macOS `phys_footprint`; Linux private pages, so pages a forked
child still shares are not counted twice.
- A spawned worker over its share is killed, and its diagrams are evaluated in the parent.
- A fork pool over its share is stopped, and the stage runs serially.
- The streamed enumeration stops with `PhaseJResourceError`.

## Spawned workers in a notebook: manual checklist

Spawned workers start a fresh interpreter with `subprocess`. They do not fork the kernel: this is how Jupyter
itself runs `!cmd`. Still, until this checklist has passed once on a machine, automatic spawned workers stay off
inside a kernel. Run it yourself, with your work saved.

1. *(no risk)* In a fresh kernel: `import daedalus`. Then
   `from daedalus.engine.fork_safety import in_jupyter_kernel; in_jupyter_kernel()` must return `True`.
2. *(no risk: threads only)* Run a model with `parallel=True`, and again with `parallel=False`. Compare `C_tau`
   byte for byte (`np.asarray(a).tobytes() == np.asarray(b).tobytes()`).
3. *(small risk: the first spawned workers from a kernel on this machine)*
   - With `%matplotlib inline` (not the `macosx` backend), call `td['total_C_batch'](points, parallel=True,
     n_workers=2, start_method='spawn')` on a run's Phase J result.
   - Compare it with `parallel=False`.
   - Watch Activity Monitor. If anything hangs, interrupt the kernel: the workers exit with it.
4. *(same as 3)* Repeat after a figure has been drawn inline.

If all four pass, `os.environ['DAEDALUS_SPAWN_IN_KERNEL'] = '1'` enables automatic spawned workers in your
notebooks. Never set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY`, and never try `start_method='fork'` on macOS.

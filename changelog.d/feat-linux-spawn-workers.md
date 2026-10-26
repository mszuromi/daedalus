#### Parallel: the automatic Phase J method no longer forks on Linux

A Phase J τ batch used to fork a worker pool on Linux whenever it had many diagrams (CPython 3.12+ warns on every such
fork of the multithreaded Sage process). The automatic method now takes threads (diagrams with large state blocks) or
the Sage-free spawned workers (a batch a timing sample predicts at 3 s or more; scripts on every platform, notebooks
on request with `DAEDALUS_SPAWN_IN_KERNEL=1`), as macOS and Windows already did. Fork is used on Linux only with
`start_method='fork'`; macOS still never forks. The values are the same bytes as `parallel=False` for every method.

- **Behaviour change (Linux).** A Linux batch of many small diagrams that forked before now runs on the main thread or
  threads unless it is predicted to take 3 s or more (then spawned workers). Type assignment and the streamed
  enumeration still fork on Linux: their workers need Sage (about 5 s and 350 MB each to start), so replacing them
  would cost more than it saves.
- **Tests.** The automatic method with fork allowed (`_fork_unsafe -> False`, i.e. the Linux decision, runs on every
  platform) spawns real workers, never calls the fork path, and matches the serial batch byte for byte; an explicit
  `'fork'` still forks where allowed. The spawn tests (`tests/test_parallel_fanout.py`) already run on the Linux CI.
- No numbers move.

#### Speed-bound tests measure CPU time, not wall time (test hygiene)

`test_analyze.py::test_fast_and_leaves_model_unchanged` (each model under 10 s), the two cold-build tests of
`test_cold_build_jump_models.py`, `test_expand_cache.py` (warm run at least 3x faster than cold),
`test_expand_nonpolynomial.py` (two time budgets) and the refusal-is-instant check of `test_memory_guards_w8.py` measured wall
seconds, so a loaded machine (several test processes, a long pipeline) failed them with unchanged code. They now use
`tests._cputime.CpuClock`: CPU seconds of the test process and its children (Sage's Maxima helper, through `psutil` when it
is installed), with the same limits. No library change.

#### Lazy k >= 3 points stay inside the run's memory budget (final check after audit wave W8)

- For k >= 3, `compute_cumulants` returns callables and the Phase J integrals run when they are called:
  `r['total_C_by_ell'][ell](t1, t2, t3)`. Those calls ran outside every memory scope, with no growth check and with
  each plan checked only on its own. One point of a cubic+quartic 1-D model at k=3, max_ell=3
  (`compute_cumulants` returned at 3.4 GB) then grew past 9.5 GB under `PHASE_J_MEM_BUDGET_GB=8` with no refusal.
  k=4, max_ell=2 did the same, reaching 12 GB.
- The returned `total_C` and `total_C_by_ell[ell]` callables now resume the run's own scope on every call: the
  footprint baseline taken when the run started, the run's single budget reading, and a running Phase J memory
  scope for the plans and states they build. A growth check runs before each state-space subset evaluation.
- A point that grows the process past the budget is refused with `PhaseJResourceError`, and the message names the
  point and the run, e.g. `total_C_by_ell[3](0, 0.3, 1.1) of <model> k=3 max_ell=3`.
- The k=3 max_ell=3 case is now refused at 8.00 GB of growth (8.32 GB peak footprint). The wrapped callables give
  the same values (the wrapped function stays at `.__wrapped__`); without a budget (`off`) they are not wrapped.
- No public number moves.

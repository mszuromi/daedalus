#### Memory guards refuse before the allocation: prediagram records, real growth, forked workers, ME forms, simulators (audit wave W8, part 2)

On an 8 GB machine a run must end in a clean refusal (`PhaseJResourceError`, a `MemoryError`), never a swap storm.
Each item below was reproduced at main first (MEM-08: verified, not fixed).

- **MEM-01, prediagram records.** `load_prediagrams` rebuilds ~11.2 kB of Sage graphs per class. It now refuses a
  cell whose records exceed the budget (`PHASE_J_MEM_BUDGET_GB`, default `default_mem_budget`) before reading or
  computing anything. The manifests' count is exact; an extrapolated count is refused only beyond twice the
  budget; the cert count is checked again once in hand.
  - Before, the shipped cells (2,4) and (5,2) (12.9 / 10.3 GB of records) loaded under any budget: 12 GB in 35 s
    under a 4 GB budget.
  - **Default change: k=2 ell=4 and k=5 ell=2 are refused at the default budget (6 GB at most)**; raise
    `PHASE_J_MEM_BUDGET_GB` to run them.
  - A cold k=4 ell=3 (81 GB of records) is refused in 0 s; it used to enumerate for 8.7 min first.
- **MEM-11 / MEM-02, real growth.** The charges cover plans and stationary states only; typing, setup objects, Sage
  expressions and caches grew runs by 0.1-1.7 GB uncharged on the zoo, and 3-30 GB on stress cases.
  - `compute_cumulants` now holds a `run_growth_scope`: the process footprint (`phys_footprint` on macOS) at the
    start of the run, and the run's single budget reading, which the pre-pass and the running scope now share.
  - `growth_guard` refuses once the footprint has grown by more than the budget. It is checked at every plan
    charge, before every diagram's Phase J integration, and every 64 prediagrams of type assignment (one
    `proc_pid_rusage` call).
  - The k=3 one-loop case (369,741 typed diagrams) grew past 10 GB with no refusal; it is now stopped at 4.00 GB
    growth under a 4 GB budget (diagram #25,235).
- **Item 57 Q4, forked workers.** On Linux each forked Phase J worker builds its own states and charged them to its
  own copy of the total. Each worker now gets an equal share of the budget's remainder (macOS never forks).
- **MEM-05, the pre-pass.** The cheap branch of the state-space pre-pass counted stateless delta lines as subset
  choices (2^k over-estimate since #38); it now counts only the subsets built. Synthetic layout cheap/enumerated
  4.01 -> 1.00; still an upper bound end to end.
- **MEM-06, all-zero edges.** An edge whose residues are all exactly 0 gives the empty plan (no tuples), as an edge
  without modes does; the full product of exact zeros was built and charged before.
- **MEM-07, chain maps.** The dense map between two moving segments of a C0 chain output is refused before
  allocation when over the budget (it was uncharged above 2,000 states: 6.4 GB at D = 2e4).
- **MEM-03, ME form caches.** The dense forms cached on matrix-exponential blocks may hold at most half the budget
  in all; they are released when their block is collected, and rebuilt on demand beyond the cap. One frequency of
  `spectrum_smooth` on a 3-field one-loop result went 1.2 -> 6.3 GB under a 4 GB budget; peak now 3.1 GB, with
  the next frequency 0.6 -> 4.9 s.
- **MEM-09.** `SideME.triple()` refuses a dense block-diagonal triple over the budget before forming it (6.4 GB at
  `MAX_DENSE_TOTAL`). The segment-decomposition cache is bounded in bytes (half the budget), not only by count.
- **MEM-13 / MEM-10, simulators.**
  - `api.pp_simulate.simulate(..., max_bytes=None)` refuses an output over `max_bytes` (default: the guards'
    default budget) before allocating it. T=1e4, dt=1e-3, 100 paths is 8 GB.
  - `simulate_coupled_rd_1d(..., max_bytes=None)` checks its snapshot buffer the same way.
  - The 2-D equal-time estimators (`structure_factor_2d`, `radial_correlator_2d`) transform 64 snapshots at a
    time; they had transformed all at once, 2-3x the snapshots' own memory. Values agree to 1e-15 relative
    (summation order).
- **RED-11, the cache.** A compressed shipped cell is copied to the local cache as it is: 2.3 MB for (2,4), where
  re-pickling it uncompressed wrote 38 MB into every working directory.
- **MEM-08, verified only.** The chain memo tables are empty on the default state-space route after a run (pole
  route: bounded at 200k entries). `_SolveData` and `SymbolicCumulant._ctx` were left as they are.
- **No public number moves.** Zoo bit-identical except base-scatter moves (dendritic_quad_soma_sigmoid 1.5e-16,
  spike_reset k2 l1 grouped 3.4e-16). No zoo entry is refused.
- Paper-session cases unchanged in value and peak footprint: ENSO-like k=3 one loop 1.08 GB; 2-neuron delayed
  spike-only one loop 1.08 GB; two private delay models 0.44 / 0.58 GB.

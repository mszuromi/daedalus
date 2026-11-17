#### Tests: a `bigmem` test the memory guard refuses is skipped, not failed

The `bigmem` tests need more memory than the default budget. When the package's own memory guard
(`SpatialMemoryBudgetError`, `PhaseJResourceError`) refuses one of them at the budget in force, the test is now
reported as skipped, with the guard's message and the variable to raise (`SPATIAL_MEM_BUDGET_GB`,
`PHASE_J_MEM_BUDGET_GB`). A real out-of-memory error, or any other failure, still fails the test. The CI `bigmem`
job therefore reports the three tests that do not fit a standard runner as skipped; they pass with a 20 GB budget.

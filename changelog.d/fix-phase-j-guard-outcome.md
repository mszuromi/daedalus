#### The memory guard always raises: a stopped run is never returned as an incomplete result

`PhaseJResourceError` (the Phase J memory budget, `PHASE_J_MEM_BUDGET_GB`) had two outcomes for the same kind of
run. A three-point cumulant of two fields, `dd.run(..., Config(k=3, external_fields=[u, u, w]))`:
- **Off balance**, the guard stopped the run itself and raised `PhaseJResourceError`.
- **At balance**, the run fitted, but the guard stopped one of the permuted-leg runs that `dd.run` makes for the
  k ≥ 3 slices. The slice evaluation's broad catch turned that into `status = 'incomplete'`, `C_tau = None` and a
  `PhaseJIncompleteWarning` ("evaluating the slices ... failed (PhaseJResourceError(...))").

The guard now raises everywhere:
- `dd.run`'s k ≥ 3 slices and its full k-point grid re-raise a memory refusal (any `MemoryError`), as
  `compute_cumulants` does.
- Covariance derivatives re-raise it unchanged, instead of wrapping it in `CovarianceDerivativeError`.

The choice: `status = 'incomplete'` names diagrams Phase J cannot evaluate, structurally (`result['phase_j_incomplete']`
lists them, and no budget changes it). A run the guard stopped is not that kind of result. Its message says what
would make it run: a larger budget, or `discard=True` for k ≥ 3 with points. The automatic Padé order of a delayed
model keeps its documented exception: a refused HIGHER order is its cost cap, and the completed lower order is
returned with `status = 'target_missed'` and a `DelayOrderWarning`. A refused first order raises.

#### Hartree closure: a given Padé order is estimated

`dd.hartree(model, delay={'order': n})` used to accept the order with "no error estimate". It now also runs `n // 2`
and reports the search's estimate in `info['delay']`: `estimate`, `orders`, and `target_met`. When `tol` is missed,
the `HARTREE_DELAY_PADE` advisory says so. The closure returned at `n` is unchanged, and `'estimate': False` runs
the order alone, as before. This matches `compute_cumulants(delay={'order': n})`.

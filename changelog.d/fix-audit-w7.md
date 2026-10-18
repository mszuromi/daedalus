#### Grouped Phase J retired, test-only references moved out of the library, dead code removed (audit wave W7, part 1)

- **Grouped Phase J is retired (RED-03, ACC-70).** `compute_cumulants(..., use_grouped_phase_j=True)` now emits a
  `FutureWarning` and runs the per-diagram integrator, which gave the same numbers (the grouped path was inert on the
  default state-space engine and, on the pole engine, agreed with the per-diagram path to <= 3.8e-13 while being
  slower). `api/_grouped_phase_j.py` and `engine/integration/time_domain/grouped_integral.py` (~2.1k lines) are
  deleted, with the routing backend `phase_j_grouped` (requesting it by name now raises "not registered") and
  `RouteRequest.grouped`. `result['config']['use_grouped_phase_j']` is always `False`; the saved `phase_j_flags`
  stamp no longer lists `USE_GROUPED_ANALYTIC_MODESUM` / `use_grouped_phase_j` (old files still load).
  A run that passed `use_grouped_phase_j=True` now gets the per-diagram numbers: zoo `spike_reset` k2 l1 moves by
  2.5e-14 relative on the pole engine and 1.4e-15 on the default engine (the grouped path's own rounding). Every
  other zoo entry is bit-identical on both engines (`DAEDALUS_PHASE_J_ENGINE` default and `poles`).
- **Test-only references are out of the library (RED-04).** No production module imported them:
  `engine.integration.symbolic` (frequency-domain Phase H), `engine.integration.spatial.{loop_dyson,
  generic_evaluator, loop_parametric, spatial_reduce, temporal_integrate}` and `engine.enumeration.orderly` now live
  in `tests/oracles/` (their tests use them there); `engine.enumeration.degree_scan`, `engine.core.serialize` (and
  their own tests) and the empty stubs `engine.core.propagator`, `engine.integration.numerical`, `engine.ui` are
  deleted. The routing table no longer lists `phase_h_symbolic` (it had no runner).
- **The frequency-domain reference refuses an undecidable pole (ACC-68).** `tests/oracles/symbolic.py` dropped a pole
  whose half-plane Sage could not decide (a parameter without an assumption) and turned a failed residue into 0, so
  the tree of `1/(i w + b)` returned 0 with status 'ok'; it now raises `PoleSearchError`.
- **`compute_correction_td(kernel_groups=...)` is retired (ACC-67).** The legacy input integrated every diagram of a
  group with the group's summed prefactor (two copies of one diagram gave four times its value); it now raises
  `TypeError` -- pass `typed_diagrams=` and `prefactors=`.
- **`eval_delta_contributions_on_tau_grid` / `_on_2d_grid` are retired (ACC-66).** They put off-grid spikes in the
  edge bin, gave ties to one side, swallowed coefficient errors, and with their defaults returned an all-zero k = 3
  heatmap; `compute_cumulants` never used them. The user-facing delta accessor is `api.cumulant` (`dd.cumulant`).
- **`G_t_entry` uses the recorded time variable (LOUD-09).** With symbolic rates whose names sort before the time
  variable it substituted a rate (`e^{-a t}` at `t -> s` gave `e^{-s t}`); it now reads `'t_var'` from
  `build_G_t_matrix`'s output and guesses only for a bare matrix. No production number moves (the production calls
  substitute numeric parameters first).
- **Dead code and duplicates (RED-06, RED-07).** 18 never-called functions and 10 unused attributes, imports and parameters deleted
  (e.g. `_vertex_combinatorial_factor`, three orphaned layout helpers, `inverse_fourier_transform`, `me_ray`,
  `_complex_quad`, `Preflight.mean_field_failed`); one copy each of the byte-identical `_ensure_warning_shown` (now
  `engine._warn_helpers`), `_iter_multi_indices` and `_opposite_rows_empty`.

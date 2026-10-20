#### Decisions after the final verification (W2, W5, F2, F6)

- **`dd.run` k ≥ 3 slices are evaluated at the exact tie (F2).** A slice point where the swept leg meets another leg
  (τ = 0, a base lag) is now evaluated with every leg at its nominal time, the tie resolved by the engine's tie order
  with the swept leg listed last (`daedalus._kpoint_slice_point`): the left limit of the slice, exactly, as the k = 2
  τ = 0 point has been since #48. Before, the pinned legs at base lag 0 sat at −1e-6 and the swept leg 1e-6 below
  the leg it met, an O(rate × 1e-6) error that was not unit invariant. Legs of different fields take one more
  `compute_cumulants` run with the permuted `external_fields` (only when a tie occurs). The same rule applies to the
  full grid (`kpoint_full_grid`: a leg at τ = 0 is at 0, the tie order decides), the moment outputs
  (`output='moment'` / `'central_moment'`, blocks of 3 or more legs) and `me_output.slice_me` (default `eps` 0).
  `DAEDALUS_TAU0=nudge` restores the old placement for k ≥ 3 too. New result key `_kpoint_slice_orders` (the leg
  order of each slice point). Checked against the exact tree third cumulant of x' = −μx − gx² + √(2D)ξ (one cubic
  vertex, mpmath): 4.8e-16 at every point, tied and untied, base lags [0, 0], [0, 0.5], [0.5, 0] (the nudge: 4e-7);
  with legs of different fields (single_population_spike_reset_test ⟨n₁ v₁ n₂⟩) the tied point equals the left
  limit of its slice, not the right one (a jump of 7x).
  **Numbers that move** (k ≥ 3 slices of `dd.run`, relative to each slice's maximum, τ ∈ {−1, 0, 0.5, 2}):
  ou_quartic k = 4 one loop: 1.1e-7 .. 4.6e-7 (5.7e-12 at τ = 0); single_population_linear_delta_spikes_test k = 3
  tree: 4.9e-9 .. 2.7e-7; single_population_spike_reset_test k = 3 tree ⟨n₁ n₂ n₁⟩: 2.7e-8 .. 2.5e-6;
  ou_quartic_coupled k = 3 (symmetric, identically 0): unchanged. Saved k ≥ 3 slice results move by the same amounts.
  `compute_cumulants`, `total_C` and every k ≤ 2 number are unchanged. Zoo A/B (separate processes, warm-up
  discarded, `use_cache=False`, base main 3053a45): every runnable entry bit-identical except
  `dendritic_quad_soma_sigmoid` 1.5e-16 (main's own cross-process scatter); the k ≥ 3 entries evaluate `total_C` at
  fixed points.
- **`Cumulants([...])` above its declared order warns (W5).** A `Cumulants` source gives cumulants only up to
  `1 + len(kappas)`; higher ones are 0 (a formal truncation). Now `Cumulants.cumulant(m)` above that order warns once
  per source (`api.noise.CumulantTruncationWarning`), `SDE.build()` records `model['noise_declared_orders']`, and
  `compute_cumulants` warns once per source when `k + 2*max_ell` can reach a noise vertex above the declared order.
  No number moves.
- **The simulators read `PHASE_J_MEM_BUDGET_GB` (F6).** `simulate(max_bytes=None)` (point-process front end) and
  `simulations.coupled_rd_1d_sim.simulate_coupled_rd_1d(max_bytes=None)` use the Phase J budget rule: the variable
  when set (`off` / `none` / `inf` / `unlimited`: no limit), else the default budget. Before, an 8 GB setting still
  refused at the 6 GB default. An explicit `max_bytes=` still wins.
- **A tracked two-field OU quartic model (W2).** `models/ou_quartic_coupled.model.py`: two OU variables with cubic
  restoring terms, coupled linearly, white noise (the model of `test_phase_j_setup_fastpath`'s inline twin, same
  numbers bit for bit). The two zoo entries of `tests/tools/phase_j_zoo_baseline.py` that ran a local, untracked model
  now run it (`ou_quartic_coupled-k2-l2-xx`, `-xy`; values bit-identical to before). They have no frozen pre-M1
  baseline value (the legacy baseline test lists them in `_ADDED_AFTER_FREEZE`); their values are pinned in
  `tests/fixtures/zoo_ou_quartic_coupled.npz`, and the analyzer's golden table gains the model. The simulator
  `simulations.ou_langevin_sim_numba.sim_ou_quartic_two_dim_numba` is renamed `sim_ou_quartic_coupled_numba`
  (no alias; scripts that import the old name need a one-line local alias), and the stored model listings of
  `notebooks/model_builder*.ipynb` / `model_runner.ipynb` name the tracked file.

#### Slow tests green again (W5 follow-up)

Tests only; no library code changed and no public number moved. The first scheduled run of
`.github/workflows/slow-tests.yml` failed 13 tests; each one was traced to a merged change:

- **Legacy-baseline entries whose values moved for a documented reason are refrozen** in
  `tests/fixtures/phase_j_legacy_baseline_refrozen_w5s.npz` (the original baseline is never rewritten). Bisected per
  commit: #44 (spatial loops at tau != 0 split the chamber time grid at the kinks) moved `spatial-allen_cahn_1d-l2` by
  <= 1.05e-3, `spatial-reaction_diffusion_conserved_1d-l1` by 3.4e-5 and `spatial-coupled_rd_2species_1d-l1` by
  1.4e-3 relative, at tau = 0.5 only (tau = 0 rows bit-identical); #58 (radial-simplex equal-time rule,
  derivative-vertex kink segments) then moved the Allen-Cahn tau = 0 rows by <= 2.2e-6 and the conserved model by
  <= 4.5e-4 (tau = 0) and 1.7e-5 (tau = 0.5). `spatial-reaction_diffusion_2d-l1` now refuses by design (#44: a
  UV-divergent tadpole at d = 2) and the test expects the `SpatialUVDivergenceError`. The two colored-noise entries'
  model fingerprint changed in #50 (CGF row semantics); their values are unchanged and the test accepts the new
  fingerprint.
- **Two k = 3 spatial tests fit a CI runner's memory budget**: since the kink-split time grid (#44) they needed
  ~11 GB per chamber. `test_k3_spectral_uniform_mass_identity` runs at n_t = 4 (both routes share the time rule;
  residual 4.8e-7 against its 5e-4 bound); `test_k3_coupled_cross_complex_modes` runs its one loop at n_t = 5,
  n_s = 6 (2.7 GB; it checks realness, exact at any grid, and a loose size bound).
- **`test_delayed_alpha_network_voltage_form_bit_identical_to_builder` expects `status='target_missed'`** on both
  forms: the automatic Pade order stops at n_max = 16 with an estimated 1.4e-5 against the 1e-8 target, which #50 marks
  as a missed target (it used to read 'ok'); the two forms still agree bit for bit, status and order included.
- **The two unequal-D Dyson loop tests are `bigmem`** (`test_unequal_D_loops_order3_runs`,
  `test_unequal_D_loop_dressing_ladder`): the coupled one loop at dressing order 2 / 3 needs 6.4 / 10.8 GB per chamber
  at the default grids; before the memory guard (#44) the same tests reached > 18.9 GB, so they never fitted a standard
  runner. (Their NaN in the x-space tree is a separate fix, fix/dyson-nan.)
- **Moves from #59 (W6) documented there**: the two-population spike-reset captures of
  `tests/test_phase_j_subset_hook.py` are refrozen (the saddle solved to round-off moved them by <= 2.9e-11 relative;
  #59 lists spike_reset k2 l1 <= 9.0e-11), and the `linear_hawkes` baseline entry accepts its new model fingerprint
  (#59 GEN-01: an explicit population; values moved 2.2e-16). `ou_quartic_two_dim_color_corr-k2-l1` is compared at
  1e-12 instead of 1e-13: its ell1 differs from the macOS baseline by 4.0e-13 on CI's stack (ubuntu / openblas) and
  has not moved at any commit since the baseline.

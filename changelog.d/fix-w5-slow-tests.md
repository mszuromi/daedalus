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

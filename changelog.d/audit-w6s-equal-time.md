#### Spatial engine: exact equal-time loops, derivative-vertex accuracy, approximate status (audit W6s follow-up)

- **Equal external times (k = 2 at tau = 0, k-point events at one time) use the radial-simplex rule.** Every edge
  weight is homogeneous there, so the radial integral is a closed-form Bessel-K (it carries the heat-kernel factor
  e^{-x^2/4B} that the product grid resolved only jointly in (t, sigma)) and the simplex is integrated spectrally
  (tanh-sinh stick breaking, adaptive order to 1e-9). Plain vertices, simplex dimension <= 4 (every one-loop k = 2
  diagram, the k = 3 tree, the phi^3 two-loop diagrams). Numbers move at tau = 0 only:
  - tracked `allen_cahn_1d_subcritical_infinite` one loop (mu 1.5, D 0.5, T 0.8, lam 0.15): was 1.2e-5 relative off
    the exact tadpole mass shift in the one-loop correction, now 1e-13;
  - zoo `spatial-allen_cahn_1d-l2`: 3 of 14 arrays (the tau = 0 values) move by up to 6.5e-6 relative (the
    (16, 14) grid's error); wall 5.3 -> 18 s;
  - a d = 2 bubble was 6e-4 off at the default grid; the k = 3 tree 1.8e-5.
  Above dimension 4 the grid stays (a 2x refinement only when it fits the memory budget and 2e7 samples, which
  it does not at the default grids). `SPATIAL_EQTIME=0` restores the grid, `SPATIAL_EQTIME_RTOL` (default 1e-9)
  trades accuracy for time (one-loop phi^2 on 64 points: 6 s, the grid 0.4 s at 1e-5);
  `result['spatial_info']['equal_time']` records the orders and changes.
- **Derivative-vertex loops (KPZ, Model B, Burgers), any tau:** the form factor makes the loop-shrink corner an
  integrable rho^(-1/2) singularity, which the product grid resolved like n^-2 (the KPZ C-C bubble at q = 0.5,
  tau = 0 was 6.3 % off at the default grid; fixed 8-node kink segments capped tau != 0 at 2e-3). These diagrams
  now use sigma = s_cap v^6 and kink segments with n_t nodes: the bubble is ~1e-4, the total C ~1e-6. Zoo
  `spatial-reaction_diffusion_conserved_1d-l1`: 2 of 11 arrays move by up to 4.5e-4 relative (toward main's own
  refined grid: main default 1.1e-4 off its (44, 48) value, now 2.7e-5); wall 1.4 -> 3.6 s.
- **A spatial result is `status='ok'` or `'approximate'`, never ok when not certified.** Approximate (with
  `result['spatial_accuracy']`, a `SpatialAccuracyWarning`, G6 `G6_SPATIAL_DERIVATIVE_LOOP` /
  `G6_SPATIAL_EQUAL_TIME`; `policy={'mode': 'strict'}` raises `SpatialAccuracyError`): any derivative-vertex loop
  (with the measured spread against a 0.7x coarser grid), equal-time diagrams left on the grid, coupled loops at
  tau = 0 and coupled k-point events at equal times (zoo `spatial-coupled_rd_2species_1d-l1`: values
  bit-identical, now approximate), `SPATIAL_INTEGRATOR=mc` / `bessel`.
- In a mixed model a diagram whose vertices are all plain no longer gets a (constant) form factor.
- **The UV power count includes derivative-vertex momenta** (`omega -= r/2`, r the even part of the form factor's
  degree along the subgraph's loop directions), at any d. Genuinely divergent loops refuse: the Model B and KPZ
  one-loop bubbles at d = 2 (omega = 0) used to return cutoff-dependent numbers. No d = 1 model changes.
- `compute_spatial_kpoint` derivative records also report the coarse-grid spread.

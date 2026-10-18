#### Spatial engine: population legs, operators, constant drift, Dyson tree, memory guard (audit wave W6s)

Models that were not affected keep their numbers bit for bit (zoo A/B below).

- **Multi-population spatial fields: a leg `(name, population)` is that population.** The leg->field map stripped
  the population digit, so `('p', 2)` computed population 1 (the default route then failed its own certification;
  `certify=False` and the bespoke tree returned population 1 silently). Each population now equals its
  single-population twin bit for bit (tree and one loop) and the closed form to 1e-13. An absent population or a
  bare name of a multi-population field raises; a one-population field still accepts any label.
- **A parameter inside a spatial operator at a non-zero saddle:** `Lap(lam*p^2)` aborted ("bilinear '__mean__'
  has no v1 lowering") while `lam*Lap(p^2)` ran. Any operator argument free of fields and coordinates is a
  constant and is annihilated (`kill_means(keep_syms=)`); the two spellings agree bit for bit and the tree matches
  the closed form with `D_eff = DD - 2*lam*p*` to 1e-12.
- **Constant drift `v*Dx(phi, 0)` is computed (it was ignored or crashed).** A drift shared by every field is
  removed exactly by the Galilean boost: every driver (k = 2 tree and loops, coupled tree and loops, k >= 3)
  evaluates the drift-free theory at `x - v*tau` (convention `C(x, tau) = <phi(0,0) phi(x,tau)>`; the k-point slot
  j at `x_j - v*tau_j`; off-grid `total_C` too). Before: the diagonal tree dropped the drift (`certify=False`,
  9.4e-3 off the exact advected OU), the default route crashed with a TypeError, the k = 2 loop guard used an
  absolute 1e-9, the coupled drivers took `.real` of the purely imaginary `V` and never read the `GradX` symbol
  (the namespace has no `GradX` attribute), so a coupled drift crashed (TypeError). Now: exact advected OU (erfc
  closed form, line and periodic image sum) to 6e-17, and its exact rate-rescaled twins (1e-6, 1e8) to 6e-17.
  Unequal drifts, cross-advection, a real `V` and `GradX^2` terms refuse, with thresholds relative to the velocity
  scale `sqrt(|eig M| |eig D|)`; a drift at d >= 2 refuses.
- **Unequal-diffusion (Dyson) coupled tree: no q-grid.** The dressed tree used a fixed `q_cut = 60` /
  `n_q = 6000` cosine transform (infinite line), a fixed 1201-mode sum (periodic) and a fixed radial grid (d >= 2).
  Each Dyson term's `q -> x` transform is now the Laguerre-Gaussian closed form and only an adaptive,
  error-checked s-integral remains (`dyson_dressing.dressed_tree_C_x`); a periodic box sums images or dual modes to
  convergence. Order 6 matches the exact unequal-D (Lyapunov) tree to 1.4e-8 on the line and at L = 3 and L = 400
  (the order-6 truncation). At order 2 (truncation 3.25e-5) main was 7.0e-5 off on the line and 4.9e-5 at L = 400
  (now 3.25e-5 on both; L = 3 was already converged) and 25-140x slower. `SPATIAL_FORCE_NUMERICAL_FT=1` keeps the grid transform as a
  cross-check.
- **k >= 3 derivative vertices (KPZ, Model B, Burgers) through `compute_cumulants(k>=3, spatial_points=)`:** the
  k-point driver dropped the form factor, so the vertex was integrated as a plain `phi^2` (Model B's conserved
  `c*Lap(p^2)` came out as exactly minus the plain `c*p^2` twin: `int dx1 dx2 kappa3 / int |kappa3|` = +1.00 where
  conservation requires 0; now -0.048 at grid step 0.5 and -0.013 at 0.25, the O(h^2) trapezoid error). The d = 1
  form factor reads each momentum symbol by its index (`q1` used to read `q[0]` when `q0` was absent; same for loop
  symbols on the fallback paths) and builds the multivariate moment over all k-1 external momenta (a form factor in
  `q0` alone had none and refused).
- **Spatial memory guard inside every chamber.** `diagram_kinematic` checks each chamber's exact sample count against
  `full_integrator.chamber_peak_bytes` (sample arrays, Symanzik blocks, the `(P, n_x)` heat kernel, derivative
  moments, the d >= 2 Gauss-Hermite grid `gh^{L d}`), so the k-point driver and the d >= 2 q-path are guarded (they
  ran unguarded: > 5 GB at k = 3, one loop, two points; 14.7 GB at d = 2, two loops, a 7-node grid). The k = 2
  up-front guard uses the same estimate (it priced one `(P, n_x)` array x 2.5, under-counting small `n_x` 2-3x).
  Measured tracemalloc peaks are 0.4-0.9 of the estimate. Runs whose chamber exceeds the budget now refuse
  (`SpatialMemoryBudgetError`); the k = 2 guard is less strict at large `n_x` (it charged 40 x 8 bytes per sample
  per output point, the real cost is ~2.5 x 8).

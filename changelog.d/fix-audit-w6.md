#### One mean-field solver, no spelling-keyed machinery, fixed Hawkes/GTaS templates (audit wave W6)

- **Every model gets the same multi-root mean-field solver (GEN-02).** Models declared with `.set_mf_equation(...)`
  went to a single-root `fsolve` iteration from four fixed unit-scale guesses (8 of the 12 zoo models); now their
  saddle equations `s - rhs(s) = 0` go through the `.equation` solver: every root (`mf_all_roots`),
  `fixed_point_index` selects among them (it was ignored), unit-free seeds, Newton refinement to round-off.
  `api._mean_field.solve_mean_field` is the one front door (`compute_cumulants`, the expand-time sector check, the
  analyzer, the symbolic routes). The same roots are selected on every tracked model; the saddles move by their old
  error (checked against 50-digit roots): ZOO_MOVES. `stability_analysis(True)` on a model with only
  `.set_mf_equation` rows now raises (there are no `Dt` terms to linearize; it was ignored). The solver dict no
  longer carries `nstar_vals` / `vstar_vals` / `phi_deriv_vals` (internal).
- **`n_populations=N` is an explicit population (GEN-01).** `ModelBuilder(name, n_populations=N)` without
  `.population(...)` now declares the population `pop` of size N (indexed fields, parameters, kernels, one-argument
  functions and `.equation` rows without `population=` are indexed by it); the flat-population branch that read
  the names `nstar`/`vstar`/`E`/`g` (and `lambda_X`/`p_part`) is retired. An exact relabeling of the tracked
  `linear_hawkes` model (`n, v, E, g -> r, u, b, k`, or `E` or `g` alone) was refused with "MF sector non-zero"; it
  now gives the same numbers (`linear_hawkes` itself: bit-identical C(tau), saddle = closed form). A two-population
  spatial model with a nonlinear coupling (`.equation` without `population=`) solved population 1 only and failed
  its MF check; both are solved (OU closed form to 1e-13). When the expand-time sector check cannot run its
  numerical check, the error now names the solver's exception (it said 'N/A').
- **Saddle = the builder's `mean_field` flag, one naming service (GEN-03, GEN-04).** New `api.naming`. A real
  parameter named `tstar` was hidden from `dd.param_names` / defaults and refused by `dd.run` (a `star` suffix
  meant "saddle"); a saddle declared under any other name (`rate_mf`) was missed by the symbolic route, the
  1-point mean, the report labels and the builder's auto-derived saddle text. Retired: the `.*star\d*` regex,
  the `'d'`-prefix stripping, the `nstar`/`vstar`/`mstar` result fallbacks, the n/v/m table of `api.access`
  (without a naming convention only internal names are accepted), the `'nstar'` default of the MF text compilers,
  the `'star'` substring test of the operator-IR mean annihilation.
- **Field names ending in digits (GEN-07).** `x1` was the ring variable `dx11`, parsed as `('dx', 11)`: every
  spelling of the leg was refused, and `dx1` pop 1 collided with `dx` pop 11. A base ending in a digit (or `_`) now
  gets a separator (`dx1_1`); letters-only names are unchanged. Digit-named models equal their letter twins bit for
  bit.
- **GTaS / Hawkes templates fixed and tested (GEN-05, GEN-06).** GTaS cumulants are those of the cluster process:
  mixed-cell kappa3 / kappa4 were exactly 0 (Monte Carlo 29.6 +- 2.2 for a Campbell value 30.6 at T = 50; now the
  Campbell integral `lambda p^2` exactly), and the kappa2 cross term ignored the leg order. `HawkesAction`,
  `ExpSynapticKernel`, `DeltaKernel` and `GTaSNoise` read only their own name fields; `add_gtas_noise` declares
  its parameters without values (it filled in 0.25 / 1.7 / 0.6 / 0 / 1 silently); the delta-kernel
  specialization survives any call order. A template-built linear Hawkes model equals the text-built
  `linear_hawkes` bit for bit (also renamed, either order).
- **Order-2 noise kernels are evaluated in their own leg orientation (found with GEN-05).** Phase J read every
  cumulant spec of a noise vertex at the first spec's tau, so an asymmetric cross kernel (`k_ij(tau) !=
  k_ij(-tau)`, e.g. the GTaS shift law) was symmetrized: `<m1(t) m2(t+tau)>` was the average of the two
  orientations. Each spec now carries its orientation (`k_ji(tau) = k_ij(-tau)`); GTaS `<m1 m2>` and `<m2 m1>`
  equal the closed form to 1e-16. Cross-field CGF rows (`response_legs=['xt', 'yt']`) were right (two errors
  cancelled) and are bit-identical.
- **Noise means with Gamma functions (LOUD-02).** `CGMY` (or any library source) with a parameter-named exponent
  stopped with `NameError: factorial` in the saddle solve; `gamma` / `factorial` are now in the solver namespaces
  (mean and `C(0)` equal the closed forms).
- **Symbolic loop route (GEN-17):** derived symbols (`phi0_i`, Taylor coefficients `phi{k}_i`) are no longer
  offered as free parameters.
- **Analyzer (W6s follow-up):** the spatial G3 check looked up `GradX` on the namespace (it lives there as a global
  symbol), so every model with a drift or gradient vertex at a non-zero saddle got `G3_SPATIAL_FORM` ("rate checks
  skipped"); the spatial rate checks now run.

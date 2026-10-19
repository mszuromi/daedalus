#### Fixed: unit-dependent thresholds on the default route (audit wave W4a)

An exact change of units (a field measured in other units, `x -> s x`, or time in other units, `t -> t/a`) no
longer changes a verdict or a number beyond rescaling it. Each absolute threshold below became relative to the
model's own scale; `tests/test_unit_invariance.py` checks twins rescaled by up to 1e+-12 (tracked `ou_quartic`,
`ou_quartic_colored`, inline models; 35 of its cases fail before this change and two more hang).

- **Mean-field DAE solver** (`api/_mean_field_dae.py`): a converged point is a root when each residual row is
  below 1e-7 of that row's linear terms at the point (a fresh finite-difference Jacobian); it was `max|R| <= 1e-7`
  absolute, which rejected every root of a model with rates ~1e9 and, in small field units, accepted points far
  from the root (a two-field model in 1e-6 units: saddle off by 3e-5, correlator by 5e-8). Roots are refined by
  Newton steps to round-off (hybr stopped at ~1e-11: population models agreed with their permuted / unrolled
  twins to 1.2e-10 only, now 6e-15); a component that is zero within its error estimate is set to an exact 0.
  Roots are merged componentwise relative to their size and their error estimates (default `atol` now 0, was
  1e-10: the roots {1, 2, 3}e-9 of a cubic merged into one). The default seed box is unchanged; a second pass
  seeds a box at the scale of every root found far off that box (the cubic in field units 1e-3 found only 3s,
  and `fixed_point_index=0` silently returned 3s instead of s), and when the default box finds nothing a ladder
  of boxes over the decades 1e-12 .. 1e12 is tried (field units 1e-6 gave G2_NO_SADDLE). Roots with a negative
  value in a field of undeclared domain stay outside `fixed_point_index` exactly as before (found from the
  mirrored seeds, with the same warning). A model whose roots lie below a fifth of its largest parameter value can
  therefore find roots it missed before, and `fixed_point_index` counts them (a cubic with roots 0.1, 0.2, 0.3 and
  K = 100 selected 0.3 before, 0.1 now). The linear-stability tolerance is 1e-12 of the model's fastest rate read
  off the pencil (was 1e-9 absolute; not relative to the largest eigenvalue, which a stiff model's fast mode or an
  algebraic mode's QZ value ~1e15 would inflate). An equation that cannot be evaluated returns 1e100 (was 1e10, below the terms of a model with
  rates ~1e11). The result carries `mf_errors` (error estimate of each selected value).
- **Mean-field sector checks** (expand time and the selected root, shared rule
  `engine.core.field_theory.mf_sector_residual`): residual <= 1e-6 x (sum of |terms|) plus the residual the
  saddle's own error estimates explain. Before: an absolute 1e-9 at expand time refused valid models in field
  units >= 1e5 or with rates >= 1e8 ("does not satisfy its own equations"), and the selected-root check's
  absolute floor 1e-7 passed a wrong equation in small units.
- **Analyzer** (`api/analyze.py`): the marginal-mode band is 1e-8 of the largest eigenvalue (or pole) with no
  floor of 1.0 (a stable model in slow time units, rates 1e-9, was refused as G2_MARGINAL_MODE); the spatial
  rate checks compare M, D and M + Vq + Dq^2 each with its own scale; the negative-variance test is relative to
  the covariance (a small noise with a negative variance passed). Unchanged and unit invariant: a mode slower
  than 1e-8 of the fastest is still reported marginal (two decoupled rates 1 and 1e9 are refused in every unit).
- **G2-K Lambert W** (`api/_kernel_stability.py`): a root strictly right of the band is 'unstable' even when
  another sits on the axis (was 'marginal'); the band is 1e-9 of the root plus the 30-digit rounding of the
  model's rates (was absolute 1e-9: slow stable delayed models were 'marginal'); the branch count no longer grows with the time unit (8e7
  branches at a 1e9 rescaling: a hang). The acausal-image and delay-dedup tests are relative.
- **Delta-contact rule** (`Cumulant.integrate_against`, `contact=None`): a kernel's jump at 0 is judged against
  the kernel's own size at 0 (a floor of 1.0 took the jump of a 1e-15 kernel for round-off and used L(0+)).
  `kernel_at_zero(..., with_scale=True)` returns that size.
- **PhaseType** (`api/noise.py`): the singularity test is relative to the sub-generator (rates below 1e-4 were
  refused), the row-sum test to each row (rows of rates ~1e6 failed on +1e-10 of round-off); the MGF is written
  `1 + theta alpha (-theta I - T)^-1 1` (the same function, exactly 1 at 0, formed without Maxima, which
  overflowed at rates ~1e6). A compound Poisson source accepts a float `M(0) = 1 +- 1e-12` with a symbolic rate
  (`CompoundPoisson('lam', Hyperexponential([1/3]*3, ...))` was refused with "K(0) != 0").
- **`symbolic_out.numeric_inverse_transform`**: the contact term is the limit of the even part along the decades
  up to 1e60 (was its value at omega = 1e9), the quadrature runs in omega / omega_s (omega_s the spectrum's
  half-width), and `epsabs=None` (new default) is 1e-12 of the integrand's own size. Before: tau = 0 was 100 % off
  for rates >= 1e6 or <= 1e-6, and 3e307 came back at 1e-9.

- **k = 2 tau grid** (`compute_cumulants`): a grid value is the equal-time point when it is 0 to round-off of the
  grid's own scale, and `total_C(t0, t1)` ties times within 1e-12 relative. Before, an absolute 1e-12: a model in
  fast time units (rates 1e12, lags ~1e-13) returned C(0) at every lag below 1e-12 with status 'ok' (a
  two-field damped rotation: 30-85 % off).

Numbers that move: only models that hit one of the cases above. Zoo A/B: 23 runnable public entries (`use_cache=False`, warm-up discarded, base main c8b869c): 17 bit-identical; `dendritic_quad_soma_sigmoid` 1.5e-16, `multipopulation_test` 1.5e-16, `quadratic_hawkes_alpha` 2.0e-16, `spike_reset` k=1 1.8e-16 and k=2 1.4e-15 (two runs of main differ by the same 1.4e-15; the branch equals one of them); analyzer codes and wall times unchanged; 5 entries refuse identically on both sides (memory budget, UV, pole-free). No zoo number moves beyond main's own scatter.

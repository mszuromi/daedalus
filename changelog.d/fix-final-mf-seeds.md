#### Mean field: seed boxes from the model's own equations, equilibrated Newton steps (final check)

Exactly rescaled twins of two tracked models picked a different mean-field root, or the right root to only ~1e-10.

- **Seed boxes** (`api._mean_field_dae._equation_scales`): each state variable's box now comes from the zero of its
  own equations. Starting from 0, each variable in turn is set to the zero of its rows nearest to 0 along that line,
  bracketed over the decades 1e-30 .. 1e30 and then bisected to full precision, for three sweeps. This scale moves
  exactly with the variable's units, so the seeds do not depend on them. The largest-parameter heuristic is kept
  only for a variable whose rows have no sign change. Before: `single_population_spike_reset_test` with the voltage
  in units of 1e3 (Em, w times 1e3) seeded the spike rates around 1e4 and returned the negative-rate root
  n1* = -3.345, which stopped the run with `G4_NEGATIVE_VARIANCE`. `dendritic_quad_soma_sigmoid` with time in units
  of 1e-3 returned an unstable root (`G2_UNSTABLE_SADDLE`). Both now return the same root as in the original units,
  to 1e-14.
- **Root order without a linearization**: for saddles declared with `.set_mf_equation`, or `.equation` rows
  without Dt terms, nothing classified stability. The roots were then ordered by their value nearest to 0, which
  can select an unstable root now that the search finds every root. They are now ordered by the relaxation
  x' = -R(x) of the saddle equations as written: a root comes first when every eigenvalue of -dR/dx has a negative
  real part (for a rate fixed point n = F(n), eig(dF/dn) < 1). This fixes a regression of the one-solver change:
  the two-population kernel model of `test_frontend_function_kernel_ft.py` was refused with `G2_UNSTABLE_SADDLE`
  (a propagator pole at -0.30i).
- **Newton refinement**: the correction is solved on the Jacobian scaled by powers of two (rows and columns to unit
  maxima). With spike rates ~1e8 next to voltages ~1, the plain least-squares cutoff dropped the small directions.
  The saddle was then right to only 1e-10 .. 2e-9, and two symmetric populations came out different by up to 1e-12.
  It is now right to round-off (1.8e-16).

Numbers that move: the saddle's last digits, and therefore the cumulants at round-off level times the problem's
conditioning. A model whose roots were previously reached from the largest-parameter box may find more roots: they
are listed in `mf_all_roots`, and the selected root comes from the stable-first order. Zoo A/B (23 runnable public
entries, `use_cache=False`, base main a46ddcf): 12 bit-identical; `dendritic_quad_soma_sigmoid` 6.3e-16,
`linear_hawkes` 2.2e-16, `linear_delta_spikes` 2.1e-16, `spike_reset` k=1 3.7e-16 relative. `spike_reset` k=2 moves
by 9.0e-14 at one loop: its saddle moves by 1-2 ulp, and a 40-digit solve puts both main's and this saddle within
2e-16 of the exact root, so this is round-off amplified about 400x by that diagram, not an error in either. Analyzer
codes are identical, and the same roots are selected.

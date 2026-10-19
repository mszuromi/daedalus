#### Unit invariance of the state-space engine and the analyzer (final check)

Found by exactly rescaled twins of the 13 tracked temporal models (time units 1e-9 .. 1e9, field units 1e+-3) and of
inline models. The state-space engine lost real digits in slow time units, and its accuracy estimate R and several
analyzer gates changed with the units. All 13 tracked models now agree with their twins to <= 4e-15 (one loop,
except `ou_quartic_colored` at a time unit of 1e9: 4.8e-13, unchanged from main). R stays within a small factor of
the unscaled run's; the mean-field root choice of two models is fixed separately (fix-final-mf-seeds).

- **Realization in the model's own units** (`api._realization`, `statespace._balance_siso`): the propagator is
  realized in a normalized frequency `s / w0`, where `w0` is a power of two near the fields' own rates (the median
  `|K0/K1|`). Its rows and columns are equilibrated by powers of two (the fields' units), and the result is mapped
  back exactly. Each entry's augmented balancing now compares `|a|` and `|b||c|` relative to their own sizes, not
  raw. Before, the coefficients of the realization carried different powers of the time unit, and balancing left
  the entries graded (`a_21 = 6e-4` next to rates 1e-9). The exponentials then lost digits: the two-field colored
  OU tree was 6e-11 off at a time unit of 1e-9, and the `single_population_quad_exp_test` one loop 3.4e-8 off. Both
  are now within 1e-15, as on the pole engine. In fast time units, a model whose spike rates carry the time unit
  and whose voltages do not no longer looks 'graded'. Its a posteriori check had added a spurious 5e-4 .. 8e-4 to R,
  raising `G6_ACCURACY_LOW` on an exact result (`linear_hawkes`, `multipopulation_test`, `quadratic_hawkes_alpha`,
  `single_population_quad_exp_test` at a time unit of 1e6 and up).
- **G6 accuracy estimate per run** (`api.compute`): the error-estimate sums are reset around a run and restored
  afterwards, as the maxima already were. Before, the run's sum was the difference of two process-wide totals, so a
  run in slow units after one in unit time got R = 0.
- **G5 regime gate** (`_tree_sigmas`): underdamped poles were merged after rounding to 9 decimal places in
  absolute terms, so a model with rates ~1e-12 rounded every pole to 0 and the sort then divided by zero. The error
  was caught as `INTERNAL_ERROR` and the regime check was skipped silently, on both engines. Poles are now merged
  at 1e-9 of the pole scale.
- **Saddle stability with fields in very different units** (`api._mean_field_dae.linear_stability`): the linearized
  pencil is equilibrated before QZ (rows and columns scaled by powers of two, which leaves its eigenvalues unchanged).
  With three fields in units 2e-6, 2.3e5 and 3.1e3, the entries spanned about 1e-11 .. 1e11. QZ is backward stable
  only normwise, so it lost the small entries: the exactly rescaled stable saddle was refused as
  `G2_UNSTABLE_SADDLE`, with eigenvalues `[+0.888, -2.94, -1.15]` instead of all negative. The eigenvalues now agree
  with the unscaled model to 1e-9, and the saddle to 1e-12.

Numbers that move: state-space results at round-off level (the realizations are formed differently), the G6
accuracy estimate R, and the linearization eigenvalues at round-off level. Zoo A/B (23 runnable public entries,
`use_cache=False`, base main a46ddcf): 15 bit-identical; `quad_exp` 2.3e-15, `dendritic_quad_soma_sigmoid`
1.5e-16, `quadratic_hawkes_alpha` 1.7e-16, `spike_reset` k=2 4.5e-16 relative; analyzer codes identical; R moves
at most 1.4 % (`quad_exp`, `quadratic_hawkes_alpha`); the 5 entries that refuse do so identically on both sides.

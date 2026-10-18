#### Mean-field roots do not depend on the field's units; stable roots first (audit wave W6, follow-up)

- **Every root is found and selectable in every unit.** A field with no declared domain is seeded on its
  non-negative half line; a negative root was selectable only when a seed happened to overshoot to it, which
  depends on the units: the cubic drift `-(x - 0.3 s)(x + 1.1 s)(x - 1.7 s)/s^2` had the roots -1.1, 0.3, 1.7 at
  s = 1 but lost -1.1 s at s = 1e-5 (and `fixed_point_index` 1 and 2 both returned 1.7e-5). The mirrored seeds'
  roots (they were only reported, `mf_unsearched_roots`, now always empty) are selectable like any other. With
  `domain='real'` the root 1.7 s was lost at s >= 1e5: the second-pass seed boxes are now repeated (up to 3
  rounds) for the root scales they discover. The roots, their order and the selection are now identical up to
  the factor s for s = 1e-8 .. 1e8.
- **`fixed_point_index` with stability analysis off: the linearly stable roots come first** (each group ordered
  by the first field's distance from 0, ties ascending; a change of units keeps the order), wherever the linearization decides (`.equation` rows with `Dt` terms, no
  kernels): the default index 0 is a stable root whenever one exists. Before, roots were ascending on the axis, so
  index 0 could be an unstable root. Where the linearization does not decide (kernels, `.set_mf_equation`
  models), the order is unchanged and the roots only the mirrored seeds reach follow the others. Every tracked
  model selects the same root as before and the zoo is unchanged (bit-identical; dendritic_quad_soma_sigmoid
  1.5e-16, run-to-run scatter); `reaction_diffusion_2d` and `reaction_diffusion_quadratic_1d` list their unstable
  negative root second (2 saddles instead of 1). Tests that selected by the ascending order were updated.
- **`.set_mf_equation` models: the iteration saddles are the unknowns again** (the compound ones substituted),
  the full system only when that finds no root. Since the W6 single solver the full system was searched first, so
  a compound saddle with no declared domain (a voltage, negative at the stable root) was seeded on its
  non-negative box: a two-neuron exponential Hawkes model written with its voltage selected the unstable upper
  rate (n* 3.407 vs the stable 0.2789) while its spike-only twin selected the stable one. Both select 0.2789
  again (the pre-W6 root). The same model written with `.equation` rows (the full system) selects it too: a
  kernel symbol in an `.equation` row is now its integral (it raised NameError), and the stable-first order uses
  the static linearization (each kernel at its integral) for models with kernels as well (ordering only: the
  unstable-saddle refusal still skips them). Tracked `.set_mf_equation` models: same roots, saddles still at round-off (50-digit
  check), C(tau) within 1.5e-16 of main on the A/B grid; zoo spike_reset k2 l1 (and grouped) 1.3e-14 (ulp-level saddle
  differences amplified by the one loop), every other entry bit-identical or <= 5.5e-16.
- **An unstable selected root is refused by name** with the analyzer off (`preflight=False`):
  `UnstableSaddleError` names the stable indices (it surfaced as `PoleFreePropagatorError`). With the analyzer on,
  `G2_UNSTABLE_SADDLE` refuses it as before.

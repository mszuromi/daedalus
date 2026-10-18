#### Mean-field roots do not depend on the field's units; stable roots first (audit wave W6, follow-up)

- **Every root is found and selectable in every unit.** A field with no declared domain is seeded on its
  non-negative half line; a negative root was selectable only when a seed happened to overshoot to it, which
  depends on the units: the cubic drift `-(x - 0.3 s)(x + 1.1 s)(x - 1.7 s)/s^2` had the roots -1.1, 0.3, 1.7 at
  s = 1 but lost -1.1 s at s = 1e-5 (and `fixed_point_index` 1 and 2 both returned 1.7e-5). The mirrored seeds'
  roots (they were only reported, `mf_unsearched_roots`, now always empty) are selectable like any other. With
  `domain='real'` the root 1.7 s was lost at s >= 1e5: the second-pass seed boxes are now repeated (up to 3
  rounds) for the root scales they discover. The roots, their order and the selection are now identical up to
  the factor s for s = 1e-8 .. 1e8.
- **`fixed_point_index` with stability analysis off: the linearly stable roots come first** (each group
  ascending by the first field), wherever the linearization decides (`.equation` rows with `Dt` terms, no
  kernels): the default index 0 is a stable root whenever one exists. Before, roots were ascending on the axis, so
  index 0 could be an unstable root. Where the linearization does not decide (kernels, `.set_mf_equation`
  models), the order is unchanged and the roots only the mirrored seeds reach follow the others. Every tracked
  model selects the same root as before and the zoo is unchanged (bit-identical; dendritic_quad_soma_sigmoid
  1.5e-16, run-to-run scatter); `reaction_diffusion_2d` and `reaction_diffusion_quadratic_1d` list their unstable
  negative root second (2 saddles instead of 1). Tests that selected by the ascending order were updated.
- **An unstable selected root is refused by name** with the analyzer off (`preflight=False`):
  `UnstableSaddleError` names the stable indices (it surfaced as `PoleFreePropagatorError`). With the analyzer on,
  `G2_UNSTABLE_SADDLE` refuses it as before.

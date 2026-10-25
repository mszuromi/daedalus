#### Spatial operator IR: a derivative vertex and a local vertex of the same fields stay separate

In a spatial model written in the operator IR (`.operator_ir()` with `Lap(...)` / `Dx(...)` in the action, or an
`SDE(..., space=d)` with `lap(u^2)`), a derivative vertex and a plain local vertex with the same fields, such as
`c1*Lap(phi^2)` and `c2*phi^2` (both `phit*phi^2`), were merged into one vertex with coefficient `c1 + c2`. The
merged vertex carried the derivative vertex's momentum form factor. So `c1 = -c2` removed the vertex (the loop run
failed with "no live loop diagrams" and the k = 3 tree was 0), and any other pair put the Laplacian on the local
coupling too, without a warning. The same happened to the local `3*lam*phi*·phit·phi^2` piece of a `lam*phi^3` term at
a non-zero saddle, next to a `Lap(phi^2)`, `Dx(phi^2)` or `Dx(phi)^2` vertex.

- **Fix.** Each term is its own species in the vertex table (`ns._operator_ir_vertex_terms`); a local species has
  form factor 1. When a vertex holds several species, the unfolded action carries the norm of their couplings,
  `sqrt(sum c_t^2)`, and each species carries the weight `c_t / sqrt(sum c_t^2)`. Couplings that sum to zero then
  no longer cancel the vertex. Terms of one species that stay apart after expansion (`c2a*phi^2 + c2b*phi^2`, or
  `c2*phi^2` and the `3*lam*phi*` piece of `lam*phi^3`) are added up. This also covers two derivative vertices of
  the same fields (Model B and KPZ, or a `Lap(phi^2)` and a `Dx(phi^2)` that cancel).
- **Unchanged.** A model with one species per vertex gives the same lowered action, the same table and the same
  numbers as before: the conserved, KPZ and Burgers models, and any model with only a local or only a derivative
  `phit*phi^2` vertex. A table weight that does not evaluate (free symbols left) is now an error, not a silent 0.
- **Refused** (they used to give wrong numbers silently): a derivative term times further plain fields
  (`phit*phi*Lap(phi)`), a derivative term with more than one response leg or an operator on a response field, and a
  composite derivative raised to a power (`phit*Lap(phi^2)^2`).
- **Moved numbers.** Hand-written mixed models (`mu = D = T = 1`; the k = 2 one-loop term at `x = 0, 1, 2`,
  tau = 0; the k = 3 tree at the slot pairs `(x, tau)` = `(0.5, 0.3), (-0.4, 0)`; `(1.2, 0.3), (0.2, 0)`;
  `(0, 0.6), (0.9, 0.2)`):

  | model | before | after |
  |---|---|---|
  | `c1 = 0.3, c2 = -0.3`, k = 2 loop | error: no live loop diagrams | 0.10310, 0.03058, 0.01065 |
  | `c1 = 0.3, c2 = 0.2`, k = 2 loop | 0.08233, -0.00816, -0.00722 | 0.01678, 0.00204, 0.00408 |
  | `c1 = 0, c2 = 0.2`, k = 2 loop | 0.01317, -0.00131, -0.00116 | 0.01444, 0.01093, 0.00621 |
  | `c1 = 0.3, c2 = 0.2`, k = 3 tree | -0.02826, -0.01694, -0.01459 | 0.00385, 0.00707, 0.00655 |

  The tracked combined Allen-Cahn + Model B + KPZ model now has three species in its `phit*phi^2` vertex (the
  third is the `phi^3` piece, 0 at its `phi* = 0` saddle). Its k = 3 tree moves in the last two digits (relative
  1e-15); its k = 2 loop is refused as UV-divergent before and after. A KPZ vertex with `lam*phi^3` at a non-zero
  saddle now refuses its k = 2 loop as UV-divergent (the `<(Dx phi)^2>` tadpole feeds the mass through
  `6*lam*phi*`); before, a finite number came from putting the KPZ form factor on the local piece. No tracked
  notebook or documented number uses such a pair.
- **Checks** (`tests/test_operator_ir_vertex_species.py`): the k = 3 tree against a direct Fourier quadrature of the
  second-order response (to 5e-10); the k = 2 tree at a non-zero saddle against the closed form with
  `m = mu - 2 c2 phi*`, `D_eff = D + 2 c1 phi*` (1e-17); the one-loop term against the lattice simulator, from paired
  runs (`slow`; within the statistical error); a KPZ gradient vertex with `c2*phi^2` and `lam*phi^3` at a non-zero
  saddle, and a split local coupling, against the same quadrature; one species switched off against the one-species
  model; and the one-species models' lowered actions, tables and loop values against values recorded before the
  change.

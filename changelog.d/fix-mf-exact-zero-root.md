#### Mean field: a saddle at an exact 0 whose rows cancel to round-off is found

`solve_mean_field_dae` refused a model whose exact saddle is the origin when the drift rows only cancel there to
round-off, for example `(a - c) + c e^{-x} - a e^{x}` with the constants split across several rate terms. Every run
stopped at `G2_NO_SADDLE` ("no MF fixed point found from 64 seeds"). This hit the SDE front end with
state-dependent `Poisson` sources coupled to several fields, and hand-written `TemporalModelBuilder` rows in the
same form. There were two causes.

- **Acceptance.** A root was accepted when each residual row is below `1e-7` of the row's linear scale
  `sum_j |J_kj| |x_j|`. At a root whose components are 0, that scale is 0 (or round-off), so a residual of a few
  ulps was refused. A row whose linear terms are themselves round-off is now satisfied when its residual is
  round-off of its own terms (16 ulps of `|lhs| + sum |rhs terms|`, the allowance the zero snap and the
  finite-difference Jacobian already use). Rows with a linear scale above round-off keep the relative test.
  Accepting round-off there would admit points far out in a decaying drift's flat tail, where `x = x + f(x)` sits at
  the round-off of `|x|`.
- **The hybrid solve.** `hybr` stops on a step relative to `x`. Converging to an exact 0 against a round-off
  residual floor, it never passes that test and reports no progress, so the start was dropped. Such a point is now
  kept only when the whole vector collapsed toward 0 (below `1e-8` of the seed) and every row is at its round-off;
  the refinement and acceptance above decide.

New tests in `tests/test_mf_exact_zero_root.py`: a two-node shot-noise model with the saddle at `(0, 0)` to
round-off, built both in the SDE front end and by hand with `set_action_text`. The one-loop results agree, and the
tree matches the Lyapunov covariance to `1e-10`. Two rate splits of the same drift are covered. Further tests cover
a round-off zero next to a nonzero field, and a round-off zero among other roots. They also check that 0 is still
refused when it is not a root, that a shifted root with terms `1e3` times the drift is found, and that no flat-tail
point displaces the stable root of a bistable drift.

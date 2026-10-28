#### Symbolic output: the one-loop equal-time covariance, diagram by diagram

`OneLoopSpectrum.equal_time(a, b, values=None, c0='rational')` returns `[(term, value)]`, one value per self-energy
diagram that reaches `(a, b)`. Each value is that diagram's part of `C^(1)_ab(tau = 0) = (1/2 pi) int S^(1)_ab(w) dw`.
The values sum to the `k = 2`, `ell = 1` value at `tau = 0` of both `dd.run` and `dd.symbolic_cumulant`. Checked to
<= 2e-15 on:
- the two-unit rate network (56 diagrams);
- a cross-coupled SDE;
- `single_population_spike_reset_test` (`v1, v2`, 82 diagrams);
- a model with a white-noise partner field.

How the values are computed:
- A bubble is a double integral over `w` and `nu`.
- All diagrams of a pair are integrated together, each to `epsrel` (default `1e-10`) of the largest.
- `values=` and the saddle rule are as in `evaluate`.

Refused: a pair whose outer lines together decay slower than `w^-2`, such as a spike train's own covariance. There
`C^(1)_ab(tau)` may jump at the tie, and its value at `tau = 0` depends on the side.

`OneLoopSpectrum.evaluate` now shares this per-diagram numeric build; no number changes.

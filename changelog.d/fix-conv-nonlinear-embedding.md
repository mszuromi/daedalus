#### Fix: kernels of nonlinear functions of the fields (`Conv(K, v^2)`, `Conv(K, phi(v))`) were wrong at loop order

- **The bug (silent, status 'ok').** `Conv(K, X)` with `X` nonlinear in the fields expands at the saddle into
  `K * (dv^2, ...)`: every leg of the product sits at the same earlier time `s`, bridged to `t` by `K(t - s)` once.
  The kernel route treated `K*dv^2` as a conductance vertex `dv (K * dv)` and delayed ONE copy of the leg. Saddles,
  k = 2 trees and everything linear in `X` were exact; loop orders (and k >= 3 trees) were wrong for every kernel
  type, not only repeated rates.
- **The fix.** The SDE front end embeds such a `Conv` as auxiliary linear fields (`SDE._embed_nonlinear_convs`):
  one stage `(b*Dt + a)*u = prev` per linear factor `a + b*(I*omega)` of the kernel image's denominator (repeated
  and near-repeated rates included), a companion block per irreducible factor of higher degree, the numerator read
  off the stages; fed by `X`, its output replaces the `Conv`. Fields `<kernel>_emb<n>_<stage>`, listed in `notes`.
  Equal to the hand-written Markov embedding to round-off (1e-15 .. 1e-14), k = 1 and 2, one and two loops. Works
  in equations and point-process rates, for unindexed kernels, kernels of the unit and kernels of the summed
  population (`sum(w[i,j]*Conv(g[j], phi(v[j])) for j in E)`). Refused by name: a delayed kernel, a kernel given
  only by `time_expr`, a non-rational image, and a kernel indexed by both the unit and the summed population.
  Affine arguments keep the kernel route (unchanged).
- **Hand-written actions** (`TemporalModelBuilder`) with such a term are refused by name in `FieldTheory.expand`
  (`engine.core.convolution.NonlinearConvError`, pointing to the SDE front end or a hand embedding) instead of
  giving wrong numbers; a `c delta` kernel (a weight) still passes.
- **Numbers that move** (old = main, new = this fix = the hand embedding; `test_conv_nonlinear_embedding.py`
  models):

  | model | quantity | old | new |
  |---|---|---|---|
  | 2-pop rate model, synapses on v^2, single pole | k=1, 1 loop, dvE | -0.657561 | -0.730332 |
  | same, separated poles (s, 1.7 s) | k=1, 1 loop, dvE | -3.267518 | -4.334712 |
  | same, repeated pole (alpha) | k=1, 1 loop, dvE | -1.257674 | -1.541525 |
  | same, numerator (1 + i w q) / two poles | k=1, 1 loop, dvE | -2.706166 | -3.609435 |
  | same, single pole | k=2, 1 loop, (dvE, dvI), tau=0 | 0.478816 | 0.706759 |
  | 1-field, Conv(K, v^2), single pole | k=1, loops 1 / 2 | 0.0219156 / 8.879e-4 | 0.0277304 / 1.0564e-3 |
  | same, repeated pole | k=1, loops 1 / 2 | 0.0161531 / 5.296e-4 | 0.0257904 / 7.520e-4 |
  | same, numerator | k=1, loops 1 / 2 | 0.0170489 / 5.661e-4 | 0.0260920 / 7.889e-4 |

  Blast radius: no tracked model (`models/`), SDE catalogue model or example notebook convolves a nonlinear
  function of a field (all their `Conv` arguments are single fields); the full default test suite run with the
  offending term logged hit only `tests/test_frontend_small_bugs.py` (a tree-level comparison, numbers correct,
  now refused for the nonlinear argument and run with an affine one) and the reducer's unit test (no kernel image,
  unchanged).
- Tests: `tests/test_conv_nonlinear_embedding.py` (two-population model with four kernel types against the hand
  embedding at k = 1 and 2; a one-field model at two loops; a rate with a kernel of the summed population; the
  refusals); `tests/test_frontend_small_bugs.py` (the builder refusal).

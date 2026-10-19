#### Cheaper Taylor norm bounds, lazy plotting imports (audit wave W8, part 3)

- **SPD-03.** The state-space Taylor action's parameters (`_onenorm_powers`) now take `||A||_1` and `||A^2||_1`
  exactly (an estimate above 300 states, as before) and bound `||A^p||` for p > 2 by
  `min(||A||^p, ||A^2||^(p//2) ||A||^(p%2))`, a rigorous upper bound, which is what the Taylor fragment needs.
  The seven further matrix powers or 1-norm estimates were about half of a state-space run.
- **SPD-06.** matplotlib (`api.report`) and IPython (`engine.core.field_theory._show`) are imported when first
  needed. `from api import compute_cumulants` no longer imports them (numba was already gone after audit W7):
  3.84 -> 3.34 s (median of 5, vs main c1cff54), and ~0.1 GB less peak footprint per run.
- **Timings** (base main f361730 and branch side by side):

  | case | base | branch |
  |---|---|---|
  | 2-neuron exp spike Hawkes, k=2 one loop, warm | 40.9 s, 1.35 GB peak | 28.1 s, 1.10 GB peak |
  | 2-neuron delayed-alpha spike-only, k=2 one loop | 612 s | 619 s (unchanged) |

- **No public number moves.** Zoo bit-identical (quad_exp, quadratic_hawkes_alpha, spike_reset, ou_quartic k4 l2
  checked again after the revert below), and the spike Hawkes C(tau) is bit-identical too.
- **Tried and reverted:** a matmul `_mode` (SPD-05, ~6 %). Its rounding depends on the tensor shape, so the
  evaluate-and-discard grid and the plain run were no longer bit-identical
  (`test_discard_is_bit_identical_and_releases`).

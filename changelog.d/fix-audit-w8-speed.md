#### Cheaper Taylor norm bounds, lazy plotting imports (audit wave W8, part 3)

- **SPD-03.** The state-space Taylor action's parameters (`_onenorm_powers`) now take `||A||_1` and `||A^2||_1`
  exactly (an estimate above 300 states, as before) and bound `||A^p||` for p > 2 by
  `min(||A||^p, ||A^2||^(p//2) ||A||^(p%2))`, a rigorous upper bound, which is what the Taylor fragment needs.
  The seven further matrix powers or 1-norm estimates were about half of a state-space run.
- **SPD-06.** matplotlib (`api.report`) and IPython (`engine.core.field_theory._show`) are imported when first
  needed. `from api import compute_cumulants` no longer imports them (numba was already gone after audit W7):
  3.84 -> 3.34 s (median of 5, vs main c1cff54), and ~0.1 GB less peak footprint per run.
- **Timings** (base and branch side by side):

  | case | base | branch |
  |---|---|---|
  | 2-neuron exp spike Hawkes, k=2 one loop, warm | 40.8 s | 30.5 s |
  | 2-neuron delayed-alpha spike-only, k=2 one loop | 612 s | 619 s (unchanged; peak 1.16 -> 1.04 GB) |

- **Moved numbers (<= 1e-15, deterministic):** zoo quad_exp k2 l1 1.39e-16, quadratic_hawkes_alpha k2 l1
  2.34e-16 (max relative over the entry's arrays). Every other zoo entry is bit-identical up to base scatter.
  The 2-neuron exp spike Hawkes C(tau) moves by <= 4e-16 relative.

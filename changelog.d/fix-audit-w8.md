#### Multi-field jump models and cold propagator builds in seconds: no Maxima zero tests on float or `I*omega` expressions (audit wave W8, part 1; SPD-04)

- **The cost was never the Taylor expansion.** On small networks of binary (Glauber) units with polynomial rates
  (2-4 units, jump action `xt Dt x - (exp(xt)-1)(1-x) r(I) - (exp(-xt)-1) x (1-r(I))`) the time went to Sage's
  `bool(c)` zero tests. Sage's random-point test cannot certify an expression with a float or `I*omega`, so each
  of these tests went to Maxima:
  - the zero test of the whole expanded action (an ECL `BINDING-STACK overflow` at three units, one loop);
  - the pivot tests of Sage's symbolic inverse of the kernel matrix (101 s for 3x3; the symbolic block then timed
    out, ~183 s wasted, and the run went on numerically);
  - the omega -> infinity limit of the inverse's entries (`numerator()` of a ~20 kB nested fraction, minutes per
    entry);
  - the source-coefficient tests of `classify_coefficient_factors` (once per typed diagram).
- **`engine.core.field_theory._sr_nonzero`** keeps `bool`'s verdict and falls back to `bool` whenever it cannot
  certify one, but certifies more cases without Maxima:
  - interval proofs also run in complex intervals (entire functions and integer powers; `log`, `arctan` and
    fractional powers go to `bool`);
  - a float expression is certified when the proof survives perturbing every float by 1e-8 AND its value clears
    2^-30 of the magnitude of its terms (an exactly nonzero value below double rounding, e.g. a complex
    `(a+b)^2 - a^2 - 2ab - b^2`, is zero to `bool` and goes to it);
  - verdicts are memoised on (GiNaC hash, `repr` digest, assumptions in force).

  0 verdict mismatches against `bool` on 10,500 random float / complex / near-cancelling expressions.
- `_sr_to_ring` no longer zero-tests the whole action. The ring monomials are independent, so the action is zero
  exactly when every coefficient is, and each coefficient is tested.
- **SPD-04: `api._propagator._sr_inverse` replaces `K_ft.inverse()`.** It runs Sage's own algorithm (augment,
  classical in-place Gauss-Jordan of `Matrix_symbolic_dense`, last-pivot check) step for step, with the certified
  test for its truth tests. The inverse is the same expressions (`repr`-identical to `K.inverse()` on every tested
  matrix: the 3x3 above, symbolic 4x4 / 5x5 with assumption parameters and `I*omega`, pivot swaps), without
  Maxima.
  - Cold-cache propagator builds (same machine load, base -> now): quad_exp 135 s (the symbolic block timed out:
    whether it completed depended on the machine's speed) -> 2.4 s, linear_hawkes 19.5 -> 0.5 s,
    linear_delta_spikes 14.6 -> 0.7 s.
  - Zoo cold-run walls: quad_exp 101.5 -> 5.6 s, quadratic_hawkes_alpha 153.8 -> 23.1 s.
- The symbolic D_delta is taken only for inverse entries of up to 4,000 characters (the zoo's largest is ~800).
  Larger ones are left to the exact tier of `compute_poles_and_residues`, which already filled D_delta whenever
  the symbolic block ran out of time.
- `classify_coefficient_factors` uses the same zero test, and memoises the product of the vertex coefficients
  per ordered tuple of vertex types: same operands, same order, same expression.
- Action text may contain numpy >= 2 scalar reprs (`f'{x!r}'` of an `np.float64` is `np.float64(0.25)`). They are
  read as the number; they raised `NameError: np`.
- **Timings, network of binary units, k = 1 mean, one process (before -> after):**
  - numeric rates: n=2 tree 2.9 -> 1.6 s, one loop 3.9 -> 2.0 s, two loops 28.4 -> 3.2 s; n=3 tree 188 -> 3.7 s,
    one loop ECL crash -> 10.5 s (0.44 GB peak footprint), two loops crash -> 786 s (7.7 GB); n=4 tree 30.8 s,
    one loop 43.9 s (expand alone was 257 s);
  - symbolic indexed coefficients: n=2 one loop 474.7 s / 5.26 GB -> 4.9 s (regression test), n=3 one loop
    1698.6 s -> TODO.
- **No public number moves.** Every zoo entry is bit-identical. The dendritic_quad_soma_sigmoid 1.4e-16 that
  shows up is base-vs-base scatter.

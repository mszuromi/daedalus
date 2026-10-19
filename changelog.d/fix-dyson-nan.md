#### Fixed: the x-space Dyson tree returned NaN for a negative eigenvalue gap (#55 regression)

`engine.integration.spatial.dyson_dressing.dressed_tree_C_x` (the exact unequal-D tree of #55) and `hcal_n` formed
each term as `exp(-m0 t) * Phi_n(t; nu)`. With an eigenvalue gap `nu < 0`, `Phi_n` alone grows like `e^{|nu| t}`, so at
large s it overflowed while the exponential underflowed: `0 * inf = NaN`. One NaN sample made the whole s-integral NaN,
and the convergence check raised (loud, not silent). Every coupled unequal-D model with two distinct masses was
affected; `tests/test_dyson_dressing.py::test_unequal_D_loops_order3_runs` and `::test_unequal_D_loop_dressing_ladder`
(slow) raised it.

- Each term is now one exponential, `(-1)^n expm(-t (Z + m0 I))[0, n]` on the Opitz matrix whose nodes are the
  eigenvalues themselves (Re > 0), which only decays. Values that were finite before are unchanged to rounding.
- The s-integral stops where its own decay bound `e^{-mu_min |tau|} e^{-2 v^2}` (times polynomials of degree
  <= 2 order + 2) is below e^{-800}, checked to be exactly 0 there, instead of sampling an infinite interval.
- Regression test `tests/test_dyson_large_gap.py`: closed-form divided differences, finite to t = 1e6 with nu < 0, and
  a mass-gap-3 unequal-D tree that converges geometrically to the exact Lyapunov tree on the line.

The two slow ladder tests still fail, now loudly on the spatial memory guard (the coupled loop at dressing order 2 / 3
needs 6.4 / 10.8 GB per chamber at the default grids, against a 6 GB budget); that is a separate #55 memory
regression (before #55 they ran at ~1.5 GB), followed up separately.

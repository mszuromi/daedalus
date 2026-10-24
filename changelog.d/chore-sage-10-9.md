#### SageMath 10.9

Daedalus now runs on, and is pinned to, the SageMath 10.9 stack. `environment.yml` (which CI uses) pins sage 10.9,
python 3.14.3, numpy 2.4.4, scipy 1.17.1, sympy 1.14.0 and numba 0.68.0 (llvmlite 0.50.0). It also requires
pip >= 26.2.2 and certifi >= 2026.10.19, the security minimums from the October 2026 pip / certifi CA incident. It
also pins setuptools 82.0.1. The conda sage 10.9 / Python 3.14 environment does not bring setuptools along, and
without it pyximport has no `distutils`. The compiled enumeration then falls back to the pure-Python path, with a
warning. The
`extras` numba range is now `>=0.61,<0.69`. The example notebooks declare the `sagemath-10.9` kernel.

- **numpy 2.4.** `np.trapz` was removed, so every use is now `np.trapezoid`, which numpy has had since 2.0. In the
  library this affects the spatial inverse transforms (`spatial_correlator`, `pipeline_bridge`) and
  `Cumulant.spectrum`.
- **Sage 10.9 Maxima bridge.** A noise expression that calls `factorial(...)` or `gamma(...)` now uses Sage's own
  function. This matters because Sage prints `gamma(1 - Y)` as `factorial(-Y)`, for example in a CGMY mean with a
  parameter-named `Y`. Before, the parser made a formal function with the same name. Sage 10.8 accepted that, but
  10.9 stops with "operator mismatch in sr-to-maxima translation". This affected CGMY and
  `TemperedStableSubordinator`. The change applies only to calls, so a parameter can still be named `gamma`. A
  `CGF(...)` text may now call `gamma(...)` / `factorial(...)`; before, these were refused as undeclared names.
- **Reserved function names (breaking).** `SDE.define_function` now refuses the names `gamma` and `factorial`.
  Rename such a function, for example to `gamma0`. In a noise expression, a call of either name now means Sage's own
  function, so a user function with that name would be silently replaced there. Before, a user function `gamma(v)`
  in a noise rate failed with a misleading "saddle does not satisfy its own equations", on Sage 10.8 as well, while
  the same function in a drift worked.
- **No number moved beyond round-off, except one documented case.** Each comparison below ran the same code under Sage 10.8 (py 3.13.7, numpy 2.3.2, scipy
  1.15.2, sympy 1.13.2, numba 0.65.1) and under Sage 10.9:
  - **Tracked zoo, default engine.** 21 entries ran. 12 are bit-identical, and 9 differ by round-off only. The
    largest relative move is 7.5e-15, on one tau = 0 element: 1.6e-18 absolute on 2.2e-4 (array max 1.5e-3). Every
    other move is at most 9e-16. Analyzer codes are identical. 6 entries refuse on both stacks with the same error
    (memory budget, UV divergence, pole-free propagator). One entry did not run: it takes more than 50 minutes.
  - **Public exact-oracle suite and frozen-baseline tests.** 105 recorded `compute_cumulants` calls. Every value is
    bit-identical or within 6.3e-16 relative. Under 10.9, the slow tests also reproduce the frozen zoo baseline to
    1e-13.
  - **Mean-field saddles near ties.** 7 drifts tested, among them a 1e-14 bistable asymmetry, a 1e-10 pitchfork, a
    near-double root and tanh / exp ties. Roots, root order, the selected saddle and the k = 2 one-loop cumulants are
    all bit-identical.
  - **Caches.** A `saved_models/` cache written under 10.8 loads under 10.9 and gives bit-identical results.
  - **One larger move.** The two-neuron delayed-alpha network of the slow tests (Padé loops at the automatic order,
    which reaches n_max = 16 with an estimated error of 1.4e-5) moves 5.1e-12 relative at one loop (tree: 3e-15).
    The cause is the automatic order's adaptive truncation. Its tolerance for order 16 is computed from the
    lower-order error estimates, which are differences of approximations and move 3e-11 between the stacks. At a
    fixed order the move is round-off: 2.0e-14 at n = 16 and 5.6e-15 at n = 8, against 8.3e-15 / 6.0e-15 for a
    1-ulp change of the delay on 10.9 alone. On the automatic route, that same 1-ulp change on 10.9 alone moves the
    one-loop value by 9.0e-13. The voltage and builder forms stay bit-identical on each stack.
- **Slow test made deterministic.** The bit-identity test for that network now runs without the wall-time cap
  (`delay={'wall_max': inf}`). On a slow CI runner, the predicted cost of the order-16 step (about 1025 s) passed
  the default 900 s for one of the two forms only, so the forms were compared at different orders.

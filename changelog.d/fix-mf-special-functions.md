#### Mean field: `erf`, `arctan` and `heaviside` in a drift

A drift (or any `.equation` right-hand side) that used `erf`, `arctan` or `heaviside` stopped the saddle solve with
`NameError: name 'erf' is not defined`, on Sage 10.8 and 10.9. The expression parser accepts these functions, but the
numpy namespace in which the mean-field residual rows are compiled (and the Sage namespace of the linear stability
analysis) did not carry them. Both tables now carry every function the parser accepts in a drift: `exp`, `log`,
`sqrt`, `sin`, `cos`, `tan`, `tanh`, `sinh`, `cosh`, `arctan`, `erf`, `heaviside`, `abs`, `gamma` and `factorial`,
plus `sign` and a bare `e`. The vertex expansion evaluates the text in Sage's own namespace and was never affected.
The `set_mf_equation` saddle path (point processes) was not affected either.

- **Bare field in a population-less model.** The linear stability now unwraps a size-1 field before calling a Sage
  function, so `erf(x)`, `sign(x)` and `abs(x)` of the bare field work (a compound argument
  such as `exp(-x)` already did).
- **Unsupported functions are refused up front.** A call of any other name in an equation (a formal function, or
  one the parser let through that has no numeric implementation) now raises
  `UnsupportedMeanFieldFunctionError` (a `ValueError`) before the first residual is evaluated, naming the function
  and listing the supported ones. It no longer fails later with a `NameError` from inside the compiled rows.
- **`heaviside(0)` is 0.** The step takes the package's Itô value Θ(0) = 0, the same one Phase J applies to a
  constant constraint row and the equal-time k = 2 point (the left limit τ → 0⁻). Sage leaves `heaviside(0)`
  unevaluated, so the action fixes no other value. The point-process simulator used 1 at 0 and now uses 0, so the
  simulator, the mean field and Phase J agree; the two differ only when a step's argument is exactly 0.
- **A saddle on the jump of a step is refused.** If a `heaviside(arg)` has `arg` equal to 0 at the selected saddle
  (to 1e-10 of its own terms), the expansion about the saddle needs the step's slope, a Dirac delta at 0, which does
  not exist. The solve now says so (`UnsupportedMeanFieldFunctionError`); before, the run stopped deep in the
  propagator with `TypeError: unable to simplify to complex approximation`. A saddle off the jump has slope 0 there.

Tests: `tests/test_mf_special_functions.py` (one-field drift per function against an independent `scipy` root and
derivative, Θ(0), the refusals, and a k = 2 tree run for `erf`, `arctan`, `heaviside` and `erf(arctan(x))`).

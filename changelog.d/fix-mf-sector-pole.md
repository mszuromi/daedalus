#### Mean-field sector check: a saddle whose closed form has a pole at the default point

`MasterEquation.to_sde()` of a process with a rate like `mu*x + c*x*(x - 1)` (and `a + b*x`) failed
`FieldTheory.expand()` with "the MF action sector is non-zero at the saddle" at default parameters where the linear part
of the drift cancels (`mu = b + c`, restoring coefficient exactly 0). The saddle (`x* = sqrt(a/c)`) was fine: the check
substitutes a symbolic form of the saddle that divides by that coefficient, the substituted coefficient could not be
evaluated (`N/A`), and an unevaluable check counted as a failure.

- **Fix.** When the substituted coefficient cannot be evaluated, the numerical check evaluates the coefficient before
  the substitution at the solver's numerical saddle (the same quantity, without the pole). A model whose saddle is wrong
  is still refused (`tests/test_leftovers_39.py`, `tests/test_mf_saddle_fixes.py` unchanged and passing).
- **Check against the exact master equation** (`tests/test_mf_sector_pole.py`, `a=1, b=0.2, mu=0.5, c=0.3`): the
  one-loop `kappa_1`, `kappa_2` against `dd.master_equation` at system size 100 and 200: relative errors 8.2e-7, 9.6e-8
  and 2.0e-7, 2.4e-8 (O(h^2): a factor 4 per doubling); the tree errors halve (1.9e-3 -> 9.4e-4, 5.0e-4 -> 2.5e-4).
- Models that built before are unchanged.

#### Symbolic output: kernels that are real in `s = i omega` once their `I` cancels

The exact tree covariance (`dd.tree_covariance`), the symbolic loops (`dd.symbolic_cumulant`) and so `dd.one_loop_mean`
and `dd.one_loop_spectrum` refused `quadratic_hawkes_alpha` as having complex coefficients. Its alpha kernel's image
reaches `K_ft` as `I w / (I omega^2 tg^2 + 2 omega tg - I)`: the realization keeps a unit `I` in both numerator and
denominator, and the structural conversion to `QQ(s, ...)` met `-I` as a number. In `s = i omega` that entry is the
real `-w / (1 + s tg)^2`. Each entry with an `I` is now first written as one fraction. Its numerator and denominator are
multiplied by the coefficient conjugate (`I -> -I`) of the denominator, which makes the denominator real. The
numerator is then real exactly when the entry is, so a genuinely complex kernel keeps its `I` and is refused as
before.

`quadratic_hawkes_alpha` now works in every form of `dd.one_loop_mean` and in `dd.one_loop_spectrum`. Both match the
pipeline to 4e-16 (mean) and 8e-15 (spectrum). A model whose kernel is half alpha and half exponential has odd powers
of `I` in a cubic denominator; it matches to 5e-16 and 2e-15. Only inputs that used to be refused change; no existing
number changes.

Two clarifications to `MeanRelation.saddle_at` (`mean_field='solve'`, from the previous change):
- At other parameters the saddle is the root `dd.run` would use there, the mean-field solve's first. It need not
  continue the build's branch: a double well `x - x^3 + h` built at `h = 0.1` sits in the left well, and at
  `h = -0.3` it moves to the right one, as `dd.run` does.
- The refusal now counts the "mean-field root(s)" the solve reports, not "stable root(s)", because that list can
  include an unstable root.

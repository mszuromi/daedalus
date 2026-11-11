#### Phase J: a δ(0) term is refused by name (`DeltaCycleError`) instead of being evaluated as 1

A δ-subset whose δ-carrying lines close a cycle has linearly dependent δ constraints. Examples are two parallel δ
lines between the same vertices, or a δ line from a vertex back to itself. Once the others are solved, the last
constraint is identically zero, so the term contains δ(0): it is infinite. Both Phase J routes dropped that
constraint and kept the subset with its δ weights, which amounts to setting δ(0) = 1, with no warning, analyzer on
or off.

Example: `Dt x = -a x + g u^2`, where `u = h + xi` is algebraic and `xi` is white, so `C_uu = 2D δ`. Take `a = 1`,
`g = 0.3`, `h = 0.5`, `D = 0.05`. The one-loop mean of `x` contains `g <u(t)^2>`, which is infinite. It came back as
`g 2D = 0.03`, and the one-loop `C_xx` as `9e-4 e^{-tau}`.

Both routes now raise `DeltaCycleError`, a `ValueError` and a `DaedalusError` with `reason = 'delta_cycle'`. It is
exported as `dd.DeltaCycleError` and names the subset's δ lines. The check sits in the subset loop that both routes
share (the state-space route evaluates the subsets that loop builds). Cycles of smooth lines are still merged as
before: their vertices share a time.

The same model with `u` an OU field (no δ part) is unchanged: its one-loop mean is `0.015`. None of the shipped
temporal models has such a subset at one loop (k = 1 and 2, both routes), and their values are unchanged bit for bit.

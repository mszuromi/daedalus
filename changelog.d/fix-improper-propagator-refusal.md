#### Phase J: a propagator entry growing at large omega is refused by name (`ImproperPropagatorError`) instead of being dropped

An entry of `G(omega)` whose reduced form `P / Q` has `deg P > deg Q` grows at large omega: `G(t)` carries `δ'(t)` (or
higher derivative) terms. The model analyzer refuses such a model (`G3_IMPROPER`). With the analyzer off
(`preflight=False`) or that error downgraded, a model that also has poles used to run to a value without that part:
- the exact propagator tier's `deg P <= deg Q` check failed inside its broad fallback;
- the numerical tier set the entry's large-omega limit to 0;
- the only sign was a `PoleResidueWarning`.

The state-space route declined the model (its realization is refused, `index_gt_1`) and handed every diagram to the
pole engine, which did the same. Example: `Dt x = -a x + u`, `u = h + xi` algebraic and white (`C_uu = 2D δ`),
`y = Dt u`. Then `G_yu = i omega`, so the tree-level `C_yx(tau)`, the lag derivative of `C_ux(tau)`, is `2 D a e^{-a/2}`
in size at one of `tau = -0.5, 0.5`: 0.061 at `a = 1, D = 0.05`. The same run gives `C_ux(0.5) = 0.061` correctly. It
came back as 0 at both lags on both routes.

The growing entries are now decided exactly, right after the exact inverse and before any tier can fall back. When
the exact tier does not reach the inverse, two numerical probes far above the poles decide them. They are recorded in
`prop['improper_entries']`. `compute_cumulants` (after the analyzer's poles stage) and `compute_correction_td` then
raise `ImproperPropagatorError` (a `ValueError`, `reason == 'improper_propagator'`) naming the entries and their
growth, on both routes.

A propagator with no pole at all still raises `PoleFreePropagatorError`, and with the analyzer on `G3_IMPROPER` still
stops the run first. Proper propagators are unchanged: the shipped models give the same values bit for bit.

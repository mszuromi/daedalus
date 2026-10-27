#### Resummation: Gaussian expectations over the nonlinear fields only

`dd.hartree` (and the screened series built on it) took its Gaussian expectations as a tensor Gauss-Hermite rule over
EVERY field, so a quadratic-rate model with 13 linear filter stages and two nonlinear voltages (15 fields) was refused
at 3^15 = 14 348 907 points. A field that enters no drift monomial of degree two or more and no noise term contributes
through its mean alone; the rule now runs over the fields that DO enter nonlinearly (`result.info['nonlinear_fields']`,
a field multiplying another field or entering the noise counts), and the same 15-field model solves in about two
seconds. The new `expectations` option of `dd.hartree` chooses the evaluation: `'quadrature'` (the tensor rule over the
nonlinear fields, refused above `MAX_QUADRATURE_POINTS` points with a message naming the dimension and its fields),
`'moments'` (the closed-form Gaussian moments of the polynomial drift and noise: no quadrature, no size limit) or
`'auto'` (the default: quadrature while the rule has at most 20 000 points, moments beyond). `gaussian_closure` takes
`nonlinear=` and `moments=` for numeric callables.

No number moves: a model whose fields are all nonlinear runs the same rule bit for bit (the coupled quartic's
covariance and mean are identical arrays); with linear fields the subspace rule, the moments and the old full-space
rule agree to 1e-13 (relative to the largest value; checked against the previous release) on the quadratic-rate model with 0, 3 and 6 filter stages, and the 15-field solution
equals an independent hand-written closure.

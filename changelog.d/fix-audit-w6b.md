#### Noise library, symbolic covariance, complex realizations, simulator arrays (audit wave W6, library group)

- **`NormalInverseGaussian` with parameter-named `alpha` / `beta` runs (LOUD-01).** Its CGF
  `sqrt(a^2 - b^2) - sqrt(a^2 - (b + theta)^2)` was expanded by Maxima into `I*sqrt(b^2 - a^2)` for the second root
  when `alpha`, `beta` are symbols, a nonzero `K(0)` the mean-field check refused. The CGF is now written with the one
  root `g = sqrt(alpha^2 - beta^2)` and a series around 1 (`K(0) = 0` identically). The mean and `C(0)` equal the
  closed forms (`delta beta / g`, `kappa_2 / (2 mu)`) to round-off; a literal NIG moves by at most 1 ulp (`C(0)`
  0.06954412153194652 -> 0.0695441215319465 on the test model).
- **Symbolic C2 `tree_covariance` of a fully coupled float model (GEN-08).** A 3-field linear SDE with float
  coefficients failed (`K * G != identity`, or `NonRationalPropagatorError` on a `^1.000` exponent): the pipeline's
  float `G` was rationalized entry by entry, which is not the exact inverse. With `K` available, `G = K^-1` is now
  computed exactly in the rational field (a supplied `G` is checked against it numerically, so a convention mismatch
  is still refused) and a float-typed integral exponent is an integer. The float model equals its rational twin and
  the Lyapunov covariance to 1.7e-15. Results of models that passed before are unchanged (their `G` was the exact
  inverse).
- **Complex-coefficient realizations get the exact Hurwitz test and the a-posteriori refusal (GEN-15).** The Routh
  test returned `None` over `QQ[I]`, so `realize_K` never checked a complex realization: the graded Pade-24 4-field
  cycle with one entry multiplied by `1 + I/1000` was accepted while its real twin was refused as `ill_conditioned`.
  The test now runs on `p(s) conj(p)(s)` (real rational coefficients, the same root real parts): both are refused.
  Agrees with the numerical eigenvalues on 200 random complex matrices.
- **`build_sim_arrays` (multi-population Hawkes simulator) reads the coupling signs from the model (GEN-16).** The
  sign of `W` came from the literal population names (`E` +, `I` -; any other name +): an A/B-named copy of an E/I
  model simulated an all-excitatory network. The sign is now that of `d v* / d w_ij` in the model's own saddle
  equation (at the saddle), so the A/B copy gives the E/I `W`. The time constant is read from `taug<P><Q>`,
  `tau_g<P><Q>`, `taug` or `tau_g` (a scalar applies to every pair); a pair whose kernel filters (an
  omega-dependent image) and has none now raises instead of a silent 1.0 (an instantaneous kernel keeps the unused
  1.0). Tracked models: `W` and `tau_g` unchanged.

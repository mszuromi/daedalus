#### Resummation: exogenous inputs (point processes, exposed noise, Gaussian jitter) through kernels

A point process, or a noise source exposed as a field (`as_field=`), is a field without dynamics (`n = noise`), so the
closure refused it ("singular Dt pencil") and a Poisson input through a synaptic kernel had to be written by hand, the
jumps driving the first filter stage. The closure now eliminates the input: its rate is the mean drive (the mean-field
saddle), a rate that depends on the fields is a drift (a Hawkes-type feedback through a kernel, with a linear or
quadratic rate, works), and its second cumulant is white noise that drives the kernel states directly (the cumulants
beyond the second are dropped, as everywhere in the closure). The input is not a field of the result
(`info['kernel_states']['inputs']` names it).

- A Poisson input through an alpha kernel, a Gaussian input exposed as a field (direct and through a kernel), a point
  process coupled directly and a field-dependent (quadratic) rate through a kernel equal the model written by hand with
  the filter stages and the jumps (Hartree mean, covariance, lag correlator and spectrum to 1e-12; observed rounding);
  the screened series (k = 2, order 1) of the Poisson-input model equals the hand model's.
- A Gaussian-jittered input goes through the rational fit of the density: for the README's jittered Poisson input into a
  low-pass `v`, the closure's spectrum equals `lam p^2 |w H_tau H_X|^2 exp(-sigma^2 w^2)` within 1e-5 of its maximum and
  the variance within 1e-5 relative (fit tolerance 1e-6); the mean is `w lam p` exactly.
- Refused, with the reason: an input that enters a nonlinear term of another equation (`n*v`), an input whose rate holds
  a kernel (a non-local noise cumulant), an algebraic field that depends on another algebraic field or a time
  derivative. No number of a model without inputs moves.

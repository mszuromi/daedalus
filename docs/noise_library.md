# Noise library (finite-moment Levy sources) and samplers

Every source is one per-unit-time CGF `K(theta; x)` (`api/noise.py`); the SDE builder derives the action term
`-K(sum_i c_i psi_i)` from it. Only infinitely divisible sources are offered. This page lists what exists and what is
deliberately not expressible.

## Sources, jump laws, composites

| Source | `K(theta)` (per unit time) | cumulant rates `kappa_m` |
|---|---|---|
| `VarianceGamma(mu, sigma, nu, drift)` | `drift*theta - log(1 - nu*mu*theta - nu*sigma^2*theta^2/2)/nu` | `d+mu`, `sigma^2+nu mu^2`, ... |
| `NormalInverseGaussian(alpha, beta, delta, mu)` | `mu*theta + delta*(sqrt(alpha^2-beta^2) - sqrt(alpha^2-(beta+theta)^2))` | `mu+delta beta/g`, `delta alpha^2/g^3`, ... (`g = sqrt(alpha^2-beta^2)`) |
| `CGMY(C, G, M, Y)` | `C*Gamma(-Y)*((M-theta)^Y - M^Y + (G+theta)^Y - G^Y)` | `C Gamma(m-Y)(M^(Y-m) + (-1)^m G^(Y-m))` |
| `TemperedStableSubordinator(alpha, c, lam)` | `c*Gamma(-alpha)*((lam-theta)^alpha - lam^alpha)` | `c Gamma(m-alpha) lam^(alpha-m)` |
| `Meixner(alpha, beta, delta, mu)` | `mu*theta + 2 delta (log cos(beta/2) - log cos((alpha theta+beta)/2))` | `mu + alpha delta tan(beta/2)`, ... |
| `BilateralGamma(alpha_p, lam_p, alpha_m, lam_m)` | `-alpha_p log(1-theta/lam_p) - alpha_m log(1+theta/lam_m)` | `(m-1)!(alpha_p/lam_p^m + (-1)^m alpha_m/lam_m^m)` |
| `Skellam(rate_plus, rate_minus)` | `r+ (e^theta-1) + r- (e^-theta - 1)` | `r+ + (-1)^m r-` |
| `NegativeBinomialProcess(shape_rate, p)` | `r (log(1-p) - log(1 - p e^theta))` | `r p/(1-p)`, `r p/(1-p)^2`, ... |
| `a + b` (`SumSource`) | `K_a + K_b` | sum |

`CGMY` with `Y < 0` is a finite-activity compound Poisson process; `0 < Y < 1` is a difference of tempered-stable
subordinators; `1 < Y < 2` is the compensated process (mean `C Gamma(1-Y)(M^(Y-1) - G^(Y-1))`). `Y = 0` is the
bilateral-gamma limit (`BilateralGamma(C, M, C, G)`) and `Y = 1` has a different CGF form; both are refused.

Jump laws for `CompoundPoisson(rate, jump)`: the existing ones plus `Geometric` (on 1, 2, ...), `PoissonJump`,
`NegativeBinomial` (failures), `Logarithmic`, `DoubleExponential` (Kou), `AsymmetricLaplace`, `Hyperexponential`,
`PhaseType(alpha, T)`, `ChiSquared` (exactly `Gamma(df/2, 2)`).

**Laws without an analytic MGF** (`LogNormal`, `Weibull`) are carried by their raw moments to `order` (default 8).
Their CGF is a truncated series: `CompoundPoisson(...).cumulant(m)` and `cgf_series(m)` raise for `m > order`, and the
action carries the order-`order` polynomial. Choose `order` at least as large as the highest cumulant order the
computation uses (`k` plus the loop corrections you ask for).

Composites: `merton_jump_diffusion`, `kou_jump_diffusion` (Gaussian + compound Poisson via `SumSource`),
`levy_driven_ou` (Barndorff-Nielsen-Shephard OU, an unbuilt `SDE`; stationary `kappa_n = kappa_n(source)/(n mu)`).

## Telegraph noise and Markov-modulated Poisson

A field `s` driven by `Poisson(rate='g', size='-2*s')` is the dichotomous process (`+a <-> -a` at rate `g`): rate and
jump size may depend on fields, so the dynamics and the action `-g (exp(-2 s psi) - 1)` are expressible, and a simulator
can draw the flips (`Poisson(rate='g', size=1).sample` gives the counts; apply `s *= (-1)^n`; `Poisson.sample`
returns `size * count`, which is exact only for at most one event per step). A Markov-modulated Poisson process is the
same with a state-dependent rate, `Poisson(rate='l01*(1-s) + l10*s', size='1-2*s')` for the hidden state, plus an
observed counting field `Poisson(rate='a0 + a1*s')`.

What is **missing** is the perturbation theory: the pipeline expands around a mean-field saddle with small fluctuations.
For a two-point state the saddle of `K'(0)` (zero for the telegraph, a cubic root for the MMPP) is not where the process
lives (`s = +-a`), so the tree and loop cumulants are not the cumulants of the process (for the telegraph the saddle
variance is `4 g s^2 -> 0`: an undriven field). Use these models for simulation, or their exact moment equations.

## Samplers (`api/noise_samplers.py`)

`source.sample(dt, size, rng, values=None)` draws `size` independent increments over `dt`; `values` binds
field-dependent parameters (scalars or arrays of length `size`). `jump.sample(rng, size)` draws jump sizes.

* Exact: Gaussian, Poisson, CompoundPoisson, GammaProcess, InverseGaussianProcess, VarianceGamma (gamma clock),
  NormalInverseGaussian (inverse-Gaussian clock), BilateralGamma, Skellam, NegativeBinomialProcess (gamma-Poisson),
  SumSource, TemperedStableSubordinator and CGMY with `Y < 1` (exponentially tilted stable rejection, the step split so
  the acceptance probability is at least `e^-1`; `Y < 0` compound Poisson).
* Controlled approximations: CGMY with `1 < Y < 2` keeps the jumps larger than `eps = CGMY_EPS/max(M,G)` exactly and
  replaces the small ones by a Gaussian with matched mean and variance (relative error of the fourth cumulant about
  `(eps max(M,G))^(4-Y)`); it needs about `eps^-Y` jumps per unit time, so it is slow. `Meixner` inverts the exact density
  on a `2^16` grid (`z = d sinh(s)` change of variable; cumulant error below 1e-3 relative in the tabulated moments).
* No sampler: `Cumulants`, `CGF` (they do not determine a scheme).
* `simulate_ou_paths` is the Levy-driven OU simulator (exact increments, Euler OU step); `euler_ou_cumulant` gives the
  exact stationary cumulants of that recursion, `kappa_n dt/(1 - (1 - mu dt)^n)`.

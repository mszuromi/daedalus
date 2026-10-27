# Resummation at strong coupling: Hartree and the screened loop series

The loop expansion is exact order by order but asymptotic. For OU quartic,
$dx = -(\mu x + \varepsilon x^3)\,dt + \sqrt{2D}\,dW$ with $g = \varepsilon D/\mu^2$, every truncation of the
equal-time variance is already 15–20 % off at $g = 0.1$, and the series diverges beyond that. Two opt-in resummations
stay accurate at any coupling. Both are off by default: without `resum=` no code path and no number changes.

| | `dd.hartree` / `Config(resum='hartree')` | `Config(resum='screened')` |
|---|---|---|
| what | self-consistent Gaussian: every cactus (daisy) diagram | the loop orders re-expanded around a variational trial propagator |
| output | mean, $\Sigma$, dressed drift $A$, $C(\tau)$, spectra | $C_N(\tau)$ for $N = 0..$`max_ell`, spread error estimate |
| non-Gaussian | none ($\kappa_{k\ge3} = 0$), one exponential in $\tau$ | the loop orders' own: sunsets, multi-exponential $C(\tau)$ |
| cost | one closure solve (milliseconds) after the expansion | about 25 runs of the loop pipeline at order $N$ (17 at $\tau = 0$ only) |

Code: `api/hartree.py`, `api/resum.py`. Tests: `tests/test_hartree.py`, `tests/test_resum.py`. The tables below come from
`docs/resummation_validation.py` (`sage -python docs/resummation_validation.py`, about 5 minutes).

## Hartree: the self-consistent Gaussian

The closure finds the Gaussian $N(m, \Sigma)$ that solves

$$\langle f\rangle = 0,\qquad A\Sigma + \Sigma A^{\mathsf T} + \langle Q\rangle = 0,\qquad A = \langle\partial f/\partial x\rangle,$$

with every average taken in that Gaussian. Here $f$ and $Q$ are the drift and the noise covariance, read off the expanded
action about the mean-field saddle: $f = M^{-1}g$ and $Q = M^{-1}Q_gM^{-\mathsf T}$ for an action
$\tilde x^{\mathsf T}(M\partial_t x - g) - \frac12\tilde x^{\mathsf T}Q_g\tilde x$.

- **Diagrams.** In MSRJD language this is the Hartree(-Fock) truncation of the 2PI effective action. It sums all cactus
  diagrams, and the mean is shifted by the dressed tadpoles. Terms with three or more response fields drop out
  (Itô: $R(0) = 0$).
- **Moment closure.** By Stein's lemma it is the Gaussian moment closure. For a jump model that is the closure of the
  master equation.
- **Lag structure.** $C(\tau) = \Sigma e^{A^{\mathsf T}\tau}$ for $\tau \ge 0$, and the spectrum is
  $S(\omega) = -\Sigma(A^{\mathsf T} + i\omega)^{-1} - (A - i\omega)^{-1}\Sigma$ (the `dd.spectra` convention
  $\int C e^{+i\omega\tau}$).

```python
h = dd.hartree(model, parameters)       # HartreeResult
h.mean, h.cov, h.A, h.Q, h.rates        # absolute mean, Sigma, dressed drift, averaged noise, -eig(A)
h.C(tau, 'x', 'y')                      # kappa(x(t), y(t + tau)), either sign of tau
h.spectra(omega=..., T=..., i=0, j=1)   # the dd.spectra keys: S, S0, integral(_neg), integral_inf(_neg), C0, tau_c
h.covariance_matrix(tau)                # the dd.covariance_matrix layout, Sigma[n, i, j]
h.solutions, h.advisories               # every distinct stable solution; notes of the solve
```

Inside a run, `Config(resum='hartree')` (`max_ell` unset or 0, `k = 1` or `2`) returns `C_tau` of the requested pair or
the mean shift, and `resummation`. It also accepts `spectra=` and `covariance=`.

**Solutions and bistability.** The closure starts from the selected saddle and from every other mean-field root, and
it keeps every distinct stable solution.

- When there are several, `HARTREE_MULTIPLE_SOLUTIONS` is added to `.advisories` and a `HartreeBistabilityWarning` is
  given.
- Example: the symmetric double well ($\mu = -1$, $\varepsilon = 1$, $D = 0.1$) has a symmetric solution and two
  broken-symmetry solutions.
- None of them describes the hopping between the wells: the exact slowest rate is 0.0335, against the Hartree rates
  0.242 (symmetric) and 1.63 (broken).
- At $D = 1$ only the symmetric solution exists, and there is no advisory.

**Scope.**

- Covered: temporal models with a constant, invertible `Dt` pencil; white noise or colored noise through its Markovian
  embedding; Poisson (jump) sources, through their first two jump moments.
- Refused (`HartreeUnsupported`): kernels (`Conv`, delays), spatial models, non-local noise cumulant kernels,
  derivative vertices, singular pencils.
- A drift that is not polynomial in the fields is replaced by its Taylor polynomial (order 6 by default, or
  `taylor_order`), with the advisory `HARTREE_TAYLOR_TRUNCATED`.
- **Expectations run over the nonlinear fields only.** A field that enters no drift monomial of degree two or more and
  no noise term (a linear filter stage, a linear read-out) contributes through its mean alone, so it adds no dimension:
  a model with 13 linear filter stages and two nonlinear voltages is a one- or two-dimensional integral, not a 15-fold
  one. The fields counted are listed in `result.info['nonlinear_fields']`. A field that multiplies another field in the
  drift, or enters the noise (a state-dependent rate), is counted. Over the nonlinear fields a polynomial action is
  integrated exactly by tensor Gauss-Hermite quadrature, or, through `hartree(..., expectations='moments')`, by the
  closed-form Gaussian (Isserlis) moments with no quadrature at all. The default `expectations='auto'` uses the
  quadrature while its rule has at most 20 000 points and the moments beyond; `expectations='quadrature'` refuses above
  `MAX_QUADRATURE_POINTS` points and names the nonlinear dimension and its fields.

## The screened (optimized-perturbation) loop series

Write the bilinear part of the action with a trial drift $A_t$ and noise $Q_t$, and treat the difference as a
two-point counterterm counted like one loop:

$$A(\lambda) = A_t + \lambda(J_0 - A_t),\qquad Q(\lambda) = Q_t + \lambda(Q_0 - Q_t),\qquad \lambda = 1 \text{ is the model.}$$

Every loop order $C_\ell$ (vertices unchanged) is evaluated with the propagator of $A(\lambda)$. The counterterm
insertions are its $\lambda$-derivatives, so there is no diagram filter and no double counting:

$$C_N = \sum_{\ell + n \le N}\frac{1}{n!}\,\partial_\lambda^n C_\ell\big|_{\lambda = 0}.$$

- **Trial line.** The trial runs along $A(\kappa) = J_0 + \kappa(A_H - J_0)$ (and the same for $Q$), through the tree
  ($\kappa = 0$) and Hartree ($\kappa = 1$). With $F_\ell(s) = C_\ell(J_0 + s\Delta, Q_0 + s\Delta_Q)$ the order-$N$
  value is $\sum_\ell\sum_{n\le N-\ell}(-\kappa)^n F_\ell^{(n)}(\kappa)/n!$.
- **Order 1 at $\kappa = 1$ is Hartree** for a model whose closure keeps the mean at the saddle. This is checked as a
  whole curve and spectrum (`test_order_one_at_hartree_mass_is_hartree`).
- **Injection.** The counterterm is a set of extra bilinear action terms
  $-\sum h_{ab}\tilde x_a\,\delta x_b - \frac12\sum q_{ab}\tilde x_a\tilde x_b$ with numeric parameters, on a copy of the
  model named `'<name> [screened]'` (its own caches). The mean field is unchanged, and every Phase J route runs as it
  is.
- **Choosing $\kappa$** (`mass=`):
  - `'pms'` (the default): the stationary point of $C_N(\kappa)$ nearest $\kappa = 1$ (minimal sensitivity). The
    observable is `'trace_cov'` (the sum of the equal-time variances) or `'C0'` (the requested pair).
  - The scan runs the loop orders at $\tau = 0$ on 17 Chebyshev nodes in $\log\kappa$ over `kappa_range` (default
    $[0.25, 6]$, narrowed to where the trial drift is stable). It interpolates $\log|F_\ell|$, whose values span many
    decades at strong coupling, and differentiates the interpolant analytically.
  - Even orders generically have no stationary point. They take the mass of the next lower order that has one (with a
    `ResumWarning`), and failing that Hartree's. `resummation['kappa_source']` records which.
  - `'hartree'` uses $\kappa = 1$. A number uses that $\kappa$.
- **The orders** come from a central stencil of runs about $\kappa$: 7 points through $N = 3$, 9 at $N = 4, 5$, relative step `step`
  (default $5\times10^{-3}$). `resummation['pms_stencil_check']` compares the stencil value with the interpolant
  ($10^{-8}$–$10^{-9}$ for OU quartic).
- **Error estimate.** `C_tau_spread` $= C_N - C_{N-1}$ at the same mass, and its maximum relative to $\max|C_N|$ is
  `resummation['spread_last_two']`. For OU quartic at three loops and equal time it is 7–280× the true error from $g = 0.01$ to $1000$.
  It is an estimate, not a bound.

```python
r = dd.run(model, dd.Config(k=2, max_ell=3, resum='screened', parameters=p, tau_grid=tau), mod)
r['C_tau']                   # order N = 3
r['C_tau_by_order']          # {0: ..., 1: ..., 2: ..., 3: ...}
r['C_tau_spread']            # C_3 - C_2
r['C_tau_loops_by_ell']      # the plain loop series (the same call a plain run makes)
r['resummation']             # kappa, kappa_source, A_trial, Q_trial, pms_curve, stationary_points, spread_last_two,
                             # hartree (the closure), counterterm, stencil, n_runs, wall
```

**The resummed mean ($k = 1$).** `Config(k=1, max_ell=N, resum='screened')` resums the one-point function of one field
with the same trial propagator, the same order-$N$ re-expansion and the same PMS choice. The counterterm is the
two-point one above, so the loop orders of the mean (the tadpoles, $\ell \ge 1$) are evaluated at the trial
propagator and re-expanded exactly like the loop orders of the covariance. The PMS criterion is the equal-time
covariance, whatever $k$ is: `observable='trace_cov'` (the default) or `'C0'` (the variance of the requested field)
are read off $k = 2$ loop orders. A $k = 1$ run and a $k = 2$ run of one model at one parameter point therefore have
the same $\kappa$ and the same trial propagator (`resummation['kappa']`, `A_trial`, `Q_trial`, `pms_curve`). The
result has the keys of $k = 2$ with the shift about the saddle in `C_tau`: `C_tau_by_order`, `C_tau_spread`,
`C_tau_loops` and `C_tau_loops_by_ell` (the plain loop series), plus the absolute `mean` and `mean_by_order`. A mean
has no `spectra=` and no `covariance=`: both are refused. For $dx = (-x + 0.3x^2 - 0.2x^3)\,dt + \sqrt{2D}\,dW$ with
$D = 0.5$ (exact mean 0.09675 from the stationary density), the plain series gives 0.150, 0.0638, 0.1056 at
$N = 1, 2, 3$ and the screened series 0.121, 0.0941, 0.0965.

With `spectra=`, `result['spectra']` holds the screened values: `S`, `S0`, the integrals and `C0` by order (they are
linear in $C$), and `tau_c` re-derived. With `covariance=True`, the covariance-matrix layout gains `Sigma_by_order`,
`Sigma_spread` and `Sigma_loops_by_ell`. The analyzer gates run on the plain loop run only. The G5 loop-parameter
advisory now names `resum=` among its remedies.

## Validation

### OU quartic, equal time

Relative error of $\langle x^2\rangle$ against the exact Boltzmann value ($\mu = D = 1$). L3 is the plain three-loop
series, and pms$N$ is the screened order $N$ at its PMS mass (pms2 at pms1's mass).

| $g$ | exact | L3 | Hartree | pms1 | pms2 | pms3 | $\kappa_3$ | spread |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.97214 | −4.2e−5 | −4.8e−4 | −2.9e−4 | +2.0e−5 | −1.1e−7 | 2.729 | 2.4e−5 |
| 0.1 | 0.81756 | −0.21 | −0.015 | −8.1e−3 | +2.7e−3 | +8.0e−6 | 2.288 | 2.2e−3 |
| 1 | 0.46792 | −589 | −0.072 | −0.036 | +0.022 | +8.6e−4 | 1.898 | 0.014 |
| 10 | 0.18890 | −1.6e6 | −0.118 | −0.057 | +0.044 | +2.8e−3 | 1.756 | 0.024 |
| 1000 | 0.021107 | −1.4e13 | −0.143 | −0.069 | +0.057 | +4.5e−3 | 1.698 | 0.031 |

These are the research prototype's numbers (L3, Hartree, pms3) to the digits shown. The prototype substituted the
mass parameter of the model by hand. Here the generic counterterm reproduces it, and the PMS $\kappa$ agrees with the
analytic series' stationary point to $10^{-8}$ (`test_pms_three_loops_strong_coupling`). Each row takes about 22 s
(25 runs at three loops).

### OU quartic, lag correlator

$\max_\tau|C - C_\text{exact}|/C_\text{exact}(0)$ on $[0, 6/m_H]$, against a detailed-balance Fokker-Planck
eigen-decomposition. In brackets: the relative error of $\tau_\text{int} = \int_0^\infty C/C(0)$.

| $g$ | Hartree | L3 | pms3 |
|---|---|---|---|
| 0.1 | 0.015 (−2.0 %) | 0.21 (−65 %) | 6.2e−4 (−0.3 %) |
| 1 | 0.072 (−9.6 %) | 589 (+468 %) | 8.3e−3 (−2.9 %) |
| 3 | 0.098 (−13 %) | 2.5e4 | 0.014 (−4.7 %) |
| 10 | 0.119 (−16 %) | 1.6e6 | 0.019 (−6.1 %) |

Hartree's single exponential decays too fast, and its error grows with $g$. pms3 holds $C(\tau)$ within 2 % up to
$g = 10$.

### Other models (tests)

- **Two coupled fields** (`ou_quartic_coupled`, symmetric couplings: a gradient system, so the reference is a 2-D
  Boltzmann quadrature), at $\varepsilon_1 = 1$, $\varepsilon_2 = 0.5$, $J = 0.5$:

  | method | max relative error over the entries of $\Sigma$ |
  |---|---|
  | Hartree | 8–13 % |
  | plain loops | ×15 at one loop, ×240 at two |
  | screened order 1 (PMS) | 9 % |
  | screened order 2 (at order 1's mass) | 2.5 % |

- **Jump model** (immigration, death, pairwise annihilation through the SDE front end's Poisson sources, against
  `dd.master_equation`):
  - Hartree equals the Gaussian closure of the exact moment equations to $10^{-10}$.
  - It moves the mean from the saddle's 1 % error to 0.01 %, and its variance is within the closure's $O(1/\Omega)$
    error (+0.9 %).
  - Screened order 2 exercises the noise counterterm ($Q_H \ne Q_0$) and is within $10^{-4}$ of the exact variance.
- **O(N)**: Hartree is exact as $N \to \infty$; the $N = 3$ closed form is tested.
- **Multiplicative noise**: Hartree matches the closed second-moment equation.
- **Colored noise**: Hartree matches the exact covariance of the embedded system in the linear limit.
- **`Dt` pencil** (`(tau*Dt + 1)*x`): $Q$ is transformed with the pencil, and Hartree equals the tree level.
- **Linear model** (two fields, cross pair): Hartree's $C(\tau)$, spectra and covariance layout equal
  `compute_cumulants` / `dd.spectra` / `dd.covariance_matrix` at tree level to $10^{-11}$. The screened series is the
  loop series (`kappa_source = 'trivial'`).

## Limitations and future work

- **Stencil derivatives.** The $\lambda$-derivatives come from a stencil of runs (step error about $10^{-9}$ at
  `step = 5e-3`). Exact derivatives are Fréchet derivatives of the state-space exponentials (block-triangular augmented
  exponentials, as in the downset sweep). They would replace the stencil with fewer runs and no step error.
- **2PI at next-to-leading order.** The self-consistent sunset (non-local $\Sigma(\tau)$, stationary Kadanoff-Baym
  equations) is a research item. It does not reuse the Phase J engine directly.
- **k ≥ 3.** The screened series is implemented for $k = 1$ (the mean) and $k = 2$. Applied to $k = 4$, the research prototype's 0-D series
  recovered 87–99 % of $\kappa_4$ at the variance's PMS mass. Hartree alone has $\kappa_{k\ge3} = 0$.
- **Not supported yet.** Kernel and delay models need their kernel states embedded first. Spatial models need a mass
  counterterm for the UV-divergent tadpole at $d \ge 2$.
- **Mean shift.** The screened series keeps the expansion point at the mean-field saddle. Hartree's own mean shift
  enters only through the loop tadpoles, so for an asymmetric model order 1 at $\kappa = 1$ is not exactly Hartree
  (it agrees to the next order in the couplings, `test_order_one_at_the_hartree_mass_is_the_hartree_mean`).
- **Multistability.** No single Gaussian, and no series around one, describes activated hopping.
- **Cost.** Each run of the scan repeats the multi-start mean-field solve. For a model whose loop order $N$ takes
  minutes, the screened series takes about 25 times that.

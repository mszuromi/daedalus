# Self-consistent Gaussian (Hartree) resummation: prototype and first results

Status: research prototype (`api/hartree.py`), not wired into `compute_cumulants` or `dd`. The numbers below come from
`docs/hartree_ou_quartic_comparison.py` (`sage -python docs/hartree_ou_quartic_comparison.py --sim`, about 2 minutes).

## What the prototype computes

For an Itô SDE $dx = f(x)\,dt + \text{noise}$ with white-noise covariance $Q(x)$, the self-consistent Gaussian is the
Gaussian $N(m, \Sigma)$ that solves

$$\langle f\rangle = 0,\qquad A\Sigma + \Sigma A^{\mathsf T} + \langle Q\rangle = 0,\qquad A = \langle \partial f/\partial x\rangle,$$

with every average taken in that Gaussian. In MSRJD language this is the Hartree(-Fock) truncation of the 2PI effective
action: the propagator is dressed by the tadpole, built from dressed lines and solved self-consistently, and the mean is
shifted by the dressed tadpoles. It sums all daisy and cactus diagrams. The lag correlator has a single matrix exponential,
$C(\tau) = \Sigma\, e^{A^{\mathsf T}\tau}$ for $\tau \ge 0$. `hartree(model, parameters)` reads $f$ and $Q$ off the
expanded action (`FieldTheory`, about the mean-field saddle). So it covers several fields and colored noise through the
Markovian embedding. Kernels, spatial models and time derivatives inside vertices are refused.

The comparison script also builds a second resummation from the existing loop pipeline. This is the loop series
reorganised around a trial mass $m$: $\mu \to m + \lambda(\mu - m)$ and $\varepsilon \to \lambda\varepsilon$, truncated at
$\lambda^N$. The literature calls this optimized or variational perturbation theory. The mass derivatives of each loop
order come from `compute_cumulants` runs at a 7-point stencil of masses. With $m = m_H$ the $N = 1$ term reproduces
Hartree exactly (checked). "scr$N$" uses $m = m_H$. "pms$N$" fixes $m$ by minimal sensitivity of the equal-time variance,
$dV_N/dm = 0$. Even orders have no stationary point, which is the known behaviour, so pms2 is undefined.

## OU quartic, $dx = -(\mu x + \varepsilon x^3)\,dt + \sqrt{2D}\,dW$, $\mu = D = 1$, $g = \varepsilon D/\mu^2$

The loop coefficients of the variance from `compute_cumulants` (max_ell = 3) are $1, -3, 24, -297$, which is the exact
Boltzmann series. Hartree's series is $1 - 3g + 18g^2 - 135g^3$. The $6g^2$ it misses is the sunset.

**Equal-time variance**, relative error against the exact Boltzmann value:

| $g$ | exact | 1 loop | 2 loops | 3 loops | Hartree | scr3 | pms3 |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.97214 | −2.2e−3 | +2.6e−4 | −4.2e−5 | −4.8e−4 | −7.8e−6 | −1.1e−7 |
| 0.1 | 0.81756 | −0.14 | +0.15 | −0.21 | −0.015 | −9.4e−3 | +8.0e−6 |
| 1 | 0.46792 | −5.3 | +46 | −589 | −0.072 | −0.32 | +8.6e−4 |
| 10 | 0.18890 | −155 | +1.3e4 | −1.6e6 | −0.118 | −1.07 | +2.8e−3 |
| 1000 | 0.021107 | −1.4e5 | +1.1e9 | −1.4e13 | −0.143 | −1.76 | +4.5e−3 |

As $g \to \infty$ the Hartree/exact ratio tends to $\Gamma(1/4)/(2\sqrt3\,\Gamma(3/4)) = 0.854$.

**Lag correlator** (exact: a detailed-balance Fokker-Planck eigen-decomposition). The first number is
$\max_\tau |C - C_\text{exact}|/C_\text{exact}(0)$ on $[0, 6/m_H]$. The second, in brackets, is the relative error of
$\tau_\text{int} = \int_0^\infty C/C(0)$:

| $g$ | Hartree | 3 loops | scr3 | pms3 |
|---|---|---|---|---|
| 0.1 | 0.015 (−2.0%) | 0.21 (−65%) | 9.4e−3 (−1.4%) | 6.2e−4 (−0.27%) |
| 1 | 0.072 (−9.6%) | 589 (+470%) | 0.32 (−59%) | 8.3e−3 (−3.0%) |
| 3 | 0.098 (−13%) | 2.5e4 | 0.68 | 0.014 (−4.7%) |
| 10 | 0.119 (−16%) | 1.6e6 | 1.07 | 0.019 (−6.1%) |

At $g = 1$ the Hartree rate $m_H = 2.303$ sits above the exact slowest Fokker-Planck rate, 2.068. The single exponential
decays too fast, and the error grows with $g$.

**Excess kurtosis** $\kappa_4/\langle x^2\rangle^2$ (the 0-D series $d_\ell = 0, -6, 126, -2682$; Daedalus reproduces
$-6, 126$ in `tests/test_all_k_boltzmann.py`):

| $g$ | exact | Hartree | 3 loops | pms2 | pms3 |
|---|---|---|---|---|---|
| 0.1 | −0.271 | 0 | −3.0 | −0.240 | −0.267 |
| 1 | −0.570 | 0 | −1.2e4 | −0.396 | −0.529 |
| 100 | −0.784 | 0 | −6.4e11 | −0.447 | −0.679 |

**Simulation** (Euler-Maruyama, 16 chains): the variance is 0.8166 ± 0.0010 at $g = 0.1$ (exact 0.8176), 0.4670 ± 0.0008
at $g = 1$ (exact 0.4679) and 0.1885 ± 0.0003 at $g = 10$ (exact 0.1889). $C(\tau)/C_\text{exact}(\tau)$ stays within
1% at the lags $0.5/m_H$, $1/m_H$ and $2/m_H$.

## Other models

- **O(N)**, $dx_i = -\mu x_i - (\varepsilon/N)\,x_i|x|^2 + \text{noise}$. Hartree is exact as $N \to \infty$. At $g = 10$
  the error falls from −12% ($N = 1$) to −2.0% ($N = 16$) and −0.14% ($N = 256$), close to $1/N$.
- **Symmetric double well**, $\mu = -1$, $\varepsilon = 1$, $D = 0.1$. Exact: $\langle x^2\rangle = 0.871$ and slowest
  rate $0.0335$ (Kramers hopping). The symmetric Hartree solution gives $\langle x^2\rangle = 0.414$ and rate 0.242. The
  broken-symmetry solution (start in one well) gives $\langle x^2\rangle = 0.877$, close to the exact value, but its rate
  is the intra-well 1.63, 49× too fast. At $D \ge 0.3$ only the symmetric solution exists. No single Gaussian describes
  barrier crossing.

## Reading

- The plain loop series is asymptotic. Already at $g = 0.1$ every truncation is off by 15–20%.
- Hartree is uniformly bounded, with an equal-time error of at most 14.3% for all $g$. It is the right first answer
  beyond the loop series at moderate and strong coupling. It misses every non-Gaussian cumulant ($\kappa_4 = 0$), the
  sunset ($6g^2$), the multi-exponential shape of $C(\tau)$ and activated hopping.
- Re-expanding the existing loop orders around a variational mass (pms3) beats Hartree everywhere tested. At $g = 1000$
  its equal-time error is 0.45%; it brings the lag correlator within 2% up to $g = 10$ and recovers most of $\kappa_4$.
  It needs no new diagrams, only the loop orders at a few masses.

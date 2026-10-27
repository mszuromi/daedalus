# Exact master-equation oracle (`dd.master_equation`)

`daedalus/api/master_equation.py` solves a lattice jump process exactly (to the box truncation, which it measures), so the jump
models of the SDE front end can be checked against something sharper than a simulation. The solver uses numpy, scipy
(sparse) and sympy (rate strings) only.

## The process

State `n` in `Z^d`, physical state `x = origin + h n` (`h` is the lattice step, the inverse system size). Each transition
`(rate, jump)` takes `n -> n + jump` at rate `rate(x)` (events per unit time). Rates are strings in the field names, the
`params` and `h` (`^` is a power), or callables `rate(x_1, ..., x_d)` of numpy arrays.

```python
import numpy as np
import daedalus as dd

# Schlogl-type birth-death in system size Om: x = n / Om
me = dd.master_equation(
    [('Om*(a*x^2 + c)', +1), ('Om*(b*x^3 + d*x)', -1)],
    fields=['x'], params=dict(Om=50.0, a=1.0, c=1.0, b=1.0, d=1.0),
    h=1 / 50.0, box=(0, 200))                 # index range n = 0..200; box='auto' uses the linear-noise width

st = me.stationary()                          # StationaryResult: p, residual, truncation_flux, edge_mass, ...
cm = me.cumulants(4)                          # cm[k]: full symmetric tensor of kappa_k; cm.cumulant('x', 'x')
cc = me.correlation(np.linspace(0, 5, 51))    # cc.C[t, j, k] = <dx_j(0) dx_k(tau_t)>
D = me.integrated_correlation()               # int_0^inf C_jk dtau
```

`box` is `[(lo, hi), ...]` (inclusive integer indices, one pair per field) or `'auto'`: the linear-noise box
`mean +- K sigma` (`K = 9`) about the mean field from `guess`, clipped by `lower` / `upper` (physical bounds; pass `lower=0` for counts, or the box reaches negative states and the rates there are refused).
`me.lna()` returns the mean-field fixed point, the Jacobian, the diffusion matrix and the Lyapunov covariance.

## Errors and diagnostics

| Quantity | Where | Meaning |
|---|---|---|
| `residual` | every result | `|Q^T p|_1 / sum |Q_nn| p_n` after the solve and two refinement steps (about 1e-16) |
| `truncation_flux` | every result | rate of jumps suppressed at the box edge divided by the total jump rate, under `p` |
| `edge_mass` | stationary, cumulants | `p` summed over the states that have a suppressed jump |
| `negative_mass` | stationary | mass clipped from negative LU round-off |
| `krylov_error`, `krylov_dim` | correlation | the last relative change between successive Krylov projections, and the dimension used (0 for `'direct'`) |

The stationary LU pins one state at `p = 1`; it pins the mean-field fixed point (a far-tail pin makes the solve meaningless), and the solve re-pins and finally refuses rather than return a vector with a large residual or negative mass. An edge where the rate vanishes anyway (the death rate at 0 particles) suppresses nothing and costs nothing. When the
flux exceeds `truncation_tol` (default 1e-10) a `TruncationWarning` is given (`on_truncation='raise'` or `'ignore'`).
The cumulant error from the box is about the flux times the distance of the edge from the mean to the power `k`; a box
of `mean +- 9 sigma` typically gives a flux near 1e-10 for a Poisson-like law and cumulants good to about 1e-8.

`policy='reflect'` (default) drops a jump that would leave the box; `'periodic'` wraps it. The stationary law is solved
on the unique closed communicating class; several closed classes (a conservation law, or a box that splits the state
space) are refused, and transient states get `p = 0`. A negative rate on the box is refused. Sage's `python` ignores
warnings by default, so read `st.truncated` or `truncation_flux` in scripts run with `sage -python`.

## Lag correlations

`C_jk(tau) = sum_n p_n f_j(n) [e^{tau Q} f_k](n)`, `f = x - <x>` (the package's `C(tau) = kappa(x_j(t), x_k(t + tau))`);
`C_jk(-tau) = C_kj(tau)`.

- `method='direct'`: marches `e^{Dt Q}` over the sorted grid with `scipy.sparse.linalg.expm_multiply` (round-off
  accurate; cost grows with `tau_max` times the largest exit rate, so it is slow for stiff networks).
- `method='krylov'`: shift-and-invert Arnoldi on `(I - gamma Q)^{-1}` with full re-orthogonalisation; one LU serves the
  whole grid. `gamma` defaults to the geometric mean of the smallest and largest positive `tau`. The dimension grows in
  steps of 20 to `m_max` (300) until the change is below `krylov_tol` (1e-10); otherwise a `KrylovWarning` is given and
  `krylov_error` says by how much.
- `method='auto'` (default) takes `'direct'` when the estimated work `nnz d (tau_max |Q|_1 + ...)` is below
  `direct_budget`, else `'krylov'`. Memory for the Krylov basis is capped by `max_memory_gb`.

A stiff two-timescale network (rates 100 and 0.03, lags from 1e-4 to 120) needs a Krylov dimension of about 160 and
agrees with the exact Lyapunov result to the box truncation (tests/test_master_equation.py).

## Diffusion (Kramers-Moyal) limit

`me.kramers_moyal(refine=(1, 2, 4))` compares the jump process with the Ito SDE of the same first two jump moments,
`a = h sum_k r_k j_k`, `D = h^2 sum_k r_k j_k j_k^T`. The transitions are grouped by direction `+-j`, with `S` the sum and
`J` the difference of the two rates; on the lattice `h/m` the diffusion is the jump process with rates
`(m^2 S +- m J)/2` in the directions `+-j`. At `m = 1` this is the jump process, and `M(m) = M_KM + c_2 m^-2 + O(m^-4)`,
so Lagrange extrapolation to `m^-2 = 0` over `refine` gives `M_KM`. The result has the extrapolated cumulants, the
values at each `m`, an error estimate (extrapolation over all of `refine` against `refine[1:]`) and, with `taus=`, the
extrapolated correlation.

The diffusion matches the jump process in the first two cumulants only: for immigration-death with rates `lam` and `mu x`,
`kappa_4 = lam/mu` for the jump process and `3 lam / (2 mu)` for the diffusion.

## Bridge to `compute_cumulants`

For string rates, `me.to_sde(kind='jump')` builds the SDE-front-end model: zero drift and one state-dependent
`Poisson(rate=r_k(x), size=h)` source per transition, coupled to field `i` with coefficient `jump_i`. `kind='diffusion'`
gives the Kramers-Moyal SDE (drift `a(x)`, Gaussian sources of variance `h^2 r_k(x)`). `h` is a model parameter.

`me.compare_with_daedalus(model=None, ks=(1, 2), max_ell=1)` runs `compute_cumulants` at `tau = 0` for every independent
entry of `kappa_k` (`k <= 3`) and returns a `ComparisonResult` (`.rows`, `.table()`): the exact value, the partial sums
over loop orders `0..max_ell` (the mean is `x* + sum_ell C^(ell)`) and the relative errors. Pass a prebuilt `model` and
`parameters=` to reuse one build over several system sizes.

**What the loop expansion should and should not match.** The expansion is in the jump size `h` about the mean-field
saddle. For `kappa_k` the tree level is `O(h^{k-1})` and every loop adds a relative factor `O(h)`, so at `max_ell = L` the
relative error against the oracle is `O(h^{L+1})` at fixed `x`. It should therefore converge to the oracle as `h -> 0`
(for the Schlogl example `x* = 1`, `Om = 25, 50, 100`: the tree error halves per doubling, the one-loop error falls
by about 4), and for linear rates the tree is exact for every cumulant, with all loops vanishing. It should not be
expected to match at fixed `h` beyond that order; near a bifurcation or for a bistable law (the saddle is not the
whole law); where the box or a state-space boundary (`x = 0` for counts) cuts the law; in the tails; or for the
lattice itself, which the oracle has and the expansion does not. Check `truncation_flux` and the saddle before reading a
disagreement as a bug in either.

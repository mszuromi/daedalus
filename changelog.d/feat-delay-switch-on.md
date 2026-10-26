#### Switch-on transients (`t_min`) now work with delay kernels

`compute_cumulants(..., t_min=, times=)` refused models with delay kernels ("computed for the stationary state only").
It now runs them: the fluctuations and the history of every delayed field are zero before `t_min`, and the tree and the
loops both use the Pade approximant of the delay on the state-space route (the exact delayed tree is the *stationary*
covariance, a boundary-value problem, so it has no transient counterpart here). The order search runs as for a
stationary run on `result['C_times']`.

- **Accuracy.** The Pade transient converges more slowly than the stationary result inside the first delay after
  `t_min` (the delayed term is still zero there): relative to the Ornstein-Uhlenbeck closed form the tree is off by
  7e-3, 2e-3 and 2e-4 at n = 4, 8 and 16 (delay 0.5, a = 1). Far from `t_min` it agrees with the stationary exact tree
  (2e-4 at n = 16). With the default `tol = 1e-8` the search therefore ends at `n_max` with
  `status = 'target_missed'` and a `DelayOrderWarning` carrying the estimate; give `delay={'tol': ...}` for the accuracy
  you need or `delay={'order': n}` for a fixed order. `result['delay']['tree_note']` says the tree is the Pade one.
- **Still refused:** `initial_covariance` with a delay kernel (`SwitchOnUnsupported`: the history of the delayed fields
  before `t_min` is zero, and the Pade realization of a delay is not a first-order model with the identity response).
- No stationary result changes.

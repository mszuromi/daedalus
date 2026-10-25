#### Covariance matrix over many fields in one call, and parameter derivatives (k = 2)

`dd.covariance_matrix(model, fields, max_ell, tau_grid=..., derivatives=[...])` (`api/covariance.py`; also
`dd.Config(covariance={...})` with `dd.run`) returns `Sigma[n, i, j] = <dx_i(0) dx_j(tau_n)>` for every pair of
`fields`, per loop order (`Sigma_by_ell`) and in total. Before this, the full matrix took one `compute_cumulants` call
per entry, and each call repeated the expansion, the propagator, the mean field and the poles.

- **Shared stages.** `api.compute.stage_memo()` is a context in which `compute_cumulants` calls at one parameter
  point (the same model dict, parameter values, Taylor order, cache setting and mean-field options) reuse the
  expansion, the propagator, the mean field and the poles and residues of the first call. It hands on the objects
  that call built, so a later call computes what it would have computed alone. Outside the block nothing is shared.
  The diagrams of each pair (one external-leg typing) are enumerated, classified and evaluated once.
  - At one loop, the two-field OU quartic matrix takes about 0.55x the CPU of the three separate calls.
  - The four-leg linear Hawkes matrix (`n_1, n_2, v_1, v_2`) takes 0.42x the CPU of the ten separate calls.
- **Bit for bit.** Every computed entry is the separate `compute_cumulants` result on the same grid, per loop order
  too. The tests check this on OU quartic two-field, a three-field linear-plus-cubic model and the linear Hawkes
  population model.
- **Lower triangle (`pairs='upper'`, default).** The pairs `i <= j` are run, and the `(j, i)` entry is `C_ij(-tau)`
  wherever `-tau` is on the grid and `tau != 0`.
  - `tau = 0` and points without a mirror get their own `(j, i)` run on those points only. At `tau = 0` the Ito left
    limit `C_ji(0-)` is the right limit `C_ij(0+)`, which differs from `C_ij(0)` when a spike train drives the other
    field. For example, with spikes `n` driving `v`, `C_nv(0) = 0.056` and `C_vn(0) = 0.222`.
  - `pairs='all'` runs every ordered pair.
- **Parameter derivatives.** `derivatives=['J1', 'w[0,1]', ...]` (or `{'parameters': [...], 'step': 1e-3,
  'abs_step': {...}}`) gives `dSigma/dtheta` by fourth-order central differences of the shared pass. The steps are
  `h` and `2h`, with `h = 1e-3 |theta|`, and each parameter costs four extra passes.
  - `error` is a conservative estimate: the truncation estimate of the second-order quotient plus a round-off floor.
  - Against the exact Lyapunov / Frechet derivative of a linear two-field OU, the error is about 1e-11.
  - At one loop it was compared with plain central differences of separate full-pipeline runs, with `mu1 = -0.4` so
    that `mu1` moves the saddle. The gap is the plain differences' own `O(h^2)` error: 1.3e-5 relative at
    `h = 1e-3 |theta|` and 1.3e-7 at `1e-4 |theta|`.
  - The saddle's sensitivity `dx*/dtheta = -J^{-1} dF/dtheta` is computed by implicit differentiation of the
    mean-field equations (`dmf`). It is checked against the stencil saddles (`dmf_fd`), and a stencil that crosses
    to another branch is refused with `CovarianceDerivativeError`.
  - Exact tangents of Phase J and a complex step were not used. The state-space realization refuses complex inputs,
    and differentiating the matrix-exponential and pole-cluster routes would mean rewriting them.
- **Refused.** Spatial models, `t_min`, `spectra`, `points`, `external_fields` and `output_*`. The other
  `compute_cumulants` options go to every run, and each run keeps its own memory guard. By default each pair's
  result is released once its values are read (`keep_entries=True` keeps them).

#### Symbolic output: the one-loop mean, diagram by diagram

`dd.one_loop_mean(model)` (`daedalus.api.symbolic_mean`) writes the `k = 1`, `ell = 1` mean shift as its tadpole
diagrams: each `Tadpole` is a quadratic drift vertex `V^(2)_{r; pq}` of the expanded action with its loop closed by the
equal-time tree covariance, attached to the external field by `G(omega = 0)`:
`<dx_a>^(1) = -sum G(0)_{a r} V^(2)_{r; pq} C0_pq(0)`. One loop needs the drift to degree two only, so the result is
exact at this order for any drift. `C0_pq(0)` comes in four forms (`form=`): `'named'` (the symbol
`kappa^(0)(dx_p, dx_q)`, which makes the sum equal `mean_relation(model).loop_order(1)`), `'integral'` (the held
Fourier integral `(1/2 pi) int C0_pq(omega) d omega` of the symbolic tree covariance), `'rational'` (its exact value,
rational in the parameters and the saddle, from the existing state-space route of `dd.symbolic_cumulant`) and
`'residues'` (the mode sum of `tree_modes` at `tau = 0`: explicit roots for factors of degree <= 2, `RootOf` above).
`G(0)` is explicit or named (`propagator='named'`, the default of `latex`). Output as Sage expressions, sympy and LaTeX;
`evaluate` gives numbers (`'integral'` by quadrature, an independent check of `'rational'`). The expressions are not
simplified towards closed forms. Every form reproduces the pipeline's `k = 1`, `ell = 1` mean to 1e-10 on a 2-unit and a
3-unit rate network with distinct `tau` (the 3-unit poles are `RootOf`), a cross vertex `c x y` whose loop is the
off-diagonal `C0_xy(0)`, an `exp` drift, and a spike-train model with exponential synaptic kernels. Refused: an entry with a contact term (no equal-time value); the
`'integral'` / `'residues'` forms where `tree_covariance` refuses the model, and `'rational'` where `symbolic_cumulant`
does (more than 12 symbols: fix some with `values=`).

`dd.mean_relation` now accepts `values=` for declared parameters that enter only the propagator (a kernel's time
constant), which it refused as unknown names. No existing number changes.

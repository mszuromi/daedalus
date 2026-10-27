#### Symbolic output: the exact stationary mean relation

`dd.mean_relation(model)` (`daedalus.api.symbolic_mean`) writes the stationary equation of motion
`<dS/d xt> = 0` symbolically: `K(omega = 0) <dx> = -sum_m V^(m) : <dx^m>`, with `K(0) = G(0)^-1` from the
symbolic propagator, the drift vertices `V^(m)` read off the (1, m) sectors of the expanded action, and the raw
moments written in connected-cumulant symbols (`kappa_dv1`, the mean shift `<v_1> - v*_1`; `kappa_dv1_dv2`;
`kappa_dv1_dv1_dv2` ...). It holds at all loop orders; the noise sectors do not enter. `loop_order(L)` gives the
`L`-loop mean shift from the lower-order cumulants (`kappa_1^(1) = -G(0) V^(2) : kappa_2^(0)`, then the `dx^2` and
third-cumulant terms at two loops). Output as Sage relations, sympy (`sympy.Eq`) and LaTeX; `evaluate` substitutes
numbers. A polynomial drift is exact (its degree is read from the mean-field sector of the action); a
non-polynomial one needs `order=` and the relation reports its Taylor truncation (`truncation`, `exact`); loop order
`L` needs `order >= 2 L`. Refused: spatial models, a time derivative or kernel on a nonlinear drift term, a
singular `K(0)`, unknown `values=` names. For `tau_l dv_l = -(v_l - b_l) + sum_k W_lk v_k^2 / a + noise` the
relation is `(1 - W diag(phi'(v*))) dv = W diag(1/a) (Var v + dv^2)` with no `1/tau`, and its one- and two-loop
orders equal the pipeline's `k = 1` means to 1e-10 (2 and 3 units, distinct `tau`); a cubic drift satisfies it with
the moments of its exact stationary density to 1e-12. No existing number changes.

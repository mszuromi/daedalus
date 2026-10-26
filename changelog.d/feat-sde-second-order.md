#### SDE front end: second- and higher-order equations, pure integrators, operator-form paper lines

- **Higher-order equations.** An lhs that is a polynomial of order `N > 1` in the time derivative acting on the field
  (`equation('x', '(m*Dt^2 + gamma*Dt + k)*x', rhs)`, or `m*dt^2*x + ...` with a declared `dt`) is embedded as a
  first-order system: the velocities `xdot` (`xddot`, ...; or `velocity='v'` / a list) are declared as fields,
  `Dt*x = xdot`, and the top equation is `m*Dt*xdot + gamma*xdot = rhs - k*x`. Noise coupled into `x` (declared
  before or after the equation) forces the top equation. The model is the one the hand-written first-order system
  builds (identical action, equations and dict, tested). Before, the engine refused any `Dt^2` term. An lhs with
  other fields or field-dependent coefficients is refused with a message.
- **Pure integrators.** `drift('x', 'v')` (an rhs without `x`) is accepted: its mean-field equation `0 = v` constrains
  the other fields and `x*` comes from the system solved together (a damped forced `v` gives `x* = F/k` exactly). An
  rhs without any field (a random walk) is still refused, now saying so.
- **Operator form of paper-style lines.** `m.sde('(tau*Dt + 1)*v = E + sum(w[i,j]*v[j] for j in pop) + s dW/dt')`: a
  linear operator on the left (on one field), the right side per unit time, noise terms `c dW/dt` (`c dN(rate)/dt`,
  `c dX_name/dt`, marks as usual). This also covers drifts with population sums, which the `dx = (f) dt` form cannot
  split. `sde_text()` now writes any lhs other than `a*Dt*x` (higher order included) in this form; it refuses
  velocities named with `velocity=` (the text names them `xdot`, ...).
- Tests (`tests/test_sde_second_order.py`): a Langevin oscillator and an RLC circuit (`<q^2> = C kT`, `<i^2> = kT/L`)
  against the exact Lyapunov covariance and C(tau) to 1e-11; a third-order equation; embedding vs the hand-written
  system (tree and one loop on a Duffing-type oscillator); the pure-integrator saddle; operator-form lines; round
  trips; the simulator on the embedded system; refusals. Existing models are unchanged (golden snapshot).

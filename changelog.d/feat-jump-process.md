## Reaction / transition builder for jump processes (`dd.JumpProcess`)

- New `api/jump_process.py`, exported as `dd.JumpProcess`: declare species, parameters, a system size `Omega`, and mass-action
  `reaction('A + B -> C', rate='k')` (combinatorics `'stochastic'` falling factorials by default, `'macroscopic'`,
  `'rate_law'`; reversible `<->`) or `transition(rate, jump)` channels. `sde()` / `build()` give the SDE-front-end model with
  zero drift and one state-dependent `Poisson(rate, size=1/Omega)` source per transition (action text identical to the
  hand-written model); `dd.run(jp, cfg)` accepts the builder directly. `to_master_equation()` and `compare()` hand the same
  transitions to the exact oracle (`dd.master_equation`), `convergence(omegas)` repeats the comparison over system sizes.
  Negative rates (probed on the lattice at the defaults) and ill-formed jumps / reactions are refused (`JumpProcessError`).
- `MasterEquation.compare_with_daedalus` no longer passes `h` to a model that does not declare it (a model written in the
  system size). No numbers change.
- Tests: `tests/test_jump_process.py` (immigration-death exact Poisson, dimerisation one-loop convergence with the tree error
  ~ 1/Omega and the one-loop error ~ 1/Omega^2, a correlated two-species linear network against Lyapunov, identical action
  text, refusals). Docs: `docs/jump_process.md`.

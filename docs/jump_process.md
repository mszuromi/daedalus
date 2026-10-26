# Reaction / transition builder (`dd.JumpProcess`)

`api/jump_process.py` is a front end for jump processes (chemical kinetics, shot noise). It emits the Ito jump action the
SDE front end already supports and validates it against the exact master-equation oracle (`docs/master_equation.md`).

```python
import daedalus as dd
jp = (dd.JumpProcess('dimerisation')            # system-size parameter 'Omega' by default
      .species('A', 'B')
      .parameter('s', 2.0).parameter('k1', 1.0).parameter('k2', 0.5)
      .system_size(50)                           # default value of Omega
      .reaction('-> A', rate='s')
      .reaction('2A -> B', rate='k1')
      .reaction('B ->', rate='k2'))
print(jp.summary())
model = jp.build()                               # for compute_cumulants; dd.run(jp, cfg) builds it too
me = jp.to_master_equation()                     # exact lattice process, h = 1/Omega, box='auto'
table = jp.compare(ks=(1, 2), max_ell=1).table() # loop-order cumulants vs the exact values
```

## The state and the rates

The state is the density `x = n / Omega` (`n` the counts); an event changes it by `jump / Omega`.

* `transition(rate, jump, name=None, per_volume=False)`: `rate` is the event rate (events per unit time) in the
  densities, the parameters and `Omega`; with `per_volume=True` it is the rate per unit system size and `Omega * rate` is
  used. `jump` is `{species: integer}` or one integer per species in declaration order.
* `reaction('A + B -> C', rate='k', combinatorics='stochastic')`: mass action, event rate `Omega * k * g(x)`,
  `g` the product over the reactants (stoichiometry `nu`) of
  * `'stochastic'` (default): `x (x - 1/Omega) ... (x - (nu-1)/Omega) / nu!`, the chemical master equation
    (`n (n-1) / 2` pairs of identical molecules);
  * `'macroscopic'`: `x^nu / nu!`;
  * `'rate_law'`: `x^nu` (`k` is the coefficient of the rate law).

  The conventions have the same mean-field limit up to the `1/nu!` of the last two; the first differs from
  `'macroscopic'` at relative order `1/Omega`, the order of the one-loop correction. Both sides may be empty (`''`, `'0'`
  or `'∅'`), stoichiometries are written `2A` or `2 A`, and `'A <-> B'` takes `rate=(forward, reverse)` and declares two
  transitions (`name_f`, `name_r`). A catalyst (on both sides) enters the rate and has no coupling.
* Transitions are named `t0, t1, ...` unless `name=` is given; the names are the noise sources of the SDE.

## What is built

`sde()` / `build()`: zero drift and, per transition, `noise(name, Poisson(rate=r(x), size='1/Omega'), couples={species:
jump component})` (zero components omitted). The model is the one written by hand with the SDE front end: the action
text, equations and declarations are equal (checked in `tests/test_jump_process.py`). Parameters with a positive default
are declared `domain='positive'` unless `parameter(..., domain=...)` says otherwise. `check()` and `latex()` pass through.

## Validation against the master equation

`to_master_equation(params=None, box='auto', **kw)` is `dd.master_equation` on the same transitions, `h = 1/Omega`
(`params` overrides parameter values, including `Omega`; `'auto'` uses the linear-noise box with lower bound 0, so give
an explicit `box` for a bistable network). `compare(params=None, box='auto', ks=(1, 2), max_ell=1)` tabulates the exact
cumulants against the loop-order partial sums. The expansion is asymptotic in `1/Omega`: at `max_ell = L` the relative
error is `O(Omega^-(L+1))` (dimerisation: tree error halves and one-loop error quarters per doubling of `Omega`), and it
need not match at a fixed small `Omega`, near a bifurcation or at count boundaries. `convergence(omegas, box=callable)`
repeats the comparison over system sizes.

## Refusals (`JumpProcessError`)

* a negative rate: each rate is probed on the lattice `n/Omega` (counts 0..400, `Omega` its default or 100) at the default
  parameters when declared (`check_rates=False` switches this off; a rate with a parameter without a default is not
  probed); the stochastic factor `x (x - 1/Omega)` is zero on the lattice and accepted;
* a jump that is not an integer vector, the zero jump, an undeclared species, the wrong number of components;
* an undeclared name in a rate, an unparsable rate, a reaction without an arrow, a reaction that changes nothing, an
  unknown `combinatorics`, a repeated name;
* `to_master_equation` / `compare` with a parameter that has no value.

A rate that is negative only at parameter values chosen later is not caught at declaration; the oracle refuses a negative
rate on its box.

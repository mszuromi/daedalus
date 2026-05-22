# Scalar-mode field coercion fix (May 2026)

## Problem

Scalar-mode theories (no `.population()` declared — auto-pop size 1)
broke whenever the user's action text passed a field through one of:

  * a Sage builtin: `exp(nt)`, `log(1 + …)`, `sqrt(…)`, etc.
  * a user-declared formal function via `.define_function(...)`:
    `f(v)`, `g(w)`, etc.
  * any `_IndexedFormalFunction.__call__` site.

The trigger: in scalar mode the framework bound fields like `xt`, `v`,
`nt` to `_FieldScalar` / `_FullPhysicalField` proxy objects that
support BOTH bare arithmetic (`xt * x`) AND `[0]` indexing (`xt[i] *
x[i]` for legacy actions).  Those proxies expose `_sage_()` to convert
to a bare SR expression, but Sage's `BuiltinFunction.__call__` does
NOT honor `_sage_()` — it calls `SR.coerce(arg)` which fails with
`no canonical coercion from <_FieldScalar / _FullPhysicalField> to
Symbolic Ring`.

This is what surfaced first in the compartmental Bernoulli theory
(`exp(nt) - 1`, `log(1 + (exp(mt)-1)*g(w))`, `f(v)`).  But it was
latent in every scalar theory the moment the action text invoked any
of those builtins.

## Fix

Three coordinated edits in `pipeline/theory_compiler.py`:

  1. Added `_unwrap_field_arg(a)` helper that detects size-1 field
     wrappers by class name and returns their bare SR expression.
     No-op for plain SR objects.

  2. Wrapped every numeric Sage builtin in `_builtin_namespace()`
     through `_unwrap_builtin(...)` so `exp`, `log`, `sin`, `cos`,
     `tan`, `sqrt`, `heaviside`, `dirac_delta` all auto-unwrap their
     arguments before passing to `BuiltinFunction.__call__`.

  3. Added `_IndexedFormalFunction.__call__(self, *args)` (previously
     only `__getitem__` was defined) so scalar-mode `f(v)` resolves to
     `sr_function(f'{name}_1')(*args)` — same as `f[0](v)` produces.
     Arguments are unwrapped via `_unwrap_field_arg` before the call.

Plus, in `pipeline/_mean_field_dae.py`:

  4. `_PhiCallableList` wraps the per-population list of phi callables
     so that scalar-mode `f(v)` works in DAE equation RHSes (the
     previous binding bound `f` to a Python list, which isn't
     callable).  Indexed access `f[i](v)` still works.

## What's preserved

Field bindings retain the original `_FieldScalar` / `_FullPhysicalField`
wrapping in size-1 mode.  Legacy actions like
`sum(xt[i]*x[i] for i in pop)` still parse — `__getitem__` on the
wrapper returns the SR var at the requested position.

Multi-pop theories are untouched: their fields are still bound as
ordinary Python lists, and the user always writes `xt[i] * x[i]` etc.

## Regression coverage

Ten existing theories smoke-tested post-fix (extract_vertex_types
succeeds for each):

  * `single_population_quad_exp_test`   — phi[i](v[i]) indexed
  * `single_population_linear_delta_spikes_test`
  * `ou_quartic_double_well`            — scalar `xt[i]*x[i]^3`
  * `ou_quartic_two_dim`                — 2D scalar
  * `ou_sextic`                         — higher-order scalar
  * `single_population_bistable_demo`
  * `single_population_linear_delta_voltage_test`
  * `single_population_conductance_test`
  * `single_population_quad_spike_reset_test`
  * `multipopulation_test`              — 2-population E/I

`tests/test_grouped_vs_perdiag.py` (6 tests) passes.

## What this unlocks

Compartmental neuronal-network actions of the form

```
nt*n - (exp(nt) - 1)*f(v)
+ mt*m - n*log(1 + (exp(mt) - 1)*g(w))
+ vt*((Dt + 1)*v - ES)
+ wt*((Dt + 1)*w - ED)
```

now parse and Taylor-expand cleanly.  The framework's *existing*
multivariate `taylor()` pass (at `msrjd/core/field_theory.py`
lines 781-786) handles the Bernoulli CGF `log(1 + (e^mt - 1)g(w))`
correctly — produces every order's contribution to all the bigrade
sectors.

For the minimal compartmental probe (single soma + dendrite, linear
`f(v) = a*v`, linear `g(w) = a*w`) the framework extracts 9 vertex
types at `taylor_order=4`:

  * `nt^k * dv`   — Poisson Taylor of soma rate f(v)         (k = 2, 3)
  * `mt^k * dn`   — Bernoulli Taylor in n (counts), order k  (k = 2, 3)
  * `mt^k * dw`   — Bernoulli Taylor in w (dendritic V)      (k = 2, 3)
  * `mt^k * dw^j` — mixed Bernoulli k-cumulant on dendritic V
  * `mt * dn * dw` — the linear conditioning vertex (coupling
    dendritic spikes to somatic spikes via dendritic voltage)
  * `mt^2 * dn * dw` — second-order conditioning

This is precisely the diagrammatic structure expected from the
Teasley & Ocker (PRX Life 2026) compartmental-neuron action; with this
fix the framework can express their k-cumulant calculations directly
from text-driven theory files.

## Known followups

  * **DAE solver stability classification** for the compartmental
    saddle currently classifies the trivial root as unstable because
    the algebraic constraints `n = f(v)`, `m = n*g(w)` contribute
    zero / undefined eigenvalues.  The MF verification falls back to
    all-ones, which is wrong for compartmental.  Workaround: skip
    `mf_bg_conditions` for compartmental tests until the DAE solver
    learns to score algebraic-vs-differential constraints differently
    (it should treat algebraic eigenvalues as automatically stable
    when the corresponding Jacobian block is invertible).

  * **Heavy-defaults theory file** for the 2-neuron-with-dendrites
    paper port — pending the stability fix above so the saddle
    verification passes without manual override.

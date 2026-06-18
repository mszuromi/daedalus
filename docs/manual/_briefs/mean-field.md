# Mean-Field Saddle & DAE Solver — Technical Brief

**Subsystem slug:** `mean-field`
**Primary files:**
- `pipeline/_mean_field.py` (the *legacy iteration* solver — 553 lines)
- `pipeline/_mean_field_dae.py` (the *DAE multi-root* solver + linear stability — 977 lines)

**Supporting files referenced:** `pipeline/compute.py` (routing), `pipeline/theory.py`
(`TheoryBuilder.equation`, `.stability_analysis`), `pipeline/theory_compiler.py`
(`_FieldScalar`, the `ns` namespace, saddle-derivative renaming), `pipeline/_propagator.py`
(`compute_poles_and_residues`, the consumer of the saddle), `pipeline/_precompute.py`
(a second consumer that re-solves the saddle).

---

## 1. Overview

### What this subsystem does, in plain language

A field theory written in the MSR-JD (Martin–Siggia–Rose–Janssen–De Dominicis)
formalism is built around a **classical background** — the value of each physical
field when *all fluctuations are switched off*. In statistical-physics language this
background is the **mean-field saddle point** of the action: the deterministic
steady state about which we then expand in loops (the perturbative diagrammatic
series). For a network of `n` neurons (or `n` populations, or a single coarse-grained
field), the saddle is the firing rate `n*`, the membrane voltage `v*`, the auxiliary
rate `m*`, etc., that solve the network's self-consistency / steady-state equations.

This subsystem's single job is: **given a fully specified model and concrete numerical
parameter values, find that saddle and package it into a substitution dictionary that
the rest of the pipeline can plug into symbolic expressions.** Concretely it produces
`num_params`, a dictionary mapping every Sage symbol that can appear in a propagator or
vertex (parameters like `E1`, `w12`; saddle values like `nstar[0]`, `vstar[0]`;
derivative-of-transfer-function symbols like `phi1_1`, `phi2_1`) to a Python `float`.

There are **two completely independent solver implementations** here, chosen by whether
the model declares its steady state as explicit `.equation(...)` calls:

1. **Legacy iteration solver** (`_mean_field.py`, `solve_mean_field`). For theories
   whose saddle is described as a *self-consistency closure*: `n*_i = phi(v*_i)` where
   `v*_i` is some algebraic function of all the `n*_j`. It iterates only over the
   "rate-like" saddle variables and treats the rest (voltages) as compound expressions
   evaluated on the fly. Single-pop and heterogeneous-pop (`E`/`I`) branches.

2. **DAE multi-root solver** (`_mean_field_dae.py`, `solve_mean_field_dae` and its
   compatibility wrapper `solve_mean_field_dae_compat`). For theories that declare a
   *differential-algebraic system* via `TheoryBuilder.equation(lhs, rhs, population)`.
   It sets the time-derivative operator `Dt → 0` on every equation, solves the resulting
   algebraic root-finding problem with **many random restarts** (multi-start Newton),
   deduplicates the roots it lands on, optionally classifies each root's **linear
   stability** via a generalized eigenvalue problem, and **selects one root** by index.

The name "DAE" (differential-algebraic equation) is used because the declared system
mixes genuinely differential equations (those carrying `Dt`, e.g. a voltage relaxation
`(tau*Dt + 1)v = ...`) with purely algebraic constraints (those without `Dt`, e.g. a
rate readout `n = phi(v)`). At the saddle the differential equations collapse to
algebraic ones (`Dt→0`), but the *differential structure is remembered* and reused for
stability analysis.

### Where it sits in the end-to-end pipeline

```
   TheoryBuilder.build()                 (pipeline/theory.py)
        │   produces the `model` dict + a compiled FieldTheory `ft`
        │   (ft._ns carries the Sage symbols; model carries the declarations)
        ▼
   compute_cumulants(...)                (pipeline/compute.py)
        │   [1] expand action → vertices/sources
        │   [2] build_propagator → K_ft, G_ft, adj_ft   (symbolic, in ω)
        │
        ├──[3] MEAN-FIELD SOLVE  ◄────────  THIS SUBSYSTEM
        │        if model['equations']:  solve_mean_field_dae_compat(ft, model, fundamental, ...)
        │        else:                    solve_mean_field(ft, model, fundamental, ...)
        │        → mf['num_params']   {SR symbol → float}
        │
        ▼
   compute_poles_and_residues(prop, num_params)   (pipeline/_propagator.py)
        │   substitutes num_params into K_ft, finds retarded poles ω_k,
        │   residue matrices C_mats → the free propagator G(τ)
        ▼
   Phase J / time-domain correction machinery → the connected cumulant C(τ)
```

**What feeds it:**
- `ft` — a compiled `FieldTheory` instance. The solver only touches `ft._ns` (the Sage
  symbol namespace) and `ft.taylor_order` (how many transfer-function derivatives to
  precompute). In the DAE path, `ft` is used *only* by the compat wrapper to map saddle
  names back onto `ft._ns` arrays.
- `model` — the theory declaration dict from `TheoryBuilder.build()`. Carries
  `parameters`, `kernels`, `populations`, `physical_fields`, `functions`,
  `phi_concrete`, `mf_bg_conditions`, `iteration_saddles`, `equations`,
  `stability_analysis`.
- `fundamental` — the user's concrete numerical parameter values keyed by parameter
  *name* (e.g. `{'E': [...], 'w': [[...]], 'tau': 1.0}`).

**What consumes its output:**
- `compute_poles_and_residues(prop, num_params)` in `pipeline/_propagator.py` — the
  *primary* consumer. It substitutes `num_params` into the symbolic propagator kernel
  `K_ft(ω)` to get a numeric matrix-valued rational function of ω, then finds its poles
  and residues. **This is the handoff from the saddle to the perturbative expansion.**
- `pipeline/_precompute.py`'s `_solve_mf_at_saddle` re-runs the solver to verify the
  action's linear term vanishes at the saddle and to surface `mf_values`.
- `compute_cumulants` itself surfaces the DAE extras (`mf_all_roots`, `mf_index_used`,
  `mf_stable_roots`, `state_var_order`, `n_seeds_converged`) onto the returned result
  dict so a notebook can introspect every fixed point and its stability.

---

## 2. The math

### 2.1 Why a saddle at all — the MSR-JD background expansion

In the MSR-JD path-integral formulation, a stochastic dynamical system (Langevin /
master-equation / spiking network) is rewritten as a generating functional

```
  Z[J] = ∫ D[φ] D[φ̃]  exp( -S[φ, φ̃] + sources )
```

where `φ` is the physical field (rate, voltage, density) and `φ̃` is the conjugate
*response field*. The action `S` is the thing the whole pipeline manipulates.
Perturbation theory expands `φ = φ* + δφ` around a **saddle** `φ*` chosen so the action
has **no linear term** in the fluctuation `δφ`. That "no linear term" condition is
exactly the classical equation of motion, i.e. the deterministic steady state. The
saddle is the *tree-level* / *mean-field* solution; loop corrections (the diagrams) are
the systematic corrections to it.

For a rate network the saddle is a self-consistency problem. A canonical neural example:

```
  v*_i = E_i + Σ_j  w_ij · g_ij ⋆ n*_j        (voltage = external drive + recurrent input)
  n*_i = φ_i(v*_i)                              (rate = transfer function of voltage)
```

Here `g_ij` is a synaptic kernel that **at the saddle integrates to 1** (kernels are
normalized: `∫ g(t) dt = 1`), so the convolution `g ⋆ n*` of a *constant* `n*_j` is just
`n*_j`. That normalization is why the legacy solver substitutes every kernel symbol → 1
(`kernel_to_one`, `g_to_one`) before evaluating: a stationary mean field sees only the
kernel's total weight, not its shape.

Substituting the voltage equation into the rate equation gives a closed fixed-point
problem in the rates alone:

```
  n*_i = φ_i( E_i + Σ_j w_ij · n*_j ) ≡ T_i(n*)
```

The legacy solver finds `n*` by driving the residual `n*_i − T_i(n*) = 0` to zero. This
is the **"iteration saddle"** (`n*`) vs **"compound saddle"** (`v*`, computed from `n*`)
split that pervades `_mean_field.py`.

### 2.2 The transfer-function derivatives `φ^(k)(v*)`

The diagrammatic vertices are obtained by Taylor-expanding the action about the saddle.
A transfer function `φ` that is nonlinear in the field contributes vertices whose
couplings are the **derivatives of `φ` evaluated at the saddle**:

```
  φ(v* + δv) = φ(v*) + φ'(v*) δv + (1/2) φ''(v*) δv² + (1/6) φ'''(v*) δv³ + ...
```

The pipeline names these `phi0_i = φ(v*_i)` (= the rate `n*_i`), `phi1_i = φ'(v*_i)`,
`phi2_i = φ''(v*_i)`, …, `phik_i = φ^(k)(v*_i)`, up to `ft.taylor_order`. The legacy
solver computes them by symbolic differentiation (Sage `diff`) of the *concrete* φ
expression `model['phi_concrete'](ns, i, v_sym)`, then evaluating at the numeric `v*_i`.
These numbers go into `num_params` so the symbolic vertices become numeric.

`phi0_i` is special: template-built theories keep `φ` as a *formal* function symbol, so
the `(2,0)` Poisson-noise sector carries the formal symbol `phi0_i` rather than `n*_i`.
The `mf_bg_conditions` dict supplies the identity `phi0_i → nstar_i`; the legacy solver
evaluates it at the solved saddle so the noise-source coefficient `−½·phi0_i` becomes
numeric. If it stayed symbolic the *entire correlator collapses to 0* — see
`_mean_field.py:260-279`.

### 2.3 The DAE formulation and `Dt → 0`

The DAE path declares the *dynamics* directly, one equation per field:

```
  (τ_i Dt + 1) v_i = E_i + Σ_j w_ij n_j          (differential — has Dt)
  n_i             = φ_i(v_i)                       (algebraic — no Dt)
```

`Dt` is a formal symbol standing for the time-derivative operator `d/dt`. At the
mean-field saddle everything is **time-stationary**, so `Dt → 0` and the system becomes
purely algebraic:

```
  v_i = E_i + Σ_j w_ij n_j
  n_i = φ_i(v_i)
```

The solver substitutes `Dt = 0` (and, for spatial theories, `Laplacian = 0`, since a
spatially-uniform background has zero Laplacian) into every equation and finds the joint
root `(v*, n*)`. The residual vector is `F_k = LHS_k − RHS_k`; the root is `F(x*) = 0`.

### 2.4 Multiple fixed points and selection

Nonlinear self-consistency equations generically have **several** solutions (bistability,
up/down states, multiple firing-rate fixed points). The DAE solver embraces this:

- **Multi-start Newton**: launch a root finder from `n_starts` random seeds, collect
  every distinct converged point.
- **Dedup**: cluster numerically-equal roots, keep one representative each
  (`_dedup_roots`).
- **Sort**: order roots ascending by the *first declared physical field's first
  population index* — this gives a deterministic, reproducible labeling so
  `fixed_point_index=0` always means "the root with the smallest first-field value."
- **Select**: `fixed_point_index` picks which root becomes the primary saddle.

### 2.5 Linear stability via the generalized eigenvalue problem

Not every fixed point is a valid expansion point: the diagrammatic (loop) expansion is
only well-defined about a **linearly stable** saddle (perturbations must decay, else the
free propagator has a growing mode and the perturbative series diverges). Stability is
read from the linearization of the DAE about the root.

Linearize `F_k(x, Dt) = 0` around `x = x*`, writing `δx ∝ e^{σ t}` so that
`Dt → σ` acting on a perturbation. To first order:

```
  Σ_j [ ∂F_k/∂x_j |_{x*, Dt=0} + σ · ∂²F_k/(∂Dt ∂x_j) |_{x*} ] δx_j = 0
```

Define two real M×M matrices (M = total number of scalar state variables):

```
  B[k, j] = ∂F_k/∂x_j        at  x*, Dt = 0       (the algebraic Jacobian)
  A[k, j] = ∂²F_k/(∂Dt ∂x_j) at  x*               (the "mass" matrix — coefficient of σ)
```

The linearized condition is `(σ A + B) δx = 0`, a **generalized eigenvalue problem**.
Rearranged: `(−B) δx = σ A δx`, i.e. eigenvalues `σ` of the matrix pencil `(−B, A)`.

- **Algebraic equations** (those without `Dt`) contribute **zero rows in A** (no `δ(Dt)`
  dependence). They produce **infinite eigenvalues** in the generalized problem — these
  are the implicit algebraic-constraint modes and get **filtered out**.
- The remaining **finite** eigenvalues are the genuine dynamical modes.
- **Stability convention:** the fixed point is stable iff *every finite eigenvalue* has
  `Re(σ) < 0` (with a small tolerance `EIG_TOL = 1e-9` so roundoff-zero eigenvalues
  aren't misclassified). A growing mode (`Re σ ≥ 0`) ⇒ unstable.

Because `A` has zero rows for algebraic equations, `A` is singular and the generalized
eigenproblem is the *correct* tool (you cannot just invert `A` and take eigenvalues of
`A⁻¹ B`). Note the `∂Dt` derivative is taken *first* (`F_k_dDt = diff(F_k, Dt_sym)`) to
drop the polynomial degree before differentiating wrt `x_j`.

### 2.6 The `stability_analysis` toggle

`model['stability_analysis']` (default `False`) gates the whole eigenvalue machinery:

- **OFF**: skip stability. `fixed_point_index` ranges over *all* converged roots. This is
  the right choice when **every** equation is algebraic — e.g. voltages integrated out,
  no `Dt` anywhere — so `A ≡ 0`, every eigenvalue is infinite, and "linear stability" is
  meaningless (a Hawkes-type rate model is the canonical example).
- **ON**: classify every root, filter to the linearly-stable subset, expose
  `fixed_point_index` over *that* subset, and raise if **no** stable root exists (the
  diagrammatic expansion would be undefined at every saddle the solver could pick).
  Required for bistable / multi-saddle theories where you must pick the stable branch.

### 2.7 How the saddle feeds the perturbative expansion

The handoff is **purely substitutional**. The propagator kernel `K_ft(ω)` and the
vertices are *symbolic* Sage expressions containing parameter symbols and saddle symbols.
The mean-field solver produces `num_params`, a dict `{SR symbol → float}` covering:

- every numeric parameter (`E1`, `w12`, `tau`, …),
- every saddle value (`nstar[i]`, `vstar[i]`, `mstar[i]`),
- every transfer-function derivative at the saddle (`phi1_i`, `phi2_i`, … — legacy path),
- `phi0_i → nstar_i` identities.

`compute_poles_and_residues(prop, num_params)` (`pipeline/_propagator.py:851`)
substitutes `num_params` into `K_ft`, obtaining a *numeric* matrix-valued rational
function of ω. It then finds the retarded poles `ω_k` (those with `Im ω > 0`) and residue
matrices `C_mats`, which assemble the free propagator `G(τ)`. From there the loop
corrections (Phase J / time-domain machinery) build the connected cumulant `C(τ)`. So the
saddle enters the expansion as *the numbers that turn symbolic diagrams into numeric
ones* — there is no further re-solve; the saddle is frozen and the loops are corrections
about it.

The symbolic-derivative renaming that *creates* the `phi1_i`, `phi2_i` symbols lives in
`theory_compiler.py` (`ns._deriv_rename_subs`, around line 1036-1048): the compiler
differentiates each formal function at its saddle expansion point and registers a rename
`<derivative value> → SR.var('phik_<i+1>')`. The legacy solver then supplies the numeric
value of each `phik_i`. The DAE compat wrapper does *not* populate `phi_deriv_vals`
(returns `{}`); the propagator builds those derivatives symbolically in that path.

---

## 3. External tools used

This subsystem touches **four** external libraries. Below, each is explained from
scratch, with the exact import lines and call sites.

### 3.1 SageMath (`sage.all`) — symbolic computer algebra

**What it is.** SageMath is a large open-source mathematics system built on top of
Python; it bundles many computer-algebra and number-theory packages behind a single
Python API. The relevant piece here is its **symbolic ring** `SR` (the "Symbolic Ring"),
which lets you build, manipulate, differentiate, and substitute into mathematical
expressions made of symbols (like `x`, `E1`, `nstar`) rather than numbers. Think of it as
"a calculator that does algebra with letters." Sage runs inside its own Python
interpreter; in this codebase the working environment is the `MSRJD_diagrams` conda env
(per the project memory).

**`SR`** is the symbolic ring object. `SR.var('name')` returns (and globally registers) a
symbolic variable with that name — calling it twice with the same name returns the *same*
symbol, which is why the solver can build a substitution dict keyed by `SR.var('E1')` and
later have it match the `E1` that appears inside a propagator expression.
`SR(expr_or_number)` coerces a Python value or another expression into the symbolic ring.

**`diff(expr, var, k)`** computes the `k`-th symbolic derivative of `expr` with respect to
`var`. The solver uses it both to Taylor-expand the transfer function and to build the
stability Jacobians.

**`.subs(dict)`** substitutes symbols → values (or symbols → other expressions) in an
expression. Chained `.subs(...).subs(...)` applies substitutions left to right. After
substituting all symbols with numbers, `float(expr)` evaluates to a Python float.

How **this code** uses Sage:

In `_mean_field.py` (top-level import):
```python
from sage.all import SR, diff                      # _mean_field.py:10
```
- Build parameter symbols and the saddle substitution dict:
  ```python
  param_subs[SR.var(f'{pname}{i+1}{j+1}')] = val[i][j]   # :69  (matrix param w12 etc.)
  param_subs[SR.var(pname)] = val                        # :87  (scalar param)
  ```
- A dummy variable for the transfer-function expansion and its symbolic derivatives:
  ```python
  v_sym = SR.var('_v_mf_')                               # :90
  phi_expr = model['phi_concrete'](ns, i, v_sym)         # :93
  phi_derivs[i][dk] = diff(phi_expr, v_sym, dk)          # :99
  ```
- Read the symbolic saddle expression, force kernels to 1, bake in params:
  ```python
  vstar_subs_dict = model.get('mf_bg_conditions', lambda ns: {})(ns)   # :105
  g_to_one = {ns.g: SR(1)}                                              # :110
  vstar_baked[i] = SR(vstar_sym[i]).subs(g_to_one).subs(param_subs)    # :111-114
  ```
- Numeric evaluation by full substitution then `float(...)`:
  ```python
  return float(phi_derivs[i][0].subs(param_subs).subs({v_sym: v_val}))  # :102
  ```

In `_mean_field_dae.py`, Sage is imported **lazily, inside `linear_stability`**
(not at module top — the DAE *root-finding* path is pure NumPy/SciPy and never imports
Sage; only stability does):
```python
from sage.all import SR, diff                                   # :676
from sage.all import (tanh as _sage_tanh, sin as _sage_sin,     # :677-681
                      cos as _sage_cos, ... pi as _sage_pi)
```
- `SR.var(f'_mfdae_{var}_{i}')` (`:700`) builds one Sage symbol per scalar state variable,
  plus `Dt_sym = SR.var('_mfdae_Dt')` (`:705`) for the time-derivative.
- The user's equation text is `eval`-ed in a namespace where math names map to the *Sage*
  versions (`sage_math_ns`, `:710-715`) so e.g. `tanh(g_gain[i]*v)` becomes a
  *differentiable* symbolic expression rather than a float.
- `diff(F_k_alg, x_j)` and `diff(F_k, Dt_sym)` (`:802, :808, :810`) build the stability
  Jacobians `A`, `B`; `.subs(root_subs)` then `float(...)` evaluates them at the root.

The **Sage math namespace vs the NumPy math namespace** is the key contrast: the
*residual* path (numeric root-finding) binds `tanh→np.tanh` (`_MATH_NS`, `:33-38`) so the
expression evaluates to a float; the *stability* path binds `tanh→sage.tanh` so the
expression stays symbolic and can be differentiated.

### 3.2 SciPy (`scipy.optimize`, `scipy.linalg`) — numerical solvers

**What it is.** SciPy is the standard scientific-computing library for Python, built on
NumPy. It provides numerical (not symbolic) algorithms: optimization, root-finding, linear
algebra, integration, etc. Nothing in SciPy knows about symbols — it operates on arrays of
floating-point numbers.

**`scipy.optimize.fsolve`** (used by the *legacy* solver) finds a root of a system of
nonlinear equations `f(x) = 0` given a function `f` and a starting guess `x0`. It uses
MINPACK's `hybrd`/`hybrj` (a modified Powell hybrid method). With `full_output=True` it
returns a 4-tuple `(x, infodict, ier, mesg)`; **`ier == 1` signals clean convergence.**
```python
from scipy.optimize import fsolve            # _mean_field.py:11
attempt = fsolve(mf_residual, x0, full_output=True)   # :174
if attempt[2] == 1:    # ier == 1 ⇒ converged                  :178
```

**`scipy.optimize.root`** (used by the *DAE* solver, imported as `_scipy_root`) is the
newer, more general root-finder. With `method='hybr'` it is the same Powell hybrid method
as `fsolve`, but it returns a richer `OptimizeResult` object with attributes `.x` (the
root), `.success` (bool), and more.
```python
from scipy.optimize import root as _scipy_root         # _mean_field_dae.py:27
sol = _scipy_root(R, x0, method='hybr')                # :484
if not sol.success: continue                           # :488
resid = R(sol.x); ...                                  # :492 (re-check residual)
```

**`scipy.linalg.eig`** (used by `linear_stability`) solves eigenvalue problems. Called
with **two** matrix arguments, `eig(C, D)`, it solves the **generalized** eigenproblem
`C v = λ D v` and returns `(eigenvalues, eigenvectors)`. The code calls `eig(-B, A)`, so
the returned eigenvalues are exactly the stability exponents `σ` of `(σA + B)δx = 0`:
```python
import scipy.linalg                                    # :682
sigmas, eigvecs = scipy.linalg.eig(-B_mat, A_mat)      # :817
```
When `A` is singular (algebraic constraints), some returned eigenvalues are `inf`/`nan`;
those are filtered with `np.isfinite`.

### 3.3 NumPy (`numpy`) — numerical arrays

**What it is.** NumPy is the foundational array library: it provides the `ndarray`
n-dimensional numeric array and elementwise math (`np.tanh`, `np.exp`, …) plus reductions
(`np.max`, `np.abs`, `np.all`, `np.isfinite`). It is the substrate SciPy stands on.

How **this code** uses NumPy (`_mean_field_dae.py`):
```python
import numpy as np                                     # :26
```
- `_MATH_NS` (`:33-38`) maps math function names to NumPy versions so user equation text
  evaluates over scalars *and* arrays uniformly (vectorized over a population).
- `_numpy_params` (`:198`) coerces parameter values into `np.asarray(..., dtype=float)`
  so indexing like `w[i, j]` (2-D) and `Em[i]` (1-D) works inside `eval`.
- The residual `R(x)` packs state slices into NumPy arrays (`:271`), evaluates each scalar
  residual, and returns a `np.empty(len(expanded))` vector (`:296`).
- The RNG for seed sampling is `np.random.default_rng(rng_seed)` (`:475`) — the modern
  NumPy random generator, seeded for reproducibility; `rng.uniform(lo, hi, size=...)`
  (`:361`) draws the random initial guesses.
- Root dedup/stability use `np.max`, `np.abs`, `np.all`, `np.isfinite`, `np.real`,
  `np.zeros`, `np.asarray`.

### 3.4 Python `eval` and the `warnings` module (standard library)

Not third-party, but worth flagging because the DAE solver's correctness hinges on them.

**`eval(source, globals_dict)`** executes a Python expression string in a supplied
namespace and returns its value. The DAE solver does *not* parse the user's equation text
into an AST — it `eval`s the (Sage→Python translated) string directly. To make this safe
and predictable it passes a **closed namespace** with `'__builtins__': {}` so no built-in
functions leak in, and binds exactly the names the expression may reference (parameters,
state vars, `phi` callables, index sets, math functions, `Dt`, `Laplacian`, `sum`, `i`).

A subtle and *load-bearing* detail (`_build_residual`, comment at `:273-279`): the
namespace is passed as the **globals** argument, not locals. Python's comprehension
scoping (PEP 3104) runs a generator expression like
`sum(w[i,j]*n[j] for j in E)` in its own function scope that sees the eval-**globals** but
*not* the eval-locals. Passing the namespace as locals would make the genexpr raise
`NameError` on `w`, `n`, `E`.

**`warnings.warn(...)`** (`:23` import; `:596` call) emits a non-fatal warning when
`fixed_point_index` is out of range; the index is then clamped rather than erroring.

---

## 4. Components

Listed in file order. Signature, inputs, outputs, and a step-by-step.

### 4.1 `pipeline/_mean_field.py`

#### `solve_mean_field(ft, model, fundamental, verbose=True)` — `_mean_field.py:14`

**The legacy iteration solver entry point** (single-population path; routes to the hetero
branch when `model['populations']` is set).

- **Takes:** `ft` (compiled FieldTheory; uses `ft._ns` and `ft.taylor_order`), `model`
  (declaration dict), `fundamental` (numeric params by name), `verbose`.
- **Returns:** dict with keys `nstar_vals` (list of floats), `vstar_vals` (list),
  `phi_deriv_vals` (`{(dk, i): float}`), `num_params` (`{SR var → float}`),
  `param_subs` (`{SR var → float}`, raw user params only).

**Step by step:**
1. **Route** (`:40-41`): if `model['populations']` truthy, delegate to
   `_solve_mean_field_hetero` and return.
2. Grab `ns = ft._ns`, `taylor_order = ft.taylor_order` (`:43-44`).
3. **Build `param_subs`** (`:54-87`): for each declared parameter present in
   `fundamental`, expand its value into per-index Sage symbols. The axis sizes come from
   `ns._pop_size` keyed by the parameter's `indexed_by` populations; legacy `indexed`
   params use `len(ns.pop)`. Matrix params → `pname{i+1}{j+1}`, vector → `pname{i+1}`,
   scalar → `pname`. **The user's value shape is the source of truth**; the spec just
   says where the symbols came from.
4. **Symbolic φ derivatives** (`:89-99`): for each pop `i`, evaluate
   `model['phi_concrete'](ns, i, v_sym)` (with `v_sym = SR.var('_v_mf_')`) and take Sage
   derivatives `diff(...)` up to `taylor_order`. `phi_num(i, v)` (`:101`) substitutes
   params and `v_sym→v`, returns a float.
5. **Symbolic `v*`** (`:104-118`): read `model['mf_bg_conditions'](ns)` (the
   `vstar_subs_dict`), pull each `ns.vstar[i]`'s RHS (default `ns.E[i]` if absent), set the
   kernel symbol `ns.g→1` and bake params. `vstar_num(i, nstar_vec)` (`:116`) substitutes
   the current `nstar` and returns a float.
6. **Build the `nstar` iteration target** (`:120-141`): if the model declares an explicit
   RHS for `ns.nstar[0]` in `mf_bg_conditions`, use it (`nstar_rhs_baked`); otherwise fall
   back to the legacy hardcode `n* = phi(v*)` (`nstar_target` at `:133`).
7. **Residual + multi-start `fsolve`** (`:143-186`): `mf_residual(nstar_vec)` returns the
   vector `[n*_i − target_i]`, wrapped so `ValueError`/`ZeroDivisionError` (symbolic
   singularities) become a large finite penalty `1e10` rather than aborting. Tries the
   initial guesses `[0.1, 0.5, 1.0, 0.01]·npop` in turn, keeping the first `ier==1`
   result; raises `RuntimeError` if all fail.
8. **Saddle sanity checks** (`:188-204`): raise if any `n*` is non-finite (fsolve dove into
   a `1/(1−n*)` pole); warn if `ier != 1`.
9. **Evaluate `v*` and φ-derivatives at the fixed point** (`:206-222`): per pop, compute
   `v*_i`, raise if non-finite, and fill `phi_deriv_vals[(dk, i)]` for `dk = 0…taylor_order`.
10. **Verbose self-check** (`:224-232`): print `v*`, `n*`, `φ(v*)` and flag if
    `|φ(v*) − n*| ≥ 1e-10` ("MISMATCH!" — the self-consistency residual).
11. **Assemble `num_params`** (`:234-279`): start from `param_subs`, add `ns.nstar[i]→n*_i`,
    `ns.vstar[i]→v*_i`; add `ns.mstar[i]→ lambda_X·p_part` for GTaS models (guarded; see
    the comment at `:240-244` about a *historical bug* where this assignment was silently
    swallowed); add `phi{dk}_{i+1}→` numeric derivatives; finally resolve any remaining
    `mf_bg_conditions` LHS (notably `phi0_i → nstar_i`) at the solved saddle.
12. **Return** the five-key dict (`:281-287`).

#### `_solve_mean_field_hetero(ft, model, fundamental, verbose=True)` — `_mean_field.py:291`

**Heterogeneous-population legacy solver** (multiple populations, e.g. E and I, each with
its own φ and saddle arrays).

- **Takes/returns:** same shape as `solve_mean_field` plus extra keys `saddle_values`
  (`{saddle_name: [vals]}`) and `saddle_info` (metadata).

**Step by step:**
1. `pop_size = {name: size}` from `model['populations']` (`:310`).
2. **`param_subs`** (`:312-332`): same expansion as the single-pop path but sized by
   `pop_size`.
3. **Identify saddles** (`:334-351`): every parameter with `mean_field=True` is a saddle;
   its `indexed_by[0]` names the population it ranges over. `saddle_info[name]` records
   `pop`, `size`, and `sr_array` (the list of Sage symbols on `ns`).
4. **Read `mf_bg`** (`:353-354`) and **classify iteration vs compound saddles**
   (`:362-378`): prefer the authoritative `model['iteration_saddles']` (computed at build
   time by `_classify_mf_eqs`); fall back to "all are iteration." Iteration saddles are the
   ones fsolve actually iterates; compound saddles (voltages) are evaluated from iteration
   values each step.
5. **`kernel_to_one`** (`:380-401`): map *every* kernel symbol on `ns` (scalar, vector,
   matrix) → `SR(1)` — the "kernels integrate to 1 at the saddle" normalization.
6. **Pre-bake RHS** (`:403-423`): `compound_rhs[name]` and `iter_rhs[name]` are each the
   per-element `mf_bg` RHS with kernels→1 and params baked in.
7. **Flatten helpers** (`:425-442`): `flat_index` maps the flat fsolve vector position ↔
   `(saddle_name, local_index)`; `_unflatten` converts a flat vector to per-saddle arrays.
8. **`_eval_compound`** (`:444-459`): given iteration values, substitute them into the
   compound RHS to get voltage-like saddle values.
9. **`_residual`** (`:461-482`): unflatten → eval compounds → build a full substitution dict
   over *all* saddle symbols → evaluate each iteration closure RHS → return
   `[iter_val − target]`. Wrapped to penalize exceptions with `1e10`.
10. **Multi-start fsolve** (`:484-501`): guesses `[0.1, 0.5, 1.0, 0.01]·total_n`.
11. **Unpack + verify finiteness** (`:503-512`), **assemble `num_params`** by mapping
    every saddle symbol (iteration *and* compound) to its solved float (`:514-523`).
12. **Legacy compat keys** (`:533-543`): concatenate iteration-saddle values into
    `nstar_concat` and compound-saddle values into `vstar_concat` (declaration order).
13. **Return** (`:544-552`) including `saddle_values` and `saddle_info`.

> Note: this hetero path returns `phi_deriv_vals: {}` (empty) — the propagator builds φ
> derivatives symbolically in the multi-pop case.

### 4.2 `pipeline/_mean_field_dae.py`

#### `_sage_to_python(text) -> str` — `_mean_field_dae.py:41`

Translate Sage-syntax to Python by replacing the power operator `^`→`**`. Sage preparses
`^` to exponentiation, but in plain Python `^` is *bitwise XOR* and raises
`TypeError: ufunc 'bitwise_xor' not supported` on a numpy float. Unconditional replacement
is safe (XOR is never meaningful in MF equations over reals). Used everywhere equation/
function text is about to be `eval`-ed.

#### `_pop_size(model, pop_name) -> int` — `_mean_field_dae.py:61`

Return the size of a named population (or `1` for `None` = scalar). Raises `KeyError` if
the name isn't declared. Note: a different helper than the legacy `ns._pop_size`.

#### `_state_variables(model) -> list[(var_name, pop_name, pop_size)]` — `:75`

Return the state variables in *declaration order* from `model['physical_fields']`.
`var_name` is the **user-facing natural name** (`'n'`, `'v'`), preferring
`f['natural_name']` over `f['name']` (the internal MSR name `'dn'`/`'dv'`). **The first
entry is the multi-root sort key.**

#### `_state_slices(state_vars) -> {var_name: (start, end)}` — `:100`

Map each state variable to its `[start, end)` slice into the flat state vector. E.g. with
`v` of size 2 then `n` of size 2: `{'v': (0,2), 'n': (2,4)}`.

#### `class _PhiCallableList` — `:114`

A list of per-population φ callables that ALSO supports a *bare scalar call* `f(v)` when
there's exactly one population position (mirroring the action-side
`_IndexedFormalFunction`). Indexed access `f[i](v)` always works. `__slots__ = ('_callables',)`.
- `__getitem__(i)` → the i-th callable; `__len__`, `__iter__` as expected.
- `__call__(*args)` (`:134`): if exactly one callable, delegate to it; else raise a clear
  `TypeError` demanding explicit `f[i](...)`.

#### `_make_phi_callables(model, params) -> {fn_name: _PhiCallableList}` — `:147`

Build *numeric* φ callables for the residual path. For each declared function with
text-form spec (`args_text`, `expression_text`), single-arg only, translate `^`→`**`, and
for each population index `i` build a closure `phi_i(x_val)` that `eval`s the expression
text in a namespace binding the arg name → `x_val`, `i` → the index, math names → NumPy,
and `'__builtins__': {}`. Multi-arg functions are skipped (the DAE solver only needs
single-arg `phi`). Returns `{name: _PhiCallableList([...])}`.

#### `_numpy_params(model, fundamental) -> dict` — `:198`

Coerce `fundamental` to a dict of NumPy arrays (`np.asarray(..., dtype=float)`) for
indexed params, or a plain `float` for scalars, so `w[i,j]` and `Em[i]` work under `eval`.

#### `_index_sets(model) -> {pop_name: [0..size-1]}` — `:222`

Return the index range per population so `for j in E` inside a comprehension iterates over
the right integers.

#### `_build_residual(model, params_np, phi_callables, index_sets, state_vars, slices)` — `:236`

Build and return `(R, expanded)` where `R(x)` is the residual function and `expanded` is
the per-scalar-residual descriptor list.
- **Pre-expand** (`:254-261`): for each declared equation, expand over its population's
  index range into `(lhs_text, rhs_text, i, pop)` tuples, translating `^`→`**` once.
- **`R(x)`** (`:268-307`): pack the flat state into per-variable NumPy arrays via `slices`,
  build the eval namespace (params, state, phi callables, index sets, NumPy math,
  `Dt=0`, `Laplacian=0`, `sum`, `'__builtins__': {}`), then for each expanded residual set
  `ns['i']=i`, `eval` LHS and RHS, store `float(lhs−rhs)`. Exceptions
  (`ValueError/ZeroDivisionError/OverflowError/FloatingPointError`) → penalty `1e10` to
  push the solver away. Returns a length-`len(expanded)` NumPy array.
- The namespace is passed as eval **globals** (the PEP-3104 comprehension-scoping fix).

#### `_seed_box_default(state_vars, model, fundamental) -> {var: (low, high)}` — `:315`

Per-variable default sampling box. `scale = max(|param value|)` over all params (fallback
`1.0`). For `domain='positive'` → `[0, 5·scale]`; otherwise (`'real'`/unspecified) →
`[−3·scale, 3·scale]`. The default domain when a field doesn't declare one is `'positive'`
(Hawkes-like, rates ≥ 0).

#### `_sample_seeds(state_vars, slices, model, fundamental, n_starts, seed_box, rng)` — `:346`

Generate `n_starts` random initial guesses (a `(n_starts, total)` array). Each variable is
drawn uniformly from its box; `seed_box` overrides defaults per-variable.

#### `_dedup_roots(roots, rtol=1e-6, atol=1e-10) -> list` — `:368`

Cluster roots within tolerance and keep one representative each. Two roots merge if
`max|r − kept| < atol + rtol·scale`, where `scale = max(max|r|, max|kept|, 1.0)`. Discovery
order preserved (sorting happens in the caller).

#### `solve_mean_field_dae(model, fundamental, *, n_starts=64, fixed_point_index=0, seed_box=None, rtol=1e-6, atol=1e-10, verbose=False, rng_seed=0) -> dict` — `:390`

**The DAE multi-root solver entry point.** Step by step:
1. **Guard** (`:443`): raise `ValueError` if no `model['equations']` (caller should use the
   legacy solver).
2. **Build the model description** (`:450-464`): `state_vars`, `slices`, `total` unknowns,
   `params_np`, `phi_callables`, `idx_sets`, and the residual `(R, expanded)`.
3. **Determinacy check** (`:466-473`): raise if `len(expanded) != total` (system under/over
   determined).
4. **Sample seeds** (`:475-477`) with a seeded RNG.
5. **Multi-start Newton** (`:479-501`): for each seed, `_scipy_root(R, x0, method='hybr')`;
   skip on exception or `not sol.success`; **re-evaluate** the residual `R(sol.x)` and
   skip if `max|resid| > 1e-7` (scipy sometimes reports success above its own tolerance);
   skip non-finite roots; collect the rest into `raw_roots`, count `n_converged`.
6. **Dedup + sort** (`:503-508`): `_dedup_roots`, then sort ascending by the first state
   variable's first index (`first_start`).
7. **No-root guard** (`:510-515`): raise if nothing converged.
8. **Build root-dict helper** (`:517-519`): `_build_root_dict(x)` →
   `{'<var>star': [vals]}`.
9. **Stability gate** (`:534`): `stab_on = bool(model.get('stability_analysis', False))`.
   - **ON** (`:537-557`): run `linear_stability` on every root (wrapped in try/except so a
     stability failure marks the root unstable rather than crashing the solve), record
     `values`, `stable`, `eigenvalues_finite`.
   - **OFF** (`:558-567`): record every root with `stable=None` (distinguishing
     "not classified" from "classified unstable") and empty eigenvalues.
10. **Selectable subset** (`:569-591`): ON → filter to stable roots and **raise** if none
    are stable (with a long, actionable message suggesting `.stability_analysis(False)` for
    all-algebraic theories); OFF → all converged roots are selectable.
11. **Index clamp** (`:593-603`): if `fixed_point_index` out of `[0, n_selectable)`, clamp
    and `warnings.warn`.
12. **Select** (`:605`) and **return** (`:607-625`) the rich result dict (see §5).

#### `linear_stability(model, fundamental, root, *, verbose=False) -> dict` — `:631`

**Classify the linear stability of one DAE fixed point** via the generalized eigenvalue
problem (math in §2.5). Step by step:
1. **Lazy Sage import** (`:676-682`) — Sage and `scipy.linalg` only load here.
2. **Guard** (`:684`): raise if no equations.
3. **Build Sage state symbols** (`:693-705`): one `SR.var('_mfdae_<var>_<i>')` per scalar
   state variable (`flat_syms`), the parallel root values (`flat_vals_at_root`),
   `M = len(flat_syms)`, and `Dt_sym = SR.var('_mfdae_Dt')`.
4. **Sage math namespace** (`:710-715`): math names → *Sage* functions so user equations
   stay differentiable.
5. **Build symbolic φ callables** (`:719-742`): like `_make_phi_callables` but returning
   *Sage* expressions; wrapped in `_PhiCallableList`.
6. **Build symbolic residuals** (`:744-789`): wrap single-position state-var symbol lists
   in `_FieldScalar` (so bare-name arithmetic works; imported from
   `pipeline.theory_compiler`), `eval` each equation's LHS/RHS in the Sage namespace with
   `Dt→Dt_sym`, `Laplacian→0`, collect `SR(lhs − rhs)`. Raise if the residual count ≠ M.
7. **Build A and B** (`:791-811`): `root_subs` maps each Sage symbol → its float root value;
   for each residual `F_k`, `B[k,j] = ∂(F_k|Dt=0)/∂x_j |root`, and
   `A[k,j] = ∂²F_k/(∂Dt ∂x_j) |root, Dt=0` (with `∂Dt` taken first).
8. **Generalized eigenproblem** (`:813-832`): `sigmas = scipy.linalg.eig(-B, A)[0]`; filter
   to `finite_mask`; `stable` iff `sigmas_finite.size > 0` AND all `Re(σ) < −EIG_TOL`
   (`EIG_TOL = 1e-9`); `unstable` = the finite σ with `Re ≥ −EIG_TOL`.
9. **Return** (`:840-847`) `stable`, `eigenvalues_finite`, `eigenvalues_all`, `A`, `B`,
   `unstable_eigenvalues`.

#### `solve_mean_field_dae_compat(ft, model, fundamental, *, fixed_point_index=0, n_starts=64, seed_box=None, verbose=True, rtol=1e-6, atol=1e-10) -> dict` — `:853`

**Drop-in replacement for the legacy `solve_mean_field`** when `model['equations']` is
populated; this is what `compute.py` actually calls. Step by step:
1. **Run** `solve_mean_field_dae` (`:886-893`) with `verbose=False`; pull `mf_values`.
2. **Verbose root report** (`:897-904`): print the number of fixed points and mark the
   selected one.
3. **Build `param_subs` in the legacy Sage-symbol convention** (`:906-941`): expand each
   parameter into `SR.var('w12')` etc., sizing matrix/vector axes from
   `ft._ns._pop_size` (falling back to `model['populations']`); handle un-indexed vector
   values too.
4. **Bake saddle values into `num_params`** (`:943-950`): for each `<var>star` saddle, look
   up the Sage array `getattr(ft._ns, saddle_name)` and map `array[i] → float(val)`.
5. **Legacy concat vectors** (`:952-954`): `nstar_concat = mf_values.get('nstar', [])`,
   `vstar_concat = mf_values.get('vstar', [])`.
6. **Return** (`:956-976`) the legacy-shape keys (`nstar_vals`, `vstar_vals`,
   `phi_deriv_vals={}`, `num_params`, `param_subs`, `saddle_values`) PLUS the DAE extras
   (`mf_values`, `mf_all_roots`, `mf_stable_roots`, `mf_unstable_roots`, `mf_index_used`,
   `state_var_order`, `n_seeds_converged`).

### 4.3 Supporting class consumed here

#### `class _FieldScalar` — `pipeline/theory_compiler.py:268`

A thin wrapper around a list of SR vars that supports BOTH bare scalar arithmetic (when
`len == 1`) AND `[i]` indexed access. `linear_stability` wraps single-position state-var
symbol lists in it so a user can write `xt * x` (size-1, scalar) and `xt[0] * x[0]`
(indexed) interchangeably. `_scalar()` raises a `TypeError` if `len != 1`. Implements all
arithmetic dunders plus `_sage_()` (so Sage can coerce it).

---

## 5. Data structures

### 5.1 The `model` dict (input)

Keys this subsystem reads (built by `TheoryBuilder.build()`):

| Key | Type | Meaning |
|---|---|---|
| `parameters` | list of pspec dicts | each `{'name', 'indexed_by'?, 'indexed'?, 'mean_field'?, 'default'?, ...}` |
| `kernels` | list of kspec dicts | `{'name', ...}` — synaptic/temporal kernels (→1 at saddle) |
| `populations` | list of `{'name','size'}` | present ⇒ heterogeneous path |
| `physical_fields` | list of fspec dicts | `{'name','natural_name','population','domain'?, ...}` |
| `functions` | list of fn dicts | `{'name','args_text','expression_text','population', ...}` (transfer fns) |
| `phi_concrete` | callable `(ns, i, v) -> SR` | legacy: concrete φ expression |
| `mf_bg_conditions` | callable `(ns) -> {SR var: SR expr}` | legacy: saddle equations (LHS var → RHS) |
| `iteration_saddles` | list[str] | hetero: authoritative iteration-vs-compound split |
| `equations` | list of eq dicts | DAE: `{'lhs_text','rhs_text','population','kind'}` (`kind` ∈ {`'differential'`,`'algebraic'`}) |
| `stability_analysis` | bool | DAE: gate the eigenvalue classification |

### 5.2 `fundamental` (input)

`{param_name: value}` where `value` is a Python scalar, a list (vector), or a list-of-lists
(matrix). Keyed by the *parameter name*, not by the indexed Sage symbol.

### 5.3 `ft._ns` (the Sage namespace)

Carries the Sage symbol objects: saddle arrays `ns.nstar`, `ns.vstar`, `ns.mstar` (each a
list of SR vars indexed by population), parameter arrays (e.g. `ns.E`), kernel symbols
(scalar `ns.g`, or vector/matrix lists), the population index list `ns.pop`, and the
`ns._pop_size` map `{pop_name: size}`. The DAE compat wrapper accesses saddle arrays via
`getattr(ft._ns, saddle_name)`.

### 5.4 Legacy solver return dict

```python
{
  'nstar_vals':     [float, ...],            # rate saddles (per pop / concatenated)
  'vstar_vals':     [float, ...],            # voltage saddles
  'phi_deriv_vals': {(dk, i): float},        # φ^(dk)(v*_i); {} on the hetero path
  'num_params':     {SR var: float},         # THE handoff dict
  'param_subs':     {SR var: float},         # raw user params only
  # hetero path also adds:
  'saddle_values':  {saddle_name: [float]},
  'saddle_info':    {saddle_name: {'pop','size','sr_array'}},
}
```

### 5.5 DAE solver return dict (`solve_mean_field_dae`)

```python
{
  'mf_values':         {'<var>star': [floats]},      # the selected root
  'mf_all_roots':      [ {'values': {...},           # every root, sorted, annotated
                          'stable': bool|None,
                          'eigenvalues_finite': np.ndarray[complex]}, ... ],
  'mf_stable_roots':   [ {'<var>star':[...]}, ... ], # selectable subset (= all if stab OFF)
  'mf_unstable_roots': [ {...}, ... ],               # only meaningful when stab ON
  'mf_index_used':     int,                          # index after clamping
  'state_var_order':   [var_name, ...],              # declaration order (sort-key origin)
  'n_seeds_converged': int,                          # pre-dedup convergence count
  'stability_analysis': bool,                        # was classification run?
}
```

### 5.6 `linear_stability` return dict

```python
{
  'stable':               bool,
  'eigenvalues_finite':   np.ndarray[complex],   # dynamical modes σ
  'eigenvalues_all':      np.ndarray[complex],   # includes ±inf (algebraic constraints)
  'A':                    np.ndarray[float] (M×M),  # ∂²F/∂Dt∂x  (mass/pencil matrix)
  'B':                    np.ndarray[float] (M×M),  # ∂F/∂x at Dt=0 (algebraic Jacobian)
  'unstable_eigenvalues': [complex, ...],        # finite σ with Re ≥ −EIG_TOL
}
```

### 5.7 `expanded` (internal, `_build_residual`)

`list[(lhs_text:str, rhs_text:str, i:int, pop:str|None)]` — one entry per scalar residual
after population expansion. Its length must equal the number of unknowns.

### 5.8 `num_params` (the output that matters)

`{SR var → float}`. The union of (raw params expanded to indexed symbols) ∪ (saddle values
`nstar[i]`/`vstar[i]`/`mstar[i]`) ∪ (φ-derivative symbols `phi1_i`, `phi2_i`, …, legacy
only) ∪ (`phi0_i → nstar_i` identities). Consumed by `compute_poles_and_residues`.

---

## 6. Data flow

**Inbound** (`compute.py:364-379`):
```python
if model.get('equations'):
    mf = solve_mean_field_dae_compat(ft, model, fundamental,
            fixed_point_index=..., n_starts=mf_dae_n_starts,
            seed_box=mf_dae_seed_box, verbose=verbose)
else:
    mf = solve_mean_field(ft, model, fundamental, verbose=verbose)
num_params = mf['num_params']
```

**Inside (DAE path):**
`fundamental {name: value}` → `_numpy_params` → `params_np {name: ndarray|float}`;
`model['functions']` → `_make_phi_callables` → `{fn: _PhiCallableList}`;
`model['equations']` + `model['physical_fields']` → `_build_residual` →
`R(x), expanded`; random seeds → `_scipy_root` per seed → `raw_roots` →
`_dedup_roots` → sort → `[per-root value dict]`; optional `linear_stability` per root.

**Inside (legacy path):**
`fundamental` → `param_subs`; `model['phi_concrete']` → symbolic `phi_derivs`;
`model['mf_bg_conditions']` → `vstar_baked`, `nstar_rhs_baked`; `fsolve(mf_residual)` →
`nstar_vals`; evaluate → `vstar_vals`, `phi_deriv_vals`; assemble → `num_params`.

**Outbound** (`compute.py:632`):
```python
compute_poles_and_residues(prop, num_params, verbose=verbose)
```
which substitutes `num_params` into the symbolic kernel `prop['K_ft'](ω)` and fills
`prop['pole_vals']` and `prop['C_mats']` — the numeric free propagator. The DAE extras
(`mf_all_roots`, `mf_index_used`, …) are surfaced onto the final `compute_cumulants`
result dict for introspection.

**Concrete example shapes.** For a 1-pop voltage+rate model with `physical_fields = [v, n]`
both scalar: `state_vars = [('v',None,1), ('n',None,1)]`, `slices = {'v':(0,1),'n':(1,2)}`,
`total = 2`. A converged root `x = [v*, n*]` becomes
`mf_values = {'vstar':[v*], 'nstar':[n*]}`. `num_params` then carries
`{E: ..., w: ..., ns.vstar[0]: v*, ns.nstar[0]: n*}`. For a 2-pop E/I rate model, the
saddle arrays each have length 2 and `nstar_concat`/`vstar_concat` are length-2 vectors.

---

## 7. Gotchas & caveats

- **Two solvers, one routing decision.** The choice is *purely* `model.get('equations')`
  truthiness (`compute.py:364`, `_precompute.py:204`). A theory that declares `.equation`
  *and* `mf_bg_conditions` will silently take the DAE path. The modules are
  "INTENTIONALLY decoupled" (`_mean_field_dae.py:16`).
- **`fsolve` initial-guess gauntlet.** The legacy solver tries `[0.1, 0.5, 1.0, 0.01]`
  in order and keeps the first `ier==1`. Theories with a saddle far outside these basins
  may converge to the *wrong* fixed point or fail. There is no multi-root machinery on the
  legacy path — it returns whatever the first converging guess lands on.
- **Symbolic singularities are penalized, not raised.** Both solvers wrap residuals so
  `ValueError/ZeroDivisionError` (e.g. spike-reset `1/(1−n*)` evaluated at `n*≈1`) become a
  `1e10` penalty, letting the solver back away. But if it *converges* into a near-pole,
  explicit finiteness checks (`math.isfinite`) raise a `RuntimeError` with guidance to
  reduce excitation or supply a custom guess (`_mean_field.py:193-217`).
- **`mstar` historical bug, now guarded.** The comment at `_mean_field.py:240-244`
  documents a prior bug: an `SR(model['mf_equations'](ns))` probe raised on the equation
  *list* inside a `try`, silently swallowing the `mstar` assignment so the GTaS rate `b_X`
  showed as missing. The current code guards on `lambda_X`/`p_part` presence.
- **`phi0_i` must be resolved or the correlator collapses to 0.** Template-built theories
  keep φ formal; the noise coefficient `−½·phi0_i` stays symbolic unless `phi0_i→nstar_i`
  is baked in (`_mean_field.py:260-279`). Easy to miss in new templates.
- **DAE: `eval` namespace must be passed as GLOBALS not locals.** The PEP-3104
  comprehension-scoping subtlety (`_build_residual` comment `:273-279`): genexprs see eval
  globals, not eval locals. Getting this wrong makes `sum(... for j in E)` raise
  `NameError`.
- **DAE residual eval has empty `__builtins__`.** `'__builtins__': {}` is intentional
  sandboxing, but it means *only* the explicitly-bound names (params, state, phi, index
  sets, `_MATH_NS`, `Dt`, `Laplacian`, `sum`) are available. A user equation referencing
  any other Python builtin (e.g. `min`, `max`, `len`) will raise — and that exception is
  swallowed into a `1e10` penalty, which can masquerade as non-convergence.
- **`Laplacian → 0` and `Dt → 0` at the saddle.** Both are hard-substituted to 0 on the
  homogeneous mean field. Stability to *spatial* (k≠0) perturbations is NOT handled here —
  it is the propagator's k-dependence (pattern formation), see the comment at
  `_mean_field_dae.py:769-773`.
- **scipy "success" is not trusted.** The DAE solver re-evaluates the residual and rejects
  any root with `max|resid| > 1e-7` even when `sol.success` is `True` (`:492-494`).
- **`stability_analysis` must be OFF for all-algebraic theories.** With no `Dt` anywhere,
  `A ≡ 0`, every eigenvalue is infinite, and turning stability ON makes the solver raise
  "NONE were classified linearly stable" (`:572-586`). The error message itself tells the
  user to call `.stability_analysis(False)`.
- **`fixed_point_index` clamps with a warning, never errors.** Out-of-range indices are
  clamped to the valid range (`:594-603`); a silent-ish `warnings.warn` is the only signal.
- **Sort key fragility.** Roots are ordered by the first declared physical field's first
  index. Reordering field declarations changes which root `fixed_point_index=0` selects.
- **`phi_deriv_vals` is `{}` on both the hetero legacy path and the entire DAE compat
  path.** Downstream must build φ derivatives symbolically in those cases. If a consumer
  ever needs explicit saddle derivatives via the DAE path, the wrapper docstring
  (`:878-881`) flags that it'll need to be populated.
- **Multi-arg functions silently skipped.** `_make_phi_callables` only handles single-arg
  functions (`len(args_text) != 1` → `continue`, `:174-175`); multi-arg CGF-style functions
  are not built as callables here.
- **`linear_stability` failures are swallowed per-root.** In `solve_mean_field_dae` the
  per-root `linear_stability` call is wrapped in `except Exception` (`:542-552`) and marks
  the root *unstable* with an `'error'` annotation rather than aborting — a stability bug
  can therefore make a genuinely stable root vanish from the selectable set.

---

## 8. Glossary

- **MSR-JD** — Martin–Siggia–Rose–Janssen–De Dominicis: the field-theoretic
  (path-integral) formulation of classical stochastic dynamics, in which a physical field
  `φ` is paired with a conjugate *response field* `φ̃`.
- **Mean field / saddle point** — the deterministic background `φ*` about which the loop
  expansion is performed; it is the value that makes the action's linear term vanish, i.e.
  the classical steady state.
- **Self-consistency equation** — the fixed-point condition the saddle must satisfy
  (e.g. `n* = φ(E + w·n*)`).
- **Transfer function (`φ`)** — the (possibly nonlinear) map from input (voltage/current)
  to output (rate); its derivatives at the saddle are the diagram couplings.
- **`φ^(k)` / `phik_i`** — the k-th derivative of the transfer function at the saddle of
  population `i`; `phi0_i = φ(v*_i) = n*_i`.
- **DAE (differential-algebraic equation)** — a system mixing differential equations
  (carrying `Dt`) with algebraic constraints (no `Dt`).
- **`Dt`** — the formal time-derivative operator symbol; substituted `→ 0` at the
  stationary saddle, and `→ σ` (the eigenvalue) in linear stability.
- **`Laplacian`** — the formal spatial second-derivative operator; `→ 0` on a
  spatially-uniform background.
- **Iteration saddle vs compound saddle** — (legacy hetero) the saddle variables fsolve
  actually iterates (rates `n*`) vs those evaluated from them each step (voltages `v*`).
- **Multi-start Newton** — running a Newton-type root finder from many random initial
  guesses to discover multiple roots of a nonlinear system.
- **Generalized eigenvalue problem** — `C v = λ D v` for matrices `C, D`; reduces to the
  ordinary eigenproblem when `D = I`. Used for stability because the "mass" matrix `A` is
  singular (algebraic constraints give zero rows).
- **Linear stability** — a fixed point is linearly stable iff every (finite) eigenvalue of
  the linearized dynamics has negative real part (perturbations decay).
- **`num_params`** — the substitution dictionary `{SR symbol → float}` that converts the
  symbolic propagator/vertices into numeric ones; the subsystem's deliverable.
- **`fundamental`** — the user's raw numeric parameter values keyed by parameter name.
- **Kernel normalization** — synaptic/temporal kernels integrate to 1, so at the
  stationary saddle every kernel symbol is replaced by 1.
- **SageMath / `SR`** — the symbolic-algebra system and its Symbolic Ring; `SR.var`,
  `diff`, `.subs`, `float`.
- **NumPy / SciPy** — numeric array library and scientific-computing library;
  `np.random.default_rng`, `scipy.optimize.root`/`fsolve`, `scipy.linalg.eig`.
- **`eval`** — Python's expression evaluator; the DAE solver `eval`s equation text in a
  sandboxed namespace.
- **`ier`** — `fsolve`'s integer convergence flag; `ier == 1` means clean convergence.
- **`_PhiCallableList`** — a list of per-population φ callables that also supports bare
  scalar call `f(v)` when there's exactly one population.
- **`_FieldScalar`** — a size-1 wrapper that supports both bare arithmetic and `[i]` access,
  letting users write `xt*x` and `xt[0]*x[0]` interchangeably.
- **GTaS** — a model family (feedforward + extra couplings) referenced in the legacy
  `mstar` handling; its mean-field rate `m*_i = λ_X · p_part`.

---

## 9. Proposed manual subsections

1. **The mean-field background in MSR-JD** — why the loop expansion needs a saddle; the
   "vanishing linear term" condition.
2. **Self-consistency vs DAE: two ways to declare a steady state** — and how the framework
   routes between the two solvers.
3. **The legacy iteration solver** — `phi_concrete`, `mf_bg_conditions`, kernel→1, the
   iteration/compound split, multi-start `fsolve`.
4. **Transfer-function derivatives at the saddle** — `phi0_i … phik_i`, how they become
   vertices, and the `phi0_i→nstar_i` rescue.
5. **The DAE formulation** — declaring `.equation(lhs, rhs, population)`, `Dt→0`,
   `Laplacian→0`, residual evaluation via sandboxed `eval`.
6. **Finding all fixed points** — multi-start Newton, seed boxes & domains, dedup, the
   deterministic sort, `fixed_point_index` selection.
7. **Linear stability** — linearizing the DAE, the `(A,B)` pencil, the generalized
   eigenvalue problem, filtering infinite eigenvalues, the stability convention.
8. **The `stability_analysis` toggle** — when to turn it on (bistable theories) vs off
   (all-algebraic theories), and the failure modes of each choice.
9. **From saddle to diagrams** — `num_params` and the handoff to
   `compute_poles_and_residues`; what is frozen and what is corrected.
10. **External tooling primer** — Sage `SR`/`diff`/`subs`, SciPy `root`/`fsolve`/`eig`,
    NumPy arrays/RNG, and the `eval`-namespace conventions.
11. **Pitfalls & diagnostics** — singularities, sort-key fragility, swallowed exceptions,
    the empty-`phi_deriv_vals` paths, reading `mf_all_roots`/eigenvalues for debugging.

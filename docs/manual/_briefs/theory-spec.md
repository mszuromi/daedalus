# Theory Specification: TheoryBuilder authoring

*Subsystem slug: `theory-spec`*

Primary source files:

- `pipeline/theory.py` (2127 lines) — the builder classes and `build()`.
- `pipeline/theory_compiler.py` (1945 lines) — text-expression → Sage-lambda compilation.
- `pipeline/theory_templates.py` (489 lines) — pre-canned action / kernel / noise templates.

Supporting file read for grounding (not the primary subject):

- `pipeline/spatial_operator_ir.py` — the `Lap()`/`Dt()`/`Dx()` operator IR that the
  builder's `operator_ir` mode lowers into.
- `pipeline/compute.py` — `compute_cumulants()`, the consumer of the model dict that
  `build()` produces.

---

## Overview

### What this subsystem is for, in plain language

The MSR-JD (Martin–Siggia–Rose–Janssen–De Dominicis) diagrammatic pipeline in this
repository computes connected cumulants (correlation/response functions) of a stochastic
field theory. To do that, the rest of the pipeline needs a **model dict** — a large Python
dictionary whose values include not just numbers and metadata but several *Sage-aware
lambda functions* (`action`, `mf_bg_conditions`, `kernel_ft_image`, `phi_concrete`,
`mf_equations`, `specializations`, …). Each lambda takes a runtime "namespace" object `ns`
that carries the symbolic field/parameter variables and returns a SageMath symbolic
expression (an `SR` object). Writing one of those dicts by hand requires deep familiarity
with both SageMath's symbolic algebra and this framework's private naming conventions.

**This subsystem is the human-friendly front door.** Instead of hand-writing the dict, a
user describes the theory with a *fluent builder API* — a chain of method calls such as
`.physical_field('x')`, `.parameter('mu', default=1.0)`, `.set_action_text('...')`,
`.equation(lhs=..., rhs=...)` — and finally calls `.build()`. The builder accumulates the
declarations, then compiles the human-typed text strings (the action, the mean-field
equations, the kernels, the noise cumulants) into the Sage lambdas the pipeline expects,
and assembles the model dict automatically.

There are three concrete builder classes (`pipeline/theory.py:2087`, `:2102`, `:2121`):

- **`TemporalTheoryBuilder`** — time-only theories (`x(t)`, no spatial extent). Carries the
  temporal-only methods (populations, kernels, colored-noise / `markovianize`, action
  templates). `build()` asserts every field is non-spatial.
- **`SpatialTheoryBuilder`** — spatial PDE / field theories (`φ(x, t)`). Carries the
  spatial-only methods (`spatial_dim`, `boundary`, `initial`, `dyson`,
  `reference_diffusion`). `build()` asserts at least one field is spatial.
- **`TheoryBuilder`** — a back-compatibility class that mixes in BOTH method sets and
  auto-detects spatial-vs-temporal at `build()` time. New code should prefer the two
  specific classes.

### Where it sits in the end-to-end pipeline

```
  ┌──────────────────┐    text strings    ┌──────────────────────┐
  │  user / UI / a    │ ─────────────────▶ │  TheoryBuilder        │
  │  *.theory.py file │  (action, eqs,     │  (pipeline/theory.py) │
  │  build() function │   kernels, noise)  │                       │
  └──────────────────┘                    └──────────┬───────────┘
                                                      │ .build()
                                  compiles text → lambdas via
                                  pipeline.theory_compiler
                                                      │
                                                      ▼
                                          ┌───────────────────────┐
                                          │  model dict           │
                                          │  (a.k.a. HAWKES_MODEL) │
                                          └──────────┬────────────┘
                                                      │
                                                      ▼
                              pipeline.compute.compute_cumulants(model, k, max_ell, …)
                              → FieldTheory expansion → diagram enumeration → Phase J
```

- **What feeds it:** A `theories/<name>.theory.py` file with a `build()` function (e.g.
  `theories/ou_quartic.theory.py`, `theories/kpz_1d.theory.py`), or a notebook / UI session
  that calls the builder methods directly. The notebook-unification engine (`nb_support`)
  loads these `.theory.py` files.
- **What consumes its output:** `pipeline.compute.compute_cumulants(model, …)`
  (`pipeline/compute.py:108`). That function reads the model dict's lambdas and metadata,
  runs the symbolic FieldTheory expansion, enumerates Feynman diagrams, and integrates them.
  The spatial blocks of the dict (`model['spatial']`, `model['boundary']`,
  `model['initial']`) are consumed by the spatial propagator builder; the DAE
  (`model['equations']`) is consumed by the multi-root mean-field solver
  (`pipeline._mean_field_dae.solve_mean_field_dae`).

So this subsystem is purely **input specification + compilation**: it never integrates a
diagram. Its single output is the model dict.

---

## The math

### MSR-JD response field theory in one page

A stochastic equation of motion (EOM) for a field `φ` (which may be a scalar `x(t)` or a
continuous field `φ(x, t)`) driven by noise `η`,

```
  ∂_t φ  =  F[φ]  +  η,        ⟨η(t) η(t')⟩ = 2D δ(t − t'),
```

is recast as a path integral by introducing a **response field** `φ̃` (the "tilde"
field). The probability of a noise realization, integrated against the dynamics enforced by
a delta functional, becomes

```
  Z = ∫ Dφ Dφ̃  exp(− S[φ, φ̃]),
```

with the MSR-JD **action**

```
  S[φ, φ̃]  =  ∫ dt [ φ̃ (∂_t φ − F[φ])  −  D φ̃² ].
```

The `φ̃ · ∂_t φ` term is the kinetic/causal structure, `φ̃ · F[φ]` the drift, and the
`−D φ̃²` term encodes the (Gaussian, white) noise as a *second cumulant* on the response
field. Non-Gaussian noise of `n`-th cumulant `κ⁽ⁿ⁾` adds a response-field monomial of
degree `n`: a term `− (κ⁽ⁿ⁾/n!) φ̃ⁿ`. This is exactly why `set_action_text`'s docstring
notes that "non-Gaussian white noise is a response-field monomial of degree ≥ 3"
(`pipeline/theory.py:1040`).

In this code the user types the **per-field integrand**, e.g. the OU-quartic action

```
  S = ∫ dt [ x̃ ((∂_t + μ) x + ε x³)  −  D x̃² ],
```

which in builder text (`theories/ou_quartic.theory.py`) is

```python
.set_action_text('sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)')
```

Here `xt` is `x̃`, `Dt` is the inert symbol for `∂_t`, and the `sum(... for i in pop)`
is the per-population sum (size 1 for a scalar theory).

### Saddle-fluctuation split

The cumulants are computed perturbatively around the deterministic steady state, the
**mean-field saddle** `φ̄` (written `xstar`, `vstar`, `nstar`, … internally). Every physical
field is split

```
  φ  =  φ̄  +  δφ,        δφ = "fluctuation field" (dx, dv, dn, …).
```

The action is Taylor-expanded in the fluctuations `δφ` around `φ̄`. The constant term, the
linear (tadpole) term, the quadratic (propagator) term, and the cubic-and-higher
(interaction-vertex) terms emerge. The **saddle equation** is the condition that the linear
(tadpole) term vanishes — physically, that `φ̄` is the deterministic fixed point. For the OU
quartic that is

```
  (∂_t + μ) x̄  =  − ε x̄³      ⟹  at steady state  μ x̄ = − ε x̄³,
```

declared in the builder as `.equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', …)`.

### Why the transfer function gets the formal-symbol treatment

For Hawkes-style neural models, the drift contains a nonlinear **transfer function**
`φ(v)` (the firing-rate nonlinearity — note the unfortunate name clash with the field
`φ`; in this codebase the transfer function is the user-declared function literally named
`phi`). Its Taylor coefficients at the saddle, `φ(v̄), φ'(v̄), φ''(v̄), …`, become the
**vertex couplings** of the diagrammatic expansion. The framework names them
`phi0_<i+1>`, `phi1_<i+1>`, `phi2_<i+1>`, … (the suffix `_<i+1>` is the 1-based population
index). The action is authored with `phi` as a *formal* Sage function so SageMath's
`taylor()` produces those formal-derivative symbols, and `specializations` /
`mf_bg_conditions` later substitute the concrete numeric values
(`pipeline/theory_templates.py:319`–`345` explains exactly why the concrete value must NOT
be inlined: doing so short-circuits the auto-Taylor pass and leaves an uncancelled `(1,0)`
tadpole in the bigrade analysis).

### The kernel Fourier image

A convolution kernel `g(t)` (e.g. an exponential synapse `g(t) = (1/τ_g) e^{−t/τ_g} Θ(t)`)
enters the action as a temporal convolution. Diagrammatic propagators live in frequency
space, so the pipeline needs the kernel's Fourier image `ĝ(ω)`. For the exponential synapse,

```
  ĝ(ω)  =  1 / (1 + i ω τ_g).
```

The builder lets the user supply either the time-domain `g(t)` (`time_expr=`) — which is
symbolically Fourier-transformed via `msrjd.core.field_theory.fourier_transform` — or the
frequency image directly (`freq_image=`, the fast path). See
`make_kernel_ft_image_lambda` (`pipeline/theory_compiler.py:1493`).

### Spatial operators and momentum form factors

For a spatial field `φ(x, t)` the action contains spatial differential operators: the
Laplacian `∇²` (diffusion `D ∇² φ`), spatial gradients `∂_{x_i} φ`, and `∂_t φ`. Their
Fourier images on a leg of wavevector `k` and frequency `ω` are

```
  ∇²  →  − k²,      ∂_t  →  − i ω,      ∂_{x_i}  →  i k_i,
```

and compositions multiply (e.g. `∇⁴ → k⁴`). These are the **form factors** the spatial
integrator attaches to each leg. The builder exposes two authoring styles:

- **Plain (v1):** `Laplacian` is a bare inert symbol used multiplicatively, exactly like
  `Dt`. Example KPZ-ish reaction-diffusion: `pt*(Dt(p) + mu*p - DD*Lap(p) + g*p^2) - T*pt^2`.
- **Operator IR (v2):** `Lap(φ)`, `Dt(φ)`, `Dx(φ, i)` are *argument-binding calls* (turn on
  with `.operator_ir()`). This is required for **derivative-inside-nonlinearity** vertices
  like KPZ's `(∂_x h)²` or Model-B's `∇²(δφ²)` / Burgers' `∂_x(δφ²)`, where a bare
  multiplicative symbol could not say *which leg* the derivative acts on. The KPZ
  distinction `(∂φ)² ≠ ∂(φ²)` is precisely what the IR captures (see
  `theories/kpz_1d.theory.py`).

The math of the IR lowering (linearity, kill-mean, derived-generator substitution
`u = δφ, v = ∇²δφ`, classification into bilinear-vs-vertex) lives in
`pipeline/spatial_operator_ir.py` and is driven from the compiler's
`_lower_operator_ir_action` (`pipeline/theory_compiler.py:720`).

### Mean-field as a DAE (differential-algebraic equation) system

Newer theories declare the EOM as a system of residuals via `.equation(lhs=..., rhs=...)`.
At the mean-field point the time derivative is set to zero (`Dt → 0`) and the resulting
algebraic system is solved by **multi-start Newton** (random seeds, then sort and pick the
`fixed_point_index`-th root). For *bistable / differential* theories one can opt into
**linear stability**: linearize the residuals keeping `Dt → −iω`, assemble the generalized
eigenvalue problem `(B − iω A) δx = 0`, and restrict the allowed roots to the linearly
stable subset (`.stability_analysis(True)`; see `pipeline/theory.py:1087`–`1213`).

### Colored noise → Markovian embedding

White noise (`⟨η η'⟩ ∝ δ(τ)`) keeps loop integrals cheap; **colored** noise
(`⟨η η'⟩ ∝ c·e^{−|τ|/τ_c}`, a single Lorentzian in frequency) does not — the loop
integrals would hang in `scipy.nquad` at loop order ≥ 1. The standard trick is to introduce
an auxiliary Ornstein–Uhlenbeck field driven by *white* noise whose stationary
autocorrelation is exactly that exponential, then couple the original field to it linearly.
The builder's `.markovianize(True)` (default on) walks the declared CGF (cumulant generating
functional) terms and rewrites every row whose kernel matches the `c·exp(−|τ|/τc)` template
into a white-noise auxiliary field + linear filter, before compiling. The rewrite itself
lives in `pipeline/colored_to_markovian.py`; the builder just toggles it.

---

## External tools used

This subsystem touches exactly one heavyweight external library — **SageMath** — plus the
Python standard library. It does NOT touch nauty, numba, scipy, or networkx directly
(those are used by *downstream* consumers of the model dict, not by the authoring layer).

### SageMath (imported as `sage.all` / `sage.*`)

**What it is, from scratch.** SageMath is a large open-source mathematics system built on
top of Python. It bundles a *computer algebra system* (CAS): the ability to hold and
manipulate **symbolic** mathematical expressions (variables, functions, polynomials,
integrals) rather than just floating-point numbers. The central object is the **Symbolic
Ring**, abbreviated **`SR`**: an algebraic structure whose elements are symbolic
expressions. `SR.var('mu')` creates a symbolic variable `mu`; arithmetic on `SR` objects
builds expression trees (`SR.var('a')*SR.var('x')**2` is the symbolic `a·x²`, not a number).
Sage can differentiate, Taylor-expand, Fourier-transform, simplify, and substitute on these
trees.

A practical wrinkle: Sage scripts are normally written in a slightly extended Python dialect
that a **preparser** rewrites into real Python before execution. The most important rewrites
are `^` → `**` (so `x^2` means "x squared", the mathematician's convention, not Python's
bitwise-XOR) and lifting integer literals like `2` into Sage `Integer` objects. This
subsystem deliberately invokes that preparser by hand (see `_safe_eval` below) so user-typed
action text can use `x^2`.

**Exactly how this code uses Sage.**

1. *Importing the symbolic ring and helpers.* (`pipeline/theory.py:68`)

   ```python
   from sage.all import SR
   ```

   and the much larger import in the compiler (`pipeline/theory_compiler.py:40`):

   ```python
   from sage.all import (
       SR, sage_eval, exp, log, sin, cos, tan, sqrt,
       heaviside, dirac_delta, I, pi, function as sr_function, diff,
   )
   ```

   - `SR` — the symbolic ring. `SR.var('name')` makes a symbolic variable;
     `SR.var('name', domain='positive')` attaches a positivity assumption that Sage's
     simplifier and Fourier transform can use. `SR(0)` / `SR(1)` are the symbolic zero/one.
     `SR.wild()` makes a *wildcard* used in pattern-substitution (`pipeline/theory_compiler.py:1389`).
   - `sage_eval` — Sage's "evaluate a string as a Sage expression in a given namespace"
     function. Used by `_IndexableCallable.__call__` (`pipeline/theory_compiler.py:652`).
   - `exp, log, sin, cos, tan, sqrt, heaviside, dirac_delta` — Sage's symbolic special
     functions. `heaviside(t)` is the step function Θ(t); `dirac_delta(τ)` is the Dirac
     δ. They are exposed to user expressions in `_builtin_namespace`
     (`pipeline/theory_compiler.py:359`).
   - `I` — the symbolic imaginary unit `i`. `pi` — symbolic π.
   - `function as sr_function` — Sage's **formal/undefined function** constructor.
     `function('phi_1')(v)` is the *unapplied* symbolic function `phi_1` evaluated at `v`;
     Sage keeps it as an opaque node that `taylor()` can differentiate formally. This is the
     machinery behind the transfer-function vertex symbols.
   - `diff` — symbolic differentiation. Used in `_augment_saddle_renames`
     (`pipeline/theory_compiler.py:1044`) to compute `∂^|α| f` at the saddle.

2. *Building variables with assumptions.* Matrix parameters become grids of SR variables,
   propagating the `domain` so Maxima (Sage's underlying CAS engine) can dispatch the right
   assumption (`pipeline/theory.py:1634`):

   ```python
   return [[SR.var(f'{_w}{i+1}{j+1}', domain=_d) for j in range(n_cols)]
           for i in range(n_rows)]
   ```

3. *The preparse-then-eval pattern.* Rather than call `sage_eval` everywhere, `_safe_eval`
   imports Sage's preparser explicitly (`pipeline/theory_compiler.py:702`):

   ```python
   import sage.all as _sage_all
   from sage.repl.preparse import preparse
   text = _normalize_expr_text(text)
   parsed = preparse(text)
   eval_globals = {**_sage_all.__dict__, **locals_dict}
   return eval(parsed, eval_globals)
   ```

   Two subtleties are spelled out in the docstring (`pipeline/theory_compiler.py:682`):
   (a) `preparse` performs the `^`→`**` and integer-lift rewrites; (b) the namespace is
   passed as Python **globals** rather than **locals** so that generator expressions inside
   `sum(... for j in pop)` can see the bound names `w`, `dn`, etc. — Python's `eval` only
   feeds its `locals` arg to the *outermost* scope, so a comprehension's inner scope would
   not see locals. Putting everything in globals fixes the comprehension scoping.

4. *Symbolic Fourier transform of kernels.* (`pipeline/theory_compiler.py:1510`)

   ```python
   from msrjd.core.field_theory import fourier_transform as _ft
   ...
   img = _ft(time_val, t_var, omega)
   ```

   This wraps Sage's `integrate` to compute `∫ g(t) e^{iωt} dt`. The docstring warns it is
   "best-effort" — Sage handles `heaviside(t)*exp(-t/tau)` without explicit positivity
   assumptions for typical neural kernels, but fails on harder forms, in which case the
   user must supply `freq_image` directly.

5. *Pattern substitution to kill saddle-times-operator terms.* `SR.wild()` plus dict-subs
   is used to enforce "operator applied to a constant saddle is zero," e.g.
   `Dt·vstar = 0`, `Laplacian·phistar = 0` (`pipeline/theory_compiler.py:1389`–`1397` and
   the mirror in `make_action_lambda` at `pipeline/theory.py`/compiler `:910`–`:938`).

6. *Inert Sage function nodes for the operator IR.* In `spatial_operator_ir.py`,
   `function('Lap')`, `function('Dt')`, `function('Dx')` create the binding nodes; the
   compiler imports `Lap, Dt, Dx` (`pipeline/theory_compiler.py:589`) and binds them into
   the action namespace so `Lap(phi)` builds the inert `Lap(...)` tree.

**Note on what is NOT here.** The `MEMORY.md` mentions nauty (graph canonicalization for
diagram enumeration), numba (JIT for hot integrals), scipy (`nquad`), and tropical
Monte-Carlo. None of those appear in the three authoring files — they are downstream of the
model dict. The brief therefore does not cover them; they belong to the enumeration /
integration subsystems.

### Python standard library

- `dataclasses` (`@dataclass`, `field`) — the spec records `FieldSpec`, `ParameterSpec`,
  `KernelSpec` (`pipeline/theory.py:65`), and the template dataclasses
  (`pipeline/theory_templates.py:51`).
- `typing` — `Any, Callable, Optional` annotations.
- `re` — regex, used for: classifying equations by `Dt` presence
  (`pipeline/theory.py:1143`+), rejecting `Conv(...)` and `Dt` on the wrong side of a DAE
  equation, auto-deriving MF equations by textual `field → fieldstar` substitution
  (`_auto_populate_mf_eqs_from_equations`, `pipeline/theory.py:1276`), and the
  single-function-call detector `_is_single_function_call` (`pipeline/theory_compiler.py:1195`).
- `textwrap` — `_normalize_expr_text` dedents triple-quoted action strings
  (`pipeline/theory_compiler.py:673`).
- `itertools.product` — `_iter_multi_indices` enumerates Taylor multi-indices
  (`pipeline/theory_compiler.py:969`); deliberately iterative (not a self-recursive
  generator) so it survives `%autoreload` hot-swaps in notebooks.
- `operator` (`operator.index`) — robust integer coercion of the Dyson `order` that accepts
  a Sage `Integer` from the notebook preparser but rejects bools/floats
  (`pipeline/theory.py:254`).
- `collections.defaultdict` — groups operator-IR vertex terms by signature
  (`pipeline/theory_compiler.py:776`).

---

## Components

This section is exhaustive. Each entry gives `file:line`, signature, inputs/outputs, and a
step-by-step. Classes that exist purely to make user-typed text behave (the wrapper
classes) are covered in full because they are the heart of how the text DSL works.

### Dataclasses (record types) — `pipeline/theory.py`

#### `FieldSpec` — `pipeline/theory.py:71`

```python
@dataclass
class FieldSpec:
    name: str
    indexed: bool
    latex: str
    description: str = ''
    natural_name: Optional[str] = None   # e.g. 'dn' has natural_name='n'
    population: Optional[str] = None      # heterogeneous-pop annotation
    spatial_dim: int = 0                  # 0 = time-only; d≥1 = φ(x,t) in d dims
```

A single physical or response field declaration. `name` is the *internal* name (e.g. `dn`
for a fluctuation field, `nt` for a response field). `natural_name` is the user-facing
letter (`n`). `spatial_dim` is **per-field** — the design decision that spatial dimension is
a field property, not a global, with `.spatial_dim(d)` as ergonomic bulk-setter.

#### `ParameterSpec` — `pipeline/theory.py:89`

```python
@dataclass
class ParameterSpec:
    name: str
    indexed: bool = False         # True = vector
    matrix: bool = False          # True = N×N matrix
    domain: Optional[str] = None  # e.g. 'positive'
    default: Any = None
    description: str = ''
    mean_field: bool = False      # True for saddle quantities (nstar, vstar, …)
    natural_name: Optional[str] = None
    indexed_by: Optional[list[str]] = None  # [] scalar, ['E'] vector, ['E','I'] matrix
```

A model parameter. `mean_field=True` flags it as a saddle-point quantity (the `*star`
variables). `indexed_by` is the modern heterogeneous-population annotation; `indexed`/`matrix`
are the legacy flags. Both are normalized at declaration time (see `parameter` below).

#### `KernelSpec` — `pipeline/theory.py:104`

```python
@dataclass
class KernelSpec:
    name: str
    sage_name: str = ''           # internal SR var name (default 'z_'+name)
    latex_name: str = ''
    frequency_image: Optional[Callable] = None
    description: str = ''
    indexed: bool = False         # vector g[i]
    matrix:  bool = False         # matrix g[i, j]
    indexed_by: Optional[list] = None
```

A convolution-kernel symbol. The actual frequency/time images for text-declared kernels are
stored separately in `self._kernel_specs` (a list of plain dicts) and compiled later; the
`KernelSpec` is the symbol-shape record the framework uses for name resolution.

### Mixin: spatial-only authoring methods — `_SpatialMethods` (`pipeline/theory.py:118`)

These methods are mixed into `SpatialTheoryBuilder` and the legacy `TheoryBuilder`.

#### `spatial_dim(self, d: int)` — `pipeline/theory.py:122`

- **Takes:** an int `d`. **Returns:** `self` (for chaining).
- **Does:** Casts `d` to int, stores it as `self._default_spatial_dim` (the default for
  fields declared *afterwards*), and retro-sets `spatial_dim = d` on every already-declared
  physical field. `d = 0` reverts to time-only; `d ∈ {1, 2, 3}` are validated end-to-end.
  All spatial fields must share one dimension (enforced later in `build`).

#### `boundary(self, mode: str, **params)` — `pipeline/theory.py:148`

- **Takes:** `mode ∈ {'infinite', 'periodic'}`; for periodic, a `length=` (a string naming
  a declared parameter, or a number for the inline shortcut). **Returns:** `self`.
- **Does:** Validates `mode` (raises `ValueError` for anything but the two supported modes;
  Dirichlet/Neumann/Robin are explicitly v2). For periodic, requires `length`. Stores
  `self._boundary = {'mode': mode, **params}`. Emitted as `model['boundary']`.

#### `initial(self, mode: str = 'stationary', **params)` — `pipeline/theory.py:182`

- **Takes:** `mode` (only `'stationary'` is supported in v1). **Returns:** `self`.
- **Does:** Validates and stores `self._initial`. Emitted as `model['initial']`. Transient
  ICs are v1.5 (Lefèvre-Biroli §2.5 `S_I` term).

#### `dyson(self, mode='fixed', order=None, tol=None)` — `pipeline/theory.py:206`

- **Takes:** `mode ∈ {'off', 'fixed'}`; for `'fixed'`, an `order` int ≥ 0. **Returns:** `self`.
- **Does:** Rejects v2+ policies (`'auto'`, `'adaptive'`, `'resum'`) with `NotImplementedError`.
  For `'fixed'`, robustly coerces `order` via `operator.index` (accepting a Sage `Integer`,
  rejecting bools/floats/negatives) and stores `self._dyson = {'mode': 'fixed', 'order': N}`.
  This is the Dyson–Duhamel truncation policy for coupled unequal-diffusion theories
  (`𝒟̂ ≠ 0`). Stored as `model['spatial']['dyson']`.

#### `dyson_order(self, order: int)` — `pipeline/theory.py:270`

- Sugar for `.dyson(mode='fixed', order=order)`.

#### `reference_diffusion(self, D0)` — `pipeline/theory.py:276`

- **Takes:** a float `D0 > 0`. **Returns:** `self`.
- **Does:** Validates positivity and stores `self._reference_diffusion`. This is the scalar
  reference `D₀` for the `𝒟 = D₀·I + 𝒟̂` split underlying the Dyson series. Emitted as
  `model['spatial']['reference_diffusion']`.

### Mixin: temporal-only authoring methods — `_TemporalMethods` (`pipeline/theory.py:300`)

Mixed into `TemporalTheoryBuilder` and legacy `TheoryBuilder`.

#### `population(self, name, *, size=1, description='')` — `pipeline/theory.py:304`

- **Takes:** a population name, a size. **Returns:** `self`.
- **Does:** Appends `{'name', 'size', 'description'}` to `self.populations` and resets
  `self.n_populations = len(self.populations)`. A *named index set with its own size* —
  heterogeneous-population theories chain one per group. **Caveat in docstring:** populations
  are recorded but NOT yet fully propagated into the symbolic/diagrammatic pipeline (the
  pipeline currently treats all populations as one combined index set of size `sum(sizes)`).

#### `kernel(self, name, frequency_image=None, sage_name='', latex_name='', description='', indexed=False, indexed_by=None)` — `pipeline/theory.py:330`

- **Takes:** kernel name + indexing flags. **Returns:** `self`.
- **Does:** Normalizes `indexed_by` (modern) over `indexed` (legacy) into `(is_vec, is_mat)`,
  then appends a `KernelSpec`. `sage_name` defaults to `'z_'+name`.

#### `define_kernel(self, name, *, time_expr=None, freq_image=None, …, indexed=False, indexed_by=None)` — `pipeline/theory.py:370`

- **Takes:** at least one of `time_expr` / `freq_image` (Sage-syntax text strings).
  **Returns:** `self`.
- **Does:** Raises if both are `None`. Registers the kernel symbol via `self.kernel(...)`
  (if not already present), then appends a *plain dict* spec
  `{'name', 'time_expr', 'freq_image', 'indexed', ['indexed_by']}` to `self._kernel_specs`.
  This dict (not the `KernelSpec`) is what the compiler's `make_kernel_ft_image_lambda`
  consumes. The text may reference `i`/`j` and any declared parameter, e.g.
  `'1/(1 + I*omega*tau_g[i, j])'`.

#### `markovianize(self, enabled=True)` — `pipeline/theory.py:429`

- **Takes:** a bool. **Returns:** `self`.
- **Does:** Sets `self._markovianize_default`. When on (default), `build()` walks
  `_cgf_terms` and rewrites colored-noise rows matching `c·exp(-|tau|/tauc)` into white-noise
  OU-auxiliary rows + a linear filter in the action. Per-row override via
  `declare_cgf_term(..., markovianize=…)`.

#### `declare_cgf_term(self, name, response_field=None, order=2, coefficient='', kernel=None, *, response_legs=None, markovianize=None)` — `pipeline/theory.py:461`

- **Takes:** A non-closed-form cumulant term. `name` groups rows (same name+order sum into
  one cumulant). `order` is the cumulant order (2 for κ², 3 for κ³…). `coefficient` is a
  Sage-syntax expression. `kernel` is an optional time-domain factor (`'dirac_delta(tau)'`
  white, `'exp(-abs(tau)/tauc)'` OU-colored). `response_field` (legacy, single) OR
  `response_legs` (per-leg list, for cross-field cumulants). **Returns:** `self`.
- **Does:** Validates at least one of `response_field`/`response_legs` is given. Normalizes
  `response_legs` from a comma-string to a list. Normalizes `markovianize` (`'auto'` →
  `None`; rejects other strings). Appends a dict to `self._cgf_terms`. Consumed by
  `make_correlated_noises_block`.

#### `correlated_noise(self, name, **kwargs)` — `pipeline/theory.py:557`

- Trivial: stores `self._correlated_noises[name] = kwargs`. A low-level escape hatch.

#### `use_synaptic_kernel(self, template)` — `pipeline/theory.py:561`

- **Does:** Applies an `ExpSynapticKernel`/`DeltaKernel` template. Sets
  `self._kernel_ft_image` from the template's `kernel_ft_image()`. For `DeltaKernel`, merges
  the kernel's `extra_specializations()` into the action template's specializations closure
  (so `g → delta_D` happens at propagator-construction time).

#### `add_gtas_noise(self, template)` — `pipeline/theory.py:591`

- **Does:** A heavyweight convenience that wires a `GTaSNoise` template in: stores it as
  `self._pending_gtas`, auto-declares the `dm`/`mt` fields (with `auto_response=False` to
  avoid declaring `mt` twice — see the long comment at `:606`), auto-declares the GTaS
  parameters with sane defaults, merges the `correlated_noises_block()`, and re-emits the
  action with the GTaS hook installed.

#### `use_action_template(self, template)` — `pipeline/theory.py:665`

- **Does:** Remembers the template object (`self._action_template`), applies any pending
  GTaS hook, then pulls the six companion lambdas off the template: `_action`,
  `_phi_concrete`, `_specializations`, `_mf_bg`, `_mf_equations`, `_mf_substitutions`, plus
  the `_functions` list. This is the *template* path (vs the *text* path).

### Base builder: `_BaseTheoryBuilder` (`pipeline/theory.py:683`)

#### `__init__(self, name, n_populations=1)` — `pipeline/theory.py:693`

Initializes every accumulator list/dict and every lambda hook to its empty/None default.
The most important state buckets:

- Declaration lists: `populations`, `response_fields`, `physical_fields`, `parameters`,
  `kernels`.
- Lambda hooks (set by `set_*` or by templates or by text compilation):
  `_action`, `_mf_bg`, `_mf_bg_solver`, `_mf_equations`, `_kernel_ft_image`,
  `_kernel_td_image`, `_phi_concrete`, `_specializations`, `_mf_substitutions`,
  `_functions`, `_operators`, `_correlated_noises`.
- Text-driven declarations (compiled at build): `_function_specs`, `_kernel_specs`,
  `_action_text`, `_operator_ir`, `_mf_eqs_text`, `_cgf_terms`, `_phi_function_name`,
  `_equations`, `_stability_analysis`, `_markovianize_default`.
- Spatial state: `_default_spatial_dim`, `_boundary`, `_initial`, `_dyson`,
  `_reference_diffusion`.

#### `response_field(self, name, indexed=True, latex='', description='', population=None)` — `pipeline/theory.py:796`

- Appends a `FieldSpec` to `self.response_fields`. Returns `self`.

#### `physical_field(self, name, indexed=True, latex='', description='', natural_name=None, auto_response=True, auto_saddle=True, population=None, spatial_dim=None)` — `pipeline/theory.py:806`

- **The most important field method.** Two calling styles distinguished by `natural_name`:
  - *New style* (`natural_name is None`): the `name` you pass IS the natural letter. The
    internal fluctuation field is derived as `d<name>` (so `physical_field('n')` →
    internal `dn`, natural `n`).
  - *Legacy style* (`natural_name` given): `name` is the literal internal name,
    `natural_name` is the separate user letter (`physical_field('dn', natural_name='n')`).
- **Spatial:** resolves `spatial_dim` (explicit kwarg wins, else inherits
  `self._default_spatial_dim`).
- **Auto-response:** unless `auto_response=False`, declares the conjugate response field
  `<natural>t` (e.g. `nt`), skipping if already present.
- **Auto-saddle:** unless `auto_saddle=False`, declares the saddle parameter `<natural>star`
  (e.g. `nstar`) with `mean_field=True`, `domain='positive'` only for `n` (rates) and free
  for others.
- Returns `self`.

#### `parameter(self, name, default=None, indexed=False, domain=None, description='', mean_field=False, natural_name=None, indexed_by=None)` — `pipeline/theory.py:914`

- Normalizes `indexed_by` over `indexed`/`matrix` into `(is_vec, is_mat)` and appends a
  `ParameterSpec`. Returns `self`.

#### `define_function(self, name, args, expression, latex=None, description='', population=None)` — `pipeline/theory.py:969`

- **Takes:** a function name, its formal arg list (e.g. `['v']`), and a Sage-syntax body
  (e.g. `'a*v^2'`). **Returns:** `self`.
- **Does:** Appends a dict `{'name', 'args', 'expression', 'latex', 'description',
  ['population']}` to `self._function_specs`. Functions may reference any declared parameter
  by name in their body.

#### `set_action_text(self, text)` — `pipeline/theory.py:1008`

- Stores `self._action_text = text`. The big docstring (`:1008`–`:1049`) is the canonical
  reference for action authoring: per-population integrand with implicit free index `i`,
  inner sums as `sum(... for j in pop)`, the plain-vs-operator-IR spatial styles, the
  non-Gaussian-white-noise monomial rule, and coupled-field cross terms.

#### `operator_ir(self, on=True)` — `pipeline/theory.py:1053`

- Stores `self._operator_ir`. When on, the action is parsed into the spatial operator IR and
  lowered to derived ring generators before expansion (default off — every existing theory
  is unaffected).

#### `set_mf_equation(self, saddle_name, rhs_text)` — `pipeline/theory.py:1069`

- Stores `self._mf_eqs_text[saddle_name] = rhs_text`. The *legacy* MF-equation surface
  (per-saddle RHS string). The framework auto-emits `phi0_<i> = nstar[i]` for EOM closure.

#### `equation(self, *, lhs, rhs, population=None)` — `pipeline/theory.py:1087`

- **Takes:** `lhs`/`rhs` Sage-syntax strings; optional `population`. **Returns:** `self`.
- **Does (step by step):**
  1. Validates `lhs`/`rhs` are non-empty strings.
  2. Rejects `Conv(...)` on either side (regex `\bConv\s*\(`, case-insensitive) — the DAE
     assumes stationary MF so kernel convolutions of constants must be pre-collapsed.
  3. Rejects `Dt` in `rhs` (regex `\bDt\b`) — derivatives belong on the LHS.
  4. Classifies `kind = 'differential' if re.search(r'\bDt\b', lhs) else 'algebraic'`.
  5. Appends `{'lhs_text', 'rhs_text', 'population', 'kind'}` to `self._equations`.
- This is the **modern DAE residual** surface (the multi-root Newton solver path).

#### `stability_analysis(self, enabled)` — `pipeline/theory.py:1187`

- Stores `self._stability_analysis`. Emitted as `model['stability_analysis']`. When True,
  the DAE solver classifies linear stability of every converged root and restricts
  `fixed_point_index` to the stable subset. Default off (vacuous for all-algebraic theories).

#### `set_transfer_function(self, name='phi')` — `pipeline/theory.py:1220`

- Stores `self._phi_function_name` — marks which declared function plays the role of the
  MSR-JD `phi_concrete` (the saddle-Taylor-expansion target). Defaults to `'phi'`.

#### Lambda-hook setters — `pipeline/theory.py:1231`–`1267`

Direct escape hatches that bypass text compilation: `set_action(fn)`,
`set_mf_bg_conditions(fn)`, `set_mf_equations(fn)`, `set_kernel_ft_image(fn)`,
`set_phi_concrete(fn)`, `set_specializations(fn)`, `set_mf_substitutions(subs)`,
`add_function(spec)`, `add_operator(spec)`. Each stores onto the corresponding `_…`
attribute and returns `self`. Used when a theory's structure is too unusual for the text DSL
or templates.

#### `_auto_populate_mf_eqs_from_equations(self)` — `pipeline/theory.py:1276`

- **Takes/Returns:** nothing (mutates `self._mf_eqs_text`).
- **Does:** Bridge from the modern `.equation(...)` API to the legacy `_mf_eqs_text` dict
  (so the FieldTheory sanity check + action-substitution chain keep working). For each
  declared equation: substitute `Dt → 0` in the LHS text; find the single state-variable
  reference (`x[i]` indexed form first, then bare `x` scalar form); infer the saddle name as
  `<var>star`; textually rewrite every state-variable name in the RHS to its `*star`
  counterpart (word-boundary regex avoids partial hits like `Em` vs `E`); for scalar LHS,
  also rewrite `xstar` → `xstar[i]` since the legacy compiler expects indexed access. Stores
  `self._mf_eqs_text[saddle_name] = rewritten_rhs`. **No-op** if `_equations` is empty or
  `_mf_eqs_text` is already populated. Raises a clear `ValueError` if the LHS can't be
  reduced to a single state-variable reference.

#### `_compile_text_declarations(self)` — `pipeline/theory.py:1367`

- **The central compilation driver.** Step by step:
  1. Calls `_auto_populate_mf_eqs_from_equations()`.
  2. Early-returns if there are no text declarations at all (template-only builders pass
     through unchanged).
  3. Imports the nine `make_*_lambda` / `make_*_block` factories from `theory_compiler`.
  4. Computes the three name lists (`field_names` = response + physical, `param_names`,
     `kernel_names`) and the `_action_naming_convention` dict (`fluctuation_fields`,
     `mean_field_saddles`, `mf_parameters`).
  5. Picks a "primary" function for `phi_concrete` plumbing — `_phi_function_name` if set,
     else the first single-arg function (multi-arg functions can't be `phi_concrete` since
     the saddle solver inverts along ONE variable).
  6. If `_action_text` set: builds `self._action = make_action_lambda(...)`.
  7. Registers EVERY declared function into `self._functions` as a formal indexed entry
     (with both an SR `expression` lambda and the original text under `expression_text` /
     `args_text` for the numerical DAE solver).
  8. If a single-arg primary exists: builds `self._phi_concrete` and `self._specializations`.
  9. If `_mf_eqs_text`: builds `self._mf_bg`, `self._mf_bg_solver`, `self._mf_equations`.
     The iteration-saddle sentinel is `'AUTO'` for heterogeneous-pop theories, else `'nstar'`.
  10. If `_kernel_specs`: builds `self._kernel_ft_image` and `self._kernel_td_image`.
  11. If `_cgf_terms`: merges `make_correlated_noises_block(...)`.
  12. For every matrix parameter, appends a `_matrix_subst` closure to
      `self._mf_substitutions` producing the `[[w_{i+1}{j+1}]]` SR-var grid (sized via
      `indexed_by` pop sizes, else `ns.pop`; domain propagated).

#### `_inject_autopop(self)` — `pipeline/theory.py:1646`

- **Scalar-mode autopop.** Appends a single-position population `pop` (size 1) and binds
  every unbound physical/response field to it. Lives on the base (not the temporal mixin)
  so `build()` can call it for any builder. Only fires for `n_populations <= 1`.

#### `_resolve_spatial_dim(self)` — `pipeline/theory.py:1670`

- Collects the set of nonzero per-field `spatial_dim` values; raises if there is more than
  one distinct nonzero dim (v1 single-dimension constraint); returns the single dim or 0.
  **Overridden** by the forward builders to assert their domain (Temporal must be 0, Spatial
  must be > 0).

#### `build(self) -> dict` — `pipeline/theory.py:1688`

- **The terminal method.** Step by step:
  1. **Scalar-mode autopop** (if no populations + has physical fields + `n_populations <= 1`).
     The long comment at `:1697` explains why this is gated on `<= 1`: firing it for a legacy
     2-pop Hawkes model would clobber the population count to 1 and zero cross-population
     correlators.
  2. **Markovianize:** if `_cgf_terms`, call `markovianize_spec(self)` from
     `pipeline.colored_to_markovian` BEFORE compiling, so the downstream compiler sees the
     rewritten spec.
  3. **Compile text declarations** (`_compile_text_declarations()`).
  4. **Validate required hooks:** `action` is always required; `phi_concrete` is required
     only when there is a single-arg function; `kernel_ft_image` is optional (DeltaKernel
     returns none). Raises `ValueError` listing what's missing.
  5. Build the `naming_convention` dict (natural↔internal maps + `mf_parameters`).
  6. Build the index sets: `pop` (flat, sized `sum(sizes)` or `n_populations`) plus
     `pop_<name>` (local per-population indices) when populations are declared.
  7. Classify iteration saddles (`_classify_mf_eqs`) → `iteration_saddles` list.
  8. **Resolve spatial dimension** and set `is_spatial`.
  9. **Reserved-name validation** (see Data structures / Gotchas) — hard error if any
     field/parameter/function/kernel name collides with a framework symbol.
  10. Build the operators list (always `Dt`; append `Laplacian` when spatial).
  11. Validate boundary/initial/dyson/reference_diffusion against `is_spatial`; resolve the
      inline-number `length` shortcut into a hidden `_pbc_length_L0` parameter.
  12. Assemble the `model` dict (see Data flow for the full key list).
  13. Conditionally attach the lambda hooks that are non-None. Note the careful naming:
      `mf_bg_conditions_action` (closure-baked, for FieldTheory.expand) and
      `mf_bg_conditions` (solver-friendly, raw concrete) are stored under DIFFERENT keys
      (`:1974`–`:1986`).
  14. Emit the `model['spatial']` / `model['boundary']` / `model['initial']` blocks only
      when spatial.
  15. Return `model`.

#### Helpers `_field_dict`, `_param_dict`, `_kernel_dict` — `pipeline/theory.py:2034`–`2084`

- Static methods that serialize each spec dataclass into the plain dict the model expects,
  including each conditional field only when set (so the dict stays sparse).

### Forward builder classes — `pipeline/theory.py:2087`–`2127`

- **`TemporalTheoryBuilder(_TemporalMethods, _BaseTheoryBuilder)`** — overrides
  `_resolve_spatial_dim` to raise if any field is spatial.
- **`SpatialTheoryBuilder(_SpatialMethods, _BaseTheoryBuilder)`** — `__init__` defaults
  `n_populations=0`; overrides `_resolve_spatial_dim` to raise if NO field is spatial.
- **`TheoryBuilder(_SpatialMethods, _TemporalMethods, _BaseTheoryBuilder)`** — both mixins,
  auto-detect; behaviorally identical to the pre-split class.

### Compiler internals — `pipeline/theory_compiler.py`

#### `_unwrap_field_arg(a)` — `pipeline/theory_compiler.py:49`

- Unwraps size-1 field wrappers (`_FullPhysicalField`, `_FieldScalar`) to their bare SR
  expression before passing into Sage's strict `BuiltinFunction.__call__` (`exp`, `log`,
  formal functions). Sage's argument coercion ignores `_sage_()`; without unwrapping,
  scalar-mode calls like `f(v)` raise `no canonical coercion` errors. No-op for plain SR.

#### `class _IndexedFormalFunction` — `pipeline/theory_compiler.py:73`

- Generates Sage **formal** function calls indexed by population. `phi[i](dv[i])` →
  `function('phi_{i+1}')(dv[i])`; `phi[i, j](dv[i])` → `function('phi_{i+1}_{j+1}')(...)`.
  Bare call `f(v)` is treated as `f[0]`. Used in the **action** namespace so FieldTheory's
  auto-Taylor pass produces `phi0_<i>`, `phi1_<i>`, … vertex symbols.

#### `class _IndexedSaddleRename` — `pipeline/theory_compiler.py:131`

- Maps formal function calls in mf_bg evaluation **directly** to the framework's Taylor-rename
  target (`phi0_<i+1>` etc.), bypassing Sage's symbolic expansion, so the action-side closure
  substitution lines up with the bigrade pass's vertex names. Handles multi-arg
  (`f00_<i+1>`). The `_build_target` suffix is `'0' * n_args`.

#### `class _FullPhysicalField` — `pipeline/theory_compiler.py:206`

- Exposes `n[i] = nstar[i] + dn[i]` (full physical field = saddle + fluctuation) in
  user-facing action text, so the user writes `nt[i] * n[i]` instead of
  `nt[i] * (nstar[i] + dn[i])`. `__getitem__(i)` returns `saddle[i] + fluct[i]`. For size-1
  theories it also forwards bare arithmetic (`xt * x`) via the dunder methods, all routing
  through `_scalar()` which raises a clear `TypeError` if size ≠ 1. `_sage_()` returns the
  scalar so Sage coercion can find it.

#### `class _FieldScalar` — `pipeline/theory_compiler.py:268`

- Thin wrapper around a *list* of SR variables that supports BOTH bare scalar arithmetic
  (when `len == 1`) and `[i]` indexed access. Binds response fields (`xt`) and internal
  fluctuation fields (`dx`) in scalar theories so `xt * x` AND `xt[0] * x[0]` both work. For
  size > 1, scalar arithmetic raises `TypeError`.

#### `class _MatrixView` — `pipeline/theory_compiler.py:316`

- Wraps a list-of-lists so it accepts both tuple subscript `w[i, j]` and chained subscript
  `w[i][j]`. Lets the user write `w[i, j]` in action text. Used for matrix parameters and
  matrix kernels.

#### `_unwrap_builtin(sage_fn)` — `pipeline/theory_compiler.py:342`

- Wraps a Sage builtin so its arguments pass through `_unwrap_field_arg` first. Without it,
  `exp(nt)` in scalar mode raises `no canonical coercion from _FieldScalar to SR`.

#### `_builtin_namespace()` — `pipeline/theory_compiler.py:359`

- Returns the dict of symbols every user expression may freely reference: wrapped
  `exp/log/sin/cos/tan/sqrt/heaviside`, `delta_function`/`dirac_delta`, plus `sum`, `I`,
  `pi`. (`delta_function` is an alias for `dirac_delta`.)

#### `_ns_var_namespace(ns, field_names, param_names, kernel_names, naming_convention=None, expand_to_full=False)` — `pipeline/theory_compiler.py:383`

- Assembles the user-name → ns-attribute dict. Each declared field/parameter/kernel name is
  exposed under its name. Size-1 fields are wrapped in `_FieldScalar`. When a naming
  convention is supplied, fields are ALSO exposed under their natural names — bound to the
  *fluctuation* alias (`expand_to_full=False`, MF context) or to `_FullPhysicalField`
  (`expand_to_full=True`, action context). Matrix params/kernels → `_MatrixView`. Operators
  `Dt`/`Laplacian` are exposed if present on `ns`.

#### `_build_namespace_for_eval(ns, *, field_names, param_names, kernel_names, functions, n_pop, transfer_function=None, transfer_function_mode='formal', i=None, naming_convention=None, extra=None)` — `pipeline/theory_compiler.py:493`

- **The full eval namespace builder.** Returns the dict to feed `_safe_eval` as `locals`.
  Includes the builtins, the ns-derived symbols, the iteration ranges
  (`pop = range(n_pop)`, `pop_<name>`, plain-name aliases), and binds every user function
  according to `transfer_function_mode`:
  - `'action'` → `_IndexedFormalFunction` (formal symbols for auto-Taylor).
  - `'mf_formal_rename'` → `_IndexedSaddleRename` (direct rename targets).
  - else (`'concrete'`) → `_make_function_callable` (evaluates the body).
  Also binds `Conv` (the formal convolution operator from `msrjd.core.convolution`) in every
  mode, and — when `transfer_function_mode == 'action'` and `ns._operator_ir` — binds
  `Lap`/`Dt`/`Dx` as the operator-IR binding calls.

#### `class _IndexableCallable` — `pipeline/theory_compiler.py:602`

- A **concrete** function exposing both `phi(v)` and `phi[i](v)`. `phi[i]` returns a copy
  carrying `fixed_idx`; calling evaluates the body via `sage_eval` with the formal args
  bound and (if indexed) `i`/`j` bound into scope. Raises `TypeError` on arity mismatch.

#### `_make_function_callable(fn_spec, parent_ns)` — `pipeline/theory_compiler.py:658`

- One-liner factory → `_IndexableCallable`.

#### `_normalize_expr_text(text)` — `pipeline/theory_compiler.py:664`

- Dedents (triple-quoted strings carry indentation that the Python parser rejects), strips,
  and collapses newlines+tabs to single spaces so a multi-line sum parses as one expression.

#### `_safe_eval(text, locals_dict, what)` — `pipeline/theory_compiler.py:682`

- The preparse-then-`eval` core (see External tools §SageMath point 3). On error, raises a
  rich `ValueError` quoting the text, the preparsed form, the original error, AND the sorted
  list of available symbols — this is the single most useful debugging affordance in the
  subsystem.

#### `_lower_operator_ir_action(s, ns, naming_convention)` — `pipeline/theory_compiler.py:720`

- **The spatial-v2 operator-IR lowering.** Step by step:
  1. Pull the fluctuation SR vars from `ns._all_field_sr_vars`.
  2. `apply_linearity` (operators distribute over sums, pull out constant coefficients).
  3. `kill_means` over the `*star*` saddle symbols (homogeneous-saddle annihilation).
  4. `to_derived_generators` → replace each atomic `Op(fluctuation)` with a fresh ring
     generator; stash `ns._operator_ir_genmap = {gen: (base, op_chain)}`.
  5. `classify_generators` → split into BILINEAR (fold into propagator) and VERTEX (carry
     per-leg momentum form factors).
  6. Build the per-vertex-type form-factor TABLE (`ns._operator_ir_vertex_terms`): for each
     derivative vertex compute base field-degree (must be 1 or 2 else `NotImplementedError`),
     its power `p`, coupling `c_t`, `n_phys = bdeg·p`, mode (`'perleg'` if bdeg==1 like
     KPZ `(∂φ)²`, `'composite'` if bdeg==2 like Model B `∇²(φ²)`), and a normalized weight
     `c_t / Σcouplings-of-same-signature`. Single-vertex theories get weight ≡ 1.
  7. Lower BILINEAR generators back to v1 bare symbols (`Lap → ns.Laplacian`,
     `Dt → ns.Dt`, first-derivative `Dx → GRADX_SYM`; raises on axis ≠ 0 for d≥2 transverse
     drift). Unfold VERTEX generators to their bare composite (drop the operator) so diagram
     enumeration sees the plain `φ̃φ²` topology.
  8. Return `SR(s).subs(subs)`.

#### `make_action_lambda(action_text, *, field_names, param_names, kernel_names, functions, n_pop, transfer_function=None, naming_convention=None)` — `pipeline/theory_compiler.py:851`

- Returns the closure `_action(ns)` that: builds the action namespace (mode `'action'`),
  `_safe_eval`s the text, optionally lowers the operator IR, **kills `op·saddle` terms**
  (when NOT operator-IR: builds a wildcard subs dict zeroing `Dt·vstar`, `Laplacian·phistar`,
  etc., on the EXPANDED form), and augments `ns._deriv_rename_subs` with saddle-point Taylor
  renames for every declared function. Returns the symbolic action.

#### `_iter_multi_indices(n_args, max_total)` — `pipeline/theory_compiler.py:958`

- Yields every multi-index `(k_1, …, k_n)` of non-negative ints summing ≤ `max_total`. Uses
  `itertools.product` (iterative, autoreload-safe).

#### `_augment_saddle_renames(ns, functions, *, naming_convention=None, taylor_order=4)` — `pipeline/theory_compiler.py:975`

- Adds saddle-point Taylor-rename entries to `ns._deriv_rename_subs` for every function.
  For an n-arg formal function, every multi-derivative `∂^|α| f` at the saddle (with
  `sum α ≤ taylor_order`) is registered as a rename to `SR.var(f'{tname}{suffix}_{i+1}')`.
  Single-arg collapses to the legacy `<tname><k>_<i+1>` naming. Resolves each argument's
  saddle array via `naming_convention['mean_field_saddles']` (fallback `<arg>star`); an
  argument with no saddle aborts that function's registration.

#### `make_phi_concrete_lambda(phi_fn_spec, *, …)` — `pipeline/theory_compiler.py:1051`

- Builds `model['phi_concrete']` from a single user-declared transfer function. The returned
  `_phi_concrete(ns, i, v_sym)` binds the function's (single) first argument to `v_sym` and
  `_safe_eval`s the body. Raises if the function takes ≠ 1 argument.

#### `make_specializations_lambda(phi_fn_spec, *, …, naming_convention=None, mf_eqs=None)` — `pipeline/theory_compiler.py:1086`

- Auto-derives `model['specializations']` for every declared function. For each function,
  each population index `i`, and each Taylor multi-index `α`, emits
  `f<α>_<i+1> → ∂^|α| f / ∂args evaluated at the saddle`. Iterates over the saddle's actual
  *size* (not `n_pop`) — the comment at `:1152` documents the single-pop-size>1 bug this
  avoids. `mf_eqs` param is retained for API compat but unused.

#### `_is_single_function_call(rhs_text)` — `pipeline/theory_compiler.py:1195`

- True iff the RHS is exactly one function call `f(arg)` with no surrounding arithmetic.
  Regex match + top-level paren-balance check. Used to detect saddle-EOM closures so the
  framework substitutes the iteration saddle with the formal rename target.

#### `_classify_mf_eqs(mf_eqs, iteration_saddle)` — `pipeline/theory_compiler.py:1230`

- Splits mf_eqs into `{'closure': …, 'compound': …}`. `iteration_saddle` may be a single
  name, a set/list of names, or the `'AUTO'` sentinel (every single-function-call RHS is a
  closure saddle). Closure saddles get formal-rename substitution; compound saddles get
  concrete substitution.

#### `make_mf_bg_conditions_lambda(mf_eqs, phi_fn_spec, *, …, iteration_saddle='nstar')` — `pipeline/theory_compiler.py:1269`

- Builds the **action-side** `mf_bg_conditions` lambda. Two-pass:
  - *Pass 1 (closure):* iteration-saddle RHS evaluated in `'mf_formal_rename'` mode →
    `nstar[i] → phi0_<i+1>` (or `+ b` for compound forms).
  - *Pass 2 (compound):* every other saddle's RHS evaluated concretely, then the closure
    subs applied to the result, so `vstar` no longer contains raw `nstar`. This makes the
    `(1,0)` tadpole cancel symbolically.
  - Finally adds the `op·saddle → 0` kill entries via `SR.wild()`. The comment at `:1378`
    stresses this MUST mirror the same rule in `make_action_lambda` or the MF solver fails
    to converge.

#### `make_mf_bg_solver_lambda(mf_eqs, phi_fn_spec, *, …)` — `pipeline/theory_compiler.py:1403`

- Builds the **solver-friendly** version: every saddle entry in raw concrete form (other
  saddles kept as raw SR symbols). `solve_mean_field` reads this as the iteration target.

#### `make_mf_equations_lambda(mf_eqs, phi_fn_spec, *, …)` — `pipeline/theory_compiler.py:1455`

- Builds `model['mf_equations']` — the list of residual equations `ns.<saddle>[i] == <rhs>`.
  `solve_mean_field` doesn't use it numerically (it uses fsolve), but FieldTheory keeps it
  for symbolic consistency checks.

#### `make_kernel_ft_image_lambda(kernel_specs, *, param_names)` — `pipeline/theory_compiler.py:1493`

- Builds `model['kernel_ft_image']`. Returns `_ft_image(ns, omega) → {ns.<kname>: SR_in_ω}`.
  For each spec: uses `freq_image` text directly (fast) or Fourier-transforms `time_expr`
  via `msrjd.core.field_theory.fourier_transform`. Handles scalar/vector/matrix kernels by
  iterating the SR symbol shape with `i`/`j` in scope.

#### `make_kernel_td_image_lambda(kernel_specs, *, param_names)` — `pipeline/theory_compiler.py:1603`

- Parallel to the FT version but emits the **time-domain** image `{ns.<kname>: SR_in_t}`.
  Used by the time-domain Phase J integrator at conductance-style vertices. Kernels declared
  only via `freq_image` (no `time_expr`) are skipped.

#### `_parse_response_legs(term, order)` — `pipeline/theory_compiler.py:1684`

- Resolves a CGF row's `response_legs` (preferred) / `response_field` (legacy) into an
  ordered per-leg list of length `order`. A single name is broadcast across all legs; a
  list must have exactly `order` entries (else `ValueError`).

#### `class _CGFKernelCallable` — `pipeline/theory_compiler.py:1727`

- A **picklable** callable (module-level, `__slots__`) that evaluates
  `coeff_text * kernel_text` against a FieldTheory namespace. Signature
  `(ns, *leg_indices, *tau_syms) -> SR`. Validates the τ-symbol count (`order - 1`), builds
  an eval namespace from `ns`'s public attributes plus the τ symbols (`tau` at order 2,
  `t1, t2, …` generally) and the standard math builtins, then `_safe_eval`s the coefficient
  and kernel. Picklable so `multiprocessing.Pool` workers can ship `TypedDiagram` objects
  carrying these callables via `NoiseSourceType`.

#### `class _CGFKernelSum` — `pipeline/theory_compiler.py:1800`

- A picklable callable that sums several `_CGFKernelCallable` outputs (when multiple CGF rows
  share `(name, order, response_legs)`).

#### `_build_cgf_kernel_callable(coeff_text, kernel_text, order)` — `pipeline/theory_compiler.py:1821`

- Thin factory → `_CGFKernelCallable`.

#### `make_correlated_noises_block(cgf_terms, *, param_names)` — `pipeline/theory_compiler.py:1833`

- Converts declared CGF terms into `model['correlated_noises']`. Groups rows by name; within
  a name, by `(name, order)`. Resolves per-leg response legs; conflicting leg-lists at the
  same order raise. Compiles each `(name, order)` group into a single `_CGFKernelCallable`
  (one row) or `_CGFKernelSum` (multiple). The per-name entry has
  `{'response_field', 'response_legs', 'cumulants', 'cumulant_text'}`. The docstring at
  `:1873` notes the old text-only path was a *structural bug* — the framework dereferences
  `cumulants[order]` as a callable.

### Templates — `pipeline/theory_templates.py`

#### `_gauss(tau, mu, sigma_sq)` — `pipeline/theory_templates.py:59`

- Unit-area Gaussian density, used by GTaSNoise's cross-cumulant kernel.

#### `@dataclass HawkesAction` — `pipeline/theory_templates.py:69`

- The canonical 2-pop Hawkes action generator. Fields: `phi ∈ {'linear',
  'linear_unit_gain', 'quadratic'}`, `gain_param='a'`, `synaptic_kernel='g'`,
  `recurrent_weight='w'`, `external_drive='E'`, `membrane_timescale='tau'`, the four field
  names, and the optional `_gtas` add-on. Methods each return a lambda:
  - `phi_concrete()` (`:123`) → `φ(v)` per the `phi` choice.
  - `specializations()` (`:133`) → the `phi_k_i → concrete derivative` dict.
  - `mf_substitutions()` (`:169`) → the w-matrix expansion + `ndot_bg`.
  - `mf_bg_conditions()` (`:187`) → `v*_i = E_i + Σ w·g·n*_j (+ GTaS)`, `phi0_i = n*_i`.
  - `mf_equations()` (`:217`) → the saddle residuals per `phi` choice.
  - `action()` (`:263`) → the full MSR-JD Hawkes action lambda (with GTaS terms when set).
  - `functions_list()` (`:319`) → `phi_i(v)` as a FORMAL indexed function (the docstring
    explains why concrete would short-circuit the auto-Taylor machinery).

#### `@dataclass ExpSynapticKernel` — `pipeline/theory_templates.py:353`

- `g(t) = (1/τ_g)·e^{-t/τ_g}·Θ(t)` with image `ĝ(ω) = 1/(1 + iω τ_g)`. `kernel_ft_image()`
  returns the substitution lambda; `extra_specializations()` returns `None`.

#### `@dataclass DeltaKernel` — `pipeline/theory_templates.py:374`

- Instantaneous `g(t) = δ(t)`. `kernel_ft_image()` returns `None`; `extra_specializations()`
  returns `lambda ns: {ns.g: ns.delta_D}` so cell-8 sees it as δ(t).

#### `@dataclass GTaSNoise` — `pipeline/theory_templates.py:400`

- Bernoulli + Gaussian GTaS external-rate process. `correlated_noises_block()` (`:436`)
  returns the κ²/κ³/κ⁴ dict (Poisson auto + Bernoulli+Gaussian cross at κ²; Poisson auto only
  at κ³/κ⁴).

---

## Data structures

### The spec dataclasses

`FieldSpec`, `ParameterSpec`, `KernelSpec` — see Components. Held in the builder's
`response_fields`, `physical_fields`, `parameters`, `kernels` lists.

### Function spec (plain dict) — `self._function_specs` entries

```python
{
  'name':        'phi',
  'args':        ['v'],
  'expression':  'a*v^2',
  'latex':       r'\varphi',
  'description': '...',
  'population':  'E',     # optional
}
```

### Compiled function entry — `self._functions` / `model['functions']` entries

```python
{
  'name':            'phi',
  'indexed':         True,
  'deriv_prefix':    'phi',
  'n_args':          1,
  'latex':           '...',
  'description':     '...',
  'expression':      <lambda i, *xs: function(f'phi_{i+1}')(*xs)>,  # formal SR
  'expression_text': 'a*v^2',     # original text, for the numerical DAE solver
  'args_text':       ['v'],
  'population':      'E',         # optional
}
```

### Kernel text spec — `self._kernel_specs` entries

```python
{
  'name':       'g',
  'time_expr':  '(1/tau_g)*exp(-t/tau_g)*heaviside(t)',  # or None
  'freq_image': '1/(1 + I*omega*tau_g)',                  # or None
  'indexed':    False,           # or True / 'vector' / 'matrix'
  'indexed_by': ['E', 'I'],      # optional
}
```

### CGF term — `self._cgf_terms` entries

```python
{
  'name':           'X',
  'response_field': 'mt',        # legacy single, may be None
  'response_legs':  ['xt','yt'], # per-leg list, None when legacy single
  'order':          2,
  'coefficient':    'lambda_X * p_part',
  'kernel':         'exp(-abs(tau)/tauc)',  # or None
  'markovianize':   None,        # True / False / None(=auto)
}
```

### DAE equation — `self._equations` / `model['equations']` entries

```python
{
  'lhs_text':   '(Dt+mu)*x[i]',
  'rhs_text':   '-eps*x[i]^3',
  'population': 'pop',      # or None
  'kind':       'differential',   # or 'algebraic' (inferred from Dt in lhs)
}
```

### Naming convention dict — `model['naming_convention']`

```python
{
  'fluctuation_fields': {'n': 'dn', 'v': 'dv'},     # natural → internal
  'mean_field_saddles': {'n': 'nstar', 'v': 'vstar'},  # natural → internal
  'mf_parameters':      ['nstar', 'vstar'],          # internal names
}
```

### Spatial blocks — `model['spatial']`, `model['boundary']`, `model['initial']`

```python
model['spatial'] = {
  'dim': 1,
  'fields_with_spatial': ['dh'],
  'dyson': {'mode': 'off'},      # or {'mode': 'fixed', 'order': N}
  'reference_diffusion': 1.0,    # only when declared
}
model['boundary'] = {'mode': 'infinite'}   # or {'mode': 'periodic', 'length': '<param>'}
model['initial']  = {'mode': 'stationary'}
```

### Wrapper objects (runtime, inside the eval namespace)

`_FullPhysicalField(saddle_array, fluct_array)`, `_FieldScalar(arr)`, `_MatrixView(rows)`,
`_IndexedFormalFunction(name)`, `_IndexedSaddleRename(name, n_args)`,
`_IndexableCallable(fn_spec, parent_ns, fixed_idx)`. These never appear in the model dict;
they live transiently in the namespace dict that `_safe_eval` consumes.

### The `ns` (namespace) object

The lambdas all take an `ns` argument — a `NamespaceForFields` built by the *consumer*
(`msrjd.core.field_theory`), not by this subsystem. Relevant attributes the lambdas read:
`ns.<fieldname>` (lists of SR vars), `ns.<paramname>`, `ns.<kernelname>`, `ns.pop`,
`ns.Dt`, `ns.Laplacian`, `ns._taylor_order`, `ns._pop_local_idx`, `ns._pop_size`,
`ns._deriv_rename_subs`, `ns._all_field_sr_vars`, `ns._operator_ir`, `ns.delta_D`.

---

## Data flow

### Inbound: what the user supplies

A `theories/<name>.theory.py` file defines `build()` returning the model dict. Concrete
temporal example (`theories/ou_quartic.theory.py`):

```python
from pipeline.theory import TemporalTheoryBuilder

def build():
    return (
        TemporalTheoryBuilder('OU Quartic (white noise)')
        .population('pop', size=1)
        .physical_field('x', population='pop', description='variable')
        .parameter('mu', default=1.0, domain='positive')
        .parameter('eps', default=0.02, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .set_action_text('sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)')
        .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='pop')
        .build()
    )
```

Concrete spatial example (`theories/kpz_1d.theory.py`):

```python
from pipeline.theory import SpatialTheoryBuilder

def build():
    return (
        SpatialTheoryBuilder('1D KPZ (per-leg gradient vertex)', n_populations=0)
        .physical_field('h', spatial_dim=1, description='interface height')
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('D',   default=1.0, domain='positive')
        .parameter('lam', default=0.3, domain='real')
        .parameter('T',   default=1.0, domain='positive')
        .equation(lhs='(Dt + mu - D*Laplacian)*h', rhs='0')
        .set_action_text('ht*(Dt(h) + mu*h - D*Lap(h) - (lam/2)*Dx(h, 0)^2) - T*ht^2')
        .operator_ir()
        .boundary('infinite')
        .initial('stationary')
        .build()
    )
```

### Internal: what the builder accumulates and compiles

- `.physical_field('x')` → appends FieldSpec `dx`(internal)/`x`(natural); auto-declares
  response `xt` and saddle `xstar` (mean_field).
- `.parameter(...)` → ParameterSpec entries.
- `.set_action_text(...)` → `_action_text` string.
- `.equation(...)` → `_equations` entries (`kind` classified by `Dt` in lhs).
- `.build()` → autopop (size-1 `pop`), `_auto_populate_mf_eqs_from_equations` derives
  `_mf_eqs_text` (e.g. `{'xstar': '-eps*xstar[i]^3 ... }`), then `_compile_text_declarations`
  turns each text string into a lambda.

### Outbound: the model dict (`model`)

Always present keys (`pipeline/theory.py:1943`):
`name`, `populations`, `iteration_saddles`, `index_sets`, `response_fields`,
`physical_fields`, `parameters`, `kernels`, `equations`, `operators`, `functions`,
`mf_substitutions`, `phi_concrete`, `action`, `naming_convention`, `stability_analysis`,
`operator_ir`.

Conditionally attached lambda keys: `mf_bg_conditions_action`, `mf_bg_conditions`,
`mf_equations`, `kernel_ft_image`, `kernel_td_image`, `specializations`,
`correlated_noises`.

Spatial-only keys (when any field is spatial): `spatial`, `boundary`, `initial`.

### Consumer: `compute_cumulants`

`pipeline.compute.compute_cumulants(model, k, max_ell, fundamental, external_fields, …)`
(`pipeline/compute.py:108`) reads these keys. Its docstring lists the *required* hooks:
`response_fields, physical_fields, parameters, kernels, operators, functions,
mf_substitutions, mf_bg_conditions, specializations, kernel_ft_image, phi_concrete,
mf_equations, action`; `correlated_noises` optional. The builder produces exactly these.
The `fundamental` dict (numeric values) comes from the theory file's `DEFAULT_FUNDAMENTAL`
constant, and `external_fields` / `tau_max` / `spatial_grid` come from `METADATA`.

---

## Gotchas & caveats

1. **The file is a self-described PROTOTYPE / STUB.** The module docstring
   (`pipeline/theory.py:1`–`62`) still describes a roadmap and an older API surface
   (`TheoryBuilder`, `transfer_function`, `use_action_template`, `set_action`). Much of the
   roadmap is now done (text compilation, templates), but the top docstring was not fully
   rewritten — do not trust it over the method bodies.

2. **Reserved names are a SCOPED hard error** (`pipeline/theory.py:1834`–`1885`). Always
   reserved: `t`, `omega`, `Dt`, `delta_D`, `delta_Dp` (FT variables + operator symbols).
   Reserved ONLY in spatial theories: `k`, `Laplacian`, `x`, `y`, `z` (wavevector +
   diffusion operator + spatial coordinates). The comment warns that `k` colliding with a
   parameter is a *silent wrong-mass bug* (`SR.var('k')` in `heat_kernel.py`). Note the irony:
   `x` is the conventional default field name in *temporal* theories but is reserved in
   *spatial* ones (where it's the coordinate). The build raises with a rename hint
   (e.g. `x → phi`).

3. **`mf_bg_conditions` is stored TWICE under two keys.** `mf_bg_conditions_action`
   (closure-baked, for `FieldTheory.expand`) and `mf_bg_conditions` (raw concrete, for
   `solve_mean_field`). The closure-baked version pre-substitutes the iteration saddle so
   the `(1,0)` tadpole cancels symbolically; the solver version keeps raw `nstar` so the
   numerical iteration can run. If only one mf_bg exists (legacy template path), it's used
   for both (`pipeline/theory.py:1974`–`1986`).

4. **The `op·saddle → 0` kill rule is duplicated and MUST stay in sync.** It appears in both
   `make_action_lambda` (`compiler`) and `make_mf_bg_conditions_lambda`. The comment at
   `pipeline/theory_compiler.py:1378` is explicit: "both lambdas walk the expression tree
   independently, so a missed operator here silently leaves a non-zero saddle term and the
   MF solver fails to converge." This is a fragile, manually-mirrored invariant.

5. **Sage's strict coercion ignores `_sage_()`.** The entire `_unwrap_field_arg` /
   `_unwrap_builtin` machinery exists solely because Sage's `BuiltinFunction.__call__` will
   not call `_sage_()` on a Python wrapper — `exp(xt)` raises `no canonical coercion` without
   the unwrap (`pipeline/theory_compiler.py:49`, `:342`). Any new builtin exposed to user
   text must be wrapped the same way or scalar-mode calls will break.

6. **Generator-expression scoping forces globals-not-locals.** `_safe_eval` passes the whole
   namespace as `eval` globals because Python's `eval` only feeds `locals` to the outermost
   scope; a comprehension's inner scope (`sum(... for j in pop)`) would not see `locals`
   (`pipeline/theory_compiler.py:682`). Consequence: user-supplied names *shadow* Sage
   builtins (e.g. a parameter named `I` would shadow the imaginary unit). This is documented
   as intentional but is a foot-gun.

7. **`_iter_multi_indices` is deliberately iterative, not recursive.** A self-recursive
   generator fails to find itself in module globals after a `%autoreload` hot-swap in
   notebooks (`pipeline/theory_compiler.py:958`).

8. **Iterate over saddle SIZE, not `n_pop`.** Multiple comments (compiler `:1152`, `:1316`,
   theory_compiler in `make_mf_bg_conditions_lambda`) warn that using `n_pop` (the *count of
   populations*) instead of `len(saddle_array)` (the *size of the population*) silently skips
   every saddle index past the first for single-pop-size>1 theories — leaving formal symbols
   unsubstituted and producing 0 candidate roots downstream.

9. **Scalar-mode autopop is gated on `n_populations <= 1`.** Firing it for a legacy 2-pop
   Hawkes model would clobber the count to 1, drop `nt2`/`vt2`/…, and zero every
   cross-population correlator. Such theories instead keep `populations` empty and use the
   flat legacy index path (`pipeline/theory.py:1697`–`1722`).

10. **`add_gtas_noise` declares `mt` with `auto_response=False` on purpose.** Leaving
    auto-response on would declare `mt` twice → a duplicate `mt1` generator in the bigrade
    `PolynomialRing` → `ValueError: variable name 'mt1' appears more than once`
    (`pipeline/theory.py:606`).

11. **`.equation()` forbids `Conv(...)` and `Dt` on the wrong side.** `Conv(...)` (any case)
    raises because the DAE assumes stationary MF (collapse kernel convolutions yourself);
    `Dt` in the `rhs` raises because derivatives belong on the LHS only
    (`pipeline/theory.py:1155`–`1172`).

12. **Operator-IR base field-degree is restricted to {1, 2}.** A derivative VERTEX of base
    field-degree other than 1 (per-leg, KPZ) or 2 (composite, Model B / Burgers) raises
    `NotImplementedError` (`pipeline/theory_compiler.py:784`). Bilinear `Dx` on axis ≠ 0
    (d≥2 transverse drift) also raises (`:837`).

13. **Populations are recorded but not fully propagated.** `population()`'s docstring
    (`pipeline/theory.py:304`) says the pipeline currently treats all populations as one
    combined index set of size `sum(sizes)`; full per-population machinery is a separate
    refactor.

14. **Markovianize default is ON and mutates the spec in `build()`.** With colored CGF rows,
    `build()` rewrites `_cgf_terms` and augments `_action_text` *before* compilation
    (`pipeline/theory.py:1735`). If a user hand-coded a colored kernel into the action they
    should set `.markovianize(False)` or use the per-row override.

15. **Best-effort symbolic Fourier transform.** `make_kernel_ft_image_lambda` will raise a
    `ValueError` telling the user to supply `freq_image` directly if Sage's `integrate`
    can't transform the `time_expr` (`pipeline/theory_compiler.py:1530`).

16. **`mf_eqs` parameter of `make_specializations_lambda` is dead.** Retained for API
    compatibility but unused — closure substitution moved to mf_bg
    (`pipeline/theory_compiler.py:1091`).

---

## Glossary

- **MSR-JD** — Martin–Siggia–Rose–Janssen–De Dominicis: the response-field path-integral
  formulation of classical stochastic dynamics. Doubles each field `φ` with a response field
  `φ̃` and encodes noise as response-field cumulant vertices.
- **Action** `S[φ, φ̃]` — the exponent of the MSR-JD path integral; the central object the
  user authors as text.
- **Response field** (`φ̃`, internal names `nt`, `vt`, `mt`, `xt`, `ht`) — the conjugate
  ("tilde") field. A degree-2 monomial in it is white Gaussian noise; degree ≥ 3 is
  non-Gaussian noise.
- **Physical field** (`φ`, natural names `n`, `v`, `x`, `h`) — the observable. Split into
  saddle + fluctuation.
- **Fluctuation field** (`dn`, `dv`, `dx`) — `δφ`, the internal field the action is
  Taylor-expanded in.
- **Saddle / mean-field** (`nstar`, `vstar`, `xstar`) — the deterministic steady state `φ̄`.
- **Tadpole** — the linear-in-fluctuation term of the expanded action. The saddle equation
  is exactly the condition that the `(1,0)` tadpole vanishes.
- **Transfer function** (`phi`) — the firing-rate nonlinearity `φ(v)`. (Name clashes with the
  field `φ`; in code it is the user function literally named `phi`.) Its Taylor coefficients
  at the saddle become vertex couplings `phi0_i, phi1_i, …`.
- **Vertex** — an interaction term of the action (cubic and higher in the fields) that
  becomes a diagram vertex.
- **Kernel** (`g`) — a convolution kernel (e.g. a synaptic filter). Needs a frequency image
  `ĝ(ω)`.
- **CGF** — cumulant generating functional. A `declare_cgf_term` row contributes one cumulant
  of the (possibly non-closed-form) noise.
- **κ⁽ⁿ⁾** — the n-th cumulant of the noise (κ² Gaussian, κ³/κ⁴ higher).
- **GTaS** — the Bernoulli+Gaussian external-rate noise process modeled by `GTaSNoise`.
- **DAE** — differential-algebraic equation. The `.equation()` residual system solved at MF.
- **Multi-start Newton** — solve the algebraic MF system from many random seeds, sort the
  roots, pick the `fixed_point_index`-th.
- **Linear stability** — classify a converged root by the generalized eigenvalue problem
  `(B − iωA)δx = 0`; opt in with `.stability_analysis(True)`.
- **Markovian embedding** — replace colored noise with a white-noise-driven auxiliary OU
  field so loop integrals stay tractable.
- **Operator IR** — the `Lap()`/`Dt()`/`Dx()` argument-binding intermediate representation
  for spatial/temporal derivative operators (`pipeline/spatial_operator_ir.py`).
- **Form factor** — the Fourier image of an operator chain on a diagram leg (`∇² → −k²`,
  `∂_t → −iω`, `∂_x → ik`).
- **Bilinear vs vertex generator** — IR generators that fold into the propagator (degree ≤ 2)
  vs those that carry per-leg momentum form factors into the integrator (degree ≥ 3).
- **per-leg vs composite** — derivative-vertex modes: `(∂φ)²` (KPZ, derivative on each leg)
  vs `∂(φ²)` / `∇²(φ²)` (Burgers / Model B, derivative on the composite).
- **Dyson–Duhamel truncation** — the fixed-order series used to dress retarded edges in
  coupled unequal-diffusion (`𝒟̂ ≠ 0`) spatial theories.
- **Population** — a named index set (group of identical units) with its own size.
- **Iteration / closure saddle** — a saddle whose MF equation RHS is a single function call
  (`nstar = phi(vstar)`); substituted via the formal rename. A **compound** saddle's RHS is
  a parameter/saddle expression; substituted concretely.
- **SR / Symbolic Ring** — SageMath's algebra of symbolic expressions; `SR.var('x')` makes a
  symbolic variable.
- **Preparse** — Sage's source rewrite that turns `^`→`**` and lifts integer literals.
- **Formal / undefined function** — `sage.function('f')(x)`: an unapplied symbolic function
  that `taylor()` can differentiate formally without a concrete body.
- **Picklable** — serializable by Python's `pickle` so `multiprocessing.Pool` can ship it
  between processes (why `_CGFKernelCallable`/`_CGFKernelSum` are module-level classes).
- **`ns` (NamespaceForFields)** — the runtime object the lambdas receive, carrying the
  symbolic field/parameter SR variables as attributes; built by the consumer, not here.
- **Phase J** — the time-domain integration stage of the pipeline (the consumer of
  `kernel_td_image`).

---

## Proposed manual subsections

1. **Why a builder?** — the model dict, its Sage lambdas, and the cost of hand-writing it.
2. **Choosing a builder class** — `TemporalTheoryBuilder` vs `SpatialTheoryBuilder` vs the
   legacy unified `TheoryBuilder`; the spatial-dim auto-detect at `build()`.
3. **Declaring fields** — `physical_field` (new vs legacy style), auto-response,
   auto-saddle, natural vs internal names; `response_field`.
4. **Declaring parameters** — scalar / vector / matrix, `domain`, `mean_field`,
   `indexed_by` vs legacy `indexed`.
5. **Writing the action as text** — `set_action_text`, the per-population integrand, the
   implicit index `i`, inner sums, full-physical-field vs explicit fluctuation form.
6. **Functions and the transfer function** — `define_function`, `set_transfer_function`,
   formal-symbol Taylor expansion, `phi0_i`/`phi1_i`/… vertex couplings.
7. **Mean-field: the two surfaces** — legacy `set_mf_equation` vs modern `.equation()` DAE
   residuals; the `Dt → 0` reduction; multi-root Newton; `stability_analysis`.
8. **Kernels** — `define_kernel`, `time_expr` vs `freq_image`, the symbolic Fourier
   transform, indexed kernels.
9. **Noise beyond white Gaussian** — `declare_cgf_term`, κ-orders, `response_legs`,
   colored noise and `markovianize`.
10. **Templates** — `HawkesAction`, `ExpSynapticKernel`, `DeltaKernel`, `GTaSNoise`, and
    `add_gtas_noise` / `use_action_template`.
11. **Spatial theories** — `spatial_dim`, `boundary`, `initial`, the plain-Laplacian style.
12. **The operator IR** — `.operator_ir()`, `Lap()/Dt()/Dx()`, per-leg vs composite
    derivative vertices, form factors (KPZ / Burgers / Model B).
13. **Coupled unequal-diffusion spatial theories** — `dyson`, `dyson_order`,
    `reference_diffusion`.
14. **How text becomes lambdas** — the `theory_compiler` namespace machinery, the wrapper
    classes, `_safe_eval`, preparse, the unwrap dance.
15. **The model dict reference** — every key, its type, when it's present.
16. **Reserved names and other pitfalls** — the scoped reserved-name table; the
    duplicated kill rule; saddle-size-vs-n_pop; scalar autopop gating.
17. **End-to-end example** — `ou_quartic.theory.py` and `kpz_1d.theory.py` walked line by
    line, from `build()` to `compute_cumulants`.

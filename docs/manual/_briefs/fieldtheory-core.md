# Subsystem brief: Action Representation, Taylor Expansion & Vertices (`fieldtheory-core`)

Source files covered (read in full):

- `msrjd/core/field_theory.py`   (1387 lines) — the `FieldTheory` expander, the cumulant-action injector, the bigrade classifier, the MF-sector verifier, the symbolic Fourier transforms.
- `msrjd/core/vertices.py`        (795 lines) — decomposition of bigrade-sector polynomials into typed vertex/source records (`VertexType`, `SourceType`, and their kernel-carrying subclasses).
- `msrjd/core/convolution.py`     (354 lines) — the formal `Conv(kernel, field)` operator and its reduction rules.
- `msrjd/core/serialize.py`       (245 lines) — save/load of an expanded theory to a directory (`metadata.json` + `symbolic_data.sobj`) and model re-import.

Worked example referenced throughout: `models/hawkes_quad_expg.py` (quadratic Hawkes, 2 populations, exponential synaptic filter). Reading that file alongside this brief is recommended — it is the canonical "model dict" shape this subsystem consumes.

---

## 1. Overview

### What this subsystem does, in plain language

In the Martin–Siggia–Rose–Janssen–De Dominicis (MSR-JD) formalism, a stochastic dynamical system is rewritten as a field theory with an **action** `S`. The action is a functional of two kinds of fields:

- **physical fields** (the actual fluctuating quantities, e.g. a voltage `v` or a spike rate `n`), and
- **response fields** (also called "tilde" or "hat" fields, the conjugate auxiliary fields introduced by the MSR-JD construction, e.g. `ṽ`, `ñ`).

To compute correlation functions perturbatively (Feynman diagrams), you must:

1. expand the action around its **mean-field (MF) saddle point** (i.e. write each field as `background + fluctuation`),
2. **Taylor-expand** every nonlinear function of the fields to a finite order, and
3. **sort the resulting polynomial terms** by how many response legs and how many physical legs each term has. The constant/linear terms encode the saddle equations; the bilinear (one-response, one-physical) term is the inverse free propagator; everything cubic and above is an **interaction vertex** that diagrams are built from.

This subsystem is the machine that does exactly that, **symbolically**, for an arbitrary model supplied as a Python dictionary. It is the "front end" of the whole pipeline: you hand it a declarative model dict, it hands back the action, sorted into sectors, and a list of typed interaction vertices and noise sources.

Two crucial design choices shape the whole subsystem:

- **The action is symbolic, not numeric.** Every field, parameter, kernel, and operator is a SageMath symbolic-ring (`SR`) variable. Nothing is bound to a number until much later in the pipeline. This is what lets the framework hand the same expanded theory to a Fourier-domain propagator extractor, a time-domain integrator, and a numerical mean-field solver.
- **The model is data, not code-per-model.** A model is a dict of strings, lists, and *lambdas*. The lambdas (the action, the substitutions, the kernels) take a *namespace object* as argument and return `SR` expressions. The framework builds the namespace, calls the lambdas, and never needs to know anything model-specific.

### Where it sits in the end-to-end pipeline

```
model dict  (models/*.py, e.g. HAWKES_MODEL)
   │
   ▼
FieldTheory(model, taylor_order).expand()          ←—  THIS SUBSYSTEM (field_theory.py)
   │   • builds the symbolic namespace
   │   • evaluates the action lambda → one big SR expression
   │   • resolves Conv(...) atoms            (convolution.py)
   │   • injects correlated-noise cumulants  (_build_cumulant_action)
   │   • multivariate Taylor to total degree taylor_order
   │   • renames formal-derivative symbols   (phi'(0) → phi1_1, …)
   │   • coerces to a PolynomialRing(SR, fields) and bigrade-classifies
   │   • verifies + zeroes the MF (saddle) sector
   ▼
ft._by_tp : { (n_tilde, n_phys) : sector polynomial }
   │
   ├── ft.free_action()  →  (1,1) sector  →  inverse free propagator     (downstream: propagator extraction, _propagator.py)
   ├── ft.noise_kernel() →  (≥2,0) sectors → noise sources
   └── ft.vertices()     →  total-degree-≥3 sectors → interaction vertices
   │
   ▼
extract_vertex_types(ft) / extract_source_types(ft)   ←—  THIS SUBSYSTEM (vertices.py)
   │   • decompose each sector polynomial into individual monomials
   │   • tag each monomial with its response-leg / physical-leg multiset
   │   • upgrade conductance vertices to ConvVertexType (carry g(τ) kernel)
   │   • upgrade non-local noise sources to NoiseSourceType (carry κ(τ) kernel)
   ▼
list[VertexType], list[SourceType]
   │
   ▼
diagram enumeration (msrjd/enumeration/degree_scan.py — calls available_degrees)
   │
   ▼
type assignment (Phase E) → integration (time-domain / spatial) → cumulants
```

**What feeds this subsystem:** the model dict authored in `models/*.py` (or compiled by `pipeline/theory_compiler.py` from `theories/*.theory.py`). The key entry call is `FieldTheory(model, taylor_order).expand()`. Callers in the repo include `pipeline/compute.py` (the main driver, line 314), `msrjd/enumeration/degree_scan.py` (line 122, re-expands a saved theory when a diagram needs a higher Taylor order), and `pipeline/_precompute.py`.

**What consumes its output:** the diagram enumerator (`degree_scan.py` uses `available_degrees` to know which vertex/source degrees exist), the propagator extractor (reads `ft.free_action()`), and the integrators in `msrjd/integration/{time_domain,spatial}/` (consume `VertexType`/`SourceType`/`ConvVertexType`/`NoiseSourceType` objects). `serialize.py` lets the expanded theory be cached to disk so the (expensive) multivariate Taylor step is not repeated.

---

## 2. The math

This section builds the relevant theory from the ground up so the code can be read against it.

### 2.1 The MSR-JD action and its fields

Start from a Langevin-type stochastic equation of motion `E[φ] = ξ` where `φ` is the physical field and `ξ` is noise. The MSR-JD construction introduces, for each physical field `φ`, a conjugate **response field** `φ̃` and writes the generating functional as a path integral `∫ DφDφ̃ e^{-S[φ,φ̃]}` (sign conventions vary; here the code keeps `S` as written by the model author and the path-integral weight is `e^{-S}`/`e^{S}` according to the model's own convention — the framework never imposes one).

For the worked Hawkes example, the action authored in `models/hawkes_quad_expg.py:236` is

```
S = Σ_i {  ñ_i (ṅ*_i + δn_i)
         − (e^{ñ_i} − 1) φ_i(δv_i)
         + ṽ_i [ (τ ∂_t + 1) δv_i + v*_i − E_i − Σ_j w_{ij} g·(n*_j + δn_j) ] }
```

Here `ñ_i, ṽ_i` are response fields (tilde fields), `δn_i, δv_i` are physical *fluctuation* fields (already written as the fluctuation around the background `n*_i, v*_i`), `g` is a synaptic kernel (a convolution operator), and `φ_i(v) = a v²` is a nonlinear transfer function.

### 2.2 Saddle (mean-field) expansion

The physical field is split into a constant background plus a fluctuation: `φ = φ* + δφ`. The background `φ*` is fixed by the **mean-field saddle equations** — the conditions that the terms in `S` linear in a single field vanish. Concretely, in the worked model the Poisson saddle is `n*_i = φ_i(v*_i) = a (v*_i)²` and the voltage saddle is `v*_i = E_i + Σ_j w_{ij} ∫g · n*_j` (the kernel integrates to 1, so `∫g = 1`).

The framework does **not** solve these equations itself — that is a downstream module (`pipeline/_mean_field.py`, `_mean_field_dae.py`). What this subsystem does is *encode them as substitutions* (`mf_bg_conditions`) and *test that the saddle sector vanishes under them* (`_verify_and_zero_mf_sector`).

### 2.3 Taylor expansion to a given order

Every nonlinear function of the fields must be replaced by a finite polynomial. For a single-argument function `f(δφ)` expanded around the saddle (`δφ = 0`),

```
f(δφ) = Σ_{k=0}^{N}  f^{(k)}(0)/k!  · δφ^k          (truncated at order N = taylor_order)
```

For a multi-argument function `f(x_1,…,x_n)` the framework uses the **multivariate Taylor series truncated at total degree N**:

```
f(x) = Σ_{|α|≤N}  (1/α!) (∂^α f)(0) · x^α
```

where `α = (k_1,…,k_n)` is a multi-index, `|α| = Σ k_j`, `α! = Π k_j!`, and `x^α = Π x_j^{k_j}`. Truncating at *total* degree (not per-variable degree) is the physically correct choice: the total degree of a monomial in the fields equals the number of legs of the diagrammatic vertex it produces, so "Taylor order N" ↔ "vertices with up to N legs."

The code realizes this with SageMath's `taylor(expr, (v1,0), (v2,0), …, N)` (one-shot multivariate). A subtle but important comment (`field_theory.py:884-892`) explains why a one-shot multivariate Taylor *replaced* an earlier sequential single-variable loop: chained `.taylor()` calls do **not** compose correctly at non-zero expansion points for multi-argument formal functions (the inner-argument substructure gets frozen by the outer call).

### 2.4 Formal functions and derivative symbols

A model author often does not want to commit to a closed-form `φ`; they write `φ_i` as a **formal Sage function** (`function('phi_1')(v)` — see `hawkes_quad_expg.py:142`). When SageMath Taylor-expands a formal function it produces ugly derivative-evaluation atoms like `D[0](phi_1)(0)` (first derivative of `phi_1` evaluated at 0) or `D[0,0](phi_1)(0)` (second derivative). The framework renames these to clean polynomial-ring-friendly symbol names:

```
phi_1(0)        → phi0_1
D[0](phi_1)(0)  → phi1_1
D[0,0](phi_1)(0)→ phi2_1
```

For a single-argument function the suffix is the derivative order `k`; for a multi-argument function it is the concatenated multi-index `k_1 k_2 … k_n` (so `f^{(2,1)}` → `f21_<i>`). The renaming dictionary is `ns._deriv_rename_subs`, built in `_build_namespace`. A model then `specializes` these abstract derivative symbols to concrete values: e.g. for `φ = a v²`, `specializations` sets `phi1_i → 2 a v*_i`, `phi2_i → 2a`, `phi_{k≥3} → 0` (`hawkes_quad_expg.py:186`).

### 2.5 Bigrade classification

After expansion the action is a polynomial in the field generators. Each monomial is classified by a **bigrade** `(n_tilde, n_phys)`:

- `n_tilde` = total exponent over the **response** (tilde) generators,
- `n_phys`  = total exponent over the **physical** (fluctuation) generators.

The physical interpretation of each sector:

| bigrade `(n_t, n_p)` | name | meaning |
|---|---|---|
| `(0,0)` | constant | must vanish (action has no constant piece at the saddle) |
| `(1,0)` | tadpole | response-linear; must vanish at the MF saddle |
| `(0,1)` | EOM residual | physical-linear; must vanish at the background solution |
| `(1,1)` | **free action** | the bilinear kernel — its inverse is the free propagator |
| `(≥2, 0)` | **noise kernel** | response-only terms ≥ quadratic — these are the noise sources |
| total ≥ 3 | **interaction vertex** | every higher-order monomial is a vertex |

The first three sectors `{(0,0),(1,0),(0,1)}` are collectively the **MF (saddle-eq) sector**. They must be zero at the saddle; the framework verifies this and then forcibly zeroes them.

### 2.6 Convolution semantics

When a synaptic kernel `g` multiplies a field `n` in the action, the `*` syntactically reads as scalar multiplication but **semantically means convolution in time**:

```
(g * n)(t) ≡ (g ⋆ n)(t) = ∫ g(t − s) n(s) ds
```

The reduction of a convolution depends on the downstream consumer:

- **stationary mean field:** a convolution against a *constant* picks up the kernel's DC gain `ĝ(0) = ∫g(t)dt`. For unit-integral kernels `ĝ(0)=1`, so `Conv(g, c) → c`.
- **Fourier-domain propagator:** by the convolution theorem `Conv(g, n)(t) → ĝ(ω) n̂(ω)`. So a kernel symbol multiplying a *fluctuation* is left as `g · n`, with `g` later substituted by `ĝ(ω)` via `model['kernel_ft_image']`.

The asymmetry of `Conv` matters: only the second argument (the field) carries the time-domain content. A conductance-style term `v · Conv(g, n)` must expand under `v=v*+δv, n=n*+δn` as `v*n* + v* g δn + δv n* + δv g δn` — a naïve `Conv→g·n` flatten gives the wrong `g·n*` bilinear coefficient. This is the entire motivation for the `Conv` operator (see `convolution.py:144-157`).

### 2.7 Correlated-noise cumulants (the `-W_m[m̃]` series)

Some models have *colored* or *correlated* external noise. Such noise contributes a cumulant-generating-functional series to the action:

```
S_cum = − Σ_n (1/n!) ∫ dt_1…dt_n Σ_{i_1…i_n} κ^{(n)}_{i_1…i_n}(τ's) · m̃_{i_1}(t_1) … m̃_{i_n}(t_n)
```

where `κ^{(n)}` is the n-th cumulant of the noise and `m̃` are the response legs the noise couples to. The framework injects this series symbolically from a declarative `model['correlated_noises']` block. Each per-leg kernel `K(τ)` is split into a **local (delta-correlated)** part `c_local · δ(τ)` and a **smooth non-local** part `K_smooth(τ)`:

```
K(τ) = c_local · δ(τ) + K_smooth(τ)
```

The local part collapses the time integral immediately and is injected as an ordinary monomial; the smooth part introduces a placeholder symbol `z_kappa_…` whose kernel function is registered for the downstream integrator. Only order n=2 smooth residuals are currently handled (see Gotchas).

---

## 3. External tools used

This subsystem touches exactly **one** heavy external library, SageMath, plus the Python standard library and IPython display. It does **not** use nauty, scipy, numba, or networkx directly — those appear in other subsystems (diagram enumeration, simulation, integration). The `multiprocessing` standard-library module is referenced indirectly (the picklability constraints in `vertices.py` are *designed around* a `multiprocessing.Pool` that lives downstream).

### 3.1 SageMath (the `sage` package)

**What it is.** SageMath is a large open-source mathematics system built on top of Python. It bundles many computer-algebra engines (Maxima, Pari, Singular, FLINT, GMP, Sympy, …) behind a unified Python API. For this subsystem the relevant part is its **symbolic ring** and its **polynomial rings**.

- **The Symbolic Ring `SR`.** `SR` is Sage's universal ring of symbolic expressions — think "anything you could write on a blackboard": `a*x^2 + sin(t) + exp(ñ)`. `SR.var('a')` creates a symbolic variable named `a`. Wrapping a value in `SR(...)` coerces it into the symbolic ring (so `SR(0)` is the symbolic zero, `SR(expr).expand()` distributes products). Crucially, SR variables can carry **assumptions** (`domain='positive'`) and **LaTeX display names** (`latex_name=...`).

- **`PolynomialRing(SR, names)`.** This builds a *polynomial ring whose coefficients live in `SR`* and whose *generators* are the named field variables. So `PolynomialRing(SR, ['nt1','vt1','dn1','dv1'])` lets the action be represented as a genuine polynomial in the four field generators, with the parameters/kernels (`a`, `w`, `g`, `τ`) living symbolically in the *coefficients*. This separation is the linchpin of bigrade classification: a polynomial element exposes its monomials via `.dict()` (a `{exponent_vector: coefficient}` map), and the exponent vector immediately gives `(n_tilde, n_phys)`.

**Exactly how this code uses Sage** — quoting the actual imports and calls:

```python
# field_theory.py:26-29
from sage.all import (
    SR, PolynomialRing, factorial, QQ, latex, LatexExpr,
    diff, function, exp, dirac_delta, heaviside, integrate, oo, I, pi, taylor
)
```

- `SR` / `SR.var(name, domain=..., latex_name=...)` — create every field, parameter, kernel, operator, and derivative symbol. Examples: `field_theory.py:1128` builds an indexed response field array `[SR.var(f"{fname}{i+1}", latex_name=...) for i in idx_list]`; `field_theory.py:1245` `ns.delta_D = SR.var('z_delta', latex_name=r'\delta')` (internal names start with `z` so they sort *after* `phi`/`tau` in any Sage product, giving a canonical factor order).
- `PolynomialRing(SR, ring_var_names)` (`field_theory.py:1164`) — the bigrade ring; `R.zero()`, `R.monomial(*exponents)`, `poly.dict()`, `poly.parent()` are all used by `_sr_to_ring`/`_collect_bigrade`.
- `taylor(S_sr, *[(v,0) for v in fields], order)` (`field_theory.py:894`) — the one-shot multivariate Taylor of the whole action around the saddle.
- `diff(deriv, arg, k)` (`field_theory.py:1310`) — symbolic partial differentiation, used to compute each formal-function partial derivative `∂^α f(0)` so its rename target can be registered.
- `factorial`, `QQ` — `_poly_taylor` (`field_theory.py:198`) builds `Σ c_n/n! x^n`; `QQ(1)/factorial(n)` is an *exact rational* (`QQ` is Sage's rational field), avoiding floating-point in the Taylor coefficients.
- `function('Conv', nargs=2)` (`convolution.py:70`) and `function(f'phi_{i+1}')(v)` (in models) — `function` builds a **formal (unevaluated) symbolic function**. `nargs=2` makes Sage validate call arity. A formal function stays inert through symbolic manipulation, which is exactly what `Conv` needs so downstream consumers can each apply their own reduction.
- `.substitute_function(Conv, _reduce_arg)` (`convolution.py:291`) — Sage's mechanism to replace every occurrence of a formal function with the result of a Python callback that receives the function's arguments. This is how `reduce_conv_in_action` rewrites every `Conv(g, n_arg)` atom in one pass.
- `dirac_delta`, `heaviside` — `dirac_delta(τ)` is used to peel the local part of a noise cumulant via `K.coefficient(dirac_delta(τ))` (`field_theory.py:463`); `heaviside(t)` is special-cased in `fourier_transform` (causal kernels).
- `integrate(integrand, t, a, b, algorithm=...)` (`field_theory.py:69,88`) — symbolic integration. The `algorithm` keyword selects the backend: `'sympy'` (tried first — handles `positive=True` parameter assumptions automatically) or `'maxima'` (Sage's default integrator). `oo` is Sage's `∞`, `I` is `√−1`, `pi` is `π`.
- `expr.expand()`, `.operands()`, `.operator()`, `.coefficient(v, d)`, `.degree(v)`, `.simplify_full()`, `.free_variables()`, `.subs(dict)`, `.has(...)`, `.is_zero()`, `.is_trivial_zero()`, `.is_numeric()` — the symbolic-expression methods the classifier and verifier lean on. Two are worth flagging:
  - **`.operator()` name-based dispatch.** Sage represents a sum not with `operator.add` but with its own `add_vararg` callable. So the code never does `op is operator.add`; it does `'add' in getattr(op, '__name__', '').lower()` (`field_theory.py:267`, `convolution.py:254`). Getting this wrong silently drops terms.
  - **`.is_trivial_zero()` vs `== 0`.** Under Sage, `expr != 0` can return *False* when `expr` contains unbound real parameters with no positivity assumptions (e.g. `rho` with `domain='real'`) — Sage cannot decide the sign and refuses. `.is_trivial_zero()` is the correct *syntactic* zero test (True iff the expression is literally zero) and is used in the cumulant injector (`field_theory.py:490`).

```python
# serialize.py:23
from sage.all import save as sage_save, load as sage_load, version as sage_version
```

- `sage_save(obj, path)` / `sage_load(path)` — Sage's binary object (`.sobj`) serialization. It pickles arbitrary Sage objects (polynomial rings, symbolic expressions, matrices) that plain `pickle`/`json` cannot. `serialize.py` uses it to persist `ft._R`, `ft._S_raw`, `ft._by_tp`, and propagator matrices. `sage_save` appends the `.sobj` extension automatically (`serialize.py:150-151`).
- `sage_version()` — recorded in metadata for reproducibility (`serialize.py:87`).
- Note: `reload_model` (`serialize.py:237`) `exec`s a model `.py` file in a fresh namespace dict to re-import the model variable. The comment says "Use SageMath's load() which handles .py files" but the code actually uses `exec(compile(code, full_path, 'exec'), ns)` — a plain-Python exec, not `sage_load`. (See Open Questions — the comment is stale.)

```python
# vertices.py:14
from sage.all import SR
```

- Only `SR` is needed here: `SR(coeff)` to re-wrap a polynomial coefficient as a standalone symbolic expression, and `SR(coeff).coefficient(sym)` to pull a placeholder symbol's prefactor out of a source coefficient (`vertices.py:726`).

### 3.2 IPython (`IPython.display`)

**What it is.** IPython is the interactive Python kernel underneath Jupyter. `IPython.display` provides helpers to render rich output (LaTeX, HTML, images) in a notebook cell.

**How this code uses it.** `field_theory.py:30` `from IPython.display import display, Math as _Math`. The single helper `_show(expr)` (`field_theory.py:95`) does `display(_Math(latex(expr)))` — i.e. takes a Sage `latex(...)` string and renders it as typeset math in a notebook. Used by `summary()` and `sanity_check()` to pretty-print sectors. The `_ConvIPBase`/`Conv`/`IP`/`_DisplaySum` display classes (`field_theory.py:104-178`) are **purely cosmetic** — they implement `_latex_()` so a notebook can display convolution/inner-product expressions nicely. They are *not* the computational `Conv` (that lives in `convolution.py`); they are a display-only mirror.

### 3.3 Python standard library

- `warnings` (`field_theory.py:24`) — non-fatal advisories: order-≥3 smooth cumulant residual dropped (`field_theory.py:551`); MF sector vanishes only numerically (`field_theory.py:648`).
- `itertools.product` / `itertools.permutations` — `_iter_multi_indices` (`field_theory.py:219`) enumerates multi-derivative indices via `product`; the cumulant injector permutes leg-field tuples via `permutations` (`field_theory.py:424`). The comment at `field_theory.py:210-215` notes `product` is used **deliberately instead of a self-recursive generator** so the code survives Jupyter `%autoreload` (a self-recursive generator loses its self-reference when module globals are hot-swapped).
- `operator` (`field_theory.py:253`, imported as `_op`) — imported but, per the comments, *not* used for identity comparison (because Sage uses `add_vararg`); kept for clarity.
- `json`, `os`, `datetime` (`serialize.py:19-21`) — `metadata.json` I/O, directory creation, timestamp.
- `functools.reduce` (`convolution.py:213`) — fold a list of factors into a product in the `Conv` mul-rule.

### 3.4 multiprocessing (indirect — a *constraint*, not a call)

This subsystem never imports `multiprocessing`, but it is engineered around it. The Phase J time-domain integrator (downstream) farms diagram evaluation across a `multiprocessing.Pool`, which **pickles** the vertex/source objects to worker processes. This forces three design decisions in `vertices.py`:

- `_NamespaceBoundKernel` (`vertices.py:25`) is a **module-level** class (not a closure) so pickle can locate it by its fully-qualified name `msrjd.core.vertices._NamespaceBoundKernel`.
- It **pre-evaluates** the user's kernel function once against the namespace and caches the *resulting SR expression* (`vertices.py:56`), because the full `_Namespace` is *unpicklable* — it contains inner closures like `_entity_axis_sizes` defined inside `FieldTheory._build_namespace`.
- The `VertexType`/`SourceType` classes implement explicit `__getstate__`/`__setstate__` because they use `__slots__` (no `__dict__`), and a subclass must reference the *base* class's `__slots__` explicitly when pickling, not `self.__slots__` (which on a subclass instance resolves only to the subclass's own slots — see the comment at `vertices.py:235-238`).

**Memory note (macOS fork hazard).** The project memory records that fork-based multiprocessing crashed the user's machine; the spatial path is serial-only and the temporal path is fork-guarded. The picklability machinery in this subsystem predates and is independent of that guard, but it is why these objects *can* cross a process boundary at all.

---

## 4. Components

Exhaustive walkthrough of every significant function/class/dataclass. Grouped by file.

### 4.1 `field_theory.py`

#### `fourier_transform(f, t, s)` — `field_theory.py:33`
Signature: `fourier_transform(f, t, s) -> SR`. Angular-frequency convention `F(s) = ∫ f(t) e^{-ist} dt` (no `2π` in the exponent, so `δ(t)→1`, `δ'(t)→iω`).
Steps: (1) coerce `f = SR(f)`; (2) detect the **causal** case `f.has(heaviside(t))` — if present, substitute `heaviside(t)→1` and integrate over `[0,∞)` instead of `(−∞,∞)` (this avoids Maxima demanding an explicit sign on `ω`, which it cannot decide where the heaviside cuts); (3) try `integrate(..., algorithm='sympy')` then `'maxima'`, returning the first that succeeds; (4) fall back to the unevaluated integral.
Returns: an `SR` frequency-domain expression. *Note:* this is a utility used by the propagator pipeline (`pipeline/_propagator.py:29`, `pipeline/theory_compiler.py:1510`), not by `expand()` itself.

#### `inverse_fourier_transform(F, s, t)` — `field_theory.py:75`
Signature: `inverse_fourier_transform(F, s, t) -> SR`. Convention `f(t) = (1/2π) ∫ F(s) e^{+ist} ds`, paired with `fourier_transform` (FT uses `e^{-ist}`, IFT uses `e^{+ist}/2π`). Tries sympy then maxima backend, divides by `2π`, falls back to unevaluated integral.

#### `_show(expr)` — `field_theory.py:95`
Renders a Sage expression (or a display `Conv`/`IP` object) as typeset LaTeX in a notebook: `display(_Math(latex(expr)))`.

#### Display classes `_ConvIPBase`, `Conv`, `IP`, `_Neg`, `_DisplaySum` — `field_theory.py:104-178`
Cosmetic-only algebra for pretty-printing. `Conv(kappa, f)` renders `(κ ∗ f)`; `IP(a, b)` renders `aᵀ b`. They overload `+`, `-`, unary `-` to build `_DisplaySum` chains whose `_latex_()` joins the parts. **Do not confuse with `convolution.Conv`** — these never participate in computation.

#### `_Namespace` — `field_theory.py:185`
An empty class. Instances are populated by `_build_namespace` with attributes for every field/parameter/kernel/operator/function plus the bookkeeping lists (`_all_field_sr_vars`, `_ring_var_names`, `_deriv_rename_subs`, etc.). It is the object every model lambda receives as `ns`.

#### `_poly_taylor(coeffs, x)` — `field_theory.py:193`
Builds `Σ_n SR(coeffs[n])·(1/n!)·x^n` as an SR expression. `coeffs[n] = f^{(n)}(background)`. Used by the legacy `taylor_coeffs` path (manually supplied derivative coefficients) in `_build_namespace`.

#### `_iter_multi_indices(n_args, max_total)` — `field_theory.py:202`
Generator yielding every multi-index `(k_1,…,k_{n_args})` of non-negative ints with `Σk_j ≤ max_total`. For `n_args=0` yields the empty tuple. For `n_args=1` collapses to `(0,),(1,),…,(max_total,)`. Implemented with `itertools.product` (autoreload-safe). Drives the formal-function rename enumeration.

#### `_multi_index_suffix(multi_idx)` — `field_theory.py:225`
Encodes a multi-index as the symbol-name suffix: `''.join(str(k) for k in multi_idx)`. Single-arg → just the derivative order (`f1`, `f2`); multi-arg → concatenated orders (`f21`).

#### `_sr_to_ring(sr_expr, R, ring_var_names)` — `field_theory.py:241`
Signature: `_sr_to_ring(sr_expr, R, ring_var_names: list) -> R element`. Converts an `SR` expression that is polynomial in the named field variables into an element of `R = PolynomialRing(SR, ring_var_names)`.
Steps: (1) `expanded = SR(sr_expr).expand()`; if zero, return `R.zero()`. (2) Split into summands: use `.operands()` for a sum (detected via name-based `'add' in op.__name__`), else wrap the single term in a list. **This sum/single-term distinction is explicit so symbolic coefficients like `τ·Dt` are never lost** (a naïve `.operands()` on a single product would iterate its *factors*, not its summands). (3) For each summand, for each ring generator `v`, read `deg = int(coeff.degree(v))` and strip it off via `coeff = coeff.coefficient(v, deg)`, accumulating the remaining symbolic coefficient. (4) Add `SR(coeff)·R.monomial(*exponents)` to the result.
Returns: the polynomial-ring representation, with parameters/kernels living in the SR coefficients and field generators in the monomials.

#### `_build_cumulant_action(ns, model)` — `field_theory.py:293`
Signature: `_build_cumulant_action(ns, model) -> SR`. Builds the `-W_m[m̃]` correlated-noise cumulant series from `model['correlated_noises']`. Returns `SR(0)` (a true no-op) when the model has no such block — so existing models are untouched.
Also initializes `ns._cumulant_kernels = {}` (the registry the downstream extractor reads).
Steps per declared noise process:
1. Resolve the optional `physical_field` (raise if declared but missing from the namespace).
2. Resolve per-leg response fields: prefer the new `response_legs` dict (keyed by cumulant order → list of field names, supports cross-field cumulants), else the legacy single `response_field` broadcast to every leg.
3. Skip `κ^{(1)}` (the mean — already absorbed into the saddle).
4. For each cumulant order `n≥2`: build `n−1` relative-time symbols `_tau_<noise>_<k>`; enumerate **distinct permutations** of the leg-field tuple (for heterogeneous fields like `[xt, yt]` the cumulant sums over both `(xt,yt)` and `(yt,xt)` orderings; the framework reconstructs the non-canonical ones here, while homogeneous fields have only one permutation since the index-product already covers index orderings).
5. For each leg-index tuple, **evaluate the user kernel** `K = SR(kernel_fn(ns, *idx_tuple, *tau_syms))` (with a back-compat path for old `(ns,i,j,tau)` order-2 signatures).
6. **Peel off the local (delta) part:** iteratively `c_local = c_local.coefficient(dirac_delta(τ_k))` for each τ. After `n−1` steps `c_local` is the multiplier of the full delta product `Πδ(τ_k)`. The residual `K_residual = K − c_local·Πδ(τ_k)`, `simplify_full`'d.
7. **Inject the local contribution:** if `c_local` is not trivially zero, add `−(1/n!)·c_local·Π(leg vars)` directly to `S_cum` (the time integrals all collapsed). Uses `.is_trivial_zero()` (not `!=0`) for the reason given in §3.1.
8. **Inject the smooth residual** (order 2 only): create a placeholder symbol `z_kappa_<noise>_2_<legA>_<i+1>_<legB>_<j+1>` with a LaTeX `κ^{(2)}_{…}`, add `−(1/n!)·sym·(leg vars)` to `S_cum`, and register the kernel in `ns._cumulant_kernels[(noise, order, tuple(zip(leg_fields, idx_tuple)))] = {symbol, kernel_fn, legs, leg_fields, tau_var}`. For order ≥3 with a smooth residual, **warn and drop** the smooth part (the local part, if any, is still injected).
Returns: the accumulated `S_cum` SR expression. Side effect: populates `ns._cumulant_kernels`.

#### `_collect_bigrade(poly, n_tilde)` — `field_theory.py:570`
Signature: `_collect_bigrade(poly, n_tilde: int) -> dict`. Splits a polynomial-ring element into `{(n_tilde, n_phys): poly}` sectors. Iterates `poly.dict().items()` (exponent-vector → coefficient); `n_t = sum(exp_vec[:n_tilde])`, `n_p = sum(exp_vec[n_tilde:])`; accumulates `SR(coeff)·R.monomial(*exp_vec)` into the keyed sector. Ring generators are ordered tilde-first, so the split index is exactly `n_tilde`.

#### `_verify_and_zero_mf_sector(by_tp, mf_subs, spec_subs, ns, R, model, mf_sector_keys, num_tol)` — `field_theory.py:585`
Applies the MF background substitutions to the saddle sectors `{(0,0),(1,0),(0,1)}` and **verifies they vanish**, then replaces each with `R.zero()` in place.
Two-tier check per monomial coefficient:
1. **Symbolic:** `coeff.subs(mf_subs)` (then `spec_subs` if present); if `.simplify_full() == 0`, pass.
2. **Numerical fallback** (only if symbolic fails): `_mf_numerical_residual(...)`; pass if `|residual| < num_tol` (default `1e-9`), and record a *soft pass* (warned, not failed).
Any monomial that passes neither is a **failure**: raises `AssertionError` listing the offending bigrade, the post-subs symbolic form, and the numerical residual. The error message says: "Either the MF solver is wrong or the action's bigrade-≤1 sector is not the saddle-eq sector." This is the structural test that catches a miswired MF solver or a mis-specified action. After all checks, each key in `mf_sector_keys` is set to `R.zero()`.

#### `_mf_numerical_residual(expr, ns, model)` — `field_theory.py:662`
Numerically evaluates an SR expression at the MF saddle for a representative parameter point. Builds a `fundamental` point (`model['fundamental_defaults']` or `_default_fundamental_point`), prefers the DAE solver (`pipeline._mean_field_dae.solve_mean_field_dae_compat`) when the model declares `equations` (the legacy iterative solver can't handle self-referential implicit equations like `xstar = −ε·xstar³`), else falls back to `pipeline._mean_field.solve_mean_field`. Substitutes the solved `num_params`, binds any remaining free symbols to `1.0`, returns `abs(complex(bound))` (or `abs(float(...))`). Returns `None` if it can't build the residual (no solver, incomplete fundamental, etc.).

#### `_default_fundamental_point(ns, model)` — `field_theory.py:730`
Builds a `fundamental` parameter dict for the numerical fallback. Prefers each parameter spec's `default` value (the theory ships these specifically because they admit a well-behaved MF solution), falling back to all-ones only when no default is declared. The docstring warns that all-ones is a poor universal fallback for nonlinear closures (e.g. `φ = a v²` with `Em=a=w=1` gives `2v²−v+1=0`, no real root). Vector/matrix params are sized by population.

#### `_MFProxyForSolver` — `field_theory.py:765`
A minimal FieldTheory-shaped proxy (`__slots__ = ('_ns','taylor_order','model')`) that `solve_mean_field` can consume *during* MF-sector verification. Needed because `self` isn't fully populated yet when `expand()` calls the verifier, so the real FieldTheory can't be passed.

#### class `FieldTheory` — `field_theory.py:784`
The central class. Constructor `__init__(self, model: dict, taylor_order: int = 4)` (`field_theory.py:800`) stores the model and order, and initializes private state to `None`: `_ns, _R, _n_tilde, _S_raw, _by_tp, _mf_sector_raw`.

##### `FieldTheory.expand(self) -> None` — `field_theory.py:814`
The main orchestration method. Step by step:
1. `ns, R, n_tilde = self._build_namespace()`; stash on `self`.
2. `S_sr = SR(self.model['action'](ns))` — evaluate the model's action lambda; result is one big SR expression.
3. **Conv reduction:** `reduce_conv_in_action(S_sr, fluct_vars, taylor_order, attachments_out=attachments)` resolves every `Conv(...)` atom (passing `taylor_order` so nonlinear Conv arguments get pre-Taylor'd at the same order). Records `ns._kernel_attachments = attachments`. Guarded by `try/except (AttributeError, TypeError)` → identity if no Conv atoms.
4. **Cumulant injection:** `S_sr = S_sr + _build_cumulant_action(ns, self.model)`.
5. **Multivariate Taylor:** if there are field vars, `S_sr = taylor(S_sr, *[(v,0) for v in ns._all_field_sr_vars], self.taylor_order)`.
6. **Derivative rename:** `S_sr = S_sr.subs(ns._deriv_rename_subs)` (e.g. `D[0](phi_1)(0)→phi1_1`).
7. **Specializations:** `S_sr = S_sr.subs(self.model['specializations'](ns))` if present (pure closure renames — applied to every sector, safe).
8. **Coerce + classify BEFORE MF subs:** `S_poly = _sr_to_ring(S_sr.expand(), R, ns._ring_var_names)`; `by_tp = _collect_bigrade(S_poly, n_tilde)`. **Critical ordering** (see the long comment `field_theory.py:913-922`): MF saddle substitutions like `vstar → (E + Σwg·nstar)/(1+τ·nstar)` would otherwise inject saddle-eq algebra into the `(1,1)` propagator kernel and the ≥2 vertices, where `vstar`/`nstar` must remain free symbolic parameters bound numerically later.
9. Stash `self._mf_sector_raw` = the *pre-zero* MF sectors (diagnostics).
10. **MF verify + zero:** pick `mf_bg_conditions_action` if present, else `mf_bg_conditions`; build `mf_subs`; re-build `spec_subs` (to resolve any `phi`-Taylor tokens the `vstar` RHS reintroduced — the mf_bg lambda builds RHSes *before* specializations runs); call `_verify_and_zero_mf_sector(...)`.
11. **Rebuild** `S_raw` from the (now MF-zeroed) `by_tp`; store `self._S_raw`, `self._by_tp`.

##### `FieldTheory.sanity_check(self, verbose=True) -> bool` — `field_theory.py:966`
Verifies the three zero sectors `(0,0)` constant, `(1,0)` tadpole, `(0,1)` EOM residual are `R.zero()`. Prints a PASS/FAIL table (always prints a FAIL, even when `verbose=False`, so failures are never silent). Returns the conjunction.

##### Accessors (all call `_require_expanded` first):
- `summary(self)` — `field_theory.py:999` — prints and `_show`s every non-zero sector with its label.
- `free_action(self)` — `field_theory.py:1010` — returns the `(1,1)` sector poly (the bilinear kernel), or `R.zero()`.
- `noise_kernel(self) -> dict` — `field_theory.py:1015` — all `(≥2, 0)` non-zero sectors.
- `vertices(self) -> dict` — `field_theory.py:1021` — all total-degree-≥3 non-zero sectors.
- `sectors(self) -> dict` — `field_theory.py:1027` — full non-zero bigrade dict.
- `ring(self)` — `field_theory.py:1033` — returns `self._R`.

##### `FieldTheory._build_namespace(self)` — `field_theory.py:1041`
Constructs `(ns, R, n_tilde)`. The longest method; step by step:
1. Bind every index set in `model['index_sets']` onto `ns`; `primary_idx` = the first index set.
2. **Population-aware sizing:** for each entry in `model['populations']`, build `[0..size-1]` index lists and bind under both `<name>` and `pop_<name>` (lets action text write `for i in E`). Legacy theories (empty `populations`) fall back to the single flat `pop`. Defines inner helpers `_field_indices(fspec)` and `_entity_axis_sizes(spec)` (resolves scalar `[]` / vector `[N]` / matrix `[N,N]` per `indexed_by`/`indexed`). Stashes `ns._populations, _pop_size, _pop_local_idx, _entity_axis_sizes`.
3. **Field variables → SR symbols.** For each `response_fields` / `physical_fields` spec: if indexed, build `[SR.var(f"{fname}{i+1}", latex_name=...)]`; else a scalar `SR.var(fname)`. Accumulate `tilde_sr_vars`/`phys_sr_vars` and names. Stash `ns._tilde_sr_vars, _phys_sr_vars, _all_field_sr_vars`.
4. **Polynomial ring:** `ring_var_names = tilde_names + phys_names` (tilde first!), `n_tilde = len(tilde_names)`, `R = PolynomialRing(SR, ring_var_names)`.
5. **Parameters → SR symbols.** Scalar params get one symbol (with `domain` assumption if declared); vector/matrix params get a flat list sized by `_entity_axis_sizes`. (Matrix 2D structure is installed later by `mf_substitutions`.)
6. **Kernels and operators → SR symbols.** Scalar kernel → one symbol named by `sage_name` (default `name`); vector → `z_<name>_<i+1>`; matrix → `z_<name>_<i+1>_<j+1>` list-of-lists. Operators → one symbol each (`sage_name` default). Sets `ns._operator_ir = bool(model.get('operator_ir', False))` (spatial-v2 authoring flag). Defines `ns.delta_D = SR.var('z_delta', …)`, `ns.delta_Dp = SR.var('z_delta_p', …)` (the `z`-prefix forces canonical factor ordering).
7. **Nonlinear functions.** For each `model['functions']` spec, two paths:
   - **`'expression'` (auto-expand):** the model supplies `expression(i, x_1,…,x_n)` returning an SR expression in formal arguments. For every multi-index `α` with `|α|≤order`, compute `∂^α expression` at the all-zero point and register a rename `ns._deriv_rename_subs[deriv_at_0] = SR.var('<prefix><suffix>_<i+1>', latex_name=...)`. Skips purely-numeric derivatives (`is_numeric()`). Installs a callable `ns.<fname>` that substitutes call-site args into the stored expression. `_deriv_latex` produces nice LaTeX (`f'`, `f''`, `f^{(k)}`, or multi-index superscripts).
   - **`'taylor_coeffs'` (legacy):** the model supplies a list of derivative coefficients; install `_poly_taylor(coeffs[:order+1], x)` as the callable.
8. Set `ns._taylor_order = self.taylor_order`.
9. **MF substitutions:** for each `model['mf_substitutions']`, `setattr(ns, sub['name'], sub['value'](ns))` (computed once at build).
Returns `(ns, R, n_tilde)`.

##### `FieldTheory._require_expanded(self)` — `field_theory.py:1376`
Raises `RuntimeError("Call expand() first.")` if `_by_tp is None`.

##### `FieldTheory._sector_label(n_t, n_p)` (static) — `field_theory.py:1380`
Maps a bigrade to a human label: `(1,1)`→"free action", `(≥2,0)`→"noise kernel", total 1→"tadpole / background", total ≥3→"vertex (order N)", else generic.

### 4.2 `vertices.py`

#### class `_NamespaceBoundKernel` — `vertices.py:25`
Picklable wrapper around a user kernel function. `__slots__ = ('_expr','_legs','_tau_var')`. Constructor `__init__(self, kernel_fn, ns, legs, tau_var)` evaluates `self._expr = SR(kernel_fn(ns, *legs, tau_var))` **once** against the full namespace and caches the SR result; stores `legs` (tuple) and `tau_var`. `__call__(self, *args)` expects `len(legs)` integer legs followed by a τ value; if the τ value *is* the cached `_tau_var` it returns `_expr` unchanged, else `_expr.subs({_tau_var: tau_value})`. The whole class exists so noise-source objects survive a `multiprocessing.Pool` round-trip without carrying the unpicklable namespace (see §3.4).

#### class `VertexType` — `vertices.py:80`
A dataclass-like record for **one monomial from an interacting-action sector (total degree ≥ 3)**. `__slots__ = ('coefficient','response_legs','physical_legs','bigrade')`.
- `coefficient`: SR expression (coupling × combinatorial prefactor).
- `response_legs`: list of `(field_base_name, population_index)`, repeated per exponent multiplicity.
- `physical_legs`: same, for physical generators.
- `bigrade`: `(n_tilde, n_phys)` tuple.
Properties: `in_degree` = number of physical legs, `out_degree` = number of response legs, `total_degree` = sum. Implements `__getstate__`/`__setstate__` for `__slots__` pickling.

#### class `ConvVertexType(VertexType)` — `vertices.py:134`
A conductance-style interaction vertex: `n_phys ≥ 1`, where one or more **physical** legs sit at independent times linked to the vertex's main time by a synaptic kernel `g(τ)` (from a `Conv(g, field)` in the original action). Adds `__slots__ = ('kernel_attachments',)`. Each attachment dict has `'symbol'` (the kernel SR symbol still in the coefficient), `'leg'` (the physical leg-tuple it attaches to), `'leg_index'` (0-based index within `physical_legs`; first match if duplicated), `'kernel_td_fn'` (callable `(tau)->SR`, the time-domain `g(τ)` bound to the kernel's indices, or `None` if only a freq image exists). Parallels `NoiseSourceType` for physical legs. Pickle state chains through `VertexType`.

#### class `SourceType` — `vertices.py:216`
A record for **one monomial from a noise-kernel sector (`n_tilde ≥ 2`, `n_phys = 0`)**. `__slots__ = ('coefficient','response_legs','bigrade')`. Property `out_degree`. Note the pickle methods reference `SourceType.__slots__` *explicitly* (not `self.__slots__`) so a subclass instance still pickles the base slots (comment `vertices.py:235`).

#### class `NoiseSourceType(SourceType)` — `vertices.py:256`
A source vertex backed by a **non-local cumulant kernel** `κ^{(n)}(τ_1,…,τ_{n-1})` — the response legs sit at independent times. Arises from the `-W_m[m̃]` series. Adds `__slots__ = ('cumulant_specs',)`. Each cumulant spec dict has `'symbol'`, `'kernel_fn'` (a `_NamespaceBoundKernel`), `'legs'`, `'leg_fields'`, `'tau_var'`, `'sign'` (the prefactor of `symbol` in the coefficient, typically `−1/2`), `'noise'`, `'order'`. Locally-correlated (delta) cumulants stay plain `SourceType` (their τ-integral collapsed inside `_build_cumulant_action`).

#### `_parse_field_name(ring_var_name)` — `vertices.py:327`
Parses a ring var name like `nt1`, `dn2`, `vt12` into `(base_name, population_index)`. Strategy: walk back from the end over trailing digits; everything before is the base, the trailing digits are the 1-based population index. If no trailing digits *or* the whole name is digits, returns `(full_name, 0)`.

#### `decompose_sector(sector_poly, n_tilde, ring_var_names)` — `vertices.py:349`
Signature: returns `list of (VertexType or SourceType)`. For each `(exp_vec, coeff)` in `sector_poly.dict()`: for each generator with non-zero exponent, parse its name to a leg-tuple and `extend` either `resp_legs` (if generator index `< n_tilde`) or `phys_legs` (else), repeated by the exponent. If `n_phys == 0` emit `SourceType(SR(coeff), resp_legs, bigrade)`, else `VertexType(SR(coeff), resp_legs, phys_legs, bigrade)`. This is the bridge from "polynomial sector" to "typed leg records."

#### `_kernel_symbol_to_pop_indices(ksym)` — `vertices.py:398`
Parses a kernel SR symbol `z_g_1_2` into `(1,2)` (matrix), `z_g_1`→`(1,)` (vector), scalar→`()`. Splits on `_`, reads trailing all-digit tokens greedily, reverses. Used to resolve which physical leg a kernel attaches to (the *last* index of a matrix kernel = the "incoming" leg).

#### `_resolve_kernel_attachment_to_leg(ksym, attached_fluct_set, physical_legs, ring_var_names, n_tilde)` — `vertices.py:427`
Picks which physical leg a kernel symbol attaches to. Strategy: (1) translate each attached fluct var into a leg-tuple `(base, idx)` and intersect with `physical_legs` — return the first match; (2) failing that, **index match**: take the kernel's *last* pop-index and return the first physical leg whose pop-idx equals it (`z_g_1_2` matches a leg with pop-idx 2). Returns `(leg_tuple, leg_index)` or `None`.

#### `_flatten_kernel_symbols(ns, model)` — `vertices.py:478`
Walks `model['kernels']`, fetches each kernel object from `ns`, and flattens scalar/vector/matrix into one list of SR symbols. Used to scope the kernel-detection scan in vertex extraction.

#### `extract_vertex_types(ft)` — `vertices.py:500`
Signature: `extract_vertex_types(ft) -> list[VertexType]` (with `ConvVertexType` mixed in). The main vertex extractor.
Steps: (1) `ft._require_expanded()`. (2) Pull `attachments = ns._kernel_attachments`; if present, gather `kernel_symbols` and pre-build the `{ksym: SR_in_tau}` map from `model['kernel_td_image']` (using a `_conv_tau_placeholder`). (3) For each total-degree-≥3 sector with `n_phys ≥ 1` (skip pure-noise `n_phys==0`), `decompose_sector` it. (4) For each `VertexType` monomial: if no attachments → keep plain. Else `kernel_attachments_in_coefficient(...)` scans the coefficient for kernel symbols; if none detected → keep plain. Else for each detected symbol, `_resolve_kernel_attachment_to_leg(...)` to a specific physical leg; build a per-call `_td_fn(tau)` closure that substitutes the placeholder τ; collect `{symbol, leg, leg_index, kernel_td_fn}`. (5) If any attachments resolved → emit a `ConvVertexType`, else plain `VertexType`. Unresolvable kernel symbols are *left in the coefficient* so downstream diagnostics flag the surviving symbol rather than silently treating it as zero.

#### `extract_source_types(ft)` — `vertices.py:629`
Signature: `extract_source_types(ft) -> list[SourceType]` (with `NoiseSourceType` mixed in). The main source extractor.
Steps: (1) `ft._require_expanded()`. (2) Read `ns._cumulant_kernels` and `model['correlated_noises']`; build `noise_resp_field` map. (3) For each `(≥2,0)` noise sector, `decompose_sector` it. (4) For each `SourceType` monomial, match its response-leg multiset (sorted) against every registered cumulant spec. The leg key has two shapes: legacy `(int,…)` (all legs on the noise's single `response_field`) or cross-field `((field,idx),…)`; build `spec_legs` accordingly (note the `+1` to convert 0-based registry indices to 1-based leg-tuples). If `spec_legs == m_leg_multiset` **and** the placeholder symbol's coefficient `sign = m.coefficient.coefficient(sym)` is non-zero, the spec contributes. (5) Build a `_NamespaceBoundKernel` and append a matched-spec dict carrying `{symbol, kernel_fn, legs, leg_fields, tau_var, sign, noise, order}`. (6) If any specs matched → emit `NoiseSourceType`, else plain `SourceType`.

#### `available_degrees(vertex_types, source_types)` — `vertices.py:777`
Returns `(interaction_degrees, source_degrees)`: a set of `(in_degree, out_degree)` pairs from the vertices, and a set of `out_degree` ints from the sources. This is what the diagram enumerator (`degree_scan.py`) consumes to know which vertex/source "shapes" exist.

### 4.3 `convolution.py`

#### `Conv` — `convolution.py:70`
`Conv = _sr_function('Conv', nargs=2)` — the formal two-argument Sage function. By convention `Conv(kernel, field)`. Never auto-evaluates; stays a formal symbol so each downstream consumer applies its own reduction. `nargs=2` makes Sage reject `Conv(g)` / `Conv(g,n,extra)`.

#### `is_convolution(expr)` — `convolution.py:73`
True iff `expr` is a `Conv(...,...)` atom (`str(expr.operator()) == 'Conv'`). Tolerant of non-symbolic inputs (returns False).

#### `kernel_of(expr)` / `field_of(expr)` — `convolution.py:90,100`
Return the first / second operand of a `Conv` atom; raise `ValueError` if not a `Conv`.

#### `reduce_conv_in_action(expr, fluct_vars, normalized=True, taylor_order=None, attachments_out=None)` — `convolution.py:110`
The reducer. Resolves every `Conv(g, n_arg)` atom by `expr.substitute_function(Conv, _reduce_arg)`. The inner `_reduce_arg(g, n_arg)` applies, in order:
- **Rule 0 (pre-Taylor):** when `taylor_order` given, `taylor(n_arg, *(v,0) for v in fluct_set, taylor_order)` — turns a nonlinear inner field `h(v,n)` into a polynomial so linearity can distribute. No-op for already-polynomial args.
- **Rule 1 (linearity):** `Conv(g, a+b) → Conv(g,a)+Conv(g,b)` (recurse over `.operands()` of a sum).
- **Rule 2 (pull constants):** `Conv(g, c·X) → c·Conv(g, X)` for time-constant `c` (a factor with no fluct vars). If *all* factors are constant → rule 3.
- **Rule 3 (DC reduction):** `Conv(g, c) → ĝ(0)·c = c` for normalized kernels (leaf with no fluct dependence).
- **Rule 4 (defer to FT):** `Conv(g, X) → g·X` for genuinely fluctuating `X`; the kernel symbol `g` is later replaced by `ĝ(ω)`. Records `attachments_out[g] |= {attached fluct vars}` (set-union accumulates across reuse) so vertex extraction can recover which leg each surviving kernel attaches to.
`_has_fluct(e)` tests whether an expression's variables intersect `fluct_set`. The docstring (`convolution.py:144-170`) works the conductance and cubic examples in full.

#### `kernel_attachments_in_coefficient(coeff, attachments, kernel_symbols=None)` — `convolution.py:294`
For a vertex coefficient, returns `{kernel_symbol: leg_var}` for each kernel symbol present in `coeff` (scanned over `kernel_symbols`, default all attachment keys). Single attached fluct → the var itself; multiple → a `frozenset` (caller disambiguates by index). Kernel symbols not in `coeff` are omitted.

`__all__` exports `Conv, is_convolution, kernel_of, field_of, reduce_conv_in_action, kernel_attachments_in_coefficient`.

### 4.4 `serialize.py`

#### `_strip_callables(spec_list)` — `serialize.py:28`
Returns a copy of a list of spec dicts with all callable values removed, keeping only JSON-serializable fields. (Model specs mix data and lambdas; only data is JSON-able.)

#### `_jsonable_index_sets(index_sets)` — `serialize.py:45`
Coerces index-set values to plain Python lists (from `range`, etc.).

#### `save_theory(path, ft, propagator_data=None, stationarity=True, model_file=None, model_var_name=None)` — `serialize.py:52`
Saves an expanded theory to a directory. Calls `ft._require_expanded()`, `os.makedirs(path, exist_ok=True)`. Writes two artifacts:
- **`metadata.json`** — plain-Python metadata: format/sage version, timestamp, model identity (`model_name`, `model_file`, `model_var_name`), expansion info (`taylor_order`, `n_tilde`, `ring_var_names`), field/param/kernel/operator specs (callables stripped), stationarity/noise structure, propagator summary, and `nonzero_sectors` (sorted list of bigrade keys for quick inspection).
- **`symbolic_data.sobj`** — Sage binary dump of `{R, S_raw, by_tp, n_tilde, K_ft, G_ft, adj_ft, D_omega, pole_vals, C_mats, G_t}` via `sage_save`.
The model dict itself (lambdas) is **not** serialized — instead the metadata records the model file path + variable name for re-import.

#### `load_theory(path)` — `serialize.py:156`
Returns `(meta, data)`: `meta` = `json.load(metadata.json)`, `data` = `sage_load(symbolic_data.sobj)`. Raises `FileNotFoundError` if either file is missing.

#### `reload_model(meta, project_root=None)` — `serialize.py:192`
Re-imports the model dict from the stored `model_file`/`model_var_name`. Resolves the path against `project_root` (or cwd), `exec`s the file in a fresh namespace dict, returns `ns[model_var]`. Raises `ValueError` (missing identity), `FileNotFoundError` (missing file), or `AttributeError` (variable not found — lists available names).

---

## 5. Data structures

### 5.1 The model dict (input)
Keys observed across `field_theory.py` and the worked model (`hawkes_quad_expg.py`):

| key | type | meaning |
|---|---|---|
| `name` | str | human-readable model name |
| `index_sets` | dict[str, list] | named index ranges; first one is `primary_idx` |
| `populations` | list[dict] (optional) | per-population `{name, size}` for heterogeneous theories |
| `response_fields` | list[dict] | each `{name, indexed, latex, ...}` — the tilde fields |
| `physical_fields` | list[dict] | same shape — the fluctuation fields |
| `parameters` | list[dict] | `{name, indexed/indexed_by, domain, default, mean_field, ...}` |
| `kernels` | list[dict] | `{name, sage_name, latex_name, indexed/indexed_by}` |
| `operators` | list[dict] | `{name, sage_name, latex_name}` (e.g. `Dt`) |
| `functions` | list[dict] | nonlinear fns: `{name, indexed, deriv_prefix, n_args, latex, expression OR taylor_coeffs}` |
| `mf_substitutions` | list[dict] | `{name, value: lambda ns -> SR}` bound onto `ns` at build time |
| `mf_bg_conditions` / `mf_bg_conditions_action` | lambda ns → dict | saddle substitutions for sector verification |
| `specializations` | lambda ns → dict | rename abstract derivative symbols to concrete values |
| `correlated_noises` | dict (optional) | colored-noise cumulant declarations |
| `kernel_ft_image` | lambda ns, ω → dict | `{kernel_sym: ĝ(ω)}` (Fourier image) |
| `kernel_td_image` | lambda ns, τ → dict | `{kernel_sym: g(τ)}` (time-domain image, for ConvVertexType) |
| `action` | lambda ns → SR | **the action** |
| `operator_ir` | bool (optional) | spatial-v2 operator-IR authoring mode flag |
| `equations` / `fundamental_defaults` | misc | drive the numerical MF residual fallback |

### 5.2 `_Namespace` (the `ns` object)
Built by `_build_namespace`. Carries (non-exhaustive): every field/param/kernel/operator/function under its model name; `_tilde_sr_vars`, `_phys_sr_vars`, `_all_field_sr_vars`; `_ring_var_names`; `_deriv_rename_subs`; `_taylor_order`; `_populations`, `_pop_size`, `_pop_local_idx`, `_entity_axis_sizes`; `_operator_ir`; `delta_D`, `delta_Dp`. After `expand()`: `_kernel_attachments` (from Conv reduction) and `_cumulant_kernels` (from cumulant injection). **Unpicklable** (inner closures).

### 5.3 `ft._by_tp` (the central output)
`dict[(n_tilde:int, n_phys:int) → PolynomialRing element]`. The MF sectors `(0,0),(1,0),(0,1)` are `R.zero()` after `expand()`. Example for the Hawkes model: `(1,1)` = free action; `(2,0)`/etc = noise kernel; `(2,1)` = the `ñ·δv²` cubic vertex from `φ''=2a`.

### 5.4 Polynomial-ring element & `.dict()`
A `PolynomialRing(SR, names)` element. Its `.dict()` is `{exponent_vector(tuple of ints): SR coefficient}`. Generators ordered tilde-first; `exp_vec[:n_tilde]` is the response part, `exp_vec[n_tilde:]` the physical part.

### 5.5 Leg tuple
`(base_name: str, population_index: int)`, e.g. `('dv', 1)`. Pop index is 1-based (0 when the field is non-indexed). Repeated in a leg list per exponent multiplicity.

### 5.6 Typed records
`VertexType{coefficient, response_legs, physical_legs, bigrade}`; `ConvVertexType` adds `kernel_attachments` (list of `{symbol, leg, leg_index, kernel_td_fn}`); `SourceType{coefficient, response_legs, bigrade}`; `NoiseSourceType` adds `cumulant_specs` (list of `{symbol, kernel_fn, legs, leg_fields, tau_var, sign, noise, order}`). All use `__slots__` + custom pickle.

### 5.7 Serialized artifacts
`metadata.json` (plain dict, see `save_theory`) + `symbolic_data.sobj` (`{R, S_raw, by_tp, n_tilde, K_ft, G_ft, adj_ft, D_omega, pole_vals, C_mats, G_t}`).

---

## 6. Data flow

**In:** a model dict → `FieldTheory(model, taylor_order)`.

**`expand()` transforms:**
```
model['action'](ns)        →  S_sr (raw SR)
  → reduce_conv_in_action   →  S_sr (Conv atoms resolved; ns._kernel_attachments set)
  → + _build_cumulant_action→  S_sr (+ -W_m series; ns._cumulant_kernels set)
  → taylor(…, order)        →  S_sr (polynomial in fields)
  → .subs(_deriv_rename_subs)→ S_sr (D[0](phi)(0) → phi1_1, …)
  → .subs(specializations)  →  S_sr (phi1_i → 2a·vstar_i, …)
  → _sr_to_ring + _collect_bigrade → by_tp = {(n_t,n_p): poly}
  → _verify_and_zero_mf_sector(by_tp, mf_subs) → MF sectors zeroed
  → ft._S_raw, ft._by_tp
```

**Out of `expand()`:** `ft._by_tp`, accessible via `free_action()`, `noise_kernel()`, `vertices()`, `sectors()`.

**Vertex/source extraction:**
```
ft.vertices()      → extract_vertex_types(ft)  → list[VertexType / ConvVertexType]
ft.noise_kernel()  → extract_source_types(ft)  → list[SourceType / NoiseSourceType]
   ↓
available_degrees(vtypes, stypes) → ({(in,out)}, {out}) → diagram enumerator
```

**Concrete example (Hawkes quadratic).** With `φ(v)=a v²`, after expansion the `(2,1)` sector contains a monomial `½·φ''·ñ·δv² = a·ñ·δv²`. `decompose_sector` yields `VertexType(coefficient=a, response_legs=[('nt',i)], physical_legs=[('dv',i),('dv',i)], bigrade=(1,2))` → `in_degree=2, out_degree=1`. The `v·Conv(g,n)` term, after `reduce_conv_in_action` rule 4, leaves the kernel symbol `g` in the bilinear/vertex coefficients with `ns._kernel_attachments[g] = {dn_j}`; in a conductance vertex this promotes to a `ConvVertexType` whose `kernel_attachments[0]['leg']` is the `δn` leg.

**Serialization round-trip:** `save_theory(path, ft, …)` → directory; `load_theory(path)` → `(meta, data)`; `reload_model(meta)` → fresh model dict → `FieldTheory(...).expand()` re-derives everything (used by `degree_scan.py` to re-expand at a higher Taylor order on demand).

---

## 7. Gotchas & caveats

1. **Sage sums are `add_vararg`, not `operator.add`.** Every place that detects a sum uses `'add' in op.__name__.lower()`, never `op is operator.add`. Getting this wrong *silently drops terms* (`_sr_to_ring` `field_theory.py:266-267`, `_reduce_arg` `convolution.py:254`, the `Conv` display class `field_theory.py:126-129`).

2. **`!= 0` is unreliable for symbolic zero.** Use `.is_trivial_zero()` (syntactic) — `expr != 0` returns `False` for unbound real params with no sign assumption (`field_theory.py:484-490`).

3. **Single-term vs sum coercion.** `_sr_to_ring` explicitly wraps a non-sum in `[expanded]` so it iterates *summands*, not factors — otherwise a single product like `τ·Dt·v` would be torn into factors and its coefficient lost (`field_theory.py:268-271`).

4. **Classify BEFORE MF subs.** `_sr_to_ring`/`_collect_bigrade` run *before* `mf_subs`. Applying MF saddle subs first would inject `E`/`nstar` algebra into the `(1,1)` propagator and ≥2 vertices, where `vstar`/`nstar` must stay free symbolic params (`field_theory.py:913-922`). The MF subs are applied **only** to the saddle sectors, as a *test*, then those sectors are zeroed.

5. **Specializations run twice.** Once after the Taylor (every sector), and again inside the MF verification (`field_theory.py:949`) because the `mf_bg` lambda builds its RHSes *before* specializations runs, so `vstar → …phi0…` reintroduces raw phi-Taylor tokens needing a second pass.

6. **One-shot multivariate Taylor is mandatory for multi-arg formal functions.** Chained single-variable `.taylor()` does **not** compose at non-zero expansion points — inner-arg substructure gets frozen (`field_theory.py:884-892`). Single-arg functions are unaffected (identical output), so this is a safe generalization.

7. **`_iter_multi_indices` uses `itertools.product`, not recursion**, specifically to survive Jupyter `%autoreload` (a self-recursive generator loses its self-reference on hot-swap) — `field_theory.py:210-215`.

8. **Correlated-noise smooth residual only at order n=2.** Order ≥3 smooth (non-δ) cumulant residuals are **silently dropped** with a `warnings.warn` (the δ-local part is still injected). This is fine for N=2 GTaS (all order-≥3 cumulants are fully local) but is a real limitation for future non-local higher-order kernels — the Phase J integrator needs an n-leg time map first (`field_theory.py:550-565`).

9. **`reduce_conv_in_action` `normalized=True` is hard-wired.** Every kernel is assumed to have `ĝ(0)=1`. Non-normalized kernels need per-kernel DC gains (not yet implemented) — `convolution.py:182-185`.

10. **Unresolvable kernel attachments are left in the coefficient.** `extract_vertex_types` deliberately keeps an unresolved kernel symbol *in* the vertex coefficient (emitting a plain `VertexType`) so downstream diagnostics flag the surviving symbol rather than silently zeroing it (`vertices.py:589-595`).

11. **Pickle + `__slots__` subtlety.** `SourceType.__getstate__` references `SourceType.__slots__` *explicitly* (not `self.__slots__`) — on a `NoiseSourceType` instance the latter resolves only to `('cumulant_specs',)`, dropping the base fields from the pickle (`vertices.py:235-238`).

12. **`_NamespaceBoundKernel` pre-evaluates eagerly.** The kernel is evaluated against the namespace at *construction* time (in `extract_source_types`), not lazily. This is required for picklability but means a kernel function that errors will throw during extraction, not during integration.

13. **`reload_model` uses `exec`, not `sage_load`, despite the comment.** The docstring/comment (`serialize.py:232`) says "Use SageMath's load() which handles .py files" but the body uses a plain Python `exec(compile(...))`. The comment is stale (see Open Questions). If a model file relies on Sage preparser syntax (e.g. `^` for power, `2/3` as a rational), this `exec` would *not* preparse it — Sage models in this repo import from `sage.all` and use Python syntax, so it works, but the discrepancy is a trap for a Sage-style model file.

14. **`save_theory` does not persist vertices/sources.** Only `R, S_raw, by_tp, n_tilde` and propagator matrices are saved. `VertexType`/`SourceType` objects are *re-extracted* from `by_tp` on load — they are cheap to rebuild and carry unpicklable-namespace-derived callables that are better reconstructed than persisted.

15. **`propagator.py` is an empty stub.** `msrjd/core/propagator.py` is a docstring only ("logic currently lives in the demo notebook"); the real propagator extraction lives in `pipeline/_propagator.py`. Do not look for K-matrix code in `core`.

16. **macOS fork hazard (project-wide).** Per project memory, fork-based multiprocessing crashed the user's machine; spatial is serial-only, temporal is fork-guarded. This subsystem's picklability design is what *allows* objects to cross a (guarded) process boundary, but the guard itself lives elsewhere.

---

## 8. Glossary

- **MSR-JD** — Martin–Siggia–Rose–Janssen–De Dominicis: the field-theoretic formulation of stochastic dynamics; introduces a conjugate *response field* per physical field.
- **Response (tilde / hat) field** — the auxiliary conjugate field `φ̃` from the MSR-JD construction. "Out-leg" in diagram language.
- **Physical (fluctuation) field** — the actual fluctuating quantity `δφ = φ − φ*` around the mean-field background. "In-leg."
- **Action `S`** — the field-theory functional whose stationary point is the classical solution and whose fluctuations give correlations.
- **Mean field (MF) / saddle point** — the constant background `φ*` solving the deterministic equations of motion; found by demanding the field-linear terms of `S` vanish.
- **Saddle (MF) sector** — bigrade `(0,0),(1,0),(0,1)`: constant, tadpole, EOM residual. Must vanish at the saddle; the framework verifies and zeroes them.
- **Bigrade `(n_tilde, n_phys)`** — the pair (number of response legs, number of physical legs) of a monomial; the primary classification axis.
- **Free action / bilinear kernel** — the `(1,1)` sector; its inverse is the free (tree-level) propagator.
- **Noise kernel** — the `(≥2, 0)` sectors (response-only, ≥quadratic); the noise sources of the diagram expansion.
- **Interaction vertex** — any total-degree-≥3 monomial; the building block of loop diagrams.
- **Taylor order** — the total-degree truncation of every nonlinear-function expansion; equals the max number of legs a vertex can have.
- **Formal function** — a Sage symbolic function (`function('phi')(v)`) left unevaluated so it can be Taylor-expanded abstractly; its derivative-at-0 atoms are renamed to clean symbols.
- **Derivative symbol** — the renamed partial-derivative-at-saddle, e.g. `phi1_2` = `φ'_2(0)`, `phi21_1` = `∂²_x1 ∂_x2 φ_1(0)`.
- **Specialization** — a model-supplied substitution replacing abstract derivative symbols with concrete values (`phi2_i → 2a`).
- **Convolution `Conv(g, n)`** — the formal time-convolution operator `∫ g(t−s) n(s) ds`; asymmetric (only arg 2 carries time content); reduced differently per consumer.
- **DC gain `ĝ(0)`** — the kernel's integral `∫g(t)dt`; for a unit-integral kernel `ĝ(0)=1`.
- **Convolution theorem** — `(g⋆n)^(ω) = ĝ(ω) n̂(ω)`; the basis for rule 4 (defer kernel symbol to FT).
- **Cumulant `κ^{(n)}`** — the n-th cumulant of a noise process; correlated noise contributes a `-W_m[m̃]` cumulant series to the action.
- **GTaS** — the correlated-input ("Gaussian-and-shifts") noise model the cumulant machinery was built for; at N=2, all order-≥3 cumulants are local.
- **Phase J integrator** — the downstream time-domain integrator that consumes the kernel-carrying vertex/source records (per-leg τ scaffolding).
- **Population** — a group of indexed fields of a given size; heterogeneous theories carry several.
- **Leg** — an attachment point of a vertex/source, tagged `(field_base_name, population_index)`; repeated per exponent.
- **SR (Symbolic Ring)** — Sage's universal ring of symbolic expressions.
- **PolynomialRing(SR, names)** — a polynomial ring over `SR` whose generators are the field names; exposes monomials via `.dict()`.
- **`.sobj`** — Sage's binary object-serialization format (`sage_save`/`sage_load`).
- **`add_vararg`** — Sage's internal n-ary addition operator (not Python's `operator.add`); detected by name.
- **`is_trivial_zero()`** — Sage's *syntactic* zero test (True iff literally zero), robust to unbound-parameter ambiguity.
- **`substitute_function`** — Sage method replacing every occurrence of a formal function with a Python-callback result; how `Conv` is reduced.
- **`__slots__`** — a Python class optimization that removes the per-instance `__dict__`; requires explicit `__getstate__`/`__setstate__` for pickling.

---

## 9. Proposed manual subsections

1. **The MSR-JD action as data** — what a model dict is; fields, parameters, kernels, operators, functions; the namespace object; the action lambda.
2. **SageMath in one page** — `SR`, `PolynomialRing(SR, …)`, formal functions, `.dict()`, the `add_vararg` and `is_trivial_zero` gotchas; why symbolic-not-numeric.
3. **From action to polynomial: the expand() pipeline** — the eight-step transform, with the Hawkes worked example carried through each step.
4. **Taylor expansion and formal-function renaming** — multivariate total-degree Taylor; `D[0](phi)(0) → phi1_1`; multi-index suffixes; specializations.
5. **The convolution operator** — why `g*n` is not multiplication; the four reduction rules; the conductance-vertex worked example; kernel attachments.
6. **Bigrade classification** — `(n_tilde, n_phys)`; the sector table; free action / noise kernel / vertices.
7. **Mean-field sector verification** — the saddle sectors; symbolic + numerical two-tier check; why classify-before-subs.
8. **Correlated-noise cumulants** — the `-W_m[m̃]` series; local vs smooth split; cross-field permutations; the order-≥3 limitation.
9. **Typed vertices and sources** — `VertexType`/`SourceType` and the kernel-carrying subclasses; `decompose_sector`; leg tuples; `available_degrees` feeding the enumerator.
10. **Picklability and the process boundary** — `__slots__` pickling, `_NamespaceBoundKernel`, the unpicklable namespace, the macOS fork caveat.
11. **Serialization** — what is saved (and what is deliberately re-derived); `metadata.json` + `.sobj`; `reload_model` and on-demand re-expansion.
12. **Where this sits in the pipeline** — feeders and consumers; the hand-off to enumeration, propagator extraction, and integration.

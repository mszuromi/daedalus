# Theory Files (`.theory.py`) & Serialization

> Subsystem slug: `theory-files`
> Primary source: `pipeline/theory_serialize.py`
> Companion sources read for this brief: `notebooks/daedalus.py` (the loader), `pipeline/_expand_cache.py` (the `saved_theories/` cache slug), `msrjd/core/serialize.py` (the **other** "save_theory" — a `.sobj` cache, not the `.theory.py` text format), `pipeline/ui/main.py` (the form that produces the spec dict), and representative theory files in `theories/`.

---

## Overview

A **theory file** (`theories/<name>.theory.py`) is the human-readable, version-controllable *single source of truth* for one physical model in the Daedalus pipeline. It is an ordinary, importable Python module with exactly three public symbols:

- `build()` — a function returning the **model dict** (the in-memory object the rest of the pipeline consumes). Internally it constructs the model with a fluent *builder* object (`TemporalTheoryBuilder` or `SpatialTheoryBuilder`) whose chained method calls (`.physical_field(...)`, `.parameter(...)`, `.set_action_text(...)`, …) declare the fields, parameters, MSR-JD action, mean-field equations, and (for spatial theories) the boundary/initial conditions.
- `DEFAULT_FUNDAMENTAL` — a plain dict of default numeric parameter values (the "working point" the calculation is evaluated at).
- `METADATA` — a plain dict of *run recommendations*: default cumulant order `k`, default loop order `ℓ`, which external fields to plot, the time grid (`tau_max`/`tau_step`), and (for spatial models) the spatial grid.

This subsystem is the **authoring + persistence boundary** of the pipeline. It has two halves, which are easy to confuse because both are called "serialization":

1. **The `.theory.py` text format** (this brief's focus, file `pipeline/theory_serialize.py`). This module is a *bidirectional bridge between a flat "spec" dict and `.theory.py` source text*:
   - **Forward** (`render_theory_file` / `save_theory_to_file`): the theory-builder UI (`pipeline/ui/main.py`, a Jupyter `ipywidgets` form) collects user input as a flat dict and emits a clean, diff-friendly `.theory.py` file.
   - **Reverse** (`load_spec_from_file`): given an existing `.theory.py`, parse its source *as text* (via Python's `ast` module — **not** by executing it) and reconstruct the original spec dict, so the UI's "Load existing theory" button can re-populate the form for editing.
2. **The `saved_theories/<slug>/` binary cache** (a *different* artefact). When a theory is actually *run*, the heavy symbolic objects (the propagator, Taylor-expanded action, per-diagram integrands) are pickled to disk under `saved_theories/<slug>/` as SageMath `.sobj` files plus a `manifest.json`. That cache is managed by `pipeline/_expand_cache.py` and `msrjd/core/serialize.py`, not by `theory_serialize.py`. This brief documents both because a reader will encounter both directories and must not conflate them.

**Where it sits in the end-to-end pipeline.**

```
  theory-builder UI (ipywidgets form)              hand-authored .theory.py
        │  _collect() → spec dict                          │
        ▼                                                   │
  theory_serialize.render_theory_file(spec)                 │
        │  emits source text                                │
        ▼                                                   ▼
  theories/<name>.theory.py   ◄────────────────────────────┘
        │
        │  daedalus.load_theory('<name>')  →  importlib exec → mod.build()
        ▼
  model dict  ──►  FieldTheory / compute_cumulants / full_integrator (the calculation)
        │                                              │
        │  (heavy symbolic objects)                    ▼
        └──────────────────────────────►  saved_theories/<slug>/*.sobj + manifest.json
                                          (binary cache — _expand_cache.py / msrjd.serialize)
```

- **Feeds it:** the UI's `_collect()` (which returns the spec dict), or a human typing a `.theory.py` directly.
- **Consumes its output:** `daedalus.load_theory(name)` imports the file and calls `build()` to get the model dict; everything downstream (field theory expansion, propagator, diagram enumeration, cumulant evaluation) consumes that model dict.

The key design promise is **round-trip fidelity**: `spec → render → file → load → spec'` should reproduce the *meaningful* content of `spec`. Because the reverse path reads the AST rather than running the module, string literals (action text, equation RHS, kernel expressions) round-trip *verbatim*, while comments and formatting are lost.

---

## The math

A theory file encodes a stochastic field theory in the **Martin–Siggia–Rose–Janssen–De Dominicis (MSR-JD)** formalism. The reader is assumed to know this physics; here is only the minimal vocabulary needed to read the code, expressed in plain (LaTeX-able) text math.

### The action

Each physical field `x` (or `phi`, `h`, `n`, `v`, …) gets a conjugate **response field** `xt` (the "tilde" or "hat" field, written `x̃`). A Langevin/Fokker–Planck process

```
    dx/dt = F[x] + ξ,        ⟨ξ(t) ξ(t')⟩ = 2 D δ(t − t')
```

is written as a path integral with the MSR-JD action `S[x, x̃]`. For the running example `ou_quartic`,

```
    dx/dt = −μ x − ε x³ + ξ,     ⟨ξ(t) ξ(t')⟩ = 2 D δ(t − t'),
```

the action (as it literally appears in `set_action_text`) is

```
    S = ∫dt [ x̃·((∂_t + μ) x + ε x³) − D x̃² ]
```

In the code the time derivative `∂_t` is the operator `Dt`, the quadratic response term `−D x̃²` encodes the white-noise vertex, and the cubic `ε x³` is the interaction. The *string* passed to `.set_action_text(...)` is exactly this expression with `xt` for `x̃` and `^` for powers:

```python
# theories/ou_quartic.theory.py:27-28
'sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)'
```

The `sum(... for i in pop)` is a population comprehension — see "indexing" below.

### Mean field / saddle point

The classical (deterministic, noise-free) trajectory is the **mean-field saddle** `x*`. Theory files declare it one of two ways:

- **DAE form** (preferred, new): `.equation(lhs=..., rhs=..., population=...)`, e.g.

  ```python
  # ou_quartic.theory.py:29
  .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='pop')
  ```

  read as "`(∂_t + μ) x = −ε x³`". This is a differential-algebraic equation (DAE) the solver integrates to a fixed point.

- **Legacy explicit-saddle form**: `.set_mf_equation('xstar', 'rhs')`, naming the saddle symbol (`<natural>star`) and giving its value, e.g. in `quadratic_hawkes_alpha`:

  ```python
  .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*g[i,j]*nstar[j] for j in E))')
  .set_mf_equation('nstar', 'phi[i](vstar[i])')
  ```

`render_theory_file` prefers `.equation(...)` when the spec has an `equations` list and falls back to `.set_mf_equation(...)` otherwise (`pipeline/theory_serialize.py:424-433`).

### Propagator, cumulants, and the loop expansion

After choosing `x*`, the theory is expanded around it. The **propagator** is the inverse of the quadratic part of `S`; **cumulants** (connected correlation functions) are computed order by order in a **loop expansion** (the small parameter is roughly `g_eff ≈ ε·D/μ²`; `ou_quartic.theory.py:34` notes `g_eff ≈ 0.02`). Two integers parametrise the request:

- `k` = the cumulant order (number of external legs); `k=2` is the two-point function.
- `ℓ` (`max_ell` / `loop_order`) = the highest loop order included; `ℓ=0` is tree level, `ℓ=1` one-loop, etc.

These appear in `METADATA` as `k_default` / `ell_default` and become the file-name keys in the binary cache (`..._k2_l1.sobj`).

### Spatial extension

For **spatial** theories the field lives on `x ∈ ℝ^d`, and the action carries the Laplacian `∇²` and, for derivative-vertex models, gradient operators. Two distinct authoring styles exist:

- **Bare-multiplicative Laplacian** (v1): the inert symbol `Laplacian` is used multiplicatively, e.g. `(Dt + mu - D*Laplacian)*phi`. No `.operator_ir()` is needed.
- **Operator-IR** (v2): the action uses *call syntax* — `Dt(h)`, `Lap(h)`, `Dx(h, 0)` — for genuine derivative *vertices*. This is **load-bearing**: KPZ's nonlinearity `(λ/2)(∂_x h)²` is a *per-leg* gradient vertex, written `Dx(h, 0)^2`, and Model-B's conserving flux `g ∇²(φ²)` is a *composite* derivative vertex `g*Lap(phi^2)`. These two differ — `(∂φ)² ≠ ∂(φ²)` — and produce different momentum **form factors** in the loop integral. A theory using call syntax **must** carry `.operator_ir()`; the serializer round-trips this toggle precisely because dropping it would silently compute the wrong physics (`pipeline/theory_serialize.py:452-459`).

KPZ in `theories/kpz_1d.theory.py`:

```
    ∂_t h  =  −μ h  +  D ∇²h  +  (λ/2)(∂_x h)²  +  η,
    ⟨η(x,t) η(x',t')⟩ = 2T δ(x−x') δ(t−t')
```

```python
.set_action_text(
    'ht*(Dt(h) + mu*h - D*Lap(h) - (lam/2)*Dx(h, 0)^2) - T*ht^2')
.operator_ir()
.boundary('infinite')
.initial('stationary')
```

The `0` in `Dx(h, 0)` is the **spatial axis index** (here d=1, so axis 0).

### Colored noise & Markovianization

White noise gives a local `−D x̃²` vertex. **Colored** (finite-correlation-time) noise instead enters as a **cumulant-generating-function (CGF) term** declared with `.declare_cgf_term(...)`, carrying an `order`, a `coefficient`, and a temporal `kernel`. From `ou_quartic_colored`:

```python
.declare_cgf_term('CXX', response_legs=['xt', 'xt'], order=2,
                  coefficient='2*D/tauc', kernel='exp(-abs(tau)/tauc)')
```

This is the Ornstein–Uhlenbeck noise spectrum `⟨ξ ξ⟩ ∝ (D/τ_c) exp(−|τ|/τ_c)`. By default the pipeline **Markovianizes** such a kernel (embeds it in an auxiliary Markovian field via `pipeline/colored_to_markovian.py`); `.markovianize(False)` opts out, and per-term `markovianize=True/False` overrides individual rows.

### Indexing / populations

A theory can be **homogeneous** (one population, scalar fields) or **heterogeneous** (multiple populations of different sizes, vector/matrix parameters). Indexing is encoded two ways:

- **New population style**: `.population('E', size=2)` declares a named population of size `N`; fields and parameters attach via `population='E'` / `indexed_by=['E']` (vector) / `indexed_by=['E','E']` (matrix). Example `quadratic_hawkes_alpha.theory.py:36-46`.
- **Legacy style**: `n_populations=N` in the constructor plus `indexed=True` (vector) / `indexed='matrix'` on parameters. Example `linear_hawkes.theory.py:24,28`.

The serializer accepts and round-trips both, with `indexed_by` winning when both are present.

---

## External tools used

This subsystem's *core file* (`pipeline/theory_serialize.py`) is deliberately **dependency-light** — it touches only the Python standard library. The heavier scientific libraries enter only in the *companion* files (the loader and the binary cache). Here is each, explained from scratch, with the actual imports/calls.

### Python standard library `ast` — Abstract Syntax Tree

**What it is.** `ast` is a built-in Python module that parses Python source *text* into a tree of node objects (the program's grammar, before execution). You can inspect this tree to learn what a program *says* without ever *running* it — which is exactly what you want when reading an untrusted-or-just-unwanted-to-execute config file.

**How this code uses it.** The entire reverse path (`load_spec_from_file`) is an AST walk. Key calls in `pipeline/theory_serialize.py`:

```python
import ast                                   # line 545
tree = ast.parse(src)                        # line 550  — text → AST
description = ast.get_docstring(tree)        # line 553  — module docstring
val = ast.literal_eval(node.value)           # line 568  — a *safe* eval of a literal
```

- `ast.parse(src)` turns the whole file into a module node; a malformed file raises `SyntaxError` here (documented at line 540).
- `ast.get_docstring(tree)` pulls the module's leading triple-quoted string (the theory's description).
- `ast.literal_eval(node)` evaluates a node **only if it is a literal** (numbers, strings, lists, dicts, tuples, booleans, `None`). It refuses function calls or names, so it can never run arbitrary code — this is why the reverse path can recover `default=[[2,3],[1,3]]` matrices safely. The code wraps it in a local helper `_lit(node, default=None)` (line 639) that returns a fallback on `ValueError`/`SyntaxError`/`TypeError`.
- The build chain is walked by recognizing node *types*: `ast.Call` (a method call), `ast.Attribute` (the `.method` part), `ast.Name` (the constructor name), `ast.Assign` (the `DEFAULT_FUNDAMENTAL = {...}` / `METADATA = {...}` module-level assignments), `ast.Return`, `ast.FunctionDef`. See the chain-walk loop at lines 588-598 and the per-method dispatch at lines 649-865.

The forward path uses **no** `ast` — it builds source by string concatenation (the `_emit_*` helpers and `render_theory_file`).

### Python standard library `re` — regular expressions

**What it is.** `re` is the built-in regular-expression engine.

**How this code uses it.** Exactly one place: `_slugify` turns a human theory name into a filesystem-safe identifier.

```python
import re                                                   # line 41
return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower() or 'theory'   # line 98
```

`re.sub` replaces every run of non-alphanumeric characters with a single underscore. (Note: the binary-cache slug in `_expand_cache.py:106` uses the *same* regex pattern, so the human-facing filename slug and the on-disk cache-dir slug agree.)

### Python standard library `textwrap`

**What it is.** Utilities for re-flowing indented text.

**How this code uses it.** `_emit_action` (line 284) imports `textwrap` and calls `textwrap.dedent(action_text)` to strip common leading whitespace *before* re-indenting the action block. This makes a load→save cycle whitespace-stable: without the dedent, every save would accumulate four more leading spaces on each multi-line action (documented at lines 281-283).

### Python standard library `os`, `importlib`, `typing`

- `os` (imported locally in `save_theory_to_file` line 511 and `load_spec_from_file` line 546): `os.path.isdir`, `os.path.join` for path handling.
- `importlib.util` is used by the **loader** `daedalus.load_theory` (not by `theory_serialize.py`) to import a `.theory.py` by file path:

  ```python
  # notebooks/daedalus.py:77-79
  spec = importlib.util.spec_from_file_location(f'theories.{name}', path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  ```

  `spec_from_file_location` builds an import "spec" for an arbitrary file path (so the file does not have to be on `sys.path`); `module_from_spec` creates an empty module object; `exec_module` actually runs the file's top-level code, defining `build`, `DEFAULT_FUNDAMENTAL`, `METADATA`. **This is the one place a `.theory.py` is executed** — so a hand-authored theory's `build()` runs real Python here, unlike the AST-only reverse path.
- `typing.Any` is used only for type hints (`_py_repr(value: Any) -> str`).

### SageMath (`sage.all`) — only in the *binary cache*, not the text format

**What it is.** SageMath is a large open-source computer-algebra system built on Python. It provides exact symbolic objects (polynomials over chosen rings, symbolic matrices, etc.). Its `save`/`load` functions pickle these objects to `.sobj` files (the extension is appended automatically).

**How this code uses it.** *Not* in `theory_serialize.py`. The companion module `msrjd/core/serialize.py` (a different "save_theory"!) uses it to persist the expanded field theory:

```python
# msrjd/core/serialize.py:23
from sage.all import save as sage_save, load as sage_load, version as sage_version
...
sage_save(sym, os.path.join(path, 'symbolic_data'))   # line 150  → symbolic_data.sobj
data = sage_load(sobj_path)                            # line 185
```

and `pipeline/_expand_cache.py` uses `sage_save` to write `propagator.sobj` / `expand_taylor<N>.sobj` (per its module docstring, lines 36-44). **Key distinction for a novice:** a `.theory.py` is plain Python text (no Sage needed to read it); a `.sobj` is a pickled Sage object (Sage *is* needed to load it). Conflating them is the single most common confusion in this subsystem.

### `json` — only in the binary cache

`json` (Python standard library) writes the human-readable `manifest.json` (`pipeline/_expand_cache.py`) and `metadata.json` (`msrjd/core/serialize.py:131`) that index the binary `.sobj` blobs. The `.theory.py` text format does *not* use JSON.

### `ipywidgets` (the UI producer) — upstream, not in this file

The spec dict that feeds `render_theory_file` is produced by `pipeline/ui/main.py`, a Jupyter `ipywidgets` form. `theory_serialize.py` never imports `ipywidgets`; it only consumes the plain dict the form's `_collect()` returns. Worth knowing because the spec-dict *shape* is dictated by that form.

---

## Components

Every significant function in `pipeline/theory_serialize.py`, plus the companion loaders. File:line and signatures are exact.

### Python-source helpers (forward path low-level)

#### `_py_repr(value: Any) -> str`  — `pipeline/theory_serialize.py:46`

Pretty-prints a Python value as a *source-code literal*, with formatting tuned for theory data. Step by step:

- `str` → `repr(value)` (adds quotes/escapes) — line 52-53.
- `bool`/`None`/`int`/`float` → `repr(value)` — line 54-55. (Note: the `bool` check comes *before* `int`, which matters because in Python `bool` is a subclass of `int`.)
- `list` → line 56-66:
  - If it's a **2D list** (every element is itself a list, i.e. a matrix) → multi-line, one row per line, each row indented four spaces (`[\n    [..],\n    [..],\n]`).
  - Else a **1D list** → single line if the joined body ≤ 60 chars, otherwise one item per line.
- `dict` → line 67-73: `{}` for empty, else one `key: value` per line, four-space indent.
- `tuple` → line 74-76: parenthesised; a 1-tuple gets a trailing comma (`(x,)`).
- Fallback → `repr(value)` (line 77).

This is what makes `default=[[2, 3], [1, 3]]` print as a readable matrix block in `quadratic_hawkes_alpha.theory.py:38-41`.

#### `_kw(name: str, value: Any) -> str`  — `pipeline/theory_serialize.py:80`

Formats one `name=value` pair: returns `f'{name}={_py_repr(value)}'`.

#### `_kw_chain(*pairs: tuple[str, Any]) -> str`  — `pipeline/theory_serialize.py:85`

Takes `(name, value)` pairs and joins the non-trivial ones with `', '`. **Omits** any pair whose value is `None` or `''` (line 90-92). This is the mechanism by which optional keyword arguments are emitted only when meaningful, keeping files "textually quiet" — a recurring design phrase. So `.parameter('mu', default=1.0)` does *not* emit `description=None`.

#### `_slugify(name: str) -> str`  — `pipeline/theory_serialize.py:96`

Filesystem-safe lowercase identifier; `re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower() or 'theory'`. Returns `'theory'` for an all-symbol name. Used by `save_theory_to_file` when `path` is a directory, and mirrored in the UI's save hint (`pipeline/ui/main.py:2193`).

### Builder-method emitters (forward path, one per builder call)

Each takes a spec sub-dict and returns one `.method(...)` source line.

#### `_emit_response_field(f: dict) -> str`  — `pipeline/theory_serialize.py:103`

Emits `.response_field('name', indexed=..., latex=..., description=...)`. Only emitted for response fields the user *explicitly* declared; the natural-name style auto-creates conjugate responses inside the builder, so most files have no `.response_field(...)` line. (The function has an odd `('', None)` placeholder in its `_kw_chain` call — see open questions.)

#### `_emit_field(method: str, f: dict, *, with_natural: bool) -> str`  — `pipeline/theory_serialize.py:112`

The workhorse for both `.response_field(...)` and `.physical_field(...)`. Logic:

1. **Population vs indexed** (mutually exclusive): if `f['population']` is set, emit `population='<name>'` (shape from population size); else emit `indexed=<bool>` (default `True`). Lines 128-131.
2. **Natural name** (only when `with_natural=True`, i.e. physical fields): emit `natural_name=` only if it differs from `name` (line 133-135). The natural name is the user-facing letter; the builder auto-prefixes `d` to get the internal fluctuation name and auto-creates the response + saddle.
3. `latex`, `description` (lines 136-139).
4. **Per-field `spatial_dim`** (line 144-145): emitted only when non-zero (`int(f['spatial_dim'])`), so time-only fields round-trip unchanged and mixed dim=0/dim=d theories are preserved without a separate `.spatial_dim(d)` line.

Returns `.<method>('name'[, kwargs])`.

#### `_emit_parameter(p: dict) -> str`  — `pipeline/theory_serialize.py:151`

Emits a `.parameter(...)` call. Important behaviours:

- **Saddle parameters are NOT emitted** — handled by the caller's filter, not here (see `render_theory_file` lines 334-339), because the framework recreates `<natural>star` from the physical-field declaration.
- **Indexing**: prefers new-style `indexed_by` (a list of population names); falls back to a legacy `type` translation (`'scalar'→None`, `'vector'→True`, `'matrix'→'matrix'`). Lines 163-175.
- Emits `default`, `indexed_by` (list-copied), `indexed`, `domain`, `mean_field` (only when truthy — `p.get('mean_field') or None`), `natural_name`, `description` (lines 176-184). Note `mean_field` should already be filtered out upstream; this is belt-and-suspenders.

#### `_emit_function(fn: dict) -> str`  — `pipeline/theory_serialize.py:189`

Emits `.define_function('name', args=[...], expression='...', population=..., latex=..., description=...)`. `args` and `expression` are mandatory (read with `fn['args']` / `fn['expression']`, not `.get`). Example: `.define_function('phi', args=['v'], expression='a[i]*v^2', population='E')`.

#### `_emit_kernel(k: dict) -> str`  — `pipeline/theory_serialize.py:200`

Emits `.define_kernel(...)` for a convolution kernel. Handles the same `indexed_by` (new) vs `indexed` (legacy `'vector'`/`'matrix'`) duality as parameters. Emits `time_expr`, `freq_image`, `latex_name` (falls back to `latex`), `sage_name`, `description`, `indexed_by`, `indexed`. A kernel may be given in the time domain (`time_expr='(t/taug[i,j]^2)*exp(-t/taug[i,j])*heaviside(t)'`) or directly as a frequency image (`freq_image='1 / (1 + I*omega*tau_g)'`).

#### `_emit_cgf_term(c: dict) -> str`  — `pipeline/theory_serialize.py:233`

Renders one colored-noise CGF row as `.declare_cgf_term(...)`. Two shapes:

- **Multi-leg** (when `response_legs` has > 1 entry): `.declare_cgf_term('CXX', response_legs=['xt','xt'], order=2, coefficient=..., kernel=...)`.
- **Legacy single-leg**: `.declare_cgf_term('name', 'mt', order=..., ...)` — second positional is the single response field (taken from `response_legs[0]` if a 1-element list, else from `response_field`).

`order` is forced to `int`. A per-row `markovianize` is emitted *only* when it's an explicit bool (lines 256-257); `None`/`'auto'` rows are serialized without the keyword.

#### `_emit_action(action_text: str) -> str`  — `pipeline/theory_serialize.py:277`

Emits `.set_action_text(...)`. Multi-line bodies are dedented (via `textwrap.dedent`) then re-indented four spaces inside a triple-quoted block (`'''\n...\n'''`); single-line bodies are emitted as a simple repr. The dedent is what keeps save→save idempotent.

#### `_emit_mf_equation(saddle: str, rhs: str) -> str`  — `pipeline/theory_serialize.py:292`

Emits the legacy `.set_mf_equation('vstar', 'rhs')`.

#### `_emit_equation(lhs: str, rhs: str, population) -> str`  — `pipeline/theory_serialize.py:297`

Emits the new DAE-form `.equation(lhs='...', rhs='...', population='...')`; `population` omitted when falsy.

### Main entry points (forward)

#### `render_theory_file(spec: dict) -> str`  — `pipeline/theory_serialize.py:310`

The forward heart: takes a spec dict and returns the **full `.theory.py` source string**. Step by step:

1. **Read spec fields** with `.get(... , default)` so partial specs still render (lines 320-351): `name`, `populations`, `n_populations`, `description`, `response_fields`, `physical_fields`, `parameters`, `functions`, `kernels`, `cgf_terms`, `action_text`, `mf_equations`, `equations`, `default_fundamental`, `metadata`.
2. **Filter saddle parameters** (lines 334-339): drop any parameter with `mean_field` truthy, and any whose name matches the auto-generated `<natural>star` convention — the framework re-creates these.
3. **Pick the builder class** (lines 370-372): `SpatialTheoryBuilder` iff any physical field has `spatial_dim ≥ 1`, else `TemporalTheoryBuilder`. (This is the "builder split"; the AST loader accepts all three constructor names, including the legacy `TheoryBuilder`.)
4. **Emit the docstring header** (lines 354-363): `"""<name> — text-driven theory file."""` plus the optional description and the boilerplate "Generated by … / Loaded by …" lines (which the reverse path later strips).
5. **Emit `from pipeline.theory import <BuilderCls>`** and `def build(): return (` (lines 376-380).
6. **Constructor**: if `populations` present, emit `BuilderCls('name')` then one `.population('name', size=N, description=...)` per population (each size explicit, declaration order visible); else emit `BuilderCls('name', n_populations=N)` (lines 381-396).
7. **Emit in fixed order**: response fields, physical fields, parameters, functions, kernels, CGF terms (lines 401-412).
8. **Action** (lines 414-420), then equations: prefer `.equation(...)` from `equations`, else fall back to `.set_mf_equation(...)` from `mf_equations` (lines 422-433).
9. **Toggles emitted only when non-default** (so files stay quiet):
   - `.stability_analysis(True)` only when on (lines 439-440);
   - `.markovianize(False)` only when the spec explicitly turned it off (lines 448-449);
   - `.operator_ir()` only when on — **load-bearing for derivative-vertex spatial theories** (lines 452-459).
10. **Spatial BC/IC**: `.boundary(mode, **params)` and `.initial(mode, **params)` only when present (lines 466-477).
11. **Dyson / reference diffusion**: `.dyson_order(N)` for a non-`'off'` policy, `.reference_diffusion(D0)` when set (lines 484-489).
12. **Close the chain** with `.build()` and the closing `)` (lines 491-492).
13. **Append module-level assignments** `DEFAULT_FUNDAMENTAL = {...}` and `METADATA = {...}` via `_py_repr` (lines 495-498).
14. Return `'\n'.join(out)`.

#### `save_theory_to_file(spec: dict, path: str) -> str`  — `pipeline/theory_serialize.py:504`

Writes the rendered source to `path` and returns the path actually written. Path normalization (lines 513-518):

- If `path` is a **directory**, the filename is derived as `<_slugify(spec['name'])>.theory.py` inside it.
- Else if `path` does not already end in `.theory.py`: if it ends in `.py` it's left alone, otherwise `.theory.py` is appended. (So `'myfoo'` → `'myfoo.theory.py'`, but `'myfoo.py'` → `'myfoo.py'`.)

Then `render_theory_file(spec)` and `open(path,'w').write(source)`.

### Main entry point (reverse)

#### `load_spec_from_file(path: str) -> dict`  — `pipeline/theory_serialize.py:528`

Parses a `.theory.py` *as text* and reconstructs the spec dict. Step by step:

1. Read the file, `ast.parse(src)` (lines 548-550).
2. **Description**: `ast.get_docstring(tree)`, then `_strip_doc_boilerplate` to remove the auto-header (lines 553-556).
3. **Module-level assignments**: scan `tree.body` for `ast.Assign` nodes targeting `DEFAULT_FUNDAMENTAL` / `METADATA`, `ast.literal_eval` their values (falling back to `{}`), keep only dicts (lines 561-574).
4. **Find `build()`**: the `ast.FunctionDef` named `build`; raise `ValueError` if missing. Find its `ast.Return`; raise if it isn't a call expression (lines 577-585).
5. **Unwind the method chain** (lines 588-598): starting at the return-call, repeatedly step into `cur.func.value` while the node is an `ast.Call`; stop at the constructor `ast.Name`. Then **reverse** so the constructor is first and methods follow in source order.
6. **Initialise the spec dict** with all expected keys and their defaults (lines 601-637), including the four toggle defaults that mirror the builder's own defaults: `stability_analysis=False`, `markovianize_default=True`, `operator_ir=False`, and the empty lists.
7. **Local helpers**: `_lit(node, default)` (safe literal eval, line 639) and `_kwargs(call)` (line 646, returns `{kw.arg: _lit(kw.value)}`).
8. **Dispatch over the chain** (lines 649-865): the constructor branch reads `name` and `n_populations`; then a long `if/elif` on `method` (the `.attr`) reconstructs each spec sub-list. Notable branches:
   - `population` → append `{name, size?, description?}`.
   - `physical_field` → append `{name, indexed?, population?, natural_name?, latex?, description?, spatial_dim?}`; **inherits** a builder-level `_default_spatial_dim` set by an earlier `.spatial_dim(d)` call if no per-field kwarg (lines 682-683).
   - `parameter`, `define_function`, `define_kernel`, `response_field` → analogous field-copying.
   - `declare_cgf_term` → handles both legacy positional `response_field` and new `response_legs` (accepts a list literal *or* a comma-separated string), plus `order`/`coefficient`/`kernel` and optional `markovianize` (lines 717-748).
   - `set_action_text` → `spec['action_text']`.
   - `set_mf_equation` → append to `mf_equations`.
   - `equation` → append `{lhs, rhs, population}` to `equations` (kwargs preferred, positional tolerated).
   - `stability_analysis` / `markovianize` / `operator_ir` → toggle booleans (with sensible no-arg defaults; `markovianize()`→True, `operator_ir()`→True).
   - `spatial_dim` → bulk-set: record `_default_spatial_dim` and `setdefault` it on every already-parsed physical field.
   - `boundary` / `initial` → `{mode, **params}` dicts.
   - `dyson` / `dyson_order` → `{mode, order}`; `dyson_order(N)` is sugar for `dyson(mode='fixed', order=N)`.
   - `reference_diffusion` → float.
   - `build` → no-op.
9. Sync `n_populations` to `len(populations)` when the populations list is non-empty (lines 868-869).
10. Return `spec`.

#### `_strip_doc_boilerplate(doc: str) -> str`  — `pipeline/theory_serialize.py:873`

Removes the auto-generated header so a reloaded theory shows only the user's own description. Drops the first line if it contains the em-dash `—` (the `"<name> — text-driven theory file."` line), stops at the `Generated by …` / `Loaded by …` lines, and trims surrounding blank lines.

### Companion: the loader

#### `daedalus.load_theory(name: str)`  — `notebooks/daedalus.py:68`

`(model, module) = load_theory('kpz_1d')`. Resolves `theories/<name>.theory.py`, imports it via `importlib.util` (executing it), and returns `(module.build(), module)`. The module exposes `DEFAULT_FUNDAMENTAL` and `METADATA`. Helper `list_theories()` (line 61) lists every `theories/*.theory.py`. `THEORIES_DIR` is `<repo_root>/theories` (line 55), where `repo_root()` (line 40) walks up until it finds a `pipeline/` directory. **This is the canonical "load by name" path** for the demo/example notebooks.

### Companion: the *binary* serializer (a different "save_theory")

These do **not** read/write `.theory.py`; they persist expanded symbolic objects. Documented here because the directory `saved_theories/` and the function name `save_theory`/`load_theory` collide with the text format and confuse readers.

- `msrjd/core/serialize.py:save_theory(path, ft, propagator_data, ...)`  — line 52. Writes `metadata.json` (JSON-able field/expansion metadata, lines 85-128) and `symbolic_data.sobj` (Sage-pickled `R`, `S_raw`, `by_tp`, `K_ft`, `G_ft`, propagator objects, lines 134-150). The model dict (with lambdas) is **not** serialized; instead `model_file`+`model_var_name` record where to re-import it (docstring lines 12-15).
- `msrjd/core/serialize.py:load_theory(path)`  — line 156. Returns `(meta, data)` by reading `metadata.json` + `symbolic_data.sobj`.
- `msrjd/core/serialize.py:reload_model(meta, project_root)`  — line 192. `exec`s the stored model `.py` into a namespace and pulls out `model_var_name`. **Caveat:** this uses raw `exec`, not the AST path — it runs arbitrary code in the model file.
- `pipeline/_expand_cache.py:_slug(model)` / `cache_dir(model)`  — lines 104/109. The on-disk cache slug is `re.sub(r'[^A-Za-z0-9]+', '_', model['name']).strip('_').lower()` — *the same pattern as `_slugify`* — under `saved_theories/<slug>/`.

---

## Data structures

### The **spec dict** (the central object)

A flat dict produced by the UI's `_collect()` and consumed by `render_theory_file`; reconstructed by `load_spec_from_file`. Keys (defaults from `load_spec_from_file` lines 601-637):

| Key | Type | Meaning |
|---|---|---|
| `name` | str | Human theory name (also the slug source). |
| `description` | str | User prose (boilerplate-stripped on reload). |
| `populations` | list[dict] | `[{'name','size','description'?}, ...]` (new heterogeneous style). |
| `n_populations` | int | Legacy population count; synced to `len(populations)` when populations present. |
| `response_fields` | list[dict] | Explicitly-declared response fields. Usually empty (auto-created). |
| `physical_fields` | list[dict] | `[{'name', 'indexed'?, 'population'?, 'natural_name'?, 'latex'?, 'description'?, 'spatial_dim'?}, ...]`. |
| `parameters` | list[dict] | `[{'name','default'?,'indexed_by'?,'indexed'?,'domain'?,'mean_field'?,'natural_name'?,'description'?}, ...]`. |
| `functions` | list[dict] | `[{'name','args','expression','population'?,'latex'?,'description'?}, ...]`. |
| `kernels` | list[dict] | `[{'name','time_expr'?,'freq_image'?,'latex_name'?,'sage_name'?,'description'?,'indexed_by'?,'indexed'?}, ...]`. |
| `cgf_terms` | list[dict] | `[{'name','response_legs'? / 'response_field'?,'order','coefficient','kernel'?,'markovianize'?}, ...]`. |
| `action_text` | str | The MSR-JD action expression (verbatim). |
| `mf_equations` | list[dict] | Legacy `[{'saddle','rhs'}, ...]`. |
| `equations` | list[dict] | DAE form `[{'lhs','rhs','population'}, ...]`. |
| `stability_analysis` | bool | Default `False`. |
| `markovianize_default` | bool | Default `True`. |
| `operator_ir` | bool | Default `False`. **Load-bearing** for derivative-vertex spatial theories. |
| `boundary` | dict | `{'mode', **params}` e.g. `{'mode':'infinite'}` or `{'mode':'periodic','length':...}`. |
| `initial` | dict | `{'mode', **params}` e.g. `{'mode':'stationary'}`. |
| `dyson` | dict | `{'mode':'fixed','order':N}`. |
| `reference_diffusion` | float | Scalar `D0` for the 𝒟 = D0·I + 𝒟̂ split. |
| `default_fundamental` | dict | Numeric defaults → emitted as `DEFAULT_FUNDAMENTAL`. |
| `metadata` | dict | Run recommendations → emitted as `METADATA`. |
| `_default_spatial_dim` | int | *Transient* helper set by a `.spatial_dim(d)` call during reverse parse; not normally re-emitted as a key. |

### `DEFAULT_FUNDAMENTAL` (module-level dict in the file)

Numeric working point. May contain scalars, vectors (lists), or matrices (lists of lists). Examples:
- `{'mu': 1.0, 'eps': 0.02, 'D': 1.0}` (`ou_quartic`)
- `{}` (empty when defaults are carried on `.parameter(default=...)` instead, as in `ou_quartic_colored` and `quadratic_hawkes_alpha`)
- `{'E': [0.78, 0.81], 'w': [[0.30,0.25],[0.30,0.35]], ...}` (`linear_hawkes`, vector/matrix entries)

### `METADATA` (module-level dict in the file)

Run recommendations. Observed keys across the read theories:

| Key | Example | Meaning |
|---|---|---|
| `k_default` | `2` | Default cumulant order. |
| `ell_default` | `0` / `1` | Default loop order. |
| `recommended_external_fields` | `[('dx',1),('dx',1)]`, `[('n',1),('n',2)]` | `(field, population_index)` legs to compute/plot. |
| `tau_max`, `tau_step` | `8.0`, `0.5` | Time grid. |
| `fixed_point_index_default` | `0` | Which saddle root to use (temporal only). |
| `spatial_grid` | `[0.0,...,6.0]` (explicit) or `[-8.0, 8.0, 65]` (`start,stop,n`) | Spatial sampling. |
| `description` | string | Optional short label. |

Note `recommended_external_fields` entries are **tuples**, which `_py_repr` renders as `('dx', 1)` and `ast.literal_eval` recovers as tuples on reload.

### The **binary cache** directory `saved_theories/<slug>/`

| File | Producer | Contents |
|---|---|---|
| `manifest.json` | `_expand_cache.py` | `{"entries":[{"key","stage","k","loop_order","saved_at"}...], "created","updated"}` — bookkeeping index of the cached artefacts (example shown below). |
| `propagator.sobj` | `_expand_cache.py` | Sage-pickled symbolic propagator (taylor-order-independent). |
| `expand_taylor<N>.sobj` | `_expand_cache.py` | Sage-pickled Taylor-expanded action at order N (`expand_taylor4`, `expand_taylor6`). |
| `unique_typed_mult_v3_<ext>_taylor<N>_k<k>_l<l>.sobj` | `_expand_cache.py` | Per-diagram integrand bundle, keyed by external legs `<ext>` (e.g. `dx1_dx1`), Taylor order, k, loop order. |
| `metadata.json` + `symbolic_data.sobj` | `msrjd/core/serialize.py` | The *other* (older) save format. |

Example `manifest.json` entry (from `saved_theories/ou_quartic_white_noise/`):

```json
{ "key": "unique_typed_mult_v3_dx1_dx1_taylor6_k2_l2",
  "stage": "unique_typed_mult_v3_dx1_dx1_taylor6",
  "k": 2, "loop_order": 2,
  "saved_at": "2026-06-17T23:03:39.062450" }
```

**The slug is `model['name']` lowercased with non-alphanumerics → `_`.** So `'OU Quartic (white noise)'` → `saved_theories/ou_quartic_white_noise/`.

---

## Data flow

### Forward (authoring → file)

1. The UI form's `_collect()` returns a **spec dict** (shape above).
2. `save_theory_to_file(spec, path)` → normalises `path` → `render_theory_file(spec)` → writes text.
3. The emitted file's content is fully determined by the spec; optional keys are dropped when `None`/`''`/default.

Concrete: the spec
```python
{'name': 'OU Quartic (white noise)',
 'populations': [{'name': 'pop', 'size': 1}],
 'physical_fields': [{'name': 'x', 'population': 'pop', 'description': 'variable'}],
 'parameters': [{'name':'mu','default':1.0,'domain':'positive'}, ...],
 'action_text': 'sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)',
 'equations': [{'lhs':'(Dt+mu)*x[i]','rhs':'-eps*x[i]^3','population':'pop'}],
 'default_fundamental': {'mu':1.0,'eps':0.02,'D':1.0},
 'metadata': {'k_default':2, ...}}
```
renders to `theories/ou_quartic.theory.py` exactly as shown in the Read of that file.

### Reverse (file → editable form)

1. `load_spec_from_file('theories/ou_quartic.theory.py')` → `ast.parse` → walk `build()`'s call chain → spec dict.
2. The UI re-populates its widgets from the spec; the user edits and re-saves.

**Round-trip guarantee:** string literals (action, equation RHS, kernel exprs) survive verbatim; Python literals (defaults, `indexed_by`, metadata tuples) survive exactly via `literal_eval`; comments and exact formatting do not survive (it is an AST round-trip, lines 538-540).

### Run-time (file → calculation → cache)

1. `daedalus.load_theory(name)` imports & runs the file, returns `(model, mod)`.
2. The model dict flows into the field-theory expansion; `_expand_cache` / `msrjd.serialize` pickle the heavy symbolic results under `saved_theories/<slug>/`.
3. On the next run, the cache is reused (keyed by slug + taylor order + k/ℓ + external legs).

---

## Gotchas & caveats

- **Two different "serialization" layers share a name and a directory.** `pipeline/theory_serialize.py` is the *text* (`.theory.py`) format. `saved_theories/<slug>/*.sobj` is a *binary Sage pickle* cache managed by `_expand_cache.py` / `msrjd/core/serialize.py`. The latter even defines its own `save_theory`/`load_theory`. Do not conflate them: you never need Sage to read a `.theory.py`; you always need Sage to load a `.sobj`.

- **`.operator_ir()` is LOAD-BEARING and silent.** Comment at `theory_serialize.py:452-457`: a re-saved KPZ/Burgers/Model-B theory that loses the `.operator_ir()` line will "silently lose the `Dt()/Lap()/Dx()` lowering and compute the wrong physics." The serializer therefore round-trips it explicitly, and it is emitted only when ON (so v1 bare-Laplacian theories stay quiet). If you hand-edit a derivative-vertex theory, keep that line.

- **The reverse path reads, the loader executes.** `load_spec_from_file` uses `ast` and never runs the file (safe; can't recover non-literal values). `daedalus.load_theory` and `msrjd.serialize.reload_model` *do* execute the file (`exec_module` / raw `exec`) — so a `.theory.py` runs real Python at run time. A hand-authored theory can therefore contain logic that the AST round-trip would not preserve, and the UI "Load" button would silently flatten.

- **Non-literal arguments are lost on reload.** `_lit` returns `None`/fallback for anything that isn't an `ast.literal_eval`-able literal. If a hand-written theory passes a computed expression (e.g. `default=2*0.4`) as a builder kwarg, `load_spec_from_file` recovers `None` for it. Defaults are typically plain literals, so this rarely bites, but it is a real edge.

- **`mean_field` / saddle parameters are filtered twice.** `render_theory_file` drops `mean_field` params and `<natural>star`-named params (lines 334-339), and `_emit_parameter` also guards `mean_field` (line 181). The framework re-creates these from the physical-field declaration; emitting them would double-declare.

- **`indexed_by` wins over legacy `indexed`/`type`.** Both `_emit_parameter` and `_emit_kernel` accept either; if both are present the new population-style `indexed_by` is used and `indexed=` is suppressed. Mixing them in a hand-written file is undefined-ish — prefer one.

- **Whitespace stability depends on the dedent.** `_emit_action` (lines 281-289) dedents before re-indenting; without it, every save adds four leading spaces. Don't "simplify" that dedent away.

- **`bool` must be checked before `int` in `_py_repr`.** Line 54 deliberately checks `isinstance(value, bool)` first because `bool ⊂ int` in Python; reordering would print `True`/`False` as `1`/`0`.

- **Builder split is inferred from `spatial_dim`, not declared.** `render_theory_file` chooses `SpatialTheoryBuilder` iff any physical field has `spatial_dim ≥ 1` (lines 370-372). A spatial theory whose fields forgot `spatial_dim` would be emitted as temporal. The AST loader is forgiving — it accepts `TheoryBuilder`, `SpatialTheoryBuilder`, *and* `TemporalTheoryBuilder` (line 652).

- **`save_theory_to_file` extension logic has a subtlety.** A path ending in `.py` but not `.theory.py` is left untouched (line 516-518), so `save_theory_to_file(spec, 'foo.py')` writes `foo.py` (not `foo.theory.py`), which `list_theories()` would then *not* discover (it requires the `.theory.py` suffix, `daedalus.py:65`).

- **`recommended_external_fields` are tuples.** They render and reload as tuples; downstream code expects `(field_name, population_index)`. Authoring them as lists in a hand-written `METADATA` would change the type on reload.

- **`DEFAULT_FUNDAMENTAL` can be empty.** Several theories (`ou_quartic_colored`, `quadratic_hawkes_alpha`) carry defaults on each `.parameter(default=...)` and leave `DEFAULT_FUNDAMENTAL = {}`. Both styles are valid; the run resolves the working point by merging.

- **`load_spec_from_file` raises on malformed files.** `ast.parse` raises `SyntaxError`; a missing `build()` or a `build()` not returning a call raises `ValueError` (lines 581, 585). The UI must catch these.

---

## Glossary

- **MSR-JD** — Martin–Siggia–Rose–Janssen–De Dominicis: a path-integral formulation of stochastic (Langevin) dynamics in which each physical field gets a conjugate *response* field.
- **Response field / tilde field** — the conjugate field `x̃` (written `xt`/`ht`/`nt` in code) that enforces the equation of motion and carries the noise vertex.
- **Action `S`** — the exponent of the MSR-JD path integral; in code, the string passed to `.set_action_text(...)`.
- **Propagator** — the inverse of the quadratic part of the action; the tree-level two-point function.
- **Cumulant** — a connected correlation function; `k` counts its external legs.
- **Loop order `ℓ` (`max_ell`)** — order in the perturbative loop expansion; `ℓ=0` tree, `ℓ=1` one-loop, …
- **Saddle / mean field `x*`** — the deterministic fixed point the theory is expanded around.
- **DAE** — differential-algebraic equation; the `.equation(lhs, rhs, population)` form used to fix the saddle.
- **CGF term** — cumulant-generating-function term; how colored (correlated) noise enters, via `.declare_cgf_term(...)` with an `order`, `coefficient`, and time `kernel`.
- **Markovianization** — embedding a colored-noise kernel into an auxiliary Markovian field so the system becomes local in time; toggled by `.markovianize(...)`.
- **Operator IR** — the v2 "intermediate representation" where derivative operators are *call-syntax* nodes (`Dt(h)`, `Lap(h)`, `Dx(h,axis)`) so derivative *vertices* carry momentum form factors; gated by `.operator_ir()`.
- **Form factor** — the momentum-dependent factor a derivative vertex contributes to a loop integral; per-leg `(∂φ)²` vs composite `∂(φ²)` give different ones.
- **Spec dict** — the flat dict the UI uses internally and that `theory_serialize` renders to / parses from.
- **Slug** — a filesystem-safe lowercase identifier derived from the theory name (`re.sub(r'[^A-Za-z0-9]+','_',name).strip('_').lower()`).
- **`ast`** — Python's standard *Abstract Syntax Tree* module; parses source to a node tree without running it.
- **`ast.literal_eval`** — safely evaluate a node *only* if it is a literal (number/string/list/dict/tuple/bool/None); never runs code.
- **`importlib.util.spec_from_file_location` / `module_from_spec` / `exec_module`** — the standard-library way to import a Python file by arbitrary path and run it.
- **SageMath / `.sobj`** — a computer-algebra system; `.sobj` is its pickled-object file format (`sage.all.save`/`load`).
- **`.theory.py`** — the human-readable theory file: `build()` + `DEFAULT_FUNDAMENTAL` + `METADATA`.
- **`saved_theories/<slug>/`** — the *binary* per-theory cache directory (`.sobj` + `manifest.json`); not the text format.
- **`manifest.json`** — the cache index listing the `.sobj` entries with their `stage`, `k`, `loop_order`, `saved_at`.

---

## Proposed manual subsections

1. **Why theory files exist** — the single-source-of-truth principle; UI ↔ file ↔ run.
2. **Anatomy of a `.theory.py`** — `build()`, `DEFAULT_FUNDAMENTAL`, `METADATA`, walked line by line on `ou_quartic`.
3. **The builder DSL** — every chainable method (`physical_field`, `parameter`, `define_function`, `define_kernel`, `declare_cgf_term`, `set_action_text`, `equation`/`set_mf_equation`, the spatial toggles).
4. **Temporal vs spatial; the builder split** — `TemporalTheoryBuilder` vs `SpatialTheoryBuilder`, and how `spatial_dim` drives the choice.
5. **Indexing & populations** — homogeneous vs heterogeneous; `population`/`indexed_by` vs legacy `n_populations`/`indexed`.
6. **Colored noise & CGF terms** — `declare_cgf_term`, kernels, Markovianization toggles.
7. **Spatial derivative vertices & the operator IR** — `Dt/Lap/Dx` call syntax, why `.operator_ir()` is load-bearing, KPZ vs Burgers vs Model-B.
8. **The spec dict** — its full shape, as the contract between the UI and `theory_serialize`.
9. **Forward serialization** — `render_theory_file` / `save_theory_to_file`, the `_emit_*` helpers, the "textually quiet" principle.
10. **Reverse serialization** — `load_spec_from_file`, the AST walk, round-trip fidelity and its limits.
11. **Discovery & loading by name** — `daedalus.load_theory`, `list_theories`, `THEORIES_DIR`.
12. **The binary cache (`saved_theories/`)** — slug, `manifest.json`, `.sobj` layout; how it differs from the text format; `_expand_cache` and `msrjd.serialize`.
13. **Gotchas & authoring rules** — the load-bearing toggles, AST literal-only limits, slug collisions.

# Subsystem brief: The `daedalus` Engine API (user-facing orchestration)

**Slug:** `engine-api`
**Primary file:** `notebooks/daedalus.py` (1102 lines, all read in full)
**Key downstream dependency:** `pipeline.compute.compute_cumulants` (`pipeline/compute.py:108`)

> Reading note for the manual author: every code citation below is `file:line` against the
> repository as read on 2026-06-17. `daedalus.py` is the single module the demo notebooks
> import as `import daedalus as dd`. It is *thin orchestration* — it does almost no physics
> itself; its whole job is to (a) load a theory, (b) translate a small, friendly `Config` into
> the big keyword-argument call that `compute_cumulants` actually wants, and (c) take the
> heterogeneous result dictionary back out and plot it in whatever shape is natural for the
> theory's "group" (temporal vs spatial, single- vs multi-field, `k=2` vs `k≥3`).

---

## 1. Overview

### What this subsystem is, in plain language

`daedalus.py` is the **front desk** of the Feynman-diagram pipeline. A physicist sitting in a
Jupyter notebook does not want to remember the exact 14-keyword signature of the heavy engine
function, nor how to reshape its output for a plot. Instead they write three lines:

```python
import daedalus as dd
model, mod = dd.load_theory('kpz_1d')          # from theories/*.theory.py
cfg = dd.Config(k=2, max_ell=1, spatial_grid=(-6, 6, 49))
res = dd.run(model, cfg, mod)                  # k/ell/Dyson all here
dd.plot_cumulant(res, cfg, model)              # adaptable, auto-dispatched
```

(That is the module's own opening docstring, `notebooks/daedalus.py:8-12`.)

The design intent, quoted from the docstring (`notebooks/daedalus.py:1-26`), is:

> "Centralises the **load → run → plot** flow so every demo notebook is thin and uniform
> regardless of its group (temporal / spatial × single / multi-field)."

So `daedalus` provides exactly three verbs and a noun:

* **`load_theory(name)`** — read a `theories/<name>.theory.py` file, return `(model, module)`.
* **`Config(...)`** — a dataclass holding *every* choice a notebook can make.
* **`run(model, cfg, module)`** — resolve the config against the theory's defaults and call the
  real engine, `compute_cumulants`. Returns the result dict with the config and model stapled on.
* **`plot_cumulant(result, cfg, model, sim=None)`** — auto-dispatch to the right plotter.

There are also introspection helpers (`describe_model`, `field_names`, `summary`,
`list_theories`) and an *output-conversion* layer that turns connected cumulants into full or
central moments (`_assemble_moment_temporal`, the `Config.output` field).

### Where it sits in the end-to-end pipeline

```
  theories/<name>.theory.py            (declarative theory text; defines build(),
        │                               DEFAULT_FUNDAMENTAL, METADATA)
        │  load_theory()
        ▼
  (model: dict, module)  ──────────►  daedalus.Config  (notebook's choices)
        │                                   │
        └──────────────┬────────────────────┘
                       │  run()  — resolves k, max_ell, external_fields, fundamental,
                       │           Dyson override, spatial-vs-temporal grids
                       ▼
              pipeline.compute_cumulants(...)        ← THE ENGINE (Phases 1–7)
                       │  returns a big result dict
                       ▼
              run() post-processes:
                 · synthesises temporal k≥3 slices / full grid from callable total_C
                 · optionally assembles moments from cumulants (set-partition formula)
                 · staples on  _cfg / _model / _resolved
                       ▼
              plot_cumulant()  → dispatches to one of seven plotters → matplotlib Figure
```

* **What feeds it:** a *theory file* (the declarative model description — see the
  `theories/*.theory.py` subsystem) plus a notebook's `Config`.
* **What it calls:** `pipeline.compute_cumulants`, the full MSR–JD diagrammatic engine. Everything
  hard (FieldTheory Taylor expansion, propagator construction, mean-field saddle solve, diagram
  enumeration, Symanzik integration / Phase-J time integration) happens *inside* that call.
  `daedalus` never touches Sage, nauty, sympy, or numba directly.
* **What consumes its output:** the demo notebooks — they take `res` and a matched simulator dict
  `sim` and hand both to `plot_cumulant`. The `res` dict is also inspected directly
  (`res['_resolved']`, `res['mf']`, `res['spatial_info']`, …).

---

## 2. The math

The reader is assumed to know MSR–JD field theory; this section ties the *code's* knobs to the
*physics objects* they control. Plain-text math throughout (LaTeX-able).

### 2.1 Connected cumulants — the engine's native output

The pipeline computes **connected k-point cumulants** of the physical field(s). For a single field
φ(x, t) the order-k connected cumulant is

```
  κ_k(x_1, t_1; … ; x_k, t_k)  =  ⟨ φ(x_1,t_1) … φ(x_k,t_k) ⟩_c
```

the connected (cluster) part of the k-point correlation. `daedalus` calls `compute_cumulants` with
a chosen `k` (number of external legs) and `max_ell` (loop order). The diagrammatic expansion is
organised by **loop order ℓ**:

```
  κ_k  =  κ_k^{(0)}  +  κ_k^{(1)}  +  κ_k^{(2)}  + …
          └ tree ┘    └ 1-loop ┘    └ 2-loop ┘
```

`Config.max_ell = L` truncates this at ℓ ≤ L. The engine returns the *per-order* pieces as
`*_by_ell` dictionaries `{ℓ: array_or_callable}`, and the cumulative sum `Σ_{ℓ≤L}` as the "total".
`daedalus`'s `cumulative_curves` (`notebooks/daedalus.py:713`) and `_order_label`
(`notebooks/daedalus.py:707`) exist precisely to render this ℓ-progression
("tree", "tree+1loop", …).

### 2.2 The k=2 case — the correlation function C(τ) and C(x,τ)

For `k=2` the connected cumulant is just the (connected) two-point function. In the **temporal**
(time-only) case the engine returns it as a curve over a lag grid τ:

```
  C(τ)  =  ⟨ φ(0) φ(τ) ⟩_c          (key 'C_tau' over 'tau_grid')
```

In the **spatial** case (a field with `spatial_dim ≥ 1`) the engine returns a real-space
correlator over a grid of separations x and lags τ:

```
  C(x, τ)  =  ⟨ φ(0,0) φ(x,τ) ⟩_c   (key 'C_tau_x', shape (n_τ, n_x))
```

The equal-time slice C(x, 0) is the headline spatial plot.

### 2.3 The k≥3 case — a function of k−1 time differences

By time-translation invariance the connected k-point cumulant depends only on the **k−1 time
differences** relative to one anchored leg:

```
  τ_j = t_j − t_0,      j = 1 … k−1     (leg 0 pinned at τ = 0)
```

So κ_k is a function κ_k(τ_1, …, τ_{k−1}). For k=2 that is one variable (the familiar C(τ)); for
k=3 it is a 2-D surface κ_3(τ_1, τ_2); for k≥4 it is a (k−1)-dimensional tensor that cannot be
drawn directly. `compute_cumulants` does **not** materialise this whole tensor for k≥3 — it returns
`C_tau = None` and instead a *callable* `total_C(*tau)` plus per-loop callables
`total_C_by_ell = {ℓ: callable}`. `daedalus.run` then synthesises two derived views (see §4 `run`):

* **Axis-parallel slices** (the default): for each leg j, sweep τ_j over the grid while pinning the
  other legs at a fixed *base* point (default all zeros):

  ```
    slice_j(τ)  =  κ_k(τ_1=base_1, …, τ_j=τ, …, τ_{k−1}=base_{k−1})
  ```

  For a single symmetric field all k−1 slices coincide — a visible check of the cumulant's
  permutation symmetry (noted at `notebooks/daedalus.py:856-867`). The canonical single slice
  `slice_1` is stored as `C_tau` so that k≥3 plots and sim-compares exactly like k=2.

* **Full grid** (opt-in via `Config.kpoint_full_grid`): the whole (k−1)-dimensional tensor
  κ_k(τ_1,…,τ_{k−1}), downsampled so the total n^{k−1} evaluations stay bounded
  (`notebooks/daedalus.py:650-670`). For k=3 this becomes the heatmap κ_3(τ_1, τ_2).

The matching simulator estimates the same slice with `lag_bins=[0, None, 0, …]` (only the swept leg
free) — documented at `notebooks/daedalus.py:588-594`.

### 2.4 Cumulants → moments (the set-partition / cluster expansion)

The engine produces *connected* cumulants κ. The full (raw) and central **moments** are
reconstructed by the **set-partition (cluster) expansion**. For the centred field, the raw k-point
moment is the sum over all set partitions π of {1,…,k} of products of cumulants over the blocks:

```
  M_k(x_1,…,x_k)  =  Σ_{π}  ∏_{B ∈ π}  κ(B)
```

This is stated verbatim in the `Config.output` comment (`notebooks/daedalus.py:261-266`) and in the
`_assemble_moment_temporal` docstring (`notebooks/daedalus.py:390-408`). Two physics subtleties the
code handles explicitly:

1. **Shared loop budget.** A naive "product of fully-dressed cumulants" would smuggle in spurious
   cross-terms — e.g. a 1-loop·1-loop product is really a 2-loop object. The code instead enforces a
   **single shared loop budget** L = `max_ell` across all multi-element blocks of a partition:

   ```
     M_k  =  Σ_π  Σ_{(ℓ_B): Σ_B ℓ_B ≤ L}  ∏_{B∈π} κ(B)^{(ℓ_B)}
   ```

   so every surviving term has *total* loop order ≤ L. The two definitions agree only at tree
   (L=0) and diverge at L≥1 for any multi-block partition (`notebooks/daedalus.py:398-408`). This
   is the `itertools.product(range(L+1), repeat=len(multis))` loop with the `sum(assign) > L`
   guard at `notebooks/daedalus.py:476-482`.

2. **Singletons = the tree mean.** A singleton block {i} contributes the field mean ⟨φ⟩, which at
   tree level is the mean-field saddle φ* (key `_external_mean`, `notebooks/daedalus.py:373`). For a
   *central* moment all singletons are dropped (only no-singleton partitions survive,
   `notebooks/daedalus.py:466`). For a *raw* moment singletons contribute μ^(#singletons), and if the
   mean is zero those terms die (`notebooks/daedalus.py:468-472`). The 1-loop tadpole shift of the
   mean is **not** yet folded in — a documented refinement (`notebooks/daedalus.py:376-377`,
   `notebooks/daedalus.py:404-405`).

`Config.output ∈ {'cumulant', 'moment', 'central_moment'}` selects this. Producing a moment costs
**k−1 extra backend runs** (one per cumulant order 2..k), because each block size needs its own
`compute_cumulants` call (`notebooks/daedalus.py:264-266`, implementation §4 `_assemble_moment_temporal`).

### 2.5 The mean-field saddle and "fundamental"/"parameters"

The pipeline first solves the **mean-field saddle** — the deterministic fixed point φ* about which
the fluctuation expansion is taken. The numeric values of the model's parameters (μ, D, λ, …) are
passed as the `fundamental` dict (engine name) = `Config.parameters` (friendly name). Saddle
parameters (names ending in `star`) are *solved by the pipeline*, not supplied. Some theories
(double-well, μ<0) have **multiple stable roots**; `Config.fixed_point_index` / `mf_dae_n_starts` /
`mf_dae_seed_box` select which root the expansion centres on (forwarded at
`notebooks/daedalus.py:559-564`).

### 2.6 Dyson dressing (coupled, unequal-diffusivity fields)

For coupled multi-field spatial theories where the fields have *unequal* bare diffusivities, the
tree-level propagator is dressed by resumming a geometric self-energy series (the **Dyson series**)
to a finite order. `Config.dyson_order` overrides the model's built-in Dyson policy at run time by
mutating `model['spatial']['dyson']` (`notebooks/daedalus.py:545-550`), optionally with a
`reference_diffusion` D₀ around which the dressing is organised.

---

## 3. External tools used

`daedalus.py` itself imports a deliberately small set of libraries. The heavy scientific tools
(SageMath, nauty, sympy, numba, networkx) are used **only inside `compute_cumulants`**, not in this
module. This section explains each, then is explicit about which are direct vs transitive.

### 3.1 NumPy — `import numpy as np` (`notebooks/daedalus.py:34`)

**What it is.** NumPy is the foundational Python array library: it provides the `ndarray`
n-dimensional array and fast vectorised math (sums, dot products, FFTs) implemented in C. Wherever
you see "array" in this codebase it means a NumPy `ndarray`.

**How this code uses it.** Pervasively, for everything numeric:

* Building grids: `np.linspace(g[0], g[1], int(g[2]))` materialises an `(lo, hi, n)` spatial grid
  tuple into an array (`notebooks/daedalus.py:327`); `np.asarray(g, dtype=float)` coerces a
  user-supplied list/array (`notebooks/daedalus.py:328`).
* Taking real parts and magnitudes for plotting: `np.real(...)`, `np.abs(...)`, `np.argmin(...)`
  (e.g. `i0 = int(np.argmin(np.abs(tau)))` picks the τ≈0 row, `notebooks/daedalus.py:1007`).
* 1-D interpolation of cumulant slices onto requested lags:
  `np.interp(abs(lag), ax, ay)` inside `_by_ell_2pt` (`notebooks/daedalus.py:428`).
* Allocating complex output buffers: `np.empty(tau.size, dtype=complex)`
  (`notebooks/daedalus.py:460`, `:625`).
* Sorting lags so interpolation is monotone: `np.argsort(a)` (`notebooks/daedalus.py:419`).

### 3.2 Matplotlib — `import matplotlib.pyplot as plt` (`notebooks/daedalus.py:35`)

**What it is.** Matplotlib is the standard Python 2-D plotting library; `pyplot` is its
MATLAB-style stateful interface. A `Figure` is the whole canvas; an `Axes` is one subplot. Methods
like `ax.plot`, `ax.bar`, `ax.errorbar`, `ax.imshow`/`ax.pcolormesh` draw curves, bars, error bars,
and heatmaps respectively.

**How this code uses it.** Every plotter calls `fig, ax = plt.subplots(...)` and returns the
`Figure`. Concrete calls:

* `plt.subplots(figsize=...)` / `plt.subplots(1, ncol, ...)` — create the figure/axes grid
  (`notebooks/daedalus.py:768`, `:1012`, `:874`).
* `ax.plot(...)` — line curves (theory) (`notebooks/daedalus.py:828`).
* `ax.errorbar(...)` — simulator points with error bars (`notebooks/daedalus.py:839`).
* `ax.bar(...)` — per-loop-order bars for temporal-k≥3 scalars and spatial k≥3 events
  (`notebooks/daedalus.py:974`, `:1067`).
* `ax.imshow(...)` — the C(x,τ) heatmap (`notebooks/daedalus.py:1041`).
* `ax.pcolormesh(...)` — the κ_3(τ_1,τ_2) heatmap (`notebooks/daedalus.py:935`).
* `ax.axhline`/`ax.axhspan` — a scalar sim value ± error band on the bar charts
  (`notebooks/daedalus.py:983-986`).
* `fig.savefig(path, dpi=130, bbox_inches='tight')` when `Config.save` is set
  (`notebooks/daedalus.py:797`).

### 3.3 Standard-library modules used directly

* `import os` (`:29`) — filesystem path manipulation in `repo_root`, `list_theories`,
  `load_theory`: `os.path.dirname`, `os.path.abspath`, `os.path.join`, `os.path.isdir`,
  `os.path.isfile`, `os.listdir`.
* `import importlib.util` (`:30`) — **dynamic import of a theory file by path** (see §3.4).
* `from dataclasses import dataclass, field` (`:31`) — `@dataclass` decorator that auto-generates
  `__init__`/`__repr__` for `Config`. (Note: `field` is imported but, as read, never used — see
  open questions.)
* `from typing import Any, Optional` (`:32`) — type hints only; no runtime effect.
* `import sys as _sys` (`:56`) — to insert `REPO_ROOT` onto `sys.path` so `import pipeline` works
  from any notebook subdirectory.
* `import itertools` — imported *locally* inside `_assemble_moment_temporal`
  (`notebooks/daedalus.py:409`) and inside the full-grid branch of `run`
  (`notebooks/daedalus.py:651`); used for `itertools.product` (cartesian product of loop-order
  assignments / grid indices).

### 3.4 `importlib.util` — dynamic loading of a `.theory.py` file (the load_theory mechanism)

**What it is.** `importlib` is Python's programmatic import system. `importlib.util` lets you import
a module *from an arbitrary file path* — not just by name from `sys.path`. The three-step idiom is:

```python
spec = importlib.util.spec_from_file_location(modname, path)   # describe the module
mod  = importlib.util.module_from_spec(spec)                   # create an empty module object
spec.loader.exec_module(mod)                                   # actually run the file's top-level code
```

**How this code uses it.** Exactly that idiom, in `load_theory` (`notebooks/daedalus.py:77-79`):

```python
spec = importlib.util.spec_from_file_location(f'theories.{name}', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
return mod.build(), mod
```

This is how a theory file (a plain Python module with `build()`, `DEFAULT_FUNDAMENTAL`, `METADATA`)
becomes a live `(model, module)` pair without the theory needing to be on the import path.

### 3.5 Transitive heavy dependencies (NOT touched by daedalus directly)

`daedalus.run` and `_assemble_moment_temporal` do `from pipeline import compute_cumulants`
(`notebooks/daedalus.py:410`, `:492`). Everything below happens **inside** that call; the manual
should make clear that `daedalus` is a *client* of them, not a user:

* **SageMath (Sage).** A computer-algebra system (CAS) built on Python. The pipeline uses Sage's
  symbolic ring (`SR`), symbolic matrices, and exact algebra to build the propagator, find ω-poles
  and residues, and solve the mean-field self-consistency exactly. In the result dict the keys
  `num_params` ("{SR symbol: float}") and the `propagator` matrices are Sage objects
  (`pipeline/compute.py:226-231`). `daedalus` only ever reads *numpy* arrays out of the result, so
  it never has to know Sage exists — except that the spatial branch deliberately *avoids* the Sage
  ω-pole-finder because it cannot consume the inert `Laplacian` symbol (`pipeline/compute.py:382-388`,
  `:619-626`).
* **nauty.** A C library (wrapped in Python) for **graph canonicalisation and isomorphism** — given
  a graph it computes a canonical labelling so that two diagrams that are the same graph up to
  relabelling are recognised as identical. The pipeline uses it to deduplicate enumerated Feynman
  prediagrams and to compute automorphism (symmetry) factors. Surfaces to `daedalus` only as the
  diagram counts in `spatial_info['n_diagrams']` / `n_live_diagrams`, which `summary` prints
  (`notebooks/daedalus.py:1098-1100`).
* **sympy.** A pure-Python symbolic-math library (lighter than Sage). Used in parts of the engine
  for symbolic manipulation and `lambdify` (turning a symbolic expression into a fast numeric
  Python/NumPy function). Not referenced by name in `daedalus`.
* **scipy.** Scientific-computing routines on top of NumPy (integration, optimisation, special
  functions). The engine uses `scipy.integrate.quad` as a fallback for 1-D mode-sum integrals and
  Bessel functions for the spatial radial transform. Not referenced in `daedalus`.
* **numba.** A just-in-time (JIT) compiler that turns numeric Python functions into fast machine
  code. Used in the engine's hot inner loops (chamber integrals, einsums). Not referenced in
  `daedalus`.
* **networkx.** A pure-Python graph library (nodes/edges, traversal, connectivity). Used in diagram
  enumeration/connectivity checks. Not referenced in `daedalus`.

The single most important takeaway for the manual: **`daedalus.py` is pure orchestration; the only
non-stdlib libraries it imports are NumPy and Matplotlib.** All CAS/graph/JIT machinery is behind
the one-line `from pipeline import compute_cumulants`.

---

## 4. Components

Exhaustive walkthrough of every significant function/class/dataclass in `notebooks/daedalus.py`,
in file order.

### 4.1 `repo_root() -> str` — `notebooks/daedalus.py:40`

**Signature:** `repo_root() -> str`. Takes nothing; returns the absolute path of the repository
root as a string.

**What it does, step by step.**
1. Find a starting directory: the directory of *this file* if `__file__` is defined, else the cwd
   (`os.path.abspath('')`) — the fallback covers running the code pasted into a bare notebook cell
   (`notebooks/daedalus.py:44-45`).
2. Walk *up* the directory tree (`root = os.path.dirname(root)`) until a directory containing a
   `pipeline/` subdirectory is found; return it (`:47-50`).
3. If the walk reaches the filesystem root without finding `pipeline/`, return the starting `here`
   as a last resort (`:51`).

This is what makes the notebooks robust to being run from any `notebooks/{legacy,temporal,spatial}/`
subdirectory.

Module-level consequence (`notebooks/daedalus.py:54-58`): `REPO_ROOT = repo_root()`,
`THEORIES_DIR = REPO_ROOT/theories`, and `REPO_ROOT` is prepended to `sys.path` so `import pipeline`
resolves.

### 4.2 `list_theories() -> list[str]` — `notebooks/daedalus.py:61`

**Signature:** `list_theories() -> list[str]`. Returns the sorted list of available theory names.

**What it does.** Lists `THEORIES_DIR`, keeps files ending in `.theory.py`, strips that suffix, and
sorts. So `kpz_1d.theory.py` → `'kpz_1d'`. Used both as the menu of loadable theories and inside the
`FileNotFoundError` message of `load_theory`.

### 4.3 `load_theory(name: str)` — `notebooks/daedalus.py:68`

**Signature:** `load_theory(name: str) -> (model: dict, module)`.

**What it takes:** `name`, the bare theory name (no `.theory.py`).
**What it returns:** a 2-tuple `(model, module)` where `model = module.build()` (a dict) and
`module` is the live imported theory module (exposing `DEFAULT_FUNDAMENTAL` and `METADATA`).

**Step by step.**
1. Build the path `THEORIES_DIR/<name>.theory.py` (`:73`).
2. If it doesn't exist, raise `FileNotFoundError` listing the available theories (`:74-76`).
3. Dynamically import the file via the `importlib.util` three-step idiom (`:77-79`, see §3.4).
4. Return `(mod.build(), mod)` (`:80`).

**Why two return values:** the model dict drives the run; the module is needed separately because
`METADATA` (run recommendations: `k_default`, `ell_default`, `tau_max`, `recommended_external_fields`,
…) and `DEFAULT_FUNDAMENTAL` (numeric parameter defaults) live on the *module*, not in the model dict.

### 4.4 `is_spatial(model) -> bool` — `notebooks/daedalus.py:85`

Returns `True` iff `model['spatial']` exists and has a truthy `dim`. The single predicate that
routes everything spatial vs temporal.

### 4.5 `spatial_dim(model) -> int` — `notebooks/daedalus.py:90`

Returns `int(model['spatial']['dim'])` or 0. The number of spatial dimensions d.

### 4.6 `is_multifield(model) -> bool` — `notebooks/daedalus.py:95`

**True iff the theory has more than one independent field channel.** Two ways to qualify
(`notebooks/daedalus.py:99-102`): more than one entry in `model['physical_fields']`, OR any
population with `size > 1`. Drives the plotting (multi-field → per-component grid).

### 4.7 `field_names(model) -> list[str]` — `notebooks/daedalus.py:105`

Returns `[f['name'] for f in model['physical_fields']]`. Used to build default external legs and to
label/locate the mean field.

### 4.8 `_fmt_default(v, maxlen=52) -> str` — `notebooks/daedalus.py:109`

A `repr()` truncated to `maxlen` chars (with a trailing ellipsis). Cosmetic helper for
`describe_model`'s parameter table.

### 4.9 `_doc_physics(doc) -> str` — `notebooks/daedalus.py:125` (+ module constants `_DOC_DROP` `:116`, `_DOC_STOP` `:121`)

**Purpose:** trim a theory's docstring down to its *physics prose*, dropping UI/provenance
boilerplate and the dev-status/cross-reference tail. `_DOC_STOP` lines (e.g. `'Phase 1 status'`,
`'status:'`, `'VERBATIM'`) *terminate* the kept block; `_DOC_DROP` substrings (e.g. `'Generated by'`,
`'.theory.py'`, `'docs/'`, ``'``build()``'``) cause that individual line to be skipped. The result is
the leading physics paragraph(s) only. Used by `describe_model` when `show_doc=True`.

### 4.10 `describe_model(model, module=None, show_doc=True) -> str` — `notebooks/daedalus.py:138`

**Signature:** `describe_model(model: dict, module=None, show_doc: bool = True) -> str`.
**What it takes:** the model dict; optionally the theory module (to surface its docstring + METADATA
run recommendations); `show_doc` toggles appending the physics docstring.
**What it returns:** the formatted multi-line text — and it **also prints it** (`:241`).

This is the canonical "what is this model?" cell. Step by step it builds a boxed report
(`notebooks/daedalus.py:147-242`):
1. A header bar with `model['name']` (`:148-151`).
2. **Domain line:** spatial PDE (`d=…`, boundary, initial) via the local `_mode` helper that reads
   either a `{'mode': …}` dict or a bare value (`:153-163`), else "temporal ODE (time-only)"
   (`:164-165`).
3. **Fields line:** one entry per physical field, via the local `_fld` helper showing natural name,
   population tag, `x∈ℝ^sd` spatial annotation, and description (`:167-177`).
4. **Response fields** (if any) (`:178-182`).
5. **Populations** (skipping the auto scalar population named `'pop'`) with size and description
   (`:184-190`).
6. **Parameters:** split into *numeric* (default-valued) and *saddle* (names ending `star`,
   "solved by the pipeline"). Numeric params print name, index annotation, default (via
   `_fmt_default`), and domain (`:192-206`).
7. **Kernels** (non-Markovian temporal convolutions) and **Functions** (n-arg transfer functions)
   (`:208-217`).
8. **Governing equation(s):** `lhs_text = rhs_text`, but *skipped* when the rhs is empty/`'0'`/`'0.0'`
   — the rationale (linear dynamics, or the nonlinearity lives in the action text like KPZ's
   gradient vertex) is in the comment at `:222-224`.
9. **Suggested run:** from `module.METADATA` — `k=…, max_ell=…` (`:228-235`).
10. Optionally appends `_doc_physics(module.__doc__)` (`:238-240`).

Important: it reads *only the model dict* (and optionally the module's METADATA/docstring), so it
stays model-independent (`:145-146`).

### 4.11 `class Config` (dataclass) — `notebooks/daedalus.py:247`

The single most important object for the manual. `@dataclass` means Python auto-generates the
constructor from the field declarations. **Every field, in declaration order:**

| Field | Default | Meaning (line) |
|---|---|---|
| `k` | `None` | Correlator order (number of external legs). If `None`, inferred from `external_fields`, else from `METADATA['k_default']` (`:255-256`). |
| `max_ell` | `None` | Loop order L (0=tree). `None` ⇒ `METADATA['ell_default']` (`:256-257`). |
| `external_fields` | `None` | The k legs, e.g. `[('x', 1), ('x', 1)]`. May name a response leg (`:258`). |
| `parameters` | `None` | Numeric parameter overrides by name — the canonical name (`:259`). |
| `fundamental` | `None` | **Deprecated alias** for `parameters` (`:260`); see `__post_init__`. |
| `output` | `'cumulant'` | `'cumulant'`\|`'moment'`\|`'central_moment'` — see §2.4 (`:261-266`). |
| `tau_max` | `None` | Temporal grid extent; `None` ⇒ METADATA/10.0 (`:269`). |
| `tau_step` | `None` | Temporal grid spacing; `None` ⇒ METADATA/0.5 (`:270`). |
| `spatial_grid` | `None` | Spatial separations: an array, an `(lo, hi, n)` tuple, or `None` (`:273`). |
| `spatial_points` | `None` | **k≥3 spatial only**: `(n_pts, k−1, 2)` of `(x_j, τ_j)` per non-anchor leg (`:274`). |
| `kpoint_base_lags` | `None` | k≥3 temporal: length k−1, the fixed τ for the non-swept legs (default all 0) (`:280-282`). |
| `kpoint_full_grid` | `False` | k≥3 temporal: compute the full (k−1)-dim tensor instead of slices (`:283-285`). |
| `dyson_order` | `None` | `None`=leave model policy; int≥0=override (`:288`). |
| `reference_diffusion` | `None` | D₀ reference for the Dyson dressing (`:289`). |
| `fixed_point_index` | `None` | Which stable mean-field root (0,1,…) for multi-root saddles (`:296`). |
| `mf_dae_n_starts` | `None` | Multi-start Newton count for the DAE saddle solve (`:297`). |
| `mf_dae_seed_box` | `None` | `{field: (lo, hi)}` start range for the saddle solve (`:298`). |
| `parallel` | `False` | Execution parallelism flag forwarded to the engine (`:301`). |
| `verbose` | `False` | Print engine progress (`:302`). |
| `show_orders` | `'cumulative'` | Plot mode: `'cumulative'`\|`'incremental'`\|`'total'` (`:305`). |
| `logy` | `False` | Log y-axis (`:306`). |
| `components` | `None` | Which (i,j)/slice to draw; `None`=auto (`:307`). **Declared but never read in this file — see open questions.** |
| `figsize` | `None` | Matplotlib figure size tuple (`:308`). |
| `title` | `None` | Plot title override (`:309`). |
| `save` | `None` | Path to `savefig`, or `None` (`:310`). |

**`__post_init__(self)`** (`notebooks/daedalus.py:312-319`): keeps `parameters` and `fundamental` in
sync — whichever the caller set is mirrored onto the other; `parameters` wins if both are given. So
old notebooks passing `fundamental=` and new ones passing `parameters=` both work.

**`resolved_grid(self)`** (`notebooks/daedalus.py:321-328`): materialises `spatial_grid`. If `None`,
returns `None`; if a 3-tuple `(lo, hi, n)`, returns `np.linspace(lo, hi, int(n))`; otherwise coerces
to a float array. This is the bridge from the friendly tuple form to the array the engine wants.

### 4.12 `_meta(mod, key, default=None)` — `notebooks/daedalus.py:331`

One-liner: `getattr(mod, 'METADATA', {}).get(key, default)`. The single accessor for theory run
recommendations. Used throughout `run`/`describe_model`.

### 4.13 `fundamental_from_model(model) -> dict` — `notebooks/daedalus.py:335` (alias `parameters_from_model` `:352`)

**Builds a `fundamental` dict from the parameters' own `default` values baked into the model.** Every
non-saddle parameter (name not ending `star`) that has a declared `default` is included; saddle
params and defaultless params are skipped (`:342-347`). This is the **bottom layer** of the layered
override (§6) that lets a loaded theory run out of the box even when its `DEFAULT_FUNDAMENTAL` is
empty. `parameters_from_model` is the canonical alias kept for existing imports.

### 4.14 `_set_partitions(items)` — `notebooks/daedalus.py:357`

A recursive generator that **yields every set partition** of a list as a list of blocks. The
standard recursion: partition the rest, then for each partition either insert the first element into
an existing block or start a new singleton block (`:366-369`). Drives the moment assembly. (For k=4
there are 15 partitions — the Bell number B_4.)

### 4.15 `_external_mean(model, res) -> float` — `notebooks/daedalus.py:373`

Returns ⟨φ⟩ for the single external field — the mean-field saddle φ* (tree-level mean; 0 for a
symmetric saddle). It reads `res['mf_values']`, finds the first field name (and, if it starts with
`'d'`, also tries the stripped form — to handle the `d`-prefixed response/fluctuation naming), and
returns the real part of the first value, defaulting to 0.0 (`:380-387`). The 1-loop tadpole shift
of the mean is not folded in (`:376-377`).

### 4.16 `_assemble_moment_temporal(model, res, kw, k, central)` — `notebooks/daedalus.py:390`

**The moment-assembly workhorse (temporal).** Signature: `(model, res, kw, k, central)` where `kw`
is the keyword dict already built for `compute_cumulants`, `k` the order, `central` a bool. Returns
a complex array `M` over `res['tau_grid']`.

**Step by step:**
1. Local imports `itertools`, `from pipeline import compute_cumulants` (`:409-410`).
2. Read the τ-grid and the first external leg `f0`; loop budget `L = max_ell`; the singleton mean
   `mu` (0 if central, else `_external_mean`) (`:411-414`).
3. **`_by_ell_2pt(r)`** (`:416-429`): returns `{ℓ: interpolator}` for the 2-point cumulant's ℓ-loop
   piece. It sorts by `|lag|` and builds a `np.interp` closure per loop order (or a zero closure if
   that order is absent). This lets the assembly evaluate κ_2^{(ℓ)} at any lag.
4. **κ_2 per order** (`:431-436`): if `k==2`, reuse `res`; else do one extra
   `compute_cumulants` run with `k=2, external_fields=[f0]*2`.
5. **κ_j per order for j=3..k** (`:438-448`): for each block size j, get its `total_C_by_ell`
   callables (reusing the order-k run when `j==k`, else one fresh run with `k=j` legs).
6. **`_kappa(block, ell, ti)`** (`:452-458`): the ℓ-loop piece of the |block|-point cumulant at the
   slice times — leg 1 swept to `tau[ti]`, the rest pinned at 0. For 2-blocks it uses the
   interpolators; for larger blocks it calls the j-point callable with the appropriate time vector.
7. **The partition sum** (`:460-483`): for each τ-index `ti`, loop over all set partitions; split
   each into singletons and multi-blocks. If `central` and there are singletons, skip the partition.
   Compute `mu_factor = mu**len(singles)`; if singletons but the mean is zero, the term dies. If no
   multi-blocks (all-singleton, raw), add `mu_factor`. Otherwise distribute the loop budget across the
   multi-blocks with `itertools.product(range(L+1), repeat=len(multis))` keeping only assignments with
   `sum ≤ L`, multiply the per-block ℓ-pieces times `mu_factor`, and accumulate. This implements the
   §2.4 shared-loop-budget formula exactly.

### 4.17 `run(model, cfg, module=None) -> dict` — `notebooks/daedalus.py:487`

**The central orchestration function.** Resolves `cfg` against theory defaults, calls
`compute_cumulants`, post-processes, and stamps the result. Returns the engine's result dict with
`_cfg`, `_model`, `_resolved` added.

**Step-by-step flow:**

1. **Import the engine** (`:492`): `from pipeline import compute_cumulants`.

2. **Resolve `max_ell`** (`:494-495`): `cfg.max_ell` if set, else `METADATA['ell_default']`, else 0.

3. **Resolve k (the crux)** (`:499-515`):
   - If `cfg.k` is set, k = cfg.k. *If* `external_fields` is also explicit and its length ≠ k, raise
     a `ValueError` — a k/legs mismatch is treated as a contradiction, not silently fixed (`:506-511`).
   - Else if `external_fields` is explicit, `k = len(ext)` (a k-point correlator has exactly k legs).
   - Else `k = METADATA['k_default']` (or 2).

4. **Resolve `external_fields`** (`:517-533`): if none given and a module exists, try
   `METADATA['recommended_external_fields']`. Then **auto-build** k copies of the first physical
   field's leg `(f0, 1)` *only when the caller gave none explicitly* AND the candidate is absent / has
   the wrong length / names a missing field. An explicit `Config.external_fields` is used verbatim
   (it may legitimately name a response leg). The validity check uses `field_names(model)` and the
   local `_nm` helper to pull the field name out of a tuple/string.

5. **Layered `fundamental`** (`:534-542`): start from `fundamental_from_model(model)` (model param
   defaults), update with `module.DEFAULT_FUNDAMENTAL`, then update with `cfg.parameters`. Each layer
   wins over the previous (§6).

6. **Dyson override** (`:544-550`): if `cfg.dyson_order is not None` *and* the model is spatial, write
   `model['spatial']['dyson'] = {'mode':'fixed','order':int(cfg.dyson_order)}` and, if given,
   `reference_diffusion`. **Note this mutates the model dict in place.**

7. **Build the `kw` dict** (`:552-553`) for `compute_cumulants`: `model, k, max_ell, fundamental,
   external_fields, parallel, verbose`.

8. **Forward MF root-selection overrides** (`:555-564`) only when set, so the engine's own defaults
   (`fixed_point_index=0`, `mf_dae_n_starts=64`, `mf_dae_seed_box=None`) are preserved otherwise.

9. **Spatial vs temporal grid wiring** (`:566-584`):
   - Spatial: if `k≠2` and no `spatial_points`, raise (k≥3 spatial needs explicit events). If `k≠2`,
     set `kw['spatial_points']`. Else (k=2) materialise the grid (default `np.linspace(-6,6,49)`) and
     set `tau_max`/`tau_step` (defaulting to 0.0 / 1.0 for spatial — equal-time-centred).
   - Temporal: `tau_max`/`tau_step` from cfg or `METADATA` (defaults 10.0 / 0.5).

10. **Call the engine** (`:586`): `res = compute_cumulants(**kw)`.

11. **Temporal k≥3 slice/grid synthesis** (`:588-672`): when not spatial, `k≥3`, and the engine
    returned `C_tau=None` with a callable `total_C`:
    - Choose a τ-grid (cap to 41 points to bound 1-D evaluations) (`:597-603`).
    - Resolve the `base` (the fixed point the axis-parallel slices pass through) from
      `cfg.kpoint_base_lags`, validating its length is k−1 (`:608-616`).
    - Local helpers `_args(vals)` (build the k-length time vector with leg 0 = 0) and `_slice(fn, j)`
      (sweep leg j over the grid, others at base) (`:618-630`).
    - Build the k−1 axis-parallel slices and their per-ℓ versions; store `C_tau_slices`,
      `C_tau_slices_by_ell`; set the canonical `C_tau = slices[1]`, `C_tau_by_ell = slices_by_ell[1]`;
      record `_kpoint_base` and a human-readable `_kpoint_slice` description (`:632-646`).
    - If `cfg.kpoint_full_grid`, also build the full (k−1)-dim tensor `C_tau_grid` and its per-ℓ
      version, with axis downsampling to keep n^{k−1} bounded; record `tau_axes` and a
      `_kpoint_grid_note` if downsampled (`:650-670`).
    - The whole block is wrapped in `try/except Exception: pass` (`:632`, `:671-672`) — on failure it
      leaves the None/scalar form for `plot_temporal_kpoint` to handle.

12. **Optional moment conversion** (`:674-692`): if `cfg.output ∈ {'moment','central_moment'}`:
    - Spatial k=2: `M(x) = κ_2(x) [+ μ² for raw]` from `C_tau_x` (`:681-684`).
    - Spatial k≥3: **`NotImplementedError`** (`:685-688`).
    - Temporal: `res['moment'] = _assemble_moment_temporal(...)` (`:690-691`).
    - Stamp `res['output_kind']`.

13. **Stamp metadata** (`:694-697`): attach `res['_cfg']`, `res['_model']`, and `res['_resolved']`
    (a dict with the resolved `k`, `max_ell`, `external_fields`, and `parameters`/`fundamental`).

14. Return `res`.

### 4.18 `_order_label(ell)` / `cumulative_curves(by_ell)` — `notebooks/daedalus.py:707`, `:713`

* `_order_label(ell)`: `'tree'` for ℓ=0, else `'tree+1loop+…+Nloop'`.
* `cumulative_curves(by_ell)`: turns per-order `{ℓ: array}` into running cumulative sums
  `{ℓ: Σ_{0..ℓ}}` (`:713-720`). The basis for the `'cumulative'` plot mode.

### 4.19 `plot_cumulant(result, cfg=None, model=None, sim=None)` — `notebooks/daedalus.py:725`

**The dispatcher.** Returns a matplotlib `Figure`. `cfg`/`model` default to the stamped `_cfg`/`_model`
if omitted. `sim` (optional) is a matched-simulator dict overlaid on the plot. Dispatch order
(`:738-756`):
1. If `output_kind ∈ {'moment','central_moment'}` and a `moment` is present → `_plot_moment`.
2. If `'C_kpoint' in result` → `plot_kpoint` (spatial k≥3 events).
3. If `is_spatial(model)` or `'C_tau_x' in result` → `plot_spatial`.
4. If `result['C_tau'] is None` → `plot_temporal_kpoint` (temporal k≥3 scalar bars).
5. If `result['C_tau_grid'] is not None` → `plot_temporal_kpoint_grid` (k=3 heatmap).
6. If `len(result['C_tau_slices']) ≥ 2` → `plot_temporal_kpoint_slices` (one panel per τ_j).
7. Else → `plot_temporal` (the C(τ) curve).

### 4.20 `_plot_moment(result, cfg, model, sim=None)` — `notebooks/daedalus.py:759`

Plots the assembled moment as a single curve (moments mix loop orders, so no per-order overlay).
Temporal → M_k(τ); spatial k=2 → M(x,0) (picks the τ≈0 row if 2-D). Overlays `sim` as points/errorbars.
y-label `$M_{k}$`. Honours `logy`, `title`, `save`.

### 4.21 `_orders_to_draw(by_ell, cfg)` — `notebooks/daedalus.py:801`

Resolves `Config.show_orders` into a list of `(label, curve)` for the line plotters:
`'incremental'` → per-order curves; `'total'` → only the top cumulative curve; `'cumulative'`
(default) → all cumulative curves. Returns `[]` if `by_ell` is empty.

### 4.22 `plot_temporal(result, cfg, model, sim=None)` — `notebooks/daedalus.py:815`

The temporal C(τ) plot with per-loop-order overlay. If `C_tau is None`, redirects to
`plot_temporal_kpoint`. Builds curves via `_orders_to_draw` (falling back to a single 'total' curve),
draws each in a rotating colour from `_ORDER_COLORS`, then overlays `sim` (handling a 2-D k≥3 sim by
taking slice 1). Standard axes/title/grid/legend/save.

### 4.23 `plot_temporal_kpoint_slices(result, cfg, model, sim=None)` — `notebooks/daedalus.py:856`

k≥3: one panel per time-difference τ_j. Reads `C_tau_slices` / `C_tau_slices_by_ell`. For each
slice j it draws the per-order overlay and (optionally) the matching sim row (`sim['C']` may be 2-D
`(k−1, n_τ)` — row p is panel p — or 1-D applied to panel 0). x-label `τ_j = t_j − t_0`, y-label
`κ_k`. Adds a suptitle noting the base if nonzero.

### 4.24 `plot_temporal_kpoint_grid(result, cfg, model, sim=None)` — `notebooks/daedalus.py:923`

k=3: a `pcolormesh` heatmap of κ_3(τ_1, τ_2) from `C_tau_grid` and `tau_axes`. If the grid is
missing or not 2-D (k≥4), falls back to `plot_temporal_kpoint_slices`. Colorbar labelled
`κ_3(τ_1,τ_2)`.

### 4.25 `plot_temporal_kpoint(result, cfg, model, sim=None)` — `notebooks/daedalus.py:948`

Temporal k≥3 equal-time scalar: a single number per loop order (the τ=0 value lives in
`C_tau_by_ell`). Draws tree/+loop **bars** honouring `show_orders`. A scalar sim overlays as a dashed
`axhline` ± an `axhspan` error band. y-label `κ_k (equal-time)`.

### 4.26 `plot_spatial(result, cfg, model, sim=None)` — `notebooks/daedalus.py:1000`

Spatial k=2: the equal-time C(x,0) slice (panel 0) plus, when a τ-grid is present (`len(tau)>1`), a
C(x,τ) `imshow` heatmap (panel 1). Reads `C_tau_x` (shape `(n_τ, n_x)`), `spatial_grid`, `tau_grid`,
and the per-order cumulative `spatial_info['C_by_order']`. The τ≈0 row is selected via
`argmin(|tau|)`. Overlays `sim` (`sim['x']`, `sim['C']`, `sim['C_err']`).

### 4.27 `plot_kpoint(result, cfg, model)` — `notebooks/daedalus.py:1054`

Spatial k≥3 at explicit events: a per-event bar chart from `C_kpoint`, with the tree/+loop
decomposition (`C_kpoint_by_ell` → `cumulative_curves`) drawn as grouped bars unless
`show_orders == 'total'`. x-label "evaluation point", y-label `κ_k`. (Note: no `sim` parameter.)

### 4.28 `summary(result) -> str` — `notebooks/daedalus.py:1087`

One-glance text summary: theory name, resolved `k`/`max_ell`, field names, `spatial_dim`, the Dyson
order if the model carries a fixed-mode Dyson policy, and the live-diagram count
`spatial_info['n_live_diagrams']` if present. Reads `result['_resolved']`, `result['_model']`, and
`result['spatial_info']`.

### 4.29 Module-level data: `_ORDER_COLORS` — `notebooks/daedalus.py:703`

A 7-colour palette indexed by loop order, reused across every plotter via modulo.

---

## 5. Data structures

### 5.1 The model dict (input)

Produced by `module.build()` in a theory file. Keys read by `daedalus` (non-exhaustive — the engine
reads many more): `name`, `spatial` (`{'dim', 'boundary', 'dyson', 'reference_diffusion'}`),
`physical_fields` (list of `{'name','natural_name','population','spatial_dim','description'}`),
`response_fields`, `populations` (`{'name','size','description'}`), `parameters` (`{'name','default',
'domain','indexed_by'}`), `kernels`, `functions`, `equations` (`{'lhs_text','rhs_text','population'}`),
`boundary`, `initial`, `naming_convention`. Note the auto scalar population is named `'pop'` and is
skipped in display (`notebooks/daedalus.py:185`, `:170`).

### 5.2 The theory module (input)

A live Python module exposing `build()`, `DEFAULT_FUNDAMENTAL: dict`, `METADATA: dict`, and a module
docstring. `METADATA` keys consumed: `k_default`, `ell_default`, `recommended_external_fields`,
`tau_max`, `tau_step` (and `spatial_grid` is declared in some theories but `run` defaults rather than
reads it). Example (`theories/kpz_1d.theory.py:60-67`): `k_default=2, ell_default=1,
recommended_external_fields=[('dh',1),('dh',1)], tau_max=2.0, tau_step=0.5`.

### 5.3 `Config` (dataclass) — §4.11 has the full field table.

### 5.4 The result dict (engine output, augmented by `run`)

The shape depends on the case. Keys `daedalus` reads/writes:

**Temporal k=2** (from `compute_cumulants`, see `pipeline/compute.py:214-241`):
* `total_C`: callable `f(*tau) -> complex`
* `total_C_by_ell`: `{ℓ: callable}`
* `C_tau`: ndarray over `tau_grid`
* `C_tau_by_ell`: `{ℓ: ndarray or None}`, with `C_tau == sum(values())`
* `tau_grid`: ndarray
* `mf_values`: `{internal_name: [v_pop1, …]}`
* `mf`: a `pipeline.access.MeanField` accessor; `params`: a `Parameters` accessor
* `num_params`: `{SR symbol: float}`; `propagator`: dict; `diagrams`: list; `config`: echo

**Temporal k≥3** (engine returns `C_tau=None` + callable `total_C`/`total_C_by_ell`). `run` adds:
`C_tau_slices` `{j: ndarray}`, `C_tau_slices_by_ell` `{j: {ℓ: ndarray}}`, `C_tau` (=slice 1),
`C_tau_by_ell` (=slice-1 per-ℓ), `_kpoint_base`, `_kpoint_slice`, and optionally `C_tau_grid`
(`(n,)*(k−1)` complex tensor), `C_tau_grid_by_ell`, `tau_axes` (list of k−1 axis arrays),
`_kpoint_grid_note`.

**Spatial k=2** (early return at `pipeline/compute.py:584-612`): `total_C` (1-or-2-arg callable),
`total_C_by_ell={0: total_C}`, `C_tau` (x=0 slice), `C_tau_x` `(n_τ, n_x)`,
`C_tau_x_by_order`, `tau_grid`, `spatial_grid`, `spatial_info` (the driver diagnostics dict),
`mf_values`, `mf`, `params`, `num_params`, `propagator`, `config`.

**Spatial k≥3** (early return at `pipeline/compute.py:463-470`): `C_kpoint` `(n_pts,)`,
`C_kpoint_by_ell` `{ℓ: (n_pts,)}`, `points` `(n_pts, k−1, 2)`, `spatial_info`, `k`, `max_ell`, `mf`.

**`spatial_info` sub-dict** (built in `msrjd/integration/spatial/pipeline_bridge.py`): keys read by
`daedalus` include `C_by_order` (`{ℓ: (n_τ, n_x)}` cumulative — `plot_spatial`), `n_live_diagrams`
and `n_diagrams` (`summary`), and internally `modes` (list of `(A, B, N)` mode tuples), `Sigma`
(1-loop self-energy), `bc_mode`, `L`, `spatial_dim`, `per_ell`, `pipeline_certified`,
`certify_max_rel`.

**`run`-added stamps (always):** `_cfg` (the `Config`), `_model` (the model dict), `_resolved`
(`{k, max_ell, external_fields, parameters, fundamental}`). Plus `moment` + `output_kind` when a
moment was requested.

### 5.5 The `sim` dict (optional plot overlay)

`plot_cumulant(..., sim=...)` accepts a matched-simulator dict. Temporal: `{'tau', 'C', 'C_err'}`;
spatial: `{'x', 'C', 'C_err'}`; temporal k≥3 multi-slice: `'C'`/`'C_err'` may be 2-D `(k−1, n_τ)`;
temporal k≥3 scalar: `{'C': <scalar>, 'C_err': <scalar or None>}` (`notebooks/daedalus.py:734-736`,
`:866-867`, `:980-986`).

---

## 6. Data flow

**In (a typical KPZ spatial 1-loop run):**

```python
model, mod = dd.load_theory('kpz_1d')
#   model = {'name':'1D KPZ (per-leg gradient vertex)', 'spatial':{'dim':1,...},
#            'physical_fields':[{'name':'h',...}], 'parameters':[mu,D,lam,T], ...}
#   mod.METADATA = {'k_default':2,'ell_default':1,
#                   'recommended_external_fields':[('dh',1),('dh',1)],...}
cfg = dd.Config(k=2, max_ell=1, spatial_grid=(-6, 6, 49))
res = dd.run(model, cfg, mod)
```

`run` resolves: `max_ell=1` (from cfg), `k=2` (from cfg), `external_fields=[('dh',1),('dh',1)]`
(recommended; valid), `fundamental={'mu':1.0,'D':1.0,'lam':0.3,'T':1.0}` (model defaults +
DEFAULT_FUNDAMENTAL, no cfg override), spatial grid `np.linspace(-6,6,49)`, `tau_max=0.0,
tau_step=1.0`. It then calls:

```python
compute_cumulants(model=model, k=2, max_ell=1,
                  fundamental={'mu':1.0,'D':1.0,'lam':0.3,'T':1.0},
                  external_fields=[('dh',1),('dh',1)],
                  spatial_grid=<49 pts>, tau_max=0.0, tau_step=1.0,
                  parallel=False, verbose=False)
```

**Out:** the spatial-k=2 dict (§5.4) with `C_tau_x` `(1, 49)`, `spatial_grid`, `spatial_info`
(`C_by_order`, `n_live_diagrams`, `modes`, `Sigma`, …), plus the stamps `_cfg`/`_model`/`_resolved`.

Then `dd.plot_cumulant(res, cfg, model, sim=mysim)`: `output_kind` absent → not a moment; no
`C_kpoint`; `is_spatial(model)` True → `plot_spatial`, which draws C(x,0) with the per-order overlay
from `spatial_info['C_by_order']` and overlays the simulator points.

**Temporal k=3 example flow:** `cfg = dd.Config(k=3, max_ell=1)` → `run` builds
`external_fields=[(f0,1)]*3`, calls the engine, gets `C_tau=None` + callable `total_C`, synthesises
the 2 axis-parallel slices (`C_tau_slices={1:…,2:…}`), sets `C_tau=slices[1]`. `plot_cumulant` then
hits branch 6 (`len(C_tau_slices) ≥ 2`) → `plot_temporal_kpoint_slices` (two panels). With
`kpoint_full_grid=True` it also gets `C_tau_grid` → branch 5 → `plot_temporal_kpoint_grid` heatmap.

---

## 7. Gotchas & caveats

1. **`run` mutates the model dict in place** (`notebooks/daedalus.py:545-550`): the Dyson override
   writes into `model['spatial']`. If the same `model` object is reused across runs with different
   `dyson_order`, the last write persists. Re-`load_theory` for a clean model.

2. **k vs external_fields contradiction is fatal** (`:506-511`): if you pass *both* `k` and
   `external_fields` and they disagree on leg count, `run` raises rather than silently choosing.

3. **Explicit `external_fields` is used verbatim** (`:519-533`): only an *absent* `external_fields`
   (or a stale METADATA recommendation) is auto-rebuilt. An explicit list that names a missing field
   will be passed straight to the engine and fail there, not in `run`.

4. **Spatial k≥3 requires `spatial_points`** (`:566-570`) — a `(n_pts, k−1, 2)` array of `(x_j, τ_j)`
   offsets. There is no grid form for k≥3 spatial. And **spatial k≥3 moments are NotImplemented**
   (`:685-688`) — only temporal any-k and spatial k=2 moment conversion exist.

5. **The temporal-k≥3 synthesis block swallows all exceptions** (`:632`, `:671-672`,
   `try/except Exception: pass`). If slice synthesis fails, the result silently retains the
   `C_tau=None`/scalar form and the plotter falls back to bars — no error surfaces. This is by design
   but can mask a real failure in `total_C`.

6. **τ-grid caps for k≥3**: slices cap the τ-grid to **41 points** (`:601-603`); the full grid
   downsamples each axis to `min(tau.size, max(2, int(4000**(1/(k−1)))))` points (`:652`) to bound
   the n^{k−1} evaluation cost. So a k≥3 plot may be coarser than the same theory's k=2 plot.

7. **Singleton/tadpole approximation**: moments use the *tree* mean φ* for singleton blocks; the
   1-loop tadpole shift of the mean is **not** folded in (`:376-377`, `:404-405`). For symmetric
   saddles (φ*=0) raw and central moments coincide and singleton terms vanish.

8. **Moments cost extra runs**: a temporal moment of order k triggers up to **k−1 additional**
   `compute_cumulants` calls (one per block size 2..k), each with its own diagram enumeration
   (`:261-266`, `:431-448`).

9. **`Config.components` is declared but never read** in `daedalus.py` — the docstring advertises it
   (`:24-25`, "honours the plot options … `components`") and the field exists (`:307`), but no plotter
   consults it. See open questions.

10. **`from dataclasses import … field` but `field` is unused** (`:31`) — every `Config` field uses a
    plain default, not `field(default_factory=…)`. Harmless but dead.

11. **`is_multifield` exists but `plot_cumulant` never branches on it** — the dispatcher routes on
    spatial/k only, not on multi-field. The docstring (`:23-25`) says dispatch is on
    "(spatial?, multi-field?, k)", but the multi-field axis is not wired in this file. See open
    questions.

12. **Spatial grid centring differs from temporal**: for spatial k=2, `run` defaults
    `tau_max=0.0, tau_step=1.0` (equal-time-centred, one τ row), whereas temporal defaults
    `tau_max=10.0, tau_step=0.5` (`:578-584`). A spatial run thus produces `C_tau_x` of shape `(1,
    n_x)` unless `tau_max` is set.

13. **`spatial_grid` METADATA is declared by theories but `run` ignores it** — `run` falls back to
    `np.linspace(-6.0, 6.0, 49)` (`:574-577`) rather than reading `METADATA['spatial_grid']` (which
    `kpz_1d.theory.py:66` provides). The notebook must pass `spatial_grid` explicitly to use the
    theory's recommendation.

14. **`describe_model` prints AND returns** (`:241-242`) — calling it in a notebook cell will both
    show the text and echo the returned string (double display) unless the result is assigned.

15. **`plot_kpoint` (spatial k≥3) takes no `sim`** — unlike every other plotter, so simulator overlay
    of spatial k≥3 events is not supported through `plot_cumulant`.

---

## 8. Glossary

* **MSR–JD field theory** — Martin–Siggia–Rose–Janssen–De Dominicis: the path-integral formulation
  of a stochastic (Langevin) dynamics. Each physical field φ gets a conjugate **response field** φ̃
  (here often `d`-prefixed, e.g. `dh` for the response to `h`); correlators and response functions
  come from the resulting action.
* **Cumulant (connected correlator)** — κ_k = ⟨φ…φ⟩_c, the connected part of the k-point correlation;
  the engine's native output.
* **Moment** — ⟨φ…φ⟩ (raw) or of the centred field (central). Reconstructed from cumulants by the
  set-partition formula (§2.4).
* **Set partition / cluster expansion** — the combinatorial identity expressing moments as sums over
  partitions of products of cumulants over blocks.
* **Loop order ℓ (`max_ell`)** — the order in the diagrammatic expansion; ℓ=0 tree, ℓ=1 one loop, etc.
* **External fields / legs** — the k field operators whose correlator is computed; `external_fields`
  is the length-k list of `(field_name, slot)` tuples.
* **k (correlator order)** — number of external legs.
* **Mean-field saddle (φ*)** — the deterministic fixed point about which fluctuations are expanded;
  found by the engine. Saddle parameters end in `star`.
* **DAE saddle solve / fixed-point index** — a multi-root Newton solver (differential-algebraic
  equations) for theories with several stable saddles (e.g. a double well); `fixed_point_index`
  picks which.
* **Dyson dressing** — resumming a self-energy series into the propagator to a finite `dyson_order`;
  used for coupled fields with unequal diffusivities.
* **fundamental / parameters** — the numeric parameter values dict; `parameters` is the canonical
  name, `fundamental` the deprecated alias (mirrored by `Config.__post_init__`).
* **spatial_grid / spatial_points** — k=2 spatial uses a grid of separations x; k≥3 spatial uses an
  explicit `(n_pts, k−1, 2)` array of `(x_j, τ_j)` events.
* **τ_j = t_j − t_0** — the k−1 time differences a k-point cumulant depends on (time-translation
  invariance).
* **slice / full grid** — for k≥3, an axis-parallel 1-D cut (sweep one leg) vs the whole
  (k−1)-dimensional tensor.
* **`*_by_ell`** — a `{loop_order: value}` dict carrying the per-loop-order decomposition.
* **`spatial_info` / modes / Sigma** — the spatial driver's diagnostic dict; `modes` are `(A, B, N)`
  triples parameterising each propagator mode's mass A, dispersion B, and noise weight N; `Sigma` is
  the 1-loop self-energy used for the tadpole mass-shift.
* **SageMath / `SR`** — a Python computer-algebra system; `SR` is its symbolic ring. The engine's
  symbolic propagator and `num_params` keys are Sage objects.
* **nauty** — graph canonicalisation / isomorphism library; deduplicates diagrams and computes
  symmetry factors.
* **sympy / lambdify** — a lighter Python CAS; `lambdify` compiles a symbolic expression to a fast
  numeric function.
* **scipy / numba / networkx** — numerical routines / JIT compiler / graph library used inside the
  engine (not in `daedalus`).
* **NumPy `ndarray`** — the n-dimensional array type underlying all numeric data here.
* **Matplotlib `Figure` / `Axes`** — the plot canvas / one subplot; every plotter returns a `Figure`.
* **`importlib.util` dynamic import** — the mechanism `load_theory` uses to import a `.theory.py`
  file from an arbitrary path.
* **`@dataclass`** — a Python decorator that auto-generates a class's constructor from declared
  fields; `Config` is one.

---

## 9. Proposed manual subsections

1. **The three-line workflow** — `load_theory → Config → run → plot_cumulant`, with the KPZ example.
2. **Loading a theory** — `load_theory`, `list_theories`, the `(model, module)` pair, what `build()`
   / `DEFAULT_FUNDAMENTAL` / `METADATA` each provide; `repo_root` and path discovery.
3. **Introspecting a model** — `describe_model`, `field_names`, `is_spatial`/`spatial_dim`/
   `is_multifield`, and how the docstring is trimmed for display.
4. **The `Config` object, field by field** — the full table, the `parameters`/`fundamental` alias,
   `resolved_grid`, and the plot-option fields.
5. **How `run` resolves a Config** — k from `external_fields`, the layered `fundamental`, the
   spatial-vs-temporal grid wiring, the MF-root and Dyson overrides; the contradiction guards.
6. **From cumulants to the result dict** — the four return shapes (temporal/spatial × k=2/k≥3) and
   the `_cfg`/`_model`/`_resolved` stamps.
7. **The k≥3 synthesis** — time differences τ_j, axis-parallel slices, the full (k−1)-dim grid, the
   τ-grid caps, and the swallowed-exception fallback.
8. **Cumulants → moments** — the set-partition expansion, the shared loop budget, singletons and the
   tree mean, the cost in extra runs; `Config.output`.
9. **Plotting and dispatch** — the `plot_cumulant` decision tree and each of the seven plotters; the
   `sim` overlay contract; `show_orders` modes and `_ORDER_COLORS`.
10. **Gotchas and limitations** — in-place model mutation, spatial-k≥3 moment gap, dead/unused
    `components`/`is_multifield` wiring, grid-centring and METADATA-vs-default mismatches.
11. **What lives behind `compute_cumulants`** — a pointer to the engine subsystem and the heavy
    libraries (Sage, nauty, sympy, scipy, numba, networkx) that `daedalus` never touches directly.

# The `compute_cumulants` Pipeline (master flow)

**Slug:** `compute-orchestration`
**Primary file:** `pipeline/compute.py`
**Entry point:** `compute_cumulants(model, k, max_ell, ...)` — `pipeline/compute.py:108`

---

## Overview

`compute_cumulants` is the **spine** of the Daedalus codebase. It is the
single function a user calls to go from a *model declaration* (a Python dict
describing an MSR-JD field theory) plus a *parameter point* (numerical values
for the model's couplings) all the way to a *numerical correlation function*
— the connected `k`-point cumulant `C(τ₁, …, τ_{k-1})` evaluated either on a
τ-grid or as a callable.

In MSR-JD (Martin–Siggia–Rose / Janssen–De Dominicis) field theory, a
stochastic dynamical system is rewritten as a path integral over a *physical*
field and a conjugate *response* field. Correlation and response functions of
the original dynamics become connected `k`-point functions of this field
theory, computed by a loop (Feynman-diagram) expansion around the mean-field
saddle point. `compute_cumulants` automates that entire expansion: it expands
the action, builds the propagator, solves for the saddle, enumerates every
Feynman diagram up to a chosen loop order, computes each diagram's symmetry
factor, and integrates each diagram numerically.

Where it sits end-to-end:

```
  model dict  (from TheoryBuilder / hand-written models/*.py)
       │
       │  + fundamental (numeric params)  + k, max_ell, external_fields
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │            compute_cumulants  (pipeline/compute.py)              │
  │                                                                 │
  │  [1] expand      FieldTheory.expand()      → bigraded action    │
  │  [2] propagator  build_propagator()        → K_ft, G_ft, …      │
  │  [3] mean_field  solve_mean_field[_dae]()  → num_params, saddle │
  │  [3.5] spatial?  → momentum integrator (early return)           │
  │  [4] poles       compute_poles_and_residues() → pole_vals,C_mats│
  │  [5] diagrams    enumerate_unique_diagrams() → typed diagrams   │
  │  [6] classify    classify_coefficient_factors() → prefactors    │
  │  [7] phase_j     compute_correction_td() per ell → C(τ)         │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼
  result dict  → notebooks (plotting), pipeline.save (npz/csv),
                 sim-comparison harnesses
```

**What feeds it:** the `model` dict (see the *theory-files* brief), built
either by `TheoryBuilder` or hand-written under `models/`. The numerical
parameter point `fundamental`, the cumulant order `k`, the max loop order
`max_ell`, and the `external_fields` (which field sits on each external leg).

**What consumes its output:** the returned `result` dict. Its `total_C`
callable and `C_tau`/`C_tau_by_ell` arrays are plotted in notebooks and
compared against direct stochastic simulations; `pipeline.save.save_npz` /
`save_csv` serialize it; the `mf` / `params` accessors expose the saddle and
parameters by natural name.

`compute_cumulants` does very little arithmetic itself. It is **orchestration**:
it sequences the seven phases, threads data dicts between them, branches
spatial-vs-temporal, assembles the per-loop-order decomposition, and records
wall-time. Each heavy phase is implemented in a sibling module (`_propagator`,
`_mean_field`, `_diagrams`, `symmetry`, `time_domain.pipeline`) which have their
own briefs; this brief documents the **glue** and the contracts between phases.

---

## The math

### 1. The MSR-JD generating functional

A Langevin / point-process dynamics for a field `φ` (e.g. a firing rate, a
density, a height field) is recast as a path integral over `φ` and a *response*
field `φ̃` (the "tilde" field). The generating functional is

```
  Z[j, j̃] = ∫ Dφ Dφ̃  exp( −S[φ, φ̃] + ∫ (j̃ φ + j φ̃) )
```

The action `S` is split into:

* a **free (Gaussian / bilinear) part** `S₀`, quadratic in the fields — its
  inverse is the bare propagator;
* an **interaction part** `S_int`, with cubic and higher vertices;
* a **mean-field / saddle part** that vanishes at the correct expansion point.

Connected correlation functions (cumulants) are obtained by functional
differentiation of `ln Z`. The `k`-point cumulant is the sum of all connected
Feynman diagrams with `k` external legs.

### 2. Fields, bigrade, and the action sectors

The code works with *fluctuation fields* — fields shifted to expand around the
saddle, `φ = φ* + δφ`. Each monomial in the expanded action carries a **bigrade**
`(n_t, n_p)` = (number of response-field `tilde` factors, number of
physical-field factors). `compute_cumulants` (via `FieldTheory`) classifies the
expanded action into sectors keyed by this bigrade:

* **Mean-field sector** — bigrades `(0,0)`, `(1,0)`, `(0,1)`. These linear/constant
  terms vanish when the saddle equation is satisfied; `_print_action_sectors`
  (`pipeline/compute.py:50`) displays them for sanity.
* **Free action** — bigrade `(1,1)`, the bilinear kernel. Its time-domain matrix
  is `K_ker`; its Fourier image is `K_ft`; its inverse `G_ft = K_ft⁻¹` is the
  bare propagator.
* **Interaction sectors** — total degree ≥ 2 excluding `(1,1)`. These are the
  vertices of the diagrammatic expansion.

### 3. The Taylor budget

Nonlinear functions in the action (e.g. a sigmoid firing-rate `φ(v)`) are
Taylor-expanded to a finite `taylor_order`. A connected diagram with `k`
external legs at `ℓ` loops needs vertices of degree at most `k + 2ℓ` (every loop
adds two legs that must terminate on vertices). So the *smallest sufficient*
Taylor order is

```
  taylor_order = max(k + 2·max_ell, 2)         (pipeline/compute.py:279)
```

The floor of 2 is structural: order 2 still produces the bilinear `(1,1)`
propagator kernel and the `(0,0)/(1,0)/(0,1)` MF saddle sectors, which downstream
code reads unconditionally even for the degenerate `k=2, max_ell=0` case (where
the tree-level pair correlator is just the bare propagator with no interaction
vertices). The docstring records that this floor was **historically 4** (chosen
for a now-obsolete cache layout) and was lowered to 2 to save ~90 min on heavy
Bernoulli theories at `k=2, max_ell=0`.

### 4. The propagator: Fourier inversion of the bilinear kernel

The free action's `(1,1)` sector is a matrix `K_ker(t)` of kernels (containing
Dirac `δ(t)`, `δ′(t)`, and model kernel symbols). Fourier-transforming gives
`K_ft(ω)`, a matrix of rational functions of `ω`. The **bare retarded
propagator** is

```
  G_ft(ω) = K_ft(ω)⁻¹            (rows = physical, cols = response)
```

Its **poles** in the upper-half ω-plane are the retarded relaxation modes; the
**residue matrices** `C_mats` at those poles give the time-domain propagator as
a sum of decaying exponentials:

```
  G(t) = Σ_k  C_k · exp(−i ω_k t) · Θ(t)   +   D_delta · δ(t)
```

where `D_delta = lim_{ω→∞} G_ft(ω)` is the instantaneous (white) part. Phases
[2] and [4] produce `pole_vals` (the `ω_k`), `C_mats` (the residue matrices),
and `D_delta`.

### 5. The mean-field saddle

The saddle point is the deterministic mean-field steady state. `solve_mean_field`
solves the self-consistency `n*_i = ⟨RHS⟩(n*_j)` by `scipy.optimize.fsolve`,
then assembles `num_params` — a dict mapping every symbolic parameter and saddle
value (`SR.var → float`) used to numerically substitute into the propagator and
the diagram integrands. For theories that declare `.equation(...)`, the
DAE-based multi-root solver `solve_mean_field_dae_compat` runs instead, returning
every fixed point with a stability annotation.

### 6. Feynman diagrams and the symmetry factor `𝒮(Γ)`

A connected `k`-point cumulant is a weighted sum of diagrams. Each diagram `Γ`
contributes

```
  weight(Γ) = 𝒮(Γ) · ∏_v c_user,v · ∏_e P_e · ∫ dt_v
```

where `c_user,v` is the literal coefficient written in the action (no implicit
`1/n!`), `P_e` is the propagator on edge `e`, the integral is over internal
vertex times, and `𝒮(Γ)` is the **combinatorial symmetry factor**.

`𝒮(Γ)` is the canonical Feynman rule (Path A in this codebase):

```
  𝒮(Γ) = ( ∏_v ∏_ℓ n_{v,ℓ}! ) / |Aut_fixed_ext(Γ)|
```

The numerator is the per-vertex Wick combinatorial (factorials of identical-leg
multiplicities at each vertex, both response and physical legs). The denominator
is the order of the full automorphism group of the colour-preserving incidence
digraph with external leaves *pinned* at their canonical positions. This single
ratio accounts for every Feynman-rule symmetry — same-type vertex swaps,
parallel-edge swaps, self-loop leg swaps — so the dedup multiplicity is **not**
multiplied back in (that would double-count).

### 7. The set-partition / per-ℓ assembly

`compute_cumulants` runs Phase J **once per loop order** `ℓ ∈ {0, …, max_ell}`,
producing a per-loop-order decomposition. The master cumulant is the sum across
loop orders:

```
  total_C(τ) = Σ_ℓ  C_ℓ(τ)          (pipeline/compute.py:802)
  C_tau      = Σ_ℓ  C_tau_by_ell[ℓ]  (pipeline/compute.py:808)
```

This per-ℓ split is what notebooks plot as "tree", "1-loop", "2-loop" curves,
and what the saver stores as `C_tree`, `C_1_loop`, etc.

> **Note on the term "set-partition".** The task framing mentions
> "moment/cumulant/central_moment set-partition assembly." The orchestrator
> *itself* (`pipeline/compute.py`) does **not** contain explicit moment↔cumulant
> set-partition (Möbius-inversion) bookkeeping — `compute_cumulants` returns the
> **connected** cumulant directly (the diagrammatic sum *is* the cumulant). The
> only assembly it performs is the additive per-ℓ sum above plus the master
> `total_C` aggregation. Moment/central-moment conversions, if needed, live in
> downstream consumers, not in this file. This is flagged in *open questions*.

---

## External tools used

This subsystem is built on **SageMath** and the scientific-Python stack. Below,
each library is explained from scratch, then exactly how `compute.py` (and its
immediate helpers) use it.

### SageMath (`sage.all`)

**What it is.** SageMath is a large open-source mathematics system built on top
of Python. It bundles dozens of computer-algebra and number-theory libraries
(Singular, Maxima, PARI/GP, FLINT, NTL, GAP, …) behind a unified Python API.
The piece used most here is its **symbolic ring** `SR` (the "Symbolic Ring"),
which represents algebraic expressions — variables, polynomials, rational
functions, transcendental functions — that you can substitute into, simplify,
differentiate, and Fourier-transform. Sage also provides exact algebraic number
fields, polynomial rings, matrices over arbitrary rings, and **graph theory**
(the `DiGraph` class and its automorphism / canonical-labeling routines).

**How `compute.py` uses it.**

```python
from sage.all import SR                          # pipeline/compute.py:19
```

* `SR(...)` wraps a Python number or string into a symbolic expression. The
  orchestrator uses it to coerce diagram prefactors to symbolic form before
  passing them to Phase J:
  ```python
  combined_prefactor = SR(info['scalar_prefactor'])   # pipeline/compute.py:693
  ```
* In the coupled-spatial branch it creates a symbolic variable and substitutes:
  ```python
  reaction_diffusion_matrices(prop['K_ft'], prop['omega'], SR.var('k'), ...)  # :440
  complex(SR(sm[a_, b_]).subs(_sub)).real                                      # :446
  ```
  `SR.var('k')` makes a fresh symbolic variable named `k`; `.subs(dict)`
  replaces symbols by numbers; `complex(...)` collapses a now-numeric symbolic
  expression to a Python complex.

The deeper Sage usage lives in the phase helpers:

* **`symmetry.py`** uses Sage's **`DiGraph`** to build a *coloured incidence
  digraph* for each diagram and calls `D.automorphism_group(partition=…)`
  (`msrjd/diagrams/symmetry.py:339`) and `D.canonical_label(partition=…,
  certificate=True)` (`:517`). `automorphism_group` returns a permutation group
  whose `.order()` is `|Aut(Γ)|` — the denominator of `𝒮(Γ)`. `canonical_label`
  returns a canonical (relabeling-invariant) form used as the dedup signature.
  The `partition=` argument is the colour classes: nodes in different classes
  are never mapped to each other, which is how vertex/edge/field *types* are
  respected.
* **`_propagator.py`** uses Sage matrices (`matrix(SR, …)`), the symbolic
  Fourier transform (`fourier_transform`), `dirac_delta`, `diff`, symbolic
  `limit`, and an **exact number field** `CyclotomicField(4)` = `ℚ[i]` together
  with `PolynomialRing(CF, 'ω')` and its `fraction_field()` to invert `K_ft`
  exactly and extract poles/residues with machine precision. It also uses
  `CDF` (Complex Double Field — Sage's wrapper around C `double complex`) for
  numerical root-finding.

> **Why Sage and not plain SymPy?** Sage's exact algebraic number fields
> (`CyclotomicField`) and polynomial fraction fields give *canonical irreducible*
> propagator entries `P(ω)/Q(ω)` with reliable GCD, so the poles are exactly the
> roots of `Q` with no spurious cancellable factors — a precision guarantee plain
> floating-point inversion cannot make. Its graph-automorphism engine (backed by
> `bliss`/`nauty`-class algorithms internally) is what makes the symmetry factor
> exact.

### nauty / bliss (via Sage's graph canonicalization)

**What it is.** `nauty` ("No AUTomorphisms, Yes?") and `bliss` are C libraries
for computing graph automorphism groups and canonical forms — the gold-standard
tools for graph isomorphism. You give them a graph (optionally with a vertex
*colouring*/partition) and they return either the automorphism group or a
canonical labeling such that two graphs are isomorphic iff their canonical forms
are byte-identical.

**How this code uses it.** Indirectly, through Sage. `compute.py` calls
`classify_coefficient_factors` (Phase [6]), which calls `combinatorial_factor`
→ `_automorphism_order` → `D.automorphism_group(partition=…)`
(`msrjd/diagrams/symmetry.py:339`). Sage routes `automorphism_group` and
`canonical_label` through its bundled graph-canonicalization backend (bliss /
nauty-family algorithms). So when the orchestrator computes `𝒮(Γ)`, the
denominator `|Aut_fixed_ext(Γ)|` is ultimately a nauty-class computation. The
dedup signature in Phase [5] (`diagram_signature` → `canonical_label`) is the
same machinery used for isomorphism testing.

### scipy (`scipy.optimize`)

**What it is.** SciPy is the standard scientific-computing library for Python.
The `scipy.optimize` submodule provides numerical root-finders and optimizers.

**How this code uses it.** The mean-field phase ([3]) uses `fsolve`:

```python
from scipy.optimize import fsolve                # pipeline/_mean_field.py:11
attempt = fsolve(mf_residual, x0, full_output=True)   # _mean_field.py:174
```

`fsolve` finds a root of the multivariate residual `mf_residual(n*) = n* −
RHS(n*)` (the saddle self-consistency). `full_output=True` returns
`(x, infodict, ier, msg)` where `ier == 1` signals clean convergence; the solver
tries a sequence of initial guesses `[0.1, 0.5, 1.0, 0.01]·npop` and keeps the
first converged result. The DAE solver (`_mean_field_dae`) layers a multi-start
global root search on top of this for theories with multiple fixed points.

### numpy (`np`)

**What it is.** NumPy is the foundational array library — fast N-dimensional
arrays and vectorized math.

**How `compute.py` uses it.**

```python
import numpy as np                               # pipeline/compute.py:18
```

* Builds the τ-grid: `np.arange(-tau_max, tau_max + tau_step*0.5, tau_step)`
  (`:712`).
* Collects the Phase-J batch evaluation into a complex array:
  `np.array(td_result_ell['total_C_batch'](...), dtype=complex)` (`:791`).
* In the spatial branch (`import numpy as _np`, `:397`): `_np.asarray`,
  `_np.zeros_like`, `_np.linspace`, `_np.real`, `_np.exp`, `_np.argmin`,
  `_np.allclose`, and the finite-difference tadpole correction.
* The propagator's numerical fallback tiers use `np.linalg.inv`, `np.roots`,
  `np.polyfit` (`_propagator.py`).

### numba

**What it is.** Numba is a just-in-time (JIT) compiler that turns numeric Python
functions into fast machine code.

**How this code uses it.** `compute.py` does **not** import numba directly.
Numba (where present) lives in the deeper integration layers (`final_integral`,
spatial integrator) that Phase J calls. The orchestrator's contract with those
layers is purely via the `propagator_data` dict and the returned callables, so
numba is an implementation detail below the orchestration spine. (Listed here for
completeness because the manual covers the full toolchain.)

### networkx

**What it is.** NetworkX is a pure-Python graph library.

**How this code uses it.** Not in `compute.py` directly. Graph work in this
subsystem (incidence digraphs, automorphisms) goes through **Sage's** `DiGraph`,
not NetworkX. The diagram *enumerator* (Phase [5], in
`msrjd/enumeration/`) may use graph structures internally, but the orchestrator
never touches networkx. (Listed for completeness.)

### Python standard library

* `os` (`compute.py:14`) — `os.cpu_count()` is referenced in the docstring's
  description of worker defaults (the actual call is in the helpers).
* `time` (`compute.py:15`) — `time.perf_counter()` for the per-phase wall-time
  tracker (`_phase_time`, `:298`).
* `signal` / `cysignals.alarm` (in `_propagator.py`) — SIGALRM-based watchdogs
  that cap pathological symbolic-inverse and `factor()` calls.
* `multiprocessing` (in `time_domain.pipeline` and `type_assignment`) — fork-based
  process pools for the two parallelizable stages, guarded by `fork_safety`.

---

## Components

The orchestrator file defines three functions; the rest of this section also
documents the **immediate callees** that define each phase's contract.

### `_trunc(s, maxlen=200)` — `pipeline/compute.py:45`

**Signature:** `_trunc(s, maxlen=200) -> str`
**Takes:** any object `s`, an integer `maxlen`.
**Returns:** `str(s)` truncated to `maxlen` chars with a trailing `'...'`.
**Does:** a display helper used by `_print_action_sectors` to keep long
symbolic coefficients readable in verbose output. Pure formatting; no side
effects.

### `_print_action_sectors(ft)` — `pipeline/compute.py:50`

**Signature:** `_print_action_sectors(ft) -> None`
**Takes:** an expanded `FieldTheory` object `ft`.
**Returns:** nothing (prints).
**Does:** pretty-prints the three action sectors after `expand()`:

1. Reads `ft._mf_sector_raw` (a dict keyed by bigrade) and prints the MF sectors
   `(0,0)`, `(1,0)`, `(0,1)` — the saddle-vanishing terms.
2. Prints `ft.free_action()` — the `(1,1)` bilinear propagator kernel.
3. Iterates `ft.sectors()` and prints every interaction bigrade `(n_t, n_p)`
   with total degree ≥ 2 (excluding `(1,1)`) — the vertices.

Inner helper `_fmt_poly(poly)` walks `poly.dict()` (a Sage polynomial's
exponent-vector → coefficient map), reconstructs each monomial string from the
ring generator names, and formats `monomial * (coeff)`. This is the verbose
"here is your action" diagnostic; it has no effect on the computation.

### `compute_cumulants(...)` — `pipeline/compute.py:108`  ← **the entry point**

**Full signature:**

```python
def compute_cumulants(
    model: dict,
    k: int,
    max_ell: int = 0,
    fundamental: dict = None,
    external_fields: list[tuple[str, int]] = None,
    *,
    tau_max: float = 50.0,
    tau_step: float = 0.5,
    spatial_grid=None,
    taylor_order: int = None,
    origin_leaf_idx: int = 0,
    output_npz: str = None,
    output_csv: str = None,
    use_cache: bool = True,
    parallel: bool = True,
    spatial_parallel: bool = True,
    spatial_n_q: int = 64,
    spatial_points=None,
    n_workers: int = None,
    use_grouped_phase_j: bool = False,
    fixed_point_index: int = 0,
    mf_dae_n_starts: int = 64,
    mf_dae_seed_box: dict | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
```

**Takes (the load-bearing arguments):**

| arg | meaning |
|-----|---------|
| `model` | the theory dict (fields, parameters, kernels, action, mf conditions, …) |
| `k` | number of external legs (the `k`-point cumulant) |
| `max_ell` | maximum loop order (0 = tree only) |
| `fundamental` | `{param_name: numeric_value}` — the parameter point |
| `external_fields` | length-`k` list of `(field_name, population)` tuples, e.g. `[('n',1),('n',2)]` |
| `tau_max`, `tau_step` | τ-grid extent and spacing |
| `spatial_grid` | x-points for a spatial model (routes to the momentum integrator) |
| `spatial_points` | `(n_pts, k-1, 2)` array of `(x_j, τ_j)` offsets for spatial `k≥3` |
| `taylor_order` | override the auto-picked Taylor budget |
| `parallel` | enable fork-based multiprocessing for the two heavy temporal stages |
| `spatial_parallel` | enable thread-based parallelism for the spatial loop integrator |
| `use_grouped_phase_j` | use the prototype grouped Phase-J integrator |
| `verbose` | print progress + record per-phase wall-time |

**Returns:** a result `dict` (see *Data structures* below). The temporal path
returns the full dict; the spatial `k=2` and spatial `k≥3` paths each return a
**different**, early-return dict.

**Step-by-step.** The body is a linear sequence of seven numbered phases with a
spatial short-circuit between [3] and [4]:

**Argument validation** (`:253`–`:289`). Defaults `fundamental` to `{}`; raises
`ValueError` if `external_fields` is missing or its length ≠ `k`. Auto-picks
`taylor_order = max(k + 2·max_ell, 2)` when not given (`:279`). Reads the model's
`naming_convention`, keeps the user's external-field form in
`external_fields_user`, and translates user-facing names to internal fluctuation
names via `normalize_external_fields` (`:288`). Initializes the per-phase
wall-time dict `phase_walls` (or `None` if not verbose) and the local closure
`_phase_time(label, t0)` (`:298`) that records `perf_counter()` deltas.

**Phase [1] — `expand`** (`:305`–`:351`). Constructs `ft = FieldTheory(model,
taylor_order=...)`. Tries the on-disk expand cache via `pipeline._expand_cache`:
`find_best_cached_order` looks for any cached order ≥ requested,
`prepare_for_load` + `load_expand` downgrade-filter to the target degree. On a
cache miss it calls `ft.expand()` (the expensive multivariate-Taylor pass) and
`save_expand`. Runs `ft.sanity_check(verbose=...)`; raises `RuntimeError` if it
fails. Extracts `vtypes = extract_vertex_types(ft)` and `stypes =
extract_source_types(ft)`, counts the `NoiseSourceType`s, and (if verbose) prints
the action sectors. Records the `expand` wall-time.

**Phase [2] — `propagator`** (`:353`–`:358`). Calls
`build_propagator(ft, model, use_cache=..., verbose=...)`, which builds (or
loads from cache) the symbolic propagator dict `prop` with keys `K_ker, K_ft,
G_ft, adj_ft, D_omega, D_delta, t_var, omega, nf, ring_gen_names`. For spatial
models it also attaches the heat-kernel block (`G_tx_sym`, `G_tx`). Records
wall-time.

**Phase [3] — `mean_field`** (`:360`–`:380`). If `model.get('equations')` is
truthy, routes to `solve_mean_field_dae_compat` (the DAE multi-root solver, with
`fixed_point_index`, `n_starts`, `seed_box`); otherwise to
`solve_mean_field`. Extracts `num_params = mf['num_params']` — the
`{SR.var → float}` substitution dict every downstream phase needs. Records
wall-time.

**Phase [3.5] — spatial short-circuit** (`:382`–`:626`). This is the
**spatial-vs-temporal branch**. Logic:

* If `spatial_grid` was passed but the model is **not** spatial, warn and ignore
  it (falls through to temporal).
* If the model **is** spatial **and** a spatial input (`spatial_grid` or
  `spatial_points`) was given, take the spatial path:
  * **k ≠ 2 (`k≥3`):** require `spatial_points`. Call `compute_spatial_kpoint(ft,
    model, prop, num_params, external_fields, spatial_points, max_ell, verbose)`.
    If it raises `NotImplementedError` containing `'single-field'`, fall to the
    **coupled multi-field** driver: build reaction-diffusion matrices `M, D, V`
    from `prop['K_ft']`, check `V == 0` (no drift supported), split the reference
    diffusion, build `tree_info_k`, and call `compute_coupled_kpoint(...)`.
    **Early-returns** a dict keyed by `C_kpoint`, `C_kpoint_by_ell`, `points`,
    `spatial_info`, `k`, `max_ell`, `mf` (`:463`–`:470`).
  * **k = 2 (the grid path):** require `spatial_grid`. Guard `max_ell ≤ 2` and
    `ic_mode == 'stationary'`. Build `tau_grid` and `spatial_grid_arr`. For
    `max_ell ≥ 1` call `compute_spatial_correlator_generic(...)` (the
    full-diagram momentum integrator) with `parallel=(parallel and
    spatial_parallel)`; for tree-level (`max_ell == 0`) call
    `compute_spatial_correlator_via_pipeline(...)`. Build the `total_C(*tau_then_x)`
    closure (1-arg → `C(τ)` at x=0; 2-arg → `C(x, τ)`), with the 1-loop tadpole
    mass-shift correction `δC = Σ·∂C₀/∂A` applied via finite difference for
    `d=1`, and a radial inverse FT for `d≥2`. **Early-returns** a spatial dict
    with `total_C`, `C_tau`, `C_tau_x`, `C_tau_x_by_order`, `spatial_info`, `mf`,
    etc. (`:584`–`:612`).
* If the model is spatial but **no** `spatial_grid` was passed, raise a clear
  `ValueError` (`:619`) — the temporal pole-finder cannot consume the inert
  `Laplacian` symbol carried in `G_ft`.

**Phase [4] — `poles`** (`:628`–`:633`). Calls
`compute_poles_and_residues(prop, num_params, verbose=...)`, which **mutates
`prop` in place** to fill `prop['pole_vals']` and `prop['C_mats']` (and fills
`D_delta` if Phase [2] left it `None`). Three-tier strategy: exact polynomial
fraction-field (`CyclotomicField(4)[ω]`), numpy-cofactor fallback, legacy
symbolic. Records wall-time.

**Phase [5] — `diagrams`** (`:635`–`:664`). Builds the field-index maps:
`ring_var_names = ft._ns._ring_var_names`, `n_tilde = ft._n_tilde`, then
`resp_idx, phys_idx = build_field_index_map(ring_var_names, n_tilde)`. Validates
that every external field is in `phys_idx`. Calls `enumerate_unique_diagrams(...)`
→ returns `(unique_by_ell, multiplicity_by_ell, all_unique)`. This runs the
four-stage diagram pipeline per `ℓ`: prediagram enumeration → typed-diagram
assignment → causal filtering → dedup. `parallel` is forwarded (fork-based
per-prediagram type assignment). Records wall-time.

**Phase [6] — `classify`** (`:666`–`:705`). Reads
`model['time_dependent_parameters']` and `model['noise_structure']`. Walks
diagrams *by ℓ* so each record carries its loop tag. For each `(td, mult)`:
calls `classify_coefficient_factors(td, time_dep_params, noise_structure)`,
extracts `combined_prefactor = SR(info['scalar_prefactor'])`, and appends to two
parallel lists:

* `kernel_groups` — `{'diagrams': [td], 'combined_prefactor': ...}` (the legacy
  Phase-J input format);
* `diagram_records` — `{'typed_diagram', 'classify', 'combined_prefactor',
  'multiplicity', 'ell'}`.

Crucially, `mult` is **recorded but not multiplied** into the prefactor — the
Aut-based `𝒮(Γ)` already carries each class's full weight (`:686`–`:692`).
Records wall-time.

**Phase [7] — `phase_j`** (`:707`–`:799`). Builds the τ-grid and the
`propagator_data` dict (the subset of `prop` Phase J consumes:
`K_ker, K_ft, G_ft, adj_ft, D_omega, D_delta, t_var, omega, nf, pole_vals,
C_mats`). Picks the τ-evaluation pattern by `k`:

* `k == 1`: `tau_points = [(t,) for t in tau_grid]`;
* `k == 2`: `tau_points = [(0.0, t) for t in tau_grid]` (leaf 0 pinned, leaf 1
  swept);
* `k ≥ 3`: `tau_points = None` (caller handles grid evaluation).

Then runs Phase J **once per ℓ**: for each loop order it filters
`records_ell`, and if empty contributes a zero callable + zero array. Otherwise
it calls either `compute_correction_td_grouped` (prototype, when
`use_grouped_phase_j`) or `compute_correction_td` (default), passing the
per-ℓ typed diagrams + prefactors + `propagator_data` + `external_fields` +
`num_params` + `origin_leaf_idx`. Stores `total_C_by_ell[ell]` and
`phase_j_by_ell[ell]`. For `k ∈ {1,2}` it also batch-evaluates the τ-grid:
`td_result_ell['total_C_batch'](tau_points, parallel=parallel, ...)` →
`C_tau_by_ell[ell]`. Records wall-time.

**Aggregation** (`:801`–`:810`). Defines the master closure `total_C(*ext_time_values)
= Σ_ℓ total_C_by_ell[ℓ](*ext)` and `C_tau = Σ_ℓ C_tau_by_ell[ℓ]` (or `None` for
`k≥3`).

**Adaptive mean-field dict** (`:812`–`:879`). Discovers which model parameters
are MF saddles (via `naming_convention['mf_parameters']`, or the per-parameter
`mean_field` flag, or the legacy `nstar/vstar/mstar` fallback), maps each to its
population index range (`_saddle_indices`, handling heterogeneous `indexed_by`),
and reads the solved value out of `num_params`. NaN-only entries (declared but
never substituted) are dropped. Produces `mf_values = {internal_name: [v1, …]}`.

**Result assembly** (`:881`–`:908`). Packs the result dict (see below). Adds
DAE-MF extras (`mf_all_roots`, `mf_stable_roots`, `mf_index_used`,
`mf_stability`) when the equation-based solver ran (`:910`–`:932`). Optionally
saves NPZ/CSV (`:934`–`:944`). Prints the final `Done.` line and the **phase
wall summary** (a per-phase `seconds (percent)` breakdown, `:946`–`:955`).
Returns `result`.

### `classify_coefficient_factors(typed_diagram, time_dep_params=None, noise_structure=None)` — `msrjd/diagrams/symmetry.py:618`

**Signature/returns:** `-> dict` with keys `'Scal', 'scalar_prefactor',
'vertex_time_factors', 'source_time_info', 'is_stationary'`.
**Takes:** one `TypedDiagram`, the model's `time_dependent_parameters` list, and
the `noise_structure` dict.
**Does:** the heart of Phase [6]. For a diagram it:

1. Computes `Scal = combinatorial_factor(typed_diagram)` — the symmetry factor
   `𝒮(Γ)` (numerator ÷ `|Aut_fixed_ext|`, via Sage automorphisms).
2. Walks every vertex. Each vertex coefficient acquires a `(−1)` because the
   weight is `∫ exp(−S)` (`coeff = -SR(vtype.coefficient)`, `:643`).
3. **Source (noise) vertices** (`_is_source_type` → no `physical_legs`): decides
   whether the amplitude can be *pulled out* of the time integral. For white
   noise with a time-independent amplitude, the constant pulls out into the
   scalar prefactor; otherwise the time-dependent part stays in `source_time_info`
   (the non-local kernel substitution happens in Phase J).
4. **Interaction vertices:** splits the coefficient into a time-independent
   *constant part* (joins `scalar_parts`) and a time-dependent part (stored in
   `vertex_time_factors`, simplified by `.simplify_rational()`).
5. Multiplies all constant parts: `scalar_prefactor = reduce(mul, scalar_parts,
   SR(1))`.
6. `is_stationary` is `True` iff there are no per-vertex time factors and no
   time-dependent noise amplitudes — telling Phase J it can use the
   time-translation-invariant fast path.

Helper `_symbols_matching_prefixes(expr, prefixes)` (`:598`) returns the free SR
variables whose names start with a given prefix (how time-dependent parameters
are detected). Helper `_is_source_type(vtype)` (`:613`) is `not
hasattr(vtype, 'physical_legs')`.

### `combinatorial_factor(typed_diagram)` — `msrjd/diagrams/symmetry.py:403`

**Returns:** `int ≥ 1` — the symmetry factor `𝒮(Γ) = (∏ Wick numerator) /
|Aut_fixed_ext|`. **Does:** computes `numer = _wick_leg_factor(td)` (product of
identical-leg factorials at each vertex) and `aut = _automorphism_order(td,
fix_external=True)` (Sage automorphism-group order), returns `numer // aut`
(integer division; defensively floors if non-divisible). This is what
`classify_coefficient_factors` calls for `Scal`.

### `enumerate_unique_diagrams(ft, model, *, k, max_ell, external_fields, G_ft, resp_idx, phys_idx, vtypes, stypes, ...)` — `pipeline/_diagrams.py:54`

**Returns:** `(unique_by_ell, multiplicity_by_ell, all_unique)` — dicts
`{ℓ: [TypedDiagram]}` and `{ℓ: [int]}` plus the flat concatenation.
**Does:** Phase [5]'s contract. For each `ℓ ∈ {0, …, max_ell}` it checks the
disk cache (`PipelineCache`, stage name embeds `k`, `ℓ`, external-field tag,
taylor order, **and** cache version `v3`); on a miss it runs the four stages —
`enumerate_prediagrams_all` → `enumerate_all_typed` (the parallelizable
type-assignment, fork-based when `parallel=True`) → `filter_causal` →
`deduplicate_with_multiplicities` — then caches and returns. The cache-version
suffix (`unique_typed_mult_v3_…`) deliberately invalidates older caches written
before the multiplicity-aware dedup and the complete-isomorphism signature.

### `build_propagator(ft, model, ...)` — `pipeline/_propagator.py:524`

**Returns:** the `prop` dict (`K_ker, K_ft, G_ft, adj_ft, D_omega, D_delta,
t_var, omega, nf, ring_gen_names`, plus `pole_vals=None, C_mats=None`
placeholders and a spatial block when applicable). **Does:** Phase [2]. Builds
`K_ker` from the `(1,1)` free action, Fourier-transforms it to `K_ft`, applies
the model's `kernel_ft_image` hook, then attempts a **budget-aware** symbolic
inverse/adjugate/det (skipped for "rich" matrices `nf ≥ 6` or > 20 free symbols,
deferring to the numerical pole-finder). Caches under
`saved_theories/<tag>/propagator.sobj`.

### `compute_poles_and_residues(prop, num_params, verbose=True)` — `pipeline/_propagator.py:851`

**Returns:** `prop` (mutated in place). **Does:** Phase [4]. Three-tier
pole/residue extraction (exact `CyclotomicField(4)[ω]` fraction-field → numpy
cofactor → legacy symbolic). Fills `prop['pole_vals']` (retarded poles, `Im ω >
0`) and `prop['C_mats']` (residue matrices). The exact path computes
`C_k[i,j] = i·P_ij(ω_k)/Q′(ω_k)`.

### `solve_mean_field(ft, model, fundamental, verbose=True)` — `pipeline/_mean_field.py:14`

**Returns:** `{'nstar_vals', 'vstar_vals', 'phi_deriv_vals', 'num_params',
'param_subs'}` (heterogeneous path adds `'saddle_values', 'saddle_info'`).
**Does:** Phase [3] (legacy/single-pop + heterogeneous branch). Assembles
`param_subs` from `fundamental`, reads the symbolic `v*` from
`mf_bg_conditions`, builds the nstar iteration target, and `fsolve`s the
self-consistency over a sequence of initial guesses. Assembles the full
`num_params` (params + saddle values + φ-derivatives `phiN_i`).

### `compute_correction_td(typed_diagrams, prefactors, propagator_data, k, num_params, external_fields, origin_leaf_idx, ...)` — `msrjd/integration/time_domain/pipeline.py:202`

**Returns:** a dict with `'total_C'` (callable), `'total_C_batch'` (callable),
`'delta_contributions'`, `'groups'`, `'skipped_kernel_ids'`, `'ext_time_vars'`.
**Does:** Phase [7]'s per-ℓ workhorse. For each typed diagram it calls
`integrate_diagram` (vertex-time numerical quadrature using the pole-residue
propagator) and accumulates per-diagram contribution callables. `total_C(*τ)`
sums them serially; `total_C_batch(τ_points, parallel=, n_workers=,
start_method=)` fans the τ-grid out over a fork-based process pool (or degrades
to serial under the notebook fork guard).

### `MeanField` / `Parameters` / `normalize_external_fields` — `pipeline/access.py`

Natural-name accessors returned in the result dict (`mf`, `params`) and the
external-field translator used at `:288`. `MeanField['v', 1]` → `v*_1`;
`Parameters['w', 1, 2]` → 1-based matrix element; `normalize_external_fields`
maps user `('n',1)` → internal `('dn',1)` via the model's naming convention (or
the n/v/m fallback).

---

## Data structures

### The `model` dict (input)

Declared by `TheoryBuilder` or hand-written. Keys the orchestrator reads:
`response_fields`, `physical_fields`, `parameters`, `kernels`, `operators`,
`functions`, `mf_substitutions`, `mf_bg_conditions`, `specializations`,
`kernel_ft_image`, `phi_concrete`, `mf_equations`, `action`. Optional:
`correlated_noises`, `naming_convention`, `equations` (routes to DAE solver),
`spatial`, `boundary`, `initial`, `populations`, `iteration_saddles`,
`time_dependent_parameters`, `noise_structure`, `name`.

### `fundamental` dict (input)

`{param_name: value}` where value is a scalar, list (vector), or list-of-lists
(matrix). Example from the docstring:

```python
{'a': 1.0, 'tau': 10.0, 'tau_g': 2.5, 'E': [1.1, 1.05],
 'w': [[0.35, 0.4], [0.3, 0.5]], 'w_X': 0.3, 'lambda_X': 2.5,
 'p_part': 0.6, 'mu_shift_diff': 0.0, 'sigma_shift_diff_sq': 1.0}
```

### `prop` dict (Phase [2]/[4] internal)

| key | type | meaning |
|-----|------|---------|
| `K_ker` | Sage `matrix(SR)` `nf×nf` | bilinear kernel in time domain (δ, δ′) |
| `K_ft` | Sage `matrix(SR)` | Fourier image of `K_ker` (rational in ω) |
| `G_ft` | Sage `matrix(SR)` or `None` | propagator `K_ft⁻¹` (rows=phys, cols=resp); `None` for rich matrices |
| `adj_ft` | Sage matrix or `None` | adjugate = `G_ft·det` |
| `D_omega` | SR or `None` | `det(K_ft)` |
| `D_delta` | Sage `matrix(SR)` | instantaneous part `lim_{ω→∞} G_ft` |
| `t_var`, `omega` | SR vars | time and frequency symbols |
| `nf` | int | number of field components |
| `ring_gen_names` | list[str] | ring generator names (first `nf` = response, rest = physical) |
| `pole_vals` | list[complex] | retarded poles, filled by Phase [4] |
| `C_mats` | list[Sage `matrix(CDF)`] | residue matrices, filled by Phase [4] |
| `G_tx_sym`, `G_tx`, `spatial_dim` | (spatial only) | heat-kernel real-space propagator |

### `propagator_data` dict (Phase [7] input)

The subset of `prop` Phase J consumes (`:714`): `K_ker, K_ft, G_ft, adj_ft,
D_omega, D_delta, t_var, omega, nf, pole_vals, C_mats`.

### `num_params` dict (Phase [3] output)

`{SR.var → float}`. Maps every symbolic parameter, saddle value (`nstar_i`,
`vstar_i`, `mstar_i`), and φ-derivative (`phiN_i`) to a number. The universal
substitution dict for all numerical evaluation.

### `diagram_records` list (Phase [6] output)

Each element:
```python
{'typed_diagram': TypedDiagram, 'classify': dict, 'combined_prefactor': SR,
 'multiplicity': int, 'ell': int}
```

### `classify` info dict (per diagram)

`{'Scal': int, 'scalar_prefactor': SR, 'vertex_time_factors': {v: SR},
'source_time_info': {v: {...}}, 'is_stationary': bool}`.

### `unique_by_ell` / `multiplicity_by_ell` (Phase [5] output)

`{ell: [TypedDiagram]}` and `{ell: [int]}` (parallel lists; multiplicity is
diagnostic only, *not* re-multiplied into the weight).

### The result dict (temporal output)

| key | type | meaning |
|-----|------|---------|
| `total_C` | callable | `Σ_ℓ C_ℓ(*τ)` |
| `total_C_by_ell` | `{ℓ: callable}` | per-loop-order |
| `C_tau` | ndarray or `None` | total on τ-grid (`k∈{1,2}`; `None` for `k≥3`) |
| `C_tau_by_ell` | `{ℓ: ndarray\|None}` | per-loop-order grid; `C_tau == Σ` |
| `tau_grid` | ndarray | τ values |
| `mf_values` | `{internal_name: [v1,…]}` | adaptive saddle values |
| `mf` | `MeanField` | natural-name saddle accessor |
| `params` | `Parameters` | natural-name parameter accessor |
| `num_params` | `{SR: float}` | numerical substitution dict |
| `propagator` | dict | the `prop` dict (with poles/residues filled) |
| `diagrams` | list | `diagram_records` |
| `kernel_groups` | list | per-diagram `{'diagrams', 'combined_prefactor'}` |
| `phase_j_by_ell` | `{ℓ: td_result}` | per-ℓ Phase-J result dicts |
| `phase_walls` | `{label: sec}` or `None` | per-phase wall-time |
| `config` | dict | echoed args (`external_fields` internal, `external_fields_in` user form, `taylor_order`, …) |
| `mf_all_roots`, `mf_stable_roots`, `mf_unstable_roots`, `mf_index_used`, `mf_state_var_order`, `mf_stability` | (DAE only) | multi-root saddle data |

### The result dict (spatial `k=2`, early return at `:584`)

Adds `C_tau_x` (the full `(n_τ, n_x)` real-space correlator), `C_tau_x_by_order`
(cumulative per-loop-order grids), `spatial_grid`, `spatial_info`; `total_C` is
the spatial closure (1-arg → `C(τ)`@x=0, 2-arg → `C(x,τ)`); `total_C_by_ell` is
`{0: total_C}`.

### The result dict (spatial `k≥3`, early return at `:463`)

`{'C_kpoint': ndarray (n_pts,), 'C_kpoint_by_ell': {ℓ: ndarray}, 'points':
ndarray (n_pts,k-1,2), 'spatial_info': dict, 'k', 'max_ell', 'mf'}`.

---

## Data flow

**In:** `model` (dict) + `fundamental` (`{name: value}`) + `k`, `max_ell`,
`external_fields` (list of `(name, pop)`).

**Threaded between phases:**

```
  model ─┬──────────────────────────────────────────────────────────────┐
         ▼                                                                │
  [1] FieldTheory(model, taylor_order).expand()  ──►  ft (._ns,._n_tilde, │
         │                                              ._by_tp, sectors)  │
         ▼                                                                 │
  [2] build_propagator(ft, model)  ──►  prop {K_ft, G_ft, D_delta, ω, nf} │
         │                                                                 │
         ▼                                                                 │
  [3] solve_mean_field(ft, model, fundamental)  ──►  num_params {SR:float}│
         │                                                                 │
         ▼ (spatial? → early return via pipeline_bridge)                   │
  [4] compute_poles_and_residues(prop, num_params)  ──► prop.pole_vals,   │
         │                                                  prop.C_mats    │
         ▼                                                                 │
  [5] enumerate_unique_diagrams(ft, model, k, ℓ, ext, G_ft, resp/phys_idx,│
         │      vtypes, stypes)  ──►  unique_by_ell {ℓ:[TypedDiagram]}     │
         ▼                                                                 │
  [6] classify_coefficient_factors(td, time_dep, noise)  ──►              │
         │      diagram_records [{td, classify, prefactor, mult, ell}]     │
         ▼                                                                 │
  [7] compute_correction_td(td_list[ℓ], prefactors[ℓ], propagator_data,   │
         │      k, external_fields, num_params)  ──►  td_result {total_C,  │
         │                                            total_C_batch, …}    │
         ▼                                                                 │
      total_C = Σ_ℓ ;  C_tau = Σ_ℓ C_tau_by_ell  ──►  result dict ────────┘
```

**Concrete example (Hawkes-like, `k=2`, tree, single τ-axis).**
`external_fields=[('n',1),('n',2)]` → normalized to `[('dn',1),('dn',2)]`. Phase
[7] sets `tau_points=[(0.0, t) for t in tau_grid]` (leaf 0 pinned to 0, leaf 1
swept over the grid). Each τ-point is passed as `total_C(0.0, t)` →
`C_tau_by_ell[0]` is the resulting complex array; with no loops, `C_tau ==
C_tau_by_ell[0]`. The notebook plots `tau_grid` vs `C_tau.real`.

---

## Gotchas & caveats

* **`compute_poles_and_residues` mutates `prop` in place** (`:632`). There is no
  return value used; the orchestrator relies on the side effect filling
  `pole_vals`/`C_mats`. The same `prop` object is later handed to Phase [7] and
  returned in the result dict.

* **Spatial models MUST be given `spatial_grid` (k=2) or `spatial_points`
  (k≥3).** A spatial `G_ft` carries the inert `Laplacian` symbol the temporal
  ω-pole-finder cannot consume. If a spatial model reaches Phase [4] without a
  spatial input, the code raises a clear `ValueError` (`:619`) rather than dying
  with a cryptic Sage `TypeError`. Conversely, passing `spatial_grid` to a
  **non-spatial** model warns and ignores it (`:390`).

* **`spatial_grid` is the k=2 grid API; `spatial_points` is the k≥3 event API.**
  Mixing them raises (`:404`, `:410`). Spatial `k≥3` returns a *different* dict
  shape (events, not a τ-grid) — callers must branch on `k`.

* **Spatial parallelism is THREAD-based, never fork-based.** The default
  `spatial_parallel=True` routes the loop integrator through a
  `ThreadPoolExecutor` (numpy releases the GIL). The memory note and the
  `fork_safety` module document that **forking after Cocoa/BLAS/matplotlib init
  inside a macOS Jupyter kernel hard-crashes the kernel and the OS** — and that
  `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` makes it *worse*. The temporal
  `parallel` flag *is* fork-based but is guarded: `fork_unsafe_in_notebook`
  (`msrjd/fork_safety.py:27`) trips only on `darwin` + a live ZMQ/Jupyter kernel
  + the `fork` start method, degrading that one case to serial with a one-time
  warning. pytest / Linux / terminal / scripts keep fork.

* **The dedup multiplicity is NOT multiplied back into the weight** (`:686`–`:692`,
  and `deduplicate_with_multiplicities`'s docstring). Under Path A the symmetry
  factor `𝒮(Γ)` (orbit–stabilizer count via Sage automorphisms) already carries
  each class's full weight; multiplying by the class size would double-count.
  `multiplicity` is kept purely for diagnostics. Historically a caller-side
  `mult` multiplication compensated for an *incomplete* signature that merged
  non-isomorphic diagrams; `diagram_signature` is now a complete isomorphism
  invariant (canonical form of the coloured incidence digraph), so that
  compensation is obsolete and removed.

* **Cache versioning matters.** The diagram cache stage name is
  `unique_typed_mult_v3_…` (`_diagrams.py:150`). Three documented bumps:
  multiplicity-aware dedup (`v1`), `NoiseSourceType` promotion (`v1→v2`), and the
  complete-isomorphism signature (`v2→v3`). Loading an older cache resurrects
  silenced bugs (collided non-isomorphic classes dropping integrals at `k≥3`),
  so the version suffix forces a rebuild. The expand cache and propagator cache
  are separate (`saved_theories/<tag>/`).

* **`taylor_order` floor of 2 (was 4).** Lowering the floor saved ~90 min on
  heavy Bernoulli theories at `k=2, max_ell=0` but means a user probing
  higher-order vertices for cache-invalidation testing must pass an explicit
  override (`:278`, docstring).

* **`per-ℓ Phase J` with empty `records_ell` contributes a zero callable + zero
  array** (`:746`–`:755`) — necessary because some `(k, ℓ)` configurations have
  no diagrams at a given loop order; otherwise `total_C` would be missing a key.

* **`num_params` must be fully numeric.** If any kernel symbol or saddle value is
  left symbolic, the propagator pole-finder silently returns 0 roots (warned in
  `compute_poles_and_residues`, `_propagator.py:976`). The mean-field phase is
  responsible for substituting *every* symbol, including formal `phi0_i`
  (resolved from `mf_bg_conditions`, `_mean_field.py:272`) — a subtle bug where a
  missing `phi0_i` collapses the whole correlator to 0.

* **`mf_values` drops NaN-only entries** (`:878`). A model may declare a saddle
  symbol (e.g. `mstar` in a non-GTaS model that still ships the declaration) that
  the solver never substitutes; those would otherwise appear as all-NaN vectors.

* **DAE `fixed_point_index` indexes the STABLE subset only** (`:910`–`:923`
  comment). `mf_stable_roots` is that subset in sort order; unstable roots are
  surfaced via `mf_unstable_roots` for inspection but are not selectable.

* **Phase-wall summary is opt-in.** `phase_walls` is `None` when `verbose=False`,
  so the per-phase timing dict in the result is only populated in verbose runs
  (`:296`).

* **Coupled spatial `k≥3` rejects drift.** If the reaction-diffusion `V` matrix
  is nonzero, it raises `NotImplementedError('coupled k>=3: drift (V != 0) not
  supported.')` (`:450`).

* **Grouped Phase J is a prototype.** `use_grouped_phase_j=True` routes to
  `compute_correction_td_grouped` (`:761`) — see `pipeline/_grouped_phase_j.py`
  for math + caveats; the default path is `compute_correction_td`.

---

## Glossary

* **MSR-JD field theory** — Martin–Siggia–Rose / Janssen–De Dominicis: the
  path-integral reformulation of a stochastic (Langevin or point-process)
  dynamics using a physical field and a conjugate *response* (tilde) field.
* **Response field (tilde field)** — the conjugate field `φ̃` in MSR-JD; sourcing
  it produces response functions.
* **Cumulant / connected correlation function** — the connected `k`-point
  function; the sum of all connected Feynman diagrams with `k` external legs.
  This is what `compute_cumulants` returns directly.
* **Bigrade `(n_t, n_p)`** — the (response-degree, physical-degree) of an action
  monomial; how the expanded action is classified into MF / free / interaction
  sectors.
* **Mean-field saddle** — the deterministic steady state about which the loop
  expansion is taken; where the linear action sectors vanish.
* **Free action / bilinear kernel** — the Gaussian `(1,1)` part of the action;
  its time-domain matrix `K_ker`, Fourier image `K_ft`, and inverse `G_ft` (the
  bare propagator).
* **Propagator `G_ft`** — `K_ft⁻¹`, the two-point Green function in frequency;
  its poles are relaxation modes, its residues `C_mats` give the time-domain
  exponentials.
* **Pole / residue (`pole_vals`, `C_mats`)** — the upper-half-plane poles of
  `G_ft(ω)` (retarded modes) and the matrix residues at each.
* **`D_delta`** — the instantaneous (white) part `lim_{ω→∞} G_ft`; the δ(t)
  coefficient in the time-domain propagator.
* **Loop order `ℓ` (`max_ell`)** — the number of independent loops in a diagram;
  `ℓ=0` is tree level. Each loop is one power of the small parameter.
* **Taylor order** — the truncation of nonlinear action functions; auto-set to
  `max(k + 2·max_ell, 2)`.
* **Typed diagram (`TypedDiagram`)** — a Feynman diagram with a definite field
  type on every leg/edge and a propagator on every edge.
* **Symmetry factor `𝒮(Γ)`** — the combinatorial weight `(∏ Wick factorials) /
  |Aut_fixed_ext|`; the canonical Feynman rule.
* **`Aut_fixed_ext` / `Aut_free`** — automorphism groups of the coloured
  incidence digraph with external leaves pinned vs free; their ratio is the
  external-Wick compensation.
* **Coloured incidence digraph** — the bipartite graph (vertex-nodes +
  edge-nodes) coloured by field/vertex/propagator type, fed to Sage's
  automorphism/canonical-label routines.
* **`num_params`** — the `{SR.var → float}` dict that makes every symbolic
  quantity numeric.
* **`SR` (Symbolic Ring)** — Sage's symbolic-expression type.
* **`CDF` (Complex Double Field)** — Sage's machine-precision complex numbers.
* **`CyclotomicField(4)`** — the exact field `ℚ[i]` used for canonical propagator
  inversion.
* **Phase J** — the time-domain integration stage (`compute_correction_td`):
  per-diagram vertex-time quadrature using the pole-residue propagator.
* **`fsolve`** — SciPy's multivariate root-finder, used for the saddle.
* **Fork vs thread parallelism** — fork copies the parent process (fast, but
  crashes a macOS Jupyter kernel after Cocoa/BLAS init); threads share memory
  (safe; numpy releases the GIL). Temporal stages fork (guarded); spatial stages
  thread.
* **Prediagram → typed → causal → unique** — the four-stage diagram enumeration:
  topological skeletons → field-typed assignments → causality-filtered → deduped.
* **`pole-residue form`** — representing the propagator as `Σ C_k e^{-iω_k t} +
  D_delta δ(t)`, the form Phase J integrates against.

---

## Proposed manual subsections

1. **Role of `compute_cumulants` in the pipeline** — the one call that ties
   every subsystem together; what feeds it, what consumes it.
2. **The MSR-JD expansion in one page** — generating functional, response field,
   action sectors, why cumulants = connected diagrams.
3. **The seven phases at a glance** — a numbered tour of [1]–[7] with the data
   each produces.
4. **Phase [1] expand & the Taylor budget** — bigraded action, `max(k+2ℓ, 2)`,
   the expand cache.
5. **Phase [2]/[4] the propagator** — `K_ker → K_ft → G_ft`, poles and residues,
   the three-tier exact path.
6. **Phase [3] the mean-field saddle** — `fsolve`, `num_params`, the DAE
   multi-root branch.
7. **Phase [5] diagram enumeration** — prediagram → typed → causal → unique;
   cache versioning.
8. **Phase [6] symmetry factors & coefficient classification** — `𝒮(Γ)`, why
   multiplicity is not re-multiplied, time-dependent vs constant factors.
9. **Phase [7] Phase J & the per-ℓ assembly** — `compute_correction_td`,
   `total_C_batch`, summing across loop orders.
10. **The spatial-vs-temporal branch** — when and how the spatial short-circuit
    fires; the two early-return dict shapes.
11. **Parallelism: fork vs thread, and the macOS notebook hazard** — the fork
    guard, `spatial_parallel`, the crash story.
12. **The result dict** — every key, the natural-name accessors, NPZ/CSV save.
13. **Performance: the phase-wall summary and where time goes**.
14. **Gotchas & failure modes** — symbolic leftovers → zero correlator, stale
    caches, spatial-input requirements.

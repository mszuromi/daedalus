# Temporal Phase-J Integration (the loop integral engine)

**Slug:** `phasej-temporal`

**Primary source files**

- `msrjd/integration/time_domain/final_integral.py` (5269 lines — the heart of the engine)
- `msrjd/integration/time_domain/pipeline.py` (the orchestration / batching layer)
- `msrjd/integration/time_domain/subgraph.py` (a near-stub for a never-finished loop-reduction route)

**Supporting files read while writing this brief** (you will need to follow these to fully understand the engine):

- `msrjd/integration/time_domain/propagator_td.py` — builds the time-domain propagator `G(t)`
- `msrjd/diagrams/symmetry.py` — `external_wick_compensation` (the combinatorial divisor)
- `msrjd/fork_safety.py` — the canonical fork-in-notebook crash guard
- `msrjd/core/vertices.py` — `NoiseSourceType`, `ConvVertexType`

---

## Overview

### What this subsystem is, in plain language

In the Martin–Siggia–Rose–Janssen–De Dominicis (MSR-JD) field theory the
quantity we ultimately want is a **time-domain correlation/response
function** of a stochastic dynamical system — for example the
auto-correlation `C(τ) = ⟨δn(t) δn(t+τ)⟩` of a Hawkes process, or a
higher-order cumulant. Perturbation theory expresses that correlator as a
sum over **Feynman diagrams**. Each diagram is a graph whose:

- **edges** are *retarded propagators* `G_R(t_v − t_u)` (the linear
  response of one field to another, carrying a Heaviside `Θ` that enforces
  causality: a response can only happen *after* its cause), and whose
- **internal vertices** are interaction points whose *times* must be
  integrated over the whole real line, subject to the causal orderings the
  Heavisides impose.

So the value of one diagram is a multi-dimensional integral of the form

```
C_Γ(external times)
   =  prefactor · ∫ ds_1 … ds_m  ∏_edges  G_R(t_head − t_tail)
```

where `s_1 … s_m` are the internal-vertex times. **This subsystem is the
machine that performs that integral**, for every diagram, at every loop
order, returning a Python callable `f(*external_times) -> complex` that you
can evaluate on any grid of external times.

In the project's internal pipeline vocabulary this is **"Phase J"** — the
final, time-domain evaluation layer of a hybrid pipeline whose earlier
phases (I and before) work in frequency space. Phase J was built as the
"MVP" tree-level proving ground but the core integrator
(`integrate_diagram`) has since been generalized to **any loop order**.

The cleverness of the engine is that it almost never falls back to brute
numerical quadrature. Because every retarded propagator is a finite sum of
decaying complex exponentials, the whole integrand is a sum of terms
`A · exp(linear combination of times)`, and the integral of such a term
over a polytope (the causal-ordering region) has a **closed form**. The
engine therefore has three dedicated *analytic* integrators — for 1, 2, and
≥3 internal-vertex integration variables — and only routes to
`scipy.integrate.quad`/`nquad` when the analytic path hits a degeneracy or
overflow.

### Where it sits in the end-to-end pipeline

**Upstream (what feeds it):**

1. A list of **typed diagrams** (`TypedDiagram` objects) from the diagram
   enumeration + typing + symmetry-dedup pipeline.
2. A **scalar prefactor** per diagram (the combinatorial/coupling weight,
   from `classify_coefficient_factors`).
3. The **retarded propagator in pole–residue form**: a dict
   `propagator_data` with keys `pole_vals` (the poles `p_α` in the upper
   half ω-plane), `C_mats` (one residue matrix per pole), and `D_delta`
   (the δ-function / instantaneous part).
4. The user's **external field list** and **numerical parameter
   substitutions** (`num_params`).

**The entry point** is `compute_correction_td` in `pipeline.py`, which
loops over diagrams calling `integrate_diagram` (in `final_integral.py`) on
each.

**Downstream (what consumes its output):**

- A callable `total_C(*tau)` and a batched `total_C_batch(tau_list)` that
  notebooks and comparison scripts call to produce theory curves to overlay
  on simulation data.
- A list of `delta_contributions` (distributional shot-noise δ-spikes) that
  a separate helper (`eval_delta_contributions_on_tau_grid` /
  `_on_2d_grid`) discretizes onto a τ grid for plotting.
- Per-diagram diagnostics (`groups`, `subset_evaluators`,
  `subset_m_values`) for debugging which analytic path fired.

A **second consumer of the same analytic integrators** is the *grouped*
Phase-J path (`grouped_integral.py`), which sums residues across diagrams
*before* integrating; it imports `_integrate_2d_polygon_modesum` and
`_integrate_nd_polytope_poset_modesum` directly. This brief documents the
per-diagram engine; the grouped path is a sibling that reuses these
building blocks.

---

## The math

This section builds the theory from the ground up so the code reads as a
literal transcription of the formulas.

### 1. The retarded propagator as a mode sum

For a *linear* response system the frequency-domain propagator
`Ĝ(ω)` is a rational matrix function of ω. Its inverse Fourier transform,
under the project's fixed convention

```
G(t) = (1 / 2π) ∫ dω  exp(iωt)  Ĝ(ω),
```

is computed by residues. The causality filter guarantees every pole
`p_α` lies in the upper half plane (`Im p_α > 0`), so each residue term
`exp(i p_α t)` **decays for t > 0** and **grows for t < 0**. The physical
retarded propagator is the analytic part multiplied by a Heaviside:

```
G_R[p, r](Δt)  =  delta_coeff[p,r] · δ(Δt)            (instantaneous part)
              +  Θ(Δt) · Σ_α  C_α[p,r] · exp(i p_α · Δt)   (smooth part)
```

- `[p, r]` are the **(physical-row, response-column)** matrix indices. The
  convention is `G[pi, ri] = ⟨φ_pi  ñ_ri⟩` — the response of physical field
  `pi` to a response-field (hatted) source `ri`. (See `propagator_td.py`
  docstring item 4: the kernel matrix K is laid out `[resp, phys]`, so you
  read `G[phys, resp]` to get the retarded entry — a transpose.)
- `delta_coeff[p,r] = lim_{ω→∞} Ĝ[p,r](ω)` is the **instantaneous response**
  — the polynomial (non-proper) part of the rational `Ĝ`. It is nonzero
  exactly for instantaneous couplings (e.g. the `ñ × δn` term in the MSR-JD
  action: a `ñ` source at time t produces an *immediate* `δn` at the same
  t). Captured in `propagator_data['D_delta']`.
- `C_α[p,r]` is the residue of `Ĝ` at pole `p_α`, position `[p,r]`. The code
  stores `λ_α := i · p_α` so a mode reads `C_α · exp(λ_α · Δt)`.

In the code this whole object per edge is the **`EdgeModeSum`** dataclass:
`(ri, pi, delta_coeff, modes)` where `modes = ((C_α, λ_α), …)`.

### 2. The diagram integral and the δ-subset expansion

A diagram with edge set E has integrand `∏_{e∈E} G_R(Δt_e)`. Substituting
the two-piece form above and **expanding the product** over which edges are
taken in their δ-form vs. their smooth-form gives a sum over subsets
`S ⊆ E`:

```
C_Γ = Σ_{S⊆E}  ∫ ds_1…ds_m
        · (∏_{e∈S} delta_coeff[e] · δ(Δt_e))           (δ edges)
        · (∏_{e∉S} Θ(Δt_e) · Σ_α C_α[e] e^{λ_α Δt_e})  (smooth edges)
        · combined_prefactor
```

For each subset:

- **δ-edge equations** `Δt_e = 0` (for `e ∈ S`) are linear in the vertex
  times. Each one is solved to **eliminate one integration variable** by
  substitution (e.g. pin an internal vertex's time to a leaf's time). This
  *reduces the dimension* m of the remaining integral.
- If a δ-edge equation has **no integration variable left to solve for**, it
  becomes a constraint *among external times only* — a **shot-noise δ(τ)
  spike**: `C_Γ` contains a `δ(a·τ + c)` term. These are not added to the
  smooth callable; they are emitted as structured `delta_contributions`.
- **Smooth edges** each contribute (i) a factor `Σ_α C_α e^{λ_α Δt_e}` and
  (ii) a **half-space constraint** `Δt_e > 0` (the retardation/Heaviside),
  which together with the other smooth edges' constraints carves out a
  **convex polytope** in the remaining integration variables.

So after δ-elimination, each subset is

```
prefactor · ∏(δ-coeffs) · ∫_{polytope}  ∏_{smooth e} [Σ_α C_α e^{λ_α Δt_e}]  ds_1…ds_m.
```

### 3. The analytic polytope integral via pole-tuple expansion

Each `Δt_e` is an *affine* function of the integration variables
`s = (s_0,…,s_{m-1})`, the free external times `t_free`, and a constant:

```
Δt_e = c0_e + Σ_v a_int_e[v]·s_v + Σ_j a_ext_e[j]·t_free[j].
```

Choose one pole `α_e` per smooth edge (a **pole tuple**). The product of the
chosen modes is a single exponential `C_prod · exp(Σ_e λ_{α_e} Δt_e)`. Group
the exponent by what it multiplies:

```
α_s[v]  = Σ_e λ_{α_e} · a_int_e[v]      (coefficient of s_v)
γ       = Σ_e λ_{α_e} · (c0_e + Σ_j a_ext_e[j]·t_free[j])   (the s-independent part)
```

so the integrand for one pole tuple is

```
prefactor · C_prod · exp(γ) · exp(Σ_v α_s[v]·s_v).
```

The whole subset integral is the sum over all `∏_e (#poles_e)` pole tuples
of `∫_{polytope} exp(Σ_v α_s[v] s_v) ds`. Each such integral is a
**closed-form exponential-over-a-polytope**. Three cases, three integrators:

#### m = 1 — interval `[L, U]`

```
∫_L^U exp(α_s · s + γ) ds = exp(γ) · (e^{α_s U} − e^{α_s L}) / α_s
```

with the `α_s → 0` limit `(U − L)·e^γ`. L and U come from intersecting the
half-line constraints. Unbounded sides (`±∞`) are allowed and evaluate to 0
*iff* the integrand decays in that direction (`sign(Re α_s)` matches);
otherwise the term diverges and the integrator returns `None` (fall back to
`scipy.quad`).

#### m = 2 — convex polygon

The polytope is a convex polygon (intersection of half-planes). The
algorithm (Stage 3a):

1. Start from a CCW bounding box `±bbox_cap` and **clip** it against each
   constraint half-plane with the **Sutherland–Hodgman** polygon-clipping
   algorithm (`_clip_polygon_to_halfplane`).
2. **Fan-triangulate** the polygon from vertex 0.
3. For each triangle, **affine-map** to the unit triangle
   `{0 ≤ u, w; u+w ≤ 1}`. The integrand becomes `exp(α₀ + p·u + q·w)`.
4. The unit-triangle integral has the closed form
   ```
   J(p, q) = ∫₀¹du ∫₀^{1-u}dw exp(p u + q w)
           = [ (e^p − e^q)/(p − q) − (e^p − 1)/p ] / q
   ```
   with **Taylor fallbacks** when any of `|p|, |q|, |p−q|` → 0 (those
   denominators would otherwise produce catastrophic cancellation — see
   `_exp_over_unit_triangle`).

#### m ≥ 3 — causal poset / chain-simplex decomposition

For deeply nested integrals the constraint set `{Δt_e > 0}` is read as a
**directed acyclic graph (DAG)** on the m integration variables: each
inter-axis constraint `s_v − s_u > 0` is an edge `u → v`, and each
single-variable constraint is a scalar bound `s_v > c` or `s_v < c`. The
full polytope is the **disjoint union of simplex regions, one per linear
extension (topological sort)** of that DAG. Each simplex region is a *chain*

```
L ≤ s_{σ(0)} ≤ s_{σ(1)} ≤ … ≤ s_{σ(m-1)} ≤ U_σ
```

The nested exponential integral over a chain has a closed form obtained by
integrating inside-out; each of the m−1 inner integrations splits one term
into two (an "upper-bound" piece that merges its coefficient into the
next-outer variable, and a "lower-bound" piece that pulls out a constant
`exp(β·L)`), giving `2^N` constant terms at the end whose sum is the
integral (`_exp_over_chain_simplex`). When a cumulative coefficient `β`
vanishes the closed form's `1/β` is singular; the **polynomial-prefactor
variant** (`_exp_over_chain_simplex_polynomial`) handles that by carrying a
polynomial-in-s antiderivative through subsequent levels. Scalar uppers at
*non-top* positions are handled by `_chain_with_intermediate_uppers`, which
splits the chain at the crossing positions and enumerates "cut tuples."

### 4. Translation invariance / origin pinning, and external-Wick sums

The correlator depends only on *time differences*, so one external leaf
(`origin_leaf_idx`, default 0) is **pinned to t = 0** during integrand
construction. The returned callable therefore really computes a function of
`(t_j − t_origin)`.

When several external legs carry the **same field type** (e.g. two `δn₁`
legs), each distinct assignment of canonical positions to leaves is a
separate **Wick contraction** and must be summed. The engine enumerates all
field-respecting position↔leaf mappings, sums the integrand over them, and
divides by the **external-Wick compensation factor**

```
comp = |Aut(Γ, leaves free)| / |Aut(Γ, leaves fixed)|
```

(`external_wick_compensation` in `symmetry.py`), which by orbit–stabilizer
is the exact count of how many times the mapping sum reproduces each
distinct pinned diagram. (Historically this was an over-dividing heuristic
based on `vertex_role_signature`; see the Gotchas section.)

### 5. Non-local kernels (noise sources and conductance vertices)

Two vertex types break the pure rational-propagator structure and need
extra integration variables:

- **`NoiseSourceType`** (non-white noise cumulants): the source's legs sit
  at *independent* times coupled by a kernel `κ^{(n)}(τ)`. The engine adds
  one τ per noise vertex, routes one leg to `anchor − τ`, and substitutes
  the kernel `κ(τ)` into the prefactor. The kernel is generally *not* a sum
  of exponentials (e.g. Gaussian `exp(−τ²/2σ²)`), so these diagrams force
  the **slow SR + `scipy.nquad`** path.

- **`ConvVertexType`** (conductance-style synaptic kernels `g(τ)`): same
  scaffold, but if `g(τ)` **single-exponential-extracts** (`τ_g·exp(−τ/τ_g)`
  is fine; polynomial-prefactor "alpha" kernels are not), the kernel becomes
  a synthetic **pseudo-edge** `(C, λ)` with constraint `Δt = +τ > 0` and the
  diagram stays on the fast **analytic** path.

---

## External tools used

This codebase straddles **SageMath** (a computer-algebra system built on
Python) and the scientific-Python stack. Here is each library, what it is,
and exactly how this subsystem uses it.

### SageMath (`sage.all`)

**What it is.** SageMath is a large open-source mathematics system that
wraps dozens of specialized libraries (Maxima for symbolic algebra, PARI,
GMP, etc.) behind a unified Python API. When you `from sage.all import …`
you get Sage's *symbolic ring* and number types. The crucial objects here:

- **`SR`** — the **Symbolic Ring**. `SR.var('x')` makes a symbolic variable;
  `SR(expr)` coerces a value into a symbolic expression. Symbolic
  expressions support `.subs(dict)` (substitute), `.expand()`, `.diff()`,
  `.coefficient(v)`, `.variables()`, `.is_zero()` / `.is_trivial_zero()`.
- **`CDF`** — the **Complex Double Field**: machine-precision complex
  numbers. `complex(CDF(SR(expr)))` is the standard idiom in this file to
  turn a fully-substituted symbolic scalar into a Python `complex`.
- **`fast_callable(expr, vars=…, domain=CDF)`** — Sage's **JIT compiler for
  symbolic expressions**. It walks the expression tree once and produces a
  fast Python-callable (backed by a bytecode interpreter / C evaluator) that
  numerically evaluates `expr` at given variable values. Used to compile the
  residual SR integrand for the `scipy.nquad` fallback path.
- **`solve` (imported as `sage_solve`)** — symbolic equation solver. Used on
  the δ-edge equations `Δt_e == 0` to solve for an integration variable.
- **`matrix`, `I`, `exp`, `heaviside`, `limit`, `oo`** (in `propagator_td.py`)
  — symbolic matrix constructor, the imaginary unit, symbolic exp/Heaviside,
  and the `ω → ∞` limit used to compute `delta_coeff` when `D_delta` is not
  precomputed.

**Exact import lines:**

```python
# final_integral.py:83
from sage.all import SR, fast_callable, CDF, solve as sage_solve
# propagator_td.py:53
from sage.all import SR, I, exp, heaviside, matrix, CDF
# pipeline.py:35
from sage.all import SR
```

**How it is used.** Sage is the *authoring/extraction* layer: the diagram's
propagator entries, the δ coefficients, and the prefactor all arrive as SR
expressions. The engine extracts the numerical pole/residue data **once**
(`_build_edge_mode_sums`, using `complex(CDF(SR(...)))`), solves the δ
equations (`sage_solve`), and reads linear coefficients off the constraint
SR expressions (`c_sr.coefficient(v)`). After that, the hot numerical loops
use **plain Python `complex` and `cmath`**, never touching Sage again — Sage
is too slow per-call.

### `cmath` / `math` (Python standard library)

`cmath.exp` is the complex exponential at the core of every analytic
integrator's inner loop; `math.inf`, `math.isinf`, `math.factorial` appear
in the bound resolution and the polynomial-prefactor chain. These are
imported locally inside the hot functions (e.g. `import cmath`) so the
module-level namespace stays clean.

### NumPy (`numpy`, imported as `np`)

**What it is.** The standard N-dimensional array library for Python.

**How it is used here.** Two roles: (1) it backs the **numba** chain-simplex
kernel, which operates on pre-allocated `np.complex128` buffers
(`np.zeros((max_terms, N), dtype=np.complex128)`), and (2) it provides the
output arrays and `np.argmin`/`np.abs` for the **δ-spike grid discretizers**
(`eval_delta_contributions_on_tau_grid`, `_on_2d_grid`).

```python
# final_integral.py:1041
import numpy as np
```

### Numba (`numba`, imported as `_numba`)

**What it is.** Numba is a **just-in-time compiler for a subset of Python +
NumPy**. You decorate a function with `@numba.njit` ("no-python JIT") and
the first time it is called Numba compiles it to native machine code via
LLVM, typically 10–100× faster than the interpreted version for tight
numeric loops.

**How it is used here.** The `2^N` chain-simplex term expansion (the
dominant cost for m ≥ 3 diagrams) is reimplemented as
`_exp_over_chain_simplex_numba_core`, an `@_numba.njit(cache=True)` function
on pre-allocated complex128 buffers with ping-pong term lists. It returns a
`(status, value)` pair (status 0 = ok, 1 = degenerate β, 2 = overflow) so
the pure-Python caller can fall back appropriately. The import is **guarded**:

```python
# final_integral.py:1043-1050
try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    _numba = None

USE_NUMBA_CHAIN_SIMPLEX = True
```

If numba is missing, or `USE_NUMBA_CHAIN_SIMPLEX = False`, the dispatcher
(`_exp_over_chain_simplex_fast`) silently uses the pure-Python reference
implementation, which is guaranteed bit-compatible on converged results.

### mpmath (`mpmath`)

**What it is.** A pure-Python **arbitrary-precision** floating-point library:
`mpmath.mpc` is a complex number with a configurable number of decimal
digits (`mp.dps`).

**How it is used here.** Two *experimental, off-by-default* precision rescue
paths for the m ≥ 3 chain simplex, where the `2^N` cancellation can lose
~14 digits of float64 precision on close-paired poles:
`_exp_over_chain_simplex_mpmath` re-runs the closed form at `dps = 50`, and
`USE_POSET_MPMATH_ACCUMULATION` accumulates the outer pole-tuple sum in
mpmath. Both imports are local and wrapped in `try/except ImportError`.
Both flags default `False` (they did not actually fix the targeted bug — see
Gotchas).

### SciPy (`scipy.integrate`)

**What it is.** SciPy is the standard scientific-computing library; its
`integrate` submodule provides **adaptive numerical quadrature**:

- `quad(f, a, b, **opts)` — adaptive 1D integration (Gauss–Kronrod /
  QUADPACK), handles `±inf` bounds natively via coordinate transforms.
- `nquad(f, [bounds…], opts=…)` — nested multi-dimensional quadrature where
  each axis's bounds may be a **callable of the outer variables**.

**How it is used here.** SciPy is the **fallback** integrator: whenever an
analytic integrator returns `None` (degenerate, overflow, mixed constraint,
unbounded-divergent), the subset closure routes to `_integrate_polytope`,
which dispatches to `_integrate_1d_polytope`/`_integrate_2d_polytope`/
`_integrate_nd_polytope`, all built on `quad`/`nquad`. Real and imaginary
parts are integrated separately (`_complex_quad`). Accuracy is tuned via the
module-level `QUAD_OPTS = {'limit': 200}` knob. Imports are local
(`from scipy.integrate import quad` / `nquad`) inside the integrator
functions.

### `multiprocessing` (Python standard library)

**What it is.** Process-based parallelism. A `Pool` of worker processes
distributes a function over a list of inputs. The **start method** controls
how workers are created: `fork` copies the parent's whole memory image
(POSIX only), while `spawn` starts a fresh interpreter and **pickles**
arguments across the boundary.

**How it is used here.** `total_C_batch` / `eval_per_diagram_batch` fan the
τ-grid evaluation out over a `Pool`. **Fork is mandatory** because the
per-diagram `contribution` closures are nested functions that stdlib
`pickle` cannot serialize — fork inherits them via memory. A module-level
`_WORKER_STATE` dict carries the closures into the workers (inherited by
fork, *not* pickled). See the fork-safety guard in the Gotchas section.

### `itertools` / `functools` / `dataclasses` / `warnings` (stdlib)

- `itertools.permutations` enumerates the external-Wick leaf mappings.
- `functools.lru_cache` memoizes the subset-sum scan
  (`_min_subset_sum_abs_cached`).
- `@dataclass(frozen=True)` defines the immutable `EdgeModeSum`,
  `_CausalPoset`, and `LoopSubgraph` records.
- `warnings.warn` emits the fork-in-notebook fallback and mixed-leaf-type
  cautions.

### NetworkX / nauty / SymPy

**Not used in this subsystem.** Graph structure is handled by SageMath's own
graph objects (`D.vertices()`, `D.edges()`, `D.num_edges()`), not NetworkX.
nauty (graph canonicalization) lives in the *enumeration* subsystem upstream,
not here. SymPy is not imported anywhere in these files — all symbolic work
is SageMath's `SR`.

---

## Components

This is the exhaustive per-function reference. Functions are grouped by
role; file:line locations are for `final_integral.py` unless noted.

### Pipeline orchestration (`pipeline.py`)

#### `compute_correction_td(...)` — `pipeline.py:202`

```python
def compute_correction_td(
    typed_diagrams=None, prefactors=None, propagator_data=None,
    k=None, num_params=None, ext_time_vars=None, origin_leaf_idx=0,
    external_fields=None, edge_mode_sums_builder_fn=None,
    kernel_groups=None,
)
```

**The public Phase-J entry point.** Takes either `typed_diagrams +
prefactors` (preferred) or a legacy `kernel_groups` list (unpacked into the
same). Steps:

1. Default `ext_time_vars` to `[t_1,…,t_k]` if not given.
2. For each `(typed_diagram, prefactor)`: compute the loop number
   (`_loop_number_from_graph`), call `integrate_diagram(...)`.
3. If the result status is `'ok'`, store the `contribution` callable, append
   its `delta_contributions`, and record per-diagram diagnostics (the
   `subset_evaluators` list lets the caller see at a glance whether the
   intended analytic path fired). Otherwise mark the diagram **skipped**.
4. Define `total_C(*ext_time_values)` = sum of all diagram callables.
5. Define `total_C_batch` and `eval_per_diagram_batch` (nested closures —
   the reason fork is required downstream).
6. **Returns** a dict with `total_C`, `total_C_batch`,
   `eval_per_diagram_batch`, `delta_contributions`, `groups`,
   `skipped_kernel_ids`, `ext_time_vars`.

**Note:** all loop orders pass through `integrate_diagram`; "loop diagrams"
are no longer skipped at the orchestration level (the docstring's
"loop diagrams are marked as skipped" is stale — the inline comment at
`pipeline.py:291` explains that the DAG structure keeps tree and loop on the
same algorithm).

#### `total_C_batch(tau_points, parallel=True, n_workers=None, start_method='fork')` — `pipeline.py:374` (nested in `compute_correction_td`)

Fans `total_C` evaluation over a τ grid. Two parallel strategies auto-chosen
by batch shape:

- **Per-τ parallelism** (`len(tau_points) ≥ n_workers`): each worker
  evaluates the full `total_C(τ_i)` summing all diagrams serially. Worker
  entry `_worker_eval_total_C`.
- **Per-(τ, diagram) nested parallelism** (`len(tau_points) < n_workers`):
  each worker does one diagram at one τ; the parent aggregates **in
  ascending diagram order** so float accumulation is bit-identical to serial.
  Worker entry `_worker_eval_one_diagram`.

Guards in order: (1) **fork-in-notebook safety** (`_fork_unsafe_in_notebook`
→ degrade to serial with a one-time warning); (2) serial fast-path when
`total_tasks < max(4, 2·n_workers)` (pool setup not worth it); (3) sets
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` *only after* the guard has cleared
fork as safe.

#### `eval_per_diagram_batch(...)` — `pipeline.py:502` (nested)

Like `total_C_batch` but returns the **full (n_tau, n_diagram) grid** of
per-diagram contributions plus a `loop_nums` list. Same fork guard, same
serial-threshold logic, always the per-(τ, diagram) task split.

#### `_worker_eval_total_C(tau_tuple)` — `pipeline.py:174`

Module-level (picklable) worker: looks up `_WORKER_STATE['total_C']`
(inherited by fork) and returns `complex(total_C(*tau_tuple))`.

#### `_worker_eval_one_diagram(task)` — `pipeline.py:186`

Worker: `task = (tau_idx, diagram_idx, tau_tuple)`, looks up
`_WORKER_STATE['tree_callables'][diagram_idx]`, returns
`(tau_idx, diagram_idx, complex_value)`.

#### `_fork_unsafe_in_notebook(start_method)` — `pipeline.py:146`

Thin alias to `msrjd.fork_safety.fork_unsafe_in_notebook` — True iff
`start_method == 'fork'` **and** macOS **and** running inside a ZMQ/Jupyter
kernel.

#### `_warn_fork_guard_once()` — `pipeline.py:156`

Emits the fork-fallback `RuntimeWarning` at most once per process.

### Core integrator (`final_integral.py`)

#### `integrate_diagram(...)` — `final_integral.py:2411` (the big one)

```python
def integrate_diagram(
    typed_diagram, propagator_data, combined_prefactor, ext_time_vars,
    num_params=None, origin_leaf_idx=0, external_fields=None,
    representative_ir=None, edge_mode_sums_builder=None,
)
```

**The per-diagram integrator.** Returns a dict whose `contribution` value is
`f(*ext_time_values) -> complex` (k positional args in canonical order; the
`origin_leaf_idx` slot is ignored, having been pinned to 0). Walkthrough of
its ~1300 lines:

1. **§1 Numerical G(t) matrix** (line 2510): build the time-domain
   propagator object `G_t_obj` via `build_G_t_matrix` with `num_params`
   substituted.
2. **§2 External-Wick contraction enumeration** (line 2514): if
   `external_fields` is supplied and matches the leaf count, enumerate all
   field-respecting canonical-position↔leaf mappings (`_all_mappings`) and
   compute the compensation divisor (`external_wick_compensation`). If
   not supplied for a mixed-field diagram, **warn** about a possible
   `τ → −τ` mirror and use the identity mapping.
3. **Vertex-time assignment** (line 2610): map each leaf to its external
   time (pin the origin leaf to `SR(0)`), give each internal vertex a fresh
   integration symbol `s_v{v}_td_`.
4. **§2b NoiseSourceType per-leg time map** (line 2628): one extra τ per
   noise vertex; route incident edges to `anchor` or `anchor − τ` by edge
   key (homogeneous legs use label ordering; heterogeneous match leg
   fields). Records `vertex_leg_time`, `vertex_leg_kind='response'`,
   `noise_source_specs`.
5. **§2c ConvVertexType per-attachment time map** (line 2750): one τ^g per
   kernel attachment, kernel-attached physical leg sits at `anchor − τ`.
   Guards against a vertex being both Noise and Conv (impossible by n_phys).
6. **§3 Per-edge info** (line 2793): for each edge resolve `(ri, pi)`
   (`_lookup_prop_indices`), the head/tail times (`_resolve_leg_time`),
   `dt_sym = t_v − t_u`, `delta_coeff`, `smooth_factor`.
7. **§3a Mode-sum cache** (line 2846): build one `EdgeModeSum` per edge via
   `edge_mode_sums_builder` (spatial hook) or `_build_edge_mode_sums`
   (default). `None` if propagator data incomplete → SR fallback path.
8. **Prefactor** `cp` (line 2863): coerce `combined_prefactor` to SR,
   substitute `num_params`.
9. **§3b / §3c kernel substitution** (lines 2868–2982): replace κ
   placeholders (noise) and `g(τ)` symbols (conv) in `cp`. For conv, attempt
   single-exponential extraction (`_extract_exp_mode`) — success sets
   `conv_kernel_extracted = True` and collects `conv_extracted_modes`,
   substituting the kernel symbol with `1`; failure substitutes the full
   `g(τ)` SR and forces the slow path.
10. **§4 External-time bookkeeping** (line 2983): `free_ext_idx` = non-origin
    positions; `free_ext_syms` = their symbols.
11. **δ-branch pre-classification** (line 2992): edges with
    `|delta_coeff| < 1e-15` are **forced-smooth**; only edges with a nonzero
    δ part are branched on — turning `2^|E|` into `2^|branch|`.
12. **The subset loop** (line 3026, `for branch_bits in range(2^n_branch)`):
    for each subset,
    - split edges into `delta_edges` and `smooth_edges`;
    - **solve δ equations** (`sage_solve`) eliminating integration variables,
      transitively resolving the substitution chain (line 3079);
    - **shot-noise detection** (line 3122): residual external-time equalities
      → emit a structured `delta_contributions` entry (single-equality MVP;
      multi-equality skipped);
    - **analytic-eligibility check** (line 3271): build the early numerical
      prefactor; if eligible (no NoiseSource leg-time, mode-sum cache
      present, numerical prefactor) **skip** the SR + `.expand()` +
      `fast_callable` build entirely;
    - **build retardation constraints** + extract linear coefficients into
      `subset_constraint_data = [(a_int, a_ext, c0), …]` (line 3401);
    - **cap τ_v** kernel variables to `±TAU_KERNEL_CAP` (conv τ lower bound 0
      for causality) (line 3419);
    - **conv pseudo-edges** (line 3481): append synthetic `EdgeModeSum`s with
      `Δt = +τ` and drop the now-redundant `τ > 0` cap;
    - **build the per-call evaluator** (line 3541): the fast numpy evaluator
      `_build_fast_subset_evaluator_from_modes` (preferred) or
      `_build_fast_subset_evaluator` (legacy), or `None` for noise diagrams;
    - **build the modesum plan** (`_build_modesum_plan`, line 3640) caching
      the τ-invariant `α_s/γ` data;
    - **wrap a subset contribution closure** `_make_subset_contrib` (line
      3651) that tries the m=1/2/≥3 analytic integrator and falls through to
      `_integrate_polytope`;
    - tag `_evaluator_label` (intent) and append to `subset_diagnostics`.
13. **Final contribution callable** (line 3737): assemble the per-mapping
    permutation list `_perms`, then `contribution(*ext_time_values)` permutes
    inputs, **re-pins the origin to 0 per permutation** (the key fix so swap
    permutations are physically distinct evaluations, not all fed
    `free_val=0`), evaluates every subset closure, and divides by `_comp`.
14. **Returns** the result dict (`status`, `contribution`,
    `delta_contributions`, `integration_vars`, `stripped_integrand`,
    `constraints`, `edge_info`, `n_subsets_evaluated`, `subset_diagnostics`,
    `cumulant_prefactor`, …).

`integrate_tree_diagram` (`final_integral.py:5269`) is a backward-compat
alias for `integrate_diagram`.

#### `_loop_number_from_graph(typed_diagram)` — `final_integral.py:2401`

`L = |E| − |V| + 1` (first Betti number of the connected graph). No
frequency dependency.

#### `_lookup_prop_indices(typed_diagram, edge_key)` — `final_integral.py:5230`

Resolve `(resp_row, phys_col)` for a prediagram edge, with a 3-level
fallback: exact `(u,v,lbl)` → `(u,v,None)` → first `(u,v)` regardless of
label. Raises `KeyError` if no match.

### Edge mode-sum data construction

#### `EdgeModeSum` (dataclass, frozen) — `final_integral.py:116`

`(ri, pi, delta_coeff, modes, dt_c0, dt_int_pairs, dt_ext_pairs)`. The
canonical numerical representation of one propagator edge (see Data
Structures). The `dt_*` fields are filled per-subset.

#### `_build_edge_mode_sums(edge_info, propagator_data)` — `final_integral.py:150`

Build one `EdgeModeSum` per edge, extracting residues from `C_mats` and
poles from `pole_vals` **once** (`λ_α = i·p_α`). Returns `None` if propagator
data is incomplete (signals the SR fallback path).

#### `_extract_exp_mode(sr_expr, tau_sym)` — `final_integral.py:203`

Extract `(C, λ)` from `C·exp(λ·τ)` using the log-derivative trick:
`λ = d/dτ log g |_{τ=0}`, `C = g(0)`. Returns `None` for
polynomial-prefactor kernels (where `g(0)=0`, e.g. the alpha kernel) — those
need multi-mode handling and force the slow path. Used for ConvVertex
kernels.

#### `_attach_subset_dt(edge_mode_sum, a_int, a_ext, c0)` — `final_integral.py:235`

Return a copy of an `EdgeModeSum` with the per-subset Δt linear form
populated as sparse `(index, coef)` pairs (dropping near-zero coefficients).

#### `_enumerate_pole_tuples(edge_mode_sums)` — `final_integral.py:529`

Generator yielding `(C_product, lambdas)` over the Cartesian product of
per-edge modes (one pole per smooth edge). Stack-based index advance to avoid
`itertools.product` overhead. Empty input → one tuple `(1.0+0j, ())`.

#### `_build_modesum_plan(smooth_edge_modes, subset_constraint_data, m, n_ext)` — `final_integral.py:595`

Pre-compute the **τ-invariant** per-pole-tuple data so the per-call inner
loop is cheap. Returns a dict with `pole_tuples`, `alphas_per_tuple`
(`m`-tuples of `α_s[v]` per pole tuple), `gamma_const_per_tuple`
(`γ` at `free_vals=0`), and `gamma_slope_per_tuple_per_ext`
(`∂γ/∂t_free[j]`). The per-τ work then reduces to
`γ = γ_const + Σ_j slope[j]·free_vals[j]`.

### Analytic exponential-over-region primitives

#### `_exp_over_unit_triangle(p, q)` — `final_integral.py:356`

Closed form `J(p,q) = ∫₀¹∫₀^{1-u} exp(pu+qw) dw du`, with **4th-order Taylor
fallbacks** for each of the four degenerate regimes (`|p|,|q|,|p−q| <
_J_TAYLOR_EPS = 1e-6`) to avoid catastrophic cancellation.

#### `_exp_over_triangle(v0, v1, v2, alpha, beta)` — `final_integral.py:406`

`∫∫_T exp(αx+βy) dA` for a triangle: affine-map to the unit triangle, call
`_exp_over_unit_triangle`, scale by `|det|·exp(v0_term)`. Returns `None` on a
**bilateral overflow guard** (`|Re p|, |Re q|, |Re v0_term| > 600`) — note
the comment recording that a one-sided relaxation (commit 7f0bf05)
introduced wrong-direction loop corrections and was reverted.

#### `_clip_polygon_to_halfplane(polygon, a, b, c)` — `final_integral.py:454`

Sutherland–Hodgman clip of a CCW polygon against `a·x+b·y+c > 0`. Boundary
vertices counted as inside (measure-zero, irrelevant to the area integral).

#### `_polygon_from_2d_constraints(constraint_data, free_ext_vals, bbox_cap)` — `final_integral.py:499`

Build the convex polygon by starting from a `±bbox_cap` box and clipping with
each `(a_int, a_ext, c0)` constraint (after substituting `free_ext_vals`).
Returns `[]` if empty.

#### `_exp_over_chain_simplex(alphas, lower, upper, eps=1e-9)` — `final_integral.py:910`

Pure-Python reference closed form of the nested chain integral
`{L ≤ s_1 ≤ … ≤ s_N ≤ U}`. Integrates inside-out, doubling the term list each
level (`2^N` terms). Returns `0` for `upper ≤ lower` (zero-measure domain,
crucial for τ < 0). Returns `None` if any cumulative coefficient `|β| < eps`
(degenerate — caller uses the polynomial variant) or on the **600 real-exp
overflow guard**.

#### `_exp_over_chain_simplex_numba_core(alphas, lower, upper, eps)` — `final_integral.py:1219` (only if numba present)

`@njit(cache=True)` translation of the above on `complex128` ping-pong
buffers. Returns `(status, value)`: status 0 ok, 1 degenerate β, 2 overflow.

#### `_exp_over_chain_simplex_fast(alphas, lower, upper, eps=1e-9)` — `final_integral.py:1303`

The **dispatcher**. Order of preference: (1) if
`USE_CHAIN_SIMPLEX_PRECISION_FIX` (default False) and N≥3 and a close-pole
condition (`_min_subset_sum_abs < 0.1`) → mpmath path; (2) numba core if
available + enabled; (3) pure-Python reference. Normalizes the numba
status/`None` semantics back to the reference's `value | None`.

#### `_exp_over_chain_simplex_mpmath(alphas, lower, upper, eps=1e-9, dps=50)` — `final_integral.py:1130`

50-digit-precision re-evaluation of the same closed form. Used only when the
(off-by-default) precision-fix gate trips. Falls back to the float64 path if
mpmath is unimportable.

#### `_exp_over_chain_simplex_polynomial(alphas, lower, upper, eps=1e-9)` — `final_integral.py:1339`

Polynomial-prefactor extension: does **not** return `None` on degenerate β.
When a level's β vanishes the antiderivative is polynomial in s; it carries a
polynomial (basis `(s−lower)^k`) through subsequent levels, using the
closed form `∫(s−lower)^k exp(βs) ds`. Returns `None` only on overflow.

#### `_min_subset_sum_abs(alphas)` / `_min_subset_sum_abs_cached(alphas_tuple)` — `final_integral.py:1092 / 1114`

`min |Σ_{i∈S} α_i|` over non-empty subsets (a cheap `2^N` scan, LRU-cached)
— detects when the chain recursion will hit a small cumulative β.

#### `_chain_with_intermediate_uppers(alphas_chain, L, upper_per_position, U_chain_top)` — `final_integral.py:1489`

Chain integral with **scalar uppers at arbitrary positions** (not just the
top). Computes per-position "effective upper" (non-decreasing), groups into
levels, and enumerates monotone "cut tuples" `(c_0 ≤ … ≤ c_{q-1})` where the
chain crosses each lower upper bound; multiplies the chain-simplex value of
each non-empty piece. Subsumes the old "maximality bail."

### The three analytic modesum integrators

#### `_integrate_1d_polytope_modesum(...)` — `final_integral.py:1976`

```python
def _integrate_1d_polytope_modesum(
    smooth_edge_modes, prefactor_complex, subset_constraint_data,
    free_ext_vals, bbox_cap=POLYGON_BBOX_CAP, pole_tuples=None, plan=None,
)
```

m=1 analytic interval integral. Resolves `[L, U]` (tracking `±inf`), then for
each pole tuple computes `α_s, γ` and the closed form
`(e^{α_s U} − e^{α_s L})/α_s · pref·C_prod·e^γ`, with the `α_s≈0` and
unbounded-endpoint special cases. Has a **plan fast-path** (uses cached
`α_s/γ`) and a legacy per-call path. Returns `None` to fall back to
`scipy.quad`. Also reachable directly from the grouped path via `pole_tuples`.

#### `_integrate_2d_polygon_modesum(...)` — `final_integral.py:2175`

m=2 analytic polygon integral. Builds the polygon once
(`_polygon_from_2d_constraints`), fan-triangulates, and for each pole tuple
computes `α_s, β_s, γ` and sums `_exp_over_triangle` over triangles. Plan +
legacy paths. `None` (or empty polygon → `0`) on failure. Imported by the
grouped path.

#### `_integrate_nd_polytope_poset_modesum(...)` — `final_integral.py:1726`

m≥3 analytic causal-poset integral. Extracts the `_CausalPoset`
(`_extract_causal_poset`), resolves the common scalar lower bound L
(`_causal_poset_consistent_scalar_lower`; physical-margin fallback
`earliest_ext − POSET_PHYSICAL_MARGIN` when none), enumerates linear
extensions, and for each pole tuple × extension sums
`pref·C_prod·e^γ · _chain_with_intermediate_uppers(α_chain, L, …)`. Plan +
legacy paths; optional (off) mpmath accumulation. Returns `None` to fall
back to `scipy.nquad`. Imported by the grouped path.

### Causal-poset machinery

#### `_CausalPoset` (dataclass, frozen) — `final_integral.py:691`

`(m, edges, scalar_lowers, scalar_uppers)` — the DAG + per-variable bounds.
Edge `(u,v)` means `s_v > s_u` (u precedes v).

#### `_extract_causal_poset(subset_constraint_data, free_ext_vals, m, tol=1e-12)` — `final_integral.py:720`

Classify each constraint: inter-axis (`+1/−1`, `c_eff≈0` → poset edge),
scalar lower (`+1` → `s>−c`), scalar upper (`−1` → `s<c`), or **anything else
→ `None`** (mixed/shifted constraint not supported, caller falls back).

#### `_enumerate_linear_extensions(poset)` — `final_integral.py:806`

Recursive Kahn-style topological-sort enumeration yielding every linear
extension (length-m tuple), in deterministic lexicographic order.

#### `_causal_poset_consistent_scalar_lower(poset, tol=1e-9)` — `final_integral.py:857`

Returns `(L, True)` if all variables that have a scalar lower agree on it (or
none have one → `(None, True)`); `(None, False)` if they disagree (caller
falls back). Variables without a lower inherit L via the chain ordering.

#### `_causal_poset_consistent_scalar_upper(poset, tol=1e-9)` — `final_integral.py:891`

Returns the per-variable min scalar upper map (only the chain-top variable in
each extension needs one).

### Fast numerical subset evaluators (for the scipy fallback)

#### `_build_fast_subset_evaluator(propagator_data, prefactor_num, smooth_edges_ri_pi, subset_constraint_data, m_sub)` — `final_integral.py:4186`

(Fix E, legacy) Build a Python closure evaluating
`P · ∏_e Σ_k C_e^{(k)} exp(i p_k Δt_e)` **without** `fast_callable.expand()`
(which would compile `|edges|^|poles|` terms). Extracts poles/residues from
`propagator_data`, sparse Δt pairs from constraints; the closure loops edges
× poles per call. Returns `None` if extraction fails.

#### `_build_fast_subset_evaluator_from_modes(prefactor_num, smooth_edge_modes, subset_constraint_data, m_sub)` — `final_integral.py:4291`

(Stage 2) Same closure semantics but consuming pre-built `EdgeModeSum`s
(residues + λ already extracted), splitting modes back into `(poles,
residues)` via `p = λ/1j`. Preferred over the legacy builder when the cache
exists.

### SciPy polytope fallback layer

#### `_integrate_polytope(integrand_callable, s_constraints, free_ext_vals, m)` — `final_integral.py:4372`

Dispatcher: m=0 (constant, just check constraints), m=1, m=2, m≥3. Increments
the `scipy_nquad_called_m*` runtime counters.

#### `_make_heaviside_filtered_integrand(integrand_callable, s_constraints, free_ext_vals, m)` — `final_integral.py:4411`

Wrap the JIT integrand so it **returns 0 whenever any `Δt_e ≤ 0`** (Θ(0)=0
convention). Necessary because the polytope *bounds* passed to nquad can
overshoot the true region (cap fallbacks, deferred cross-axis constraints),
and for retarded poles `G^sm(Δt)` *grows* for Δt<0 — without the filter the
integrand would add a spurious positive contribution. Constraints
pre-extracted in sparse form; pure-constant infeasible → constant-zero
closure.

#### `_complex_quad(integrand_callable, s_slot_index, other_args, lower, upper)` — `final_integral.py:4520`

1D `scipy.quad` of one slot (real + imag separately), other args fixed.

#### `_resolve_1d_bounds(s_constraints, s_index)` — `final_integral.py:4550`

Intersect half-line constraints into `[L,U]`; returns a **degenerate
`(0.0, 0.0)`** (not `(inf,-inf)`) on infeasibility — important so an outer
`scipy.quad(f, 0, 0)` returns 0 rather than the sign-flipped full-line
integral.

#### `_integrate_1d_polytope(...)` — `final_integral.py:4599`

m=1 via `scipy.quad` on `[L,U]`. Skips the Heaviside filter (bounds exact)
unless `DEBUG_HEAVISIDE_GUARD`.

#### `_integrate_2d_polytope(...)` — `final_integral.py:4646`

m=2 via `scipy.nquad` with `s_0` inner (bounds depend on `s_1`) and `s_1`
outer. Pre-splits constraints into pure_0 / mixed / s1_only / constant. Uses
the **±200 cap** + Heaviside filter when no pure-`s_1` bound exists. Carefully
distinguishes a genuine pure-`s_1` constraint from a pure-external constraint
(the latter must NOT set `pure_s1_found`, else an unbounded axis is sampled —
the source of ~12% k=4 overshoot).

#### `_integrate_nd_polytope(...)` — `final_integral.py:4805`

m≥3 via nested `scipy.nquad`. `_make_bound_fn(k_var)` builds each axis's
bounds callable, **skipping constraints that still couple to a more-inner
axis** (deferred-constraint correctness — regression test
`test_phase_J_nd_polytope_preserves_deferred_constraints`). `OUTER_CAP` is
200 (or `POSET_PHYSICAL_MARGIN=50` when `USE_POSET_CAP_MATCH_SCIPY`). Always
Heaviside-filtered.

#### `_outer_bounds(s_constraints, k_var)` — `final_integral.py:5026`

Bounds on the outermost variable from pure-`k_var` constraints; `(-inf,+inf)`
if none, `(inf,-inf)` if infeasible.

### δ-spike grid discretizers

#### `eval_delta_contributions_on_tau_grid(delta_contributions, tau_grid, free_ext_dim=1, vary_index=0, fixed_values=None)` — `final_integral.py:3828`

Convert symbolic δ-spike contributions to a 1D τ-grid array. For each
contribution with equality `a·x+c=0`, solve `τ_fire = −c'/a_vary`, evaluate
the coefficient callable, check retardation half-spaces, deposit
`coeff/|a_vary|/dtau` into the nearest bin. **Degenerate case** (`a_vary≈0`,
`c'≈0`): the δ fires along the whole slice → deposit the *continuous*
`coeff_fc(τ)` with no `1/dtau` divisor.

#### `eval_delta_contributions_on_2d_grid(delta_contributions, tau1_grid, tau2_grid, free_ext_dim=3, grid_axes=(1,2), fixed_values=None)` — `final_integral.py:4004`

2D analogue: the δ-line `a1·τ1+a2·τ2+c=0` is swept along whichever axis has
the larger coefficient, depositing into the nearest cell.

### Propagator support (`propagator_td.py`)

#### `build_G_t_matrix(propagator_data, t_var, num_params=None)` — `propagator_td.py:135`

Build `G_R(t) = D_delta·δ(t) + Θ(t)·Σ_k C_k exp(i p_k t)` as a dict
`{'smooth': SR matrix, 'delta': SR matrix, 't_var': …}`. Normalizes raw
`complex`/`ComplexDoubleElement` into `a + b·I` SR form (the `_to_sr_ab`
helper) **before** `num_params` subs to avoid a GiNaC `'<' not supported
between complex` crash. Optional `simplify_full` pass gated by
`USE_SIMPLIFY_FULL_IN_GT` (default False — dead weight on the analytic path).

#### `G_t_entry(G_t_obj, phys_idx, resp_idx, t_expr, include_heaviside=True)` — `propagator_td.py:379`

Look up the **smooth** entry `smooth[phys, resp]`, substitute the time
argument, optionally multiply by `heaviside`. Applies the resp↔phys
transpose convention.

#### `G_t_delta_coeff(G_t_obj, phys_idx, resp_idx)` — `propagator_td.py:433`

Return the δ(t) coefficient as a Python complex (or real if imag negligible),
0 if no delta info.

#### `_infer_omega_variable` / `_infer_time_variable` — `propagator_td.py:353 / 459`

Recover the ω symbol (for the `ω→∞` δ-limit) / the time symbol (the only free
variable left in a numerically-substituted G(t) entry), avoiding equality
tests that would trip Maxima on embedded complex coefficients.

### Loop subgraph stub (`subgraph.py`)

#### `LoopSubgraph` (dataclass) — `subgraph.py:34`

`(loop_vars, edges, internal_vertices, attachment_vertices, attachment_kind)`
— the planned record for a reduced loop kernel. Effectively unused by the
current engine.

#### `identify_loop_subgraphs(representative_ir, typed_diagram)` — `subgraph.py:66`

**A stub.** Returns `[]` when `free_freqs` is empty (tree-level) and
**raises `NotImplementedError`** otherwise. The whole frequency-domain
loop-reduction route this file was meant to enable was *superseded* — the
current `integrate_diagram` handles loops directly in the time domain (it
treats every internal vertex as an integration variable regardless of loop
order), so this module is never exercised on the live path. Its docstring
documents the *planned* algorithm only.

---

## Data structures

### `propagator_data` (input dict)

The pole–residue description of the retarded propagator. Keys:

| key | type | meaning |
|---|---|---|
| `pole_vals` | list of SR | the poles `p_α`, all with `Im > 0` |
| `C_mats` | list of Sage SR matrices | residue matrix `C_α[i,j]`, one per pole |
| `D_delta` | Sage SR matrix (optional) | δ-coefficients `lim_{ω→∞} Ĝ[i,j]` |
| `G_ft` | Sage SR matrix (optional) | full `Ĝ(ω)`, used to compute `D_delta` if missing |
| `nf` | int (optional) | matrix size (inferred from C_mats/D_delta otherwise) |

### `EdgeModeSum` (`final_integral.py:116`, frozen dataclass)

| field | type | meaning |
|---|---|---|
| `ri` | int | response-column index `G[pi, ri]` |
| `pi` | int | physical-row index |
| `delta_coeff` | complex | δ(Δt) coefficient (= `ω→∞` limit) |
| `modes` | tuple of `(C_α, λ_α)` | pole-residue pairs; `λ_α = i·p_α` |
| `dt_c0` | float | constant of the Δt linear form (per-subset) |
| `dt_int_pairs` | tuple of `(int idx, float coef)` | sparse coefficients on integration vars |
| `dt_ext_pairs` | tuple of `(int idx, float coef)` | sparse coefficients on free external times |

### `subset_constraint_data` (list)

One tuple `(a_int, a_ext, c0)` per smooth edge (and per cap / pseudo-edge),
encoding `Δt = c0 + a_int·s + a_ext·t_free` and the retardation `Δt > 0`.
`a_int` has length m (= number of surviving integration variables), `a_ext`
length n_ext.

### modesum **plan** dict (`_build_modesum_plan`)

| key | type | meaning |
|---|---|---|
| `pole_tuples` | tuple of `(C_prod, lambdas)` | materialized pole-tuple product |
| `alphas_per_tuple` | tuple of m-tuples of complex | `α_s[v]` per tuple |
| `gamma_const_per_tuple` | tuple of complex | `γ` at `free_vals=0` per tuple |
| `gamma_slope_per_tuple_per_ext` | tuple of tuples | `∂γ/∂t_free[j]` per tuple |

### `_CausalPoset` (`final_integral.py:691`, frozen)

`(m: int, edges: tuple[(u,v)], scalar_lowers: tuple[(var,c)], scalar_uppers:
tuple[(var,c)])`.

### `edge_info` (list of dict, internal)

Per-edge: `{'u','v','lbl','ri','pi','dt_sym' (SR), 'delta_coeff',
'smooth_factor' (SR)}`.

### `delta_contributions` (list of dict, output)

| key | meaning |
|---|---|
| `coeff_fc` | `fast_callable` coefficient `f(*free_ext_vals) -> complex` |
| `equality_a`, `equality_c` | linear δ-surface `a·x + c = 0` |
| `equality_symbolic` | the SR equality |
| `retardation_data` | list of `(a_ext, c0)` half-space constraints |
| `delta_edges`, `free_ext_idx` | bookkeeping |

### `integrate_diagram` result dict (output)

`status` (`'ok'`/`'empty_polytope'`/`'failed'`), `contribution` (callable),
`delta_contributions`, `integration_vars`, `stripped_integrand`,
`constraints`, `edge_info`, `n_subsets_evaluated`, `n_delta_contributions`,
`n_shotnoise_skipped`, `subset_diagnostics`, `cumulant_prefactor`,
`has_cumulant_kernel`.

### `_RUNTIME_COUNTERS` (`final_integral.py:315`)

A module-level diagnostic dict counting how often each path was *attempted*
vs *returned None* vs *fell to scipy.nquad* (per m). Reset with
`_reset_runtime_counters()` before a timed run; "intent" labels live in
`subset_diagnostics['evaluator']`, "runtime" lives here.

---

## Data flow

**In:** `compute_correction_td(typed_diagrams, prefactors, propagator_data,
k, num_params, external_fields, …)`.

**Per diagram** (`integrate_diagram`):

1. `propagator_data` → `build_G_t_matrix` → `G_t_obj` (smooth SR matrix +
   delta matrix).
2. Each edge → `(ri, pi)` + `dt_sym` + `delta_coeff` + `smooth_factor`
   (`edge_info`), then → one `EdgeModeSum` (`edge_mode_sums`).
3. Subset loop produces, per surviving subset, a `subset_constraint_data`
   list, a `_smooth_edge_modes` list, a `_modesum_plan`, and a
   `_make_subset_contrib` closure. δ-only subsets with external-time
   equalities → `delta_contributions`.
4. `contribution(*ext_times)` sums all subset closures over all Wick
   permutations / divided by `_comp`.

**Out of `integrate_diagram`:** the result dict (above).

**Out of `compute_correction_td`:** `{total_C, total_C_batch,
eval_per_diagram_batch, delta_contributions, groups, skipped_kernel_ids,
ext_time_vars}`.

**Concrete call shape (from the docstrings):** for k=2,
`total_C(t_1, t_2)` returns `C` at `τ = t_2 − t_1` (position i is the time of
`external_fields[i]`; `external_fields[0]` is pinned). For a τ sweep,
`total_C_batch([(0.0, τ) for τ in grid])` returns a list of complex values.

**Inside one subset closure** `_contrib(free_vals)`:

```
α_s[v], γ  ←  pole tuples × constraint data  (cached in `plan`)
if   m==1:  try _integrate_1d_polytope_modesum  → value or None
elif m==2:  try _integrate_2d_polygon_modesum   → value or None
elif m>=3:  try _integrate_nd_polytope_poset_modesum → value or None
if None:    resolve c_eff per constraint, call _integrate_polytope (scipy)
```

**δ-spike flow:** `delta_contributions` → `eval_delta_contributions_on_tau_grid`
or `_on_2d_grid` → numpy array added to the smooth `total_C` curve.

---

## Gotchas & caveats

- **m ≥ 3 chain-simplex precision (the OPEN BUG).** On close-paired poles
  (e.g. spike-reset poles `0.329i, 0.351i`, difference `0.022i`) the `2^N`
  closed form suffers cancellation-driven precision loss — empirically a ~4×
  aggregate overestimate of the 1-loop value at spike-reset k=2 ell=1. The
  two attempted fixes are **both off by default** because **neither actually
  cured it**: `USE_CHAIN_SIMPLEX_PRECISION_FIX` (mpmath dispatch on close
  poles) and `USE_POSET_MPMATH_ACCUMULATION` (mpmath outer accumulation,
  cancellation factor only ~21×). The audit (`docs/m_ge3_precision_bug_audit.md`)
  now points at a **bookkeeping difference in pole-tuple construction between
  the grouped and per-diagram paths** (grouped pre-sums cancelled-within-group
  residues; per-diag sums after integration; by linearity they should agree,
  so the 4× points to a construction bug). **This is unresolved.**

- **Cap-mismatch flag is a near-miss, also off.** `USE_POSET_CAP_MATCH_SCIPY`
  was meant to align the analytic poset's unbounded-below fallback
  (`L = earliest_ext − 50`) with the scipy `OUTER_CAP=200` for
  marginal-stability theories (`Re β ≈ 0`, integral genuinely cap-dependent).
  The comment records that aligning the caps "does **not** bring per-diag
  into agreement with grouped" — so the dominant error is elsewhere. Default
  False.

- **The overflow guard MUST stay bilateral.** `_exp_over_triangle`'s
  `EXP_REAL_LIMIT = 600` guard checks `|Re|` on both signs. A Stage-4a
  one-sided relaxation (commit 7f0bf05) "introduced wrong-direction loop
  corrections in spike-reset k=2 ell=1" and was **reverted**. The reason: the
  unit-triangle formula has structural cancellation `(e^p−e^q)/(p−q) −
  (e^p−1)/p`, which becomes float-noise-dominated when one exp underflows and
  the other doesn't.

- **Fork-in-notebook can crash the OS.** Forking after Cocoa/BLAS init in a
  macOS Jupyter kernel has hard-crashed the dev machine twice.
  `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` makes it **worse** (removes the
  guard rail, not the hazard). `total_C_batch`/`eval_per_diagram_batch` now
  consult `_fork_unsafe_in_notebook` and **silently degrade to serial** on
  macOS + ZMQ kernel + fork. The `test_phase_J_total_C_batch_*` bit-identity
  tests pass under **pytest** (a plain interpreter, never a ZMQ kernel), so
  they certify *numerical determinism* but **NOT notebook crash-safety**.

- **Windows is unsupported for parallel batch.** Windows has no `fork`; the
  per-diagram closures are unpicklable nested functions, so
  `total_C_batch(parallel=True)` raises `ValueError: cannot find context for
  'fork'`. The header documents two future fixes (cloudpickle, or top-level
  picklable evaluator classes). POSIX-only for now.

- **Missing `external_fields` can silently mirror the correlator.** For a
  diagram with **mixed leaf field types**, omitting `external_fields` falls
  back to identity leaf→position mapping, which "may produce a τ → −τ mirror
  image of the physical correlator." The code **warns** but proceeds. Always
  pass `external_fields` for k ≥ 2.

- **The external-Wick divisor is subtle.** It must be the exact
  orbit–stabilizer index `|Aut_free|/|Aut_fixed|` from
  `external_wick_compensation`. A previous `∏N!`-over-role-signature heuristic
  over-divided (×⅓/×½ deficits on k=4 OU+εx³ 1-loop cascades) because same
  role signature does not imply the leaf swap is an automorphism. The
  `_mapping_fallback` branch (field-count mismatch) must use `_compensation =
  1`, not the heuristic.

- **Origin re-pinning per permutation.** Each Wick permutation re-subtracts
  the permuted origin time so the origin leaf returns to 0. Before this fix,
  the swap permutation was fed `free_val=0`, producing spurious asymmetry in
  `C(τ)` for every non-tree k=2 identical-externals case (e.g. cubic-vertex
  tadpoles where `V(τ) + V(−τ)` must be summed, not counted).

- **NoiseSourceType forces the slow path.** Gaussian/non-rational noise
  kernels can't be absorbed into the closed forms; those subsets are
  `_analytic_eligible = False` and route through SR + `fast_callable` +
  `scipy.nquad`. `ou_quartic_colored` at max_ell=2 is pathologically slow for
  this reason (per-MEMORY notes; use the white theory instead).

- **τ_v kernel cap is a quadrature band-aid.** Without `TAU_KERNEL_CAP=50`,
  `scipy.quad` over `(retard_L, +∞)` lets the tan-substitution compress a
  Gaussian kernel's peak near the boundary where adaptive sampling
  intermittently misses it, producing spurious spikes. Conv τ uses lower
  bound 0 (causal), Noise τ uses ±CAP (symmetric).

- **`POLYGON_BBOX_CAP=200` vs `POSET_PHYSICAL_MARGIN=50`.** The 200 box is too
  loose for the chain simplex: combined with retarded poles
  (`|Re β|>3`), `exp(β·L)` at `L=−200` overflows the 600 guard and bails to
  scipy. The poset path uses the tighter physical margin 50 (≈ several
  correlation times) for its lower-bound fallback.

- **`_resolve_1d_bounds` degenerate-empty sentinel.** It returns `(0.0, 0.0)`
  not `(inf, -inf)` on infeasibility, because `scipy.quad(f, +inf, -inf)`
  returns the **sign-flipped full-line integral**, silently poisoning any
  outer `nquad`.

- **Pure-external constraint must not set `pure_s1_found`.** In
  `_integrate_2d_polytope`, a constraint with both `a_int[0]≈0` and
  `a_int[1]≈0` is a pure-external inequality, not an `s_1` bound; treating it
  as one skips the `±200` cap fallback and oversamples an unbounded axis
  (~12% of the k=4 theory-vs-sim overshoot).

- **Stale docstrings.** `pipeline.py`'s module docstring and
  `compute_correction_td`'s say loop diagrams are *skipped*; the live code
  evaluates all loop orders through `integrate_diagram` (inline comment at
  `pipeline.py:291`). `subgraph.py` describes an algorithm that was never
  wired in (the time-domain integrator supersedes the planned
  frequency-domain loop reduction).

- **`USE_SIMPLIFY_FULL_IN_GT` is dead weight on the fast path.** The
  `simplify_full` pass in `build_G_t_matrix` only helps the SR/scipy fallback
  (NoiseSourceType); for rational theories the simplified smooth matrix is
  built and never read. Default False (−19% to −41% wall time on regression
  fixtures).

---

## Glossary

- **MSR-JD field theory** — Martin–Siggia–Rose–Janssen–De Dominicis: the path
  integral formulation of classical stochastic dynamics, with a *physical*
  field and a *response* (hatted) field per degree of freedom.
- **Retarded propagator `G_R`** — linear response of a physical field to a
  response-field source, carrying a Heaviside `Θ(Δt)` (effect after cause).
- **Pole / residue (`p_α`, `C_α`)** — the rational frequency-domain
  propagator's poles (here all `Im > 0`) and matrix residues; the
  time-domain propagator is `Σ_α C_α exp(i p_α t)`.
- **δ-coefficient / instantaneous part (`D_delta`)** — the `ω→∞` limit of
  `Ĝ`, the weight of a `δ(t)` term (an immediate response).
- **δ-edge subset / shot-noise δ** — expanding `∏(δ + smooth)` over which
  edges are taken in δ-form; a residual constraint among external times alone
  is a `δ(τ)` spike.
- **Polytope / chamber** — the convex region carved out by the causal
  `Δt > 0` constraints; the integration domain.
- **Causal poset (DAG)** — the m≥3 constraint set read as a partial order on
  integration times; integers `(u,v)` mean `s_v > s_u`.
- **Linear extension** — a topological sort of the poset = one chain-simplex
  region; the polytope is the disjoint union over all of them.
- **Chain simplex** — `L ≤ s_1 ≤ … ≤ s_N ≤ U`; its exponential integral has a
  `2^N`-term closed form.
- **Pole tuple** — a choice of one pole per smooth edge; the product is a
  single exponential whose polytope integral is closed-form.
- **Mode sum** — the per-edge representation `Σ_α C_α exp(λ_α Δt)`,
  `λ_α = i p_α`; stored as `EdgeModeSum`.
- **m (after delta)** — number of integration variables surviving δ
  elimination in a subset; selects the 1/2/≥3 analytic integrator.
- **External-Wick compensation** — divisor `|Aut_free|/|Aut_fixed|` removing
  overcounting in the same-field leaf-permutation sum.
- **Origin pinning** — fixing one external leaf at t=0 (translation
  invariance); the callable then computes a function of time differences.
- **NoiseSourceType / ConvVertexType** — vertices with non-local kernels
  (`κ(τ)` / `g(τ)`) needing extra τ integration variables.
- **`fast_callable`** — Sage's JIT for symbolic expressions (the slow-path
  integrand compiler).
- **`@njit`** — Numba's no-python JIT decorator (native machine code for the
  chain-simplex inner loop).
- **`mp.dps`** — mpmath's decimal-digit precision setting (50 in the rescue
  paths).
- **`quad` / `nquad`** — SciPy adaptive 1D / nested-multi-D quadrature.
- **Sutherland–Hodgman** — the polygon-clipping algorithm building the m=2
  integration region.
- **`Θ(0) = 0` convention** — the boundary `Δt = 0` is strictly excluded from
  retarded support, enforced by strict inequalities and the Heaviside filter.
- **Fork / spawn (start method)** — process-creation strategies;
  fork inherits memory (POSIX, used here), spawn pickles arguments (would
  fail on the unpicklable closures).

---

## Proposed manual subsections

1. **What Phase J computes** — the diagram integral
   `C_Γ = pref·∫∏G_R`, and its place at the end of the hybrid pipeline.
2. **The retarded propagator as a mode sum** — Fourier convention, poles in
   the upper half plane, the δ + smooth decomposition, the resp↔phys
   transpose, `build_G_t_matrix` / `EdgeModeSum`.
3. **From a diagram to a polytope integral** — vertex times, retardation
   Heavisides, the δ-subset expansion, δ-edge elimination, shot-noise spikes.
4. **Translation invariance and external-Wick sums** — origin pinning, the
   leaf↔position mapping enumeration, the `|Aut_free|/|Aut_fixed|` divisor,
   the per-permutation re-pinning fix.
5. **Pole-tuple expansion and the analytic plan** — `α_s`, `γ`, the
   τ-invariant `_build_modesum_plan` cache.
6. **The m=1 interval integrator** — closed form, unbounded endpoints.
7. **The m=2 polygon integrator** — Sutherland–Hodgman clipping,
   fan-triangulation, the unit-triangle `J(p,q)` and its Taylor fallbacks,
   the bilateral overflow guard.
8. **The m≥3 causal-poset integrator** — DAG extraction, linear extensions,
   the chain-simplex closed form, polynomial-prefactor and intermediate-upper
   variants, the numba fast path.
9. **The scipy fallback layer** — when analytic returns `None`, the
   Heaviside-filtered nquad path, bound-resolution subtleties (deferred
   constraints, pure-external traps, degenerate-empty sentinel).
10. **Non-local kernels** — NoiseSourceType and ConvVertexType, kernel
    substitution, single-exponential pseudo-edges, the τ_v cap.
11. **Batching and parallelism** — `total_C_batch` strategies, the unpicklable
    closures, the fork-in-notebook crash guard, Windows limitations.
12. **δ-spike discretization** — `eval_delta_contributions_on_tau_grid` /
    `_on_2d_grid`.
13. **Diagnostics and tuning knobs** — `_RUNTIME_COUNTERS`, `subset_diagnostics`,
    `QUAD_OPTS`, `TAU_KERNEL_CAP`, `USE_*` flags.
14. **Known precision caveats and open bugs** — the m≥3 close-pole
    overestimate, the cap-mismatch and grouped/per-diag discrepancy.

# Grouped Phase-J (fully-analytic mode-sum) — Technical Brief

**Slug:** `grouped-phasej`

**Primary source files:**

- `pipeline/_grouped_phase_j.py` — the per-prediagram *grouping wrapper* (`compute_correction_td_grouped`)
- `msrjd/integration/time_domain/grouped_integral.py` — the *grouped evaluator* (`integrate_grouped_diagram`) and two grouped closed-form helpers (`_evaluate_grouped_m0_modesum`, `_integrate_grouped_m1_modesum`)

**Supporting source files read for this brief:**

- `msrjd/integration/time_domain/final_integral.py` — the per-diagram analytic evaluators (`_integrate_1d_polytope_modesum`, `_integrate_2d_polygon_modesum`, `_integrate_nd_polytope_poset_modesum`, `_integrate_polytope`), `EdgeModeSum`, `_enumerate_pole_tuples`, `_loop_number_from_graph`, `_lookup_prop_indices`, and the flag/cap constants
- `msrjd/integration/time_domain/propagator_td.py` — `build_G_t_matrix`, `G_t_entry`, `G_t_delta_coeff`
- `msrjd/diagrams/symmetry.py` — `external_wick_compensation`, `vertex_role_signature`, `_automorphism_order`
- `pipeline/compute.py` — the dispatch site (`use_grouped_phase_j` flag), `msrjd/integration/time_domain/pipeline.py` — `compute_correction_td` (the per-diagram path this mirrors)

---

## Overview

### What "Phase J" is

In the Daedalus pipeline a connected correlation function (cumulant) of an MSR-JD
(Martin–Siggia–Rose–Janssen–De Dominicis) field theory is computed as a sum over
**Feynman diagrams**. The end-to-end flow has named phases. **Phase I** is the
*frequency-domain* propagator/vertex assembly; **Phase J** is the *time-domain
vertex-time integration* step — the part that takes each diagram's product of
retarded propagators and vertex coefficients and performs the integral over the
internal vertex times to produce a number `C(t_1, …, t_k)` (the contribution of
that diagram to the `k`-point cumulant as a function of the external times).

The per-diagram Phase-J entry point is `integrate_diagram` (aliased
`integrate_tree_diagram`) in `final_integral.py`; the loop that calls it once
per typed diagram lives in `compute_correction_td`
(`msrjd/integration/time_domain/pipeline.py`). "Grouped Phase-J" is an
*optimisation* of that loop.

### The problem grouped Phase-J solves

The enumerator emits many **typed diagrams** that share one parent
**prediagram**. A *prediagram* is the bare graph topology — its vertices, its
leaves (external legs), its edges, and which internal vertices are integration
variables. A *typed diagram* additionally fixes, per edge, which propagator
matrix entry `(ri, pi)` flows along that edge (which field-component pair), and
the per-vertex coefficients. Typed diagrams of one prediagram therefore have
**identical graph topology, identical leaves, identical internal vertices, and
identical integration variables** — they differ *only* in per-edge
`(ri, pi)` indices and per-vertex coefficients (docstring of
`grouped_integral.py`, lines 9–13).

The naive per-diagram loop repays the entire per-diagram setup cost — graph
walk, vertex-time symbol allocation, the `2^|E|` δ-edge subset bookkeeping, the
`fast_callable` JIT compile, the polytope/pole extraction — **once per typed
diagram**. The docstring records the motivating measurement
(`grouped_integral.py`, lines 18–21):

> "For (k=2, ell=1) multipop the user saw 7736 typed diagrams across 9
> prediagrams — ~860 diagrams per prediagram, all paying full setup cost."

The key mathematical observation is **linearity of integration**: because the
integration *domain* is identical across all typed diagrams of one prediagram,
you may sum the integrands first and integrate once
(`grouped_integral.py`, lines 33–41):

```
Σ_td ∫ ds ⋯ td-integrand_td   ==   ∫ ds ⋯ Σ_td td-integrand_td
```

This is *exact*, not an approximation. So the grouped path builds the
prediagram-level scaffolding **once**, sums the per-edge integrand factors over
all typed diagrams of the group, and runs a single integration per δ-edge
subset.

### Where it sits in the pipeline

```
   enumeration → type assignment → propagator (Phase I)
                                          │
                                          ▼
            compute_cumulants  (pipeline/compute.py)
                                          │
              records grouped by loop order ℓ, per ℓ:
                                          │
                  use_grouped_phase_j? ───┴───────────────┐
                       True                               False
                        │                                  │
                        ▼                                  ▼
   compute_correction_td_grouped          compute_correction_td  (per-diagram)
   (pipeline/_grouped_phase_j.py)         (…/time_domain/pipeline.py)
                        │                                  │
       group by (prediagram, ext_legs)                    │
                        │                                  │
                        ▼                                  │
   integrate_grouped_diagram  ── per group ──►  one `contribution` callable
   (…/time_domain/grouped_integral.py)                    │
                        │                                  │
       (returns total_C / total_C_batch, same shape) ◄─────┘
                        │
                        ▼
            τ-grid evaluation + saving (unchanged)
```

- **Feeds it:** `compute_cumulants` in `pipeline/compute.py`. At the dispatch
  site (`compute.py:757`), for each loop order `ell` it builds a list of typed
  diagrams (`r['typed_diagram']`) and their prefactors
  (`r['combined_prefactor']`) and, if `use_grouped_phase_j=True`, calls
  `compute_correction_td_grouped` instead of `compute_correction_td`
  (`compute.py:764`). It also passes `propagator_data` (the Phase-I output:
  poles, residue matrices, δ-coefficients), `external_fields`, `num_params`
  (numerical parameter substitutions), and `origin_leaf_idx`.
- **Consumes its output:** the same `compute_cumulants` code. The grouped wrapper
  returns the **same dict shape** as `compute_correction_td` — keys `total_C`
  (a callable `f(*t) -> complex`), `total_C_batch`, `ext_time_vars` — so the
  downstream τ-grid evaluation (`compute.py:790`) and saving code never know
  which path produced the numbers (`_grouped_phase_j.py` docstring, lines 17–23).

### "Fully-analytic mode-sum" (Stage 4a)

The original grouped prototype still used `scipy.nquad` (adaptive numerical
quadrature, precision floor ~`1e-8`) on the summed integrand. **Stage 4a**
(opt-in via the module flag `USE_GROUPED_ANALYTIC_MODESUM`, default `True`)
replaces that with a **merged-residue analytic mode-sum**: each propagator is
written as a finite sum of single-exponential pole terms, the products of these
exponentials are integrated *in closed form* over the causal polytope, and the
residue weights are pre-summed across the whole group. The precision floor drops
from scipy's ~`1e-8` to machine ε (~`1e-15`)
(`grouped_integral.py` docstring, lines 84–86).

---

## The math

This section builds up the integral that Phase-J evaluates, from the
field-theory side down to the closed forms the code implements.

### 1. The retarded propagator in the time domain

After Phase I, every propagator is known in the frequency domain as a rational
matrix `Ĝ(ω)`. Daedalus performs a partial-fraction / pole decomposition (in
`pipeline/_propagator.py`) and hands Phase J a `propagator_data` dict containing

- `pole_vals` — the poles `p_k` of `Ĝ(ω)` (each guaranteed `Im(p_k) > 0` by the
  causality filter),
- `C_mats` — one residue matrix `C_k` per pole, so `C_k[i,j]` is the residue of
  `Ĝ[i,j](ω)` at `p_k`,
- optionally `D_delta` — the `ω→∞` limits (the instantaneous δ-response weights).

`build_G_t_matrix` (`propagator_td.py:135`) assembles the **time-domain retarded
propagator** as (lines 140–141):

```
G_R[i,j](t) = delta_coeffs[i,j] · δ(t)
            + Θ(t) · ( Σ_k C_mats[k][i,j] · exp(I · p_k · t) )
```

The first term is the *instantaneous* δ-response; the second is the *smooth*
pole-residue sum. With the Fourier convention used pipeline-wide
(`build_G_t_matrix` docstring, lines 157–161), each summand `C_k · exp(I p_k t)`
*decays* for `t > 0` and *grows* for `t < 0`; the Heaviside `Θ(t)` is what makes
the retarded propagator well-defined.

Define the **mode** convention used throughout the analytic code:

```
λ_k = i · p_k         (a complex "rate"; Re λ_k < 0 for a decaying retarded mode)
C_α = C_mats[α][pi, ri]   (the residue at pole α for the edge's (pi, ri) entry)
```

so the smooth time-domain propagator on an edge is `Σ_α C_α · exp(λ_α · Δt)`
(`EdgeModeSum` docstring, lines 128–133). Here `Δt` is the time difference across
the edge: for an edge `u → v`, `Δt = t_v − t_u`.

### 2. One diagram's integral

Assign a time to each internal vertex (the integration variables `s_v`) and to
each leaf (the external time `t_cp`, with one leaf pinned to `t = 0` — the
"origin leaf"). The contribution of one typed diagram is

```
C_Γ(t_1,…,t_k) = (prefactor) · ∫ ∏_v ds_v   ∏_edges  G_R[pi_e, ri_e](Δt_e)
```

Each retarded factor `G_R(Δt_e) = δ-part + Θ(Δt_e) · smooth-part`. Expanding the
product of `|E|` retarded factors over the binary choice "δ-piece vs smooth-piece
on each edge" gives a sum over `2^|E|` **subsets**. In a given subset, the δ-edges
carry a `δ(Δt_e)` (which *collapses one integration variable* via the linear
equality `Δt_e = 0`), and the smooth edges carry `Σ_α C_α exp(λ_α Δt_e)` with a
retardation constraint `Δt_e > 0`.

After δ-elimination, suppose `m` integration variables `s_1,…,s_m` survive. The
smooth retardation constraints `Δt_e > 0` are **linear inequalities** in the
`s_i` and the free external times — i.e. they cut out a **convex polytope** `P`.
The subset's contribution is

```
I_subset = (prefactor) · ∫_P  ds_1 … ds_m   ∏_{e ∈ smooth} [ Σ_α C_α^{(e)} exp(λ_α Δt_e) ]
```

### 3. Pole expansion → analytic mode-sum

Expand the product of mode-sums into a sum over **pole tuples**
`α = (α_1, …, α_{n_smooth})` (one pole index per smooth edge). Each tuple
contributes a single exponential whose exponent is *linear* in the `s_i`:

```
∏_e [Σ_α C_α exp(λ_α Δt_e)]  =  Σ_α  ( ∏_e C_{α_e} )  · exp( Σ_e λ_{α_e} Δt_e )
```

Writing `Δt_e = c0_e + Σ_i a_int_{e,i}·s_i + Σ_j a_ext_{e,j}·t_free_j`, the
exponent becomes `γ_α + Σ_i α^{(i)}_s s_i` where

```
α^{(i)}_s = Σ_e λ_{α_e} · a_int_{e,i}        (slope along integration var i)
γ_α       = Σ_e λ_{α_e} · ( c0_e + Σ_j a_ext_{e,j} · t_free_j )   (constant part)
```

So `I_subset = Σ_α (∏_e C_{α_e}) exp(γ_α) ∫_P exp(Σ_i α^{(i)}_s s_i) ds`. The
remaining geometric integral `∫_P exp(linear) ds` has a **closed form** that
depends only on `m`:

- **m = 0** — no integration; just evaluate the exponentials after checking the
  constraints `Δt_e > 0`. (`_evaluate_grouped_m0_modesum`.)
- **m = 1** — a 1-D interval `[L, U]`; `∫_L^U exp(α_s s + γ) ds =
  (exp(α_s U + γ) − exp(α_s L + γ)) / α_s`, with the `α_s ≈ 0` limit handled
  separately and `±∞` endpoints zeroed when the integrand decays in that
  direction. (`_integrate_grouped_m1_modesum` and per-diagram
  `_integrate_1d_polytope_modesum`.)
- **m = 2** — a convex polygon; clip the half-planes, fan-triangulate, and
  integrate `exp(α x + β y)` over each triangle with the closed form
  `J(p,q) = ∫₀¹∫₀^{1-u} exp(pu + qw) dw du` (`_exp_over_unit_triangle`,
  `final_integral.py:356`). (`_integrate_2d_polygon_modesum`.)
- **m ≥ 3** — a polytope handled by a **causal-poset chain-simplex**
  decomposition: read off the partial order `s_u < s_v` from the retardation
  constraints, enumerate linear extensions (each a nested simplex
  `L ≤ s_{σ(0)} ≤ … ≤ s_{σ(m-1)} ≤ U`), and integrate `exp(Σ λ s)` over each
  nested simplex. (`_integrate_nd_polytope_poset_modesum`.)

### 4. The grouping math — merged residues

For *one* prediagram with `N` typed diagrams indexed by `j`, **all `N` share the
same pole spectrum `λ_α` and the same polytope** (same constraints, since the
constraints come from the shared `Δt_e` symbols). Only the residues `C_α` and the
prefactor `cp_j` differ between typed diagrams (via their `(ri, pi)` edge
indices). So by linearity the entire group's subset contribution is

```
I_subset^group  =  Σ_j cp_j · Σ_α ( ∏_{δ-edge} δ_coeff_{j,e} ) ( ∏_{smooth e} C^{(j)}_{α_e, e} )
                                  · exp(γ_α) · ∫_P exp(Σ α^{(i)}_s s_i) ds
```

Crucially, the `exp(γ_α)·∫_P …` factor does **not** depend on `j`. So the code
*pre-sums the residue weights over `j` inside the pole tuple*, defining the
**merged-residue tensor** (`grouped_integral.py` docstring, lines 71–75; the
construction is `grouped_integral.py:925–948`):

```
B_α  =  Σ_j  ( cp_j · ∏_{δ-edge} δ_coeff_{j,e} )  ·  ∏_{smooth e} C^{(j)}_{α_e, e}
```

Then `I_subset^group = Σ_α B_α exp(γ_α) ∫_P exp(…)`. This is *exactly* the same
closed form the per-diagram analytic evaluators already use, except the per-tuple
coefficient — which for the per-diagram path is the Cartesian product
`∏_e C_{α_e}` (`_enumerate_pole_tuples`, `final_integral.py:529`) — is **replaced
by the summed tensor `B_α`**. The "underlying analytic integration is identical"
(`grouped_integral.py` docstring, lines 79–86); only the residue accumulation
moved *inside* the tuple sum.

### 5. The grouped-vs-perdiag equivalence (and the caveat)

Mathematically, **summing the residues before integrating equals summing the
integrals after** — this is what makes the grouped path a drop-in. The codebase
treats agreement to floating-point round-off as a **correctness contract**
(`grouped_integral.py` docstring, lines 94–98):

> "If anything other than the simple Hawkes / multipop tree+1-loop case is
> exercised, this prototype should be carefully cross-checked against
> `integrate_diagram` (sum N per-diagram results vs 1 grouped result — they MUST
> match to floating-point round-off)."

The one subtlety — and the subject of an *open precision investigation* — is at
**m ≥ 3**. The grouped path sums `B_α` *before* the chain-simplex integration, so
intra-group cancellations happen at small magnitude; the per-diagram path sums
*after* integration. By linearity these are equal in exact arithmetic, but in
float64 they can disagree when individual pole-tuple terms are huge
(close-paired poles → `O(1/Δp)` residues, products of 5 such → `O(10^8)`,
catastrophic cancellation down to the `O(1)` integral). This is documented in
`docs/m_ge3_precision_bug_audit.md` and recorded in the `USE_POSET_CAP_MATCH_SCIPY`
and `USE_POSET_MPMATH_ACCUMULATION` comment blocks (`final_integral.py:1663–1723`)
as an **OPEN BUG** affecting spike-reset `k≥2 ell≥1`.

### 6. The external-Wick compensation (vertex_role_signature history)

The integrators enumerate every field-respecting assignment of canonical
external positions to diagram leaves and **sum the integrand over all of them**
(the `_all_mappings` permutation sum). Some of those permutations are
*redundant* — they map the diagram to itself via a graph automorphism — so the
sum over-counts each distinct pinned-external diagram. The exact divisor is the
**orbit–stabilizer index**

```
comp = |Aut(Γ, leaves free)| / |Aut(Γ, leaves fixed)|
```

(`external_wick_compensation`, `symmetry.py:343–400`). The grouped path divides
its final summed contribution by `comp` (`grouped_integral.py:1074`,
`total / _comp`).

The `vertex_role_signature` (`symmetry.py:247`) is a *depth-1 graph invariant*
that captures a vertex's "role" (its type/coefficient/bigrade, its external-leaf
attachment pattern, and its internal-edge incidence pattern). **History:** the old
compensation heuristic approximated the index as `∏ N_{sig,field}!` over leaves
grouped by the `vertex_role_signature` of their attachment vertex. That
over-divides whenever two attachment vertices *share a role signature* but are
*not* related by any automorphism — e.g. the `k=4` OU+εx³ 1-loop cascades, where
three noise vertices all read "noise feeding a cubic" at depth 1 yet hang off
structurally distinct cubics. The role-signature heuristic broke every `k≥3`
1-loop cumulant while leaving `k=2` exact (at `k=2` the heuristic happens to
coincide with the true index). It was replaced by the **exact Sage automorphism
ratio** (`symmetry.py:368–381`). The role-signature function survives in the
codebase (still referenced in `final_integral.py:2573`) but the grouped path now
uses the exact `external_wick_compensation` for its divisor.

---

## External tools used

This subsystem leans almost entirely on **SageMath**. There is no direct use of
nauty, sympy, numba, or networkx *in the two primary files* — but SageMath
internally embeds several of those, and the per-diagram analytic evaluators reach
into scipy. Below is each library, explained from scratch, with the actual import
lines and calls.

### SageMath (`sage`)

**What it is.** SageMath is a large open-source mathematics system built on top
of Python. It bundles dozens of specialised libraries (GiNaC/Pynac for symbolic
algebra, Maxima for simplification, FLINT/NTL/PARI for number theory, the `bliss`
/ `nauty` graph-automorphism engines, NumPy/SciPy for numerics) behind a single
Python API. When this code writes `from sage.all import …` it is importing
Sage's "everything" namespace. The crucial Sage objects this subsystem touches:

- **`SR` — the Symbolic Ring.** `SR` is Sage's ring of symbolic expressions
  (powered by Pynac, Sage's fork of the GiNaC C++ CAS). `SR.var('x')` makes a
  symbolic variable; `SR(3)` wraps a number; `SR(expr)` coerces; arithmetic
  builds expression trees you can `.subs()`, `.expand()`, `.simplify_full()`,
  `.coefficient()`, `.variables()`, `.diff()`. Think of it as Sage's equivalent
  of SymPy's `Symbol`/`Expr`, but a different engine.
  - Import: `from sage.all import SR, fast_callable, CDF, solve as sage_solve`
    (`grouped_integral.py:101`); also `from sage.all import SR`
    (`_grouped_phase_j.py:34`).
  - Used to allocate the internal time symbol `t_sym = SR.var('_t_td_')`
    (`grouped_integral.py:378`), the vertex-time symbols
    `SR.var(f's_v{v}_td_', …)` (`:467`), the per-noise-source τ symbols (`:497`),
    and the default external time variables in the wrapper
    (`SR.var(f't_{j+1}', …)`, `_grouped_phase_j.py:74`).
  - Used to build the symbolic per-edge `Δt` (`dt = SR(t_v - t_u)`, `:549`), the
    summed subset integrand `subset_factor_group = SR(0)` accumulated via `+`
    and `*` (`:760–770`), and to coerce numbers/symbols throughout.
  - **Method calls**: `.subs(substitutions)` (δ-elimination back-substitution,
    `:769`), `.expand()` (`:773`), `.is_trivial_zero()` (`:779`),
    `.variables()` (`:798`), `.coefficient(v)` (linear-coefficient extraction,
    `:836`), `.simplify_full()` (kernel substitution cleanup, `:588`),
    `.is_zero()` (shot-noise detection, `:745`).

- **`fast_callable` — a JIT-style expression compiler.** A symbolic `SR`
  expression evaluated naively (by `.subs()` then float-cast) is *slow* because
  it re-walks the GiNaC tree every call. `fast_callable(expr, vars=[…],
  domain=CDF)` compiles `expr` into a fast byte-code/closure that takes the
  listed variables as positional arguments and returns a value in the chosen
  domain. It is Sage's analogue of SymPy's `lambdify`.
  - Import: as above (`grouped_integral.py:101`).
  - Call: `integrand_fc = fast_callable(subset_factor_group, vars=fc_vars_sub,
    domain=CDF)` (`grouped_integral.py:815–818`). This is built **once per
    subset over the SUMMED integrand** — the central win: one compile instead of
    `N` compiles. The compiled callable is used only on the **scipy.nquad
    fallback** path (when the analytic merged-residue path returns `None`); the
    analytic path never touches it.

- **`CDF` — the Complex Double Field.** `CDF` is Sage's ring of complex numbers
  backed by hardware `double` precision (a thin wrapper over C `double complex`).
  `complex(CDF(SR(expr)))` is the idiom for "evaluate this symbolic expression to
  a fast machine-precision Python `complex`." `CDF` is chosen as the
  `fast_callable` domain (so the compiled integrand returns complex doubles) and
  as the cast target when converting poles/residues to numeric `complex`.
  - Import: top-level (`:101`) and re-imported locally as `_CDF` inside the
    analytic block (`from sage.all import CDF as _CDF`, `:611`, `:902`).
  - Calls: `complex(_CDF(SR(p))) * 1j` for poles (`:612`),
    `complex(_CDF(SR(C_mats[k][pi, ri])))` for residues (`:626`),
    `complex(_CDF(SR(cp)))` and `complex(_CDF(SR(ei[di]['delta_coeff'])))` for
    the merged prefactor (`:908`, `:911`).

- **`solve` (imported as `sage_solve`) — symbolic equation solver.** Sage's
  `solve(eqn, var, solution_dict=True)` solves an equation symbolically (backed
  by Maxima) and returns a list of solution dicts. The grouped path uses it to
  **eliminate** an integration variable on each δ-edge: a δ-edge imposes
  `Δt_e = 0`, a *linear* equation; solving it for one surviving integration
  variable removes that variable from the integral.
  - Import: `solve as sage_solve` (`:101`).
  - Call: `sol = sage_solve(eq_expr == 0, int_var_to_eliminate,
    solution_dict=True)` (`grouped_integral.py:719–722`); `sol[0][var]` is the
    back-substitution expression stored in `substitutions`.

#### nauty / bliss (via Sage) — graph automorphisms

The compensation factor needs the *order of the automorphism group* of a
coloured graph. Sage computes this with its built-in graph-isomorphism engine
(historically `bliss`/`nauty`-class algorithms). The grouped path reaches this
**indirectly** through `external_wick_compensation` →
`_automorphism_order(typed_diagram, fix_external=…)`
(`symmetry.py:328–340`), which builds a coloured incidence digraph and calls
`D.automorphism_group(partition=partition).order()`. The import in the grouped
file is lazy: `from msrjd.diagrams.symmetry import external_wick_compensation`
(`grouped_integral.py:434`). The novice takeaway: *nauty/bliss answer the
question "how many ways can I relabel this graph's vertices and leave it
unchanged, respecting the colours?", and that count is the divisor that prevents
the external-leaf permutation sum from over-counting.*

### scipy (via the per-diagram evaluators) — adaptive numerical quadrature

scipy is the standard Python scientific-computing library; `scipy.integrate.nquad`
performs **adaptive multidimensional numerical quadrature** (it samples the
integrand on an adaptively-refined grid until a relative tolerance is met,
default `epsrel ≈ 1e-8`). The grouped path **does not import scipy directly**,
but its fallback routes through `_integrate_polytope`
(`final_integral.py:4372`), which dispatches to `_integrate_1d_polytope` /
`_integrate_2d_polytope` / `_integrate_nd_polytope` (scipy-backed). The grouped
path only hits scipy when the analytic merged-residue path returns `None` (e.g.
unbounded-divergent endpoint, non-rational kernel). This is the slower, lower-
precision path; Stage 4a exists precisely to avoid it.

### Python standard library

- **`collections.defaultdict`** — `from collections import defaultdict`
  (`_grouped_phase_j.py:32`). A `dict` subclass that auto-creates a default value
  for missing keys. Used to bucket typed diagrams into groups:
  `groups = defaultdict(lambda: {'tds': [], 'cps': []})`
  (`_grouped_phase_j.py:96`), so `groups[key]['tds'].append(td)` works without a
  membership check.
- **`itertools`** — `import itertools as _itertools`
  (`grouped_integral.py:326`). Used for `_itertools.permutations(lfs)`
  (`:406`) when enumerating the external-leaf Wick mappings.
- **`cmath` / `math`** — local imports inside the hot helpers. `cmath.exp` is the
  complex exponential used in every closed-form term (`grouped_integral.py:149`,
  `:205–206`; `_integrate_1d_polytope_modesum` etc.). `math.inf` / `math.isinf`
  track unbounded interval sides exactly (`:229–230`).
- **`NoiseSourceType`** — `from msrjd.core.vertices import NoiseSourceType`
  (`grouped_integral.py:120`). Used in `isinstance(vtype, NoiseSourceType)`
  checks (`:487`) to detect non-local cumulant (noise-source) vertices that
  introduce extra τ integration variables and disable the analytic path.

### What is NOT used here

- **numba** — not imported in either primary file. (numba is used elsewhere in
  the codebase for the spatial path; the analytic mode-sum is pure Python-level
  complex arithmetic, fast enough without it.)
- **networkx** — not used; all graph work goes through Sage `Graph` objects
  carried on the prediagram (`td.prediagram[0]`, a Sage graph `D`).
- **sympy** — not used; Daedalus's CAS is Sage's `SR`, not SymPy.

---

## Components

Listed file-by-file, top to bottom. Signatures are quoted verbatim; `file:line`
points to the `def`.

### `pipeline/_grouped_phase_j.py`

#### Module-level constant `_SUBSET_SKIP_STATUSES` — `_grouped_phase_j.py:49`

```python
_SUBSET_SKIP_STATUSES = frozenset({
    'shotnoise_skipped_in_prototype',
    'dt_sym_mismatch_skipped',
})
```

The two **subset-level statuses** that mean "the grouped evaluator silently
*dropped* this subset rather than integrating it." A group can return overall
`status='ok'` yet have a subset diagnostic with one of these — which makes the
group total numerically *wrong* (a subset's integrand is missing). The wrapper
inspects every subset diagnostic against this frozenset to decide whether to fall
back to the per-diagram path. The comment (`:42–48`) is explicit that this is
"the production bug we're fixing," and lists them explicitly so a *new* skip
status added later doesn't silently regress.

#### `compute_correction_td_grouped(...)` — `_grouped_phase_j.py:55`

```python
def compute_correction_td_grouped(
    typed_diagrams,
    prefactors,
    propagator_data,
    k,
    num_params=None,
    ext_time_vars=None,
    origin_leaf_idx=0,
    external_fields=None,
):
```

- **Takes:** a flat list of `typed_diagrams` (all at one loop order `ell`), a
  parallel list of `prefactors` (the `combined_prefactor` per diagram),
  `propagator_data` (Phase-I poles/residues/δ-coeffs), `k` (number of external
  legs), optional `num_params` (numeric substitutions), `ext_time_vars`
  (external time symbols — auto-created if `None`), `origin_leaf_idx` (which leaf
  is pinned to `t=0`, default 0), and `external_fields` (canonical leaf→field
  list).
- **Returns:** a dict with the same shape as `compute_correction_td`:
  `total_C` (callable `f(*ext_time_values) -> complex`), `total_C_batch`
  (callable over a list of τ-points), `eval_per_diagram_batch` (`None` — not
  supported in the prototype), `delta_contributions` (`[]` — not supported),
  plus grouped bookkeeping: `groups` (per-group metadata), `skipped_kernel_ids`,
  `fallback_to_perdiag`, and `ext_time_vars`.
- **Step by step:**
  1. **Default external times** (`:72–76`): if `ext_time_vars is None`, build
     `[t_1, …, t_k]` as `SR` variables with LaTeX names.
  2. **Group by (prediagram identity, external-legs signature)** (`:87–100`).
     The grouping key is `(id(td.prediagram), _ext_legs_key(td))`. The nested
     `_ext_legs_key(td)` (`:87`) returns a hashable sorted tuple of
     `(leaf, (field_base, pop_idx))` entries from `td.external_legs`. Subdividing
     by external-legs is **required**: two typed diagrams that differ only in
     which leaf carries which external field have different `_all_mappings` /
     `_compensation` inside the evaluator and *cannot share a sum* (`:82–86`).
     The per-diagram path handles each independently; the grouped path subdivides
     at the finer key.
  3. **Per-group dispatch loop** (`:125–228`), three tiers:
     - **Tier 1 — grouped evaluator** (`:134–164`): call
       `integrate_grouped_diagram`. If `status == 'ok'` *and* no subset
       diagnostic is in `_SUBSET_SKIP_STATUSES` (the `has_skipped_subset` check
       at `:147–153`), accept the group: append `result['contribution']` to
       `group_callables`, record metadata with `'handled_by':
       'grouped_evaluator'`.
     - **Tier 2 — per-diagram fallback on silent subset skip** (`:165–191`): if
       `status == 'ok'` but a subset was silently dropped, re-compute the
       *entire group* via the per-diagram path (`_perdiag_fallback`), append that
       callable, and record `'handled_by': 'perdiag_fallback'` with the skip
       reasons. This trades the grouped speedup for correctness on that group
       (the same numbers `use_grouped_phase_j=False` would give).
     - **Tier 3 — per-diagram fallback on group-level failure** (`:192–228`): if
       the grouped evaluator failed entirely (e.g. early validation), also fall
       back to per-diagram. If *that* throws too, record a *loud* `skipped` entry
       (`:213–228`) rather than silently dropping the whole group.
  4. **Build `total_C` / `total_C_batch`** (`:231–241`): `total_C(*t)` sums every
     group callable; `total_C_batch(tau_points, …)` serially evaluates `total_C`
     at each τ-point (parallelism deferred — `:237–241`).
  5. **Return the result dict** (`:243–252`).

#### Nested `_ext_legs_key(td)` — `_grouped_phase_j.py:87`

Returns `tuple(sorted((lf, fld) for lf, fld in td.external_legs.items()))` — the
hashable external-legs signature used as the second component of the group key.
The comment notes the `leaf id()` defensiveness is moot in practice (leaves are
ints).

#### Nested `_perdiag_fallback(tds, cps)` — `_grouped_phase_j.py:107`

Calls `compute_correction_td(typed_diagrams=tds, prefactors=cps, …)` (the
per-diagram path) and returns `(pd_result['total_C'], len(tds))`. Isolates the
fallback so the returned callable has the same
`(*ext_time_values) -> complex` signature as the grouped `contribution`.

#### Nested `total_C(*ext_time_values)` — `_grouped_phase_j.py:231`

Sums `complex(fn(*ext_time_values))` over all `group_callables`.

#### Nested `total_C_batch(tau_points, parallel=False, n_workers=None)` — `_grouped_phase_j.py:237`

Returns `[complex(total_C(*pt)) for pt in tau_points]`. Matches
`compute_correction_td`'s API signature but ignores `parallel`/`n_workers` in the
prototype.

---

### `msrjd/integration/time_domain/grouped_integral.py`

#### Module-level constant `USE_GROUPED_ANALYTIC_MODESUM` — `grouped_integral.py:130`

```python
USE_GROUPED_ANALYTIC_MODESUM = True
```

Master switch for the analytic merged-residue path inside
`integrate_grouped_diagram`. When `True` (and the underlying
`USE_POLYGON_M2_INTEGRATOR` / `USE_POSET_INTEGRATOR` flags in `final_integral`
are also on), each subset gets a `pole_tuples` iterator with merged residues
`B_α`, giving machine precision. When `False`, every subset falls through to
scipy.nquad (~`1e-8` rel).

#### Module-level constant `_GROUPED_EXP_REAL_LIMIT` — `grouped_integral.py:135`

```python
_GROUPED_EXP_REAL_LIMIT = 600.0
```

Overflow guard threshold. `cmath.exp(z)` raises `OverflowError` when `z.real >
709` on IEEE doubles; clip at 600 for headroom. Mirrors `EXP_REAL_LIMIT` in
`final_integral.py`.

#### `_evaluate_grouped_m0_modesum(pole_tuples, subset_constraint_data, free_ext_vals)` — `grouped_integral.py:138`

```python
def _evaluate_grouped_m0_modesum(pole_tuples, subset_constraint_data, free_ext_vals):
```

- **What it is:** the analytic closed form for `m = 0` (no integration variables
  survived δ-elimination).
- **Takes:** `pole_tuples` — iterable of `(B_alpha, lambdas)` merged-residue
  pairs; `subset_constraint_data` — list of `(a_int, a_ext, c0)` linear forms for
  each smooth-edge `Δt`; `free_ext_vals` — the numeric free external-time values.
- **Returns:** `Σ_α B_α · exp(Σ_e λ_α_e · Δt_e)` as a `complex`; `0` if any
  constraint `Δt_e > 0` is violated (Θ(0) = 0 convention — `c_eff <= 0` → outside);
  `None` on overflow (caller falls back to scipy).
- **Steps:**
  1. **Constraint check** (`:150–156`): for each `(a_int, a_ext, c0)` compute
     `c_eff = c0 + Σ_i a_ext[i]·free_ext_vals[i]`; if `c_eff <= 0` return
     `0.0+0.0j` (the half-space is strictly open at `Δt = 0`).
  2. **Per-edge Δt** (`:158–164`): for `m=0`, `a_int` is empty, so
     `Δt_e = c_eff` exactly.
  3. **Pole-tuple sum** (`:165–176`): for each `(B_alpha, lambdas)`, accumulate
     `γ = Σ_e λ_e · Δt_e`; overflow-guard `|γ.real| > 600` → `None`; else add
     `B_alpha · cmath.exp(γ)`.

#### `_integrate_grouped_m1_modesum(pole_tuples, subset_constraint_data, free_ext_vals, bbox_cap)` — `grouped_integral.py:179`

```python
def _integrate_grouped_m1_modesum(pole_tuples, subset_constraint_data, free_ext_vals, bbox_cap):
```

- **What it is:** the analytic closed form for `m = 1` (one surviving integration
  variable `s`).
- **Takes:** `pole_tuples`, `subset_constraint_data`, `free_ext_vals` as above,
  plus `bbox_cap` (a finite bounding-box cap — passed but only used to *not* clip;
  see below).
- **Returns:** `∫_L^U Σ_α B_α exp(α_s s + γ_α) ds` as a `complex`; `0` if the
  interval is empty/infeasible; `None` on overflow or an unbounded interval with
  a non-decaying integrand.
- **Steps:**
  1. **Resolve the feasible interval `[L, U]`** (`:208–228`). Initialise
     `L = -inf, U = +inf`. For each constraint `a·s + c_eff > 0`: if `|a| <
     1e-15` it's `s`-independent — `c_eff <= 0` ⇒ infeasible (return 0), else
     skip. Otherwise `bound = -c_eff/a`; `a > 0` tightens the lower bound (`L =
     max(L, bound)`), `a < 0` tightens the upper bound (`U = min(U, bound)`). If
     `L >= U`, empty → return 0. Track `L_inf`, `U_inf` flags.
  2. **Pole-tuple sum** (`:233–279`). For each `(B_alpha, lambdas)`, compute
     `α_s = Σ_e λ_e·a_int_e[0]` and `γ = Σ_e λ_e·c_ext_e`. Overflow-guard `γ`.
     Then:
     - `|α_s| < 1e-15` (constant in `s`): `(U−L)·exp(γ)`, but if either side is
       infinite the integral diverges → `None`.
     - else `(exp(α_s U + γ) − exp(α_s L + γ)) / α_s`, with each boundary term
       **zeroed** when the corresponding side is `±∞` *and* the integrand decays
       there (`U_inf` with `α_s.real < 0` → `term_U = 0`; `L_inf` with
       `α_s.real > 0` → `term_L = 0`). If it would *diverge* (`U_inf,
       α_s.real >= 0` or `L_inf, α_s.real <= 0`) → `None` (caller falls back to
       scipy.nquad, which handles the unbounded endpoint via adaptive
       substitution). The docstring (`:196–202`) explains *why ±∞ rather than
       ±bbox_cap*: clipping at `±bbox_cap` leaves a residual `exp(α_s·bbox_cap) ≈
       1e-9` boundary term for typical Hawkes timescales (measured on
       `quad_exp_k2_ell0` before the fix).

#### `integrate_grouped_diagram(...)` — `grouped_integral.py:283`

```python
def integrate_grouped_diagram(
    typed_diagrams,
    combined_prefactors,
    propagator_data,
    ext_time_vars,
    num_params=None,
    origin_leaf_idx=0,
    external_fields=None,
):
```

The heart of the subsystem. Sums-then-integrates over one prediagram group.

- **Takes:** `typed_diagrams` (length-`N`, **all sharing one prediagram** —
  asserted), `combined_prefactors` (length-`N`, per-td full prefactors),
  `propagator_data` (must have `pole_vals`, `C_mats`, optionally `D_delta`),
  `ext_time_vars`, `num_params`, `origin_leaf_idx`, `external_fields`.
- **Returns:** a dict with `status` (`'ok'`/`'failed'`/…), `contribution`
  (`f(*ext_time_values) -> complex`, the summed group contribution),
  `n_diagrams`, `integration_vars`, `loop_number`, `subset_diagnostics`, and a
  `reason` on failure.
- **Steps (with line refs):**
  1. **Validation** (`:328–358`): empty list → fail; length mismatch → fail;
     **all td's must share the same prediagram object by identity**
     (`all(td.prediagram is pd0 …)`, `:350`) → fail on mismatch.
  2. **Shared scaffolding** (`:360–376`): `loop_number =
     _loop_number_from_graph(td0)`; extract the Sage graph `D = td0.prediagram[0]`
     and leaves `td0.prediagram[2]`; check `len(ext_time_vars) == len(leaves)`.
  3. **Build the smooth/δ propagator matrix once** (`:378–379`):
     `t_sym = SR.var('_t_td_')`, `G_t_obj = build_G_t_matrix(propagator_data,
     t_sym, num_params)`. Shared across the whole group.
  4. **Wick contraction enumeration `_all_mappings` + compensation `_compensation`**
     (`:381–452`). If `external_fields` is provided and matches the leaf count,
     enumerate every field-respecting bijection of canonical external positions
     to leaves: bucket external positions and leaves by field (`_cp_by_field`,
     `_leaves_by_field`), take `itertools.permutations` within each field
     (`:406`), and build the Cartesian product of per-field permutations into
     `_all_mappings`. If any field's counts mismatch, fall back to the identity
     mapping and `_compensation = 1` (`_mapping_fallback`, `:430–432`).
     Otherwise compute the **exact** compensation via
     `external_wick_compensation` on every group member and **assert all members
     agree** (`:434–449`) — a mismatch returns `status='failed'` because a shared
     divisor would mis-weight the group. (See "The math §6".)
  5. **Vertex-time map** (`:454–469`): for the first mapping, assign each leaf its
     external time (pinning the origin leaf to `SR(0)`); allocate a fresh
     `s_v…` symbol per internal vertex and collect them in `integration_vars`.
  6. **Per-td noise-source per-leg time maps** (`:471–520`): for each td, scan its
     `vertex_assignments` for `NoiseSourceType` vertices with `cumulant_specs`;
     each introduces a per-td `τ` symbol and a per-leg time offset
     `anchor_time − τ`. The union of all τ symbols becomes
     `integration_vars_grouped` (`:514–520`). For plain (no-noise-source) groups
     this loop is a no-op and the analytic path stays enabled.
  7. **Per-td edge info + prefactor** (`:522–591`) — the *only* genuinely per-td
     work. For each td and each prediagram edge `(u, v, lbl)`: look up `(ri, pi)`
     via `_lookup_prop_indices`; resolve the head/tail times (accounting for
     noise-source leg offsets); build `dt = SR(t_v − t_u)`,
     `delta_coeff = G_t_delta_coeff(...)`, `smooth_factor = G_t_entry(...,
     include_heaviside=False)`; store in `edge_info`. Then coerce the per-td
     `combined_prefactor` to `SR`, substitute `num_params`, and (if noise-source
     kernels are present) substitute the cumulant-kernel symbols and
     `simplify_full`.
  8. **Analytic mode-sum precomputation** (`:593–637`). Enable the analytic path
     only if `USE_GROUPED_ANALYTIC_MODESUM` *and* `pole_vals`/`C_mats` present
     *and* **no** td has a non-empty `vertex_leg_time` (noise-source kernels make
     the integrand non-rational and disable the closed form, `:602–607`). If
     enabled, precompute `lambdas_per_pole = (complex(_CDF(SR(p)))·1j for p)`
     (`:612`) and the per-td/per-edge/per-pole residue tensor
     `residues_per_td_edge` (`:619–628`). Built **once**, reused across all
     subsets.
  9. **External-time bookkeeping** (`:639–648`): the free external indices
     `free_ext_idx` (all leaves except the pinned origin), their symbols
     `free_ext_syms`, `n_edges`, and `n_subsets_total = 2**n_edges`.
  10. **The `2^|E|` subset loop** (`:654–1033`) — the core. For each
      `subset_bits`:
      - Split edges into `delta_edges` (bit set) and `smooth_edges` (bit clear)
        (`:655–658`).
      - **Per-td filter** (`:663–671`): a td with a near-zero δ-coefficient on any
        δ-edge contributes nothing → drop it; keep `contributing_td`. If none
        contribute, `continue`.
      - **δ-sym consistency check** (`:673–700`): all contributing td's must agree
        on the symbolic `dt_sym` of each δ-edge (they differ only for
        noise-source vertices). On mismatch, record a diagnostic with status
        `dt_sym_mismatch_skipped` and `continue` (this is one of the
        `_SUBSET_SKIP_STATUSES` the wrapper watches).
      - **δ-elimination** (`:702–734`): for each δ-edge, try to solve its
        `Δt_e = 0` for a surviving integration variable via `sage_solve`,
        recording the back-substitution in `substitutions` and removing the
        variable from `remaining_int_vars`. If a δ-edge's equation involves no
        integration variable, it becomes an *external-time equality*
        (shot-noise). Infeasible solve → mark `subset_infeasible` and skip.
      - **Shot-noise skip** (`:736–757`): any non-trivially-zero external-time
        equality means a residual δ-spike; the prototype records status
        `shotnoise_skipped_in_prototype` and `continue` (the second
        `_SUBSET_SKIP_STATUSES` member).
      - **Build the SUMMED subset integrand** (`:759–783`): for each contributing
        td, `term = cp · ∏_{δ-edge} delta_coeff · ∏_{smooth} smooth_factor`,
        back-substituted, accumulated into `subset_factor_group`; `.expand()`;
        skip if trivially zero.
      - **Smooth-edge retardation constraints** (`:785–791`): one per smooth edge,
        the back-substituted `dt_sym` (shared across contributing td's).
      - **`fast_callable` compile** (`:793–828`): compute `m_sub =
        len(remaining_int_vars)`, the variable list, a sanity check that the
        summed integrand has no unexpected free symbols (else `status='failed'`),
        and compile `integrand_fc = fast_callable(subset_factor_group, …,
        domain=CDF)`. (Used only on the scipy fallback.)
      - **Linear-coefficient extraction** (`:830–877`): for each smooth-edge
        constraint, extract `a_int` (coeffs on `remaining_int_vars`), `a_ext`
        (coeffs on `free_ext_syms`), `c0` (constant), building
        `subset_constraint_data`. Non-linear constraint → `status='failed'`.
        Appends finite τ-caps for any noise-source τ in `remaining_int_vars`.
      - **Merged-residue pole-tuple construction** (`:879–964`). If the analytic
        path is enabled, `m_sub` is in the supported set, and there are no extra
        τ vars: build `merged_pf_per_j = cp_j · ∏_{δ-edge} delta_coeff_j`
        (`:904–913`), the per-td smooth-edge residue arrays (`:915–921`), and then
        enumerate the multi-index `α` over per-edge poles, summing
        `B_alpha = Σ_j (merged_pf_per_j · ∏_e residue_{j,e,α_e})` and pairing it
        with `lambdas = (λ_{α_e})_e` into `grouped_pole_tuples`
        (`:925–948`). Also build `grouped_dummy_modes` — a tuple of
        `EdgeModeSum` objects whose only role is to pass the `n_smooth` length
        check inside the per-diagram evaluators (`:949–961`); their modes come
        from td[0]'s residues but are *never read* on the analytic path (the
        `pole_tuples` override carries the real weights).
      - **Build the subset contribution closure** (`:966–1025`) via
        `_make_subset_contrib`. Each closure tries the analytic path first
        (dispatching by `m_val` to `_evaluate_grouped_m0_modesum`,
        `_integrate_grouped_m1_modesum`, `_integrate_2d_polygon_modesum`, or
        `_integrate_nd_polytope_poset_modesum` — the last two with
        `pole_tuples=grouped_pole_tuples`), and on `None` falls back to the
        scipy `_integrate_polytope` on the compiled `integrand_fc`.
      - Append the closure to `subset_contributions` and a diagnostic record.
  11. **Final contribution callable (Wick-permutation sum)** (`:1035–1074`). From
      `_all_mappings`, build the permutation list `_perms` mapping canonical
      external positions through each mapping. The returned `contribution(*ext)`
      loops over permutations; for each it (a) permutes the external times, (b)
      time-shifts so the (permuted) origin leaf returns to 0 (the integrand
      computes the diagram as a function of *time differences*; this symmetrises
      asymmetric integrands and is a no-op for symmetric ones), (c) extracts the
      free-time values, and (d) sums every subset closure. Finally **divides by
      `_compensation`** (`:1074`). Returns the result dict (`:1076–1083`).

#### Nested `_make_subset_contrib(fc, cdata, m_val, pole_tuples=None, dummy_modes=None)` — `grouped_integral.py:967`

Closure factory. Returns `_contrib(free_vals)` which: tries the matching analytic
evaluator for `m_val ∈ {0, 1, 2, ≥3}` (returning early if it yields a non-`None`
value), otherwise resolves the constraints with the numeric `free_vals` and calls
`_integrate_polytope(fc, resolved, free_vals, m_val)` (scipy fallback). Note the
m=2/m≥3 calls pass `prefactor_complex=1.0+0.0j` because the prefactor is already
folded into `B_α` via `merged_pf_per_j`.

#### Nested `contribution(*ext_time_values)` — `grouped_integral.py:1049`

The returned group callable described in step 11 above.

---

### Supporting components (read for context, in `final_integral.py` / `propagator_td.py` / `symmetry.py`)

#### `EdgeModeSum` (dataclass) — `final_integral.py:117`

`@dataclass(frozen=True)`. Numerical mode-sum of one propagator edge. Fields:
`ri, pi` (response-column / physical-row indices), `delta_coeff` (complex δ(Δt)
weight), `modes` (tuple of `(C_α, λ_α)` pole-residue pairs), and the per-subset
`dt_c0`, `dt_int_pairs`, `dt_ext_pairs` sparse-linear-form fields (filled at the
subset stage). In the grouped path, the `grouped_dummy_modes` are `EdgeModeSum`
instances whose `modes` exist only to satisfy `n_smooth` length checks.

#### `_enumerate_pole_tuples(edge_mode_sums)` — `final_integral.py:529`

The **per-diagram** Cartesian-product iterator: yields `(C_product, lambdas)` per
pole tuple where `C_product = ∏_e C_{α_e}`. The grouped path's whole purpose is to
*replace* this with the merged `B_α` sum — both feed the same analytic evaluators.

#### `_integrate_1d_polytope_modesum(...)` — `final_integral.py:1976`

The per-diagram m=1 analytic interval integrator (focus function per the task
brief). Computes `∫_L^U Π_e [Σ_α C_α exp(λ_α Δt_e)]·pref ds` by the same pole
expansion as `_integrate_grouped_m1_modesum`, but accepts an optional
`pole_tuples` override (the grouped merged residues) and an optional `plan`
(τ-invariant per-tuple `α_s`/`γ` decomposition cached at subset setup,
`:2052–2105`). The grouped path's `_integrate_grouped_m1_modesum`
(`grouped_integral.py:179`) is a *stand-alone twin* of this function — same
interval resolution, same closed form — rather than a call into it; the m=2 / m≥3
grouped paths *do* call the shared per-diagram evaluators with the override.

#### `_integrate_2d_polygon_modesum(...)` — `final_integral.py:2175`

m=2 analytic polygon integrator. Accepts `pole_tuples` override (grouped) — the
grouped path calls it with `smooth_edge_modes=list(dummy_modes)`,
`prefactor_complex=1.0+0.0j`, and `pole_tuples=grouped_pole_tuples`.

#### `_integrate_nd_polytope_poset_modesum(...)` — `final_integral.py:1726`

m≥3 analytic causal-poset chain-simplex integrator. Same override contract. This
is the path with the known float64 cancellation issue at close-paired poles
(see Gotchas).

#### `_integrate_polytope(integrand_callable, s_constraints, free_ext_vals, m)` — `final_integral.py:4372`

The scipy-backed **fallback** dispatcher (`m=0` direct eval, `m=1/2/≥3` →
`_integrate_1d/2d/nd_polytope`). Hit only when the analytic path returns `None`.

#### `_loop_number_from_graph(typed_diagram)` — `final_integral.py:2401`

`L = |E| − |V| + 1` from the prediagram graph. No frequency-domain dependency.

#### `_lookup_prop_indices(typed_diagram, edge_key)` — `final_integral.py:5230`

Resolves `(resp_row, phys_col)` propagator indices for a prediagram edge, with a
three-level fallback on the `(u, v, lbl)` edge key (exact → `(u,v,None)` →
first-matching-`(u,v)`). Used per td per edge in the grouped path.

#### `build_G_t_matrix / G_t_entry / G_t_delta_coeff` — `propagator_td.py:135 / 379 / 433`

- `build_G_t_matrix` assembles the `{'smooth', 'delta', 't_var'}` dict (smooth =
  `Σ_k C_k exp(I p_k t)` SR matrix; delta = `lim_{ω→∞} Ĝ` constants).
- `G_t_entry(G_t_obj, phys_idx, resp_idx, t_expr, include_heaviside)` reads the
  smooth `[phys, resp]` entry (the **transpose** of the kernel's `(resp, phys)`
  convention — both Phase I and Phase J must agree on this transpose) with the
  time variable substituted; the grouped path calls it with
  `include_heaviside=False` (causality is enforced by the polytope constraints,
  not by an explicit Θ).
- `G_t_delta_coeff(G_t_obj, phys_idx, resp_idx)` returns the δ(t) coefficient (the
  `ω→∞` limit), as a real if the imaginary part is negligible, else complex; `0`
  when there is no δ component.

#### `external_wick_compensation(typed_diagram)` — `symmetry.py:343`

Returns the exact orbit–stabilizer index `|Aut(Γ, leaves free)| / |Aut(Γ, leaves
fixed)|` via `_automorphism_order` (Sage automorphism-group orders). Raises
`RuntimeError` if `|Aut_free|` is not divisible by `|Aut_fixed|` (a Sage edge case
— fail loudly rather than mis-weight). Replaces the old `vertex_role_signature`
`∏N!` heuristic.

#### `vertex_role_signature(vertex, typed_diagram)` — `symmetry.py:247`

A hashable depth-1 role signature for an internal vertex (type/coefficient/bigrade
+ external-leaf attachment pattern + internal-edge incidence pattern). **Historical
basis of the old (now-superseded) compensation heuristic.** Still referenced in
`final_integral.py:2573`; not on the grouped path's critical numerical route.

---

## Data structures

### `propagator_data` (dict) — Phase-I output, consumed throughout

| key | type | meaning |
|---|---|---|
| `pole_vals` | list of `SR` (or numeric) | poles `p_k` of `Ĝ(ω)`, `Im(p_k) > 0` |
| `C_mats` | list of Sage matrices (one per pole) | `C_mats[k][i,j]` = residue of `Ĝ[i,j]` at `p_k` |
| `D_delta` | Sage matrix (optional) | `ω→∞` limits → δ(t) coefficients |
| `G_ft` | Sage matrix (optional) | frequency-domain propagator (used by `build_G_t_matrix`) |
| `nf` | int (optional) | propagator matrix size |

### `TypedDiagram` (attributes the grouped path reads)

- `td.prediagram` — a tuple `(D, …, leaves, …)`. `D = td.prediagram[0]` is the
  Sage graph; `td.prediagram[2]` is the leaf list. Grouped path keys on
  `id(td.prediagram)` and asserts `td.prediagram is pd0`.
- `td.external_legs` — dict `{leaf: (field_base, pop_idx)}`. Source of the
  group key's second component and the canonical leaf→field reference.
- `td.edge_types` — dict `{(u,v,lbl): (resp_leg, phys_leg)}`; `resp_leg[1]-1`
  gives the population index for noise-source leg routing.
- `td.propagator_indices` — dict `{(u,v,lbl): (ri, pi)}`, read by
  `_lookup_prop_indices`.
- `td.vertex_assignments` — dict `{vertex: VertexType}`; scanned for
  `NoiseSourceType` to detect non-local cumulant kernels.

### `groups` bucket (`defaultdict` value)

`{'tds': [TypedDiagram,…], 'cps': [prefactor,…]}` — keyed by
`(id(td.prediagram), ext_legs_signature)`.

### `edge_info` entry (per td per edge) — `grouped_integral.py:553`

```python
{'u', 'v', 'lbl', 'ri', 'pi', 'dt_sym', 'delta_coeff', 'smooth_factor'}
```

`dt_sym` is the symbolic `Δt` (an `SR` expression); `delta_coeff` is the complex
δ-weight; `smooth_factor` is the Heaviside-stripped smooth SR propagator entry.

### `subset_constraint_data` entry — `(a_int, a_ext, c0)`

A linear form for one smooth-edge `Δt = c0 + a_int·s + a_ext·t_free`: `a_int` is a
list of coeffs on the surviving integration vars; `a_ext` on the free external
times; `c0` the constant. Defines the half-space `Δt > 0`.

### Merged pole tuple — `(B_alpha, lambdas)`

`B_alpha` is the complex merged residue `Σ_j cp_j ∏_{δ} δ ∏_{smooth} C^{(j)}_{α_e}`;
`lambdas` is the tuple `(λ_{α_e})_e` of per-smooth-edge modes. This is the
grouped path's replacement for `_enumerate_pole_tuples`'s `(C_product, lambdas)`.

### Subset diagnostic record — `grouped_integral.py:1026` / `:695` / `:752`

`{'delta_edges', 'smooth_edges', 'status', 'm_after_delta', 'n_contributing_td',
'analytic_modesum'}` on success, or a 3-field skip record with `status ∈
{'dt_sym_mismatch_skipped', 'shotnoise_skipped_in_prototype'}`.

### `integrate_grouped_diagram` return dict

`{'status', 'contribution', 'n_diagrams', 'integration_vars', 'loop_number',
'subset_diagnostics'}` (or a failure variant with `'reason'`).

### `compute_correction_td_grouped` return dict

`{'total_C', 'total_C_batch', 'eval_per_diagram_batch' (None),
'delta_contributions' ([]), 'groups', 'skipped_kernel_ids', 'fallback_to_perdiag',
'ext_time_vars'}`. Each `groups` entry:
`{'kernel_id', 'loop_number', 'n_diagrams', 'handled_by'
∈ {'grouped_evaluator','perdiag_fallback','skipped'}, 'reason', 'representation'}`.

---

## Data flow

**In (from `compute.py:764`):**

```python
td_result_ell = compute_correction_td_grouped(
    typed_diagrams = [r['typed_diagram'] for r in records_ell],   # one loop order
    prefactors     = [r['combined_prefactor'] for r in records_ell],
    k              = k,
    propagator_data= propagator_data,        # pole_vals, C_mats, D_delta, …
    external_fields= external_fields,         # canonical leaf→field list
    num_params     = num_params,              # numeric substitutions
    origin_leaf_idx= origin_leaf_idx,         # which leaf pinned to t=0
)
```

**Grouping:** `(id(td.prediagram), sorted-external-legs)` buckets the flat list
into groups of ~hundreds of td's each (e.g. ~860 per group in the k=2 ell=1
multipop case).

**Per group:** `integrate_grouped_diagram` →
- builds `G_t_obj` once, `_all_mappings`/`_compensation` once, residue tensor once;
- loops `2^|E|` subsets, summing the integrand and (when analytic) building `B_α`;
- returns one `contribution(*t) -> complex`.

**Out (to `compute.py:787`):**

```python
total_C_by_ell[ell] = td_result_ell['total_C']
C_tau_by_ell[ell]   = np.array(
    td_result_ell['total_C_batch'](tau_points, parallel=…, n_workers=…),
    dtype=complex,
)
```

The master `total_C` sums across loop orders (`compute.py:802`) and `C_tau =
Σ_ell C_tau_by_ell[ell]` (`compute.py:808`). The grouped bookkeeping keys
(`groups`, `skipped_kernel_ids`, `fallback_to_perdiag`) are available for
diagnostics but are not required by the downstream τ-grid/saving path.

**Concrete example of the equivalence test** (from the docstring contract): for a
fixture like `quad_exp_k2_ell0` or `spike_reset_k2_ell0`, the regression harness
`tests/test_grouped_vs_perdiag.py` computes `Σ_N` per-diagram results
(`integrate_diagram`) and the single grouped result
(`integrate_grouped_diagram`) and asserts they match to ~`1e-14` rel on the
analytic path.

---

## Gotchas & caveats

1. **Silent subset drops are the production bug this wrapper guards against.**
   `integrate_grouped_diagram` can return `status='ok'` while a subset diagnostic
   says `shotnoise_skipped_in_prototype` or `dt_sym_mismatch_skipped` — meaning a
   piece of the integrand was dropped and the group total is *wrong*. The
   isolation test cited in the comment found "5/7 multi-diagram groups dropped to
   0" at spike-reset k=1 ell=2 (`_grouped_phase_j.py:131–133`). The wrapper
   detects these and falls back to per-diagram (Tier 2). **If you add a new skip
   status to the evaluator, you MUST add it to `_SUBSET_SKIP_STATUSES`** or the
   wrapper will accept incomplete integrands.

2. **Shot-noise δ-spikes are not handled in the prototype.** Residual
   external-time equalities after δ-elimination (τ=0 same-population legs in
   heterogeneous Hawkes) are skipped — the per-diagram path accounts for them in
   `delta_contributions`, which the grouped path does *not* rebuild
   (`grouped_integral.py:736–757`, docstring lines 63–66). The wrapper falls
   back to per-diagram when this fires.

3. **`delta_contributions` and `eval_per_diagram_batch` are unsupported** in the
   grouped return (`None` / `[]`, `_grouped_phase_j.py:246–248`). Any caller that
   needs per-diagram breakdowns or the δ-spike accounting must use the
   per-diagram path.

4. **`total_C_batch` is serial in the prototype** — it ignores `parallel` /
   `n_workers` (`_grouped_phase_j.py:237–241`). (Note: the codebase memory warns
   that fork-based parallelism crashes the user's macOS kernel; serial-only here
   is the safe default.)

5. **The analytic path is disabled by any noise-source kernel.** If *any* td in
   the group has a non-empty `vertex_leg_time` (a `NoiseSourceType` cumulant
   kernel), the integrand is non-rational and the merged-residue closed form
   cannot fire — it falls through to `fast_callable` + scipy
   (`grouped_integral.py:602–607`, docstring lines 88–93).

6. **The group must be homogeneous in external-legs *and* compensation.** The
   wrapper subdivides by external-legs (`_grouped_phase_j.py:78–94`). Inside the
   evaluator, if `external_wick_compensation` differs across group members,
   `integrate_grouped_diagram` returns `status='failed'`
   (`grouped_integral.py:437–448`) — the shared divisor would mis-weight the
   group; use the per-diagram path for that correlator. This is the MVP scope
   limit called out in the docstring (lines 50–57): "Wick contraction
   enumeration assumes all typed diagrams in the group share the same
   `external_legs` mapping … if a future enumerator emits different external-leg
   assignments for the same prediagram, regroup at a finer key."

7. **m ∈ {0, 1} analytic helpers are grouped-only twins, not shared code.**
   `_evaluate_grouped_m0_modesum` and `_integrate_grouped_m1_modesum` live in
   `grouped_integral.py` and re-implement the closed forms; the m=2/m≥3 paths
   *do* call the shared `final_integral.py` evaluators via the `pole_tuples`
   override. The MVP docstring (lines 58–62) flagged m=0/1 as needing "a separate
   pole-residue closure" — Stage 4a added exactly that.

8. **OPEN PRECISION BUG at m≥3 (close-paired poles).** Grouped and per-diagram
   *should* agree by linearity, but the spike-reset k=2 ell=1 case shows a ~4×
   discrepancy (per-diag +2.66e-3 vs grouped +6.38e-4). The
   `USE_POSET_CAP_MATCH_SCIPY` comment (`final_integral.py:1690–1700`) concludes
   the bbox-cap mismatch is *not* the dominant cause and points to "a bookkeeping
   difference in pole-tuple construction between the two paths." The grouped path
   pre-sums (cancelled-within-group) pole tuples → smaller magnitude → different
   float64 value than per-diag's sum-after-integration. The
   `USE_POSET_MPMATH_ACCUMULATION` flag (off by default) was an attempted fix that
   "did NOT address the spike-reset bug." Authoritative writeup:
   `docs/m_ge3_precision_bug_audit.md`. (Memory note:
   `project_chain_simplex_precision_bug.md`.)

9. **Spike-reset test failures are fixture/theory drift, NOT an integrator bug.**
   The four `spike_reset_*` fails in `tests/test_grouped_vs_perdiag.py` come from
   `_FUNDAMENTAL_SPIKE_RESET` still passing a `taug` param that the theory file
   dropped when it switched to a delta synaptic kernel; the failure happens in
   `compute_poles_and_residues` *before* any Phase-J integration. Use
   `quad_exp_k2_ell0` as the working regression fixture. (Memory note:
   `project_spike_reset_fixture_drift.md`.)

10. **Θ(0) = 0 convention everywhere.** A constraint `Δt > 0` is *strict*: the
    boundary `Δt = 0` is OUTSIDE the feasible region (`c_eff <= 0` → reject).
    This appears in `_evaluate_grouped_m0_modesum` (`:155`),
    `_integrate_grouped_m1_modesum` (`:217`), and `_integrate_polytope`'s m=0
    branch (`final_integral.py:4388–4392`). Mismatched conventions between paths
    would shift boundary contributions.

11. **Unbounded interval handling is subtle (and load-bearing for precision).**
    `_integrate_grouped_m1_modesum` tracks `±∞` exactly and zeroes the boundary
    term only when the integrand *decays* in that direction; otherwise it returns
    `None` to force the scipy fallback. Clipping at `±bbox_cap` instead would leave
    a residual `exp(α_s·bbox_cap) ≈ 1e-9` term for typical Hawkes timescales — the
    very error this code was written to remove (`grouped_integral.py:196–202`,
    measured on `quad_exp_k2_ell0`).

12. **Overflow guard, not error.** All four closed-form helpers return `None` (not
    raise) when `|γ.real| > 600` (`_GROUPED_EXP_REAL_LIMIT`), so the caller can
    fall back to scipy rather than crash. 600 is a deliberate margin below the
    709 hard ceiling of `cmath.exp` on IEEE doubles.

13. **`grouped_dummy_modes` carry fake residues.** The `EdgeModeSum` objects built
    at `grouped_integral.py:949–961` exist *only* to satisfy the `n_smooth` length
    check inside the per-diagram m=2/m≥3 evaluators; their `modes` are td[0]'s
    residues and are never read when `pole_tuples` is supplied. Do not mistake
    them for the actual weights — those live in `grouped_pole_tuples`.

14. **`USE_POSET_INTEGRATOR` / `USE_POLYGON_M2_INTEGRATOR` are read *live* via the
    module alias.** `grouped_integral.py:119` imports `final_integral as _fi_mod`
    so per-call reads pick up notebook overrides of those flags. If a notebook
    flips `_fi_mod.USE_POSET_INTEGRATOR = False`, the grouped m≥3 analytic path
    silently turns off and routes to scipy.

15. **`vertex_role_signature` is a deprecated heuristic basis.** It is NOT the
    current compensation mechanism; the exact `external_wick_compensation`
    (Sage automorphism ratio) replaced it. The history matters because the old
    `∏N!` heuristic *over-divided* at k≥3 (broke every k≥3 1-loop cumulant) while
    coincidentally matching at k=2 (`symmetry.py:368–381`).

16. **Prototype status.** Not wired into `compute_cumulants` by default — opt in
    with `use_grouped_phase_j=True` (`compute.py:129`, default `False`;
    `_grouped_phase_j.py` docstring lines 26–29).

---

## Glossary

- **MSR-JD field theory** — Martin–Siggia–Rose–Janssen–De Dominicis: a
  path-integral formulation of classical stochastic dynamics with a *physical*
  field `φ` and a *response* (hatted) field `ñ`. Propagators are entries of a
  matrix `G[φ, ñ]`; the response/physical index split is the `(ri, pi)` pair.
- **Cumulant** — connected `k`-point correlation function; the object Phase-J
  computes as a sum over diagrams.
- **Prediagram** — bare graph topology (vertices, leaves, edges, integration
  variables), *before* field-types and coefficients are assigned to edges.
- **Typed diagram** — a prediagram with per-edge propagator indices `(ri, pi)`
  and per-vertex coefficients fixed. Many typed diagrams share one prediagram.
- **Leaf / external leg** — a graph vertex carrying an external field at a fixed
  external time; one leaf is "pinned" to `t = 0` (the origin leaf).
- **Internal vertex** — a non-leaf vertex; its time is an integration variable.
- **Loop number `ℓ`** — `|E| − |V| + 1` (topological cycles in the undirected
  graph). Tree = 0, one loop = 1, etc.
- **Retarded propagator `G_R(t) = δ-part + Θ(t)·smooth-part`** — the
  causal Green's function; `Θ` is the Heaviside step enforcing `t > 0`.
- **Pole / residue** — `Ĝ(ω)` is rational; `pole_vals[k]` are its poles `p_k`
  and `C_mats[k]` the matrix residues. Time domain: `Σ_k C_k exp(i p_k t)`.
- **Mode `(C_α, λ_α)`** — one pole's residue and rate, with `λ_α = i p_α` so the
  smooth propagator is `Σ_α C_α exp(λ_α Δt)`.
- **δ-edge / smooth-edge subset** — the `2^|E|` expansion of the product of
  retarded propagators into "instantaneous δ vs smooth" on each edge.
- **δ-elimination** — a δ-edge imposes `Δt = 0`, a linear equation that removes
  one integration variable (solved by `sage_solve`).
- **Polytope / causal polytope** — the convex region cut out by the smooth-edge
  retardation inequalities `Δt_e > 0`; the integration domain.
- **`m`** — number of integration variables surviving δ-elimination in a subset;
  selects the closed-form path (0/1/2/≥3).
- **Pole tuple `α`** — a choice of one pole per smooth edge; expanding the product
  of mode-sums yields a sum over pole tuples.
- **Merged-residue tensor `B_α`** — the grouped path's per-tuple coefficient,
  `Σ_j cp_j ∏ δ ∏ C^{(j)}`, replacing the per-diagram Cartesian product
  `∏_e C_{α_e}`.
- **Causal poset / chain simplex** — for `m ≥ 3`, the partial order `s_u < s_v`
  read from constraints; integrating over its linear extensions (nested
  simplices) gives the polytope integral.
- **Fan triangulation** — splitting a convex polygon into triangles from vertex 0
  (the m=2 path).
- **External-Wick compensation `comp`** — the orbit–stabilizer divisor
  `|Aut(Γ,free)|/|Aut(Γ,fixed)|` that removes over-counting from the external-leaf
  permutation sum.
- **`vertex_role_signature`** — a depth-1 graph invariant of a vertex; basis of
  the *old* (superseded) compensation heuristic.
- **`SR` (Symbolic Ring)** — Sage's symbolic-expression type (Pynac/GiNaC).
- **`fast_callable`** — Sage's expression-to-fast-closure compiler (like
  `lambdify`).
- **`CDF` (Complex Double Field)** — Sage's hardware-`double` complex numbers.
- **`sage_solve`** — Sage's symbolic equation solver (Maxima-backed).
- **nauty / bliss** — graph-automorphism engines (used by Sage internally for
  `automorphism_group`).
- **scipy.nquad** — adaptive numerical multidimensional quadrature (the fallback,
  ~`1e-8` precision).
- **Shot-noise δ-spike** — a residual external-time equality (`τ = 0` same-pop)
  after δ-elimination; a separate δ-contribution the prototype skips.
- **NoiseSourceType / non-local cumulant kernel** — a noise vertex whose kernel
  introduces extra `τ` integration variables and a non-rational integrand,
  disabling the analytic path.
- **`origin_leaf_idx`** — index of the external leaf pinned to `t = 0`; the
  integrand is a function of time *differences*.
- **Stage 4a** — the milestone that made the grouped path fully analytic
  (merged-residue mode-sum), dropping the precision floor to machine ε.

---

## Proposed manual subsections

1. **Why group? — the per-diagram redundancy.** The 7736-diagram / 9-prediagram
   measurement; identical topology/domain across a prediagram's typed diagrams.
2. **Linearity of integration — the exactness argument.** `Σ∫ = ∫Σ` on a shared
   domain; why grouping is exact, not approximate.
3. **From retarded propagators to a polytope integral.** `G_R = δ + Θ·smooth`,
   the `2^|E|` subset expansion, δ-elimination, retardation inequalities → convex
   polytope.
4. **The mode-sum and pole tuples.** `Σ_α C_α exp(λ_α Δt)`, expansion into pole
   tuples, the linear exponent `γ_α + Σ α^{(i)}_s s_i`.
5. **The merged-residue tensor `B_α`.** Pre-summing residues across the group;
   identical analytic kernel, different per-tuple coefficient.
6. **Closed forms by `m`.** m=0 (constraint check), m=1 (interval with ±∞
   handling), m=2 (polygon + unit-triangle `J(p,q)`), m≥3 (causal-poset chain
   simplex).
7. **The grouping wrapper.** Group key `(prediagram, ext_legs)`, the three-tier
   fallback (grouped → per-diag on silent skip → per-diag on failure → loud skip),
   `_SUBSET_SKIP_STATUSES`.
8. **The grouped evaluator walkthrough.** Shared scaffolding once; per-td edge
   info; the subset loop; merged pole-tuple construction; the Wick-permutation
   sum and `/comp`.
9. **External-Wick compensation.** Orbit–stabilizer index, why role-signature
   `∏N!` failed at k≥3, the exact Sage automorphism ratio.
10. **Tooling primer for the physicist.** SageMath (`SR`, `fast_callable`, `CDF`,
    `solve`), nauty/bliss via `automorphism_group`, scipy.nquad as fallback.
11. **Precision: analytic vs scipy, grouped vs per-diagram.** Machine-ε analytic
    path; the m≥3 close-paired-pole cancellation open bug; the regression
    contract and `quad_exp_k2_ell0`.
12. **Limitations and the fallback matrix.** Noise-source kernels, shot-noise
    spikes, unsupported `delta_contributions`, serial batch; when the per-diagram
    path is required.
13. **Known drift and gotchas.** Spike-reset fixture/theory drift, Θ(0)=0
    convention, overflow guards, live flag reads.

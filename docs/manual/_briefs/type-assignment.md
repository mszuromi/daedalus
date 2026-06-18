# Subsystem brief: Type Assignment, Causality, Symmetry factor, Filter

Slug: `type-assignment`

Files covered (read in full):

- `msrjd/diagrams/filter.py`            (Build Phase D)
- `msrjd/diagrams/type_assignment.py`   (Build Phase E)
- `msrjd/diagrams/causality.py`         (Build Phase F)
- `msrjd/diagrams/symmetry.py`          (Build Phase G)
- `msrjd/fork_safety.py`                (shared fork-safety guard)

Supporting files read to resolve the data structures and the data flow:

- `msrjd/core/vertices.py`              (defines `VertexType` / `SourceType` / `ConvVertexType` / `NoiseSourceType`; Build Phase B)
- `pipeline/_diagrams.py`               (the orchestrator that wires Phases D→E→F→G together with disk caching)
- `msrjd/enumeration/loop_diagram_enumeration.py` (Build Phase C — produces the *prediagrams* this subsystem consumes)

---

## 1. Overview

### What this subsystem does, in plain language

The end-to-end pipeline computes correlation/response functions of a stochastic
field theory by (a) writing down the MSR-JD action, (b) enumerating all Feynman
diagrams up to some loop order, and (c) integrating each diagram. This subsystem
is the **bridge between "a bare topological graph" and "a fully-labelled Feynman
diagram you can actually integrate."** It does four jobs, in pipeline order:

1. **Filter (Phase D, `filter.py`).** Throws away topological graphs
   ("prediagrams") whose vertices have *degrees* that no vertex or source in the
   theory could ever supply. This is a cheap set-membership check that runs
   *before* the expensive labelling step, so we never waste time on a graph that
   is hopeless from the start.

2. **Type assignment (Phase E, `type_assignment.py`).** The heart of the
   subsystem. Takes one bare directed graph and produces **every valid way to
   decorate it** with field types: which monomial of the action sits at each
   internal vertex, which external field sits on each external leg, and — for
   each directed edge — which response field (`ñ`, "tilde") feeds the tail and
   which physical field (`φ`) feeds the head, i.e. which propagator `G_{ij}` the
   edge carries. It is a constraint-satisfaction / backtracking engine: degrees
   must match, fields must point the right way, and the chosen propagator must
   not be identically zero. Output: a stream of `TypedDiagram` objects.

3. **Causality (Phase F, `causality.py`).** Prunes typed diagrams that violate
   the retarded boundary condition of the MSR-JD formalism. Structurally this
   means "the graph must be a DAG (no directed cycles)"; analytically it means
   "all poles of the propagator must sit in the upper half of the complex
   frequency plane." For the typed-diagram set we generate, the structural DAG
   check is the operative one.

4. **Symmetry factor & dedup (Phase G, `symmetry.py`).** Two things. First, it
   deduplicates the typed diagrams — many of the diagrams Phase E emits are the
   *same* diagram up to relabelling — using a complete graph-isomorphism
   invariant computed via Sage's canonical-labelling machinery. Second, it
   computes the **symmetry factor** `𝒮(Γ)` for each surviving diagram, the
   purely combinatorial rational number (here always an integer) that every
   Feynman rule carries, computed as

   ```
   𝒮(Γ) = (∏_v ∏_ℓ n_{v,ℓ}!) / |Aut_fixed_ext(Γ)|
   ```

   where the numerator counts distinguishable Wick pairings at each vertex and
   the denominator is the order of the diagram's automorphism group, obtained
   from Sage's `automorphism_group(...)`.

`fork_safety.py` is a small shared safety guard. The type-assignment step is
"embarrassingly parallel" across prediagrams, so it offers an optional
fork-based multiprocessing path. But forking a process on macOS *after* the
Cocoa/BLAS/matplotlib libraries have initialized — which is always the case
inside a Jupyter kernel — can **hard-crash the kernel and the entire OS**. This
guard detects "am I inside a macOS Jupyter kernel?" and, if so, silently
degrades the parallel path to serial.

### Where it sits in the pipeline

Upstream (feeds this subsystem):

- **`msrjd/core/vertices.py` (Phase B)** produces the lists of `VertexType` and
  `SourceType` objects — one per monomial of the interacting action and the
  noise kernel. These describe "what kinds of vertices the theory offers and
  what legs they have."
- **`msrjd/enumeration/loop_diagram_enumeration.py` (Phase C)** produces the
  *prediagrams*: bare directed multigraphs with `k` external leaves and `ℓ`
  loops, isomorph-deduplicated, with no field types assigned yet. Each
  prediagram is a 4-tuple `(D, G, leaves, internal)` (see §5).
- The **symbolic propagator matrix `G_ft`** (built elsewhere from the quadratic
  action) is consumed read-only — only its *zero/nonzero pattern* matters here.

Downstream (consumes this subsystem's output):

- **`msrjd/integration/...` (Phase J)** — the integrators. They take each unique
  `TypedDiagram`, read its `edge_types` / `propagator_indices` / `external_legs`
  / `vertex_assignments`, multiply by `𝒮(Γ)` and the vertex coefficients, sum
  over external-leaf assignments (dividing by `external_wick_compensation`), and
  perform the actual time/frequency/momentum integral.
- **`msrjd/diagrams/symmetry.py::classify_coefficient_factors`** is also called
  by the integrators (`integration/symbolic.py`, the spatial
  `pipeline_bridge.py`) to split each vertex coefficient into a scalar prefactor
  that can be pulled outside the integral vs. a time-dependent factor that must
  stay inside.

The single function that wires all four phases together is
`pipeline/_diagrams.py::enumerate_unique_diagrams`, which adds disk caching:

```python
# pipeline/_diagrams.py:176-186
_, _, prediagrams, _ = enumerate_prediagrams_all(k=k, ell=ell, verbose=False)
typed = enumerate_all_typed(prediagrams, external_fields, vtypes, stypes,
                            G_ft=G_ft, resp_index=resp_idx, phys_index=phys_idx,
                            parallel=parallel, n_workers=n_workers)
causal, n_disc, _ = filter_causal(typed)
unique, multiplicities = deduplicate_with_multiplicities(causal)
```

Note that the `filter_prediagrams` (Phase D) call is *not* in this orchestrator —
the loop enumerator already produces only degree-valid topologies for the chosen
`(k, ℓ)`, so the explicit filter is exercised mostly by tests
(`tests/test_filter.py`) and by callers that enumerate prediagrams more loosely.
It remains the canonical cheap pre-screen and is documented here in full.

---

## 2. The math

A reader who knows MSR-JD field theory will recognize all of this; we restate it
from the ground up so the code's choices are unambiguous.

### 2.1 The MSR-JD setup and the two field species

A Langevin equation `∂_t φ = F[φ] + ξ` with noise `ξ` is rewritten, à la
Martin–Siggia–Rose / Janssen–De Dominicis (MSR-JD), as a path integral over
*two* fields per physical field:

- the **physical field** `φ` (in the code's naming, generators like `vt`, `nt`,
  `xt`; the "t" is part of the base name, not "tilde"), and
- the **response field** `ñ` ("tilde", in the code base names like `dv`, `dn`,
  `dx` — the `d` prefix is the response/conjugate field).

The generating functional is `Z = ∫ D[φ] D[ñ] exp(−S[φ, ñ])`. The Gaussian
(quadratic) part of `S` defines the **propagators**; the higher-order part
defines the **interaction vertices**; the noise enters as **source vertices**
(pure response-field monomials from the noise cumulant generating functional).

### 2.2 Edges, propagators, and the response/physical convention

Every internal line of a diagram is a propagator. In MSR-JD there is exactly one
"forward in time" propagator family: the **retarded response** `⟨φ ñ⟩` (and the
**correlation** `⟨φ φ⟩`, which factors into two responses joined by a noise
vertex). The code encodes this with a strict directionality convention,
stated at `type_assignment.py:11-13`:

> each directed edge `u → v` carries a propagator `G_{ij}` where `i` is the
> response-field index contributed by vertex `u` (tail) and `j` is the
> physical-field index contributed by vertex `v` (head).

So a directed edge is a *Wick contraction* between a response leg `ñ_i` at its
tail and a physical leg `φ_j` at its head. The propagator value is the entry of
the propagator matrix `G_ft = K_ft^{-1}` (the inverse of the quadratic-form
matrix `K`). The crucial index convention is restated repeatedly in the code
(`type_assignment.py:293-296`, `:490-502`):

> `G_ft = K_ft^{-1}` has rows = physical, cols = response, so
> `G_ft[i, j] = ⟨φ_i ñ_j⟩`. The propagator on an edge resp-tail → phys-head is
> `⟨φ_phys ñ_resp⟩ = G_ft[pi, ri]`.

i.e. you index `G_ft` with **(physical-row, response-col)**, in that order. This
ordering is load-bearing: §7 records a comment about a historical `[ri, pi]` bug
that only manifested for off-diagonal couplings.

### 2.3 Retarded causality

The retarded Green's function `G_{ij}(t)` is nonzero only for `t > 0`. In
Fourier space this is the statement that **all poles of `det(K̂(ω))` lie in the
upper half of the complex `ω` plane** (`Im(ω_k) > 0`), so that closing the
contour the "correct" way yields the retarded boundary condition. Two equivalent
manifestations:

- **Structural:** a diagram built only from retarded responses and noise
  vertices is a **directed acyclic graph** — there is no way to return to a
  vertex by following time-forward arrows. A directed cycle would require
  closing a loop of strictly-retarded propagators, which integrates to zero (or
  signals an acausal construction). This is the operative check, implemented as
  `D.is_directed_acyclic()`.
- **Analytic:** if explicit pole locations are supplied, every pole must have
  `Im > 0`. Implemented in `check_pole_structure`, but in the current pipeline
  the typed-diagram path passes no `pole_vals`, so only the DAG check runs.

### 2.4 The symmetry factor `𝒮(Γ)` (the heart of Phase G)

This is the part most worth getting exactly right, because a wrong `𝒮(Γ)`
silently produces a numerically-plausible but wrong answer. The code implements
the **canonical Feynman rule** ("Path A"):

```
𝒮(Γ)  =  (∏_v ∏_ℓ n_{v,ℓ}!)  /  |Aut_fixed_ext(Γ)|
```

(`symmetry.py:403-446`). Two pieces:

**Numerator — the Wick leg factor.** For each vertex `v`, and each leg-type `ℓ`
(a leg-type lumps *direction* — response vs physical — with the *field*), let
`n_{v,ℓ}` be the number of legs of type `ℓ` at `v`. The numerator is

```
∏_v [ (∏_r n_{v, resp_r}!) × (∏_p n_{v, phys_p}!) ]
```

This is "the Wick combinatorial you'd get if every leg at every vertex were
distinguishable." A degree-3 vertex `φ²ñ` contributes `2!` from its two
identical `φ` legs; etc. Implemented in `_wick_leg_factor`
(`symmetry.py:115-146`).

**Denominator — the automorphism group order.** `|Aut_fixed_ext(Γ)|` is the
order of the automorphism group of the *fully coloured* diagram, with the
external legs **pinned** at their canonical positions ("fixed_ext"). An
automorphism is a relabelling of internal vertices and edges that maps the
diagram to itself while preserving every label that physics cares about: vertex
type, vertex coefficient, the propagator on each edge, and the direction of each
edge. It captures three families of redundancy:

- same-type internal vertex swaps,
- parallel-edge swaps between the same vertex pair,
- self-loop / tadpole leg-pair swaps.

The order is computed by Sage (see §3). Dividing the Wick numerator by the
automorphism order removes exactly the over-counting that those redundancies
introduce, yielding the canonical symmetry factor.

The per-diagram weight that the integrator builds (`symmetry.py:421-423`) is:

```
weight(Γ)  =  𝒮(Γ)  ×  ∏_v c_user,v  ×  ∏_e P_e  ×  ∫ dt_v
```

where `c_user,v` is the **literal** coefficient written in the action (no
implicit `1/n!`), `P_e` is the propagator on edge `e`, and the integral is over
internal vertex times.

### 2.5 The two automorphism orders: fixed vs. free external legs

There are two automorphism counts, differing only in whether external leaves may
be permuted:

- `|Aut_fixed_ext(Γ)|` — each external leaf gets its own colour class, so leaves
  can never be permuted. This is the denominator of `𝒮(Γ)`.
- `|Aut_free_ext(Γ)|` — same-field external leaves share a colour, so they *may*
  be permuted by an automorphism.

`Aut_fixed_ext` is a subgroup of `Aut_free_ext`, so `|free|` is always an integer
multiple of `|fixed|`. Their ratio is the **external Wick compensation**:

```
comp  =  |Aut_free_ext(Γ)| / |Aut_fixed_ext(Γ)|
```

(`external_wick_compensation`, `symmetry.py:343-400`). Why it exists: the
integrators enumerate **every** field-respecting assignment of canonical
external positions to diagram leaves and sum the integrand over all of them
(the `_all_mappings` sum). By the **orbit–stabilizer theorem**, two assignments
give the same *pinned-external* diagram iff they differ by a leaf-permuting
automorphism, and the stabilizer of any assignment is exactly `Aut_fixed_ext`.
So each distinct pinned diagram is counted `|free|/|fixed|` times by the mapping
sum; dividing by `comp` makes the sum equal the sum over distinct pinned
diagrams — which is what the Feynman rules require, since each pinned diagram
already carries its full weight through `𝒮(Γ)` (whose denominator is the *same*
`Aut_fixed_ext`).

### 2.6 Per-vertex combinatorial factor (the alternative `M_v` formula)

`symmetry.py` also contains an older, vertex-local way to count Wick pairings,
`_vertex_combinatorial_factor` (`symmetry.py:51-112`). For a single vertex `v`
with outgoing edges each carrying a pairing `(r, t)` (response type `r`, target
`t`), with:

```
n_r     = number of response legs of type r at v
n[r][t] = number of outgoing edges with pairing (r, t)
m_t     = number of outgoing edges targeting t
```

the per-vertex factor is

```
M_v = [ ∏_r  n_r! / ∏_t n[r][t]! ] × [ ∏_t m_t! ]
```

and `𝒮(Γ) = ∏_v M_v` in the older "Path B" formulation. The leg-matching
enumerator `_leg_matchings` (§4) relies on this `M_v` semantics to justify
generating only the *canonical* leg ordering (one representative) per vertex
rather than all `N!` index permutations. The production `combinatorial_factor`
used today is the **Path A** Aut-based formula of §2.4, not this `M_v` product;
`_vertex_combinatorial_factor` is retained for documentation/cross-checks.

---

## 3. External tools used

This subsystem leans heavily on **SageMath** (specifically its symbolic ring and
its graph theory built on **nauty**), plus one helper from **SymPy** and a few
Python-stdlib pieces. None of NumPy/SciPy/numba/networkx is used directly here.

### 3.1 SageMath — what it is

**SageMath** ("Sage") is a large open-source mathematics system that bundles many
specialized libraries behind one uniform Python API. When you `from sage.all
import ...`, you get symbolic algebra, exact arithmetic, graph theory, group
theory, linear algebra, etc. The code runs *inside* a Sage-provided Python
(`sage -python ...`), which is why the project's working interpreter is the
`MSRJD_diagrams` conda env (see project memory). Two Sage facilities are used
here:

**(a) The Symbolic Ring `SR`.** `SR` is Sage's ring of symbolic expressions
(think: a CAS expression tree). `SR(x)` coerces a Python number or another
expression into a symbolic expression. Methods like `.is_zero()`,
`.simplify_full()`, `.variables()`, `.coefficient(sym)`, `.subs({...})` operate
on it. Imports and uses in this subsystem:

```python
# type_assignment.py:19
from sage.all import SR
```
```python
# causality.py:16
from sage.all import SR, imag_part
```
```python
# symmetry.py:46
from sage.all import SR
```

- In `type_assignment.py`, `SR` appears mainly via the comment about
  `SR(G_ft[pi, ri]).is_zero()` being expensive; the code deliberately *avoids*
  calling it and instead does the structural string check
  `str(G_ft[_i,_j]) == '0'` (`type_assignment.py:165`) — see §7. So `SR` is
  imported but the symbolic-simplification path is intentionally bypassed for
  speed.
- In `causality.py`, `SR(pole)` coerces each pole into a symbolic expression and
  `imag_part(...)` extracts its imaginary part; `im.simplify_full()` tries to
  simplify it so the sign test `bool(im_simplified > 0)` can decide. `imag_part`
  is Sage's symbolic "take the imaginary part" function.
- In `symmetry.py`, `SR(...)` is used to coerce the combinatorial factor and
  vertex coefficients into symbolic expressions in
  `classify_coefficient_factors`, and `expr.variables()`,
  `coeff.subs({...})`, `const_part.is_one()`, `is_zero()`,
  `td_part.simplify_rational()` are used to split coefficients into
  pull-out-able vs. integrand parts.

**(b) Sage graph theory: `DiGraph`, automorphism groups, canonical labelling.**
Sage's `DiGraph` is a directed-graph object. Crucially, Sage's graph
isomorphism / automorphism / canonical-form algorithms are powered by **nauty**.

### 3.2 nauty — what it is, and how Sage uses it here

**nauty** ("No AUTomorphisms, Yes?") by Brendan McKay is the gold-standard C
library for graph canonical labelling and automorphism-group computation. Given a
graph (optionally vertex-coloured via a *partition*), nauty computes:

- a **canonical form** — a relabelling of the vertices into a normal order such
  that two graphs are isomorphic **iff** their canonical forms are identical, and
- the **automorphism group** — all relabellings that map the graph to itself.

Sage wraps nauty behind two `DiGraph` methods this subsystem calls. The
"partition" argument is how you pass a vertex colouring: nauty is told it may
only map a vertex to another vertex *in the same partition cell*. This is exactly
how the code forbids, e.g., mapping a noise source onto an interaction vertex.

**Call 1 — automorphism-group order** (`symmetry.py:339-340`):

```python
grp = D.automorphism_group(partition=partition)
return int(grp.order())
```

`D` is the coloured incidence digraph (see §4), `partition` is the list of colour
classes. `grp` is a Sage permutation group; `grp.order()` is its size. We only
need the order (an integer), so we cast it and discard the group itself.

**Call 2 — canonical labelling** (`symmetry.py:517`):

```python
C, cert = D.canonical_label(partition=partition, certificate=True)
```

`C` is the canonical relabelled copy of `D`; `cert` (the "certificate") is the
dict mapping each original vertex id to its canonical-label id. The code then
builds a hashable signature from (the colour keys, which canonical ids each
colour class occupies, the canonical edge list) so that two diagrams hash equal
**iff** they are isomorphic coloured graphs. This is `diagram_signature`
(`symmetry.py:467-524`), the dedup key.

`DiGraph` itself is imported lazily inside the builder:

```python
# symmetry.py:180
from sage.all import DiGraph
...
# symmetry.py:239
D = DiGraph(multiedges=False, loops=False)
D.add_vertices(nodes_added)
D.add_edges(edges_directed)
```

It also queries graph structure throughout the subsystem:
`D.vertices()`, `D.edges()`, `D.in_degree(v)`, `D.out_degree(v)`,
`D.incoming_edges(v)`, `D.outgoing_edges(v)`, `D.is_directed_acyclic()`. These
are the prediagram's own Sage `DiGraph` (built upstream in Phase C). Note the
incidence graph built in `symmetry.py` is a **fresh, separate** `DiGraph` — a
bipartite "vertex-nodes + edge-nodes" graph constructed solely to feed nauty
(see §4) — not the prediagram itself.

### 3.3 SymPy — what it is, and the one place it's used

**SymPy** is a pure-Python computer-algebra library (independent of Sage). This
subsystem uses exactly one helper from it, `multiset_permutations`, inside
`_leg_matchings`:

```python
# type_assignment.py:406-410
from sympy.utilities.iterables import multiset_permutations
resp_perms = (list(multiset_permutations(list(resp_legs)))
              if resp_legs else [[]])
phys_perms = (list(multiset_permutations(list(phys_legs)))
              if phys_legs else [[]])
```

`multiset_permutations(seq)` yields each **distinct** ordering of a multiset
exactly once. For a multiset of length `N` with multiplicities `n_r`, that is
`N! / ∏_r n_r!` orderings (a multinomial), *not* the full `N!`. Using it instead
of `itertools.permutations` is the optimization that keeps the typed-diagram
count from blowing up by `𝒮(Γ)`× (see §4, `_leg_matchings`). The leg values are
`(field_base, pop_idx)` tuples (hashable), so the multiset comparison works.

### 3.4 Python standard library

- **`itertools.permutations` and `itertools.product`** (`type_assignment.py:18`).
  `permutations(seq)` yields all orderings; the local helper
  `_distinct_permutations` wraps it in a `set(...)` to dedupe identical external
  fields. `product(*lists)` is the Cartesian product, used to enumerate every
  combination of vertex-type choices across the internal vertices.
- **`collections.Counter` / `defaultdict`** (`symmetry.py:41`, `:301`). `Counter`
  tallies multiplicities (leg-type counts, pairing counts); `defaultdict(list)`
  groups edges by their other endpoint.
- **`functools.reduce`, `operator.mul`, `math.factorial`** (`symmetry.py:42-44`).
  `factorial(n)` is `n!`; `reduce(mul, parts, SR(1))` multiplies a list of
  symbolic parts into one product.
- **`multiprocessing`** (imported lazily in `type_assignment.py:648`) — the
  optional fork-based parallel path; see §3.5.
- **`IPython.get_ipython`, `sys`, `warnings`** (`fork_safety.py`) — used to
  detect a Jupyter kernel and emit the fallback warning.

### 3.5 Why fork is lethal here, and the guard (`fork_safety.py`)

POSIX `fork()` clones the current process. The stdlib `multiprocessing` "fork"
start method uses it to spawn workers that **inherit the parent's memory** —
which is exactly why `type_assignment.enumerate_all` chooses fork: Sage `DiGraph`
objects, `VertexType`/`SourceType` instances, and the propagator matrix don't all
pickle cleanly through stdlib `pickle`, so passing them to workers via pickling
(the "spawn" method) would fail. With fork, the workers just *have* them.
(`type_assignment.py:551-560`.)

The catastrophe: on **macOS**, Apple's runtime forbids using many Objective-C /
Cocoa APIs after a `fork()` without an `exec()`. Once Cocoa, BLAS, or matplotlib
has initialized in the parent — which always happens inside a Jupyter kernel —
forking and then touching those libraries in the child can **hard-crash the
kernel and the entire OS**. (The project memory records this crashing the dev
machine twice.) The "fix" people reach for,
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`, makes it **worse** — it converts a
controlled abort into a hard crash (`fork_safety.py:8-11`).

So the guard `fork_unsafe_in_notebook` returns `True` only in the precise lethal
situation and `False` everywhere fork is safe:

```python
# fork_safety.py:27-44
def fork_unsafe_in_notebook(start_method='fork'):
    if start_method != 'fork':
        return False
    if sys.platform != 'darwin':
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        return False
    return ip is not None and ip.__class__.__name__ == 'ZMQInteractiveShell'
```

The Jupyter detection is the class-name check `ZMQInteractiveShell` (the IPython
shell class that *only* a ZMQ/Jupyter kernel uses; terminal IPython uses
`TerminalInteractiveShell`, plain scripts and pytest return `None`). When the
guard fires, `enumerate_all` warns once and degrades to serial
(`type_assignment.py:630-634`):

```python
from msrjd.fork_safety import fork_unsafe_in_notebook, warn_fork_fallback_once
if (parallel and len(prediagrams) > 1 and fork_unsafe_in_notebook(start_method)):
    warn_fork_fallback_once('diagram type-assignment enumeration')
    parallel = False
```

`warn_fork_fallback_once(where)` emits a single `RuntimeWarning` per call-site
label per process (`fork_safety.py:47-61`), deduped via the module-level
`_WARNED` set.

---

## 4. Components

Listed in pipeline order: Phase D (filter), Phase E (type assignment), Phase F
(causality), Phase G (symmetry). For each: name, file:line, signature, inputs,
outputs, and step-by-step behaviour.

### Phase D — `msrjd/diagrams/filter.py`

#### `classify_prediagram_vertices(D, leaves)` — `filter.py:12-40`

```python
def classify_prediagram_vertices(D, leaves):
```

- **Takes:** `D` a Sage `DiGraph` (the prediagram), `leaves` a list of leaf
  (external-leg) vertex ids.
- **Returns:** `(source_vertices, interaction_vertices)`, two lists of non-leaf
  vertex ids.
- **Steps:** builds `leaf_set = set(leaves)`; iterates `D.vertices()`; skips
  leaves; a non-leaf with `in_degree == 0` is a **source** (a noise vertex — no
  incoming response edges), otherwise an **interaction** vertex.

#### `filter_prediagrams(prediagrams, vertex_types, source_types)` — `filter.py:43-81`

```python
def filter_prediagrams(prediagrams, vertex_types, source_types):
```

- **Takes:** `prediagrams` a list of `(D, G, leaves, internal)` 4-tuples;
  `vertex_types` a list of `VertexType`; `source_types` a list of `SourceType`.
- **Returns:** `(kept, n_discarded)` — the filtered list and the count removed.
- **Steps:**
  1. `available_degrees(vertex_types, source_types)` (from `core/vertices.py`)
     returns `int_degs` (a set of `(in_degree, out_degree)` pairs the theory's
     interaction vertices can supply) and `src_degs` (a set of `out_degree`
     values its sources can supply).
  2. For each prediagram, classify its vertices, then check: every source vertex
     must have `out_degree ∈ src_degs`; every interaction vertex must have
     `(in_degree, out_degree) ∈ int_degs`.
  3. The first failing vertex sets `valid = False` and short-circuits; only
     prediagrams that pass *every* check are kept.
- **Why:** this is the cheap pre-screen — pure degree/set-membership, no field
  labelling, no Sage isomorphism. A prediagram with a degree-4 internal vertex in
  a theory with only cubic vertices is rejected here in O(vertices) time instead
  of after a fruitless backtracking search in Phase E.

### Phase E — `msrjd/diagrams/type_assignment.py`

#### `class TypedDiagram` — `type_assignment.py:29-71`

The output object of Phase E. A `__slots__` class (so instances are memory-lean
and you cannot accidentally add attributes). Fields (see §5 for shapes):
`prediagram`, `vertex_assignments`, `edge_types`, `external_legs`,
`propagator_indices`. Implements `__getstate__`/`__setstate__` so it pickles
across the fork pool (`type_assignment.py:60-65`), and a `__repr__` summarizing
vertex/edge/assignment counts.

#### `_distinct_permutations(seq)` — `type_assignment.py:22-24`

```python
def _distinct_permutations(seq):
    return set(permutations(seq))
```

Yields each distinct ordering of `seq` once, by materializing all permutations
into a set. Used to enumerate external-field assignments across leaves without
double-counting when two external fields are identical.

#### `build_field_index_map(ring_var_names, n_tilde)` — `type_assignment.py:76-110`

```python
def build_field_index_map(ring_var_names, n_tilde):
```

- **Takes:** `ring_var_names` the ordered list of polynomial-ring generator
  names (e.g. `['vt1','vt2','nt1','nt2','dv1','dv2','dn1','dn2']`); `n_tilde` the
  number of response-field generators (the **first** `n_tilde` names are response
  fields).
- **Returns:** `(resp_index, phys_index)`, two dicts mapping
  `(field_base, pop_idx) → matrix_row_or_col`. `resp_index` numbers the response
  fields `0,1,...`; `phys_index` numbers the physical fields `0,1,...`.
- **Steps:** parse each generator name with `_parse_field_name` (from
  `core/vertices.py`) into `(base, pop_idx)`; if its position `i < n_tilde` it's a
  response field (row counter), else a physical field (col counter). These maps
  are the bridge from "a leg `(base, pop_idx)`" to "a row/col of `G_ft`."

#### `enumerate_typed_diagrams(prediagram, external_fields, vertex_types, source_types, G_ft, resp_index, phys_index)` — `type_assignment.py:115-266`

The top-level enumerator for **one** prediagram. A generator yielding
`TypedDiagram`s.

- **Takes:**
  - `prediagram` = `(D, G_graph, leaves, internal)`.
  - `external_fields` = the length-`k` ordered list of external-leg field tuples
    in the correlation function.
  - `vertex_types`, `source_types` = the theory's available `VertexType` /
    `SourceType` lists.
  - `G_ft` = the propagator matrix (or `None` to skip the zero-pattern check).
  - `resp_index`, `phys_index` = the field-index maps.
- **Yields:** every valid `TypedDiagram`.
- **Steps:**
  1. **Precompute the zero/nonzero mask of `G_ft`** (`:158-165`). For each
     `(i, j)` entry, record `g_zero_mask[(i,j)] = (str(G_ft[i,j]) == '0')`. This is
     a deliberately *structural* (string-based) test, not the semantically-correct
     `is_zero()`, to avoid triggering Sage's expensive symbolic simplification on
     long rational propagators (a multi-second cost per entry on rich theories).
     It catches only literal `SR(0)` cells. A mathematically-zero-after-cancellation
     entry slips through and merely produces a few extra diagrams that integrate
     to 0 downstream (a harmless over-generation, never a wrong answer). See §7.
  2. **Classify non-leaf vertices** (`:167-177`) into `source_verts` (in_degree 0)
     and `interaction_verts` (in_degree > 0); `ordered_internal = source_verts +
     interaction_verts` fixes a deterministic processing order.
  3. **Precompute per-vertex incident-edge lists** (`:179-184`) via
     `D.outgoing_edges(v)` / `D.incoming_edges(v)`. Edges are Sage 3-tuples
     `(u, v, label)` so that *multi-edges* (parallel edges between the same pair)
     stay distinct.
  4. **Build candidate type lists per vertex** (`:186-201`). A source vertex of
     out-degree `od` gets all `SourceType`s with `st.out_degree == od`; an
     interaction vertex of bidegree `(ind, od)` gets all `VertexType`s with
     matching `(in_degree, out_degree)`. If any vertex has **no** candidate, the
     whole prediagram is impossible → `return` (yields nothing).
  5. **Determine leaf direction constraints** (`:204-213`). A leaf with only an
     outgoing edge can carry only a response field (`'resp'`); only-incoming →
     `'phys'`; both → `'both'`.
  6. **Enumerate external-field permutations** (`:223-242`). For every distinct
     ordering of `external_fields` across the leaves, check that each leaf's field
     is compatible with its direction (`resp` field must be in `resp_index`, etc).
     Why enumerate here: the upstream prediagram dedup treats all leaves as
     interchangeable, so one prediagram stands in for every leaf permutation; we
     must materialize them so diagrams differing in *which leaf carries which
     field* are all generated. Truly-identical ones are merged later by
     `deduplicate_typed_diagrams`.
  7. **Dispatch on whether there are internal vertices.** If `ordered_internal`
     is empty (`:244-251`), all vertices are leaves and edges just connect leaves —
     delegate to `_try_build_diagram_no_internal`. Otherwise (`:253-266`), take the
     Cartesian `product(*candidate_lists)` over each internal vertex's candidate
     types; for each combination build `vert_assignment` and delegate to
     `_try_build_diagram`.

#### `_try_build_diagram_no_internal(...)` — `type_assignment.py:269-305`

Handles the degenerate case of a diagram with no internal vertices (e.g. a single
edge directly connecting two external leaves — a bare propagator). For each edge
`u → v`: the tail `u` contributes the response leg (its external field), the head
`v` contributes the physical leg; verify both are direction-compatible
(`resp_field in resp_index`, `phys_field in phys_index`); look up
`ri = resp_index[resp_field]`, `pi = phys_index[phys_field]`; reject if
`g_zero_mask[(pi, ri)]` (note the **(pi, ri)** order). On success, record
`edge_types[edge] = (resp_field, phys_field)` and `prop_indices[edge] = (ri, pi)`,
then `yield` a `TypedDiagram` with empty `vertex_assignments`.

#### `_try_build_diagram(...)` — `type_assignment.py:308-337`

Given fixed external + vertex-type assignments, set up the per-vertex leg-matching
options and launch the backtracker. For each internal vertex, call
`_leg_matchings(vtype, out_edges, in_edges)` to get all distinct
`(resp_map, phys_map)` options; if any vertex yields zero options the combination
is dead (`return`). Then call `_backtrack(... vertex_idx=0, assigned_resp={},
assigned_phys={})`.

#### `_leg_matchings(vertex_type, out_edges, in_edges)` — `type_assignment.py:340-416`

The leg-to-edge bijection enumerator — and one of the most carefully-reasoned
functions in the subsystem.

- **Takes:** a `vertex_type`, the vertex's `out_edges` and `in_edges`.
- **Yields:** distinct `(resp_map, phys_map)` pairs, each a dict `{edge: leg}`,
  where each outgoing edge gets a response leg and each incoming edge gets a
  physical leg.
- **Steps:**
  1. Read `resp_legs = vertex_type.response_legs`, `phys_legs =
     vertex_type.physical_legs` (sources lack `physical_legs`, so default to `[]`).
  2. If the counts don't match the edge counts, `return` (no options).
  3. Use SymPy's `multiset_permutations` (§3.3) to generate the
     `N!/∏ n_r!` **distinct** orderings of each leg multiset (materialized to
     lists so the inner loop can re-iterate). For each `(rp, pp)` ordering pair,
     `zip` legs onto edges and yield the maps.
- **Why the multiset (not full `N!`) matters — the canonical-leg-matching
  change.** Earlier code enumerated all `N!` index permutations and relied on the
  downstream `deduplicate_typed_diagrams` to discard the over-count, which blew
  the typed-diagram count up to ~15–20× the unique count before dedup. The
  docstring (`:356-387`) carefully proves numerical equivalence: the over-count
  factor is exactly `𝒮(Γ)`; the old code generated `𝒮(Γ)` identical copies each
  carrying full weight, dedup kept one, the integrator multiplied by `𝒮(Γ)`. The
  new code generates the single canonical copy directly, the integrator multiplies
  by the same `𝒮(Γ)` — **bit-identical output**, far smaller intermediate set.
  Pinned by `test_leg_matchings_canonical` and
  `test_enumerate_typed_signatures_match_pre_change`.

#### `_backtrack(...)` — `type_assignment.py:419-538`

The recursive constraint-satisfaction core.

- **State:** `vertex_idx` (which internal vertex we're assigning), `assigned_resp`
  (`{edge: resp_leg}` so far), `assigned_phys` (`{edge: phys_leg}` so far).
- **Base case** (`:431-472`): all internal vertices assigned. For each edge,
  resolve its response leg (from `assigned_resp` if an internal tail set it, else
  from the external field if the tail is a leaf) and its physical leg
  (symmetrically). Look up `ri`/`pi`; if either is missing or the propagator is
  zero (`g_zero_mask[(pi, ri)]`), `return`. On full success, `yield` the
  `TypedDiagram` with `dict(vert_assignment)`, `edge_types`, `ext_assignment`,
  `prop_indices`.
- **Recursive step** (`:474-538`): for each `(resp_map, phys_map)` option at
  `vertex_idx`, merge into fresh `new_resp`/`new_phys`; run an **early propagator
  consistency check** on every edge that is now fully determined (both ends
  known) — checking both internal-internal edges and edges where one end is a
  leaf — and prune immediately if any propagator is zero. Only if `consistent`,
  recurse to `vertex_idx + 1`. Early pruning is what keeps the search tractable on
  large diagrams.
- **Convention note in code** (`:490-502`): explicitly documents that the correct
  index order is `g_zero_mask[(pi, ri)]` (phys-row, resp-col) and warns that the
  historically-buggy `[ri, pi]` order happened to work for diagonal couplings but
  silently filtered out valid diagrams for off-diagonal ones (e.g. GTaS).

#### Parallel convenience layer

##### `_worker_enumerate_one_prediagram(pd_idx)` — `type_assignment.py:565-579`

Module-level worker entry point (must be module-level so `multiprocessing` can
pickle it). Reads its inputs from the module global `_ENUM_WORKER_STATE`
(populated by the parent *before* forking, so children inherit it — not pickled
via `initargs`), runs `enumerate_typed_diagrams` on prediagram `pd_idx`, and
returns the list of `TypedDiagram`s.

##### `enumerate_all(prediagrams, external_fields, vertex_types, source_types, G_ft, resp_index, phys_index, parallel=False, n_workers=None, start_method='fork')` — `type_assignment.py:582-684`

The cross-prediagram driver. Concatenates the per-prediagram results in
prediagram-index order — **bit-identical between serial and parallel paths**
(pinned by `test_enumerate_all_parallel_matches_serial`).

- **Fork-safety gate** (`:630-634`): if `parallel` and `>1` prediagram and
  `fork_unsafe_in_notebook(start_method)`, warn once and force `parallel = False`.
- **Serial path** (`:636-646`): default. Iterate prediagrams, extend the result
  list.
- **Parallel path** (`:648-684`): only for a *legitimate* fork (the notebook case
  already fell back). Sets `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` via
  `os.environ.setdefault` (needed so objc doesn't abort the legitimate fork in
  scripts), populates `_ENUM_WORKER_STATE` **before** forking, gets a fork context
  `mp.get_context('fork')`, caps workers at `min(cpu_count, len(prediagrams))`,
  and `Pool.map`s the worker over prediagram indices. `Pool.map` preserves input
  order so the flattened output matches the serial order. Only `'fork'` is
  supported (`'spawn'` would require pickling Sage graph/matrix inputs).

### Phase F — `msrjd/diagrams/causality.py`

#### `check_pole_structure(pole_vals, omega=None)` — `causality.py:19-94`

```python
def check_pole_structure(pole_vals, omega=None):
```

- **Takes:** `pole_vals` a list of pole locations (`SR` expressions or numbers);
  `omega` unused (kept for interface symmetry).
- **Returns:** `(passed: bool, details: str, conditions: list)`. `passed` is
  `True` if all poles are verified retarded (`Im > 0`), or if the check is
  *inconclusive* but no pole definitively fails (in which case `conditions`
  carries the symbolic `Im(pole) > 0` constraints that must hold). `False` if any
  pole has `Im ≤ 0` or `Im == 0`.
- **Steps:** empty list → trivially causal. For each pole: coerce with `SR(pole)`,
  take `imag_part`, try `simplify_full`. Then attempt sign decisions in order via
  `bool(im_simplified > 0)` → pass; `bool(... <= 0)` → fail; `bool(... == 0)` →
  fail (a pole on the real axis is not retarded). Each `bool(...)` is wrapped in
  `try/except (TypeError, ValueError)` because Sage raises when it can't decide a
  symbolic inequality; an undecidable pole is recorded as **conditional**. Final
  verdict: any `failed` → `False`; else any `conditional` → `True` with the
  symbolic conditions; else all-pass.

#### `check_causality(typed_diagram, pole_vals=None)` — `causality.py:97-123`

```python
def check_causality(typed_diagram, pole_vals=None):
```

- **Takes:** a `TypedDiagram`; optional `pole_vals`.
- **Returns:** `(passed, details, conditions)`.
- **Steps:** pull `D = typed_diagram.prediagram[0]`. **Structural check:** if
  `not D.is_directed_acyclic()` → reject (`'directed cycle — acausal'`). If
  `pole_vals` is supplied and nonempty, defer to `check_pole_structure`.
  Otherwise pass with "Structural DAG check passed." In the current pipeline no
  `pole_vals` are passed, so the **DAG check is the operative gate**.

#### `filter_causal(typed_diagrams, pole_vals=None)` — `causality.py:126-152`

```python
def filter_causal(typed_diagrams, pole_vals=None):
```

- **Takes:** a list of `TypedDiagram`; optional `pole_vals`.
- **Returns:** `(kept, n_discarded, discarded_details)` — the surviving list, the
  count removed, and one human-readable reason string per discarded diagram.
- **Steps:** run `check_causality` on each; partition into `kept` /
  `discarded_details`. This is the function the orchestrator calls
  (`pipeline/_diagrams.py:185`).

### Phase G — `msrjd/diagrams/symmetry.py`

#### `_vertex_combinatorial_factor(vertex, typed_diagram)` — `symmetry.py:51-112`

Computes the per-vertex factor `M_v` of §2.6 (the older Path-B count). Collects
the `(resp_leg, phys_leg)` pairing of every outgoing edge from `vertex`; tallies
`resp_counts` (`n_r`), `pair_counts` (`n[r][t]`), `target_counts` (`m_t`) with
`Counter`; computes `M_v = [∏_r n_r!/∏_t n[r][t]!] × [∏_t m_t!]`. Returns ≥ 1.
External leaves are treated as unique targets (distinguished by leaf position);
internal targets by `(resp, phys)` pairing. *Not used by the production
`combinatorial_factor`*, which uses the Aut-based Path A; retained as the
per-vertex semantic referenced by `_leg_matchings`.

#### `_wick_leg_factor(typed_diagram)` — `symmetry.py:115-146`

The **numerator** of the Path-A symmetry factor. For each assigned vertex, tally
`Counter(vtype.response_legs)` and (if present) `Counter(vtype.physical_legs)`;
multiply in `factorial(c)` for each leg-type count `c`. Returns
`∏_v [(∏_r n_{v,resp_r}!) × (∏_p n_{v,phys_p}!)]`.

#### `_colored_incidence_digraph(typed_diagram, fix_external=True)` — `symmetry.py:149-244`

Builds the **coloured bipartite incidence digraph** that is fed to nauty. This is
the central construction that makes the symmetry factor and the dedup signature
both correct. Returns `(D, partition, vertex_label_map, color_groups)`.

- **Why bipartite "vertex-nodes + edge-nodes":** a plain `DiGraph` can't directly
  colour edges, and Sage/nauty colour *vertices*, not edges. So each Feynman
  vertex becomes a node `('V', v)` and **each propagator edge becomes its own
  node** `('E', ek)`. An original edge `u → v` is encoded as two bipartite edges
  `('V', u) → ('E', ek)` and `('E', ek) → ('V', v)`. Now an edge's *colour* lives
  on its edge-node, and nauty can see it.
- **Colouring (the `partition`):**
  - Internal vertex node: colour
    `('vertex', type(vt).__name__, str(vt.coefficient), vt.bigrade)` — so only
    same-type, same-coefficient, same-bigrade vertices may be swapped.
  - External leaf node: if `fix_external=True`, colour `('leaf', v, field)` (own
    colour per leaf position → leaves cannot move → `Aut_fixed_ext`); if
    `fix_external=False`, colour `('leaf', field)` (same-field leaves share a
    colour → swappable → `Aut_free_ext`).
  - Edge node: colour `('edge', resp_leg, phys_leg, prop)` where `prop` is the
    `(pi/ri)` propagator-index tuple — so an automorphism must preserve the
    propagator type and direction on every edge.
- `partition = list(color_groups.values())` is the list of colour classes, ready
  for `D.automorphism_group(partition=...)` / `D.canonical_label(partition=...)`.
  Parallel edges between the same pair with identical colour become genuinely
  swappable, which is exactly the symmetry the Feynman rules want counted.

#### `vertex_role_signature(vertex, typed_diagram)` — `symmetry.py:247-325`

A hashable signature for an *internal* vertex that is invariant under graph
relabelling — two vertices in the same Aut-orbit get the same signature. It
encodes: the vertex's type tag `(class, coefficient, bigrade)`, its
external-leaf attachment pattern (sorted multiset of `(field, prop)` for incident
leaves), and its internal-edge incidence pattern (multisets of `('out'/'in',
other-vertex-type-tag, count, sorted prop-indices)` grouped by the other
endpoint so multi-edges collapse correctly). Leaf vertices return a unique
`('leaf', vertex, field)` tag. **History note:** Phase J once used a `∏N!` over
leaves grouped by this signature as the external-Wick divisor; that
depth-1 heuristic over-divided when two structurally-distinct vertices shared a
signature, and has been **superseded** by the exact `external_wick_compensation`
(see §7). The function survives but the production divisor no longer uses it.

#### `_automorphism_order(typed_diagram, fix_external=True)` — `symmetry.py:328-340`

Builds the coloured incidence digraph and returns
`int(D.automorphism_group(partition=partition).order())` — i.e. `|Aut_fixed_ext|`
when `fix_external=True`, `|Aut_free_ext|` when `False`. This is the single point
where nauty's automorphism counting is invoked.

#### `external_wick_compensation(typed_diagram)` — `symmetry.py:343-400`

```python
def external_wick_compensation(typed_diagram):
```

- **Takes:** a `TypedDiagram`.
- **Returns:** the integer index `|Aut_free| / |Aut_fixed|` (always ≥ 1; equals 1
  when no leaf permutation is an automorphism, e.g. all-distinct external fields).
- **Steps:** compute `aut_free = _automorphism_order(..., fix_external=False)` and
  `aut_fixed = _automorphism_order(..., fix_external=True)`. Since `Aut_fixed` is
  a subgroup of `Aut_free`, `aut_free % aut_fixed` must be 0; **if not, it raises
  `RuntimeError`** rather than silently mis-weighting (`:391-399`) — deliberately
  fail-loud, since silent mis-weighting is the exact bug class this function
  exists to fix. Used by the integrators to divide their external-leaf mapping
  sum (see §2.5).

#### `combinatorial_factor(typed_diagram)` — `symmetry.py:403-446`

```python
def combinatorial_factor(typed_diagram):
```

- **Returns:** the integer symmetry factor `𝒮(Γ) = numer // aut`, where
  `numer = _wick_leg_factor(td)` and `aut = _automorphism_order(td,
  fix_external=True)`. Defensive fallback to `numer // max(aut, 1)` if a Sage
  edge case ever makes `numer` not divisible by `aut` (it should always divide —
  finite group acting freely on a finite set). This is **the** production
  symmetry factor.

#### `compute_all_combinatorial_factors(typed_diagrams)` — `symmetry.py:449-462`

`[combinatorial_factor(td) for td in typed_diagrams]` — vectorized convenience.

#### `diagram_signature(td)` — `symmetry.py:467-524`

The **dedup key**: a hashable canonical signature such that two typed diagrams
get the same signature **iff** they are isomorphic coloured graphs (with
`fix_external=False`, so same-field leaves may permute).

- **Steps:** build the incidence digraph with `fix_external=False`; sort colour
  classes by `str(key)` for a label-independent, deterministic cell order; call
  `D.canonical_label(partition=partition, certificate=True)` → `(C, cert)`. Build
  `cells` = for each colour key, the sorted tuple of canonical ids
  `cert[v]` its members map to (this is essential — the canonical edge list alone
  would confuse two graphs of the same shape but different colourings). Return
  `(tuple(str(k) for k in keys), cells, tuple(sorted(C.edges(labels=False))))`.
- **Why it must be a *complete* invariant:** the dedup multiplicity is **not**
  multiplied back into the weight (Path-A `𝒮(Γ)` already carries each class's full
  weight). So if the signature ever *collided* two non-isomorphic classes, one
  class's integral would be silently dropped. The history note (`:484-498`)
  records exactly such a past bug (the old hand-rolled depth-1 invariant collided
  4 of 11 a³-sector classes at k=3, giving `κ₃ = −68/3 a³` instead of `−32 a³`); a
  canonical-form invariant cannot collide, closing that failure mode for all
  `(k, ℓ, theory)`.

#### `deduplicate_typed_diagrams(typed_diagrams)` — `symmetry.py:527-550`

Thin wrapper: returns just the `unique` list from
`deduplicate_with_multiplicities`.

#### `deduplicate_with_multiplicities(typed_diagrams)` — `symmetry.py:553-593`

```python
def deduplicate_with_multiplicities(typed_diagrams):
```

- **Returns:** `(unique, multiplicities)` — one representative per signature in
  first-seen order, and the equivalence-class size of each. Implementation: a
  `sig_to_idx` dict; first time a signature is seen, append the diagram and a
  multiplicity of 1; thereafter increment the matching multiplicity.
- **Important:** `multiplicities` is **diagnostic only** under Path A — the weight
  is carried entirely by `combinatorial_factor`, so multiplying by class size
  would double-count (the docstring `:556-568` is explicit). This is the function
  the orchestrator actually calls (`pipeline/_diagrams.py:186`).

#### `_symbols_matching_prefixes(expr, prefixes)` — `symmetry.py:598-610`

Returns the set of free `SR` variables in `expr` whose name starts with any given
prefix (via `expr.variables()`). Used by `classify_coefficient_factors` to find
the time-dependent symbols in a coefficient.

#### `_is_source_type(vtype)` — `symmetry.py:613-615`

`return not hasattr(vtype, 'physical_legs')` — distinguishes a `SourceType` (no
physical legs) from a `VertexType`.

#### `classify_coefficient_factors(typed_diagram, time_dep_params=None, noise_structure=None)` — `symmetry.py:618-698`

```python
def classify_coefficient_factors(typed_diagram, time_dep_params=None,
                                 noise_structure=None):
```

- **Purpose:** split each vertex's coefficient into a **scalar prefactor** (can be
  pulled outside the integral) vs **factors that must stay inside** (time-dependent
  couplings, non-white noise amplitudes). Consumed by the integrators.
- **Returns:** a dict with keys `'Scal'` (= `combinatorial_factor(td)`),
  `'scalar_prefactor'` (the product of all pull-out-able pieces, including
  `𝒮(Γ)`), `'vertex_time_factors'` (`{vertex: time-dependent SR factor}`),
  `'source_time_info'` (per-source metadata: legs, temporal type, amplitude,
  whether it stays in the integrand), and `'is_stationary'` (bool).
- **Steps (high level):** start `scalar_parts = [SR(Scal)]`. For each vertex,
  take `coeff = -SR(vtype.coefficient)` (the minus from `Z = ∫ exp(-S)`, so each
  `S_V` factor acquires a sign). For **sources**, white noise with a
  time-independent amplitude can be pulled out; coloured/time-dependent amplitudes
  stay inside, recorded in `source_time_info`. For **interaction vertices**, split
  off the constant part (substitute time-dep symbols → 1) into `scalar_parts` and
  store the residual `td_part = coeff / const_part` (simplified with
  `simplify_rational`) in `vertex_time_factors`. Finally
  `scalar_prefactor = reduce(mul, scalar_parts, SR(1))` and compute
  `is_stationary` (no vertex time factors and no time-dependent / non-standard
  noise amplitudes).

---

## 5. Data structures

### The prediagram 4-tuple `(D, G, leaves, internal)`

Produced upstream (Phase C, `loop_diagram_enumeration.py`), consumed by every
function here.

- `D` — a Sage **`DiGraph`** (directed, possibly with multi-edges and self-loops).
  Vertices are small integer ids; edges are 3-tuples `(u, v, label)` (the label
  disambiguates parallel edges). This is the *oriented* prediagram.
- `G` — the underlying **undirected** graph (the un-oriented topology from which
  `D` was obtained by `enumerate_orientations`). Carried along but not used by
  this subsystem directly.
- `leaves` — a list of vertex ids that are external legs (degree-1 "stubs"). Their
  index order is the canonical position order used to line up with
  `external_fields`.
- `internal` — list of non-leaf vertex ids.

### Field / leg tuple `(field_base, pop_idx)`

A single field-leg identity, e.g. `('nt', 1)` or `('dn', 2)`. `field_base` is the
generator base name (letters), `pop_idx` is the 1-based population index parsed
off the trailing digits by `_parse_field_name`. Index `0` means "no trailing
digits." Response and physical fields are distinguished by *which generator
list* the base name came from (the first `n_tilde` generators are response).

### `VertexType` (defined in `core/vertices.py:80-131`)

One monomial of the **interacting action** (total degree ≥ 3, `n_phys ≥ 1`).
`__slots__` = `coefficient` (`SR`), `response_legs` (list of `(base, idx)`),
`physical_legs` (list of `(base, idx)`), `bigrade` (`(n_tilde, n_phys)`).
Properties `in_degree = len(physical_legs)`, `out_degree = len(response_legs)`.
Subclass `ConvVertexType` adds `kernel_attachments` for conductance-style
synaptic-kernel vertices.

### `SourceType` (defined in `core/vertices.py:216-253`)

One monomial of the **noise kernel** (`n_tilde ≥ 2`, `n_phys = 0` — a pure
response-field source). `__slots__` = `coefficient`, `response_legs`, `bigrade`.
Property `out_degree = len(response_legs)`. **Has no `physical_legs`** — this is
how `_is_source_type` and `_leg_matchings` distinguish sources from interactions.
Subclass `NoiseSourceType` adds `cumulant_specs` for non-local correlated noise.

### `TypedDiagram` (defined in `type_assignment.py:29-71`)

The Phase-E output. `__slots__`:

- `prediagram` — the `(D, G, leaves, internal)` 4-tuple this was built from.
- `vertex_assignments` — `{vertex_id: VertexType | SourceType}` for internal
  vertices (empty for the no-internal case).
- `edge_types` — `{(u, v, label): (resp_leg, phys_leg)}`; 3-tuple edge keys keep
  multi-edges distinct. Each value is a pair of `(field_base, pop_idx)` legs.
- `external_legs` — `{leaf_vertex: (field_base, pop_idx)}`.
- `propagator_indices` — `{(u, v, label): (resp_matrix_idx, phys_matrix_idx)}`,
  i.e. `(ri, pi)` — the row/col into `G_ft`. (Stored as `(ri, pi)`; the *lookup*
  into `g_zero_mask`/`G_ft` uses `(pi, ri)` — see §7.)

### The propagator zero mask `g_zero_mask`

`{(i, j): bool}` over the full `G_ft` grid; `True` means "entry `G_ft[i, j]` is
the literal `SR(0)`." Computed once per prediagram in `enumerate_typed_diagrams`,
threaded through the backtracker. Indexed as `g_zero_mask[(pi, ri)]`.

### The dedup signature (`diagram_signature` return)

A 3-tuple `(colour_key_strings, colour_cells, canonical_edges)`, fully hashable,
used as a dict key for dedup. See `diagram_signature` in §4.

---

## 6. Data flow (concrete)

### Inputs arriving at the subsystem

From the orchestrator `enumerate_unique_diagrams` (`pipeline/_diagrams.py`):

- `prediagrams` — list of `(D, G, leaves, internal)` from
  `enumerate_prediagrams_all(k=k, ell=ell)`.
- `external_fields` — length-`k` list like `[('nt', 1), ('dn', 1)]`.
- `vtypes`, `stypes` — `VertexType` / `SourceType` lists from `extract_*_types`.
- `G_ft`, `resp_idx`, `phys_idx` — the propagator matrix and its index maps
  (`build_field_index_map`).

### The chain, with what flows between stages

```
prediagrams ──[filter_prediagrams]──► degree-valid prediagrams        (Phase D, cheap pre-screen)
            ──[enumerate_all_typed]──► list[TypedDiagram]  (typed)     (Phase E)
            ──[filter_causal]────────► list[TypedDiagram]  (causal)    (Phase F, DAG check)
            ──[deduplicate_with_multiplicities]──► (unique, multiplicities)   (Phase G)
```

A concrete worked micro-example (from `tests/test_type_assignment.py`):

```python
# A single edge connecting two external leaves (a bare propagator):
pd = (DiGraph([(1, 2)]), None, [1, 2], [])
external_fields = [('nt', 1), ('dn', 1)]
results = list(enumerate_typed_diagrams(pd, external_fields, [], [],
                                        G_ft, resp_idx, phys_idx))
# -> one TypedDiagram with:
#    edge_types         = {(1, 2, None): (('nt',1), ('dn',1))}
#    external_legs      = {1: ('nt',1), 2: ('dn',1)}
#    propagator_indices = {(1, 2, None): (ri, pi)}
#    vertex_assignments = {}
```

### Outputs leaving the subsystem

`enumerate_unique_diagrams` returns `(unique_by_ell, multiplicity_by_ell,
all_unique)`. The integrators (Phase J) then, per `TypedDiagram`:

1. read `vertex_assignments` for coefficients (split via
   `classify_coefficient_factors`),
2. read `edge_types` + `propagator_indices` to place propagators
   `G_ft[pi, ri]` on edges,
3. read `external_legs` to line leaves up with `external_fields`,
4. multiply by `𝒮(Γ) = combinatorial_factor(td)`,
5. sum over external-leaf mappings, dividing by
   `external_wick_compensation(td)`,
6. integrate over internal vertex times / frequencies / momenta.

---

## 7. Gotchas & caveats

1. **Propagator index order is `(pi, ri)`, not `(ri, pi)`.** `G_ft = K_ft^{-1}`
   has rows = physical, cols = response, so the edge propagator
   `⟨φ_phys ñ_resp⟩ = G_ft[pi, ri]`. The code stores `propagator_indices =
   (ri, pi)` but **looks up** the mask/matrix with `(pi, ri)`. The comment at
   `type_assignment.py:490-502` records a historical bug where `[ri, pi]` was used
   for the lookup: it worked for *diagonal* couplings (where the indices coincide)
   but silently filtered out valid diagrams for *off-diagonal* couplings (e.g.
   GTaS `m̃ → δn`, phys idx 0, resp idx 4). Any new propagator-touching code must
   index `(pi, ri)`.

2. **The zero-check is intentionally a *string* test, not `is_zero()`.**
   `g_zero_mask[(i,j)] = (str(G_ft[i,j]) == '0')` (`type_assignment.py:165`). The
   semantically-correct `SR(G_ft[pi,ri]).is_zero()` triggers full Sage symbolic
   simplification on long rational propagators — multiple seconds per entry. The
   string test catches **only literal `SR(0)`** cells (what `build_propagator`
   uses for empty cells). A mathematically-zero-after-cancellation entry is *not*
   caught, so an extra diagram or two may be generated that simply integrate to 0
   downstream — harmless over-generation, never a wrong number. Be aware the typed
   set can therefore contain a few "vanishing" diagrams.

3. **`_leg_matchings` emits the canonical leg ordering only.** It uses SymPy's
   `multiset_permutations` (the `N!/∏n_r!` distinct orderings), *not* all `N!`.
   This is correct only because the integrator multiplies by `𝒮(Γ)` afterward
   (the over-count factor it no longer generates is exactly `𝒮(Γ)`). If you ever
   change how the symmetry factor is applied, revisit this. Pinned by
   `test_leg_matchings_canonical` /
   `test_enumerate_typed_signatures_match_pre_change`.

4. **Dedup multiplicity is diagnostic only — never multiply it back.** Under
   Path A the entire weight is in `combinatorial_factor`. Multiplying by the
   `deduplicate_with_multiplicities` class size would double-count
   (`symmetry.py:556-568`). The cache version was bumped (`unique_typed` →
   `unique_typed_mult_v3`) specifically to invalidate caches that predate the
   complete-invariant signature, because loading them would resurrect the
   class-collision bug.

5. **`diagram_signature` MUST stay a complete isomorphism invariant.** The old
   hand-rolled depth-1 invariant collided non-isomorphic classes (4 of 11
   a³-sector classes at k=3), silently dropping integrals (`κ₃ = −68/3 a³` instead
   of `−32 a³`). The current canonical-form-based signature cannot collide. Don't
   "optimize" it back to a shallow heuristic.

6. **`external_wick_compensation` fails loud on a non-divisor.** If
   `|Aut_free| % |Aut_fixed| != 0` it raises `RuntimeError` rather than guessing
   (`symmetry.py:391-399`). `combinatorial_factor`, by contrast, falls back to
   floor-division (`symmetry.py:440-445`) — a deliberate asymmetry, since a wrong
   compensation silently mis-weights whereas the Wick factor's divisibility is a
   theorem that "should always hold."

7. **`vertex_role_signature` is superseded for weighting.** A prior Phase-J
   external divisor used `∏N!` over leaves grouped by this depth-1 signature; it
   over-divided (×⅓/×½ deficits on the k=4 OU+εx³ 1-loop cascades) because two
   structurally-distinct vertices can share a role signature without any
   automorphism relating them. The exact `external_wick_compensation` replaced it.
   The function is kept but the production weight path no longer uses it — don't
   reintroduce the heuristic divisor (see `final_integral.py:2560-2586`).

8. **The fork path is OFF by default and unsafe in notebooks.** `parallel=False`
   is the default (`type_assignment.py:584`). Even when `parallel=True`,
   `fork_unsafe_in_notebook` forces serial inside a macOS Jupyter kernel —
   because fork-after-Cocoa/BLAS-init can hard-crash the kernel *and the OS*, and
   `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` makes it worse. The guard's Jupyter
   detection hinges on the IPython shell class being exactly
   `ZMQInteractiveShell`; terminal IPython, pytest, plain scripts, and Linux are
   all (correctly) treated as fork-safe. Because pytest is *not* a ZMQ kernel, the
   parallel-vs-serial bit-identity tests run on the fork path and therefore never
   certified notebook safety — only the guard protects the notebook.

9. **`enumerate_all`'s `_ENUM_WORKER_STATE` is a module global and not
   thread-safe.** It is populated *before* the fork so children inherit it. Do not
   call `enumerate_all(parallel=True)` from two threads concurrently
   (`type_assignment.py:657-660`). Only `start_method='fork'` is supported;
   `'spawn'` would need to pickle Sage graphs/matrices and is not guaranteed to
   work.

10. **`check_pole_structure` returns `passed=True` on inconclusive symbolic
    poles** (recording them as `conditions`). A pole exactly on the real axis
    (`Im == 0`) is treated as a *failure*, not a conditional. In the current
    pipeline no `pole_vals` are passed to `filter_causal`, so only the structural
    DAG check actually gates diagrams; the pole machinery is latent.

11. **`filter_prediagrams` (Phase D) is not in the production orchestrator.** The
    loop enumerator already produces only degree-valid prediagrams for a given
    `(k, ℓ)`, so `enumerate_unique_diagrams` skips the explicit filter. The filter
    is still the canonical cheap pre-screen (and is unit-tested), but a reader
    tracing the live pipeline won't see it called — it's exercised by tests and by
    looser prediagram producers.

---

## 8. Glossary

- **MSR-JD** — Martin–Siggia–Rose / Janssen–De Dominicis. The field-theory
  rewriting of a Langevin SDE as a path integral over a physical field `φ` and a
  conjugate **response field** `ñ`.
- **Response field (`ñ`, "tilde")** — the auxiliary conjugate field of MSR-JD; in
  the code its generator base names carry the `d` prefix (`dv`, `dn`, `dx`).
  Outgoing edge legs are response legs.
- **Physical field (`φ`)** — the actual dynamical field; base names like `vt`,
  `nt`, `xt`. Incoming edge legs are physical legs.
- **Prediagram** — a bare directed (multi)graph with `k` external leaves and `ℓ`
  loops, *before* any field types are assigned. Carried as
  `(D, G, leaves, internal)`.
- **Typed diagram** — a prediagram fully decorated with vertex types, external
  fields, and per-edge `(resp_leg, phys_leg)` propagator types. A `TypedDiagram`.
- **Leg** — one field "stub" at a vertex, identified by `(field_base, pop_idx)`.
  Response legs feed outgoing edges; physical legs feed incoming edges.
- **Bigrade `(n_tilde, n_phys)`** — the count of response and physical legs of a
  vertex/monomial. A noise source has `n_phys = 0`; an interaction vertex has
  `n_phys ≥ 1` and total degree ≥ 3.
- **Source / source vertex** — a pure response-field monomial from the noise
  kernel (in-degree 0; a `SourceType`). It injects noise.
- **Interaction vertex** — a higher-order monomial of the action (`VertexType`).
- **Propagator `G_{ij}`** — a two-point function on an edge; here the retarded
  response `⟨φ_phys ñ_resp⟩`, read from `G_ft[pi, ri]` (phys-row, resp-col).
- **`G_ft = K_ft^{-1}`** — the propagator matrix, the inverse of the quadratic
  (Gaussian) form `K`. Rows index physical fields, columns index response fields.
- **Retarded / causal** — a propagator nonzero only for `t > 0`; equivalently all
  poles in the upper-half `ω`-plane; structurally, the diagram is a DAG.
- **DAG** — Directed Acyclic Graph (no directed cycle).
- **Symmetry / combinatorial factor `𝒮(Γ)`** — the rational/integer Feynman-rule
  multiplicity, `(∏ Wick leg factorials) / |Aut_fixed_ext(Γ)|`.
- **Automorphism group `Aut(Γ)`** — the relabellings of a (coloured) graph onto
  itself. `Aut_fixed_ext` pins external leaves; `Aut_free_ext` lets same-field
  leaves permute.
- **Orbit–stabilizer theorem** — group-theory fact used to justify the external
  Wick compensation: `|orbit| = |group| / |stabilizer|`.
- **External Wick compensation** — `|Aut_free| / |Aut_fixed|`, the divisor the
  integrator applies to its external-leaf mapping sum.
- **Incidence digraph (coloured)** — the bipartite "vertex-nodes + edge-nodes"
  graph built so nauty can colour edges (as nodes) and compute automorphisms /
  canonical forms.
- **Canonical label / certificate** — nauty's normal-form relabelling (`C`) and
  the original→canonical id map (`cert`); equal canonical forms ⇔ isomorphic.
- **SageMath / `SR`** — the math system; `SR` is its symbolic-expression ring.
- **nauty** — McKay's C library for graph canonical labelling and automorphism
  groups, wrapped by Sage's `automorphism_group` / `canonical_label`.
- **`multiset_permutations` (SymPy)** — yields the `N!/∏n_r!` distinct orderings
  of a multiset.
- **Partition (nauty/Sage sense)** — a vertex colouring passed to the graph
  algorithms; vertices may only map within the same colour class.
- **Fork (POSIX) / fork start method** — cloning a process; workers inherit
  parent memory (no input pickling). Lethal after Cocoa/BLAS init in a macOS
  Jupyter kernel.
- **`ZMQInteractiveShell`** — the IPython shell class used only by a ZMQ/Jupyter
  kernel; the fork-safety guard's notebook fingerprint.
- **`pop_idx` (population index)** — the 1-based subscript on a field generator
  (e.g. the `2` in `nt2`), distinguishing populations/components.

---

## 9. Proposed manual subsections

1. **Where typed diagrams come from** — the Phase B→C→D→E→F→G arc; what a
   prediagram is and what a typed diagram is.
2. **The response/physical edge convention** — `u → v` carries
   `⟨φ_head ñ_tail⟩`; the `G_ft[pi, ri]` index order and why it matters.
3. **Phase D — the degree filter** — `available_degrees`,
   `classify_prediagram_vertices`, `filter_prediagrams`; the cheap pre-screen.
4. **Phase E — type assignment as constraint satisfaction** — candidate types,
   external-field permutations, leg matchings, the backtracker, early propagator
   pruning.
5. **The canonical-leg-matching optimization** — multiset vs. `N!` permutations;
   the proof of bit-identical output.
6. **The propagator zero mask** — why a structural string test, not `is_zero()`.
7. **Phase F — causality** — retarded poles, the DAG check, the latent
   pole-structure machinery.
8. **Phase G — the symmetry factor** — Wick numerator, the coloured incidence
   digraph, nauty automorphism orders, `combinatorial_factor`.
9. **Deduplication and the complete invariant** — `diagram_signature`, canonical
   labelling, the class-collision bug it closed, multiplicity-is-diagnostic.
10. **External Wick compensation and the integrator hand-off** — fixed vs. free
    automorphisms, orbit–stabilizer, how the integrator combines `𝒮(Γ)`, `comp`,
    coefficients, and the mapping sum.
11. **Coefficient classification** — pull-out scalars vs. integrand factors,
    stationarity.
12. **Parallelism and fork-safety** — the embarrassingly-parallel structure, why
    fork on macOS Jupyter is lethal, the `fork_safety` guard, serial fallback.
13. **External tools primer** — Sage/`SR`, nauty (via Sage graphs), SymPy
    `multiset_permutations`, stdlib pieces.
14. **Caching and cache invalidation** — `enumerate_unique_diagrams`, the
    `unique_typed_mult_v3` stage tag, version-bump history.
15. **Gotchas reference** — consolidated list of the index-order, string-zero,
    multiplicity, fail-loud, and fork caveats.

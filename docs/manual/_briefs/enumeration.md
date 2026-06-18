# Diagram Enumeration: Prediagrams, nauty, loop-order bound

*Subsystem slug: `enumeration`. Source of truth read in full:*
*`msrjd/enumeration/loop_diagram_enumeration.py` and `msrjd/enumeration/degree_scan.py`.*

---

## Overview

### What this subsystem does, in one sentence

Given two integers — `k` (we want a `k`-point correlation/cumulant function) and
`ell` (the loop order of the perturbative correction we are computing) — this
subsystem produces the **complete, deduplicated list of all bare graph
topologies** that could possibly contribute to that correlation function at that
loop order, *before any physical field labels are attached*. Those bare graphs
are called **prediagrams**.

### Why "pre"-diagrams?

In MSR-JD (Martin–Siggia–Rose–Janssen–De Dominicis) field theory, a genuine
Feynman diagram is a graph where every edge is a specific *propagator* (e.g. a
response/physical propagator `⟨φ φ̃⟩` versus a correlation propagator `⟨φ φ⟩`)
and every internal vertex is a specific *interaction monomial* from the action
(e.g. the cubic vertex `φ̃ φ²`). A *prediagram* strips all of that away. It keeps
only:

- the **shape** of the graph (which vertices connect to which),
- the **distinction leaf-vs-internal** (which vertices are external legs of the
  correlation function, and which are internal interaction points), and
- an **orientation** on every edge (an arrow `u → v`), because MSR-JD propagators
  are causal/directed — a response propagator flows in a definite time direction.

So a prediagram is "a Feynman diagram with the physics not yet filled in." This
subsystem answers the purely *combinatorial/topological* question — *what shapes
are even possible?* — and leaves the *physical* question — *which of those shapes
the theory actually populates, and with what propagators and vertices* — to the
next subsystem.

### Where it sits in the end-to-end pipeline

The project is organized into lettered "build phases" (see
`docs/BUILD_PHASE_OUTLINES.md`). This subsystem is the topological generator that
feeds the typing machinery:

```
  Phase 1 (FieldTheory.expand)        Phase 2 (THIS SUBSYSTEM)         Phase D/E
  ───────────────────────────        ────────────────────────         ──────────
  Action → Taylor-expanded            enumerate_prediagrams(k, ell)    filter_prediagrams
  vertices + noise kernel,        ──► trees → topologies →         ──► + type_assignment
  saved to disk as a "theory"         prediagrams (D,G,leaves,internal)   .enumerate_typed_diagrams
                                            │
                                            ▼
                                   Phase C: degree_scan.ensure_taylor_order
                                   (feedback: did Phase 1 expand far enough?)
```

- **What feeds it:** essentially nothing but the two integers `k` and `ell`. The
  topology generation is *theory-agnostic* — it does not look at the action at
  all. (The theory only enters later, when prediagram vertices get matched to
  interaction monomials.)
- **What consumes its output:**
  1. **`msrjd/diagrams/type_assignment.py`** is the direct downstream consumer.
     Its `enumerate_typed_diagrams(prediagram, ...)` takes one prediagram tuple
     `(D, G, leaves, internal)` and produces every valid fully-typed
     `TypedDiagram` for it (see `type_assignment.py:115`,
     `type_assignment.py:140` where it unpacks `D, G_graph, leaves, internal = prediagram`).
  2. **`msrjd/enumeration/degree_scan.py`** (Build Phase C) scans the same
     prediagram tuples for their maximum vertex degree, and uses that to decide
     whether the saved theory (Phase 1) was Taylor-expanded to a high enough
     order. If a prediagram has a degree-5 internal vertex but the theory only
     expanded the action to cubic order, the theory has no degree-5 vertex to
     offer and must be re-expanded. This is the "bridge" that closes the loop
     between the *topological* world (this subsystem) and the *algebraic* world
     (Phase 1).

### The three-stage refinement

The generator works in three successive refinements, each one a *filter +
deduplicate* on the previous:

1. **Trees** — connected acyclic graphs with the right number of leaves. This is
   the "spanning skeleton" of every possible diagram. Trees have zero loops, so
   to reach `ell` loops we will later add `ell` extra edges.
2. **Topologies** — undirected multigraphs obtained by adding exactly `ell` extra
   edges to a tree (creating exactly `ell` independent cycles = loops), then
   keeping only the ones that satisfy MSR-JD's structural constraints, then
   removing isomorphic duplicates.
3. **Prediagrams** — topologies with every edge given an arrow (orientation),
   keeping only orientations that are physically admissible (acyclic, correct
   source/sink structure), then removing isomorphic duplicates again.

The public entry points (`enumerate_all`, `enumerate_prediagrams`, etc.) wrap
these three stages.

---

## The math

This section builds the relevant graph theory and the famous completeness bound
from the ground up. A reader who knows MSR-JD field theory but not the
combinatorics should be able to follow every step.

### Vocabulary: graphs, degrees, leaves

A **graph** is a set of **vertices** (points) and **edges** (connections between
pairs of points). The **degree** of a vertex `v`, written `deg(v)`, is the number
of edge-ends touching it. (If two vertices are joined by a *double edge* — a
"multi-edge" — that contributes `2` to each of their degrees.)

We partition vertices by degree:

- **Leaves** = degree-1 vertices. In our setting these are the **external legs**
  of the correlation function. A `k`-point function has exactly `k` of them in
  the final diagram.
- **Internal vertices** = degree `≥ 2`. These are the interaction points.
- We further split internal vertices into **degree-2** vertices (`V₂`) and
  **degree-≥3** vertices (`V₃`, somewhat loosely written `degree_3plus` in the
  code).

In the code this partition is computed by `classify_vertices_sage`
(`loop_diagram_enumeration.py:32`), which returns the four lists
`(leaves, internal, degree_2, degree_3plus)`.

### Trees and the loop number

A **tree** is a connected graph with **no cycles** (no closed loops). The
defining counting identity of a tree is

```
    |E| = |V| − 1            (a tree on |V| vertices has exactly |V|−1 edges)
```

For *any* connected graph `G`, the number of **independent cycles** — equivalently
the **loop number** or **first Betti number** — is

```
    L(G) = |E| − |V| + 1
```

This is exactly `count_cycles_sage` (`loop_diagram_enumeration.py:68`):
`G.size() - G.order() + 1`, where in SageMath `size()` = number of edges `|E|`
and `order()` = number of vertices `|V|`. (The function returns `-1` as a sentinel
when the graph is disconnected, since the formula is only meaningful for connected
graphs.)

**Consequence used everywhere:** starting from a tree (`L = 0`) and adding `ell`
extra edges *without adding new vertices* raises `|E|` by `ell` and leaves `|V|`
unchanged, so it raises the loop number by exactly `ell`. This is the engine of
the whole topology stage: *take a tree, add `ell` edges, demand `L(G) = ell`*.

### Where the leaf count comes from: `k` and `j`

The final diagram must have exactly `k` external legs (leaves). But the
*intermediate trees* may temporarily have more than `k` leaves, because some
"extra" tree-leaves will be consumed when we add the loop edges (an added edge can
attach to a former leaf, raising its degree above 1 so it stops being a leaf).

The code calls this surplus `j`: a tree is generated with `k + j` leaves, where
`j ≥ 0` is the number of leaves that will be "eaten" by the `ell` added edges.

**How large can `j` be?** Each added edge has two endpoints, so `ell` added edges
provide `2·ell` endpoint-attachments. In the most leaf-consuming case, attaching
edges to leaves is what converts them to internal vertices. The code bounds

```
    j_max = ell + ell//2 = floor(3·ell / 2)
```

(`loop_diagram_enumeration.py:124`, `j_max = ell + (ell // 2)`). Intuitively this
is the maximum number of tree-leaves that the `ell` loop-edges can absorb across
all valid decompositions; the `3·ell/2` form is the integer ceiling on how many
degree-1 nodes the loop structure can "use up." For `ell = 1` this gives
`j_max = 1`; for `ell = 2`, `j_max = 3`; for `ell = 3`, `j_max = 4`.

### The degree constraints (why some trees are pruned)

Not every tree with `k + j` leaves can be the skeleton of a valid MSR-JD diagram.
Two structural bounds prune the search.

**(a) Bound on degree-≥3 vertices (`v3_max`).**

```
    |V₃| ≤ k + j − 2                         (loop_diagram_enumeration.py:128)
```

This is the standard bound for a tree with `k + j` leaves: a tree with `m` leaves
has at most `m − 2` vertices of degree ≥ 3 (the extreme case being a binary tree).
Here `m = k + j`, hence `v3_max = k + j − 2`.

**(b) Bound on degree-2 vertices (`v2_max`) — the headline completeness bound.**

This is the subtle one, the *enumeration completeness bound* referenced
throughout the project. The code (`loop_diagram_enumeration.py:139`) sets

```
    v2_max = 2·v3_max + 3·ell − k − 3·j + 3
```

and the comment (`loop_diagram_enumeration.py:129`) records that this algebraically
equals the **proven** bound on the number of degree-2 vertices of the *final
topology* `G`:

```
    |V₂ᵀ| ≤ k + 3·ell − j − 1
```

Let us verify the algebra. Substitute `v3_max = k + j − 2`:

```
    2·v3_max + 3·ell − k − 3·j + 3
  = 2·(k + j − 2) + 3·ell − k − 3·j + 3
  = 2k + 2j − 4 + 3·ell − k − 3·j + 3
  = k − j − 1 + 3·ell
  = k + 3·ell − j − 1.    ✓
```

So the code's `v2_max` is exactly the stated bound `k + 3·ell − j − 1`.

**Why is `k + 3·ell − j − 1` the right bound?** The MEMORY note
`project_enumeration_bound_fix.md` and the in-code comment summarize the proof
ingredients (full proof lives in the paper appendix):

1. a **degree-partition identity** (the handshake/incidence count relating leaves,
   `V₂`, `V₃`, edges, and loop number),
2. the **orientability constraint** `|V₂ᴳ| ≤ k + ell − 1` (a degree-2 internal
   vertex in an MSR-JD diagram must be orientable as one-in/one-out, and there is
   a hard ceiling on how many such vertices a valid causal orientation admits),
   and
3. an **incidence-counting** bound `θ ≤ 2·ell − j` (where `θ` counts certain
   doubled-edge / multi-edge incidences).

Combining these yields `|V₂ᵀ| ≤ k + 3·ell − j − 1`.

**The bug that was fixed (June 2026, commit `40454e7`).** An *earlier* version of
the code used a tighter bound with an extra `− ⌊j/3⌋` term:

```
    v2_max_OLD = 2·v3_max + 3·ell − k − 3·j − (j//3) + 3   # = k + 3·ell − j − 1 − ⌊j/3⌋
```

That extra tightening came from a claimed lemma `θ ≤ 2·ell − j − ⌊j/3⌋`, which is
**FALSE at `ell ≥ 3`**. The counterexample (recorded in the comment at
`loop_diagram_enumeration.py:133` and in `scratch/bound_check.py`): *three
doubled-edge bubbles on a hub*, `|V| = 8`, where every decomposition has `j = 3`
and `θ = 3 > 2·ell − j − ⌊j/3⌋`. The false lemma never actually *bound* at the
small orders that had been tested (slack ≥ 2 verified by exhaustive enumeration at
`(k,ell) ∈ {(2,1),(3,1),(2,2),(3,2),(4,1),(2,3)}`), so the *output was identical*
at all tested orders — but only the proven bound `k + 3·ell − j − 1` *guarantees
completeness* at `ell ≥ 3`. The lesson recorded in MEMORY: **the prune must follow
ONLY a proven bound; empirical slack at small orders is not a proof.**

### The MSR-JD structural constraints on topologies

Beyond the degree *counts*, the topology must satisfy *local* structural rules so
that it is reducible to a unique canonical diagram. These are enforced by
`check_topology_constraints` (`loop_diagram_enumeration.py:180`):

- **Correct leaf count:** `|leaves| == k` (the final topology must have exactly
  `k` external legs).
- **Connected:** `G.is_connected()` — a Feynman diagram contributing to a
  *connected* cumulant must be connected.
- **No two adjacent degree-2 vertices** (`has_adjacent_degree2_sage`): a chain of
  two degree-2 vertices in a row is a redundant subdivision of a single
  propagator line; only one canonical representative is kept.
- **Every degree-≥3 vertex has at least one non-degree-2 neighbor**
  (`check_deg3_has_non_deg2_neighbor`).
- **Leaf neighbors are not all degree-2** (`check_leaf_neighbors_not_all_deg2`):
  an external leg should not attach only through a chain of degree-2 subdivisions.

Crucially these three "degree-2 pruning" checks are *skipped at tree level*
(`ell == 0`), because then "the tree IS the topology — no contraction ambiguity"
(comment at `loop_diagram_enumeration.py:188`). The skip is gated by
`if ell is None or ell > 0:`.

### The orientation constraints (turning a topology into prediagrams)

An MSR-JD edge is a *directed* propagator. So each undirected topology must be
oriented. With `|E|` edges there are `2^{|E|}` raw orientations; the admissible
ones are selected by `check_orientation_constraints`
(`loop_diagram_enumeration.py:275`):

- **Acyclic:** `D.is_directed_acyclic()` — MSR-JD response propagators are causal,
  so a valid diagram has no directed cycle (you cannot return to your own past).
- **A degree-1 vertex (leaf) must be an *incoming* leg:** if `in_deg + out_deg == 1`
  then we require `in_deg == 1` (the external leg carries the field *in*; the arrow
  points *into* the leaf).
- **A sink with `in_deg ≥ 2` and `out_deg == 0` is forbidden** (`out_deg == 0 and
  in_deg ≥ 2`): a pure sink absorbing two or more lines is not an admissible
  MSR-JD vertex.
- **A pure pass-through degree-2 vertex (`in_deg == 1 and out_deg == 1`) is
  forbidden:** this would be a trivial line-subdivision again.
- **No two *sources* are adjacent:** a *source* is a vertex with `in_deg == 0`
  (and not a leaf). Two adjacent sources (`if any(u in sources for u in
  D.neighbors_out(v))`) is forbidden — sources represent noise-injection points,
  and two of them cannot be directly wired together.

The physical reading: **sources** (in-degree 0, not a leaf) are *noise-kernel
insertion points*; **interaction vertices** (both in- and out-edges) are
*action-vertex insertion points*; **leaves** are *external legs*. This source /
interaction / leaf trichotomy is exactly what `degree_scan.py` and
`type_assignment.py` re-derive downstream.

### Graph isomorphism: why dedup is needed and what it means (nauty intro)

After adding edges and after orienting, the same *abstract* graph gets produced
many times with different vertex *labels* (vertex `5` here is vertex `7` there).
Two graphs are **isomorphic** if one can be turned into the other by *relabeling
vertices* — i.e. there is a bijection of vertices that maps edges to edges. For
Feynman-diagram bookkeeping, isomorphic graphs are the *same diagram* and must be
counted once.

Deciding isomorphism naively means trying all `n!` relabelings — infeasible. The
classic engine that does this efficiently is **nauty** ("No AUTomorphisms, Yes?"),
by Brendan McKay. Nauty computes a *canonical form*: a deterministic relabeling
such that two graphs are isomorphic **iff** their canonical forms are *literally
identical*. It does this by iteratively *refining a coloring* of the vertices
(partitioning them by degree, then by neighbors' degrees, and so on) and
backtracking through the residual symmetries, pruning with the automorphisms it
discovers along the way. SageMath ships nauty and exposes it through
`Graph.is_isomorphic` / `Graph.canonical_label`; this code uses the former
(`is_isomorphic`) inside `graphs_isomorphic_with_labels` and
`directed_graphs_isomorphic_with_labels`.

**Vertex-colored isomorphism — the key subtlety.** A plain isomorphism is allowed
to map *any* vertex to *any* vertex. But in our diagrams, **leaves are not
interchangeable with internal vertices** — a relabeling that turns a leaf into an
internal vertex is *not* a legal symmetry. The trick used here is **vertex
coloring**: assign color `0` to leaves and color `1` to internal vertices, and ask
for a *color-preserving* isomorphism. Nauty/Sage honors a vertex *partition* (the
coloring) and only searches over relabelings that keep colors fixed. In the code
this is done by `set_vertex(v, 0_or_1)` before calling `is_isomorphic`
(`loop_diagram_enumeration.py:84-96` and `:99-111`).

---

## External tools used

This subsystem touches **SageMath** heavily, the Python standard library
(`itertools`, `collections`, `concurrent.futures`), and — *only transitively*,
through `degree_scan.py`'s re-expansion path — the project's own serialization and
`FieldTheory`. It does **not** itself import sympy, scipy, numba, or networkx.
Each is explained from scratch below.

### SageMath (`sage`)

**What it is.** SageMath is a large open-source mathematics system built on top of
Python. It bundles dozens of specialized math libraries (Pari, GAP, Singular,
**nauty**, NetworkX, NumPy, …) behind a uniform Python API, and adds its own types
for exact arithmetic, symbolic expressions (the "Symbolic Ring" `SR`), and — most
relevant here — **graph theory** (`Graph`, `DiGraph`, and the `graphs.*`
generators). Because Sage replaces some Python built-ins (e.g. integer literals
become Sage `Integer`s) and needs its bundled libraries on the path, *this module
can only run inside a SageMath kernel/interpreter* — see the module docstring:
`"Requires a SageMath kernel."` (`loop_diagram_enumeration.py:7`), and the tests
run via `sage -python -m pytest` (`test_degree_scan.py:7`).

**The import.** The whole API is pulled in with one wildcard import:

```python
from sage.all import *      # loop_diagram_enumeration.py:22
```

This is the conventional Sage idiom; it brings `Graph`, `DiGraph`, `graphs`,
`Integer`, etc. into scope. `degree_scan.py` and `test_degree_scan.py` instead
import the specific names they need:

```python
from sage.all import DiGraph, Graph    # test_degree_scan.py:14
from sage.all import SR                # type_assignment.py:19 (downstream)
```

**Exactly how this code uses Sage.** Every graph object in this subsystem is a
Sage graph. The specific Sage features used:

- **`graphs.trees(n)`** (`loop_diagram_enumeration.py:149`) — a *generator* that
  yields *every* non-isomorphic tree on `n` vertices, exactly once each. This is
  the brute-force engine for the tree stage; we then filter by leaf count and
  degree bounds. (Sage generates these efficiently; the user never has to enumerate
  trees by hand.)
- **`Graph(tree, multiedges=True, loops=False)`** (`loop_diagram_enumeration.py:174`)
  — constructs an undirected graph that *allows multiple edges between the same
  pair of vertices* (`multiedges=True`, needed because loop-edges can double an
  existing tree edge to make a "bubble") but *forbids self-loops* (`loops=False`,
  an edge from a vertex to itself, which is never physical here).
- **`DiGraph(multiedges=True, loops=False)`** (`loop_diagram_enumeration.py:265`)
  — the directed analog, used to hold an oriented topology (a prediagram).
- **Graph inspection methods:** `G.vertices()`, `G.edges(labels=...)`,
  `G.degree(v)`, `G.neighbors(v)`, `G.is_connected()`, `G.order()` (= `|V|`),
  `G.size()` (= `|E|`), `G.add_edge(u, v)`, `G.copy()`, `G.relabel(dict,
  inplace=False)`.
- **`G.set_vertex(v, value)`** (`loop_diagram_enumeration.py:93-95`,
  `:107-109`) — attaches an arbitrary *payload* (here the color `0`/`1`) to a
  vertex. Sage's isomorphism test treats these payloads as a vertex coloring.
- **`G.is_isomorphic(H)`** (`loop_diagram_enumeration.py:96`, `:111`) — the
  **nauty-backed** isomorphism test. Because we set vertex colors first, this
  becomes a *color-preserving* isomorphism test (leaves only map to leaves).
- **DiGraph-specific:** `D.add_vertices(...)`, `D.add_edge(u, v, label)`,
  `D.in_degree(v)`, `D.out_degree(v)`, `D.is_directed_acyclic()`,
  `D.neighbors_out(v)`, `D.incoming_edges(v)`, `D.outgoing_edges(v)`,
  `D.to_undirected()`.
- **Plotting** (`sage.plot.plot.graphics_array`, `G.plot(...)`, `.show()`,
  `.save()`) — used only by the `_plot_*` helpers and the `show_*` API for inline
  display in a notebook. Not part of the data pipeline.

**nauty specifically.** The reader will not see the string "nauty" anywhere in the
code — it is *inside* Sage. Every `is_isomorphic` call routes to nauty's
canonical-form machinery. That is the *only* graph-isomorphism technology this
subsystem relies on, and it is what makes deduplication tractable. (See the
glossary and the "graph isomorphism from scratch" passage in **The math** above.)

### `itertools` (Python standard library)

**What it is.** A standard-library module of memory-efficient iterators for
combinatorial generation.

**Import & use:**

```python
from itertools import combinations, combinations_with_replacement   # :23
```

- **`combinations(iterable, r)`** — yields every `r`-subset (no repeats, order
  ignored). Used in `generate_edge_multisets` (`loop_diagram_enumeration.py:169`):
  `combinations(all_vertices, 2)` enumerates every *possible* edge (unordered pair
  of distinct vertices).
- **`combinations_with_replacement(iterable, r)`** — like `combinations` but
  *allows the same element more than once*. Used at `:170`:
  `combinations_with_replacement(possible_edges, ell)` enumerates every *multiset*
  of `ell` edges to add — "with replacement" is essential because adding the *same*
  edge twice creates a double-edge (a bubble), which is a legitimate loop topology.

`type_assignment.py` (downstream) additionally uses `permutations` and `product`
from `itertools`, but those are out of scope for this subsystem.

### `collections` (Python standard library)

```python
from collections import Counter, defaultdict   # :24
```

- **`defaultdict(list)`** — a dict that auto-creates an empty list on first access
  to a missing key. Used in `_plot_trees` (`loop_diagram_enumeration.py:349`) to
  bucket trees by their `j` value for grouped display.
- **`Counter`** is imported but, in the current file, **not actually referenced**
  anywhere (a harmless dead import — see Gotchas).

### `concurrent.futures` (Python standard library)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed   # :25
```

- **`ThreadPoolExecutor(max_workers=n)`** — runs callables on a pool of OS
  *threads* (not processes). Used in `_enumerate_topologies_raw`
  (`loop_diagram_enumeration.py:248`) and `_enumerate_prediagrams_raw` (`:332`) to
  fan tree-processing / orientation-processing out across threads when
  `n_threads > 1`.
- **`as_completed(futures)`** — yields each submitted future *as it finishes*
  (not in submission order), so results are collected as soon as they are ready.

**Why threads and not processes here?** This is a deliberate, safety-critical
choice in this project. Per the MEMORY note on macOS fork-safety, **fork-based
multiprocessing crashes the user's machine** when a kernel forks after
BLAS/Cocoa/matplotlib init. Threads avoid `fork` entirely. The trade-off is the
Python GIL — but Sage's graph routines spend most of their time in compiled C
(nauty, the graph backend), which releases the GIL, so threads can still
parallelize the heavy isomorphism work. Note the *default* is `n_threads=1`
everywhere (serial), so the thread pool only activates when the caller opts in.

### Transitive tools via `degree_scan.py` (the re-expansion path only)

`degree_scan.py` imports project internals, *not* third-party libraries:

```python
from msrjd.core.serialize import load_theory, save_theory, reload_model   # degree_scan.py:13
from msrjd.core.field_theory import FieldTheory                           # degree_scan.py:14
```

- **`load_theory(path) -> (meta, data)`** / **`save_theory(path, ft, ...)`** —
  read/write a previously-expanded theory to/from disk. `meta` is a dict carrying
  among other things `meta['taylor_order']`, `meta['stationarity']`,
  `meta['model_file']`, `meta['model_var_name']`.
- **`reload_model(meta, project_root=...)`** — re-imports the *model definition*
  (the Python file describing the action) from `meta['model_file']` /
  `meta['model_var_name']`.
- **`FieldTheory(model, taylor_order=...)` + `.expand()`** — the Phase 1 engine
  that Taylor-expands the action's interaction terms to the requested polynomial
  order. `degree_scan.ensure_taylor_order` invokes this *only* when the saved order
  is too low for the prediagrams at hand.

These are reached only on the re-expansion branch; in the common case
(`sufficient == True`) `ensure_taylor_order` returns immediately without touching
`FieldTheory`.

---

## Components

Exhaustive, in file order. Signatures are quoted verbatim; line numbers are the
`def` line in `loop_diagram_enumeration.py` unless noted.

### Core graph utilities

#### `classify_vertices_sage(G)` — line 32
**Signature:** `classify_vertices_sage(G)`
**Takes:** a Sage `Graph` or `DiGraph` `G`.
**Returns:** a 4-tuple `(leaves, internal, degree_2, degree_3plus)`, each a list of
vertices.
**Steps:** one list-comprehension per category, filtering `G.vertices()` by
`G.degree(v)`: `== 1` (leaves), `> 1` (internal), `== 2` (degree_2), `>= 3`
(degree_3plus). Note `internal = degree_2 ∪ degree_3plus`. This is the single most
called helper in the module.

#### `has_adjacent_degree2_sage(G, degree_2_vertices)` — line 41
**Signature:** `has_adjacent_degree2_sage(G, degree_2_vertices)`
**Takes:** graph `G` and the precomputed list of its degree-2 vertices.
**Returns:** `True` iff *some* degree-2 vertex has a degree-2 *neighbor*.
**Steps:** builds a set of the degree-2 vertices, then double-loops over each
degree-2 vertex's neighbors; returns `True` on the first neighbor that is also
degree-2 (and not the vertex itself). Used to prune redundant line-subdivisions.

#### `check_deg3_has_non_deg2_neighbor(G, deg3plus_vertices, deg2_vertices)` — line 50
**Returns:** `False` iff some degree-≥3 vertex has *all* of its neighbors degree-2
(an "isolated hub" surrounded only by subdivision chains); `True` otherwise.
**Steps:** for each degree-≥3 vertex, `all(n in deg2_set for n in G.neighbors(v))`
→ if true for any, return `False`.

#### `check_leaf_neighbors_not_all_deg2(G, leaves, deg2_vertices)` — line 58
**Returns:** `False` iff every neighbor of every leaf is degree-2 (an external leg
that attaches only through subdivision chains); `True` otherwise.
**Steps:** collect the set of all leaf-neighbors; if it is non-empty and entirely
degree-2, return `False`.

#### `count_cycles_sage(G)` — line 68
**Returns:** the loop number `|E| − |V| + 1` for a connected `G`, or `-1` if `G` is
disconnected.
**Steps:** `if not G.is_connected(): return -1`, else `return G.size() - G.order()
+ 1`. This is the formula gate that the topology stage uses to demand exactly
`ell` loops.

#### `relabel_leaves_first(G)` — line 74
**Returns:** `(G_relabeled, relabel_dict)` where the relabeled graph has its
**leaves numbered `0 .. |L|−1` and internal vertices numbered `|L| .. |V|−1`**.
**Steps:** classify; build a dict mapping each leaf (in sorted order) to a small
index and each internal vertex (in sorted order) to an index offset by the leaf
count; call `G.relabel(dict, inplace=False)` (Sage returns a *new* relabeled graph
when `inplace=False`). This canonical leaf-first labeling is applied to every
accepted topology so that downstream code can rely on "leaves come first."

#### `graphs_isomorphic_with_labels(G1, leaves1, G2, leaves2)` — line 84
**Returns:** `True` iff `G1` and `G2` are isomorphic **as leaf-colored graphs**.
**Steps:**
1. Fast rejects: different `order()` (|V|), different `size()` (|E|), or different
   leaf counts → `False` immediately (cheap invariants before the expensive test).
2. Copy both graphs and color each vertex `0` if it is a leaf, else `1`, via
   `set_vertex`.
3. Return `G1c.is_isomorphic(G2c)` — the nauty-backed, color-preserving test.

This is the *undirected* dedup predicate, used for topologies.

#### `directed_graphs_isomorphic_with_labels(D1, leaves1, D2, leaves2)` — line 99
**Returns:** the directed analog of the above — `True` iff two prediagrams (DiGraphs)
are isomorphic as leaf-colored *directed* graphs.
**Steps:** identical structure to the undirected version; the only difference is
that `D1c.is_isomorphic(D2c)` is now a directed isomorphism (arrows must match).

### Tree generation

#### `generate_trees_with_constraints(k, ell, max_vertices_search=50)` — line 118
**Returns:** a list of triples `(tree, j, num_leaves)` — every non-isomorphic tree
satisfying the degree constraints, tagged with its leaf-surplus `j` and its leaf
count `num_leaves = k + j`.
**Steps (the heart of the completeness bound):**
1. `j_max = ell + ell//2` (line 124).
2. Loop `j` from `0` to `j_max`. For each `j`:
   - `num_leaves = k + j`.
   - `v3_max = k + j − 2` (degree-≥3 bound).
   - `v2_max = 2·v3_max + 3·ell − k − 3·j + 3` (= the proven `k + 3·ell − j − 1`).
   - Compute the vertex-count search window:
     - `min_n = num_leaves` if `num_leaves == 1` else `num_leaves + 1`
       (a single leaf is the degenerate one-vertex tree; otherwise you need at
       least one internal vertex).
     - `max_n = min(num_leaves + v2_max + v3_max, max_vertices_search)` then
       further capped by `min(max_n, num_leaves + min(10, v2_max + v3_max))`
       (line 143) — **the `+10` hard cap.** See Gotchas: this cap is a performance
       guard that can in principle undercut the proven bound at high order.
   - If `max_n < min_n`, skip this `j`.
3. For each `n` in `[min_n, max_n]`, iterate `graphs.trees(n)` (all non-isomorphic
   trees on `n` vertices) and **keep** a tree iff:
   - its leaf count equals `num_leaves`,
   - `|degree_3plus| ≤ v3_max`,
   - `|degree_2| ≤ v2_max`.
4. Accumulate `(tree, j, num_leaves)` for every survivor; return the list.

### Edge addition

#### `generate_edge_multisets(all_vertices, ell)` — line 166
**Returns:** a list of `ell`-multisets of edges (each multiset is a tuple of `ell`
vertex-pairs) to *add* to a tree, or `[]` if there are fewer than 2 vertices.
**Steps:** `possible_edges = combinations(all_vertices, 2)` (every distinct pair),
then `combinations_with_replacement(possible_edges, ell)` (every multiset of `ell`
edges, repeats allowed → permits double-edges/bubbles).

#### `add_edges_to_tree(tree, edge_multiset)` — line 173
**Returns:** a fresh undirected multigraph = the tree plus the `ell` extra edges.
**Steps:** `G = Graph(tree, multiedges=True, loops=False)` (copy with multi-edge
support), then `G.add_edge(...)` for each edge in the multiset.

#### `check_topology_constraints(G, k, ell=None)` — line 180
**Returns:** `True` iff `G` is an admissible *topology*.
**Steps:** classify; require `|leaves| == k` and `G.is_connected()`; then, *only
if `ell is None or ell > 0`*, apply the three degree-2 pruning checks
(`has_adjacent_degree2_sage`, `check_deg3_has_non_deg2_neighbor`,
`check_leaf_neighbors_not_all_deg2`). At `ell == 0` those are skipped (tree = its
own topology).

### Topology enumeration

#### `process_tree_parallel(args)` — line 203
**Signature:** `process_tree_parallel(args)` where `args = (tree, j, num_leaves, k,
ell)`.
**Returns:** a list of `(G_relabeled, leaves_final, internal_final)` candidate
topologies derived from this one tree.
**Steps:** for each edge-multiset from `generate_edge_multisets`:
add the edges; require `count_cycles_sage(G) == ell` (exactly `ell` loops);
require `check_topology_constraints(G, k, ell=ell)`; then `relabel_leaves_first`
and re-classify; append the candidate. (The name says "parallel" because it is the
unit of work submitted to the thread pool, but it runs perfectly well serially.)

#### `_remove_isomorphic_undirected(candidates)` — line 219
**Returns:** the deduplicated list of unique topologies.
**Steps:** a classic *O(n²)* dedup: walk the candidates; keep one only if it is
**not** `graphs_isomorphic_with_labels` to any already-kept representative. Quadratic
in the number of candidates, but each comparison is cheap (invariant fast-rejects
first, nauty only when those pass).

#### `_enumerate_topologies_raw(k, ell, n_threads=1, max_vertices_search=50, verbose=True, trees_with_j=None)` — line 228
**Returns:** the list of unique topologies `(G, leaves, internal)`.
**Steps:**
1. If `trees_with_j` is not supplied, call `generate_trees_with_constraints`
   (allows the caller to share the tree list across stages — see `enumerate_all`).
2. Process each tree (serially if `n_threads == 1`, else through a
   `ThreadPoolExecutor`) via `process_tree_parallel`, accumulating candidates.
3. Dedup with `_remove_isomorphic_undirected`; return.
   Verbose mode prints progress and the candidate-vs-unique counts.

### Orientation enumeration

#### `orient_edges(G, orientation_bits)` — line 264
**Returns:** a `DiGraph` `D` = `G` with each edge oriented per the bit vector.
**Steps:** create an empty multi-edge DiGraph, add all vertices, then for the
`i`-th undirected edge `(u, v)`: add `u → v` if `orientation_bits[i] == 0`, else
`v → u`. **The edge index `i` is used as the edge label** (`D.add_edge(u, v, i)`),
which preserves multi-edge identity. (Note the per-edge ordering comes from
`G.edges(labels=False)`, which is deterministic for a given graph object.)

#### `check_orientation_constraints(D, leaves)` — line 275
**Returns:** `True` iff the orientation `D` is an admissible prediagram.
**Steps:** (see "orientation constraints" in **The math**) require
`is_directed_acyclic`; reject degree-1 vertices that are not pure-incoming; reject
`out_deg==0, in_deg≥2` sinks; reject `in_deg==1, out_deg==1` pass-throughs; reject
any two adjacent sources.

#### `enumerate_orientations(G, leaves)` — line 295
**Returns:** the list of all admissible oriented `DiGraph`s for `G`.
**Steps:** iterate `bits` over `0 .. 2^{|E|}−1`, decode `bits` into a length-`|E|`
orientation vector via `(bits >> i) & 1`, build the DiGraph, and keep it iff
`check_orientation_constraints` passes.
**Note (mild inefficiency):** the comprehension at `:297-301` calls `orient_edges`
**twice** per candidate — once inside `check_orientation_constraints(...)` and once
to materialize the kept result. See Gotchas.

#### `remove_isomorphic_directed(directed_diagrams)` — line 304
**Returns:** deduplicated list of unique prediagrams `(D, G, leaves, internal)`.
**Steps:** same O(n²) dedup pattern as the undirected case, but using
`directed_graphs_isomorphic_with_labels`.

#### `process_orientation_parallel(args)` — line 313
**Signature:** `process_orientation_parallel(args)` where `args = (G, leaves,
internal)`.
**Returns:** `[(D, G, leaves, internal) for D in enumerate_orientations(G,
leaves)]` — the thread-pool work unit for the orientation stage.

#### `_enumerate_prediagrams_raw(topologies, n_threads=1, verbose=True)` — line 318
**Returns:** the deduplicated list of unique prediagrams.
**Steps:** for each topology, enumerate orientations (serially or via the thread
pool), accumulate `(D, G, leaves, internal)`, then `remove_isomorphic_directed`.

### Visualization helpers (notebook display only)

These produce no pipeline data; they render Sage `graphics_array` plots inline (and
optionally save PNGs). Color codes are documented in the corresponding `show_*`
docstrings.

- **`_plot_trees(trees_with_j, k, ell, save=False)` — line 347.** Groups trees by
  `j` via `defaultdict`, colors leaves black / degree-2 lightblue / degree-3+ red,
  titles each with `|V|, |L|, |V₂|, |V₃|`.
- **`_plot_topologies(topologies, k, ell, save=False)` — line 382.** Leaves black,
  internal lightgray.
- **`_plot_prediagrams(prediagrams, k, ell, save=False)` — line 403.** Leaves
  black, **sources red**, other internal lightblue. Sources are recomputed here as
  `in_degree == 0`.

### Public API

All return `(payload, count)` or `(…multiple payloads…, counts_dict)`; the `show_*`
variants additionally render and return just the count(s).

- **`enumerate_all(k, ell, n_threads=1, max_vertices_search=50, verbose=True)` —
  line 434.** Runs the full pipeline *once*, sharing the tree list across stages
  (it passes `trees_with_j=trees` into `_enumerate_topologies_raw` so trees are not
  regenerated). Returns `(trees, topologies, prediagrams, counts)` where `counts =
  {n_trees, n_topologies, n_prediagrams}`.
- **`show_all(k, ell, ...)` — line 462.** Calls `enumerate_all`, then plots each
  non-empty stage. Returns `counts`.
- **`enumerate_trees(k, ell, max_vertices_search=50, verbose=False)` — line 478.**
  Returns `(trees, count)`. Thin wrapper over `generate_trees_with_constraints`.
- **`enumerate_topologies(k, ell, ...)` — line 498.** Returns `(topologies,
  count)`.
- **`enumerate_prediagrams(k, ell, ...)` — line 515.** Returns `(prediagrams,
  count)` — runs topologies then orientations. **This is the canonical entry point
  the rest of the pipeline calls.**
- **`show_trees` / `show_topologies` / `show_prediagrams`** — lines 535 / 552 /
  569. Display + return count.

### `degree_scan.py` (Build Phase C — the feedback bridge)

#### `max_vertex_degree(prediagrams)` — `degree_scan.py:17`
**Takes:** `list of (D, G, leaves, internal)`.
**Returns:** the maximum `in_degree(v) + out_degree(v)` over all **non-leaf**
vertices across all prediagrams; `0` if none.
**Steps:** for each prediagram, skip leaves (set-membership), and track the running
max total degree. This is the *required Taylor order* — a degree-`d` internal
vertex needs a `d`-leg interaction monomial, which only exists if the action was
expanded to order `d`.

#### `scan_source_vertices(prediagrams)` — `degree_scan.py:45`
**Returns:** the *set of out-degrees* exhibited by **source** vertices (non-leaf,
`in_degree == 0`) across all prediagrams.
**Steps:** for each non-leaf vertex with `in_degree == 0`, add its `out_degree` to
the set. These out-degrees tell Phase E which *noise-kernel* arities are needed.

#### `check_taylor_order(meta, max_degree)` — `degree_scan.py:70`
**Returns:** `(sufficient, current_order, required_order)` where `sufficient =
(meta['taylor_order'] >= max_degree)`, `current = meta['taylor_order']`, `required
= max_degree`.

#### `ensure_taylor_order(theory_path, prediagrams, project_root=None)` — `degree_scan.py:91`
**Returns:** `(meta, data)` — either the already-loaded theory (if sufficient) or
the freshly re-expanded-and-reloaded theory.
**Steps:**
1. `load_theory(theory_path)` → `(meta, data)`.
2. `max_deg = max_vertex_degree(prediagrams)`; `check_taylor_order`.
3. If sufficient, return `(meta, data)` unchanged.
4. Otherwise print a re-expansion message, `reload_model(meta, project_root=...)`,
   build `FieldTheory(model, taylor_order=required)`, `.expand()`, `save_theory(...)`
   (passing `propagator_data=None` — **the propagator must be recomputed
   separately**, as the print at `:133` warns), and finally
   `return load_theory(theory_path)` (re-read from disk to get the updated meta).

---

## Data structures

The subsystem passes around a small number of tuple shapes (no dataclasses are
defined here — the `TypedDiagram` dataclass lives downstream in
`type_assignment.py`).

- **Tree record:** `(tree, j, num_leaves)`
  - `tree` — a Sage `Graph` (acyclic).
  - `j : int` — leaf surplus; `num_leaves = k + j`.
  - `num_leaves : int` — the tree's leaf count.

- **Topology record:** `(G, leaves, internal)`
  - `G` — a Sage `Graph` (undirected, multi-edges allowed, exactly `ell` loops),
    **relabeled leaves-first** (`relabel_leaves_first`).
  - `leaves : list[int]` — vertex IDs `0 .. k−1` after relabeling.
  - `internal : list[int]` — the remaining vertex IDs.

- **Prediagram record:** `(D, G, leaves, internal)` — *the* output object
  - `D` — a Sage `DiGraph` (the oriented graph; edge labels carry the original
    edge index for multi-edge identity).
  - `G` — the underlying undirected topology (same as the topology stage).
  - `leaves : list[int]` / `internal : list[int]` — as above.
  - This 4-tuple is exactly what `type_assignment.enumerate_typed_diagrams` unpacks
    (`type_assignment.py:140`) and what `degree_scan.*` iterate over.

- **Counts dict:** `{'n_trees': int, 'n_topologies': int, 'n_prediagrams': int}`
  (built in `enumerate_all`, `loop_diagram_enumeration.py:454`).

- **Edge-multiset:** a tuple of `ell` vertex-pairs, e.g. `((0,3),(0,3))` for a
  double-edge bubble between vertices 0 and 3.

- **Orientation vector:** a length-`|E|` list of bits (`0` = keep `(u,v)`,
  `1` = flip to `(v,u)`), produced by `(bits >> i) & 1`.

- **`meta` dict (from `load_theory`, used by `degree_scan`):** carries at least
  `meta['taylor_order'] : int`, and on the re-expansion path also
  `meta['stationarity']`, `meta['model_file']`, `meta['model_var_name']`.

---

## Data flow

### Inputs

The *only* genuine inputs to the generator are the two integers `k` and `ell`
(plus tuning knobs `n_threads`, `max_vertices_search`, `verbose`). No theory data,
no action, no field labels.

### Internal flow (one call to `enumerate_prediagrams(k, ell)`)

```
  (k, ell)
     │
     ▼  generate_trees_with_constraints
  [ (tree, j, num_leaves), ... ]                     # trees passing degree bounds
     │
     ▼  process_tree_parallel  (add ell edges, demand L==ell, apply constraints,
     │                          relabel leaves-first)
  [ (G, leaves, internal), ... ]  (with isomorphic duplicates)
     │
     ▼  _remove_isomorphic_undirected   (nauty, leaf-colored)
  [ (G, leaves, internal), ... ]  unique TOPOLOGIES
     │
     ▼  enumerate_orientations  (2^|E| orientations, keep admissible)
  [ (D, G, leaves, internal), ... ]  (with isomorphic duplicates)
     │
     ▼  remove_isomorphic_directed   (nauty, leaf-colored, directed)
  [ (D, G, leaves, internal), ... ]  unique PREDIAGRAMS   ───►  downstream
```

### Outputs and who reads which key

- `type_assignment.enumerate_typed_diagrams(prediagram, ...)` reads the whole
  4-tuple: `D` for the directed structure, `leaves` for the external-leg set,
  `internal` for the interaction/source vertices (`type_assignment.py:140`).
- `degree_scan.max_vertex_degree` / `scan_source_vertices` read `D` (for in/out
  degrees) and `leaves` (to exclude external legs).

### A concrete example of counts (anchor numbers from MEMORY/code comment)

At the tested small orders, the deduplicated *topology* counts are recorded as:
`(k,ell) = (2,1) → 9 topologies`, `(3,1) → 67`, `(2,2) → 289`
(`loop_diagram_enumeration.py:138` comment and `project_enumeration_bound_fix.md`).
These are the same before and after the June-2026 bound fix (the fix changed only
the *proof*, not the observed output at these orders).

### Re-expansion feedback (Phase C)

```
  prediagrams ──► max_vertex_degree ──► required_order
                                            │
  saved theory.meta['taylor_order'] ───────►│ check_taylor_order
                                            ▼
                          sufficient? ── yes ──► return (meta, data)
                                  │
                                  no ──► reload_model → FieldTheory(...).expand()
                                         → save_theory(propagator_data=None)
                                         → reload → (meta, data)  [propagator stale!]
```

---

## Gotchas & caveats

1. **`Counter` is imported but unused.** `from collections import Counter,
   defaultdict` (`:24`) — only `defaultdict` is referenced. Harmless dead import.

2. **The `+10` vertex-search cap can in principle undercut completeness at high
   order.** `generate_trees_with_constraints` caps the tree-vertex search at
   `max_n = min(max_n, num_leaves + min(10, v2_max + v3_max))`
   (`loop_diagram_enumeration.py:143`). The *proven* completeness bound only
   guarantees that all needed topologies fit within `num_leaves + v2_max + v3_max`
   vertices; this extra cap clamps the *internal*-vertex budget to **at most 10**.
   For small `(k, ell)` the proven budget is well under 10, so the cap is inert —
   but at large enough orders `v2_max + v3_max > 10`, and then the cap could *skip
   trees the proven bound says we need*, silently dropping topologies. This is a
   performance guard, not a proven-safe bound — flag for review at high order. (See
   open questions.)

3. **The whole module requires a SageMath kernel.** `from sage.all import *`
   (`:22`) will not import under a plain CPython interpreter. Tests run via `sage
   -python -m pytest`. Do not try to import this module in a non-Sage environment.

4. **Thread-only parallelism is intentional and load-bearing for safety.** The
   thread pool (not a process/fork pool) is a deliberate choice driven by the macOS
   fork-crash history (see project MEMORY). Do **not** "optimize" this to
   `ProcessPoolExecutor` / fork — it can crash the user's machine. Threads are
   correct here because the heavy work (nauty isomorphism, Sage graph backend) runs
   in GIL-releasing C. Default `n_threads=1` (serial) everywhere.

5. **`enumerate_orientations` builds each candidate DiGraph twice.**
   (`:297-301`) — once for the constraint check, once to keep it. Functionally
   correct, but doubles the orientation-construction cost; a clean rewrite would
   build once and reuse.

6. **`O(n²)` isomorphism dedup.** Both `_remove_isomorphic_undirected` and
   `remove_isomorphic_directed` compare each new candidate against every kept
   representative. The cheap invariant fast-rejects (order/size/leaf-count) keep
   this affordable at small orders, but it is quadratic in the candidate count and
   will dominate at high order. A canonical-form bucket (`Graph.canonical_label`)
   would make it near-linear; not done here.

7. **The `j`-loop bound `j_max = ell + ell//2` and the degree-2 bound are
   *separate* prunes.** Both must be correct for completeness; the headline fix
   (June 2026) corrected only the degree-2 bound (`v2_max`), removing a false
   `−⌊j/3⌋` term. If anyone re-tightens either bound, it must come from a *proven*
   statement — empirical slack at small orders is explicitly called out as **not a
   proof** (`:133` comment, MEMORY note).

8. **`check_topology_constraints` skips degree-2 pruning at `ell == 0`.** Be aware
   that the constraint set is *different* at tree level vs loop level (gate at
   `:188`). A tree-level call with `ell=None` also skips the pruning. Passing the
   wrong `ell` here would change which topologies survive.

9. **`relabel_leaves_first` is applied to topologies but the leaf-first invariant
   is assumed downstream.** Downstream code (`type_assignment`, `degree_scan`)
   treats `leaves` as the canonical external-leg set; it relies on the relabeling
   having been done. Topologies are relabeled inside `process_tree_parallel`
   (`:213`), so any code that builds prediagrams *without* going through this path
   (e.g. the hand-built mocks in `test_degree_scan.py`) will not have leaf-first
   labels — those tests only exercise `degree_scan`, which does not assume it.

10. **`ensure_taylor_order` leaves the propagator stale after re-expansion.** It
    passes `propagator_data=None` to `save_theory` and prints "propagator data must
    be recomputed" (`degree_scan.py:131-133`). A caller that re-expands and then
    immediately uses the propagator without recomputing will get a wrong/missing
    propagator. This is by design (propagator computation is a separate phase) but
    is an easy footgun.

11. **`count_cycles_sage` returns `-1` for disconnected graphs**, which is *not*
    `ell` for any `ell ≥ 0`, so a disconnected graph is automatically rejected by
    the `== ell` test in `process_tree_parallel`. The `-1` is a deliberate sentinel,
    not an error.

12. **Edge labels in `orient_edges` carry the original edge index** (`D.add_edge(u,
    v, i)`), which is what preserves multi-edge identity through orientation and
    into the prediagram `D`. Downstream multi-edge handling (`type_assignment`
    iterates `D.edges()` as 3-tuples `(u, v, label)`) depends on these labels being
    present and distinct.

---

## Glossary

- **MSR-JD field theory** — Martin–Siggia–Rose–Janssen–De Dominicis path-integral
  formulation of stochastic dynamics. Diagrams have *directed* (causal) propagators
  and a doubled field content (a "physical" field and a "response"/"hatted" field).
- **`k`-point function** — a correlation/cumulant of `k` fields; its diagrams have
  `k` external legs.
- **`ell` (loop order)** — the number of independent loops in the diagram; the
  perturbative order of the correction.
- **Prediagram** — a *bare, oriented* graph topology: leaves vs internal vertices +
  arrows on edges, but **no** propagator types or vertex monomials assigned yet.
  The output object of this subsystem (`(D, G, leaves, internal)`).
- **Topology** — an *undirected* prediagram (no arrows yet); the intermediate
  output of the topology stage.
- **Tree** — a connected acyclic graph; the loop-free skeleton from which
  topologies are built by adding edges.
- **Leaf / external leg** — a degree-1 vertex; an external argument of the
  correlation function.
- **Internal vertex** — a degree-≥2 vertex; an interaction point.
- **Source** — an internal vertex with **in-degree 0** (and not a leaf); a
  noise-kernel insertion point in MSR-JD.
- **Sink** — a vertex with out-degree 0.
- **Degree** — number of edge-ends at a vertex; multi-edges count their multiplicity.
- **Loop number / first Betti number** — `|E| − |V| + 1` for a connected graph;
  the number of independent cycles.
- **`j` (leaf surplus)** — number of "extra" tree-leaves that the `ell` added
  loop-edges will consume; trees are generated with `k + j` leaves.
- **`v2_max` / `v3_max`** — search bounds on the number of degree-2 / degree-≥3
  vertices; `v3_max = k+j−2`, `v2_max = k + 3·ell − j − 1`.
- **Completeness bound (`k + 3·ell − j − 1`)** — the proven upper bound on degree-2
  vertices that guarantees the tree search does not miss any valid topology.
- **Multi-edge (multigraph)** — two or more edges between the same vertex pair; a
  double-edge forms a "bubble" loop. Enabled via `multiedges=True`.
- **Self-loop** — an edge from a vertex to itself; disabled here (`loops=False`),
  never physical.
- **Isomorphism (of graphs)** — a vertex relabeling that maps one graph onto another
  edge-for-edge; isomorphic diagrams are "the same diagram."
- **Vertex coloring / colored isomorphism** — assigning colors to vertices (here
  `0`=leaf, `1`=internal) and requiring an isomorphism to *preserve* colors, so
  leaves can only map to leaves. Implemented via Sage `set_vertex`.
- **Canonical form / canonical label** — a deterministic relabeling such that two
  graphs are isomorphic iff their canonical forms are identical; the core idea
  behind efficient isomorphism testing.
- **nauty** — Brendan McKay's C library ("No AUTomorphisms, Yes?") for graph
  canonical labeling and isomorphism, bundled inside SageMath and used implicitly by
  every `is_isomorphic` call here.
- **SageMath** — open-source mathematics system bundling many math libraries (incl.
  nauty) behind a Python API; provides `Graph`, `DiGraph`, `graphs.trees`, etc. This
  module requires a Sage kernel.
- **`graphs.trees(n)`** — Sage generator yielding every non-isomorphic tree on `n`
  vertices exactly once.
- **`Graph` / `DiGraph`** — Sage undirected / directed graph classes.
- **`ThreadPoolExecutor` / `as_completed`** — Python stdlib thread-pool primitives
  used for opt-in parallelism (threads, never fork, on macOS-safety grounds).
- **`combinations` / `combinations_with_replacement`** — itertools generators for
  subsets / multisets; the latter (with replacement) is what permits adding the same
  edge twice (bubbles).
- **Taylor order** — the polynomial order to which the action's interaction terms
  were expanded in Phase 1; must be ≥ the max prediagram vertex degree, which is the
  job of `degree_scan.ensure_taylor_order`.
- **`FieldTheory` / `expand()`** — Phase 1 engine that Taylor-expands the action;
  re-invoked by `ensure_taylor_order` only when the saved order is too low.

---

## Proposed manual subsections

1. **What a prediagram is, and why we strip the physics first** — motivation;
   prediagram vs Feynman diagram; place in the build-phase pipeline.
2. **Graph-theory primer for field theorists** — vertices, edges, degree, leaves,
   trees, the loop-number formula `|E|−|V|+1`.
3. **The three-stage refinement: trees → topologies → prediagrams** — the
   filter-and-dedup pattern, with a worked `(k,ell)=(2,1)` walk-through.
4. **The leaf surplus `j` and the loop budget** — why intermediate trees carry
   `k+j` leaves; derivation of `j_max = ⌊3·ell/2⌋`.
5. **The degree bounds and the completeness theorem** — `v3_max = k+j−2`;
   `v2_max = k+3·ell−j−1`; the orientability + incidence ingredients; the
   algebraic check that the code's expression equals the bound.
6. **The bound-correction story (the false `−⌊j/3⌋` lemma)** — the `ell≥3`
   counterexample, why outputs were unchanged, and the "only proven bounds prune"
   principle.
7. **Structural constraints on topologies** — connectivity, leaf count, the three
   degree-2 pruning rules, and the `ell==0` skip.
8. **Orientation and the MSR-JD causality constraints** — acyclicity, leaf-incoming,
   forbidden sinks/pass-throughs, no adjacent sources; the source/interaction/leaf
   trichotomy.
9. **Graph isomorphism and nauty from scratch** — what isomorphism means, why naive
   `n!` fails, canonical labeling, colored isomorphism, and how Sage `set_vertex` +
   `is_isomorphic` implement leaf-preserving dedup.
10. **Parallelism and the macOS fork-safety constraint** — threads vs processes, the
    GIL, why this subsystem is thread-only.
11. **API reference** — `enumerate_*` / `show_*`, return shapes, the shared-tree
    optimization in `enumerate_all`.
12. **The degree-scan feedback bridge (Phase C)** — `max_vertex_degree`,
    `scan_source_vertices`, `ensure_taylor_order`, and the stale-propagator caveat.
13. **Data structures and downstream contract** — the `(D,G,leaves,internal)` tuple
    and how `type_assignment` consumes it.
14. **Performance notes and known limitations** — `O(n²)` dedup, double-build in
    `enumerate_orientations`, the `+10` vertex cap.

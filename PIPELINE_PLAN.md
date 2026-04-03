# Automated Feynman Diagram Pipeline — Architecture & Build Plan

**Last updated:** 2026-04-03
**Status:** Phases A–H implemented and debugged. Tree-level 2-point function validated against simulation for linear Hawkes. 1-loop evaluation implemented but awaiting validation. See `CHANGELOG.md` for critical bug fixes applied 2026-04-03.

---

## Overview

A model-agnostic, SageMath-based framework for computing perturbative corrections to k-point functions in MSRJD field theories. The pipeline has two phases:

1. **Theory Specification & Propagator Precomputation** — the user defines a stochastic system; the framework expands the action, extracts the free action, and precomputes the propagator as far as symbolically feasible.
2. **Diagram Computation** — given a k-point function and loop level, the framework enumerates all valid Feynman diagrams, assigns field types, checks causality, computes symmetry factors, and integrates.

All symbolic computation targets SageMath. SymPy is used only as a fallback integration backend.

---

## Phase 1: Theory Specification & Propagator Precomputation

### 1.1 User Input

| What | Description |
|---|---|
| **Fields** | Response fields (e.g. n_tilde, v_tilde) and physical fluctuation fields (e.g. delta_n, delta_v), each with index sets. Response fields are full integration variables; physical fields are expanded around the MF background. |
| **Parameters** | Scalar symbolic parameters (tau, w_ij, nstar_i, ...) with optional domains (positive, real, etc.). |
| **Functions** | Nonlinear functions (phi, ...) either as formal SageMath function symbols (auto-expanded) or with explicit Taylor coefficients. |
| **Filters** | Kernels required to integrate to 1 (e.g. synaptic filter g(t) with integral(g) = 1). |
| **Operators** | Differential operators selected from a menu: partial_t, gradient, Laplacian, etc. Each is an algebraic placeholder symbol in the action. |
| **Mean-field equations** | Background conditions (e.g. phi_i(v_i*) = nstar_i) and the fluctuation expansion (which fields are expanded around what). Applied as SR substitutions after Taylor expansion. |
| **Stationarity** | Whether the system is stationary in time. Enables frequency conservation at vertices during integration. |
| **Action** | Either: (a) the full MSRJD action S[fields] as a symbolic callable, or (b) a manually entered propagator matrix (skipping automatic extraction). |

### 1.2 Processing

1. Build the SageMath namespace: SR symbolic variables for all fields, parameters, kernels, operators, functions.
2. Evaluate the action callable to get an SR expression.
3. Taylor-expand all nonlinear functions of field variables (sequential `.taylor()` calls).
4. Rename formal derivative symbols to clean names (e.g. `D[0](phi_1)(0)` -> `phi1_1`).
5. Apply MF background conditions and optional specializations as SR substitutions.
6. Convert to `PolynomialRing(SR, field_names)` for bigrade classification.
7. Classify every monomial by bigrade (n_tilde, n_phys) via exponent vectors.
8. Sanity checks: verify (0,0), (1,0), (0,1) sectors vanish.
9. Extract free action = (1,1) sector.
10. Build kernel matrix K from free action monomials.
11. Convert K entries to kernel/distribution form (Dt -> delta', constants -> delta).
12. Fourier transform K symbolically: K_hat(omega).
13. Compute propagator G_hat(omega) = K_hat^{-1}(omega) with safeguards:
    - **Dimension gate** (MAX_DIM = 20): skip symbolic inverse for large matrices.
    - **Timeout** (INVERSE_TIMEOUT = 60s): kill `.inverse()` via SIGALRM if too slow.
    - **Complexity gate** (MAX_NOPS = 5000): reject result if expression tree is too large.
    - Always compute `adj_ft` and `D_omega = det(K_hat)` as implicit representation.
14. Inverse Fourier transform with branching logic:
    - **Branch 1:** det(K_hat) independent of omega -> G(t) = G_hat * delta(t).
    - **Branch 2:** Poles found via `solve()` -> residue theorem: G(t) = sum_k C_k exp(i omega_k t).
    - **Branch 3:** No poles, G_hat explicit -> attempt direct symbolic inverse FT entry-by-entry.
    - **Branch 4:** Nothing works -> leave in frequency domain. Still valid for Feynman diagram computation.

### 1.3 Output — Theory File

Saved to disk for loading in Phase 2:

| Object | Description |
|---|---|
| `model` | The full model dict including the action callable, so re-expansion to higher Taylor order is possible. |
| `K_ft` | Fourier-domain kernel matrix. |
| `G_ft` | Explicit propagator matrix (if computed), or None. |
| `adj_ft`, `D_omega` | Adjugate and determinant of K_hat (always available as implicit G_hat representation). |
| `pole_vals`, `C_mats` | Pole locations and residue coefficient matrices (if Branch 2 succeeded). |
| `G_t` | Time-domain propagator matrix (if inverse FT succeeded), or None. |
| `S_int` | The interacting action: all bigrade sectors with total degree >= 3, stored as a polynomial ring element. Not yet decomposed into individual vertices. |
| `noise_sectors` | Noise kernel sectors: bigrade (n_tilde >= 2, n_phys = 0). |
| `field_metadata` | Field names, index sets, response/physical classification, latex names, ring generator order. |
| `param_symbols` | All parameter, kernel, and operator SR symbols. |
| `stationarity` | Boolean flag. |
| `taylor_order` | The Taylor order used for expansion. |

**What is NOT done here:** Vertex decomposition. The interacting action is stored as a polynomial sum, not split into individual vertices. That decomposition happens in Phase 2 once the required Taylor order is determined by the prediagram structure.

---

## Phase 2: Diagram Computation

### Step 1 — User Input

- **k-point function:** the integer k and which specific field variables to correlate (e.g. "2-point function of delta_v_1 and delta_v_2").
- **Maximum loop level:** ell.

### Step 2 — Prediagram Enumeration

Run `enumerate_prediagrams(k, ell_i)` for each loop level ell_i = 0, 1, ..., ell.

Uses the existing `loop_diagram_enumeration.py`:
- Generate spanning trees with degree constraints.
- Add edges to form ell-loop topologies.
- Remove isomorphic undirected topologies.
- Orient edges to form prediagrams (DAGs).
- Remove isomorphic directed prediagrams.

Each prediagram is a tuple `(D, G, leaves, internal)` where D is a SageMath DiGraph, leaves are external legs, internal are interaction/source vertices.

### Step 3 — Determine Required Taylor Order

Scan all prediagrams across all loop levels:
- For each internal vertex v, compute `in_degree(v) + out_degree(v)`.
- Take the maximum across all vertices and all prediagrams.
- This is the minimum total field degree needed from S_int.

If the theory file was expanded to a Taylor order lower than this maximum, re-expand from the stored model dict (action callable) to the required order. This is why Phase 1 stores the full model dict.

### Step 4 — Extract Typed Vertices from S_int

Decompose each bigrade sector of S_int into individual monomials. Each monomial becomes a **vertex type**:

```
VertexType:
    coefficient   : SR expression (coupling constant * combinatorial prefactor)
    response_legs : list of (field_name, index)  — the n_tilde response fields
    physical_legs : list of (field_name, index)  — the n_phys physical fields
    in_degree     : n_phys  (physical fields = incoming edges)
    out_degree    : n_tilde (response fields = outgoing edges)
    bigrade       : (n_tilde, n_phys)
```

Similarly decompose noise kernel sectors into **source vertex types**:

```
SourceType:
    coefficient   : SR expression
    response_legs : list of (field_name, index)  — all legs are response fields
    out_degree    : n_tilde
```

### Step 5 — Filter Prediagrams by Vertex Availability

1. Build the set of available `(in_degree, out_degree)` pairs from the theory's vertex types.
2. Build the set of available source out-degrees from the noise kernel source types.
3. For each prediagram:
   - Check every internal (non-source) vertex: does its `(in_deg, out_deg)` appear in the available set?
   - Check every source vertex (in_degree = 0): does its out_degree appear in the available source set?
   - If any vertex has no match, discard the prediagram.

This is a fast set-membership filter that can eliminate many prediagrams before the expensive type-assignment step.

### Step 6 — Type Assignment (Prediagram -> Diagrams)

For each surviving prediagram, enumerate all valid fully-typed assignments. This is the combinatorial core of the pipeline.

**What needs to be assigned:**

1. **External legs:** Leaf vertex i is assigned `external_fields[i]` (fixed, not permuted). External legs are labeled and correspond to specific positions in the k-point function.

2. **Interaction vertices:** For each internal vertex with `(in_deg, out_deg)`, try all vertex types from Step 4 that match those degrees.

3. **Source vertices:** For each source vertex (in_degree = 0), try all source types from the noise kernel that match its out_degree.

4. **Edges:** Each directed edge u -> v connects:
   - A response-field out-leg of vertex u (the tail)
   - A physical-field in-leg of vertex v (the head)

   For each candidate assignment of vertices, check every edge: look up G_ft[physical_field_index, response_field_index] (transposed — the retarded propagator). If that entry is zero, the assignment is invalid.

5. **Leg matching at each vertex:** The specific field legs of the chosen vertex type must be consistently assignable to the edges incident on that vertex. This is a constraint-satisfaction problem at each vertex.

Each valid, fully-consistent assignment produces one **diagram**.

### Step 7 — Causality Test

For each typed diagram:
- Retarded propagators enforce time-ordering: the response field event must precede the physical field event along each directed edge.
- The DAG structure (enforced during prediagram enumeration) already prevents closed causal loops.
- Verify that for each assigned propagator component G_{ij}, the pole structure is consistent with retarded (causal) boundary conditions — poles should lie in the upper half-plane for the contour closure used in Branch 2.
- Reject any diagram where causality is violated.

### Step 8 — Symmetry Factors

For each valid diagram, compute the symmetry factor S:
- S = |Aut(diagram)| where Aut is the automorphism group of the fully-typed labeled diagram.
- Automorphisms must preserve: vertex types, edge types (propagator components), edge orientations, and external leg labels.
- The combinatorial prefactors from the action (1/n! from Taylor expansion) are already absorbed into the vertex coefficients from Step 4.
- The diagram contributes with weight 1/S times the product of vertex coefficients.

### Step 9 — Integration

For each diagram, construct the integral expression:

**Stationary systems (frequency domain):**
- Each edge carries a frequency variable omega_e.
- Each vertex imposes frequency conservation: delta(sum of incoming omega - sum of outgoing omega).
- After imposing conservation, there are exactly ell independent loop frequencies.
- The integrand is: product over edges of G_hat_{ij}(omega_e) * product over vertices of (vertex coefficient) * product over sources of (source coefficient).
- Integrate over the ell loop frequencies.

**Non-stationary systems (time domain):**
- Each edge carries a time difference.
- The integrand involves convolutions of G_{ij}(t) along the diagram structure.
- Integrate over internal time variables.

**Evaluation strategy:**
1. Attempt full symbolic integration. If successful, return the symbolic k-point function correction at this loop level.
2. If symbolic integration is not feasible, report this to the user with an explanation (e.g. "the 2-loop integral involves a non-elementary function of the parameters").
3. If the user supplies numerical parameter values, perform numerical integration. Offer choice of method (adaptive quadrature for low-dimensional integrals, Monte Carlo for higher-dimensional).

---

## Codebase

| File | What it does | Status |
|---|---|---|
| `msrjd/core/field_theory.py` | SageMath expansion framework: namespace builder, Taylor expansion, bigrade classification, `fourier_transform`, `inverse_fourier_transform` | Complete. Core of Phase 1. |
| `msrjd/core/vertices.py` | Vertex/source type extraction from expanded action. `VertexType`, `SourceType` data structures. | Complete. Phase B. |
| `msrjd/core/serialize.py` | Save/load theory to disk (JSON + `.sobj`). | Complete. Phase A. |
| `msrjd/core/cache.py` | Pipeline cache for intermediate results (prediagrams, typed diagrams). | Complete. |
| `msrjd/enumeration/loop_diagram_enumeration.py` | Prediagram enumeration: trees → topologies → oriented DAGs. | Complete. Phase 2 Step 2. |
| `msrjd/diagrams/filter.py` | Filter prediagrams by vertex availability. | Complete. Phase D. |
| `msrjd/diagrams/type_assignment.py` | Enumerate all valid typed assignments (vertex types, edge types, external legs). | Complete. Phase E. Fixed: external legs not permuted, multi-edge support. |
| `msrjd/diagrams/causality.py` | Causality filter: check retarded propagator consistency. | Complete. Phase F. |
| `msrjd/diagrams/symmetry.py` | Symmetry factors, deduplication, coefficient classification. | Complete. Phase G. |
| `msrjd/integration/symbolic.py` | Symbolic integration: frequency conservation, integrand construction, kernel grouping, loop signatures. | Complete. Phase H. Fixed: propagator transposition, conservation for k=1. |
| `models/hawkes_sage.py` | Nonlinear Hawkes 2-population model (quadratic φ). | Complete. Fixed: action sign. |
| `models/hawkes_linear_sage.py` | Linear Hawkes 2-population model (φ(v) = v). | Complete. For validation. |
| `notebooks/hawkes_2pt_pipeline_demo.ipynb` | Full pipeline demo: enumeration → integration → numerical evaluation → simulation comparison. | Complete. Nonlinear model. |
| `notebooks/hawkes_linear_phi_test.ipynb` | Linear model pipeline + simulation validation. | Complete. Tree-level validated. |

---

## Build Order

| Phase | Task | Depends on | Status |
|---|---|---|---|
| **A** | **Serialization:** save/load theory to disk. SageMath `.sobj` for symbolic objects + JSON sidecar for metadata. Design the theory file format. | `field_theory_sage.py` | ✅ Complete |
| **B** | **Vertex decomposition:** new function to split bigrade polynomial sectors into individual typed monomials with field-leg metadata (VertexType / SourceType data structures). | `field_theory_sage.py` | ✅ Complete |
| **C** | **Bridge — max-degree scan & Taylor order feedback:** scan prediagrams for max vertex degree, compare to stored Taylor order, re-expand if needed. Connects enumeration output to theory data. | A, B, `loop_diagram_enumeration.py` | ✅ Complete |
| **D** | **Prediagram filtering:** remove prediagrams with vertex degrees not available in the theory's vertex/source sets. Fast set-membership check. | B, C | ✅ Complete |
| **E** | **Type assignment engine:** enumerate all valid field-type assignments on edges, vertices, and external legs. Constraint-satisfaction over the prediagram structure. This is the hardest algorithmic piece. | B, D | ✅ Complete |
| **F** | **Causality filter:** check retarded propagator consistency and pole-structure compatibility for each typed diagram. | E | ✅ Complete |
| **G** | **Symmetry factor computation:** automorphism group of labeled typed diagrams. | E | ✅ Complete |
| **H** | **Diagram integration — symbolic:** construct and evaluate integral expressions. Frequency domain for stationary systems. Frequency conservation at vertices. | E, G | ✅ Complete |
| **I** | **Numerical integration:** user supplies fundamental parameters; MF solver derives n*, phi derivatives; FFT-based spectral grid + IFT for k≥2; scalar loop integral for k=1. Factored evaluation: precompute unique loop integrands, multiply by external propagators. | H | ✅ Complete (tree validated, loop awaiting validation) |
| **J** | **SageMath-native specification UI:** replace/rework the SymPy Theory Builder with a SageMath-native interface. Lowest priority — the model dict format already works. | A | Not started |

**Detailed outlines for each phase:** see [`BUILD_PHASE_OUTLINES.md`](BUILD_PHASE_OUTLINES.md).

**Parallelism:** A and B are independent and can be built simultaneously. F and G are independent once E is complete. Everything else is sequential along the dependency chain.

---

## Design Decisions & Conventions

- **Fourier transform convention:** angular frequency, no 2π factor. FT: exp(−iωt), IFT: exp(+iωt)/(2π). Gives δ(t) → 1, δ'(t) → iω.
- **Bigrade convention:** (n_tilde, n_phys) where n_tilde = number of response fields, n_phys = number of physical fields. Ring generators ordered [tilde_gens..., phys_gens...].
- **Kernel matrix K:** `S_free = ã^T K a` where ã = response fields, a = physical fields. K has rows = response, cols = physical.
- **Propagator G = K⁻¹:** Same layout (rows=response, cols=physical). `G[resp_i, phys_j]` gives `⟨phys_j resp_i⟩`. **Important:** the *retarded* propagator (how physical field j responds to response-field source i) is `G^R_{j←i} = G[j, i]` — the TRANSPOSED entry. The `_get_propagator_entry` function handles this transposition.
- **External legs are labeled:** Leaf vertex `i` is always assigned `external_fields[i]`. External fields are NOT permuted — permuting would compute a different correlator.
- **MSR-JD action sign:** `S = ñ ṅ − (e^ñ − 1)φ + ṽ[...]`. The Poisson term has a MINUS sign. Saddle condition: ṅ* = +φ(v*) = +n*.
- **Time convention for IFT:** The MSR-JD phase is `exp(+iω(t₁−t₂))`. The IFT naturally gives C(t₁−t₂). We flip the output to get C(t₂−t₁) matching the simulation convention (positive τ = second field later).
- **Vacuum diagrams:** assumed to cancel in normalized correlation functions. Not computed.
- **Connected diagrams only:** enforced by the connectivity requirement in prediagram enumeration.
- **SageMath throughout:** all symbolic computation in SageMath. SymPy used only as a fallback integration backend via `algorithm='sympy'`.
- **Model file structure:** contains fields, response fields, parameters, functions (phi), kernels, operators, MF equations, the full action, concrete phi for numerical evaluation, and specializations. The model declares *what* phi is; the notebook computes derivatives and solves MF equations generically.

---

## Open Questions

1. **Optimal serialization format:** `.sobj` handles SageMath expressions natively but is opaque. Alternative: pickle the model dict + polynomial ring data. Need to test round-trip fidelity with SR expressions, polynomial ring elements, and function symbols.

2. **Degenerate poles:** the residue formula C_k = i * adj(omega_k) / det'(omega_k) assumes simple poles. Near bifurcation points, poles can collide (det'(omega_k) = 0). Should we detect and handle this with higher-order residue formulas, or flag it as an error?

3. **Stationarity detection:** user declares stationarity, but could we also verify it from the action structure? A stationary system has time-translation-invariant kernels, which means K_hat(omega) is the only relevant object (no separate time dependence).

4. **Scaling of type assignment (Step 6):** for theories with many field types and large prediagrams, the combinatorial explosion of valid assignments could be severe. May need pruning heuristics or constraint propagation to keep this tractable.

5. **Multi-dimensional loop integrals:** for ell >= 2, the loop integrals are multi-dimensional. Symbolic evaluation becomes unlikely. Need robust numerical infrastructure (possibly leveraging SageMath's numerical integration or external libraries like Cuba/Vegas).

---

## Directory Structure

### Current layout (flat, ad hoc)

```
Automated Feynman Calculations/
    Enumeration Code/
        loop_diagram_enumeration.py
        enumeration_ui.ipynb
        loop_diagram_enumeration.ipynb
        load_diagram_data.ipynb
        diagrams_k2_ell1/               # saved diagram data (.npz)
    Field Theory Framework/
        field_theory_sage.py
        field_theory.py                  # legacy SymPy version
        field_theory_demo.ipynb          # legacy SymPy demo
        field_theory_sage_demo.ipynb
        models/
            hawkes_sage.py
            hawkes.py                    # legacy SymPy version
    Theory Builder/
        theory_builder.py               # SymPy-based UI (to be reworked)
        theory_builder.ipynb
        theory_builder_ui.ipynb
        hawkes_mf_expansion.ipynb
    Test System Code/
        create_notebook.py
        *.ipynb                          # parameter searches, simulations
    PIPELINE_PLAN.md
    *.pdf, *.docx                        # reference documents
```

### Target layout (consolidated, module-based)

```
Automated Feynman Calculations/
│
├── PIPELINE_PLAN.md                     # this file
│
├── msrjd/                               # main Python/SageMath package
│   ├── __init__.py
│   │
│   ├── core/                            # Phase 1: theory specification & expansion
│   │   ├── __init__.py
│   │   ├── field_theory.py              # FieldTheory class, namespace builder,
│   │   │                                #   Taylor expansion, bigrade classification,
│   │   │                                #   fourier_transform, inverse_fourier_transform
│   │   ├── propagator.py                # K matrix extraction, FT, inverse,
│   │   │                                #   branching logic (delta, residue, direct, freq-domain)
│   │   ├── serialize.py                 # save/load theory files  [Build Phase A]
│   │   └── vertices.py                  # vertex decomposition: bigrade sectors -> typed
│   │                                    #   monomials (VertexType, SourceType)  [Build Phase B]
│   │
│   ├── enumeration/                     # Phase 2 Step 2: prediagram enumeration
│   │   ├── __init__.py
│   │   ├── loop_diagram_enumeration.py  # existing enumeration code (trees, topologies,
│   │   │                                #   prediagrams, isomorphism removal)
│   │   └── degree_scan.py              # max-degree scan across prediagrams,
│   │                                    #   Taylor order feedback  [Build Phase C]
│   │
│   ├── diagrams/                        # Phase 2 Steps 4-8: diagram construction
│   │   ├── __init__.py
│   │   ├── filter.py                    # prediagram filtering by vertex
│   │   │                                #   availability  [Build Phase D]
│   │   ├── type_assignment.py           # combinatorial field-type assignment
│   │   │                                #   engine  [Build Phase E]
│   │   ├── causality.py                 # retarded propagator / pole-structure
│   │   │                                #   consistency check  [Build Phase F]
│   │   └── symmetry.py                  # automorphism group, symmetry factor
│   │                                    #   computation  [Build Phase G]
│   │
│   ├── integration/                     # Phase 2 Step 9: diagram evaluation
│   │   ├── __init__.py
│   │   ├── symbolic.py                  # symbolic integration (freq domain for
│   │   │                                #   stationary, time domain otherwise,
│   │   │                                #   frequency conservation)  [Build Phase H]
│   │   └── numerical.py                 # numerical fallback (quadrature, Monte Carlo,
│   │                                    #   parameter substitution)  [Build Phase I]
│   │
│   └── ui/                              # optional: SageMath-native specification UI
│       ├── __init__.py                  #   [Build Phase J — lowest priority]
│       └── theory_builder.py
│
├── models/                              # model specification dicts
│   ├── __init__.py
│   ├── hawkes_sage.py                   # Hawkes 2-population (reference model)
│   └── ...                              # future models
│
├── notebooks/                           # demo & working notebooks
│   ├── field_theory_sage_demo.ipynb     # Phase 1 demo (propagator pipeline)
│   ├── enumeration_demo.ipynb           # prediagram enumeration demo
│   ├── full_pipeline_demo.ipynb         # end-to-end demo (future)
│   └── ...
│
├── tests/                               # test suite
│   ├── test_field_theory.py             # expansion, bigrade, sanity checks
│   ├── test_propagator.py              # K extraction, FT, inverse, branching
│   ├── test_enumeration.py             # tree/topology/prediagram counts
│   ├── test_vertices.py                # vertex decomposition
│   ├── test_type_assignment.py         # diagram construction
│   ├── test_integration.py            # symbolic + numerical integration
│   └── ...
│
├── saved_theories/                      # serialized theory files (Phase A output)
│   ├── hawkes_2pop.sobj                 # or whatever format we choose
│   └── ...
│
├── legacy/                              # old code kept for reference, not imported
│   ├── field_theory_sympy.py            # old SymPy expansion framework
│   ├── hawkes_sympy.py                  # old SymPy model spec
│   ├── theory_builder_sympy.py          # old SymPy UI
│   └── ...
│
├── test_systems/                        # stochastic simulations for validation
│   ├── create_notebook.py
│   ├── hawkes_2neuron_param_search.ipynb
│   └── ...
│
└── docs/                                # reference documents
    ├── contributing_diagrams.pdf
    ├── feynman_graph_counting.pdf
    └── ...
```

### Key principles

1. **Single importable package (`msrjd/`):** all library code lives here. Notebooks and scripts import from `msrjd`. SageMath `load()` calls are replaced by proper imports (or a thin `load()` wrapper that calls `import`).

2. **Separation of concerns:** `core/` handles the symbolic field theory (action -> propagator). `enumeration/` handles the combinatorial graph theory (prediagrams). `diagrams/` handles the bridge between them (typing, filtering, symmetry). `integration/` handles evaluation. Each subpackage can be developed and tested independently.

3. **Models outside the package:** model specification dicts (like `hawkes_sage.py`) live in `models/` at the top level, not inside the library. Users add new models here without touching library code.

4. **Notebooks outside the package:** all `.ipynb` files live in `notebooks/`. They import from `msrjd` and `models`. No library code lives in notebooks.

5. **Tests mirror the package:** `tests/` has one test file per module. Tests should cover known analytic results (e.g. Hawkes 2-pop propagator poles, known diagram counts for small k and ell).

6. **Legacy quarantine:** old SymPy code moves to `legacy/`. Not deleted (may contain useful reference logic) but not imported by anything.

7. **Saved theories are data, not code:** `saved_theories/` holds serialized output from Phase 1. Gitignored if large.

### Migration plan

The restructuring does not need to happen all at once. Recommended order:

1. Create `msrjd/` and `msrjd/core/`, move `field_theory_sage.py` -> `msrjd/core/field_theory.py`.
2. Move `models/hawkes_sage.py` -> top-level `models/hawkes_sage.py`.
3. Move `loop_diagram_enumeration.py` -> `msrjd/enumeration/loop_diagram_enumeration.py`.
4. Update the demo notebook imports.
5. Move legacy SymPy files to `legacy/`.
6. Create new modules (`propagator.py`, `serialize.py`, `vertices.py`, etc.) as each build phase begins.
7. Move notebooks to `notebooks/` once imports are stable.

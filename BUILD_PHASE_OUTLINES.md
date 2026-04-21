# Build Phase Outlines — Detailed Implementation Plans

**Last updated:** 2026-04-03

Each section below gives a detailed implementation outline for one phase of the build order defined in `PIPELINE_PLAN.md`. Outlines are written before implementation begins and updated as design decisions solidify. Phases A–I are now complete; see `CHANGELOG.md` for critical bug fixes applied 2026-04-03.

---

## Phase A — Serialization ✅ COMPLETE

**File:** `msrjd/core/serialize.py`
**Tests:** `tests/test_serialize.py` (12 tests, all passing)

### What was built

Three public functions:

1. **`save_theory(path, ft, propagator_data, ...)`**
   - Creates a directory with two files:
     - `metadata.json` — plain-Python data (field names, taylor_order, model identity, stationarity, nonzero sectors, propagator info). Callable values stripped via `_strip_callables()`.
     - `symbolic_data.sobj` — SageMath `.sobj` containing all symbolic objects (R, S_raw, by_tp, K_ft, G_ft, adj_ft, D_omega, pole_vals, C_mats, G_t).
   - Stores `model_file` and `model_var_name` paths for re-importing the model dict (since lambdas cannot be pickled).

2. **`load_theory(path)`** → `(meta, data)` dicts.

3. **`reload_model(meta, project_root)`** → re-imports the model dict from the stored `.py` file path via `exec()`.

### Design decisions
- JSON for metadata (human-inspectable, diffable) + `.sobj` for symbolic data (native SageMath serialization).
- Model dict is NOT serialized directly — too many lambdas. Instead, model file path is stored for re-import.

---

## Phase B — Vertex Decomposition

**File:** `msrjd/core/vertices.py`
**Tests:** `tests/test_vertices.py`
**Depends on:** `field_theory.py` (FieldTheory.sectors(), FieldTheory.vertices(), FieldTheory.noise_kernel())

### Goal

Split the polynomial sectors returned by `FieldTheory.vertices()` and `FieldTheory.noise_kernel()` into individual typed monomials. Each monomial becomes a `VertexType` or `SourceType` — the atomic building blocks that Phase E (type assignment) will map onto prediagram vertices.

### Data structures

```python
class VertexType:
    """One monomial from an interacting-action sector (total degree >= 3)."""
    coefficient    : SR           # coupling constant * combinatorial prefactor
    response_legs  : list[tuple]  # [(field_name, population_index), ...]
    physical_legs  : list[tuple]  # [(field_name, population_index), ...]
    bigrade        : tuple[int, int]  # (n_tilde, n_phys)

    @property
    def in_degree(self) -> int:   # n_phys (physical fields = incoming edges)
    @property
    def out_degree(self) -> int:  # n_tilde (response fields = outgoing edges)
    @property
    def total_degree(self) -> int:
```

```python
class SourceType:
    """One monomial from a noise-kernel sector (n_tilde >= 2, n_phys = 0)."""
    coefficient    : SR
    response_legs  : list[tuple]  # all legs are response fields
    bigrade        : tuple[int, int]  # (n_tilde, 0)

    @property
    def out_degree(self) -> int:  # n_tilde
```

### Implementation steps

1. **`decompose_sector(sector_poly, n_tilde, ring_var_names)`**
   - Input: one bigrade sector as a `PolynomialRing(SR, ...)` element, plus the number of tilde generators and the ring variable name list.
   - Iterate over `sector_poly.dict().items()` — each `(ETuple, SR_coeff)` pair is one monomial.
   - For each ETuple, read off which ring generators have nonzero exponents. Map generator index to `(field_name, population_index)` using the ring variable name list.
   - Generators `0..n_tilde-1` are response fields; `n_tilde..` are physical fields.
   - Handle repeated fields: if `nt1` appears with exponent 2, that's two response legs of the same type. Represent as two entries in `response_legs`.
   - Return a list of `VertexType` or `SourceType` objects.

2. **`extract_vertex_types(ft)`**
   - Call `ft.vertices()` to get all sectors with total degree >= 3.
   - For each sector, call `decompose_sector(...)`.
   - Collect into a flat list of `VertexType` objects.
   - Return the list.

3. **`extract_source_types(ft)`**
   - Call `ft.noise_kernel()` to get all (n_tilde >= 2, n_phys = 0) sectors.
   - Decompose each into `SourceType` objects.
   - Return the list.

4. **`available_degrees(vertex_types, source_types)`**
   - Compute the set of `(in_degree, out_degree)` pairs from vertex types.
   - Compute the set of `out_degree` values from source types.
   - Return both sets — used by Phase D for fast filtering.

### Key considerations

- **Exponent multiplicity:** A monomial like `nt1^2 * dn1` has two response legs of type `(nt, 1)` and one physical leg of type `(dn, 1)`. The ETuple exponents encode this directly.
- **Coefficient extraction:** The SR coefficient from `.dict()` already contains parameters, kernel symbols, and any 1/n! factors from Taylor expansion. No further manipulation needed.
- **Ring variable name parsing:** Names follow the pattern `{field_base_name}{population_index}` (e.g., `nt1`, `dn2`). Need to map back to the original field spec name and index. The namespace `ns._ring_var_names` list and `ns._tilde_sr_vars`/`ns._phys_sr_vars` provide this mapping.
- **Ordering:** Leg lists should have a canonical order for later comparison during type assignment.

### Tests

- `test_vertex_type_fields` — decompose a known (2,1) sector monomial, verify response_legs and physical_legs.
- `test_source_type_fields` — decompose a (2,0) noise kernel monomial.
- `test_coefficient_extraction` — verify the SR coefficient matches the original polynomial coefficient.
- `test_exponent_multiplicity` — a monomial with repeated field gets the correct number of legs.
- `test_extract_from_hawkes` — expand Hawkes model, extract all vertex types, verify expected bigrade set matches `ft.vertices()` keys.
- `test_available_degrees` — verify degree sets against manual inspection.
- `test_total_degree` — total_degree property equals sum of legs.
- `test_empty_theory` — a free theory (no interaction terms) returns empty lists.

---

## Phase C — Bridge: Max-Degree Scan & Taylor Order Feedback

**File:** `msrjd/enumeration/degree_scan.py`
**Tests:** `tests/test_degree_scan.py`
**Depends on:** Phase A (serialize), Phase B (vertices), `loop_diagram_enumeration.py`

### Goal

Given a set of prediagrams and a saved theory, determine whether the theory was expanded to a high enough Taylor order. If not, reload the model and re-expand.

### Implementation steps

1. **`max_vertex_degree(prediagrams)`**
   - Input: list of prediagrams from `enumerate_prediagrams(k, ell)`. Each prediagram is a tuple `(D, G, leaves, internal)` where `D` is a SageMath DiGraph.
   - For each prediagram, for each internal vertex `v` (not a leaf), compute `D.in_degree(v) + D.out_degree(v)`.
   - Return the maximum across all prediagrams and all internal vertices.

2. **`check_taylor_order(meta, max_degree)`**
   - Input: metadata dict from `load_theory()` and the max degree from step 1.
   - Compare `meta['taylor_order']` to `max_degree`.
   - Return `(sufficient: bool, current_order: int, required_order: int)`.

3. **`ensure_taylor_order(theory_path, prediagrams, project_root=None)`**
   - High-level convenience function.
   - Load theory metadata. Compute max vertex degree from prediagrams.
   - If current order is sufficient, load and return `(meta, data)`.
   - If not: call `reload_model(meta)` to get the model dict, construct a new `FieldTheory(model, taylor_order=max_degree)`, call `expand()`, re-save via `save_theory()`, then return the new `(meta, data)`.
   - Print a message: "Re-expanding theory from order {old} to {new}..."

4. **`scan_source_vertices(prediagrams)`**
   - Identify source vertices (in_degree = 0, not a leaf) in each prediagram.
   - Return the set of out-degrees needed for source vertices.
   - This is used alongside `max_vertex_degree` to understand what the theory needs.

### Key considerations

- **Prediagram format:** The enumeration code returns `(D, G, leaves, internal)` where `D` is a DiGraph and `leaves`/`internal` are vertex lists. Need to verify this interface — read the enumeration code to confirm the exact return format.
- **Re-expansion cost:** Taylor expansion is fast (seconds for order ~10 in a 2-field system). Re-saving overwrites the theory directory, which is fine.
- **Source vertices vs internal vertices:** In the prediagram, leaves are external legs. Source vertices (no incoming edges, not leaves) need noise-kernel types. Internal vertices with both in and out edges need interaction vertices.
- **Edge case:** If `model_file` is not set in metadata, re-expansion is impossible — raise a clear error.

### Tests

- `test_max_degree_simple` — hand-built prediagram with known vertex degrees.
- `test_check_taylor_order_sufficient` — order 4 theory with max degree 3 → sufficient.
- `test_check_taylor_order_insufficient` — order 2 theory with max degree 4 → insufficient.
- `test_ensure_reexpands` — save a theory at order 2, provide prediagrams needing order 4, verify re-expansion happens and new theory has correct order.
- `test_source_vertex_scan` — verify source vertex identification in a known prediagram.

---

## Phase D — Prediagram Filtering

**File:** `msrjd/diagrams/filter.py`
**Tests:** `tests/test_filter.py`
**Depends on:** Phase B (vertex types → available degree sets), Phase C (Taylor order ensured)

### Goal

Eliminate prediagrams whose vertex degree signatures cannot be matched by any vertex or source type in the theory. This is a fast pass that avoids the expensive type-assignment step for impossible prediagrams.

### Implementation steps

1. **`filter_prediagrams(prediagrams, vertex_types, source_types)`**
   - Compute available interaction degrees: `{(vt.in_degree, vt.out_degree) for vt in vertex_types}`.
   - Compute available source degrees: `{st.out_degree for st in source_types}`.
   - For each prediagram `(D, G, leaves, internal)`:
     - Classify each non-leaf vertex as either source (in_degree = 0) or interaction (in_degree > 0).
     - For source vertices: check `D.out_degree(v) in available_source_degrees`.
     - For interaction vertices: check `(D.in_degree(v), D.out_degree(v)) in available_interaction_degrees`.
     - If all vertices pass, keep the prediagram; otherwise discard.
   - Return the filtered list and optionally the count of discarded prediagrams.

2. **`classify_prediagram_vertices(D, leaves)`**
   - Helper that returns `(source_vertices, interaction_vertices)` for a single prediagram DiGraph, excluding leaves.
   - Source: non-leaf with `D.in_degree(v) == 0`.
   - Interaction: non-leaf with `D.in_degree(v) > 0`.

### Key considerations

- **Speed:** This should be O(V) per prediagram — just set membership checks. No symbolic computation.
- **Leaf vertices:** External legs are leaves. They are not checked against vertex types — they represent the external fields of the k-point function.
- **Degree-2 interaction vertices:** These correspond to (1,1) bigrade — the free action, not the interacting action. These should not appear in prediagrams (the enumeration code requires internal vertices to have degree >= 3). Verify this assumption.
- **No filtering on field types yet:** This step only checks degree signatures, not which specific field types are available. Type-level filtering happens in Phase E.

### Tests

- `test_filter_keeps_valid` — prediagram with degrees matching the theory passes.
- `test_filter_removes_invalid` — prediagram with a degree-5 vertex in a theory with max order 4 is removed.
- `test_filter_checks_sources` — source vertex with out_degree 3 in a theory with only (2,0) noise kernel is removed.
- `test_filter_all_pass` — when all prediagrams are valid, output equals input.
- `test_filter_all_fail` — when no prediagram matches, output is empty.
- `test_classify_vertices` — verify source vs interaction classification on a hand-built DiGraph.

---

## Phase E — Type Assignment Engine

**File:** `msrjd/diagrams/type_assignment.py`
**Tests:** `tests/test_type_assignment.py`
**Depends on:** Phase B (VertexType/SourceType), Phase D (filtered prediagrams)

### Goal

For each filtered prediagram, enumerate all valid fully-typed Feynman diagrams. This is the combinatorial core — the hardest algorithmic piece.

### Data structures

```python
class TypedDiagram:
    """A fully typed Feynman diagram."""
    prediagram       : tuple       # (D, G, leaves, internal)
    vertex_assignments : dict      # {vertex_id: VertexType or SourceType}
    edge_assignments   : dict      # {(u, v): (response_field_leg, physical_field_leg)}
    external_legs      : dict      # {leaf_vertex: (field_name, population_index)}
    propagator_key     : dict      # {(u,v): (resp_idx, phys_idx)} — indices into G_ft
```

### Implementation steps

1. **`assign_external_legs(prediagram, external_fields)`**
   - Input: prediagram and the user's k-point function specification (list of k field variables).
   - Each leaf vertex has exactly one edge (either in or out). The field variable assigned to the leaf determines whether it's a response or physical external leg.
   - Generate all permutations of external_fields → leaf vertices.
   - Filter: if a leaf has only outgoing edges, it must be assigned a response field. If only incoming, a physical field.
   - Return list of valid external-leg assignments.

2. **`assign_vertices(prediagram, vertex_types, source_types, external_assignment)`**
   - For each non-leaf vertex:
     - If source vertex (in_degree = 0): find all `SourceType` with matching out_degree.
     - If interaction vertex: find all `VertexType` with matching `(in_degree, out_degree)`.
   - Take the Cartesian product of choices across all vertices.
   - Return generator of vertex assignment dicts (can be large — use lazy iteration).

3. **`match_legs_at_vertex(vertex, vertex_type, incident_edges, edge_directions)`**
   - The chosen `VertexType` has specific legs: e.g., `response_legs = [(nt, 1), (nt, 2)]`, `physical_legs = [(dn, 1)]`.
   - The vertex has specific edges: some outgoing (carrying response fields), some incoming (carrying physical fields).
   - Enumerate all bijections from the vertex type's response legs to outgoing edges and physical legs to incoming edges.
   - Each bijection assigns a specific field identity to each edge endpoint at this vertex.
   - Return list of valid leg matchings.

4. **`check_propagator_consistency(edge_assignment, G_ft)`**
   - For each directed edge `(u, v)`:
     - The tail (at u) carries a response field with index `i`.
     - The head (at v) carries a physical field with index `j`.
     - Look up `G_ft[i, j]` (or `adj_ft[i, j]`). If the entry is identically zero, this edge cannot carry a propagator — the assignment is invalid.
   - Return True/False.

5. **`enumerate_typed_diagrams(prediagram, external_fields, vertex_types, source_types, G_ft)`**
   - Top-level function combining steps 1–4.
   - For each external leg assignment:
     - For each vertex assignment (lazy):
       - For each consistent leg matching at all vertices:
         - Check propagator consistency.
         - If valid, yield a `TypedDiagram`.

6. **`enumerate_all(prediagrams, external_fields, vertex_types, source_types, G_ft)`**
   - Map `enumerate_typed_diagrams` across all prediagrams.
   - Return flat list (or generator) of all `TypedDiagram` objects.

### Key considerations

- **Combinatorial explosion:** For theories with many field types or large prediagrams, the product over vertices can be huge. Pruning is critical:
  - Prune early: check propagator consistency as soon as two adjacent vertices are assigned, not after the full assignment.
  - Constraint propagation: once an edge's field types are determined by one endpoint, the other endpoint is constrained.
  - Consider implementing as backtracking search with forward checking rather than brute-force Cartesian product.

- **Propagator matrix:** We need a mapping from `(response_field_name, phys_field_name)` to `(matrix_row, matrix_col)`. Build this index map once from the ring variable names and `n_tilde`.

- **Edge direction convention:** Edges go from response field (source) to physical field (sink). The edge `u -> v` means vertex u contributes a response-type leg and vertex v contributes a physical-type leg.

- **Isomorphic typed diagrams:** Two typed diagrams that differ only by a relabeling of internal vertices are isomorphic. Deduplication could happen here or be deferred to Phase G (symmetry). For now, generate all and let symmetry factors account for overcounting.

- **External legs with specific field types:** The user specifies which fields to correlate (e.g., "2-point function of delta_v_1 and delta_v_2"). These constrain which leaves can receive which field assignments.

### Tests

- `test_external_leg_assignment` — 2-point function with 2 leaves, verify both permutations generated.
- `test_vertex_matching_simple` — single interaction vertex with one matching VertexType.
- `test_leg_matching_bijection` — vertex with 2 response legs of different types, verify both orderings tried.
- `test_propagator_zero_rejects` — edge connecting response field i to physical field j where G_ft[i,j] = 0 → rejected.
- `test_full_enumeration_hawkes` — Hawkes 2-pop at k=2, ell=0 (tree level), verify expected number of typed diagrams.
- `test_backtracking_prunes` — verify that adding propagator checks reduces the enumeration count vs brute force.
- `test_source_vertex_assignment` — source vertex matched to SourceType correctly.
- `test_no_valid_assignments` — prediagram where no type assignment works returns empty list.

---

## Phase F — Causality Filter

**File:** `msrjd/diagrams/causality.py`
**Tests:** `tests/test_causality.py`
**Depends on:** Phase E (TypedDiagram)

### Goal

Verify that each typed diagram is consistent with retarded (causal) propagators. The MSRJD formalism uses retarded Green's functions — each propagator G_{ij}(t) is nonzero only for t > 0, meaning the response field event precedes the physical field event.

### Implementation steps

1. **`check_causality(typed_diagram, pole_vals=None)`**
   - The DAG structure of the prediagram already enforces acyclicity — no closed causal loops.
   - For each edge, verify that the assigned propagator component is consistent with retarded boundary conditions.
   - If `pole_vals` is available (Branch 2 of inverse FT): check that all poles for the relevant propagator component lie in the upper half of the complex omega plane (Im(omega_k) > 0). This ensures the contour closure gives a retarded propagator (nonzero only for t > 0).
   - If poles are not available (frequency domain fallback): the causality check is structural only — verify the DAG is a valid causal ordering. Flag that pole-based verification was not possible.
   - Return `(passed: bool, details: str)`.

2. **`check_pole_structure(pole_vals, D_omega, omega)`**
   - Analyze the poles of det(K_hat):
     - For each pole, compute `Im(pole)` symbolically.
     - If all poles have strictly positive imaginary parts, the system is retarded.
     - If some poles have `Im <= 0`, flag the specific poles.
   - Note: pole locations may depend on parameters symbolically. In that case, return a conditional result: "retarded if Im(omega_k) > 0 for all k" with the symbolic conditions.

3. **`filter_causal(typed_diagrams, pole_vals=None)`**
   - Apply `check_causality` to each diagram.
   - Return the list of diagrams that pass.

### Key considerations

- **Symbolic imaginary parts:** For a general theory, poles may be symbolic expressions like `omega = i/tau`. Whether `Im(omega) > 0` depends on the sign of `tau`. SageMath's `imag_part()` may not simplify without assumptions. Strategy: attempt symbolic evaluation; if inconclusive, keep the diagram and annotate with the required parameter conditions.
- **DAG sufficiency:** For many practical cases, the DAG structure alone is sufficient — every prediagram is already acyclic by construction. The pole check is an additional physical consistency verification.
- **Non-stationary systems:** In time-domain, causality is enforced by the retarded kernel directly. The check becomes: is G_{ij}(t) = 0 for t < 0? If we have G_t from inverse FT, this can be verified. Otherwise, trust the formalism.

### Tests

- `test_dag_is_causal` — all prediagrams from enumeration are DAGs → pass structural check.
- `test_upper_half_plane_poles` — Hawkes model poles have positive imaginary parts → pass.
- `test_lower_half_pole_fails` — synthetic pole at `omega = -i` → fail.
- `test_symbolic_pole_conditional` — pole at `omega = i/tau` → conditional pass.
- `test_filter_causal` — mix of passing and failing diagrams filtered correctly.

---

## Phase G — Combinatorial Factor & Field-Type Deduplication ✅ COMPLETE

**File:** `msrjd/diagrams/symmetry.py`
**Tests:** `tests/test_symmetry.py` (18 tests, all passing)
**Depends on:** Phase E (TypedDiagram)

### Goal

Compute the combinatorial factor M for each typed diagram and deduplicate equivalent diagrams. M counts response-leg permutations at each vertex that preserve the `(resp_type, phys_type)` field-type pairing — the within-vertex Wick contractions that give the same integrand by commutativity. The diagram contributes with weight `M × ∏(-vertex_coefficient)` (sign flip for exp(-S) convention).

### What was built

**Field-type deduplication** (`diagram_signature` + `deduplicate_typed_diagrams`):
- Signature encodes, for each internal vertex: the sorted multiset of `(field_type, propagator_index)` for every leaf attached to that vertex, plus vertex type info and internal edges.
- Two TypedDiagrams that differ only by permuting same-type leaves (within or across internal vertices) are merged into a single representative.
- The inter-vertex Wick contractions that get merged here are re-enumerated in Phase J integration.

**Combinatorial factor M** (`_vertex_combinatorial_factor`):
- Pairings: `(resp_type, phys_type)` for every outgoing edge.
- Formula: `M_v = [∏_r n_r! / ∏_p n[r][p]!] × [∏_p m_p!]` where `n_r` = total count of response type r, `n[r][p]` = count of (resp r, phys p) pairings, `m_p` = count of phys type p.
- M counts within-vertex permutations of identical response legs that preserve the field-type pairing — these give the same integrand by commutativity of the product.

**Coefficient classification** (`classify_coefficient_factors`):
- Partitions each diagram's prefactor into scalar (parameter-only) and frequency/time-dependent parts.
- Extracts `scalar_prefactor = M × ∏(-vertex_coefficient)` for each vertex.

### Design rationale (2026-04-14)

An earlier revision (2026-04-13) attempted position-aware dedup to distinguish diagrams where same-type leaves connect to different internal vertices.  This kept 14 TypedDiagrams for k=3 linear Hawkes, but the pipeline's canonical remapping in `integrate_tree_diagram` caused groups of these diagrams to evaluate identically — accidentally double-counting Mapping_1 Wick contractions while missing Mapping_2.

The current design (2026-04-14) puts deduplication at the field-type level (merging to 10 diagrams) and handles inter-vertex Wick contractions explicitly in Phase J integration (see `integrate_tree_diagram` docs).  This is cleaner because:
- The "which dn₁ leg is at which vertex" distinction is handled where it matters — in the integration — rather than duplicated in the diagram list.
- `integrate_tree_diagram` enumerates ALL canonical-to-leaf mappings and sums them, with a compensation factor that removes overcounting of same-vertex permutations (which M already accounts for).
- This correctly gives Mapping_1 + Mapping_2 for Configuration B (cross-vertex) and single-integrand-with-compensation for Configuration A (same-vertex).

### Implementation steps

1. **`build_colored_graph(typed_diagram)`**
   - Construct a vertex- and edge-colored graph from the typed diagram:
     - **Vertex colors:** each vertex is colored by its assigned type (VertexType or SourceType identity + specific leg matching). External legs get their own color (by assigned external field).
     - **Edge colors:** each directed edge is colored by its propagator component `(i, j)`.
   - Return a SageMath `DiGraph` with color data suitable for automorphism computation.

2. **`compute_automorphism_group(colored_digraph)`**
   - Use SageMath's `DiGraph.automorphism_group()` with the coloring partition.
   - The automorphism must preserve: vertex colors (type assignments), edge colors (propagator components), edge directions, and external leg labels (external legs are fixed points).
   - Return the group and its order.

3. **`symmetry_factor(typed_diagram)`**
   - Call `build_colored_graph`, then `compute_automorphism_group`.
   - Return `S = group.order()`.

4. **`compute_all_symmetry_factors(typed_diagrams)`**
   - Map `symmetry_factor` over all diagrams.
   - Return dict `{diagram_index: S}`.

### Key considerations

- **External legs are fixed:** Automorphisms must fix external (leaf) vertices because they represent specific observables. This is enforced by giving each external leg a unique color.
- **SageMath's automorphism machinery:** `DiGraph.automorphism_group(partition=...)` uses `bliss` or `nauty` internally. The partition must separate vertices by color (type). Edges are colored by their propagator type, which must also be respected.
- **Vertex coefficient prefactors:** The 1/n! factors from Taylor expansion are already absorbed into the `VertexType.coefficient`. The symmetry factor S accounts for overcounting from equivalent internal vertex permutations.
- **Tree-level diagrams:** Tree-level (ell=0) diagrams typically have S=1 unless there are identical vertices with identical edge types — a useful test case.
- **Loop diagrams:** Loops can create nontrivial automorphisms (e.g., exchanging the two edges of a self-energy bubble).

### Tests

- `test_tree_symmetry_factor_one` — tree-level diagram with distinct vertex types → S = 1.
- `test_bubble_symmetry_factor` — self-energy bubble (two identical edges between same vertices) → S = 2.
- `test_external_legs_fixed` — verify automorphisms do not permute external legs.
- `test_colored_graph_construction` — verify colors are assigned correctly from TypedDiagram.
- `test_distinct_types_no_symmetry` — all vertices have different types → S = 1.

---

## Phase H — Symbolic Integration ✅ COMPLETE

**File:** `msrjd/integration/symbolic.py`
**Tests:** `tests/test_integration.py`
**Depends on:** Phase E (TypedDiagram), Phase G (symmetry factors)

### Critical implementation notes (from debugging 2026-04-03)

- **Propagator transposition:** `_get_propagator_entry(i, j, ...)` reads `G_ft[j, i]` (transposed). The kernel matrix K has rows=response, cols=physical, so G=K⁻¹ has the same layout. The retarded propagator G^R_{phys_j ← resp_i} = G[j,i]. This transposition is essential for correct amplitudes.
- **Frequency conservation for k=1:** The guard `len(ext_freqs) >= 2` was removed so that overall conservation (ω_ext = 0) is applied for tadpole diagrams.
- **Kernel grouping:** `group_diagrams_by_kernel` groups diagrams by full kernel signature (external + loop). `loop_only_signature` further groups by loop-only part. The factored evaluation precomputes unique loop integrals and multiplies by diagram-specific external propagators.

### Goal

Construct and evaluate the integral expression for each typed diagram's contribution to the k-point function.

### Implementation steps

1. **`build_integrand_stationary(typed_diagram, G_ft, vertex_types, source_types, symmetry_factor)`**
   - **Stationary (frequency domain) case.**
   - Assign a frequency variable `omega_e` to each internal edge.
   - Each directed edge contributes a factor: `G_ft[i, j](omega_e)` — the `(i,j)` propagator evaluated at frequency `omega_e`.
   - Each interaction vertex contributes its `coefficient`.
   - Each source vertex contributes its `coefficient`.
   - Each vertex imposes frequency conservation: `delta(sum_in - sum_out)` where `sum_in` is the sum of frequencies on incoming edges and `sum_out` on outgoing edges.
   - After imposing conservation deltas, eliminate dependent frequency variables. The number of remaining independent integration variables = ell (number of loops).
   - The overall prefactor includes `1/symmetry_factor`.
   - Return `(integrand: SR expression, loop_freqs: list of SR variables, overall_prefactor: SR)`.

2. **`build_integrand_nonstationary(typed_diagram, G_t, vertex_types, source_types, symmetry_factor)`**
   - **Non-stationary (time domain) case.**
   - Assign a time variable `t_v` to each internal vertex.
   - Each directed edge `u -> v` contributes `G_t[i, j](t_v - t_u)`.
   - Each vertex contributes its coefficient.
   - Integration is over all internal time variables: `integral ... dt_1 ... dt_n` from `-oo` to `+oo` (or 0 to oo if retarded propagators are explicit).
   - Return `(integrand, time_vars, prefactor)`.

3. **`integrate_diagram_symbolic(integrand, integration_vars)`**
   - Attempt symbolic integration via SageMath's `integrate()`.
   - Strategy: try each variable sequentially, simplifying between steps.
   - For frequency-domain integrals with rational integrands, try residue theorem directly.
   - Use SymPy backend as fallback: `integrate(..., algorithm='sympy')`.
   - If integration succeeds, return the symbolic result.
   - If integration fails (returns unevaluated integral), return `None` and flag for numerical evaluation.

4. **`frequency_conservation(vertex, incoming_edges, outgoing_edges, freq_vars)`**
   - Build the constraint `sum(freq_vars[e] for e in incoming) == sum(freq_vars[e] for e in outgoing)`.
   - Use these constraints to eliminate `n_vertices - 1` frequency variables (one overall conservation is redundant for connected diagrams), leaving exactly `ell` loop frequencies.
   - Return the substitution dict and the list of independent loop frequencies.

5. **`compute_correction(typed_diagrams, symmetry_factors, G_ft_or_G_t, stationarity, ...)`**
   - Top-level function: sum over all diagrams at a given loop level.
   - For each diagram: build integrand, integrate, multiply by 1/S.
   - Return the total symbolic correction (or a list of per-diagram results if some fail).

### Key considerations

- **Frequency conservation reduces dimensionality:** For a connected diagram with V internal vertices and E internal edges, frequency conservation gives V-1 constraints (one per vertex, minus one for overall conservation). The number of independent loop frequencies is `ell = E - V + 1`, which equals the loop number by Euler's formula.
- **External frequencies:** External legs carry fixed external frequencies (the Fourier-conjugate variables of the k-point function's time arguments). These enter the conservation equations but are not integrated over.
- **Symbolic integration feasibility:** Tree-level (ell=0) has no loop integrals — the result is purely algebraic. One-loop (ell=1) is a single frequency integral of a rational function — residue theorem. Two-loop and above: symbolic evaluation is unlikely for general theories.
- **Residue theorem shortcut:** For stationary one-loop diagrams, the integrand is a rational function of omega. Can use `partial_fraction()` + known residue formulas rather than generic `integrate()`.

### Tests

- `test_frequency_conservation_tree` — tree-level diagram: no loop frequencies, purely algebraic result.
- `test_frequency_conservation_one_loop` — one-loop diagram: verify exactly 1 loop frequency remains.
- `test_integrand_construction` — verify the integrand contains the correct propagator and vertex factors.
- `test_tree_level_evaluation` — tree-level Hawkes 2-point: symbolic result matches manual calculation.
- `test_one_loop_residue` — one-loop self-energy with known poles: verify residue computation.
- `test_nonstationary_time_domain` — time-domain integrand has correct G_t convolution structure.

---

## Phase I — Numerical Integration ✅ COMPLETE (in notebook)

**File:** `msrjd/integration/numerical.py`
**Tests:** `tests/test_numerical.py`
**Depends on:** Phase H (integrand construction)

### Goal

When symbolic integration fails, allow the user to supply numerical parameter values and evaluate loop integrals numerically.

### Implementation steps

1. **`substitute_parameters(integrand, param_values)`**
   - Input: symbolic integrand from Phase H and a dict `{param_name: numerical_value}`.
   - Substitute all symbolic parameters with numerical values.
   - Return a numerical-ready integrand (should be a function of the loop frequencies only).

2. **`numerical_integrate_1d(integrand, loop_freq, bounds=(-oo, oo), method='adaptive')`**
   - For one-loop integrals.
   - Use SageMath's `numerical_integral()` (wraps GSL adaptive quadrature).
   - Handle infinite bounds via variable substitution (e.g., `omega = tan(theta)`).
   - Return `(value, error_estimate)`.

3. **`numerical_integrate_nd(integrand, loop_freqs, bounds, method='monte_carlo')`**
   - For multi-loop integrals (ell >= 2).
   - Options:
     - SageMath's `numerical_integral` with nested calls (low ell).
     - Monte Carlo via SageMath or external library (high ell).
   - Return `(value, error_estimate)`.

4. **`evaluate_diagram_numerical(typed_diagram, param_values, G_ft, symmetry_factor, stationarity)`**
   - Convenience function: build integrand (Phase H), substitute parameters, integrate numerically.
   - Return numerical contribution of this diagram.

5. **`evaluate_correction_numerical(diagrams, param_values, ...)`**
   - Sum numerical contributions over all diagrams.
   - Report per-diagram and total values with error estimates.

### Key considerations

- **Parameter completeness check:** Before numerical evaluation, verify all symbolic parameters have been assigned values. Raise a clear error listing any unsubstituted symbols.
- **Convergence:** Loop integrals may have singularities or slow convergence. Adaptive quadrature handles most one-loop cases. For multi-loop, Monte Carlo with importance sampling may be needed.
- **Complex integrands:** The propagator may have poles in the complex plane. For real-frequency integration, the integrand is real along the real axis (for physical correlation functions), but care is needed near poles.
- **Units/scaling:** The user's parameter values determine the scale of the integral. No automatic unit handling — the user is responsible for consistent units.
- **SageMath numerical tools:** `numerical_integral()` returns `(value, error)`. For complex integrands, integrate real and imaginary parts separately.

### Tests

- `test_substitute_parameters` — all symbols replaced, result is numerical.
- `test_missing_parameter_error` — omitting a parameter raises ValueError with the missing symbol name.
- `test_1d_gaussian` — integrate a known Gaussian → verify against analytic result.
- `test_1d_lorentzian` — integrate 1/(1+omega^2) → pi, verifying against known result.
- `test_hawkes_one_loop_numerical` — one-loop Hawkes correction with specific parameter values, compare to independent numerical calculation.
- `test_error_estimate` — verify error estimate is reasonable (much smaller than the value).

---

## Phase J — Time-Domain Integration ✅ COMPLETE

**Files:**
- `msrjd/integration/time_domain/__init__.py`
- `msrjd/integration/time_domain/pipeline.py` — orchestrator
- `msrjd/integration/time_domain/final_integral.py` — tree integration
- `msrjd/integration/time_domain/propagator_td.py` — G(t) decomposition
- `msrjd/integration/time_domain/subgraph.py` — loop reduction (WIP)

**Depends on:** Phases B, E, G (TypedDiagram, scalar prefactor, combinatorial factor M).

### Goal

Evaluate tree-level k-point functions via direct time-domain integration, bypassing the frequency-domain IFT.  Critical for systems where the frequency-domain propagator has an instantaneous δ component (e.g. the ñ × δn coupling in the MSR-JD Hawkes action), which creates polynomial divergences in the FFT-based path.

### What was built

**Propagator decomposition** (`build_G_t_matrix`):
- `G^R_{p,r}(t) = c_δ · δ(t) + Θ(t) · G^{sm}_{p,r}(t)`
- Smooth part = pole-residue sum `Σ_k C_k[p,r] · exp(I · pole_k · t)` from Phase 1.2 propagator.
- Delta coefficient `c_δ = lim_{ω→∞} Ĝ(ω)` captures instantaneous couplings.
- Heaviside retardation is enforced via polytope bounds during integration (not via symbolic `heaviside(t)`, which `fast_callable` cannot compile).

**Subset enumeration** (`integrate_tree_diagram`):
- For each tree diagram with |E| edges, expand the product `∏_e G^R_e = ∏_e (c_δ δ + Θ G^{sm})`.
- Enumerate `2^|E|` subsets; each subset chooses δ or smooth per edge.
- δ-chosen edges pin integration variables via sifting: `δ(t_head - t_tail) = 0` → t_tail = t_head.  If a δ remains after variable elimination, it's a residual external-time equality (shot noise) — separated into `delta_contributions` for later discrete evaluation on τ grid.
- Smooth-chosen edges contribute `Θ(Δt_e)` constraints, forming a polytope for the remaining integration variables.
- Each subset's integrand is JIT-compiled via `fast_callable` (Sage → CDF) and integrated with `scipy.integrate.quad`/`nquad` using analytical polytope bounds (`_resolve_1d_bounds`, `_integrate_2d_polytope`).

**Inter-vertex Wick contraction enumeration** (added 2026-04-14):
- For correlators with repeated external field types (e.g. two `dn₁` legs at different spacetime points), each distinct canonical-to-leaf mapping is a separate Wick contraction.
- Enumerate all permutations of canonical positions within each same-type field group.  For each mapping, evaluate the integrand with appropriately permuted positional arguments.
- Sum over mappings, divide by compensation factor = product over internal vertices V of `(∏_f n_{V,f}!)`, where `n_{V,f}` is the number of same-type-f leaves at V.
- **Same-vertex case** (e.g. both `dn₁` at one vertex): mappings give same integrand by commutativity.  Compensation removes the overcounting → same result as before.
- **Cross-vertex case** (e.g. one `dn₁` at source, one at interaction): mappings give genuinely different integrands (different time arguments at different vertices).  Compensation = 1 → correct sum of distinct Wick contractions.

**Shot-noise separation**:
- Residual δ on external times → `delta_contributions` list.
- Each entry stores: coefficient callable, linear form `a·τ + c = 0` for the equality, retardation data.
- Evaluated discretely on a τ grid via `eval_delta_contributions_on_tau_grid`.

**Pipeline orchestration** (`compute_correction_td`):
- Iterates over all typed diagrams (tree-level only; loops skipped).
- Collects per-diagram `contribution` callables and δ-contribution structures.
- Returns `total_C(*ext_time_values) → complex` that sums all diagram contributions.

### Critical fixes (chronological)

- **2026-04-08**: 2D polytope bounds sentinel fix (non-star trees were diverging at large |τ|).  Fixed `_resolve_1d_bounds` infeasibility sentinel and 2D polytope bound propagation.
- **2026-04-13**: Leaf-time assignment respects per-diagram leaf-field mapping (was: naive positional mapping that broke for diagrams with same-type leaves at different vertices).
- **2026-04-14**: Inter-vertex Wick contraction enumeration for repeated external field types.  Fixes ~15% theory underestimate at large negative τ for k=3 linear Hawkes with `[dn₁, dn₁, dn₂]` external fields.
- **2026-04-15 (k=4 hardening session)**: four bugs in the polytope integration path fixed together, validated end-to-end at k=4 for all tested external-field configurations:
  - **`_integrate_nd_polytope._make_bound_fn` cross-axis filter**: constraints that still couple to a more-inner axis (j < k_var) are now skipped at the current nesting level and deferred to the deeper level where that axis becomes the resolution target.  Without the filter, `_resolve_1d_bounds` treated such constraints as pure residuals and spuriously declared the polytope infeasible.  First exercised at k=4 with m=3 trees (3 internal vertices).
  - **Heaviside-filtered integrand wrapper** (`_make_heaviside_filtered_integrand`): polytope bounds are now an optimization (tightening the quadrature domain for speed); correctness is guaranteed by an explicit Heaviside product check inside the integrand.  Previously, when deferred constraints forced a ±OUTER_CAP fallback, scipy would sample outside the true polytope where `G^sm(Δt)` GROWS (retarded poles with Im(ω)>0 give `exp(-γ Δt)` which grows for `Δt < 0`), producing cap-sensitive spurious contributions.  The filter zeros out any sample outside the true polytope.
  - **`_integrate_2d_polytope` `pure_s1_found` fix**: the flag was being set as soon as `a_int[0] ≈ 0` without verifying `a_int[1] ≠ 0`.  A pure-external constraint (both coefficients zero) was wrongly flagging, skipping the `OUTER_CAP` fallback and pushing scipy.nquad onto an unbounded outer axis where its tanh-sinh variable transform biases the integral.  At k=4, δ-sifting can pin an integration variable to external times and leave residual pure-external constraints in subsets that reach this path — the triggering condition.
  - **`_two_point` factorial-correction branch** (in `models/cumulant_estimator.py`): for same-pop same-ft='dn' same-lag spike pairs, the subtraction now uses `n(n-1)/dt² − ⟨n/dt⟩²` instead of ordinary `(n−mean)²`.  Matches the 4-point product's factorial-corrected handling of coincident spike legs.  None of the configurations tested to date trigger the path; it's a latent consistency patch.

### Known caveats (not bugs)

- **Bin-averaging bias in the simulation estimator**: the sim measures the BIN-AVERAGED cumulant density over a 4-dimensional box of side `dt_bin`, while theory evaluates the POINT value.  For smooth κ_n with timescale τ, the systematic shift is `O((dt_bin/τ)²)` per axis.  At `dt_bin = 2.0, τ = 10` this is ~4% per axis, compounding to ~10-15% at k=4.  Halving `dt_bin` reduces it ~4×.  The cell 31 default is now `dt_bin = 1.0`.

### Known limitations

- **Loop-level diagrams** (`loop_number > 0`): skipped.  Loop reduction via the frequency-domain Phase 2 pipeline is the fallback.
- **Non-linear retardation**: only handles edges with linear `a·s + b` retardation constraints.  Extension to arbitrary polyhedra would require a proper polytope library.
- **Hand-integration match** (notebook cell 26): analytical polytope bounds match pipeline for 1D integrals (stars).  For 2D/3D integrals, hand-integration via `nquad` is slower and less accurate than the pipeline's unified polytope integrator.  Pipeline values are authoritative.

### Tests

- `tests/test_time_domain.py` — regression tests:
  - `test_phase_J_nd_polytope_preserves_deferred_constraints` — 3D polytope volume with cross-axis constraint (catches `_make_bound_fn` cross-axis-filter regression).
  - `test_phase_J_nd_polytope_simplex_gaussian` — quantitative 3D Gaussian on ordered simplex.
  - `test_phase_J_heaviside_filter_kills_overshoot` — 3D exponential on ordered simplex (catches any Heaviside-filter regression).
  - `test_phase_J_2d_polytope_pure_external_constraint` — 2D box with pure-external residual (catches `pure_s1_found` regression).
  - Plus earlier tests for contribution callable, delta contribution structure, retardation handling, Configuration A/B distinction.

### Validated configurations (as of 2026-04-15)

All within ~1σ of theory/sim = 1.0 with randomized seeds:
- k=2: `[dn₁, dn₂]`, mixed `[dn, dv]`
- k=3: `[dn₁, dn₁, dn₂]`, mixed
- k=4: `[dn₁, dn₂, dv₁, dv₂]` (all-distinct), `[dn₁, dn₁, dn₂, dn₂]` (two same-type pairs), `[dn₁, dn₂, dn₂, dn₁]` (interleaved pairs)

---

## Phase K — SageMath-Native Specification UI

**File:** `msrjd/ui/theory_builder.py`
**Tests:** `tests/test_ui.py`
**Depends on:** Phase A (serialize)

### Goal

Replace the legacy SymPy-based Theory Builder with a SageMath-native interface for specifying new models interactively. Lowest priority — the model dict format already works for programmatic use.

### Implementation steps

1. **`TheoryBuilder` class**
   - Interactive, step-by-step model specification.
   - Methods for each component:
     - `add_field(name, type='response'|'physical', indexed=True, latex=None)`
     - `add_parameter(name, indexed=False, domain=None)`
     - `add_kernel(name, latex=None)`
     - `add_operator(name, type='Dt'|'grad'|'laplacian')`
     - `add_function(name, expression=None, taylor_coeffs=None)`
     - `set_action(action_callable)`
     - `set_mf_conditions(conditions_callable)`
     - `set_index_sets(index_sets_dict)`

2. **`build_model()`**
   - Validate the specification: all referenced fields exist, action is callable, index sets are consistent.
   - Return a model dict in the same format expected by `FieldTheory()`.

3. **`save_model(path)`**
   - Write the model dict to a `.py` file that can be `load()`-ed or imported.
   - The generated file should define a single dict variable (e.g., `MY_MODEL = {...}`).
   - Handles lambda serialization by writing them as `def` functions in the generated file.

4. **Notebook widget interface (optional)**
   - If running in Jupyter/SageMath notebook, provide `ipywidgets`-based UI for field/parameter entry.
   - Low priority — the programmatic API is sufficient.

### Key considerations

- **Validation before build:** Catch common mistakes (typos in field names, missing index sets, action referencing undefined fields) before the user tries to expand.
- **Lambda serialization for save_model:** The model dict contains lambdas (action, mf_bg_conditions, etc.). When saving to a `.py` file, these must be written as named functions. Use `inspect.getsource()` if possible, or require the user to define named functions instead of lambdas.
- **Backward compatibility:** The generated model dict must work with the existing `FieldTheory` constructor without modification.
- **This is lowest priority** because users can already write model dicts directly (as in `hawkes_sage.py`). The UI is a convenience layer.

### Tests

- `test_build_minimal_model` — one field, one parameter, trivial action → valid model dict.
- `test_build_hawkes_equivalent` — reconstruct the Hawkes model via TheoryBuilder, verify it matches `HAWKES_MODEL`.
- `test_validation_missing_field` — action references undefined field → clear error.
- `test_save_and_reload` — save model to .py, reload via `reload_model()`, verify round-trip.
- `test_expand_from_builder` — build model, pass to FieldTheory, expand, sanity check passes.

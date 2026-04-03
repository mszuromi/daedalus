# Changelog

All notable fixes, features, and known issues for the MSR-JD Feynman diagram pipeline.

---

## 2026-04-03 — Critical bug fixes and simulation validation

### Critical fixes (affect all numerical results)

1. **Propagator index transposition** (`msrjd/integration/symbolic.py`)
   - **Bug:** `_get_propagator_entry(i, j, ...)` read `G_ft[i, j]` where `i`=response row, `j`=physical col. But the retarded propagator "response of physical field j to response-field source i" is `G^R_{j←i} = G[j, i]` (transposed).
   - **Impact:** Every diagram integrand used the wrong propagator entries. For asymmetric networks this produced wrong amplitudes (factor ~1.4–5× depending on parameters) and wrong time-domain asymmetry.
   - **Fix:** Transposed the lookup: `G_ft[i, j]` → `G_ft[j, i]`.
   - **Verification:** Pipeline `S₁₂(0)` now exactly matches the analytical formula `[(I − W ĥ)⁻¹ diag(n*) (I − W ĥ)⁻ᵀ]₁₂` (ratio = 1.0000).

2. **Propagator matrix ordering mismatch** (`notebooks/hawkes_*_pipeline_demo.ipynb`, cell 8)
   - **Bug:** Cell 8 hardcoded `resp_names = ['vt1','vt2','nt1','nt2']`, but `build_field_index_map` uses the ring variable ordering `['nt1','nt2','vt1','vt2']`. The kernel matrix `K_ft` rows/cols were permuted relative to what the propagator indices expected.
   - **Impact:** `G_ft[0,0]` was `G[vt1,dv1]` in the matrix but the type assignment thought it was `G[nt1,dn1]`. Produced symmetric integrands (losing cross-correlation asymmetry) and wrong amplitudes.
   - **Fix:** Derive `resp_names` and `phys_names` from `ring_gen_names[:n_tilde]` and `ring_gen_names[n_tilde:]`.

3. **External leg permutation** (`msrjd/diagrams/type_assignment.py`)
   - **Bug:** `enumerate_typed_diagrams` permuted external field assignments across all leaf vertices (`for ext_perm in permutations(...)`). This generated diagrams for all orderings of external fields (e.g., both ⟨dn₁ dn₂⟩ and ⟨dn₂ dn₁⟩).
   - **Impact:** The "swapped" diagrams have opposite imaginary parts, so summing them cancelled the asymmetry, producing a symmetric integrand for cross-correlators.
   - **Fix:** External legs are labeled — leaf `i` always gets `external_fields[i]`. Removed the permutation loop.

4. **Action sign: Poisson term** (`models/hawkes_sage.py`)
   - **Bug:** The MSR-JD action had `+(e^{ñ} − 1)φ` but the correct sign is `−(e^{ñ} − 1)φ`.
   - **Impact:** Flipped the sign of the entire tree-level spectrum. For all-excitatory networks, the cross-correlation was negative (physically impossible).
   - **Fix:** Changed to `−(e^{ñ} − 1)φ` and updated `ndot_bg` from `−n*` to `+n*`.

5. **Conservation equation guard for k=1** (`msrjd/integration/symbolic.py`)
   - **Bug:** `build_integrand_stationary` had `if overall_cons is not None and len(ext_freqs_all) >= 2:` which skipped applying ω_ext = 0 for k=1 tadpole.
   - **Impact:** k=1 diagrams retained a spurious external frequency variable instead of evaluating to a scalar.
   - **Fix:** Removed the `len >= 2` guard.

### Other fixes

6. **Multi-edge support in type assignment** (`msrjd/diagrams/type_assignment.py`)
   - `D.neighbors_out(v)` collapsed multi-edges; switched to `D.outgoing_edges(v)`.
   - Assigned unique integer labels in `orient_edges` to prevent dict key collisions.

7. **k variable shadowing** (multiple notebook cells)
   - Loop variables `for k in ...` overwrote the config `k` (cumulant order). Renamed to `kern`, `idx`, `pk`, `dk` as appropriate.

8. **IFT time convention** (notebook cell 28)
   - The MSR-JD phase is `exp(+iω(t₁−t₂))`, so the natural IFT gives `C(t₁−t₂)`. Flip the output array to get `C(t₂−t₁)` matching the simulation convention (positive τ = second field later).

9. **Simulation covariance normalization** (notebook cell 30)
   - Binned-rate cross-correlation had an extra `1/dt_bin` factor relative to the continuous covariance density. Multiply by `dt_bin`.

10. **Sage Integer/RealNumber contamination** (notebook simulation cell)
    - Sage wraps all numeric literals as `Integer()`/`RealNumber()` which numpy rejects. All values passed to numpy are now explicitly cast via `float()`/`int()`.

### Features added

- **Model-agnostic MF solver** (notebook cell 28): Reads `phi_concrete` from the model, differentiates symbolically to the required Taylor order, solves MF self-consistency equations numerically via `fsolve`. No hardcoded parameter names.
- **Linear Hawkes model** (`models/hawkes_linear_sage.py`): `φ(v) = v` with specializations `phi1=1`, `phi2=...=0`. Vertices arise only from `exp(ñ)` Poisson nonlinearity.
- **Model-specific cache directories**: Cache path includes model name to prevent cross-contamination between models.
- **Adaptive evaluation by k**: k=1 (scalar mean), k=2 (spectrum + IFT), k≥3 (2D slices).
- **Euler-Poisson simulation** for validation against analytical results.

### Known issues / future work

- **Higher-loop evaluation**: The factored evaluation (precompute unique loop integrands, multiply by external propagators) is implemented for k≥2 but not yet verified against simulation for the nonlinear model.
- **Fourier artifacts**: Sharp features near τ=0 (Poisson shot noise delta function) cause Gibbs ringing. Mitigated by increasing `Delta_tau` (finer grid) but not eliminated.
- **`_build_factor_product` in notebook**: The factored loop evaluation uses `G_ft[ri, pi]` directly from `prop_factors` — this needs to be checked against the transposed convention. May need updating for loop-level diagrams.

---

## 2026-03-27 — Initial pipeline build

- Phases A–H implemented: serialization, vertex decomposition, prediagram enumeration, type assignment, causality filter, symmetry/deduplication, symbolic integration, numerical evaluation.
- 118 tests passing.
- Validated on 2-population nonlinear Hawkes process with quadratic φ.

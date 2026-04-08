# Changelog

All notable fixes, features, and known issues for the MSR-JD Feynman diagram pipeline.

---

## 2026-04-07 — k=3 support, residue-based IFT, and structured residue exploration

### Multi-frequency (k=3) numerical evaluation

- **Generalized `spectrum_tree`** to handle multiple external frequencies. For
  k=2 (n_ext=1) returns a 1D array; for k=3 (n_ext=2) evaluates on an N×N grid.
  Falls back to per-slice evaluation when fine 2D grids would be too costly.
- **Generalized `inverse_fourier`** to handle 1D (`ifft`) and 2D (`ifft2`)
  spectra with appropriate `(N·Δω/(2π))^n_ext` scaling.
- **Adaptive grid by k** in cell 28: k=2 uses `T_max=80, Δτ=0.05` (N≈4096);
  k≥3 currently set to the same fine grid (`Δτ=0.02 → N=8192`, ~67M 2D points).
- **k=3 plotting**: extracts 1D slices `C(τ₁, τ₂=0)` and `C(τ₁=0, τ₂)` from the
  full 2D `C(τ₁,τ₂)` surface. Two-panel layout matching the n_τ slices.
- **k=3 simulation cumulant**: For each slice, compute the connected 3rd cumulant
  via FFT — slice 0 cross-correlates `dn_a · dn_c` (product) with `dn_b`, slice 1
  uses `dn_a · dn_b` × `dn_c`. The product trick reduces the 3-point cumulant
  to a 2-point correlation since means are subtracted.
- **Adaptive comparison plot**: simulation cell now reads `external_fields` from
  the config and computes the appropriate auto/cross/3-point statistic. Same
  notebook handles k=1, 2, 3 with no edits.

### Residue-based IFT for k=2 (exact, no Gibbs ringing)

- **`find_spectrum_poles(propagator_data, num_params)`**: Returns all poles of
  the spectrum from the propagator. Poles of det(K(ω))=0 are already known
  symbolically; the spectrum has additional poles at their negatives from
  det(K(−ω))=0. Substitutes parameters for numerical pole values.
- **`compute_numerical_residues(f, poles)`**: Computes residues at simple poles
  via the limit `(z − pole) · f(z)` evaluated at `z = pole + ε`.
- **`ift_via_residues(f, poles, tau_grid)`**: Closes contour in the upper
  half-plane for τ>0 (returns `+i · Σ_upper residue · exp(iωτ)`) and lower
  half-plane for τ<0. Exact, no truncation artifacts, evaluable at any τ.
- **Delta-spike detection**: For auto-correlators, the Poisson shot noise
  contributes `n* · δ(τ)`. This shows up as a constant `S(ω→∞)` with no poles.
  Detected by evaluating `S` at large ω and added as `S∞ / Δτ` at τ=0 to match
  the binned simulation convention.
- **Validation**: For linear Hawkes k=2 cross-correlator, the residue IFT
  matches the FFT IFT to ~0.1% across all τ values (smooth part) and the
  delta-spike heights match exactly when the τ grids share `Δτ`.

### Sequential residue integration prototype (k=3, partial)

Explored a fully exact residue-based path for k=3 ("Path C2"). The goal:
integrate over external frequencies one at a time via residues, eliminating
both FFTs entirely. Two implementations were tested:

1. **Pure-symbolic chained**: Build `J(ω₀, t₂) = i·Σ_upper res(ω₁=p_k(ω₀))
   · exp(i·p_k(ω₀)·t₂)` symbolically, then attempt second integration. Sage's
   `solve()` and `simplify_rational()` choked on rational expressions with
   embedded exponentials — calls timed out at ~10 minutes.

2. **Structured `Term` objects**: Each term tracks `(rational_part, exp_factors)`
   where `exp_factors` is a list of `(linear_combo, time_var)` representing
   `exp(i·linear_combo·time_var)` factors accumulated over residue substitutions.
   The rational part stays rational in the surviving omega vars, so `solve()`
   works at every step. Successfully completed both integrations for k=3 without
   symbolic blowup. Inner integration (over ω₁) yielded 4 upper / 2 lower poles,
   each with the expected shift structure (`p_intrinsic − ω₀` from mixed
   propagator factors, `±p_intrinsic` from single-variable factors).

**Status**: The architecture works (terms propagate cleanly through both
integrations) but the **contour direction logic for the outer integral has
bugs**. The effective time at the outer step is `t₁ + (coefficient_of_ω₀_in_existing_phases)·t₂`,
and different terms have different effective signs depending on (t₁, t₂).
Test values for k=3 were off by varying factors (0.5–0.85) and sometimes wrong
sign. The architecture needs more debugging on the sign accumulation across
the two residue closures.

**Verification at k=2**: The same machinery works perfectly for k=2 (matches
FFT to 0.1%), confirming the basic residue-via-`N(p)/D'(p)` and Term substitution
logic are correct. The k=3 issues are specific to handling the second
contour direction with carried-over exp factors.

### Pipeline architecture

- **Adaptive evaluation cell** (`hawkes_linear_phi_test.ipynb` cell 28): now
  computes residue-based C(τ) alongside FFT-based C(τ) for k=2 and overlays
  both in the comparison plot. Three-curve overlay (sim, FFT-tree, residue-tree).
- **`_param_subs` model-agnostic phi differentiation**: previously hardcoded
  `ns.a[i]` substitutions in the MF solver; now iterates `HAWKES_MODEL['parameters']`
  and substitutes any fundamental parameter into the symbolic phi derivative
  expressions. Works for any phi form without code changes.
- **Cache directory keys**: now include `external_fields` so switching from
  `[(dn,1),(dn,2)]` to `[(dn,1),(dn,1)]` doesn't pull stale diagrams.

### Documentation

- **CHANGELOG.md** updated with all 2026-04-03 critical fixes and 2026-04-07 work
- **PIPELINE_PLAN.md**: status updated to reflect Phases A–I complete; design
  decisions section now documents propagator transposition, external leg labeling,
  action sign convention, and IFT time convention
- **BUILD_PHASE_OUTLINES.md**: Phases H and I marked complete with critical
  implementation notes from debugging

### Known issues / open questions

- **k=3 sequential residue (Path C2)**: contour-direction sign bug, see above.
  Architecture is correct but implementation needs debugging of sign accumulation
  across sequential residue closures with mixed effective times.
- **k≥3 evaluation cost**: full 2D FFT is the only working option, ~67M points
  per evaluation at the current grid. Acceptable but slow.
- **Fourier artifacts at τ≈0**: Sharp features (delta-function shot noise)
  cause Gibbs ringing in the FFT path. Residue path has no ringing for k=2.
- **Time-domain integration not yet attempted**: For systems with known
  symbolic time-domain propagators, direct vertex-time or edge-duration
  integration would sidestep the residue-chasing complexity entirely. See
  user notes in 2026-04-07 design discussion (spanning-tree time reduction,
  V−1 independent time variables, polyhedral integration regions for
  exponential propagators).

### Design discussion: time-domain integration as the primary path

User has proposed shifting the priority of the integration backend.
For stationary connected diagrams, the following equivalence holds:

- **Frequency-space path**: assign edge frequencies → conservation at vertices →
  reduce to k−1 external + ℓ loop frequencies → IFT to time domain.
- **Time-domain path**: assign vertex times → fix one as origin (global
  translation invariance) → use V−1 independent vertex times OR equivalently
  E−ℓ edge durations after applying loop closure constraints.

Both routes give the same number of independent integrations (V−1 for a
connected diagram with V vertices). The time-domain route is preferred when:
- Symbolic G(t) is available (e.g., sums of exponentials, our linear Hawkes case)
- Causality / Heaviside structure restricts the integration region
- Pole structure across multiple frequencies becomes hard to track

The time-domain route gives a polyhedral integration region with products of
exponentials, which often admits exact piecewise formulas. This is the
proposed Phase J for v2 of the framework. Not yet started.

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

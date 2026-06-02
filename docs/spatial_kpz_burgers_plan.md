# KPZ & Burgers vertices — END TO END (June 2026)

> **STATUS: Burgers + KPZ now compile and run end-to-end through
> `compute_cumulants`.** The e2e blocker below (the "v2 k-explicit kernel") was
> resolved by *drift-generalizing the existing heat kernel* (cleaner than a
> separate `propagator_k.py`): the bilinear `Dx` lowers to a bare `GradX`
> symbol, and `extract_mass_diffusion` reads off a **drift** `V` (the `k¹`
> coefficient) instead of rejecting it. For a gradient nonlinearity the only
> `Dx` reaching the bilinear sector is the saddle cross-term `∝ φ*`, so `V → 0`
> at the homogeneous saddle and the propagator is the pure heat kernel. Steps
> 1–4 below are **done**; step 5 (a dedicated KPZ/Burgers simulator) is the
> remaining gold-standard validation.
>
> **Validation (μ=D=T=1, λ=0.3):** tree `C(0,0)=0.50000` (exact, = validated
> Allen-Cahn baseline); 1-loop **mode-dependent**: Burgers (composite)
> `→0.49987`, KPZ (per-leg) `→0.50109` — opposite signs from the distinct
> self-energy structure; `imag_frac=0` (real correlator). Drift heat kernel vs
> analytic advection-diffusion `2.78e-17`; per-leg/composite form factor vs
> brute `∫dℓ` `9.5e-12`. Theories: `theories/{burgers,kpz}_1d.theory.py`;
> tests: `tests/test_propagator_spatial.py` (5 new), `test_full_integrator.py`.

*Branch `spatial-extension`, June 2026.* Target theories with **gradient
nonlinearities**:

- **Burgers** `∂_tφ = ν∇²φ − λφ∂_xφ + η = ν∇²φ − (λ/2)∂_x(φ²) + η` — a `∂_x` on the
  `φ²` *composite*.
- **KPZ** `∂_th = ν∇²h + (λ/2)(∂_xh)² + η` — a `∂_x` on each of the two *physical*
  legs (a genuine per-leg structure, `(∂φ)² ≠ ∂(composite)`).

## What is DONE (the form-factor machinery — validated)

The integrator side is built and validated to machine precision, generic and
non-bespoke:

1. **Per-leg form factors** (`pipeline_bridge.diagram_form_factor(td, chain, mode=)`):
   - `mode='composite'` — chain on the response-leg momentum (`∇²(φⁿ)`, `½∂(φ²)`).
   - `mode='perleg'` — chain on **each incoming physical-leg** momentum
     (`∏_legs i·p_leg`) — the KPZ structure. On a φ̃φ² bubble it yields
     `F = ℓ²q(ℓ−q)` (the `ℓ·(q−ℓ)` dot-product KPZ signature).
2. **Complex form factors** (`∂_x → ik` is imaginary): `_formfactor_average` and
   `_formfactor_callable` now carry `complex`; `diagram_kinematic` returns complex
   when a form factor is present (real otherwise — bit-identical to before). The
   physical `C(q,τ)` is real; the imaginary part cancels in the diagram sum / is
   dropped at the real-space output.
3. **Validated** (`tests/test_full_integrator.test_perleg_and_complex_form_factor`):
   on a φ̃φ² bubble, the per-leg KPZ form factor and a complex odd-`∂` integrand
   both match a brute `∫dℓ` to **9.5e-12**.

So: *given a compiled KPZ/Burgers diagram, the integrator computes it.*

## The remaining blocker — the v2 k-explicit propagator kernel (Phase 3)

KPZ/Burgers cannot yet COMPILE through `compute_cumulants`. Exact failure
(`pipeline/theory_compiler.py:796`):

> `NotImplementedError: operator IR: bilinear 'Dx' has no v1 lowering yet (only Lap/Dt); a k-explicit kernel builder is needed.`

**Why.** Expanding the vertex about the saddle produces a *bilinear* cross-term:
`∂_x(φ²) → 2φ*∂_x(δφ) + ∂_x(δφ²)`, so `φ*·∂_x(δφ)` is a degree-2 (bilinear) `Dx`
generator. The v1 propagator kernel `K(ω,k)` is built from the **inert `Laplacian`
symbol** (`Lap → −k²`, real, even); it has no lowering for `Dx → ik` (odd,
imaginary). Even though that term's coefficient `∝ φ*` vanishes at the homogeneous
saddle `φ*=0`, the kernel is assembled **symbolically** (before the MF solve), and
the heat-kernel mode extraction (`diagonal_modes_from_propagator`, the real
`A + B·k²` split) cannot represent an `ik` term. This is the v2 "momentum-native
kernel" (`docs/spatial_v2_architecture.md` §4, Phase 3 — not built).

## Plan to finish (ordered, each with a validation gate)

1. ✅ **Drift-generalized kernel** (`heat_kernel.py`, *not* a separate
   `propagator_k.py`): the bilinear `Dx` lowers to a bare `GradX` symbol
   (`spatial_operator_ir.GRADX_SYM`, the `∂_x`-analogue of `Laplacian`);
   `extract_mass_diffusion` substitutes `GradX → i·k` and reads the **drift**
   `V` = the `k¹` coefficient (instead of rejecting it), returning `(A, B, V)`.
   `gaussian_heat_kernel`/`image_sum` carry `V` as a Galilean shift
   `x → x − v t` (`v = V/i`); `V=0` is bit-identical to the pure heat kernel.
   *Gate ✅:* a genuine drift `v·∂_xφ` gives `extract → (μ, D, i·v)` and the
   drift kernel matches the analytic advection-diffusion Green's function to
   `2.78e-17`.
2. ✅ **Saddle handling**: the drift is carried *symbolically* (`ac_drift[i]
   ∝ φ*`); at the integrator the bridge substitutes the numeric saddle — for
   `φ*=0` (Burgers/KPZ) `V→0` so `m_k=μ+Dk²` is exact, and a **drift guard**
   raises cleanly if `V≠0` at the saddle (a genuine advection, not yet in the
   integrator). *Gate ✅:* Burgers `ac_drift = I·λ·φ*₁` → 0 at `φ*=0`; KPZ has
   no bilinear `Dx` at all (`∂_x` of the homogeneous mean is 0).
3. ✅ **Per-leg vertex chain through the compiler**: the degree-2/single-type
   gate (`theory_compiler.py`) now allows base-degree 1 (`perleg`) and 2
   (`composite`) and stashes `ns._operator_ir_vertex_mode`. *Gate ✅:* Burgers
   → `'composite'`, KPZ → `'perleg'`.
4. ✅ **Wire + e2e**: `compute_spatial_correlator_generic` reads the mode, passes
   it to `_formfactor_callable`, and takes `Re` at the real-space output (records
   `imag_frac`). *Gate ✅:* Burgers/KPZ `C(x,τ)` run end-to-end, finite, real
   (`imag_frac=0`); tree `0.50000`, 1-loop `0.49987` (Burgers) / `0.50109` (KPZ).
5. ⏳ **Sim validation** (remaining): add the KPZ `(∂_xφ)²` / Burgers `φ∂_xφ`
   forcing to the 1-D spectral simulator (multiply the appropriate spectral
   derivative `ik` per mode) and compare `C(x,0)` / `S(q)`. The equal-time
   variance is a *weak* KPZ observable at small `λ` (the corrections above are
   <0.3%); the discriminating signature is the `q²`-dependence of the 1-loop
   self-energy (it renormalizes `ν=D`). The form-factor machinery is already
   validated vs brute `∫dℓ` to `9.5e-12`, and the Model-B sibling matches the
   sim-validated `loop_dyson` oracle to ~1%.

Steps 1–4 are **done** (June 2026). The drift generalization is non-bespoke: it
makes *any* advection-bearing theory's heat kernel correct (validated at the
oracle level), while the φ*=0 gradient theories (KPZ/Burgers) run fully e2e.

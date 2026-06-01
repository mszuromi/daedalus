# KPZ & Burgers vertices — status + the remaining e2e blocker

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

1. **k-explicit kernel** (`…/spatial/propagator_k.py`, new): build `K(ω,k)` with a
   gradient symbol `ik` (analogous to the `Laplacian` symbol) so `Dt→−iω`,
   `Lap→−k²`, `Dx→ik` all lower. *Gate:* a free theory with a genuine drift
   `v·∂_xφ` reproduces its analytic `G(ω,k)=1/(−iω+ν k²+iv k)`.
2. **Saddle substitution before mode extraction**: substitute `φ*` into `K` first,
   so for `φ*=0` the spurious `φ*·ik` bilinear drops and the propagator is the
   real heat kernel again. *Gate:* Burgers/KPZ at `φ*=0` give the same free
   propagator as `ν∇²`, no `ik` survives.
3. **Per-leg vertex chain through the compiler**: drop the degree-2/single-type
   gates (`theory_compiler.py:767-782`); stash, per vertex generator,
   `(base-degree → mode, chain)` so `_formfactor_callable` is called with
   `mode='perleg'` for base-degree-1 generators (KPZ) and `'composite'` for
   base-degree-≥2 (Model B / Burgers). *Gate:* the enumerated KPZ vertex extracts
   `F = ∏_legs i·p_leg`.
4. **Wire + e2e**: `compute_spatial_correlator_generic` picks the mode per diagram
   and applies the (now complex) form factor; take `Re` at the real-space output.
   *Gate:* KPZ/Burgers `C(q,τ)` runs end-to-end, finite.
5. **Sim validation**: add the KPZ `(∂_xφ)²` / Burgers `φ∂_xφ` forcing to the 1-D
   spectral simulator (multiply the appropriate spectral derivative per mode) and
   compare `C(x,0)` / `S(q)` (the KPZ coupling renormalizes `ν`; the 1-loop
   self-energy `∝ q²` correction is the signature).

Steps 1–2 are the substantial piece (a real propagator-kernel change touching the
validated Lap-based spatial propagator — done carefully, behind the v2 gate, not
disturbing existing theories). 3–5 reuse the validated form-factor machinery above.

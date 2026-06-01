# Spatiotemporal convolutions in SPDEs — forms & pipeline integration

*Branch `spatial-extension`, June 2026. A survey of the convolution / nonlocal
interaction forms that appear in stochastic PDEs, and how each maps onto the
Daedalus MSR-JD pipeline. The unifying observation — and the reason much of this
is **already supported** — is that in the momentum–frequency representation every
convolution is a multiplicative factor, i.e. a **form factor** `𝔣(k,ω)`.*

## The unifying principle

A convolution in real space is a multiplication in Fourier space:

```
(w ⊛ φ)(x,t) = ∫ w(x−x', t−t') φ(x',t') dx' dt'   ⟶   ŵ(k,ω) · φ̂(k,ω).
```

The MSR-JD pipeline is **momentum–frequency native** in exactly the places this
matters:

1. the **propagator** is built as `K(ω,k)`, `G = K⁻¹` — a convolution in the
   *linear* dynamics just adds `ŵ(k,ω)` to `K`;
2. an interaction **vertex** carries per-leg **form factors** `𝔣_ℓ(k)` (the
   machinery added for derivative/∇ vertices, June 2026) — a convolution on a
   vertex leg is the form factor `𝔣 = ŵ(k)`;
3. the loop integral averages the product of leg form factors over the
   loop-momentum Gaussian by **Gauss–Hermite**, which is *exact for polynomials
   and fast-converging for any smooth kernel* (validated below).

So "integrate a convolution" reduces to "supply `ŵ(k,ω)`" — the same three seams
(`form_factor`, `propagator_k`, `temporal_integrate`) the spatial-v2 architecture
already exposes (`docs/spatial_v2_architecture.md`).

## The forms (with where they appear)

| Form | Real-space | Fourier `ŵ` | Physics |
|---|---|---|---|
| **Spatial-convolution vertex** | `f(φ) ⊛ w(x)` (e.g. `∫w(x−x')f(φ(x'))dx'`) | `ŵ(k)` on the leg | **neural fields** (Amari/Wilson–Cowan), nonlocal Allen–Cahn nonlinearity |
| **Nonlocal linear diffusion** | `∫w(x−x')[φ(x')−φ(x)]dx'` | `μ + (ŵ(0)−ŵ(k))` in `K` | nonlocal Cahn–Hilliard, aggregation–diffusion |
| **Fractional Laplacian** | `(−∇²)^α φ` | `D|k|^{2α}` in `K` | anomalous / Lévy diffusion, fractional Allen–Cahn |
| **Temporal memory kernel** | `∫K(t−t')(…)dt'` | `K̃(ω)` | delayed neural fields, non-Markovian / mode-coupling |
| **Full spatiotemporal kernel** | `G(x−x', t−t')` | `G̃(k,ω) = ŵ(k)·K̃(ω)` (separable) | retarded nonlocal coupling |

Representative kernels: Gaussian `w(x)=e^{−x²/2σ²}/√(2πσ²) → ŵ(k)=e^{−σ²k²/2}`;
exponential `w(x)=e^{−|x|/d}/2d → ŵ(k)=1/(1+d²k²)` (Lorentzian); Mexican-hat
(difference of Gaussians); power-law `|k|^{2α}`.

## What the pipeline supports today

| Case | Status | Mechanism |
|---|---|---|
| **Spatial-convolution *vertex*, smooth `ŵ(k)`, at 1-loop** | ✅ **integrator-ready** (authoring TODO) | `full_integrator._formfactor_average` takes an arbitrary callable `F(ℓ,q)`; a convolution kernel is `F=ŵ(ℓ)`. **Validated** vs brute `∫dℓ`: Gaussian → 1e-11, Lorentzian → 5e-5, `1/(1+d²k²)²` → 3e-4 (Gauss–Hermite `gh_order≈16`). Needs a `Conv(field, kernel)` author surface + a `ŵ(k)` form-factor extractor (the IR already reserves a `Conv` node). |
| **Tree-level, ANY linear convolution / fractional Laplacian** | ✅ supportable | the heat-kernel propagator `G_R(k,t)=e^{−m_k t}` with `m_k = μ + ŵ(k)` is exact for any `ŵ(k)`; the q→x FT is numeric. Only the propagator builder needs to accept a general `m_k`. |
| **Temporal memory kernel = sum of exponentials** | ✅ **already built** | the Markovian-embedding preprocessor (`pipeline/colored_to_markovian.py`) turns `e^{−t/τ}` memory into an auxiliary field — exact. (Built for colored noise; same trick for memory kernels.) |
| **Loops with a NON-Gaussian `m_k`** (fractional `|k|^{2α}`, generic nonlocal `K`) | ⚠️ **gap** | the analytic Symanzik reduction assumes `m_k = μ + Dk²` (Gaussian in `k`), so `∫dᵈℓ` is closed-form. A non-quadratic `m_k` makes the loop non-Gaussian ⇒ the analytic step breaks. Needs a **direct momentum quadrature** at loop level (a Regime-1 cutoff grid — the `d≥2` path already does direct `∫dᵈℓ`, so the seam exists). |
| **Non-separable `G̃(k,ω)`** | ⚠️ research | couples the spatial form factor and the temporal backend; fine for separable `ŵ(k)K̃(ω)`, open otherwise. |

## Validation (this session)

The decisive new fact: the form-factor loop average is **not limited to the
polynomial (derivative-vertex) case** — it integrates any smooth `ŵ(ℓ)` over the
loop Gaussian by Gauss–Hermite. On the standard bubble routing:

```
kernel  ŵ(k)                       GH(n=6)   GH(n=16)   vs brute ∫dℓ
Gaussian  e^{−σ²k²/2}              3.4e-6    1.4e-11    (exact: Gaussian×Gaussian)
Lorentzian 1/(1+d²k²)             1.5e-2    7.0e-5
rational   1/(1+d²k²)²            6.1e-2    2.3e-4
```

Gaussian kernels are essentially exact; rational/decaying kernels converge to
1e-4 with `gh_order≈16` (vs the `gh_order=6` that is exact for the low-degree
derivative-vertex polynomials). So **convolution vertices need only a larger
`gh_order`** — no new integrator.

## Roadmap to first-class support

1. **`Conv(field, kernel)` IR node + `ŵ(k)` form factor.** The operator IR already
   parses a `Conv` token; add its `form_factor → ŵ(k)` (a registry of named
   kernels: `gaussian(σ)`, `exponential(d)`, `mexican_hat`, `power(α)`), and route
   it through `_formfactor_callable` exactly like the `Lap`/`Dx` chain. Bump the
   default `gh_order` when a non-polynomial kernel is present.
2. **General `m_k` in the propagator** (tree + the linear-nonlocal/fractional case):
   let `build_propagator` accept `m_k = μ + ŵ(k)` symbolically; tree-level then
   works for any `ŵ`.
3. **Direct loop momentum quadrature** for non-Gaussian `m_k` (fractional Laplacian
   loops): a `spatial_reduce` strategy that does `∫dᵈℓ` numerically on a
   cutoff grid instead of the Gaussian Symanzik — reuse the `d≥2` direct-`∫dᵈℓ`
   path. This is the one genuine integrator extension.
4. **Sim cross-check:** a neural-field (Gaussian/Mexican-hat `w`) or nonlocal
   Allen–Cahn theory + a convolution-aware simulator (spectral: multiply by `ŵ(k)`
   per mode — trivial in the existing pseudo-spectral sim).

## Sources

- Neural field / nonlocal interaction kernels: [Numerical solution of the stochastic neural field equation](https://www.sciencedirect.com/science/article/abs/pii/S0378437122001741); [Large Deviations for Nonlocal Stochastic Neural Fields](https://pmc.ncbi.nlm.nih.gov/articles/PMC3991906/); [Dynamics of neural fields with exponential temporal kernel](https://link.springer.com/article/10.1007/s12064-024-00414-7) (the "nonlocal ≡ higher spatial derivatives ≡ spatial integrals ≡ auxiliary diffusive fields" equivalence).
- Nonlocal / fractional phase-field: [A nonlocal stochastic Cahn–Hilliard equation](https://arxiv.org/pdf/1510.07923); [Non-local Allen–Cahn with rough kernels](https://arxiv.org/pdf/1510.02812); [Fractional Cahn–Hilliard, Allen–Cahn and porous-medium equations](https://arxiv.org/pdf/1502.06383); [From nonlocal to local Cahn–Hilliard](https://arxiv.org/pdf/1803.09729).

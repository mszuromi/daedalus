# Notebooks

Organized by era and domain. **Pipeline-era** notebooks call `compute_cumulants`
(from `pipeline/`) and compare the diagrammatic theory against a matched simulator;
each is self-contained (build theory → compute orders → simulate → plot).

Every pipeline-era notebook opens with a **depth-robust root cell** (walks up to the
`pipeline/` package, then `chdir`s back to `notebooks/`), so it runs correctly from
these subdirectories and its relative data paths still resolve.

## `spatial/` — spatial field theories (Laplacian; `C(x, τ)` correlators)

| Notebook | What |
|---|---|
| `pipeline_allen_cahn_1d_full_loop_sim_compare` | φ⁴ Allen-Cahn, d=1. Config cell `MAX_ELL ∈ {0,1,2}`; tree → 1-loop → 2-loop vs SPDE sim, with cumulative per-loop progression and a `VERBOSE` staged `[1/7]…[7/7]` trace. |
| `pipeline_allen_cahn_quintic_1d_full_loop_sim_compare` | **φ⁶ generalization test** (Allen-Cahn + `−γφ⁵`). The new `φ̃φ⁵` degree-6 vertex first enters at 2 loops; default `MAX_ELL=2`. |
| `pipeline_linear_diffusion_1d_sim_compare` | free 1D diffusion (Gaussian `C₀` check). |
| `pipeline_linear_field_2d_sim_compare` | free 2D field. |
| `pipeline_reaction_diffusion_2d_loop_sim_compare` | 2D reaction-diffusion, 1-loop. |
| `pipeline_reaction_diffusion_conserved_1d_sim_compare` | **Model B** — conserved `∇²(φ²)`, the **composite-∇** derivative vertex (`mode='composite'`, `F∝q²`; conservation-suppressed variance). |
| `pipeline_burgers_1d_sim_compare` | **Burgers** — `−(λ/2)∂ₓ(φ²)`, the **composite-∂ₓ** vertex (imaginary form factor `∂ₓ→ik`; saddle-drift heat kernel). |
| `pipeline_kpz_1d_sim_compare` | **KPZ** — `(λ/2)(∂ₓh)²`, the **per-leg-∂ₓ** vertex (`mode='perleg'`, `F=ℓ²q(ℓ−q)`); includes the excess-velocity `v∞` cross-check (~1%). |
| `pipeline_kpz_2d_sim_compare` | **KPZ in d=2** — `(λ/2)(∇h)²=Σ_i(∂_i h)²`: exercises the d≥2 form-factor machinery (the `L·d`-dim transverse-moment GH average, validated vs brute `∫d²ℓ` to 1e-14) + a 2-D simulator; excess-velocity check ~0.1%. |
| `pipeline_combined_allencahn_modelb_kpz_1d_sim_compare` | **Combined model (d=1)** — `−λφ³ + g∇²(φ²) + (κ/2)(∂ₓφ)²` (Allen-Cahn ⊕ Model B ⊕ KPZ). Now **computes** through the per-vertex form-factor table (3 vertex types, mixed composite/per-leg modes in one theory) + the full 3-D SPDE sim. |
| `pipeline_combined_allencahn_modelb_kpz_2d_sim_compare` | **Combined model in d=2** — the same 3 vertices at two dimensions (per-vertex table + `L·d`-dim transverse-moment loop average + 2-D sim). Runs fast (~65s total); 1-loop is UV-divergent (same-signature Model B × KPZ cross, milder than d=3), reported honestly. |
| `pipeline_coupled_rd_2species_1d_sim_compare` | **Coupled 2-species RD** — matrix reaction coupling (complex mode pair, damped predator–prey), cubic nonlinearities: tree (matrix-Lyapunov analytic IFT) + 1-loop (spectral-assignment driver) for `C_aa` AND the τ-asymmetric cross-correlator `C_ab`, vs the 2-species SPDE sim; plus the **unequal-D Dyson-dressed tree** with the truncation ladder N=0→1→3 converging geometrically (5.7e-2 → 4.6e-3 → 5.3e-5) to the exact matrix-heat-kernel oracle. |
| `pipeline_combined_allencahn_modelb_kpz_3d_sim_compare` | **Combined model in d=3** — the same 3 vertices at three spatial dimensions: stresses the per-vertex table + the `L·d`-dim transverse-moment loop average + the 3-D simulator all at once. Runs end-to-end (tree ~2s, 1-loop ~150s); the 1-loop is UV-divergent (the same-signature Model B × KPZ cross), reported honestly (off-scale, cutoff-set). |

The notebooks cover one representative per model class: **free** (linear diffusion / 2D field),
**polynomial reaction `φⁿ`** (Allen-Cahn φ⁴/φ⁶, 2D reaction-diffusion), **composite-derivative**
`∇²(φⁿ)`/`∂ₓ(φ²)` (Model B, Burgers), and **per-leg-derivative** `(∂ₓφ)²` (KPZ).

See [`docs/spatial_pipeline.md`](../docs/spatial_pipeline.md) and, for the gradient vertices,
[`docs/spatial_kpz_burgers_plan.md`](../docs/spatial_kpz_burgers_plan.md).

## `temporal/` — time-only theories (OU, Hawkes, neural), via the ω-domain Phase J

- `pipeline_ou_quartic_*`, `pipeline_ou_sextic_*` — nonlinear Ornstein-Uhlenbeck (white, colored, 2D, cross-correlated noise).
- `pipeline_singlepop_*`, `pipeline_multipop_*` — single/multi-population neural (quadratic/cubic, spike-reset).
- `pipeline_conductance_synapse_compare`, `pipeline_dendritic_linear_sim_compare` — neural variants.
- `pipeline_quad_expg_*` — quadratic Hawkes with exponential filter.
- `pipeline_bistable_mf_demo` — multi-root mean-field + linear-stability classification.
- `pipeline_linear_delta_third_moment_compare`, `pipeline_demo` — third-moment / general pipeline demo.

## `legacy/` — pre-pipeline experiments (archival)

Early diagram-enumeration and time-domain Hawkes exploration from before the
`pipeline/` API (`enumeration_*`, `field_theory_sage_demo`, `load_diagram_data`,
`hawkes_td_*`). Kept for provenance; **several import now-removed modules and will not
run as-is.**

## (root) — UI tooling

- `theory_builder.ipynb` — interactive theory-input UI.
- `theory_runner.ipynb` — run a saved theory through the pipeline.

`saved_results/` and `saved_theories/` are shared output directories (their contents
are gitignored).

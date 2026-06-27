# Example notebooks

Ten curated examples, each headlining **one** capability of the pipeline and together spanning
its range — temporal SDEs to spatial SPDEs, single- to multi-field, Gaussian / colored /
correlated / point-process noise, and the spatial vertex classes.

Each example follows the same three steps:

1. **The model** — `dd.load_theory(...)` + `dd.describe_model(...)` print the structure.
2. **The pipeline → cumulants** — one `dd.run(...)` drives the chain (enumerate → propagator →
   mean-field → loop integrals → cumulant); the plot is the theory alone.
3. **Independent simulation** — a from-scratch simulator overlaid on the pipeline curve as an
   external check (not part of the pipeline).

They are split into [`SDE/`](SDE) (temporal — SDEs / point processes) and [`SPDE/`](SPDE) (spatial PDEs).

## Temporal (SDE / point process)

| Notebook | Capability |
|---|---|
| [`OU_quartic_white`](SDE/OU_quartic_white.ipynb) | White Gaussian noise — single-field quartic OU; full 2-loop `C(τ)`. |
| [`OU_sextic_white`](SDE/OU_sextic_white.ipynb) | A φ⁶ nonlinearity — the degree-6 vertex. |
| [`OU_quartic_correlated_colored`](SDE/OU_quartic_correlated_colored.ipynb) | Two coupled OU fields with **correlated, colored** noise — Markovian embedding of a finite-τc cross-correlated drive; nonzero cross-correlator. |
| [`Hawkes_quadratic_alpha`](SDE/Hawkes_quadratic_alpha.ipynb) | A nonlinear **Hawkes point process** with an α-function synaptic kernel (a rise-then-decay convolution). |
| [`Dendritic_nonlinear`](SDE/Dendritic_nonlinear.ipynb) | Two-compartment neuron — quadratic soma + sigmoidal (probabilistic) dendrite; a non-Markovian multi-field DAE. |

## Spatial (SPDE)

| Notebook | Capability |
|---|---|
| [`Allen_Cahn_phi4`](SPDE/Allen_Cahn_phi4.ipynb) | **Polynomial vertex** φ⁴ — the spatial MSR-JD machinery (heat kernels, Symanzik / causal-chamber loop integrals). |
| [`KPZ`](SPDE/KPZ.ipynb) | **Per-leg gradient vertex** `(∂ₓh)²` — a `∂ₓ → ik` form factor on each leg; excess-velocity cross-check. |
| [`Model_B_conserved`](SPDE/Model_B_conserved.ipynb) | **Composite-∇² vertex** `∇²(φ²)` — conserved order parameter, q²-suppressed variance. |
| [`RD_2D`](SPDE/RD_2D.ipynb) | **d=2 with a UV-finite loop** — the cubic-vertex bubble converges for d < 4; no cutoff. |
| [`RD_2_species`](SPDE/RD_2_species.ipynb) | **Coupled multi-field** — matrix reaction coupling → auto + cross correlators; Dyson dressing for unequal diffusion. |

## What each notebook loads

Each example loads its theory by name (`dd.load_theory(...)`) and overlays a matched
simulator. The simulators are deliberately shared where the physics is the same — one OU
simulator serves all three OU notebooks, one 1-D spatial simulator serves the three 1-D
spatial notebooks.

| Notebook | theory (`theories/…`) | simulator (`simulations/…`) |
|---|---|---|
| `OU_quartic_white` | `ou_quartic` | `ou_langevin_sim_numba` |
| `OU_sextic_white` | `ou_sextic` | `ou_langevin_sim_numba` |
| `OU_quartic_correlated_colored` | `ou_quartic_two_dim_color_corr` | `ou_langevin_sim_numba` |
| `Hawkes_quadratic_alpha` | `quadratic_hawkes_alpha` | `hawkes_sim_multipop_numba` |
| `Dendritic_nonlinear` | `dendritic_quad_soma_sigmoid` | `dendritic_quad_sigmoid_sim_numba` |
| `Allen_Cahn_phi4` | `allen_cahn_1d_subcritical_infinite` | `spatial_field_1d_sim` |
| `KPZ` | `kpz_1d` | `spatial_field_1d_sim` |
| `Model_B_conserved` | `reaction_diffusion_conserved_1d` | `spatial_field_1d_sim` |
| `RD_2D` | `reaction_diffusion_2d` | `spatial_field_2d_sim` |
| `RD_2_species` | `coupled_rd_2species_1d` | `coupled_rd_1d_sim` |

(The temporal sims also import `cumulant_estimator`, a small correlator helper.)

## Notes

- **Connected cumulants.** The pipeline produces *connected* correlators; the simulator cells
  subtract the mean where it is nonzero before comparing.
- **Speed.** Simulations use reduced Monte-Carlo / SPDE settings so each notebook runs in
  roughly 10–60 s — showcases, not precision benchmarks. `numba` is required for the simulation
  cells (see the top-level [README](../../README.md)).

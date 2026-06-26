# Example notebooks

Ten curated examples, each headlining **one** capability of the pipeline and together spanning
its range — temporal ODEs to spatial PDEs, single- to multi-field, Gaussian / colored /
correlated / point-process noise, and the spatial vertex classes.

Each example follows the same three steps:

1. **The model** — `dd.load_theory(...)` + `dd.describe_model(...)` print the structure.
2. **The pipeline → cumulants** — one `dd.run(...)` drives the chain (enumerate → propagator →
   mean-field → loop integrals → cumulant); the plot is the theory alone.
3. **Independent simulation** — a from-scratch simulator overlaid on the pipeline curve as an
   external check (not part of the pipeline).

## Temporal (ODE / point process)

| Notebook | Capability |
|---|---|
| [`OU_quartic_white`](OU_quartic_white.ipynb) | White Gaussian noise — single-field quartic OU; full 2-loop `C(τ)`. |
| [`OU_sextic_white`](OU_sextic_white.ipynb) | A φ⁶ nonlinearity — the degree-6 vertex. |
| [`OU_quartic_correlated_colored`](OU_quartic_correlated_colored.ipynb) | Two coupled OU fields with **correlated, colored** noise — Markovian embedding of a finite-τc cross-correlated drive; nonzero cross-correlator. |
| [`Hawkes_quadratic_alpha`](Hawkes_quadratic_alpha.ipynb) | A nonlinear **Hawkes point process** with an α-function synaptic kernel (a rise-then-decay convolution). |
| [`Dendritic_nonlinear`](Dendritic_nonlinear.ipynb) | Two-compartment neuron — quadratic soma + sigmoidal (probabilistic) dendrite; a non-Markovian multi-field DAE. |

## Spatial (PDE)

| Notebook | Capability |
|---|---|
| [`Allen_Cahn_phi4`](Allen_Cahn_phi4.ipynb) | **Polynomial vertex** φ⁴ — the spatial MSR-JD machinery (heat kernels, Symanzik / causal-chamber loop integrals). |
| [`KPZ`](KPZ.ipynb) | **Per-leg gradient vertex** `(∂ₓh)²` — a `∂ₓ → ik` form factor on each leg; excess-velocity cross-check. |
| [`Model_B_conserved`](Model_B_conserved.ipynb) | **Composite-∇² vertex** `∇²(φ²)` — conserved order parameter, q²-suppressed variance. |
| [`RD_2D`](RD_2D.ipynb) | **d=2 with a UV-finite loop** — the cubic-vertex bubble converges for d < 4; no cutoff. |
| [`RD_2_species`](RD_2_species.ipynb) | **Coupled multi-field** — matrix reaction coupling → auto + cross correlators; Dyson dressing for unequal diffusion. |

## Notes

- **Connected cumulants.** The pipeline produces *connected* correlators; the simulator cells
  subtract the mean where it is nonzero before comparing.
- **Speed.** Simulations use reduced Monte-Carlo / SPDE settings so each notebook runs in
  roughly 10–60 s — showcases, not precision benchmarks. `numba` is required for the simulation
  cells (see the top-level [README](../../README.md)).

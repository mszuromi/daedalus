#### Spatial routes: `SPATIAL_INTEGRATOR` is never ignored silently

`SPATIAL_INTEGRATOR=mc` / `bessel` (with `SPATIAL_MC_N`) selects a backend of the single-field k = 2 analytic-IFT
route only. Several other spatial routes have only the deterministic product grid, and they used to ignore the
switch without a word:
- the coupled multi-field route (spectral assignments; k = 2 and k-point), e.g. two-species reaction-diffusion at
  d = 1. Its results were identical with and without `mc`;
- the single-field k-point route;
- the numerical-FT route (derivative vertices at d ≥ 2, or `SPATIAL_FORCE_NUMERICAL_FT=1`). It also recorded
  `integrator='mc'`, so the result was labelled a Monte-Carlo estimate although it was the grid value.

These routes still run the grid, with the same values bit for bit. Now:
- a `SpatialIntegratorWarning` names the route and says that the grid is used and that `SPATIAL_MC_N` is ignored.
  It is shown even under the Sage launcher's `PYTHONWARNINGS=ignore`;
- the result's info records `integrator='grid'` and `integrator_requested`;
- an unknown `SPATIAL_INTEGRATOR` value raises `ValueError` on every route, as it already did on the k = 2 route.

#### Coupled spatial loops: the memory refusal comes first and names a grid that fits

Each coupled loop chamber was checked only when it was about to run, so a two-loop refusal came after the whole
one-loop pass. Now every chamber of every loop diagram is checked before any is evaluated.

The refusal (`SpatialMemoryBudgetError`) now gives the grid in use and the finest common
`SPATIAL_GRID_NT = SPATIAL_GRID_NS` that fits the budget (down to 8), with its estimate. It also says that `mc` does
not apply to this route. Example: a two-species cubic model at d = 1, two loops, 5 output points, under the default
6 GB budget. It is refused in about 4 s:
- the grid in use (n_t = 16, n_s = 14) needs about 19.7 GB per chamber;
- `SPATIAL_GRID_NT=11 SPATIAL_GRID_NS=11` needs about 6.0 GB and fits.

A coarser grid trades accuracy. The message asks for a comparison with a second, coarser grid.

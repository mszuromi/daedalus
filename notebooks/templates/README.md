# Templates

Clean per-group starting points. All four share **one** skeleton — load a
theory from `theories/*.theory.py` → one `nb.Config` → `nb.run` →
`nb.plot_cumulant` — so the demos stay uniform. Copy the one matching your
theory's shape and change `THEORY` + the config cell; everything else is
identical on purpose.

| Template | Group | Ships with |
|---|---|---|
| `template_temporal_single.ipynb` | temporal · single-field | `ou_quartic_double_well` |
| `template_temporal_multi.ipynb` | temporal · multi-field | `ou_quartic_two_dim` |
| `template_spatial_single.ipynb` | spatial · single-field | `kpz_1d` |
| `template_spatial_multi.ipynb` | spatial · multi-field | `coupled_rd_2species_1d` |

**Sim-vs-theory references** — the same core with a matched simulator overlaid
via `nb.plot_cumulant(..., sim=...)`. The only addition over the plain template
is the simulator cell; the `sim` dict is `{tau, C, C_err}` (temporal) or
`{x, C, C_err}` (spatial):

| Reference | Overlay |
|---|---|
| `template_temporal_single_sim_compare.ipynb` | OU + cubic vs Euler–Maruyama; `C(τ)` |
| `template_spatial_single_sim_compare.ipynb` | Allen–Cahn vs 1-D SPDE; equal-time `C(x,0)` |

See [`../README.md`](../README.md) for the config knobs (arbitrary `k` / loop
order, Dyson dressing, plotting options) and the full notebook catalog.

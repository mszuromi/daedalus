# Notebooks

Every demo computes a cumulant from a symbolic (S)PDE action with
`compute_cumulants` (the `pipeline/` package) and compares the diagrammatic
theory against a matched simulator. They all share **one** flow and **one**
source of truth for theories.

## Start here

| File | Use it to |
|---|---|
| [`templates/`](templates/) | Copy a clean per-group starting point (see the four groups below). |
| [`theory_runner.ipynb`](theory_runner.ipynb) | Run **any** `theories/*.theory.py` with no edits beyond one config cell — temporal or spatial, single- or multi-field, any `k`, any loop order, with/without Dyson. |
| [`theory_builder.ipynb`](theory_builder.ipynb) | Author a new theory in the interactive UI; it writes `theories/<name>.theory.py`. |
| [`daedalus.py`](daedalus.py) | The shared engine every template, the runner, and (the migrated) demos import. |

### The shared flow — load → run → plot

Theories live in **`theories/*.theory.py`** (the single source of truth — no
notebook re-builds a model inline). Every notebook is thin:

```python
import daedalus as dd
model, mod = dd.load_theory('kpz_1d')              # from theories/*.theory.py
cfg = dd.Config(k=2, max_ell=1, spatial_grid=(-6, 6, 49))
res = dd.run(model, cfg, mod)                       # k / ℓ / Dyson all here
dd.plot_cumulant(res, cfg, model)                   # auto-dispatched, adaptable
```

`dd.Config` holds everything a demo chooses; leave a field `None` to inherit the
theory file's `METADATA` / `DEFAULT_FUNDAMENTAL`:

* **Arbitrary `k` / loop order** — `k` (2 = ⟨··⟩, 3 = ⟨···⟩, …) and `max_ell`
  (0 = tree, 1, 2, …) are free. Spatial `k ≥ 3` uses `spatial_points`
  (an `(n_pts, k-1, 2)` array of `(x_j, τ_j)` offsets).
* **Dyson dressing** — `dyson_order` (any order ≥ 0) + `reference_diffusion`
  override the model's policy at run time (for coupled, unequal-diffusion fields).
* **Output quantity** — `output` ∈ `'cumulant'` (default) | `'moment'` |
  `'central_moment'`. The diagrammatics give connected cumulants κ; the moment
  options assemble the full k-point moment `⟨φ(x₁)…φ(x_k)⟩` (resp. of the
  centred field) via the set-partition expansion and return it as
  `res['moment']` (temporal any `k`, spatial `k=2`; costs `k−1` extra backend
  runs). The chosen `max_ell` is one **shared loop budget** across the whole
  moment — `M = Σ_π Σ_{Σℓ_B ≤ max_ell} ∏_B κ(B)^{(ℓ_B)}` — so it is the
  perturbatively-consistent L-loop moment (no partial higher-order cross-terms,
  e.g. a 1-loop moment never contains a 1-loop×1-loop = 2-loop piece). Raw
  moments add the (tree) mean ⟨φ⟩ on singleton blocks; central moments drop
  them.
* **Adaptable plotting** — `show_orders` (`'cumulative'` | `'incremental'` |
  `'total'`), `logy`, `components`, `figsize`, `title`, `save`. `plot_cumulant`
  dispatches on (spatial?, multi-field?, k): `C(τ)` for temporal, equal-time
  `C(x,0)` (+ a `C(x,τ)` heatmap when a τ grid is present) for spatial `k=2`,
  and a per-event bar chart for spatial `k ≥ 3`. Pass a real simulator with
  `plot_cumulant(..., sim={'tau'|'x', 'C', 'C_err'})` to overlay it.

### The four groups & their templates

A theory's **structure** (spatial vs temporal × single- vs multi-field) picks
the plot, so the demos are grouped by it. Multi-field = more than one physical
field **or** a population of size > 1.

| Group | Template | Ships with | Plot |
|---|---|---|---|
| temporal · single | [`template_temporal_single.ipynb`](templates/template_temporal_single.ipynb) | `ou_quartic_double_well` | `C(τ)` + per-loop overlay |
| temporal · multi | [`template_temporal_multi.ipynb`](templates/template_temporal_multi.ipynb) | `ou_quartic_two_dim` | `C(τ)`; pick legs via `external_fields` |
| spatial · single | [`template_spatial_single.ipynb`](templates/template_spatial_single.ipynb) | `kpz_1d` | `C(x,0)` (+ heatmap); `k≥3` bar chart |
| spatial · multi | [`template_spatial_multi.ipynb`](templates/template_spatial_multi.ipynb) | `coupled_rd_2species_1d` | `C(x,0)`; Dyson for unequal `D` |

Two **sim-vs-theory references** show how a deep-dive notebook overlays a real
simulator on the same core (the only addition is the simulator cell + passing
`sim=` to `plot_cumulant`):
[`template_temporal_single_sim_compare.ipynb`](templates/template_temporal_single_sim_compare.ipynb)
and [`template_spatial_single_sim_compare.ipynb`](templates/template_spatial_single_sim_compare.ipynb).

Each `*_sim_compare` notebook below is a per-theory deep dive: the theory side is
this same flow; only the simulator differs per model.  **27 of these have been
migrated onto the `daedalus` core** — they `dd.load_theory()` + `dd.run()`
in place of an inline/importlib model build, keeping their simulator, plots, and
validated diagnostics byte-for-byte (physics verified bit-identical).  Six are
not migrated, all due to **pre-existing pipeline/theory-file breakage** (not the
migration): the three 2-population Hawkes demos (`pipeline_demo`,
`quad_expg_loop_rate`, `quad_expg_sim_compare` — legacy `indexed=` models fail in
the propagator/MF solver) and `linear_delta_third_moment` (a corrupted voltage
theory file) currently do not run; `singlepop_quad_spike_reset` and `kpz_1d`
are pending a follow-up pass.

## `spatial/` — spatial field theories (Laplacian; `C(x, τ)` correlators)

| Notebook | Group | What |
|---|---|---|
| `pipeline_allen_cahn_1d_full_loop_sim_compare` | spatial·single | φ⁴ Allen-Cahn, d=1. `MAX_ELL ∈ {0,1,2}`; tree → 1-loop → 2-loop vs SPDE sim, cumulative per-loop progression + `VERBOSE` staged trace. |
| `pipeline_allen_cahn_quintic_1d_full_loop_sim_compare` | spatial·single | **φ⁶ generalization test** (Allen-Cahn + `−γφ⁵`). The `φ̃φ⁵` degree-6 vertex first enters at 2 loops; default `MAX_ELL=2`. |
| `pipeline_linear_diffusion_1d_sim_compare` | spatial·single | free 1D diffusion (Gaussian `C₀` check). |
| `pipeline_linear_field_2d_sim_compare` | spatial·single | free 2D field. |
| `pipeline_reaction_diffusion_2d_loop_sim_compare` | spatial·single | 2D reaction-diffusion, 1-loop. |
| `pipeline_reaction_diffusion_conserved_1d_sim_compare` | spatial·single | **Model B** — conserved `∇²(φ²)`, the **composite-∇** vertex (`mode='composite'`, `F∝q²`; conservation-suppressed variance). |
| `pipeline_burgers_1d_sim_compare` | spatial·single | **Burgers** — `−(λ/2)∂ₓ(φ²)`, the **composite-∂ₓ** vertex (imaginary form factor `∂ₓ→ik`; saddle-drift heat kernel). |
| `pipeline_kpz_1d_sim_compare` | spatial·single | **KPZ** — `(λ/2)(∂ₓh)²`, the **per-leg-∂ₓ** vertex (`mode='perleg'`, `F=ℓ²q(ℓ−q)`); excess-velocity `v∞` cross-check (~1%). |
| `pipeline_kpz_2d_sim_compare` | spatial·single | **KPZ in d=2** — `(λ/2)(∇h)²`: the d≥2 form-factor machinery (`L·d`-dim transverse-moment GH average, vs brute `∫d²ℓ` to 1e-14) + 2-D sim; excess-velocity ~0.1%. |
| `pipeline_combined_allencahn_modelb_kpz_1d_sim_compare` | spatial·single | **Combined (d=1)** — `−λφ³ + g∇²(φ²) + (κ/2)(∂ₓφ)²`. Per-vertex form-factor table (3 vertex types, mixed composite/per-leg modes) + 3-D SPDE sim. |
| `pipeline_combined_allencahn_modelb_kpz_2d_sim_compare` | spatial·single | **Combined (d=2)** — same 3 vertices at d=2 (per-vertex table + `L·d`-dim transverse-moment average + 2-D sim). 1-loop UV-divergent (Model B × KPZ cross), reported honestly. |
| `pipeline_combined_allencahn_modelb_kpz_3d_sim_compare` | spatial·single | **Combined (d=3)** — same 3 vertices at d=3 (tree ~2s, 1-loop ~150s). 1-loop UV-divergent, reported honestly (cutoff-set). |
| `pipeline_coupled_rd_2species_1d_sim_compare` | spatial·**multi** | **Coupled 2-species RD** — matrix reaction coupling (complex mode pair, damped predator–prey) + cubic nonlinearities: tree (matrix-Lyapunov analytic IFT) + 1-loop for `C_aa` AND the τ-asymmetric cross `C_ab`, vs the 2-species SPDE sim; plus the **unequal-D Dyson-dressed tree** (truncation ladder N=0→1→3 converging 5.7e-2 → 4.6e-3 → 5.3e-5 to the matrix-heat-kernel oracle). Theory file: [`coupled_rd_2species_1d`](../theories/coupled_rd_2species_1d.theory.py). |

Coverage spans one representative per model class: **free** (linear diffusion /
2D field), **polynomial reaction `φⁿ`** (Allen-Cahn φ⁴/φ⁶, 2D RD),
**composite-derivative** `∇²(φⁿ)`/`∂ₓ(φ²)` (Model B, Burgers), and
**per-leg-derivative** `(∂ₓφ)²` (KPZ).

See [`docs/spatial_pipeline.md`](../docs/spatial_pipeline.md) and, for the
gradient vertices, [`docs/spatial_kpz_burgers_plan.md`](../docs/spatial_kpz_burgers_plan.md).

## `temporal/` — time-only theories (OU, Hawkes, neural), via the ω-domain Phase J

| Notebook(s) | Group | What |
|---|---|---|
| `pipeline_ou_quartic_sim_compare`, `pipeline_ou_sextic_sim_compare`, `pipeline_ou_quartic_colored_sim_compare` | temporal·single | nonlinear Ornstein-Uhlenbeck (white / colored). |
| `pipeline_ou_quartic_two_dim_sim_compare`, `..._two_dim_corr_...`, `..._two_dim_color_corr_...` | temporal·**multi** | 2-field OU (coupled, cross-correlated / colored noise). |
| `pipeline_singlepop_*` (quad, cubic-alpha, spike-reset, …) | temporal·single | single-population neural. |
| `pipeline_multipop_sim_compare`, `pipeline_multipop_spike_reset_sim_compare` | temporal·**multi** | multi-population neural (E/I). |
| `pipeline_conductance_synapse_compare`, `pipeline_dendritic_linear_sim_compare` | temporal·single | neural variants. |
| `pipeline_quad_expg_sim_compare`, `pipeline_quad_expg_loop_rate_compare` | temporal·single | quadratic Hawkes with exponential filter. |
| `pipeline_bistable_mf_demo` | temporal·single | multi-root mean-field + linear-stability classification. |
| `pipeline_linear_delta_third_moment_compare`, `pipeline_demo` | temporal·single | third-moment / general pipeline demo. |

## `legacy/` — pre-pipeline experiments (archival)

Early diagram-enumeration and time-domain Hawkes exploration from before the
`pipeline/` API (`enumeration_*`, `field_theory_sage_demo`, `load_diagram_data`,
`hawkes_td_*`). Kept for provenance; **several import now-removed modules and
will not run as-is.**

---

`saved_results/` and `saved_theories/` are shared output directories (their
contents are gitignored). Every notebook opens with a **depth-robust root cell**
that walks up to the `pipeline/` package and puts `notebooks/` on the path, so it
runs correctly from any of these subdirectories.

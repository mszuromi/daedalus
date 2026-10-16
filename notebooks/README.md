# Notebooks

All notebooks run on the shared `daedalus` engine (imported as `dd`) — see the top-level
[README](../README.md) for installation.

## Start here

| File | Use it to |
|---|---|
| [`tutorials/`](tutorials/) | **The tutorial series.** Models are built with the SDE / point-process front end (`api.sde.SDE`): quick start, the front end, the noise library, point processes, delays, outputs, hand-written actions, spatial models. Start with [`00_quick_start.ipynb`](tutorials/00_quick_start.ipynb). |
| [`examples/`](examples/) | Worked examples, one per capability, each overlaying a from-scratch simulation on the pipeline result. See [`examples/README.md`](examples/README.md). |
| [`model_builder.ipynb`](model_builder.ipynb) | Author a model in the interactive `ModelUI` form; it writes `models/<name>.model.py`. |
| [`model_runner.ipynb`](model_runner.ipynb) | Load and run **any** `models/*.model.py` with a single config cell — temporal or spatial, single- or multi-field, any `k`, any loop order, with/without Dyson dressing. |
| [`legacy/model_builder_tutorial.ipynb`](legacy/model_builder_tutorial.ipynb) | The earlier guided tour of the hand-written `TemporalModelBuilder` (superseded by the tutorials; kept for reference). |
| [`daedalus.py`](../daedalus.py) | The shared front-end (`dd`) every notebook imports (at the repo root). |

## Tutorials

| | Notebook | Covers | Run time |
|---|---|---|---|
| 00 | [Quick start](tutorials/00_quick_start.ipynb) | one SDE end to end; simulation and exact variance | < 1 min |
| 01 | [The SDE front end](tutorials/01_sde_front_end.ipynb) | populations, indexed parameters, functions, kernels, Itô / Stratonovich | < 1 min |
| 02 | [The noise library](tutorials/02_noise_library.ipynb) | every source, jump law and composite; Lévy-driven OU | < 1 min |
| 03 | [Point processes](tutorials/03_point_processes.ipynb) | linear and nonlinear Hawkes, `SDE.simulate`, δ components | < 1 min |
| 04 | [Delays](tutorials/04_delays.ipynb) | hard and distributed delays, Padé loops, a delay in a nonlinear term | ~1 min |
| 05 | [Outputs](tutorials/05_outputs.ipynb) | `dd.cumulant`, spectra, kernel integrals, `k = 3`, symbolic results, transients | < 1 min |
| 06 | [Writing the action by hand](tutorials/06_actions_by_hand.ipynb) | `TemporalModelBuilder`; model files; `dd.Config` / `dd.run` | < 1 min |
| 07 | [Spatial models](tutorials/07_spatial.ipynb) | stochastic PDEs; pointers to the spatial examples | ~1 min |

## The shared flow — load → run → plot

```python
import daedalus as dd
model, mod = dd.load_model('allen_cahn_1d_subcritical_infinite')   # from models/*.model.py
cfg = dd.Config(k=2, max_ell=1, chi_grid=(-6, 6, 49))
res = dd.run(model, cfg, mod)                        # k / loop order / Dyson all here
dd.plot_cumulant(res, cfg, model)                   # auto-dispatched
```

`dd.Config` holds every run choice; leave a field `None` to inherit the model file's
`METADATA` / `DEFAULT_FUNDAMENTAL`. Run **`dd.config_options()`** for the full annotated list,
or see the [outputs tutorial](tutorials/05_outputs.ipynb).

A model can be built four ways, all giving the same model dict: the SDE / point-process front
end (`api.sde.SDE`, the tutorials), the GUI (`model_builder.ipynb`), the hand-written builder
(`TemporalModelBuilder`, tutorial 06), or by loading a saved `models/*.model.py` file
(`dd.load_model`).

---

`saved_results/` and `saved_models/` are shared output directories (gitignored). Every
notebook opens with a depth-robust root cell that locates the `api/` package and puts
`notebooks/` on the path, so it runs from any working directory.

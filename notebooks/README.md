# Notebooks

All notebooks run on the shared `daedalus` engine (imported as `dd`) — see the top-level
[README](../README.md) for installation.

## Start here

| File | Use it to |
|---|---|
| [`theory_builder_tutorial.ipynb`](theory_builder_tutorial.ipynb) | **Guided tour** — every component of a temporal theory, the graphical and code builders, and how to run one. Start here. |
| [`theory_builder.ipynb`](theory_builder.ipynb) | Author a theory in the interactive `TheoryUI` form; it writes `theories/<name>.theory.py`. |
| [`theory_runner.ipynb`](theory_runner.ipynb) | Load and run **any** `theories/*.theory.py` with a single config cell — temporal or spatial, single- or multi-field, any `k`, any loop order, with/without Dyson dressing. |
| [`examples/`](examples/) | Eight worked examples, one per capability, each overlaying a from-scratch simulation on the pipeline result. See [`examples/README.md`](examples/README.md). |
| [`daedalus.py`](../daedalus.py) | The shared front-end (`dd`) every notebook imports (at the repo root). |

## The shared flow — load → run → plot

```python
import daedalus as dd
model, mod = dd.load_theory('allen_cahn_1d_subcritical_infinite')   # from theories/*.theory.py
cfg = dd.Config(k=2, max_ell=1, chi_grid=(-6, 6, 49))
res = dd.run(model, cfg, mod)                        # k / loop order / Dyson all here
dd.plot_cumulant(res, cfg, model)                   # auto-dispatched
```

`dd.Config` holds every run choice; leave a field `None` to inherit the theory file's
`METADATA` / `DEFAULT_FUNDAMENTAL`. Run **`dd.config_options()`** for the full annotated list,
or see §5 of the tutorial for what each argument does.

A theory can be built three ways, all feeding the same `dd.run`: the GUI
(`theory_builder.ipynb`), the Python builder (`dd.TemporalTheoryBuilder`, shown in the
tutorial), or by loading a saved `theories/*.theory.py` file (`dd.load_theory`).

---

`saved_results/` and `saved_theories/` are shared output directories (gitignored). Every
notebook opens with a depth-robust root cell that locates the `api/` package and puts
`notebooks/` on the path, so it runs from any working directory.

# Architecture

Daedalus turns an MSR–JD action for a stochastic (S)PDE into mean fields, multi-point cumulants,
and loop corrections — analytically. Four tiers:

```
dd  (daedalus.py)   notebook/script front-end: Config, run, plot, load_model, describe_model
      │
api/                user-facing orchestration: compute_cumulants, ModelBuilder, report, save
      │
engine/             the core: symbolic field theory → diagram enumeration → loop integration
      │
simulations/        independent numerical validators (NOT part of the analytic pipeline)
```

## `dd` — `daedalus.py`

The single entry point (`import daedalus as dd`). Wraps `api` with notebook-friendly helpers:
`Config` (one dataclass holding every run choice), `run` (layers the config over the model and
calls the engine), `plot_cumulant`, `load_model` / `list_models`, `describe_model`,
`save_npz` / `save_csv`, the prediagram helpers, and the graphical `ModelUI`. Works from scripts
too (`sage -pip install -e .`, or put the repo root on `sys.path`).

## `api/`

The orchestration layer `dd` drives:

- `compute.py` — `compute_cumulants`: the full chain (FT expand → propagator → mean field →
  enumerate diagrams → integrate → cumulant).
- `model.py` (+ `model_compiler.py`, `model_serialize.py`, `model_templates.py`) — the
  declarative `ModelBuilder` authoring system that emits a model dict.
- `_mean_field*.py`, `_propagator.py`, `_grouped_phase_j.py`, `_diagrams.py` — pipeline helpers.
- `report.py`, `save.py`, `access.py` — PDF reports, `.npz` / `.csv` output, natural-name accessors.
- `ui/` — the `ModelUI` graphical builder.

## `engine/`

The model-agnostic physics core:

- `core/` — `field_theory.py` (the MSR–JD action algebra), `vertices.py`, `propagator.py`.
- `enumeration/` — Feynman-diagram enumeration.
- `diagrams/` — typed-diagram bookkeeping, symmetry factors, prediagram plotting.
- `integration/{time_domain,spatial}/` — the loop-integral evaluators: the temporal Phase J
  integrator, and the spatial Symanzik / causal-chamber / heat-kernel integrator.

## `simulations/`

From-scratch numerical integrators (Euler–Maruyama for SDEs, spectral ETD1 for SPDEs) used to
validate the analytic results in the example notebooks. They are **not** consumed by the
analytic pipeline.

## `models/`

`*.model.py` specs, each building a model dict via the `api` builder. Loaded by path with
`dd.load_model(name)` (they are data, not importable packages).

## Data flow of a run

`dd.load_model(name)` → `(model, module)`. `dd.run(model, cfg, module)` layers `cfg` over the
model's `DEFAULT_FUNDAMENTAL` / `METADATA` and calls `api.compute_cumulants`, which drives the
`engine` and returns a result dict (`C_tau`, `tau_grid` / `chi_grid`, the mean field, the
per-loop-order breakdown, the diagram records). `dd.plot_cumulant` renders it; `dd.save_npz`
persists it.

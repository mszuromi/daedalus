# Architecture

Daedalus computes stationary statistics of Itô SDEs/SPDEs: it turns the MSR–JD action into
mean fields, multi-point cumulants, and loop corrections — analytically. Four tiers:

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
  enumerate diagrams → integrate → cumulant), with the model analyzer's gates at the stage
  boundaries and the integrator chosen by the backend registry.
- `analyze.py` — the model analyzer: traits, findings (`error` / `warning` / `info`, each with
  a code and a remedy) and the assumptions ledger; standalone (`analyze(model)`) or wired into
  `compute_cumulants` (`Preflight`: gates G0–G4 before the enumeration, G6 after the run).
- `routing.py` — the backend registry: each integrator backend declares
  `applies(request) -> (bool, reasons)` and a priority; the router takes the first that applies.
- `config_run.py` — the YAML entry point behind `daedalus run config.yaml`: strict validation of a
  config, then the same pipeline as `dd.run`, with the requested outputs written (`--dry-run`
  prints the resolved config and the analyzer report instead of computing).
- `sde.py` + `noise.py` — the SDE front-end: declare fields, a drift and noise sources; derive the
  MSR action and the mean-field equations and emit the same model dict as `ModelBuilder`.
  `noise.py` is the noise-source library, each source given by its conditional cumulant generating
  function (Gaussian, Poisson, compound Poisson, gamma and inverse-Gaussian processes, variance-gamma, NIG,
  CGMY / tempered stable, Meixner, bilateral gamma, Skellam, negative-binomial, sums, or a user-given CGF).
  `noise_samplers.py` draws increments of every named source and jump law (`source.sample(dt, n, rng)`).
  `SDE.point_process` declares spike trains whose rates filter any field (`Conv`); `pp_simulate.py`
  simulates such models (`SDE.simulate`).
- `cumulant.py` — `Cumulant`: a computed cumulant as its smooth part per loop order plus the
  shot-noise δ components (the spikes on a hyperplane of the external times), read off the result
  without recomputing. `cumulant_plot.py` draws it (`dd.plot_cumulant(..., show_delta=True)`).
- `spectra.py` — the exact spectral and integrated outputs of a k = 2 result (`spectra=`, `dd.spectra`):
  `S(ω)`, `S(0)`, `∫_0^T C`, `∫_0^∞ C` and `τ_c` per loop order, from the triples of `me_output.py`
  (resolvents and augmented matrix exponentials), with a quadrature fallback that reports error estimates.
- `master_equation.py` — the exact master-equation oracle for lattice jump processes (`dd.master_equation`): sparse
  stationary solve, cumulants `k <= 4`, Krylov / `expm_multiply` lag correlations, truncation diagnostics, the
  Kramers-Moyal extrapolation, and the bridge to `compute_cumulants` (`docs/master_equation.md`).
- `symbolic_out.py` — the tree-level covariance as symbolic expressions in the Fourier domain,
  with exports (sympy, LaTeX, a numpy callable, JSON) and its time-domain inverse.
- `_pole_clusters.py` — near-degenerate and repeated propagator poles: replaces a nearly
  cancelling pole sum by a contour average around the cluster, so close poles keep their precision.
- `_expand_cache.py` — a disk cache for `FieldTheory.expand()` results (the dominant cost for
  large models), reused across Taylor orders.
- `_realization.py` — the state-space form of the propagator and of the kernels: turns the Fourier
  kernel `K_ft` into exact matrices (A, B, C, D) with `G(t) = C e^{At} B` for t > 0 (plus an
  instantaneous part D), without computing a root, a pole or a residue. Rates that coincide or
  nearly coincide are an ordinary case for it. Each propagator entry and each conductance kernel
  gets its own minimal realization; an entry that is not rational in the frequency, or a model with
  space, is refused by name and left to the pole engine.
- `colored_to_markovian.py` — rewrites a colored-noise model as a linear filter driven by white
  noise, so the temporal integrator can handle it.
- `model.py` (+ `model_compiler.py`, `model_serialize.py`, `model_templates.py`) — the
  declarative `ModelBuilder` authoring system that emits a model dict.
- `_mean_field*.py`, `_propagator.py`, `_diagrams.py` — pipeline helpers.
- `report.py`, `save.py`, `access.py` — PDF reports, `.npz` / `.csv` output, natural-name accessors.
- `ui/` — the `ModelUI` graphical builder.

## `engine/`

The model-agnostic physics core:

- `core/` — `field_theory.py` (the MSR–JD action algebra), `vertices.py`.
- `enumeration/` — Feynman-diagram enumeration.
- `diagrams/` — typed-diagram bookkeeping, symmetry factors, prediagram plotting.
- `integration/{time_domain,spatial}/` — the loop-integral evaluators: the temporal Phase J
  integrator, and the spatial Symanzik / causal-chamber / heat-kernel integrator.
  Within `time_domain/`, besides the main Phase J chain (`pipeline.py`, `final_integral.py`):
  - `expdd.py` — divided-difference kernels: every closed-form Phase J integral of an
    exponential over a simplex written as a divided difference of `exp`, evaluated stably.
  - `boxfree.py` — the exact integrals built on those kernels, evaluated in the log domain, with
    an explicit overflow status instead of a silent `inf`.
  - `statespace.py` — the state-space route for a Phase J integral: an integral of a product of
    propagator and kernel lines over a region of internal times is evaluated as a matrix
    computation on the realizations of `api/_realization.py` (a sweep over the sub-orders of the
    diagram, with no poles, residues or pole tuples), which stays accurate where poles come close.
    **Which engine runs.** It is the default engine: every subset it can take goes through it, and
    a subset it cannot take (a noise-kernel box, a spatial model, a symbolic prefactor) or that it
    refuses at call time (a stiff or oversized system, a non-finite value) is evaluated by the pole
    engine instead, never silently: the reason is counted and reported in
    `result['phase_j_engine']`. `DAEDALUS_PHASE_J_ENGINE=poles` switches back to the pole engine
    everywhere (the numbers of the release before the change); `default` or `statespace` name the
    default; any other value raises. `PHASE_J_MEM_BUDGET_GB` (environment variable, default 6, or
    `off`) caps the memory of Phase J as a whole: the stationary states of this route are charged
    against it, and a run over it raises `PhaseJResourceError` before allocating.
  - `dbm_integral.py` — the exact integral of an exponential-polynomial over a polytope whose
    rows are difference bounds (scalar bounds and shifted order rows).

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
per-loop-order breakdown, the diagram records, the route and the analyzer report).
`dd.plot_cumulant` renders it; `dd.save_npz` persists it with a version / Phase J convention /
route / assumptions-ledger stamp, and `dd.load_npz` reads it back (warning when the stamp
predates 0.2.0; `dd.load_provenance` returns the stamp).

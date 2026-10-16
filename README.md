# Daedalus

**Automated loop corrections for stochastic dynamics.** Write a model as stochastic differential
equations, a point process, or a stochastic PDE; Daedalus returns its stationary mean,
correlation functions and higher cumulants, at tree level and with one, two or more loop
corrections, as functions of time lag (and space). It builds the Martin–Siggia–Rose–Janssen–De
Dominicis (MSRJD) action, enumerates the Feynman diagrams and evaluates every diagram's time
integral exactly, with no hand derivation. It is built on [SageMath](https://www.sagemath.org/).

```python
m = (SDE('quartic OU')
     .parameter('mu', default=1.0).parameter('eps', default=0.02).parameter('D', default=1.0)
     .field('x')
     .drift('x', '-mu*x - eps*x^3')
     .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
res = compute_cumulants(m.build(), k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)])
```

## Why

Linear-noise and Gaussian approximations describe fluctuations about a fixed point by linearizing
the dynamics. Nonlinearity, non-Gaussian noise (shot noise, jumps, spikes) and their interplay show
up first in the **loop corrections**: shifted means, changed variances and time scales, and nonzero
higher cumulants. Computing them by hand means enumerating diagrams and doing nested time integrals
for each model. Daedalus does both automatically, and the integrals are done in closed form (as
matrix computations on the state-space form of the propagator), so a result is exact for its loop
order and can be pushed to two or three loops where the diagrams allow.

## What it does

- **Models from equations.** `SDE(...).field().drift().noise()` derives the action and the
  mean-field equations from the stochastic equations (Itô or Stratonovich); populations with
  indexed parameters, named functions, convolution kernels and general linear operators
  (`(tau*Dt + 1)*v`) are part of the syntax. The action can also be written by hand.
- **Point processes** with any conditional intensity (`.point_process('n', rate)`), including
  intensities that filter their own spike train (nonlinear Hawkes processes), mixed with
  continuous fields, plus a simulator of the same model (`SDE.simulate`).
- **A noise library** of 15 source types, 20 jump laws and named composites (Gaussian, Poisson,
  compound Poisson, gamma, inverse-Gaussian, variance-gamma, NIG, CGMY, Meixner, Skellam, ...),
  each with its cumulant generating function and, for most, an exact sampler.
- **Delays and kernels:** hard delays `x(t - d)` (exact tree, Padé loops with an automatically
  chosen order), distributed delays and any rational kernel, also inside nonlinear terms.
- **Outputs:** cumulants of any order `k`; `C(τ)` at any lag, spectra and kernel integrals in
  closed form; shot-noise δ components reported separately; loop corrections as exact rational
  functions of the parameters (`dd.symbolic_cumulant`); switch-on transients.
- **Spatial fields** (stochastic PDEs such as Allen–Cahn and reaction–diffusion) in one to three
  dimensions.
- **A model report** before any heavy computation (`--dry-run`): which checks the model passes,
  an estimate of the loop-expansion parameter, and warnings when a result would be unreliable.

Every example notebook compares the result with an independent simulation.

---

## Quick start

With conda or mamba installed ([Miniforge](https://github.com/conda-forge/miniforge) if you
have neither):

```bash
git clone https://github.com/mszuromi/daedalus.git
cd daedalus
conda env create -f environment.yml     # or:  mamba env create -f environment.yml   (faster)
conda activate daedalus
```

[`environment.yml`](environment.yml) pulls SageMath 10.8 and the extras from conda-forge —
that is the entire setup. Then, from the repository root, the connected two-point function of
a nonlinear Ornstein–Uhlenbeck process `dx = (-mu x - eps x^3) dt + sqrt(2D) dW`, tree level
plus one loop:

```bash
python - <<'PY'
from api.sde import SDE
from api.noise import Gaussian
from api import compute_cumulants

m = (SDE('OU quartic')
     .parameter('mu', default=1.0, domain='positive')
     .parameter('eps', default=0.02, domain='positive')
     .parameter('D', default=1.0, domain='positive')
     .field('x')
     .drift('x', '-mu*x - eps*x^3')
     .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
res = compute_cumulants(m.build(), k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)], verbose=False)
tree, loop = res['C_tau_by_ell'][0], res['C_tau_by_ell'][1]       # arrays over the lag grid; index 100 is lag 0
print(f"C(0): tree {tree[100].real:.5f}, one-loop {loop[100].real:+.5f}, total {res['C_tau'][100].real:.5f}")
PY
```

It prints `C(0): tree 1.00000, one-loop -0.06000, total 0.94000` in a few seconds: the one-loop
term is the exact first-order correction `-3 eps D^2 / mu^3` to the stationary variance. (Inside
the conda environment `python` is Sage's interpreter; with a native Sage install use
`sage -python` and run from the repository root.)

## Tutorials

The tutorial series in [`notebooks/tutorials/`](notebooks/tutorials/) builds every model with the
SDE / point-process front end. Each notebook is short, self-contained and ships executed.

| | Tutorial | Covers |
|---|---|---|
| 00 | [Quick start](notebooks/tutorials/00_quick_start.ipynb) | one SDE end to end: tree, one and two loops, simulation, exact variance |
| 01 | [The SDE front end](notebooks/tutorials/01_sde_front_end.ipynb) | fields, populations and indexed parameters, functions, kernels, Itô / Stratonovich, `show()` |
| 02 | [The noise library](notebooks/tutorials/02_noise_library.ipynb) | every source, jump law and composite in a table; a Lévy-driven OU process vs its exact cumulants and a sampler |
| 03 | [Point processes](notebooks/tutorials/03_point_processes.ipynb) | linear Hawkes (exact check), nonlinear Hawkes with loops vs `SDE.simulate`, δ components |
| 04 | [Delays](notebooks/tutorials/04_delays.ipynb) | hard delays (exact tree, Padé loops, automatic order), distributed delays, a delay in a nonlinear term |
| 05 | [Outputs](notebooks/tutorials/05_outputs.ipynb) | the result dict, `dd.cumulant` (spectra, kernel integrals, modes), `k = 3`, symbolic results, switch-on transients |
| 06 | [Writing the action by hand](notebooks/tutorials/06_actions_by_hand.ipynb) | `TemporalModelBuilder`, how the SDE builder maps onto it, model files and `dd.run` |
| 07 | [Spatial models](notebooks/tutorials/07_spatial.ipynb) | stochastic PDEs; pointers to the spatial examples |

[`notebooks/examples/`](notebooks/examples/) holds worked examples, one per capability, each with a
from-scratch simulation overlay: white, colored and correlated noise, a nonlinear Hawkes process,
delayed feedback, a delayed product nonlinearity, delayed-kernel point processes, a two-compartment
neuron model, and spatial fields.

## The noise library at a glance

A noise source is a process with independent stationary increments, given by its cumulant
generating function per unit time `K(θ)`; it enters the action as `-K(Σ c_x x̃)`. Only infinitely
divisible laws with finite moments are offered. Full table with parameters and cumulant rates:
[tutorial 02](notebooks/tutorials/02_noise_library.ipynb) and [`docs/noise_library.md`](docs/noise_library.md).

| Kind | Members (`api.noise`) |
|---|---|
| Sources | `Gaussian`, `Poisson`, `CompoundPoisson`, `GammaProcess`, `InverseGaussianProcess`, `VarianceGamma`, `NormalInverseGaussian`, `CGMY`, `TemperedStableSubordinator`, `Meixner`, `BilateralGamma`, `Skellam`, `NegativeBinomialProcess`, `Cumulants` (cumulant rates), `CGF` (any expression in `theta`) |
| Jump laws (for `CompoundPoisson`) | `Dirac`, `Exponential`, `Gamma`, `GaussianJump`, `Laplace`, `Uniform`, `Bernoulli`, `Binomial`, `FinitePMF`, `Geometric`, `PoissonJump`, `NegativeBinomial`, `Logarithmic`, `DoubleExponential`, `AsymmetricLaplace`, `Hyperexponential`, `PhaseType`, `ChiSquared`, `LogNormal`, `Weibull` (the last two by their moments) |
| Composites | `a + b` (sum of sources), `merton_jump_diffusion`, `kou_jump_diffusion`, `levy_driven_ou` |

Parameters may depend on fields (state-dependent rates and jump sizes), and `source.sample(dt, n, rng)`
draws exact increments for all sources except `Cumulants` / `CGF`.

## Capabilities and limits

| | Supported | Limits |
|---|---|---|
| Expansion | loop expansion about a stable mean-field fixed point; any number of fields and populations | perturbative: needs a small loop parameter (the model report estimates it); no barrier crossing between wells, no limit cycles |
| Statistics | stationary cumulants of any order `k`; switch-on transients from a mean-field initial state | the practical size is set by the diagram count: e.g. `k = 2` up to three loops for one white-noise field, `k = 4` at two loops, `k = 6` at one loop |
| Noise | Gaussian (white, colored, correlated), Poisson / point processes, finite-moment Lévy sources, state-dependent parameters | heavy-tailed (infinite-moment) stable noise; Stratonovich conversion for Gaussian noise only; two-state (telegraph) noise is simulable but not expandable |
| Time structure | rational kernels, hard and distributed delays (exact tree, Padé loops) | time-independent dynamics: no explicit `t` in the equations |
| Space | stochastic PDEs in `d = 1..3`, `k = 2`, up to two loops | derivative (transport) vertices such as KPZ or conserved Model B are outside the stationary-saddle cumulants |
| Outputs | `C(τ)` on any grid or at any lag, spectra, kernel integrals, δ components, symbolic rational results | symbolic results for state-space-representable models only (no conductance kernels, size limits) |

## Requirements

The one prerequisite is **SageMath 10.8**. Sage provides Python 3.13 and almost the entire
stack — numpy, scipy, sympy, matplotlib, mpmath, networkx, cysignals, the Jupyter stack
(JupyterLab / notebook / ipykernel / **ipywidgets**), and a `SageMath` Jupyter kernel. Only a
couple of extras are added on top (`numba`, optionally `pydot`).

> Other Sage 10.x releases will likely work, but 10.8 is what the code and the embedded
> notebook kernel (`sagemath-10.8`) are tested against.

## Installation

### Option A — conda environment file (recommended: reproducible, cross-platform)

Works on macOS, Linux, and Windows (via WSL2); needs conda or mamba
([Miniforge](https://github.com/conda-forge/miniforge) if you have neither). From the cloned
repo root:

```bash
conda env create -f environment.yml      # or:  mamba env create -f environment.yml   (faster)
conda activate daedalus
```

[`environment.yml`](environment.yml) installs SageMath 10.8 plus the `numba`/`pydot`/`graphviz`
extras from conda-forge (validated to resolve on macOS/Linux). For the optional GUI clipboard,
also `pip install pyperclip`.

In the activated environment plain `python` is Sage's interpreter, and the conda-forge `sage`
launcher has no `-python` flag. Wherever this README or CONTRIBUTING says `sage -python`, run
`python` instead (for example `python -m pytest tests/ -q`, `python -m daedalus run config.yaml`).
The conda environment registers a Jupyter kernel named `sagemath`; if a notebook asks for
`sagemath-10.8`, pick it from the kernel menu.

### Option B — native SageMath + extras

1. Install **SageMath 10.8** from the official downloads:
   <https://www.sagemath.org/download.html> (macOS app, Linux binary, or WSL2 on Windows).
   This registers the `sagemath-10.8` Jupyter kernel the notebooks expect.
2. Add the extras into Sage's own Python:

   ```bash
   sage -pip install numba pydot      # pyperclip is optional (GUI copy buttons)
   ```

   `pydot` also needs the system `graphviz` binary for rendering
   (`brew install graphviz` / `apt install graphviz`); it is only used for optional
   diagram plots, with a graceful fallback if absent.

## Get the code

```bash
git clone https://github.com/mszuromi/daedalus.git
cd daedalus
```

It runs straight from the clone — the notebooks add the repo to the Python path automatically
(each opens with a depth-robust cell that locates the repo root). To import it from anywhere
with **no path setup**, optionally install it editable into Sage's Python:

```bash
sage -pip install -e .      # then `import daedalus` / `import api` work from any directory
```

(SageMath itself stays an external prerequisite — this only registers the in-repo packages.)
From a script without installing, see *Using it from a script* below.

## Package layout

| path | what it is |
|---|---|
| [`daedalus.py`](daedalus.py) | the `dd` front-end — `import daedalus as dd`; the entry point for notebooks **and** scripts |
| [`api/`](api/) | the user-facing layer `dd` wraps (`compute_cumulants`, `ModelBuilder`, `generate_report`, `save_npz`) |
| [`engine/`](engine/) | the core engine: symbolic field theory → diagram enumeration → loop integration |
| [`simulations/`](simulations/) | independent numerical simulators used to validate the analytic results |
| [`models/`](models/) | `*.model.py` specs loaded by `dd.load_model(name)` |

## Using it from a script

`dd` runs from a plain `sage -python` script, not just notebooks — put the repo root and its
`notebooks/` directory on the path:

```python
import sys; sys.path[:0] = ['/path/to/daedalus', '/path/to/daedalus/notebooks']
import daedalus as dd
model, mod = dd.load_model('allen_cahn_1d_subcritical_infinite')
res = dd.run(model, dd.Config(k=2, max_ell=1, chi_grid=(-6, 6, 49)), mod)
```

For lower-level access the API layer is importable directly, e.g. `from api import compute_cumulants`.

Models can also be declared line by line, without writing the action (`api/sde.py`): each line
contributes its own action term (`.drift` / `.equation`, `.noise`, `.point_process`). A spike-only
exponential Hawkes network whose rates filter the spike trains themselves:

```python
from api.sde import SDE
from api.delay import Kernel
from api import compute_cumulants
m = (SDE('Hawkes').population('E', size=2)
     .parameter('a', default=[1.0, 1.0], indexed_by=['E'])
     .parameter('Em', default=[-1.5, -1.5], indexed_by=['E'])
     .parameter('tau', default=2.0, domain='positive')
     .parameter('w', default=[[0.4, 0.4], [0.4, 0.4]], indexed_by=['E', 'E'])
     .define_kernel('r', kernel=Kernel.exponential('tau'))
     .point_process('n', 'a*exp(Em + sum(w[i,j]*Conv(r, n[j]) for j in E))', population='E'))
res = compute_cumulants(m.build(), k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)])
rec = m.simulate(T=500, dt=0.01, n_paths=200, bin=0.25)      # a simulator of the same model
```

## Running from a config file

The `daedalus run` command takes a YAML file instead of a script (needs PyYAML, which the conda environment has):

```bash
sage -python -m daedalus run config.yaml --dry-run   # validate, print the resolved config and the analyzer report, compute nothing
sage -python -m daedalus run config.yaml             # compute and write the outputs
daedalus run config.yaml                             # the same, after `sage -pip install -e .` (console script `daedalus`)
```

Run these from the repository root, or from anywhere after `sage -pip install -e .` (see *Get the code*); otherwise Python cannot find the `daedalus` module.

```yaml
# config.yaml: the connected 2-point function of the OU quartic model, tree + 1 loop
model: ou_quartic                  # a models/<name>.model.py, or a path to a *.model.py
parameters:                        # numbers, or lists of numbers for indexed parameters
  mu: 1.0
  eps: 0.02
question:
  k: 2
  max_ell: 1
  external_fields: [[x, 1], [x, 1]]          # [field, population index], k of them
  tau_grid: {start: -6, stop: 6, num: 25}    # or an explicit increasing list
  taylor_order: 4                            # optional; default is chosen from k and max_ell
policy:
  use_cache: true
  parallel: false
  verbose: false
outputs:                           # relative paths are relative to this file
  npz: results/ou.npz
  csv: results/ou.csv
  plot: results/ou.png
  # report_pdf: results/ou.pdf     # temporal models only
```

Every section but `model` is optional; an omitted `k`, `max_ell`, `external_fields` or `tau_grid` takes
the model's own default, exactly as `dd.run` does. `points` (spatial models with k >= 3) is the
`(n_points, k-1, 2)` list of `(x_j, tau_j)` events. Unknown keys and bad values stop the run with the
line number, e.g. `config.yaml: line 7: unknown key 'taylor' in 'question'`. A run also writes
`config_resolved.yaml` (every default filled in; itself a valid config) and `run_info.json` (package
version, git commit, resolved options, wall time) next to the first output.

More keys. `question.chi_grid` (alias `spatial_grid`) is the spatial-difference grid of a spatial model,
written like `tau_grid`: a `{start, stop, num}` mapping is a range, a list is literal points, and a
3-element list that looks like a range (`[-5, 5, 51]`) is an error, as in `dd.Config`. `output: results/`
(a directory) or `output: results/ou1` (a file stem) writes `.npz`, `.csv` and `.png` there, with
`config_resolved.yaml` and `run_info.json` beside them (prefixed with the stem when it is one); an explicit
`outputs:` entry wins for its kind. A list-valued parameter must have the shape the model declares (the sizes of
its populations), otherwise the run stops with the line number.

An inline SDE can replace `model:` (built with `api.sde.SDE`, which derives the action and the mean-field
equations; the noise `type` is a class of `api.noise`, the other keys are its arguments):

```yaml
sde:
  name: OU quartic
  parameters: {mu: {default: 1.0, domain: positive}, eps: 0.02, D: {default: 1.0, domain: positive}}
  fields: [x]
  drift: {x: "-mu*x - eps*x^3"}
  noise:
    xi: {type: Gaussian, var: "2*D", couples: {x: 1}}
  stability_analysis: true
```

`--dry-run` also prints the model analyzer's report table for the question (gates G0-G5: structure, mean
field, poles, noise class, the loop-parameter estimate `g_eff` and the Taylor check; no diagram
enumeration, no integrals) as YAML comments, so the output is still a valid config. The same table is
`api.analyze.analyze(model).table()` (or `result['analysis'].table()` after a run) and `dd.run` prints it when
`verbose=True`:

```text
Model report: OU Quartic (white noise)                                    [OK; 0 warning(s), 0 info]
----------------------------------------------------------------------------------------------------
question   k=2  max_ell=1  external=[('x', 1), ('x', 1)]
route      phase_j_per_diagram (predicted)
model      temporal; 1 physical / 1 response component(s); 4 parameter(s); Taylor order 4;
           time-homogeneous yes
vertices   polynomial yes; species [(1, 2), (1, 3)]; max degree 4; state-dependent noise no
noise      white; Gaussian yes; cross-correlated no
saddle     1 found, 1 stable; selected #0 stable yes; marginal modes 0
poles      n=1; rate min/max 1 / 1; fast/slow 1; repeated no; close pairs 0; delta part no
regime     g_eff 0.06 (small, warn > 0.2)
cost       analysis 12.3 ms
----------------------------------------------------------------------------------------------------
findings   none
```

## Verify the install

From the repository root:

```bash
sage -python - <<'PY'
import sys; sys.path[:0] = ['.', 'notebooks']
import daedalus as dd
model, mod = dd.load_model('ou_quartic')
res = dd.run(model, dd.Config(k=2, max_ell=0), mod)
print('OK —', dd.summary(res).splitlines()[0])
PY
```

Expected output: `OK — model : 'OU Quartic (white noise)'`.

## Choosing the Phase J engine

The time integrals of the loop diagrams are evaluated, by default, as matrix computations on the
state-space form of the propagator (no poles or residues), which keeps its accuracy where the
decay rates of a model are equal or close. A case that route cannot take (a noise-kernel box, a
spatial model) uses the pole engine automatically, and the result records which engine served
each part (`result['phase_j_engine']`). To use the pole engine everywhere, set
`DAEDALUS_PHASE_J_ENGINE=poles` before the run (for example `DAEDALUS_PHASE_J_ENGINE=poles python
script.py`); this gives the numbers of the release before the change. Phase J refuses to exhaust
memory: `PHASE_J_MEM_BUDGET_GB` (default 6, or `off`) sets the cap, and a run over it stops with
`PhaseJResourceError` and the numbers to raise it. See [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Prediagram cache

The package ships the diagram skeletons for 21 `(k, ell)` cells (up to `k = 6, ell = 1` and `k = 2, ell = 4`); the first use of a cell rebuilds its records from the stored classes, under a second for the small ones and growing with the class count. For any other cell, `dd.build_cache(cells, dest=None, dry_run=False)` first prints what exists (shipped / local / missing), the class count (known or extrapolated), and the estimated build time and memory, and then builds the missing cells into `dest` (default `saved_prediagrams`). `dry_run=True` stops after the report and writes nothing. Note that using a cell later needs about 11 kB of memory per class.

```python
import daedalus as dd
dd.build_cache([(3, 4), (4, 3)], dry_run=True)   # what would it cost?
```

Two large cells, `(4, 3)` (7.3 million classes, 15 MB compressed) and `(6, 2)` (12.5 million classes,
29 MB), are too big for git and are published as release assets. Nothing is downloaded unless you ask:
`dd.fetch_cache(cells='all')` (or `cells=[(4, 3)]`, `cells=[(6, 2)]`) downloads the cells into `saved_prediagrams/` and
checks size, SHA-256, format stamp and class count against
[`engine/enumeration/fetch_manifest.json`](engine/enumeration/fetch_manifest.json) before it accepts a file.
Using such a cell later needs about 11 kB of memory per class (about 80 GB for `(4, 3)`), so fetching is
mostly for machines that can hold it.

## Run the notebooks

```bash
sage -n jupyter          # native Sage install
# or, inside the conda environment:
jupyter lab
```

Then open one of:

| notebook | purpose |
|---|---|
| [`notebooks/tutorials/`](notebooks/tutorials/) | the tutorial series (start with `00_quick_start.ipynb`) |
| [`notebooks/model_builder.ipynb`](notebooks/model_builder.ipynb) | point-and-click model builder (the `ModelUI` form) |
| [`notebooks/model_runner.ipynb`](notebooks/model_runner.ipynb) | load and run any saved `models/*.model.py` |
| [`notebooks/examples/`](notebooks/examples/) | one notebook per capability, each with a simulation overlay |
| [`notebooks/legacy/`](notebooks/legacy/) | the earlier guided tour of the hand-written model builder |

If a notebook reports its kernel (`sagemath-10.8`) is missing — e.g. your Sage registered a
differently-named kernel — just pick your **SageMath** kernel from Jupyter's kernel menu.

## Dependencies (tested versions)

| component | tested version | source | needed for |
|---|---|---|---|
| **SageMath** | **10.8** | *install this* | the whole framework |
| Python | 3.13 | Sage | — |
| numpy / scipy / sympy / mpmath | 2.3.2 / 1.15.2 / 1.13.2 / 1.3.0 | Sage | core computation |
| matplotlib | 3.10.5 | Sage | plotting |
| networkx | 3.5 | Sage | diagram / report bookkeeping |
| cysignals | 1.12.6 | Sage | interruptible long computations |
| JupyterLab / notebook / ipykernel | 4.5 / 7.5 / 7.1 | Sage | running the notebooks |
| ipywidgets | 8.1.1 | Sage | the graphical model builder (`ModelUI`) |
| **numba** (+ llvmlite) | 0.65.1 (+ 0.47.0) | **extra** (`sage -pip` / conda) | loop-integrator speed; **required** by the numba simulation cells in `examples/` (the pipeline itself falls back to pure Python without it) |
| pydot | 4.0.1 | extra (optional) | optional diagram / prediagram plots (needs the `graphviz` binary) |
| pyperclip | — | extra (optional) | optional copy buttons in the GUI |

Everything marked *Sage* comes with a standard SageMath 10.8 install; only the **extra** rows
are installed separately (Option A folds them into `environment.yml`; Option B uses
`sage -pip install`).

## Tests (optional)

```bash
sage -python -m pytest tests/ -q
```

## Changes between versions

See [`CHANGELOG.md`](CHANGELOG.md). Version 0.2.0 changes the numbers of some models with
instantaneous (δ) propagator parts, and integrates every Phase J region exactly, without an
integration box, which moves models with slow modes (for example the one-loop of
`single_population_quad_exp_test` by up to 6 %). `DAEDALUS_PHASE_J_DD_KERNELS=0` restores the
boxed integration and `DAEDALUS_PHASE_J_LEGACY=1` the pre-0.2.0 numbers. Files saved with
`dd.save_npz` record the version that wrote them, and `dd.load_npz` warns when a file predates
the change.

## Citing

If you use Daedalus in your research, please cite the software ([`CITATION.cff`](CITATION.cff);
GitHub's "Cite this repository" button gives the same in other formats):

```bibtex
@software{daedalus,
  author  = {Szuromi, Matthew and Ocker, Gabriel K.},
  title   = {Daedalus: automated loop corrections for stochastic dynamics (MSRJD field theory framework)},
  url     = {https://github.com/mszuromi/daedalus},
  note    = {version as in CITATION.cff},
  year    = {2026}
}
```

## License

BSD 3-Clause; see [`LICENSE`](LICENSE).

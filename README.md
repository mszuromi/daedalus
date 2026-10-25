# Daedalus

A model-agnostic framework for computing stationary statistics of Itô stochastic
differential equations. The framework performs automated evaluation of Feynman diagrams
for perturbative MSR–JD (Martin–Siggia–Rose–Janssen–De Dominicis) field theories, built
on SageMath. Given an SDE/SPDE written as an MSR–JD action, it computes the mean field,
multi-point cumulants, and loop corrections (action → diagram enumeration → causal
integration). The notebook front-end is the `daedalus` module, imported as `dd`.

---

## Quick start

With conda or mamba installed ([Miniforge](https://github.com/conda-forge/miniforge) if you
have neither):

```bash
git clone https://github.com/mszuromi/daedalus.git
cd daedalus
conda env create -f environment.yml     # or:  mamba env create -f environment.yml   (faster)
conda activate daedalus
jupyter lab                             # then open notebooks/model_builder_tutorial.ipynb
```

[`environment.yml`](environment.yml) pulls SageMath 10.9 and the extras from conda-forge —
that is the entire setup. The sections below explain each step, give a non-conda alternative,
and list tested versions.

## Requirements

The one prerequisite is **SageMath 10.9**. Sage provides Python 3.14 and almost the entire
stack — numpy, scipy, sympy, matplotlib, mpmath, networkx, cysignals, the Jupyter stack
(JupyterLab / notebook / ipykernel / **ipywidgets**), and a `SageMath` Jupyter kernel. Only a
couple of extras are added on top (`numba`, optionally `pydot`).

> Other Sage 10.x releases will likely work, but 10.9 is what the code and the embedded
> notebook kernel (`sagemath-10.9`) are tested against (10.8 was, up to v0.3.1).

## Installation

### Option A — conda environment file (recommended: reproducible, cross-platform)

Works on macOS, Linux, and Windows (via WSL2); needs conda or mamba
([Miniforge](https://github.com/conda-forge/miniforge) if you have neither). From the cloned
repo root:

```bash
conda env create -f environment.yml      # or:  mamba env create -f environment.yml   (faster)
conda activate daedalus
```

[`environment.yml`](environment.yml) installs SageMath 10.9 plus the `numba`/`pydot`/`graphviz`
extras from conda-forge (validated to resolve on macOS/Linux). For the optional GUI clipboard,
also `pip install pyperclip`.

In the activated environment plain `python` is Sage's interpreter, and the conda-forge `sage`
launcher has no `-python` flag. Wherever this README or CONTRIBUTING says `sage -python`, run
`python` instead (for example `python -m pytest tests/ -q`, `python -m daedalus run config.yaml`).
The conda environment registers a Jupyter kernel named `sagemath`; if a notebook asks for
`sagemath-10.9`, pick it from the kernel menu.

### Option B — native SageMath + extras

1. Install **SageMath 10.9** from the official downloads:
   <https://www.sagemath.org/download.html> (macOS app, Linux binary, or WSL2 on Windows).
   This registers the `sagemath-10.9` Jupyter kernel the notebooks expect.
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
model, mod = dd.load_model('kpz_1d')
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

Equations name their operators (`api/operators.py`). The time derivative is `Dt` by default, so the models above
need no declaration; `.operator('dt', TimeDerivative())` renames it. An operator is written in product style
(`(tau*dt + 1)*v`, `D*lap*u`) or call style (`dt(v)`, `lap(u)`). An operator name may not collide with a field,
parameter or function. A `Laplacian()` together with `SDE(name, space=d)` makes the model an SPDE in `d` dimensions,
with space-time white noise `Gaussian(var=..., space='white')`. It builds the same model as a hand-written
`SpatialModelBuilder` action. One-loop stochastic Allen-Cahn, $\partial_t\phi = D\nabla^2\phi - \mu\phi - \lambda\phi^3 + \eta$:

```python
from api.sde import SDE
from api.operators import TimeDerivative, Laplacian
from api.noise import Gaussian
from api import compute_cumulants
m = (SDE('Allen-Cahn', space=1)                      # space=d: an SPDE in d dimensions
     .operator('dt', TimeDerivative()).operator('lap', Laplacian())
     .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
     .parameter('lam', default=0.1, domain='positive').parameter('T', default=1.0, domain='positive')
     .field('phi')
     .equation('phi', lhs='dt(phi)', rhs='D*lap(phi) - mu*phi - lam*phi^3')
     .noise('eta', Gaussian(var='2*T', space='white'), into={'phi': 1}))
res = compute_cumulants(m.build(), k=2, max_ell=1, external_fields=[('phi', 1), ('phi', 2)],
                        tau_max=1.0, tau_step=0.5, spatial_grid=(0.0, 4.0, 9))
```

A Laplacian of an expression (`g*lap(phi^2)`, a conserved derivative vertex) is supported, through the spatial
operator IR. Temporal-only features with a Laplacian are refused, not dropped: populations, kernels and delays, point
processes, non-Gaussian noise, and the Stratonovich and Marcus rules.

Before computing anything, an `SDE` can show what it derived. `m.check()` prints and returns a report (a dict) with the
declarations, the derived action split by origin (one term per field and per noise source), the mean-field equations,
the interpretation corrections (the Stratonovich or Marcus terms, each labelled with its source), the vertex table (the
numbers of response and field legs, the coefficient at the saddle, the degree and the loop orders a vertex can enter),
the propagator structure, and warnings (a truncated Marcus series, terms that enter the mean field only, the
analyzer's findings). It integrates nothing; it costs the expansion of the action, about a second for a small model.
`m.latex()` returns LaTeX strings (`'sde'`, `'noise'`, `'action'`, `'mean_field'`) and compiles nothing. A malformed
expression is refused when it is declared, with a caret under the failing character, the field or noise it belongs
to, and a hint:

```python
from api.sde import SDE
from api.noise import Gaussian
m = (SDE('OU quartic').parameter('mu', default=1.0, domain='positive')
     .parameter('eps', default=0.02, domain='positive').parameter('D', default=1.0, domain='positive')
     .field('x').drift('x', '-mu*x - eps*x^3').noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
report = m.check()                  # prints the report below; report['vertices']['rows'] is the vertex table
tex = m.latex()                     # tex['sde'] is dx = (-eps x^3 - mu x) dt + sqrt(2 D) dW_xi, as LaTeX
```

```text
Vertices (Taylor order 4; loops listed for k=2, max_ell=1)
  resp  field  degree  kind              loops  monomial -> coefficient
     1      2       3  interaction           1  xt1*dx1^2 -> 3*eps*xstar1
     1      3       4  interaction           1  xt1*dx1^3 -> eps
```

```text
equation('x') rhs: undeclared symbol 'epsilon'
    -mu*x - epsilon*x^3
            ^
    hint: did you mean 'eps'? (otherwise declare 'epsilon')
```

### Spectra and integrated correlators (k = 2)

`spectra=` asks for exact spectral and integrated outputs, computed from the state-space triples of the run with no
lag grid and no quadrature (`api/spectra.py`). They are given per loop order and in total:

```python
import numpy as np
import daedalus as dd
from api import compute_cumulants
model = dd.load_model('ou_quartic')[0]
res = compute_cumulants(model, k=2, max_ell=1, external_fields=[('dx', 1)] * 2,
                        tau_grid=np.array([0.0]),             # the equal-time point is enough
                        spectra={'omega': (0.0, 10.0, 201), 'T': [1.0, 10.0]})
sp = res['spectra']       # also dd.Config(spectra=...), dd.spectra(res, omega, T), YAML question.spectra
sp['S'], sp['S_by_ell']   # S(omega) on the grid
sp['S0']                  # S(0)
sp['integral']            # int_0^T C for each T (integral_neg: int_{-T}^0)
sp['integral_inf']        # int_0^inf C
sp['tau_c']               # int_0^inf C / C(0)
```

- **Conventions.** `C(tau) = kappa(x_1(t), x_2(t + tau))`, as `C_tau`. The spectrum is
  `S(omega) = int C(tau) exp(+i omega tau) dtau`, with no `1/2pi`; for a cross pair it is `<x1^(omega) x2^(omega)*>`,
  and `S_21 = conj(S_12)`. `Cumulant.spectrum` uses the opposite sign of the exponent.
- **Shot-noise spikes.** The spike `w delta(tau)` at `tau = 0` is in `S` (as a flat floor `w`) and in `delta0`, and in
  neither window, so `S(0) = integral_inf + integral_inf_neg + delta0`.
- **Equal times.** `C0` is the Ito left limit `C(0-)`.
- **Truncated correlation time.** `tau_c_by_ell[l]` is the correlation time of the cumulant truncated at order `l`.
- **Fallback.** Loop orders without triples (pole engine, pole-cluster deformation) use adaptive quadrature, report
  error estimates and give an `MEFallbackWarning`; `method='exact'` refuses them instead.
- **Refused.** Spatial models, switch-on runs and `discard=True` are refused.

### Covariance matrix over many fields, and parameter derivatives (k = 2)

`dd.covariance_matrix` returns `Sigma[n, i, j] = <dx_i(0) dx_j(tau_n)>` for every pair of a list of fields in one call
(`api/covariance.py`). The mean field, propagator and poles are computed once for the whole matrix, and each pair's
diagrams are enumerated and evaluated once. Every computed entry is the separate `compute_cumulants` result on the
same grid, bit for bit:

```python
import numpy as np
import daedalus as dd
model = dd.load_model('ou_quartic_coupled')[0]
cm = dd.covariance_matrix(model, ['x', 'y'], 1, tau_grid=np.array([-2.0, 0.0, 2.0]),
                          derivatives=['J1'])     # also dd.Config(covariance={...})
cm['Sigma']                    # (n_tau, 2, 2), complex like C_tau; per loop order: cm['Sigma_by_ell']
cm['Sigma0']                   # the tau = 0 matrix
d = cm['derivatives']['J1']
d['dSigma'], d['error']        # dSigma/dJ1 and its error estimate
d['dmf']                       # dx*/dJ1 of the saddle (implicit differentiation)
```

- **Lower triangle.** The pairs `i <= j` are run. The `(j, i)` entry is `C_ij(-tau)` where `-tau` is on the grid and
  `tau != 0`. The other points get their own `(j, i)` run: `tau = 0` is the Ito left limit `C_ji(0-)`, which differs
  from `C_ij(0)` when `C_ij` jumps at 0, as a spike train's cross-covariance can. `pairs='all'` runs every ordered pair.
- **Derivatives.** They are fourth-order central differences (steps `h` and `2h`, `h = 1e-3 |theta|`) of the shared
  pass, with four extra passes per parameter. `error` is a conservative estimate. A stencil whose saddles leave the
  base point's branch is refused with `CovarianceDerivativeError`.
- **Options.** The other `compute_cumulants` options are passed to every run, and each run keeps its own memory
  guard. Spatial models, `t_min` and `spectra` are refused.

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

## Run the notebooks

```bash
sage -n jupyter          # native Sage install
# or, inside the conda environment:
jupyter lab
```

Then open one of:

| notebook | purpose |
|---|---|
| [`notebooks/model_builder_tutorial.ipynb`](notebooks/model_builder_tutorial.ipynb) | guided tour — build a temporal model and run it |
| [`notebooks/model_builder.ipynb`](notebooks/model_builder.ipynb) | point-and-click model builder (the `ModelUI` form) |
| [`notebooks/model_runner.ipynb`](notebooks/model_runner.ipynb) | load and run any saved `models/*.model.py` |
| [`notebooks/examples/`](notebooks/examples/) | one notebook per capability, each with a simulation overlay |

If a notebook reports its kernel (`sagemath-10.9`) is missing — e.g. your Sage registered a
differently-named kernel — just pick your **SageMath** kernel from Jupyter's kernel menu.

## Dependencies (tested versions)

| component | tested version | source | needed for |
|---|---|---|---|
| **SageMath** | **10.9** | *install this* | the whole framework |
| Python | 3.14 | Sage | — |
| numpy / scipy / sympy / mpmath | 2.4.4 / 1.17.1 / 1.14.0 / 1.4.1 | Sage | core computation |
| matplotlib | 3.10.8 | Sage | plotting |
| networkx | 3.6.1 | Sage | diagram / report bookkeeping |
| cysignals | 1.12.6 | Sage | interruptible long computations |
| JupyterLab / notebook / ipykernel | 4.6 / 7.6 / 7.3 | Sage | running the notebooks |
| ipywidgets | 8.1.8 | Sage | the graphical model builder (`ModelUI`) |
| **numba** (+ llvmlite) | 0.68.0 (+ 0.50.0) | **extra** (`sage -pip` / conda) | loop-integrator speed; **required** by the numba simulation cells in `examples/` (the pipeline itself falls back to pure Python without it) |
| pydot | 4.0.1 | extra (optional) | optional diagram / prediagram plots (needs the `graphviz` binary) |
| pyperclip | — | extra (optional) | optional copy buttons in the GUI |

Everything marked *Sage* comes with a standard SageMath 10.9 install; only the **extra** rows
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

If you use this software, please cite it — see [`CITATION.cff`](CITATION.cff).

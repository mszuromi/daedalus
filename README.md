# MSRJD Field Theory Framework

A model-agnostic framework for automated Feynman-diagram computation in MSR–JD
(Martin–Siggia–Rose–Janssen–De Dominicis) field theories, built on SageMath. Given a
stochastic (S)ODE/PDE written as an MSR–JD action, it computes the mean field, multi-point
cumulants, and loop corrections **analytically** (action → diagram enumeration → causal
integration). The notebook front-end is the `daedalus` module, imported as `dd`.

---

## Quick start

With conda or mamba installed ([Miniforge](https://github.com/conda-forge/miniforge) if you
have neither):

```bash
git clone <repository-url>
cd <repository>
conda env create -f environment.yml     # or:  mamba env create -f environment.yml   (faster)
conda activate daedalus
jupyter lab                             # then open notebooks/theory_builder_tutorial.ipynb
```

[`environment.yml`](environment.yml) pulls SageMath 10.8 and the extras from conda-forge —
that is the entire setup. The sections below explain each step, give a non-conda alternative,
and list tested versions.

## Requirements

The one prerequisite is **SageMath 10.8**. Sage provides Python 3.12+ and almost the entire
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
git clone <repository-url>
cd <repository>
```

There is **no build or install step for the framework itself** — it runs straight from the
clone. The notebooks add the repository to the Python path automatically; from a script, do
`sys.path.insert(0, '<repo>'); sys.path.insert(0, '<repo>/notebooks')`.

## Verify the install

From the repository root:

```bash
sage -python - <<'PY'
import sys; sys.path[:0] = ['.', 'notebooks']
import daedalus as dd
model, mod = dd.load_theory('ou_quartic')
res = dd.run(model, dd.Config(k=2, max_ell=0), mod)
print('OK —', dd.summary(res).splitlines()[0])
PY
```

Expected output: `OK — theory : 'OU Quartic (white noise)'`.

## Run the notebooks

```bash
sage -n jupyter          # native Sage install
# or, inside the conda environment:
jupyter lab
```

Then open one of:

| notebook | purpose |
|---|---|
| [`notebooks/theory_builder_tutorial.ipynb`](notebooks/theory_builder_tutorial.ipynb) | guided tour — build a temporal theory and run it |
| [`notebooks/theory_builder.ipynb`](notebooks/theory_builder.ipynb) | point-and-click theory builder (the `TheoryUI` form) |
| [`notebooks/theory_runner.ipynb`](notebooks/theory_runner.ipynb) | load and run any saved `theories/*.theory.py` |
| [`notebooks/examples/`](notebooks/examples/) | one notebook per capability, each with a simulation overlay |

If a notebook reports its kernel (`sagemath-10.8`) is missing — e.g. your Sage registered a
differently-named kernel — just pick your **SageMath** kernel from Jupyter's kernel menu.

## Dependencies (tested versions)

| component | tested version | source | needed for |
|---|---|---|---|
| **SageMath** | **10.8** | *install this* | the whole framework |
| Python | 3.12 (conda) / 3.13 (native) | Sage | — |
| numpy / scipy / sympy / mpmath | 2.3.2 / 1.15.2 / 1.13.2 / 1.3.0 | Sage | core computation |
| matplotlib | 3.10.5 | Sage | plotting |
| networkx | 3.5 | Sage | diagram / report bookkeeping |
| cysignals | 1.12.6 | Sage | interruptible long computations |
| JupyterLab / notebook / ipykernel | 4.5 / 7.5 / 7.1 | Sage | running the notebooks |
| ipywidgets | 8.1.1 | Sage | the graphical theory builder (`TheoryUI`) |
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

## Citing

If you use this software, please cite it — see [`CITATION.cff`](CITATION.cff).

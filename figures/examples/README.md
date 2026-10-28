# Worked examples: figures and data

Each folder regenerates one worked-example figure of the paper, together with every number behind it, from the model
as typed in the SDE front end.

| folder | model | figure |
|---|---|---|
| [`reaction_diffusion/`](reaction_diffusion/) | two-species reaction–diffusion SPDE in $d=1$, cubic self-interaction | `fig_rd_two_species` |
| [`hawkes_delay/`](hawkes_delay/) | two-population exponential Hawkes network with delayed alpha synapses | `fig_hawkes_delay` |
| [`cmos_neat_rn/`](cmos_neat_rn/) | two subthreshold CMOS inverters joined by a capacitance matrix (exact jump process) | `fig_cmos_scaling` |

The running example (the quartic Ornstein–Uhlenbeck process, Fig. 1) is in [`notebooks/paper_figures.ipynb`](../../notebooks/paper_figures.ipynb).

## Layout of a folder

- `model.py` — the model in the SDE front end (`build()` returns the unbuilt `SDE`; the action and mean-field equations
  are derived from it).
- `compute.py` — every pipeline run and simulation behind the figure, as functions and as a command line
  (`sage -python compute.py ...`; the docstring lists the jobs and their cost).
- `simulator.py` — an independent simulator (no diagrams) where the package has none.
- `data/` — the JSON (and NPZ) files the figure is drawn from, written by `compute.py` with Daedalus v0.4.0.
- `<name>.ipynb` — the notebook: prints the model, loads (or, with the `RECOMPUTE` flags, regenerates) the data,
  prints the comparison tables, and draws and saves the figure (`fig_*.pdf`, `fig_*.png`).

## Running

The notebooks use the SageMath kernel of the pinned environment (`environment.yml`) and find the repository root by
walking up from their own folder, so no installation is needed:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=sagemath-10.9 figures/examples/reaction_diffusion/reaction_diffusion.ipynb
```

With the `RECOMPUTE` flags off (the default) a notebook runs in under a minute from the shipped data. The long jobs
are easier to run from a terminal, for example

```bash
cd figures/examples/reaction_diffusion && SPATIAL_MEM_BUDGET_GB=20 PYTHONHASHSEED=0 sage -python compute.py pipeline eq2
```

Seeds are fixed, so the simulations reproduce the shipped files bit for bit on the same platform; pipeline values are
deterministic with `PYTHONHASHSEED=0`. The Padé order of each delay is pinned in `compute.py` and recorded in the
data files: the automatic order raises the order only while the predicted run time of the next one stays under a wall
limit, so left automatic it would depend on how fast and how busy the machine is.

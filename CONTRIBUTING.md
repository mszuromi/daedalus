# Contributing to Daedalus

## Setup

See the [README](README.md): install **SageMath 10.8** (conda `environment.yml`, or native),
then optionally `sage -pip install -e .` so `import daedalus` works from anywhere.

All Python runs under Sage's interpreter — use `sage -python ...`, never plain `python`.

## Running the tests

```bash
sage -python -m pytest tests/ -q       # default suite (the slow tests are deselected)
sage -python -m pytest -m slow         # the minutes-long ones (coupled-Dyson loops, k>=3 spatial)
```

`pytest.ini` sets `addopts = -m "not slow"`, so a bare run finishes in a few minutes and stays
green. Mark any new test that takes minutes `@pytest.mark.slow`.

## Adding a theory

A theory is a `theories/<name>.theory.py` file that builds a model dict with
`TemporalTheoryBuilder` / `SpatialTheoryBuilder` — see
[`notebooks/theory_builder_tutorial.ipynb`](notebooks/theory_builder_tutorial.ipynb) and
`api/theory.py`. It must expose `build()`, `DEFAULT_FUNDAMENTAL`, and `METADATA`. Load it with
`dd.load_theory('<name>')`; it then appears in `dd.list_theories()`. Confirm it runs at its own
defaults (`dd.run(*dd.load_theory('<name>'))`) before committing — a shipped theory that fails
on run is worse than no theory.

To validate it against simulation, add a matching simulator under `simulations/`
(Euler–Maruyama for SDEs, spectral ETD1 for SPDEs) and overlay it in an example notebook
(`notebooks/examples/`).

## Layout

See [ARCHITECTURE.md](ARCHITECTURE.md) for the `dd → api → engine → simulations` tiers.

## Issues

Report bugs and requests on the [issue tracker](https://github.com/mszuromi/daedalus/issues).

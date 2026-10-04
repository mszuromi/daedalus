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

Diagram lists have no fixed order: it depends on where the prediagrams came from (a cache file,
the in-memory streaming enumerator that spatial runs use, or the eager enumerator that temporal
`use_cache=False` runs use; `DAEDALUS_PREDIAGRAM_EAGER=1` or `=0` forces one of the last two).
A test must pick a diagram by its structure (`_pick_live` in `tests/_diagram_order.py`), never by
its position in a list. To probe for position dependence, run tests with the diagram lists
reordered: `DAEDALUS_TEST_DIAGRAM_ORDER=shuffle sage -python -m pytest tests/...` shuffles the
prediagram records and, for each loop order, the typed diagrams with their multiplicities (so the
diagrams of one prediagram move among themselves too); `=reverse` reverses each loop order's
typed diagrams; `DAEDALUS_TEST_DIAGRAM_SEED` picks the shuffle. One permutation can leave a
positional pick in place, so a passing run is evidence, not proof. A single test can ask for the
same with the `shuffle_diagram_order` fixture (or `apply_diagram_order`), which replaces the
whole-run order for that test rather than composing with it; a module-scoped fixture computed
before it still carries the whole-run order. `tests/test_prediagram_cache.py` pins the exact
record order and undoes the reordering for its own tests (`unwrapped_order_patches`); do the
same in any module that asserts list order (see `tests/conftest.py`).

Temporal `use_cache=False` runs keep the eager enumerator only because Phase J's `scipy.nquad`
fallback is not yet representative-independent (`TEMPORAL_CACHE_OFF_STREAMS` in
`engine/enumeration/prediagram_cache.py`). After any change to that fallback, run its slow gate,
`sage -python -m pytest -m slow tests/test_prediagram_cache.py -k fallback_model` (about 7 min;
`-n 3` runs its three cases side by side). Each case is a strict expected failure on one model
that reaches the fallback; flip the flag only when every case fails as an unexpected pass.

## Adding a model

A model is a `models/<name>.model.py` file that builds a model dict with
`TemporalModelBuilder` / `SpatialModelBuilder` — see
[`notebooks/model_builder_tutorial.ipynb`](notebooks/model_builder_tutorial.ipynb) and
`api/model.py`. It must expose `build()`, `DEFAULT_FUNDAMENTAL`, and `METADATA`. Load it with
`dd.load_model('<name>')`; it then appears in `dd.list_models()`. Confirm it runs at its own
defaults (`dd.run(*dd.load_model('<name>'))`) before committing — a shipped model that fails
on run is worse than no model.

To validate it against simulation, add a matching simulator under `simulations/`
(Euler–Maruyama for SDEs, spectral ETD1 for SPDEs) and overlay it in an example notebook
(`notebooks/examples/`).

## Layout

See [ARCHITECTURE.md](ARCHITECTURE.md) for the `dd → api → engine → simulations` tiers.

## Issues

Report bugs and requests on the [issue tracker](https://github.com/mszuromi/daedalus/issues).

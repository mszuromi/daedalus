#### One `daedalus` package: `api`, `engine`, `simulations` and `models` move under it

The library used to install three generic top-level packages (`api`, `engine`, `simulations`) next to a
`daedalus.py` module. Those names clash with other packages in a shared `site-packages`, which blocked a PyPI
release. Everything is now one package: `daedalus/__init__.py` is the former `daedalus.py` (`import daedalus as dd`
is unchanged), and the library lives in `daedalus.api`, `daedalus.engine` and `daedalus.simulations`. The shipped
model files are now in `daedalus/models/`. The PyPI distribution is named `daedalus-msrjd`
(`sage -pip install daedalus-msrjd`); the import name stays `daedalus`.

- **Deprecated old names, for one release.** `import api`, `import engine.<...>` and `import simulations.<...>` still
  work, with one `DeprecationWarning` per name, and give the same module objects as `daedalus.<...>` (no second copy:
  module state, classes and exceptions are shared). Old pickles and caches that name `engine.*` / `api.*` classes
  still load once `daedalus` has been imported (a bare `pickle.load` in a fresh process does not import it, so `import
  daedalus` first; the package's own loaders do). The modules keep their own `__spec__`, so `importlib.reload` and
  autoreload work. A user's own top-level `api` / `engine` / `simulations` package always takes precedence (a bare
  folder of that name does not count). In a source checkout, the top-level `api/`, `engine/` and `simulations/`
  folders now hold only a forwarding `__init__.py` and are not installed; a local module git does not track in one of
  them (`simulations/my_sim.py`) still imports under its old name.
- **Model files.** `dd.load_model(name)` / `dd.list_models()` search, in order: the directories in the new
  `DAEDALUS_MODELS_PATH` (separated like `PATH`), a clone's top-level `models/` folder (your own local model files
  keep loading after the tracked ones moved), then the shipped `daedalus/models/`. A name found in two places
  warns, and the first one wins. `load_model` also accepts a path to a `*.model.py` file. New: `dd.model_dirs()`,
  `dd.MODELS_PATH_ENV`, `dd.PACKAGE_DIR`. `dd.MODELS_DIR` is now the shipped `daedalus/models/`, and
  `dd.REPO_ROOT` is the folder that holds the package (the clone's root).
- **Installed package.** The shipped prediagram cells, their manifests, the model files and the Cython sources
  are package data, found through `importlib.resources`. pyximport builds the compiled enumerator into
  `<clone>/.pyxbld` as before, or into `~/.cache/daedalus/pyxbld` for an installed package (override:
  `DAEDALUS_BUILD_DIR`). It recompiles once after this change, because the module names changed.
  `python -m daedalus run config.yaml` now works through `daedalus/__main__.py`. `import daedalus` no longer
  imports `matplotlib.pyplot`; it is imported when a plot is first drawn.
- **Generated model files** (`ModelUI`, `model_serialize`) now write `from daedalus.api.model import ...`. The
  `ModelUI` Load list shows the models of every model directory (the shipped ones included), not only its save
  folder.
- **Tests.** `tests/test_namespace_compat.py` covers the alias (identity, one warning per name, a user's own `api`
  package wins, old pickles, the forwarding folders), the packaging config (only `daedalus*`; every tracked data
  file under `daedalus/` is listed as package data), the Sage-free and pyplot-free spawn-worker import,
  `python -m daedalus`, and the model search path. The tests that walk the source tree (ticket-number guard, OBJC
  fork-safety guard, error-class hierarchy) now assert a minimum number of files scanned, so a later move cannot
  silently empty them.
- No numbers move. 21 Phase J zoo entries (`tests/tools/phase_j_zoo_baseline.py`: OU quartic k = 2 to ℓ = 3, k = 3
  and k = 4 to ℓ = 2, coupled and colored OU, spike reset, quadratic-exponential, linear Hawkes, multi-population,
  delta spikes, dendritic, quadratic Hawkes alpha, spatial ℓ ≤ 1) give equal values, counters and diagram sources
  before and after the move, and the same three refusals. `dd.run` on `ou_quartic` from an installed wheel matches
  main byte for byte. Zoo records now show `daedalus/models/<name>.model.py` as the model path, and a new model
  file hash (only its import line changed).

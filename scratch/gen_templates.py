"""Generate the 4 group template notebooks under notebooks/templates/.

All four share ONE skeleton (load → run → plot via daedalus); only the
title/description markdown and the single Config cell differ per group, so
the demos are maximally uniform.  Run with `sage -python`.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'notebooks', 'templates')
os.makedirs(OUT, exist_ok=True)


def md(*lines):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': list(_split(lines))}


def code(*lines):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': list(_split(lines))}


def _split(lines):
    """Join the given line-strings with newlines into nbformat source list
    (each element ends with \n except the last)."""
    text = '\n'.join(lines)
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


# ── the shared cells (identical across all four templates) ───────────────────

ROOT_CELL = code(
    "%matplotlib inline",
    "import os, sys",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "# depth-robust repo root: walk up until the 'pipeline' package is found",
    "_root = os.path.abspath('')",
    "while _root != os.path.dirname(_root) and not os.path.isdir("
    "os.path.join(_root, 'pipeline')):",
    "    _root = os.path.dirname(_root)",
    "sys.path.insert(0, _root)",
    "sys.path.insert(0, os.path.join(_root, 'notebooks'))",
    "import daedalus as dd",
    "print('daedalus \\u2192', dd.REPO_ROOT)",
)

LOAD_CELL = code(
    "model, mod = dd.load_theory(THEORY)",
    "print('loaded :', model.get('name'))",
    "print('fields :', dd.field_names(model),",
    "      '| spatial_dim:', dd.spatial_dim(model),",
    "      '| multi-field:', dd.is_multifield(model))",
    "print('params :', [p['name'] for p in (model.get('parameters') or [])])",
)

RUN_CELL = code(
    "res = dd.run(model, cfg, mod)",
    "print(dd.summary(res))",
)

PLOT_CELL = code(
    "fig = dd.plot_cumulant(res, cfg, model)",
    "plt.show()",
)


def header_md(title, group, shape, swap_hint):
    return md(
        f"# {title} — pipeline demo template",
        "",
        f"The **{group}** template.  Every demo in this group follows the same "
        "three steps — **load → run → plot** — all driven by "
        "[`daedalus.py`](../daedalus.py):",
        "",
        "1. **Load** a theory from `theories/<name>.theory.py` (the single "
        "source of truth — no inline model building).",
        "2. **Run** it with one `dd.Config(...)`: the correlator order `k`, the "
        "loop order `max_ell`, the Dyson order, the grids, and every plotting "
        "option live there.",
        "3. **Plot** with `dd.plot_cumulant`, which auto-dispatches to the form "
        "natural to this group.",
        "",
        f"This template ships with **`{shape}`**.  To demo a different "
        f"{group.split(' ')[0]} theory, {swap_hint}  Everything else is "
        "identical across all four group templates on purpose — common "
        "thematics for the demos.",
    )


SETUP_MD = md("## 1. Setup")
CONFIG_MD = md(
    "## 2. Choose the theory & configure the run",
    "",
    "This is the only cell you edit.  `k` and `max_ell` are free (arbitrary "
    "correlator order and loop order); the plotting options "
    "(`show_orders`, `logy`, `figsize`, …) are adaptable.",
)
RUN_MD = md("## 3. Run")
PLOT_MD = md(
    "## 4. Plot",
    "",
    "`show_orders` controls the per-loop-order overlay: `'cumulative'` "
    "(tree, tree+1-loop, …), `'incremental'` (each order alone), or "
    "`'total'` (the summed result only).",
)


def build_nb(header, config_cell):
    cells = [header, SETUP_MD, ROOT_CELL, CONFIG_MD, config_cell, LOAD_CELL,
             RUN_MD, RUN_CELL, PLOT_MD, PLOT_CELL]
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'SageMath 10.8',
                           'language': 'sage', 'name': 'sagemath-10.8'},
            'language_info': {'name': 'sage'},
        },
        'nbformat': 4, 'nbformat_minor': 5,
    }


# ── the four group-specific (header, config) pairs ───────────────────────────

templates = {}

templates['temporal_single'] = (
    header_md(
        'Temporal · single-field', 'temporal single-field',
        "ou_quartic_double_well",
        "set `THEORY` below to any single-field temporal "
        "`theories/*.theory.py` (e.g. `ou_sextic`, "
        "`single_population_spike_reset_test`)."),
    code(
        "THEORY = 'ou_quartic_double_well'   # any single-field temporal theory",
        "",
        "cfg = dd.Config(",
        "    # --- what to compute (arbitrary k and loop order) ---",
        "    k=2,            # 2 = ⟨xx⟩, 3 = ⟨xxx⟩, …",
        "    max_ell=1,      # 0 = tree, 1 = +1-loop, 2 = +2-loop, …",
        "    # external_fields=[('dx', 1), ('dx', 1)],   # None → auto from theory",
        "    # output='cumulant', # 'cumulant' (default) | 'moment' | 'central_moment'",
        "",
        "    # --- temporal grid ---",
        "    tau_max=4.0,",
        "    tau_step=1.0,",
        "",
        "    # --- plotting (adaptable) ---",
        "    show_orders='cumulative',   # 'cumulative' | 'incremental' | 'total'",
        "    logy=False,",
        ")",
    ),
)

templates['temporal_multi'] = (
    header_md(
        'Temporal · multi-field', 'temporal multi-field',
        "ou_quartic_two_dim",
        "set `THEORY` below to any multi-field/multi-population temporal "
        "`theories/*.theory.py` (e.g. `ou_quartic_two_dim_corr`, "
        "`multipopulation_test`)."),
    code(
        "THEORY = 'ou_quartic_two_dim'   # any multi-field temporal theory",
        "",
        "cfg = dd.Config(",
        "    k=2,",
        "    max_ell=0,      # tree; bump to 1 for the 1-loop correction",
        "    # Multi-field: choose which legs to correlate.  None → auto",
        "    # ⟨field0 field0⟩.  For the cross-correlator ⟨xy⟩:",
        "    # external_fields=[('dx', 1), ('dy', 1)],",
        "    # output='cumulant', # 'cumulant' (default) | 'moment' | 'central_moment'",
        "",
        "    tau_max=8.0,",
        "    tau_step=0.5,",
        "",
        "    show_orders='cumulative',",
        "    logy=False,",
        ")",
    ),
)

templates['spatial_single'] = (
    header_md(
        'Spatial · single-field', 'spatial single-field',
        "kpz_1d",
        "set `THEORY` below to any single-field spatial "
        "`theories/*.theory.py` (e.g. `allen_cahn_1d_subcritical_infinite`, "
        "`burgers_1d`, `reaction_diffusion_quadratic_1d`)."),
    code(
        "THEORY = 'kpz_1d'   # any single-field spatial theory",
        "",
        "cfg = dd.Config(",
        "    k=2,",
        "    max_ell=1,      # 0 = tree, 1 = +1-loop, …",
        "    # output='cumulant', # 'cumulant' (default) | 'moment' | 'central_moment' (k=2)",
        "",
        "    # --- spatial grid: (lo, hi, n_points) or an explicit array ---",
        "    spatial_grid=(-6.0, 6.0, 49),",
        "    tau_max=0.0,    # > 0 adds a C(x, τ) heatmap panel",
        "    tau_step=1.0,",
        "",
        "    show_orders='cumulative',",
        "    logy=False,",
        "    # For k ≥ 3 set spatial_points=(n_pts, k-1, 2) of (x_j, τ_j)",
        "    # offsets instead of spatial_grid (see the k≥3 note below).",
        ")",
    ),
)

templates['spatial_multi'] = (
    header_md(
        'Spatial · multi-field', 'spatial multi-field',
        "coupled_rd_2species_1d",
        "set `THEORY` below to any multi-field spatial "
        "`theories/*.theory.py`."),
    code(
        "THEORY = 'coupled_rd_2species_1d'   # any multi-field spatial theory",
        "",
        "cfg = dd.Config(",
        "    k=2,",
        "    max_ell=1,      # 0 = tree, 1 = +1-loop, …",
        "    # output='cumulant', # 'cumulant' (default) | 'moment' | 'central_moment' (k=2)",
        "",
        "    spatial_grid=(-8.0, 8.0, 65),",
        "    tau_max=0.0,",
        "",
        "    # --- Dyson dressing (only when fields have UNEQUAL diffusion) ---",
        "    # The theory default is equal D (no Dyson).  For Da ≠ Db set the",
        "    # diffusion via `parameters` and pick a Dyson order ≥ 0:",
        "    # parameters={'Da': 0.8, 'Db': 0.5},",
        "    # dyson_order=1,",
        "",
        "    show_orders='cumulative',",
        "    logy=False,",
        ")",
    ),
)

KPOINT_MD = md(
    "### Arbitrary `k` (spatial)",
    "",
    "For 3-point and higher spatial cumulants, replace `spatial_grid` with "
    "explicit evaluation events: `spatial_points` is an `(n_pts, k-1, 2)` "
    "array giving, for each of the `k-1` non-anchor legs, its `(x_j, τ_j)` "
    "offset from the anchor.  `dd.plot_cumulant` then draws the per-event bar "
    "chart automatically.  Example (`k=3`, two events):",
    "",
    "```python",
    "cfg = dd.Config(k=3, max_ell=0, spatial_points=[",
    "    [[0.5, 0.0], [1.0, 0.0]],   # event 1: legs at x=0.5 and x=1.0",
    "    [[1.0, 0.0], [2.0, 0.0]],   # event 2",
    "])",
    "```",
)

for name, (header, config_cell) in templates.items():
    nb_obj = build_nb(header, config_cell)
    if name.startswith('spatial'):
        nb_obj['cells'].append(KPOINT_MD)
    path = os.path.join(OUT, f'template_{name}.ipynb')
    with open(path, 'w') as f:
        json.dump(nb_obj, f, indent=1)
    print('wrote', os.path.relpath(path, ROOT))

print('done')

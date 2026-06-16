"""Generate notebooks/templates/template_temporal_single_sim_compare.ipynb —
the worked reference showing how a deep-dive sim-vs-theory notebook adopts the
nb_support core (load -> run -> plot) and overlays a real simulator via
nb.plot_cumulant(..., sim=...).  Run with `sage -python`."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'notebooks', 'templates',
                   'template_temporal_single_sim_compare.ipynb')


def _split(lines):
    text = '\n'.join(lines)
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


def md(*lines):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': list(_split(lines))}


def code(*lines):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': list(_split(lines))}


cells = [
    md(
        "# Temporal · single-field — **sim vs theory** reference",
        "",
        "The same **load → run → plot** core as "
        "[`template_temporal_single.ipynb`](template_temporal_single.ipynb), "
        "with a matched simulator overlaid.  The only addition over the plain "
        "template is section 3 (the simulator) and passing its result to "
        "`nb.plot_cumulant(..., sim=...)`.",
        "",
        "Every `*_sim_compare` notebook in [`../temporal/`](../temporal) and "
        "[`../spatial/`](../spatial) follows this shape: the theory side is "
        "`nb_support`; only the simulator differs per model.",
    ),
    md("## 1. Setup"),
    code(
        "%matplotlib inline",
        "import os, sys, time",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "_root = os.path.abspath('')",
        "while _root != os.path.dirname(_root) and not os.path.isdir("
        "os.path.join(_root, 'pipeline')):",
        "    _root = os.path.dirname(_root)",
        "sys.path.insert(0, _root)",
        "sys.path.insert(0, os.path.join(_root, 'notebooks'))",
        "import nb_support as nb",
        "# simulator + cumulant estimator for THIS model",
        "from models.ou_langevin_sim_numba import sim_ou_quartic_numba",
        "from models.cumulant_estimator import compute_kpoint_slice",
        "print('nb_support \\u2192', nb.REPO_ROOT)",
    ),
    md(
        "## 2. Theory side (identical to the plain template)",
        "",
        "Load the theory from its file, set one `nb.Config`, run.",
    ),
    code(
        "THEORY = 'ou_quartic_double_well'",
        "",
        "cfg = nb.Config(",
        "    k=2,                # 2 = ⟨xx⟩ C(τ) curve; 3+ = equal-time κ_k scalar",
        "    max_ell=1,          # tree + 1-loop",
        "    tau_max=8.0,",
        "    tau_step=0.5,",
        "    show_orders='cumulative',",
        ")",
        "",
        "model, mod = nb.load_theory(THEORY)",
        "res = nb.run(model, cfg, mod)",
        "print(nb.summary(res))",
    ),
    md(
        "## 3. Simulation side (matched parameters)",
        "",
        "Pull the exact parameters and τ-grid the theory used out of the "
        "result (`res['_resolved']['fundamental']`, `res['tau_grid']`, the "
        "mean-field saddle `res['mf']`), run a few Euler–Maruyama trajectories, "
        "and estimate `C(τ)`.  The output is a plain "
        "`{tau, C, C_err}` dict — the only thing `nb.plot_cumulant` needs.",
    ),
    code(
        "fund = res['_resolved']['fundamental']",
        "mu, eps, D = float(fund['mu']), float(fund['eps']), float(fund['D'])",
        "x_init = float(res['mf']['x'][0])      # expand around the MF saddle",
        "",
        "# --- simulation knobs (kept small so this runs in seconds) ---",
        "N_RUNS = 4",
        "T_sim  = float(5e5)",
        "dt_sim = 0.05",
        "dt_bin = 0.5",
        "tau_max = float(cfg.tau_max)",
        "",
        "n_steps        = int(T_sim / dt_sim)",
        "bin_size_steps = max(int(round(dt_bin / dt_sim)), 1)",
        "dt_bin_eff     = bin_size_steps * dt_sim",
        "n_bins         = n_steps // bin_size_steps",
        "max_lag_bins   = int(tau_max / dt_bin_eff)",
        "tau_sim_grid   = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_bin_eff",
        "",
        "_ = sim_ou_quartic_numba(int(1000), float(dt_sim), mu, eps, D, x_init,",
        "                         int(bin_size_steps), int(100), int(0))  # JIT",
        "",
        "C_runs, var_runs, mean_runs = [], [], []",
        "t0 = time.perf_counter()",
        "for run in range(N_RUNS):",
        "    x_bins = sim_ou_quartic_numba(",
        "        int(n_steps), float(dt_sim), mu, eps, D, x_init,",
        "        int(bin_size_steps), int(n_bins), int(1234 + run))",
        "    var_runs.append(float((x_bins**2).mean()))",
        "    mean_runs.append(float(x_bins.mean()))",
        "    if int(cfg.k) == 2:                       # 2-point C(τ) estimate",
        "        _, C_run = compute_kpoint_slice(",
        "            binned_counts=np.zeros_like(x_bins), dt_bin=float(dt_bin_eff),",
        "            pop_indices=[0, 0], lag_bins=[0, None],",
        "            max_lag_bins=int(max_lag_bins), field_types=['dv', 'dv'],",
        "            voltage_bins=x_bins)",
        "        C_runs.append(np.asarray(C_run).real)",
        "",
        "print(f'sim: {N_RUNS} runs x T={T_sim:.0g} in "
        "{time.perf_counter()-t0:.1f}s')",
        "print(f'  <x>   = {np.mean(mean_runs):+.4f}    "
        "<x^2> = {np.mean(var_runs):.4f}')",
        "",
        "if int(cfg.k) == 2:",
        "    # 2-point correlator: overlay the simulated C(τ) on the theory curve.",
        "    C_runs = np.array(C_runs)",
        "    sim = {'tau': tau_sim_grid, 'C': C_runs.mean(axis=0),",
        "           'C_err': C_runs.std(axis=0, ddof=1) / np.sqrt(N_RUNS)}",
        "    _mid = len(sim['C']) // 2",
        "    _i0  = int(np.argmin(np.abs(np.asarray(res['tau_grid']))))",
        "    print(f'  sim    C(0) = {sim[\"C\"][_mid]:.4f}  "
        "(bin-averaged, dt_bin={float(dt_bin):.2f})')",
        "    print(f'  theory C(0) = {float(np.real(res[\"C_tau\"])[_i0]):.4f}  "
        "(tree + {cfg.max_ell}-loop)')",
        "else:",
        "    # k≥3: there is no C(τ) curve — the connected k-point cumulant is a",
        "    # SCALAR per loop order (res['C_tau'] is None; res['total_C'] is a",
        "    # callable).  The 2-point OU C(τ) overlay above is k=2-specific, so",
        "    # we report the THEORY κ_k here and skip the sim overlay.",
        "    sim = None",
        "    _be = {e: (0.0 if v is None else float(np.real(np.asarray(v))))",
        "           for e, v in (res['C_tau_by_ell'] or {}).items()}",
        "    _cum = 0.0",
        "    for e in sorted(_be):",
        "        _cum += _be[e]",
        "        _lab = 'tree' if e == 0 else f'tree+{e}loop'",
        "        print(f'  theory κ{int(cfg.k)}(τ=0)  {_lab:12s} = {_cum:+.4e}')",
        "    print('  (odd cumulants vanish at the symmetric saddle x*=0 → κ₃=0;'",
        "          ' set Config.fixed_point_index to sit in one well)')",
    ),
    md(
        "## 4. Theory vs simulation",
        "",
        "`nb.plot_cumulant` auto-dispatches on `k`: for **k=2** it draws the "
        "per-loop-order theory `C(τ)` overlay (set by `cfg.show_orders`) with "
        "the simulator points + error bars; for **k≥3** there is no `C(τ)` "
        "curve, so it draws the equal-time connected k-point cumulant κ_k as "
        "per-loop-order bars (`sim=None` — the 2-point estimator above is "
        "k=2-specific).",
    ),
    code(
        "fig = nb.plot_cumulant(res, cfg, model, sim=sim)",
        "plt.show()",
    ),
]

nb_obj = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'SageMath 10.8', 'language': 'sage',
                       'name': 'sagemath-10.8'},
        'language_info': {'name': 'sage'},
    },
    'nbformat': 4, 'nbformat_minor': 5,
}
with open(OUT, 'w') as f:
    json.dump(nb_obj, f, indent=1)
print('wrote', os.path.relpath(OUT, ROOT), '(', len(cells), 'cells )')

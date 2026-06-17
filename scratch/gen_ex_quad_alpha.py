"""Generate notebooks/examples/temporal_quadratic_hawkes_alpha.ipynb.

A concise "bonafide example" for the ``quadratic_hawkes_alpha`` theory: a
quadratic nonlinear Hawkes process with an ALPHA-FUNCTION synaptic kernel.
Mirrors the committed reference ``spatial_allen_cahn_phi4_1d.ipynb`` cell-for-
cell (title / §0 Setup / §1 model / §2 pipeline / §3 sim / Summary).

The ``md`` / ``code`` / ``_split`` helpers and the SETUP cell are copied
verbatim from ``scratch/gen_examples.py``.  Run with

    sage -python scratch/gen_ex_quad_alpha.py

then execute with ``sage -python scratch/exec_nb.py <path>``.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'notebooks', 'examples')
os.makedirs(OUT, exist_ok=True)


def md(*lines):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': list(_split(lines))}


def code(*lines):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': list(_split(lines))}


def _split(lines):
    text = '\n'.join(lines)
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


# ── shared cells (verbatim from scratch/gen_examples.py) ──────────────────────

SETUP = code(
    "%matplotlib inline",
    "import os, sys, time",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "# depth-robust repo root: walk up until the 'pipeline' package is found",
    "_root = os.path.abspath('')",
    "while _root != os.path.dirname(_root) and not os.path.isdir("
    "os.path.join(_root, 'pipeline')):",
    "    _root = os.path.dirname(_root)",
    "sys.path.insert(0, _root)",
    "sys.path.insert(0, os.path.join(_root, 'notebooks'))",
    "os.chdir(os.path.join(_root, 'notebooks'))  # cwd=notebooks/ for models/ data paths",
    "import daedalus as dd",
)

MODEL_MD = md(
    "## 1. The model",
    "",
    "`dd.describe_model` prints the structure straight from the theory file — "
    "domain, fields, parameters, kernels (the alpha-function synaptic filter "
    "`g`), and the quadratic transfer `phi`.",
)

THEORY_MD = md(
    "## 2. The pipeline → theoretical cumulants",
    "",
    "One `dd.run` drives the whole MSR-JD chain (enumerate diagrams → "
    "propagator → mean-field saddle → loop integrals → cumulant). The "
    "alpha-kernel enters as a genuine non-Markovian convolution `g * n` (double "
    "Fourier poles), and the quadratic transfer gives the gain "
    "$\\phi'(v^*) = 2a v^*$. The plot is the **theory only** — the simulation "
    "is added in §3.",
)

SIM_MD = md(
    "## 3. Independent simulation",
    "",
    "A direct Euler-step integration of the point process — written from "
    "scratch, with no reference to the diagrammatics. The alpha kernel is "
    "realised exactly as a **two-stage exponential cascade** "
    "($\\alpha = $ two identical low-pass filters convolved, unit DC gain), and "
    "the spike rate is the quadratic $\\lambda_i = a_i v_i^2$. Overlaying it on "
    "the pipeline curve is the validation.",
)


# ── the notebook ─────────────────────────────────────────────────────────────

TITLE = ('Quadratic Hawkes with an alpha-function synaptic kernel — a '
         'non-Markovian temporal convolution')

HEADLINE = (
    "**Showcases:** an **alpha-function synaptic kernel** (a non-Markovian "
    "temporal convolution that *rises then decays*, not a bare exponential) "
    "combined with a **quadratic Hawkes transfer** $\\phi(v)=a\\,v^2$. The "
    "pipeline carries the colored-in-time filter `g * n` through the MSR-JD "
    "machinery as a genuine convolution (double Fourier poles), and the "
    "quadratic nonlinearity gives a nonzero curvature $\\phi''(v^*)=2a$ — the "
    "cubic vertex that drives loop corrections. A size-2 excitatory "
    "population; the fields are the spike train $n$ and the synaptic voltage "
    "$v$. Here it is run at **tree level** ($\\langle nn\\rangle$).\n\n"
    "$$n_i = \\phi(v_i)=a\\,v_i^2,\\qquad "
    "\\tau_i\\dot v_i = -(v_i-E_i) + \\sum_j w_{ij}\\,(g_{ij}*n_j),\\qquad "
    "g_{ij}(t)=\\frac{t}{\\tau_{g,ij}^2}\\,e^{-t/\\tau_{g,ij}}\\,\\Theta(t).$$")

# §2 — config (theory) cell.
CONFIG = code(
    "cfg = dd.Config(",
    "    k=2, max_ell=0,                          # two-point ⟨nn⟩, tree level",
    "    external_fields=[('dn', 1), ('dn', 2)],  # the two neurons' spike trains",
    "    tau_max=20.0, tau_step=2.5,              # C(τ) on a lag grid",
    "    parallel=False,                          # serial (no fork in notebooks)",
    ")")

# §2 — run + theory-only plot.  Also prints the MF saddle (the alpha-kernel
# closure n* = a·v*²) so its physicality is visible.
RUN = code(
    "res = dd.run(model, cfg, mod)",
    "print(dd.summary(res))",
    "mf = res['mf_values']",
    "print('mean-field saddle:  n* =', np.round(mf['nstar'], 4),",
    "      '  v* =', np.round(mf['vstar'], 4))",
    "a = np.asarray(res['_resolved']['fundamental']['a'])",
    "print('closure n* − a·v*² =',",
    "      list(np.round(np.asarray(mf['nstar']) - a*np.asarray(mf['vstar'])**2, 12)))",
    "dd.plot_cumulant(res, cfg, model)   # theory only",
    "plt.show()")

# §3 — independent simulator → sim={'tau','C','C_err'} + overlay.
#
# Quadratic-transfer alpha-kernel point-process sim.  ``...quad_alpha_numba``
# is the v²-transfer sibling of the cubic ``...cubic_alpha_numba`` simulator —
# identical two-stage alpha-cascade plumbing, only the rate exponent differs
# (v² vs v³).  Under the Sage kernel every numba argument is wrapped in
# int()/float() (Sage's preparser makes Integer/RealLiteral, which @njit
# rejects).
SIM = code(
    "# Independent point-process simulation (direct Euler integration) — NOT the",
    "# pipeline.  Same physical parameters as the theory (read from the model).",
    "from models.hawkes_sim_multipop_numba import (",
    "    sim_hawkes_multipop_quad_alpha_numba, build_sim_arrays, flat_index_of)",
    "from models.cumulant_estimator import compute_kpoint_slice",
    "",
    "external_fields = res['_resolved']['external_fields']",
    "fundamental     = res['_resolved']['fundamental']",
    "arr = build_sim_arrays(model, fundamental, mf)   # flat per-neuron / per-pair arrays",
    "N, tau_v, a_gain   = arr['N'], arr['tau_v'], arr['a_gain']",
    "E_drive, W         = arr['E_drive'], arr['W']",
    "tau_g_arr, v_init  = arr['tau_g'], arr['v_init']",
    "pop_offsets        = arr['pop_offsets']",
    "",
    "# Fast simulation knobs (whole notebook runs in well under a minute).",
    "N_RUNS, T_sim, dt_sim, dt_bin = 5, 8.0e5, 0.01, 0.25",
    "tau_max        = float(cfg.tau_max)",
    "n_steps        = int(T_sim / dt_sim)",
    "bin_size_steps = max(int(round(dt_bin / dt_sim)), 1)",
    "dt_bin_eff     = bin_size_steps * dt_sim",
    "n_bins         = n_steps // bin_size_steps",
    "max_lag_bins   = int(tau_max / dt_bin_eff)",
    "tau_sim_grid   = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_bin_eff",
    "pop_indices    = [flat_index_of(model, pop_offsets, ef[0], ef[1])",
    "                  for ef in external_fields]",
    "field_types    = [ef[0] for ef in external_fields]",
    "",
    "# JIT warmup (Sage literals → int()/float() so @njit can type them).",
    "_ = sim_hawkes_multipop_quad_alpha_numba(",
    "    int(1000), float(dt_sim), tau_v, a_gain, E_drive, W, tau_g_arr,",
    "    v_init.copy(), int(bin_size_steps), int(100), int(0))",
    "",
    "C_runs, rate_runs = [], []",
    "t0 = time.perf_counter()",
    "for run in range(N_RUNS):",
    "    bc, vb, ts = sim_hawkes_multipop_quad_alpha_numba(",
    "        int(n_steps), float(dt_sim), tau_v, a_gain, E_drive, W, tau_g_arr,",
    "        v_init.copy(), int(bin_size_steps), int(n_bins), int(1234 + run))",
    "    rate_runs.append([float(ts[i]) / T_sim for i in range(N)])",
    "    _, C_run = compute_kpoint_slice(",
    "        bc, float(dt_bin_eff), [int(p) for p in pop_indices],",
    "        [0, None], int(max_lag_bins), field_types=field_types, voltage_bins=vb)",
    "    C_runs.append(C_run)",
    "C_runs = np.array(C_runs)",
    "C_sim  = C_runs.mean(axis=0)",
    "C_err  = C_runs.std(axis=0, ddof=1) / np.sqrt(N_RUNS)",
    "print('sim: %d runs × T=%.0g took %.1fs' % (N_RUNS, T_sim, time.perf_counter() - t0))",
    "print('sim rates =', np.round(np.array(rate_runs).mean(axis=0), 4),",
    "      '   theory n* =', np.round(mf['nstar'], 4))",
    "",
    "sim = {'tau': tau_sim_grid, 'C': C_sim, 'C_err': C_err}",
    "mid = len(tau_sim_grid) // 2",
    "C_th0 = float(np.interp(0.0, res['tau_grid'], np.real(res['C_tau'])))",
    "print('C(0):  sim = %.5f (± %.5f)   theory = %.5f'",
    "      % (C_sim[mid], C_err[mid], C_th0))",
    "dd.plot_cumulant(res, cfg, model, sim=sim)",
    "plt.show()")

SUMMARY = (
    "The tree-level two-point cumulant $C(\\tau)=\\langle n\\,n\\rangle$ "
    "carries the **alpha-function synaptic filter** — its rise-then-decay "
    "shape (double Fourier poles, not a single exponential) sets the lag "
    "structure of the correlator, and the cross-coupling $w_{ij}$ makes it "
    "asymmetric in $\\tau$. The independent point-process simulation — with the "
    "alpha kernel built as a two-stage exponential cascade and the **quadratic** "
    "rate $\\lambda=a v^2$ — reproduces both the mean-field rates ($n^*=a v^{*2}$) "
    "and the $C(\\tau)$ curve. Turning on `max_ell=1` would add the 1-loop "
    "correction from the cubic vertex $\\phi''(v^*)=2a$.")


def build_nb():
    cells = [
        md(f"# {TITLE}", "", HEADLINE),
        md("## 0. Setup"),
        SETUP,
        MODEL_MD,
        code("THEORY = 'quadratic_hawkes_alpha'",
             "model, mod = dd.load_theory(THEORY)",
             "dd.describe_model(model, mod)"),
        THEORY_MD, CONFIG, RUN,
        SIM_MD, SIM,
        md("## Summary", "", SUMMARY),
    ]
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'SageMath 10.8',
                           'language': 'sage', 'name': 'sagemath-10.8'},
            'language_info': {'name': 'sage'},
        },
        'nbformat': 4, 'nbformat_minor': 5,
    }


if __name__ == '__main__':
    nb = build_nb()
    path = os.path.join(OUT, 'temporal_quadratic_hawkes_alpha.ipynb')
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print('wrote', os.path.relpath(path, ROOT))
    print('done')

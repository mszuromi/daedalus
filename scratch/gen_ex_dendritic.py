"""Generate the bonafide example notebook
``notebooks/examples/temporal_dendritic_linear.ipynb``.

Mirrors the structure of ``notebooks/examples/spatial_allen_cahn_phi4_1d.ipynb``
(title → §0 Setup → §1 model → §2 pipeline/theory → §3 sim overlay → Summary),
but for the two-compartment ``single_pop_dendritic_linear`` neural theory.  The
distinctive features it showcases: the DAE mean-field solver, non-Markovian
exponential synaptic kernels, and the multi-field soma/dendrite Bernoulli CGF.

Run:   sage -python scratch/gen_ex_dendritic.py
Exec:  sage -python scratch/exec_nb.py notebooks/examples/temporal_dendritic_linear.ipynb
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


# ── shared setup cell (verbatim from scratch/gen_examples.py) ────────────────

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


# ── notebook ────────────────────────────────────────────────────────────────

TITLE = md(
    "# Soma + dendrite (linear compartmental) — a non-Markovian, "
    "multi-field DAE theory",
    "",
    "**Showcases:** the *temporal* side of the framework on a genuinely "
    "non-trivial neural model. Three features beyond the plain-Langevin "
    "examples appear at once: (i) the **DAE mean-field solver** — both "
    "governing relations are *algebraic* (no $\\partial_t$), so the saddle "
    "is a self-consistent fixed point found by multi-start Newton, not an "
    "ODE steady state; (ii) **non-Markovian exponential synaptic kernels** "
    "$g(t)=\\tfrac1\\tau e^{-t/\\tau}$ that convolve the inputs (memory in "
    "time); and (iii) a **multi-field soma/dendrite** structure whose "
    "dendritic spike is a *Bernoulli* event per somatic spike, handled by the "
    "framework's CGF machinery $\\log\\!\\big(1+(e^{\\tilde n_D}-1)\\,p_D\\big)$.",
    "",
    "$$\\lambda^S_i=\\Big[E^S_i+\\textstyle\\sum_j w^{SS}_{ij}\\,(g^S_i* n^S_j)"
    "+\\sum_j w^{SD}_{ij}\\,(g^S_i* n^D_j)\\Big]_+,\\qquad "
    "p^D_i=\\mathrm{clip}\\Big(E^D_i+\\textstyle\\sum_j w^{DS}_{ij}\\,"
    "(g^D_i* n^S_j)+\\sum_j w^{DD}_{ij}\\,(g^D_i* n^D_j),\\,0,1\\Big).$$",
)

MODEL_MD = md(
    "## 1. The model",
    "",
    "`dd.describe_model` prints the structure straight from the theory file — "
    "the two populations (size-2 excitatory), the somatic/dendritic spike "
    "fields, the four weight matrices, the two exponential kernels, and the "
    "self-consistent (algebraic) governing relations.",
)

MODEL_CODE = code(
    "THEORY = 'single_pop_dendritic_linear'",
    "model, mod = dd.load_theory(THEORY)",
    "dd.describe_model(model, mod)",
)

THEORY_MD = md(
    "## 2. The pipeline → theoretical cumulants",
    "",
    "One `dd.run` drives the whole MSR-JD chain. Because both relations are "
    "*algebraic*, the mean-field step runs the **DAE solver** (multi-start "
    "Newton on the self-consistency, not an ODE integration); it returns the "
    "unique low-rate physical saddle $n^S_*,\\,n^D_*$. The propagator is built "
    "in frequency space from the exponential kernels' Lorentzian factors "
    "$1/(1+i\\omega\\tau)$. We take the connected two-point cumulant "
    "$C^{(2)}(\\tau)$ between the two neurons' **somatic** spike trains "
    "(`('nS',1)` × `('nS',2)`) at tree level. The plot is the **theory "
    "only** — the simulation is added in §3.",
)

CONFIG_CODE = code(
    "cfg = dd.Config(",
    "    k=2, max_ell=0,                          # two-point ⟨nS nS⟩, tree level",
    "    external_fields=[('nS', 1), ('nS', 2)],  # somatic spikes, neuron 1 × neuron 2",
    "    tau_max=20.0, tau_step=0.5,              # C(τ) on τ ∈ [0, 20]",
    ")",
    "# parameters: the model's own parameter defaults are used (weights, τ's,",
    "# resting drives baked into the theory file).  Pin them here for clarity",
    "# and to guarantee the theory and the simulator share identical numbers.",
    "fundamental = {",
    "    'wSS': [[0.0, 0.0], [0.0, 0.0]],   'wSD': [[0.1, 0.0], [0.0, 0.1]],",
    "    'wDS': [[0.1, 0.1], [0.1, 0.1]],   'wDD': [[0.0, 0.0], [0.0, 0.0]],",
    "    'tauS': [1.0, 1.0], 'tauD': [1.0, 1.0],",
    "    'ES':   [0.5, 0.5], 'ED':   [0.5, 0.5],",
    "}",
    "cfg.parameters = fundamental",
)

RUN_CODE = code(
    "res = dd.run(model, cfg, mod)",
    "print(dd.summary(res))",
    "print('mean-field saddle: nS* = %s,  nD* = %s'",
    "      % (np.round(res['mf_values']['nSstar'], 5).tolist(),",
    "         np.round(res['mf_values']['nDstar'], 5).tolist()))",
    "dd.plot_cumulant(res, cfg, model)   # theory only",
    "plt.show()",
)

SIM_MD = md(
    "## 3. Independent simulation",
    "",
    "A direct event-driven Monte-Carlo of the spiking network — written from "
    "scratch (numba), with no reference to the diagrammatics. Each soma fires "
    "as an inhomogeneous Poisson process with the rectified-linear rate above; "
    "each dendrite fires per somatic spike with the clipped-linear Bernoulli "
    "probability $p_D$. The synaptic drive is the running exponential filter of "
    "past spikes. We estimate the *connected* $C^{(2)}(\\tau)$ slice between the "
    "two somatic trains (the estimator returns the connected cumulant directly — "
    "no mean subtraction needed) and overlay it on the pipeline curve.",
)

SIM_CODE = code(
    "from models.dendritic_linear_sim_numba import (",
    "    sim_dendritic_linear_numba, build_sim_arrays, flat_index_of,",
    "    stack_binned_counts)",
    "from models.cumulant_estimator import estimate_kpoint_slices",
    "",
    "# Reduced Monte-Carlo budget for a fast showcase (~1 min); the off-diagonal",
    "# cross-correlation is a small signal, so we run several independent seeds",
    "# to pin the statistical band.  This numba kernel is SERIAL (no fork) — safe.",
    "# (Native int()/float() casts: the Sage kernel preparses bare literals to",
    "#  Sage types, which numba's nopython compiler cannot type — so cast.)",
    "T_sim, dt_sim, dt_bin, N_RUNS = 1_500_000.0, 0.01, 0.5, 6",
    "tau_max = float(cfg.tau_max)",
    "",
    "# Translate model + saddle into the per-array sim inputs.",
    "arr = build_sim_arrays(model, fundamental, res['mf_values'])",
    "N   = arr['N']",
    "wSS, wSD, wDS, wDD = arr['wSS'], arr['wSD'], arr['wDS'], arr['wDD']",
    "tauS, tauD, ES, ED = arr['tauS'], arr['tauD'], arr['ES'], arr['ED']",
    "stack_offsets      = arr['stack_offsets']",
    "",
    "# Discretization + τ grid (the sim's own binned grid).",
    "n_steps    = int(T_sim / dt_sim)",
    "bin_steps  = max(int(round(dt_bin / dt_sim)), 1)",
    "dt_bin_eff = float(bin_steps * dt_sim)",
    "n_bins     = int(n_steps // bin_steps)",
    "max_lag_bins = int(tau_max / dt_bin_eff)",
    "tau_sim    = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_bin_eff",
    "",
    "# The k external legs → flat sim rows (leg 0 = the lag-0 anchor; legs 1..k−1",
    "# are swept one at a time by the estimator).  pop_indices / field_types have",
    "# length k, exactly as estimate_kpoint_slices expects.",
    "ext         = cfg.external_fields",
    "pop_indices = [flat_index_of(stack_offsets, e[0], e[1]) for e in ext]",
    "field_types = [e[0] for e in ext]",
    "CAP_PD_AT_ONE = True   # paper convention: dendritic prob clipped at 1",
    "",
    "# k−1 cumulant slices, matching the theory's res['C_tau_slices'] and the",
    "# k≥3 panels of dd.plot_cumulant.  The base point for the non-swept legs is",
    "# cfg.kpoint_base_lags (default all-0 → the slices pass through the origin).",
    "k    = int(res['_resolved']['k'])",
    "base = list(cfg.kpoint_base_lags) if cfg.kpoint_base_lags else [0.0] * (k - 1)",
    "base_bins = [int(round(b / dt_bin_eff)) for b in base]",
    "",
    "# JIT warm-up (first numba call compiles), then N_RUNS independent seeds.",
    "_ = sim_dendritic_linear_numba(int(1000), float(dt_sim), wSS, wSD, wDS, wDD,",
    "                               tauS, tauD, ES, ED, int(bin_steps), int(100),",
    "                               int(0), CAP_PD_AT_ONE)",
    "t0 = time.perf_counter()",
    "C_runs, rate_S = [], []",
    "for run in range(N_RUNS):",
    "    nS_b, nD_b, _lam, _pD, totS, totD = sim_dendritic_linear_numba(",
    "        int(n_steps), float(dt_sim), wSS, wSD, wDS, wDD, tauS, tauD, ES, ED,",
    "        int(bin_steps), int(n_bins), int(11 + run), CAP_PD_AT_ONE)",
    "    rate_S.append([float(totS[i]) / T_sim for i in range(N)])",
    "    # Dendritic legs are spike trains (field_types like 'dn'/'nS') → the",
    "    # stacked spike-count array is the relevant data (no voltage channel).",
    "    binned_stacked = stack_binned_counts(nS_b, nD_b)",
    "    tau_sim, Cj = estimate_kpoint_slices(",
    "        float(dt_bin_eff), [int(p) for p in pop_indices], field_types,",
    "        base_bins, int(max_lag_bins), binned_counts=binned_stacked)",
    "    C_runs.append(np.asarray(Cj))             # (k-1, n_tau)",
    "C_arr = np.array(C_runs).real",
    "C_sim = C_arr.mean(axis=0); C_err = C_arr.std(axis=0, ddof=1) / np.sqrt(N_RUNS)",
    "sim   = {'tau': tau_sim, 'C': C_sim, 'C_err': C_err}   # 2-D: one row per slice",
    "print('sim: %d runs × %g steps in %.1fs   somatic rate ≈ %s (MF: %s)'",
    "      % (N_RUNS, n_steps, time.perf_counter() - t0,",
    "         np.round(np.mean(rate_S, axis=0), 4).tolist(),",
    "         np.round(arr['nS_mf'], 4).tolist()))",
    "",
    "# At k=2 there is a single slice (row 0); report C(0) and the cross-corr peak.",
    "tg, Cth = res['tau_grid'], np.real(res['C_tau'])",
    "ipk = np.argmax(Cth)",
    "print('theory C(0) = %.5f    sim C(0) = %.5f ± %.5f'",
    "      % (Cth[np.argmin(np.abs(tg))],",
    "         C_sim[0][max_lag_bins], C_err[0][max_lag_bins]))",
    "print('cross-correlation peak  τ ≈ %.1f:  theory = %.5f   sim = %.5f ± %.5f'",
    "      % (tg[ipk], Cth[ipk],",
    "         np.interp(tg[ipk], tau_sim, C_sim[0]),",
    "         np.interp(tg[ipk], tau_sim, C_err[0])))",
    "dd.plot_cumulant(res, cfg, model, sim=sim)",
    "plt.show()",
)

SUMMARY = md(
    "## Summary",
    "",
    "A two-compartment (soma + dendrite) linear compartmental network, run "
    "through the full MSR-JD pipeline at tree level. Three pieces of machinery "
    "carried the calculation: the **DAE mean-field solver** (the saddle is a "
    "self-consistent fixed point of two *algebraic* relations), the "
    "**non-Markovian exponential synaptic kernels** (Lorentzian propagator "
    "factors $1/(1+i\\omega\\tau)$), and the **multi-field Bernoulli CGF** for "
    "the dendritic-spike channel.",
    "",
    "The somatic cross-correlation $C^{(2)}_{12}(\\tau)$ is a **delayed bump** — "
    "near zero at $\\tau=0$ (two different neurons are uncorrelated at equal "
    "time) and peaking at $\\tau\\!\\approx\\!1$–$2$, the lag set by the "
    "synaptic filter time constant as the two somata talk through their shared "
    "dendritic pathway. The event-driven simulation reproduces this delayed "
    "shape within statistics.",
    "",
    "**Caveat (physics).** The off-diagonal cross-correlation is a *small* "
    "signal (peak $\\sim10^{-3}$), so the honest comparison is the **curve / "
    "peak**, not the single point $C(0)$: $\\tau=0$ sits at a near-node of the "
    "cross-correlation, where the finite-bin Poisson self-variance and the "
    "Monte-Carlo floor dominate the genuinely-tiny equal-time value. A second, "
    "known mismatch source is the simulator's $p_D$ **cap at 1** (paper "
    "convention), which the MSR-JD Bernoulli action does not impose — "
    "negligible here at low drive, but it grows if `ED` or the weights push "
    "$p_D$ toward the boundary.",
)


def build_nb():
    cells = [
        TITLE,
        md("## 0. Setup"), SETUP,
        MODEL_MD, MODEL_CODE,
        THEORY_MD, CONFIG_CODE, RUN_CODE,
        SIM_MD, SIM_CODE,
        SUMMARY,
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
    path = os.path.join(OUT, 'temporal_dendritic_linear.ipynb')
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print('wrote', os.path.relpath(path, ROOT))

"""Generate notebooks/examples/temporal_dendritic_quad_sigmoid.ipynb.

A concise "bonafide example" for ``dendritic_quad_soma_sigmoid``: a two-neuron,
two-compartment model with a QUADRATIC somatic transfer and a SIGMOIDAL
dendritic transfer whose output is a Bernoulli PROBABILITY.  Mirrors
``temporal_quadratic_hawkes_alpha.ipynb`` (title / 0 Setup / 1 model / 2 pipeline
/ 3 sim / Summary).

Run:    sage -python scratch/gen_ex_dendritic_quad_sigmoid.py
Execute: sage -python scratch/exec_nb.py notebooks/examples/temporal_dendritic_quad_sigmoid.ipynb
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'notebooks', 'examples')
os.makedirs(OUT, exist_ok=True)
MAX_ELL = 0   # STAGE 1 tree (cache-build + fallback); stage 2 flips to 1


def md(*lines):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': list(_split(lines))}


def code(*lines):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': list(_split(lines))}


def _split(lines):
    text = '\n'.join(lines)
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


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

TITLE = ('Two-compartment neurons: a quadratic soma and a sigmoidal '
         '(probabilistic) dendrite')

HEADLINE = (
    "**Showcases:** a **two-compartment** point-process neuron pair where the "
    "two compartments carry *different* nonlinearities. Each neuron has a "
    "somatic voltage $v_S$ and a dendritic voltage $v_D$, each an exponential "
    "low-pass filter of its synaptic input. The **soma** fires as a Poisson "
    "process with a **quadratic** rate $\\phi_S(v_S)=a_S v_S^2$; the "
    "**dendrite** fires as a **Bernoulli** gate (one trial per somatic spike) "
    "whose success probability is a **sigmoid** $\\sigma(v_D)=1/(1+e^{-v_D})\\in"
    "(0,1)$ --- so the dendritic transfer's output is a genuine *probability*. "
    "The Bernoulli structure enters the action as $n_S\\log(1+(e^{\\tilde n_D}-1)"
    "\\sigma(v_D))$. Two neurons coupled by $w_{SD}$ (dendrite$\\to$soma) and "
    "$w_{DS}$ (soma$\\to$dendrite). Run to **"
    + ("1-loop" if MAX_ELL >= 1 else "tree level") + "**.\n\n"
    "$$\\tau_S\\dot v_{S,i}=-(v_{S,i}-E_{S,i})+\\textstyle\\sum_j w_{SD,ij}n_{D,j},"
    "\\quad n_{S,i}\\sim\\mathrm{Pois}(a_S v_{S,i}^2),$$\n"
    "$$\\tau_D\\dot v_{D,i}=-(v_{D,i}-E_{D,i})+\\textstyle\\sum_j w_{DS,ij}n_{S,j},"
    "\\quad n_{D,i}\\sim\\mathrm{Binom}(n_{S,i},\\,\\sigma(v_{D,i})).$$")

MODEL_MD = md(
    "## 1. The model",
    "",
    "`dd.describe_model` prints the structure from the theory file. Four fields "
    "per neuron --- somatic/dendritic spike trains $n_S,n_D$ and "
    "somatic/dendritic voltages $v_S,v_D$. The quadratic soma and the sigmoid "
    "dendrite are written *inline* in the action (rather than via a named "
    "`define_function`) so the framework's symbolic mean-field check can verify "
    "the saddle through the nested $\\log/\\exp/\\sigma$ Bernoulli term.",
)

THEORY_MD = md(
    "## 2. The pipeline → theoretical cumulants",
    "",
    "One `dd.run` drives the whole MSR-JD chain. We request the somatic "
    "cross-correlator between the two neurons, "
    "`external_fields=[('dnS', 1), ('dnS', 2)]`. The quadratic soma gives a "
    "curvature $\\phi_S''=2a_S$ and the sigmoid dendrite a curvature "
    "$\\sigma''(v_D^*)$ --- the vertices that drive the loop corrections. The "
    "plot is **theory only**; the simulation is added in §3.",
)

CONFIG = code(
    "cfg = dd.Config(",
    "    k=2, max_ell=%d,                          # somatic cross-correlation" % MAX_ELL,
    "    external_fields=[('dnS', 1), ('dnS', 2)],  # the two neurons' somatic spikes",
    "    tau_max=20.0, tau_step=2.5,",
    "    parallel=False,                          # serial (no fork in notebooks)",
    ")")

RUN = code(
    "res = dd.run(model, cfg, mod)",
    "print(dd.summary(res))",
    "mf = res['mf_values']",
    "print('mean-field saddle:  nS* =', np.round(mf['nSstar'], 4),",
    "      '  nD* =', np.round(mf['nDstar'], 4))",
    "print('                    vS* =', np.round(mf['vSstar'], 4),",
    "      '  vD* =', np.round(mf['vDstar'], 4),",
    "      '   p_D = sigma(vD*) =', np.round(1/(1+np.exp(-np.asarray(mf['vDstar'], float))), 4))",
    "dd.plot_cumulant(res, cfg, model)   # theory only",
    "plt.show()")

SIM_MD = md(
    "## 3. Independent simulation",
    "",
    "A direct Euler-step simulation of the two-compartment point process --- "
    "written from scratch, no reference to the diagrammatics. Each step: the "
    "soma fires $\\mathrm{Poisson}(a_S v_S^2\\,dt)$, the dendrite fires "
    "$\\mathrm{Binomial}(n_S,\\sigma(v_D))$ (the sigmoid keeps the probability in "
    "$(0,1)$ with no clipping), then the voltages are kicked by the spikes and "
    "decay. We estimate the somatic cross-correlator and overlay it.",
)

SIM = code(
    "# Independent two-compartment point-process simulation (Euler step) — NOT the pipeline.",
    "from models.dendritic_quad_sigmoid_sim_numba import (",
    "    sim_dendritic_quad_sigmoid_numba, build_sim_arrays, flat_index_of, stack_binned_counts)",
    "from models.cumulant_estimator import estimate_kpoint_slices",
    "",
    "external_fields = res['_resolved']['external_fields']",
    "fundamental     = res['_resolved']['parameters']",
    "arr = build_sim_arrays(model, fundamental, mf)",
    "N            = arr['N']",
    "aS, ES, ED   = arr['aS'], arr['ES'], arr['ED']",
    "tauS, tauD   = arr['tauS'], arr['tauD']",
    "wSD, wDS     = arr['wSD'], arr['wDS']",
    "stack_offsets = arr['stack_offsets']",
    "",
    "# Fast simulation knobs (whole notebook runs in well under a minute).",
    "N_RUNS, T_sim, dt_sim, dt_bin = 5, 1.0e6, 0.01, 0.25",
    "tau_max        = float(cfg.tau_max)",
    "n_steps        = int(T_sim / dt_sim)",
    "bin_size_steps = max(int(round(dt_bin / dt_sim)), 1)",
    "dt_bin_eff     = bin_size_steps * dt_sim",
    "n_bins         = n_steps // bin_size_steps",
    "max_lag_bins   = int(tau_max / dt_bin_eff)",
    "",
    "# leg ('dnS', i) -> stacked row of the somatic spike train of neuron i; spike type 'dn'.",
    "pop_indices = [flat_index_of(stack_offsets, ef[0][1:], ef[1]) for ef in external_fields]",
    "field_types = ['dn'] * len(external_fields)",
    "k         = int(res['_resolved']['k'])",
    "base      = list(cfg.kpoint_base_lags) if cfg.kpoint_base_lags else [0.0] * (k - 1)",
    "base_bins = [int(round(b / dt_bin_eff)) for b in base]",
    "",
    "# JIT warmup (Sage literals -> int()/float() so @njit can type them).",
    "_ = sim_dendritic_quad_sigmoid_numba(",
    "    int(1000), float(dt_sim), aS, ES, ED, tauS, tauD, wSD, wDS,",
    "    int(bin_size_steps), int(100), int(0))",
    "",
    "C_runs, rate_runs = [], []",
    "t0 = time.perf_counter()",
    "for r in range(N_RUNS):",
    "    bnS, bnD, _rS, _pD, tnS, tnD = sim_dendritic_quad_sigmoid_numba(",
    "        int(n_steps), float(dt_sim), aS, ES, ED, tauS, tauD, wSD, wDS,",
    "        int(bin_size_steps), int(n_bins), int(2024 + r))",
    "    rate_runs.append([float(tnS[i]) / T_sim for i in range(N)])",
    "    binned_counts = stack_binned_counts(bnS, bnD)",
    "    tau_sim, Cj = estimate_kpoint_slices(",
    "        float(dt_bin_eff), [int(p) for p in pop_indices], field_types,",
    "        base_bins, int(max_lag_bins), binned_counts=binned_counts)",
    "    C_runs.append(np.asarray(Cj))",
    "C_arr = np.array(C_runs)",
    "C_sim = C_arr.mean(axis=0)",
    "C_err = C_arr.std(axis=0, ddof=1) / np.sqrt(N_RUNS)",
    "print('sim: %d runs x T=%.0g took %.1fs' % (N_RUNS, T_sim, time.perf_counter() - t0))",
    "print('sim somatic rates =', np.round(np.array(rate_runs).mean(axis=0), 4),",
    "      '   theory nS* =', np.round(mf['nSstar'], 4))",
    "",
    "sim = {'tau': tau_sim, 'C': C_sim, 'C_err': C_err}",
    "mid   = len(tau_sim) // 2",
    "C_th0 = float(np.interp(0.0, res['tau_grid'], np.real(res['C_tau'])))",
    "print('C_nSnS(0):  sim = %.5f (+/- %.5f)   theory = %.5f'",
    "      % (C_sim[0][mid], C_err[0][mid], C_th0))",
    "dd.plot_cumulant(res, cfg, model, sim=sim)",
    "plt.show()")

SUMMARY = (
    "The somatic cross-correlator $C_{n_S n_S}(\\tau)$ between the two neurons is "
    "shaped by both compartments: the **quadratic** soma sets the spiking gain, "
    "while the **sigmoidal** dendrite contributes a bounded, saturating "
    "probability that feeds back through $w_{SD}$. Because the dendritic output "
    "is a true probability $\\sigma(v_D)\\in(0,1)$, no clipping is ever needed "
    "(unlike a linear dendritic gate). The "
    + ("1-loop self-energy --- fed by the soma's $\\phi_S''=2a_S$ and the "
       "dendrite's $\\sigma''(v_D^*)$ vertices --- moves the theory off the tree "
       "value into agreement with the direct two-compartment simulation."
       if MAX_ELL >= 1 else
       "tree-level correlator already captures the compartmental coupling; "
       "`max_ell=1` adds the loop correction from the $\\phi_S''$ and "
       "$\\sigma''$ vertices.")
    + " The voltage formulation (transfers acting on local voltages) is what "
    "keeps the nested sigmoid Bernoulli term symbolically tractable.")


def build_nb():
    cells = [
        md(f"# {TITLE}", "", HEADLINE),
        md("## 0. Setup"),
        SETUP,
        MODEL_MD,
        code("THEORY = 'dendritic_quad_soma_sigmoid'",
             "model, mod = dd.load_theory(THEORY)",
             "dd.describe_model(model, mod)",
             "print('\\nfields:', dd.field_names(model))"),
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
    path = os.path.join(OUT, 'temporal_dendritic_quad_sigmoid.ipynb')
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print('wrote', os.path.relpath(path, ROOT))
    print('done')

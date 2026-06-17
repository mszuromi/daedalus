"""Generate the bonafide example notebooks under notebooks/examples/.

Each example follows ONE structure (the user's spec):

    1. The model      — dd.load_theory + dd.describe_model  (model info first)
    2. The pipeline    — dd.run + dd.plot_cumulant           (theoretical cumulants)
    3. The simulation  — an INDEPENDENT simulator, appended at the end, overlaid
                         via dd.plot_cumulant(..., sim=...)   (not part of the pipeline)

Concise showcase: light narrative, one headline feature per notebook.  Run with
``sage -python scratch/gen_examples.py`` then execute with scratch/exec_nb.py.
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


# ── shared cells ─────────────────────────────────────────────────────────────

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
    "domain, fields, parameters, kernels, and the governing equation.",
)
def model_code(theory):
    return code(
        f"THEORY = {theory!r}",
        "model, mod = dd.load_theory(THEORY)",
        "dd.describe_model(model, mod)",
    )

THEORY_MD = md(
    "## 2. The pipeline → theoretical cumulants",
    "",
    "One `dd.run` drives the whole MSR-JD chain (enumerate diagrams → propagator "
    "→ mean-field saddle → loop integrals → cumulant). The plot is the **theory "
    "only** — the simulation is added in §3.",
)


def _spatial1d_sim(coupling, cparam, L, N, nsteps, burn):
    """A spatial-1d simulation cell (direct ETD1 SPDE integration) for the
    given per-vertex coupling kwarg (``lam`` φ⁴ / ``lam_kpz`` / ``g_lap``)."""
    return code(
        "# Independent SPDE simulation (direct ETD1 integration) — NOT the",
        "# pipeline.  Same physical parameters as the theory (read from the model).",
        "from models.spatial_field_1d_sim import simulate, equal_time_correlator",
        "fp = dd.parameters_from_model(model)",
        f"snaps, x_grid, meta = simulate(L={L}, N={N}, mu=fp['mu'], D=fp['D'], "
        f"T=fp['T'],",
        f"                               {coupling}=fp['{cparam}'],",
        f"                               n_steps={nsteps}, burn_in={burn}, "
        "record_every=20, seed=1)",
        "mean = float(np.mean(snaps))      # ⟨φ⟩: ≈0 if symmetric, the excess velocity for KPZ",
        "Cx = equal_time_correlator(snaps) - mean**2   # CONNECTED (the pipeline gives connected)",
        "half = len(x_grid) // 2 + 1",
        "sim = {'x': x_grid[:half], 'C': Cx[:half]}",
        "mid = res['C_tau_x'].shape[0] // 2",
        "print('theory C(0) = %.4f   sim C(0) = %.4f   (sim mean = %.3f)'",
        "      % (np.real(res['C_tau_x'])[mid][0], sim['C'][0], mean))",
        "dd.plot_cumulant(res, cfg, model, sim=sim)",
        "plt.show()",
    )


SIM_MD = md(
    "## 3. Independent simulation",
    "",
    "A direct numerical integration of the SPDE — written from scratch, with no "
    "reference to the diagrammatics. Overlaying it on the pipeline curve is the "
    "validation.",
)


# ── example specifications ───────────────────────────────────────────────────

EXAMPLES = []

EXAMPLES.append(dict(
    fname='spatial_allen_cahn_phi4_1d',
    theory='allen_cahn_1d_subcritical_infinite',
    title='Allen–Cahn (φ⁴) in d=1 — a polynomial spatial vertex',
    headline=(
        "**Showcases:** the spatial MSR-JD machinery on the simplest interacting "
        "field — a φ⁴ (polynomial) vertex. The pipeline builds heat-kernel "
        "propagators, reduces the loop integral via Symanzik polynomials over "
        "causal time-chambers, and inverse-Fourier-transforms back to real "
        "space. Here it is run to **1-loop**.\n\n"
        "$$\\partial_t\\phi = D\\,\\partial_x^2\\phi - \\mu\\phi - "
        "\\lambda\\phi^3 + \\eta,\\qquad \\langle\\eta\\eta\\rangle = "
        "2T\\,\\delta\\,\\delta.$$"),
    config=code(
        "cfg = dd.Config(",
        "    k=2, max_ell=1,                 # two-point ⟨φφ⟩, tree + 1-loop",
        "    external_fields=[('dphi', 1), ('dphi', 1)],",
        "    spatial_grid=(0.0, 6.0, 25),    # equal-time C(x) on x ∈ [0, 6]",
        "    tau_max=0.0,",
        ")"),
    sim=_spatial1d_sim('lam', 'lam', 40.0, 256, 120000, 20000),
    summary=(
        "The 1-loop φ⁴ self-energy lifts the equal-time variance C(0) above the "
        "free (tree) value; the direct SPDE simulation confirms it. This is the "
        "spatial pipeline's reference case — every other spatial example swaps "
        "only the vertex."),
))

EXAMPLES.append(dict(
    fname='spatial_kpz_1d',
    theory='kpz_1d',
    title='KPZ in d=1 — a per-leg gradient vertex',
    headline=(
        "**Showcases:** a **derivative** vertex. KPZ's nonlinearity "
        "$(\\lambda/2)(\\partial_x h)^2$ attaches a momentum factor to each leg "
        "of the vertex, which the pipeline handles with a per-leg gradient "
        "form-factor (not the plain polynomial path of Allen–Cahn).\n\n"
        "$$\\partial_t h = D\\,\\partial_x^2 h + "
        "\\tfrac{\\lambda}{2}(\\partial_x h)^2 + \\eta.$$"),
    config=code(
        "cfg = dd.Config(",
        "    k=2, max_ell=1,",
        "    external_fields=[('dh', 1), ('dh', 1)],",
        "    spatial_grid=(0.0, 10.0, 30),",
        "    tau_max=0.0,",
        ")"),
    sim=_spatial1d_sim('lam_kpz', 'lam', 20.0, 128, 150000, 30000),
    extra_md=md(
        "### The genuine nonlinear signature: excess velocity",
        "",
        "In **d=1**, KPZ shares the linear (Edwards–Wilkinson) *stationary* "
        "measure, so the equal-time `C(x)` above is **λ-independent** — the "
        "vertex leaves the static variance untouched (the pipeline and sim "
        "agree on the EW value). Where the per-leg gradient vertex *does* show "
        "up is the **excess velocity** "
        "$\\langle\\dot h\\rangle = (\\lambda/2\\mu)\\,\\langle(\\partial_x "
        "h)^2\\rangle$ — a drift absent in the linear theory. The sim's mean "
        "height-rate matches the tree-level lattice prediction:"),
    extra=code(
        "# Tree-level lattice prediction for the KPZ excess velocity vs the sim drift.",
        "dx, Nsim, Lsim = meta['dx'], meta['N'], meta['L']",
        "ks   = 2.0 * np.pi * np.fft.fftfreq(Nsim, d=dx)",
        "disp = fp['mu'] + (2.0 * fp['D'] / dx**2) * (1.0 - np.cos(ks * dx))",
        "ddh2 = (fp['T'] / Lsim) * np.sum((np.sin(ks * dx) / dx) ** 2 / disp)",
        "exc_theory = (fp['lam'] / (2.0 * fp['mu'])) * ddh2",
        "print('KPZ excess velocity  ⟨h⟩ = (λ/2μ)⟨(∂ₓh)²⟩')",
        "print('   sim    = %.4f' % mean)",
        "print('   theory = %.4f   (ratio %.3f)' % (exc_theory, mean / exc_theory))",
    ),
    summary=(
        "KPZ is the **per-leg gradient vertex**: the same diagram machinery as "
        "Allen–Cahn, but with a `∂ₓ → ik` form-factor on each leg. In d=1 the "
        "static `C(x)` is λ-independent (EW stationary measure) — pipeline and "
        "sim agree — and the nonlinearity reveals itself in the excess velocity, "
        "which matches the tree-level lattice prediction to ~1%."),
))

EXAMPLES.append(dict(
    fname='spatial_model_b_conserved_1d',
    theory='reaction_diffusion_conserved_1d',
    title='Model B (conserved ∇²φ²) in d=1 — a composite-derivative vertex',
    headline=(
        "**Showcases:** a **composite-derivative**, conservation-law vertex. "
        "Model B's flux form $g\\,\\partial_x^2(\\phi^2)$ carries an overall "
        "$\\partial_x^2 \\to -q^2$ on the whole vertex, so the long-wavelength "
        "variance is conservation-suppressed.\n\n"
        "$$\\partial_t\\phi = D\\,\\partial_x^2\\phi - \\mu\\phi + "
        "g\\,\\partial_x^2(\\phi^2) + \\eta.$$"),
    config=code(
        "cfg = dd.Config(",
        "    k=2, max_ell=1,",
        "    external_fields=[('dphi', 1), ('dphi', 1)],",
        "    spatial_grid=(0.0, 8.0, 30),",
        "    tau_max=0.0,",
        ")"),
    sim=_spatial1d_sim('g_lap', 'g', 20.0, 200, 120000, 20000),
    summary=(
        "The composite ∇² vertex multiplies the form-factor by q², suppressing "
        "the loop correction at small momentum — the signature of a conserved "
        "order parameter. The SPDE simulation reproduces the suppressed "
        "variance."),
))


EXAMPLES.append(dict(
    fname='spatial_reaction_diffusion_2d',
    theory='reaction_diffusion_2d',
    title='Reaction–diffusion in d=2 — a UV-finite loop above d=1',
    headline=(
        "**Showcases:** the pipeline in **d=2** with a *convergent* loop. The "
        "quadratic nonlinearity gives a cubic MSR vertex, so the 1-loop "
        "self-energy is a genuine momentum-dependent **bubble** $\\propto g^2$; "
        "the integral $\\int d^2\\ell\\,$ behaves as $\\int d^2\\ell/\\ell^4$ in "
        "the UV, which converges for $d<4$ (the φ⁴ upper critical dimension). So "
        "the d=2 correction is **finite — no cutoff**.\n\n"
        "$$\\partial_t\\phi = D\\,\\nabla^2\\phi - \\mu\\phi - g\\phi^2 + "
        "\\eta,\\qquad x\\in\\mathbb{R}^2.$$"),
    body=[
        md("## 2. The pipeline → theoretical cumulants (tree vs 1-loop)", "",
           "Two runs at the same separations: tree, and tree+1-loop. The "
           "difference `dC(r)` is the finite d=2 bubble correction."),
        code(
            "rs = np.linspace(0.4, 4.0, 14)        # radial separations r ≥ 0",
            "common = dict(k=2, external_fields=[('dphi', 1), ('dphi', 1)],",
            "              spatial_grid=rs, tau_max=1.0, tau_step=1.0)",
            "tree = dd.run(model, dd.Config(max_ell=0, **common), mod)",
            "loop = dd.run(model, dd.Config(max_ell=1, **common), mod)",
            "mid = tree['C_tau_x'].shape[0] // 2   # τ = 0 row",
            "C0 = np.real(tree['C_tau_x'])[mid]",
            "C1 = np.real(loop['C_tau_x'])[mid]",
            "dC = C1 - C0",
            "print('tree C(r0) = %.4f   1-loop C(r0) = %.4f   dC = %.4f'",
            "      % (C0[0], C1[0], dC[0]))",
            "print('1-loop correction all finite? ', bool(np.all(np.isfinite(dC))),",
            "      '  max|dC| = %.4g' % np.max(np.abs(dC)))",
            "plt.plot(rs, C0, '-o', ms=4, label='tree')",
            "plt.plot(rs, C1, '-s', ms=4, label='tree + 1-loop')",
            "plt.xlabel('r'); plt.ylabel('C(r)'); plt.legend()",
            "plt.title('d=2 reaction–diffusion: tree vs 1-loop'); plt.show()"),
        md("## 3. Independent simulation", "",
           "Two checks. (a) The free ($g=0$) **2-D structure factor** $S(q)$ "
           "against the tree propagator $T/(\\mu+Dq^2)$. (b) The **$g^2$-scaling** "
           "of the loop: doubling $g$ quadruples `dC`, confirming a genuine "
           "$O(g^2)$ bubble (tree is $g$-independent since $\\phi^*=0$)."),
        code(
            "from models.spatial_field_2d_sim import simulate_2d, "
            "radial_structure_factor_2d",
            "fp = dd.parameters_from_model(model)",
            "snaps, meta = simulate_2d(L=20.0, N=64, mu=fp['mu'], D=fp['D'], "
            "T=fp['T'], g=0.0,",
            "                          n_steps=50000, burn_in=10000, "
            "record_every=15, seed=3)",
            "kc, Sq = radial_structure_factor_2d(snaps, meta, n_bins=30)",
            "plt.plot(kc, Sq, 'o', ms=4, label='2-D simulation (free)')",
            "plt.plot(kc, fp['T'] / (fp['mu'] + fp['D'] * kc**2), '-', "
            "label=r'tree $T/(\\mu+Dq^2)$')",
            "plt.xlim(0, 3); plt.xlabel('q'); plt.ylabel('S(q)'); plt.legend()",
            "plt.title('d=2 structure factor: tree vs simulation'); plt.show()",
            "",
            "# g²-scaling of the loop correction (an O(g²) bubble → ratio 4).",
            "loop2 = dd.run(model, dd.Config(max_ell=1,",
            "               parameters={**fp, 'g': 2 * fp['g']}, **common), mod)",
            "dC2 = np.real(loop2['C_tau_x'])[mid] - C0",
            "print('dC(2g)/dC(g) at r0 = %.3f   (O(g²) bubble → 4.0)'",
            "      % (dC2[0] / dC[0]))"),
    ],
    summary=(
        "The d=2 reaction–diffusion bubble is the pipeline working **above one "
        "dimension with a finite loop**: the cubic vertex's momentum-dependent "
        "self-energy converges (d=2 < 4), the correction needs no cutoff, the "
        "free structure factor matches the 2-D simulation, and the loop scales "
        "as g² as a genuine bubble must."),
))


# ── assemble ─────────────────────────────────────────────────────────────────

def build_nb(spec):
    cells = [
        md(f"# {spec['title']}", "", spec['headline']),
        md("## 0. Setup"),
        SETUP,
        MODEL_MD, model_code(spec['theory']),
    ]
    if spec.get('body'):                       # fully-custom §2/§3 (bespoke examples)
        cells += spec['body']
    else:
        cells += [
            THEORY_MD, spec['config'],
            code("res = dd.run(model, cfg, mod)",
                 "print(dd.summary(res))",
                 "dd.plot_cumulant(res, cfg, model)   # theory only",
                 "plt.show()"),
            SIM_MD, spec['sim'],
        ]
        if spec.get('extra_md') and spec.get('extra'):
            cells += [spec['extra_md'], spec['extra']]
    cells.append(md("## Summary", "", spec['summary']))
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
    for spec in EXAMPLES:
        nb = build_nb(spec)
        path = os.path.join(OUT, f"{spec['fname']}.ipynb")
        with open(path, 'w') as f:
            json.dump(nb, f, indent=1)
        print('wrote', os.path.relpath(path, ROOT))
    print('done')

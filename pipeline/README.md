# `pipeline/` — user-facing API for the MSR-JD Feynman pipeline

This package wraps the existing `msrjd/` machinery into single-call
entry points so users can:

1. Compute a cumulant slice and save it to `.npz` / `.csv`,
2. Generate a multi-page PDF report showing each diagram and its
   numerical contribution,
3. (eventually) declare a theory at a higher level than hand-writing
   a Sage-aware Python dict.

## Status (prototype)

| piece | what's there | what's deferred |
|---|---|---|
| `compute_cumulants(...)` | Full pipeline end-to-end: FT expand → propagator → MF → diagrams → Phase J → τ-grid eval. NPZ save built in. | k≥3 currently returns the `total_C` callable but doesn't auto-evaluate on a grid (caller does that). CSV output is stretch. |
| `generate_report(...)` | Multi-page PDF: cover (params, MF, total slice) + one page per diagram (graph, vertex types, contribution plot). | LaTeX-rendered action / vertex coefficients (currently raw `str()` of SR). Per-diagram k≥3 plots. |
| `TheoryBuilder` (in `theory.py`) | Class scaffolding for declarative theory input via `.response_field(...).parameter(...).set_action(...)` chain. | Action templates (HawkesAction(phi='linear'), etc.) — user still has to supply the action lambda. YAML / JSON loader. |

## Quick start

```python
from models.hawkes_linear_expg_gtas import HAWKES_MODEL
from pipeline import compute_cumulants, generate_report

fundamental = {
    'E': [1.1, 1.05], 'w': [[0.35, 0.4], [0.3, 0.5]],
    'tau': 10.0, 'a': 1.0, 'tau_g': 2.5,
    'w_X': 0.3, 'lambda_X': 2.5, 'p_part': 0.6,
    'mu_shift_diff': 0.0, 'sigma_shift_diff_sq': 1.0,
}

result = compute_cumulants(
    model           = HAWKES_MODEL,
    k               = 2,
    max_ell         = 0,
    fundamental     = fundamental,
    external_fields = [('dn', 1), ('dn', 2)],
    tau_max         = 50.0,
    tau_step        = 0.5,
    output_npz      = 'linear_gtas_k2.npz',
)

generate_report(
    model           = HAWKES_MODEL,
    k               = 2,
    fundamental     = fundamental,
    external_fields = [('dn', 1), ('dn', 2)],
    output_pdf      = 'linear_gtas_k2_report.pdf',
    result          = result,    # reuse compute
)
```

End-to-end example: `pipeline/examples/run_linear_gtas.py`.

Run with:

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES sage -python \
    pipeline/examples/run_linear_gtas.py
```

## Result dict

`compute_cumulants(...)` returns:

```python
{
    'total_C':       callable(*tau_values) -> complex,
    'C_tau':         ndarray (complex, k=2 only),
    'tau_grid':      ndarray (float),
    'mf_values': {
        'nstar': [n*_1, n*_2, ...],
        'vstar': [v*_1, v*_2, ...],
        'mstar': [b_X, b_X, ...] or None,
    },
    'num_params':    {SR symbol: float},
    'diagrams':      [{'typed_diagram': td, 'classify': info,
                       'combined_prefactor': SR}, ...],
    'kernel_groups': [...],
    'phase_j_result': td_result,
    'propagator':    {'G_ft': matrix, 'pole_vals': [...], 'C_mats': [...], ...},
    'config':        {...},
}
```

NPZ save (when `output_npz=...` is set) writes:

| key | dtype | shape | description |
|---|---|---|---|
| `tau_grid` | float | (n_tau,) | τ values |
| `C_tau_real` | float | (n_tau,) | Re C(τ) |
| `C_tau_imag` | float | (n_tau,) | Im C(τ) |
| `nstar` | float | (npop,) | mean-field firing rates |
| `vstar` | float | (npop,) | mean-field voltages |
| `mstar` | float | (npop,) | external rate (GTaS only) |
| `k`, `max_ell` | int | (1,) | run config |
| `model_name` | str | (1,) | for cross-reference |

## How it maps to the notebook

| pipeline call | notebook cells |
|---|---|
| `FieldTheory(model).expand()` | cell 4 |
| `extract_vertex_types`, `extract_source_types` | cell 5 |
| `pipeline._propagator.build_propagator` | cell 8 (propagator construction + factor() + δ-coeff matrix) |
| `pipeline._mean_field.solve_mean_field` | cell 23 (MF residual, fsolve, num_params assembly) |
| `compute_poles_and_residues` | cell 23 deferred / cell 24 |
| `enumerate_prediagrams_all + enumerate_all_typed + filter_causal + deduplicate_typed_diagrams` | cells 11, 17, 18 |
| `classify_coefficient_factors` | cell 22 |
| `compute_correction_td` | cell 25 (Phase J) |
| τ-grid evaluation of `total_C` | cell 28 |

## Theory builder (roadmap)

The `TheoryBuilder` in `pipeline/theory.py` is a **stub** — it'll
accumulate field/parameter/kernel declarations and emit a model dict,
but you still have to provide the action / mf_bg_conditions /
phi_concrete as Sage lambdas via `.set_action(...)` etc.

The intended end state is something like:

```python
from pipeline.theory import TheoryBuilder, HawkesAction

t = (TheoryBuilder('My Hawkes', n_populations=2)
     .response_field('nt', latex=r'\tilde{n}')
     .response_field('vt', latex=r'\tilde{v}')
     .physical_field('dn', latex=r'\delta\dot{n}')
     .physical_field('dv', latex=r'\delta v')
     .parameter('a', default=1.0)
     .parameter('tau', default=10.0, domain='positive')
     .parameter('tau_g', default=2.5, domain='positive')
     .parameter('E', default=[1.1, 1.05], indexed='vector')
     .parameter('w', default=[[0.35, 0.4], [0.3, 0.5]], indexed='matrix')
     .kernel('g', frequency_image=lambda omega, p:
                  1 / (1 + 1j * omega * p['tau_g']))
     .use_action_template(
         HawkesAction(phi='linear', external_drive='E',
                      synaptic_kernel='g', recurrent_weight='w')
     ))
model = t.build()
```

The `HawkesAction` template would generate the action lambda from the
declared building blocks.  Adding GTaS noise: `t.correlated_noise('X', ...)`.

For the plain-vanilla case this is mostly mechanical translation of
the existing model files; the trickier piece is allowing user-defined
custom action terms while keeping the high-level interface.

## Stretch items

- CSV output (alongside NPZ) — trivial.
- LaTeX rendering of action / vertex coefficients in the PDF report
  via `matplotlib.text` mathtext or external `pdflatex`.
- Web UI (Flask/Streamlit) wrapping `TheoryBuilder` for interactive
  theory entry.
- Theory file `*.theory.yaml` schema with a small expression DSL.
- Unit tests for `compute_cumulants` against the existing notebooks'
  Phase J output.

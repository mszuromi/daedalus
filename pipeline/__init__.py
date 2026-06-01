"""
pipeline — high-level user-facing API for the MSR-JD Feynman pipeline.

This package wraps the lower-level msrjd machinery (FieldTheory expansion,
type assignment, propagator construction, Phase J integration) into
single-function calls suitable for scripts, batch runs, and notebook
sanity checks.

Core entry points
-----------------

  from pipeline import compute_cumulants, generate_report

  result = compute_cumulants(
      model            = HAWKES_MODEL,         # model dict
      k                = 2,                    # k-point cumulant
      max_ell          = 0,                    # tree-level only
      fundamental      = {...},                # numerical parameters
      external_fields  = [('dn', 1), ('dn', 2)],
      tau_max          = 50.0,                 # τ grid extent
      tau_step         = 0.5,
      output_npz       = 'result.npz',         # optional save
  )

  generate_report(
      model            = HAWKES_MODEL,
      k                = 2,
      max_ell          = 0,
      fundamental      = {...},
      external_fields  = [('dn', 1), ('dn', 2)],
      output_pdf       = 'report.pdf',
  )

The result dict includes:

  - ``total_C``      : callable f(*tau_values) → complex
  - ``C_tau``        : ndarray of values on the τ grid
  - ``tau_grid``     : ndarray of τ values
  - ``mf_values``    : {'nstar': [...], 'vstar': [...], 'mstar': [...]}
  - ``num_params``   : {SR symbol: float}
  - ``diagrams``     : list of TypedDiagram with prefactors
  - ``propagator``   : {'G_ft': matrix, 'pole_vals': [...], 'C_mats': [...]}
  - ``config``       : the call args echoed back

Status (prototype):
  ✓ compute_cumulants — full pipeline up to and including Phase J
  ✓ generate_report   — multi-page PDF with prediagrams + per-diagram values
  ✗ theory builder API (declarative theory input)  — design sketched in
    pipeline/theory.py; full implementation pending.
"""
from pipeline.compute     import compute_cumulants
from pipeline.report      import generate_report
from pipeline.save        import save_npz, save_csv, params_slug
from pipeline.access      import MeanField, Parameters, normalize_external_fields
from pipeline._precompute import precompute

__all__ = [
    'compute_cumulants', 'generate_report',
    'save_npz', 'save_csv', 'params_slug',
    'MeanField', 'Parameters', 'normalize_external_fields',
    'precompute',
]

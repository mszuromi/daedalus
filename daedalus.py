"""daedalus.py — the notebook front-end (``dd``) of Daedalus: stationary
statistics of Itô SDEs/SPDEs via automated Feynman-diagram evaluation of
perturbative MSR–JD field theories.

Centralises the **load → run → plot** flow so every demo notebook is thin and
uniform regardless of its group (temporal / spatial × single / multi-field).
The four group templates (``notebooks/templates/``) and every
``*_sim_compare`` notebook import this module so the thematics are common.

    import daedalus as dd
    model, mod = dd.load_model('kpz_1d')          # from models/*.model.py
    cfg = dd.Config(k=2, max_ell=1, chi_grid=(-6, 6, 49))   # χ = spatial sep.
    res = dd.run(model, cfg, mod)                  # k/ell/Dyson all here
    dd.plot_cumulant(res, cfg, model)              # adaptable, auto-dispatched

Design choices
--------------
* **Single source of truth for models.**  ``load_model`` imports a
  ``models/<name>.model.py`` file and returns ``(model, module)``; demo
  notebooks never re-build a model inline.
* **Arbitrary k and ℓ.**  ``Config.k`` / ``Config.max_ell`` are free; the
  spatial k≥3 path is wired through ``Config.spatial_points``.
* **Dyson dressing exposed.**  ``Config.dyson_order`` (+ ``reference_diffusion``)
  overrides the model's Dyson policy at run time — any order ≥ 0.
* **Adaptable plotting.**  ``plot_cumulant`` auto-dispatches on
  (spatial?, multi-field?, k) and honours the plot options on ``Config``
  (``show_orders``, ``logy``, ``components``, ``figsize``, …).
"""
from __future__ import annotations

import math
import os
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

#: Daedalus version (keep in sync with pyproject.toml and api/__init__.py;
#: CITATION.cff describes the latest RELEASE and is bumped, with its
#: date-released, when this version is released; tests/
#: test_save_version_stamp.py checks all of it).
__version__ = '0.3.3'


# ── Repo / model discovery ──────────────────────────────────────────────────

def repo_root() -> str:
    """Walk up from this file (or cwd, in a bare notebook) until the
    ``api/`` package is found.  Robust to running from any of the
    ``notebooks/`` subdirectories."""
    here = os.path.dirname(os.path.abspath(__file__)) \
        if '__file__' in globals() else os.path.abspath('')
    root = here
    while root != os.path.dirname(root):
        if os.path.isdir(os.path.join(root, 'api')):
            return root
        root = os.path.dirname(root)
    return here


REPO_ROOT = repo_root()
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
import sys as _sys
if REPO_ROOT not in _sys.path:
    _sys.path.insert(0, REPO_ROOT)


# ── Lazy re-exports of the user-facing pipeline API ──────────────────────────
# Every major pipeline symbol a user touches is reachable as ``dd.<name>``,
# alongside the notebook helpers defined below (``Config``, ``run``, ``plot_*``,
# ``load_model``, …).  Resolved lazily on first access (PEP 562) so
# ``import daedalus`` stays light and only pulls in Sage when something is
# actually used.  The canonical ``from api... import <name>`` forms keep
# working — these are aliases; that explicit form is the one to use inside a
# saved ``models/*.model.py`` file (which cannot import this notebook helper).
_PIPELINE_EXPORTS = {
    # authoring — build a model in code (pipeline.model)
    'ModelBuilder':             'api.model',
    'TemporalModelBuilder':     'api.model',
    'SpatialModelBuilder':      'api.model',
    # noise template (api.model_templates) for builder.add_gtas_noise(...)
    'GTaSNoise':                 'api.model_templates',
    # graphical builder (pipeline.ui)
    'ModelUI':                  'api.ui',
    # deprecated aliases (pre-July-2026 name: "theory") — kept so existing
    # scripts and notebooks keep working; new code should use the Model* names
    'TheoryBuilder':             'api.model',
    'TemporalTheoryBuilder':     'api.model',
    'SpatialTheoryBuilder':      'api.model',
    'TheoryUI':                  'api.ui',
    # compute / report / precompute / persistence / access (pipeline top level)
    'compute_cumulants':         'api',
    'generate_report':           'api',
    'precompute':                'api',
    'save_npz':                  'api',
    'save_csv':                  'api',
    'load_npz':                  'api',
    'load_provenance':           'api',
    # model analyzer (Feature D) and the backend registry
    'analyze':                   'api.analyze',
    'Question':                  'api.analyze',
    'Policy':                    'api.analyze',
    'AnalyzerError':             'api.analyze',
    # a run that left diagrams out (result['status'] == 'incomplete')
    'PhaseJIncompleteWarning':   'api.compute',
    'PhaseJIncompleteError':     'api.compute',
    # Phase J over its memory budget (PHASE_J_MEM_BUDGET_GB)
    'PhaseJResourceError':       'api.compute',
    # a discard=True run asked for a point it did not evaluate
    'DiscardedDiagramsError':    'api.compute',
    'backends':                  'api.routing',
    'params_slug':               'api',
    'MeanField':                 'api',
    # the common base of every exception the package raises on purpose
    'DaedalusError':             'api',
    'Parameters':                'api',
    'normalize_external_fields': 'api',
    # δ components of a cumulant (Feature B): dd.cumulant(result)
    'cumulant':                  'api.cumulant',
    'Cumulant':                  'api.cumulant',
    'DeltaComponent':            'api.cumulant',
    'ModeTerm':                  'api.cumulant',
    # matrix-exponential triples of the smooth part (C0 output):
    # dd.cumulant(result).me / .modes / .integrate_against / .spectrum
    'CumulantME':                'api.me_output',
    'RationalKernel':            'api.me_output',
    'MEUnavailable':             'api.me_output',
    'MEFallbackWarning':         'api.me_output',
    'MEConditioningWarning':     'api.me_output',
    'slice_me':                  'api.me_output',
    # exact master-equation oracle for lattice jump processes
    'master_equation':           'api.master_equation',
    'MasterEquation':            'api.master_equation',
    'TruncationWarning':         'api.master_equation',
    # reaction / transition builder for jump processes
    'JumpProcess':               'api.jump_process',
    'JumpProcessError':          'api.jump_process',
    # exact S(omega), S(0), int_0^T C, tau_c of a k = 2 result
    'spectra':                   'api.spectra',
    'SpectraWarning':            'api.spectra',
    # the k = 2 covariance matrix over many fields, and dSigma/dtheta
    'covariance_matrix':         'api.covariance',
    'CovarianceDerivativeError': 'api.covariance',
    'CovarianceWarning':         'api.covariance',
    # opt-in resummations: the self-consistent Gaussian (Hartree) and the
    # screened (optimized-perturbation) loop series, Config(resum=...)
    'hartree':                   'api.hartree',
    'HartreeResult':             'api.hartree',
    'HartreeError':              'api.hartree',
    'HartreeUnsupported':        'api.hartree',
    'HartreeBistabilityWarning': 'api.hartree',
    'HartreeAdvisoryWarning':    'api.hartree',
    'screened':                  'api.resum',
    'ResumWarning':              'api.resum',
    # symbolic tree-level outputs (C1 Fourier domain, C2 time domain)
    'tree_covariance':           'api.symbolic_out',
    'tree_covariance_from_result': 'api.symbolic_out',
    'TreeCovariance':            'api.symbolic_out',
    'tree_modes':                'api.symbolic_out',
    'ModeSum':                   'api.symbolic_out',
    'PairingKernel':             'api.symbolic_out',
    'Kernel':                    'api.delay',          # the kernel-shape builder (the pairing kernel is PairingKernel)
    'KernelIntegral':            'api.symbolic_out',
    'integrate_against':         'api.symbolic_out',
    'inverse_transform':         'api.symbolic_out',
    'pair_with_kernel':          'api.symbolic_out',
    'AssumptionViolation':       'api.symbolic_out',
    # symbolic loop level (C3): equal-time cumulants, kernel pairings and
    # matrix-exponential triples as rational functions of the parameters
    'symbolic_cumulant':         'api.symbolic_loop',
    'symbolic_triple':           'api.symbolic_loop',
    'SymbolicCumulant':          'api.symbolic_loop',
    'SymbolicTriple':            'api.symbolic_loop',
    'SymbolicLoopRefused':       'api.symbolic_loop',
    'SaddleWarning':             'api.symbolic_loop',
}


def __getattr__(name):                   # PEP 562 — lazy module-level attribute
    """Resolve ``dd.<name>`` for the re-exported pipeline symbols on first use."""
    target = _PIPELINE_EXPORTS.get(name)
    if target is not None:
        import importlib
        return getattr(importlib.import_module(target), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PIPELINE_EXPORTS))


def list_models() -> list[str]:
    """Every ``models/<name>.model.py`` available to load."""
    return sorted(f[:-len('.model.py')]
                  for f in os.listdir(MODELS_DIR)
                  if f.endswith('.model.py'))


def load_model(name: str):
    """Import ``models/<name>.model.py`` and return ``(model, module)``.

    ``model = module.build()`` is the model dict; ``module`` exposes
    ``DEFAULT_FUNDAMENTAL`` and ``METADATA`` (the run defaults)."""
    path = os.path.join(MODELS_DIR, f'{name}.model.py')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'No model file at {path}. Available: {list_models()}')
    spec = importlib.util.spec_from_file_location(f'models.{name}', path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod.build(), mod
    except Exception as e:
        raise RuntimeError(
            f'Failed to load model {name!r} from {path}: {e}') from e


# ── Model introspection ──────────────────────────────────────────────────────

# — deprecated aliases (pre-July-2026 names) — new code: load_model / list_models
load_theory = load_model
list_theories = list_models


def is_spatial(model: dict) -> bool:
    sp = model.get('spatial')
    return bool(sp and sp.get('dim'))


def spatial_dim(model: dict) -> int:
    sp = model.get('spatial') or {}
    return int(sp.get('dim') or 0)


def is_multifield(model: dict) -> bool:
    """True iff the model has more than one independent field channel —
    either >1 physical field or a population of size >1.  Drives the
    plotting (multi-field → per-component grid)."""
    if len(model.get('physical_fields') or []) > 1:
        return True
    return any(int(p.get('size') or 1) > 1
               for p in (model.get('populations') or []))


def field_names(model: dict) -> list[str]:
    # Natural (authored) name, e.g. 'x' — not the internal fluctuation
    # name 'dx' — so summary() matches describe_model() and the result is
    # safe to read back into external_fields.
    return [f.get('natural_name') or f['name']
            for f in (model.get('physical_fields') or [])]


def _field_pop_size(model: dict, natural_name: str) -> int:
    """How many population components the physical field ``natural_name``
    carries (1 for a scalar / single-population field).

    Used to spread an auto-built ``external_fields`` over distinct
    (field, pop) legs for a MULTI-population model — otherwise every leg
    collapses onto population 1 and the framework computes the wrong
    correlator (e.g. ⟨n₁ n₁⟩ instead of the intended cross-population
    ⟨n₁ n₂⟩)."""
    pf = next((f for f in (model.get('physical_fields') or [])
               if (f.get('natural_name') or f.get('name')) == natural_name),
              None)
    if pf is None:
        return 1
    # Named-population field → its declared population's size.
    pop = pf.get('population')
    if pop:
        for p in (model.get('populations') or []):
            if p.get('name') == pop:
                return max(1, int(p.get('size') or 1))
    # Anonymous indexed field (legacy ``n_populations`` style) → the size
    # of the global ``pop`` index set the saddle solver built.
    if pf.get('indexed'):
        idx = (model.get('index_sets') or {}).get('pop')
        if idx:
            return max(1, len(idx))
    return 1


def _ext_is_valid(model: dict, ext, k: int) -> bool:
    """True iff ``ext`` is a usable k-leg external_fields list for ``model``:
    present, of length ``k``, every leg names a real physical field, and a
    leg's 1-based population index is one of that field's populations.

    A stale METADATA recommendation (wrong length, naming a field that
    no longer exists, or a population the field does not have) fails this
    check so :func:`run` auto-builds a fresh one instead of feeding garbage
    to ``compute_cumulants``."""
    if not ext or len(ext) != k:
        return False
    valid = set(field_names(model))

    def _ok(e):
        if not isinstance(e, (tuple, list)):
            return e in valid
        if not e or e[0] not in valid:
            return False
        if len(e) < 2:
            return True
        pop = e[1]
        if isinstance(pop, (bool, np.bool_)) or not hasattr(pop, '__index__'):
            return False
        return 1 <= int(pop) <= _field_pop_size(model, e[0])
    return all(_ok(e) for e in ext)


def _auto_external_fields(model: dict, k: int) -> list:
    """Build a default k-leg ``external_fields`` for ``model``.

    The legs sit on the FIRST physical field, spread over its populations:
    ``[(f0, 1), (f0, 2), …]`` cycling modulo the population count.  For a
    MULTI-population model this gives distinct (field, pop) legs (e.g. k=2
    → ``[('n', 1), ('n', 2)]``) instead of collapsing every leg onto
    population 1 — which would silently compute the WRONG correlator
    (⟨n₁ n₁⟩ rather than the intended cross-population ⟨n₁ n₂⟩).  A
    single-population field keeps the old behaviour (npop=1 ⇒ every leg is
    pop 1, i.e. ``[(f0, 1)] * k``)."""
    fld = field_names(model)
    f0 = fld[0] if fld else 'phi'
    npop = _field_pop_size(model, f0)
    return [(f0, (i % npop) + 1) for i in range(k)]


def param_names(model: dict) -> list[str]:
    """The parameter names you can override via ``Config(parameters={...})`` —
    the declared parameters, excluding the mean-field saddle symbols.  Use this
    to get the exact spellings (a mistyped name now raises in ``dd.run``)."""
    from api.naming import free_parameter_names
    return free_parameter_names(model)


def _fmt_default(v, maxlen: int = 52) -> str:
    s = repr(v)
    return s if len(s) <= maxlen else s[:maxlen - 1] + '…'


# Provenance / dev-status boilerplate to strip from a model docstring so
# describe_model shows the model's physics, not its changelog.
_DOC_DROP = ('text-driven model file', 'Generated by', 'Loaded by',
             '.model.py', '====', '----', 'Schema', 'exposes:', 'docs/',
             '``build()``', '``DEFAULT_FUNDAMENTAL``', '``METADATA``',
             'returning a ``model``', 'suggested numerical', 'free-form info',
             'ready for', 'pattern).')
_DOC_STOP = ('Phase 1 status', 'Phase 2 status', 'Phase 3 status',
             'See ``docs', 'Mirrors the inline', 'status:', 'VERBATIM')


def _doc_physics(doc: str) -> str:
    """Trim a model docstring to its physics prose — drop UI-provenance
    boilerplate and the dev-status / cross-reference tail."""
    keep = []
    for ln in doc.splitlines():
        if any(s in ln for s in _DOC_STOP):
            break
        if any(s in ln for s in _DOC_DROP):
            continue
        keep.append(ln)
    return '\n'.join(keep).strip()


def describe_model(model: dict, module=None, show_doc: bool = True) -> str:
    """Pretty-print the structure of a loaded model — the *model information
    from the model file*: domain, fields, populations, parameters, kernels,
    transfer functions, and the governing equation(s).  Prints and returns the
    text.  Pass the model ``module`` (the 2nd value of :func:`load_model`) to
    also surface its docstring + ``METADATA`` run recommendations.

    This is the canonical "what is this model?" cell for the example
    notebooks — it reads only the model dict, so it stays model-independent.
    An unbuilt ``api.sde.SDE`` / builder is built first."""
    from api.compute import as_model_dict
    model = as_model_dict(model)
    out = []
    bar = '─' * 72
    out.append(bar)
    out.append(f"  {model.get('name', '(unnamed model)')}")
    out.append(bar)

    def _mode(v, default):
        if isinstance(v, dict):
            return v.get('mode', default)
        return v or default

    if is_spatial(model):
        sp = model.get('spatial') or {}
        bc = _mode(model.get('boundary') or sp.get('boundary'), 'infinite')
        ic = _mode(model.get('initial'), 'stationary')
        out.append(f"Domain         : SPDE · d={spatial_dim(model)} · "
                   f"boundary={bc} · initial={ic}")
    else:
        out.append("Domain         : SDE")

    def _fld(f):
        nm = f.get('natural_name') or f.get('name')
        pop = f.get('population')
        tag = f"[{pop}]" if pop and pop != 'pop' else ''
        sd = f.get('spatial_dim')
        sdt = f" (x∈ℝ^{sd})" if sd else ''
        desc = f.get('description') or ''
        return f"{nm}{tag}{sdt}" + (f" — {desc}" if desc else '')

    pf = model.get('physical_fields') or []
    out.append("Fields         : " + ('; '.join(_fld(f) for f in pf) or '(none)'))
    rf = []
    for r in (model.get('response_fields') or []):
        rf.append(r.get('name') if isinstance(r, dict) else str(r))
    if rf:
        out.append("Response fields: " + ', '.join(rf))

    pops = [p for p in (model.get('populations') or [])
            if p.get('name') != 'pop']            # skip auto scalar population
    if pops:
        out.append("Populations    : " + ', '.join(
            f"{p.get('name')} (size {p.get('size')})"
            + (f" — {p['description']}" if p.get('description') else '')
            for p in pops))

    params = model.get('parameters') or []
    from api.naming import is_saddle as _is_saddle   # the mean_field flag
    numeric = [p for p in params if not _is_saddle(p)]
    saddle = [p.get('name') for p in params if _is_saddle(p)]
    if saddle:
        out.append("Mean-field saddle: " + ', '.join(saddle))

    ker = model.get('kernels') or []
    for kr in ker:
        ix = kr.get('indexed_by')
        ixt = f"[{','.join(ix)}]" if ix else ''
        out.append(f"Kernel         : {kr.get('name')}{ixt} — "
                   "non-Markovian temporal convolution")
    fns = model.get('functions') or []
    for fn in fns:
        na = fn.get('n_args')
        if na is None:                # template-built fns carry 'args', not 'n_args'
            na = len(fn.get('args') or []) or '?'
        out.append(f"Function       : {fn.get('name')}(·) "
                   f"— {na}-arg transfer")

    eqs = model.get('equations') or []
    for eq in eqs:
        rhs = (eq.get('rhs_text') or '').strip()
        # Skip a trivial/empty rhs: either the dynamics are linear, or the
        # nonlinearity lives in the action text (e.g. KPZ's gradient vertex)
        # rather than a simple rhs — the notebook's markdown states it in full.
        if eq.get('lhs_text') and rhs and rhs not in ('0', '0.0'):
            out.append(f"Governing eqn  : {eq['lhs_text']} = {rhs}")

    act = (model.get('action_text') or '').strip()
    if act:
        act_lines = act.splitlines()
        out.append(f"Action  S      : {act_lines[0]}")
        for ln in act_lines[1:]:
            out.append(f"                 {ln}")

    if numeric:
        out.append("Parameters (defaults):")
        for p in numeric:
            ix = p.get('indexed_by')
            ixt = f" [{','.join(ix)}]" if ix else ''
            dom = f"  ({p['domain']})" if p.get('domain') else ''
            out.append(f"    {p.get('name')}{ixt} = "
                       f"{_fmt_default(p.get('default'))}{dom}")

    meta = (getattr(module, 'METADATA', {}) or {}) if module else {}
    rec = []
    if meta.get('k_default') is not None:
        rec.append(f"k={meta['k_default']}")
    if meta.get('ell_default') is not None:
        rec.append(f"max_ell={meta['ell_default']}")
    if rec:
        out.append("Suggested run  : " + ', '.join(rec))

    text = '\n'.join(out)
    doc = _doc_physics((getattr(module, '__doc__', '') or '')) if module else ''
    if show_doc and doc:
        text += '\n\n' + doc
    print(text)
    return text


# ── Run configuration ────────────────────────────────────────────────────────

@dataclass
class Config:
    """Everything a demo notebook chooses, in one object.

    Leave a field ``None`` to inherit the model file's ``METADATA`` /
    ``DEFAULT_FUNDAMENTAL`` default.
    """
    # ── what to compute ──
    k: Optional[int] = None                 # cumulant order; inferred from
                                            # external_fields if left None
    max_ell: Optional[int] = None           # loop order (0=tree, 1, 2, …)
    external_fields: Optional[list] = None  # e.g. ['x', 'x'] or [('v', 1), ('v', 2)]
    parameters: Optional[dict] = None        # numeric parameter overrides, by name
    fundamental: Optional[dict] = None       # deprecated alias for ``parameters``
    # Output quantity.  The diagrammatics produce CONNECTED cumulants
    # κ(x₁…x_k); 'moment'/'central_moment' assemble the full k-point moment
    # ⟨φ(x₁)…φ(x_k)⟩ (resp. of the centred field) via the set-partition
    # formula M = Σ_π ∏_B κ(B) — see _assemble_moment.  Costs k−1 extra
    # backend runs (one per order 2..k).  Temporal k≥3: the moment is the
    # curve ⟨φ(0) φ(τ) φ(0) … φ(0)⟩ over the τ grid (leg 1 swept, the other
    # legs at 0; ``kpoint_base_lags`` is not used), and its cumulant blocks
    # of 3 or more legs are evaluated at the times of slice 1 with the
    # default base lags (``_kpoint_slice_times``), so its τ = 0 point is the
    # left limit, like that of ``C_tau``.
    output: str = 'cumulant'                # 'cumulant'|'moment'|'central_moment'

    # ── temporal grid (the τ axis; also the τ axis of spatial C(χ,τ)) ──
    # Give EITHER tau_max+tau_step (uniform symmetric grid −tau_max…tau_max) OR
    # an explicit tau_grid (array | (lo,hi,n) tuple, like chi_grid) — the
    # latter is used verbatim and wins if both are set.
    tau_max: Optional[float] = None
    tau_step: Optional[float] = None
    tau_grid: Any = None                    # array | (lo, hi, n) | None

    # ── spatial ──
    # chi_grid is the grid of spatial DIFFERENCES χ = x_j − x_k (the paper's
    # real-space variable, eq. B47); spatial_grid is the backward-compat alias.
    chi_grid: Any = None                    # array | (lo, hi, n) | None
    spatial_grid: Any = None                # deprecated alias for chi_grid
    spatial_points: Any = None              # k≥3: (n_pts, k-1, 2) of (x_j, τ_j)

    # ── temporal k≥3 cumulant slicing ──
    # The connected k-point cumulant depends on k−1 time differences
    # τ_j = t_j − t_0.  By default run() returns the k−1 axis-parallel slices
    # through the origin (sweep leg j, others at 0).  A slice point where the
    # swept leg meets another leg (τ = 0 or a base lag) is the LEFT limit of
    # that slice, evaluated exactly: every leg at its nominal time, the tie
    # resolved by the engine's tie order with the swept leg listed last
    # (``_kpoint_slice_point``; times and leg orders in
    # ``result['_kpoint_slice_times']`` / ``['_kpoint_slice_orders']``).
    # DAEDALUS_TAU0=nudge restores the pre-Oct-2026 placement 1e-6 below the
    # leg met (``_kpoint_slice_times``).  The full grid keeps exact ties on
    # its diagonals (the engine's tie order), so its origin can differ from a
    # slice's τ = 0 point.
    kpoint_base_lags: Optional[list] = None  # length k−1: the fixed τ for the
                                             # non-swept legs (the base point the
                                             # slices pass through); default 0
    kpoint_full_grid: bool = False           # compute the full (k−1)-dim tensor
                                             # C(τ_1..τ_{k−1}) instead of slices
                                             # (cost ~n^{k−1}; k=3 → heatmap)

    # ── switch-on transients (temporal; non-stationary stage 1) ──
    # ``t_min`` (absolute time): the fluctuations about the mean-field saddle
    # start at zero at t_min, and the cumulant is a function of the ABSOLUTE
    # external times ``times``: shape (n, k) (one row per point, legs in the
    # order of external_fields), or 1-D / an (lo, hi, n) TUPLE (every leg at
    # t: the equal-time cumulant).  Values: result['C_times'] (by loop order:
    # result['C_times_by_ell']).  None (default): the stationary state.
    # ``initial_covariance`` (k=2, tree level only): the covariance C0 of the
    # fluctuations at t_min, adding R C0 R^T (compute_cumulants).
    t_min: Optional[float] = None
    times: Any = None
    initial_covariance: Any = None

    # ── delay kernels (api.delay) ──
    # options of the Pade route for models with delays, as
    # ``compute_cumulants(delay=...)``: e.g. {'order': 8} (fixed order) or
    # {'tol': 1e-6, 'n_max': 16, 'wall_max': 900, 'tree': 'exact'}.  None:
    # the route's defaults (automatic order).
    delay: Optional[dict] = None

    # ── exact spectral outputs (k = 2; api.spectra) ──
    # ``compute_cumulants(spectra=...)``: None (off), True (S(0), the
    # integrals to infinity, tau_c) or {'omega': (lo, hi, n) | array,
    # 'T': number | array, 'method': 'auto' | 'exact' | 'quad'}.  The values
    # land in result['spectra']; they do not use the lag grid
    # (tau_grid=np.array([0.0]) is enough).  S(omega) = int C e^{+i omega tau}.
    spectra: Any = None

    # ── the k = 2 covariance matrix over many fields (api.covariance) ──
    # ``covariance_matrix(...)`` instead of one compute_cumulants call: None
    # (off), True (every physical field leg) or {'fields': [...], 'pairs':
    # 'upper' | 'all', 'derivatives': [parameter, ...] or {'parameters': [...],
    # 'step': 1e-3}, 'keep_entries': bool}.  run() then returns the
    # covariance_matrix dict (Sigma[n, i, j] = <dx_i(0) dx_j(tau_n)>, per
    # loop order in Sigma_by_ell, derivatives in 'derivatives').
    covariance: Any = None

    # ── opt-in resummation (api.hartree, api.resum) ──
    # None (off: the loop series, unchanged), 'hartree' (the self-consistent
    # Gaussian on its own; max_ell unset or 0, k = 1 or 2) or 'screened' / a
    # dict {'method': 'screened', 'mass': 'pms' | 'hartree' | kappa,
    # 'observable': 'trace_cov' | 'C0' | [fields], 'step': 5e-3,
    # 'kappa_range': (0.25, 6.0), 'n_scan': 17}: the loop orders re-expanded
    # around a variational trial propagator (k = 2, max_ell = the screened
    # order N >= 1) -> result['C_tau'] (order N), ['C_tau_by_order'],
    # ['C_tau_spread'], ['C_tau_loops_by_ell'], ['resummation'].
    resum: Any = None

    # ── numerical knobs shared with compute_cumulants (None = its default) ──
    taylor_order: Optional[int] = None      # vertex Taylor order (None: max(k + 2 max_ell, 2))
    use_cache: Optional[bool] = None        # on-disk caches (None: on)
    accuracy: Optional[float] = None        # target relative accuracy of the G6 estimate
    backend: Optional[str] = None           # an explicit backend name (api.routing)
    discard: Any = None                     # False | True | 'auto': release each diagram after evaluating it

    # ── Dyson dressing (coupled unequal-D) ──
    dyson_order: Optional[int] = None       # None=leave model; int≥0=override
    reference_diffusion: Optional[float] = None

    # ── mean-field DAE root selection (multi-root models) ──
    # Multi-root saddles (e.g. the double-well regime mu<0, with two
    # stable wells) must choose which root to expand around.  Leave
    # ``None`` to inherit ``compute_cumulants``' own defaults
    # (fixed_point_index=0, 64 multi-starts, no seed box).
    fixed_point_index: Optional[int] = None   # which stable root (0, 1, …)
    mf_dae_n_starts: Optional[int] = None     # multi-start Newton count
    mf_dae_seed_box: Optional[dict] = None    # {field: (lo, hi)} start range

    # ── execution ──
    parallel: bool = False
    # Worker count for the parallel backend (forwarded to compute_cumulants).
    # None → the backend's own default (spatial threads: min(8, cores);
    # temporal batch: up to 8 threads, or 4 spawned workers for a batch of
    # seconds).  The spatial backend and the temporal batch's thread fan-out
    # (diagrams with large state blocks) are safe in a Jupyter kernel; fork
    # (type assignment; the temporal batch only on request) is Linux-only: macOS never forks
    # (engine.fork_safety).  The numbers do not depend on it.
    # DAEDALUS_PARALLEL=0 turns every pool off.
    n_workers: Optional[int] = None
    verbose: bool = False

    # ── model analyzer (api.analyze) ──
    # ``policy``: an ``api.analyze.Policy`` or dict (mode 'strict' /
    # 'permissive', consents, ``downgrade`` codes); ``preflight``: None = the
    # gates run (unless DAEDALUS_PREFLIGHT=0), False = off.
    policy: Any = None
    preflight: Optional[bool] = None

    # ── plotting options (adaptable) ──
    show_orders: str = 'cumulative'         # 'cumulative' | 'incremental' | 'total'
    logy: bool = False
    components: Any = None                  # which (i,j)/slice to draw; None=auto
    tau_slice_chi: Optional[float] = None   # spatial: the fixed χ₀ for the C(χ₀,τ)
                                            # temporal-slice panel; None → χ closest to 0
    figsize: Optional[tuple] = None
    title: Optional[str] = None
    save: Optional[str] = None              # path to savefig, or None

    def __post_init__(self):
        # ``parameters`` is the canonical name; ``fundamental`` is kept as a
        # backward-compatible alias.  Mirror whichever the caller set onto the
        # other (``parameters`` wins if both are given) so either reads fine.
        if self.parameters is not None:
            self.fundamental = self.parameters
        elif self.fundamental is not None:
            self.parameters = self.fundamental
        # ``chi_grid`` (the paper's spatial-difference grid χ=x_j−x_k) is the
        # canonical name; ``spatial_grid`` is the backward-compat alias.  Mirror
        # whichever was set onto the other (``chi_grid`` wins if both are given).
        if self.chi_grid is not None:
            self.spatial_grid = self.chi_grid
        elif self.spatial_grid is not None:
            self.chi_grid = self.spatial_grid
        # Fail early on a mistyped output kind (otherwise it silently
        # degrades to the connected cumulant — a wrong-kind result).
        if self.output not in ('cumulant', 'moment', 'central_moment'):
            raise ValueError(
                "Config.output must be 'cumulant', 'moment', or "
                f"'central_moment'; got {self.output!r}.")
        # Validate the grids eagerly so the list-vs-tuple range slip
        # (tau_grid=[-5,5,51] → 3 literal points, not a 51-pt range) is caught
        # at Config construction, not deep inside run().
        self.resolved_tau_grid()
        self.resolved_grid()
        self._check_numbers()
        self._check_switch_on()

    def _check_numbers(self):
        """The integer / positive options, checked at construction (and
        again by :func:`run`, as fields may be set afterwards) instead of
        deep inside the run."""
        from api.compute import _entry_int
        from api._grids import is_real_number
        for _n, _lo in (('k', 1), ('max_ell', 0), ('n_workers', 1),
                        ('fixed_point_index', 0), ('mf_dae_n_starts', 1),
                        ('taylor_order', 1)):
            _v = getattr(self, _n)
            if _v is None:
                continue
            if _entry_int(f'Config.{_n}', _v) < _lo:
                raise ValueError(f"Config.{_n} must be >= {_lo}; got {_v!r}.")
        for _n in ('tau_max', 'tau_step'):
            _v = getattr(self, _n)
            if _v is not None and not np.isfinite(float(_v)):
                raise ValueError(f"Config.{_n} must be finite; got {_v!r}.")
        if self.use_cache is not None and not isinstance(
                self.use_cache, (bool, np.bool_)):
            raise ValueError(
                f"Config.use_cache must be True, False or None; got "
                f"{self.use_cache!r}.")
        if self.accuracy is not None and not (
                is_real_number(self.accuracy)
                and np.isfinite(float(self.accuracy))
                and float(self.accuracy) > 0):
            raise ValueError(
                f"Config.accuracy must be a finite number > 0; got "
                f"{self.accuracy!r}.")
        if self.discard is not None and not (
                isinstance(self.discard, (bool, np.bool_))
                or self.discard == 'auto'):
            raise ValueError(
                f"Config.discard must be False, True or 'auto'; got "
                f"{self.discard!r}.")
        if self.spectra is not None:
            from api.spectra import resolve_spectra_options
            if resolve_spectra_options(self.spectra, 'Config.spectra') \
                    is not None and self.output != 'cumulant':
                raise ValueError(
                    f"Config.spectra with output={self.output!r}: the "
                    f"spectral outputs are those of the connected cumulant "
                    f"(output='cumulant').")
        if self.covariance is not None:
            from api.covariance import resolve_covariance_options
            if resolve_covariance_options(
                    self.covariance, 'Config.covariance') is not None:
                for _bad, _why in (
                        (self.output != 'cumulant',
                         f'output={self.output!r} (the covariance is the '
                         f'connected cumulant)'),
                        (self.k not in (None, 2), f'k={self.k!r} (it is k = 2)'),
                        (self.spectra is not None, 'spectra'),
                        (self.t_min is not None, 't_min'),
                        (self.spatial_points is not None, 'spatial_points')):
                    if _bad:
                        raise ValueError(
                            f'Config.covariance with {_why} is not '
                            f'supported.')
        if self.resum is not None:
            from api.resum import resolve_resum_options, check_config
            from api.covariance import resolve_covariance_options
            check_config(resolve_resum_options(self.resum, 'Config.resum'),
                         k=self.k, max_ell=self.max_ell, output=self.output,
                         t_min=self.t_min, spatial_points=self.spatial_points,
                         covariance=resolve_covariance_options(
                             self.covariance, 'Config.covariance'))
        if self.tau_step is not None and not float(self.tau_step) > 0:
            raise ValueError(
                f"Config.tau_step must be > 0; got {self.tau_step!r}.")
        if self.tau_max is not None and not float(self.tau_max) >= 0:
            raise ValueError(
                f"Config.tau_max must be >= 0; got {self.tau_max!r}.")

    def _check_switch_on(self):
        if self.t_min is None and (self.times is not None
                                   or self.initial_covariance is not None):
            raise ValueError(
                'Config: times / initial_covariance need t_min (a switch-on '
                'transient); without t_min the run is stationary.')
        self.resolved_times()

    def resolved_times(self):
        """Materialise ``times`` (an ``(lo, hi, n)`` tuple → ``linspace`` of
        equal-time points; a list/array → literal, 1-D or ``(n, k)``), or
        None."""
        if self.times is None:
            return None
        if isinstance(self.times, tuple):
            return self._resolve_grid(self.times, 'times')
        arr = np.asarray(self.times, dtype=float)
        self._require_finite(arr, 'times')
        return arr

    def resolved_grid(self):
        """Materialise ``chi_grid`` — the spatial-difference grid χ = x_j − x_k
        (accepts an ``(lo, hi, n)`` tuple).  ``spatial_grid`` is the alias."""
        return self._resolve_grid(self.chi_grid, 'chi_grid')

    def resolved_tau_grid(self):
        """Materialise ``tau_grid`` (accepts an ``(lo, hi, n)`` tuple), or None."""
        return self._resolve_grid(self.tau_grid, 'tau_grid')

    @staticmethod
    def _resolve_grid(g, name='grid'):
        """``(lo, hi, n)`` TUPLE → ``linspace``; list/array → literal points;
        None → None.  The rule is shared with ``compute_cumulants``
        (:func:`api._grids.resolve_grid`); a 3-element *list* ending in an
        integer count (``[-5, 5, 51]``) is the "I meant a range" slip and
        raises, pointing at the tuple form."""
        from api._grids import resolve_grid
        return resolve_grid(g, name)

    @staticmethod
    def _require_finite(arr, name):
        from api._grids import require_finite
        require_finite(arr, name)


def _meta(mod, key, default=None):
    return (getattr(mod, 'METADATA', {}) or {}).get(key, default)


def fundamental_from_model(model: dict) -> dict:
    """Build a ``fundamental`` dict from the parameters' own ``default``
    values baked into the model — every non-saddle parameter with a
    declared default.  Lets a loaded model run out of the box even when
    its ``DEFAULT_FUNDAMENTAL`` is empty.  Saddle params (``mean_field=True``,
    solved by the mean-field step) and defaultless params are skipped."""
    from api.naming import is_saddle
    out = {}
    for p in model.get('parameters') or []:
        if is_saddle(p) or p.get('default') is None:
            continue
        out[p['name']] = p['default']
    return out


# Canonical alias (matches the ``Config.parameters`` rename); the original
# name is kept for the existing notebooks that import it.
parameters_from_model = fundamental_from_model


def fetch_cache(cells='core', dest=None, base_url=None, **kw):
    """Download the prediagram cells that are too big for git (see
    ``engine.enumeration.prediagram_cache.fetch_cache``).  Explicit only:
    nothing is fetched unless you call this; the base URL comes from
    ``base_url=`` or the environment variable ``DAEDALUS_CACHE_URL``."""
    from engine.enumeration import prediagram_cache as _pc
    return _pc.fetch_cache(cells, dest=dest, base_url=base_url, **kw)


def build_cache(cells, dest=None, dry_run=False, **kw):
    """Estimate, then build, prediagram cells that are neither shipped nor
    already under *dest* (see ``engine.enumeration.prediagram_cache.build_cache``).
    The report (status, class count, estimated time and memory) is printed
    first; ``dry_run=True`` stops there and writes nothing."""
    from engine.enumeration import prediagram_cache as _pc
    return _pc.build_cache(cells, dest=dest, dry_run=dry_run, **kw)


def config_options(spatial=None):
    """Print every ``dd.Config`` argument — grouped, with its live default and a
    one-line description — so the full set of knobs is discoverable from any
    notebook.

    ``spatial=True`` / ``False`` tailors the grid/slicing section to the spatial
    / temporal case; ``None`` (default) prints both.  Pass a model dict and it
    auto-detects: ``dd.config_options(dd.is_spatial(model))``.

    The grid/slicing knobs (what the cumulant is evaluated *over*):

      temporal  C(τ)         tau_max, tau_step
      temporal  k≥3 cumulant kpoint_base_lags (fix the non-swept legs),
                             kpoint_full_grid (full (k−1)-D tensor vs slices)
      spatial   C(χ,τ)       chi_grid (χ=x_j−x_k axis), tau_max/tau_step (τ axis)
                             • full (χ,τ) grid : ranged chi_grid + tau_max>0
                             • equal-time C(χ,0): tau_max=0.0
                             • fixed-χ  C(χ₀,τ) : chi_grid=[χ0], tau_max>0
      spatial   k≥3          spatial_points = (n_pts, k−1, 2) explicit (x_j, τ_j) events,
                             OR chi_grid (+ tau_grid) → full 2(k−1)-D grid of all legs
    """
    import dataclasses
    d = {f.name: f.default for f in dataclasses.fields(Config)}

    def row(name, desc):
        return '    %-17s = %-11s  %s' % (name, repr(d.get(name)), desc)

    common_grid = [
        ('what to compute', [
            ('k',               'cumulant order: 1=mean ⟨φ⟩, 2=covariance, ≥3=higher cumulant'),
            ('max_ell',         'loop order: 0=tree, 1=+1-loop, 2=+2-loop, …'),
            ('external_fields', "the k legs, e.g. ['x', 'x'] (one-unit fields) or [('v',1),('v',2)]; "
                                "sets k if k is None; at k=1 the loop shift <x> - x* (output='moment': the mean)"),
            ('parameters',      "numeric overrides {name: value} (dd.param_names(model) lists the valid names); None → model defaults  [alias: fundamental]"),
            ('output',          "'cumulant' | 'moment' | 'central_moment'"),
        ]),
    ]
    temporal_grid = ('temporal grid / slicing (τ axis)', [
        ('tau_grid',         'τ (lag) grid: (lo,hi,n) TUPLE→linspace, or an array of points — the primary τ knob'),
        ('tau_max',          'under-the-hood: τ extent (symmetric −tau_max…tau_max) if no tau_grid'),
        ('tau_step',         'under-the-hood: τ spacing (paired with tau_max)'),
        ('kpoint_base_lags', 'k≥3: [k−1 floats] fix the non-swept legs (slices cross here; where the swept leg meets a leg, the point is the left limit)'),
        ('kpoint_full_grid', 'k≥3: True → full (k−1)-D tensor C(τ₁…) instead of axis slices'),
        ('t_min',            'switch-on transient: fluctuations start at 0 at this absolute time (None → stationary)'),
        ('times',            'with t_min: absolute times, (n, k) array, or 1-D / (lo,hi,n) TUPLE (all legs at t) → result["C_times"]'),
        ('initial_covariance', 'with t_min, k=2, tree: covariance C0 of the fluctuations at t_min (adds R C0 Rᵀ)'),
    ])
    spatial_grid_ = ('spatial grid / slicing (χ and τ axes)', [
        ('chi_grid',      'grid χ=x_j−x_k: (lo,hi,n) TUPLE→linspace, or array; [χ0] fixes χ  [alias: spatial_grid]'),
        ('tau_grid',      'τ grid: (lo,hi,n) TUPLE→linspace, or array; [0.0] → equal-time C(χ,0)'),
        ('tau_max',       'under-the-hood: τ extent if no tau_grid (0.0 → equal-time)'),
        ('tau_step',      'under-the-hood: τ spacing (paired with tau_max)'),
        ('spatial_points','k≥3: explicit (n_pts, k−1, 2) (x_j, τ_j) events — OR omit and pass chi_grid(+tau_grid) to sweep the full 2(k−1)-D grid → result["C_kpoint_grid"]'),
    ])
    rest = [
        ('mean-field root (multi-root models)', [
            ('fixed_point_index', 'which stable saddle root to expand around (0,1,…)'),
            ('mf_dae_n_starts',   'multi-start Newton count for the saddle solve'),
            ('mf_dae_seed_box',   '{field: (lo,hi)} start range for the saddle search'),
        ]),
        ('numerics (the compute_cumulants options)', [
            ('taylor_order', 'vertex Taylor order; None → max(k + 2 max_ell, 2)'),
            ('use_cache',    'on-disk caches (diagram, propagator, expand); None → on'),
            ('accuracy',     'target relative accuracy of the G6 error estimate; None → 1e-10'),
            ('backend',      'an explicit backend name (api.routing); None → the router picks'),
            ('discard',      "False | True | 'auto': release each diagram after evaluating it at the run's points (memory; k≥3: the slice points)"),
        ]),
        ('exact spectra (k=2; api.spectra)', [
            ('spectra', "None | True | {'omega': (lo,hi,n) or array, 'T': number or array, 'method': 'auto'|'exact'|'quad'} → result['spectra']: S(ω)=∫C e^{+iωτ}dτ, S(0), ∫_0^T C, τ_c per loop order (no lag grid needed)"),
        ]),
        ('covariance matrix (k=2; api.covariance)', [
            ('covariance', "None | True | {'fields': [...], 'pairs': 'upper'|'all', 'derivatives': [param, ...] or {'parameters': [...], 'step': 1e-3}, 'keep_entries': bool} → run() returns Sigma[n,i,j]=<dx_i(0) dx_j(τ_n)> over every pair (one shared MF/propagator/pole pass) and dSigma/dθ"),
        ]),
        ('resummation (api.hartree, api.resum)', [
            ('resum', "None | 'hartree' (self-consistent Gaussian; max_ell 0) | 'screened' or {'method': 'screened', 'mass': 'pms'|'hartree'|kappa, 'observable': 'trace_cov'|'C0'|[fields], 'step', 'kappa_range', 'n_scan'} (k=2: loop orders re-expanded around a variational propagator, order N = max_ell) → C_tau, C_tau_by_order, C_tau_spread, resummation"),
        ]),
        ('delay kernels (api.delay)', [
            ('delay', "options of the Padé route: {'order': 8} or {'tol': 1e-6, 'n_max': 16, 'wall_max': 900, 'tree': 'exact'}; None → automatic order"),
        ]),
        ('Dyson dressing (coupled unequal-D)', [
            ('dyson_order',         'None=leave model; int≥0 overrides the Dyson order'),
            ('reference_diffusion', 'reference D for the Dyson expansion'),
        ]),
        ('execution', [
            ('parallel',  'enable the parallel backend (threads everywhere, fork on Linux only; same numbers; DAEDALUS_PARALLEL=0 turns every pool off)'),
            ('n_workers', 'worker count (spatial and temporal threads; fork workers on Linux)'),
            ('verbose',   'print backend progress'),
        ]),
        ('model analyzer (api.analyze)', [
            ('policy',    "Policy / dict: mode 'strict'|'permissive', consents, downgrade=[codes]; None → permissive"),
            ('preflight', 'None/True → analyzer gates before the enumeration; False → off'),
        ]),
        ('plotting', [
            ('show_orders', "'cumulative' | 'incremental' | 'total'"),
            ('logy',        'log-scale the y axis'),
            ('components',  'which (i,j)/slice to draw (multi-field); None=auto'),
            ('tau_slice_chi','spatial: fixed χ₀ for the C(χ₀,τ) panel; None → χ≈0'),
            ('figsize',     'matplotlib figure size, e.g. (7.5, 4.6)'),
            ('title',       'override the plot title'),
            ('save',        'path to savefig, or None'),
        ]),
    ]

    groups = list(common_grid)
    if spatial is None:
        groups += [temporal_grid, spatial_grid_]
    elif spatial:
        groups += [spatial_grid_]
    else:
        groups += [temporal_grid]
    groups += rest

    print('dd.Config arguments  (defaults shown; leave None to inherit the '
          "model's METADATA):\n")
    for title, items in groups:
        print('  ' + title)
        for name, desc in items:
            print(row(name, desc))
        print()


# ── Cumulants → moments (full multivariate set-partition assembly) ──────────

def _set_partitions(items):
    """Yield every set partition of ``items`` (a list) as a list of blocks."""
    items = list(items)
    if not items:
        yield []
        return
    if len(items) == 1:
        yield [items]
        return
    first, rest = items[0], items[1:]
    for sub in _set_partitions(rest):
        for i in range(len(sub)):
            yield sub[:i] + [[first] + sub[i]] + sub[i + 1:]
        yield [[first]] + sub


def _result_meta(result, key, default=None):
    """Read a run-config field (k, max_ell, external_fields) from a result —
    from dd.run's ``_resolved``, falling back to compute_cumulants' ``config``
    so the plotters also work on a raw ``dd.compute_cumulants`` result."""
    rv = result.get('_resolved')
    if rv and key in rv:
        return rv[key]
    return (result.get('config') or {}).get(key, default)


def _leg_parts(leg):
    """``(field name, 1-based population)`` of an external leg (a
    ``(name, pop)`` tuple or a bare name, population 1)."""
    if isinstance(leg, (tuple, list)):
        return str(leg[0]), (int(leg[1]) if len(leg) > 1 else 1)
    return str(leg), 1


def _external_mean(model: dict, res: dict, leg=None) -> float:
    """Mean-field saddle φ* of one external leg's field and population — the
    TREE-level value of the 1-point function ⟨φ⟩ on that leg.

    The cumulant pipeline expands around the saddle, so the *fluctuation*
    field's tree-level 1-point function is 0; the physical mean is the saddle
    value plus the ℓ≥1 loop tadpole shifts (:func:`_leg_mean_by_ell`).  The
    saddle is stored under the field's saddle parameter (``api.naming``;
    ``nSstar``, ``xstar`` …) in the temporal ``res['mf_values']`` or the
    spatial ``res['mf']['mf_values']`` (a list of per-component dicts), one
    value per population.  ``leg`` is a ``(field, pop)`` external leg; None
    means the result's first external leg.  Returns 0.0 for a symmetric/zero
    saddle, or if none is found."""
    # Collect every candidate saddle dict {<saddle>: [vals]} — temporal
    # exposes one at the top level; spatial nests them under res['mf'].
    dicts = []
    mv = res.get('mf_values')
    if isinstance(mv, dict):
        dicts.append(mv)
    spat = res.get('mf')
    if isinstance(spat, dict):
        smv = spat.get('mf_values') or spat.get('saddle_values')
        if isinstance(smv, dict):
            dicts.append(smv)
        elif isinstance(smv, (list, tuple)):
            dicts.extend(d for d in smv if isinstance(d, dict))
    if not dicts:
        return 0.0
    # External leg's field → base physical field (strip the response/fluct 'd').
    if leg is None:
        ext = _result_meta(res, 'external_fields')
        if ext:
            leg = ext[0]
        else:
            names = field_names(model)
            leg = names[0] if names else ''
    fld, pop = _leg_parts(leg)
    from api.naming import saddle_for_field
    sad = saddle_for_field(model, fld)       # natural or internal leg name
    keys = [k for k in (sad, fld) if k]
    for d in dicts:
        for key in keys:
            if key in d:
                try:
                    vals = np.real(np.asarray(d[key], dtype=complex)).ravel()
                except Exception:
                    continue
                if vals.size == 0:
                    continue
                if vals.size == 1:
                    val = float(vals[0])
                elif 1 <= pop <= vals.size:
                    val = float(vals[pop - 1])
                else:
                    raise ValueError(
                        f'external leg {leg!r}: population {pop} is outside '
                        f'the {vals.size} saddle values of {key!r}')
                return 0.0 if abs(val) < 1e-14 else val   # kill subnormals
    return 0.0


def _leg_mean_by_ell(model, res, kw, legs, L, compute_cumulants):
    """``{leg: [μ⁽⁰⁾, …, μ⁽ᴸ⁾]}``: the mean of every distinct external leg
    per loop order.  ``μ⁽⁰⁾`` is the mean-field saddle of the leg's field and
    population (:func:`_external_mean`); ``μ⁽ℓ⁾`` (ℓ ≥ 1) is the ℓ-loop
    1-point cumulant of the fluctuation (the tadpoles), from one ``k = 1``
    run per leg (the order-k run itself when ``k = 1``)."""
    out = {}
    for leg in legs:
        if leg in out:
            continue
        mu = [_external_mean(model, res, leg)]
        if L >= 1:
            if len(kw['external_fields']) == 1:
                r1 = res
            else:
                kw1 = dict(kw)
                kw1.update(k=1, external_fields=[leg],
                           tau_grid=np.array([0.0]))
                kw1.pop('points', None)
                r1 = compute_cumulants(**kw1)
            be = r1.get('C_tau_by_ell') or {}
            for e in range(1, L + 1):
                arr = be.get(e)
                mu.append(0.0 if arr is None else
                          float(np.real(np.asarray(arr).ravel()[0])))
        out[leg] = mu
    return out


def _assemble_moment_temporal(model, res, kw, k, central):
    """Full / central k-point MOMENT along the same 1-D slice as ``C_tau``:

        M(τ) = ⟨φ_0(0) φ_1(τ) φ_2(0) … φ_{k−1}(0)⟩
             = Σ_π  Σ_{(ℓ_B): Σℓ_B ≤ L}  ∏_{B∈π} κ(B)^{(ℓ_B)}

    — the set-partition (cluster) expansion truncated at ONE shared loop budget
    ``L = max_ell``.  This is the perturbatively-consistent L-loop moment: the
    total loop order of every surviving term is ≤ L, so a 1-loop moment never
    smuggles in a partial 2-loop ℓ_B₁·ℓ_B₂ cross-term (which the naive
    product-of-fully-dressed-cumulants would).  The two agree only at tree
    (L=0); they diverge at L≥1 for any multi-block partition.

    Every block is the cumulant of ITS OWN legs (``external_fields`` entries:
    field and population) at those legs' times.  A singleton block {i} is the
    mean of leg i per loop order (``_leg_mean_by_ell``: the saddle at ℓ=0,
    the tadpoles at ℓ≥1), and it shares the loop budget like any block; the
    central moment keeps only the partitions without singletons.  A block of
    two legs is their cross-cumulant at their lag, from a ``k = 2`` run on
    those two legs over the lags it needs (the order-k run itself when
    ``k = 2``; ``_pair_key`` orders the pair so that τ = 0 is read on the
    slice's side of the tie); a
    block of three or more legs is the ``total_C_by_ell`` callable of a run on
    its legs (the order-k run for the full block).  ``res['moment_means']``
    records the leg means used."""
    import itertools
    from api import compute_cumulants as _compute

    def compute_cumulants(**kw_):
        # an extra run that left diagrams out makes the moment incomplete too
        r = _compute(**kw_)
        if isinstance(r, dict) and r.get('status') == 'incomplete':
            res['status'] = 'incomplete'
            res.setdefault('phase_j_incomplete_moment_runs', []).append(
                {'k': kw_.get('k'), 'phase_j_incomplete':
                 r.get('phase_j_incomplete')})
        return r
    tau = np.asarray(res['tau_grid'], dtype=float)
    legs = [tuple(e) if isinstance(e, (tuple, list)) else e
            for e in kw['external_fields']]
    L = int(kw.get('max_ell', 0) or 0)
    parts = list(_set_partitions(list(range(k))))

    means = {}
    if not central:
        means = _leg_mean_by_ell(model, res, kw, legs, L, compute_cumulants)
        res['moment_means'] = {leg: list(v) for leg, v in means.items()}

    # ── 2-leg blocks: one k = 2 run per ordered pair of legs (key) over the
    # lags it needs, read as C(τ) = ⟨φ_first(0) φ_second(τ)⟩.  A block with
    # the swept leg 1 is keyed (other leg, leg 1) at lag τ: at τ = 0 leg 1
    # is the EARLIEST leg of the tie on the slice (``_kpoint_slice_point``;
    # 1e-6 below the others under DAEDALUS_TAU0=nudge),
    # and the k = 2 grid's τ = 0 point is the same side (the second leg
    # earlier).  A block of two legs at 0 is keyed in leg order at lag 0 (the
    # anchor is the latest leg; tied pinned legs: the later-listed one counts
    # as earlier).  A pair of identical legs is even in the lag
    # (stationarity), so for k ≥ 3 it is run over |τ| only.
    def _pair_key(a, b, ti):
        if b == 1:                        # a == 0
            key, lag = (legs[a], legs[b]), float(tau[ti])
        elif a == 1:
            key, lag = (legs[b], legs[a]), float(tau[ti])
        else:
            key, lag = (legs[a], legs[b]), 0.0
        if key[0] == key[1] and k >= 3:
            lag = abs(lag)
        return key, lag

    needed = {}
    for part in parts:
        for bl in part:
            if len(bl) == 2:
                a, b = sorted(bl)
                for ti in range(tau.size):
                    key, lag = _pair_key(a, b, ti)
                    needed.setdefault(key, set()).add(lag)

    def _table(r):
        """(lag → index, {ℓ: real array}) of a k = 2 run's grid."""
        t = np.asarray(r['tau_grid'], dtype=float)
        be = r.get('C_tau_by_ell') or {0: r.get('C_tau')}
        arrs = {e: (None if be.get(e) is None else np.asarray(be[e]))
                for e in range(L + 1)}
        return t, arrs

    pair_tab = {}
    if k == 2 and (legs[0], legs[1]) in needed:
        pair_tab[(legs[0], legs[1])] = _table(res)       # the run itself
    for key, lags in needed.items():
        if key in pair_tab:
            continue
        kw2 = dict(kw)
        kw2.update(k=2, external_fields=list(key),
                   tau_grid=np.array(sorted(lags), dtype=float))
        kw2.pop('points', None)
        kw2.pop('tau_max', None)
        kw2.pop('tau_step', None)
        pair_tab[key] = _table(compute_cumulants(**kw2))

    def _pair_value(key, lag, ell):
        t, arrs = pair_tab[key]
        idx = np.flatnonzero(np.abs(t - lag) <= 1e-12 * max(1.0, abs(lag)))
        if idx.size == 0:
            raise RuntimeError(
                f'moment assembly: lag {lag!r} of the 2-point block {key} is '
                f'not on its grid {t.tolist()}')
        arr = arrs.get(ell)
        return 0.0 + 0.0j if arr is None else complex(arr[int(idx[0])])

    # ── blocks of 3 or more legs: the per-ell callables of a run on the
    # block's legs (in leg order), evaluated at the legs' slice times.
    kj = {}
    if k >= 3 and res.get('total_C_by_ell'):
        kj[tuple(legs)] = res['total_C_by_ell']

    def _block_fns(bl_legs):
        if bl_legs not in kj:
            kwj = dict(kw)
            kwj.update(k=len(bl_legs), external_fields=list(bl_legs))
            if 'points' in kwj:
                # evaluate-and-discard: the block's own points
                kwj['points'] = _moment_block_points(k, tau, legs)[bl_legs]
            kj[bl_legs] = compute_cumulants(**kwj).get('total_C_by_ell') or {}
        return kj[bl_legs]

    # The k external times of every point: those of ``run``'s slice 1 with
    # the default base lags (leg 1 swept, the other legs at 0), from
    # ``_kpoint_slice_times``.  A block of 3 or more legs is evaluated at its
    # legs' entries, so at τ = 0 (where leg 1 meets the other legs) it is the
    # left limit of the slice, as ``C_tau`` is, and the k = 3 central moment
    # is evaluated at the same times as slice 1 of κ₃: at a tie with leg 1
    # listed last (the exact rule, ``_kpoint_slice_point``), or with the pinned
    # legs at 0⁻ = −_ITO_EPS under DAEDALUS_TAU0=nudge.  For k ≥ 4 the pinned
    # legs coincide with each other, as on the slices.  A block of 2 legs
    # reads the k = 2 grid at τ or 0 (``_pair_key``), whose τ = 0 point is
    # the k = 2 equal-time value on the slice's side of the tie.
    slice_times = None
    slice_orders = None
    if k >= 3:
        slice_times, slice_orders = _moment_slice(k, tau)

    def _kappa(block, ell, ti):
        """ℓ-loop piece of the block's cumulant at the slice times."""
        b = sorted(block)
        if len(b) == 1:
            return means[legs[b[0]]][ell]
        if len(b) == 2:
            key, lag = _pair_key(b[0], b[1], ti)
            return _pair_value(key, lag, ell)
        # the block's legs in the point's evaluation order (leg 1 last at an
        # exact tie, so it is the earliest of the legs it meets)
        b = [i for i in slice_orders[ti] if i in block]
        fn = _block_fns(tuple(legs[i] for i in b)).get(ell)
        if not callable(fn):
            return 0.0 + 0.0j
        return complex(fn(*[slice_times[ti][idx] for idx in b]))

    M = np.empty(tau.size, dtype=complex)
    for ti in range(tau.size):
        tot = 0.0 + 0.0j
        for part in parts:
            if central and any(len(bl) == 1 for bl in part):
                continue                      # central drops every singleton
            if any(len(bl) == 1 and not any(means[legs[bl[0]]])
                   for bl in part):
                continue                      # a zero-mean leg kills the term
            # share the loop budget L across all the blocks (Σℓ_B ≤ L)
            for assign in itertools.product(range(L + 1), repeat=len(part)):
                if sum(assign) > L:
                    continue
                prod = 1.0 + 0.0j
                for bl, e in zip(part, assign):
                    prod *= _kappa(bl, e, ti)
                    if prod == 0.0:
                        break
                tot += prod
        M[ti] = tot
    return M


# ── Temporal k≥3 slices: the external times of each slice point ─────────────
#
# DEFAULT since Oct 2026 (DAEDALUS_TAU0=exact): ``_kpoint_slice_point`` --
# every leg at its nominal time; a point where the swept leg meets another leg
# is an exact tie, evaluated with the swept leg listed LAST, which the
# engine's tie order (a later-listed leg is infinitesimally earlier) makes the
# left limit of the slice exactly (the k = 2 τ = 0 rule; the k = 3 tree of an
# OU + quadratic model matches its closed form to 5e-16 at every tie).  Legs of
# different fields take one more run with the permuted ``external_fields``.
# What follows describes the DAEDALUS_TAU0=nudge placement
# (``_kpoint_slice_times``), the default before Oct 2026:
#
# ``run`` turns a temporal k≥3 cumulant (a function of the k external times;
# leg 0 is the anchor, t₀ = 0) into k−1 curves over ``tau_grid``: slice j
# sweeps leg j, and the other non-anchor legs (the PINNED legs) sit at
# ``Config.kpoint_base_lags`` (default 0).  Every point of a slice is the value
# of that curve itself, never a value decided by a tie between the swept leg
# and another leg.  At a coincident point that is the curve's LEFT limit, the
# same convention as the k=2 grid, whose τ = 0 point is sampled at
# τ = −``api.compute._ITO_EPS`` (the Itô left limit):
#
# * A pinned leg sits at its base lag, or at 0⁻ = −_ITO_EPS when that lag is 0
#   (|b| ≤ 1e−12), as the k=2 grid's τ = 0 point does.
# * The swept leg sits at τ, except where τ coincides (|τ − x| ≤ 1e−12) with
#   the anchor (x = 0) or with a pinned leg's base lag or evaluation time.
#   There it is placed one more _ITO_EPS below the earliest leg it coincides
#   with: the swept leg is infinitesimally EARLIER than every leg it meets,
#   and the point is the left limit τ → τ₀⁻ of its own curve.  With the
#   default base lags, the τ = 0 point of every slice has the swept leg at
#   −2·_ITO_EPS and the pinned legs at −_ITO_EPS; a ``tau_grid`` point at
#   τ = −_ITO_EPS meets the pinned legs' evaluation time and gets the same
#   times.  If the new time lands on yet another leg, the step is repeated;
#   where t − _ITO_EPS rounds back to t (|t| ≥ 2³⁴ ≈ 1.7e10), the step is
#   the next float below t instead.  So the swept leg never ties.
#
# Resolution.  The point is evaluated a finite _ITO_EPS below the leg it
# meets, so it equals the curve's left limit at τ₀ only up to the curve's
# change over that offset: 2·_ITO_EPS at τ = 0 with the default base lags
# (_ITO_EPS for the k=2 grid's τ = 0 point), i.e. O(1e−6) relative for rates
# of order 1.  That matters only where the jump at τ₀ is itself that small
# (``multipopulation_test`` in CHANGELOG 0.2.0).  τ-grid points closer than
# about 2·_ITO_EPS to 0 or to a base lag, and base lags within a few _ITO_EPS
# of 0 or of each other, are below the resolution of the nudge: e.g. a τ
# strictly between a pinned leg's evaluation time −_ITO_EPS and its lag 0 is
# evaluated as given, with the swept leg between that leg and the anchor.
#
# Pinned legs that coincide among themselves (for k ≥ 4 with the default base
# lags, all of them at −_ITO_EPS) stay tied.  The engine then resolves the tie
# by its external-time tie order (the later-listed leg counts as
# infinitesimally earlier; ``_tie_order_sign`` in
# ``engine/integration/time_domain/final_integral.py``).  The pinned legs do
# not move along a slice, so that order is the same at every point of the
# curve and adds no jump to it.  But every point of such a slice is then a
# tie, and the value at a tie is that one-sided limit only as accurately as
# the engine evaluates the tie: a tie can send a two-time region to the
# scipy.nquad fallback, or a region with three or more integration times to
# the poset integrator, whose known lower-bound inheritance error then
# applies.  For single_population_spike_reset_test k = 4 (tree) the default
# slices are 1.0-8.5% (about 1e-3 absolute) off the tie-order limit at
# τ = 0.5 (CHANGELOG 0.2.0, known issue).  Distinct, nonzero
# ``kpoint_base_lags`` keep the pinned legs apart.
#
# The full grid (``Config.kpoint_full_grid``) is not a set of curves: each axis
# value is mapped on its own (0 → −_ITO_EPS, ``_kpoint_pinned_time``), and
# legs with equal values stay tied: the engine's tie order decides on the
# grid's diagonals.  So at the origin ``C_tau_grid`` holds the tie-order
# value, which differs from a slice's τ = 0 point where that slice's swept
# leg is not the highest-indexed of the tied legs (slice 1 at k = 3).
#
# The temporal k≥3 moment outputs (``Config.output`` = 'moment' or
# 'central_moment', ``_assemble_moment_temporal``) evaluate their cumulant
# blocks of 3 or more legs at the times of slice 1 with the default base
# lags, so all of the above applies to them too (including the tied pinned
# legs for k ≥ 4).
_KPOINT_TIE_ATOL = 1e-12


def _kpoint_pinned_time(b, eps):
    """Evaluation time of a non-swept leg of a k≥3 slice at base lag ``b``:
    ``b`` itself, or the Itô left limit ``-eps`` when ``b`` is 0
    (``|b| <= 1e-12``)."""
    b = float(b)
    # ``not (<=)``: a NaN base lag stays NaN (the engine refuses it) instead
    # of being read as the anchor time.
    return -eps if abs(b) <= _KPOINT_TIE_ATOL else b


def _kpoint_below(x, eps):
    """``x - eps``, or the next float below ``x`` where ``x - eps`` rounds
    back to ``x`` (``|x| >= 2**34`` for ``eps = 1e-6``): a time strictly
    earlier than ``x``."""
    y = x - eps
    return y if y < x else math.nextafter(x, -math.inf)


def _kpoint_slice_times(base, j, tau, eps):
    """The k external times ``[t_0, …, t_{k−1}]`` (``t_0 = 0``) at which
    :func:`run` evaluates point ``tau`` of the k≥3 slice ``j``.

    Leg ``j`` is swept; every other non-anchor leg sits at its entry of
    ``base`` (length k−1, entry ``i − 1`` for leg ``i``), via
    :func:`_kpoint_pinned_time`.  Leg ``j`` is at ``tau``, unless ``tau`` is
    within 1e−12 of the anchor or of a pinned leg's base lag or evaluation
    time.  Then it goes ``eps`` below the earliest of the legs it meets (the
    next float below where ``eps`` is under the float spacing, see
    :func:`_kpoint_below`), so the value is the left limit of the slice at
    ``tau`` (repeated if that time meets another leg; the swept leg never
    ties).  See the comment above for the convention and its resolution."""
    k = len(base) + 1
    if not 1 <= j < k:
        raise ValueError(f'slice index j={j} must be in 1..{k - 1}')
    times = [0.0] * k
    pinned = [leg for leg in range(1, k) if leg != j]
    for leg in pinned:
        times[leg] = _kpoint_pinned_time(base[leg - 1], eps)
    # (nominal time, evaluation time) of the anchor and of each pinned leg
    others = [(0.0, 0.0)] + [(float(base[leg - 1]), times[leg])
                             for leg in pinned]
    t = float(tau)
    met = [e for b, e in others
           if abs(t - b) <= _KPOINT_TIE_ATOL or abs(t - e) <= _KPOINT_TIE_ATOL]
    if met:
        t = _kpoint_below(min(met), eps)
        for _ in range(k):              # never land on another leg
            met = [e for _b, e in others if abs(t - e) <= _KPOINT_TIE_ATOL]
            if not met:
                break
            t = _kpoint_below(min(met), eps)
    times[j] = t
    return times


def _kpoint_slice_point(base, j, tau, scale):
    """``(times, order)``: point ``tau`` of the k≥3 slice ``j`` at the EXACT
    tie (the default since Oct 2026, ``DAEDALUS_TAU0=exact``; the −_ITO_EPS
    placement of :func:`_kpoint_slice_times` is ``DAEDALUS_TAU0=nudge``).

    Every leg sits at its nominal time: the anchor at 0, the pinned legs at
    their base lags, the swept leg ``j`` at ``tau``, snapped exactly onto a
    leg it meets (within ``1e-12 * scale``; ``scale`` is the largest |time| of
    the grid and the base lags, so the tie is unit invariant).  A tie is
    resolved by the engine's tie order (``_tie_order_sign``: a later-listed
    leg counts as infinitesimally EARLIER), so a tied point is evaluated with
    the legs in ``order`` = anchor, pinned legs, then ``j`` LAST: the swept
    leg is then earlier than every leg it meets, and the value is the left
    limit of the slice, exactly (the k = 2 τ = 0 rule).  Pinned legs
    keep their relative order, so a pinned leg at base lag 0 counts as
    earlier than the anchor, as its 0⁻ placement did.  ``order`` is the
    identity (``range(k)``) at points without a tie."""
    k = len(base) + 1
    if not 1 <= j < k:
        raise ValueError(f'slice index j={j} must be in 1..{k - 1}')
    atol = 1e-12 * float(scale)
    times = [0.0] * k
    pinned = [leg for leg in range(1, k) if leg != j]
    for leg in pinned:
        b = float(base[leg - 1])
        times[leg] = 0.0 if abs(b) <= atol else b
    t = float(tau)
    met = [times[leg] for leg in [0] + pinned if abs(t - times[leg]) <= atol]
    if met:
        times[j] = min(met, key=lambda x: abs(t - x))
        return times, [0] + pinned + [j]
    times[j] = t
    return times, list(range(k))


def _kpoint_incomplete(res, cfg, exc, msg):
    """Mark a k≥3 result incomplete and report ``msg``: a
    ``PhaseJIncompleteWarning``, or ``PhaseJIncompleteError`` under
    ``policy={'mode': 'strict'}``."""
    import warnings
    from api.compute import (PhaseJIncompleteError, PhaseJIncompleteWarning,
                             _policy_is_strict)
    from api.save import _ensure_warning_shown
    res['status'] = 'incomplete'
    if _policy_is_strict(cfg.policy):
        raise PhaseJIncompleteError(msg, result=res) from exc
    _ensure_warning_shown(PhaseJIncompleteWarning, 'always', message=msg,
                          stacklevel=3)
    warnings.warn(msg, PhaseJIncompleteWarning, stacklevel=3)



def _kpoint_plan(cfg, k, tau):
    """Where :func:`run` reads a temporal k≥3 cumulant: the τ grid of the
    slices (``tau``, the run's grid; at most 41 points, ``[0.0]`` when
    empty), the base lags, the tie rule, the external times and leg order of
    every slice point (``_kpoint_slice_point`` / ``_kpoint_slice_times``)
    and the full grid's axis and point map (``kpoint_full_grid``).  Made
    before the run, so evaluate-and-discard knows the points
    (:func:`_kpoint_run_points`)."""
    from api.compute import _ITO_EPS, _tau0_mode
    plan = {'replaced': False, 'resampled_from': None}
    tau = None if tau is None else np.asarray(tau, dtype=float)
    if tau is None or tau.size == 0:
        tau = np.array([0.0])
        plan['replaced'] = True
    elif tau.size > 41:                      # bound the # of 1-D evaluations
        plan['resampled_from'] = tau.size
        tau = np.linspace(float(tau.min()), float(tau.max()), 41)
        plan['replaced'] = True
    plan['tau'] = tau

    # The connected k-point cumulant depends on k−1 time differences
    # τ_j = t_j − t_0 (leg 0 anchored at τ=0).  ``base`` is the fixed point
    # the axis-parallel slices pass through (the non-swept legs sit here).
    base = cfg.kpoint_base_lags
    if base is not None:
        base = [float(b) for b in base]
        if not all(math.isfinite(b) for b in base):
            raise ValueError(
                f"kpoint_base_lags must be finite; got {base!r}.")
        if len(base) != k - 1:
            raise ValueError(
                f"kpoint_base_lags must have k−1 = {k-1} entries (one per "
                f"non-anchor leg); got {len(base)}.")
    else:
        base = [0.0] * (k - 1)
    plan['base'] = base

    exact_tie = _tau0_mode() == 'exact'
    plan['exact_tie'] = exact_tie
    # the tie tolerance's scale: the largest |time| of the grid and base
    tie_scale = max([abs(float(t)) for t in tau] + [abs(b) for b in base]
                    + [0.0])

    def _args(vals):                # full grid: vals = τ of legs 1..k−1
        # Every leg on its own; legs with equal values stay tied and the
        # engine's tie order decides (a later-listed leg is earlier, so a
        # leg at the anchor time is its Itô left limit 0⁻).  'nudge':
        # a leg at τ = 0 is evaluated at −_ITO_EPS instead.
        if exact_tie:
            return [0.0] + [0.0 if abs(float(v)) <= 1e-12 * tie_scale
                            else float(v) for v in vals]
        return [0.0] + [_kpoint_pinned_time(v, _ITO_EPS) for v in vals]
    plan['grid_args'] = _args

    # The external times of every slice point, computed once: slice j
    # sweeps leg j, the other non-anchor legs sit at ``base``, and a
    # point where leg j meets another leg is the LEFT limit of the
    # slice (``_kpoint_slice_times``).  A failure is kept
    # (``slice_error``): ``run`` reports it as a failed slice evaluation.
    plan['slice_error'] = None
    try:
        if exact_tie:
            pts = {j: [_kpoint_slice_point(base, j, t, tie_scale)
                       for t in tau] for j in range(1, k)}
            plan['slice_times'] = {j: [p[0] for p in v]
                                   for j, v in pts.items()}
            plan['slice_orders'] = {j: [p[1] for p in v]
                                    for j, v in pts.items()}
        else:
            plan['slice_times'] = {
                j: [_kpoint_slice_times(base, j, t, _ITO_EPS) for t in tau]
                for j in range(1, k)}
            plan['slice_orders'] = {j: [list(range(k))] * tau.size
                                    for j in range(1, k)}
    except Exception as e:          # as broad as the slice evaluation's catch
        plan['slice_error'] = e
        plan['slice_times'] = plan['slice_orders'] = None
    # the full grid's axis: downsampled so the n^{k−1} evaluations stay
    # bounded
    n = min(tau.size, max(2, int(4000 ** (1.0 / (k - 1)))))
    plan['grid_tau'] = (tau if n >= tau.size else
                        np.linspace(float(tau.min()), float(tau.max()), n))
    return plan


def _kpoint_run_points(plan, legs, k, full_grid, moment):
    """``{external legs: [points]}``: the external-time tuples at which
    :func:`run` reads the callables of each k≥3 run it makes (the run on
    ``legs`` and the runs with permuted legs; with ``moment``, the cumulant
    blocks of three or more legs), from ``plan`` (:func:`_kpoint_plan`).
    Evaluate-and-discard runs are evaluated exactly there."""
    import itertools
    legs = tuple(legs)
    runs = {legs: []}
    for j in range(1, k):
        for times, order in zip(plan['slice_times'][j],
                                plan['slice_orders'][j]):
            runs.setdefault(tuple(legs[o] for o in order), []).append(
                tuple(times[o] for o in order))
    if full_grid:
        g = plan['grid_tau']
        try:
            grid = [tuple(plan['grid_args']([g[m] for m in idx]))
                    for idx in itertools.product(range(g.size), repeat=k - 1)]
        except Exception:
            grid = []       # the grid's own evaluation fails and reports it
        runs[legs].extend(grid)
    if moment:
        try:
            blocks = _moment_block_points(k, plan['tau'], legs)
        except Exception:
            blocks = {}     # likewise for the moment (``_kappa``)
        for key, pts in blocks.items():
            runs.setdefault(key, []).extend(pts)
    return runs


def _moment_slice(k, tau):
    """The external times and leg orders of the temporal k≥3 moment's points
    (:func:`_assemble_moment_temporal`): those of ``run``'s slice 1 with the
    default base lags."""
    from api.compute import _ITO_EPS, _tau0_mode
    if _tau0_mode() == 'exact':
        # exact ties; a tied point lists leg 1 last (``_kpoint_slice_point``)
        scale = max([abs(float(t)) for t in tau] + [0.0])
        pts = [_kpoint_slice_point([0.0] * (k - 1), 1, float(t), scale)
               for t in tau]
        return [p[0] for p in pts], [p[1] for p in pts]
    return ([_kpoint_slice_times([0.0] * (k - 1), 1, float(t), _ITO_EPS)
             for t in tau], [list(range(k))] * len(tau))


def _moment_block_points(k, tau, legs):
    """``{block legs: [points]}``: where the temporal k≥3 moment evaluates
    each cumulant block of three or more legs (``_kappa`` in
    :func:`_assemble_moment_temporal`)."""
    slice_times, slice_orders = _moment_slice(k, tau)
    out = {}
    for part in _set_partitions(list(range(k))):
        for block in part:
            if len(block) < 3:
                continue
            for ti in range(len(tau)):
                b = [i for i in slice_orders[ti] if i in block]
                out.setdefault(tuple(legs[i] for i in b), []).append(
                    tuple(slice_times[ti][idx] for idx in b))
    return out

def _spatial_raw_moment_message(max_ell, mu_a=None, mu_b=None):
    why = ('' if mu_a is None else f' (saddles {mu_a}, {mu_b})')
    return (f"Config.output='moment' for a spatial k=2 run at "
            f"max_ell={max_ell}: the raw moment needs the loop corrections "
            f"of the means, which the spatial integrator does not compute"
            f"{why}.  Use output='central_moment' (= the cumulant at k=2) "
            f"or max_ell=0.")


def _refuse_spatial_raw_moment(model, kw, ext, max_ell):
    """Refuse a spatial raw k=2 moment that needs the means' loop corrections
    BEFORE the (expensive) loop run, not after it.

    The rule: with a loop budget L the raw moment needs mu_a^(la) mu_b^(lb)
    for la + lb <= L, i.e. the means' tadpole shifts as soon as L >= 2, or
    L >= 1 with a nonzero saddle.  L >= 2 is decided from the budget alone;
    L = 1 needs the saddle, read from a one-event tree-level k = 1 run (the
    same call the k = 1 path makes), per distinct external leg."""
    from api import compute_cumulants
    if max_ell >= 2:
        raise NotImplementedError(
            _spatial_raw_moment_message(max_ell))
    base = {kk: v for kk, v in kw.items()
            if kk not in ('spatial_grid', 'tau_max', 'tau_step', 'tau_grid',
                          'spatial_points', 'external_fields', 'k', 'max_ell')}
    mus = []
    seen = {}
    for leg in ext:
        key = tuple(leg) if isinstance(leg, (tuple, list)) else leg
        if key not in seen:
            r1 = compute_cumulants(
                **base, k=1, max_ell=0, external_fields=[leg],
                spatial_points=np.zeros((1, 0, 2), dtype=float))
            seen[key] = _external_mean(model, r1, leg)
        mus.append(seen[key])
    if any(mus):
        raise NotImplementedError(
            _spatial_raw_moment_message(max_ell, *mus[:2]))


def run(model: dict, cfg: Config, module=None) -> dict:
    """Resolve the config against the model's defaults and call
    ``compute_cumulants``.  Handles temporal vs spatial, the spatial
    k≥3 event path, and the Dyson-order override.  Returns the result
    dict with ``cfg`` / ``model`` attached under ``'_cfg'`` / ``'_model'``."""
    from api import compute_cumulants

    cfg._check_numbers()              # fields may be set after construction
    # an api.sde.SDE (incl. point-process models) is built here, as by hand
    if not isinstance(model, dict) and callable(getattr(model, 'build', None)):
        model = model.build()
    max_ell = (cfg.max_ell if cfg.max_ell is not None
               else (_meta(module, 'ell_default', 0) if module else 0))
    ext = cfg.external_fields
    ext_is_explicit = ext is not None

    # Resolve the cumulant order k.  Explicit ``cfg.k`` wins; otherwise infer
    # it from an explicit ``external_fields`` (a k-point cumulant has exactly
    # k legs); otherwise fall back to the model's ``k_default``.  If BOTH are
    # given they must agree — a mismatch is a contradiction, not something to
    # silently paper over.
    if cfg.k is not None:
        k = cfg.k
        if ext_is_explicit and len(ext) != k:
            raise ValueError(
                f"Config mismatch: k={k} but external_fields lists "
                f"{len(ext)} leg(s) {ext}.  A k-point cumulant needs exactly "
                f"k legs.  Drop k (it is inferred from external_fields), set "
                f"k={len(ext)}, or pass {k} legs.")
    elif ext_is_explicit:
        k = len(ext)
    else:
        k = _meta(module, 'k_default', 2) if module else 2

    if ext is None and module is not None:
        ext = _meta(module, 'recommended_external_fields')
    # Auto-build a k-matching external_fields ONLY when the caller gave none
    # explicitly: k copies of the first physical field's leg.  Triggers when
    # ext is absent, or a stale METADATA *recommendation* is the wrong length
    # or names a missing field.  An explicit Config.external_fields is used
    # verbatim (it may legitimately name a response leg); any k-mismatch was
    # caught above.
    if not ext_is_explicit and not _ext_is_valid(model, ext, k):
        ext = _auto_external_fields(model, k)
    # a bare field name ('x') is the leg ('x', 1) of a one-unit field
    from api.access import resolve_external_fields
    ext = resolve_external_fields(model, ext)
    # Layered fundamental: model param defaults ← model DEFAULT_FUNDAMENTAL
    # ← explicit cfg override.  Each layer wins over the one before, so a
    # loaded model runs out of the box and the notebook can override any
    # subset.
    fundamental = fundamental_from_model(model)
    fundamental.update(getattr(module, 'DEFAULT_FUNDAMENTAL', {}) or {}
                       if module else {})
    if cfg.parameters:
        # Reject mistyped override names instead of silently inserting a
        # key the backend ignores (which would run with the DEFAULT value).
        from api.naming import free_parameter_names
        _valid = set(free_parameter_names(model))
        _unknown = [kk for kk in cfg.parameters if kk not in _valid]
        if _valid and _unknown:
            raise ValueError(
                f"Unknown parameter override(s) {_unknown}. "
                f"Declared parameters: {sorted(_valid)}.")
        fundamental.update(cfg.parameters)

    # Dyson override: inject into the model's spatial policy at run time.
    # Copy first so we never mutate the caller's model dict — load_model
    # returns it un-copied and the documented flow is load-once-run-many,
    # so an in-place write would leak the Dyson policy into later runs.
    if cfg.dyson_order is not None and model.get('spatial'):
        model = dict(model)
        model['spatial'] = dict(model['spatial'])
        model['spatial']['dyson'] = {'mode': 'fixed',
                                     'order': int(cfg.dyson_order)}
        if cfg.reference_diffusion is not None:
            model['spatial']['reference_diffusion'] = \
                float(cfg.reference_diffusion)

    kw = dict(model=model, k=k, max_ell=max_ell, fundamental=fundamental,
              external_fields=ext, parallel=cfg.parallel, verbose=cfg.verbose,
              n_workers=cfg.n_workers)

    # Mean-field DAE root-selection overrides (multi-root models such as
    # the double-well regime mu<0).  The Config value wins, then the model
    # file's METADATA ``fixed_point_index_default`` / ``seed_box_default``
    # (written by the model builder UI); forward each only when set, so that
    # ``compute_cumulants``' own defaults (fixed_point_index=0,
    # mf_dae_n_starts=64, mf_dae_seed_box=None) are preserved otherwise.
    fpi = cfg.fixed_point_index
    if fpi is None and module is not None:
        fpi = _meta(module, 'fixed_point_index_default')
        if fpi is not None:
            from api.compute import _entry_int
            if _entry_int("METADATA['fixed_point_index_default']", fpi) < 0:
                raise ValueError(
                    f"METADATA['fixed_point_index_default'] must be >= 0; "
                    f"got {fpi!r}.")
    if fpi is not None:
        kw['fixed_point_index'] = int(fpi)
    if cfg.mf_dae_n_starts is not None:
        kw['mf_dae_n_starts'] = int(cfg.mf_dae_n_starts)
    seed_box = cfg.mf_dae_seed_box
    if seed_box is None and module is not None:
        seed_box = _meta(module, 'seed_box_default')
        if seed_box is not None and not isinstance(seed_box, dict):
            raise ValueError(
                f"METADATA['seed_box_default'] must be a dict "
                f"{{state variable: (lo, hi)}}; got {seed_box!r}.  Fix it in "
                f"the model file (or pass Config(mf_dae_seed_box=...)).")
    if seed_box is not None:
        kw['mf_dae_seed_box'] = seed_box
    for _opt in ('taylor_order', 'use_cache', 'accuracy', 'backend', 'discard'):
        if getattr(cfg, _opt) is not None:
            kw[_opt] = getattr(cfg, _opt)
    if cfg.policy is not None:
        kw['policy'] = cfg.policy
    if getattr(cfg, 'delay', None) is not None:
        kw['delay'] = dict(cfg.delay)
    if getattr(cfg, 'spectra', None) not in (None, False):
        kw['spectra'] = (dict(cfg.spectra) if isinstance(cfg.spectra, dict)
                         else cfg.spectra)
    if cfg.preflight is not None:
        kw['preflight'] = bool(cfg.preflight)
    cfg._check_switch_on()            # fields may be set after construction
    if cfg.t_min is not None:
        # switch-on transient: absolute times, temporal cumulants only
        if is_spatial(model):
            raise NotImplementedError(
                'Config.t_min (switch-on transients) is implemented for '
                'temporal models only')
        if (getattr(cfg, 'output', 'cumulant') or 'cumulant') != 'cumulant':
            raise NotImplementedError(
                f"Config.t_min with output={cfg.output!r}: only the cumulant "
                f"is assembled for switch-on runs (assemble moments from "
                f"result['C_times'] of the lower orders)")
        kw['t_min'] = float(cfg.t_min)
        kw['times'] = cfg.resolved_times()
        if cfg.initial_covariance is not None:
            kw['initial_covariance'] = cfg.initial_covariance

    _kpoint_grid = None            # set below if a k≥3 spatial grid is built
    if is_spatial(model):
        if k == 1:
            # k=1 spatial mean ⟨φ⟩: by translation invariance the mean is
            # x-independent (= the mean-field saddle φ*), so a single event
            # suffices.  The spatial loop integrator does not assemble the k=1
            # tadpole, so the 1-point mean is the tree-level (mean-field) value.
            if max_ell:
                print('  [k=1 spatial] loop tadpoles for the 1-point mean are '
                      'not yet supported by the spatial integrator — showing '
                      'the mean-field (tree) value.')
            kw['max_ell'] = 0
            # Keep the LOCAL max_ell in lockstep so result['_resolved']
            # reports what was actually computed (tree only), not the
            # requested order the spatial integrator silently dropped.
            max_ell = 0
            kw['spatial_points'] = np.zeros((1, 0, 2), dtype=float)
        elif k != 2 and cfg.spatial_points is None:
            # No explicit events → sweep the FULL 2(k-1)-D grid: each of the k-1
            # non-anchor legs ranges over chi_grid (× tau_grid).  Pure convenience
            # over the same k-point evaluator (events built from the grid axes);
            # the flat result is reshaped to the (τ,χ)×(k-1) grid after the run.
            chi_g = cfg.resolved_grid()
            if chi_g is None:
                raise ValueError(
                    'spatial k≥3 needs EITHER Config.spatial_points='
                    '(n_pts, k-1, 2) explicit (x_j, τ_j) events, OR a chi_grid '
                    '(+ optional tau_grid) to sweep the full 2(k-1)-D grid over '
                    'the k-1 non-anchor legs.')
            tau_g = cfg.resolved_tau_grid()
            if tau_g is None:
                tau_g = np.array([0.0])              # equal-time grid (τ axes = 1 pt)
            legs = [(float(x), float(t)) for t in tau_g for x in chi_g]
            n_ev = len(legs) ** (k - 1)
            if n_ev > 20000:
                raise ValueError(
                    f'k={k} grid would be {n_ev} events = '
                    f'({len(chi_g)}*{len(tau_g)})^(k-1); too many — k-point grids '
                    'scale combinatorially. Shrink chi_grid/tau_grid, or pass '
                    'explicit Config.spatial_points.')
            if cfg.verbose or n_ev > 400:
                print(f'  [k={k} grid] {n_ev} events = '
                      f'({len(chi_g)}*{len(tau_g)})^(k-1); tree ~{n_ev * 0.3:.0f}s'
                      ', 1-loop is far slower.')
            import itertools as _it
            kw['spatial_points'] = np.array(
                [[list(p) for p in combo]
                 for combo in _it.product(legs, repeat=k - 1)], dtype=float)
            _kpoint_grid = (np.asarray(tau_g), np.asarray(chi_g), k - 1)
        elif k != 2:
            kw['spatial_points'] = np.asarray(cfg.spatial_points, dtype=float)
        else:
            grid = cfg.resolved_grid()
            if grid is None:
                grid = np.linspace(-6.0, 6.0, 49)
            kw['spatial_grid'] = grid
            kw['tau_max'] = 0.0 if cfg.tau_max is None else cfg.tau_max
            kw['tau_step'] = 1.0 if cfg.tau_step is None else cfg.tau_step
    else:
        kw['tau_max'] = (_meta(module, 'tau_max', 10.0)
                         if cfg.tau_max is None else cfg.tau_max)
        kw['tau_step'] = (_meta(module, 'tau_step', 0.5)
                          if cfg.tau_step is None else cfg.tau_step)

    # An explicit tau_grid (array | (lo,hi,n)) overrides tau_max/tau_step on the
    # paths that build a τ grid — temporal (any k) and spatial k=2 C(χ,τ).  The
    # spatial event paths (k=1 mean, k≥3 spatial_points) carry their own times.
    tg = cfg.resolved_tau_grid()
    if tg is not None and not (is_spatial(model) and k != 2):
        kw['tau_grid'] = tg

    from api.covariance import resolve_covariance_options
    _cov = resolve_covariance_options(getattr(cfg, 'covariance', None),
                                      'Config.covariance')
    from api.resum import resolve_resum_options
    _resum = resolve_resum_options(getattr(cfg, 'resum', None),
                                   'Config.resum')
    if _resum is not None:
        # opt-in resummation (api.hartree / api.resum); the loop path below
        # is untouched when resum is None
        return _run_resummed(model, cfg, kw, k, max_ell, ext, fundamental,
                             _resum, _cov)
    if _cov is not None:
        # every pair of the fields in one shared pass (api.covariance)
        from api.covariance import covariance_matrix
        if is_spatial(model):
            raise NotImplementedError(
                'Config.covariance is for temporal models')
        _opts = {key: v for key, v in kw.items()
                 if key not in ('model', 'k', 'max_ell', 'fundamental',
                                'external_fields')}
        res = covariance_matrix(
            model, _cov['fields'], max_ell, parameters=fundamental,
            pairs=_cov['pairs'], derivatives=_cov['derivatives'],
            keep_entries=_cov['keep_entries'], **_opts)
        res['_cfg'] = cfg
        res['_model'] = model
        res['_resolved'] = dict(k=2, max_ell=max_ell, fields=res['fields'],
                                parameters=fundamental,
                                fundamental=fundamental)
        return res

    if (is_spatial(model) and k == 2 and max_ell >= 1
            and (getattr(cfg, 'output', 'cumulant') or 'cumulant') == 'moment'):
        _refuse_spatial_raw_moment(model, kw, ext, max_ell)

    # Temporal k≥3: where the result will be read (the slices below), known
    # before the run; with evaluate-and-discard the run is evaluated there
    _kplan = _kpoint_runs = None
    if not is_spatial(model) and k >= 3 and cfg.t_min is None:
        _kplan = _kpoint_plan(cfg, k, kw['tau_grid'] if 'tau_grid' in kw
                              else np.arange(-kw['tau_max'], kw['tau_max']
                                             + kw['tau_step'] * 0.5,
                                             kw['tau_step']))
        if kw.get('discard') not in (None, False) and (
                _kplan['slice_error'] is not None):
            # no slice points: the slices fail (reported below), and the
            # run keeps its callables
            kw['discard'] = False
        elif kw.get('discard') not in (None, False):
            _kpoint_runs = _kpoint_run_points(
                _kplan, [tuple(e) if isinstance(e, (list, tuple)) else e
                         for e in kw['external_fields']], k,
                bool(cfg.kpoint_full_grid),
                (getattr(cfg, 'output', 'cumulant') or 'cumulant')
                in ('moment', 'central_moment'))
            kw['points'] = _kpoint_runs[tuple(
                tuple(e) if isinstance(e, (list, tuple)) else e
                for e in kw['external_fields'])]

    res = compute_cumulants(**kw)

    # Spatial k≥3 grid: reshape the flat per-event k-point cumulant back into the
    # full 2(k-1)-D grid — one (τ, χ) pair per non-anchor leg, axis order
    # (τ₂, χ₂, τ₃, χ₃, …) — so it slices like the k=2 C(χ,τ).
    if _kpoint_grid is not None and isinstance(res, dict):
        tau_g, chi_g, nlegs = _kpoint_grid
        shp = (len(tau_g), len(chi_g)) * nlegs
        if res.get('C_kpoint') is not None:
            res['C_kpoint_grid'] = np.asarray(res['C_kpoint']).reshape(shp)
        be = res.get('C_kpoint_by_ell') or {}
        if be:
            res['C_kpoint_grid_by_ell'] = {
                e: np.asarray(v).reshape(shp) for e, v in be.items()}
        res['kpoint_grid_axes'] = {'tau': tau_g, 'chi': chi_g, 'n_legs': nlegs}

    # Temporal k≥3: compute_cumulants returns ``C_tau=None`` and callable
    # ``total_C`` / ``total_C_by_ell`` (the connected k-point cumulant is a
    # function of the k external times).  Synthesise the natural 1-D slice
    # C_k(τ) = ⟨φ(0) φ(τ) φ(0) … φ(0)⟩_c — sweep leg 1, pin the others at
    # τ=0 — and store it as ``C_tau`` / ``C_tau_by_ell`` so k≥3 plots and
    # compares exactly like k=2 (a curve over ``tau_grid``).  The simulator
    # estimates the matching slice with ``lag_bins=[0, None, 0, …]``.  The
    # times of each slice point (the Itô nudges at coincident legs) are set by
    # ``_kpoint_slice_times``; see the comment above it.
    if (_kplan is not None and res.get('C_tau') is None
            and res.get('t_min') is None
            and callable(res.get('total_C'))):
        tau = _kplan['tau']
        if _kplan['replaced']:
            res['tau_grid'] = tau
        if _kplan['resampled_from'] is not None:
            import warnings
            n_in = _kplan['resampled_from']
            res['_kpoint_tau_resampled'] = (
                f'tau_grid of {n_in} points replaced by 41 evenly spaced '
                f'points over [{tau[0]}, {tau[-1]}] for the k={k} slices')
            from api.save import _ensure_warning_shown
            _msg = (f'k={k}: the {n_in}-point tau_grid was replaced by 41 '
                    f'evenly spaced points over [{tau[0]}, {tau[-1]}] for the '
                    f'slices (result["tau_grid"]); evaluate result["total_C"] '
                    f'directly for other times.')
            _ensure_warning_shown(UserWarning, 'always', message=_msg,
                                  stacklevel=2)
            warnings.warn(_msg, UserWarning, stacklevel=2)
        base = _kplan['base']
        _args = _kplan['grid_args']

        # The callables of the run with its legs in another order: a point
        # where the swept leg ties with another leg is evaluated with the
        # swept leg LAST (``_kpoint_slice_point``).  The cumulant is symmetric
        # under a joint permutation of legs and times, so legs of one field
        # reuse this run's callables; legs of different fields take one more
        # run with the permuted ``external_fields``.
        _legs = [tuple(e) if isinstance(e, (list, tuple)) else e
                 for e in kw['external_fields']]
        _perm_runs = {}

        def _perm_fns(order):
            order = tuple(order)
            pf = [_legs[o] for o in order]
            if pf == _legs:
                return res.get('total_C_by_ell') or {}, res.get('total_C')
            key = tuple(pf)
            if key not in _perm_runs:
                kwp = dict(kw)
                kwp['external_fields'] = list(pf)
                if _kpoint_runs is not None:
                    # evaluate-and-discard: this run's own points
                    kwp['points'] = _kpoint_runs[key]
                rp = compute_cumulants(**kwp)
                if isinstance(rp, dict) and rp.get('status') == 'incomplete':
                    # an extra run that left diagrams out makes the slices
                    # incomplete too (as the moment's extra runs do)
                    res['status'] = 'incomplete'
                    res.setdefault('phase_j_incomplete_perm_runs', []).append(
                        {'external_fields': list(pf),
                         'phase_j_incomplete': rp.get('phase_j_incomplete')})
                _perm_runs[key] = (rp.get('total_C_by_ell') or {},
                                   rp.get('total_C'))
            return _perm_runs[key]

        def _slice(pick, j):            # sweep leg j over tau, others at base
            out = np.empty(tau.size, dtype=complex)
            for i, times in enumerate(slice_times[j]):
                order = slice_orders[j][i]
                fn = pick(order)
                out[i] = complex(fn(*[times[o] for o in order]))
            return out

        try:
            # The external times of every slice point (``_kpoint_plan``).
            # Kept in the result.
            if _kplan['slice_error'] is not None:
                raise _kplan['slice_error']
            slice_times = _kplan['slice_times']
            slice_orders = _kplan['slice_orders']
            tcb = {e: f for e, f in (res.get('total_C_by_ell') or {}).items()
                   if callable(f)}
            # The k−1 axis-parallel slices through ``base``.  When total_C is
            # the plain sum of the per-ell callables (``api.compute``), its
            # slice is that sum of their slices, in total_C's order, instead
            # of a second evaluation of every diagram (half the
            # Phase J evaluations of a k≥3 run).
            slices_by_ell = {j: {e: _slice(lambda o, e=e: _perm_fns(o)[0][e], j)
                                 for e in tcb}
                             for j in range(1, k)}
            parts = getattr(res['total_C'], 'sum_of_by_ell', None)
            if (parts is not None and list(parts) == list(tcb)
                    and all(parts[e] is tcb[e] for e in tcb)):
                slices = {}
                for j in range(1, k):
                    out = np.empty(tau.size, dtype=complex)
                    for i in range(tau.size):
                        out[i] = sum(complex(slices_by_ell[j][e][i])
                                     for e in tcb)
                    slices[j] = out
            else:
                slices = {j: _slice(lambda o: _perm_fns(o)[1], j)
                          for j in range(1, k)}
            _perm_runs.clear()          # free the permuted runs' plans
            res['C_tau_slices'] = slices
            res['C_tau_slices_by_ell'] = slices_by_ell
            res['C_tau'] = slices[1]                    # canonical single slice
            res['C_tau_by_ell'] = slices_by_ell[1]
            res['_kpoint_base'] = base
            res['_kpoint_slice_times'] = {
                j: np.array(v, dtype=float) for j, v in slice_times.items()}
            # the leg order each point was evaluated in (the swept leg last
            # at an exact tie; the identity elsewhere and under 'nudge')
            res['_kpoint_slice_orders'] = {
                j: np.array(v, dtype=int) for j, v in slice_orders.items()}
            res['_kpoint_slice'] = (
                f'{k-1} slices: leg j swept over tau_grid (τ_j = t_j − t_0), '
                f'legs ≠ j fixed at base={base}, for j = 1..{k-1}; a point '
                f'where leg j meets another leg is the left limit of its '
                f'slice (times in _kpoint_slice_times)')

        except Exception as e:
            # The slices could not be evaluated: the result keeps the
            # callables (``total_C``) but has no curve, so it is marked
            # incomplete and the failure is reported (raised under a strict
            # policy), never only printed.  (The catch stays broad —
            # lambdified callables have a wide failure surface.)
            res['_kpoint_slice_error'] = repr(e)
            _kpoint_incomplete(
                res, cfg, e,
                f'k={k}: evaluating the slices of the k-point cumulant '
                f'failed ({e!r}); result["C_tau"] is None (no curve). '
                f'result["total_C"] is the cumulant as a function of the '
                f'{k} external times.')

        if cfg.kpoint_full_grid and res.get('C_tau') is not None:
            try:
                # Optional: the FULL (k−1)-dim tensor C(τ_1..τ_{k−1}).
                # Downsample the axis so the total n^{k−1} evaluations stay
                # bounded.
                import itertools
                gtau = _kplan['grid_tau']

                def _grid(fn):
                    out = np.empty((gtau.size,) * (k - 1), dtype=complex)
                    for idx in itertools.product(range(gtau.size),
                                                 repeat=k - 1):
                        out[idx] = complex(fn(*_args([gtau[m] for m in idx])))
                    return out

                res['C_tau_grid'] = _grid(res['total_C'])
                res['C_tau_grid_by_ell'] = {e: _grid(f)
                                            for e, f in tcb.items()}
                res['tau_axes'] = [gtau] * (k - 1)
                if gtau.size < tau.size:
                    res['_kpoint_grid_note'] = (
                        f'τ downsampled to {gtau.size} pts/axis '
                        f'({gtau.size ** (k - 1)} evals) to bound grid cost')
            except Exception as e:
                res['_kpoint_grid_error'] = repr(e)
                _kpoint_incomplete(
                    res, cfg, e,
                    f'k={k}: evaluating the full k-point grid '
                    f'(kpoint_full_grid) failed ({e!r}); the slices '
                    f'(result["C_tau"]) are complete, result["C_tau_grid"] '
                    f'is missing.')

    # Optional output conversion: assemble the full / central k-point MOMENT
    # from the cumulants (the set-partition / cluster expansion).
    # compute_cumulants stays pure — this is post-processing on top.
    out_kind = getattr(cfg, 'output', 'cumulant') or 'cumulant'
    if out_kind in ('moment', 'central_moment') and res.get(
            '_kpoint_slice_error') is not None:
        # the cumulant blocks could not be evaluated (reported above)
        res['moment'] = None
        res['output_kind'] = out_kind
    elif out_kind in ('moment', 'central_moment'):
        central = out_kind == 'central_moment'
        if is_spatial(model):
            if k == 1:                      # M₁ = ⟨φ⟩: the raw first moment
                # The 1-point MOMENT is the mean itself (the mean-field
                # saddle φ*); its CENTRAL counterpart ⟨φ−⟨φ⟩⟩ is 0 by
                # definition.  (Without this special-case k==1 fell into
                # the k≥3 branch and raised a misleading "spatial k≥3 not
                # implemented" error.)
                res['moment'] = 0.0 if central else _external_mean(model, res)
            elif k == 2:                    # M(x) = κ₂(x) [+ μ_a μ_b for raw]
                mu_ab = 0.0
                if not central:
                    leg_a, leg_b = (tuple(e) if isinstance(e, (tuple, list))
                                    else e for e in ext)
                    mu_a = _external_mean(model, res, leg_a)
                    mu_b = _external_mean(model, res, leg_b)
                    # The spatial integrator has no 1-point tadpoles, so the
                    # loop corrections of the means are unavailable: with a
                    # loop budget L the raw moment needs μ_a⁽ℓa⁾ μ_b⁽ℓb⁾ for
                    # ℓa + ℓb ≤ L, which involves them as soon as L ≥ 1 with
                    # a nonzero saddle, or L ≥ 2.
                    # (refused up front by _refuse_spatial_raw_moment; this
                    # is the same rule on the finished result)
                    if max_ell >= 2 or (max_ell >= 1 and (mu_a or mu_b)):
                        raise NotImplementedError(
                            _spatial_raw_moment_message(max_ell, mu_a, mu_b))
                    mu_ab = mu_a * mu_b
                    res['moment_means'] = {leg_a: [mu_a], leg_b: [mu_b]}
                res['moment'] = (np.real(np.asarray(res['C_tau_x'])) + mu_ab)
            else:
                raise NotImplementedError(
                    f"Config.output={out_kind!r} for spatial k≥3 is not "
                    "implemented yet (temporal any-k and spatial k=2 are).")
        else:
            res['moment'] = _assemble_moment_temporal(
                model, res, kw, k, central)
        res['output_kind'] = out_kind

    # D4: the analyzer's one-table report (``result['analysis'].table()``)
    if cfg.verbose and isinstance(res, dict):
        _rep = res.get('analysis')
        if hasattr(_rep, 'table'):
            print(_rep.table())
    res['_cfg'] = cfg
    res['_model'] = model
    res['_resolved'] = dict(k=k, max_ell=max_ell, external_fields=ext,
                            parameters=fundamental, fundamental=fundamental)
    return res


# ── Per-loop-order decomposition ─────────────────────────────────────────────

_ORDER_COLORS = ['#3F00FF', '#1F9FCC', '#16A085', '#E67E22', '#E74C3C',
                 '#8E44AD', '#7F8C8D']


def _run_resummed(model, cfg, kw, k, max_ell, ext, fundamental, opts, cov):
    """``dd.run`` with ``Config.resum``: the Hartree closure or the screened
    series, on the τ grid :func:`run` resolved."""
    from api.hartree import refusal_reason, HartreeUnsupported
    why = refusal_reason(model)
    if why is not None:
        raise HartreeUnsupported(f'Config.resum: {why}.')
    from api.resum import check_config
    check_config(opts, k=k, max_ell=max_ell, output=cfg.output,
                 t_min=cfg.t_min, spatial_points=cfg.spatial_points,
                 covariance=cov)
    tau = kw.get('tau_grid')
    if tau is None:
        tau = np.arange(-kw['tau_max'], kw['tau_max'] + kw['tau_step'] * 0.5,
                        kw['tau_step'])
    tau = np.asarray(tau, dtype=float)
    mf_kw = {key: kw[key] for key in ('fixed_point_index', 'mf_dae_n_starts',
                                      'mf_dae_seed_box') if key in kw}
    spec = kw.get('spectra')
    if spec is not None:
        from api.spectra import resolve_spectra_options
        spec = resolve_spectra_options(spec, 'Config.spectra')
    if opts['method'] == 'hartree':
        from api.hartree import hartree, run_result
        h = hartree(model, fundamental, taylor_order=opts.get('taylor_order'),
                    **mf_kw)
        res = run_result(h, ext, k, tau, spectra=spec, covariance=cov)
    else:
        from api.resum import screened
        run_kw = {key: kw[key] for key in (
            'use_cache', 'parallel', 'n_workers', 'policy', 'preflight',
            'backend', 'accuracy', 'taylor_order') if key in kw}
        res = screened(model, max_ell, fundamental, external_fields=ext,
                       options={key: v for key, v in opts.items()
                                if key != 'method'},
                       tau_grid=tau, spectra=spec, covariance=cov,
                       verbose=cfg.verbose, **mf_kw, **run_kw)
    res['_cfg'] = cfg
    res['_model'] = model
    res['_resolved'] = dict(k=k, max_ell=max_ell, external_fields=ext,
                            parameters=fundamental, fundamental=fundamental,
                            resum=opts['method'])
    return res


def _order_label(ell: int) -> str:
    if ell == 0:
        return 'tree'
    return 'tree+' + '+'.join(f'{e}loop' for e in range(1, ell + 1))


def cumulative_curves(by_ell: dict) -> dict:
    """``{ell: array}`` per-order → cumulative ``{ell: Σ_{0..ell}}``."""
    out, running = {}, None
    for ell in sorted(by_ell):
        v = np.real(np.asarray(by_ell[ell]))
        running = v.copy() if running is None else running + v
        out[ell] = running.copy()
    return out


# ── Plotting (adaptable, auto-dispatched) ────────────────────────────────────

def plot_cumulant(result: dict, cfg: Config = None, model: dict = None,
                  sim: dict = None, residual: bool = False,
                  show_delta: bool = False, delta_style: str = 'arrow'):
    """Plot the cumulant in the form natural to the model's group.

    Dispatch:
      * k=1 (any group)     → :func:`plot_temporal_mean` (the 1-point mean
                              ⟨φ⟩; tree level = the mean-field saddle φ*)
      * spatial k≥3 events  → :func:`plot_kpoint`
      * spatial k=2         → :func:`plot_spatial`
      * temporal            → :func:`plot_temporal`

    ``sim`` (optional) overlays a matched simulator: a dict with
    ``tau``/``C``/``C_err`` (temporal) or ``x``/``C``/``C_err`` (spatial); for
    k=1 a scalar mean ``{'C': value, 'C_err': err}``.

    ``residual=True`` (temporal C(τ) with a ``sim`` and at least one loop order)
    adds a companion panel plotting C(τ) − C_tree(τ): the per-order loop
    contributions (1-loop, 1+2-loop, …) against the simulated residual
    sim − tree, so a loop that is a tiny fraction of the tree becomes the whole
    signal.  Ignored where it does not apply.
    ``show_delta=True`` (temporal cumulants, k = 2 or 3) draws, instead of
    the plot above, the cumulant with its shot-noise δ components
    (:func:`api.cumulant_plot.plot_cumulant_delta`; ``delta_style`` 'arrow'
    or 'bin'); not with ``residual`` or a moment output.
    Returns the Matplotlib ``Figure``.
    """
    cfg = cfg or result.get('_cfg') or Config()
    model = model or result.get('_model') or {}
    if result.get('t_min') is not None:
        raise NotImplementedError(
            'plot_cumulant: switch-on results (t_min) are functions of the '
            "absolute times; plot result['C_times'] against result['times']")
    if show_delta:
        if residual or result.get('output_kind') in ('moment',
                                                     'central_moment'):
            raise ValueError('show_delta draws the cumulant with its δ '
                             'components: not with residual=True or a '
                             'moment output')
        from api.cumulant_plot import plot_cumulant_delta
        return plot_cumulant_delta(result, cfg, model, sim, delta_style)
    if result.get('output_kind') in ('moment', 'central_moment') \
            and result.get('moment') is not None:
        return _plot_moment(result, cfg, model, sim)
    # k=1 is a 1-POINT function: the mean ⟨φ⟩ of the external field.  Its TREE
    # level is the mean-field saddle φ*; ℓ≥1 add the loop tadpole shifts.  There
    # is no C(τ)/C(χ) curve, so draw it as tree/+loop bars (works for both
    # temporal and the spatial mean-field, which is x-independent).
    if _result_meta(result, 'k') == 1:
        return plot_temporal_mean(result, cfg, model, sim)
    if result.get('C_kpoint_grid') is not None:      # k≥3 spatial grid → slice heatmap
        return plot_kpoint_grid(result, cfg, model, sim)
    if 'C_kpoint' in result:
        return plot_kpoint(result, cfg, model)
    if is_spatial(model) or 'C_tau_x' in result:
        return plot_spatial(result, cfg, model, sim)
    # Temporal k≥3: there is no C(τ) curve — the connected k-point cumulant is
    # the equal-time scalar per loop order (compute_cumulants returns
    # ``C_tau=None`` and a callable ``total_C``).  Draw it as per-order bars.
    if result.get('C_tau') is None:
        return plot_temporal_kpoint(result, cfg, model, sim)
    if result.get('C_tau_grid') is not None:         # k≥3 full grid → heatmap
        return plot_temporal_kpoint_grid(result, cfg, model, sim)
    if len(result.get('C_tau_slices') or {}) >= 2:   # k≥3: one panel per τ_j
        return plot_temporal_kpoint_slices(result, cfg, model, sim)
    return plot_temporal(result, cfg, model, sim, residual=residual)


def _plot_moment(result, cfg, model, sim=None):
    """Plot the assembled full / central k-point MOMENT (``Config.output`` ≠
    'cumulant').  Temporal → M_k(τ) along the slice; spatial k=2 → M(x,0).
    A single curve (the moment mixes loop orders, so no per-order overlay)."""
    cfg = cfg or Config()
    kind = result.get('output_kind', 'moment')
    k = _result_meta(result, 'k', '?')
    lab = 'central moment' if kind == 'central_moment' else 'raw moment'
    M = np.real(np.asarray(result['moment']))
    fig, ax = plt.subplots(figsize=cfg.figsize or (7.5, 4.6))
    if is_spatial(model) or 'C_tau_x' in result:
        xs = np.asarray(result['spatial_grid'])
        tau = np.asarray(result.get('tau_grid', [0.0]))
        m = M[int(np.argmin(np.abs(tau)))] if M.ndim == 2 else M
        ax.plot(xs, m, '-', lw=1.8, color='#8E44AD',
                label=f'theory: {lab}')
        ax.set_xlabel(r'$\chi$')
    else:
        ax.plot(np.asarray(result['tau_grid']), M, '-', lw=1.8,
                color='#8E44AD', label=f'theory: {lab}')
        ax.set_xlabel(r'$\tau$')
    if sim is not None and sim.get('C') is not None:
        sx = np.asarray(sim.get('tau', sim.get('x')))
        se = sim.get('C_err')
        if se is not None:
            ax.errorbar(sx, np.asarray(sim['C']), yerr=np.asarray(se), fmt='o',
                        ms=3, color='#222', alpha=0.6, capsize=2, label='sim')
        else:
            ax.plot(sx, np.asarray(sim['C']), 'o', ms=3, color='#222',
                    alpha=0.6, label='sim')
    ax.set_ylabel(r'$M_{%s}$' % k)
    if cfg.logy:
        ax.set_yscale('log')
    ax.set_title(cfg.title or f"{model.get('name','')}: {lab} (k={k})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def _orders_to_draw(by_ell: dict, cfg: Config):
    """Resolve ``Config.show_orders`` into a list of (label, curve)."""
    if not by_ell:
        return []
    if cfg.show_orders == 'incremental':
        return [(f'{ell}-loop' if ell else 'tree',
                 np.real(np.asarray(by_ell[ell]))) for ell in sorted(by_ell)]
    cum = cumulative_curves(by_ell)
    if cfg.show_orders == 'total':
        top = max(cum)
        return [(_order_label(top), cum[top])]
    return [(_order_label(ell), cum[ell]) for ell in sorted(cum)]


def plot_temporal(result, cfg, model, sim=None, residual=False):
    """C(τ) with a per-loop-order overlay (the temporal group plot).

    ``residual=True`` (with a ``sim`` and at least one loop order present) adds a
    companion panel on the right plotting C(τ) − C_tree(τ): the cumulative loop
    contributions (1-loop, 1+2-loop, …) against the simulated residual
    sim − tree.  A loop that is only a few percent of the tree — and so invisible
    where the curves overlap on the left — becomes the whole signal on the right.
    """
    cfg = cfg or Config()
    if result.get('C_tau') is None:          # temporal k≥3 → scalar k-point
        return plot_temporal_kpoint(result, cfg, model, sim)
    tau = np.asarray(result['tau_grid'])
    by_ell = {e: v for e, v in (result.get('C_tau_by_ell') or {}).items()
              if v is not None}

    def _sim_curve():
        """sim (C, C_err) as 1-D arrays on sim['tau'] (collapse a k≥3 slice)."""
        st = np.asarray(sim['tau']); sc = np.asarray(sim['C'])
        se = sim.get('C_err'); se = np.asarray(se) if se is not None else None
        if sc.ndim == 2:                  # a k≥3 multi-slice sim → show slice 1
            sc = sc[0]
            se = se[0] if se is not None else None
        return st, sc, se

    # The residual companion needs a tree (ℓ=0), ≥1 loop order, and a sim.
    want_resid = bool(residual and sim is not None and 0 in by_ell
                      and any(e >= 1 for e in by_ell))
    if want_resid:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=cfg.figsize or (12.0, 4.6))
    else:
        fig, ax = plt.subplots(figsize=cfg.figsize or (7.5, 4.6))

    # ── left panel: the usual per-order overlay ──────────────────────────────
    curves = _orders_to_draw(by_ell, cfg)
    if not curves:
        curves = [('total', np.real(np.asarray(result['C_tau'])))]
    for i, (lab, c) in enumerate(curves):
        ax.plot(tau, c, '-', lw=1.8,
                color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                label=f'theory: {lab}')
    if sim is not None:
        st, sc, se = _sim_curve()
        if se is not None:
            ax.errorbar(st, sc, yerr=se, fmt='o', ms=3,
                        color='#222', alpha=0.6, capsize=2, label='sim')
        else:
            ax.plot(st, sc, 'o', ms=3, color='#222', alpha=0.6, label='sim')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$C(\tau)$')
    if cfg.logy:
        ax.set_yscale('log')
    ax.set_title(cfg.title or f"{model.get('name','')}: C(τ)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # ── right panel: residual C(τ) − C_tree(τ), loop contributions isolated ──
    if want_resid:
        tree = np.real(np.asarray(by_ell[0]))
        cum = cumulative_curves(by_ell)
        ax2.axhline(0.0, color='gray', lw=0.8)
        for ell in [e for e in sorted(cum) if e >= 1]:
            lab = '+'.join(str(e) for e in range(1, ell + 1)) + '-loop'
            ax2.plot(tau, np.real(cum[ell]) - tree, '-', lw=1.8,
                     color=_ORDER_COLORS[ell % len(_ORDER_COLORS)],
                     label=f'theory: {lab}')
        st, sc, se = _sim_curve()
        sc_i = np.interp(tau, st, sc)         # sim onto the model τ-grid
        se_i = np.interp(tau, st, se) if se is not None else None
        if se_i is not None:
            ax2.errorbar(tau, sc_i - tree, yerr=se_i, fmt='o', ms=3,
                         color='#222', alpha=0.6, capsize=2, label='sim − tree')
        else:
            ax2.plot(tau, sc_i - tree, 'o', ms=3, color='#222', alpha=0.6,
                     label='sim − tree')
        ax2.set_xlabel(r'$\tau$')
        ax2.set_ylabel(r'$C(\tau) - C_{\mathrm{tree}}(\tau)$')
        ax2.set_title('residual: loop contributions isolated')
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8)

    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_temporal_kpoint_slices(result, cfg, model, sim=None):
    """k≥3: the connected k-point cumulant has k−1 independent time differences
    τ_j = t_j − t_0.  Draw one panel per difference — slice j sweeps leg j over
    the τ-grid with the other legs pinned at their base lags (default τ=0) —
    each with the per-loop-order overlay and (optionally) the matching
    simulator slice.  A point where the swept leg meets another leg (τ = 0
    or a base lag) is the left limit of its curve, evaluated at the exact tie
    (``run``, ``_kpoint_slice_point``).

    For a single symmetric field the k−1 slices coincide (a visible check of the
    cumulant's permutation symmetry); when the external legs are different
    fields they genuinely differ.

    ``sim['C']`` / ``sim['C_err']`` may be 2-D ``(k−1, n_tau)`` — row p is the
    sim for panel p — or 1-D (applied to the first panel only)."""
    cfg = cfg or Config()
    tau = np.asarray(result['tau_grid'])
    slices = result.get('C_tau_slices') or {}
    sl_by_ell = result.get('C_tau_slices_by_ell') or {}
    js = sorted(slices)
    k = len(js) + 1
    fig, axes = plt.subplots(1, len(js),
                             figsize=cfg.figsize or (4.4 * len(js), 4.3),
                             squeeze=False)
    axes = axes[0]
    sim_t = np.asarray(sim['tau']) if sim is not None else None
    sim_C = np.asarray(sim['C']) if sim is not None else None
    sim_E = (np.asarray(sim['C_err'])
             if (sim is not None and sim.get('C_err') is not None) else None)
    for p, j in enumerate(js):
        ax = axes[p]
        by_ell = {e: v for e, v in (sl_by_ell.get(j) or {}).items()
                  if v is not None}
        curves = _orders_to_draw(by_ell, cfg) or \
            [('total', np.real(np.asarray(slices[j])))]
        for i, (lab, c) in enumerate(curves):
            ax.plot(tau, c, '-', lw=1.8,
                    color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                    label=f'theory: {lab}')
        if sim_C is not None:
            row = sim_C[p] if sim_C.ndim == 2 else (sim_C if p == 0 else None)
            err = None
            if sim_E is not None:
                err = sim_E[p] if sim_E.ndim == 2 else (sim_E if p == 0 else None)
            if row is not None:
                if err is not None:
                    ax.errorbar(sim_t, row, yerr=err, fmt='o', ms=3,
                                color='#222', alpha=0.6, capsize=2, label='sim')
                else:
                    ax.plot(sim_t, row, 'o', ms=3, color='#222', alpha=0.6,
                            label='sim')
        ax.set_xlabel(r'$\tau_{%d}=t_{%d}-t_0$' % (j, j))
        ax.set_ylabel(r'$\kappa_{%d}$' % k)
        ax.set_title(r'slice %d: sweep leg %d' % (j, j))
        if cfg.logy:
            ax.set_yscale('log')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    base = result.get('_kpoint_base')
    base_s = (f'  (others fixed at base={base})'
              if base is not None and any(abs(b) > 1e-12 for b in base) else '')
    fig.suptitle(cfg.title or
                 f"{model.get('name', '')}: k={k} cumulant, "
                 f"{len(js)} slices{base_s}")
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_temporal_kpoint_grid(result, cfg, model, sim=None):
    """k=3: heatmap of the full 2-D cumulant κ_3(τ_1, τ_2) (the whole grid of
    the two independent time differences).  For k≥4 there is no 2-D view, so
    fall back to the k−1 axis-parallel slices."""
    cfg = cfg or Config()
    axes_ = result.get('tau_axes') or []
    grid = result.get('C_tau_grid')
    if grid is None or len(axes_) != 2:              # k≥4 (or no grid)
        return plot_temporal_kpoint_slices(result, cfg, model, sim)
    t1, t2 = np.asarray(axes_[0]), np.asarray(axes_[1])
    Z = np.real(np.asarray(grid))                    # (τ_1, τ_2)
    fig, ax = plt.subplots(figsize=cfg.figsize or (6.4, 5.0))
    im = ax.pcolormesh(t1, t2, Z.T, shading='auto', cmap='RdBu_r')
    fig.colorbar(im, ax=ax, label=r'$\kappa_3(\tau_1,\tau_2)$')
    ax.set_xlabel(r'$\tau_1 = t_1 - t_0$')
    ax.set_ylabel(r'$\tau_2 = t_2 - t_0$')
    note = result.get('_kpoint_grid_note')
    ax.set_title((cfg.title or f"{model.get('name', '')}: k=3 cumulant grid")
                 + (f'\n{note}' if note else ''))
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_temporal_kpoint(result, cfg, model, sim=None):
    """Temporal k≥3: the equal-time connected k-point cumulant is a single
    scalar per loop order (``compute_cumulants`` returns ``C_tau=None`` and a
    callable ``total_C``; the τ=0 value lives in ``C_tau_by_ell``).  Draw the
    tree / +loop bars, honouring ``cfg.show_orders``; overlay a scalar sim
    value if given as ``sim={'C': <scalar>, 'C_err': <scalar or None>}``."""
    cfg = cfg or Config()
    by_ell = {e: float(np.real(np.asarray(v)))
              for e, v in (result.get('C_tau_by_ell') or {}).items()
              if v is not None}
    fig, ax = plt.subplots(figsize=cfg.figsize or (6.0, 4.3))
    if by_ell:
        if cfg.show_orders == 'incremental':
            labs = [('%d-loop' % e if e else 'tree', by_ell[e])
                    for e in sorted(by_ell)]
        else:
            cum, run = {}, 0.0
            for e in sorted(by_ell):
                run += by_ell[e]
                cum[e] = run
            if cfg.show_orders == 'total':
                top = max(cum)
                labs = [(_order_label(top), cum[top])]
            else:
                labs = [(_order_label(e), cum[e]) for e in sorted(cum)]
        idx = np.arange(len(labs))
        ax.bar(idx, [v for _, v in labs], width=0.6,
               color=[_ORDER_COLORS[i % len(_ORDER_COLORS)]
                      for i in range(len(labs))])
        ax.set_xticks(idx)
        ax.set_xticklabels([l for l, _ in labs], fontsize=8,
                           rotation=15, ha='right')
    if sim is not None and sim.get('C') is not None:
        sv = float(np.real(np.asarray(sim['C']).ravel()[0]))
        se = sim.get('C_err')
        ax.axhline(sv, color='#222', lw=1.5, ls='--', label='sim')
        if se is not None:
            se = float(np.asarray(se).ravel()[0])
            ax.axhspan(sv - se, sv + se, color='#222', alpha=0.12)
        ax.legend(fontsize=8)
    k = _result_meta(result, 'k', '?')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel(r'$\kappa_{%s}$ (equal-time)' % k)
    ax.set_title(cfg.title or f"{model.get('name','')}: "
                 f"k={k} cumulant at τ=0")
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_temporal_mean(result, cfg, model, sim=None):
    """k=1: the 1-point cumulant is the MEAN ⟨φ⟩ of the external field — a single
    τ-independent number.

    Its TREE level is the **mean-field saddle φ\\*** (the mean-field solution);
    the ℓ≥1 orders add the loop tadpole shifts, so the cumulative tree+loops is
    the perturbative ⟨φ⟩ that the simulation mean should match.  (The pipeline
    expands around the saddle, so the *fluctuation* tree 1-point is 0 — the
    physical tree value is φ*, injected here from the mean-field data.)  Drawn
    as tree / +loop bars (``Config.show_orders`` honoured); a scalar simulation
    mean ``sim={'C': value, 'C_err': err}`` overlays as a dashed line.  Works
    for temporal models and the spatial mean-field (x-independent; loop tadpoles
    for the spatial 1-point are not yet assembled, so spatial shows tree only).
    There is no C(τ) curve to draw — for a correlation use k≥2."""
    cfg = cfg or Config()
    tau = np.asarray(result.get('tau_grid', [0.0]))
    i0 = int(np.argmin(np.abs(tau))) if tau.size else 0
    by_ell = {}
    for e, v in (result.get('C_tau_by_ell') or {}).items():
        if v is None:
            continue
        arr = np.real(np.asarray(v)).ravel()
        by_ell[e] = float(arr[i0] if arr.size > i0 else arr[0])
    # TREE level of the 1-point function IS the mean-field saddle φ*: the
    # fluctuation's tree 1-point is 0 (saddle condition), so anchor tree at φ*
    # and let ℓ≥1 stack the tadpole shifts on top.  ⟨φ⟩ = φ* + Σ_{ℓ≥1} tadpole.
    mf_star = _external_mean(model, result)
    by_ell[0] = mf_star + by_ell.get(0, 0.0)
    fig, ax = plt.subplots(figsize=cfg.figsize or (6.0, 4.3))
    if by_ell:
        if cfg.show_orders == 'incremental':
            labs = [('%d-loop' % e if e else 'tree', by_ell[e])
                    for e in sorted(by_ell)]
        else:
            cum, run = {}, 0.0
            for e in sorted(by_ell):
                run += by_ell[e]
                cum[e] = run
            if cfg.show_orders == 'total':
                top = max(cum)
                labs = [(_order_label(top), cum[top])]
            else:
                labs = [(_order_label(e), cum[e]) for e in sorted(cum)]
        idx = np.arange(len(labs))
        ax.bar(idx, [v for _, v in labs], width=0.6,
               color=[_ORDER_COLORS[i % len(_ORDER_COLORS)]
                      for i in range(len(labs))])
        ax.set_xticks(idx)
        ax.set_xticklabels([l for l, _ in labs], fontsize=8, rotation=15, ha='right')
    if sim is not None and sim.get('C') is not None:
        sv = float(np.real(np.asarray(sim['C']).ravel()[0]))
        ax.axhline(sv, color='#222', lw=1.5, ls='--', label='sim')
        se = sim.get('C_err')
        if se is not None:
            se = float(np.asarray(se).ravel()[0])
            ax.axhspan(sv - se, sv + se, color='#222', alpha=0.12)
        ax.legend(fontsize=8)
    fld = _result_meta(result, 'external_fields') or [('phi', 1)]
    fb = fld[0][0] if isinstance(fld[0], (tuple, list)) else str(fld[0])
    from api.naming import field_natural_name
    base = field_natural_name(model, fb)                # the physical field
    ax.axhline(0, color='gray', lw=0.5)
    # Mark the mean-field tree value φ* so the loop shift off it is visible
    # (especially under show_orders='total', where the tree bar is hidden).
    if abs(mf_star) > 1e-12:
        ax.axhline(mf_star, color='#3F00FF', lw=1.0, ls=':',
                   label=r'mean-field $%s^\ast=%.4g$' % (base, mf_star))
    ax.set_ylabel(r'$\langle\,%s\,\rangle$  (1-point mean; tree $=%s^\ast$)'
                  % (base, base))
    ax.set_title(cfg.title or f"{model.get('name','')}: 1-point mean (k=1)")
    if abs(mf_star) > 1e-12 or (sim is not None and sim.get('C') is not None):
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_spatial(result, cfg, model, sim=None):
    """Spatial correlator C(χ,τ).

    Always shows the **equal-time C(χ,0)** χ-slice (per-loop overlay).  When a τ
    grid is present it ALSO shows the **temporal C(χ₀,τ) at one fixed χ** with the
    same per-loop (tree / tree+1loop…) overlay — the τ-sweep counterpart of the
    equal-time panel (χ₀ defaults to the χ closest to 0; ``Config.tau_slice_chi``
    picks another), the **model C(χ,τ) heatmap** (which contains *every* loop
    order in the run — the
    title states the order), and — when ``sim`` carries a 2-D correlator
    (``sim={'x','tau','C'}`` with a 2-D ``C(τ,χ)``) — the matching **simulation
    heatmap on a shared colour scale**.  A 1-D ``sim={'x','C'[,'C_err']}`` still
    overlays on the equal-time panel."""
    cfg = cfg or Config()
    if spatial_dim(model) >= 2:                     # d≥2: χ is a VECTOR → C(τ, χ¹, χ²)
        return _plot_spatial_2d(result, cfg, model, sim)
    C = np.real(np.asarray(result['C_tau_x']))     # (n_tau, n_x) — total, all loops
    xs = np.asarray(result['spatial_grid'])
    tau = np.asarray(result['tau_grid'])
    i0 = int(np.argmin(np.abs(tau)))
    si = result.get('spatial_info') or {}
    by_order = si.get('C_by_order')                # {ell: (n_tau, n_x)} cumulative
    ell_top = (max(by_order) if by_order
               else (_result_meta(result, 'max_ell', 0) or 0))
    loop_lab = _order_label(ell_top)               # e.g. 'tree+1loop'
    sim2d = (sim is not None and sim.get('tau') is not None
             and np.ndim(np.asarray(sim.get('C'))) == 2)
    has_tau = tau.size > 1

    if not has_tau:                                # equal-time only → single panel
        fig, axA = plt.subplots(figsize=cfg.figsize or (6.0, 4.3))
        axB = axH = axS = None
    elif sim2d:                                    # slices + model & sim heatmaps
        fig, ax = plt.subplots(2, 2, figsize=cfg.figsize or (11.5, 8.2))
        axA, axB, axH, axS = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    else:                                          # slices + model heatmap
        fig, ax = plt.subplots(1, 3, figsize=cfg.figsize or (15.0, 4.4))
        axA, axB, axH = ax
        axS = None

    # ── Panel A: equal-time C(χ, 0) — χ-slice at fixed τ=0 ──
    if by_order and cfg.show_orders != 'total':
        for i, ell in enumerate(sorted(by_order)):
            c = np.real(np.asarray(by_order[ell]))
            c = c[i0] if c.ndim == 2 else c
            axA.plot(xs, c, '-', lw=1.8,
                     color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                     label=f'theory: {_order_label(ell)}')
    else:
        axA.plot(xs, C[i0], '-', lw=1.8, color='#1F9FCC',
                 label=f'theory: {loop_lab}')
    if sim2d:
        st = np.asarray(sim['tau']); sj0 = int(np.argmin(np.abs(st)))
        axA.plot(np.asarray(sim['x']), np.real(np.asarray(sim['C']))[sj0],
                 'o', ms=3, color='#222', alpha=0.6, label='sim')
    elif sim is not None and sim.get('C') is not None:
        sx, sc = np.asarray(sim['x']), np.real(np.asarray(sim['C']))
        se = sim.get('C_err')
        if se is not None:
            axA.errorbar(sx, sc, yerr=np.asarray(se), fmt='o', ms=3,
                         color='#222', alpha=0.6, capsize=2, label='sim')
        else:
            axA.plot(sx, sc, 'o', ms=3, color='#222', alpha=0.6, label='sim')
    axA.set_xlabel(r'$\chi$'); axA.set_ylabel(r'$C(\chi,\,0)$')
    if cfg.logy:
        axA.set_yscale('log')
    axA.set_title(cfg.title or f"{model.get('name','')}: equal-time C(χ,0)")
    axA.grid(alpha=0.25); axA.legend(fontsize=8)

    # ── Panel B: temporal C(χ₀, τ) at ONE fixed χ — per-loop overlay (mirrors Panel A) ──
    # τ-sweep counterpart of A: same tree / tree+1loop decomposition, at a single
    # χ₀ (default the χ closest to 0 — the on-site temporal decay; set
    # ``Config.tau_slice_chi`` to pick another separation).
    if has_tau:
        j0 = (int(np.argmin(np.abs(xs - float(cfg.tau_slice_chi))))
              if cfg.tau_slice_chi is not None else int(np.argmin(np.abs(xs))))
        chi0 = float(xs[j0])
        if by_order and cfg.show_orders != 'total':
            for i, ell in enumerate(sorted(by_order)):
                c = np.real(np.asarray(by_order[ell]))
                cc = c[:, j0] if c.ndim == 2 else c
                axB.plot(tau, cc, '-', lw=1.8,
                         color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                         label=f'theory: {_order_label(ell)}')
        else:
            axB.plot(tau, C[:, j0], '-', lw=1.8, color='#1F9FCC',
                     label=f'theory: {loop_lab}')
        if sim2d:
            sxa = np.asarray(sim['x']); sj = int(np.argmin(np.abs(sxa - chi0)))
            axB.plot(np.asarray(sim['tau']),
                     np.real(np.asarray(sim['C']))[:, sj],
                     'o', ms=3, color='#222', alpha=0.6, label='sim')
        axB.set_xlabel(r'$\tau$'); axB.set_ylabel(r'$C(\chi_0,\,\tau)$')
        if cfg.logy:
            axB.set_yscale('log')
        axB.set_title(r'$C(\chi_0,\tau)$ at $\chi_0=%.2g$' % chi0)
        axB.grid(alpha=0.25); axB.legend(fontsize=8)

    # ── model (+ sim) C(χ,τ) heatmaps, on a shared colour scale ──
    if has_tau:
        vmin, vmax = float(np.min(C)), float(np.max(C))
        im = axH.imshow(C, aspect='auto', origin='lower',
                        extent=[xs.min(), xs.max(), tau.min(), tau.max()],
                        cmap='viridis', vmin=vmin, vmax=vmax)
        axH.set_xlabel(r'$\chi$'); axH.set_ylabel(r'$\tau$')
        axH.set_title(r'theory $C(\chi,\tau)$  [%s]' % loop_lab)
        fig.colorbar(im, ax=axH, fraction=0.046, pad=0.04)
        if sim2d:
            sxa, sta = np.asarray(sim['x']), np.asarray(sim['tau'])
            sC = np.real(np.asarray(sim['C']))
            im2 = axS.imshow(sC, aspect='auto', origin='lower',
                             extent=[sxa.min(), sxa.max(), sta.min(), sta.max()],
                             cmap='viridis', vmin=vmin, vmax=vmax)
            axS.set_xlabel(r'$\chi$'); axS.set_ylabel(r'$\tau$')
            axS.set_title(r'sim $C(\chi,\tau)$  (shared scale)')
            fig.colorbar(im2, ax=axS, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def _plot_spatial_2d(result, cfg, model, sim=None):
    """Spatial correlator in d≥2: the separation χ is a VECTOR, so the k=2
    correlator depends on (τ, χ¹, χ²).  The pipeline computes the isotropic
    radial C(r,τ); this presents it as the three axis SLICES (χ¹, χ², τ — each
    with the tree / tree+1loop per-order overlay) and the three pairwise
    HEATMAPS over {χ¹, χ², τ}, via C(χ¹,χ²,τ) = C_rad(√((χ¹)²+(χ²)²), τ)
    (rotational invariance, so the χ¹ and χ² views coincide — the visual
    signature of isotropy).  A ``sim`` dict ``{'chi','tau','C'}`` with a 3-D
    ``C[i,a,b]`` = C(χ¹=chi[a], χ²=chi[b], τ=tau[i]) adds sim points on the
    slices and a matching row of sim heatmaps on the shared colour scale."""
    cfg = cfg or Config()
    r   = np.asarray(result['spatial_grid'], float)        # radial separations ≥ 0
    tau = np.asarray(result['tau_grid'], float)
    it0 = int(np.argmin(np.abs(tau)))                       # τ = 0 row
    si  = result.get('spatial_info') or {}
    by  = si.get('C_by_order') or {}
    ell_top = max(by) if by else (_result_meta(result, 'max_ell', 0) or 0)
    loop_lab = _order_label(ell_top)
    full = np.real(np.asarray(result['C_tau_x']))          # (n_tau, n_r) total
    R = float(r.max())
    has_tau = tau.size > 1

    def rad(arr2d, rq, it):                                 # radial interpolant at lag-index it
        return np.interp(np.clip(np.abs(rq), float(r.min()), R), r, np.real(arr2d)[it])

    orders = sorted(by) if (by and cfg.show_orders != 'total') else [None]

    def overlay(ax, xvals, fn, simxy=None):                # per-order theory curves (+ sim dots)
        for i, o in enumerate(orders):
            arr = np.real(np.asarray(by[o])) if o is not None else full
            ax.plot(xvals, fn(arr), '-', lw=1.7,
                    color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                    label=('theory: %s' % _order_label(o)) if o is not None else 'theory')
        if simxy is not None:
            ax.plot(simxy[0], simxy[1], 'o', ms=3, color='#222', alpha=.6, label='sim')
        ax.grid(alpha=.25); ax.legend(fontsize=8)

    chi = np.linspace(-R, R, 121)
    name = model.get('name', '')
    Xb = R / np.sqrt(2.0); cb = np.linspace(-Xb, Xb, 81)
    CX, CY = np.meshgrid(cb, cb)
    M_xy = rad(full, np.sqrt(CX**2 + CY**2), it0)           # model C(χ¹,χ²,0)
    M_xt = (np.array([rad(full, chi, it) for it in range(tau.size)])
            if has_tau else None)                          # model C(χ¹,0,τ)

    have_sim = (sim is not None and sim.get('C') is not None
                and np.ndim(np.asarray(sim['C'])) == 3 and has_tau)
    if have_sim:
        sC = np.real(np.asarray(sim['C'])); sx = np.asarray(sim['chi'], float)
        st = np.asarray(sim['tau'], float)
        a0 = int(np.argmin(np.abs(sx))); is0 = int(np.argmin(np.abs(st)))
        km = np.abs(sx) <= R; sxm = sx[km]
        if not km.any():               # no sim separation inside the model radius
            import warnings as _w
            _w.warn(f'sim correlator: no separation within the model radius '
                    f'(max |χ|={R:g}); skipping the sim overlay.', stacklevel=2)
            have_sim = False

    if not has_tau:                                         # equal-time only → 1×3
        fig, ax = plt.subplots(1, 3, figsize=cfg.figsize or (15.0, 4.4))
        overlay(ax[0], chi, lambda a: rad(a, chi, it0))
        ax[0].set_xlabel(r'$\chi^1$'); ax[0].set_ylabel(r'$C(\chi^1,0,0)$')
        ax[0].set_title(r'slice $\chi^1$  ($\chi^2{=}0$)')
        overlay(ax[1], chi, lambda a: rad(a, chi, it0))
        ax[1].set_xlabel(r'$\chi^2$'); ax[1].set_ylabel(r'$C(0,\chi^2,0)$')
        ax[1].set_title(r'slice $\chi^2$  ($\chi^1{=}0$)')
        im = ax[2].imshow(M_xy, origin='lower', extent=[-Xb, Xb, -Xb, Xb],
                          aspect='auto', cmap='viridis')
        ax[2].set_xlabel(r'$\chi^1$'); ax[2].set_ylabel(r'$\chi^2$')
        ax[2].set_title(r'$C(\chi^1,\chi^2,0)$  [%s]' % loop_lab)
        fig.colorbar(im, ax=ax[2], fraction=.046, pad=.04)
        fig.suptitle(r'%s: equal-time $C(\chi^1,\chi^2)$' % name); fig.tight_layout()
        if cfg.save:
            fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
        return fig

    nrows = 3 if have_sim else 2
    fig, ax = plt.subplots(nrows, 3, figsize=cfg.figsize or (15.5, 4.3 * nrows + 0.4))

    # ── row 0: the three axis slices (model per-order + optional sim points) ──
    overlay(ax[0, 0], chi, lambda a: rad(a, chi, it0),
            simxy=(sxm, sC[is0][km][:, a0]) if have_sim else None)
    ax[0, 0].set_xlabel(r'$\chi^1$'); ax[0, 0].set_ylabel(r'$C(\chi^1,0,0)$')
    ax[0, 0].set_title(r'slice $\chi^1$  ($\chi^2{=}0,\tau{=}0$)')
    overlay(ax[0, 1], chi, lambda a: rad(a, chi, it0),
            simxy=(sxm, sC[is0][a0][km]) if have_sim else None)
    ax[0, 1].set_xlabel(r'$\chi^2$'); ax[0, 1].set_ylabel(r'$C(0,\chi^2,0)$')
    ax[0, 1].set_title(r'slice $\chi^2$  ($\chi^1{=}0,\tau{=}0$)')
    overlay(ax[0, 2], tau, lambda a: np.array([rad(a, 0.0, it) for it in range(tau.size)]),
            simxy=(st, sC[:, a0, a0]) if have_sim else None)
    ax[0, 2].set_xlabel(r'$\tau$'); ax[0, 2].set_ylabel(r'$C(0,0,\tau)$')
    ax[0, 2].set_title(r'slice $\tau$  ($\chi^1{=}\chi^2{=}0$)')

    # ── model heatmaps (row 1) + matching sim heatmaps (row 2), shared scale/col ──
    model_maps = [
        (M_xy, [-Xb, Xb, -Xb, Xb], r'$\chi^1$', r'$\chi^2$', r'$C(\chi^1,\chi^2,0)$'),
        (M_xt, [chi.min(), chi.max(), tau.min(), tau.max()], r'$\chi^1$', r'$\tau$', r'$C(\chi^1,0,\tau)$'),
        (M_xt, [chi.min(), chi.max(), tau.min(), tau.max()], r'$\chi^2$', r'$\tau$', r'$C(0,\chi^2,\tau)$')]
    if have_sim:
        sim_maps = [
            # .T: sim C[a,b]=C(χ¹=a,χ²=b) has row=χ¹/col=χ²; the model M_xy and the
            # reused χ¹(x)/χ²(y) labels want row=χ²/col=χ¹ (no-op for an isotropic sim).
            (sC[is0][np.ix_(km, km)].T, [sxm.min(), sxm.max(), sxm.min(), sxm.max()]),
            (sC[:, :, a0][:, km], [sxm.min(), sxm.max(), st.min(), st.max()]),
            (sC[:, a0, :][:, km], [sxm.min(), sxm.max(), st.min(), st.max()])]
    for c, (M, ext, xl, yl, base) in enumerate(model_maps):
        vmin, vmax = float(np.nanmin(M)), float(np.nanmax(M))
        im = ax[1, c].imshow(M, origin='lower', extent=ext, aspect='auto',
                             cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, c].set_xlabel(xl); ax[1, c].set_ylabel(yl)
        ax[1, c].set_title(base + r'  [theory, %s]' % loop_lab)
        fig.colorbar(im, ax=ax[1, c], fraction=.046, pad=.04)
        if have_sim:
            Ms, exts = sim_maps[c]
            im2 = ax[2, c].imshow(Ms, origin='lower', extent=exts, aspect='auto',
                                  cmap='viridis', vmin=vmin, vmax=vmax)
            ax[2, c].set_xlabel(xl); ax[2, c].set_ylabel(yl)
            ax[2, c].set_title(base + r'  [sim]')
            fig.colorbar(im2, ax=ax[2, c], fraction=.046, pad=.04)
    sub = ('isotropic d=2; sim on the shared model colour scale'
           if have_sim else 'isotropic d=2 — χ¹ and χ² views coincide')
    fig.suptitle(r'%s: $C(\chi^1,\chi^2,\tau)$ (%s)' % (name, sub))
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_kpoint(result, cfg, model):
    """Spatial k≥3 cumulant at explicit events: a per-event bar chart
    with the tree / +loop decomposition stacked alongside."""
    cfg = cfg or Config()
    vals = np.real(np.asarray(result['C_kpoint']))
    by_ell = result.get('C_kpoint_by_ell') or {}
    n = len(vals)
    idx = np.arange(n)
    fig, ax = plt.subplots(figsize=cfg.figsize or (max(6, 1.3 * n), 4.3))
    if by_ell and cfg.show_orders != 'total':
        cum = cumulative_curves({e: np.asarray(v) for e, v in by_ell.items()})
        w = 0.8 / max(len(cum), 1)
        for i, ell in enumerate(sorted(cum)):
            ax.bar(idx + i * w, np.real(cum[ell]), width=w,
                   color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                   label=f'theory: {_order_label(ell)}')
    else:
        ax.bar(idx, vals, width=0.6, color='#1F9FCC', label='theory: total')
    ax.set_xlabel('evaluation point')
    ax.set_ylabel(r'$\kappa_k$')
    ax.set_xticks(idx)
    ax.set_title(cfg.title or f"{model.get('name','')}: k={result.get('k')}"
                 " cumulant at events")
    ax.grid(alpha=0.25, axis='y')
    ax.legend(fontsize=8)
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_kpoint_grid(result, cfg, model, sim=None):
    """Spatial k≥3 cumulant swept over the full 2(k-1)-D (χ,τ) grid.  The object
    is 2(k-1)-dimensional, so this draws a 2-D SLICE as a heatmap: at equal time
    (a single τ) the spatial map κ_k(χ₂,χ₃); otherwise κ_k(χ₂,τ₂) with the other
    legs pinned at the grid origin (χ≈0, τ≈0).  The full array is in
    ``result['C_kpoint_grid']`` (axes in ``result['kpoint_grid_axes']``) for any
    other slice you want."""
    cfg = cfg or Config()
    g = np.real(np.asarray(result['C_kpoint_grid']))
    ax = result.get('kpoint_grid_axes') or {}
    tau = np.asarray(ax.get('tau')); chi = np.asarray(ax.get('chi'))
    nlegs = int(ax.get('n_legs', g.ndim // 2))
    k = result.get('k', nlegs + 1)
    it0 = int(np.argmin(np.abs(tau))); ix0 = int(np.argmin(np.abs(chi)))
    if tau.size == 1:                          # equal time → κ_k(χ₂,χ₃) map
        sl = [0, slice(None)]                   # leg2: τ=0, χ₂ varies (rows)
        sl += [0, slice(None)] if nlegs >= 2 else []   # leg3: τ=0, χ₃ varies (cols)
        sl += [0, ix0] * max(0, nlegs - 2)     # legs 4..k pinned at origin
        M = g[tuple(sl)]
        extent = [chi.min(), chi.max(), chi.min(), chi.max()]
        xlab, ylab = r'$\chi_3$', r'$\chi_2$'
        sub = r'$\kappa_%d(\chi_2,\chi_3)$ at $\tau=0$' % k
    else:                                      # κ_k(χ₂,τ₂), other legs pinned
        sl = [slice(None), slice(None)]        # leg2: τ₂ (rows), χ₂ (cols)
        sl += [it0, ix0] * max(0, nlegs - 1)   # legs 3..k pinned at (τ≈0,χ≈0)
        M = g[tuple(sl)]
        extent = [chi.min(), chi.max(), tau.min(), tau.max()]
        xlab, ylab = r'$\chi_2$', r'$\tau_2$'
        sub = r'$\kappa_%d(\chi_2,\tau_2)$  (other legs at $\chi{=}0,\tau{=}0$)' % k
    fig, axp = plt.subplots(figsize=cfg.figsize or (6.6, 4.8))
    im = axp.imshow(np.real(M), aspect='auto', origin='lower', extent=extent,
                    cmap='viridis')
    axp.set_xlabel(xlab); axp.set_ylabel(ylab)
    axp.set_title(cfg.title or f"{model.get('name','')}: {sub}")
    fig.colorbar(im, ax=axp, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


# ── Diagram structure (prediagram topologies, not numbers) ───────────────────

def plot_prediagrams(model, k, max_ell, save=None, ncol=None):
    """Draw the *contributing prediagrams* — the MSR-JD directed topologies that
    survive the model's vertex/source filter — for the k-point cumulant up to
    loop order ``max_ell``, grouped by topology family.

    Layout: time flows right → left, with sources on the right,
    interaction (internal) vertices in the middle, and external legs on the
    left.  Labels are GENERIC and role-distinct: sources *i, ii, …*; internal
    vertices *a, b, c, …*; external legs *1, 2, …*.  Propagators carry no symbol
    of their own — each is named by its endpoints (e.g. *a→b*, with a mid-edge
    arrowhead for the direction).  The specific typing of any one diagram
    (propagator a→b → Δ_xy, source i → K⁽²⁾, a → φ″, leg 1 → field) is a
    separate label-mapping table.

    ``model`` is a model dict (from :func:`load_model`) or a model-name
    string.  Layout uses graphviz ``dot`` when installed, else a built-in
    fallback.  Returns the Matplotlib ``Figure`` (and writes a PNG if ``save``
    is given).  Practical range: ``k + max_ell ≤ 4``.
    """
    from engine.diagrams.prediagram_plot import plot_prediagrams as _pp
    return _pp(model, int(k), int(max_ell), save=save, ncol=ncol)


def prediagram_mappings(model, k, max_ell, external_fields=None,
                        use_propagator=False, max_typings=6, printout=True):
    """The label-mapping tables that accompany :func:`plot_prediagrams`.

    For each contributing prediagram (same numbering as the figure), list its
    typed realizations as maps from the GENERIC labels to the actual field
    types::

        source i        → K⁽²⁾⟨x̃ x̃⟩        (noise cumulant on response legs)
        vertex a        → coeff · φ-vertex   (the interaction monomial)
        propagator a→b  → G[φ ← φ̃]           (bare response propagator)
        external leg 1  → φ                  (the correlator's external field)

    ``model`` is a model dict (from :func:`load_model`) or a model name.
    Returns ``(result, text)`` and prints ``text`` by default; ``result`` is
    ``{ell: [entry…]}`` keeping every typing.  ``max_typings`` caps how many are
    printed per prediagram.  ``use_propagator=True`` builds the propagator and
    drops identically-zero typings (the exact contributing set).
    """
    from engine.diagrams.prediagram_plot import prediagram_mappings as _pm
    return _pm(model, int(k), int(max_ell), external_fields=external_fields,
               use_propagator=use_propagator, max_typings=max_typings,
               printout=printout)


# ── Run summary (printed in every notebook) ──────────────────────────────────

def summary(result: dict) -> str:
    """One-glance text summary of what was computed."""
    r = result.get('_resolved', {})
    model = result.get('_model', {})
    lines = [f"model : {model.get('name','?')!r}",
             f"k      : {r.get('k')}    max_ell : {r.get('max_ell')}",
             f"fields : {field_names(model)}"
             f"   spatial_dim : {spatial_dim(model)}"]
    sp = (model.get('spatial') or {}).get('dyson')
    if sp and sp.get('mode') == 'fixed':
        lines.append(f"dyson  : order {sp.get('order')}")
    si = result.get('spatial_info') or {}
    if si.get('n_live_diagrams') is not None:
        lines.append(f"diagrams (live): {si.get('n_live_diagrams')}")
    return '\n'.join(lines)


# ── diagram figures (tikz-feynman) ──────────────────────────────────────
def export_tikz(result, *, index=None, path=None, standalone=None,
                ell=None, **kw):
    """Emit ``tikz-feynman`` source for the diagrams of a completed run.

    Parameters
    ----------
    result : dict or list
        A result from :func:`run` / ``compute_cumulants`` (its ``'diagrams'``
        records are used), a list of records, or a single typed diagram or
        prediagram.
    index : int or None
        ``None`` (default) draws every diagram as a panel array; an integer
        draws just that one.  It is a position in the records' own order
        (after the ``ell`` filter), and so are the panels of the array.  For
        a result, that is the order of ``result['diagrams']``, which depends
        on where the prediagrams came from (a cache file, the eager
        enumerator of ``use_cache=False``, or the in-memory streaming one
        under ``DAEDALUS_PREDIAGRAM_EAGER=0``), so the same index can name
        another diagram in another run.
    path : str or None
        Write the source here as well as returning it.  A ``.tex`` path
        defaults ``standalone=True`` so the file compiles on its own.
    standalone : bool or None
        Wrap in a compilable document.  ``None`` means "True if writing a
        ``.tex`` file, else False".
    ell : int or None
        Keep only diagrams at this loop order.
    **kw
        Forwarded to the emitter / assembler: ``propagator_label``,
        ``show_factors``, ``symbol_map``, ``scale``, ``ncol``,
        ``panel_width``, ``caption``, ``label``, ``max_panels``.

    Returns
    -------
    str
        The LaTeX source.

    Notes
    -----
    A lone diagram labels its propagators (``G``) because it is usually
    pedagogical; an array suppresses them because repeating one symbol on
    every edge of every panel crowds the vertex factors without adding
    information.  Pass ``propagator_label`` explicitly to override either.
    """
    from engine.diagrams.tikz_export import (
        to_tikz_feynman, diagram_to_standalone,
    )
    from engine.diagrams.tikz_figure import (
        diagrams_to_figure, diagrams_to_standalone,
    )

    if isinstance(result, dict) and 'diagrams' in result:
        records = list(result['diagrams'] or [])
    elif isinstance(result, (list, tuple)) and not (
            len(result) == 4 and hasattr(result[0], 'vertices')):
        records = list(result)
    else:
        records = [result]

    if ell is not None:
        records = [r for r in records
                   if isinstance(r, dict) and r.get('ell') == ell]
        if not records:
            raise ValueError(f'no diagrams at ell={ell}')

    if standalone is None:
        standalone = bool(path) and str(path).endswith('.tex')

    if index is not None:
        rec = records[index]
        one = rec.get('typed_diagram', rec) if isinstance(rec, dict) else rec
        single_kw = {k: v for k, v in kw.items()
                     if k not in ('ncol', 'panel_width', 'caption',
                                  'label', 'max_panels')}
        tex = (diagram_to_standalone(one, **single_kw) if standalone
               else to_tikz_feynman(one, **single_kw))
    else:
        tex = (diagrams_to_standalone(records, **kw) if standalone
               else diagrams_to_figure(records, **kw))

    if path:
        with open(path, 'w') as fh:
            fh.write(tex)
    return tex


def main(argv=None):
    """Command line: ``sage -python -m daedalus run config.yaml [--dry-run]``
    (logic in ``api/config_run.py``)."""
    from api.config_run import main as _cli
    status = _cli(argv)
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()

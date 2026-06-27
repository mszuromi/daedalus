"""daedalus.py — shared scaffolding for the pipeline demo notebooks.

Centralises the **load → run → plot** flow so every demo notebook is thin and
uniform regardless of its group (temporal / spatial × single / multi-field).
The four group templates (``notebooks/templates/``) and every
``*_sim_compare`` notebook import this module so the thematics are common.

    import daedalus as dd
    model, mod = dd.load_theory('kpz_1d')          # from theories/*.theory.py
    cfg = dd.Config(k=2, max_ell=1, chi_grid=(-6, 6, 49))   # χ = spatial sep.
    res = dd.run(model, cfg, mod)                  # k/ell/Dyson all here
    dd.plot_cumulant(res, cfg, model)              # adaptable, auto-dispatched

Design choices
--------------
* **Single source of truth for theories.**  ``load_theory`` imports a
  ``theories/<name>.theory.py`` file and returns ``(model, module)``; demo
  notebooks never re-build a theory inline.
* **Arbitrary k and ℓ.**  ``Config.k`` / ``Config.max_ell`` are free; the
  spatial k≥3 path is wired through ``Config.spatial_points``.
* **Dyson dressing exposed.**  ``Config.dyson_order`` (+ ``reference_diffusion``)
  overrides the model's Dyson policy at run time — any order ≥ 0.
* **Adaptable plotting.**  ``plot_cumulant`` auto-dispatches on
  (spatial?, multi-field?, k) and honours the plot options on ``Config``
  (``show_orders``, ``logy``, ``components``, ``figsize``, …).
"""
from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

#: Daedalus version (keep in sync with pyproject.toml + CITATION.cff).
__version__ = '0.1.0'


# ── Repo / theory discovery ──────────────────────────────────────────────────

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
THEORIES_DIR = os.path.join(REPO_ROOT, 'theories')
import sys as _sys
if REPO_ROOT not in _sys.path:
    _sys.path.insert(0, REPO_ROOT)


# ── Lazy re-exports of the user-facing pipeline API ──────────────────────────
# Every major pipeline symbol a user touches is reachable as ``dd.<name>``,
# alongside the notebook helpers defined below (``Config``, ``run``, ``plot_*``,
# ``load_theory``, …).  Resolved lazily on first access (PEP 562) so
# ``import daedalus`` stays light and only pulls in Sage when something is
# actually used.  The canonical ``from api... import <name>`` forms keep
# working — these are aliases; that explicit form is the one to use inside a
# saved ``theories/*.theory.py`` file (which cannot import this notebook helper).
_PIPELINE_EXPORTS = {
    # authoring — build a theory in code (pipeline.theory)
    'TheoryBuilder':             'api.theory',
    'TemporalTheoryBuilder':     'api.theory',
    'SpatialTheoryBuilder':      'api.theory',
    # noise template (api.theory_templates) for builder.add_gtas_noise(...)
    'GTaSNoise':                 'api.theory_templates',
    # graphical builder (pipeline.ui)
    'TheoryUI':                  'api.ui',
    # compute / report / precompute / persistence / access (pipeline top level)
    'compute_cumulants':         'api',
    'generate_report':           'api',
    'precompute':                'api',
    'save_npz':                  'api',
    'save_csv':                  'api',
    'params_slug':               'api',
    'MeanField':                 'api',
    'Parameters':                'api',
    'normalize_external_fields': 'api',
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


def list_theories() -> list[str]:
    """Every ``theories/<name>.theory.py`` available to load."""
    return sorted(f[:-len('.theory.py')]
                  for f in os.listdir(THEORIES_DIR)
                  if f.endswith('.theory.py'))


def load_theory(name: str):
    """Import ``theories/<name>.theory.py`` and return ``(model, module)``.

    ``model = module.build()`` is the model dict; ``module`` exposes
    ``DEFAULT_FUNDAMENTAL`` and ``METADATA`` (the run defaults)."""
    path = os.path.join(THEORIES_DIR, f'{name}.theory.py')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'No theory file at {path}. Available: {list_theories()}')
    spec = importlib.util.spec_from_file_location(f'theories.{name}', path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod.build(), mod
    except Exception as e:
        raise RuntimeError(
            f'Failed to load theory {name!r} from {path}: {e}') from e


# ── Model introspection ──────────────────────────────────────────────────────

def is_spatial(model: dict) -> bool:
    sp = model.get('spatial')
    return bool(sp and sp.get('dim'))


def spatial_dim(model: dict) -> int:
    sp = model.get('spatial') or {}
    return int(sp.get('dim') or 0)


def is_multifield(model: dict) -> bool:
    """True iff the theory has more than one independent field channel —
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


def _fmt_default(v, maxlen: int = 52) -> str:
    s = repr(v)
    return s if len(s) <= maxlen else s[:maxlen - 1] + '…'


# Provenance / dev-status boilerplate to strip from a theory docstring so
# describe_model shows the model's physics, not its changelog.
_DOC_DROP = ('text-driven theory file', 'Generated by', 'Loaded by',
             '.theory.py', '====', '----', 'Schema', 'exposes:', 'docs/',
             '``build()``', '``DEFAULT_FUNDAMENTAL``', '``METADATA``',
             'returning a ``model``', 'suggested numerical', 'free-form info',
             'ready for', 'pattern).')
_DOC_STOP = ('Phase 1 status', 'Phase 2 status', 'Phase 3 status',
             'See ``docs', 'Mirrors the inline', 'status:', 'VERBATIM')


def _doc_physics(doc: str) -> str:
    """Trim a theory docstring to its physics prose — drop UI-provenance
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
    """Pretty-print the structure of a loaded theory — the *model information
    from the theory file*: domain, fields, populations, parameters, kernels,
    transfer functions, and the governing equation(s).  Prints and returns the
    text.  Pass the theory ``module`` (the 2nd value of :func:`load_theory`) to
    also surface its docstring + ``METADATA`` run recommendations.

    This is the canonical "what is this model?" cell for the example
    notebooks — it reads only the model dict, so it stays model-independent."""
    out = []
    bar = '─' * 72
    out.append(bar)
    out.append(f"  {model.get('name', '(unnamed theory)')}")
    out.append(bar)

    def _mode(v, default):
        if isinstance(v, dict):
            return v.get('mode', default)
        return v or default

    if is_spatial(model):
        sp = model.get('spatial') or {}
        bc = _mode(model.get('boundary') or sp.get('boundary'), 'infinite')
        ic = _mode(model.get('initial'), 'stationary')
        out.append(f"Domain         : spatial PDE · d={spatial_dim(model)} · "
                   f"boundary={bc} · initial={ic}")
    else:
        out.append("Domain         : temporal ODE (time-only)")

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
    numeric = [p for p in params if not str(p.get('name', '')).endswith('star')]
    saddle = [p.get('name') for p in params
              if str(p.get('name', '')).endswith('star')]
    if numeric:
        out.append("Parameters     :")
        for p in numeric:
            ix = p.get('indexed_by')
            ixt = f" [{','.join(ix)}]" if ix else ''
            dom = f"  ({p['domain']})" if p.get('domain') else ''
            out.append(f"    {p.get('name')}{ixt} = "
                       f"{_fmt_default(p.get('default'))}{dom}")
    if saddle:
        out.append("Mean-field saddle (solved by the pipeline): "
                   + ', '.join(saddle))

    ker = model.get('kernels') or []
    for kr in ker:
        ix = kr.get('indexed_by')
        ixt = f"[{','.join(ix)}]" if ix else ''
        out.append(f"Kernel         : {kr.get('name')}{ixt} — "
                   "non-Markovian temporal convolution")
    fns = model.get('functions') or []
    for fn in fns:
        out.append(f"Function       : {fn.get('name')}(·) "
                   f"— {fn.get('n_args')}-arg transfer")

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

    Leave a field ``None`` to inherit the theory file's ``METADATA`` /
    ``DEFAULT_FUNDAMENTAL`` default.
    """
    # ── what to compute ──
    k: Optional[int] = None                 # correlator order; inferred from
                                            # external_fields if left None
    max_ell: Optional[int] = None           # loop order (0=tree, 1, 2, …)
    external_fields: Optional[list] = None  # e.g. [('x', 1), ('x', 1)]
    parameters: Optional[dict] = None        # numeric parameter overrides, by name
    fundamental: Optional[dict] = None       # deprecated alias for ``parameters``
    # Output quantity.  The diagrammatics produce CONNECTED cumulants
    # κ(x₁…x_k); 'moment'/'central_moment' assemble the full k-point moment
    # ⟨φ(x₁)…φ(x_k)⟩ (resp. of the centred field) via the set-partition
    # formula M = Σ_π ∏_B κ(B) — see _assemble_moment.  Costs k−1 extra
    # backend runs (one per order 2..k).
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
    # through the origin (sweep leg j, others at 0).
    kpoint_base_lags: Optional[list] = None  # length k−1: the fixed τ for the
                                             # non-swept legs (the base point the
                                             # slices pass through); default 0
    kpoint_full_grid: bool = False           # compute the full (k−1)-dim tensor
                                             # C(τ_1..τ_{k−1}) instead of slices
                                             # (cost ~n^{k−1}; k=3 → heatmap)

    # ── Dyson dressing (coupled unequal-D) ──
    dyson_order: Optional[int] = None       # None=leave model; int≥0=override
    reference_diffusion: Optional[float] = None

    # ── mean-field DAE root selection (multi-root theories) ──
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
    # temporal batch: os.cpu_count()).  NOTE: the SPATIAL backend is
    # thread-based (safe in a Jupyter kernel) and is the one this actually
    # tunes in a notebook; the TEMPORAL batch is fork-based and is force-
    # serialized on macOS+Jupyter (see the fork guard), so n_workers there
    # only takes effect outside a notebook (scripts / pytest / Linux).
    n_workers: Optional[int] = None
    verbose: bool = False

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
        if self.parameters is None and self.fundamental is not None:
            self.parameters = self.fundamental
        elif self.fundamental is None and self.parameters is not None:
            self.fundamental = self.parameters
        # ``chi_grid`` (the paper's spatial-difference grid χ=x_j−x_k) is the
        # canonical name; ``spatial_grid`` is the backward-compat alias.  Mirror
        # whichever was set onto the other (``chi_grid`` wins if both are given).
        if self.chi_grid is None and self.spatial_grid is not None:
            self.chi_grid = self.spatial_grid
        elif self.spatial_grid is None and self.chi_grid is not None:
            self.spatial_grid = self.chi_grid
        # Fail early on a mistyped output kind (otherwise it silently
        # degrades to the connected cumulant — a wrong-kind result).
        if self.output not in ('cumulant', 'moment', 'central_moment'):
            raise ValueError(
                "Config.output must be 'cumulant', 'moment', or "
                f"'central_moment'; got {self.output!r}.")

    def resolved_grid(self):
        """Materialise ``chi_grid`` — the spatial-difference grid χ = x_j − x_k
        (accepts an ``(lo, hi, n)`` tuple).  ``spatial_grid`` is the alias."""
        return self._resolve_grid(self.chi_grid)

    def resolved_tau_grid(self):
        """Materialise ``tau_grid`` (accepts an ``(lo, hi, n)`` tuple), or None."""
        return self._resolve_grid(self.tau_grid)

    @staticmethod
    def _resolve_grid(g):
        """``(lo, hi, n)`` → ``linspace``; array/list → array; None → None."""
        if g is None:
            return None
        if isinstance(g, tuple):
            if len(g) != 3:
                raise ValueError(
                    f"grid tuple must be (lo, hi, n); got a {len(g)}-tuple "
                    f"{g!r}. For an arbitrary grid pass a list/array instead.")
            return np.linspace(g[0], g[1], int(g[2]))
        return np.asarray(g, dtype=float)


def _meta(mod, key, default=None):
    return (getattr(mod, 'METADATA', {}) or {}).get(key, default)


def fundamental_from_model(model: dict) -> dict:
    """Build a ``fundamental`` dict from the parameters' own ``default``
    values baked into the model — every non-saddle parameter with a
    declared default.  Lets a loaded theory run out of the box even when
    its ``DEFAULT_FUNDAMENTAL`` is empty.  Saddle params (``*star``,
    solved by the mean-field step) and defaultless params are skipped."""
    out = {}
    for p in model.get('parameters') or []:
        name = p.get('name') or ''
        if name.endswith('star') or p.get('default') is None:
            continue
        out[name] = p['default']
    return out


# Canonical alias (matches the ``Config.parameters`` rename); the original
# name is kept for the existing notebooks that import it.
parameters_from_model = fundamental_from_model


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
      spatial   k≥3 events   spatial_points = (n_pts, k−1, 2) of (x_j, τ_j)
    """
    import dataclasses
    d = {f.name: f.default for f in dataclasses.fields(Config)}

    def row(name, desc):
        return '    %-17s = %-11s  %s' % (name, repr(d.get(name)), desc)

    common_grid = [
        ('what to compute', [
            ('k',               'correlator order: 1=mean ⟨φ⟩, 2=correlation, ≥3=cumulant'),
            ('max_ell',         'loop order: 0=tree, 1=+1-loop, 2=+2-loop, …'),
            ('external_fields', "the k legs, e.g. [('x',1),('x',1)]; sets k if k is None"),
            ('parameters',      "numeric overrides {name: value}; None → theory defaults"),
            ('output',          "'cumulant' | 'moment' | 'central_moment'"),
        ]),
    ]
    temporal_grid = ('temporal grid / slicing (τ axis)', [
        ('tau_grid',         'τ (lag) grid for C(τ): array | (lo,hi,n)  — the primary τ knob'),
        ('tau_max',          'under-the-hood: τ extent (symmetric −tau_max…tau_max) if no tau_grid'),
        ('tau_step',         'under-the-hood: τ spacing (paired with tau_max)'),
        ('kpoint_base_lags', 'k≥3: [k−1 floats] fix the non-swept legs (slices cross here)'),
        ('kpoint_full_grid', 'k≥3: True → full (k−1)-D tensor C(τ₁…) instead of axis slices'),
    ])
    spatial_grid_ = ('spatial grid / slicing (χ and τ axes)', [
        ('chi_grid',      'spatial-difference grid χ=x_j−x_k: (lo,hi,n) or array; [χ0] fixes χ'),
        ('tau_grid',      'τ grid for C(χ,τ): array | (lo,hi,n); [0.0] → equal-time C(χ,0)'),
        ('tau_max',       'under-the-hood: τ extent if no tau_grid (0.0 → equal-time)'),
        ('tau_step',      'under-the-hood: τ spacing (paired with tau_max)'),
        ('spatial_points','k≥3: (n_pts, k−1, 2) array of explicit (x_j, τ_j) events'),
    ])
    rest = [
        ('mean-field root (multi-root theories)', [
            ('fixed_point_index', 'which stable saddle root to expand around (0,1,…)'),
            ('mf_dae_n_starts',   'multi-start Newton count for the saddle solve'),
            ('mf_dae_seed_box',   '{field: (lo,hi)} start range for the saddle search'),
        ]),
        ('Dyson dressing (coupled unequal-D)', [
            ('dyson_order',         'None=leave model; int≥0 overrides the Dyson order'),
            ('reference_diffusion', 'reference D for the Dyson expansion'),
        ]),
        ('execution', [
            ('parallel',  'enable the parallel backend (temporal fork-parallelism is force-serialized in a macOS Jupyter kernel)'),
            ('n_workers', 'worker count (spatial threads; temporal batch outside Jupyter)'),
            ('verbose',   'print backend progress'),
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
          "theory's METADATA):\n")
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


def _external_mean(model: dict, res: dict) -> float:
    """Mean-field saddle φ* for the external leg's field — the TREE-level value
    of the 1-point function ⟨φ⟩.

    The cumulant pipeline expands around the saddle, so the *fluctuation*
    field's tree-level 1-point function is 0; the physical mean is the saddle
    value plus the ℓ≥1 loop tadpole shifts.  The saddle is stored under the
    ``'<base>star'`` convention (``nSstar``, ``xstar``, ``hstar`` …) in the
    temporal ``res['mf_values']`` or the spatial ``res['mf']['mf_values']``
    (a list of per-component dicts).  Returns the saddle for the external
    leg's base physical field (0.0 for a symmetric/zero saddle, or if none is
    found)."""
    # Collect every candidate saddle dict {'<base>star': [vals]} — temporal
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
    ext = (res.get('_resolved') or {}).get('external_fields')
    if ext:
        e0 = ext[0]
        fld = e0[0] if isinstance(e0, (tuple, list)) else str(e0)
    else:
        names = field_names(model)
        fld = names[0] if names else ''
    base = fld[1:] if fld.startswith('d') else fld
    keys = [base + 'star', fld + 'star', base, fld]
    for d in dicts:
        for key in keys:
            if key in d:
                try:
                    val = float(np.real(np.asarray(d[key]).ravel()[0]))
                    return 0.0 if abs(val) < 1e-14 else val   # kill subnormals
                except Exception:
                    pass
    return 0.0


def _assemble_moment_temporal(model, res, kw, k, central):
    """Full / central k-point MOMENT along the same 1-D slice as ``C_tau``:

        M(τ) = ⟨φ(0) φ(τ) φ(0) … φ(0)⟩
             = Σ_π  Σ_{(ℓ_B): Σℓ_B ≤ L}  ∏_{B∈π} κ(B)^{(ℓ_B)}

    — the set-partition (cluster) expansion truncated at ONE shared loop budget
    ``L = max_ell``.  This is the perturbatively-consistent L-loop moment: the
    total loop order of every surviving term is ≤ L, so a 1-loop moment never
    smuggles in a partial 2-loop ℓ_B₁·ℓ_B₂ cross-term (which the naive
    product-of-fully-dressed-cumulants would).  The two agree only at tree
    (L=0); they diverge at L≥1 for any multi-block partition.

    Singleton blocks contribute the tree mean ⟨φ⟩ at ℓ=0 (raw) or drop out
    (central → only no-singleton partitions survive); their 1-loop tadpole is
    not folded in (a documented refinement).  The per-order cumulants
    κ_j^{(ℓ)} come straight from the ``*_by_ell`` increments — one extra
    ``compute_cumulants`` run per order 2..k, each delivering every loop order
    at once (the order-k run is reused)."""
    import itertools
    from api import compute_cumulants
    tau = np.asarray(res['tau_grid'], dtype=float)
    f0 = kw['external_fields'][0]
    L = int(kw.get('max_ell', 0) or 0)
    mu = 0.0 if central else _external_mean(model, res)

    def _by_ell_2pt(r):
        """{ℓ: even-lag interpolator} of the 2-point cumulant's ℓ-loop piece."""
        be = r.get('C_tau_by_ell') or {0: r.get('C_tau')}
        t = np.asarray(r['tau_grid']); a = np.abs(t); o = np.argsort(a); ax = a[o]
        out = {}
        for e in range(L + 1):
            arr = be.get(e)
            if arr is None:
                out[e] = (lambda lag: 0.0 + 0.0j)
            else:
                ay = np.real(np.asarray(arr))[o]
                out[e] = (lambda lag, ax=ax, ay=ay:
                          complex(np.interp(abs(lag), ax, ay)))
        return out

    # κ_2 per loop order (reuse the order-k run when k==2, else one 2-pt run).
    if k == 2:
        k2 = _by_ell_2pt(res)
    else:
        kw2 = dict(kw); kw2.update(k=2, external_fields=[f0] * 2)
        k2 = _by_ell_2pt(compute_cumulants(**kw2))

    # κ_j per loop order for j = 3..k (reuse the order-k ``*_by_ell``).
    kj = {}
    for j in range(3, k + 1):
        if j == k and res.get('total_C_by_ell'):
            be = res['total_C_by_ell']
        else:
            kwj = dict(kw); kwj.update(k=j, external_fields=[f0] * j)
            be = compute_cumulants(**kwj).get('total_C_by_ell') or {}
        kj[j] = {e: (be.get(e) if callable(be.get(e))
                     else (lambda *t: 0.0 + 0.0j))
                 for e in range(L + 1)}

    parts = list(_set_partitions(list(range(k))))

    def _kappa(block, ell, ti):
        """ℓ-loop piece of the |block|-point cumulant at the slice times."""
        b = sorted(block)
        if len(b) == 2:                       # leg 1 swept (τ); rest pinned (0)
            return k2[ell](float(tau[ti]) if 1 in b else 0.0)
        times = [float(tau[ti]) if idx == 1 else 0.0 for idx in b]
        return complex(kj[len(b)][ell](*times))

    M = np.empty(tau.size, dtype=complex)
    for ti in range(tau.size):
        tot = 0.0 + 0.0j
        for part in parts:
            singles = [bl for bl in part if len(bl) == 1]
            multis = [bl for bl in part if len(bl) > 1]
            if central and singles:           # central drops every singleton
                continue
            mu_factor = mu ** len(singles)    # singletons: tree mean, ℓ=0 only
            if singles and mu_factor == 0.0:  # zero mean → singleton terms die
                continue
            if not multis:                    # all-singleton (raw): pure μ^k
                tot += mu_factor
                continue
            # share the loop budget L across the multi-blocks (Σℓ_B ≤ L);
            # singletons consume none (only the ℓ=0 mean is available).
            for assign in itertools.product(range(L + 1), repeat=len(multis)):
                if sum(assign) > L:
                    continue
                prod = mu_factor
                for bl, e in zip(multis, assign):
                    prod *= _kappa(bl, e, ti)
                tot += prod
        M[ti] = tot
    return M


def run(model: dict, cfg: Config, module=None) -> dict:
    """Resolve the config against the theory's defaults and call
    ``compute_cumulants``.  Handles temporal vs spatial, the spatial
    k≥3 event path, and the Dyson-order override.  Returns the result
    dict with ``cfg`` / ``model`` attached under ``'_cfg'`` / ``'_model'``."""
    from api import compute_cumulants

    max_ell = (cfg.max_ell if cfg.max_ell is not None
               else (_meta(module, 'ell_default', 0) if module else 0))
    ext = cfg.external_fields
    ext_is_explicit = ext is not None

    # Resolve the correlator order k.  Explicit ``cfg.k`` wins; otherwise infer
    # it from an explicit ``external_fields`` (a k-point correlator has exactly
    # k legs); otherwise fall back to the theory's ``k_default``.  If BOTH are
    # given they must agree — a mismatch is a contradiction, not something to
    # silently paper over.
    if cfg.k is not None:
        k = cfg.k
        if ext_is_explicit and len(ext) != k:
            raise ValueError(
                f"Config mismatch: k={k} but external_fields lists "
                f"{len(ext)} leg(s) {ext}.  A k-point correlator needs exactly "
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
    valid = set(field_names(model))
    def _nm(e):
        return e[0] if isinstance(e, (tuple, list)) else e
    if not ext_is_explicit and (
            ext is None or len(ext) != k
            or not all(_nm(e) in valid for e in ext)):
        fld = field_names(model)
        f0 = fld[0] if fld else 'phi'
        ext = [(f0, 1)] * k
    # Layered fundamental: model param defaults ← theory DEFAULT_FUNDAMENTAL
    # ← explicit cfg override.  Each layer wins over the one before, so a
    # loaded theory runs out of the box and the notebook can override any
    # subset.
    fundamental = fundamental_from_model(model)
    fundamental.update(getattr(module, 'DEFAULT_FUNDAMENTAL', {}) or {}
                       if module else {})
    if cfg.parameters:
        # Reject mistyped override names instead of silently inserting a
        # key the backend ignores (which would run with the DEFAULT value).
        _valid = {p['name'] for p in (model.get('parameters') or [])
                  if not str(p.get('name', '')).endswith('star')
                  and not p.get('mean_field')}
        _unknown = [kk for kk in cfg.parameters if kk not in _valid]
        if _valid and _unknown:
            raise ValueError(
                f"Unknown parameter override(s) {_unknown}. "
                f"Declared parameters: {sorted(_valid)}.")
        fundamental.update(cfg.parameters)

    # Dyson override: inject into the model's spatial policy at run time.
    # Copy first so we never mutate the caller's model dict — load_theory
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

    # Mean-field DAE root-selection overrides (multi-root theories such as
    # the double-well regime mu<0).  Forward each only when set, so that
    # ``compute_cumulants``' own defaults (fixed_point_index=0,
    # mf_dae_n_starts=64, mf_dae_seed_box=None) are preserved otherwise.
    if cfg.fixed_point_index is not None:
        kw['fixed_point_index'] = int(cfg.fixed_point_index)
    if cfg.mf_dae_n_starts is not None:
        kw['mf_dae_n_starts'] = int(cfg.mf_dae_n_starts)
    if cfg.mf_dae_seed_box is not None:
        kw['mf_dae_seed_box'] = cfg.mf_dae_seed_box

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
            kw['spatial_points'] = np.zeros((1, 0, 2), dtype=float)
        elif k != 2 and cfg.spatial_points is None:
            raise ValueError(
                'spatial k≥3 needs Config.spatial_points = (n_pts, k-1, 2) '
                'array of (x_j, τ_j) offsets per non-anchor external slot.')
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

    res = compute_cumulants(**kw)

    # Temporal k≥3: compute_cumulants returns ``C_tau=None`` and callable
    # ``total_C`` / ``total_C_by_ell`` (the connected k-point cumulant is a
    # function of the k external times).  Synthesise the natural 1-D slice
    # C_k(τ) = ⟨φ(0) φ(τ) φ(0) … φ(0)⟩_c — sweep leg 1, pin the others at
    # τ=0 — and store it as ``C_tau`` / ``C_tau_by_ell`` so k≥3 plots and
    # compares exactly like k=2 (a curve over ``tau_grid``).  The simulator
    # estimates the matching slice with ``lag_bins=[0, None, 0, …]``.
    if (not is_spatial(model) and k >= 3 and res.get('C_tau') is None
            and callable(res.get('total_C'))):
        tau = np.asarray(res.get('tau_grid'))
        if tau is None or tau.size == 0:
            tau = np.array([0.0])
            res['tau_grid'] = tau
        elif tau.size > 41:                      # bound the # of 1-D evaluations
            tau = np.linspace(float(tau.min()), float(tau.max()), 41)
            res['tau_grid'] = tau

        # The connected k-point cumulant depends on k−1 time differences
        # τ_j = t_j − t_0 (leg 0 anchored at τ=0).  ``base`` is the fixed point
        # the axis-parallel slices pass through (the non-swept legs sit here).
        base = cfg.kpoint_base_lags
        if base is not None:
            base = [float(b) for b in base]
            if len(base) != k - 1:
                raise ValueError(
                    f"kpoint_base_lags must have k−1 = {k-1} entries (one per "
                    f"non-anchor leg); got {len(base)}.")
        else:
            base = [0.0] * (k - 1)

        def _args(vals):                # vals: τ for legs 1..k−1 (leg 0 = 0)
            a = [0.0] * k
            for leg in range(1, k):
                t = float(vals[leg - 1])
                # Itô LEFT-limit (Itô ⇒ left-continuous): a leg coinciding with
                # the anchor (t=0) is nudged to 0⁻ so a coincident causal ordering
                # survives and the equal-time step discontinuity isn't sampled —
                # same convention as the k=2 path (pipeline.compute._ITO_EPS).
                a[leg] = t if abs(t) > 1e-12 else -1e-6
            return a

        def _slice(fn, j):              # sweep leg j over tau, others at base
            out = np.empty(tau.size, dtype=complex)
            for i in range(tau.size):
                v = list(base)
                v[j - 1] = float(tau[i])
                out[i] = complex(fn(*_args(v)))
            return out

        try:
            tcb = {e: f for e, f in (res.get('total_C_by_ell') or {}).items()
                   if callable(f)}
            # The k−1 axis-parallel slices through ``base``.
            slices = {j: _slice(res['total_C'], j) for j in range(1, k)}
            slices_by_ell = {j: {e: _slice(f, j) for e, f in tcb.items()}
                             for j in range(1, k)}
            res['C_tau_slices'] = slices
            res['C_tau_slices_by_ell'] = slices_by_ell
            res['C_tau'] = slices[1]                    # canonical single slice
            res['C_tau_by_ell'] = slices_by_ell[1]
            res['_kpoint_base'] = base
            res['_kpoint_slice'] = (
                f'{k-1} slices: leg j swept over tau_grid (τ_j = t_j − t_0), '
                f'legs ≠ j fixed at base={base}, for j = 1..{k-1}')

            # Optional: the FULL (k−1)-dim tensor C(τ_1..τ_{k−1}).  Downsample
            # the axis so the total n^{k−1} evaluations stay bounded.
            if cfg.kpoint_full_grid:
                import itertools
                n = min(tau.size, max(2, int(4000 ** (1.0 / (k - 1)))))
                gtau = (tau if n >= tau.size else
                        np.linspace(float(tau.min()), float(tau.max()), n))

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
            # Don't swallow silently: record it and surface a note when verbose.
            # (The catch stays broad — lambdified callables have a wide failure
            # surface — but we instrument rather than hide it.)
            res['_kpoint_slice_error'] = repr(e)
            if cfg.verbose:
                print(f'  [k>=3 slice synthesis failed: {e!r}; '
                      f'using the equal-time fallback]')

    # Optional output conversion: assemble the full / central k-point MOMENT
    # from the cumulants (the set-partition / cluster expansion).
    # compute_cumulants stays pure — this is post-processing on top.
    out_kind = getattr(cfg, 'output', 'cumulant') or 'cumulant'
    if out_kind in ('moment', 'central_moment'):
        central = out_kind == 'central_moment'
        if is_spatial(model):
            if k == 2:                      # M(x) = κ₂(x) [+ μ² for raw]
                mu = 0.0 if central else _external_mean(model, res)
                res['moment'] = (np.real(np.asarray(res['C_tau_x']))
                                 + (0.0 if central else mu * mu))
            else:
                raise NotImplementedError(
                    f"Config.output={out_kind!r} for spatial k≥3 is not "
                    "implemented yet (temporal any-k and spatial k=2 are).")
        else:
            res['moment'] = _assemble_moment_temporal(
                model, res, kw, k, central)
        res['output_kind'] = out_kind

    res['_cfg'] = cfg
    res['_model'] = model
    res['_resolved'] = dict(k=k, max_ell=max_ell, external_fields=ext,
                            parameters=fundamental, fundamental=fundamental)
    return res


# ── Per-loop-order decomposition ─────────────────────────────────────────────

_ORDER_COLORS = ['#3F00FF', '#1F9FCC', '#16A085', '#E67E22', '#E74C3C',
                 '#8E44AD', '#7F8C8D']


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
                  sim: dict = None):
    """Plot the cumulant in the form natural to the theory's group.

    Dispatch:
      * k=1 (any group)     → :func:`plot_temporal_mean` (the 1-point mean
                              ⟨φ⟩; tree level = the mean-field saddle φ*)
      * spatial k≥3 events  → :func:`plot_kpoint`
      * spatial k=2         → :func:`plot_spatial`
      * temporal            → :func:`plot_temporal`

    ``sim`` (optional) overlays a matched simulator: a dict with
    ``tau``/``C``/``C_err`` (temporal) or ``x``/``C``/``C_err`` (spatial); for
    k=1 a scalar mean ``{'C': value, 'C_err': err}``.
    Returns the Matplotlib ``Figure``.
    """
    cfg = cfg or result.get('_cfg') or Config()
    model = model or result.get('_model') or {}
    if result.get('output_kind') in ('moment', 'central_moment') \
            and result.get('moment') is not None:
        return _plot_moment(result, cfg, model, sim)
    # k=1 is a 1-POINT function: the mean ⟨φ⟩ of the external field.  Its TREE
    # level is the mean-field saddle φ*; ℓ≥1 add the loop tadpole shifts.  There
    # is no C(τ)/C(χ) curve, so draw it as tree/+loop bars (works for both
    # temporal and the spatial mean-field, which is x-independent).
    if (result.get('_resolved') or {}).get('k') == 1:
        return plot_temporal_mean(result, cfg, model, sim)
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
    return plot_temporal(result, cfg, model, sim)


def _plot_moment(result, cfg, model, sim=None):
    """Plot the assembled full / central k-point MOMENT (``Config.output`` ≠
    'cumulant').  Temporal → M_k(τ) along the slice; spatial k=2 → M(x,0).
    A single curve (the moment mixes loop orders, so no per-order overlay)."""
    cfg = cfg or Config()
    kind = result.get('output_kind', 'moment')
    k = (result.get('_resolved') or {}).get('k', '?')
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


def plot_temporal(result, cfg, model, sim=None):
    """C(τ) with a per-loop-order overlay (the temporal group plot)."""
    cfg = cfg or Config()
    if result.get('C_tau') is None:          # temporal k≥3 → scalar k-point
        return plot_temporal_kpoint(result, cfg, model, sim)
    tau = np.asarray(result['tau_grid'])
    by_ell = {e: v for e, v in (result.get('C_tau_by_ell') or {}).items()
              if v is not None}
    fig, ax = plt.subplots(figsize=cfg.figsize or (7.5, 4.6))
    curves = _orders_to_draw(by_ell, cfg)
    if not curves:
        curves = [('total', np.real(np.asarray(result['C_tau'])))]
    for i, (lab, c) in enumerate(curves):
        ax.plot(tau, c, '-', lw=1.8,
                color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                label=f'theory: {lab}')
    if sim is not None:
        st, sc = np.asarray(sim['tau']), np.asarray(sim['C'])
        se = sim.get('C_err')
        se = np.asarray(se) if se is not None else None
        if sc.ndim == 2:                  # a k≥3 multi-slice sim → show slice 1
            sc = sc[0]
            se = se[0] if se is not None else None
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
    fig.tight_layout()
    if cfg.save:
        fig.savefig(cfg.save, dpi=130, bbox_inches='tight')
    return fig


def plot_temporal_kpoint_slices(result, cfg, model, sim=None):
    """k≥3: the connected k-point cumulant has k−1 independent time differences
    τ_j = t_j − t_0.  Draw one panel per difference — slice j sweeps leg j over
    the τ-grid with the other legs pinned at τ=0 — each with the per-loop-order
    overlay and (optionally) the matching simulator slice.

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
    k = (result.get('_resolved') or {}).get('k', '?')
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
    fld = (result.get('_resolved') or {}).get('external_fields') or [('phi', 1)]
    fb = fld[0][0] if isinstance(fld[0], (tuple, list)) else str(fld[0])
    base = fb[1:] if fb.startswith('d') else fb        # physical field (drop 'd')
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
    picks another), the **theory C(χ,τ) heatmap** (which contains *every* loop
    order in the run — the
    title states the order), and — when ``sim`` carries a 2-D correlator
    (``sim={'x','tau','C'}`` with a 2-D ``C(τ,χ)``) — the matching **simulation
    heatmap on a shared colour scale**.  A 1-D ``sim={'x','C'[,'C_err']}`` still
    overlays on the equal-time panel."""
    cfg = cfg or Config()
    C = np.real(np.asarray(result['C_tau_x']))     # (n_tau, n_x) — total, all loops
    xs = np.asarray(result['spatial_grid'])
    tau = np.asarray(result['tau_grid'])
    i0 = int(np.argmin(np.abs(tau)))
    si = result.get('spatial_info') or {}
    by_order = si.get('C_by_order')                # {ell: (n_tau, n_x)} cumulative
    ell_top = (max(by_order) if by_order
               else ((result.get('_resolved') or {}).get('max_ell', 0) or 0))
    loop_lab = _order_label(ell_top)               # e.g. 'tree+1loop'
    sim2d = (sim is not None and sim.get('tau') is not None
             and np.ndim(np.asarray(sim.get('C'))) == 2)
    has_tau = tau.size > 1

    if not has_tau:                                # equal-time only → single panel
        fig, axA = plt.subplots(figsize=cfg.figsize or (6.0, 4.3))
        axB = axH = axS = None
    elif sim2d:                                    # slices + theory & sim heatmaps
        fig, ax = plt.subplots(2, 2, figsize=cfg.figsize or (11.5, 8.2))
        axA, axB, axH, axS = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    else:                                          # slices + theory heatmap
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

    # ── theory (+ sim) C(χ,τ) heatmaps, on a shared colour scale ──
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


# ── Diagram structure (prediagram topologies, not numbers) ───────────────────

def plot_prediagrams(model, k, max_ell, save=None, ncol=None):
    """Draw the *contributing prediagrams* — the MSR-JD directed topologies that
    survive the theory's vertex/source filter — for the k-point cumulant up to
    loop order ``max_ell``, grouped by topology family.

    Buice/Ocker convention: time flows right → left, with sources on the right,
    interaction (internal) vertices in the middle, and external legs on the
    left.  Labels are GENERIC and role-distinct: sources *i, ii, …*; internal
    vertices *a, b, c, …*; external legs *1, 2, …*.  Propagators carry no symbol
    of their own — each is named by its endpoints (e.g. *a→b*, with a mid-edge
    arrowhead for the direction).  The specific typing of any one diagram
    (propagator a→b → Δ_xy, source i → K⁽²⁾, a → φ″, leg 1 → field) is a
    separate label-mapping table.

    ``model`` is a model dict (from :func:`load_theory`) or a theory-name
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

    ``model`` is a model dict (from :func:`load_theory`) or a theory name.
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
    lines = [f"theory : {model.get('name','?')!r}",
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

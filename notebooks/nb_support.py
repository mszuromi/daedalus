"""notebooks/nb_support.py — shared scaffolding for the pipeline demo notebooks.

Centralises the **load → run → plot** flow so every demo notebook is thin and
uniform regardless of its group (temporal / spatial × single / multi-field).
The four group templates (``notebooks/templates/``) and every
``*_sim_compare`` notebook import this module so the thematics are common.

    import nb_support as nb
    model, mod = nb.load_theory('kpz_1d')          # from theories/*.theory.py
    cfg = nb.Config(k=2, max_ell=1, spatial_grid=(-6, 6, 49))
    res = nb.run(model, cfg, mod)                  # k/ell/Dyson all here
    nb.plot_cumulant(res, cfg, model)              # adaptable, auto-dispatched

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


# ── Repo / theory discovery ──────────────────────────────────────────────────

def repo_root() -> str:
    """Walk up from this file (or cwd, in a bare notebook) until the
    ``pipeline/`` package is found.  Robust to running from any of the
    ``notebooks/`` subdirectories."""
    here = os.path.dirname(os.path.abspath(__file__)) \
        if '__file__' in globals() else os.path.abspath('')
    root = here
    while root != os.path.dirname(root):
        if os.path.isdir(os.path.join(root, 'pipeline')):
            return root
        root = os.path.dirname(root)
    return here


REPO_ROOT = repo_root()
THEORIES_DIR = os.path.join(REPO_ROOT, 'theories')
import sys as _sys
if REPO_ROOT not in _sys.path:
    _sys.path.insert(0, REPO_ROOT)


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
    spec.loader.exec_module(mod)
    return mod.build(), mod


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
    return [f['name'] for f in (model.get('physical_fields') or [])]


# ── Run configuration ────────────────────────────────────────────────────────

@dataclass
class Config:
    """Everything a demo notebook chooses, in one object.

    Leave a field ``None`` to inherit the theory file's ``METADATA`` /
    ``DEFAULT_FUNDAMENTAL`` default.
    """
    # ── what to compute ──
    k: Optional[int] = None                 # correlator order (1, 2, 3, …)
    max_ell: Optional[int] = None           # loop order (0=tree, 1, 2, …)
    external_fields: Optional[list] = None  # e.g. [('x', 1), ('x', 1)]
    fundamental: Optional[dict] = None       # numeric parameter overrides

    # ── temporal grid ──
    tau_max: Optional[float] = None
    tau_step: Optional[float] = None

    # ── spatial ──
    spatial_grid: Any = None                # array | (lo, hi, n) | None
    spatial_points: Any = None              # k≥3: (n_pts, k-1, 2) of (x_j, τ_j)

    # ── Dyson dressing (coupled unequal-D) ──
    dyson_order: Optional[int] = None       # None=leave model; int≥0=override
    reference_diffusion: Optional[float] = None

    # ── execution ──
    parallel: bool = False
    verbose: bool = False

    # ── plotting options (adaptable) ──
    show_orders: str = 'cumulative'         # 'cumulative' | 'incremental' | 'total'
    logy: bool = False
    components: Any = None                  # which (i,j)/slice to draw; None=auto
    figsize: Optional[tuple] = None
    title: Optional[str] = None
    save: Optional[str] = None              # path to savefig, or None

    def resolved_grid(self):
        """Materialise ``spatial_grid`` (accepts an ``(lo, hi, n)`` tuple)."""
        g = self.spatial_grid
        if g is None:
            return None
        if isinstance(g, tuple) and len(g) == 3:
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


def run(model: dict, cfg: Config, module=None) -> dict:
    """Resolve the config against the theory's defaults and call
    ``compute_cumulants``.  Handles temporal vs spatial, the spatial
    k≥3 event path, and the Dyson-order override.  Returns the result
    dict with ``cfg`` / ``model`` attached under ``'_cfg'`` / ``'_model'``."""
    from pipeline import compute_cumulants

    k = cfg.k if cfg.k is not None else (_meta(module, 'k_default', 2)
                                         if module else 2)
    max_ell = (cfg.max_ell if cfg.max_ell is not None
               else (_meta(module, 'ell_default', 0) if module else 0))
    ext = cfg.external_fields
    ext_is_explicit = ext is not None
    if ext is None and module is not None:
        ext = _meta(module, 'recommended_external_fields')
    # Auto-build a k-matching external_fields when none usable was given:
    # k copies of the first physical field's leg.  Triggers when ext is
    # absent, the wrong length for k, or (for a METADATA *recommendation*
    # only) names a field the model doesn't have — so a stale recommended
    # list never crashes the run.  An explicit Config.external_fields is
    # respected verbatim (it may legitimately name a response leg).
    valid = set(field_names(model))
    def _nm(e):
        return e[0] if isinstance(e, (tuple, list)) else e
    if (ext is None or len(ext) != k
            or (not ext_is_explicit
                and not all(_nm(e) in valid for e in ext))):
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
    if cfg.fundamental:
        fundamental.update(cfg.fundamental)

    # Dyson override: inject into the model's spatial policy at run time.
    if cfg.dyson_order is not None and model.get('spatial'):
        model['spatial']['dyson'] = {'mode': 'fixed',
                                     'order': int(cfg.dyson_order)}
        if cfg.reference_diffusion is not None:
            model['spatial']['reference_diffusion'] = \
                float(cfg.reference_diffusion)

    kw = dict(model=model, k=k, max_ell=max_ell, fundamental=fundamental,
              external_fields=ext, parallel=cfg.parallel, verbose=cfg.verbose)

    if is_spatial(model):
        if k != 2 and cfg.spatial_points is None:
            raise ValueError(
                'spatial k≥3 needs Config.spatial_points = (n_pts, k-1, 2) '
                'array of (x_j, τ_j) offsets per non-anchor external slot.')
        if k != 2:
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

    res = compute_cumulants(**kw)
    res['_cfg'] = cfg
    res['_model'] = model
    res['_resolved'] = dict(k=k, max_ell=max_ell, external_fields=ext)
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
      * spatial k≥3 events  → :func:`plot_kpoint`
      * spatial k=2         → :func:`plot_spatial`
      * temporal            → :func:`plot_temporal`

    ``sim`` (optional) overlays a matched simulator: a dict with
    ``tau``/``C``/``C_err`` (temporal) or ``x``/``C``/``C_err`` (spatial).
    Returns the Matplotlib ``Figure``.
    """
    cfg = cfg or result.get('_cfg') or Config()
    model = model or result.get('_model') or {}
    if 'C_kpoint' in result:
        return plot_kpoint(result, cfg, model)
    if is_spatial(model) or 'C_tau_x' in result:
        return plot_spatial(result, cfg, model, sim)
    return plot_temporal(result, cfg, model, sim)


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
        if se is not None:
            ax.errorbar(st, sc, yerr=np.asarray(se), fmt='o', ms=3,
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


def plot_spatial(result, cfg, model, sim=None):
    """Equal-time C(x,0) slice + per-loop overlay, and a C(x,τ) heatmap
    when a τ grid is present (the spatial group plot)."""
    cfg = cfg or Config()
    C = np.real(np.asarray(result['C_tau_x']))
    xs = np.asarray(result['spatial_grid'])
    tau = np.asarray(result['tau_grid'])
    i0 = int(np.argmin(np.abs(tau)))
    si = result.get('spatial_info') or {}
    by_order = si.get('C_by_order')          # {ell: (n_tau, n_x)} cumulative

    ncol = 2 if len(tau) > 1 else 1
    fig, axes = plt.subplots(1, ncol, figsize=cfg.figsize or (5.6 * ncol, 4.3))
    ax0 = axes[0] if ncol > 1 else axes

    if by_order and cfg.show_orders != 'total':
        for i, ell in enumerate(sorted(by_order)):
            c = np.real(np.asarray(by_order[ell]))
            c = c[i0] if c.ndim == 2 else c
            ax0.plot(xs, c, '-', lw=1.8,
                     color=_ORDER_COLORS[i % len(_ORDER_COLORS)],
                     label=f'theory: {_order_label(ell)}')
    else:
        ax0.plot(xs, C[i0], '-', lw=1.8, color='#1F9FCC', label='theory')
    if sim is not None:
        sx, sc = np.asarray(sim['x']), np.asarray(sim['C'])
        se = sim.get('C_err')
        if se is not None:
            ax0.errorbar(sx, sc, yerr=np.asarray(se), fmt='o', ms=3,
                         color='#222', alpha=0.6, capsize=2, label='sim')
        else:
            ax0.plot(sx, sc, 'o', ms=3, color='#222', alpha=0.6, label='sim')
    ax0.set_xlabel('x')
    ax0.set_ylabel(r'$C(x,\,0)$')
    if cfg.logy:
        ax0.set_yscale('log')
    ax0.set_title(cfg.title or f"{model.get('name','')}: equal-time C(x,0)")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8)

    if ncol > 1:
        im = axes[1].imshow(C, aspect='auto', origin='lower',
                            extent=[xs.min(), xs.max(), tau.min(), tau.max()],
                            cmap='viridis')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel(r'$\tau$')
        axes[1].set_title(r'$C(x,\tau)$')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
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

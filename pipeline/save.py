"""
pipeline.save — NPZ + CSV serialization of compute_cumulants results.

The schema is k-/ell-/model-adaptive:

* **Per-loop-order curves**: one entry per ell from 0 to ``max_ell``.
  Tree (ell=0) is keyed ``C_tree``; loops are keyed ``C_<n>_loop``.
  ``C_total`` is the sum over all ells.

* **Mean-field array**: whatever quantities ``compute_cumulants``
  populated under ``result['mf_values']`` get one ``mf_<name>`` entry
  per quantity, plus a ``mf_keys`` index array.  Hawkes models give
  ``nstar``, ``vstar``; GTaS adds ``mstar``; future models can add
  more without changing this code.

* **Parameter metadata**: keys of ``fundamental`` saved as
  ``parameter_keys`` for cross-reference.

NPZ shape conventions
---------------------
For ``k ∈ {1, 2}``: each ``C_*`` is a 1D complex array whose shape
matches ``tau_grid``.  For k=1 the array is constant (rate is τ-
independent) — saved on the grid for shape consistency with k=2.

For k ≥ 3: ``compute_cumulants`` returns ``C_tau = None`` (caller
evaluates ``total_C`` themselves), so the saved ``C_*`` arrays are
empty.  ``tau_grid`` is still saved so callers know the grid the
pipeline would have used.
"""
from __future__ import annotations

import hashlib
import os
import re
import warnings
from typing import Any

import numpy as np


# ── filesystem-safe parameter slugs ───────────────────────────────────
#
# Output paths must never be built from ``str(params_dict)``.  A Sage /
# sympy substitution dict stringifies to ``{Em1: 1.0, tau1: 10.0, ...}``
# (symbol keys → no quotes) or ``{'mu': 0.1, ...}`` — full of braces,
# colons and spaces.  Used as a directory name that produces junk dirs
# like ``{Em1: 1.0, ..., vstar2: 1.0}/`` at the repo root (one per
# working point), which is exactly what the ``{*}/`` ``.gitignore`` rule
# was added to paper over.
#
# ``params_slug`` is the canonical, stable, readable encoding: keys
# sorted by ``str(key)``, each pair rendered ``key=value`` with floats
# truncated, pairs joined by ``__``.  ``_sanitize_output_path`` is the
# defensive backstop wired into every saver below: if a path component
# ever arrives as a stringified dict repr, it is slugified in place (with
# a warning) so no ``{...}/`` directory is ever materialised.

# Characters allowed verbatim in a path component: alphanumerics plus a
# small set that every common filesystem accepts.
_SAFE_RE = re.compile(r'[^A-Za-z0-9._=+-]+')


def _sanitize_token(s: Any) -> str:
    """Collapse any run of filesystem-unsafe characters to ``_``."""
    return _SAFE_RE.sub('_', str(s)).strip('_')


def _fmt_value(v: Any) -> str:
    """Compact, readable rendering of a single parameter value.

    Integers stay integers (``10.0 → '10'``); other floats use 6 sig
    figs (``0.1 → '0.1'``, ``6.70732 → '6.70732'``); list / array values
    flatten to ``a-b-c``; everything else is sanitized via ``str``.
    """
    if isinstance(v, (list, tuple, np.ndarray)):
        flat = np.asarray(v, dtype=object).ravel()
        return '-'.join(_fmt_value(x) for x in flat) or '0'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _sanitize_token(v)
    if np.isfinite(f) and f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return _sanitize_token(f'{f:.6g}')


def params_slug(params: Any, *, max_len: int = 180) -> str:
    """Return a filesystem-safe, deterministic slug for a parameter dict.

    Keys are sorted by ``str(key)`` (so symbol- and string-keyed dicts
    slugify identically and stably), each pair is rendered ``key=value``
    with floats truncated, and pairs are joined by ``__``.  Overlong
    slugs are truncated and disambiguated with a short content hash.

    >>> params_slug({'mu': 0.1, 'eps': 0.1, 'D': 1.0})
    'D=1__eps=0.1__mu=0.1'
    """
    if not isinstance(params, dict):
        return _sanitize_token(params) or 'params'
    parts = [f'{_sanitize_token(k)}={_fmt_value(params[k])}'
             for k in sorted(params, key=str)]
    slug = _SAFE_RE.sub('_', '__'.join(parts))
    if len(slug) > max_len:
        digest = hashlib.sha1(slug.encode('utf-8')).hexdigest()[:8]
        slug = slug[:max_len - 9].rstrip('_') + '_' + digest
    return slug or 'params'


def _looks_like_dict_repr(component: str) -> bool:
    """True if a path component is a stringified Python dict repr."""
    return (len(component) >= 2 and component[0] == '{'
            and component[-1] == '}' and ':' in component)


def _slug_from_dict_repr(component: str) -> str:
    """Recover a stable slug from an already-stringified dict repr such as
    ``{Em1: 1.0, tau1: 10.0}`` (symbol keys) or ``{'mu': 0.1}``.

    Falls back to a plain sanitize of the whole component if the repr
    does not parse as flat ``key: value`` pairs (e.g. nested commas)."""
    inner = component[1:-1].strip()
    if not inner:
        return 'params'
    pairs: dict[str, str] = {}
    for chunk in inner.split(', '):
        if ': ' not in chunk:
            return _sanitize_token(component)
        key, val = chunk.split(': ', 1)
        pairs[key.strip().strip('\'"')] = val.strip()
    return params_slug(pairs)


def _sanitize_output_path(path: str) -> str:
    """Slugify any directory component that is a leaked dict repr.

    This is the backstop that stops a junk ``{Em1: 1.0, ...}/`` directory
    from ever being created: if a caller builds an output path from
    ``str(params_dict)`` instead of :func:`params_slug`, the offending
    component is rewritten in place (with a warning) before the path is
    used to create directories or files.
    """
    comps = str(path).split(os.sep)
    fixed = [(_slug_from_dict_repr(c) if _looks_like_dict_repr(c) else c)
             for c in comps]
    if fixed == comps:
        return path
    new_path = os.sep.join(fixed)
    warnings.warn(
        f'save path component looked like a stringified params dict; '
        f'slugified to avoid a junk directory: {path!r} -> {new_path!r}. '
        f'Build output paths with pipeline.save.params_slug(...) instead '
        f'of str(params).',
        stacklevel=3,
    )
    return new_path


# ── helpers ───────────────────────────────────────────────────────────

def _ell_key(ell: int) -> str:
    """0 → 'C_tree', 1 → 'C_1_loop', 2 → 'C_2_loop', ..."""
    return 'C_tree' if ell == 0 else f'C_{int(ell)}_loop'


def _curve_or_empty(arr) -> np.ndarray:
    """Coerce a per-ell curve into an NPZ-friendly array.  ``None``
    becomes an empty complex array (k≥3 case)."""
    if arr is None:
        return np.array([], dtype=complex)
    return np.asarray(arr, dtype=complex)


def _format_value(v: Any) -> str:
    """Format a parameter / mf value for the CSV header comment line.
    Lists / arrays become ``[a, b, c]``; scalars use 6-sig-fig repr."""
    try:
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if arr.dtype.kind in 'fc':
                return np.array2string(arr, formatter={'float_kind':
                                                       lambda x: f'{x:.6g}'})
            return repr(list(v))
        if isinstance(v, float):
            return f'{v:.6g}'
        return repr(v)
    except Exception:
        return repr(v)


# ── NPZ ───────────────────────────────────────────────────────────────

def save_npz(result: dict, path: str, extra: dict | None = None) -> str:
    """Serialize a ``compute_cumulants`` result dict to ``.npz``.

    Parameters
    ----------
    result : dict
        The dict returned by ``pipeline.compute_cumulants``.  Must
        contain ``tau_grid``, ``C_tau``, ``C_tau_by_ell``,
        ``mf_values``, ``config``.
    path : str
        Output ``.npz`` path.  Parent directory is created if missing.
    extra : dict or None
        Optional ``{key: array}`` mapping merged into the payload after
        the pipeline schema is built — useful for notebooks that want
        to colocate simulation sidecar data (e.g. ``C_sim_mean``,
        ``rates_sim_mean``) in the same file.  Pipeline keys are not
        protected — passing ``'C_total'`` in ``extra`` will overwrite
        the theoretical curve.

    Returns
    -------
    str
        The output path (echoed for convenience).
    """
    path = _sanitize_output_path(path)

    cfg     = result['config']
    k       = int(cfg['k'])
    max_ell = int(cfg['max_ell'])

    payload: dict[str, np.ndarray] = {
        'k':              np.array([k],       dtype=int),
        'max_ell':        np.array([max_ell], dtype=int),
        'tau_grid':       np.asarray(result['tau_grid'], dtype=float),
        'C_total':        _curve_or_empty(result.get('C_tau')),
    }

    # Per-loop-order curves: C_tree, C_1_loop, C_2_loop, ...
    C_tau_by_ell = result.get('C_tau_by_ell', {}) or {}
    for ell in sorted(C_tau_by_ell.keys()):
        payload[_ell_key(ell)] = _curve_or_empty(C_tau_by_ell[ell])

    # Mean-field — adaptive, one entry per quantity the model declared
    mf_values = result.get('mf_values', {}) or {}
    mf_keys = []
    for name in mf_values:
        vals = mf_values[name]
        if vals is None:
            continue
        payload[f'mf_{name}'] = np.asarray(vals, dtype=float)
        mf_keys.append(name)
    payload['mf_keys'] = np.asarray(mf_keys)

    # Parameter metadata — keys + per-name values.  Scalars stay
    # scalar; vectors / matrices retain their shape.  Saving each
    # parameter under ``param_<name>`` makes ``np.load(...)['param_E']``
    # round-trip the user's input directly.
    fundamental = cfg.get('fundamental', {}) or {}
    payload['parameter_keys'] = np.asarray(list(fundamental.keys()))
    for pname, pval in fundamental.items():
        try:
            payload[f'param_{pname}'] = np.asarray(pval, dtype=float)
        except (TypeError, ValueError):
            # Non-numeric (rare) — fall back to dtype=object so it
            # still round-trips even if numpy can't unify the dtype.
            payload[f'param_{pname}'] = np.asarray(pval, dtype=object)

    # Cross-reference metadata
    payload['model_name']   = np.asarray([cfg.get('model_name', '')])
    payload['ext_field_kinds'] = np.asarray(
        [ef[0] for ef in cfg.get('external_fields', []) or []])
    payload['ext_field_pops']  = np.asarray(
        [int(ef[1]) for ef in cfg.get('external_fields', []) or []],
        dtype=int)

    # Merge user-supplied extras (sim sidecars etc.) — last writer wins
    if extra:
        for key, val in extra.items():
            payload[key] = np.asarray(val)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, **payload)
    return path


# ── CSV ───────────────────────────────────────────────────────────────

def save_csv(result: dict, path: str) -> str:
    """Serialize a ``compute_cumulants`` result dict to a wide-format
    CSV with parameter / mean-field metadata as ``#``-comment lines.

    Layout::

        # k=2, max_ell=1, model='Quadratic Hawkes ...'
        # parameters:
        #   E: [0.78, 0.81]
        #   ...
        # mean-field:
        #   nstar: [0.5236, 0.6151]
        #   ...
        tau,C_tree.real,C_tree.imag,C_1_loop.real,C_1_loop.imag,C_total.real,C_total.imag
        -50.000000,...

    For k ≥ 3 (``C_tau is None``) only the header comments are written
    — the τ-grid table is empty.

    Parameters
    ----------
    result : dict
        Same dict consumed by ``save_npz``.
    path : str
        Output ``.csv`` path.

    Returns
    -------
    str
        The output path.
    """
    path = _sanitize_output_path(path)

    cfg     = result['config']
    k       = int(cfg['k'])
    max_ell = int(cfg['max_ell'])

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    tau_grid     = np.asarray(result['tau_grid'], dtype=float)
    C_tau        = result.get('C_tau')
    C_tau_by_ell = result.get('C_tau_by_ell', {}) or {}

    with open(path, 'w') as f:
        # Header comments — context for the saved curves
        f.write(f'# k={k}, max_ell={max_ell}, '
                f'model={cfg.get("model_name", "")!r}\n')

        ext = cfg.get('external_fields', []) or []
        f.write(f'# external_fields: {ext}\n')

        f.write('# parameters:\n')
        for name, val in (cfg.get('fundamental') or {}).items():
            f.write(f'#   {name}: {_format_value(val)}\n')

        f.write('# mean-field:\n')
        for name, vals in (result.get('mf_values') or {}).items():
            if vals is None:
                continue
            f.write(f'#   {name}: {_format_value(vals)}\n')

        f.write(f'# tau_grid: {len(tau_grid)} points '
                f'in [{tau_grid.min():.4g}, {tau_grid.max():.4g}]\n')

        # Build column list
        col_names = ['tau']
        col_data: list[np.ndarray] = [tau_grid]

        # Per-ell curves first (tree, 1_loop, 2_loop, ...)
        for ell in sorted(C_tau_by_ell.keys()):
            curve = C_tau_by_ell[ell]
            if curve is None:
                continue
            base = _ell_key(ell)
            arr = np.asarray(curve, dtype=complex)
            col_names.extend([f'{base}.real', f'{base}.imag'])
            col_data.extend([arr.real, arr.imag])

        # Total at the end
        if C_tau is not None:
            arr = np.asarray(C_tau, dtype=complex)
            col_names.extend(['C_total.real', 'C_total.imag'])
            col_data.extend([arr.real, arr.imag])

        # Write header + rows
        f.write(','.join(col_names) + '\n')
        if len(col_data) > 1 and len(col_data[0]) > 0:
            n_rows = len(col_data[0])
            for i in range(n_rows):
                row = [f'{col_data[j][i]:.10g}'
                       for j in range(len(col_data))]
                f.write(','.join(row) + '\n')

    return path

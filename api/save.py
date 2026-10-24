"""
api.save — NPZ + CSV serialization of compute_cumulants results.

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

* **Provenance stamp** (since 0.2.0): ``daedalus_version`` (the package
  version that wrote the file), ``phase_j_convention`` (the Phase J
  equal-time rule the numbers were computed with; ``PHASE_J_CONVENTION``
  for the default rules) and ``phase_j_flags`` (M7: the value of every
  ``final_integral`` switch that can move a Phase J number, as
  ``NAME=value`` pairs joined by ``;``; files written by 0.2.1 and
  earlier also list the retired ``grouped_integral`` switch and
  ``use_grouped_phase_j``;
  recorded for provenance only, :func:`load_npz` does not check
  it).  ``compute_cumulants`` records that rule in
  ``result['config']['phase_j_convention']`` when it computes the numbers,
  and the stamp copies it, so changing the Phase J flags between computing
  and saving does not mislabel a file; only a result dict without that
  entry (e.g. hand-built) is stamped with the rule in force when the file
  is written.  The NPZ stores each as a one-element string array; the CSV
  writes them as ``#`` header lines.  :func:`load_npz` warns
  (:class:`StaleResultWarning`, once per file and process) when the stamp
  is missing, older than 0.2.0, or names the pre-0.2.0 convention.

* **Route and assumptions** (D2, written when ``compute_cumulants`` recorded
  them, i.e. with the model analyzer on): ``route`` (the integrator backend
  that computed the numbers, ``api.routing``), ``assumptions`` (the
  analyzer's assumptions ledger: every consent, convention, assertion and
  approximation the numbers depend on) and ``analyzer_findings`` (the
  findings' codes and levels), the last two as JSON text.  Files written
  by 0.2.0, or with the analyzer off, lack them and load unchanged;
  :func:`load_provenance` reads them from any file (``None`` when absent).

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
import sys
import warnings
from typing import Any

import numpy as np


# ── provenance stamp (daedalus_version / phase_j_convention) ──────────
#
# 0.2.0 changed a number-producing convention for the constant constraint
# rows that δ-elimination leaves (a smooth propagator whose time difference
# is a constant; CHANGELOG.md, 0.2.0):
#
# * a row whose value is exactly 0 and does not depend on the external
#   times is weighted Θ(0) = 0 (Itô) in every Phase J integrator; before
#   0.2.0 the two-time polygon integrator kept it (Θ(0) = 1).  This moves
#   numbers at every τ;
# * a row that compares two external times that coincide exactly is decided
#   by the external-time tie order (exactly one orientation holds), so a
#   k ≥ 3 cumulant at coincident external times is a one-sided limit.  This
#   moves such points, e.g. the diagonals of ``dd.run``'s full k-point grid
#   and the coincident non-swept legs of its k ≥ 4 slices.  (The
#   later-listed leg counts as infinitesimally earlier, so at coincident
#   times of legs of different fields the value depends on the order of
#   ``external_fields``, as the k = 2 τ = 0 point has since 0.1.0.)
# * a point of a ``dd.run`` k ≥ 3 slice where the swept leg meets another
#   leg (e.g. τ = 0) is now the left limit of its slice: the swept leg is
#   placed one ``_ITO_EPS`` below the leg it meets
#   (``daedalus._kpoint_slice_times``), so those points moved too, and so
#   did the k ≥ 3 moment outputs (``Config.output = 'moment'`` or
#   ``'central_moment'``), now evaluated at the times of slice 1.
#   (Since Oct 2026 those points are evaluated at the exact tie instead,
#   ``daedalus._kpoint_slice_point``: every k ≥ 3 slice point moves by
#   O(rate x 1e-6) once more.)
#
# Saved results of models with δ / instantaneous propagator parts may
# therefore differ from a fresh computation.  Every saved file carries the
# version and the convention it was computed with.

#: Label of the default Phase J convention of this version (the 0.2.0
#: constant-row rules: Θ(0) = 0 for a zero row that does not depend on the
#: external times, the external-time tie order for coincident times).
PHASE_J_CONVENTION = 'theta0_ito_const_rows'

# The retired ``final_integral.THETA0_CONST_ROW_MODE`` -> the
# stamp label it wrote; 'legacy_clip' files (``DAEDALUS_PHASE_J_LEGACY=1``
# etc., before the retirement) still load, with a warning.
_PHASE_J_CONVENTION_BY_MODE = {
    'ito':         PHASE_J_CONVENTION,
    'legacy_clip': 'theta0_legacy_clip',
}
# Labels whose numbers follow the pre-0.2.0 rule (a load warns).
_LEGACY_PHASE_J_CONVENTIONS = frozenset({'theta0_legacy_clip'})

# Oldest package version whose saved numbers use the current convention.
_STAMP_MIN_VERSION = (0, 2, 0)

_FI_MODULE = 'engine.integration.time_domain.final_integral'


from engine._warn_helpers import (                    # noqa: E402
    ensure_warning_shown as _ensure_warning_shown)


class StaleResultWarning(UserWarning):
    """A loaded result file predates (or opted out of) the 0.2.0 Phase J
    equal-time convention and may need to be recomputed."""


_INIT_VERSION_RE = re.compile(r"""^__version__\s*=\s*['"]([^'"]+)['"]""",
                              re.M)


def _daedalus_version() -> str:
    """The running package version (``api.__version__``), read at call time.

    ``api/__init__`` imports this module before it defines ``__version__``,
    so the value is looked up lazily.  When the package is not imported
    (this module loaded standalone, e.g. by a numpy-only test), it is read
    from the ``api/__init__.py`` source next to this file instead of
    importing the Sage-heavy package.  ``'unknown'`` if neither works.
    """
    v = getattr(sys.modules.get('api'), '__version__', None)
    if v is None:
        try:
            init_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '__init__.py')
            with open(init_py, encoding='utf-8') as f:
                m = _INIT_VERSION_RE.search(f.read())
            v = m.group(1) if m else None
        except OSError:
            v = None
    return str(v) if v else 'unknown'


def _phase_j_convention() -> str:
    """Label of the Phase J convention in force: ``PHASE_J_CONVENTION``, the
    only one since the 'legacy_clip' Θ(0) mode was retired (the
    labels of ``_LEGACY_PHASE_J_CONVENTIONS`` remain for loading files
    written before)."""
    return PHASE_J_CONVENTION


#: The ``final_integral`` switches and tolerances that can move a Phase J
#: number, in stamp order (M7; the memo flags are pure speed switches and
#: are not listed).
_PHASE_J_STAMP_FLAGS = (
    'PHASE_J_T_MIN', 'POSET_PHYSICAL_MARGIN', 'USE_POSET_CAP_MATCH_SCIPY',
    'USE_POSET_MPMATH_ACCUMULATION', 'USE_POLYGON_M2_INTEGRATOR',
    'USE_1D_INTEGRATOR', 'USE_POSET_INTEGRATOR', 'TAU_KERNEL_CAP',
    'QUAD_OPTS', 'NQUAD_EPSABS_FACTOR', 'NQUAD_EPSREL', 'NQUAD_LIMIT',
    'NQUAD_TAIL_K', 'NQUAD_UNCERTIFIED_CAP')
# (The retired switches ``THETA0_CONST_ROW_MODE``,
# ``STRUCTURAL_ZEROS``, ``NQUAD_HARDENED``, ``USE_DBM_FALLBACK``,
# ``USE_PHASE_J_DD_KERNELS``, ``CHAIN_UPPERS_KERNEL``, ``SEMIINF_CHAIN``,
# ``UNBOUNDED_2D``, ``USE_SHOTNOISE_INTEGRATED`` and the boxed-route
# constants ``POLYGON_BBOX_CAP``, ``USE_CHAIN_SIMPLEX_PRECISION_FIX``,
# ``USE_NUMBA_CHAIN_SIMPLEX``; a file written before carries them in its
# ``phase_j_flags``.)


def _phase_j_flags() -> str:
    """The Phase J flags in force (call time) as ``NAME=value;...``
    (``_PHASE_J_STAMP_FLAGS``), read from ``final_integral`` when it is
    loaded; ``'final_integral not loaded'``
    otherwise (the stamp then only says that no Phase J ran in this
    process)."""
    fi = sys.modules.get(_FI_MODULE)
    if fi is None:
        return 'final_integral not loaded'
    parts = []
    for name in _PHASE_J_STAMP_FLAGS:
        v = getattr(fi, name, '<absent>')
        if isinstance(v, bool):
            v = int(v)
        parts.append(f'{name}={v}')
    return ';'.join(parts)


def _analysis_stamp(result: dict | None) -> dict[str, str]:
    """``{'route': ..., 'assumptions': <json>, 'analyzer_findings':
    <json>}`` from a ``compute_cumulants`` result, each entry only when the
    result recorded it (``config['route']`` / ``config['assumptions']`` /
    ``result['analysis']``)."""
    import json
    out: dict[str, str] = {}
    if not isinstance(result, dict):
        return out
    cfg = result.get('config') if isinstance(result.get('config'), dict) else {}
    route = cfg.get('route') or (result.get('route') or {}).get('backend')
    if route:
        out['route'] = str(route)
    led = cfg.get('assumptions')
    if led is not None:
        out['assumptions'] = json.dumps(led, sort_keys=True, default=str)
    rep = result.get('analysis')
    if rep is not None and hasattr(rep, 'findings'):
        out['analyzer_findings'] = json.dumps(
            [{'code': f.code, 'level': f.level} for f in rep.findings],
            sort_keys=True)
    if result.get('status') == 'incomplete':
        # diagrams Phase J could not evaluate are missing from the numbers
        # (``result['phase_j_incomplete']``); written only then, so a
        # complete result's file is unchanged
        out['phase_j_status'] = json.dumps(
            {str(ell): [r.get('diagram') for r in recs] for ell, recs in
             sorted((result.get('phase_j_incomplete') or {}).items())},
            sort_keys=True, default=str)
    return out


def load_provenance(data: dict) -> dict:
    """The provenance of a loaded file (``load_npz``'s dict): the stamp
    entries as strings, ``assumptions`` and ``analyzer_findings`` parsed
    from JSON; ``None`` for every entry the file does not carry (files
    written by 0.2.0 have no route or assumptions)."""
    import json
    out = {}
    for key in ('daedalus_version', 'phase_j_convention', 'phase_j_flags',
                'route'):
        out[key] = _stamp_value(data, key)
    for key in ('assumptions', 'analyzer_findings', 'phase_j_status'):
        raw = _stamp_value(data, key)
        try:
            out[key] = json.loads(raw) if raw else None
        except ValueError:
            out[key] = None
    return out


def _result_stamp(result: dict | None = None) -> dict[str, str]:
    """``{'daedalus_version': ..., 'phase_j_convention': ...,
    'phase_j_flags': ...}``, plus ``route`` / ``assumptions`` /
    ``analyzer_findings`` when the result recorded them
    (:func:`_analysis_stamp`).

    The convention and the flags are the ones ``compute_cumulants`` recorded
    in ``result['config']`` when it computed the numbers; for a result
    without those entries, the ones in force now (call time).  The version
    is the running package's (the writer's)."""
    conv = None
    flags = None
    if isinstance(result, dict):
        cfg = result.get('config')
        if isinstance(cfg, dict):
            conv = cfg.get('phase_j_convention')
            flags = cfg.get('phase_j_flags')
    if not conv:
        conv = _phase_j_convention()
    if not flags:
        flags = _phase_j_flags()
    stamp = {'daedalus_version': _daedalus_version(),
             'phase_j_convention': str(conv),
             'phase_j_flags': str(flags)}
    stamp.update(_analysis_stamp(result))
    return stamp


_VERSION_RE = re.compile(r'^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?')


def _parse_version(v: Any) -> tuple[int, int, int] | None:
    """``'0.2.0'`` -> ``(0, 2, 0)``; a missing minor / patch counts as 0 and
    a pre-release / local suffix is ignored.  ``None`` if unparseable."""
    m = _VERSION_RE.match(str(v))
    if not m:
        return None
    return tuple(int(g) if g is not None else 0 for g in m.groups())


def _stamp_value(data: Any, key: str) -> str | None:
    """A stamp entry as ``str`` (``None`` when absent or empty).  NPZ stores
    it as a one-element string array; a plain string is accepted too."""
    try:
        raw = data[key]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    arr = np.asarray(raw).ravel()
    if arr.size == 0:
        return None
    s = str(arr[0]).strip()
    return s or None


def _stale_reason(version: str | None, convention: str | None) -> str | None:
    """Why a saved result may be stale under the current convention, or
    ``None`` when its stamp is current.  An unknown (newer) convention
    label is not flagged."""
    if version is None:
        return 'it has no daedalus_version stamp (written before 0.2.0)'
    parsed = _parse_version(version)
    if parsed is None:
        return f'its daedalus_version stamp {version!r} is not a version'
    if parsed < _STAMP_MIN_VERSION:
        return f'it was written by daedalus {version}'
    if convention is None:
        return 'it has no phase_j_convention stamp'
    if convention in _LEGACY_PHASE_J_CONVENTIONS:
        return (f'it was computed with the pre-0.2.0 Phase J convention '
                f'{convention!r} (legacy flags)')
    return None


# (source, reason) pairs already warned about in this process.
_STALE_WARNED: set[tuple[str, str]] = set()


def _warn_if_stale(version: str | None, convention: str | None,
                   source: str, *, stacklevel: int = 3) -> bool:
    """Emit one :class:`StaleResultWarning` per (source, reason) and
    process; return True when the stamp is stale (warned now or before)."""
    reason = _stale_reason(version, convention)
    if reason is None:
        return False
    key = (str(source), reason)
    if key in _STALE_WARNED:
        return True
    _STALE_WARNED.add(key)
    _msg = (
        f'{source}: {reason}.  Daedalus 0.2.0 changed how Phase J decides '
        f'the constant constraint rows left by δ-elimination: a row whose '
        f'value is exactly 0 without depending on the external times is '
        f'weighted Θ(0) = 0 (Itô) in every integrator (earlier versions '
        f'used Θ(0) = 1 in the two-time polygon integrator), and a row '
        f'comparing two coincident external times follows the external-time '
        f'tie order, so k ≥ 3 values at coincident times are one-sided '
        f'limits; the points of dd.run\'s k ≥ 3 slices where the swept leg '
        f'meets another leg (e.g. τ = 0) are now the left limit of their '
        f'slice, and the k ≥ 3 moment outputs follow slice 1.  It also '
        f'integrates every Phase J region exactly, without an integration '
        f'box, which moves models with slow modes.  Results for models with '
        f'δ (instantaneous) propagator parts or slow modes may be stale and '
        f'should be recomputed; see CHANGELOG.md (0.2.0).')
    _ensure_warning_shown(StaleResultWarning, 'always', message=_msg,
                          stacklevel=stacklevel)   # once per key above
    warnings.warn(_msg, StaleResultWarning, stacklevel=stacklevel)
    return True


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
        f'Build output paths with api.save.params_slug(...) instead '
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


def _delta_table(result: dict) -> dict | None:
    """The ``delta_components`` table of a ``compute_cumulants`` result
    (Feature B, :func:`api.cumulant.delta_table`), or ``None`` when the
    result carries no Phase J records (spatial, k = 1 without records, a
    hand-made dict)."""
    if not isinstance(result, dict) or 'phase_j_by_ell' not in result:
        return None
    from api.cumulant import delta_table
    try:
        return delta_table(result)
    except Exception as exc:                      # noqa: BLE001
        # the curves are still worth saving; say why the table is missing
        warnings.warn(f'delta_components table not written: '
                      f'{type(exc).__name__}: {exc}', stacklevel=3)
        return None


def _delta_payload(tab: dict) -> dict[str, np.ndarray]:
    """NPZ arrays of a δ table: ``delta_weight`` (complex; NaN when the
    weight depends on other times), ``delta_location`` (k = 2 τ, else NaN),
    ``delta_ell``, ``delta_diagram``, ``delta_equality``,
    ``delta_provenance`` (JSON) and ``delta_complete``."""
    return {
        'delta_weight': np.asarray(tab['weight'], dtype=complex),
        'delta_location': np.asarray(tab['location'], dtype=float),
        'delta_ell': np.asarray(tab['ell'], dtype=int),
        'delta_diagram': np.asarray(tab['diagram'], dtype=str),
        'delta_equality': np.asarray(tab['equality'], dtype=str),
        'delta_provenance': np.asarray(tab['provenance'], dtype=str),
        'delta_complete': np.asarray([bool(tab['complete'])]),
    }


# ── NPZ ───────────────────────────────────────────────────────────────

def _spectra_payload(sp):
    """``spectra_*`` arrays of ``result['spectra']`` (:mod:`api.spectra`):
    ``spectra_ells``; per quantity the total (``spectra_<name>``) and the
    loop orders stacked in the order of ``spectra_ells``
    (``spectra_<name>_by_ell``: shape ``(n_ell,)`` for the numbers,
    ``(n_ell, n)`` over ``spectra_omega`` / ``spectra_T``);
    ``spectra_status``, ``spectra_conventions``; a refused request keeps
    ``spectra_reason`` only."""
    out = {'spectra_status': np.asarray([str(sp.get('status', ''))]),
           'spectra_conventions': np.asarray([str(sp.get('conventions', ''))])}
    if sp.get('status') != 'ok':
        out['spectra_reason'] = np.asarray([str(sp.get('reason', ''))])
        return out
    ells = sorted(sp.get('C0_by_ell') or {})
    out['spectra_ells'] = np.asarray(ells, dtype=int)
    out['spectra_method_by_ell'] = np.asarray(
        [str((sp.get('method_by_ell') or {}).get(e, '')) for e in ells])
    for name in ('S0', 'integral_inf', 'integral_inf_neg', 'delta0', 'C0',
                 'tau_c'):
        out[f'spectra_{name}'] = np.asarray([sp[name]], dtype=complex)
        out[f'spectra_{name}_by_ell'] = np.asarray(
            [sp[f'{name}_by_ell'][e] for e in ells], dtype=complex)
    for grid, names in (('omega', ('S',)), ('T', ('integral', 'integral_neg'))):
        if sp.get(grid) is None:
            continue
        out[f'spectra_{grid}'] = np.asarray(sp[grid], dtype=float)
        for name in names:
            out[f'spectra_{name}'] = np.asarray(sp[name], dtype=complex)
            out[f'spectra_{name}_by_ell'] = np.asarray(
                [sp[f'{name}_by_ell'][e] for e in ells], dtype=complex)
    return out


def save_npz(result: dict, path: str, extra: dict | None = None) -> str:
    """Serialize a ``compute_cumulants`` result dict to ``.npz``.

    Parameters
    ----------
    result : dict
        The dict returned by ``api.compute_cumulants``.  Must
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
        the theoretical curve.  The provenance stamp
        (``daedalus_version``, ``phase_j_convention``, ``phase_j_flags``)
        is written last
        and cannot be overridden through ``extra``.

    Returns
    -------
    str
        The output path (echoed for convenience).
    """
    if isinstance(result, dict) and result.get('t_min') is not None:
        raise NotImplementedError(
            'a switch-on result (t_min) is not saved yet: save '
            "result['times'] and result['C_times'] instead")
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

    # Feature B: the shot-noise δ components (not in C_total; see
    # api.cumulant).  Absent for results without Phase J records.
    tab = _delta_table(result)
    if tab is not None:
        payload.update(_delta_payload(tab))

    # the exact spectral outputs (``compute_cumulants(spectra=)``)
    if isinstance(result.get('spectra'), dict):
        payload.update(_spectra_payload(result['spectra']))

    # Merge user-supplied extras (sim sidecars etc.) — last writer wins
    if extra:
        for key, val in extra.items():
            payload[key] = np.asarray(val)

    # Provenance stamp — after the extras so it always describes THIS
    # computation (a re-save must not inherit an old file's stamp).
    for key, val in _result_stamp(result).items():
        payload[key] = np.asarray([val])

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, **payload)
    return path


def load_npz(path: str, *, allow_pickle: bool = False,
             warn_stale: bool = True) -> dict[str, np.ndarray]:
    """Load a ``.npz`` written by :func:`save_npz` into a plain dict.

    Checks the provenance stamp: when ``daedalus_version`` is missing or
    older than 0.2.0, or ``phase_j_convention`` is missing or names the
    pre-0.2.0 rule, a :class:`StaleResultWarning` is emitted once per file
    and process.  Such results were computed before the 0.2.0 rules for
    constant constraint rows (Θ(0) = 0 (Itô) for a zero row that does not
    depend on the external times; the external-time tie order, i.e. a
    one-sided limit, at coincident external times of a k ≥ 3 cumulant; see
    CHANGELOG.md, 0.2.0); for models with δ (instantaneous) propagator
    parts they may differ from a fresh computation and should be
    recomputed.  The arrays are returned unchanged either way.

    Parameters
    ----------
    path : str
        The ``.npz`` file.
    allow_pickle : bool
        Passed to ``np.load``.  Only needed for files holding object
        arrays (non-numeric parameters); never enable it for untrusted
        files.
    warn_stale : bool
        Set False to skip the stamp check.

    Returns
    -------
    dict
        ``{key: array}`` for every entry in the file.
    """
    with np.load(path, allow_pickle=allow_pickle) as z:
        data = {key: z[key] for key in z.files}
    if warn_stale:
        _warn_if_stale(_stamp_value(data, 'daedalus_version'),
                       _stamp_value(data, 'phase_j_convention'),
                       os.path.abspath(str(path)))
    return data


# ── CSV ───────────────────────────────────────────────────────────────

def save_csv(result: dict, path: str) -> str:
    """Serialize a ``compute_cumulants`` result dict to a wide-format
    CSV with parameter / mean-field metadata as ``#``-comment lines.

    Layout::

        # k=2, max_ell=1, model='Quadratic Hawkes ...'
        # daedalus_version: 0.2.0
        # phase_j_convention: theta0_ito_const_rows
        # phase_j_flags: PHASE_J_T_MIN=-inf;POSET_PHYSICAL_MARGIN=50.0;...
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
    if isinstance(result, dict) and result.get('t_min') is not None:
        raise NotImplementedError(
            'a switch-on result (t_min) is not saved yet: save '
            "result['times'] and result['C_times'] instead")
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

        for key, val in _result_stamp(result).items():
            f.write(f'# {key}: {val}\n')

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

        # Feature B: the δ components, one JSON object per line
        tab = _delta_table(result)
        if tab is not None:
            import json
            n = len(tab['weight'])
            f.write(f'# delta_components: {n} (complete: '
                    f'{bool(tab["complete"])}; not in the C columns)\n')
            for i in range(n):
                w = complex(tab['weight'][i])
                loc = tab['location'][i]

                def _num(v):                    # strict JSON: NaN -> null
                    return None if v != v else v
                f.write('#   ' + json.dumps({
                    'weight': [_num(w.real), _num(w.imag)],
                    'location': _num(loc),
                    'ell': tab['ell'][i], 'diagram': tab['diagram'][i],
                    'equality': tab['equality'][i],
                    'provenance': json.loads(tab['provenance'][i])}) + '\n')

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
                # round-trip exact (.10g rounded to 5e-11 relative)
                row = [f'{col_data[j][i]:.17g}'
                       for j in range(len(col_data))]
                f.write(','.join(row) + '\n')

    return path

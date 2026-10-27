"""
daedalus.api.covariance — the k = 2 covariance matrix of a list of fields in one call,
and its parameter derivatives.

``covariance_matrix(model, fields, max_ell, tau_grid=...)`` returns
``Sigma[n, i, j] = C_ij(tau_n) = <dx_i(0) dx_j(tau_n)>`` for every pair of
``fields``, per loop order and in total.  It runs ``compute_cumulants`` once
per pair inside :func:`daedalus.api.compute.stage_memo`, so the stages that depend on
the parameter point only -- the expansion, the propagator, the mean field and
the poles / residues -- run once for the whole matrix; the diagrams of each
pair (one external-leg typing) are enumerated, classified and evaluated once.
Each computed entry is the per-entry ``compute_cumulants`` result on the same
grid, bit for bit.

Lower triangle (``pairs='upper'``, the default).  Stationarity gives
``C_ji(tau) = C_ij(-tau)``, so the ``(j, i)`` entry (``i < j``) is read from
the ``(i, j)`` run at ``-tau`` wherever ``-tau`` is a grid point and
``tau != 0``.  The other points -- ``tau = 0`` and every ``tau`` whose mirror
is not on the grid -- come from a ``(j, i)`` run on those points only, with
the poles resolved for the whole grid's extent (the full-grid ``(j, i)`` run's
values at those rows; a separate call on just those points can differ in the
last bits when the poles are clustered).  Models with delay kernels share
nothing (the delay route recomputes every stage per pair).  At
``tau = 0`` the value is the Ito left limit ``C_ji(0-)``, which is the RIGHT
limit ``C_ij(0+)``; it differs from ``C_ij(0)`` when ``C_ij`` jumps at 0 (a
spike-train cross-covariance, for instance), so it is never reflected.
``pairs='all'`` runs every ordered pair on the whole grid.

Parameter derivatives (``derivatives=``) are fourth-order central differences
(Richardson: steps ``h`` and ``2h``) of the shared pass, with an error
estimate.  Phase J is a numerical integrator over matrix exponentials of
parameter-dependent generators, and its realization refuses complex inputs, so
neither an exact tangent nor a complex step is available without rewriting it.
The saddle's sensitivity ``dx*/dtheta = -J^{-1} dF/dtheta`` is computed
separately by implicit differentiation of the mean-field equations and checked
against the saddles of the stencil runs: a stencil that crosses to another
branch of saddles is refused instead of differenced.  The differences assume
``Sigma`` is smooth in the parameter: at a lag where it has a kink in it (a
delay ``d`` at ``tau = +-d``) the error estimate is not a bound.
"""
from __future__ import annotations

import re
import time
import warnings

import numpy as np

from daedalus.engine.errors import DaedalusError

#: relative step of the difference stencil (``h = STEP * |theta|``, or STEP
#: itself at ``theta = 0``): the fourth-order truncation is ~STEP^4 and the
#: round-off ~1e-13 / STEP relative, both well under 1e-9 at 1e-3
STEP = 1e-3

#: the relative round-off floor of one entry of the shared pass, used in the
#: derivative's error estimate (Phase J's per-entry agreement with exact
#: references is ~1e-15 .. 1e-13)
ROUNDOFF = 1e-13

#: relative agreement asked of the stencil saddles' difference quotient and
#: the implicit-function derivative before a stencil is differenced
SADDLE_RTOL = 1e-4

# options that fix what compute_cumulants returns for ONE pair: the matrix
# call sets them itself or does not support them
_REFUSED = {
    'k': 'the covariance matrix is k = 2',
    'external_fields': 'the legs are the pairs of ``fields``',
    'spatial_grid': 'spatial models are not supported',
    'chi_grid': 'spatial models are not supported',
    'spatial_points': 'spatial models are not supported',
    'points': 'k = 2 is read on tau_grid',
    't_min': 'switch-on transients are not supported',
    'times': 'switch-on transients are not supported',
    'initial_covariance': 'switch-on transients are not supported',
    'spectra': 'call daedalus.api.spectra on a kept entry (keep_entries=True)',
    'output_npz': 'save the entries one by one (keep_entries=True)',
    'output_csv': 'save the entries one by one (keep_entries=True)',
    'origin_leaf_idx': 'the matrix is C_ij(tau) = <dx_i(0) dx_j(tau)>',
}


class CovarianceDerivativeError(DaedalusError, RuntimeError):
    """A derivative stencil left the branch of saddles of the base point (or
    a stencil point has no saddle)."""


class CovarianceWarning(UserWarning):
    """A derivative step was shortened to stay inside a parameter's domain."""


# ── fields and parameters ───────────────────────────────────────────────────

def _pop_size(model, natural):
    pf = next((f for f in (model.get('physical_fields') or [])
               if (f.get('natural_name') or f.get('name')) == natural), None)
    if pf is None:
        return 1
    pop = pf.get('population')
    if pop:
        for p in (model.get('populations') or []):
            if p.get('name') == pop:
                return max(1, int(p.get('size') or 1))
    if pf.get('indexed'):
        idx = (model.get('index_sets') or {}).get('pop')
        if idx:
            return max(1, len(idx))
    return 1


def default_fields(model):
    """Every physical field leg of ``model``: ``(name, population)`` for each
    field (natural name) and each of its populations, in declaration order."""
    out = []
    for f in (model.get('physical_fields') or []):
        name = f.get('natural_name') or f['name']
        out.extend((name, p) for p in range(1, _pop_size(model, name) + 1))
    return out


def _legs(fields):
    out = []
    for f in fields:
        if isinstance(f, str):
            f = (f, 1)
        if not (isinstance(f, (tuple, list)) and len(f) == 2):
            raise ValueError(f'fields entries are field names or (field, '
                             f'population) pairs; got {f!r}')
        out.append((str(f[0]), int(f[1])))
    if not out:
        raise ValueError('fields is empty')
    if len(set(out)) != len(out):
        raise ValueError(f'fields lists a leg twice: {out}')
    return out


def _resolved_parameters(model, given):
    """compute_cumulants' parameter defaulting (declared defaults, then the
    given values)."""
    specs = model.get('parameters', []) or []
    out = {p['name']: p['default'] for p in specs
           if p.get('default') is not None and not p.get('mean_field')}
    out.update(dict(given or {}))
    return out


_INDEX = re.compile(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*(?:\[([0-9,\s]+)\])?\s*$')


def _parameter_targets(spec, values, model):
    """``[(label, name, index)]`` for the derivative parameters ``spec``: a
    name (every component of an array parameter), ``'w[0,1]'`` or
    ``('w', (0, 1))``."""
    if isinstance(spec, (str, tuple)):
        spec = [spec]
    domains = {p['name']: p.get('domain')
               for p in (model.get('parameters', []) or [])}
    out = []
    for s in spec:
        if isinstance(s, tuple):
            name, idx = str(s[0]), s[1]
            idx = ((int(idx),) if np.ndim(idx) == 0
                   else tuple(int(i) for i in idx))
        else:
            m = _INDEX.match(str(s))
            if m is None:
                raise ValueError(f'derivative parameter {s!r}: expected a '
                                 f"name, 'name[i]' or 'name[i,j]'")
            name = m.group(1)
            idx = (None if m.group(2) is None else
                   tuple(int(i) for i in m.group(2).split(',') if i.strip()))
        if name not in values:
            raise ValueError(
                f'derivative parameter {name!r} is not a parameter of the '
                f'model with a value; known: {sorted(values)}')
        arr = np.asarray(values[name], dtype=float)
        if idx is None:
            idxs = ([()] if arr.ndim == 0 else
                    [tuple(int(i) for i in ix) for ix in np.ndindex(arr.shape)])
        else:
            if len(idx) != arr.ndim or any(
                    not 0 <= i < n for i, n in zip(idx, arr.shape)):
                raise ValueError(f'derivative parameter {s!r}: index {idx} '
                                 f'outside the shape {arr.shape} of {name!r}')
            idxs = [idx]
        for ix in idxs:
            label = name if not ix else f"{name}[{','.join(map(str, ix))}]"
            if label not in [o[0] for o in out]:
                out.append((label, name, ix, domains.get(name)))
    return out


def _component(values, name, ix):
    return float(np.asarray(values[name], dtype=float)[ix] if ix
                 else np.asarray(values[name], dtype=float))


def _shifted(values, name, ix, delta):
    """A copy of ``values`` with component ``ix`` of ``name`` moved by
    ``delta`` (lists stay lists, scalars stay scalars)."""
    out = dict(values)
    v = values[name]
    if not ix:
        out[name] = float(v) + delta
        return out
    a = np.array(v, dtype=float)
    a[ix] += delta
    out[name] = a.tolist() if isinstance(v, (list, tuple)) else a
    return out


# ── the matrix pass ──────────────────────────────────────────────────────────

def _plan(n_fields, grid, pairs):
    """The runs of one pass: ``[(i, j, rows)]`` (``rows`` the grid indices
    run, ``None`` for the whole grid) and the reflections
    ``[(i, j, n, m)]`` (``Sigma[n, j, i] = C_ij[m]``)."""
    runs, refl = [], []
    pos = {}
    for m, t in enumerate(grid):
        pos.setdefault(float(t), m)
    for i in range(n_fields):
        for j in range(i, n_fields):
            runs.append((i, j, None))
    for i in range(n_fields):
        for j in range(i + 1, n_fields):
            if pairs == 'all':
                runs.append((j, i, None))
                continue
            rest = []
            for n, t in enumerate(grid):
                m = pos.get(-float(t))
                if float(t) != 0.0 and m is not None:
                    refl.append((i, j, n, m))
                else:
                    rest.append(n)
            if rest:
                runs.append((j, i, None if len(rest) == len(grid) else rest))
    return runs, refl


def _matrix_pass(model, legs, max_ell, params, grid, pairs, keep_entries,
                 verbose, options, want_ns=False):
    from daedalus.api.compute import compute_cumulants, stage_memo
    n = len(legs)
    runs, refl = _plan(n, grid, pairs)
    sigma = np.zeros((len(grid), n, n), dtype=complex)
    by_ell = {}
    filled = np.zeros((len(grid), n, n), dtype=bool)
    records, entries, first = [], {}, None
    t0, c0 = time.perf_counter(), time.process_time()
    with stage_memo() as memo:
        for (i, j, rows) in runs:
            g = grid if rows is None else grid[rows]
            idx = np.arange(len(grid)) if rows is None else np.asarray(rows)
            t1 = time.perf_counter()
            r = compute_cumulants(
                model, 2, max_ell, external_fields=[legs[i], legs[j]],
                parameters=params, tau_grid=g, verbose=verbose, **options)
            if r.get('C_tau') is None:
                raise RuntimeError(
                    f'covariance_matrix: the ({legs[i]}, {legs[j]}) run '
                    f'returned no C_tau (route {r.get("route")})')
            sigma[idx, i, j] = r['C_tau']
            filled[idx, i, j] = True
            for ell, v in r['C_tau_by_ell'].items():
                by_ell.setdefault(int(ell), np.zeros_like(sigma))[idx, i, j] = v
            if first is None:             # (not the whole result: memory)
                hit = (next(iter(memo.entries.values()), None) if want_ns
                       else None)
                first = {'mf_values': r.get('mf_values'),
                         'params': r.get('params'),
                         '_ns': hit['ft']._ns if hit else None}
            records.append({
                'fields': (legs[i], legs[j]), 'indices': (i, j),
                'tau_grid': 'all' if rows is None else 'points',
                'n_tau': len(g), 'status': r.get('status'),
                'phase_j_incomplete': r.get('phase_j_incomplete') or {},
                'n_diagrams': len(r.get('diagrams') or ()),
                'wall': time.perf_counter() - t1})
            if keep_entries:
                entries[(i, j) if rows is None else (i, j, 'points')] = r
            del r
        hits, misses = memo.hits, memo.misses
    for (i, j, nn, m) in refl:
        sigma[nn, j, i] = sigma[m, i, j]
        for v in by_ell.values():
            v[nn, j, i] = v[m, i, j]
        filled[nn, j, i] = True
    assert filled.all(), 'covariance_matrix: an entry was not computed'
    statuses = {rec['status'] for rec in records}
    return {
        'Sigma': sigma, 'Sigma_by_ell': dict(sorted(by_ell.items())),
        'first': first, 'runs': records, 'entries': entries,
        'reflected': len(refl),
        'status': ('incomplete' if 'incomplete' in statuses else
                   'ok' if statuses <= {'ok'} else ','.join(
                       sorted(str(s) for s in statuses))),
        'shared_stages': {'hits': hits, 'misses': misses},
        'wall': time.perf_counter() - t0, 'cpu': time.process_time() - c0,
    }


# ── saddle sensitivity (implicit differentiation) ────────────────────────────

def _saddle_residual(model, values, ns):
    """``(state_vars, slices, R, keys)`` of the mean-field system at the
    parameter values ``values`` (``daedalus.api._mean_field_dae``'s own residual)."""
    from daedalus.api import _mean_field_dae as D
    if model.get('equations'):
        sv = D._state_variables(model)
        sl = D._state_slices(sv)
        p = D._numpy_params(model, values)
        if model.get('kernels'):
            p.update(D._kernel_values(model, values, ns))
        R, _ = D._build_residual(model, p, D._make_phi_callables(model, p),
                                 D._index_sets(model), sv, sl)
        return sv, sl, R, D._saddle_keys(model, sv)
    sv, sl, R, _, keys, _ = D._saddle_system(model, values, ns, reduced=False)
    return sv, sl, R, keys


def _five_point(f, x, h):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def saddle_sensitivity(model, values, mf_values, target, ns=None):
    """``dx*/dtheta = -J^{-1} dF/dtheta`` at the saddle ``mf_values`` (a
    result's ``mf_values``) for the parameter component ``target`` =
    ``(name, index)``: ``{saddle: [d value / d theta]}``.  ``F`` is the
    mean-field residual the solver uses; ``J`` and ``dF/dtheta`` are
    five-point differences of it (it is analytic in both)."""
    name, ix = target
    sv, sl, R0, keys = _saddle_residual(model, values, ns)
    x = np.zeros(sum(size for _, _, size in sv))
    for var, _, size in sv:
        a, b = sl[var]
        vals = mf_values.get(keys[var])
        if vals is None or len(vals) != b - a:
            raise KeyError(f'no saddle value for {keys[var]!r}')
        x[a:b] = np.asarray(vals, dtype=float)
    J = np.empty((x.size, x.size))
    for c in range(x.size):
        h = 1e-4 * max(1.0, abs(x[c]))
        e = np.zeros_like(x)
        e[c] = 1.0
        J[:, c] = _five_point(lambda s: np.asarray(R0(x + s * e), dtype=float),
                              0.0, h)
    th = _component(values, name, ix)
    h = 1e-4 * max(1.0, abs(th))
    cache = {}

    def F(s):
        if s not in cache:
            cache[s] = np.asarray(_saddle_residual(
                model, _shifted(values, name, ix, s), ns)[2](x), dtype=float)
        return cache[s]
    dx = -np.linalg.solve(J, _five_point(F, 0.0, h))
    return {keys[var]: dx[sl[var][0]:sl[var][1]].tolist() for var, _, _ in sv}


def _flat_mf(mf_values):
    return {k: np.asarray(v, dtype=float) for k, v in (mf_values or {}).items()}


# ── derivatives ──────────────────────────────────────────────────────────────

def _derivative_options(derivatives):
    if isinstance(derivatives, dict):
        opts = dict(derivatives)
        unknown = set(opts) - {'parameters', 'step', 'abs_step'}
        if unknown:
            raise ValueError(f'derivatives: unknown option(s) {sorted(unknown)}'
                             f"; known: 'parameters', 'step', 'abs_step'")
        if 'parameters' not in opts:
            raise ValueError("derivatives={...} needs 'parameters'")
    else:
        opts = {'parameters': derivatives}
    step = float(opts.get('step', STEP))
    if not (np.isfinite(step) and 0 < step < 0.5):
        raise ValueError(f'derivatives step must be in (0, 0.5); got {step}')
    return opts['parameters'], step, dict(opts.get('abs_step') or {})


def _derivatives(model, legs, max_ell, values, grid, pairs, verbose, options,
                 base, spec):
    params, step, abs_step = _derivative_options(spec)
    targets = _parameter_targets(params, values, model)
    ns = base['first'].get('_ns')
    mf0 = _flat_mf(base['first'].get('mf_values'))
    sig0 = base['Sigma']
    out = {}
    for label, name, ix, domain in targets:
        th = _component(values, name, ix)
        h = float(abs_step.get(label, abs_step.get(name,
                  step * abs(th) if th != 0 else step)))
        if domain == 'positive' and th - 2 * h <= 0:
            h = th / 4.0
            warnings.warn(f'derivative of {label}: step shortened to {h:g} to '
                          f'keep {label} - 2h > 0', CovarianceWarning,
                          stacklevel=3)
        if not (h > 0 and np.isfinite(h)):
            raise ValueError(f'derivative of {label}: step {h!r} is not > 0')
        pts = {}
        for s in (-2, -1, 1, 2):
            v = _shifted(values, name, ix, s * h)
            try:
                pts[s] = _matrix_pass(model, legs, max_ell, v, grid, pairs,
                                      False, verbose, options)
            except Exception as exc:
                raise CovarianceDerivativeError(
                    f'derivative of {label}: the run at {label} = '
                    f'{th + s * h!r} (base {th!r}, step {s:+d}h) failed: '
                    f'{type(exc).__name__}: {exc}') from exc

        def rich(key, sub=None):
            def get(s):
                a = pts[s][key]
                return a if sub is None else a[sub]
            d1 = (get(1) - get(-1)) / (2 * h)
            d2 = (get(2) - get(-2)) / (4 * h)
            return (4 * d1 - d2) / 3, np.abs(d1 - d2) / 3
        dS, trunc = rich('Sigma')
        err = trunc + ROUNDOFF * float(np.max(np.abs(sig0))) / h
        dS_ell, err_ell = {}, {}
        for ell in base['Sigma_by_ell']:
            d, t = rich('Sigma_by_ell', ell)
            dS_ell[ell] = d
            err_ell[ell] = t + ROUNDOFF * float(
                np.max(np.abs(base['Sigma_by_ell'][ell]))) / h
        # the saddles of the stencil: one branch, consistent with the
        # implicit-function derivative
        mfs = {s: _flat_mf(pts[s]['first'].get('mf_values')) for s in pts}
        dmf_fd = {}
        for key, x0 in mf0.items():
            if not all(key in mfs[s] and mfs[s][key].shape == x0.shape
                       for s in mfs):
                continue
            d1 = (mfs[1][key] - mfs[-1][key]) / (2 * h)
            d2 = (mfs[2][key] - mfs[-2][key]) / (4 * h)
            dmf_fd[key] = ((4 * d1 - d2) / 3, d1, d2, x0)
        try:
            dmf = saddle_sensitivity(model, values, base['first']['mf_values'],
                                     (name, ix), ns)
            dmf_note = None
        except Exception as exc:          # no residual to differentiate
            dmf, dmf_note = None, f'{type(exc).__name__}: {exc}'
        for key, (d, d1, d2, x0) in dmf_fd.items():
            # round-off of the stencil saddles' quotient (a branch change
            # moves a saddle by O(1), the quotient by O(1 / h))
            floor = 1e-9 * (np.abs(x0) + 1.0) / h
            if dmf is not None and key in dmf:
                ref = np.asarray(dmf[key], dtype=float)
                bad = np.abs(d - ref) > SADDLE_RTOL * (
                    np.abs(ref) + np.abs(d1)) + floor
            else:
                bad = np.abs(d1 - d2) > 0.1 * np.abs(d1) + floor
            if np.any(bad):
                raise CovarianceDerivativeError(
                    f'derivative of {label}: the saddle {key!r} is not on one '
                    f'branch across the stencil {label} = {th:g} '
                    f'{{-2h, -h, +h, +2h}} (h = {h:g}): values '
                    + ', '.join(f'{s:+d}h: {mfs[s][key].tolist()}'
                                for s in (-2, -1, 1, 2))
                    + f', base {x0.tolist()}'
                    + ('' if dmf is None or key not in dmf else
                       f'; implicit dx*/d{label} = {dmf[key]}')
                    + '. Pass a smaller step (derivatives={"parameters": '
                      '[...], "step": ...}) or move away from the '
                      'bifurcation.')
        out[label] = {
            'dSigma': dS, 'error': err, 'dSigma_by_ell': dS_ell,
            'error_by_ell': err_ell, 'step': h, 'value': th,
            'dmf': dmf, 'dmf_fd': {k: v[0].tolist() for k, v in dmf_fd.items()},
            'dmf_note': dmf_note,
            'status': ('ok' if all(pts[s]['status'] == 'ok' for s in pts)
                       else 'incomplete'),
            'wall': sum(pts[s]['wall'] for s in pts),
            'cpu': sum(pts[s]['cpu'] for s in pts),
        }
    return out


# ── public entry point ───────────────────────────────────────────────────────

#: the keys of ``dd.Config(covariance={...})``
COVARIANCE_KEYS = ('fields', 'pairs', 'derivatives', 'keep_entries')


def resolve_covariance_options(value, where='covariance'):
    """``dd.Config.covariance`` as the keyword arguments of
    :func:`covariance_matrix` (``None`` when off): ``True`` (every physical
    field leg) or a dict with ``fields``, ``pairs``, ``derivatives``,
    ``keep_entries``."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be None, True or a dict with keys '
                         f'{list(COVARIANCE_KEYS)}; got {value!r}')
    unknown = sorted(set(value) - set(COVARIANCE_KEYS))
    if unknown:
        raise ValueError(f'{where}: unknown key(s) {unknown}; known: '
                         f'{list(COVARIANCE_KEYS)}')
    out = {'fields': value.get('fields'),
           'pairs': value.get('pairs', 'upper'),
           'derivatives': value.get('derivatives'),
           'keep_entries': bool(value.get('keep_entries', False))}
    if out['pairs'] not in ('upper', 'all'):
        raise ValueError(f"{where}: pairs must be 'upper' or 'all'; got "
                         f"{out['pairs']!r}")
    if out['fields'] is not None:
        _legs(out['fields'])
    if out['derivatives'] is not None:
        _derivative_options(out['derivatives'])
    return out



def covariance_matrix(model, fields=None, max_ell=0, *, parameters=None,
                      fundamental=None, tau_grid=None, tau_max=50.0,
                      tau_step=0.5, pairs='upper', derivatives=None,
                      keep_entries=False, verbose=False, **options):
    """The k = 2 covariance matrix ``Sigma[n, i, j] = <dx_i(0) dx_j(tau_n)>``
    of ``fields`` on ``tau_grid``, per loop order and in total, in one call.

    Parameters
    ----------
    model : dict, SDE or builder
        A temporal model (spatial models are refused).
    fields : list or None
        The legs: field names (``'x'``, for a field with one unit) or
        ``(field, unit)`` pairs, e.g. ``['x', 'y']`` or ``[('v', 1), ('v', 2)]``.  ``None``: every physical
        field leg (:func:`default_fields`).
    max_ell : int
        Maximum loop order.
    parameters : dict or None
        Parameter values, as ``compute_cumulants`` (``fundamental`` is the
        older name).
    tau_grid, tau_max, tau_step
        The lag grid, as ``compute_cumulants`` (an ``(lo, hi, n)`` tuple is a
        range); every pair runs on the same explicit grid.
    pairs : 'upper' or 'all'
        ``'upper'``: the pairs ``i <= j`` on the grid, the lower triangle by
        ``C_ji(tau) = C_ij(-tau)`` where ``-tau`` is on the grid and
        ``tau != 0``, the remaining points by a ``(j, i)`` run on them (module
        docstring).  ``'all'``: every ordered pair on the grid.
    derivatives : None, list or dict
        Parameter derivatives ``dSigma/dtheta``: a list of parameters (a name
        -- every component of an array parameter --, ``'w[0,1]'`` or
        ``('w', (0, 1))``), or a dict ``{'parameters': [...], 'step': 1e-3,
        'abs_step': {label: h}}``.  Fourth-order central differences with
        steps ``h`` and ``2h`` (``h = step * |theta|``, ``step`` at
        ``theta = 0``); four extra passes per parameter.
    keep_entries : bool
        Keep each pair's full ``compute_cumulants`` result in
        ``result['entries']`` (memory: every entry keeps its diagrams and
        callables).  Default False: each result is released after its
        values are read.
    verbose : bool
        ``compute_cumulants``' progress output for every run.
    **options
        Passed to every ``compute_cumulants`` call (``use_cache``,
        ``parallel``, ``n_workers``, ``fixed_point_index``, ``policy``,
        ``preflight``, ``backend``, ``accuracy``, ``discard``, ``delay``,
        ``taylor_order``, ``mf_dae_n_starts``, ``mf_dae_seed_box``).  Each
        run keeps its own memory guard (``PHASE_J_MEM_BUDGET_GB``); macOS
        never forks.

    Returns
    -------
    dict
        ``'Sigma'`` (complex ``(n_tau, N, N)``, as ``C_tau``),
        ``'Sigma_by_ell'`` ``{ell: (n_tau, N, N)}`` (the loop-order
        increments; ``Sigma`` is their sum), ``'Sigma0'`` (the ``tau = 0``
        matrix when 0 is on the grid, else None), ``'fields'``,
        ``'tau_grid'``, ``'pairs'``, ``'runs'`` (one record per
        ``compute_cumulants`` call), ``'reflected'`` (entries filled by
        symmetry), ``'shared_stages'`` (stage-memo hits / misses),
        ``'status'``, ``'mf_values'``, ``'params'``, ``'wall'``, ``'cpu'``,
        ``'entries'`` (with ``keep_entries``) and, with ``derivatives``,
        ``'derivatives'``: ``{label: {'dSigma', 'error', 'dSigma_by_ell',
        'error_by_ell', 'step', 'value', 'dmf', 'dmf_fd', 'dmf_note',
        'status', 'wall', 'cpu'}}`` (``error``: the truncation estimate
        ``|D_h - D_2h| / 3`` of the second-order quotient, an upper bound for
        the fourth-order one, plus a round-off floor ``ROUNDOFF max|Sigma| /
        h``; ``dmf``: the implicit-function derivative of the saddles).
    """
    from daedalus.api.compute import as_model_dict
    from daedalus.api._grids import resolve_grid
    model = as_model_dict(model)
    for key in options:
        if key in _REFUSED:
            raise ValueError(f'covariance_matrix: {key}= is not an option '
                             f'here ({_REFUSED[key]})')
    if model.get('spatial'):
        raise NotImplementedError(
            'covariance_matrix is for temporal models; a spatial model has '
            'C(tau, chi) per pair (call compute_cumulants per pair)')
    if pairs not in ('upper', 'all'):
        raise ValueError(f"pairs must be 'upper' or 'all'; got {pairs!r}")
    if parameters is None:
        parameters = fundamental
    from daedalus.api.access import resolve_external_fields
    legs = _legs(default_fields(model) if fields is None
                 else resolve_external_fields(model, list(fields)))
    grid = resolve_grid(tau_grid, 'tau_grid')
    if grid is None:
        if not (tau_step > 0 and tau_max >= 0 and np.isfinite(tau_step)
                and np.isfinite(tau_max)):
            raise ValueError(f'need finite tau_step > 0 and tau_max >= 0 '
                             f'(got {tau_step}, {tau_max}), or a tau_grid')
        grid = np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step)
    grid = np.asarray(grid, dtype=float).ravel()
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError('tau_grid must be a non-empty finite grid')
    if len(set(grid.tolist())) != grid.size:
        raise ValueError('tau_grid has repeated points')
    options = dict(options, tau_max=tau_max, tau_step=tau_step)
    values = _resolved_parameters(model, parameters)
    base = _matrix_pass(model, legs, max_ell, parameters, grid, pairs,
                        keep_entries, verbose, options,
                        want_ns=derivatives is not None)
    first = base['first']
    zero = np.flatnonzero(grid == 0.0)
    out = {
        'Sigma': base['Sigma'], 'Sigma_by_ell': base['Sigma_by_ell'],
        'Sigma0': base['Sigma'][zero[0]] if zero.size else None,
        'fields': legs, 'tau_grid': grid, 'pairs': pairs, 'max_ell': max_ell,
        'runs': base['runs'], 'reflected': base['reflected'],
        'shared_stages': base['shared_stages'], 'status': base['status'],
        'mf_values': first.get('mf_values'), 'params': first.get('params'),
        'wall': base['wall'], 'cpu': base['cpu'],
    }
    if keep_entries:
        out['entries'] = base['entries']
    if derivatives is not None:
        out['derivatives'] = _derivatives(
            model, legs, max_ell, values, grid, pairs, verbose, options,
            base, derivatives)
    return out

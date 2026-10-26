"""Cumulants of observables, functions of the fields (``SDE.observable``).

An observable ``O = f(x)`` declared on an :class:`api.sde.SDE` is an external
leg of :func:`api.compute_cumulants` (``external_fields=['rate']``, or
``('rate', u)`` for unit ``u`` of a population observable; field legs may
be mixed in).  Its cumulants are the composite-operator expansion about the
mean-field saddle ``x*``,

    O = f* + f_a dx_a + (1/2) f_ab dx_a dx_b + ...,

assembled from the cumulants of the field fluctuations ``dx`` that the engine
computes (one ``compute_cumulants`` call per pair or tuple of field legs with a
nonzero coefficient):

* an observable LINEAR in the fields (``x + 2*y``, ``a*v`` with a linear
  ``phi``) is the same linear combination of field legs: exact at every ``k``
  and loop order;
* a nonlinear observable, with ``dx = O(sqrt(hbar))`` in the loop counting
  (``<dx> = O(hbar)``, ``C = O(hbar)``, the ``n``-th cumulant
  ``O(hbar^(n-1))``):

  - ``k = 1``: the mean shift ``<O> - f*``, zero at tree level (as for a
    field) and at one loop ``f_a <dx_a>_1 + (1/2) f_ab C_ab(0)`` (the tree
    covariance at equal times); the mean is ``f* + C_tau``
    (``result['observable']['mean']``);
  - ``k = 2``: the tree ``f_a f'_b C_ab(tau)``.

  Beyond these orders the expansion needs the coincident-time cumulants of the
  composite operator (``<dx_a(0) dx_b(0) dx_c(tau)>`` and up), which the engine
  does not compute: refused with that reason.  A spike train or exposed source
  enters an observable only linearly (``SDE.observable`` refuses otherwise).

Conventions as for field legs: ``C(tau) = <O_1(0) O_2(tau)>`` in leg order,
``C(0)`` the Ito left limit, the loop-order increments in ``C_tau_by_ell``.
"""
from __future__ import annotations

import numpy as np

__all__ = ['ObservableError', 'is_observable_call', 'observable_cumulants']


class ObservableError(ValueError):
    """An observable question the composite-operator expansion cannot answer."""


def _legs(model, external_fields):
    obs = model.get('observables') or {}
    out = []
    for leg in external_fields:
        if isinstance(leg, str):
            name, unit = leg, None
        elif isinstance(leg, (tuple, list)) and len(leg) == 2 and isinstance(leg[0], str):
            name, unit = leg[0], leg[1]
        else:
            return None
        out.append({'name': name, 'unit': unit, 'obs': name in obs})
    return out


def is_observable_call(model, external_fields) -> bool:
    """True when ``external_fields`` names an observable of ``model``."""
    if not (model.get('observables') and external_fields):
        return False
    legs = _legs(model, external_fields)
    return bool(legs) and any(l['obs'] for l in legs)


def _value(params, name, idx, unit, where):
    """The numeric value of parameter ``name`` read with ``idx`` index
    letters (``a[i]`` -> one), at unit ``unit``."""
    raw = params[name]
    n = len(idx) or (1 if isinstance(raw, (list, tuple, np.ndarray)) else 0)
    if n and unit is None:
        raise ObservableError(f'{where}: parameter {name!r} is indexed; the observable needs '
                              f'population= and a leg (name, unit).')
    v = params[(name,) + (unit,) * n] if n else raw
    if isinstance(v, (list, tuple, np.ndarray)):
        raise ObservableError(f'{where}: parameter {name!r} is a matrix or vector here; index it '
                              f'(e.g. {name}[i]).')
    return float(v)


def _expansion(model, spec, leg, params, mf, field_units, where):
    """``(f*, {field leg: f_a}, {(a, b): f_ab})`` of one observable leg at
    the saddle; field legs are ``(natural name, unit)``."""
    from sage.all import SR
    from api.noise import parse_expr
    e = parse_expr(spec['inlined'], what=where)
    pop, unit = spec['population'], leg['unit']
    if pop is None and unit not in (None, 1):
        raise ObservableError(f'{where}: the observable has no population; give its name alone '
                              f'(got unit {unit}).')
    if pop is not None and unit is None:
        raise ObservableError(f'{where}: a population observable is read at one unit: give '
                              f'({leg["name"]!r}, u).')
    fields = set(field_units)
    sub, legs = {}, {}
    for v in e.variables():
        parts = str(v).split('__')
        base, idx = parts[0], parts[1:]
        if any(i != 'i' for i in idx):
            raise ObservableError(f'{where}: only the unit index i is supported in an observable '
                                  f'(got {base}[{", ".join(idx)}]).')
        if base in fields:
            n = field_units[base]
            if n > 1:
                if pop is None or spec['field_pops'].get(base) != pop:
                    raise ObservableError(
                        f'{where}: field {base!r} has {n} units; read it in a population '
                        f'observable (population= its population) at a unit.')
                u = int(unit)
            else:
                u = 1
            sym = SR.var(f'obsleg_{base}_{u}')
            sub[v] = sym
            legs[sym] = (base, u)
        else:
            sub[v] = SR(_value(params, base, idx, None if pop is None else int(unit), where))
    f = e.subs(sub)
    at = {s: SR(float(mf[f'{b}star'][u - 1])) for s, (b, u) in legs.items()}
    syms = list(legs)
    f0 = complex(f.subs(at).n())
    g = {legs[a]: complex(f.diff(a).subs(at).n()) for a in syms}
    H = {(legs[a], legs[b]): complex(f.diff(a).diff(b).subs(at).n()) for a in syms for b in syms}
    real = lambda z: z.real if abs(z.imag) <= 1e-14 * max(1.0, abs(z.real)) else z
    return real(f0), {k: real(v) for k, v in g.items() if v != 0}, \
        {k: real(v) for k, v in H.items() if v != 0}


def observable_cumulants(model, call: dict) -> dict:
    """``compute_cumulants(**call)`` for a question whose external legs name
    observables (see the module docstring)."""
    from api.compute import compute_cumulants
    specs = model['observables']
    k, max_ell = int(call['k']), int(call.get('max_ell') or 0)
    legs = _legs(model, call['external_fields'])
    where = 'compute_cumulants(observable legs)'
    if legs is None or len(legs) != k:
        raise ObservableError(f'{where}: give k = {k} legs, each a name or (name, unit).')
    for key in ('t_min', 'spectra', 'output_npz', 'output_csv', 'spatial_points'):
        if call.get(key) is not None:
            raise ObservableError(f'{where}: {key}= is not supported with observable legs.')
    if model.get('spatial'):
        raise ObservableError(f'{where}: observables of an SPDE are not supported.')
    nonlinear = [l['name'] for l in legs if l['obs'] and not specs[l['name']]['linear']]
    if nonlinear and not ((k == 1 and max_ell <= 1) or (k == 2 and max_ell == 0)):
        raise ObservableError(
            f'{where}: observable(s) {sorted(set(nonlinear))} are nonlinear in the fields; the '
            f'composite-operator expansion is available for the mean to one loop (k = 1, '
            f'max_ell <= 1) and the tree of k = 2 (max_ell = 0).  k = {k}, max_ell = {max_ell} '
            f'needs coincident-time cumulants of the composite operator, which the engine does '
            f'not compute.')
    import inspect
    accepted = set(inspect.signature(compute_cumulants).parameters)
    base = {kk: v for kk, v in call.items() if kk in accepted
            and kk not in ('model', 'k', 'max_ell', 'external_fields')}

    def run(kk, ell, ext, **over):
        kw = dict(base)
        kw.update(over)
        return compute_cumulants(model, kk, ell, external_fields=list(ext), **kw)

    # the saddle and the parameter values (one cheap run: the tree of k = 1)
    first_field = None
    for l in legs:
        nm = specs[l['name']]['fields'][0] if l['obs'] and specs[l['name']]['fields'] else \
            (None if l['obs'] else l['name'])
        if nm:
            first_field = nm
            break
    if first_field is None:
        raise ObservableError(f'{where}: the observables read no field (constants have no '
                              f'cumulants).')
    probe = run(1, 0, [(first_field, 1)])
    mf, params = probe['mf_values'], probe['params']
    natural = {f.get('natural_name') or f['name']: f for f in model['physical_fields']}
    field_units = {n: len(mf.get(f'{n}star', [0])) for n in natural}

    # each leg as {field leg: coefficient} (+ its saddle value and Hessian)
    lin, hess, saddle = [], [], []
    for l in legs:
        if l['obs']:
            f0, g, H = _expansion(model, specs[l['name']], l, params, mf, field_units,
                                  f'observable {l["name"]!r}')
        else:
            u = 1 if l['unit'] is None else int(l['unit'])
            f0 = float(mf[f'{l["name"]}star'][u - 1]) if f'{l["name"]}star' in mf else None
            g, H = {(l['name'], u): 1.0}, {}
        lin.append(g)
        hess.append(H)
        saddle.append(f0)

    terms = []                       # (coefficient, field legs, sub-result)

    def combos(i):
        if i == k:
            yield 1.0, ()
            return
        for a, c in lin[i].items():
            for cc, rest in combos(i + 1):
                yield c * cc, (a,) + rest

    for c, ext in combos(0):
        terms.append((c, ext, run(k, max_ell, ext)))
    quad = []                        # k = 1 one loop: (1/2) f_ab C_ab(0)
    if k == 1 and max_ell >= 1 and hess[0]:
        for (a, b), h in hess[0].items():
            r = run(2, 0, (a, b), tau_grid=np.array([0.0]))
            quad.append((0.5 * h, (a, b), r))

    # (every coefficient may vanish: f_a = 0 at a symmetric saddle; the k = 1
    # tree run then gives the grid)
    template = terms[0][2] if terms else probe
    out = {kk: v for kk, v in template.items()
           if kk in ('mf_values', 'mf', 'params', 'num_params', 'tau_grid', 'propagator',
                     'route', 'analysis')}
    ells = sorted({e for _, _, r in terms for e in (r.get('C_tau_by_ell') or {})}
                  | ({1} if quad else set()) | set(range(max_ell + 1)))
    if k <= 2 and template.get('C_tau') is not None:
        shape = np.shape(template['C_tau'])
        by_ell = {e: np.zeros(shape, dtype=complex) for e in ells}
        for c, _, r in terms:
            for e, v in (r.get('C_tau_by_ell') or {}).items():
                if v is not None:
                    by_ell[e] = by_ell[e] + c * np.asarray(v)
        for c, _, r in quad:
            by_ell[1] = by_ell[1] + c * complex(np.asarray(r['C_tau']).ravel()[0])
        out['C_tau_by_ell'] = by_ell
        out['C_tau'] = sum(by_ell.values()) if by_ell else np.zeros(shape, dtype=complex)
    else:
        out['C_tau'], out['C_tau_by_ell'] = None, {e: None for e in ells}

    def total(*tau):
        v = sum(c * r['total_C'](*tau) for c, _, r in terms)
        return v + sum(c * complex(np.asarray(r['C_tau']).ravel()[0]) for c, _, r in quad)

    def by(e):
        def f(*tau):
            v = sum(c * r['total_C_by_ell'][e](*tau) for c, _, r in terms
                    if e in (r.get('total_C_by_ell') or {}))
            if e == 1:
                v = v + sum(c * complex(np.asarray(r['C_tau']).ravel()[0]) for c, _, r in quad)
            return v
        return f

    out['total_C'] = total
    out['total_C_by_ell'] = {e: by(e) for e in ells}
    statuses = [r.get('status', 'ok') for _, _, r in terms + quad]
    out['status'] = 'ok' if all(s == 'ok' for s in statuses) else 'incomplete'
    rec = {'legs': [l['name'] if l['unit'] is None else (l['name'], l['unit']) for l in legs],
           'saddle': saddle, 'linear': lin, 'hessian': hess,
           'terms': [{'coefficient': c, 'fields': list(ext)} for c, ext, _ in terms],
           'quadratic_terms': [{'coefficient': c, 'fields': list(ext)} for c, ext, _ in quad],
           'components': [r for _, _, r in terms + quad]}
    if k == 1 and out['C_tau'] is not None and saddle[0] is not None:
        rec['mean'] = saddle[0] + out['C_tau']
    out['observable'] = rec
    out['external_fields'] = rec['legs']
    out['k'], out['max_ell'] = k, max_ell
    return out

"""The text form of an SDE model: the YAML ``sde:`` block of a config.

:meth:`api.sde.SDE.to_text` writes a model as the ``sde:`` block that
``daedalus run`` reads (:mod:`api.config_run`), and
:meth:`api.sde.SDE.from_text` reads one back through the same parser and
builder, so a config's inline model and the text of a Python-built one are
the same format::

    text = m.to_text()                 # 'sde:\\n  name: ...\\n  parameters: ...'
    m2 = SDE.from_text(text)           # the same model (same action, equations, dict)

The block holds the declarations as given (the equations as written, a
delayed field as ``x(t - d)``, a higher-order lhs with its velocities, the
sources with their arguments); everything they derive (embedded velocities,
coloured-noise states, delay kernels, the Ito corrections) is derived again
on reading.  The builder declares in a fixed order (operators, populations,
parameters, functions, kernels; ``fields`` in order, where an entry that is
a point process or a field a noise source creates declares that; the
equations; the other sources; the settings; the observables), and
:meth:`~api.sde.SDE.to_text` checks that reading the text back gives the
same model, byte for byte: a model declared in an order the block cannot
record (a field created by one equation before another field is declared,
say) is refused with that reason.  Refused as well: sources the block has
no form for (``SumSource``, objects outside
:mod:`api.noise`), and parameter values that are not numbers.
"""
from __future__ import annotations

import re

from api.noise import GaussianVector, expr_to_text
from api.sde import _unwrap

__all__ = ['sde_spec', 'to_text', 'from_text']


def _err(msg):
    from api.sde import SDEError
    return SDEError(f'to_text(): {msg}')


def _num_tree(v, what):
    """A parameter value as plain numbers / nested lists."""
    if hasattr(v, 'tolist'):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_num_tree(x, what) for x in v]
    if isinstance(v, bool):
        raise _err(f'{what}: {v!r} is not a number.')
    if isinstance(v, int):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise _err(f'{what}: {v!r} is not a number (the text form holds numeric '
                   f'defaults).') from None
    return int(v) if isinstance(v, int) else f


def _coef(c):
    """A coupling coefficient as written (a string, a number, or a list)."""
    if isinstance(c, (list, tuple)):
        return [_coef(x) for x in c]
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return c
    return str(c)


def _text(v):
    """An expression as written in the block (a plain number as a number)."""
    t = expr_to_text(v)
    if re.fullmatch(r'-?\d+', t):
        return int(t)
    m = re.fullmatch(r'\(?(-?\d+\.\d*(?:e[-+]?\d+)?)\)?', t)
    return float(m.group(1)) if m else t


def _law(obj, what):
    """``{type: ..., <arg>: ...}`` of a jump law or source whose parameters
    are its constructor's arguments (checked by rebuilding it); arguments at
    their default are left out."""
    import inspect
    sig = inspect.signature(type(obj).__init__).parameters
    args = {k: _text(v) for k, v in obj.params.items()
            if not (k in sig and sig[k].default is not inspect.Parameter.empty
                    and str(sig[k].default) == str(_text(v)))}
    try:
        again = type(obj)(**args)
    except Exception:
        again = None
    if again is None or repr(again) != repr(obj):
        raise _err(f'{what}: {type(obj).__name__} has no text form (its parameters are not its '
                   f'arguments).')
    return {'type': type(obj).__name__, **args}


def _source(name, s):
    """The ``noise:`` entry of one source."""
    from api.noise import CGF, CompoundPoisson, Cumulants, Gaussian
    what = f'source {name!r}'
    col = s.get('colored')
    src = s['source']
    if col is not None:
        d = {'type': 'Gaussian', 'var': _text(src.params['var'])}
        if expr_to_text(src.params['mean']) != '0':
            d['mean'] = _text(src.params['mean'])
        d['kernel'] = col['kernel'].freq_image
        d['into'] = {f: _coef(c) for f, c in col['into'].items()}
        if col['field'] != name:
            d['as_field'] = col['field']
        return d
    t = type(src)
    if t is Gaussian:
        d = {'type': 'Gaussian', 'var': _text(src.params['var'])}
        if expr_to_text(src.params['mean']) != '0':
            d['mean'] = _text(src.params['mean'])
        if src.space is not None:
            d['space'] = src.space
    elif t is GaussianVector:
        d = {'type': 'GaussianVector', 'cov': [[_text(c) for c in r] for r in src.cov]}
        if any(expr_to_text(m) != '0' for m in src.mean_vector):
            d['mean'] = [_text(m) for m in src.mean_vector]
    elif t is CompoundPoisson:
        d = {'type': 'CompoundPoisson', 'rate': _text(src.params['rate']),
             'jump': _law(src.jump, what)}
    elif t is Cumulants:
        d = {'type': 'Cumulants', 'kappas': [_text(src.params[f'kappa{m}'])
                                            for m in range(2, src.declared_order + 1)]}
        if expr_to_text(src.params['mean']) != '0':
            d['mean'] = _text(src.params['mean'])
    elif t is CGF:
        d = {'type': 'CGF', 'expression': expr_to_text(src._K)}
    else:
        from api.noise import SOURCE_TYPES
        if t.__name__ not in SOURCE_TYPES:
            raise _err(f'{what}: {t.__name__} has no text form (the sde: block takes the '
                       f'sources {", ".join(SOURCE_TYPES)}).')
        d = _law(src, what)
    if s['couples']:
        d['into'] = {f: _coef(c) for f, c in s['couples'].items()}
    return d


def sde_spec(sde) -> dict:
    """The plain-data ``sde:`` block of ``sde`` (ordered as the builder
    reads it)."""
    from api.operators import Laplacian, TimeDerivative
    out = {'name': sde.name}
    if sde._n_pop_default != 1:
        out['n_populations'] = sde._n_pop_default
    if sde._space is not None:
        out['space'] = sde._space
    ops = {}
    for n, op in sde._operators.items():
        if n == 'Dt' and sde._dt_implicit:
            continue
        kind = {TimeDerivative: 'TimeDerivative', Laplacian: 'Laplacian'}.get(type(op))
        if kind is None:
            raise _err(f'operator {n!r} ({type(op).__name__}) has no text form.')
        ops[n] = kind
    if ops:
        out['operators'] = ops
    if sde._pops:
        out['populations'] = {p['name']: (p['size'] if not p.get('description') else
                                          {'size': p['size'], 'description': p['description']})
                              for p in sde._pops}
    if sde._params:
        params = {}
        for (name,), kw in sde._params:
            e = {}
            if kw.get('default') is not None:
                e['default'] = _num_tree(kw['default'], f'parameter {name!r}')
            for k in ('domain', 'description', 'indexed', 'indexed_by'):
                if kw.get(k) not in (None, '', False):
                    e[k] = list(kw[k]) if k == 'indexed_by' else kw[k]
            params[name] = e['default'] if list(e) == ['default'] else e
        out['parameters'] = params
    if sde._functions:
        funcs = {}
        for (name, args, expr), kw in sde._functions:
            e = {'args': list(args), 'expression': expr}
            for k in ('latex', 'description', 'population'):
                if kw.get(k) not in (None, ''):
                    e[k] = kw[k]
            funcs[name] = e
        out['functions'] = funcs
    kernels = {}
    for name, k in sde._kernels.items():
        if name in sde._auto_kernels or name in sde._composite_kernels:
            continue                        # (declared again by x(t - d) / Conv(a*b, x))
        jit = sde._jitter.get(name)
        if jit and 'factors' not in jit:    # Kernel.gaussian (fitted again on reading)
            kernels[name] = {'gaussian': {'sigma': _unwrap(jit['sigma']), 'tol': jit['tol']}}
            continue
        e = {}
        for key in ('time_expr', 'freq_image', 'delay', 'latex_name', 'indexed', 'indexed_by'):
            if k.get(key) not in (None, '', False):
                e[key] = list(k[key]) if key == 'indexed_by' else k[key]
        kernels[name] = e
    if kernels:
        out['kernels'] = kernels
    emb = set((getattr(sde, '_conv_embedding', None) or {}).get('fields', []))
    fields, pps = [], {}
    for f, info in sde._fields.items():
        if f in emb or info.get('embedded_of'):
            continue                        # (declared by their equation / kernel)
        src = sde._sources.get(info['exposed']) if info.get('exposed') else None
        if src is not None and src.get('pp'):
            d = sde._sources[f]['declared']
            e = {'rate': d['rate']}
            if str(d['size']).strip() != '1':
                e['size'] = _coef(d['size'])
            for k in ('population', 'domain'):
                if info.get(k) is not None:
                    e[k] = info[k]
            if info.get('description') not in ('', f'point process {f}'):
                e['description'] = info['description']
            pps[f] = e
            fields.append(f)
            continue
        if info.get('exposed') or info.get('colored_of'):
            fields.append(f)                # (declared by its noise source)
            continue
        attrs = {k: info[k] for k in ('population', 'description', 'domain', 'latex')
                 if info.get(k) not in (None, '')}
        fields.append({f: attrs} if attrs else f)
    if fields:
        out['fields'] = fields
    if pps:
        out['point_processes'] = pps
    t0 = (sde._time_names() or ['Dt'])[0]
    eqs = {}
    for f, d in sde._declared_eqs.items():
        if d['velocity'] is None and d['lhs'].replace(' ', '') == f'{t0}*{f}':
            eqs[f] = d['rhs']
        else:
            e = {'lhs': d['lhs'], 'rhs': d['rhs']}
            if d['velocity'] is not None:
                e['velocity'] = d['velocity']
            eqs[f] = e
    if eqs:
        out['drift' if all(isinstance(v, str) for v in eqs.values()) else 'equations'] = eqs
    noise = {}
    for name, s in sde._sources.items():
        if s.get('pp'):
            continue
        d = _source(name, s)
        if s['expose']:
            d['as_field'] = s['expose']
            dom = sde._fields[s['expose']].get('domain')
            if dom is not None:
                d['domain'] = dom
        if s.get('population') is not None:
            d['population'] = s['population']
        if 'interp' in s:
            d['interpretation'] = s['interp']
        noise[name] = d
    if noise:
        out['noise'] = noise
    if sde._interp != 'ito':
        out['interpretation'] = sde._interp
    if sde._series_order != 8:
        out['series_order'] = sde._series_order
    if sde._marcus_route != 'auto':
        out['marcus_route'] = sde._marcus_route
    if sde._stability is not None:
        out['stability_analysis'] = sde._stability
    if sde._extra:
        out['extra_action'] = list(sde._extra)
    if sde._boundary is not None:
        out['boundary'] = dict(sde._boundary)
    if sde._dyson is not None:
        out['dyson'] = {k: v for k, v in sde._dyson.items() if v is not None}
    if sde._observables:
        out['observables'] = {n: (o['expr'] if o['population'] is None and not o['description']
                                  else {k: v for k, v in (('expr', o['expr']),
                                                          ('population', o['population']),
                                                          ('description', o['description']))
                                        if v not in (None, '')})
                              for n, o in sde._observables.items()}
    return out


def _dump(spec) -> str:
    import yaml

    class _Dumper(yaml.SafeDumper):
        pass

    def flow_list(dumper, data):
        # leaf lists (names, numbers, rows) inline; lists of mappings as blocks
        flow = all(not isinstance(x, dict) for x in data)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=flow)

    _Dumper.add_representer(list, flow_list)
    return yaml.dump({'sde': spec}, Dumper=_Dumper, sort_keys=False, allow_unicode=True,
                     default_flow_style=False, width=100)


def _same(a, b):
    from api.sde import _canon_model
    return (a.action_text() == b.action_text() and a.equations() == b.equations()
            and _canon_model(a.build()) == _canon_model(b.build()))


def to_text(sde, verify: bool = True) -> str:
    """The ``sde:`` block of ``sde`` (see the module docstring)."""
    text = _dump(sde_spec(sde))
    if verify:
        try:
            again = from_text(text)
            ok = _same(sde, again)
        except Exception as e:
            raise _err(f'the text does not read back ({type(e).__name__}: {e}).') from None
        if not ok:
            diff = [f'{what} {list(a)} (read back: {list(b)})'
                    for what, a, b in (('fields', sde._fields, again._fields),
                                       ('equations', sde._eqs, again._eqs),
                                       ('sources', sde._sources, again._sources))
                    if list(a) != list(b)]
            raise _err('reading the text back gives a different model: the model was declared '
                       'in an order the sde: block does not record'
                       + (f' ({"; ".join(diff)})' if diff else '') + '.  Declare the fields, '
                       'then the equations, then the noise sources.')
    return text


def from_text(text: str, source: str = '<text>'):
    """The :class:`api.sde.SDE` of an ``sde:`` block (a config holding
    ``sde:``, or the block's mapping alone), read with the parser and the
    builder of ``daedalus run``; errors are line-numbered
    (:class:`api.config_run.ConfigError`)."""
    from api.config_run import sde_block_from_text, sde_from_block
    return sde_from_block(sde_block_from_text(text, source), source)

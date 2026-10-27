"""Paper-style equations for the SDE front end: :meth:`api.sde.SDE.sde`.

One equation per line (or per ``;``), written as in a paper::

    dx = (-mu*x - eps*x^3) dt + sqrt(2*D) dW
    tau*dv = (E - v) dt + sigma ∘ dW_a + J dN(r*exp(v))

Each side is read into the existing declarations, nothing else:

* the left side is ``d<field>``, optionally with a constant factor
  (``tau*dv``, ``tau dv``); the field is declared if it is not yet;
* the right side is a sum of terms, each ending in a differential: ``dt``
  (the drift, ``(f) dt`` or ``f*dt``), ``dW`` (a standard Wiener process,
  ``Gaussian(var=1)``), ``dN(rate)`` (a Poisson counting process,
  ``Poisson(rate)``) or ``d<X>_<name>`` for a source given in ``noises=``
  (``noises={'L': GammaProcess(...)}`` and ``dL_L``, or ``dW_a`` with a
  Gaussian of another variance).  The factor in front is the coupling;
* a subscript names the process (``dW_a``, ``dW_{a}``, ``dN_b(rate)``): one
  name used in several equations is ONE source coupled to all of them.  An
  unnamed ``dW`` / ``dN`` is a source of its own equation (``W_x``,
  ``N_x``);
* a circle in front of the differential (``∘ dW``, or ASCII ``o dW``; the
  diamond ``◇``, ``⋄`` reads the same) marks the term Stratonovich for a
  Gaussian source and Marcus for a jump source; a plain term is Ito.  The
  interpretation is per source (``noise(..., interpretation=)``), so a model
  may mix them; one source takes one interpretation.

* the operator form puts a linear operator on the left and writes the
  right side per unit time, a noise term as ``c dW/dt`` (``c dN(rate)/dt``,
  ``c dX_name/dt``): ``(tau*Dt + 1)*v = E + sum(w[i,j]*v[j] for j in pop) +
  s dW/dt`` or ``(m*Dt^2 + gamma*Dt + k)*x = sqrt(2*gamma*T) dW/dt`` (a
  higher-order lhs is embedded, see :meth:`api.sde.SDE.equation`).  The
  left side acts on one field; the deterministic terms are the rhs of
  :meth:`~api.sde.SDE.equation` (a population sum is fine here).

So ``m.sde('dx = (-mu*x) dt + sqrt(2*D) dW')`` is
``m.field('x').drift('x', '-mu*x').noise('W_x', Gaussian(var=1),
into={'x': 'sqrt(2*D)'}, interpretation='ito')``, byte for byte.  A malformed
line raises :class:`api.sde.SDEError` with a caret under the failing
character.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Optional

from api.noise import (Gaussian, NoiseSource, Poisson, expr_to_text,
                       expression_syntax_problem, format_caret_error)

__all__ = ['parse_paper', 'apply_paper', 'paper_text']

# the circle marks (a Stratonovich / Marcus term); 'o' only as a token of its own
_MARKS = ('∘', '◦', '○', '◇', '⋄', '◊')
_FUNCS = {'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'arctan', 'abs',
          'heaviside', 'erf', 'sum', 'for', 'in', 'pi', 'e', 'I', 'Conv', 'i', 'j'}
_DIFF = re.compile(r'd(?:t|[A-Z](?:_\{?\w+\}?)?)$')


def _err(SDEError, line, at, headline, hint=''):
    return SDEError(format_caret_error(line, at, f'sde(): {headline}', hint))


def _statements(text: str):
    """``[(statement, line)]``: one per line or ``;``; a line inside an open
    bracket, starting with ``+`` / ``-`` or with no ``=`` continues the one
    before; ``#`` starts a comment."""
    out = []
    for raw in str(text).replace(';', '\n').split('\n'):
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        depth = 0
        if out:
            depth = sum({'(': 1, '[': 1, ')': -1, ']': -1}.get(c, 0) for c in out[-1])
        if out and (depth > 0 or line.lstrip()[:1] in '+-' or '=' not in line
                    or out[-1].rstrip()[-1:] in '+-*/^('):
            out[-1] = out[-1] + ' ' + line.strip()
        else:
            out.append(' '.join(line.split()))
    return out


def _split_terms(rhs: str, offset: int):
    """The top-level terms of ``rhs``: ``[(sign, text, start)]`` (``start`` the
    index of ``text`` in the statement).  A sign is binary after an operand
    (not after an operator or an exponent ``1e``)."""
    terms, depth, start, sign = [], 0, 0, '+'
    k = 0
    while k < len(rhs):
        ch = rhs[k]
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch in '+-' and depth == 0:
            before = rhs[:k].rstrip()
            binary = (before and (before[-1].isalnum() or before[-1] in ')]_.')
                      and not re.search(r'(?<![\w.])\d+\.?\d*[eE]$', before))
            if binary or not before:
                body = rhs[start:k]
                if body.strip():
                    terms.append((sign, body, offset + start))
                elif before:
                    pass
                sign, start = ch, k + 1
        k += 1
    terms.append((sign, rhs[start:], offset + start))
    out = []
    for sign, body, at in terms:
        lead = len(body) - len(body.lstrip())
        if body.strip():
            out.append((sign, body.strip(), at + lead))
    return out


def _strip_parens(text: str) -> str:
    """``(f)`` -> ``f`` when the brackets enclose all of ``text``."""
    t = text.strip()
    while t.startswith('(') and t.endswith(')'):
        depth = 0
        for k, ch in enumerate(t):
            depth += {'(': 1, ')': -1}.get(ch, 0)
            if depth == 0 and k < len(t) - 1:
                return t
        t = t[1:-1].strip()
    return t


def _split_differential(term: str):
    """``(coefficient, mark, differential, argument)`` of one term, or
    ``None`` when it does not end in a differential.  ``argument`` is the
    text of ``dN(...)`` (``None`` without brackets)."""
    t, arg = term.rstrip(), None
    if t.endswith(')'):
        depth = 0
        for k in range(len(t) - 1, -1, -1):
            depth += {')': 1, '(': -1}.get(t[k], 0)
            if depth == 0:
                break
        head = t[:k].rstrip()
        m = re.search(r'(?<![\w.])d[A-Z](?:_\{?\w+\}?)?$', head)
        if m:
            arg, t = t[k + 1:-1].strip(), head
    m = re.search(r'(?<![\w.])(' + _DIFF.pattern[:-1] + r')$', t)
    if not m:
        return None
    diff, coef = m.group(1), t[:m.start()].rstrip()
    mark, c = '', coef.rstrip(' *')
    if c.endswith(_MARKS) or re.search(r'(?:^|[\s*)\]])o$', c):
        mark, coef = c[-1], c[:-1]
    coef = coef.rstrip()
    if coef.endswith('*'):
        coef = coef[:-1].rstrip()
    return coef, mark, re.sub(r'_\{(\w+)\}$', r'_\1', diff), arg, m.start(1)


def _operator_field(lhs: str, fields, context):
    """The field an operator-form left side acts on (``(tau*Dt + 1)*v`` ->
    ``v``), or ``None`` when ``lhs`` holds no time derivative.  Raises
    ``ValueError`` (with the candidates) unless exactly one name is a field
    or undeclared."""
    tnames = context.get('tnames') or ['Dt']
    if not any(re.search(rf'(?<!\w){re.escape(n)}(?!\w)', lhs) for n in tnames):
        return None
    known = set(context.get('known', ())) | set(tnames) | _FUNCS
    cands = []
    for n in re.findall(r'(?<![\w.])[^\W\d]\w*', lhs):
        if n not in known and n not in cands:
            cands.append(n)
    if len(cands) != 1:
        raise ValueError(cands)
    return cands[0]


def parse_paper(text: str, fields, SDEError, context: Optional[dict] = None):
    """The equations of ``text``: ``[{'field', 'lhs_factor', 'drift',
    'terms': [{'name', 'kind', 'coef', 'mark', 'arg', 'diff'}], 'line'}]``."""
    eqs = []
    for line in _statements(text):
        depth, eq_at = 0, -1
        for k, ch in enumerate(line):
            depth += {'(': 1, '[': 1, ')': -1, ']': -1}.get(ch, 0)
            if ch == '=' and depth == 0 and line[k - 1:k] not in ('<', '>', '!', '=') \
                    and line[k + 1:k + 2] != '=':
                if eq_at >= 0:
                    raise _err(SDEError, line, k, 'a second "=" in one equation',
                               'one equation per line (or separate them with ";")')
                eq_at = k
        if eq_at < 0:
            raise _err(SDEError, line, len(line), 'an equation needs "="',
                       'write dx = (drift) dt + (coupling) dW')
        lhs = line[:eq_at].strip()
        m = re.fullmatch(r'(?:(.*?)\s*\*?\s*)?(?<![\w.])d([A-Za-z]\w*)', lhs)
        op_lhs = None
        if not m or m.group(2) == 't':
            try:
                field = _operator_field(lhs, fields, context or {})
            except ValueError as exc:
                raise _err(SDEError, line, 0,
                           f'the operator left side {lhs!r} must act on one field (found '
                           f'{", ".join(exc.args[0]) or "none"})',
                           'write the operator on the field alone, e.g. (tau*Dt + 1)*v') from None
            if field is None:
                raise _err(SDEError, line, 0, f'the left side {lhs!r} is not d<field>',
                           'write dx = ... (or tau*dx = ... with a constant factor, or an operator '
                           'form such as (m*Dt^2 + gamma*Dt + k)*x = ... + c dW/dt)')
            prob = expression_syntax_problem(lhs)
            if prob:
                raise _err(SDEError, line, prob[0], f'cannot read the left side {lhs!r}: {prob[1]}',
                           prob[2])
            op_lhs, m = lhs, None
        factor = (m.group(1) or '').strip() or None if m else None
        field = m.group(2) if m else field
        if factor is not None:
            prob = expression_syntax_problem(factor)
            if prob:
                raise _err(SDEError, line, prob[0], f'cannot read the factor {factor!r} of d{field}: '
                           f'{prob[1]}', prob[2])
        rhs_at = eq_at + 1 + (len(line[eq_at + 1:]) - len(line[eq_at + 1:].lstrip()))
        drift, terms, seen = [], [], {}
        for sign, body, at in _split_terms(line[eq_at + 1:].strip(), rhs_at):
            if op_lhs is not None:
                # operator form: a noise term is c dW/dt, every other term is deterministic
                per = re.search(r'\s*/\s*dt\s*$', body)
                sp = _split_differential(body[:per.start()]) if per else _split_differential(body)
                if per is None:
                    if sp is not None:
                        raise _err(SDEError, line, at + sp[4],
                                   f'in the operator form the term {body!r} is per unit time',
                                   'write the drift without dt and a noise term as c dW/dt')
                    drift.append((sign, body))
                    continue
                if sp is None or sp[2] == 'dt':
                    raise _err(SDEError, line, at, f'the term {body!r} is not c dW/dt',
                               'a noise term of the operator form is c dW/dt, c dN(rate)/dt or '
                               'c dX_name/dt')
                body = body[:per.start()]
            else:
                sp = _split_differential(body)
            if sp is None:
                raise _err(SDEError, line, at + len(body) - 1 if body else at,
                           f'the term {body!r} of d{field} has no differential',
                           'every term ends in dt, dW, dN(rate) or d<X>_<name>, separated from its '
                           'factor by a space or "*", e.g. (-mu*x) dt or sqrt(2*D) dW')
            coef, mark, diff, arg, dpos = sp
            coef = _strip_parens(coef) if coef else '1'
            if coef != '1':
                prob = expression_syntax_problem(coef)
                if prob:
                    raise _err(SDEError, line, at + body.find(coef[:1]) + prob[0]
                               if coef[:1] and body.find(coef[:1]) >= 0 else at,
                               f'cannot read the coefficient {coef!r}: {prob[1]}', prob[2])
            if diff == 'dt':
                if mark:
                    raise _err(SDEError, line, at + body.rfind(mark),
                               'a circle marks a noise term, not the drift',
                               'write the drift as (f) dt; mark the noise: g ∘ dW')
                if arg is not None:
                    raise _err(SDEError, line, at, 'dt takes no argument')
                drift.append((sign, coef))
                continue
            letter, _, name = diff[1:].partition('_')
            if letter == 'N' and arg is not None:
                prob = expression_syntax_problem(arg)
                if prob:
                    raise _err(SDEError, line, at + body.rfind('(') + 1 + prob[0],
                               f'cannot read the rate {arg!r}: {prob[1]}', prob[2])
            elif arg is not None:
                raise _err(SDEError, line, at + body.rfind('('),
                           f'{diff} takes no argument (only dN(rate) does)',
                           f'give the source in noises= and write {diff}')
            key = name or f'{letter}_{field}'
            if key in seen:
                raise _err(SDEError, line, at + dpos,
                           f'{diff} appears twice in the equation of d{field}',
                           'add the coefficients in one term, or name independent processes '
                           '(dW_a, dW_b)')
            seen[key] = True
            terms.append({'name': key, 'named': bool(name), 'letter': letter,
                          'coef': coef if sign == '+' else _neg(coef),
                          'mark': mark, 'arg': arg, 'diff': diff, 'at': at + dpos})
        if not drift and not terms:
            raise _err(SDEError, line, rhs_at, f'd{field} has no terms')
        if op_lhs is not None and len(drift) == 1:
            drift = [(drift[0][0], _strip_parens(drift[0][1]))]
        eqs.append({'field': field, 'factor': factor, 'drift': _join_drift(drift),
                    'terms': terms, 'line': line, 'op_lhs': op_lhs})
    if not eqs:
        raise SDEError('sde(): no equation in the text (write dx = (drift) dt + ... dW).')
    names = [e['field'] for e in eqs]
    for f in names:
        if names.count(f) > 1:
            raise SDEError(f'sde(): d{f} has two equations.')
    return eqs


def _neg(c: str) -> str:
    """``-c``, bracketed when ``c`` is a sum (or starts with a sign)."""
    return f'-{c}' if not _is_sum(c) else f'-({c})'


def _is_sum(c: str) -> bool:
    return c.lstrip().startswith(('-', '+')) or len(_split_terms(c, 0)) > 1


def _atomic(t: str) -> bool:
    return re.fullmatch(r'[\w.]+(\[[^\]]*\])?|[\w.]+\([^()]*\)', t) is not None \
        and not t.startswith('-')


def _join_drift(drift) -> str:
    """The drift text: one ``dt`` term as written, several joined with their
    signs (``-mu*x*dt - eps*x^3*dt`` -> ``-mu*x - eps*x^3``)."""
    if not drift:
        return '0'
    out = ''
    for k, (sign, c) in enumerate(drift):
        if k == 0:
            out = c if sign == '+' else _neg(c)
        else:
            out += f' {sign} ' + (f'({c})' if _is_sum(c) else c)
    return out


def apply_paper(sde, text: str, noises: Optional[dict] = None):
    """Declare what ``text`` says on the builder ``sde`` (see the module
    docstring): fields not yet declared, one equation per field, one noise
    source per process with its couplings and interpretation."""
    from api.sde import SDEError
    if sde._interp != 'ito':
        raise SDEError(f"sde(): the model has interpretation({sde._interp!r}); paper-style "
                       f"terms carry their own (a plain dW is Ito, ∘dW Stratonovich, ∘dN "
                       f"Marcus): drop interpretation(...) and mark the terms.")
    text = sde._unicode_dt(str(text))          # ∂t: the time derivative of an operator form
    noises = dict(noises or {})
    for k, v in noises.items():
        if not isinstance(v, NoiseSource):
            raise SDEError(f'sde(noises={{{k!r}: ...}}): give a noise source from api.noise '
                           f'(Gaussian, Poisson, CompoundPoisson, ...); got {type(v).__name__}.')
    known = (set(sde._param_names) | set(sde._function_names) | set(sde._kernels)
             | set(sde._operators) | {p['name'] for p in sde._pops})
    eqs = parse_paper(text, sde._fields, SDEError,
                      {'tnames': sde._time_names(), 'known': known})
    # the sources, in order of first appearance: name -> (source, couplings, mark)
    srcs: 'OrderedDict[str, dict]' = OrderedDict()
    for e in eqs:
        for t in e['terms']:
            name, line = t['name'], e['line']
            if name in noises:
                if t['arg'] is not None:
                    raise _err(SDEError, line, t['at'],
                               f'{t["diff"]}({t["arg"]}): the source {name!r} is given in '
                               f'noises=, so it takes no rate here', f'write {t["diff"]}')
                src = noises[name]
            elif t['letter'] == 'W':
                src = Gaussian(var=1)
            elif t['letter'] == 'N':
                if t['arg'] is None:
                    raise _err(SDEError, line, t['at'] + len(t['diff']),
                               f'{t["diff"]} needs its rate', f'write {t["diff"]}(rate), e.g. '
                               f'dN(lam)')
                src = None
            else:
                raise _err(SDEError, line, t['at'],
                           f'{t["diff"]}: no source {name!r} is given',
                           f"pass it as noises={{{name!r}: <source>}} (dW and dN(rate) need none)")
            have = srcs.get(name)
            if have is None:
                if name in sde._sources:
                    raise _err(SDEError, line, t['at'],
                               f'a noise source named {name!r} is already declared',
                               'name the process differently (dW_b)' if t['named'] else
                               'name the process (dW_a)')
                srcs[name] = {'source': src, 'rate': t['arg'], 'couples': OrderedDict(),
                              'mark': t['mark'], 'diff': t['diff']}
                have = srcs[name]
            else:
                if bool(have['mark']) != bool(t['mark']):
                    raise _err(SDEError, line, t['at'],
                               f'the process {name!r} is marked in one equation and not in '
                               f'another', 'one source takes one interpretation: mark every '
                               'term of it (∘) or none')
                if have['rate'] != t['arg'] and name not in noises:
                    raise _err(SDEError, line, t['at'],
                               f'the process {name!r} has the rate {have["rate"]!r} in one '
                               f'equation and {t["arg"]!r} here', 'one source has one rate')
            have['couples'][e['field']] = t['coef']
    for e in eqs:
        if e['field'] not in sde._fields:
            sde.field(e['field'])
    for e in eqs:
        f, drift = e['field'], e['drift']
        if e['op_lhs'] is not None:
            sde.equation(f, lhs=e['op_lhs'], rhs=drift)
        elif e['factor'] is None:
            sde.drift(f, drift)
        else:
            t = sde._time_names()
            fac = _strip_parens(e['factor'])
            fac = f'({fac})' if _is_sum(fac) else fac
            sde.equation(f, lhs=f'{fac}*{t[0] if t else "Dt"}*{f}', rhs=drift)
    for name, s in srcs.items():
        src = s['source'] if s['source'] is not None else Poisson(rate=s['rate'])
        interp = 'ito' if not s['mark'] else ('stratonovich' if src.is_gaussian else 'marcus')
        sde.noise(name, src, into=dict(s['couples']), interpretation=interp)
        sde._sources[name]['paper'] = True
    return sde


def paper_text(sde) -> str:
    """The equations of ``sde`` written as :meth:`api.sde.SDE.sde` reads them
    (``a*Dt*x = rhs`` as ``a*dx = (rhs) dt + ...``, any other lhs, a
    higher-order one included, in the operator form; sources coupled by
    ``into=``, Gaussian sources of variance 1, Poisson sources of size 1, or
    sources named in the returned text's ``noises``): ``(text, noises)``."""
    from api.sde import SDEError, _DT
    lines, noises = [], {}
    dt = (sde._time_names() or ['Dt'])[0]
    for name, s in sde._sources.items():
        if s.get('pp') or s['expose']:
            raise SDEError(f'paper_text(): source {name!r} is a point process or exposed; '
                           f'it has no paper-style term.')
        if s.get('colored') or type(s['source']).__name__ == 'GaussianVector':
            raise SDEError(f'paper_text(): source {name!r} is coloured or a GaussianVector; it has '
                           f'no paper-style term.')
    for f, eq in sde._eqs.items():
        if sde._fields[f].get('embedded_of'):
            continue                          # (written with its higher-order equation)
        emb, key, head = sde._embedded.get(f), f, None
        if emb is not None:
            if emb['fields'] != [f + 'd' * j + 'ot' for j in range(1, emb['order'])]:
                raise SDEError(f'paper_text(): the velocities {", ".join(emb["fields"])} of {f!r} '
                               f'were named with velocity=; the paper-style line names them '
                               f'{f}dot, ... (rename them to write the text).')
            lhs, rhs, key = emb['lhs_raw'].strip(), emb['rhs_raw'].strip(), emb['top']
        else:
            lhs, rhs = eq['lhs_raw'].strip(), sde._redelay(eq['rhs_raw'].strip())
            L = sde._parse(eq['lhs'], f'lhs of {f}').expand()
            a = L.coefficient(_DT, 1).coefficient(sde._sym(f), 1)
            if not a.is_zero() and (L - a * _DT * sde._sym(f)).expand() == 0:
                if lhs in (f'{dt}*{f}', f'{dt}({f})'):
                    head = f'd{f}'
                else:
                    m = re.fullmatch(rf'(.+?)\*{re.escape(dt)}\*{re.escape(f)}', lhs)
                    if m:
                        head = f'{_strip_parens(m.group(1))}*d{f}'
        per = '' if head is not None else '/dt'      # the operator form: noise per unit time
        parts = ([] if rhs == '0' else [f'({rhs}) dt' if head is not None else f'({rhs})'])
        for name, s in sde._sources.items():
            if key not in s['couples']:
                continue
            src, c = s['source'], str(s['couples'][key])
            interp = sde._src_interp(name)
            mark = '∘ ' if interp != 'ito' else ''
            # an unnamed process (W_x / N_x of its own equation) is written bare
            bare = len(s['couples']) == 1
            if type(src).__name__ == 'Gaussian' and str(src.params['var']) == '1' \
                    and str(src.params['mean']) == '0' and getattr(src, 'space', None) is None:
                diff = 'dW' if bare and name == f'W_{f}' else f'dW_{name}'
            elif type(src).__name__ == 'Poisson' and str(src.params.get('size', 1)) == '1':
                diff = ('dN' if bare and name == f'N_{f}' else f'dN_{name}') \
                    + f'({expr_to_text(src.params["rate"])})'
            else:
                letter = 'W' if src.is_gaussian else 'L'
                diff = f'd{letter}_{name}'
                noises[name] = src
            c = c.strip()
            parts.append(('' if c == '1' else f'({c}) ' if _is_sum(c) else f'{c} ') + mark + diff
                         + per)
        if head is None:
            lines.append(f'{lhs} = ' + (' + '.join(parts) if parts else '0'))
        else:
            lines.append(f'{head} = ' + (' + '.join(parts) if parts else '0 dt'))
    return '\n'.join(lines), noises

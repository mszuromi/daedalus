"""LaTeX for an :class:`api.sde.SDE`: the SDE as written, the noise sources, the
derived action and the mean-field equations (:meth:`api.sde.SDE.latex`).

Strings only: nothing is compiled.  Expressions go through Sage's ``latex()``
on symbols whose names are fixed here (``_ltx_p_<name>`` for parameters,
``_ltx_q_<name>`` for fields, ``_ltx_r_...`` for an operator with its field),
so the output depends on the model alone and not on what the Sage session
defined earlier.  Sage orders the factors and terms of an expression by symbol
name, which makes the output the same on every run, parameters before fields.
The operators are typeset as ``\\partial_t`` and ``\\nabla^2`` and stand in front
of the one field they act on (``D \\nabla^2 u``).

Population sums (``sum(... for j in E)``) and ``Conv(K, x)`` are typeset here
(Sage does not parse them).  An expression the typesetter cannot read is
returned as ``\\texttt{...}`` rather than raised: the derivation itself has
already accepted it.
"""
from __future__ import annotations

import re
import zlib

from sage.all import SR, latex, sage_eval, function

from api.noise import _MATH_FUNCS, _SAGE_PRINTED_CALLS, _flatten_indices, expr_to_text

__all__ = ['render_latex']

# short names that Sage would print as upright words
_GREEK = {'eps': r'\varepsilon', 'lam': r'\lambda', 'sig': r'\sigma', 'gam': r'\gamma',
          'del': r'\delta', 'kap': r'\kappa'}
_IMPLICIT_POPS = ('pop', 'pop_all')
_FUNCS: dict = {}


def _crc(text: str) -> str:
    return format(zlib.crc32(text.encode()) & 0xffffffff, '08x')


def _ident(text: str) -> str:
    return re.sub(r'[^0-9A-Za-z_]', '_', text)


def _name_latex(name: str) -> str:
    """``tau_g`` -> ``\\tau_{g}``, ``mu`` -> ``\\mu``, ``eps`` -> ``\\varepsilon``,
    ``s0`` -> ``s_{0}``."""
    return _GREEK[name] if name in _GREEK else str(latex(SR.var(name)))


def _escape(text: str) -> str:
    special = {'\\': r'\textbackslash{}', '^': r'\^{}', '~': r'\~{}'}
    return r'\texttt{' + ''.join(special.get(c, '\\' + c if c in '_#%&{}$' else c)
                                 for c in text) + '}'


def _func(name: str, ln: str, template):
    """A formal function that prints through ``template(*argument_latex)``.
    One per (name, latex): pynac keeps functions by name for the session."""
    key = f'{name}_{_crc(ln)}'
    if key not in _FUNCS:
        _FUNCS[key] = function(f'_ltx_f_{_ident(key)}',
                               print_latex_func=lambda self, *a: template(*[latex(q) for q in a]))
    return _FUNCS[key]


def _conv(a, b):
    return r'\left(' + a + r' \ast ' + b + r'\right)'


def _dt_call(x):
    return r'\partial_t ' + x


def _lap_call(x):
    atom = re.fullmatch(r'[\\\w{}^_]+', x.strip()) is not None
    return r'\nabla^2 ' + (x if atom else r'\left(' + x + r'\right)')


class _Typesetter:
    """Expression texts of one SDE to LaTeX."""

    def __init__(self, sde, d, star: bool = False):
        self.sde = sde
        self.star = star
        self.spatial = bool(d.get('spatial'))
        pops = sde._populations()
        # one unit: the population index is not shown
        self.trivial = self.spatial or (len(pops) == 1 and pops[0]['size'] == 1)
        self.fields = list(sde._fields)
        self.field_latex = {f: (info['latex'] or _name_latex(f))
                            for f, info in sde._fields.items()}
        self.func_latex = {n: (kw.get('latex') or _name_latex(n))
                           for (n, _a, _e), kw in sde._functions}
        self.sym_ln: dict = {}            # symbol name -> latex name
        self.bare_ok: set = set()         # latex names whose braces may be dropped
        self._ph: dict = {}               # placeholder text -> symbol (typeset sums)

    # ── symbols ──
    def _sym(self, name: str, ln: str, group: str):
        # the groups order the factors of a product: parameters, responses,
        # fields, operators with their field, sums
        s = f'_ltx_{group}_{_ident(name)}_{_crc(ln)}'
        self.sym_ln[s] = ln
        return SR.var(s, latex_name=ln)

    def _indexed(self, ln: str, idx: list, lead: tuple = ()) -> str:
        idx = [] if (self.trivial and idx == ['i']) else list(idx)
        idx = list(lead) + idx
        if not idx:
            return ln
        sub = ''.join(idx) if all(len(i) == 1 for i in idx) else ','.join(idx)
        head = ln if re.fullmatch(r'\\[A-Za-z]+|[A-Za-z]', ln) else '{' + ln + '}'
        return f'{head}_{{{sub}}}'

    def _plain(self, n: str):
        """A symbol of the flattened text: ``x__i`` / ``xt__i`` / ``w__i__j`` / ``mu``."""
        base, *idx = n.split('__')
        if base in self.fields:
            return self._sym(n, self._indexed(self.field_latex[base], idx,
                                              (r'\ast',) if self.star else ()), 'r')
        if base.endswith('t') and base[:-1] in self.fields:
            return self._sym(n, self._indexed(r'\tilde{' + self.field_latex[base[:-1]] + '}', idx), 'q')
        return self._sym(n, self._indexed(_name_latex(base), idx), 'p')

    def _called(self, n: str):
        base, *idx = n.split('__')
        ln = self._indexed(self.func_latex.get(base) or _name_latex(base), idx)
        return _func(n, ln, lambda *a, _ln=ln: _ln + r'\left(' + ', '.join(a) + r'\right)')

    def _locals(self, text: str) -> dict:
        called = set(re.findall(r'\b([A-Za-z_]\w*)\s*\(', text))
        loc = {}
        for n in set(re.findall(r'\b([A-Za-z_]\w*)\b', text)):
            if n in self._ph:
                loc[n] = self._ph[n]
            elif n == 'Dt' and n not in called:
                loc[n] = self._sym(n, r'\partial_t', 't')
            elif n == 'Laplacian':
                loc[n] = self._sym(n, r'\nabla^2', 't')
            elif n == 'Conv' and n in called:
                loc[n] = _func('Conv', '', _conv)
            elif n in ('Dt', 'Lap') and n in called:
                loc[n] = _func(n, '', _dt_call if n == 'Dt' else _lap_call)
            elif n in _MATH_FUNCS:
                loc[n] = _MATH_FUNCS[n]
            elif n in called and n in _SAGE_PRINTED_CALLS and n not in self.sde._function_names:
                loc[n] = _SAGE_PRINTED_CALLS[n]
            elif n in called:
                loc[n] = self._called(n)
            elif n not in ('pi', 'I', 'e'):
                loc[n] = self._plain(n)
        return loc

    # ── sums ──
    def _sums(self, text: str) -> str:
        """Replace each ``sum(body for J in P)`` by a placeholder symbol that
        prints as the typeset sum."""
        out, pos = '', 0
        while True:
            m = re.compile(r'(?<![\w.])sum\s*\(').search(text, pos)
            if not m:
                return out + text[pos:]
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1, '[': 1, ']': -1}.get(text[k], 0)
                k += 1
            body, cut, depth = text[m.end():k - 1], -1, 0
            for q in range(len(body)):
                depth += {'(': 1, ')': -1, '[': 1, ']': -1}.get(body[q], 0)
                if depth == 0 and re.match(r'\s+for\s+', body[q:]):
                    cut = q
            hm = re.fullmatch(r'\s+for\s+([A-Za-z_]\w*)\s+in\s+([A-Za-z_]\w*)\s*',
                              body[cut:]) if cut >= 0 else None
            if not hm:
                raise ValueError('cannot read the sum')
            j, pop = hm.groups()
            rng = f'{j}=1}}^{{N' if pop in _IMPLICIT_POPS else f'{j} \\in {pop}'
            ln = rf'\sum_{{{rng}}}\left[{self.expr(body[:cut])}\right]'
            ph = f'_phsum{_crc(ln)}'
            self._ph[ph] = self._sym(ph, ln, 'u')
            self.bare_ok.add(ln)
            out += text[pos:m.start()] + ph
            pos = k

    def _delays(self, text: str) -> str:
        """``Conv(K, x)`` with a pure delay kernel (``x(t - d)`` as written)
        -> a placeholder symbol that prints as ``x(t - d)``."""
        sde, out, pos = self.sde, '', 0
        try:
            spans = sde._conv_spans(text)
        except Exception:                                    # noqa: BLE001
            return text
        for a, b, c in spans:
            args = [q.strip() for q in text[b:c - 1].split(',', 1)]
            k = sde._kernels.get(args[0]) if len(args) == 2 else None
            if k is None or not k.get('delay') or k.get('freq_image') != '1':
                continue
            ln = rf'{self.expr(args[1])}(t - {self.expr(k["delay"])})'
            ph = f'_phdel{_crc(ln)}'
            self._ph[ph] = self._sym(ph, ln, 'r')
            self.bare_ok.add(ln)
            out += text[pos:a] + ph
            pos = c
        return out + text[pos:]

    # ── operators ──
    def _group_ops(self, E, loc):
        """Put each operator in front of the one field it acts on:
        ``tau*Dt*v + v`` -> ``tau*(dt v) + v`` with ``dt v`` one symbol."""
        from sage.symbolic.operators import add_vararg
        ops = {n: loc[n] for n in ('Dt', 'Laplacian') if n in loc and E.has(loc[n])}
        if not ops:
            return E
        dt, lap = ops.get('Dt'), ops.get('Laplacian')
        fsyms = [s for n, s in loc.items()
                 if n.split('__')[0] in self.fields and str(s) in self.sym_ln]

        def parts(e):
            return e.operands() if e.operator() is add_vararg else [e]
        out = SR(0)
        for t in parts(E):
            if not any(t.has(s) for s in ops.values()):
                out += t
                continue
            for st in parts(t.expand()):
                a = st.degree(dt) if dt is not None else 0
                b = st.degree(lap) if lap is not None else 0
                c = st.subs({s: 1 for s in ops.values()})
                xs = [x for x in fsyms if c.has(x)]
                if len(xs) == 1 and c.degree(xs[0]) == 1 and (a or b):
                    pre = ((r'\partial_t' + (f'^{{{a}}}' if a > 1 else '')) if a else '') \
                        + (r'\nabla^{%d}' % (2 * b) if b else '')
                    opln = pre + ' ' + self.sym_ln[str(xs[0])]
                    atom = self._sym(f'op{a}{b}_{xs[0]}', opln, 't')
                    self.bare_ok.add(opln)
                    out += c.coefficient(xs[0], 1) * atom
                else:
                    out += st
        return out

    # ── expressions ──
    def _parse(self, text: str, zero_ops: bool = False):
        t2 = _flatten_indices(self._sums(self._delays(' '.join(str(text).split()))))
        loc = self._locals(t2)
        E = SR(sage_eval(t2, locals=loc))
        if zero_ops:
            return E.subs({s: 0 for n, s in loc.items() if n in ('Dt', 'Laplacian')}), loc
        return self._group_ops(E, loc), loc

    def expr(self, text, *, zero_ops: bool = False) -> str:
        """LaTeX of the expression ``text`` (the engine's or the user's spelling)."""
        try:
            return self._tidy(latex(self._parse(text, zero_ops)[0]))
        except Exception:                                    # noqa: BLE001
            return _escape(' '.join(str(text).split()))

    def is_sum(self, text) -> bool:
        from sage.symbolic.operators import add_vararg
        try:
            return self._parse(text)[0].operator() is add_vararg
        except Exception:                                    # noqa: BLE001
            return True

    def _tidy(self, s: str) -> str:
        """Drop the braces Sage puts around a symbol with a LaTeX name when
        that cannot change the meaning (no superscript in the name)."""
        for ln in sorted(set(self.sym_ln.values()), key=len, reverse=True):
            pat = r'(?<![A-Za-z0-9}])\{' + re.escape(ln) + r'\}'
            if ln in self.bare_ok or '^' not in ln:
                s = re.sub(pat, lambda m, _l=ln: _l, s)
            else:                       # (a name with its own superscript keeps the group)
                s = re.sub(pat + r'(?![\^_])', lambda m, _l=ln: _l, s)
        return s


# ── the outputs ──────────────────────────────────────────────────────────

def _letter(s) -> str:
    src = s.get('source')
    if src is None or type(src).__name__ in ('Poisson', 'CompoundPoisson'):
        return 'N'
    return 'W' if src.is_gaussian else 'L'


def _join(terms) -> str:
    """``a + b - c`` from terms that may start with a minus sign."""
    out = ''
    for t in terms:
        out += (' - ' + t[1:].lstrip()) if t.startswith('-') else ' + ' + t
    return out


def _block(rows) -> str:
    return r'\begin{aligned}' + '\n' + ' \\\\\n'.join(rows) + '\n' + r'\end{aligned}'


def render_latex(sde) -> dict:
    """``{'sde', 'noise', 'action', 'mean_field'}`` of ``sde`` as LaTeX strings."""
    from api.sde import _DT
    d = sde._derive()
    ts = _Typesetter(sde, d)
    idx = (lambda t: t) if ts.trivial else sde._index
    sub = {n: _name_latex(n) for n in sde._sources}

    def paren(text, ln, neg=False):
        return r'\left(' + ln + r'\right)' if ts.is_sum(idx(text)) or (neg and ln.startswith('-')) else ln

    # the SDE as written: dx = (...) dt + c sqrt(var) dW + c dN
    rows = []
    for f, eq in sde._eqs.items():
        lhs_txt, rhs_txt = eq['lhs'], eq['rhs']
        rhs_ln = ts.expr(idx(rhs_txt))
        diff_terms, dot_terms = [], []
        for name, s in sde._sources.items():
            if f not in s['couples'] or s.get('pp'):
                continue
            c, src, ltr = str(s['couples'][f]), s['source'], _letter(s)
            lead = '' if c.strip() == '1' else paren(c, ts.expr(idx(c))) + r'\,'
            if src.is_gaussian:
                v = expr_to_text(src.params['var'])
                if v.strip() != '1':
                    lead += r'\sqrt{' + ts.expr(idx(v)) + r'}\,'
                if not SR(src.params['mean']).is_zero():
                    m = expr_to_text(src.params['mean'])
                    diff_terms.append(ts.expr(idx(f'({c})*({m})')) + r'\,\mathrm{d}t')
            diff_terms.append(lead + rf'\mathrm{{d}}{ltr}_{{{sub[name]}}}')
            dot_terms.append(lead + rf'\dot{{{ltr}}}_{{{sub[name]}}}')
        f_ln = ts.field_latex[f] + ('' if ts.trivial else '_{i}')
        try:
            dt_form = (sde._parse(lhs_txt, f'lhs of {f}') - _DT * sde._sym(f)).expand().is_zero()
        except Exception:                                    # noqa: BLE001
            dt_form = False
        if dt_form:
            rows.append(rf'\mathrm{{d}}{f_ln} &= {paren(rhs_txt, rhs_ln)}\,\mathrm{{d}}t'
                        + _join(diff_terms))
        else:
            rows.append(f'{ts.expr(idx(lhs_txt))} &= {rhs_ln}' + _join(dot_terms))
    out = {'sde': _block(rows)}

    # the noise sources
    rows = []
    for name, s in sde._sources.items():
        ltr = _letter(s)
        nm = f'{ltr}_{{{sub[name]}}}'
        dot = rf'\dot{{{ltr}}}_{{{sub[name]}}}'
        src = s.get('source')
        tail = (rf'\quad (\text{{the field }} {ts.field_latex[s["expose"]]})'
                if s['expose'] else '')
        if s.get('pp'):
            size = s['pp']['size'].strip()
            rows.append(rf'{ts.field_latex[s["expose"]]} &= {dot},\quad '
                        rf'\mathbb{{E}}[\mathrm{{d}}{nm} \mid \text{{past}}] = '
                        + ('' if size == '1' else ts.expr(size) + r'\,')
                        + ts.expr(idx(s['pp']['rate'])) + r'\,\mathrm{d}t')
        elif src.is_gaussian:
            space = r'\,\delta(x - x^{\prime})' if getattr(src, 'space', None) == 'white' else ''
            rows.append(rf'\langle {dot}(t)\,{dot}(t^{{\prime}}) \rangle &= '
                        rf'{ts.expr(expr_to_text(src.params["var"]))}{space}\,'
                        rf'\delta(t - t^{{\prime}}){tail}')
        else:
            ps = ', '.join(rf'\mathrm{{{k.replace("_", chr(92) + "_")}}}={ts.expr(expr_to_text(v))}'
                           for k, v in (getattr(src, 'params', None) or {}).items())
            rows.append(rf'{nm} &\sim \text{{{type(src).__name__}}}({ps}){tail}')
    out['noise'] = _block(rows) if rows else ''

    # the derived action, in the order of action_text()
    measure = r'\mathrm{d}t' + (r'\,\mathrm{d}^{%d}x' % sde._space if ts.spatial else '')
    rank = {'dynamics': 0, 'noise': 1, 'extra': 2}
    terms = []
    for t in sorted(sde._action_log, key=lambda t: rank[t['kind']]):
        if t['kind'] == 'dynamics':
            resp = ts.expr(f"{t['field']}t[i]")
            ln = (resp + r'\left(' + ts.expr(t['lhs']) + ' - '
                  + paren(t['rhs'], ts.expr(t['rhs']), neg=True) + r'\right)')
            pop = None if ts.spatial or ts.trivial else sde._pop_of_field(t['field'])
        else:
            ln = ts.expr(t['text'])
            pop = (sde._pop_of_source(t['source'].split(' (part')[0])
                   if t['kind'] == 'noise' and not ts.spatial and not ts.trivial else None)
        if pop is not None:
            under = 'i=1}^{N' if pop in _IMPLICIT_POPS else f'i \\in {pop}'
            ln = rf'\sum_{{{under}}}\left[{ln}\right]'
        terms.append(ln)
    rows = [rf'S &= \int {measure}\,\Big[ {terms[0]}']
    rows += [r'&\qquad ' + ('- ' + t[1:].lstrip() if t.startswith('-') else '+ ' + t)
             for t in terms[1:]]
    rows[-1] += r' \Big]'
    out['action'] = _block(rows)

    # the mean-field equations: stationary and homogeneous, fields at the saddle
    tm = _Typesetter(sde, d, star=True)
    out['mean_field'] = _block([f"{tm.expr(e['lhs'], zero_ops=True)} &= "
                                f"{tm.expr(e['rhs'], zero_ops=True)}" for e in d['equations']])
    return out

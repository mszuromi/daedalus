"""SDE front-end: declare fields, drift and noise SOURCES; derive the MSR
action and the mean-field equations; emit the same model dict
:class:`api.model.TemporalModelBuilder` produces.

Each field ``x`` obeys one equation in the repository's operator form,

    lhs_x = rhs_x + sum_a c_xa * eta_a'

where ``lhs_x`` is the linear differential operator on ``x`` (contains
``Dt``: ``'Dt*x'``, ``'(tau*Dt + 1)*v'``), ``rhs_x`` is everything
deterministic (fields, parameters, defined functions, kernels), and the
noise sources ``eta_a`` (:mod:`api.noise`) couple in through
``couples={field: coefficient}``.  The derived action is

    S = sum_x xt*(lhs_x - rhs_x) - sum_a K_a( sum_x c_xa*xt )

summed over each population's index (``sum(... for i in pop)``), and the
mean-field equations are ``lhs_x|_{Dt=0} = rhs_x + sum_a c_xa*K_a'(0)``.

A source may be *exposed* as a field (``expose='n'``): the field ``n``
(e.g. a spike train) and its response ``nt`` are declared, the pairing
``nt*n - K(nt)`` enters the action, the equations use ``n`` like any
field, and the mean-field system gains ``n = K'(0)``.

Example (OU quartic)::

    m = (SDE('OU quartic')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('eps', default=0.02, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .field('x')
         .drift('x', '-mu*x - eps*x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
         .build())

Expressions may use bare field names (``x``) or indexed ones
(``x[i]``); bare field names and bare calls of declared functions are
indexed by the population index ``i`` automatically.  Population sums
(``sum(w[i,j]*g*n[j] for j in E)``) are written as in the tracked
models.

Conventions: Ito by default (``.interpretation('stratonovich')``
converts Gaussian noise to Ito with the drift correction
``f_i += 1/2 sum_a sum_j [V_a b_ja d_j b_ia + 1/2 b_ja b_ia d_j V_a]``,
``b_ia = c_ia / a_i`` with ``a_i`` the ``Dt`` coefficient of ``lhs_i``
and ``V_a`` the variance rate; it raises for jump and exposed sources).
Time-independent sources only.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Optional

from sage.all import SR, integrate, oo, assume, forget, function

from api.noise import (NoiseError, NoiseSource, THETA, parse_expr,
                       expr_to_text, expr_names, _MATH_FUNCS)

__all__ = ['SDE', 'SDEError']


class SDEError(NoiseError):
    """Raised for an invalid SDE declaration."""


_KEYWORDS = {'sum', 'for', 'in', 'Dt', 'Conv', 'I', 'pi', 'e',
             'if', 'else', 'range'} | set(_MATH_FUNCS)
_DT = SR.var('Dt')


def _split_top_level(text: str, sep: str = ','):
    out, depth, cur = [], 0, ''
    for ch in text:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def _wrap(text) -> str:
    t = str(text).strip()
    return t if re.fullmatch(r'[\w\[\],.\s]*', t) else f'({t})'


class SDE:
    """Builder for an SDE model; see the module docstring."""

    def __init__(self, name: str, n_populations: Optional[int] = None):
        self.name = name
        self._n_pop_default = int(n_populations) if n_populations else 1
        self._pops: list[dict] = []
        self._fields: 'OrderedDict[str, dict]' = OrderedDict()
        self._params: list[tuple[tuple, dict]] = []
        self._param_names: list[str] = []
        self._functions: list[tuple[tuple, dict]] = []
        self._function_names: list[str] = []
        self._kernels: 'OrderedDict[str, dict]' = OrderedDict()
        self._eqs: dict[str, dict] = {}
        self._sources: 'OrderedDict[str, dict]' = OrderedDict()
        self._interp = 'ito'
        self._stability: Optional[bool] = None
        self._extra: list[str] = []
        self._series_order = 8
        self.notes: list[str] = []

    # ── declarations (passed through to TemporalModelBuilder) ─────────
    def population(self, name: str, *, size: int = 1, description: str = ''):
        self._pops.append({'name': name, 'size': max(int(size), 1),
                           'description': description})
        return self

    def field(self, name: str, *, population: Optional[str] = None,
              description: str = '', domain: Optional[str] = None,
              latex: str = ''):
        """Declare a physical field (its response field is ``<name>t``)."""
        self._check_new_name(name)
        self._fields[name] = {'population': population, 'description': description,
                              'domain': domain, 'latex': latex, 'exposed': None}
        return self

    def parameter(self, name: str, default=None, indexed=False,
                  domain: Optional[str] = None, description: str = '',
                  indexed_by: Optional[list] = None):
        if name == 'theta':
            raise SDEError("'theta' is reserved for the CGF variable.")
        self._params.append(((name,), dict(default=default, indexed=indexed,
                                           domain=domain, description=description,
                                           indexed_by=indexed_by)))
        self._param_names.append(name)
        return self

    def define_function(self, name: str, args: list, expression: str,
                        latex: Optional[str] = None, description: str = '',
                        population: Optional[str] = None):
        self._functions.append(((name, list(args), expression),
                                dict(latex=latex, description=description,
                                     population=population)))
        self._function_names.append(name)
        return self

    def define_kernel(self, name: str, *, time_expr: Optional[str] = None,
                      freq_image: Optional[str] = None, latex_name: str = '',
                      indexed=False, indexed_by: Optional[list] = None):
        if time_expr is None and freq_image is None:
            raise SDEError(f'define_kernel({name!r}): give time_expr= or freq_image=.')
        self._kernels[name] = dict(time_expr=time_expr, freq_image=freq_image,
                                   latex_name=latex_name, indexed=indexed,
                                   indexed_by=indexed_by)
        return self

    def stability_analysis(self, enabled: bool):
        self._stability = bool(enabled)
        return self

    def series_order(self, order: int):
        """Taylor order used in the action for CGFs with a removable
        singularity at 0 (the uniform jump mgf).  Default 8."""
        self._series_order = int(order)
        return self

    # ── dynamics ──────────────────────────────────────────────────────
    def equation(self, field: str, lhs: str, rhs: str = '0'):
        """``lhs = rhs + noise`` for ``field``; ``lhs`` holds the linear
        differential operator (contains ``Dt``), ``rhs`` no noise."""
        if field not in self._fields:
            raise SDEError(f'equation({field!r}): declare the field first with .field().')
        if field in self._eqs:
            raise SDEError(f'equation({field!r}): the field already has an equation.')
        if not re.search(r'\bDt\b', lhs):
            raise SDEError(f'equation({field!r}): lhs={lhs!r} must contain the time '
                           f'derivative Dt (a dynamical equation).')
        if re.search(r'\bDt\b', rhs):
            raise SDEError(f'equation({field!r}): Dt found in rhs; it belongs on the lhs.')
        if re.search(r'\bt\b', rhs) or re.search(r'\bt\b', lhs):
            raise SDEError(f'equation({field!r}): explicit time t is not supported '
                           f'(time-independent dynamics only).')
        self._eqs[field] = {'lhs': lhs, 'rhs': rhs}
        return self

    def drift(self, field: str, f: str):
        """Shorthand for ``equation(field, lhs='Dt*field', rhs=f)``."""
        return self.equation(field, lhs=f'Dt*{field}', rhs=f)

    def noise(self, name: str, source: NoiseSource, couples: Optional[dict] = None,
              *, expose: Optional[str] = None, domain: Optional[str] = None,
              population: Optional[str] = None):
        """Declare a noise source ``name`` feeding ``couples={field: c}``
        (``dx_field += c * d eta``).  ``expose='n'`` exposes it as the field
        ``n`` (use ``n`` in the equations' rhs)."""
        if not isinstance(source, NoiseSource):
            raise SDEError(
                f'noise({name!r}): source must be a noise source from api.noise '
                f'(Gaussian, Poisson, CompoundPoisson, GammaProcess, '
                f'InverseGaussianProcess, Cumulants, CGF); got {type(source).__name__}. '
                f'Per-step distributions are not processes.')
        if name in self._sources:
            raise SDEError(f'noise({name!r}): a source with this name exists.')
        couples = dict(couples or {})
        if not couples and expose is None:
            raise SDEError(f'noise({name!r}): give couples={{field: coefficient}} '
                           f'or expose=<field name>.')
        if expose is not None:
            self._check_new_name(expose)
            self._fields[expose] = {'population': population, 'description':
                                    f'exposed noise source {name}', 'domain': domain,
                                    'latex': '', 'exposed': name}
        self._sources[name] = {'source': source, 'couples': couples,
                               'expose': expose, 'population': population}
        return self

    def interpretation(self, mode: str):
        mode = str(mode).lower()
        if mode not in ('ito', 'stratonovich'):
            raise SDEError(f"interpretation({mode!r}): use 'ito' or 'stratonovich'.")
        self._interp = mode
        return self

    def extra_action(self, text: str):
        """Append ``text`` to the derived action (written with its own
        ``sum(... for i in pop)`` wrapper if it is indexed)."""
        self._extra.append(text)
        return self

    def set_action_text(self, text: str):
        raise SDEError(
            'SDE derives the full action from the declared equations and noise '
            'sources; append terms with extra_action(...), or replace the action '
            'on the builder returned by to_builder().')

    # ── helpers ───────────────────────────────────────────────────────
    def _check_new_name(self, name):
        if not re.fullmatch(r'[A-Za-z]\w*', name) or '__' in name:
            raise SDEError(f'{name!r} is not a valid field name.')
        if name in self._fields:
            raise SDEError(f'field {name!r} is declared twice.')
        if name in ('Dt', 't', 'theta', 'i', 'j'):
            raise SDEError(f'{name!r} is a reserved name.')

    def _populations(self) -> list[dict]:
        if self._pops:
            for p in self._pops:
                if p['name'] == 'pop' and p['size'] > 1:
                    raise SDEError(
                        "population('pop', size>1): the model compiler binds the name "
                        "'pop' to the legacy population count; name the population "
                        "differently (or use SDE(name, n_populations=N)).")
            return self._pops
        return [{'name': self._implicit_pop(), 'size': self._n_pop_default,
                 'description': ''}]

    def _implicit_pop(self) -> str:
        # The compiler binds ``pop`` to range(number of populations), which is
        # range(1) once a population is declared, so an implicit population of
        # size N > 1 gets its own name and ``pop`` in the texts is renamed.
        return 'pop' if self._n_pop_default == 1 else 'pop_all'

    def _rename_pop(self, text: str) -> str:
        if self._pops or self._implicit_pop() == 'pop':
            return text
        return re.sub(r'\bpop\b', self._implicit_pop(), text)

    def _pop_of_field(self, f: str) -> str:
        p = self._fields[f]['population']
        pops = self._populations()
        if p is None:
            if len(pops) > 1:
                raise SDEError(f'field {f!r}: give population= (several populations '
                               f'are declared).')
            return pops[0]['name']
        if p not in [q['name'] for q in pops]:
            raise SDEError(f'field {f!r}: unknown population {p!r}.')
        return p

    def _pop_of_source(self, name: str) -> str:
        s = self._sources[name]
        pops = set()
        if s['expose']:
            pops.add(self._pop_of_field(s['expose']))
        for f in s['couples']:
            if f not in self._fields:
                raise SDEError(f'noise({name!r}): couples to undeclared field {f!r}.')
            pops.add(self._pop_of_field(f))
        if s['population'] is not None:
            pops.add(s['population'])
        if len(pops) != 1:
            raise SDEError(f'noise({name!r}): a source must couple to fields of ONE '
                           f'population (got {sorted(pops)}).')
        return pops.pop()

    def _allowed_names(self) -> set:
        names = set(self._fields) | {f + 't' for f in self._fields}
        names |= set(self._param_names) | set(self._function_names) | set(self._kernels)
        names |= {p['name'] for p in self._populations()} | {'pop'}
        names |= {'i', 'j', 'k', 'l'} | _KEYWORDS
        return names

    def _check_text(self, text: str, what: str):
        toks = set(re.findall(r'\b([A-Za-z_]\w*)\b', text))
        unknown = sorted(toks - self._allowed_names())
        if unknown:
            raise SDEError(f'{what}: undeclared symbol(s) {unknown}; every symbol must '
                           f'be a declared field, parameter, function, kernel or index.')

    def _index(self, text) -> str:
        """Bare field names -> ``x[i]``; bare function calls -> ``phi[i](``."""
        text = expr_to_text(text) if not isinstance(text, str) else text
        text = self._rename_pop(text)
        for f in self._fields:
            text = re.sub(rf'\b{re.escape(f)}\b(?!\s*[\[\(])', f'{f}[i]', text)
        for fn in self._function_names:
            text = re.sub(rf'\b{re.escape(fn)}\s*\(', f'{fn}[i](', text)
        return text

    def _parse(self, text, what):
        return parse_expr(self._index(text), what=what)

    def _sym(self, f: str):
        return SR.var(f'{f}__i')

    def _kernel_dc(self, name: str):
        """The kernel's integral (its frequency image at omega = 0)."""
        k = self._kernels[name]
        if k['freq_image'] is not None:
            return parse_expr(k['freq_image'], what=f'kernel {name}').subs(
                {SR.var('omega'): 0})
        e = parse_expr(k['time_expr'], what=f'kernel {name}')
        t = SR.var('t')
        dd = SR.var('_dirac_')
        e = e.subs({function('dirac_delta')(t): dd})
        from sage.all import dirac_delta
        e = e.subs({dirac_delta(t): dd})
        delta_part = e.coefficient(dd, 1)
        rest = (e - delta_part * dd).simplify_full()
        assumed = [v > 0 for v in rest.variables() if str(v) != 't']
        for a in assumed:
            assume(a)
        # A retarded kernel carries heaviside(t): Sage/Maxima does not always evaluate the integral over (-oo, oo)
        # with it (the result can be an unevaluated ``cases``), so integrate its t > 0 part over (0, oo) instead.
        from sage.all import heaviside
        lower = -oo
        if rest.has(heaviside(t)):
            rest = rest.subs({heaviside(t): SR(1)})
            lower = SR(0)
        try:
            val = integrate(rest, t, lower, oo) if not rest.is_zero() else SR(0)
        finally:
            for a in assumed:
                forget(a)
        if 'integrate' in str(val):
            raise SDEError(f'kernel {name}: cannot integrate time_expr symbolically '
                           f'for the mean-field equations; give freq_image=.')
        return (val + delta_part).simplify_full()

    def _collapse_kernels(self, text: str) -> str:
        """Replace kernel symbols by their integral (stationary mean field)."""
        if not self._kernels:
            return text
        # Conv(K, X) -> (K)*(X)
        while True:
            m = re.search(r'\bConv\s*\(', text)
            if not m:
                break
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1}.get(text[k], 0)
                k += 1
            args = _split_top_level(text[m.end():k - 1])
            if len(args) != 2:
                raise SDEError(f'cannot read Conv(...) in {text!r}.')
            text = text[:m.start()] + f'({args[0].strip()})*({args[1].strip()})' + text[k:]
        for name in self._kernels:
            dc = self._kernel_dc(name)

            def rep(m, _dc=dc):
                val = expr_to_text(_dc)
                if m.group(1):
                    idx = [s.strip() for s in m.group(1).split(',')]
                    for old, new in zip(('i', 'j'), idx):
                        val = re.sub(rf'\[([^\]]*)\b{old}\b', lambda q: q.group(0)[:-1] + '\0' + new, val)
                    val = val.replace('\0', '')
                return f'({val})'
            text = re.sub(rf'\b{re.escape(name)}\b(?:\s*\[([^\]]*)\])?', rep, text)
        return text

    # ── derivation ────────────────────────────────────────────────────
    def _derive(self) -> dict:
        """Derive the action terms and the mean-field equations."""
        self.notes = []
        for f, info in self._fields.items():
            if info['exposed'] is None and f not in self._eqs:
                raise SDEError(f'field {f!r} has no equation; call .equation() or .drift().')
        allowed_src = (set(self._fields) | set(self._param_names)
                       | set(self._function_names) | {'i', 'j'})
        for name, s in self._sources.items():
            s['source'].validate(allowed_src)
            for f, c in s['couples'].items():
                if f not in self._fields:
                    raise SDEError(f'noise({name!r}): couples to undeclared field {f!r}.')
                self._check_text(str(c), f'noise({name!r}) coefficient for {f!r}')
                if 't' in expr_names(parse_expr(self._index(str(c)))):
                    raise SDEError(f'noise({name!r}): coefficient depends on t.')
        for f, eq in self._eqs.items():
            self._check_text(eq['lhs'], f'equation({f!r}) lhs')
            self._check_text(eq['rhs'], f'equation({f!r}) rhs')
        if self._interp == 'stratonovich':
            for name, s in self._sources.items():
                if not s['source'].is_gaussian:
                    raise SDEError(
                        f"interpretation('stratonovich'): source {name!r} is a jump/"
                        f"non-Gaussian source; the Stratonovich rule is defined here "
                        f"for Gaussian noise only (jump SDEs need the Marcus rule).")
                if s['expose']:
                    raise SDEError(
                        f"interpretation('stratonovich'): exposed source {name!r} is "
                        f"not supported; couple it with couples= instead.")

        extra_rhs = {f: [] for f in self._eqs}       # deterministic additions (action + MF)
        mf_rhs = {f: [] for f in self._eqs}          # MF-only additions (non-Gaussian means)
        source_terms = {}                            # per population: action text terms
        exposed_eqs = []                             # (field, rhs text)

        for name, s in self._sources.items():
            src = s['source']
            pop = self._pop_of_source(name)
            coup = {f: self._parse(str(c), f'coefficient of {name} in {f}')
                    for f, c in s['couples'].items()}
            for f in coup:
                if self._fields[f]['exposed'] is not None:
                    raise SDEError(f'noise({name!r}): cannot couple to the exposed '
                                   f'source field {f!r}.')
            K_full = parse_expr(self._index(expr_to_text(
                src.action_cgf(self._series_order))), what=f'K of {name}')
            mean = parse_expr(self._index(expr_to_text(src.mean())), what=f'mean of {name}')
            if s['expose']:
                n = s['expose']
                nt = SR.var(f'{n}t__i')
                term = f'{n}t[i]*{n}[i] - {_wrap(expr_to_text(K_full.subs({THETA: nt})))}'
                source_terms.setdefault(pop, []).append(term)
                exposed_eqs.append((n, expr_to_text(mean)))
                for f, c in coup.items():
                    extra_rhs[f].append(c * SR.var(f'{n}__i'))
                continue
            if src.is_gaussian:
                gm = src.params['mean']
                if not SR(gm).is_zero():
                    gm_i = parse_expr(self._index(expr_to_text(gm)))
                    for f, c in coup.items():
                        extra_rhs[f].append(c * gm_i)
                    self.notes.append(f'Gaussian source {name!r}: mean '
                                      f'{expr_to_text(gm)} moved into the drift of '
                                      f'{", ".join(coup)}.')
                    # The action keeps only the variance part of K.
                    K_full = parse_expr(self._index(expr_to_text(src.params['var'])),
                                        what=f'{name}.var') * THETA ** 2 / 2
            else:
                for f, c in coup.items():
                    if not SR(mean).is_zero():
                        mf_rhs[f].append(c * mean)
            theta_val = sum(c * SR.var(f'{f}t__i') for f, c in coup.items())
            K_term = K_full.subs({THETA: theta_val})
            source_terms.setdefault(pop, []).append(f'-{_wrap(expr_to_text(K_term))}')

        # Stratonovich -> Ito drift correction (Gaussian sources only).
        if self._interp == 'stratonovich':
            dtcoef = {}
            for f, eq in self._eqs.items():
                L = self._parse(eq['lhs'], f'lhs of {f}').expand()
                a = L.coefficient(_DT, 1).coefficient(self._sym(f), 1)
                if a.is_zero() or set(map(str, a.variables())) & {g + '__i' for g in self._fields}:
                    raise SDEError(f"stratonovich: cannot read a constant Dt "
                                   f"coefficient off lhs={eq['lhs']!r}.")
                dtcoef[f] = a
            for name, s in self._sources.items():
                V = self._parse(expr_to_text(s['source'].params['var']), f'{name}.var')
                b = {f: self._parse(str(c), 'coefficient') / dtcoef[f]
                     for f, c in s['couples'].items()}
                for fi in b:
                    corr = SR(0)
                    for fj in b:
                        xj = self._sym(fj)
                        corr += (V * b[fj] * b[fi].diff(xj)
                                 + b[fj] * b[fi] * V.diff(xj) / 2) / 2
                    corr = corr.simplify_full()
                    if not corr.is_zero():
                        extra_rhs[fi].append(dtcoef[fi] * corr)
                        self.notes.append(
                            f'Stratonovich -> Ito: drift of {fi!r} += '
                            f'{expr_to_text(dtcoef[fi] * corr)} (source {name!r}).')

        # Per-field action terms and mean-field equations.
        field_terms, mf_eqs = {}, []
        for f, eq in self._eqs.items():
            pop = self._pop_of_field(f)
            lhs_txt, rhs_txt = self._index(eq['lhs']), self._index(eq['rhs'])
            add = sum(extra_rhs[f], SR(0))
            rhs_act = rhs_txt if add.is_zero() else f'{rhs_txt} + {_wrap(expr_to_text(add))}'
            field_terms.setdefault(pop, []).append(
                f'{f}t[i]*({lhs_txt} - {_wrap(rhs_act)})')
            add_mf = sum(mf_rhs[f], SR(0))
            rhs_mf = rhs_act if add_mf.is_zero() else f'{rhs_act} + {_wrap(expr_to_text(add_mf))}'
            lhs_mf, rhs_mf = self._normalise_lhs(f, lhs_txt, rhs_mf)
            mf_eqs.append({'field': f, 'lhs': lhs_mf,
                           'rhs': self._collapse_kernels(rhs_mf), 'population': pop})
        for n, rhs in exposed_eqs:
            mf_eqs.append({'field': n, 'lhs': f'{n}[i]',
                           'rhs': self._collapse_kernels(rhs),
                           'population': self._pop_of_field(n)})
        # Order the MF equations by field declaration order.
        order = list(self._fields)
        mf_eqs.sort(key=lambda e: order.index(e['field']))

        parts = []
        for p in self._populations():
            terms = field_terms.get(p['name'], []) + source_terms.get(p['name'], [])
            if not terms:
                continue
            body = '\n    + '.join(terms).replace('+ -', '- ')
            parts.append(f"sum(\n    {body}\n    for i in {p['name']})")
        action = '\n+ '.join(parts)
        for e in self._extra:
            action += f'\n+ ({self._rename_pop(" ".join(e.split()))})'
        return {'action': action, 'equations': mf_eqs}

    def _normalise_lhs(self, f, lhs, rhs):
        """If ``lhs`` vanishes at Dt = 0 (``Dt*x``), move the linear restoring
        term of ``rhs`` (field-independent coefficient) onto the lhs so the
        mean-field equation defines ``x*``; the residual is unchanged."""
        L = parse_expr(lhs, what=f'lhs of {f}')
        if not L.subs({_DT: 0}).simplify_full().is_zero():
            return lhs, rhs
        try:
            R = parse_expr(rhs, what=f'rhs of {f}')
        except NoiseError as exc:
            raise SDEError(
                f'equation({f!r}): lhs={lhs!r} vanishes at Dt=0 and rhs cannot be '
                f'split symbolically ({exc}); write the restoring term on the lhs, '
                f"e.g. lhs='(Dt + mu)*{f}'.") from exc
        x = self._sym(f)
        Re = R.expand()
        c1 = Re.coefficient(x, 1)
        fieldsyms = {g + '__i' for g in self._fields}
        if c1.is_zero() or set(map(str, c1.variables())) & fieldsyms:
            raise SDEError(
                f'equation({f!r}): lhs={lhs!r} vanishes at Dt=0 and rhs has no linear '
                f'term in {f} with a field-independent coefficient, so the mean-field '
                f'equation cannot define {f}*; write the equation with a restoring '
                f'term on the lhs.')
        new_lhs = expr_to_text((L - c1 * x).expand())
        new_rhs = expr_to_text((Re - c1 * x).expand())
        self.notes.append(f'mean-field equation of {f!r}: linear term '
                          f'{expr_to_text(c1 * x)} moved to the lhs.')
        return new_lhs, (new_rhs if new_rhs.strip() else '0')

    # ── outputs ───────────────────────────────────────────────────────
    def action_text(self) -> str:
        return self._derive()['action']

    def equations(self) -> list[dict]:
        """The derived ``.equation(lhs=..., rhs=..., population=...)`` calls."""
        return [{k: e[k] for k in ('lhs', 'rhs', 'population')}
                for e in self._derive()['equations']]

    def to_builder(self):
        """A :class:`TemporalModelBuilder` with the derived action and
        mean-field equations set (continue by hand from here)."""
        from api.model import TemporalModelBuilder
        d = self._derive()
        b = TemporalModelBuilder(self.name)
        pops = self._populations()
        for p in pops:
            b.population(p['name'], size=p['size'], description=p['description'])
        single = pops[0]['name'] if len(pops) == 1 else None
        for f, info in self._fields.items():
            kw = dict(population=self._pop_of_field(f), description=info['description'])
            if info['domain'] is not None:
                kw['domain'] = info['domain']
            if info['latex']:
                kw['latex'] = info['latex']
            b.physical_field(f, **kw)
        for (name,), kw in self._params:
            kw = dict(kw)
            if kw['indexed_by'] is None and kw['indexed'] and single:
                kw['indexed_by'] = ([single, single] if kw['indexed'] == 'matrix'
                                    else [single])
            b.parameter(name, **{k: v for k, v in kw.items() if k != 'indexed'
                                 or kw['indexed_by'] is None})
        for (name, args, expr), kw in self._functions:
            kw = dict(kw)
            if kw['population'] is None and single:
                kw['population'] = single
            b.define_function(name, args=args, expression=expr,
                              **{k: v for k, v in kw.items() if v is not None})
        for name, k in self._kernels.items():
            kw = {kk: v for kk, v in k.items() if v not in (None, '', False)}
            b.define_kernel(name, **kw)
        b.set_action_text(d['action'])
        for e in d['equations']:
            b.equation(lhs=e['lhs'], rhs=e['rhs'], population=e['population'])
        if self._stability is not None:
            b.stability_analysis(self._stability)
        return b

    def build(self) -> dict:
        return self.to_builder().build()

    def show(self) -> str:
        """Print (and return) the derived action and mean-field equations as
        TemporalModelBuilder calls, copyable into a hand-written model."""
        d = self._derive()
        lines = [f"# SDE {self.name!r} ({self._interp})",
                 ".set_action_text('''", d['action'], "''')"]
        for e in d['equations']:
            lines.append(f".equation(lhs={e['lhs']!r}, rhs={e['rhs']!r}, "
                         f"population={e['population']!r})")
        for n in self.notes:
            lines.append(f'# note: {n}')
        out = '\n'.join(lines)
        print(out)
        return out

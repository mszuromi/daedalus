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
(``x[i]``); bare field names, bare parameters indexed by one population
and bare calls of declared functions are indexed by the population index
``i`` automatically.  Population sums
(``sum(w[i,j]*g*n[j] for j in E)``) are written as in the tracked
models.

Conventions: Ito by default (``.interpretation('stratonovich')``
converts Gaussian noise to Ito with the drift correction
``f_i += 1/2 sum_a sum_j [V_a b_ja d_j b_ia + 1/2 b_ja b_ia d_j V_a]``,
``b_ia = c_ia / a_i`` with ``a_i`` the ``Dt`` coefficient of ``lhs_i``
and ``V_a`` the variance rate; it raises for jump and exposed sources).
Time-independent sources only.

``.interpretation('marcus')`` extends Stratonovich to jump sources: a jump
``z`` of a source coupled through the vector field ``G = sum_i b_i(x) d/dx_i``
moves the state along the flow of ``G`` for time ``z``,
``x -> exp(z G) x``.  The action stays Ito-ordered (it is the generator's):
the noise term of a source with cumulant rates ``kappa_k`` is
``sum_k kappa_k/k! D_k``, ``D_k = e^{-xt.x} G^k e^{xt.x}``, in place of
``K(sum_i c_i xt_i)``, so enumeration and Phase J are unchanged.  Gaussian
sources (and Gaussian parts of a sum) take the Stratonovich conversion above
(a Gaussian-only model gives exactly the Stratonovich action); a jump source
with a constant coupling is unchanged (the Ito action).  A jump source with an
affine coupling whose flow is closed form (each coupling affine in its own
field, ``x -> e^{bz} x + (a/b)(e^{bz} - 1)``) gets its noise term exactly,
from the jump law's mgf, kept to ``SDE.series_order`` response legs:
``lam E[(e^{bz} - 1)^n]/n!`` per ``(xt (x + a/b))^n``, with no moment series in
``z`` (that series diverges for exponential-tailed jumps at degree
``n > 1/(|b| scale)``).  Other state-dependent couplings carry the series to
``SDE.series_order`` (``marcus_route('series')`` forces it).
``compute_cumulants`` reports, over the vertices ``(k, max_ell)`` can use,
the size of the next series term, or the rounding of the exact coefficients,
at the saddle (``result['marcus_truncation']``, G6), and refuses an exact
route whose jump law has no mgf where a usable vertex needs it.  A
state-dependent rate stays outside
the flow (the pre-jump rate).  Refused under Marcus: exposed sources, a
point process whose spikes enter with a state-dependent coefficient,
couplings that read other units of a population (joint jumps), and
state-dependent ``Cumulants``/``CGF`` sources.

Mean-field equation: ``lhs|_{Dt=0} = rhs`` must define ``x*``, so a lhs such
as ``Dt*x`` is completed.  The linear term of ``rhs`` (field-independent
coefficient) moves onto the lhs; where there is none (``-x/(1 + eps*x^2)``,
``-x*exp(-eps*x^2)``, a linear coefficient that holds a field) ``c*x`` with
``c = 1`` is added to both sides.  The residual, the saddle and the linear
stability do not depend on ``c`` and the action is untouched; ``SDE.notes``
records the step.  Writing ``lhs='(Dt + c)*x'``, ``rhs='drift + c*x'`` by hand
gives the same model.
"""
from __future__ import annotations

import keyword
import re
from collections import OrderedDict
from typing import Optional

from sage.all import SR, integrate, oo, assume, assumptions, forget, function

from api.noise import (NoiseError, NoiseSource, THETA, parse_expr,
                       expr_to_text, expr_names, _MATH_FUNCS, _SAGE_PRINTED_CALLS)

__all__ = ['SDE', 'SDEError']


class SDEError(NoiseError):
    """Raised for an invalid SDE declaration."""


_KEYWORDS = {'sum', 'for', 'in', 'Dt', 'Conv', 'I', 'pi', 'e',
             'if', 'else', 'range'} | set(_MATH_FUNCS)
_DT = SR.var('Dt')
_NO_DECLARE = _KEYWORDS | {'theta', 't'}
_NO_POPULATION = {'Dt', 'Conv', 'theta', 't'}


def _check_declarable(name, kind):
    """Refuse a declared name that is a built-in constant, keyword or math
    function.  ``e``, ``pi`` and ``I`` are the Sage constants inside noise
    parameters, so a declared ``e`` was read as Euler's number there but as
    the parameter elsewhere (a silently wrong action and saddle)."""
    bad = (name in _NO_POPULATION or keyword.iskeyword(name)) if kind == 'population' \
        else (name in _NO_DECLARE or keyword.iskeyword(name))
    if bad:
        raise SDEError(
            f'{kind} name {name!r} is reserved (built-in constant, keyword or '
            f'function); choose another name (e.g. {name}0).')


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


def _compositions(n: int, parts: int):
    """The tuples of ``parts`` non-negative integers that sum to ``n``."""
    if parts == 1:
        yield (n,)
        return
    for first in range(n, -1, -1):
        for rest in _compositions(n - first, parts - 1):
            yield (first,) + rest


def _is_rational_in(e, t) -> bool:
    num, den = SR(e).numerator_denominator()
    return num.is_polynomial(t) and den.is_polynomial(t)


def _cgf_shapes(K):
    """``K(theta)`` as ``sum coefficient * shape(theta)`` for the exact Marcus
    route: ``([(kind, key, coefficient)], bases)``.  ``kind`` is ``'exp'``
    (``key = alpha``: ``e^{alpha theta}``), ``'poly'`` (``key = p``:
    ``theta^p``) or ``'generic'`` (``key`` the shape); constants are dropped
    (every difference kills them).  ``bases`` are the subexpressions of the
    generic shapes at which the closed form stops being the mgf (bases of
    negative or non-integer powers, log arguments): the mgf exists at ``s``
    while none of them changes sign or vanishes on ``[0, s]``."""
    import operator
    from sage.all import exp as _exp
    from sage.symbolic.operators import add_vararg, mul_vararg
    K0, K = SR(K), SR(K).expand()
    terms = K.operands() if K.operator() is add_vararg else [K]

    def shape(g):
        op = g.operator()
        if g is THETA or (op is None and g.has(THETA)):
            return 'poly', 1, SR(1)
        if op is operator.pow:
            base, ex = g.operands()
            if base.operator() is not None and str(base.operator()) == 'exp' and not ex.has(THETA):
                return shape(_exp(base.operands()[0] * ex))
            if (base - THETA).is_zero() and ex.is_integer() and bool(ex > 0):
                return 'poly', int(ex), SR(1)
        if op is not None and str(op) == 'exp':
            arg = g.operands()[0]
            if arg.is_polynomial(THETA) and arg.degree(THETA) == 1:
                return 'exp', arg.coefficient(THETA, 1), _exp(arg.subs({THETA: 0}))
        return 'generic', g, SR(1)

    groups = OrderedDict()
    for t in terms:
        if not t.has(THETA):
            continue
        facs = t.operands() if t.operator() is mul_vararg else [t]
        cf, g = SR(1), SR(1)
        for q in facs:
            if q.has(THETA):
                g *= q
            else:
                cf *= q
        kind, key, extra = shape(g)
        slot = groups.setdefault((kind, str(key)), [kind, key, SR(0)])
        slot[2] += cf * extra
    shapes = [(kind, key, cf) for kind, key, cf in groups.values() if not SR(cf).is_zero()]

    bases = []

    def walk(e):
        op = e.operator()
        if op is None:
            return
        ops = e.operands()
        if op is operator.pow:
            base, ex = ops
            if base.has(THETA) and not (ex.is_integer() and bool(ex >= 0)):
                if not any((base - b).is_zero() for b in bases):
                    bases.append(base)
        elif str(op) == 'log' and ops[0].has(THETA):
            if not any((ops[0] - b).is_zero() for b in bases):
                bases.append(ops[0])
        for q in ops:
            walk(q)
    if any(kind == 'generic' for kind, _, _ in shapes):
        # the unexpanded K keeps (1 - c theta)^(-2) as such: expanded, its base
        # is a square that touches 0 at the pole without changing sign
        walk(K0)
    for kind, key, _ in shapes:
        if kind == 'generic':
            walk(key)
    return shapes, bases


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
        self._marcus_route = 'auto'
        self._marcus_terms: 'OrderedDict[str, dict]' = OrderedDict()
        self._ito_drift: dict = {}
        self.notes: list[str] = []

    # ── declarations (passed through to TemporalModelBuilder) ─────────
    def population(self, name: str, *, size: int = 1, description: str = ''):
        _check_declarable(name, 'population')
        self._pops.append({'name': name, 'size': max(int(size), 1),
                           'description': description})
        return self

    def field(self, name: str, *, population: Optional[str] = None,
              description: str = '', domain: Optional[str] = None,
              latex: str = ''):
        """Declare a physical field (its response field is ``<name>t``)."""
        self._check_new_name(name)
        _check_declarable(name, 'field')
        self._fields[name] = {'population': population, 'description': description,
                              'domain': domain, 'latex': latex, 'exposed': None}
        return self

    def parameter(self, name: str, default=None, indexed=False,
                  domain: Optional[str] = None, description: str = '',
                  indexed_by: Optional[list] = None):
        _check_declarable(name, 'parameter')
        if name in self._param_names:
            raise SDEError(f'parameter {name!r} is declared twice.')
        self._params.append(((name,), dict(default=default, indexed=indexed,
                                           domain=domain, description=description,
                                           indexed_by=indexed_by)))
        self._param_names.append(name)
        return self

    def define_function(self, name: str, args: list, expression: str,
                        latex: Optional[str] = None, description: str = '',
                        population: Optional[str] = None):
        _check_declarable(name, 'function')
        if name in _SAGE_PRINTED_CALLS:
            # A call of it means Sage's own function in noise expressions and the saddle solve.
            raise SDEError(f'function name {name!r} is reserved (Sage\'s {name}); choose another '
                           f'name (e.g. {name}0).')
        self._functions.append(((name, list(args), expression),
                                dict(latex=latex, description=description,
                                     population=population)))
        self._function_names.append(name)
        return self

    def define_kernel(self, name: str, *, time_expr: Optional[str] = None,
                      freq_image: Optional[str] = None, latex_name: str = '',
                      indexed=False, indexed_by: Optional[list] = None,
                      delay: Optional[str] = None, kernel=None):
        """A convolution kernel ``Conv(name, x)`` (see
        :meth:`api.model.TemporalModelBuilder.define_kernel`): ``time_expr`` or
        ``freq_image``, optionally delayed by ``delay`` (``g(t - d)``; the hard
        delay ``delta(t - d)`` alone), or ``kernel=Kernel.delay('d') *
        Kernel.exponential('tau')`` (:class:`api.delay.Kernel`)."""
        _check_declarable(name, 'kernel')
        if kernel is not None:
            if time_expr is not None or freq_image is not None or delay is not None:
                raise SDEError(f'define_kernel({name!r}): give kernel= alone.')
            freq_image, delay = kernel.freq_image, kernel.delay
        if time_expr is not None and delay is None:
            from api.delay import split_shifted_time_expr
            sh = split_shifted_time_expr(str(time_expr))
            if sh is not None:
                delay, time_expr = sh
        if delay is not None and time_expr is None and freq_image is None:
            freq_image = '1'
        if time_expr is None and freq_image is None:
            raise SDEError(f'define_kernel({name!r}): give time_expr=, freq_image= '
                           f'or delay=.')
        self._kernels[name] = dict(time_expr=time_expr, freq_image=freq_image,
                                   latex_name=latex_name, indexed=indexed,
                                   indexed_by=indexed_by,
                                   delay=None if delay is None else str(delay))
        return self

    def stability_analysis(self, enabled: bool):
        self._stability = bool(enabled)
        return self

    def series_order(self, order: int):
        """Taylor order used in the action for CGFs with a removable
        singularity at 0 (the uniform jump mgf), and the order of the Marcus
        series of a jump source with a state-dependent coupling
        (``interpretation('marcus')``; for an affine coupling, the number of
        response legs its exact noise term is kept to).  Default 8."""
        self._series_order = int(order)
        return self

    def marcus_route(self, route: str):
        """How ``interpretation('marcus')`` builds the noise term of a jump
        source with a state-dependent coupling.  ``'auto'`` (default): exact
        for an affine coupling whose flow is closed form (each coupling affine
        in its own field), from the jump law's mgf, else the moment series of
        ``sum_k kappa_k/k! D_k``.  ``'series'``: the moment series always
        (it converges fast for a small slope times jump size, where the exact
        coefficients can lose digits to cancellation)."""
        route = str(route).lower()
        if route not in ('auto', 'series'):
            raise SDEError(f"marcus_route({route!r}): use 'auto' or 'series'.")
        self._marcus_route = route
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
        rhs = self._delayed_fields(rhs, field)
        rhs = self._compose_kernels(rhs, f'equation({field!r})')
        if re.search(r'\bt\b', rhs) or re.search(r'\bt\b', lhs):
            raise SDEError(f'equation({field!r}): explicit time t is not supported '
                           f'(time-independent dynamics only; a delayed field is '
                           f'written x(t - d)).')
        self._eqs[field] = {'lhs': lhs, 'rhs': rhs}
        return self

    def _delayed_fields(self, text: str, where: str) -> str:
        """``x(t - d)`` (``x`` a declared field, ``d`` a parameter expression)
        -> ``Conv(delay_<d>, x)`` with the hard-delay kernel ``delay_<d>``
        declared on the fly (:mod:`api.delay`)."""
        if not self._fields:
            return text
        names = '|'.join(re.escape(f) for f in sorted(self._fields, key=len, reverse=True))
        ind = re.search(rf'\b({names})\s*\[[^\]]*\]\s*\(\s*t\b', text)
        if ind:
            raise SDEError(f'equation({where!r}): an indexed delayed field '
                           f'{text[ind.start():ind.end()]}...) is not supported; write '
                           f"Conv(K, {ind.group(1)}[...]) with define_kernel('K', delay='d').")
        pat = re.compile(rf'\b({names})\s*\(\s*t\s*-\s*')
        out, pos = '', 0
        while True:
            m = pat.search(text, pos)
            if not m:
                break
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1}.get(text[k], 0)
                k += 1
            if depth:
                raise SDEError(f'equation({where!r}): unbalanced parentheses in '
                               f'{text[m.start():]!r}.')
            dexpr = text[m.end():k - 1].strip()
            if not dexpr or re.search(r'\bt\b', dexpr):
                raise SDEError(f'equation({where!r}): cannot read the delay in '
                               f'{text[m.start():k]!r} (write x(t - d), d free of t).')
            self._check_text(dexpr, f'equation({where!r}) delay {dexpr!r}')
            kname = 'delay_' + (re.sub(r'\W+', '_', dexpr).strip('_') or 'd')
            have = self._kernels.get(kname)
            if have is None:
                if kname in self._allowed_names():
                    raise SDEError(f'equation({where!r}): the delay kernel name '
                                   f'{kname!r} is taken; declare the kernel with '
                                   f"define_kernel(..., delay={dexpr!r}).")
                self.define_kernel(kname, delay=dexpr)
            elif have.get('delay') != dexpr or have.get('freq_image') != '1':
                raise SDEError(f'equation({where!r}): kernel {kname!r} exists and is '
                               f'not the delay {dexpr!r}.')
            out += text[pos:m.start()] + f'Conv({kname}, {m.group(1)})'
            pos = k
        return out + text[pos:]

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
            _check_declarable(expose, 'exposed field')
            self._fields[expose] = {'population': population, 'description':
                                    f'exposed noise source {name}', 'domain': domain,
                                    'latex': '', 'exposed': name}
        self._sources[name] = {'source': source, 'couples': couples,
                               'expose': expose, 'population': population}
        return self

    def point_process(self, name: str, rate: str, *, population: Optional[str] = None,
                      size='1', domain: Optional[str] = None, description: str = ''):
        """Declare the spike train (counting process) ``name`` with
        conditional intensity ``rate``: ``P(spike of name[i] in dt) =
        rate_i dt``.  The field ``name`` (and its response ``<name>t``) is
        declared, the action gains ``nt*n - (exp(size*nt) - 1)*rate`` and the
        mean field ``n* = size*rate*`` (kernels at omega = 0).  ``rate`` is an
        expression like a drift: fields, parameters, declared functions
        (``phi[i](v[i])``), population sums (``sum(w[i,j]*Conv(g[i,j], n[j])
        for j in E)``) and convolutions ``Conv(K, x)`` of ANY field, the spike
        train ``name`` itself included (a spike-only Hawkes process); a
        product of declared kernels inside ``Conv`` (``Conv(h[i]*g[i,j],
        n[j])``) is their convolution.  ``size`` is the jump of ``name`` at a
        spike (a deterministic mark, default 1).  A spike train enters a rate
        only through ``Conv`` (its instantaneous value is a delta)."""
        self._check_new_name(name)
        _check_declarable(name, 'point process')
        if name in self._sources:
            raise SDEError(f'point_process({name!r}): a source with this name exists.')
        rate = ' '.join(str(rate).split())
        if not rate:
            raise SDEError(f'point_process({name!r}): empty rate.')
        if re.search(r'\bDt\b', rate) or re.search(r'\bt\b', rate):
            raise SDEError(f'point_process({name!r}): the rate may not contain Dt or '
                           f'the time t (a delayed field is Conv(K, x) with '
                           f"define_kernel('K', delay='d')).")
        self._fields[name] = {'population': population, 'description':
                              description or f'point process {name}',
                              'domain': domain, 'latex': '', 'exposed': name}
        self._sources[name] = {'source': None, 'couples': {}, 'expose': name,
                               'population': population,
                               'pp': {'rate': rate, 'size': str(size).strip() or '1'}}
        return self

    def interpretation(self, mode: str):
        """``'ito'`` (default), ``'stratonovich'`` (Gaussian sources) or
        ``'marcus'`` (Gaussian and jump sources; see the module docstring)."""
        mode = str(mode).lower()
        if mode not in ('ito', 'stratonovich', 'marcus'):
            raise SDEError(f"interpretation({mode!r}): use 'ito', 'stratonovich' or "
                           f"'marcus'.")
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
        """Bare field names -> ``x[i]``; bare function calls -> ``phi[i](``;
        bare vector parameters -> ``p[i]``."""
        text = expr_to_text(text) if not isinstance(text, str) else text
        text = self._rename_pop(text)
        for f in self._fields:
            text = re.sub(rf'\b{re.escape(f)}\b(?!\s*[\[\(])', f'{f}[i]', text)
        for fn in self._function_names:
            text = re.sub(rf'\b{re.escape(fn)}\s*\(', f'{fn}[i](', text)
        for pn in self._vector_params():
            text = re.sub(rf'\b{re.escape(pn)}\b(?!\s*[\[\(])', f'{pn}[i]', text)
        return text

    def _vector_params(self):
        """Parameters indexed by one population (a bare name means ``p[i]``)."""
        out = []
        for (name,), kw in self._params:
            ib = kw.get('indexed_by')
            if (ib is not None and len(ib) == 1) or (ib is None and kw.get('indexed') in (True, 'vector')):
                out.append(name)
        return out

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
        delta_raw = delta_part
        # ``(c*dirac_delta(t) + h(t))*heaviside(t)``: the delta of a causal kernel
        # sits on the boundary and counts with its full weight c (the kernel's
        # frequency image is c + h~(omega), as for the same kernel written
        # ``c*dirac_delta(t) + h(t)*heaviside(t)``), so the step is not carried
        # into the delta part; a t-dependent factor is evaluated at the delta (t = 0).
        from sage.all import heaviside as _H
        delta_part = delta_part.subs({_H(t): SR(1)}).subs({t: SR(0)})
        # assume only what is not assumed already, and forget only that: a
        # parameter that a model builder declared positive keeps its global
        # assumption (forgetting it leaked into every later computation)
        # (a positivity already implied by a stronger held assumption, e.g.
        # taug > 1, is 'redundant' to Sage: skipped too; every assume sits in
        # the try, so a failure part-way still forgets what was added)
        held = {str(a) for a in assumptions()}
        assumed = []
        # A retarded kernel carries heaviside(t): Sage/Maxima does not always evaluate the integral over (-oo, oo)
        # with it (the result can be an unevaluated ``cases``), so integrate its t > 0 part over (0, oo) instead.
        from sage.all import heaviside
        lower = -oo
        causal = e - delta_raw * dd             # unsimplified: simplify_full mangles sums of exponentials
        if rest.has(heaviside(t)):
            rest = rest.subs({heaviside(t): SR(1)})
            lower = SR(0)
        val = None
        try:
            for v in rest.variables():
                if str(v) == 't' or str(v > 0) in held:
                    continue
                try:
                    assume(v > 0)
                except ValueError:                 # redundant
                    continue
                assumed.append(v > 0)
            try:
                val = integrate(rest, t, lower, oo) if not rest.is_zero() else SR(0)
            except (ValueError, TypeError):
                # Maxima cannot order the rates of e.g. a difference of exponentials
                # ('requested additional constraints'): the structural closed form below
                val = None
        finally:
            for a in assumed:
                forget(a)
        if val is None or 'integrate' in str(val):
            val = self._kernel_dc_causal(name, causal, t)
        return (val + delta_part).simplify_full()

    def _kernel_dc_causal(self, name, causal, t):
        """The integral of a causal sum of ``t^n exp(-a t)`` terms by the closed form
        ``n!/a^(n+1)`` (``model_compiler.causal_exponential_ft`` at omega = 0), the
        decay rates being positive by declaration (``domain='positive'``); no
        assumption is made on any symbol."""
        from api.model_compiler import causal_exponential_ft
        declared = {p[0][0] for p in self._params if p[1].get('domain') == 'positive'}
        positive = {str(v) for v in causal.variables()
                    if str(v).split('__')[0] in declared}
        try:
            return SR(causal_exponential_ft(causal, t, SR(0), positive=positive))
        except ValueError as exc:
            raise SDEError(f'kernel {name}: cannot integrate time_expr symbolically '
                           f'for the mean-field equations ({exc}); declare '
                           f"domain='positive' on its time constants, or give freq_image=.") from exc

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
                    mp = dict(zip(('i', 'j'), idx))
                    # one simultaneous pass inside each [...] (a sequential i->j, j->i
                    # rename would cancel a swap)
                    val = re.sub(r'\[([^\]]*)\]', lambda q: '[' + re.sub(
                        r'\b([ij])\b', lambda r: mp.get(r.group(1), r.group(1)), q.group(1)) + ']', val)
                return f'({val})'
            text = re.sub(rf'\b{re.escape(name)}\b(?:\s*\[([^\]]*)\])?', rep, text)
        return text

    def _check_pp(self, name, pp):
        what = f'point_process({name!r}) rate'
        rate = self._compose_kernels(pp['rate'], f'point_process({name!r})')
        self._check_text(rate, what)
        self._check_text(pp['size'], f'point_process({name!r}) size')
        spikes = {f for f, info in self._fields.items() if info['exposed'] is not None}
        if re.search(r'\b(' + '|'.join(map(re.escape, sorted(spikes))) + r')\b',
                     pp['size']):
            raise SDEError(f'point_process({name!r}): size may not depend on a spike train.')
        bare = set(re.findall(r'\b([A-Za-z_]\w*)\b', self._strip_conv(rate))) & spikes
        if bare:
            raise SDEError(
                f'{what}: the spike train(s) {sorted(bare)} appear outside Conv(...); a '
                f'spike train enters a rate only filtered, Conv(K, {sorted(bare)[0]}).')

    @staticmethod
    def _conv_spans(text):
        """``(start, open_paren_end, close_end)`` of each top-level ``Conv(``."""
        out, pos = [], 0
        while True:
            m = re.compile(r'\bConv\s*\(').search(text, pos)
            if not m:
                return out
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1}.get(text[k], 0)
                k += 1
            if depth:
                raise SDEError(f'unbalanced parentheses in {text!r}.')
            out.append((m.start(), m.end(), k))
            pos = k

    def _strip_conv(self, text):
        out, pos = '', 0
        for a, _, c in self._conv_spans(text):
            out += text[pos:a] + ' 0 '
            pos = c
        return out + text[pos:]

    def _compose_kernels(self, text: str, where: str) -> str:
        """``Conv(h[i]*g[i,j], x)`` (a product of declared kernels) ->
        ``Conv(h_g[i,j], x)`` with the convolution ``h_g`` (product of the
        frequency images, sum of the delays) declared on the fly."""
        out, pos = '', 0
        for a, b, c in self._conv_spans(text):
            args = _split_top_level(text[b:c - 1])
            if len(args) != 2:
                raise SDEError(f'{where}: cannot read {text[a:c]!r} (Conv(kernel, field)).')
            facs = [f.strip() for f in _split_top_level(args[0], '*')]
            if len(facs) == 1:
                out += text[pos:c]
                pos = c
                continue
            refs = []
            for f in facs:
                m = re.fullmatch(r'([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?', f)
                if not m or m.group(1) not in self._kernels:
                    raise SDEError(f'{where}: {args[0].strip()!r} in Conv(...) must be a '
                                   f'declared kernel or a product of declared kernels '
                                   f'({f!r} is not one).')
                idx = [q.strip() for q in m.group(2).split(',')] if m.group(2) else []
                refs.append((m.group(1), idx))
            sites = []
            for _, idx in refs:
                for q in idx:
                    if q not in sites:
                        sites.append(q)
            # the composite's own index order: the population index i first
            sites.sort(key=lambda q: {'i': 0, 'j': 1}.get(q, 2))
            if len(sites) > 2:
                raise SDEError(f'{where}: a kernel product carries at most two indices '
                               f'(got {sites}).')
            local = dict(zip(sites, ('i', 'j')))
            images, delays, pops = [], [], [None] * len(sites)
            for kname, idx in refs:
                k = self._kernels[kname]
                if k['freq_image'] is None:
                    raise SDEError(f'{where}: kernel {kname!r} has only a time_expr; a '
                                   f'kernel product needs frequency images (give '
                                   f'freq_image= or kernel=).')
                kidx = list(k.get('indexed_by') or [])
                if len(idx) != len(kidx):
                    raise SDEError(f'{where}: kernel {kname!r} takes {len(kidx)} '
                                   f'index(es), got {idx}.')
                mp = {old: local[new] for old, new in zip(('i', 'j'), idx)}
                for new, popn in zip(idx, kidx):
                    pops[sites.index(new)] = popn

                def remap(t, _mp=mp):
                    def sub(mm):
                        inner = re.sub(r'\b([ij])\b',
                                       lambda q: '\0' + _mp.get(q.group(1), q.group(1)),
                                       mm.group(1))
                        return '[' + inner + ']'
                    return re.sub(r'\[([^\]]*)\]', sub, t).replace('\0', '')
                if k['freq_image'] != '1':
                    images.append(f'({remap(k["freq_image"])})')
                if k['delay']:
                    delays.append(f'({remap(k["delay"])})')
            cname = '_'.join(r[0] for r in refs)
            spec = dict(time_expr=None, freq_image='*'.join(images) or '1', latex_name='',
                        indexed=False, indexed_by=pops if sites else None,
                        delay=' + '.join(delays) if delays else None)
            have = self._kernels.get(cname)
            if have is None:
                if cname in self._allowed_names():
                    raise SDEError(f'{where}: the composite kernel name {cname!r} is taken.')
                self._kernels[cname] = spec
            elif any(have.get(q) != spec[q] for q in ('freq_image', 'delay', 'indexed_by')):
                raise SDEError(f'{where}: kernel {cname!r} exists and is not the convolution '
                               f'{args[0].strip()!r} (index the factors the same way at '
                               f'every use, or declare the product with define_kernel).')
            ref = cname + (f'[{",".join(sites)}]' if sites else '')
            out += text[pos:a] + f'Conv({ref}, {args[1].strip()})'
            pos = c
        return out + text[pos:]

    # ── derivation ────────────────────────────────────────────────────
    def _derive(self) -> dict:
        """Derive the action terms and the mean-field equations."""
        self.notes = []
        for f, info in self._fields.items():
            if info['exposed'] is None and f not in self._eqs:
                raise SDEError(f'field {f!r} has no equation; call .equation() or .drift().')
        if self._interp == 'marcus':
            self._check_marcus_joint()
        allowed_src = (set(self._fields) | set(self._param_names)
                       | set(self._function_names) | {'i', 'j'})
        for name, s in self._sources.items():
            if s.get('pp'):
                self._check_pp(name, s['pp'])
                continue
            try:
                s['source'].validate(allowed_src)
            except NoiseError as exc:
                if s['expose'] and 'Conv' in str(exc):
                    raise SDEError(f'noise({name!r}): {exc} A spike train whose rate filters '
                                   f'fields (Conv) is declared with point_process({s["expose"]!r}, '
                                   f'rate=...).') from exc
                raise
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
                if s.get('pp') or not s['source'].is_gaussian:
                    raise SDEError(
                        f"interpretation('stratonovich'): source {name!r} is a jump/"
                        f"non-Gaussian source; the Stratonovich rule is defined here "
                        f"for Gaussian noise only; jump SDEs take the Marcus rule, "
                        f"interpretation('marcus').")
                if s['expose']:
                    raise SDEError(
                        f"interpretation('stratonovich'): exposed source {name!r} is "
                        f"not supported; couple it with couples= instead.")
        if self._interp == 'marcus':
            self._check_marcus()

        extra_rhs = {f: [] for f in self._eqs}       # deterministic additions (action + MF)
        mf_rhs = {f: [] for f in self._eqs}          # MF-only additions (non-Gaussian means)
        source_terms = {}                            # per population: action text terms
        exposed_eqs = []                             # (field, rhs text)
        gauss_units = []                             # (name, Gaussian source, couples): Stratonovich
        self._marcus_terms = OrderedDict()
        self._ito_drift = {}                         # field -> Stratonovich corrections (simulator)

        for name, s in self._sources.items():
            src = s['source']
            pop = self._pop_of_source(name)
            if s.get('pp'):
                n, size = s['expose'], s['pp']['size']
                rate = self._index(self._compose_kernels(s['pp']['rate'], f'point_process({n!r})'))
                jump = f'{n}t[i]' if size == '1' else f'{_wrap(size)}*{n}t[i]'
                source_terms.setdefault(pop, []).append(
                    f'{n}t[i]*{n}[i] - (exp({jump}) - 1)*{_wrap(rate)}')
                exposed_eqs.append((n, rate if size == '1' else f'{_wrap(size)}*{_wrap(rate)}'))
                continue
            coup = {f: self._parse(str(c), f'coefficient of {name} in {f}')
                    for f, c in s['couples'].items()}
            for f in coup:
                if self._fields[f]['exposed'] is not None:
                    raise SDEError(f'noise({name!r}): cannot couple to the exposed '
                                   f'source field {f!r}.')
            if self._interp == 'marcus' and not s['expose']:
                units = self._marcus_units(name, s, coup)
            else:
                units = [(name, src, False)]
            for uname, src, series in units:
                if series:
                    self._marcus_series(uname, src, coup, pop, source_terms, mf_rhs)
                    continue
                K_full = parse_expr(self._index(expr_to_text(
                    src.action_cgf(self._series_order))), what=f'K of {uname}')
                mean = parse_expr(self._index(expr_to_text(src.mean())), what=f'mean of {uname}')
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
                    gauss_units.append((uname, src, s['couples']))
                    if not SR(gm).is_zero():
                        gm_i = parse_expr(self._index(expr_to_text(gm)))
                        for f, c in coup.items():
                            extra_rhs[f].append(c * gm_i)
                        self.notes.append(f'Gaussian source {uname!r}: mean '
                                          f'{expr_to_text(gm)} moved into the drift of '
                                          f'{", ".join(coup)}.')
                        # The action keeps only the variance part of K.
                        K_full = parse_expr(self._index(expr_to_text(src.params['var'])),
                                            what=f'{uname}.var') * THETA ** 2 / 2
                else:
                    for f, c in coup.items():
                        if not SR(mean).is_zero():
                            mf_rhs[f].append(c * mean)
                theta_val = sum(c * SR.var(f'{f}t__i') for f, c in coup.items())
                K_term = K_full.subs({THETA: theta_val})
                source_terms.setdefault(pop, []).append(f'-{_wrap(expr_to_text(K_term))}')

        # Stratonovich -> Ito drift correction (Gaussian sources; under Marcus
        # the Gaussian sources and Gaussian parts, whose Marcus rule it is).
        if self._interp == 'stratonovich' or (self._interp == 'marcus' and gauss_units):
            dtcoef = self._dt_coefficients()
            for name, src, couples in gauss_units:
                V = self._parse(expr_to_text(src.params['var']), f'{name}.var')
                b = {f: self._parse(str(c), 'coefficient') / dtcoef[f]
                     for f, c in couples.items()}
                for fi in b:
                    corr = SR(0)
                    for fj in b:
                        xj = self._sym(fj)
                        corr += (V * b[fj] * b[fi].diff(xj)
                                 + b[fj] * b[fi] * V.diff(xj) / 2) / 2
                    corr = corr.simplify_full()
                    if not corr.is_zero():
                        extra_rhs[fi].append(dtcoef[fi] * corr)
                        self._ito_drift.setdefault(fi, []).append(dtcoef[fi] * corr)
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
        """If ``lhs`` vanishes at Dt = 0 (``Dt*x``), make the mean-field
        equation define ``x*``; the residual ``lhs - rhs`` is unchanged.

        Preferred: move the linear restoring term of ``rhs`` (field-independent
        coefficient) onto the lhs.  Fallback, where ``rhs`` has no such term
        (a rational or exponential drift, or a linear coefficient that contains
        a field): add ``c*x`` to both sides, ``lhs -> lhs + c*x`` and
        ``rhs -> rhs + c*x``, with ``c = 1``.  The mean-field solve is a root
        solve on ``lhs|_{Dt=0} - rhs`` in which ``c*x`` cancels, and linear
        stability reads ``d(lhs - rhs)/dx`` (also ``c``-free), so ``c`` only
        names the saddle text ``x* = (rhs + c*x*)/c``; 1 keeps it unscaled."""
        workaround = (f"write the restoring term on the lhs, e.g. lhs='(Dt + 1)*{f}' "
                      f"with rhs = <drift> + {f}")
        try:
            L = parse_expr(lhs, what=f'lhs of {f}')
        except NoiseError as exc:
            raise SDEError(
                f'equation({f!r}): lhs={lhs!r} cannot be read symbolically ({exc}); '
                f'if it vanishes at Dt=0, {workaround}.') from exc
        if not L.subs({_DT: 0}).simplify_full().is_zero():
            return lhs, rhs
        try:
            R = parse_expr(rhs, what=f'rhs of {f}')
        except NoiseError as exc:
            raise SDEError(
                f'equation({f!r}): lhs={lhs!r} vanishes at Dt=0 and rhs cannot be '
                f'split symbolically ({exc}); {workaround}.') from exc
        x = self._sym(f)
        Re = R.expand()
        c1 = Re.coefficient(x, 1)
        fieldsyms = {g + '__i' for g in self._fields}
        if not (c1.is_zero() or set(map(str, c1.variables())) & fieldsyms):
            new_lhs = expr_to_text((L - c1 * x).expand())
            new_rhs = expr_to_text((Re - c1 * x).expand())
            self.notes.append(f'mean-field equation of {f!r}: linear term '
                              f'{expr_to_text(c1 * x)} moved to the lhs.')
            return new_lhs, (new_rhs if new_rhs.strip() else '0')
        if str(x) not in set(map(str, R.variables())):
            raise SDEError(
                f'equation({f!r}): lhs={lhs!r} vanishes at Dt=0 and rhs does not '
                f'depend on {f}, so the mean-field equation cannot define {f}*; '
                f'{workaround}.')
        # No syntactic field-independent linear term: add and subtract c*x.
        cx = expr_to_text(x)
        new_lhs = expr_to_text((L + x).expand())
        new_rhs = f'{_wrap(rhs)} + {cx}'
        self.notes.append(
            f'mean-field equation of {f!r}: rhs has no field-independent linear '
            f'term in {f}; added {cx} to both sides (lhs {new_lhs!r}, c = 1; the '
            f'residual and the linear stability are unchanged).')
        return new_lhs, new_rhs

    # ── Marcus rule ───────────────────────────────────────────────────
    def _dt_coefficients(self) -> dict:
        """The constant ``Dt`` coefficient ``a`` of each equation's lhs
        (``a*Dt*x + ...``): a coupling ``c`` in lhs units is the vector field
        ``c/a`` of the field."""
        dtcoef = {}
        for f, eq in self._eqs.items():
            L = self._parse(eq['lhs'], f'lhs of {f}').expand()
            dt_terms = L.coefficient(_DT, 1)
            a = dt_terms.coefficient(self._sym(f), 1)
            # the x part of the Dt terms must be exactly a*x with a free of the fields: ``(1 + x^2)*Dt*x`` has
            # the Dt-coefficient x + x^3, whose x-part alone (1) would be taken for a
            if (a.is_zero() or set(map(str, a.variables())) & {g + '__i' for g in self._fields}
                    or (dt_terms - a * self._sym(f)).expand().has(self._sym(f))):
                raise SDEError(f"{self._interp}: cannot read a constant Dt "
                               f"coefficient off lhs={eq['lhs']!r}.")
            dtcoef[f] = a
        return dtcoef

    def _field_dependent(self, expr) -> bool:
        """True when ``expr`` (indexed form, ``x__i``) holds a field."""
        return any('__' in str(v) and str(v).split('__')[0] in self._fields
                   for v in SR(expr).variables())

    def _check_marcus(self):
        """Refuse what the Marcus rule is not defined for here: exposed
        sources, and point processes whose spikes enter an equation with a
        state-dependent coefficient (a constant one is the same jump in every
        interpretation)."""
        for name, s in self._sources.items():
            if s.get('pp'):
                self._check_marcus_pp(name, s['pp'])
            elif s['expose']:
                raise SDEError(
                    f"interpretation('marcus'): exposed source {name!r} is not "
                    f"supported (it enters the equations as a field, so no jump "
                    f"flow is defined); couple it with couples= instead.")

    def _check_marcus_joint(self):
        """A coupling that reads other units of the population (``x[j]``, a
        population sum) would move several units with one jump: a joint
        (multivariate) jump, which needs mixed moments.  Refused before the
        coupling is parsed (the parser refuses it under every interpretation,
        with a message about parsing)."""
        names = '|'.join(re.escape(f) for f in sorted(self._fields, key=len, reverse=True))
        for name, s in self._sources.items():
            for f, c in s['couples'].items():
                c = str(c)
                if ((names and re.search(rf'\b({names})\s*\[\s*(?!i\s*\])', c))
                        or re.search(r'\bsum\s*\(', c)):
                    raise SDEError(
                        f"interpretation('marcus'): the coupling {c!r} of source "
                        f"{name!r} to {f!r} reads other units of the population, so "
                        f"one jump would move several units jointly; joint "
                        f"(multivariate) jumps need mixed moments and are not "
                        f"supported. One scalar source coupled to several fields of "
                        f"the same unit (couples={{'x': ..., 'y': ...}}) is a jump "
                        f"along a vector field and is supported; independent sources "
                        f"per unit are too.")

    def _check_marcus_pp(self, n, pp):
        fix = (f"declare the jump as noise(..., Poisson(...) or CompoundPoisson(...), "
               f"couples={{field: coefficient}}), whose state-dependent coupling "
               f"takes the Marcus flow, or use interpretation('ito')")
        bare_in = []
        for f, eq in self._eqs.items():
            rhs = self._strip_conv(eq['rhs'])     # a filtered train Conv(K, n) is no jump
            if not re.search(rf'\b{re.escape(n)}\b', rhs):
                continue
            bare_in.append(f)
            try:
                R = self._parse(rhs, f'rhs of {f}')
            except NoiseError:
                raise SDEError(
                    f"interpretation('marcus'): spike train {n!r} enters "
                    f"equation({f!r}) outside Conv(...) in a form whose coefficient "
                    f"cannot be read (a population sum?); a point process with a "
                    f"state-dependent coupling is not supported under Marcus: "
                    f"{fix}.") from None
            nsym = SR.var(f'{n}__i')
            coef = R.diff(nsym)
            if coef.has(nsym) or self._field_dependent(coef):
                raise SDEError(
                    f"interpretation('marcus'): spike train {n!r} enters "
                    f"equation({f!r}) with the state-dependent coefficient "
                    f"{expr_to_text(coef)!r}; a point process with a state-dependent "
                    f"coupling is not supported under Marcus (a constant coupling "
                    f"is the same jump in every interpretation): {fix}.")
        if bare_in and self._field_dependent(self._parse(pp['size'], f'size of {n}')):
            raise SDEError(
                f"interpretation('marcus'): spike train {n!r} has the state-dependent "
                f"size {pp['size']!r} and enters equation({bare_in[0]!r}) as a jump; a "
                f"point process with a state-dependent coupling is not supported "
                f"under Marcus: {fix}.")

    def _marcus_units(self, name, s, coup):
        """Split a coupled source into the pieces the Marcus rule treats
        apart: ``[(name, source, series)]``.  Gaussian sources and Gaussian
        parts of a sum take the Stratonovich route (``series`` False); a jump
        source (or the jump parts of a sum) with a state-dependent coupling
        takes the Marcus series (``series`` True); with a constant coupling
        and no state-dependent Gaussian part the source is unchanged (Marcus
        = Ito: the flow of a constant field is a translation)."""
        from api.noise import SumSource, Cumulants, CGF
        src = s['source']
        fields = list(self._fields)
        parts = list(src.parts) if isinstance(src, SumSource) else [src]
        for q in parts:
            if isinstance(q, (Cumulants, CGF)) and q.field_dependent(fields):
                raise SDEError(
                    f"interpretation('marcus'): source {name!r} is a state-dependent "
                    f"{type(q).__name__} source, whose Gaussian part cannot be told "
                    f"from its jumps (the two take different Marcus rules); declare "
                    f"it as Gaussian(...) + a jump source, or use interpretation('ito').")
        if len(parts) == 1 and src.is_gaussian:
            return [(name, src, False)]
        sd = any(self._field_dependent(c) for c in coup.values())
        gauss = [q for q in parts if q.is_gaussian]
        jumps = [q for q in parts if not q.is_gaussian]
        if not sd and not any(q.field_dependent(fields) for q in gauss):
            return [(name, src, False)]
        units = [(name if len(parts) == 1 else f'{name} (part {k + 1})', q, False)
                 for k, q in enumerate(parts) if q.is_gaussian]
        if jumps:
            j = jumps[0] if len(jumps) == 1 else SumSource(*jumps)
            units.append((name, j, sd))
        return units

    def _marcus_series(self, name, src, coup, pop, source_terms, mf_rhs):
        """The Marcus noise term of the jump source ``src`` coupled through the
        vector field ``G = sum_f b_f d/dx_f`` (``b_f = c_f/a_f``):
        ``H = sum_{k<=N} kappa_k/k! D_k``, ``D_k = e^{-xt.x} G^k e^{xt.x}``
        (``D_1 = theta = sum_f c_f ft``, ``D_{k+1} = G D_k + theta D_k``),
        ``N = SDE.series_order``.  The action gets ``-H`` (its theta-linear
        part, the mean drift, included); the mean-field equations get that
        linear part (MF only), as for a non-Gaussian source on the Ito route.
        The cumulant rates ``kappa_k`` stay outside the flow (a state-dependent
        rate is the pre-jump rate)."""
        from sage.all import factorial
        N = int(self._series_order)
        src._check_order(N, f"the Marcus series of source {name!r} (series order {N})")
        a = self._dt_coefficients()
        slope = (self._marcus_affine_flow(coup, a)
                 if self._marcus_route == 'auto' and src._max_order() is None else None)
        if slope is not None:
            self._marcus_exact(name, src, coup, slope, pop, source_terms, mf_rhs)
            return
        resp = {f: SR.var(f'{f}t__i') for f in coup}
        theta = sum((coup[f] * resp[f] for f in coup), SR(0))
        b = {f: coup[f] / a[f] for f in coup}
        top = src._max_order()
        n_kap = N + 1 if top is None or top >= N + 1 else N
        kap = {k: parse_expr(self._index(expr_to_text(src.cumulant(k))),
                             what=f'cumulant {k} of {name}')
               for k in range(1, n_kap + 1)}
        D = theta.expand()
        H, nxt = kap[1] * D, None
        for k in range(2, n_kap + 1):
            D = (sum((b[f] * D.diff(self._sym(f)) for f in coup), SR(0)) + theta * D).expand()
            term = kap[k] / factorial(k) * D
            if k <= N:
                H += term
            else:
                nxt = term
        H = H.expand()
        zero = {r: 0 for r in resp.values()}
        for f in coup:
            lin = H.diff(resp[f]).subs(zero)
            if not lin.is_zero():
                mf_rhs[f].append(lin)
        source_terms.setdefault(pop, []).append(f'-{_wrap(expr_to_text(H))}')
        self.notes.append(
            f"Marcus: source {name!r} acts through the flow of its coupling; the "
            f"action carries sum_k kappa_k/k! D_k to order {N} (SDE.series_order); "
            f"the size of the next term at the saddle is result['marcus_truncation'].")
        self._marcus_terms[name] = {
            'route': 'series', 'order': N, 'population': pop,
            'responses': [f'{f}t' for f in coup], 'fields': list(self._fields),
            'kept': expr_to_text(H),
            'next': None if nxt is None else expr_to_text(nxt.expand())}

    def _marcus_affine_flow(self, coup, a):
        """The slopes ``s_f = d b_f/d x_f`` of the coupling ``G = sum_f b_f
        d/dx_f`` (``b_f = c_f/a_f``) when its flow is closed form: every
        ``b_f`` affine in its own field and free of the other coupled fields
        (a diagonal linear vector field; fields the source does not move may
        enter, being constant along the flow).  Then
        ``phi_z(x)_f - x_f = (b_f/s_f)(e^{s_f z} - 1)``, or ``b_f z`` where
        ``s_f = 0``.  ``None`` otherwise (the series route)."""
        moved = {f: self._sym(f) for f in coup}
        slope = {}
        for f, c in coup.items():
            g = SR(c) / a[f]
            if any(g.has(xh) for h, xh in moved.items() if h != f):
                return None
            s = g.diff(moved[f])
            if any(s.has(xh) for xh in moved.values()):
                return None
            slope[f] = s
        return slope

    def _marcus_exact(self, name, src, coup, slope, pop, source_terms, mf_rhs):
        """The Marcus noise term of a jump source with a closed-form flow
        (:meth:`_marcus_affine_flow`), exact.  With ``p = sum_f a_f ft x_f``,
        ``H = e^{-p} K(G) e^{p} = sum_n T[(sum_f w_f (e^{s_f z} - 1) + v z)^n]/n!``,
        ``w_f = c_f ft/s_f`` (``s_f != 0``), ``v = sum_{s_f = 0} c_f ft``, where
        the functional ``T`` takes ``z^m e^{sz}`` to ``K^{(m)}(s)``.  It is kept
        to ``n <= SDE.series_order`` response legs (as the Ito action keeps an
        irregular CGF): every kept vertex is exact, with no moment series in
        ``z``.  ``T`` acts term by term on ``K``: ``A e^{alpha theta}`` gives
        ``A (sum_f w_f (e^{alpha s_f} - 1) + v alpha)^n`` (a jump of size
        ``alpha``), ``A theta^p`` the ``p``-th ``z``-derivative at 0, and any
        other term the alternating binomial sum of its values at
        ``sum_f j_f s_f``, put over one denominator when rational (which removes
        its cancellation).  ``compute_cumulants`` refuses parameters at which
        the jump law's mgf does not exist where a usable vertex needs it, and
        sizes the rounding of the kept coefficients
        (``result['marcus_truncation']``)."""
        from itertools import product as _iprod
        from sage.all import binomial, factorial, exp as _exp
        N = int(self._series_order)
        resp = {f: SR.var(f'{f}t__i') for f in coup}
        moving = [f for f in coup if not SR(slope[f]).is_zero()]
        still = [f for f in coup if SR(slope[f]).is_zero()]
        K = parse_expr(self._index(expr_to_text(src.cgf())), what=f'K of {name}')
        shapes, bases = _cgf_shapes(K)
        z = SR.var('z__marcus')
        v = sum((coup[f] * resp[f] for f in still), SR(0))

        def T(mu, m):
            n = sum(mu) + m
            tot = SR(0)
            for kind, key, cf in shapes:
                if kind == 'exp':
                    t = key ** m
                    for f, q in zip(moving, mu):
                        t *= (_exp(key * slope[f]) - 1) ** q
                elif kind == 'poly':
                    if key < n:                 # the z-power series starts at z^n
                        continue
                    e = z ** m
                    for f, q in zip(moving, mu):
                        e *= (_exp(slope[f] * z) - 1) ** q
                    t = e.taylor(z, 0, key).coefficient(z, key) * factorial(key)
                else:
                    gm = key.diff(THETA, m) if m else key
                    t = SR(0)
                    for js in _iprod(*[range(q + 1) for q in mu]):
                        w = SR(1)
                        for q, j in zip(mu, js):
                            w *= binomial(q, j) * (-1) ** (q - j)
                        t += w * gm.subs({THETA: sum((j * slope[f] for f, j in zip(moving, js)),
                                                     SR(0))})
                    if _is_rational_in(key, THETA):
                        t = t.factor()
                tot += cf * t
            return tot

        H, coefs = SR(0), {}
        for n in range(1, N + 1):
            for mu in _compositions(n, len(moving) + (1 if still else 0)):
                mu, m = (mu[:-1], mu[-1]) if still else (mu, 0)
                P = T(mu, m)
                for f, q in zip(moving, mu):
                    P = P / slope[f] ** q
                if SR(P).is_zero():
                    continue
                mono = v ** m / factorial(m)
                for f, q in zip(moving, mu):
                    mono *= (coup[f] * resp[f]) ** q / factorial(q)
                coefs[','.join(map(str, tuple(mu) + ((m,) if still else ())))] = expr_to_text(P)
                H += P * mono
        zero = {r: 0 for r in resp.values()}
        for f in coup:
            lin = H.diff(resp[f]).subs(zero)
            if not lin.is_zero():
                mf_rhs[f].append(lin)
        source_terms.setdefault(pop, []).append(f'-{_wrap(expr_to_text(H))}')
        self.notes.append(
            f"Marcus: source {name!r} acts through the closed-form flow of its affine "
            f"coupling; the action carries its exact noise term to {N} response legs "
            f"(SDE.series_order), with no moment series.")
        self._marcus_terms[name] = {
            'route': 'exact', 'order': N, 'population': pop,
            'responses': [f'{f}t' for f in coup], 'fields': list(self._fields),
            'slopes': {f: expr_to_text(SR(slope[f])) for f in coup},
            'cgf': expr_to_text(K), 'mgf_bases': [expr_to_text(b) for b in bases],
            'coefficients': coefs}

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
        star = self._star_equations(d['equations'])
        if star is not None:
            for sname, rhs in star:
                b.set_mf_equation(sname, rhs)
        else:
            for e in d['equations']:
                b.equation(lhs=e['lhs'], rhs=e['rhs'], population=e['population'])
        if self._stability is not None:
            b.stability_analysis(self._stability)
        return b

    def _star_equations(self, eqs):
        """For a model with a point process whose every mean-field equation
        reads ``x = rhs`` at Dt = 0: the saddle equations ``xstar = rhs*``
        (``set_mf_equation``), the form hand-written point-process models use
        (the same saddle, solved by the same route, bit for bit).  ``None``
        otherwise (the residual ``.equation`` form)."""
        if not any(s.get('pp') for s in self._sources.values()):
            return None
        names = sorted(self._fields, key=len, reverse=True)
        pat = re.compile(r'\b(' + '|'.join(map(re.escape, names)) + r')\s*\[')
        out = []
        for e in eqs:
            f = e['field']
            try:
                L = parse_expr(e['lhs'], what=f'lhs of {f}').subs({_DT: 0})
            except NoiseError:
                return None
            if not (L - self._sym(f)).simplify_full().is_zero():
                return None
            if (self._fields[f]['exposed'] is None
                    and re.search(rf'\b{re.escape(f)}\s*\[', e['rhs'])):
                # a continuous field with x = F(x, ...) (e.g. a conductance
                # term (Es - v)*Conv(g, n)): an implicit saddle equation, not
                # an explicit xstar = ... (which the engine substitutes as
                # written); the residual ``.equation`` form solves it.  (A
                # spike train's nstar = F(nstar) is the iteration saddle the
                # hand-written point-process models use: unchanged.)
                return None
            out.append((f'{f}star', pat.sub(lambda m: m.group(1) + 'star[', e['rhs'])))
        return out

    def simulate(self, **kw):
        """Simulate the model's point processes, drifts and Gaussian sources
        (:func:`api.pp_simulate.simulate`; keywords ``T``, ``dt``, ``n_paths``,
        ``burn_in``, ``bin``, ``params``, ``seed``)."""
        from api.pp_simulate import simulate
        return simulate(self, **kw)

    def _check_derived_names(self):
        """Refuse a parameter, function or kernel named like a symbol the
        compiler derives from a field ``f``: its response field ``ft``, its
        fluctuation ``df`` or its saddle ``fstar`` (the clash was otherwise
        found only by the analyzer at run time)."""
        derived = {}
        for f in self._fields:
            for n, what in ((f + 't', 'response field'),
                            ('d' + f, 'fluctuation'),
                            (f + 'star', 'mean-field saddle')):
                derived[n] = (what, f)
        for kind, names in (('parameter', self._param_names),
                            ('function', self._function_names),
                            ('kernel', list(self._kernels))):
            for n in names:
                if n in derived:
                    what, f = derived[n]
                    raise SDEError(
                        f'{kind} name {n!r} is the {what} the compiler derives '
                        f'from field {f!r}; choose another name.')

    def build(self) -> dict:
        self._check_derived_names()
        m = self.to_builder().build()
        trunc = sorted(set(n for n, s in self._sources.items()
                           if not s.get('pp') and not getattr(s['source'], 'regular', True))
                       | set(n for n, t in self._marcus_terms.items() if t['route'] == 'series'))
        exact = sorted(n for n, t in self._marcus_terms.items() if t['route'] == 'exact')
        if exact:
            # the exact Marcus terms are kept to SDE.series_order response
            # legs; compute_cumulants refuses k + max_ell above it
            m['marcus_exact_order'] = {'order': self._series_order, 'sources': exact}
        if self._interp == 'marcus':
            m['interpretation'] = 'marcus'
        if self._marcus_terms:
            # the Marcus series of each jump source with a state-dependent
            # coupling, kept to SDE.series_order, and its next term:
            # compute_cumulants sizes that term at the saddle
            m['marcus_series'] = {n: dict(t) for n, t in self._marcus_terms.items()}
        if trunc:
            # the action holds these CGFs as Taylor polynomials of this order;
            # compute_cumulants refuses (k, max_ell) that would need more
            m['noise_series_order'] = {'order': self._series_order, 'sources': trunc}
        declared = {n: s['source'].declared_order for n, s in self._sources.items()
                    if s.get('source') is not None and hasattr(s['source'], 'declared_order')}
        if declared:
            # Cumulants([...]) sources: cumulants above the declared order are 0;
            # compute_cumulants warns when (k, max_ell) can reach them
            m['noise_declared_orders'] = declared
        return m

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

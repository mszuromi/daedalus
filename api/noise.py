"""Noise-source library for the SDE front-end (:mod:`api.sde`).

Every noise source is specified by ONE primitive: its per-unit-time
conditional cumulant generating function (CGF) ``K(theta; x)``,

    E[exp(theta * d eta) | past] = exp(K(theta; x) * dt),

so ``K(0) = 0`` and the ``m``-th cumulant rate of ``d eta`` is
``d^m K / d theta^m`` at ``theta = 0``.  The MSR action of
``L_i x_i = r_i(x) + sum_a c_ia(x) * eta_a'`` is

    S = sum_i psi_i (L_i x_i - r_i) - sum_a K_a( sum_i c_ia psi_i ; x ),

(``psi_i`` is the response field of ``x_i``; in Daedalus the response
field of ``x`` is ``xt``).  With this sign and normalisation
``Gaussian(var='2*D')`` gives the familiar ``-D*xt^2`` and
``Poisson(rate='phi')`` gives ``-(exp(xt) - 1)*phi``.

Only infinitely divisible (Levy-type) sources are offered: Gaussian,
Poisson, compound Poisson, the Gamma and inverse-Gaussian subordinators,
and two escape hatches (:class:`Cumulants`, :class:`CGF`).  There is no
"per-step distribution" constructor on purpose: an arbitrary per-step
law does not define a continuous-time process.

Parameters are Sage-syntax expressions (``str``), numbers, or Sage
symbolic expressions.  They may reference declared parameters and
fields with the repository's index notation (``x[i]``, ``w[i,j]``,
``phi[i](v[i])``).  Symbols are checked against the declared names by
:meth:`NoiseSource.validate` (the SDE builder calls it); an explicit
time ``t`` is always rejected (time-independent sources only).
"""
from __future__ import annotations

import re
import warnings
from math import factorial
from typing import Iterable, Optional, Sequence

from sage.all import SR, Integer, exp, log, sqrt, function, sage_eval
from sage.all import gamma as _gamma, cos as _cos, matrix as _matrix
from engine.errors import DaedalusError

__all__ = [
    'NoiseError', 'NoiseSource', 'CumulantTruncationWarning',
    'Gaussian', 'GaussianVector', 'Poisson', 'CompoundPoisson', 'GammaProcess',
    'InverseGaussianProcess', 'Cumulants', 'CGF',
    'JumpDistribution', 'Dirac', 'Exponential', 'Gamma', 'GaussianJump',
    'Laplace', 'Uniform', 'Bernoulli', 'Binomial', 'FinitePMF',
    'VarianceGamma', 'NormalInverseGaussian', 'CGMY', 'TemperedStableSubordinator',
    'Meixner', 'BilateralGamma', 'Skellam', 'NegativeBinomialProcess', 'SumSource',
    'Geometric', 'PoissonJump', 'NegativeBinomial', 'Logarithmic', 'DoubleExponential',
    'AsymmetricLaplace', 'Hyperexponential', 'PhaseType', 'ChiSquared', 'LogNormal',
    'Weibull', 'merton_jump_diffusion', 'kou_jump_diffusion', 'levy_driven_ou',
    'SOURCE_TYPES', 'THETA', 'parse_expr', 'expr_to_text', 'expr_names',
]


class NoiseError(DaedalusError, ValueError):
    """Raised for an invalid noise declaration."""


class CumulantTruncationWarning(UserWarning):
    """A :class:`Cumulants` source was asked for a cumulant above the order it
    declares; that cumulant is taken as 0 (a formal truncation)."""


# The CGF variable.  Reserved: a parameter or field may not be called theta.
THETA = SR.var('theta')

# Names of elementary functions accepted in expressions (Sage meaning).
_MATH_FUNCS = {
    'exp': exp, 'log': log, 'sqrt': sqrt,
}
import sage.all as _sa                                     # noqa: E402
for _n in ('sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'arctan',
           'heaviside', 'erf'):
    _MATH_FUNCS[_n] = getattr(_sa, _n)
_MATH_FUNCS['abs'] = _sa.abs_symbolic
# Sage's own functions for names it prints itself (gamma(1 - Y) prints as factorial(-Y), e.g. in a
# CGMY mean), used only where the name is called, so a parameter may still be named gamma.  A formal
# function of the same name breaks Sage 10.9's Maxima conversion ("operator mismatch").
_SAGE_PRINTED_CALLS = {'factorial': _sa.factorial, 'gamma': _sa.gamma}

_INDEX_RE = re.compile(
    r'\b([A-Za-z_]\w*)\s*\[\s*([A-Za-z_]\w*)\s*(?:,\s*([A-Za-z_]\w*)\s*)?\]')


# ── Expression helpers (text <-> Sage SR) ──────────────────────────────

def _flatten_indices(text: str) -> str:
    """``x[i]`` -> ``x__i``, ``w[i,j]`` -> ``w__i__j``."""
    return _INDEX_RE.sub(
        lambda m: m.group(1) + '__' + m.group(2)
        + ('__' + m.group(3) if m.group(3) else ''), text)


def _unflatten_name(name: str) -> str:
    parts = name.split('__')
    if len(parts) == 1:
        return name
    return f'{parts[0]}[{",".join(parts[1:])}]'


_PY_KEYWORDS = frozenset(__import__('keyword').kwlist) - {'for', 'in', 'if', 'else'}
_TOKEN_RE = re.compile(r'\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+)|([^\W\d]\w*)|(\*\*|[-+*/^%<>=!]=?|.))')


def caret_block(text: str, index: int, *, indent: str = '    ') -> str:
    """``text`` (the line holding character ``index``) and a ``^`` line under
    that character."""
    index = max(0, min(int(index), len(text)))
    start = text.rfind('\n', 0, index) + 1
    end = text.find('\n', index)
    line = text[start:end if end >= 0 else len(text)]
    return f'{indent}{line}\n{indent}{" " * (index - start)}^'


def format_caret_error(text: str, index: int, headline: str, hint: str = '') -> str:
    """``headline``, then ``text`` with a caret under character ``index``,
    then the hint."""
    out = f'{headline}\n{caret_block(text, index)}'
    return out + (f'\n    hint: {hint}' if hint else '')


def _bracket_problem(text: str):
    """The first unmatched bracket: ``(index, reason, hint)`` or ``None``."""
    pairs, stack = {')': '(', ']': '['}, []
    for k, ch in enumerate(text):
        if ch in '([':
            stack.append((ch, k))
        elif ch in ')]':
            if not stack or stack[-1][0] != pairs[ch]:
                return (k, f"unmatched {ch!r}",
                        f"remove the extra {ch!r}"
                        + (f" or close the {stack[-1][0]!r} opened at column {stack[-1][1] + 1}"
                           if stack else ' (nothing is open here)'))
            stack.pop()
    if stack:
        ch, k = stack[-1]
        return (k, f'{ch!r} is never closed',
                f"add the closing {')' if ch == '(' else ']'!r} (brackets must balance)")
    return None


def expression_syntax_problem(text: str):
    """Where ``text`` stops being an expression: ``(index, reason, hint)`` for
    the first syntax problem (``index`` a character of ``text``), or ``None``
    when it reads as one.  No Sage is involved; the grammar is the one the
    parser accepts (Python's, with ``^`` for the power)."""
    import ast
    text = str(text)
    if not text.strip():
        return (0, 'the expression is empty', 'give an expression, e.g. -mu*x')
    prob = _bracket_problem(text)
    if prob:
        return prob
    toks = [(m.start(m.lastindex), m.group(m.lastindex)) for m in _TOKEN_RE.finditer(text)
            if m.lastindex]
    for k, tok in toks:
        if tok in _PY_KEYWORDS:
            return (k, f'{tok!r} is a Python keyword and cannot be a name',
                    f'rename the symbol (e.g. {tok[:3]} or {tok}0)')
    for (a, t1), (b, t2) in zip(toks, toks[1:]):
        if t2 == '(' and (re.fullmatch(r'\d[\w.]*|\.\d+', t1) or t1 == ')'):
            return (b, f'missing operator between {t1!r} and {t2!r}',
                    f'write the product with "*", e.g. {t1}*(...)')
    py, idx = [], []
    for k, ch in enumerate(text):
        piece = '**' if ch == '^' else (' ' if ch == '\n' else ch)
        py.append(piece)
        idx.extend([k] * len(piece))
    try:
        ast.parse(''.join(py), mode='eval')
        return None
    except SyntaxError as exc:
        off = (exc.offset or 0) - 1
        msg = str(exc.msg)
        if exc.lineno and exc.lineno > 1:                    # (never: newlines are blanked)
            off = 0
    except (ValueError, MemoryError, RecursionError):
        return (0, 'the expression cannot be read', '')
    end = len(text.rstrip())
    k = end if off < 0 or off >= len(idx) else idx[off]
    operand = re.compile(r'[\w.(]')
    prev = [(a, t) for a, t in toks if a < k]
    cur = next(((a, t) for a, t in toks if a >= k and t.strip()), None)
    p_tok = prev[-1][1] if prev else ''
    c_tok = cur[1] if cur else ''
    if k >= end:
        if p_tok and re.fullmatch(r'\*\*|[-+*/^%]', p_tok):
            return (end - 1 if p_tok != '**' else end - 2,
                    f'the expression ends after the operator {p_tok!r}',
                    'an operator needs a right-hand operand')
        return (end, 'the expression ends too early', 'is something missing at the end?')
    if c_tok == '=':
        return (cur[0], "'=' (assignment) in an expression",
                'an expression has no "="; an equation is written '
                "equation('x', lhs='Dt*x', rhs='...')")
    if c_tok and not re.fullmatch(r'[\w.]+|\*\*|[-+*/^%<>=!]=?|[()\[\],]', c_tok):
        return (cur[0], f'unexpected character {c_tok!r}', 'remove it')
    if re.fullmatch(r'\*\*|[-+*/^%]', p_tok) and re.fullmatch(r'\*\*|[*/^%]', c_tok):
        return (cur[0], f'two operators in a row ({p_tok!r} then {c_tok!r})',
                'an operator needs an operand on each side')
    if re.search(r'\bfor\b', text) and not re.search(r'\bfor\s+\w+\s+in\b', text):
        at = re.search(r'\bfor\b(?:\s+\w+\s*)?', text).end()
        return (at, "a population sum needs 'in' after the index",
                'write sum(expression for j in pop)')
    if 'forgot a comma' in msg:
        for (a, t1), (b, t2) in zip(toks, toks[1:]):
            if a >= k and t2 == 'in' and re.fullmatch(r'\w+', t1):
                return (b, "a population sum needs 'for' before the index",
                        'write sum(expression for j in pop)')
            if a >= k and re.fullmatch(r'[\w.]+', t1) and re.fullmatch(r'[\w.]+', t2):
                return (b, f'missing operator between {t1!r} and {t2!r}',
                        f'write the product with "*", e.g. {t1}*{t2}')
    if c_tok == 'in' and p_tok and re.fullmatch(r'\w+', p_tok):
        return (cur[0], "a population sum needs 'for' before the index",
                'write sum(expression for j in pop)')
    if p_tok and re.fullmatch(r'[\w.]+', p_tok) and operand.match(c_tok):
        return (cur[0], f'missing operator between {p_tok!r} and {c_tok!r}',
                f'write the product with "*", e.g. {p_tok}*{c_tok}')
    if 'decimal literal' in msg:
        m = re.match(r'[\d.]+(?:[eE][+-]?\d+)?', text[k:])
        at = k + (m.end() if m else 0)
        return (at, 'a number runs into a name', 'write the product with "*", e.g. 2*x')
    if re.fullmatch(r'\*\*|[-+*/^%]', p_tok) and c_tok in (')', ']', ','):
        return (cur[0], f'the operator {p_tok!r} has no right-hand operand',
                'an operator needs an operand on each side')
    if p_tok in ('(', '[', ',') and c_tok in (')', ']', ','):
        return (cur[0], f'nothing between {p_tok!r} and {c_tok!r}', 'an argument is missing')
    return (k, msg, '')


def parse_failure_message(value, what: str, exc=None) -> str:
    """The text of a parse failure for ``value``: the reason, the string with
    a caret under the failing character, and a hint."""
    txt = ' '.join(str(value).split())
    prob = expression_syntax_problem(txt)
    if prob:
        index, reason, hint = prob
        return format_caret_error(txt, index, f'{what}: cannot parse the expression: {reason}',
                                  hint)
    return f'{what}: cannot parse {value!r}' + (f' ({exc})' if exc is not None else '')


def parse_expr(value, *, what: str = 'expression'):
    """Parse a parameter value (str / number / SR) into a Sage expression.

    Indexed names become flat symbols (``x[i]`` -> ``x__i``); calls of
    unknown names become formal Sage functions (``phi[i](v[i])`` ->
    ``phi__i(v__i)``).  Population sums (``sum(... for j in E)``) cannot be
    parsed here (noise parameters, separated drift terms).
    """
    if isinstance(value, str):
        txt = ' '.join(value.split())
        if not txt:
            raise NoiseError(f'{what}: empty expression')
        if re.search(r'\bfor\b', txt):
            raise NoiseError(
                f'{what}: population sums ("sum(... for j in pop)") cannot '
                f'be split or parsed in this expression ({value!r}); they are '
                f'accepted where the model is written as a whole (an '
                f'equation(lhs=..., rhs=...) whose lhs carries the restoring '
                f'term, a point_process rate), '
                f'not as a noise parameter or a separated term.  Declare a '
                f'parameter or function '
                f'for the summed quantity instead (a spike train whose rate '
                f'sums or filters fields: SDE.point_process).')
        bad = expression_syntax_problem(txt)
        if bad:
            raise NoiseError(parse_failure_message(txt, what))
        txt = _flatten_indices(txt)
        called = set(re.findall(r'\b([A-Za-z_]\w*)\s*\(', txt))
        names = set(re.findall(r'\b([A-Za-z_]\w*)\b', txt))
        loc = {}
        for n in names:
            if n in _MATH_FUNCS:
                loc[n] = _MATH_FUNCS[n]
            elif n in called and n in _SAGE_PRINTED_CALLS:
                loc[n] = _SAGE_PRINTED_CALLS[n]
            elif n in called:
                loc[n] = function(n)
            elif n == 'theta':
                loc[n] = THETA
            elif n in ('pi', 'I', 'e'):
                continue
            else:
                loc[n] = SR.var(n)
        try:
            return SR(sage_eval(txt, locals=loc))
        except Exception as exc:                       # noqa: BLE001
            raise NoiseError(f'{what}: cannot parse {value!r} ({exc})') from exc
    try:
        return SR(value)
    except Exception as exc:                           # noqa: BLE001
        raise NoiseError(f'{what}: cannot convert {value!r} to an '
                         f'expression ({exc})') from exc


def _numeric(v):
    """``float(v)`` for a real numeric expression, else ``None`` (symbolic)."""
    try:
        v = SR(v)
        if v.is_numeric() and not v.variables():
            return float(v)
    except Exception:                                  # noqa: BLE001
        pass
    return None


def _require(who, value, name, ok, why):
    """Raise :class:`NoiseError` when a NUMERIC parameter violates ``ok``
    (symbolic parameters are not checked)."""
    x = _numeric(value)
    if x is not None and not ok(x):
        raise NoiseError(f'{who}: parameter {name}={x:g} is invalid; {why}.')


def _seq(who, what, arg):
    """``list(arg)``, refusing a string (``list('Dn')`` would silently give
    the expressions ``D`` and ``n``)."""
    if isinstance(arg, (str, bytes)):
        raise NoiseError(f'{who}: {what} must be a list of expressions, not the '
                         f'string {arg!r}.')
    return list(arg)


def expr_names(expr) -> set[str]:
    """Bare names (index suffixes stripped) of the free symbols and formal
    functions in ``expr``: ``x__i`` -> ``x``, ``phi__i(...)`` -> ``phi``."""
    out = {str(v).split('__')[0] for v in SR(expr).variables()}
    for f in _formal_functions(SR(expr)):
        out.add(f.split('__')[0])
    return out


def _formal_functions(expr) -> set[str]:
    out = set()

    def walk(e):
        op = e.operator()
        if op is None:
            return
        name = getattr(op, 'name', None)
        if callable(name):
            nm = op.name()
            if nm not in _MATH_FUNCS and nm not in ('exp', 'log'):
                if hasattr(op, '__module__') and 'function_factory' in str(type(op)):
                    out.add(nm)
                elif type(op).__name__ == 'NewSymbolicFunction':
                    out.add(nm)
        for o in e.operands():
            walk(o)
    walk(SR(expr))
    return out


def _num_text(c) -> str:
    c = SR(c)
    if c.is_integer():
        return str(c)
    try:
        if c.is_rational():
            q = c.pyobject()
            return f'({q.numerator()}/{q.denominator()})'
    except Exception:                                  # noqa: BLE001
        pass
    return f'({c})'


def expr_to_text(expr) -> str:
    """Print a Sage expression in the repository's action-text syntax
    (``exp(...)`` rather than ``e^...``, ``x[i]`` rather than ``x__i``)."""
    import operator as _op
    from sage.symbolic.operators import add_vararg, mul_vararg
    e = SR(expr)
    op = e.operator()
    if op is None:
        if e.is_symbol():
            return _unflatten_name(str(e))
        if e.is_numeric() or e.is_constant():
            if e == SR('e'):
                return 'exp(1)'
            if e.is_numeric() and (e.is_real() if hasattr(e, 'is_real') else True):
                return _num_text(e)
            return f'({e})'
        return f'({e})'
    ops = e.operands()
    if op is _op.add or op is add_vararg:
        parts = [expr_to_text(o) for o in ops]
        s = parts[0]
        for p in parts[1:]:
            s += (' - ' + p[1:]) if p.startswith('-') else (' + ' + p)
        return s
    if op is _op.mul or op is mul_vararg:
        num, den = [], []
        coeff = SR(1)
        for o in ops:
            if o.operator() is _op.pow and SR(o.operands()[1]).is_numeric() \
                    and SR(o.operands()[1]) < 0:
                b, k = o.operands()
                den.append(b if k == -1 else b ** (-k))
            elif o.is_numeric():
                coeff *= o
            else:
                num.append(o)
        sign = ''
        if coeff.is_numeric() and coeff < 0:
            sign, coeff = '-', -coeff
        factors = [] if coeff == 1 else [_num_text(coeff)]
        for o in num:
            t = expr_to_text(o)
            factors.append(f'({t})' if o.operator() in (_op.add, add_vararg) or t.startswith('-') else t)
        s = '*'.join(factors) if factors else '1'
        if den:
            ds = [expr_to_text(d) for d in den]
            dd = '*'.join(f'({d})' for d in ds)
            s = f'{s}/({dd})' if len(ds) > 1 else f'{s}/({ds[0]})'
        return sign + s
    if op is _op.pow:
        b, k = ops
        if SR(k) == SR(1) / 2:
            return f'sqrt({expr_to_text(b)})'
        if SR(k) == -1:
            return f'1/({expr_to_text(b)})'
        bt = expr_to_text(b)
        if b.operator() is not None or bt.startswith('-'):
            bt = f'({bt})'
        kt = expr_to_text(k)
        if not (SR(k).is_integer() and SR(k) >= 0):
            kt = f'({kt})'
        return f'{bt}^{kt}'
    name = op.name() if callable(getattr(op, 'name', None)) else str(op)
    if name == 'exp':
        return f'exp({expr_to_text(ops[0])})'
    args = ', '.join(expr_to_text(o) for o in ops)
    return f'{_unflatten_name(name)}({args})'


# ── Base class ─────────────────────────────────────────────────────────

class NoiseSource:
    """A named-parameter noise source with a closed-form per-unit-time CGF.

    Subclasses implement :meth:`_cgf` (the CGF in :data:`THETA`).
    """

    #: True for the Gaussian source (Stratonovich conversion allowed).
    is_gaussian = False
    #: True when the CGF is analytic at theta = 0 in closed form (a
    #: removable singularity, e.g. the uniform jump mgf, is False).
    regular = True
    #: spatial structure of the noise of an SPDE (``'white'``); None: time-only
    space = None

    def __init__(self, **params):
        self.params = {k: parse_expr(v, what=f'{type(self).__name__}({k}=...)')
                       for k, v in params.items()}
        self._check_domain()
        self._check_common()

    def _check_domain(self):
        """Parameter-domain checks (numeric values only); default none."""

    def _require(self, name, ok, why):
        _require(type(self).__name__, self.params[name], name, ok, why)

    # -- sampling (api.noise_samplers) -------------------------------------
    def sample(self, dt, size, rng, values=None):
        """``size`` i.i.d. increments of the process over a step ``dt``
        (exact where a closed-form representation is known; see
        :mod:`api.noise_samplers`).  ``values`` binds field-dependent
        parameters (name -> float or array of length ``size``)."""
        from api.noise_samplers import sample_source
        return sample_source(self, dt, size, rng, values)

    # -- the primitive -----------------------------------------------------
    def _cgf(self):                                    # pragma: no cover
        raise NotImplementedError

    def cgf(self, theta=None):
        """``K(theta)`` as a Sage expression (default variable :data:`THETA`)."""
        K = self._cgf()
        if theta is not None:
            K = K.subs({THETA: SR(theta)})
        return K

    def _max_order(self):
        """Highest exact cumulant order (``None``: unlimited).  Finite when
        the CGF contains a truncated moment series (see :class:`_SeriesJump`)."""
        return None

    def _check_order(self, m, what):
        top = self._max_order()
        if top is not None and int(m) > top:
            raise NoiseError(
                f'{type(self).__name__}: its CGF is a series truncated at order '
                f'{top} (a moment-series jump law); {what} needs order {int(m)}. '
                f'Raise the law\'s order= (or lower SDE.series_order).')

    def cumulant(self, m: int):
        """``d^m K / d theta^m`` at 0: the ``m``-th cumulant per unit time."""
        m = int(m)
        if m < 1:
            raise NoiseError('cumulant order must be >= 1')
        self._check_order(m, f'cumulant {m}')
        K = self._cgf()
        if self.regular:
            return (K.diff(THETA, m)).subs({THETA: 0}).simplify_full()
        ser = K.taylor(THETA, 0, m)
        return (ser.coefficient(THETA, m) * factorial(m)).simplify_full()

    def mean(self):
        """Mean rate ``K'(0)`` (the drift contributed by this source)."""
        return self.cumulant(1)

    def cgf_series(self, order: int):
        """Taylor polynomial of ``K`` in theta up to ``order``."""
        self._check_order(order, f'a series of order {int(order)}')
        return sum(self.cumulant(m) * THETA ** m / factorial(m)
                   for m in range(1, int(order) + 1))

    def action_cgf(self, order: int = 8):
        """The CGF as written into the action: the closed form when it is
        regular at 0, else the Taylor polynomial to ``order``.  Refused when
        a truncated moment series carries fewer than ``order`` terms."""
        self._check_order(order, f'the action (series order {int(order)})')
        return self._cgf() if self.regular else self.cgf_series(order)

    # -- validation --------------------------------------------------------
    def _check_common(self):
        for k, v in self.params.items():
            names = expr_names(v)
            if 't' in names:
                raise NoiseError(
                    f'{type(self).__name__}: parameter {k}={v} depends on the '
                    f'time t; only time-independent sources are supported.')
            if THETA in SR(v).variables():
                raise NoiseError(
                    f'{type(self).__name__}: parameter {k} may not contain '
                    f'theta (reserved for the CGF variable).')
        if self._k0_roundoff():
            return
        K0 = self._value_at_zero()
        k0 = _numeric(K0)
        if k0 is not None:                              # float round-off at K(0), relative to the largest parameter
            scale = max([1.0] + [abs(x) for x in map(_numeric, self.params.values()) if x is not None]) ** 2
            if abs(k0) <= 1e-12 * scale:
                return
        if not SR(K0).simplify_full().is_zero():
            raise NoiseError(
                f'{type(self).__name__}: K(0) = {K0} != 0; a per-unit-time '
                f'CGF must vanish at theta = 0.')

    def _k0_roundoff(self):
        """True when ``K(0)`` is float round-off by a criterion of the
        source's own (dimensionless) quantities; default False."""
        return False

    def _value_at_zero(self):
        K = self._cgf()
        if self.regular:
            return K.subs({THETA: 0})
        return K.limit(theta=0)

    def free_names(self) -> set[str]:
        """Bare names this source's parameters reference."""
        out = set()
        for v in self.params.values():
            out |= expr_names(v)
        return out

    def validate(self, allowed: Iterable[str]):
        """Raise :class:`NoiseError` if a parameter references a name that is
        not in ``allowed`` (declared fields, parameters, functions, indices)."""
        allowed = set(allowed)
        unknown = sorted(self.free_names() - allowed)
        if unknown:
            raise NoiseError(
                f'{type(self).__name__}: undeclared symbol(s) {unknown}; every '
                f'symbol must be a declared field, parameter, function or '
                f'population index.')
        return self

    def field_dependent(self, field_names: Iterable[str]) -> bool:
        return bool(self.free_names() & set(field_names))

    def __repr__(self):
        args = ', '.join(f'{k}={expr_to_text(v)!r}' for k, v in self.params.items())
        return f'{type(self).__name__}({args})'


# ── Gaussian / Poisson-type / Levy sources ─────────────────────────────

class Gaussian(NoiseSource):
    """White Gaussian noise: ``K = mean*theta + var*theta^2/2``.

    ``var`` (the variance per unit time) may depend on the fields
    (multiplicative noise, Ito by default).  A nonzero ``mean`` is moved
    into the drift by the SDE builder: ``Gaussian(mean=phi, var=phi^2)``
    is ``d eta = phi dt + phi dW``.

    ``space='white'`` makes it space-time white noise for an SPDE
    (``SDE(..., space=d)``): ``<eta(x,t) eta(x',t')> = var delta(x - x')
    delta(t - t')``.  (Conserved noise, ``div`` of a white vector noise, is not
    supported yet.)

    ``kernel=`` makes it coloured: the white noise of intensity ``var``
    filtered by a causal kernel ``h`` (an :class:`api.delay.Kernel`:
    ``from api.delay import Kernel``, ``Gaussian(var, kernel=Kernel.exponential('tau'))``; ``Kernel.alpha('tau')``,
    ``Kernel.gamma(n, 'mean')`` or ``Kernel.image(text)`` with a rational
    image), ``eta = h * xi``, so ``<eta(t) eta(t')> = var (h * h~)(t - t')``
    and the integral of the covariance is ``var |H(0)|^2`` (``var`` for a
    kernel of unit area: ``Kernel.exponential(tau)`` gives the
    Ornstein-Uhlenbeck covariance ``var/(2 tau) exp(-|t - t'|/tau)``, white
    noise of intensity ``var`` as ``tau -> 0``).  The SDE builder embeds the
    filter as fields (a Markov embedding, exact); its image must be rational
    in ``I*omega`` and proper, and ``var`` free of the fields (a state-dependent
    coupling goes in ``into=``).
    """
    is_gaussian = True

    def __init__(self, var, mean=0, *, space=None, kernel=None):
        if space not in (None, 'white'):
            if space == 'conserved':
                raise NoiseError("Gaussian(space='conserved'): conserved noise (the divergence "
                                 "of a white vector noise) is not supported yet; use "
                                 "space='white'.")
            raise NoiseError(f"Gaussian(space={space!r}): use space='white' (space-time white "
                             f"noise of an SPDE) or omit it (time-only noise).")
        self.space = space
        if kernel is not None:
            from api.delay import Kernel
            if not isinstance(kernel, Kernel):
                raise NoiseError(f'Gaussian(kernel=...): give an api.delay.Kernel (e.g. '
                                 f"Kernel.exponential('tau')); got {type(kernel).__name__}.")
            if kernel.delay is not None:
                raise NoiseError('Gaussian(kernel=...): a delayed kernel has no Markov embedding '
                                 '(no finite state); filter the noise with a delay-free rational '
                                 'kernel.')
            if space is not None:
                raise NoiseError('Gaussian(kernel=...): coloured noise is temporal (not with '
                                 "space='white').")
        self.kernel = kernel
        super().__init__(var=var, mean=mean)

    def __repr__(self):
        out = super().__repr__()
        return out if self.kernel is None else f'{out[:-1]}, kernel={self.kernel!r})'

    def _check_domain(self):
        self._require('var', _nonneg, 'need var >= 0 (a variance per unit time)')

    def _cgf(self):
        return self.params['mean'] * THETA + self.params['var'] * THETA ** 2 / 2


class GaussianVector(NoiseSource):
    """Correlated white Gaussian noise of ``n`` components,
    ``<d eta_k d eta_l> = cov[k][l] dt`` (``cov`` a symmetric ``n x n`` matrix
    of numbers or parameter expressions, free of the fields; ``mean`` an
    optional vector).  Coupled with ``into={field: [c_1, ..., c_n]}`` (one
    coefficient per component, ``dx += sum_k c_k d eta_k``); its action term is
    ``-(1/2) theta^T cov theta`` with ``theta_k = sum_x c_xk xt``.  Under the
    Stratonovich (or Marcus) rule the drift correction is
    ``1/2 sum_kl cov_kl sum_j b_jl d_j b_ik`` (``b = c/a``)."""
    is_gaussian = True

    def __init__(self, cov, mean=None):
        try:
            rows = [list(r) for r in cov]
        except TypeError:
            raise NoiseError('GaussianVector(cov=...): give a square matrix (a list of rows).') from None
        n = len(rows)
        if n < 1 or any(len(r) != n for r in rows):
            raise NoiseError(f'GaussianVector(cov=...): cov must be square; got rows of lengths '
                             f'{[len(r) for r in rows]}.')
        C = [[parse_expr(v, what=f'GaussianVector(cov[{k}][{l}])') for l, v in enumerate(r)]
             for k, r in enumerate(rows)]
        for k in range(n):
            for l in range(k):
                if not (SR(C[k][l]) - SR(C[l][k])).simplify_full().is_zero():
                    raise NoiseError(f'GaussianVector(cov=...): cov must be symmetric; cov[{k}][{l}] '
                                     f'= {C[k][l]} but cov[{l}][{k}] = {C[l][k]}.')
        mean = [0] * n if mean is None else list(mean)
        if len(mean) != n:
            raise NoiseError(f'GaussianVector(mean=...): {n} components; got {len(mean)} means.')
        self.n = n
        params = {f'cov_{k + 1}_{l + 1}': rows[k][l] for k in range(n) for l in range(k, n)}
        params.update({f'mean_{k + 1}': m for k, m in enumerate(mean)})
        super().__init__(**params)
        self.cov = [[self.params[f'cov_{min(k, l) + 1}_{max(k, l) + 1}'] for l in range(n)]
                    for k in range(n)]
        self.mean_vector = [self.params[f'mean_{k + 1}'] for k in range(n)]
        num = [[_numeric(c) for c in r] for r in self.cov]
        if all(v is not None for r in num for v in r):
            import numpy as _np
            if _np.linalg.eigvalsh(_np.array(num, dtype=float)).min() < -1e-12 * max(
                    1.0, abs(_np.array(num, dtype=float)).max()):
                raise NoiseError('GaussianVector(cov=...): cov must be positive semidefinite.')

    def _k0_roundoff(self):
        return True                         # (K(0) = 0 by construction)

    def _cgf(self):
        raise NoiseError('GaussianVector has no scalar CGF: its action term is '
                         '-(1/2) theta^T cov theta over its components.')

    def __repr__(self):
        cov = [[expr_to_text(c) for c in r] for r in self.cov]
        return f'GaussianVector(cov={cov!r})'


class Poisson(NoiseSource):
    """Poisson counting process of rate ``rate`` with jump ``size``:
    ``K = rate*(exp(size*theta) - 1)``.  ``rate`` and ``size`` may depend
    on the fields (``rate='a*exp(v)'`` gives an exponential Hawkes unit)."""

    def __init__(self, rate, size=1):
        super().__init__(rate=rate, size=size)

    def _check_domain(self):
        self._require('rate', _nonneg, 'need rate >= 0')

    def _cgf(self):
        return self.params['rate'] * (exp(self.params['size'] * THETA) - 1)


class GammaProcess(NoiseSource):
    """Gamma subordinator: ``K = -shape_rate*log(1 - scale*theta)``;
    cumulant rates ``shape_rate*(m-1)!*scale^m``."""

    def __init__(self, shape_rate, scale):
        super().__init__(shape_rate=shape_rate, scale=scale)

    def _check_domain(self):
        # a negative scale is the reflected (negative-jump) subordinator: allowed
        self._require('shape_rate', _pos, 'need shape_rate > 0')

    def _cgf(self):
        return -self.params['shape_rate'] * log(1 - self.params['scale'] * THETA)


class InverseGaussianProcess(NoiseSource):
    """Inverse-Gaussian subordinator with mean ``mean_rate`` and shape
    ``shape`` per unit time (``X_t ~ IG(mean_rate*t, shape*t^2)``):
    ``K = (shape/mean_rate)*(1 - sqrt(1 - 2*mean_rate^2*theta/shape))``.
    Cumulant rates: ``mean_rate``, ``mean_rate^3/shape``,
    ``3*mean_rate^5/shape^2``, ``15*mean_rate^7/shape^3``, ..."""

    def __init__(self, mean_rate, shape):
        super().__init__(mean_rate=mean_rate, shape=shape)

    def _check_domain(self):
        self._require('mean_rate', _pos, 'need mean_rate > 0')
        self._require('shape', _pos, 'need shape > 0')

    def _cgf(self):
        mu, lam = self.params['mean_rate'], self.params['shape']
        return (lam / mu) * (1 - sqrt(1 - 2 * mu ** 2 * THETA / lam))


class Cumulants(NoiseSource):
    """Escape hatch: a source given by its cumulant rates,
    ``K = mean*theta + sum_{m>=2} kappa_m theta^m / m!``.  ``kappas`` is
    ``[kappa2, kappa3, ...]`` (expressions, may depend on the fields)."""

    def __init__(self, kappas: Sequence, mean=0):
        kappas = _seq('Cumulants', 'kappas', kappas)
        if not kappas:
            raise NoiseError('Cumulants: give at least kappa2.')
        params = {'mean': mean}
        for m, k in enumerate(kappas, start=2):
            params[f'kappa{m}'] = k
        self._order = len(kappas) + 1
        super().__init__(**params)

    def _check_domain(self):
        self._require('kappa2', _nonneg, 'need kappa2 >= 0 (a variance rate)')

    def _cgf(self):
        K = self.params['mean'] * THETA
        for m in range(2, self._order + 1):
            K += self.params[f'kappa{m}'] * THETA ** m / factorial(m)
        return K

    @property
    def declared_order(self) -> int:
        """The highest cumulant order given (``1 + len(kappas)``)."""
        return self._order

    def warn_truncation(self, m: int, where: str = '') -> None:
        """Warn (once per source) that cumulant order ``m`` lies above the
        declared order: the CGF is a polynomial, so that cumulant is 0."""
        if int(m) <= self._order or getattr(self, '_truncation_warned', False):
            return
        self._truncation_warned = True
        from engine._warn_helpers import ensure_warning_shown
        _msg = (f'{self!r}: cumulant order {int(m)} requested{where}, but only '
                f'cumulants up to order {self._order} are declared; the higher '
                f'cumulants are taken as 0. Give more kappas if they are not zero.')
        ensure_warning_shown(CumulantTruncationWarning, 'always', message=_msg,
                             stacklevel=3)
        warnings.warn(_msg, CumulantTruncationWarning, stacklevel=3)

    def cumulant(self, m: int):
        self.warn_truncation(m)
        return super().cumulant(m)


class CGF(NoiseSource):
    """Escape hatch: a source given by its per-unit-time CGF, an expression
    in ``theta`` (and parameters / fields), e.g. ``CGF('lam*(cosh(a*theta)-1)')``.
    It must vanish at ``theta = 0`` and be analytic there."""

    def __init__(self, expression):
        self._K = parse_expr(expression, what='CGF(expression)')
        if THETA not in self._K.variables():
            raise NoiseError('CGF: the expression must depend on theta.')
        # Analyticity first: the generic K(0) check in the base constructor
        # evaluates the expression at 0 and used to let the raw Sage error
        # (e.g. 'division by zero' for CGF('1/theta')) escape unnamed.
        try:
            self._K.subs({THETA: 0})
        except Exception as exc:                       # noqa: BLE001
            raise NoiseError(f'CGF: not analytic at theta = 0 ({exc}).') from exc
        super().__init__()

    def _cgf(self):
        return self._K

    def free_names(self) -> set[str]:
        return expr_names(self._K) - {'theta'}

    def _check_common(self):
        if 't' in expr_names(self._K):
            raise NoiseError('CGF: the expression depends on the time t; only '
                             'time-independent sources are supported.')
        super()._check_common()

    def __repr__(self):
        return f'CGF({expr_to_text(self._K)!r})'


# ── Jump distributions and compound Poisson ────────────────────────────

class JumpDistribution:
    """A jump-size law with a closed-form moment generating function."""
    regular = True

    #: True for laws whose MGF is a truncated moment series (no analytic MGF)
    truncated = False

    def __init__(self, **params):
        self.params = {k: parse_expr(v, what=f'{type(self).__name__}({k}=...)')
                       for k, v in params.items()}
        for k, v in self.params.items():
            # as NoiseSource._check_common: theta in a
            # jump parameter would silently make a different CGF
            if THETA in SR(v).variables():
                raise NoiseError(
                    f'{type(self).__name__}: parameter {k} may not contain '
                    f'theta (reserved for the CGF variable).')
        self._check_domain()

    def _check_domain(self):
        """Parameter-domain checks (numeric values only); default none."""

    def _require(self, name, ok, why):
        _require(type(self).__name__, self.params[name], name, ok, why)

    def sample(self, rng, size, values=None):
        """``size`` i.i.d. draws of the jump size (:mod:`api.noise_samplers`)."""
        from api.noise_samplers import sample_jump
        return sample_jump(self, rng, size, values)

    def mgf(self, theta=None):
        M = self._mgf()
        if theta is not None:
            M = M.subs({THETA: SR(theta)})
        return M

    def _mgf(self):                                    # pragma: no cover
        raise NotImplementedError

    def __repr__(self):
        args = ', '.join(f'{k}={expr_to_text(v)!r}' for k, v in self.params.items())
        return f'{type(self).__name__}({args})'


class Dirac(JumpDistribution):
    """Deterministic jump ``a``: ``M = exp(a*theta)``."""
    def __init__(self, a):
        super().__init__(a=a)

    def _mgf(self):
        return exp(self.params['a'] * THETA)


class Exponential(JumpDistribution):
    """Exponential jump with mean ``mean``: ``M = 1/(1 - mean*theta)``."""
    def __init__(self, mean):
        super().__init__(mean=mean)

    def _mgf(self):
        return 1 / (1 - self.params['mean'] * THETA)


class Gamma(JumpDistribution):
    """Gamma jump (shape ``shape``, scale ``scale``; Erlang for integer
    shape): ``M = (1 - scale*theta)^(-shape)``."""
    def __init__(self, shape, scale):
        super().__init__(shape=shape, scale=scale)

    def _check_domain(self):
        # a negative scale is the reflected (negative) jump: allowed
        self._require('shape', _pos, 'need shape > 0')

    def _mgf(self):
        return (1 - self.params['scale'] * THETA) ** (-self.params['shape'])


class GaussianJump(JumpDistribution):
    """Normal jump ``N(mu, sigma^2)``: ``M = exp(mu*theta + sigma^2*theta^2/2)``."""
    def __init__(self, mu, sigma):
        super().__init__(mu=mu, sigma=sigma)

    def _mgf(self):
        return exp(self.params['mu'] * THETA
                   + self.params['sigma'] ** 2 * THETA ** 2 / 2)


class Laplace(JumpDistribution):
    """Centred Laplace jump with scale ``b``: ``M = 1/(1 - b^2*theta^2)``."""
    def __init__(self, b):
        super().__init__(b=b)

    def _mgf(self):
        return 1 / (1 - self.params['b'] ** 2 * THETA ** 2)


class Uniform(JumpDistribution):
    """Uniform jump on ``[lo, hi]``:
    ``M = (exp(hi*theta) - exp(lo*theta))/((hi - lo)*theta)`` (removable
    singularity at 0, so the action carries its Taylor polynomial)."""
    regular = False

    def __init__(self, lo, hi):
        super().__init__(lo=lo, hi=hi)

    def _check_domain(self):
        lo, hi = _numeric(self.params['lo']), _numeric(self.params['hi'])
        if (lo is not None and hi is not None and lo == hi) or \
                SR(self.params['hi'] - self.params['lo']).is_trivial_zero():
            raise NoiseError(f'Uniform: need lo != hi (got lo = hi = '
                             f'{self.params["lo"]}); a fixed jump is Dirac(a).')

    def _mgf(self):
        lo, hi = self.params['lo'], self.params['hi']
        return (exp(hi * THETA) - exp(lo * THETA)) / ((hi - lo) * THETA)


class Bernoulli(JumpDistribution):
    """Jump 1 with probability ``p`` else 0: ``M = 1 - p + p*exp(theta)``."""
    def __init__(self, p):
        super().__init__(p=p)

    def _check_domain(self):
        self._require('p', lambda x: 0 <= x <= 1, 'need 0 <= p <= 1')

    def _mgf(self):
        p = self.params['p']
        return 1 - p + p * exp(THETA)


class Binomial(JumpDistribution):
    """Binomial jump ``B(n, p)``: ``M = (1 - p + p*exp(theta))^n``."""
    def __init__(self, n, p):
        super().__init__(n=n, p=p)

    def _mgf(self):
        p = self.params['p']
        return (1 - p + p * exp(THETA)) ** self.params['n']


class FinitePMF(JumpDistribution):
    """Jump ``values[j]`` with probability ``probs[j]``:
    ``M = sum_j probs[j]*exp(values[j]*theta)``."""

    def __init__(self, values: Sequence, probs: Sequence):
        values, probs = _seq('FinitePMF', 'values', values), _seq('FinitePMF', 'probs', probs)
        if len(values) != len(probs) or not values:
            raise NoiseError('FinitePMF: values and probs must be non-empty '
                             'and of equal length.')
        params = {}
        for j, (a, p) in enumerate(zip(values, probs)):
            params[f'a{j}'] = a
            params[f'p{j}'] = p
        self._n = len(values)
        super().__init__(**params)
        for j in range(self._n):
            self._require(f'p{j}', _nonneg, f'need probs[{j}] >= 0')
        total = sum(self.params[f'p{j}'] for j in range(self._n))
        x = _numeric(total)
        if x is not None:
            # the same round-off allowance as Hyperexponential: float
            # probabilities that sum to 1 up to 1e-12 are a distribution
            if abs(x - 1) > 1e-12:
                raise NoiseError(f'FinitePMF: probabilities sum to {total}, not 1.')
        elif not total.variables():
            if not (SR(total) - 1).simplify_full().is_zero():
                raise NoiseError(f'FinitePMF: probabilities sum to {total}, not 1.')

    def _mgf(self):
        return sum(self.params[f'p{j}'] * exp(self.params[f'a{j}'] * THETA)
                   for j in range(self._n))


class CompoundPoisson(NoiseSource):
    """Compound Poisson process: events at ``rate`` carry i.i.d. jumps drawn
    from ``jump`` (a :class:`JumpDistribution`):
    ``K = rate*(M_J(theta) - 1)``; cumulant rates ``rate*E[J^m]``."""

    def __init__(self, rate, jump: JumpDistribution):
        if not isinstance(jump, JumpDistribution):
            raise NoiseError(
                f'CompoundPoisson: jump must be a jump distribution (an '
                f'api.noise.JumpDistribution such as Dirac, Exponential, Gamma, '
                f'Geometric, PhaseType, ...), got {type(jump).__name__}.')
        self.jump = jump
        self.regular = jump.regular
        super().__init__(rate=rate)

    def _max_order(self):
        return self.jump.order if self.jump.truncated else None

    def _cgf(self):
        return self.params['rate'] * (self.jump.mgf() - 1)

    def free_names(self) -> set[str]:
        out = super().free_names()
        for v in self.jump.params.values():
            out |= expr_names(v)
        return out

    def _check_common(self):
        for k, v in self.jump.params.items():
            if 't' in expr_names(v):
                raise NoiseError(
                    f'CompoundPoisson: jump parameter {k} depends on the time '
                    f't; only time-independent sources are supported.')
        super()._check_common()

    def _k0_roundoff(self):
        """``K(0) = rate (M(0) - 1)``: round-off when the jump law's
        dimensionless ``M(0)`` is 1 to 1e-12.  (The generic test needs a
        numeric ``K(0)``: float probabilities summing to 1 - 1e-16 times a
        symbolic rate were refused, and the rate's units entered the
        tolerance --.)"""
        try:
            m0 = _numeric(SR(self.jump.mgf()).subs({THETA: 0}))
        except (ValueError, ZeroDivisionError, ArithmeticError, TypeError,
                RuntimeError):
            return False
        return m0 is not None and abs(m0 - 1) <= 1e-12

    def __repr__(self):
        return f'CompoundPoisson(rate={expr_to_text(self.params["rate"])!r}, jump={self.jump!r})'


# ═══ Round 2: finite-moment Levy sources, jump laws, composites ═══
#
# Every class below is a per-unit-time CGF like the rest of the module and
# has an exact (or documented-approximate) sampler in api.noise_samplers.

def _pos(x):
    return x > 0


def _nonneg(x):
    return x >= 0


def _renamed_keyword(who, old, new, old_value, new_value, default=0):
    """The value of a keyword that was renamed ``old -> new``: ``new_value``,
    or ``old_value`` with one ``DeprecationWarning`` (both given: an error)."""
    if old_value is None:
        return default if new_value is None else new_value
    if new_value is not None:
        raise NoiseError(f'{who}: give {new}= only ({old}= is its deprecated name).')
    import warnings
    warnings.warn(f'{who}({old}=) is renamed {new}= (the location parameter); {old}= is '
                  f'accepted for one more release', DeprecationWarning, stacklevel=3)
    return old_value


class VarianceGamma(NoiseSource):
    """Variance-gamma process (Brownian motion with drift ``mu`` and volatility
    ``sigma`` run on a gamma clock of mean rate 1 and variance rate ``nu``),
    plus an optional linear location ``loc`` (``drift=`` is its deprecated name):
    ``K = loc*theta - log(1 - nu*mu*theta - nu*sigma^2*theta^2/2)/nu``.
    Cumulant rates: ``kappa_1 = loc + mu``, ``kappa_2 = sigma^2 + nu*mu^2``."""

    def __init__(self, mu, sigma, nu, loc=None, *, drift=None):
        super().__init__(mu=mu, sigma=sigma, nu=nu,
                         loc=_renamed_keyword('VarianceGamma', 'drift', 'loc', drift, loc))

    def _check_domain(self):
        self._require('sigma', _pos, 'need sigma > 0')
        self._require('nu', _pos, 'need nu > 0 (variance rate of the gamma clock)')

    def _cgf(self):
        P = self.params
        return (P['loc'] * THETA
                - log(1 - P['nu'] * P['mu'] * THETA
                      - P['nu'] * P['sigma'] ** 2 * THETA ** 2 / 2) / P['nu'])


class NormalInverseGaussian(NoiseSource):
    """Normal-inverse-Gaussian process ``NIG(alpha, beta, delta, loc)`` per unit
    time: ``K = loc*theta + delta*(sqrt(alpha^2 - beta^2) -
    sqrt(alpha^2 - (beta + theta)^2))`` (need ``alpha > |beta|``, ``delta > 0``;
    ``mu=`` is the deprecated name of ``loc=``)."""

    def __init__(self, alpha, beta, delta, loc=None, *, mu=None):
        super().__init__(alpha=alpha, beta=beta, delta=delta,
                         loc=_renamed_keyword('NormalInverseGaussian', 'mu', 'loc', mu, loc))

    def _check_domain(self):
        self._require('alpha', _pos, 'need alpha > 0')
        self._require('delta', _pos, 'need delta > 0')
        a, b = _numeric(self.params['alpha']), _numeric(self.params['beta'])
        if a is not None and b is not None and not abs(b) < a:
            raise NoiseError(f'NormalInverseGaussian: need |beta| < alpha '
                             f'(alpha={a:g}, beta={b:g}).')

    def _cgf(self):
        # written with the one square root g = sqrt(alpha^2 - beta^2) and a
        # series in theta around 1: K(0) is then 0 identically and every
        # derivative carries the same branch of g.  (The form
        # sqrt(a^2 - b^2) - sqrt(a^2 - (b + theta)^2) was expanded by Maxima
        # into I*sqrt(b^2 - a^2) for the second root when alpha, beta are
        # symbols, a nonzero K(0) the mean-field check refused.)
        P = self.params
        a, b = P['alpha'], P['beta']
        g2 = a ** 2 - b ** 2
        return P['loc'] * THETA + P['delta'] * sqrt(g2) * (
            1 - sqrt(1 - (2 * b * THETA + THETA ** 2) / g2))


class CGMY(NoiseSource):
    """CGMY (tempered-stable, Koponen-Boyarchenko-Levendorskii) process with
    Levy density ``C*exp(-M x)/x^(1+Y)`` for ``x > 0`` and
    ``C*exp(-G|x|)/|x|^(1+Y)`` for ``x < 0``:
    ``K = C*Gamma(-Y)*((M - theta)^Y - M^Y + (G + theta)^Y - G^Y)``.
    For ``1 < Y < 2`` this is the compensated process (mean
    ``C*Gamma(1-Y)*(M^(Y-1) - G^(Y-1))``).  ``Y`` may not be 0 (variance
    gamma / bilateral gamma) or 1 (the limit has a different form); need
    ``C, G, M > 0`` and ``Y < 2``.  Cumulant rates
    ``kappa_m = C*Gamma(m - Y)*(M^(Y-m) + (-1)^m G^(Y-m))``."""

    def __init__(self, C, G, M, Y):
        super().__init__(C=C, G=G, M=M, Y=Y)

    def _check_domain(self):
        for n in ('C', 'G', 'M'):
            self._require(n, _pos, f'need {n} > 0')
        y = _numeric(self.params['Y'])
        if y is not None and (y >= 2 or y in (0.0, 1.0)):
            raise NoiseError(
                f'CGMY: need Y < 2 and Y not in {{0, 1}} (got Y={y:g}); Y=0 is the '
                f'variance-gamma / BilateralGamma limit.')

    def _cgf(self):
        P = self.params
        C, G, M, Y = P['C'], P['G'], P['M'], P['Y']
        return C * _gamma(-Y) * ((M - THETA) ** Y - M ** Y + (G + THETA) ** Y - G ** Y)


class TemperedStableSubordinator(NoiseSource):
    """Tempered-stable subordinator (one-sided CGMY, ``0 < alpha < 1``): Levy
    density ``c*exp(-lam x)/x^(1+alpha)``,
    ``K = c*Gamma(-alpha)*((lam - theta)^alpha - lam^alpha)``;
    ``kappa_m = c*Gamma(m - alpha)*lam^(alpha - m)``."""

    def __init__(self, alpha, c, lam):
        super().__init__(alpha=alpha, c=c, lam=lam)

    def _check_domain(self):
        self._require('alpha', lambda x: 0 < x < 1, 'need 0 < alpha < 1')
        self._require('c', _pos, 'need c > 0')
        self._require('lam', _pos, 'need lam > 0')

    def _cgf(self):
        P = self.params
        al, c, lam = P['alpha'], P['c'], P['lam']
        return c * _gamma(-al) * ((lam - THETA) ** al - lam ** al)


class Meixner(NoiseSource):
    """Meixner process ``Meixner(alpha, beta, delta, loc)`` per unit time:
    ``K = loc*theta + 2*delta*(log cos(beta/2) - log cos((alpha*theta + beta)/2))``
    (``alpha > 0``, ``|beta| < pi``, ``delta > 0``); ``kappa_1 = loc +
    alpha*delta*tan(beta/2)``, ``kappa_2 = alpha^2*delta/(2*cos^2(beta/2))``
    (``mu=`` is the deprecated name of ``loc=``)."""

    def __init__(self, alpha, beta, delta, loc=None, *, mu=None):
        super().__init__(alpha=alpha, beta=beta, delta=delta,
                         loc=_renamed_keyword('Meixner', 'mu', 'loc', mu, loc))

    def _check_domain(self):
        from math import pi
        self._require('alpha', _pos, 'need alpha > 0')
        self._require('delta', _pos, 'need delta > 0')
        self._require('beta', lambda x: abs(x) < pi, 'need |beta| < pi')

    def _cgf(self):
        P = self.params
        a, b, d = P['alpha'], P['beta'], P['delta']
        return P['loc'] * THETA + 2 * d * (log(_cos(b / 2)) - log(_cos((a * THETA + b) / 2)))


class BilateralGamma(NoiseSource):
    """Bilateral gamma process (a gamma process minus an independent one):
    ``K = -alpha_p*log(1 - theta/lam_p) - alpha_m*log(1 + theta/lam_m)``;
    ``kappa_m = (m-1)!*(alpha_p/lam_p^m + (-1)^m alpha_m/lam_m^m)``."""

    def __init__(self, alpha_p, lam_p, alpha_m, lam_m):
        super().__init__(alpha_p=alpha_p, lam_p=lam_p, alpha_m=alpha_m, lam_m=lam_m)

    def _check_domain(self):
        for n in self.params:
            self._require(n, _pos, f'need {n} > 0')

    def _cgf(self):
        P = self.params
        return (-P['alpha_p'] * log(1 - THETA / P['lam_p'])
                - P['alpha_m'] * log(1 + THETA / P['lam_m']))


class Skellam(NoiseSource):
    """Skellam process (difference of two independent Poisson processes):
    ``K = rate_plus*(exp(theta) - 1) + rate_minus*(exp(-theta) - 1)``."""

    def __init__(self, rate_plus, rate_minus):
        super().__init__(rate_plus=rate_plus, rate_minus=rate_minus)

    def _check_domain(self):
        self._require('rate_plus', _nonneg, 'need rate_plus >= 0')
        self._require('rate_minus', _nonneg, 'need rate_minus >= 0')

    def _cgf(self):
        P = self.params
        return (P['rate_plus'] * (exp(THETA) - 1)
                + P['rate_minus'] * (exp(-THETA) - 1))


class NegativeBinomialProcess(NoiseSource):
    """Negative-binomial (Polya) process: ``N_t ~ NB(shape_rate*t, p)`` counting
    failures, ``K = shape_rate*(log(1 - p) - log(1 - p*exp(theta)))`` (a
    compound Poisson process with logarithmic jumps); mean
    ``shape_rate*p/(1 - p)``."""

    def __init__(self, shape_rate, p):
        super().__init__(shape_rate=shape_rate, p=p)

    def _check_domain(self):
        self._require('shape_rate', _pos, 'need shape_rate > 0')
        self._require('p', lambda x: 0 < x < 1, 'need 0 < p < 1')

    def _cgf(self):
        P = self.params
        return P['shape_rate'] * (log(1 - P['p']) - log(1 - P['p'] * exp(THETA)))


class SumSource(NoiseSource):
    """Independent sum of sources: ``K = sum_a K_a`` (``a + b`` on sources
    builds one).  Used for jump-diffusions: ``Gaussian + CompoundPoisson``.
    Always handled through the general CGF path of the SDE builder (a
    Gaussian part is not moved into the Stratonovich conversion)."""

    def __init__(self, *parts):
        flat = []
        for q in parts:
            flat.extend(q.parts if isinstance(q, SumSource) else [q])
        if len(flat) < 2 or not all(isinstance(q, NoiseSource) for q in flat):
            raise NoiseError('SumSource: give at least two noise sources.')
        self.parts = flat
        self.regular = all(q.regular for q in flat)
        super().__init__()

    def _cgf(self):
        return sum(q._cgf() for q in self.parts)

    def _max_order(self):
        tops = [t for t in (q._max_order() for q in self.parts) if t is not None]
        return min(tops) if tops else None

    def free_names(self) -> set[str]:
        out = set()
        for q in self.parts:
            out |= q.free_names()
        return out

    def __repr__(self):
        return ' + '.join(repr(q) for q in self.parts)


NoiseSource.__add__ = lambda self, other: SumSource(self, other)


# ── more jump laws ─────────────────────────────────────────────────────

class Geometric(JumpDistribution):
    """Geometric jump on ``{1, 2, ...}`` (number of trials), success
    probability ``p``: ``M = p*exp(theta)/(1 - (1-p)*exp(theta))``."""

    def __init__(self, p):
        super().__init__(p=p)

    def _check_domain(self):
        self._require('p', lambda x: 0 < x <= 1, 'need 0 < p <= 1')

    def _mgf(self):
        p = self.params['p']
        return p * exp(THETA) / (1 - (1 - p) * exp(THETA))


class PoissonJump(JumpDistribution):
    """Poisson-distributed jump (a count) with mean ``mean``:
    ``M = exp(mean*(exp(theta) - 1))``."""

    def __init__(self, mean):
        super().__init__(mean=mean)

    def _check_domain(self):
        self._require('mean', _nonneg, 'need mean >= 0')

    def _mgf(self):
        return exp(self.params['mean'] * (exp(THETA) - 1))


class NegativeBinomial(JumpDistribution):
    """Negative-binomial jump on ``{0, 1, ...}`` (failures before ``r``
    successes; failure probability ``p``): ``M = ((1 - p)/(1 - p*exp(theta)))^r``."""

    def __init__(self, r, p):
        super().__init__(r=r, p=p)

    def _check_domain(self):
        self._require('r', _pos, 'need r > 0')
        self._require('p', lambda x: 0 < x < 1, 'need 0 < p < 1')

    def _mgf(self):
        r, p = self.params['r'], self.params['p']
        return ((1 - p) / (1 - p * exp(THETA))) ** r


class Logarithmic(JumpDistribution):
    """Logarithmic (log-series) jump on ``{1, 2, ...}``,
    ``P(k) = -p^k/(k*log(1-p))``: ``M = log(1 - p*exp(theta))/log(1 - p)``."""

    def __init__(self, p):
        super().__init__(p=p)

    def _check_domain(self):
        self._require('p', lambda x: 0 < x < 1, 'need 0 < p < 1')

    def _mgf(self):
        p = self.params['p']
        return log(1 - p * exp(THETA)) / log(1 - p)


class DoubleExponential(JumpDistribution):
    """Kou's double-exponential jump: ``+Exp(rate eta_up)`` with probability
    ``p``, ``-Exp(rate eta_down)`` otherwise:
    ``M = p*eta_up/(eta_up - theta) + (1-p)*eta_down/(eta_down + theta)``."""

    def __init__(self, p, eta_up, eta_down):
        super().__init__(p=p, eta_up=eta_up, eta_down=eta_down)

    def _check_domain(self):
        self._require('p', lambda x: 0 <= x <= 1, 'need 0 <= p <= 1')
        self._require('eta_up', _pos, 'need eta_up > 0')
        self._require('eta_down', _pos, 'need eta_down > 0')

    def _mgf(self):
        P = self.params
        return (P['p'] * P['eta_up'] / (P['eta_up'] - THETA)
                + (1 - P['p']) * P['eta_down'] / (P['eta_down'] + THETA))


class AsymmetricLaplace(JumpDistribution):
    """Asymmetric Laplace jump ``loc + scale_up*E1 - scale_down*E2`` (``E1``,
    ``E2`` i.i.d. unit exponentials):
    ``M = exp(loc*theta)/((1 - scale_up*theta)*(1 + scale_down*theta))``."""

    def __init__(self, scale_up, scale_down, loc=0):
        super().__init__(scale_up=scale_up, scale_down=scale_down, loc=loc)

    def _check_domain(self):
        self._require('scale_up', _pos, 'need scale_up > 0')
        self._require('scale_down', _pos, 'need scale_down > 0')

    def _mgf(self):
        P = self.params
        return exp(P['loc'] * THETA) / ((1 - P['scale_up'] * THETA)
                                         * (1 + P['scale_down'] * THETA))


class Hyperexponential(JumpDistribution):
    """Mixture of exponentials: mean ``means[j]`` with probability
    ``probs[j]``: ``M = sum_j probs[j]/(1 - means[j]*theta)``."""

    def __init__(self, probs: Sequence, means: Sequence):
        probs = _seq('Hyperexponential', 'probs', probs)
        means = _seq('Hyperexponential', 'means', means)
        if len(probs) != len(means) or not probs:
            raise NoiseError('Hyperexponential: probs and means must be non-empty '
                             'and of equal length.')
        self._n = len(probs)
        params = {}
        for j, (q, m) in enumerate(zip(probs, means)):
            params[f'p{j}'] = q
            params[f'm{j}'] = m
        super().__init__(**params)

    def _check_domain(self):
        tot = 0.0
        numeric = True
        for j in range(self._n):
            self._require(f'p{j}', _nonneg, f'need probs[{j}] >= 0')
            self._require(f'm{j}', _pos, f'need means[{j}] > 0')
            x = _numeric(self.params[f'p{j}'])
            if x is None:
                numeric = False
            else:
                tot += x
        if numeric and abs(tot - 1) > 1e-12:
            raise NoiseError(f'Hyperexponential: probabilities sum to {tot:g}, not 1.')

    def _mgf(self):
        return sum(self.params[f'p{j}'] / (1 - self.params[f'm{j}'] * THETA)
                   for j in range(self._n))


class PhaseType(JumpDistribution):
    """Phase-type jump ``PH(alpha, T)``: absorption time of a finite Markov
    chain with sub-generator ``T`` (``n x n``) started in phase ``j`` with
    probability ``alpha[j]`` (an atom ``1 - sum(alpha)`` at 0 if the sum is
    below 1).  The MGF is rational,
    ``M = 1 - sum(alpha) + alpha (-theta I - T)^(-1) t``, ``t = -T 1``;
    it covers Erlang, hyperexponential and Coxian laws.  Moments
    ``E[X^m] = m! alpha (-T)^(-m) 1``."""

    def __init__(self, alpha: Sequence, T: Sequence):
        alpha = _seq('PhaseType', 'alpha', alpha)
        T = [_seq('PhaseType', 'T rows', r) for r in _seq('PhaseType', 'T', T)]
        n = len(alpha)
        if n == 0 or len(T) != n or any(len(r) != n for r in T):
            raise NoiseError('PhaseType: alpha must have length n and T must be n x n.')
        self._n = n
        params = {f'a{i}': alpha[i] for i in range(n)}
        for i in range(n):
            for j in range(n):
                params[f'T{i}_{j}'] = T[i][j]
        super().__init__(**params)

    def _matrix(self):
        n = self._n
        return _matrix(SR, n, n, [self.params[f'T{i}_{j}']
                                  for i in range(n) for j in range(n)])

    def _check_domain(self):
        n = self._n
        a = [_numeric(self.params[f'a{i}']) for i in range(n)]
        if any(x is not None and x < 0 for x in a):
            raise NoiseError('PhaseType: alpha entries must be >= 0.')
        if all(x is not None for x in a) and sum(a) > 1 + 1e-12:
            raise NoiseError(f'PhaseType: sum(alpha) = {sum(a):g} exceeds 1.')
        for i in range(n):
            row = [_numeric(self.params[f'T{i}_{j}']) for j in range(n)]
            if any(x is None for x in row):
                continue
            if row[i] >= 0:
                raise NoiseError(f'PhaseType: T[{i}][{i}] must be negative.')
            if any(row[j] < 0 for j in range(n) if j != i):
                raise NoiseError(f'PhaseType: off-diagonal entries of T must be >= 0 (row {i}).')
            # round-off relative to the row (an absolute 1e-12 refused valid
            # rows of rates ~1e6 whose float sum is +1e-10)
            if sum(row) > 1e-12 * sum(abs(x) for x in row):
                raise NoiseError(f'PhaseType: row {i} of T sums to {sum(row):g} > 0 '
                                 f'(rows of a sub-generator sum to <= 0).')
        Tn = [[_numeric(self.params[f'T{i}_{j}']) for j in range(n)] for i in range(n)]
        if all(x is not None for r in Tn for x in r):
            import numpy as _np
            Tm = _np.array(Tn)
            # relative to the matrix's own scale (the floor max(1, |T|)^n
            # refused every sub-generator with rates below ~1e-4)
            if abs(_np.linalg.det(Tm)) <= 1e-12 * float(_np.abs(Tm).max()) ** n:
                raise NoiseError('PhaseType: T is singular, so the chain never absorbs '
                                 '(a sub-generator must be invertible).')

    def _mgf(self):
        # M = 1 - sum(alpha) + alpha (-theta I - T)^(-1) t, t = -T 1, written
        # as 1 + theta alpha (-theta I - T)^(-1) 1 (the same function): exactly
        # 1 at theta = 0 whatever the float round-off of alpha and T (a
        # residual 2e-16 * rate was refused as K(0) != 0), and formed from the
        # adjugate and the determinant (Maxima's rational simplification
        # overflowed on float rates ~1e6).
        n = self._n
        T = self._matrix()
        al = _matrix(SR, 1, n, [self.params[f'a{i}'] for i in range(n)])
        A = -THETA * _matrix(SR, n, n, lambda i, j: 1 if i == j else 0) - T
        num = SR((al * A.adjugate() * _matrix(SR, n, 1, [1] * n))[0, 0]).expand()
        den = SR(A.det()).expand()
        return 1 + THETA * num / den


class ChiSquared(Gamma):
    """Chi-squared jump with ``df`` degrees of freedom; exactly
    ``Gamma(shape=df/2, scale=2)`` (analytic MGF ``(1 - 2*theta)^(-df/2)``)."""

    def __init__(self, df):
        JumpDistribution.__init__(self, shape=SR(parse_expr(df)) / 2, scale=2)
        self.df = parse_expr(df)

    def _check_domain(self):
        self._require('shape', _pos, 'need df > 0')


class _SeriesJump(JumpDistribution):
    """A jump law without an analytic MGF, carried by its raw moments to a
    requested ``order``: ``M(theta) = sum_{m<=order} E[J^m] theta^m/m!`` is
    a TRUNCATED SERIES, so a source built from it has exact cumulants only up
    to ``order`` and refuses any use that needs more (cumulant, series
    order of the action).  ``order`` must cover every cumulant order the
    computation will use (``k`` plus loop corrections)."""
    truncated = True

    def __init__(self, order=8, **params):
        order = int(order)
        if order < 1:
            raise NoiseError(f'{type(self).__name__}: order must be >= 1.')
        self.order = order
        super().__init__(**params)

    def moment(self, m: int):                          # pragma: no cover
        raise NotImplementedError

    def _mgf(self):
        return sum(self.moment(m) * THETA ** m / factorial(m)
                   for m in range(self.order + 1))


class LogNormal(_SeriesJump):
    """Log-normal jump ``exp(N(mu, sigma^2))`` (no MGF): raw moments
    ``exp(m*mu + m^2*sigma^2/2)`` to ``order`` (truncated series)."""

    def __init__(self, mu, sigma, order=8):
        super().__init__(order, mu=mu, sigma=sigma)

    def _check_domain(self):
        self._require('sigma', _pos, 'need sigma > 0')

    def moment(self, m):
        return exp(m * self.params['mu'] + m ** 2 * self.params['sigma'] ** 2 / 2)


class Weibull(_SeriesJump):
    """Weibull jump (``scale``, ``shape``; no MGF for ``shape < 1``): raw
    moments ``scale^m*Gamma(1 + m/shape)`` to ``order`` (truncated series)."""

    def __init__(self, scale, shape, order=8):
        super().__init__(order, scale=scale, shape=shape)

    def _check_domain(self):
        self._require('scale', _pos, 'need scale > 0')
        self._require('shape', _pos, 'need shape > 0')

    def moment(self, m):
        return self.params['scale'] ** m * _gamma(1 + SR(m) / self.params['shape'])


#: every noise-source class (config files / SDE accept these by name)
SOURCE_TYPES = (
    'Gaussian', 'GaussianVector', 'Poisson', 'CompoundPoisson', 'GammaProcess',
    'InverseGaussianProcess', 'Cumulants', 'CGF', 'VarianceGamma',
    'NormalInverseGaussian', 'CGMY', 'TemperedStableSubordinator', 'Meixner',
    'BilateralGamma', 'Skellam', 'NegativeBinomialProcess')


# ── named composites (thin constructors over existing pieces) ──────────

def merton_jump_diffusion(drift, sigma, rate, jump_mean, jump_sd):
    """Merton jump-diffusion increments: a Brownian part (``drift``,
    variance ``sigma^2``) plus compound Poisson with Gaussian jumps
    ``N(jump_mean, jump_sd^2)``."""
    return (Gaussian(var=f'({sigma})^2', mean=drift)
            + CompoundPoisson(rate, GaussianJump(jump_mean, jump_sd)))


def kou_jump_diffusion(drift, sigma, rate, p, eta_up, eta_down):
    """Kou jump-diffusion increments: Brownian part plus compound Poisson
    with double-exponential jumps."""
    return (Gaussian(var=f'({sigma})^2', mean=drift)
            + CompoundPoisson(rate, DoubleExponential(p, eta_up, eta_down)))


def levy_driven_ou(source, rate='mu', *, field='x', name='Levy-driven OU',
                   rate_default=1.0):
    """Barndorff-Nielsen-Shephard Levy-driven OU, ``d x = -rate*x dt + d L``
    with ``L`` the Levy process ``source`` (an unbuilt :class:`api.sde.SDE`;
    add parameters of ``source`` with ``.parameter``).  Stationary
    cumulants of ``x`` are ``kappa_n(source)/(n*rate)``."""
    from api.sde import SDE
    s = SDE(name)
    s.parameter(rate, default=rate_default, domain='positive')
    s.field(field)
    s.drift(field, f'-{rate}*{field}')
    s.noise('L', source, into={field: 1})
    return s

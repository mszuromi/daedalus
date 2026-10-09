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
from math import factorial
from typing import Iterable, Optional, Sequence

from sage.all import SR, Integer, exp, log, sqrt, function, sage_eval

__all__ = [
    'NoiseError', 'NoiseSource',
    'Gaussian', 'Poisson', 'CompoundPoisson', 'GammaProcess',
    'InverseGaussianProcess', 'Cumulants', 'CGF',
    'JumpDistribution', 'Dirac', 'Exponential', 'Gamma', 'GaussianJump',
    'Laplace', 'Uniform', 'Bernoulli', 'Binomial', 'FinitePMF',
    'THETA', 'parse_expr', 'expr_to_text', 'expr_names',
]


class NoiseError(ValueError):
    """Raised for an invalid noise declaration."""


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
_RESERVED = {'sum', 'for', 'in', 'pi', 'I', 'e', 'Dt', 'theta'} | set(_MATH_FUNCS)

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


def parse_expr(value, *, what: str = 'expression'):
    """Parse a parameter value (str / number / SR) into a Sage expression.

    Indexed names become flat symbols (``x[i]`` -> ``x__i``); calls of
    unknown names become formal Sage functions (``phi[i](v[i])`` ->
    ``phi__i(v__i)``).  Population sums (``sum(... for j in E)``) are
    not supported inside noise parameters.
    """
    if isinstance(value, str):
        txt = ' '.join(value.split())
        if not txt:
            raise NoiseError(f'{what}: empty expression')
        if re.search(r'\bfor\b', txt):
            raise NoiseError(
                f'{what}: population sums are not supported inside a noise '
                f'parameter ({value!r}); declare a parameter or function '
                f'for the summed quantity instead.')
        txt = _flatten_indices(txt)
        called = set(re.findall(r'\b([A-Za-z_]\w*)\s*\(', txt))
        names = set(re.findall(r'\b([A-Za-z_]\w*)\b', txt))
        loc = {}
        for n in names:
            if n in _MATH_FUNCS:
                loc[n] = _MATH_FUNCS[n]
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

    def __init__(self, **params):
        self.params = {k: parse_expr(v, what=f'{type(self).__name__}({k}=...)')
                       for k, v in params.items()}
        self._check_common()

    # -- the primitive -----------------------------------------------------
    def _cgf(self):                                    # pragma: no cover
        raise NotImplementedError

    def cgf(self, theta=None):
        """``K(theta)`` as a Sage expression (default variable :data:`THETA`)."""
        K = self._cgf()
        if theta is not None:
            K = K.subs({THETA: SR(theta)})
        return K

    def cumulant(self, m: int):
        """``d^m K / d theta^m`` at 0: the ``m``-th cumulant per unit time."""
        m = int(m)
        if m < 1:
            raise NoiseError('cumulant order must be >= 1')
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
        return sum(self.cumulant(m) * THETA ** m / factorial(m)
                   for m in range(1, int(order) + 1))

    def action_cgf(self, order: int = 8):
        """The CGF as written into the action: the closed form when it is
        regular at 0, else the Taylor polynomial to ``order``."""
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
        K0 = self._value_at_zero()
        if not SR(K0).simplify_full().is_zero():
            raise NoiseError(
                f'{type(self).__name__}: K(0) = {K0} != 0; a per-unit-time '
                f'CGF must vanish at theta = 0.')

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
    """
    is_gaussian = True

    def __init__(self, var, mean=0):
        super().__init__(var=var, mean=mean)

    def _cgf(self):
        return self.params['mean'] * THETA + self.params['var'] * THETA ** 2 / 2


class Poisson(NoiseSource):
    """Poisson counting process of rate ``rate`` with jump ``size``:
    ``K = rate*(exp(size*theta) - 1)``.  ``rate`` and ``size`` may depend
    on the fields (``rate='a*exp(v)'`` gives an exponential Hawkes unit)."""

    def __init__(self, rate, size=1):
        super().__init__(rate=rate, size=size)

    def _cgf(self):
        return self.params['rate'] * (exp(self.params['size'] * THETA) - 1)


class GammaProcess(NoiseSource):
    """Gamma subordinator: ``K = -shape_rate*log(1 - scale*theta)``;
    cumulant rates ``shape_rate*(m-1)!*scale^m``."""

    def __init__(self, shape_rate, scale):
        super().__init__(shape_rate=shape_rate, scale=scale)

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

    def _cgf(self):
        mu, lam = self.params['mean_rate'], self.params['shape']
        return (lam / mu) * (1 - sqrt(1 - 2 * mu ** 2 * THETA / lam))


class Cumulants(NoiseSource):
    """Escape hatch: a source given by its cumulant rates,
    ``K = mean*theta + sum_{m>=2} kappa_m theta^m / m!``.  ``kappas`` is
    ``[kappa2, kappa3, ...]`` (expressions, may depend on the fields)."""

    def __init__(self, kappas: Sequence, mean=0):
        kappas = list(kappas)
        if not kappas:
            raise NoiseError('Cumulants: give at least kappa2.')
        params = {'mean': mean}
        for m, k in enumerate(kappas, start=2):
            params[f'kappa{m}'] = k
        self._order = len(kappas) + 1
        super().__init__(**params)

    def _cgf(self):
        K = self.params['mean'] * THETA
        for m in range(2, self._order + 1):
            K += self.params[f'kappa{m}'] * THETA ** m / factorial(m)
        return K


class CGF(NoiseSource):
    """Escape hatch: a source given by its per-unit-time CGF, an expression
    in ``theta`` (and parameters / fields), e.g. ``CGF('lam*(cosh(a*theta)-1)')``.
    It must vanish at ``theta = 0`` and be analytic there."""

    def __init__(self, expression):
        self._K = parse_expr(expression, what='CGF(expression)')
        if THETA not in self._K.variables():
            raise NoiseError('CGF: the expression must depend on theta.')
        super().__init__()
        try:
            self._K.subs({THETA: 0})
        except Exception as exc:                       # noqa: BLE001
            raise NoiseError(f'CGF: not analytic at theta = 0 ({exc}).') from exc

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

    def __init__(self, **params):
        self.params = {k: parse_expr(v, what=f'{type(self).__name__}({k}=...)')
                       for k, v in params.items()}

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

    def _mgf(self):
        lo, hi = self.params['lo'], self.params['hi']
        return (exp(hi * THETA) - exp(lo * THETA)) / ((hi - lo) * THETA)


class Bernoulli(JumpDistribution):
    """Jump 1 with probability ``p`` else 0: ``M = 1 - p + p*exp(theta)``."""
    def __init__(self, p):
        super().__init__(p=p)

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
        values, probs = list(values), list(probs)
        if len(values) != len(probs) or not values:
            raise NoiseError('FinitePMF: values and probs must be non-empty '
                             'and of equal length.')
        params = {}
        for j, (a, p) in enumerate(zip(values, probs)):
            params[f'a{j}'] = a
            params[f'p{j}'] = p
        self._n = len(values)
        super().__init__(**params)
        total = sum(self.params[f'p{j}'] for j in range(self._n))
        if total.is_numeric() or not total.variables():
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
                f'CompoundPoisson: jump must be a jump distribution (Dirac, '
                f'Exponential, Gamma, GaussianJump, Laplace, Uniform, '
                f'Bernoulli, Binomial, FinitePMF), got {type(jump).__name__}.')
        self.jump = jump
        self.regular = jump.regular
        super().__init__(rate=rate)

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

    def __repr__(self):
        return f'CompoundPoisson(rate={expr_to_text(self.params["rate"])!r}, jump={self.jump!r})'

"""
field_theory_sage.py
====================
Model-agnostic MSRJD field theory expansion framework — SageMath version.

The action is specified as a callable returning an SR expression. All field
variables are SR symbolic variables. The framework automatically Taylor-expands
every nonlinear function of the fields (exp, tanh, formal functions, ...) to
the requested order, renames formal-function derivative symbols to clean names,
applies MF background conditions, then classifies terms by bigrade (n_tilde, n_phys)
using PolynomialRing(SR, ...) exponent vectors.

Usage
-----
    load('field_theory_sage.py')
    load('daedalus/simulations/hawkes_sage.py')

    ft = FieldTheory(HAWKES_MODEL, taylor_order=4)
    ft.expand()
    ft.sanity_check()
    ft.summary()
"""

import sys
import warnings

from sage.all import (
    SR, PolynomialRing, factorial, QQ, latex,
    diff, function, exp, dirac_delta, heaviside, integrate, oo, I, pi, taylor
)
from daedalus.engine.errors import DaedalusError


def fourier_transform(f, t, s):
    r"""
    Symbolic Fourier transform (angular-frequency convention):

        F(s) = \int_{-\infty}^{\infty} f(t) e^{-i s t} dt

    No 2π in the exponent.  Gives  δ(t) → 1,  δ'(t) → iω.

    For causal integrands of the form ``g(t) * heaviside(t)`` — which
    is virtually every neural-style kernel — we replace ``heaviside(t)``
    with ``1`` and restrict the integration to ``[0, ∞)``.  This avoids
    Maxima's request for an explicit sign on ``omega`` (it can't decide
    where the heaviside argument cuts unless told).  For non-causal
    integrands the full real line is used.

    Tries the SymPy backend first (it handles ``positive=True``
    assumptions on time constants automatically) and falls back to
    Maxima's default integrator.
    """
    f = SR(f)

    # Detect the causal case: replace heaviside(t) with 1 and split
    # the integration domain at zero.  Bounds are wrapped in SR() so
    # the SymPy integrator (which calls ``a._sympy_()``) accepts them.
    has_heaviside_t = bool(f.has(heaviside(t)))
    if has_heaviside_t:
        integrand = f.subs({heaviside(t): 1}) * exp(-I * s * t)
        a, b = SR(0), oo
    else:
        integrand = f * exp(-I * s * t)
        a, b = -oo, oo

    # SymPy first (better at handling symbolic-positive parameters);
    # then Maxima.  If both fail, return the unevaluated integral.
    for algo in ('sympy', 'maxima'):
        try:
            return integrate(integrand, t, a, b, algorithm=algo)
        except (ValueError, RuntimeError, TypeError):
            continue
    return integrate(integrand, t, a, b)


def _show(expr):
    """Display a SageMath expression or Conv/IP object as rendered LaTeX."""
    from IPython.display import display, Math as _Math   # on use
    display(_Math(latex(expr)))


# ---------------------------------------------------------------------------
# Display helpers — Conv and IP with _latex_() for show()/display()
# ---------------------------------------------------------------------------

class _ConvIPBase:
    def __add__(self, other):
        return _DisplaySum([self, other])
    def __radd__(self, other):
        return _DisplaySum([self]) if other == 0 else _DisplaySum([other, self])
    def __sub__(self, other):
        return _DisplaySum([self, _Neg(other)])
    def __neg__(self):
        return _Neg(self)


class Conv(_ConvIPBase):
    """Conv(kappa, f)  — convolution (κ∗f)(t)."""
    def __init__(self, kappa, f):
        self.kappa = kappa
        self.f     = f
    def _latex_(self):
        kappa_str = latex(self.kappa)
        # Wrap sum kernels in parens so (κ₁+κ₂)∗f reads unambiguously.
        # SageMath may return operator.add or its own add_vararg; check __name__.
        # Fallback: if latex contains '+' the kernel is a sum.
        is_sum = False
        try:
            op = self.kappa.operator()
            is_sum = op is not None and 'add' in getattr(op, '__name__', '')
        except AttributeError:
            pass
        if not is_sum:
            is_sum = ('+' in kappa_str)
        if is_sum:
            kappa_str = r'\left(' + kappa_str + r'\right)'
        return r'\left(' + kappa_str + r' \ast ' + latex(self.f) + r'\right)'
    def __repr__(self):
        return f'Conv({self.kappa!r}, {self.f!r})'


class IP(_ConvIPBase):
    """IP(a, b)  — inner product ∫ a(t) b(t) dt."""
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def _latex_(self):
        return latex(self.a) + r'^\top ' + latex(self.b)
    def __repr__(self):
        return f'IP({self.a!r}, {self.b!r})'


class _Neg(_ConvIPBase):
    def __init__(self, inner):
        self.inner = inner
    def _latex_(self):
        return r'-' + latex(self.inner)
    def __repr__(self):
        return f'-{self.inner!r}'


class _DisplaySum:
    def __init__(self, terms):
        self.terms = list(terms)
    def __add__(self, other):
        return _DisplaySum(self.terms + (other.terms if isinstance(other, _DisplaySum) else [other]))
    def __radd__(self, other):
        return self if other == 0 else _DisplaySum([other] + self.terms)
    def __sub__(self, other):
        return _DisplaySum(self.terms + [_Neg(other)])
    def _latex_(self):
        parts = []
        for t in self.terms:
            s = latex(t)
            if parts and not s.startswith('-'):
                parts.append('+')
            parts.append(s)
        return ' '.join(parts)
    def __repr__(self):
        return ' + '.join(repr(t) for t in self.terms)


# ---------------------------------------------------------------------------
# Namespace object
# ---------------------------------------------------------------------------

class _Namespace:
    pass


# ---------------------------------------------------------------------------
# SR polynomial helper  (used for legacy 'taylor_coeffs' path)
# ---------------------------------------------------------------------------

def _poly_taylor(coeffs, x):
    """
    Build Taylor polynomial  sum_n  (coeffs[n] / n!)  x^n  as an SR expression.
    coeffs[n] = f^(n)(background)  — the n-th derivative at the expansion point.
    """
    return sum(SR(c) * QQ(1)/factorial(n) * x**n
               for n, c in enumerate(coeffs))


def _iter_multi_indices(n_args, max_total):
    """Yield every multi-index (k_1, ..., k_{n_args}) of non-negative
    integers with  sum k_j  ≤ ``max_total``.

    Used by the formal-function rename machinery to enumerate every
    partial derivative up to a given total order.  For ``n_args=1``
    this collapses to ``(0,), (1,), ..., (max_total,)`` — i.e. the
    single-argument behavior the framework previously hardcoded.

    Implemented iteratively (via ``itertools.product``) rather than
    recursively so it survives ``%autoreload`` in Jupyter — a
    self-recursive generator would lose its reference to itself
    when the module's globals get hot-swapped.
    """
    if n_args == 0:
        yield ()
        return
    from itertools import product
    for combo in product(range(max_total + 1), repeat=n_args):
        if sum(combo) <= max_total:
            yield combo


def _multi_index_suffix(multi_idx):
    """Encode a multi-index as the rename-target suffix used in
    ``f<suffix>_<i+1>`` symbol names.

    For single-arg (``n_args=1``) the suffix is just the derivative
    order ``<k>`` — keeping the legacy ``f0_1``, ``f1_1``, ...
    naming intact.  For multi-arg we concatenate the per-argument
    derivative orders, giving ``f<k1><k2>...<kn>_<i+1>``.
    """
    return ''.join(str(k) for k in multi_idx)


# ---------------------------------------------------------------------------
# Truncated Taylor expansion of the action (polynomial structure + atoms)
# ---------------------------------------------------------------------------
#
# The legacy route is ONE multivariate ``taylor()`` of the whole SR action in
# every field variable, then ``.expand()``.  ``taylor()`` itself is quick, but
# Maxima returns every coefficient over the common denominator of ALL the
# non-polynomial atoms of the action: for a few sigmoids ``1/(1+exp(L))`` of
# linear combinations L of two fields with a symbolic saddle, the expanded
# order-2 action is ~0.5 MB (order 4: ~18 MB), single coefficients ~120 kB
# (even the free-propagator entry ``Dt*tau + 1`` comes out as a sum over that
# denominator).  ``PolynomialRing(SR)`` then zero-tests each of them, which
# goes to Maxima when a coefficient holds a symbol with an assumption (a
# ``domain='positive'`` parameter): ~100 s at order 2, never done at order 3.
#
# The truncated route walks the expression tree once.  Every field-dependent
# node becomes its Taylor polynomial ``{exponent tuple: SR coefficient}``,
# truncated at total degree ``order``: sums add, products and positive
# integer powers multiply with truncation at every step (response fields,
# which enter polynomially, cost nothing), and each distinct non-polynomial
# ATOM (``exp``, ``log``, a negative or fractional power, a function call,
# ...) gets one ``taylor()`` of its own, in the field variables it depends on,
# memoised.  The polynomial is the same as the legacy one; each atom keeps
# the form ``taylor()`` gives it alone (its double-precision rounding is as
# good as the legacy form's), and atoms are never put over a common
# denominator with each other.

_EXPAND_LEGACY_ENV = 'DAEDALUS_EXPAND_LEGACY'


def _expand_legacy_requested() -> bool:
    """Always ``False``: ``DAEDALUS_EXPAND_LEGACY`` (the 0.2.1 expansion, one
    multivariate ``taylor()`` of the whole action, which dropped degree-3
    terms) is retired; setting it only warns."""
    from daedalus.engine import _env
    _env.retired(_EXPAND_LEGACY_ENV,
                 'the per-atom truncated Taylor expansion is the only route')
    return False


class DerivativeAtomTaylorError(DaedalusError, Exception):
    """A derivative of an abstract function (``D[0](f)(x)``) in the action,
    in a place where Maxima's multivariate ``taylor()`` would truncate it
    (it stops ``k`` orders early on a ``k``-th derivative atom: ``taylor(
    x*D[0](f)(x), (x, 0), 3)`` has no ``x^3`` term).  Raised instead of a
    silently truncated expansion; not caught by the fall-backs."""


def _has_derivative_atom(e, field_vars=None) -> bool:
    """Whether ``e`` contains ``D[...](f)(...)`` of an abstract function
    (with ``field_vars``: one whose arguments depend on a field variable;
    an atom of parameters only is a constant to ``taylor()``)."""
    from sage.symbolic.operators import FDerivativeOperator
    names = (None if field_vars is None
             else {repr(v) for v in field_vars})
    stack = [SR(e)]
    while stack:
        x = stack.pop()
        op = x.operator()
        if op is None:
            continue
        if isinstance(op, FDerivativeOperator):
            if names is None or any(repr(u) in names for u in x.variables()):
                return True
            continue
        stack.extend(x.operands())
    return False


class _TruncatedTaylorUnsupported(Exception):
    """An atom whose ``taylor()`` is not a polynomial in the field variables
    (Laurent / Puiseux terms, or field variables left in a coefficient).
    The caller falls back to the legacy ``taylor()`` of the whole action, so
    such an action behaves exactly as before."""


def _sr_op_name(op) -> str:
    return (getattr(op, '__name__', None)
            or (op.name() if hasattr(op, 'name') else type(op).__name__))


def _truncated_taylor(expr, field_vars, order: int) -> dict:
    """Taylor polynomial of ``expr`` around ``field_vars = 0``, truncated at
    total degree ``order``, as ``{exponent tuple: SR coefficient}``.

    Same polynomial as ``taylor(expr, *[(v, 0) for v in field_vars], order)``;
    the coefficients are left unexpanded.  Raises
    :class:`_TruncatedTaylorUnsupported` for an atom whose expansion is not
    polynomial.
    """
    # SR objects are keyed by ``repr``: ``==`` between expressions builds a
    # relation whose truth test can go to Maxima (assumption symbols).
    fvars = list(field_vars)
    n = len(fvars)
    pos = {repr(v): i for i, v in enumerate(fvars)}
    zero = (0,) * n
    one = SR(1)
    memo: dict = {}

    def _add(acc, e, c):
        acc[e] = (acc[e] + c) if e in acc else c

    def _mul(A, B):
        out: dict = {}
        for ea, ca in A.items():
            da = sum(ea)
            for eb, cb in B.items():
                if da + sum(eb) <= order:
                    _add(out, tuple(x + y for x, y in zip(ea, eb)), ca * cb)
        return out

    def _depends(e):
        return any(repr(v) in pos for v in e.variables())

    def _atom(e):
        """One ``taylor()`` of a non-polynomial atom, in the field variables
        it depends on, read into an exponent dict."""
        names = {repr(u) for u in e.variables()}
        vs = [v for v in fvars if repr(v) in names]
        if _has_derivative_atom(e, vs):
            # the multivariate form truncates derivative atoms early; the
            # univariate form is exact (one field variable only)
            if len(vs) != 1:
                raise DerivativeAtomTaylorError(
                    f'cannot expand {e}: a derivative of an abstract function '
                    f'of several field variables ({", ".join(map(str, vs))}); '
                    f'give the function an expression')
            T = SR(taylor(e, vs[0], 0, order)).expand()
        else:
            T = SR(taylor(e, *[(v, 0) for v in vs], order)).expand()
        opT = T.operator()
        terms = (T.operands() if opT is not None
                 and _sr_op_name(opT) == 'add_vararg' else [T])
        out: dict = {}
        for term in terms:
            ex = [0] * n
            c = term
            for v in vs:
                d = c.degree(v)
                if d != int(d) or d < 0:
                    raise _TruncatedTaylorUnsupported(
                        f'non-polynomial expansion of {e}')
                if d:
                    ex[pos[repr(v)]] = int(d)
                    c = c.coefficient(v, d)
            if _depends(c):
                raise _TruncatedTaylorUnsupported(
                    f'field variables left in a coefficient of {e}')
            if sum(ex) <= order:
                _add(out, tuple(ex), c)
        return out

    def _go(e):
        e = SR(e)
        if not _depends(e):
            return {zero: e}
        key = repr(e)
        if key in memo:
            return memo[key]
        op = e.operator()
        name = _sr_op_name(op) if op is not None else None
        if op is None:                                   # a field variable
            ex = [0] * n
            ex[pos[key]] = 1
            r = {tuple(ex): one}
        elif name == 'add_vararg':
            r = {}
            for o in e.operands():
                for ee, c in _go(o).items():
                    _add(r, ee, c)
        elif name == 'mul_vararg':
            r = {zero: one}
            for o in e.operands():
                r = _mul(r, _go(o))
        elif (name == 'pow' and e.operands()[1].is_integer()
              and e.operands()[1] > 0):
            B = _go(e.operands()[0])
            r = {zero: one}
            for _ in range(int(e.operands()[1])):
                r = _mul(r, B)
        else:
            r = _atom(e)
        memo[key] = r
        return r

    return _go(expr)


def _taylor_action(S_sr, field_vars, order: int):
    """Truncated multivariate Taylor of the action in ``field_vars`` around
    0 (total degree <= ``order``), as an SR expression.

    Uses :func:`_truncated_taylor` unless an atom has no polynomial
    expansion at 0; then one ``taylor()`` of the whole action (which then
    succeeds or fails as before).
    """
    if not _expand_legacy_requested():         # (warns on the retired switch)
        try:
            poly = _truncated_taylor(S_sr, field_vars, order)
        except (_TruncatedTaylorUnsupported, ArithmeticError, ValueError,
                RuntimeError, TypeError):
            poly = None
        if poly is not None:
            out = SR(0)
            for ex, c in poly.items():
                mono = SR(1)
                for v, k in zip(field_vars, ex):
                    if k:
                        mono = mono * v ** k
                out = out + c * mono
            return out
    if _has_derivative_atom(S_sr, field_vars):
        raise DerivativeAtomTaylorError(
            'the action contains a derivative of an abstract function '
            '(D[...](f)(...)) and the expansion would take the multivariate '
            'taylor() route (an atom without a polynomial expansion), which '
            'truncates such atoms early; give '
            'the function an expression')
    return taylor(S_sr, *[(v, 0) for v in field_vars], order)


# ---------------------------------------------------------------------------
# Zero test and construction of PolynomialRing(SR) elements
# ---------------------------------------------------------------------------
#
# ``R(d)`` on ``R = PolynomialRing(SR, ...)`` zero-tests every coefficient
# (``PolyDict.remove_zeros`` -> ``bool(c)``), and every ring operation does it
# again for every coefficient of the result.  ``bool(c)`` on an expression
# containing a symbol with an assumption (``domain='positive'``) skips Sage's
# random-point test and asks Maxima, which takes seconds on a large
# coefficient.  :func:`_sr_nonzero` reaches the same verdict as ``bool(c)``
# but settles the common case -- a coefficient that is visibly nonzero at one
# point that satisfies every assumption -- by interval arithmetic, and
# :func:`_ring_from_nonzero_dict` then builds the ring element without a
# second test.

_NONZERO_PROBE_DOMAINS = frozenset({'real', 'positive', 'complex'})

# Operators whose interval extension in MPFI is rigorous; a coefficient with
# anything else (erf, Bessel, formal functions, ...) goes to ``bool(c)``.
_NONZERO_PROBE_OPERATORS = frozenset({
    'add_vararg', 'mul_vararg', 'pow', 'exp', 'log', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'arctan',
})


def _is_float_literal(po) -> bool:
    """A floating-point constant: Python ``float`` / ``complex`` or a Sage
    real or complex number (``0.5*I`` folds into one complex float)."""
    from sage.rings.real_double import RealDoubleElement
    from sage.rings.real_mpfr import RealNumber
    from sage.rings.complex_double import ComplexDoubleElement
    from sage.rings.complex_mpfr import ComplexNumber
    return isinstance(po, (float, complex, RealNumber, RealDoubleElement,
                           ComplexNumber, ComplexDoubleElement))


def _interval_safe(c, floats_ok: bool = False) -> bool:
    """Only rigorous operators and exact constants.  A float literal is
    excluded (unless ``floats_ok``): Sage compares floats approximately
    (``0.1*q - q/10`` is zero for ``bool``), exact interval arithmetic would
    not."""
    op = c.operator()
    if op is None:
        if c.is_numeric():
            try:
                po = c.pyobject()
                if _is_float_literal(po):
                    return floats_ok
                return bool(po.parent().is_exact())
            except (AttributeError, TypeError, ValueError):
                return False
        return True
    if _sr_op_name(op) not in _NONZERO_PROBE_OPERATORS:
        return False
    return all(_interval_safe(o, floats_ok) for o in c.operands())


# Complex interval arithmetic is rigorous for these without a branch cut
# (log, arctan and fractional powers are left to ``bool(c)``).
_NONZERO_PROBE_OPERATORS_COMPLEX = frozenset({
    'add_vararg', 'mul_vararg', 'pow', 'exp', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh',
})


def _complex_interval_safe(c) -> bool:
    """Operators whose complex interval extension has no branch cut, and
    integer powers only (the constants were checked by ``_interval_safe``)."""
    op = c.operator()
    if op is None:
        return True
    name = _sr_op_name(op)
    if name not in _NONZERO_PROBE_OPERATORS_COMPLEX:
        return False
    if name == 'pow':
        ex = c.operands()[1]
        if not (ex.is_numeric() and ex.is_integer()):
            return False
    return all(_complex_interval_safe(o) for o in c.operands())


def _probe_value(k: int):
    """Deterministic probe values in (0.2, 0.8): low-discrepancy (golden
    ratio) sequence, so distinct variables get unrelated values."""
    x = (0.5 + 0.6180339887498949 * (k + 1)) % 1.0
    return 0.2 + 0.6 * x


def _probe_point(c, floats_ok: bool = False):
    """``(vs, val_of)``: the sorted variables of ``c`` and a probe value for
    each, or ``None`` when ``c`` is not interval-safe or a probe value
    violates an assumption on one of its variables."""
    from sage.symbolic.assumptions import assumptions, GenericDeclaration
    if not _interval_safe(c, floats_ok):
        return None
    # Variables are keyed by name: ``==`` between SR variables builds
    # a relation whose truth test may itself go to Maxima.
    vs = sorted(c.variables(), key=repr)
    val_of = {repr(v): _probe_value(k) for k, v in enumerate(vs)}
    for a in assumptions():
        if isinstance(a, GenericDeclaration):
            if (repr(a._var) in val_of
                    and a._assumption not in _NONZERO_PROBE_DOMAINS):
                return None
            continue
        av = {repr(v): v for v in a.variables()}
        if any(k in val_of for k in av):
            if (not all(k in val_of for k in av)
                    or not bool(a.subs({v: val_of[k]
                                        for k, v in av.items()}))):
                return None
    return vs, val_of


def _perturb_floats(c, signs, margin):
    """``c`` with the k-th float literal (in tree order) multiplied by
    ``1 + signs[k % len(signs)] * margin``."""
    counter = [0]

    def walk(e):
        op = e.operator()
        if op is None:
            if e.is_numeric() and _is_float_literal(e.pyobject()):
                sg = signs[counter[0] % len(signs)]
                counter[0] += 1
                return e * (1 + sg * margin)
            return e
        return op(*[walk(o) for o in e.operands()])
    return walk(c)


_FLOAT_MARGIN = SR(1) / 10 ** 8
# Perturbation patterns: every float up, every float down, and the two
# alternating ones (two floats that differ by an ulp are moved apart).
_FLOAT_PATTERNS = ((1,), (-1,), (1, -1), (-1, 1))


def _proved_nonzero(c, floats_ok: bool = False,
                    above_float_dust: bool = False) -> bool:
    """True only if ``c`` is shown to be a nonzero function on its
    assumption domain: interval arithmetic (128 bits) at a probe point that
    satisfies every assumption excludes 0.  Never raises; False means "not
    proved".

    With ``floats_ok`` float literals are read as the exact rationals they
    are, and the proof is kept only if it survives perturbing the floats by
    ``1 +- 1e-8`` (all up, all down, alternating; same sign each time), so an
    expression that cancels to float dust (``exp(-0.3*x)*exp(0.1*x) -
    exp(-0.2*x)``, floats an ulp apart) is not claimed nonzero: Sage and
    Maxima call it zero.

    ``above_float_dust`` (with ``floats_ok``, when ``c`` has float
    literals): also require ``|c| > 2^-30`` times the magnitude of its terms
    at the point (:func:`_term_magnitude`).  A float expression whose exact
    value is below the double-precision rounding of its own terms (an
    ``(a+b)^2 - a^2 - 2ab - b^2`` of complex floats) is zero to ``bool``.
    """
    try:
        from sage.ext.fast_callable import fast_callable
        from sage.rings.real_mpfi import RealIntervalField
        pp = _probe_point(c, floats_ok)
        if pp is None:
            return False
        vs, val_of = pp
        RIF128 = RealIntervalField(128)
        args = [RIF128(val_of[repr(v)]) for v in vs]

        def sign_of(expr):
            val = fast_callable(expr, vars=vs, domain=RIF128)(*args)
            if val.contains_zero():
                return 0
            return 1 if val.lower() > 0 else -1

        try:
            sg = sign_of(c)
        except (TypeError, ValueError, ArithmeticError):
            # a complex constant (``I*omega`` of a Fourier image): the same
            # proof in complex interval arithmetic, entire functions only
            if not _complex_interval_safe(c):
                return False
            from sage.rings.complex_interval_field import (
                ComplexIntervalField)
            CIF128 = ComplexIntervalField(128)
            cargs = [CIF128(val_of[repr(v)]) for v in vs]

            def sign_of(expr):
                val = fast_callable(expr, vars=vs, domain=CIF128)(*cargs)
                re, im = val.real(), val.imag()
                out = (0 if re.contains_zero() else (1 if re.lower() > 0
                                                     else -1),
                       0 if im.contains_zero() else (1 if im.lower() > 0
                                                     else -1))
                return 0 if out == (0, 0) else out

            sg = sign_of(c)
        if sg == 0:
            return False
        if floats_ok and not _interval_safe(c, False):
            for pat in _FLOAT_PATTERNS:
                if sign_of(_perturb_floats(c, pat, _FLOAT_MARGIN)) != sg:
                    return False
            if above_float_dust:
                from sage.rings.complex_interval_field import (
                    ComplexIntervalField)
                CIF128 = ComplexIntervalField(128)
                cargs = [CIF128(val_of[repr(v)]) for v in vs]
                mag = fast_callable(c, vars=vs, domain=CIF128)(*cargs).abs()
                scale = _term_magnitude(c, vs, cargs, CIF128)
                if not mag.lower() > _FLOAT_DUST_REL * scale.upper():
                    return False
        return True
    except Exception:
        return False


# A float expression is not claimed nonzero below this fraction of the
# magnitude of its terms (double rounding is ~1e-16 of it per operation).
_FLOAT_DUST_REL = 2.0 ** -30


def _term_magnitude(c, vs, cargs, CIF):
    """Upper magnitude of ``c`` with every sum replaced by the sum of the
    magnitudes of its terms (products and integer powers of term
    magnitudes; any other operator is the magnitude of its own value), at
    the point ``cargs``: the scale its float rounding is relative to."""
    from sage.ext.fast_callable import fast_callable
    memo = {}

    def mag(e):
        key = repr(e)
        if key in memo:
            return memo[key]
        op = e.operator()
        name = _sr_op_name(op) if op is not None else None
        if name == 'add_vararg':
            r = sum(mag(o) for o in e.operands())
        elif name == 'mul_vararg':
            r = 1
            for o in e.operands():
                r = r * mag(o)
        elif (name == 'pow' and e.operands()[1].is_numeric()
              and e.operands()[1].is_integer() and e.operands()[1] >= 0):
            r = mag(e.operands()[0]) ** int(e.operands()[1])
        else:
            r = fast_callable(e, vars=vs, domain=CIF)(*cargs).abs()
        memo[key] = r
        return r
    return mag(SR(c))


def _sr_nonzero(c) -> bool:
    """``bool(c)`` for an SR coefficient (True unless Sage can show it is
    zero), with a fast exact certificate for the nonzero case.

    If ``c`` is built from operators with a rigorous interval extension and
    evaluates, in 128-bit interval arithmetic, to an interval excluding 0 at
    a real point inside (0.2, 0.8)^n that satisfies every assumption on its
    variables, then ``c`` is not zero (as a function on the assumption
    domain), which is what ``bool(c)`` would conclude after asking Maxima.
    Otherwise the decision is Sage's own ``bool(c)``.
    """
    c = SR(c)
    if c.is_trivial_zero():
        return False
    # the same large coefficient is tested once per typed diagram
    # (``classify_coefficient_factors``): the verdict is memoised on its
    # GiNaC hash (exact float bits), a digest of its ``repr`` and the
    # assumptions in force (never ``==`` on SR, which can itself go to
    # Maxima)
    import hashlib
    from sage.symbolic.assumptions import assumptions
    key = (hash(c), hashlib.sha1(repr(c).encode()).digest(),
           repr(assumptions()))
    hit = _NONZERO_MEMO.get(key)
    if hit is not None:
        return hit
    if _proved_nonzero(c):
        out = True
    # Float coefficients (numeric rates baked into the action): Sage's
    # random-point test cannot certify a float expression, so ``bool(c)``
    # always asks Maxima, seconds per large coefficient (and an ECL stack
    # overflow on a whole multi-field action).  A proof that survives
    # perturbing every float by 1e-8 (``floats_ok``) is one ``bool(c)``
    # reaches too: a float expression that cancels to dust fails it.
    elif _proved_nonzero(c, floats_ok=True, above_float_dust=True):
        out = True
    else:
        out = bool(c)
    if len(_NONZERO_MEMO) >= _NONZERO_MEMO_MAX:
        _NONZERO_MEMO.clear()
    _NONZERO_MEMO[key] = out
    return out


_NONZERO_MEMO: dict = {}
_NONZERO_MEMO_MAX = 4096


def _ring_from_nonzero_dict(R, d: dict):
    """``R(d)`` for a dict whose coefficients are all known to be nonzero
    (already zero-tested), without Sage testing each of them again.  The
    element is the one ``R(d)`` builds (same class, same ``PolyDict``)."""
    if not d:
        return R.zero()
    try:
        from sage.rings.polynomial.multi_polynomial_element import (
            MPolynomial_polydict, MPolynomial_element)
        from sage.rings.polynomial.polydict import PolyDict
        if type(R.gen(0)) is not MPolynomial_polydict:
            return R(d)
        K = R.base_ring()
        el = MPolynomial_polydict.__new__(MPolynomial_polydict)
        MPolynomial_element.__init__(
            el, R, PolyDict({k: K(a) for k, a in d.items()}))
        return el
    except ImportError:
        return R(d)


def _sum_of_polys(R, polys):
    """``R.zero() + p_1 + p_2 + ...`` for ring elements with pairwise
    disjoint monomials (the bigrade sectors), without re-testing every
    coefficient at each ``+``.  The dict order is the one the running sum
    gives (``PolyDict.__add__`` copies the larger operand and appends the
    other's keys)."""
    acc: dict = {}
    for p in polys:
        d = p.dict()
        big, small = (d, acc) if len(acc) < len(d) else (acc, d)
        merged = dict(big)
        for e, c in small.items():
            if e in merged:                     # not disjoint: plain ring sum
                return sum(polys, R.zero())
            merged[e] = c
        acc = merged
    return _ring_from_nonzero_dict(R, acc)


# ---------------------------------------------------------------------------
# SR → PolynomialRing conversion
# ---------------------------------------------------------------------------

def _sr_to_ring(sr_expr, R, ring_var_names: list):
    """
    Convert an SR expression that is polynomial in ring_var_names to a
    PolynomialRing(SR, ring_var_names) element.

    For each summand of the expanded SR expression, extracts the degree in each
    ring variable via .degree() / .coefficient(v, d), accumulating SR coefficients
    (parameters, kernels, operator symbols) separately from ring monomials.

    Handles the sum/single-term distinction explicitly so that symbolic
    coefficients like tau*Dt are never lost.
    """
    import operator as _op
    gen_sr = [SR.var(name) for name in ring_var_names]
    result = R.zero()

    expanded = SR(sr_expr).expand()
    if expanded.is_trivial_zero():
        # The whole-action zero test is not needed: the ring monomials are
        # independent, so the action is zero exactly when every coefficient
        # is, and each coefficient is tested below.  (Testing the whole
        # expanded action sent one huge expression to Maxima: an ECL
        # BINDING-STACK overflow on a three-field jump model.)
        return result

    # Collect summands.  For a sum, .operands() gives the addends correctly.
    # For a non-sum (single product, single atom), wrap in a list so we don't
    # accidentally iterate over factors instead of summands.
    # Use name-based check (not identity) because SageMath uses add_vararg,
    # not operator.add — identity check with _op.add silently fails.
    _op_obj = expanded.operator()
    if _op_obj is not None and 'add' in getattr(_op_obj, '__name__', '').lower():
        summands = expanded.operands()
    else:
        summands = [expanded]

    # Accumulate into a {exponent_tuple: SR coeff} dict, then build the ring
    # element in ONE shot via R(dict).  The old incremental
    # ``result += SR(coeff) * R.monomial(*exponents)`` re-normalises the whole
    # growing polynomial on every step -- O(N^2), and brutal when the SR
    # coefficients are large (e.g. the shared exp-denominators that a
    # sigmoid/Bernoulli mean-field saddle bakes into every monomial: the
    # dendritic order-3 expand spent ~36 min almost entirely here).  The dict
    # build is O(N) and yields a bit-identical polynomial.
    acc = {}
    for term in summands:
        exponents = []
        coeff = SR(term)
        for v_sr in gen_sr:
            deg = int(coeff.degree(v_sr))
            exponents.append(deg)
            if deg > 0:
                coeff = coeff.coefficient(v_sr, deg)
        key = tuple(exponents)
        acc[key] = (acc[key] + SR(coeff)) if key in acc else SR(coeff)
    if not acc:
        return result
    # Same zero decisions as ``R(acc)``, one (fast-path) test per coefficient.
    return _ring_from_nonzero_dict(
        R, {k: c for k, c in acc.items() if _sr_nonzero(c)})


# ---------------------------------------------------------------------------
# Bigrade classification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Correlated-noise cumulant injection  (Option A: read declarative
# ``correlated_noises`` block, append -W_m[mt] to the action symbolically)
# ---------------------------------------------------------------------------

class _PermutedKernel:
    """A cumulant kernel read for a reconstructed leg-field permutation
    ``perm`` of its row (``perm[k]``: the row's leg sitting at position
    ``k``): the leg indices are put back in the row's own order, and at
    order 2 the swapped pair reads ``K(j, i, -tau)`` (``tau = t_first -
    t_second``; a delta of ``-tau`` written back as one of ``tau``, the form
    the delta extraction matches).  At order >= 3 only the fully local
    (delta-product) part is used, which no permutation of the times
    changes.  Module level so it pickles like the kernel it wraps."""
    __slots__ = ('inner', 'perm')

    def __init__(self, inner, perm):
        self.inner = inner
        self.perm = tuple(perm)

    def __call__(self, ns, *args):
        n = len(self.perm)
        idx, taus = args[:n], list(args[n:])
        canon = [None] * n
        for k, p in enumerate(self.perm):
            canon[p] = idx[k]
        if n == 2 and self.perm == (1, 0) and len(taus) == 1:
            t = SR(taus[0])
            K = SR(self.inner(ns, *canon, -t))
            return K.subs({dirac_delta(-t): dirac_delta(t)})
        return self.inner(ns, *canon, *taus)


def _build_cumulant_action(ns, model):
    r"""
    Construct the symbolic contribution

        S_cum  =  - sum_n  (1/n!)  \int dt_1 ... dt_n
                    \sum_{i_1...i_n}  kappa^{(n)}_{i_1...i_n}(\tau\text{'s})
                                      \tilde m_{i_1}(t_1) ... \tilde m_{i_n}(t_n)

    for each noise process declared in ``model['correlated_noises']``.

    Per ordered leg-tuple, the user-supplied kernel function is called
    with a placeholder relative-time variable ``\tau``.  The result is
    decomposed as

        K(tau)  =  c_local * dirac_delta(tau)  +  K_smooth(tau)

    via ``K.coefficient(dirac_delta(tau))``:

      * ``c_local`` (dirac residue) collapses the time integral and
        contributes  -(1/2) * c_local * mt[i] * mt[j]  directly to S_cum
        (a *local* noise-kernel term — same format as the cortical
        Poisson  -1/2 nstar_i * nt_i^2);
      * ``K_smooth`` introduces an SR coefficient symbol
        ``z_kappa_<noise>_<order>_<i+1>_<j+1>`` (analogous to ``z_g``
        for the synaptic filter) representing the implicit two-time
        integration; the kernel function is registered on
        ``ns._cumulant_kernels[(noise, order, (i, j))]`` for the
        downstream pipeline to consume.

    Only order n = 2 is implemented in this commit (covers the GTaS
    Bernoulli + Gaussian case).  Higher orders fire a warning and are
    skipped — that's fine for N=2 GTaS where κ^(n)=0 for n≥3.

    Returns an SR expression; SR(0) when the model has no
    ``correlated_noises`` declaration (i.e. plain quad_expg, linear, …).
    """
    cn = model.get('correlated_noises', None)
    ns._cumulant_kernels = {}
    if not cn:
        return SR(0)

    S_cum = SR(0)

    for noise_name, spec in cn.items():
        # ``physical_field`` is OPTIONAL — present for GTaS-style external
        # noise that introduces its own physical field (MSR bilinear
        # ``mt · m`` is registered elsewhere), absent for inline
        # cumulants on existing response fields (no new field involved).
        phys_name = spec.get('physical_field')
        if phys_name and not hasattr(ns, phys_name):
            raise ValueError(
                f"correlated_noises[{noise_name!r}]: physical_field "
                f"{phys_name!r} is not declared in physical_fields"
            )

        # Per-leg response field resolution:
        #   * ``response_legs`` (NEW; dict keyed by order, value = list of
        #     response-field names of length ``order``) — supports
        #     cross-field cumulants (legs on multiple response fields).
        #   * ``response_field`` (legacy; single string broadcast to all
        #     legs at every order) — kept for back-compat.
        response_legs_by_order = spec.get('response_legs') or {}
        legacy_resp_name = spec.get('response_field')
        if legacy_resp_name and not hasattr(ns, legacy_resp_name):
            raise ValueError(
                f"correlated_noises[{noise_name!r}]: response_field "
                f"{legacy_resp_name!r} is not declared in response_fields"
            )

        # κ^(1) (mean) is informational — already absorbed into the saddle
        # by the model's mf_bg_conditions.  Skip explicit injection.

        for order, kernel_fn in spec.get('cumulants', {}).items():
            if order < 2:
                continue

            # Resolve per-leg response field names for THIS order.  If
            # the new ``response_legs`` block has an entry, use it;
            # otherwise fall back to the legacy single-field broadcast.
            leg_field_names = response_legs_by_order.get(order)
            if leg_field_names is None:
                if not legacy_resp_name:
                    raise ValueError(
                        f"correlated_noises[{noise_name!r}] order={order}: "
                        f"no response_legs and no legacy response_field — "
                        f"can't resolve which response field each leg "
                        f"sits on."
                    )
                leg_field_names = [legacy_resp_name] * int(order)

            if len(leg_field_names) != int(order):
                raise ValueError(
                    f"correlated_noises[{noise_name!r}] order={order}: "
                    f"response_legs has {len(leg_field_names)} entries "
                    f"but order is {order}.")

            # Per-leg SR-var arrays and index ranges.  All entries are
            # wrapped to lists so a non-indexed (scalar) response field
            # still iterates uniformly.
            leg_arrays: list = []
            leg_ranges: list = []
            for fname in leg_field_names:
                if not hasattr(ns, fname):
                    raise ValueError(
                        f"correlated_noises[{noise_name!r}] order={order}: "
                        f"response-leg field {fname!r} not declared.")
                arr = getattr(ns, fname)
                if not isinstance(arr, list):
                    arr = [arr]
                leg_arrays.append(arr)
                leg_ranges.append(range(len(arr)))

            # Cumulant order n needs n-1 relative time variables.
            tau_syms = [SR.var(f'_tau_{noise_name}_{k}',
                               latex_name=rf'\tau_{{{k}}}')
                        for k in range(order - 1)]
            n_fact = factorial(order)
            from itertools import product as _iter_product
            from itertools import permutations as _iter_perms

            # Enumerate distinct permutations of the leg-field tuple.
            # For homogeneous leg-fields (e.g. ['xt', 'xt']) there's
            # only one — the existing leg-INDEX product over the field's
            # range already covers all index orderings (the 1/n! factor
            # symmetrizes them).  For heterogeneous leg-fields
            # (e.g. ['xt', 'yt']) the cumulant series sums over BOTH
            # (xt, yt) and (yt, xt) ordered tuples but the user only
            # writes one canonical row; the framework reconstructs the
            # other orderings here.
            seen_perms = set()
            distinct_field_perms = []
            for perm_indices in _iter_perms(range(order)):
                perm_fields = tuple(leg_field_names[p] for p in perm_indices)
                if perm_fields in seen_perms:
                    continue
                seen_perms.add(perm_fields)
                # Record the per-leg array, range, and original-position
                # mapping for this permutation.
                perm_arrays = [leg_arrays[p] for p in perm_indices]
                perm_ranges = [leg_ranges[p] for p in perm_indices]
                distinct_field_perms.append({
                    'fields': perm_fields,
                    'arrays': perm_arrays,
                    'ranges': perm_ranges,
                    'perm': tuple(perm_indices),
                })

            for perm_spec in distinct_field_perms:
              perm_leg_arrays = perm_spec['arrays']
              perm_leg_ranges = perm_spec['ranges']
              perm_leg_fields = list(perm_spec['fields'])
              # The row's kernel K(i, j, tau) is written for its own leg
              # order (tau = t_first - t_second); a reconstructed order is
              # read with the indices put back (order 2: K(j, i, -tau)), so
              # every registered spec carries its own orientation (Phase J
              # evaluates each spec in it).
              kfn = (kernel_fn if perm_spec['perm'] == tuple(range(order))
                     else _PermutedKernel(kernel_fn, perm_spec['perm']))
              for idx_tuple in _iter_product(*perm_leg_ranges):
                # Evaluate the kernel at placeholder τ symbols.
                try:
                    K = SR(kfn(ns, *idx_tuple, *tau_syms))
                except TypeError:
                    # Backward-compat: order 2 callers may use the old
                    # (ns, i, j, tau) signature with a single τ; expand.
                    if order == 2 and len(tau_syms) == 1:
                        K = SR(kfn(ns, idx_tuple[0], idx_tuple[1],
                                   tau_syms[0]))
                    else:
                        raise
                K = K.expand()

                # Iteratively peel off δ(τ_k) coefficients.  After n-1
                # steps, what remains in ``c_local`` is the multiplier
                # of the FULL delta product  ∏ δ(τ_k).
                c_local = K
                for tau_k in tau_syms:
                    try:
                        c_local = c_local.coefficient(dirac_delta(tau_k))
                    except (AttributeError, TypeError):
                        c_local = SR(0)
                        break
                # Residual = K minus the all-delta contribution
                delta_product = SR(1)
                for tau_k in tau_syms:
                    delta_product = delta_product * dirac_delta(tau_k)
                K_residual = (K - c_local * delta_product)
                try:
                    K_residual = K_residual.simplify_full()
                except (ValueError, RuntimeError, AttributeError):
                    pass

                # ── Local (fully delta-correlated) contribution ──────
                # All time integrals collapse;  -(1/n!) c_local m̃_{i₁}…m̃_{iₙ}
                # at a single time gets injected directly into S_cum.
                # Each leg's SR variable comes from THAT leg's response
                # field — leg_arrays[k] is the SR-var list for the k-th
                # leg's named response field.
                #
                # Zero check: ``c_local != 0`` returns False under
                # Sage when the expression contains unbound parameters
                # without positivity assumptions (e.g. ``rho`` with
                # ``domain='real'``).  ``is_trivial_zero()`` is the
                # right test: True iff the expression is syntactically
                # zero, regardless of param assumptions.
                if not SR(c_local).is_trivial_zero():
                    factor = -SR(1) / n_fact * SR(c_local)
                    for k, k_idx in enumerate(idx_tuple):
                        factor = factor * perm_leg_arrays[k][k_idx]
                    S_cum = S_cum + factor

                # ── Smooth (non-local) residual ──────────────────────
                # Currently only handled at order 2 (the integrator uses
                # one τ_v per noise vertex).  For order ≥ 3 with a
                # smooth residual, warn — the integrator's per-leg time
                # map is ordered-by-leg; we'd need an n-leg time map.
                # For Bernoulli + Gaussian GTaS at N=2, all order-≥3
                # cumulants are FULLY LOCAL so this branch never fires
                # in practice; future non-local higher-order kernels
                # need integrator extension before they can be used.
                #
                # Symbol naming for cross-field at order 2: encode each
                # leg's field name in the suffix so the same noise can
                # produce both ``z_kappa_X_2_mt_1_mt_2`` (auto, cells 1/2
                # of mt) and ``z_kappa_X_2_xt_1_yt_1`` (cross between
                # different response fields).
                if not SR(K_residual).is_trivial_zero():
                    if order == 2:
                        leg_a_name = perm_leg_fields[0]
                        leg_b_name = perm_leg_fields[1]
                        sym_name = (
                            f'z_kappa_{noise_name}_{order}'
                            f'_{leg_a_name}_{idx_tuple[0]+1}'
                            f'_{leg_b_name}_{idx_tuple[1]+1}'
                        )
                        latex_name = (
                            rf'\kappa^{{({order})}}_'
                            rf'{{{leg_a_name}_{idx_tuple[0]+1},'
                            rf'{leg_b_name}_{idx_tuple[1]+1}}}'
                        )
                        sym = SR.var(sym_name, latex_name=latex_name)
                        # The leg-FIELD permutation outer loop has
                        # already enumerated (xt, yt) and (yt, xt) as
                        # distinct iterations, so the (1/n!) factor and
                        # the iteration's symmetry combine to give the
                        # correct overall coefficient.  At the canonical
                        # ordered single-field case (legs = ['mt', 'mt']),
                        # only ONE permutation exists; the index-product
                        # iteration over (i, j) + (j, i) provides the
                        # within-field symmetry as before.
                        S_cum = S_cum + (
                            -SR(1) / n_fact * sym
                            * perm_leg_arrays[0][idx_tuple[0]]
                            * perm_leg_arrays[1][idx_tuple[1]]
                        )
                        ns._cumulant_kernels[
                            (noise_name, order,
                             tuple(zip(perm_leg_fields, idx_tuple)))
                        ] = {
                            'symbol':    sym,
                            'kernel_fn': kfn,
                            'legs':      tuple(idx_tuple),
                            'leg_fields': tuple(perm_leg_fields),
                            'tau_var':   tau_syms[0],
                        }
                    else:
                        warnings.warn(
                            f"correlated_noises[{noise_name!r}] "
                            f"cumulant order {order} legs "
                            f"{idx_tuple}: kernel has a non-local "
                            f"smooth residual.  The Phase J integrator "
                            f"only supports smooth (non-δ) kernels at "
                            f"order n=2 — at n≥3 the smooth part is "
                            f"silently dropped (the δ-local part of "
                            f"the same kernel, if any, IS still "
                            f"injected correctly).  See "
                            f"docs/correlated_noise_capabilities.md "
                            f"§1 for the integrator-extension cost "
                            f"and workarounds.",
                            stacklevel=3,
                        )

    return S_cum


def _collect_bigrade(poly, n_tilde: int) -> dict:
    """
    Split a PolynomialRing element into {(n_tilde, n_phys): poly} sectors.
    Ring generators are ordered [tilde_gens..., phys_gens...].
    """
    R = poly.parent()
    sectors: dict = {}
    # The coefficients of a ring element are nonzero already: split the
    # dict and build each sector once (the incremental ``+`` re-tested every
    # coefficient of the growing sector at each step).
    parts: dict = {}
    for exp_vec, coeff in poly.dict().items():
        key = (int(sum(exp_vec[:n_tilde])), int(sum(exp_vec[n_tilde:])))
        parts.setdefault(key, {})[exp_vec] = SR(coeff)
    for key, d in parts.items():
        sectors[key] = _ring_from_nonzero_dict(R, d)
    return sectors


#: Relative tolerance of the mean-field sector checks (expand time, and the
#: selected root in ``daedalus.api._mean_field_dae._check_selected_root``): a wrong
#: equation leaves a residual of the size of its own terms, a converged
#: saddle ~1e-16 of it.  (Was an absolute 1e-9 at expand time: valid models
#: in large field units or fast time units were refused --.)
MF_SECTOR_RTOL = 1e-6


#: Round-off allowance of the sector checks: this many ulps of the largest
#: term scale of the whole mean-field sector (a float residue such as a
#: noise source's K(0) = 3e-17, collapsed to a bare number, has no terms of
#: its own to be judged against).
MF_SECTOR_ULPS = 64


def _terms_scale(expr, num_subs, unbound=None):
    """Sum of the magnitudes of the expanded terms of ``expr`` at
    ``num_subs`` (0 when it cannot be evaluated)."""
    try:
        terms = SR(expr).expand()
        ops = (terms.operands()
               if getattr(terms.operator(), '__name__', '') == 'add_vararg'
               else [terms])
        tot = 0.0
        for t in ops:
            v = SR(t).subs(num_subs)
            if v.free_variables():
                if unbound is None:
                    return 0.0
                v = v.subs({f: unbound for f in v.free_variables()})
            tot += abs(complex(v))
        return tot
    except Exception:                                   # noqa: BLE001
        return 0.0


def mf_sector_residual(expr, num_subs, errs=None, unbound=None,
                       raw=None, floor=0.0):
    """``(resid, tol, scale)`` of a mean-field sector coefficient at a
    numerical point, or ``None`` when it cannot be evaluated.

    ``resid = |sum of the terms|`` and ``scale = sum of |terms|`` (the terms
    of the expanded coefficient, each evaluated at ``num_subs``);
    ``tol = MF_SECTOR_RTOL * scale + 10 * sum_v |d expr / d v| err_v`` over
    the saddle values with an error estimate ``errs = {symbol: err}`` (a
    root at 0 is known to ~its own size only: its terms vanish with it).
    Every part scales with the coefficient, so the verdict does not depend
    on the units.  ``unbound``: the value given to a symbol ``num_subs``
    does not bind (``None``: such a coefficient is not evaluated).  ``raw``:
    the coefficient before the saddle substitution (its terms set the scale
    too: a closed-form saddle collapses ``-mu x* + m`` to a bare float);
    ``floor``: an absolute round-off allowance added to ``tol``."""
    c = SR(expr)
    try:
        # exactly zero at the point: no scale needed (and no expansion)
        whole = c.subs(num_subs)
        if unbound is not None and whole.free_variables():
            whole = whole.subs({f: unbound for f in whole.free_variables()})
        if not whole.free_variables() and complex(whole) == 0:
            return 0.0, floor, 0.0
    except Exception:                                   # noqa: BLE001
        pass
    try:
        terms = c.expand()
        ops = (terms.operands()
               if getattr(terms.operator(), '__name__', '') == 'add_vararg'
               else [terms])
        vals = [SR(t).subs(num_subs) for t in ops]
        free = set()
        for v in vals:
            free.update(v.free_variables())
        if free:
            if unbound is None:
                return None
            fill = {f: unbound for f in free}
            vals = [v.subs(fill) for v in vals]
        vals = [complex(v) for v in vals]
    except Exception:                                   # noqa: BLE001
        return None
    resid = abs(sum(vals))
    scale = sum(abs(v) for v in vals)
    if raw is not None:
        scale = max(scale, _terms_scale(raw, num_subs, unbound))
    tol = MF_SECTOR_RTOL * scale + floor
    if not resid <= tol and errs:
        sens = 0.0
        for sym, e in errs.items():
            if not e:
                continue
            try:
                if not c.has(sym):
                    continue
                d = SR(c.diff(sym)).subs(num_subs)
                if unbound is not None and d.free_variables():
                    d = d.subs({f: unbound for f in d.free_variables()})
                sens += abs(complex(d)) * float(e)
            except Exception:                           # noqa: BLE001
                continue
        tol += 10.0 * sens
    return resid, tol, scale


def _verify_and_zero_mf_sector(by_tp, mf_subs, spec_subs, ns, R, model,
                               mf_sector_keys=((0, 0), (1, 0), (0, 1))):
    """Apply ``mf_subs`` to the saddle-eq (MF) sector and verify it
    vanishes.  After verification each MF sector entry is replaced by
    ``R.zero()`` in-place — at the saddle the sector is zero by
    construction, and downstream consumers (sanity_check, propagator
    extractor) rely on it.

    Two-tier check, per the user's spec:

      1. **Symbolic**: ``coeff.subs(mf_subs).simplify_full() == 0``.
         Cheap and complete for closed-form saddles.
      2. **Numerical fallback** (only when symbolic fails): bind every
         remaining free symbol to a numerical saddle from the model's
         own MF solver and check the residual RELATIVE to the size of the
         coefficient's terms (:func:`mf_sector_residual`).  Catches the
         case where the MF equation is solved iteratively rather than
         in closed form, so ``simplify_full`` can't see the
         cancellation but the numerics confirm it.

    Failures raise :class:`AssertionError` with the offending bigrade,
    the post-subs symbolic form, and the numerical residual when
    available.  This is the structural test that catches both a
    miswired MF solver and an action whose bigrade-≤1 sector is not
    actually the saddle-eq sector.
    """
    failures: list = []
    soft_passes: list = []
    mf_memo: dict = {}      # the saddle solve is the same for every coefficient
    dc_gains = getattr(ns, '_kernel_dc_gains', None)
    pending: list = []      # numerically checked, judged after the sector scale

    for key in mf_sector_keys:
        sector_poly = by_tp.get(key)
        if sector_poly is None or sector_poly == R.zero():
            continue
        for exp_vec, coeff in sector_poly.dict().items():
            c_subbed = SR(coeff).subs(mf_subs)
            if spec_subs:
                c_subbed = c_subbed.subs(spec_subs)
            # A kernel symbol left at the saddle (a hand-written
            # mean-field text ``w*g*nstar``, or a legacy ``g*n`` action
            # term) is the kernel's integral there, as in the solvers.
            if dc_gains is not None:
                k_subs = dc_gains.subs_dict(c_subbed)
                if k_subs:
                    c_subbed = c_subbed.subs(k_subs)
            if not _proved_nonzero(c_subbed, floats_ok=True):
                try:
                    c_simpl = c_subbed.simplify_full()
                    if c_simpl == 0:
                        continue
                except Exception:
                    c_simpl = c_subbed
            else:
                # Shown nonzero at a probe point (interval arithmetic), so
                # ``simplify_full`` cannot reach 0 and is skipped; it is
                # only run to word a failure (below).
                c_simpl = None
            raw = SR(coeff)
            if dc_gains is not None:
                k_raw = dc_gains.subs_dict(raw)
                if k_raw:
                    raw = raw.subs(k_raw)
            num = _mf_numerical_residual(c_subbed, ns, model, mf_memo, raw)
            pending.append((key, exp_vec, c_simpl, c_subbed, num))

    # round-off allowance: MF_SECTOR_ULPS ulps of the largest term scale of
    # the sector (a float residue collapsed to a bare number has none)
    sector_scale = max([0.0] + [num[2] for *_, num in pending
                                if num is not None])
    floor = MF_SECTOR_ULPS * sys.float_info.epsilon * sector_scale
    for key, exp_vec, c_simpl, c_subbed, num in pending:
        num_resid = None if num is None else num[0]
        if num is not None and num[0] <= num[1] + floor:
            soft_passes.append((key, exp_vec, c_simpl, num_resid))
        else:
            if c_simpl is None:
                try:
                    c_simpl = c_subbed.simplify_full()
                except Exception:
                    c_simpl = c_subbed
            failures.append((key, exp_vec, c_simpl, num_resid))

    if failures:
        lines = [
            "This model's mean-field saddle does not satisfy its own "
            "equations — the MF action sector is non-zero at the saddle.",
            "Likely causes:",
            "  - a .set_mf_equation(...) / .mf_residual(...) that doesn't match "
            "the action's deterministic drift; or",
            "  - parameters with no default= (only DEFAULT_FUNDAMENTAL), so "
            "the saddle can't be checked numerically — the 'numerical "
            "residual' below then reads 'N/A'.  Add default= to those "
            "parameters.",
            "Diagnostics (internal):",
        ]
        for key, exp_vec, c_simpl, num_resid in failures:
            num_str = (f"{num_resid:.3e}"
                       if num_resid is not None else "N/A")
            lines.append(
                f"  bigrade={key}  monomial_exponents={exp_vec}\n"
                f"    symbolic residual: {c_simpl}\n"
                f"    numerical residual: {num_str}")
        solve_exc = mf_memo.get('num_params')
        if isinstance(solve_exc, Exception):
            # the numerical check could not run: say why (it was swallowed
            # into a bare 'N/A' before)
            lines.append(
                f"  the numerical check could not run: the mean-field solve "
                f"at the model's default parameter point raised "
                f"{type(solve_exc).__name__}: {solve_exc}")
            raise AssertionError('\n'.join(lines)) from solve_exc
        raise AssertionError('\n'.join(lines))

    if soft_passes:
        warnings.warn(
            f"MF sector vanishes only numerically (not symbolically) for "
            f"{len(soft_passes)} monomial(s).  This is normal when the "
            f"MF equation is solved iteratively rather than in closed "
            f"form; flagging for visibility.",
            stacklevel=3,
        )

    for key in mf_sector_keys:
        if key in by_tp:
            by_tp[key] = R.zero()


def kernel_dc_gains_of(ns, model):
    """The model's kernel integrals (:class:`daedalus.engine.core.convolution.
    KernelDCGains`), made once per namespace and kept on it as
    ``ns._kernel_dc_gains`` (the mean-field check and the solvers read the
    same object)."""
    from daedalus.engine.core.convolution import KernelDCGains
    gains = getattr(ns, '_kernel_dc_gains', None)
    if gains is None:
        gains = KernelDCGains(ns, model)
        try:
            ns._kernel_dc_gains = gains
        except AttributeError:
            pass
    return gains


def _mf_numerical_residual(expr, ns, model, memo=None, raw=None):
    """``(residual, tolerance, scale)`` of an SR expression at the MF saddle
    (:func:`mf_sector_residual`; a symbol the saddle point does not bind
    is set to 1.0).

    Binds every free symbol to its numerical value at the saddle for a
    representative parameter point (``model['fundamental_defaults']`` if
    provided, else all-ones).  Returns ``None`` if the residual can't be
    built (no MF solver, fundamental incomplete, etc.) — the caller falls
    back to ``failures`` rather than ``soft_passes`` in that case.
    """
    if memo is not None and 'num_params' in memo:
        # The solve below does not depend on ``expr`` (deterministic: fixed
        # seed), so one verification solves once, not once per coefficient.
        num_params = memo['num_params']
    else:
        num_params = _mf_saddle_values(ns, model)
        if memo is not None:
            memo['num_params'] = num_params
    if num_params is None or isinstance(num_params, Exception):
        return None
    vals, errs = num_params
    out = mf_sector_residual(expr, vals, errs, unbound=1.0, raw=raw)
    if out is None and raw is not None:
        # The symbolic saddle substitution can put a pole at the default point
        # (a restoring coefficient that vanishes there: ``mu - b - c = 0``
        # divides the closed form of the saddle).  The check is about the
        # coefficient at the NUMERICAL saddle, so evaluate it before the
        # substitution.
        out = mf_sector_residual(raw, vals, errs, unbound=1.0, raw=raw)
    return out


def _mf_saddle_values(ns, model):
    """``(num_params, errors)``: the numerical saddle used by the residual
    check and the error estimates of its values (``{}`` when the solver
    gives none); ``None`` without a parameter point, the solver's exception
    when the solve fails."""
    fundamental = (model.get('fundamental_defaults') or
                   _default_fundamental_point(ns, model))
    if fundamental is None:
        return None

    # the one multi-root solver (``.equation`` rows or saddle equations);
    # a failure is recorded, not swallowed: the sector check names it
    try:
        from daedalus.api._mean_field import solve_mean_field
        ft_proxy = _MFProxyForSolver(
            ns, model, taylor_order=getattr(ns, '_taylor_order', 4))
        mf = solve_mean_field(ft_proxy, model, fundamental, verbose=False)
    except Exception as exc:                            # noqa: BLE001
        return exc
    return dict(mf.get('num_params', {})), dict(mf.get('num_params_err') or {})


def _default_fundamental_point(ns, model):
    """Build a ``fundamental`` dict for the verification's numerical
    fallback.  Prefers the model author's per-parameter ``default``
    values (which the model ships specifically because they admit a
    well-behaved MF solution); falls back to all-ones only when a
    parameter has no default declared.

    All-ones is a poor universal fallback for non-linear closures —
    e.g. quad ``phi(v) = a·v²`` with ``Em=a=w=1`` yields the saddle
    equation ``2v² - v + 1 = 0`` (no real root).  The model's own
    defaults sidestep this.
    """
    fundamental: dict = {}
    pop_size_map = getattr(ns, '_pop_size', {}) or {}
    for pspec in model.get('parameters', []):
        if pspec.get('mean_field'):
            continue
        pname = pspec['name']
        default = pspec.get('default')
        if default is not None:
            fundamental[pname] = default
            continue
        ib = pspec.get('indexed_by') or []
        if not ib:
            fundamental[pname] = 1.0
        elif len(ib) == 1:
            n = pop_size_map.get(ib[0], 2)
            fundamental[pname] = [1.0] * n
        else:
            n_rows = pop_size_map.get(ib[0], 2)
            n_cols = pop_size_map.get(ib[1], 2)
            fundamental[pname] = [[1.0] * n_cols for _ in range(n_rows)]
    # A model that declares no parameters (every coefficient a literal) has
    # the empty point, which is still a valid point: returning ``None`` here
    # would switch the numerical fallback off, and a coupled implicit saddle
    # (``x* = 0.8 y*``, ``y* = -0.5 x*``) is only verifiable through it (the
    # symbolic substitution above is simultaneous, so it cannot close a loop).
    if not fundamental and model.get('parameters') is None:
        return None
    return fundamental


class _MFProxyForSolver:
    """Minimal FieldTheory-shaped proxy that ``solve_mean_field`` can
    consume during MF-sector verification.  We can't call
    ``solve_mean_field(self, ...)`` from inside ``expand()`` because
    ``self`` isn't fully populated yet — this proxy exposes just the
    ``_ns``, ``taylor_order``, and ``model`` attributes the solver needs.
    """
    __slots__ = ('_ns', 'taylor_order', 'model')

    def __init__(self, ns, model, taylor_order=4):
        self._ns = ns
        self.model = model
        self.taylor_order = taylor_order


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FieldTheory:
    """
    Model-agnostic MSRJD field theory expander (SageMath version).

    The action callable in the model dict returns an SR expression. Field
    variables are SR symbolic variables. Nonlinear functions (exp, formal
    function symbols, etc.) are Taylor-expanded automatically.

    Parameters
    ----------
    model : dict
        Model specification dict (see daedalus/simulations/hawkes_sage.py).
    taylor_order : int
        Truncation order for all Taylor expansions.
    """

    def __init__(self, model: dict, taylor_order: int = 4):
        self.model        = model
        self.taylor_order = taylor_order
        self._ns      = None
        self._R       = None
        self._n_tilde = 0
        self._S_raw   = None
        self._by_tp   = None
        self._mf_sector_raw = None
        # Interaction sectors written with per-field kernel aliases (see
        # ``expand``); ``None`` when no kernel convolves several fields.
        self._vertex_alias_by_tp = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self) -> None:
        """
        Build the namespace, evaluate the action as an SR expression,
        auto-expand all nonlinear functions of the field variables via
        sequential taylor(), rename formal derivative symbols, apply MF
        background conditions and specializations, then classify by bigrade.
        """
        ns, R, n_tilde = self._build_namespace()
        self._ns      = ns
        self._R       = R
        self._n_tilde = n_tilde

        # Evaluate action — result is an SR expression
        S_sr = SR(self.model['action'](ns))

        # ── Conv(kernel, field) reduction ─────────────────────────────
        # Resolve every ``Conv(g, X)`` atom according to the time-domain
        # convolution rules:
        #   * linearity in arg 2  ⇒  Conv(g, a+b) = Conv(g, a) + Conv(g, b)
        #   * pull constant prefactors out of arg 2
        #   * Conv(g, constant) = ĝ(0) · constant   (ĝ(0) = the kernel's
        #     integral, read off ``kernel_ft_image`` at omega = 0; a
        #     normalised kernel leaves the constant untouched)
        #   * Conv(g, fluctuation) → g · fluctuation   (kernel symbol stays
        #     for the FT pipeline to substitute ĝ(ω) later).
        #
        # The asymmetry between Conv's two arguments matters for any
        # action term where a kernel is convolved with a field that
        # ALSO multiplies another field outside the convolution — e.g.
        # conductance-style ``v · Conv(g, n)``, which must expand under
        # v=vstar+dv, n=nstar+dn as
        #    vstar·nstar + vstar·g·dn + dv·nstar + dv·g·dn
        # The naïve ``Conv(g, n) → g·n`` flatten gives the wrong
        # ``g·nstar`` bilinear coefficient on the dv side.
        from daedalus.engine.core.convolution import reduce_conv_in_action
        dc_gains = kernel_dc_gains_of(ns, self.model)
        S_alias = None
        self._vertex_alias_by_tp = None
        try:
            fluct_vars = set(ns._all_field_sr_vars)
            # Pass taylor_order so nonlinear Conv arguments
            # (``Conv(g, h(v, n))`` with non-polynomial h) get Taylor-
            # expanded at the same order the action itself is about to
            # be expanded at downstream.  For polynomial-of-degree-≤N
            # arguments the Taylor is a no-op, so existing models
            # using simple ``Conv(g, n[j])`` form keep bit-identical
            # output.
            #
            # ``attachments`` collects ``kernel_symbol → set of attached
            # fluct vars`` for every rule-4 ``Conv(g, fluct) → g·fluct``
            # emission.  Stashed on ``ns`` for downstream vertex
            # extraction (``extract_vertex_types``) to identify which
            # leg each surviving kernel symbol in an interaction-vertex
            # coefficient is attached to — needed for the time-domain
            # Phase J integrator's per-leg τ scaffolding.
            attachments = {}
            S_conv = S_sr
            S_sr = reduce_conv_in_action(
                S_conv, fluct_vars, taylor_order=self.taylor_order,
                attachments_out=attachments, dc_gains=dc_gains,
            )
            ns._kernel_attachments = attachments
            # A kernel convolving several fields (``Conv(g, x)`` and
            # ``Conv(g, y)``, or a shared ``Conv(r, n[j])`` over a
            # population) leaves vertex coefficients that cannot say which
            # leg each factor of ``g`` came from.  Reduce once more with one
            # alias per (kernel, field) emission; its interaction sectors
            # (below) feed vertex extraction only, every other consumer
            # keeps the sectors written with ``g``.
            shared = [g for g, fs in attachments.items() if len(fs) > 1]
            if shared:
                aliases = {}
                S_alias = reduce_conv_in_action(
                    S_conv, fluct_vars, taylor_order=self.taylor_order,
                    dc_gains=dc_gains, alias_kernels=shared,
                    aliases_out=aliases,
                )
                if aliases:
                    ns._kernel_aliases = aliases
                else:
                    S_alias = None
        except (AttributeError, TypeError):
            # No Conv atoms / non-SR action / missing ns attribute —
            # treat as identity transformation.
            pass

        # Inject -W_m[mt] cumulant series from correlated_noises declarations
        # (Option A: user writes ns.dm[i] / ns.mt[i] in the action, the
        # framework appends the non-local cumulant terms here from the
        # declarative spec).  Returns SR(0) when the model has no
        # correlated_noises block, so existing models are untouched.
        S_cum = _build_cumulant_action(ns, self.model)
        S_sr = S_sr + S_cum
        by_tp = self._sr_to_bigrade(S_sr, ns, R, n_tilde)
        if S_alias is not None:
            self._vertex_alias_by_tp = {
                key: poly for key, poly in self._sr_to_bigrade(
                    S_alias + S_cum, ns, R, n_tilde).items()
                if key[0] + key[1] >= 3 and key[1] >= 1
                and poly != R.zero()}

        # Apply MF background conditions ONLY to the MF sector — the
        # bigrade-≤1-in-each-index slots (0,0), (1,0), (0,1) that hold
        # the saddle-eq residual.  At the saddle this sector must
        # vanish; the substitution is the *test* of that condition,
        # not a structural rewrite of the propagator.
        mf_bg_key = ('mf_bg_conditions_action'
                     if 'mf_bg_conditions_action' in self.model
                     else 'mf_bg_conditions')
        # Preserve the pre-zero MF sector polynomials for diagnostics.
        # Downstream code reads ``_by_tp`` (which has the MF sector
        # zeroed by construction); ``_mf_sector_raw`` lets a debug
        # print show what was there before verification.
        self._mf_sector_raw = {
            key: by_tp.get(key, R.zero())
            for key in [(0, 0), (1, 0), (0, 1)]
        }

        if mf_bg_key in self.model:
            mf_subs = self.model[mf_bg_key](ns)
            # Re-apply specializations after mf_subs to resolve any
            # phi-Taylor symbols (``phi0_<i+1>``, ``phi1_<i+1>``, …)
            # reintroduced by the vstar substitution's RHS — the
            # mf_bg lambda builds RHSes BEFORE specializations runs,
            # so ``vstar → (Em + Σwg·phi0_<j+1>)/(1+τ·phi0_<i+1>)``
            # carries raw phi-Taylor tokens that need a second pass.
            spec_subs = (self.model['specializations'](ns)
                         if 'specializations' in self.model else None)
            _verify_and_zero_mf_sector(
                by_tp, mf_subs, spec_subs, ns, R, self.model,
                mf_sector_keys=[(0, 0), (1, 0), (0, 1)],
            )

        # Rebuild S_poly from by_tp (MF sector now zeroed) so
        # downstream consumers that read self._S_raw see a consistent
        # action whose (≤1, ≤1) sectors are exactly zero.
        # The sectors partition the monomials (the running ``+`` re-tested
        # every coefficient at each step).
        S_raw = _sum_of_polys(R, list(by_tp.values()))

        self._S_raw = S_raw
        self._by_tp = by_tp

        # First-order-in-time invariant.  Every downstream consumer of
        # ``_by_tp`` (propagator kernel, MF linearization pencil,
        # vertex form factors) assumes each ``Dt`` appears at most
        # linearly; a ``Dt^2`` term would silently vanish from the
        # linearization instead of being rejected.  Check once, here,
        # where the whole action is in hand.
        self.check_first_order_in_dt()

    def _sr_to_bigrade(self, S_sr, ns, R, n_tilde):
        """Taylor-expand the reduced action ``S_sr`` (Conv atoms resolved,
        cumulant terms added), rename and specialise, and classify it by
        bigrade: ``{(n_tilde, n_phys): polynomial}`` (MF sector untouched)."""
        # Multivariate Taylor in all field variables around 0 (one-shot).
        # Truncates at TOTAL degree taylor_order, which matches the
        # diagrammatic vertex-leg-count interpretation.  Replaces the
        # previous sequential single-variable loop, which broke for
        # multi-arg formal functions (chained .taylor() doesn't compose
        # at non-zero expansion points — the inner-arg substructures
        # get frozen by the outer call).  For single-arg formal
        # functions, the result is identical to the old sequential
        # loop (since each formal call only depends on one fluctuation
        # variable, multivariate Taylor at total order N produces the
        # same monomials as per-variable order N).
        #
        # ``_taylor_action`` builds that polynomial atom by atom with
        # truncation at every product (see ``_truncated_taylor``).
        if ns._all_field_sr_vars:
            S_sr = _taylor_action(
                S_sr, ns._all_field_sr_vars, self.taylor_order)

        # Rename formal function derivative symbols to clean SR names
        # e.g. D[0](phi_1)(0) → phi1_1,  phi_1(0) → phi0_1
        if ns._deriv_rename_subs:
            S_sr = S_sr.subs(ns._deriv_rename_subs)

        # Apply specializations (e.g. phi0_<i+1> → a*vstar_<i+1>,
        # phi1_<i+1> → a, quadratic phi, delta g).  These are pure
        # closure renames and are safe to apply to every sector — they
        # do not introduce parameter dependence that should be confined
        # to the saddle-eq sector.
        if 'specializations' in self.model:
            S_sr = S_sr.subs(self.model['specializations'](ns))

        # Coerce to polynomial ring and bigrade-classify FIRST, BEFORE
        # applying MF saddle substitutions.  This is critical: MF subs
        # of the form ``vstar → (Em + Σwg·nstar)/(1+τ·nstar)`` rewrites
        # ``vstar`` everywhere it appears, which would otherwise inject
        # Em-dependence (and other saddle-eq algebra) into the (1,1)
        # bilinear propagator kernel and the ≥2 interaction vertices,
        # where ``vstar``/``nstar`` should remain as free symbolic
        # parameters bound numerically downstream via num_params.
        S_poly = _sr_to_ring(S_sr.expand(), R, ns._ring_var_names)
        by_tp = _collect_bigrade(S_poly, n_tilde)
        return by_tp

    def check_first_order_in_dt(self) -> None:
        """Reject an action with any term of ``Dt``-degree ≥ 2.

        Raises :class:`daedalus.engine.core.dt_order.NonlinearDtError` (a
        ``ValueError``).  Called at the end of :meth:`expand`, and again
        by the expand-cache loader so a bundle written before this guard
        existed cannot sneak a second-order action back in.
        """
        from daedalus.engine.core.dt_order import check_action_first_order_in_dt
        ns = self._ns
        if ns is None or self._by_tp is None:
            return
        check_action_first_order_in_dt(
            self._by_tp,
            getattr(ns, 'Dt', None),
            getattr(ns, '_ring_var_names', []),
            model_name=self.model.get('name', '<unnamed>'),
        )

    def sanity_check(self, verbose=True) -> bool:
        """
        Verify zero sectors:
          (0,0) — constant
          (1,0) — tadpole (must vanish at MF saddle)
          (0,1) — EOM residual (must vanish at background solution)

        Prints the per-sector PASS/FAIL table only when ``verbose`` (default
        True, preserving the historical behaviour for direct callers); a
        FAILED sector is ALWAYS printed so a failure is never silent.
        """
        self._require_expanded()
        checks = [
            ((0, 0), 'constant term'),
            ((1, 0), 'tadpole — must vanish at MF saddle'),
            ((0, 1), 'linear physical-only — must vanish at EOM'),
        ]
        results = []
        all_pass = True
        for key, label in checks:
            val = self._by_tp.get(key, self._R.zero())
            ok  = (val == self._R.zero())
            results.append((ok, key, label, val))
            all_pass = all_pass and ok
        if verbose or not all_pass:
            print('=== Sanity checks ===')
            for ok, key, label, val in results:
                print(f'  [{"PASS" if ok else "FAIL"}]  '
                      f'(n_tilde={key[0]}, n_phys={key[1]})  {label}')
                if not ok:
                    _show(val)
        return all_pass

    def summary(self) -> None:
        """Print and display all non-zero bigrade sectors."""
        self._require_expanded()
        print('=== Action sectors ===')
        for (n_t, n_p), expr in sorted(self._by_tp.items()):
            if expr == self._R.zero():
                continue
            label = self._sector_label(n_t, n_p)
            print(f'  (n_tilde={n_t}, n_phys={n_p})  [{label}]:')
            _show(expr)

    def free_action(self):
        """Return the (1,1) sector polynomial, or ``R.zero()`` if absent."""
        self._require_expanded()
        return self._by_tp.get((1, 1), self._R.zero())

    def noise_kernel(self) -> dict:
        """Return all (≥2, 0) sectors."""
        self._require_expanded()
        return {k: v for k, v in self._by_tp.items()
                if k[0] >= 2 and k[1] == 0 and v != self._R.zero()}

    def vertices(self) -> dict:
        """Return all sectors with total degree ≥ 3."""
        self._require_expanded()
        return {k: v for k, v in self._by_tp.items()
                if k[0] + k[1] >= 3 and v != self._R.zero()}

    def sectors(self) -> dict:
        """Return full bigrade dict (non-zero sectors only)."""
        self._require_expanded()
        return {k: v for k, v in self._by_tp.items()
                if v != self._R.zero()}

    def ring(self):
        self._require_expanded()
        return self._R

    # ------------------------------------------------------------------
    # Namespace builder
    # ------------------------------------------------------------------

    def _build_namespace(self):
        """
        Construct the _Namespace with:
          - SR symbolic variables for all field variables
          - SR symbolic variables for all parameters, kernels, operators
          - Callables for nonlinear functions (auto-expand or taylor_coeffs)
          - _deriv_rename_subs: {formal_deriv_at_0 → nice_SR_var} for renaming
          - _all_field_sr_vars: list of SR vars to expand in

        Also builds PolynomialRing(SR, ring_var_names) for bigrade analysis.
        Returns (ns, R, n_tilde).
        """
        m   = self.model
        ns  = _Namespace()
        idx = m['index_sets']
        for name, lst in idx.items():
            setattr(ns, name, list(lst))
        primary_idx = list(list(idx.values())[0])

        # ── Population-aware sizing ────────────────────────────
        # For heterogeneous models, every population in
        # ``model['populations']`` has its own index range
        # ``[0, ..., size-1]`` exposed on the namespace under both
        # ``pop_<name>`` and (for ergonomic action text) ``<name>``.
        # Legacy models have an empty ``populations`` list and
        # fall back to the single flat ``pop`` index.
        populations = m.get('populations') or []
        pop_size = {p['name']: int(p.get('size', 1)) for p in populations}
        pop_local_idx = {name: list(range(sz))
                         for name, sz in pop_size.items()}
        # Bind plain-name iterables for use in action text:
        # ``for i in E`` and ``for j in I`` work as written.
        for pname, plist in pop_local_idx.items():
            if not hasattr(ns, pname):
                setattr(ns, pname, list(plist))
            # Also bind under the ``pop_<name>`` alias for callers
            # that prefer the explicit form.
            alias = f'pop_{pname}'
            if not hasattr(ns, alias):
                setattr(ns, alias, list(plist))

        def _field_indices(fspec):
            """Indices over which a field's SR vars range.  Uses
            ``fspec['population']`` when given (per-pop sizing); falls
            back to ``primary_idx`` for legacy fields."""
            pop = fspec.get('population')
            if pop and pop in pop_local_idx:
                return pop_local_idx[pop]
            return primary_idx

        def _entity_axis_sizes(spec):
            """Resolve the per-axis size list for a parameter / kernel.

            Returns ``[]`` for scalar, ``[N_a]`` for vector, ``[N_a, N_b]``
            for matrix.  ``indexed_by`` (heterogeneous-pop) wins over
            legacy ``indexed=`` when both are present.
            """
            ib = spec.get('indexed_by')
            if ib:
                return [pop_size.get(p, len(primary_idx)) for p in ib]
            indexed = spec.get('indexed', False)
            if indexed == 'matrix':
                return [len(primary_idx), len(primary_idx)]
            if indexed is True or indexed == 'vector':
                return [len(primary_idx)]
            return []

        # Stash for downstream code (mf_substitutions, _mean_field, etc.).
        ns._populations    = populations
        ns._pop_size       = pop_size
        ns._pop_local_idx  = pop_local_idx
        # Expose the axis-size helper so per-pop param / kernel iterators
        # in pipeline code can reuse the same resolution rules.
        ns._entity_axis_sizes = _entity_axis_sizes

        # ---- Field variables as SR symbols ----
        from daedalus.engine.core.vertices import field_symbol_name
        tilde_sr_vars: list = []
        phys_sr_vars:  list = []
        tilde_names:   list = []
        phys_names:    list = []

        for fspec in m['response_fields']:
            fname   = fspec['name']
            indexed = fspec.get('indexed', True)
            lx      = fspec.get('latex', fname)
            if indexed:
                idx_list = _field_indices(fspec)
                arr = [SR.var(field_symbol_name(fname, i + 1),
                              latex_name=f'{lx}_{{{i+1}}}')
                       for i in idx_list]
                setattr(ns, fname, arr)
                tilde_sr_vars.extend(arr)
                tilde_names.extend(field_symbol_name(fname, i + 1)
                                  for i in idx_list)
            else:
                v = SR.var(field_symbol_name(fname), latex_name=lx)
                setattr(ns, fname, v)
                tilde_sr_vars.append(v)
                tilde_names.append(field_symbol_name(fname))

        for fspec in m['physical_fields']:
            fname   = fspec['name']
            indexed = fspec.get('indexed', True)
            lx      = fspec.get('latex', fname)
            if indexed:
                idx_list = _field_indices(fspec)
                arr = [SR.var(field_symbol_name(fname, i + 1),
                              latex_name=f'{lx}_{{{i+1}}}')
                       for i in idx_list]
                setattr(ns, fname, arr)
                phys_sr_vars.extend(arr)
                phys_names.extend(field_symbol_name(fname, i + 1)
                                  for i in idx_list)
            else:
                v = SR.var(field_symbol_name(fname), latex_name=lx)
                setattr(ns, fname, v)
                phys_sr_vars.append(v)
                phys_names.append(field_symbol_name(fname))

        ns._tilde_sr_vars    = tilde_sr_vars
        ns._phys_sr_vars     = phys_sr_vars
        ns._all_field_sr_vars = tilde_sr_vars + phys_sr_vars

        # ---- Polynomial ring (tilde generators first) ----
        ring_var_names       = tilde_names + phys_names
        ns._ring_var_names   = ring_var_names
        n_tilde              = len(tilde_names)
        R                    = PolynomialRing(SR, ring_var_names)

        # ---- SR parameters ----
        # Vector params get a flat list of SR vars sized by their
        # ``indexed_by`` population (or the legacy primary_idx).
        # Matrix params get a 1D placeholder list; the actual 2D
        # element-naming grid is set later by ``mf_substitutions``,
        # which also knows about ``indexed_by`` for per-pop sizing.
        # Mean-field saddle names (the expression namespace binds a size-1
        # saddle as a scalar too: bare ``vstar`` in a scalar model).
        ns._mf_saddle_names = frozenset(
            p['name'] for p in m.get('parameters', []) if p.get('mean_field'))
        for pspec in m.get('parameters', []):
            pname   = pspec['name']
            domain  = pspec.get('domain', None)
            axis_sizes = _entity_axis_sizes(pspec)
            if not axis_sizes:
                sym = SR.var(pname, domain=domain) if domain else SR.var(pname)
                setattr(ns, pname, sym)
            else:
                # For vector: axis_sizes = [N_a].
                # For matrix: take the first axis; the 2D structure
                # gets installed by mf_substitutions.
                n = axis_sizes[0]
                arr = ([SR.var(f"{pname}{i+1}", domain=domain)
                        for i in range(n)]
                       if domain else
                       [SR.var(f"{pname}{i+1}") for i in range(n)])
                setattr(ns, pname, arr)

        # ---- Kernels and operators ----
        # Kernels can be:
        #   * scalar (default):  one SR symbol  ns.g  used as ``g`` in action.
        #   * vector (indexed=True/'vector' or indexed_by=['X']):
        #     N_X SR symbols ``g_<i+1>``, namespace exposes
        #     ``ns.g = [g_1, ...]`` for ``g[i]`` syntax.
        #   * matrix (indexed='matrix' or indexed_by=['X', 'Y']):
        #     N_X × N_Y SR symbols ``g_<i+1>_<j+1>``, namespace
        #     exposes ``ns.g`` as a list-of-lists for ``g[i, j]`` syntax.
        #     The companion ``kernel_ft_image`` lambda evaluates the
        #     frequency image per (i, j) pair so per-pair parameters
        #     like ``tau_g[i, j]`` resolve correctly.
        for kspec in m.get('kernels', []):
            kname    = kspec['name']
            klx_base = kspec.get('latex_name', None) or kname
            axis_sizes = _entity_axis_sizes(kspec)
            if len(axis_sizes) == 2:
                n_rows, n_cols = axis_sizes
                rows = []
                for i in range(n_rows):
                    row = []
                    for j in range(n_cols):
                        sn = (f'{kspec.get("sage_name", "z_" + kname)}'
                              f'_{i+1}_{j+1}')
                        lx = f'{klx_base}_{{{i+1}{j+1}}}'
                        row.append(SR.var(sn, latex_name=lx))
                    rows.append(row)
                setattr(ns, kname, rows)
            elif len(axis_sizes) == 1:
                arr = []
                for i in range(axis_sizes[0]):
                    sn = f'{kspec.get("sage_name", "z_" + kname)}_{i+1}'
                    lx = f'{klx_base}_{{{i+1}}}'
                    arr.append(SR.var(sn, latex_name=lx))
                setattr(ns, kname, arr)
            else:
                kn_sage = kspec.get('sage_name', kname)
                setattr(ns, kname,
                        SR.var(kn_sage, latex_name=klx_base)
                        if klx_base else SR.var(kn_sage))
        for ospec in m.get('operators', []):
            oname = ospec.get('sage_name', ospec['name'])
            olx   = ospec.get('latex_name', None)
            setattr(ns, ospec['name'],
                    SR.var(oname, latex_name=olx) if olx else SR.var(oname))

        # Spatial v2: flag the operator-IR authoring mode (Lap()/Dt()/Dx()
        # binding syntax) so the action-eval namespace and the action lambda
        # route through ``pipeline.spatial_operator_ir``.  Default False —
        # the bare multiplicative ``Dt``/``Laplacian`` symbols above are
        # untouched for every existing model.
        ns._operator_ir = bool(m.get('operator_ir', False))

        # Internal names start with 'z' so they sort after tau ('t') and phi ('p')
        # in any SR product, giving the canonical order: φ … τ … δ/δ'.
        ns.delta_D  = SR.var('z_delta',   latex_name=r'\delta')
        ns.delta_Dp = SR.var('z_delta_p', latex_name=r"\delta'")

        # ---- Nonlinear functions ----
        # Accumulate derivative-rename substitutions for formal function symbols.
        # The rename machinery is multi-arg-aware: for an n-argument
        # function, every partial derivative (k_1, ..., k_n) with
        # sum ≤ taylor_order gets registered as a rename to
        # ``<prefix><k_1><k_2>...<k_n>_<i+1>``.  Single-arg
        # (``n_args=1``, the default) collapses to the legacy
        # ``<prefix><k>_<i+1>`` naming, so existing models are
        # unaffected.
        ns._deriv_rename_subs = {}
        order = self.taylor_order

        for fspec in m.get('functions', []):
            fname        = fspec['name']
            indexed      = fspec.get('indexed', True)
            deriv_prefix = fspec.get('deriv_prefix', fname)
            n_args       = int(fspec.get('n_args', 1))

            if 'expression' in fspec:
                # ---- Auto-expand path ----
                # ``fspec['expression'](i, x_1, ..., x_n)`` returns an SR
                # expression in the n formal arguments.  Partial
                # derivatives at the all-zero point get renamed to
                # ``<prefix><multi-idx-suffix>_<i+1>``.
                fn_latex = fspec.get('latex', deriv_prefix)
                arg_dums = [SR.var(f'_xdum_{deriv_prefix}_{j}')
                            for j in range(n_args)]

                def _deriv_latex(base, multi_idx, sub):
                    """LaTeX name for the partial-derivative symbol."""
                    total = sum(multi_idx)
                    if total == 0:
                        return (f'{base}_{{{sub}}}' if sub != '' else base)
                    if len(multi_idx) == 1:
                        # Legacy single-arg notation:  f', f'', f^{(k)}.
                        k = multi_idx[0]
                        primes = "'" if k == 1 else ("''" if k == 2 else None)
                        if primes:
                            return (f"{base}{primes}_{{{sub}}}"
                                    if sub != '' else f"{base}{primes}")
                        return (f'{base}^{{({k})}}_{{{sub}}}'
                                if sub != '' else f'{base}^{{({k})}}')
                    # Multi-arg: superscript shows the multi-index tuple.
                    idx_str = ','.join(str(k) for k in multi_idx)
                    return (f'{base}^{{({idx_str})}}_{{{sub}}}'
                            if sub != '' else f'{base}^{{({idx_str})}}')

                def _build_target(multi_idx, sub_label):
                    suffix = _multi_index_suffix(multi_idx)
                    base   = f'{deriv_prefix}{suffix}'
                    return (f'{base}_{sub_label}' if sub_label != ''
                            else base)

                def _register_renames(fe, sub_label):
                    """Register a rename for every multi-derivative of
                    ``fe`` (an SR expression in the ``arg_dums``)
                    evaluated at the all-zero expansion point."""
                    zero_subs = {arg_dums[j]: 0 for j in range(n_args)}
                    for multi_idx in _iter_multi_indices(n_args, order):
                        deriv = fe
                        for j, kj in enumerate(multi_idx):
                            if kj > 0:
                                deriv = diff(deriv, arg_dums[j], kj)
                        try:
                            deriv_at_0 = SR(deriv).subs(zero_subs)
                            if SR(deriv_at_0).is_numeric():
                                continue
                            lname = _deriv_latex(fn_latex, multi_idx, sub_label)
                            nice  = SR.var(_build_target(multi_idx, sub_label),
                                           latex_name=lname)
                            ns._deriv_rename_subs[deriv_at_0] = nice
                        except Exception:
                            pass

                if indexed:
                    fn_exprs = []
                    for i in primary_idx:
                        fe = fspec['expression'](i, *arg_dums)
                        fn_exprs.append(fe)
                        _register_renames(fe, str(i + 1))

                    def _make_fn(exprs, dums):
                        def fn(i, *xs):
                            sub = {dums[j]: xs[j] for j in range(len(dums))}
                            return exprs[i].subs(sub)
                        return fn
                    setattr(ns, fname, _make_fn(fn_exprs, arg_dums))

                else:
                    fe = fspec['expression'](*arg_dums)
                    _register_renames(fe, '')
                    def _make_scalar_fn(fe_, dums):
                        def fn(*xs):
                            sub = {dums[j]: xs[j] for j in range(len(dums))}
                            return fe_.subs(sub)
                        return fn
                    setattr(ns, fname, _make_scalar_fn(fe, arg_dums))

            elif 'taylor_coeffs' in fspec:
                # ---- Legacy path: manually supplied derivative coefficients ----
                coeffs = fspec['taylor_coeffs']
                if indexed:
                    def _make_tc(c_list, o):
                        def fn(x):
                            return _poly_taylor(c_list[:o+1], x)
                        return fn
                    fns = [_make_tc(coeffs[i], order) for i in range(len(primary_idx))]
                    setattr(ns, fname, lambda i, x, _fns=fns: _fns[i](x))
                else:
                    def _make_scalar(c_list, o):
                        def fn(x):
                            return _poly_taylor(c_list[:o+1], x)
                        return fn
                    setattr(ns, fname, _make_scalar(coeffs, order))

        # Expose taylor_order on namespace (used by specializations lambdas)
        ns._taylor_order = self.taylor_order

        # ---- MF substitutions (computed once at namespace build) ----
        for sub in m.get('mf_substitutions', []):
            setattr(ns, sub['name'], sub['value'](ns))

        return ns, R, n_tilde

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_expanded(self):
        if self._by_tp is None:
            raise RuntimeError(
                "FieldTheory.expand() has not been run yet: the sectors, vertices and propagators are built by expand(); call it before reading them.")

    @staticmethod
    def _sector_label(n_t: int, n_p: int) -> str:
        total = n_t + n_p
        if (n_t, n_p) == (1, 1):  return 'free action'
        if n_t >= 2 and n_p == 0: return 'noise kernel'
        if total == 1:             return 'tadpole / background'
        if total >= 3:             return f'vertex (order {total})'
        return f'sector ({n_t},{n_p})'

"""Declared names as Sage symbols for the SDE front end (:mod:`daedalus.api.sde`).

Each declaration of :class:`daedalus.api.sde.SDE` that introduces a name
(:meth:`~daedalus.api.sde.SDE.field`, :meth:`~daedalus.api.sde.SDE.parameter`,
:meth:`~daedalus.api.sde.SDE.operator`, :meth:`~daedalus.api.sde.SDE.define_function`,
:meth:`~daedalus.api.sde.SDE.define_kernel`, :meth:`~daedalus.api.sde.SDE.population`,
:meth:`~daedalus.api.sde.SDE.point_process`, and :meth:`~daedalus.api.sde.SDE.noise` with
``as_field=``) returns a :class:`Symbol`: the Sage symbol of that name, so
expressions can be written as Sage objects::

    m = SDE('OU quartic')
    mu, eps, D = (m.parameter(n, default=v) for n, v in (('mu', 1.0), ('eps', 0.02), ('D', 1.0)))
    x = m.field('x')
    m.drift(x, -mu*x - eps*x**3).noise('xi', Gaussian(var=2*D), into={x: 1})

A :class:`Symbol` is also the builder: any method of the builder called on
it is the builder's (``m.field('x').drift('x', ...)`` chains as before), and
it returns what the builder returns.  Arithmetic gives plain Sage
expressions; a call (``lap(u)``, ``phi(v)``, ``x(t - d)``) is the call of
that name, ``x[i]`` / ``w[i, j]`` its indexed form.  Every method that takes
an expression accepts a string or a Sage expression; a Sage expression is
written as text with :func:`daedalus.api.noise.expr_to_text` (Sage's order of terms)
and read like that string.  Strings keep working unchanged, byte for byte.
"""
from __future__ import annotations

from sage.all import SR, function
from sage.symbolic.expression import Expression

from daedalus.api.noise import expr_to_text

__all__ = ['Symbol', 'Conv', 'as_text', 'as_name']

#: ``Conv(K, x)`` as a Sage function (a kernel ``K`` convolved with ``x``)
Conv = function('Conv')


class Symbol(Expression):
    """The Sage symbol of a declared name that also forwards the builder's
    methods (see the module docstring).  ``kind`` is what was declared
    (``'field'``, ``'parameter'``, ``'operator'``, ``'function'``,
    ``'kernel'``, ``'population'``)."""

    def __init__(self, sde, name: str, kind: str):
        Expression.__init__(self, SR, SR.var(name))
        self._sde, self._decl_name, self._kind = sde, name, kind

    def __getattr__(self, attr):
        if attr.startswith('__') or attr in ('_sde', '_decl_name', '_kind'):
            raise AttributeError(attr)
        return getattr(self._sde, attr)

    # the two builder methods whose names Sage expressions also have
    def operator(self, *args, **kw):
        """The builder's :meth:`daedalus.api.sde.SDE.operator` when called with
        arguments; without, Sage's ``Expression.operator()``."""
        if args or kw:
            return self._sde.operator(*args, **kw)
        return Expression.operator(self)

    def show(self):
        """The builder's :meth:`daedalus.api.sde.SDE.show` (the derived action)."""
        return self._sde.show()

    def __call__(self, *args):
        if self._kind == 'parameter':
            raise TypeError(f'parameter {self._decl_name!r} is not callable (a function is '
                            f'declared with define_function).')
        return function(self._decl_name)(*args)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        return SR.var('__'.join([self._decl_name] + [as_name(k) for k in idx]))

    def __reduce__(self):
        return (SR.var, (self._decl_name,))


def as_text(value):
    """``value`` as the builder reads it: a string unchanged, a Sage
    expression (a :class:`Symbol` too) through :func:`expr_to_text`, anything
    else unchanged."""
    if isinstance(value, str):
        return value
    if isinstance(value, Expression):
        return expr_to_text(value)
    return value


def as_name(value) -> str:
    """The name of a declared :class:`Symbol` (or a plain Sage symbol), or
    ``value`` itself (a string)."""
    if isinstance(value, Symbol):
        return value._decl_name
    if isinstance(value, Expression) and value.is_symbol():
        return str(value)
    return value

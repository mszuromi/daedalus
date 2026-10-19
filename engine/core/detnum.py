"""Process-independent numeric substitution into Sage expressions.

Pynac (Sage's symbolic engine) orders the operands of a product or sum by a hash
of the symbols that is drawn afresh in every process.  ``expr.subs({a: 1.1, b: 2.3,
c: 3.7})`` on ``a*b*c`` multiplies the floats in that process-dependent order, so
the last bit of the result differs from one process to the next (1-4 ulp scatter
of the Phase J prefactors, and hence of every cumulant built from them, identical
within a process but not across processes, whatever ``PYTHONHASHSEED`` is).

:func:`numeric_subs` evaluates the expression itself, in Python doubles, with the
operands in a canonical order (sorted by their text) and sums rounded exactly
(``math.fsum``), so the value depends on the expression and the parameter values
only.  An expression with a free symbol the substitution does not cover (or a node
this evaluator does not know) falls back to ``expr.subs``.
"""
from __future__ import annotations

import cmath
import math
import operator

from sage.all import SR

__all__ = ['numeric_subs']


class _Fallback(Exception):
    """The expression has a symbol or node the deterministic evaluator cannot do."""


def _number(v):
    """A Python float or complex from a parameter value (Sage, numpy or Python)."""
    c = complex(v)
    return c.real if c.imag == 0.0 else c


def _ev(e, vals):
    if e.is_symbol():
        try:
            return vals[str(e)]
        except KeyError:
            raise _Fallback from None
    op = e.operator()
    if op is None:
        if e.is_numeric():
            return _number(e)
        raise _Fallback
    args = e.operands()
    if op is operator.add:
        xs = [_ev(a, vals) for a in args]
        re = math.fsum(x.real if isinstance(x, complex) else x for x in xs)
        im = math.fsum(x.imag if isinstance(x, complex) else 0.0 for x in xs)
        return complex(re, im) if im != 0.0 else re
    if op is operator.mul:
        acc = 1.0
        for a in sorted(args, key=str):
            acc = acc * _ev(a, vals)
        return acc
    if op is operator.pow:
        base, ex = _ev(args[0], vals), args[1]
        if ex.is_numeric() and ex.is_integer():
            return base ** int(ex)
        return base ** _ev(ex, vals)
    # a function of the arguments (exp, sqrt via pow, ...): the Sage value of the
    # function at double arguments (a single deterministic call)
    try:
        sv = [SR(_ev(a, vals)) for a in args]
        return _number(op(*sv))
    except Exception:
        raise _Fallback from None


def numeric_subs(expr, subs):
    """``expr`` with the parameter values of ``subs`` substituted, as an SR number
    computed in the same order in every process; ``expr.subs(subs)`` when ``expr`` is
    not a closed form in the symbols of ``subs``.  ``subs``: ``{symbol: number}``."""
    expr = SR(expr)
    if not subs:
        return expr
    try:
        vals = {str(k): _number(v) for k, v in subs.items()}
    except (TypeError, ValueError):
        return expr.subs(subs)
    try:
        free = expr.variables()
        if any(str(s) not in vals for s in free):
            return expr.subs(subs)
        return SR(_ev(expr, vals))
    except (_Fallback, OverflowError, ZeroDivisionError, ValueError, TypeError):
        return expr.subs(subs)
